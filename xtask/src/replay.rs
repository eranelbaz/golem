//! Offline placement-policy replay over routing traces (§6.3): RAM and VRAM
//! hit rates for CLOCK, LRU and the §6.2 decayed score against pool size,
//! and from them d, the share of routed expert bytes read from disk.
//!
//! Mirrors production: VRAM is a per-layer K-slot slab (P2) whose victims
//! are picked among the layer's non-routed slots; RAM is one pool split into
//! per-unit-size classes (§5.4) and sees only VRAM misses (P3), with every
//! unit claimed by the current compute leased until its copy lands (§6.6).
//! Policy metadata (LRU stamps, decayed scores) is booked on decode steps
//! only (§6.2); CLOCK reference bits are set on any tier hit (`residency.rs`).

use std::fmt::Write as _;
use std::path::Path;

use golem_placement::pool::{GgufPreadSource, class_slots};
use golem_placement::{HALF_LIFE_STEPS as HALF_LIFE, decay_factor};

/// §6.3 default min-dwell in steps.
const MIN_DWELL: u64 = 256;
/// §6.3 O10 estimate of duplicated capacity under inclusive residency.
const EXCLUSIVE_EXTRA_GIB: f64 = 10.6;
const GIB: f64 = (1u64 << 30) as f64;
/// Decode steps per trace in the cold-start window (`Stats::d_early`).
const EARLY_STEPS: usize = 32;
/// Prediction margins beyond top-k scored against disk misses (§6.5).
const MARGINS: [usize; 3] = [0, 2, 4];

/// Expert geometry the replay needs: per-layer unit bytes (gate + up +
/// down of one expert) and the per-unit-size classes of the RAM pool.
#[derive(Debug, Clone)]
pub struct Geo {
    pub unit: Vec<u64>,
    pub n_experts: u32,
    pub classes: Vec<(u64, Vec<u32>)>,
}

impl Geo {
    /// From the model's first shard (split GGUFs walk every shard).
    pub fn from_gguf(path: &Path) -> Result<Self, String> {
        let lead = lead_of(path)?;
        let geo = GgufPreadSource::open(path, lead)
            .and_then(|s| s.geometry())
            .map_err(|e| e.to_string())?
            .ok_or("model has no expert tensors")?;
        Ok(Self {
            unit: geo.layers.iter().map(|l| l.slot_bytes()).collect(),
            n_experts: geo.n_experts,
            classes: geo.classes(),
        })
    }

    fn n_layers(&self) -> u32 {
        self.unit.len() as u32
    }

    fn total_bytes(&self) -> u64 {
        self.unit.iter().sum::<u64>() * u64::from(self.n_experts)
    }
}

/// The model's dense lead: trace blk numbers minus it are expert layers.
fn lead_of(path: &Path) -> Result<u32, String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let kvs = golem_gguf::read_kvs(&mut f).map_err(|e| e.to_string())?;
    Ok(golem_gguf::live_layers(&kvs)
        .map_err(|e| e.to_string())?
        .start)
}

/// One `llama_decode` compute: a prefill chunk or a decode step.
#[derive(Debug, Clone, PartialEq)]
pub struct Compute {
    pub decode: bool,
    pub step: u64,
    /// (expert layer, routed ids) per MoE layer, in execution order.
    pub layers: Vec<(u32, Vec<u16>)>,
    /// (expert layer, predicted ids in descending router score) from `pr`
    /// records (§6.5 P5), made before that layer ran.
    pub predicted: Vec<(u32, Vec<u16>)>,
}

#[derive(Debug, Clone)]
pub struct Trace {
    pub name: String,
    pub computes: Vec<Compute>,
    /// Header `top_k=`: the routed width a `pr` record's margin is cut on.
    pub top_k: Option<usize>,
}

impl Trace {
    /// Parses a `# golem-routing-trace v1` file. Blk numbers map to expert
    /// layers by subtracting `lead`; a new compute starts whenever the blk
    /// number does not increase or the step/phase changes. `pr` rows attach
    /// to the decode step they were made in, never start a compute.
    pub fn parse(name: &str, text: &str, lead: u32, geo: &Geo) -> Result<Self, String> {
        let mut lines = text.lines();
        let header = lines.next().ok_or("empty trace")?;
        if !header.starts_with("# golem-routing-trace v1") {
            return Err(format!("{name}: not a routing trace: {header}"));
        }
        let top_k = header
            .split_whitespace()
            .find_map(|kv| kv.strip_prefix("top_k="))
            .and_then(|v| v.parse().ok());
        let mut computes: Vec<Compute> = Vec::new();
        // `pr` rows seen before their step's first `dc` row.
        let mut pending: Vec<(u64, u32, Vec<u16>)> = Vec::new();
        let mut prev_blk = u32::MAX;
        for (n, line) in lines.enumerate() {
            let line = line.trim_end_matches('\r');
            if line.is_empty() {
                continue;
            }
            let bad = |why: &str| format!("{name}:{}: {why}: {line}", n + 2);
            let cols: Vec<&str> = line.split('\t').collect();
            let [step, phase, blk, ids] = cols[..] else {
                return Err(bad("want 4 columns"));
            };
            let step: u64 = step.parse().map_err(|_| bad("step"))?;
            let decode = match phase {
                "dc" | "pr" => true,
                "pf" => false,
                _ => return Err(bad("phase")),
            };
            let blk: u32 = blk.parse().map_err(|_| bad("blk"))?;
            let layer = blk
                .checked_sub(lead)
                .filter(|&l| l < geo.n_layers())
                .ok_or_else(|| bad("blk outside the live expert layers"))?;
            let mut ids = ids
                .split(',')
                .map(|s| {
                    s.parse::<u16>()
                        .ok()
                        .filter(|&i| u32::from(i) < geo.n_experts)
                })
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| bad("ids"))?;
            if phase == "pr" {
                // The prediction for blk is made after blk - lookahead ran,
                // so it belongs to the step's open compute, if any.
                match computes.last_mut().filter(|c| c.step == step && c.decode) {
                    Some(c) => c.predicted.push((layer, ids)),
                    None => pending.push((step, layer, ids)),
                }
                continue;
            }
            // A multi-token prefill chunk repeats ids; P2 works on the set.
            ids.sort_unstable();
            ids.dedup();
            let same = computes
                .last()
                .is_some_and(|c| c.step == step && c.decode == decode && blk > prev_blk);
            if !same {
                let predicted = pending
                    .drain(..)
                    .filter(|&(s, _, _)| decode && s == step)
                    .map(|(_, l, p)| (l, p))
                    .collect();
                computes.push(Compute {
                    decode,
                    step,
                    layers: Vec::new(),
                    predicted,
                });
            }
            computes
                .last_mut()
                .expect("pushed")
                .layers
                .push((layer, ids));
            prev_blk = blk;
        }
        Ok(Self {
            name: name.to_owned(),
            computes,
            top_k,
        })
    }

    fn decode_steps(&self) -> usize {
        self.computes.iter().filter(|c| c.decode).count()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Policy {
    Clock,
    Lru,
    /// §6.2 decayed score; victims prefer keys resident longer than `min_dwell`.
    Decay {
        min_dwell: u64,
    },
    /// Frequency-weighted: lowest score decayed at a long `half_life` first
    /// (#120 O3; `u32::MAX` ≈ LFU).
    Freq {
        half_life: u32,
    },
}

impl Policy {
    fn label(self) -> String {
        match self {
            Self::Clock => "CLOCK".into(),
            Self::Lru => "LRU".into(),
            Self::Decay { min_dwell } => format!("decay H={HALF_LIFE} D={min_dwell}"),
            Self::Freq {
                half_life: u32::MAX,
            } => "LFU".into(),
            Self::Freq { half_life } => format!("freq H={half_life}"),
        }
    }
}

const ABSENT: u32 = u32::MAX;

/// One tier's residency over flat `layer × n_experts` keys, partitioned
/// (RAM: unit class, VRAM: layer) with a slot cap per partition.
struct Tier {
    policy: Policy,
    part: Vec<u32>,
    cap: Vec<u64>,
    members: Vec<Vec<u32>>,
    pos: Vec<u32>,
    clock_keys: Vec<Vec<u32>>,
    hand: Vec<usize>,
    refbit: Vec<bool>,
    stamp: Vec<u64>,
    score: Vec<f64>,
    scored_at: Vec<u64>,
    admitted: Vec<u64>,
    lease: Vec<u64>,
    epoch: u64,
    half_life: f64,
    /// Pinned hot set: resident for the whole run, never a victim.
    pinned: Vec<bool>,
}

impl Tier {
    /// `clock_keys[p]` is partition `p`'s keys in CLOCK walk order.
    fn new(policy: Policy, n_keys: usize, clock_keys: Vec<Vec<u32>>, cap: Vec<u64>) -> Self {
        let mut part = vec![0; n_keys];
        for (p, keys) in clock_keys.iter().enumerate() {
            for &k in keys {
                part[k as usize] = p as u32;
            }
        }
        Self {
            policy,
            part,
            members: vec![Vec::new(); cap.len()],
            hand: vec![0; cap.len()],
            cap,
            pos: vec![ABSENT; n_keys],
            clock_keys,
            refbit: vec![false; n_keys],
            stamp: vec![0; n_keys],
            score: vec![0.0; n_keys],
            scored_at: vec![0; n_keys],
            admitted: vec![0; n_keys],
            lease: vec![0; n_keys],
            epoch: 1,
            half_life: match policy {
                Policy::Freq { half_life } => f64::from(half_life),
                _ => HALF_LIFE,
            },
            pinned: vec![false; n_keys],
        }
    }

    fn resident(&self, key: u32) -> bool {
        self.pos[key as usize] != ABSENT
    }

    fn leased(&self, key: u32) -> bool {
        self.lease[key as usize] == self.epoch || self.pinned[key as usize]
    }

    fn decayed(&self, key: u32, now: u64) -> f64 {
        let k = key as usize;
        self.score[k] * decay_factor(now.saturating_sub(self.scored_at[k]), self.half_life)
    }

    /// Books one routed access (decode steps only, §6.2).
    fn observe(&mut self, key: u32, now: u64, tick: u64) {
        let k = key as usize;
        self.stamp[k] = tick;
        self.score[k] = self.decayed(key, now) + 1.0;
        self.scored_at[k] = now;
    }

    fn lookup(&mut self, key: u32, touch: bool) -> bool {
        if !self.resident(key) {
            return false;
        }
        self.refbit[key as usize] |= touch;
        self.lease[key as usize] = self.epoch;
        true
    }

    fn insert(&mut self, key: u32, now: u64) {
        debug_assert!(!self.resident(key), "key {key} inserted twice");
        let k = key as usize;
        let p = self.part[k] as usize;
        self.pos[k] = self.members[p].len() as u32;
        self.members[p].push(key);
        self.refbit[k] = false;
        self.admitted[k] = now;
        self.lease[k] = self.epoch;
    }

    fn remove(&mut self, key: u32) {
        let k = key as usize;
        let p = self.part[k] as usize;
        let i = self.pos[k] as usize;
        self.members[p].swap_remove(i);
        if let Some(&moved) = self.members[p].get(i) {
            self.pos[moved as usize] = i as u32;
        }
        self.pos[k] = ABSENT;
    }

    fn full(&self, key: u32) -> bool {
        let p = self.part[key as usize] as usize;
        self.members[p].len() as u64 >= self.cap[p]
    }

    fn pick(&mut self, p: usize, now: u64) -> Option<u32> {
        let unleased = |t: &Self, k: u32| !t.leased(k);
        match self.policy {
            Policy::Clock => {
                let n = self.clock_keys[p].len();
                for _ in 0..2 * n + 1 {
                    let k = self.clock_keys[p][self.hand[p]];
                    self.hand[p] = (self.hand[p] + 1) % n;
                    if !self.resident(k) || self.leased(k) {
                        continue;
                    }
                    if std::mem::take(&mut self.refbit[k as usize]) {
                        continue;
                    }
                    return Some(k);
                }
                None
            }
            Policy::Lru => self.members[p]
                .iter()
                .copied()
                .filter(|&k| unleased(self, k))
                .min_by_key(|&k| (self.stamp[k as usize], k)),
            Policy::Decay { min_dwell } => {
                let lowest = |dwelled: bool| {
                    self.members[p]
                        .iter()
                        .copied()
                        .filter(|&k| unleased(self, k))
                        .filter(|&k| {
                            !dwelled || now.saturating_sub(self.admitted[k as usize]) > min_dwell
                        })
                        .min_by(|&a, &b| self.decayed(a, now).total_cmp(&self.decayed(b, now)))
                };
                lowest(true).or_else(|| lowest(false))
            }
            Policy::Freq { .. } => self.members[p]
                .iter()
                .copied()
                .filter(|&k| unleased(self, k))
                .min_by(|&a, &b| {
                    self.decayed(a, now)
                        .total_cmp(&self.decayed(b, now))
                        .then(a.cmp(&b))
                }),
        }
    }

    /// Admits `key`, evicting a victim when its partition is full.
    fn admit(&mut self, key: u32, now: u64) -> Result<Option<u32>, String> {
        let victim = if self.full(key) {
            let p = self.part[key as usize] as usize;
            let v = self
                .pick(p, now)
                .ok_or_else(|| format!("every slot of partition {p} is leased"))?;
            self.remove(v);
            Some(v)
        } else {
            None
        };
        self.insert(key, now);
        Ok(victim)
    }

    /// Exclusive residency's D2H landing of a VRAM victim (§6.3 O10): the
    /// decayed scorer drops it when the pool's victim scores higher.
    fn admit_demoted(&mut self, key: u32, now: u64) {
        if self.full(key) {
            let p = self.part[key as usize] as usize;
            let Some(v) = self.pick(p, now) else { return };
            if matches!(self.policy, Policy::Decay { .. } | Policy::Freq { .. })
                && self.decayed(v, now) > self.decayed(key, now)
            {
                return;
            }
            self.remove(v);
        }
        self.insert(key, now);
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Config {
    pub pool_bytes: u64,
    pub k: u32,
    pub ram: Policy,
    pub vram: Policy,
    pub exclusive: bool,
    /// Background RAM → VRAM promotions per decode step (§6.3; 0 = off).
    pub promote: u32,
    /// One VRAM arena per unit class holding `layers × K` slots, victims
    /// chosen across layers, instead of K slots per layer (§3.4 P2).
    pub global_vram: bool,
    /// Prefill RAM hits set the CLOCK reference bit (`lease_if_resident`).
    pub prefill_ref: bool,
}

/// Decode-step totals (prefill computes move residency but are not counted).
#[derive(Debug, Clone, Copy, Default)]
pub struct Stats {
    pub accesses: u64,
    pub vram_hits: u64,
    pub ram_hits: u64,
    pub bytes: u64,
    pub disk_bytes: u64,
    /// `bytes` / `disk_bytes` over each trace's first [`EARLY_STEPS`].
    pub early_bytes: u64,
    pub early_disk_bytes: u64,
    /// Routed bytes / disk bytes of prefill computes (time to first token).
    pub prefill_bytes: u64,
    pub prefill_disk_bytes: u64,
    /// Background promotions issued (each one unit of off-critical-path H2D).
    pub promoted: u64,
    /// Decode layers with a `pr` prediction: routed units neither tier
    /// held when the layer ran (would-be disk misses), and their bytes.
    pub pred_disk_units: u64,
    pub pred_disk_bytes: u64,
    /// Per [`MARGINS`]: of those, units the top (k + margin) named, and
    /// bytes of named units neither tier held that the layer did not route.
    pub pred_caught: [u64; MARGINS.len()],
    pub pred_wasted_bytes: [u64; MARGINS.len()],
    /// `pr` records whose layer did not run later in the same compute.
    pub pred_unmatched: u64,
}

impl Stats {
    pub fn vram_hit(&self) -> f64 {
        pct(self.vram_hits, self.accesses)
    }

    /// Share of VRAM misses served from RAM.
    pub fn ram_hit(&self) -> f64 {
        pct(self.ram_hits, self.accesses - self.vram_hits)
    }

    /// d: share of routed expert bytes read from disk.
    pub fn d(&self) -> f64 {
        pct(self.disk_bytes, self.bytes)
    }

    /// d over each trace's first [`EARLY_STEPS`] decode steps (cold start).
    pub fn d_early(&self) -> f64 {
        pct(self.early_disk_bytes, self.early_bytes)
    }

    /// d over prefill computes.
    pub fn d_prefill(&self) -> f64 {
        pct(self.prefill_disk_bytes, self.prefill_bytes)
    }

    /// Recall at `MARGINS[m]` over would-be disk misses.
    pub fn pred_recall(&self, m: usize) -> f64 {
        pct(self.pred_caught[m], self.pred_disk_units)
    }

    /// Wasted prefetch bytes at `MARGINS[m]` as a share of demand disk bytes.
    pub fn pred_waste(&self, m: usize) -> f64 {
        pct(self.pred_wasted_bytes[m], self.pred_disk_bytes)
    }

    fn add(&mut self, o: Stats) {
        self.promoted += o.promoted;
        self.pred_disk_units += o.pred_disk_units;
        self.pred_disk_bytes += o.pred_disk_bytes;
        self.pred_unmatched += o.pred_unmatched;
        for m in 0..MARGINS.len() {
            self.pred_caught[m] += o.pred_caught[m];
            self.pred_wasted_bytes[m] += o.pred_wasted_bytes[m];
        }
        self.prefill_bytes += o.prefill_bytes;
        self.prefill_disk_bytes += o.prefill_disk_bytes;
        self.early_bytes += o.early_bytes;
        self.early_disk_bytes += o.early_disk_bytes;
        self.accesses += o.accesses;
        self.vram_hits += o.vram_hits;
        self.ram_hits += o.ram_hits;
        self.bytes += o.bytes;
        self.disk_bytes += o.disk_bytes;
    }
}

fn pct(n: u64, d: u64) -> f64 {
    if d == 0 {
        0.0
    } else {
        100.0 * n as f64 / d as f64
    }
}

pub struct Sim<'g> {
    geo: &'g Geo,
    exclusive: bool,
    ram: Tier,
    vram: Tier,
    tick: u64,
    prefill_ref: bool,
    /// Step offset so steps stay monotonic when caches carry across traces.
    base: u64,
    promote_budget: u32,
}

impl<'g> Sim<'g> {
    pub fn new(geo: &'g Geo, cfg: Config) -> Self {
        let e = geo.n_experts;
        let n_keys = (geo.n_layers() * e) as usize;
        let slots = class_slots(&geo.classes, cfg.pool_bytes);
        let ram_keys: Vec<Vec<u32>> = geo
            .classes
            .iter()
            .map(|(_, layers)| {
                layers
                    .iter()
                    .flat_map(|&l| (0..e).map(move |x| l * e + x))
                    .collect()
            })
            .collect();
        // MUL_MAT_ID wants one plane type per `as` tensor, so a shared arena
        // is per unit class; the routed layer's keys are leased, which keeps
        // its top-k floor.
        let (vram_keys, vram_cap) = if cfg.global_vram {
            let cap = geo
                .classes
                .iter()
                .map(|(_, layers)| layers.len() as u64 * u64::from(cfg.k))
                .collect();
            (ram_keys.clone(), cap)
        } else {
            (
                (0..geo.n_layers())
                    .map(|l| (0..e).map(|x| l * e + x).collect())
                    .collect(),
                vec![u64::from(cfg.k); geo.n_layers() as usize],
            )
        };
        Self {
            geo,
            exclusive: cfg.exclusive,
            ram: Tier::new(cfg.ram, n_keys, ram_keys, slots),
            vram: Tier::new(cfg.vram, n_keys, vram_keys, vram_cap),
            tick: 0,
            prefill_ref: cfg.prefill_ref,
            base: 0,
            promote_budget: cfg.promote,
        }
    }

    /// §6.3 promotion between steps (inclusive residency): the best-scored
    /// RAM-resident experts not in VRAM displace their layer's weakest slot
    /// when they beat it by the 20% swap margin, at most `promote` per step.
    /// The copy lands before the next step (≤ 8 × 15.2 MB ≈ 4.5 ms at 27
    /// GB/s), so a promoted slot serves that step.
    fn promote(&mut self, now: u64) -> u64 {
        const SWAP_MARGIN: f64 = 1.2;
        if self.promote_budget == 0 || self.exclusive {
            return 0;
        }
        self.vram.epoch += 1; // no routed set between steps: nothing leased
        let mut cand: Vec<(f64, u32)> = self
            .ram
            .members
            .iter()
            .flatten()
            .copied()
            .filter(|&k| !self.vram.resident(k))
            .map(|k| (self.vram.decayed(k, now), k))
            .filter(|&(s, _)| s > 0.0)
            .collect();
        cand.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let mut done = 0;
        for (score, k) in cand {
            if done == self.promote_budget {
                break;
            }
            let layer = self.vram.part[k as usize] as usize;
            let weakest = self.vram.members[layer].iter().copied().min_by(|&a, &b| {
                self.vram
                    .decayed(a, now)
                    .total_cmp(&self.vram.decayed(b, now))
            });
            if let Some(w) = weakest
                && self.vram.members[layer].len() as u64 >= self.vram.cap[layer]
            {
                if score < SWAP_MARGIN * self.vram.decayed(w, now) {
                    continue;
                }
                self.vram.remove(w);
            }
            self.vram.insert(k, now);
            done += 1;
        }
        u64::from(done)
    }

    /// Production refuses a pool with fewer slots in a class than its
    /// in-flight floor, layers × K + 1 (`p3_glue::class_layout`).
    pub fn below_floor(geo: &Geo, pool_bytes: u64, k: u32) -> bool {
        let slots = class_slots(&geo.classes, pool_bytes);
        geo.classes
            .iter()
            .zip(slots)
            .any(|((_, layers), s)| s < layers.len() as u64 * u64::from(k) + 1)
    }

    /// First-start pre-fill from a routing profile (per-key decode access
    /// counts, §6.3): the top K per layer into VRAM, then the top keys per
    /// class into RAM (exclusive: VRAM-resident keys skipped). Hottest last.
    pub fn prefill(&mut self, counts: &[u64]) {
        let fill = |t: &mut Tier, skip: &dyn Fn(u32) -> bool| {
            for p in 0..t.cap.len() {
                let mut keys: Vec<u32> = t.clock_keys[p]
                    .iter()
                    .copied()
                    .filter(|&k| counts[k as usize] > 0 && !skip(k))
                    .collect();
                keys.sort_by_key(|&k| {
                    std::cmp::Reverse((counts[k as usize], std::cmp::Reverse(k)))
                });
                keys.truncate(t.cap[p] as usize);
                for &k in keys.iter().rev() {
                    t.insert(k, 0);
                }
            }
            t.epoch += 1;
        };
        fill(&mut self.vram, &|_| false);
        let vram = &self.vram;
        let exclusive = self.exclusive;
        fill(&mut self.ram, &|k| exclusive && vram.resident(k));
    }

    /// [`Sim::prefill`], then pins the top `frac` of each RAM class's slots
    /// by `counts` (a long-horizon routing profile): never evicted, so the
    /// rest of the class runs the tier's policy on what is left (#120 O3).
    /// Inclusive residency, cold start only: an exclusive RAM hit would
    /// drop a pinned key.
    pub fn pin(&mut self, counts: &[u64], frac: f64) {
        self.prefill(counts);
        let t = &mut self.ram;
        for p in 0..t.cap.len() {
            let mut keys: Vec<u32> = t.clock_keys[p]
                .iter()
                .copied()
                .filter(|&k| counts[k as usize] > 0)
                .collect();
            keys.sort_by_key(|&k| std::cmp::Reverse((counts[k as usize], std::cmp::Reverse(k))));
            keys.truncate((t.cap[p] as f64 * frac) as usize);
            for k in keys {
                t.pinned[k as usize] = true;
            }
        }
        t.epoch += 1;
    }

    /// Books `layer`'s `pr` predictions against the units it routes that
    /// neither tier holds, before the layer's accesses move residency. The
    /// prefetch reads themselves are not simulated, so waste is a lower
    /// bound. Each record is cut at `k + margin`; with lookahead > 1 a layer
    /// is scored on the union of its records, as `protect` shields it.
    fn score_prediction(
        &self,
        c: &Compute,
        k: usize,
        layer: u32,
        ids: &[u16],
        bytes: u64,
        st: &mut Stats,
    ) {
        let e = self.geo.n_experts;
        let cold = |id: u16| {
            let key = layer * e + u32::from(id);
            !self.vram.resident(key) && !self.ram.resident(key)
        };
        let preds: Vec<&[u16]> = c
            .predicted
            .iter()
            .filter(|(l, _)| *l == layer)
            .map(|(_, p)| p.as_slice())
            .collect();
        if preds.is_empty() {
            return;
        }
        let named = |id: u16, m: usize| {
            preds
                .iter()
                .any(|p| p[..(k + m).min(p.len())].contains(&id))
        };
        for &id in ids.iter().filter(|&&id| cold(id)) {
            st.pred_disk_units += 1;
            st.pred_disk_bytes += bytes;
            for (i, &m) in MARGINS.iter().enumerate() {
                st.pred_caught[i] += u64::from(named(id, m));
            }
        }
        for (i, &m) in MARGINS.iter().enumerate() {
            let mut seen: Vec<u16> = Vec::new();
            for p in &preds {
                for &id in &p[..(k + m).min(p.len())] {
                    if !seen.contains(&id) && !ids.contains(&id) && cold(id) {
                        st.pred_wasted_bytes[i] += bytes;
                    }
                    seen.push(id);
                }
            }
        }
    }

    pub fn run(&mut self, trace: &Trace) -> Result<Stats, String> {
        let e = self.geo.n_experts;
        let mut st = Stats::default();
        let mut now = self.base;
        let mut n_decode = 0;
        for c in &trace.computes {
            let early = c.decode && n_decode < EARLY_STEPS;
            n_decode += usize::from(c.decode);
            self.ram.epoch += 1;
            self.tick += 1;
            now = self.base + c.step;
            for (layer, ids) in &c.layers {
                self.vram.epoch += 1;
                let bytes = self.geo.unit[*layer as usize];
                if c.decode {
                    let k = trace.top_k.unwrap_or(ids.len());
                    self.score_prediction(c, k, *layer, ids, bytes, &mut st);
                }
                for &id in ids {
                    let key = layer * e + u32::from(id);
                    self.vram.lease[key as usize] = self.vram.epoch;
                    if c.decode {
                        self.vram.observe(key, now, self.tick);
                        self.ram.observe(key, now, self.tick);
                    }
                }
                for &id in ids {
                    let key = layer * e + u32::from(id);
                    if c.decode {
                        st.accesses += 1;
                        st.bytes += bytes;
                    }
                    if early {
                        st.early_bytes += bytes;
                    }
                    if !c.decode {
                        st.prefill_bytes += bytes;
                    }
                    if self.vram.lookup(key, true) {
                        st.vram_hits += u64::from(c.decode);
                        continue;
                    }
                    let ram_hit = self.ram.lookup(key, c.decode || self.prefill_ref);
                    if ram_hit {
                        st.ram_hits += u64::from(c.decode);
                    } else {
                        if c.decode {
                            st.disk_bytes += bytes;
                        }
                        if early {
                            st.early_disk_bytes += bytes;
                        }
                        if !c.decode {
                            st.prefill_disk_bytes += bytes;
                        }
                        if !self.exclusive {
                            self.ram.admit(key, now)?;
                        }
                    }
                    let demoted = self.vram.admit(key, now)?;
                    if self.exclusive {
                        if ram_hit {
                            self.ram.remove(key);
                        }
                        if let Some(v) = demoted {
                            self.ram.admit_demoted(v, now);
                        }
                    }
                }
            }
            if c.decode {
                st.promoted += self.promote(now);
            }
            for (i, (l, _)) in c.predicted.iter().enumerate() {
                let ran = c.layers.iter().any(|(r, _)| r == l);
                let earlier = c.predicted[..i].iter().any(|(p, _)| p == l);
                st.pred_unmatched += u64::from(!ran && !earlier);
            }
        }
        self.base = now + 1;
        Ok(st)
    }
}

/// Runs `traces` under `cfg`, each from a cold cache (`warm`: caches carry
/// over from one trace to the next), optionally pre-filled per trace by
/// `profile(i)`. Returns pooled decode stats.
fn replay(
    geo: &Geo,
    traces: &[Trace],
    cfg: Config,
    warm: bool,
    profile: Option<&dyn Fn(usize) -> Vec<u64>>,
) -> Result<Stats, String> {
    replay_pinned(geo, traces, cfg, warm, profile, 0.0)
}

/// [`replay`] with the top `pin` of each RAM class pinned from `profile(i)`
/// (pre-filled, never a victim) instead of pre-filled.
fn replay_pinned(
    geo: &Geo,
    traces: &[Trace],
    cfg: Config,
    warm: bool,
    profile: Option<&dyn Fn(usize) -> Vec<u64>>,
    pin: f64,
) -> Result<Stats, String> {
    if pin > 0.0 {
        let floor_ok = |frac: f64| {
            !Sim::below_floor(geo, (cfg.pool_bytes as f64 * (1.0 - frac)) as u64, cfg.k)
        };
        if warm || cfg.exclusive || !floor_ok(pin) {
            return Err(format!(
                "pin {pin}: needs a cold, inclusive run and unpinned slots above the floor"
            ));
        }
    }
    let mut total = Stats::default();
    let mut sim = Sim::new(geo, cfg);
    for (i, t) in traces.iter().enumerate() {
        if !warm && i > 0 {
            sim = Sim::new(geo, cfg);
        }
        match profile {
            Some(p) if pin > 0.0 => sim.pin(&p(i), pin),
            Some(p) => sim.prefill(&p(i)),
            None => {}
        }
        total.add(sim.run(t).map_err(|e| format!("{}: {e}", t.name))?);
    }
    Ok(total)
}

/// Per-key decode access counts over `traces`, skipping index `skip`.
fn counts(geo: &Geo, traces: &[Trace], skip: usize) -> Vec<u64> {
    let mut out = vec![0u64; (geo.n_layers() * geo.n_experts) as usize];
    for (_, t) in traces.iter().enumerate().filter(|&(i, _)| i != skip) {
        for c in t.computes.iter().filter(|c| c.decode) {
            for (layer, ids) in &c.layers {
                for &id in ids {
                    out[(layer * geo.n_experts + u32::from(id)) as usize] += 1;
                }
            }
        }
    }
    out
}

fn gib(g: f64) -> u64 {
    (g * GIB) as u64
}

/// The full markdown report: policy comparison, capacity curve with d,
/// exclusive residency, routing-profile pre-fill and a VRAM K sweep. Pools
/// production refuses at this K (`Sim::below_floor`) are simulated and
/// marked †.
pub fn report(geo: &Geo, traces: &[Trace], k: u32) -> Result<String, String> {
    let cur = |pool: f64| Config {
        pool_bytes: gib(pool),
        k,
        ram: Policy::Clock,
        vram: Policy::Lru,
        exclusive: false,
        promote: 0,
        global_vram: false,
        prefill_ref: true,
    };
    let run = |cfg: Config, warm: bool| replay(geo, traces, cfg, warm, None);
    let mark = |pool: f64| {
        let dag = if Sim::below_floor(geo, gib(pool), k) {
            "†"
        } else {
            ""
        };
        format!("{pool:.1}{dag}")
    };
    let pct1 = |v: f64| format!("{v:.1}");
    let decay = Policy::Decay {
        min_dwell: MIN_DWELL,
    };
    let policies = [
        Policy::Clock,
        Policy::Lru,
        decay,
        Policy::Decay { min_dwell: 0 },
    ];
    let mut o = String::new();
    let total_gib = geo.total_bytes() as f64 / GIB;
    let classes: Vec<String> = geo
        .classes
        .iter()
        .map(|(u, l)| format!("{u} B × {}", l.len()))
        .collect();
    let _ = writeln!(
        o,
        "Geometry: {} expert layers × {} experts, {total_gib:.1} GiB of expert units; \
         RAM classes {}; VRAM K = {k} slots/layer. † = below the production pool floor \
         (layers × K + 1 slots per class) at this K: simulated, but `golem run` refuses it.\n",
        geo.n_layers(),
        geo.n_experts,
        classes.join(", ")
    );
    let _ = writeln!(
        o,
        "| trace | prefill computes | decode steps |\n|---|---|---|"
    );
    for t in traces {
        let d = t.decode_steps();
        let _ = writeln!(o, "| {} | {} | {d} |", t.name, t.computes.len() - d);
    }

    let _ = writeln!(
        o,
        "\n### Policy per tier, cold, pooled over traces\n\n\
         | pool GiB | residency | RAM policy | VRAM policy | VRAM hit % | RAM hit % of VRAM misses | d % |\n\
         |---|---|---|---|---|---|---|"
    );
    let (mut ram_gap, mut vram_gap): (f64, f64) = (0.0, 0.0);
    for pool in [10.0, 40.0, 63.0] {
        let base = run(cur(pool), false)?;
        let mut rows: Vec<(Policy, Policy)> = policies.iter().map(|&p| (p, Policy::Lru)).collect();
        rows.extend(
            policies
                .iter()
                .filter(|&&p| p != Policy::Lru)
                .map(|&p| (Policy::Clock, p)),
        );
        rows.push((decay, decay));
        let n_inclusive = rows.len();
        rows.extend([
            (Policy::Clock, Policy::Lru),
            (Policy::Clock, decay),
            (decay, decay),
        ]);
        for (i, (ram, vram)) in rows.into_iter().enumerate() {
            let exclusive = i >= n_inclusive;
            let r = run(
                Config {
                    ram,
                    vram,
                    exclusive,
                    ..cur(pool)
                },
                false,
            )?;
            if !exclusive && (ram, vram) == (decay, Policy::Lru) {
                ram_gap = ram_gap.max((r.ram_hit() - base.ram_hit()).abs());
            }
            if !exclusive && (ram, vram) == (Policy::Clock, decay) {
                vram_gap = vram_gap.max((r.vram_hit() - base.vram_hit()).abs());
            }
            let current = !exclusive && (ram, vram) == (Policy::Clock, Policy::Lru);
            let tag = if current { " (current)" } else { "" };
            let _ = writeln!(
                o,
                "| {} | {} | {} | {}{tag} | {} | {} | {} |",
                mark(pool),
                if exclusive { "exclusive" } else { "inclusive" },
                ram.label(),
                vram.label(),
                pct1(r.vram_hit()),
                pct1(r.ram_hit()),
                pct1(r.d())
            );
        }
    }
    let verdict = if ram_gap <= 1.0 && vram_gap <= 1.0 {
        "within ≈1 point at both tiers: keep CLOCK, drop the scorer"
    } else {
        "outside ≈1 point at a tier: keep the scorer"
    };
    let _ = writeln!(
        o,
        "\nDecayed score (D = {MIN_DWELL}) vs current, max over pools: |Δ| RAM hit \
         {ram_gap:.2} points, VRAM hit {vram_gap:.2} points → {verdict}."
    );

    let _ = writeln!(
        o,
        "\n### Background VRAM promotion (inclusive, RAM CLOCK, VRAM decay H={HALF_LIFE} \
         D={MIN_DWELL}, 20% swap margin), cold\n\n\
         | pool GiB | promotions/step cap | VRAM hit % | promotions/step | d % |\n|---|---|---|---|---|"
    );
    for pool in [40.0, 63.0] {
        for promote in [0, 8, 32] {
            let cfg = Config {
                vram: decay,
                promote,
                ..cur(pool)
            };
            let r = run(cfg, false)?;
            let steps: usize = traces.iter().map(Trace::decode_steps).sum();
            let _ = writeln!(
                o,
                "| {} | {promote} | {} | {:.2} | {} |",
                mark(pool),
                pct1(r.vram_hit()),
                r.promoted as f64 / steps.max(1) as f64,
                pct1(r.d())
            );
        }
    }

    let _ = writeln!(
        o,
        "\n### Capacity curve (current policies; warm = caches carried across traces)\n\n\
         | pool GiB | RAM hit % cold | d % cold | d % warm | d % exclusive cold | d % exclusive warm |\n\
         |---|---|---|---|---|---|"
    );
    let mut pools = vec![10.0, 20.0, 30.0, 40.0, 50.0, 63.0];
    pools.extend([63.0 + EXCLUSIVE_EXTRA_GIB, 90.0, 120.0, 150.0]);
    pools.push((total_gib * 10.0).ceil() / 10.0);
    for pool in pools {
        let x = Config {
            exclusive: true,
            ..cur(pool)
        };
        let c = run(cur(pool), false)?;
        let _ = writeln!(
            o,
            "| {} | {} | {} | {} | {} | {} |",
            mark(pool),
            pct1(c.ram_hit()),
            pct1(c.d()),
            pct1(run(cur(pool), true)?.d()),
            pct1(run(x, false)?.d()),
            pct1(run(x, true)?.d())
        );
    }

    let _ = writeln!(
        o,
        "\n### Routing-profile pre-fill (leave-one-out: each trace pre-filled from the \
         others' decode counts), cold; early = first {EARLY_STEPS} decode steps\n\n\
         | pool GiB | d % prefill none | d % prefill profile | d % early none | d % early profile | d % none | d % profile |\n\
         |---|---|---|---|---|---|---|"
    );
    let loo = |i: usize| counts(geo, traces, i);
    for pool in [10.0, 40.0, 63.0] {
        let a = run(cur(pool), false)?;
        let b = replay(geo, traces, cur(pool), false, Some(&loo))?;
        let _ = writeln!(
            o,
            "| {} | {} | {} | {} | {} | {} | {} |",
            mark(pool),
            pct1(a.d_prefill()),
            pct1(b.d_prefill()),
            pct1(a.d_early()),
            pct1(b.d_early()),
            pct1(a.d()),
            pct1(b.d())
        );
    }

    let _ = writeln!(
        o,
        "\n### Frequency-weighted RAM eviction and pinned hot set (#120 O3; inclusive, \
         VRAM LRU), cold. Profile = leave-one-out decode counts; pin = that share of each \
         class pinned from it, never evicted\n\n\
         | pool GiB | RAM policy | start | RAM hit % | d % early | d % |\n|---|---|---|---|---|---|"
    );
    let freq = [
        Policy::Decay { min_dwell: 0 },
        Policy::Freq { half_life: 4096 },
        Policy::Freq {
            half_life: u32::MAX,
        },
    ];
    // (RAM policy, start label, profile?, pinned share)
    let mut rows: Vec<(Policy, &str, bool, f64)> = vec![(Policy::Clock, "cold", false, 0.0)];
    rows.extend(freq.iter().map(|&p| (p, "cold", false, 0.0)));
    rows.push((Policy::Clock, "profile pre-fill", true, 0.0));
    for (frac, name) in [(0.25, "pin 25%"), (0.5, "pin 50%"), (0.75, "pin 75%")] {
        rows.push((Policy::Clock, name, true, frac));
    }
    rows.push((Policy::Freq { half_life: 4096 }, "pin 50%", true, 0.5));
    rows.push((Policy::Clock, "cold, prefill sets no ref bit", false, 0.0));
    for pool in [63.0, 78.0] {
        for &(ram, start, profiled, pin) in &rows {
            let profile = profiled.then_some(&loo as &dyn Fn(usize) -> Vec<u64>);
            let r = replay_pinned(
                geo,
                traces,
                Config {
                    ram,
                    prefill_ref: !start.contains("no ref bit"),
                    ..cur(pool)
                },
                false,
                profile,
                pin,
            )?;
            let _ = writeln!(
                o,
                "| {} | {} | {start} | {} | {} | {} |",
                mark(pool),
                ram.label(),
                pct1(r.ram_hit()),
                pct1(r.d_early()),
                pct1(r.d())
            );
        }
    }

    let _ = writeln!(
        o,
        "\n### VRAM slots per layer (RAM: CLOCK 63 GiB), cold\n\n\
         | K | VRAM hit % | d % |\n|---|---|---|"
    );
    for kk in [k, 2 * k, 4 * k] {
        let r = run(Config { k: kk, ..cur(63.0) }, false)?;
        let _ = writeln!(o, "| {kk} | {} | {} |", pct1(r.vram_hit()), pct1(r.d()));
    }

    let _ = writeln!(
        o,
        "\n### VRAM arena: K per layer vs one arena per unit class at equal slots \
         (63 GiB), cold\n\n\
         | residency | RAM policy | VRAM policy | per-layer VRAM hit % | arena VRAM hit % | per-layer d % | arena d % |\n\
         |---|---|---|---|---|---|---|"
    );
    for (exclusive, ram, vram) in [
        (false, Policy::Clock, Policy::Clock),
        (false, Policy::Clock, Policy::Lru),
        (false, Policy::Clock, decay),
        (true, decay, decay),
    ] {
        let base = Config {
            ram,
            vram,
            exclusive,
            ..cur(63.0)
        };
        let a = run(base, false)?;
        let b = run(
            Config {
                global_vram: true,
                ..base
            },
            false,
        )?;
        let _ = writeln!(
            o,
            "| {} | {} | {} | {} | {} | {} | {} |",
            if exclusive { "exclusive" } else { "inclusive" },
            ram.label(),
            vram.label(),
            pct1(a.vram_hit()),
            pct1(b.vram_hit()),
            pct1(a.d()),
            pct1(b.d())
        );
    }

    let pred_cfg = Config {
        vram: decay,
        ..cur(63.0)
    };
    let p = run(pred_cfg, false)?;
    let _ = writeln!(
        o,
        "\n### P5 prediction vs would-be disk misses (RAM CLOCK, VRAM {}, 63 GiB), cold\n",
        decay.label()
    );
    if p.pred_disk_units == 0 {
        let _ = writeln!(
            o,
            "No `pr` records in these traces: capture with `--prefetch protect --routing-trace`."
        );
    } else {
        let _ = writeln!(
            o,
            "{} would-be disk-miss units in predicted layers ({:.2} GB); {} `pr` \
             records for a layer that did not run in the same compute. With \
             lookahead > 1 a layer is scored on the union of its records. Waste \
             counts named units neither tier held and the layer did not route \
             (lower bound: prefetch-induced evictions are not simulated).\n\n\
             | margin | ids per record | recall of disk misses % | wasted bytes % of demand disk bytes |\n\
             |---|---|---|---|",
            p.pred_disk_units,
            p.pred_disk_bytes as f64 / 1e9,
            p.pred_unmatched
        );
        for (i, m) in MARGINS.iter().enumerate() {
            let _ = writeln!(
                o,
                "| {m} | top-k + {m} | {} | {} |",
                pct1(p.pred_recall(i)),
                pct1(p.pred_waste(i))
            );
        }
    }
    Ok(o)
}

/// `xtask replay --model <shard-1.gguf> --k <slots> <trace.tsv>...`
pub fn run(args: Vec<String>) {
    let die = |m: String| -> ! {
        eprintln!("replay: {m}");
        std::process::exit(2);
    };
    let mut model = None;
    let mut k = None;
    let mut paths = Vec::new();
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--model" => model = it.next(),
            "--k" => {
                let v = it.next().unwrap_or_default();
                k = Some(
                    v.parse::<u32>()
                        .unwrap_or_else(|e| die(format!("--k {v:?}: {e}"))),
                );
            }
            f if f.starts_with("--") => die(format!("unknown flag {f}")),
            _ => paths.push(a),
        }
    }
    let model = model.unwrap_or_else(|| die("--model <shard-1.gguf> is required".into()));
    // The trace run's resolved K (`golem run` prints it): a smaller K cannot
    // hold that run's prefill chunks, a larger one overstates VRAM hits.
    let k = k.unwrap_or_else(|| die("--k <slots per layer> is required".into()));
    if !(1..=1024).contains(&k) {
        die(format!("--k {k} outside 1..=1024"));
    }
    if paths.is_empty() {
        die("no trace files".into());
    }
    let geo = Geo::from_gguf(Path::new(&model)).unwrap_or_else(|e| die(e));
    let lead = lead_of(Path::new(&model)).unwrap_or_else(|e| die(e));
    let traces: Vec<Trace> = paths
        .iter()
        .map(|p| {
            let text = std::fs::read_to_string(p).unwrap_or_else(|e| die(format!("{p}: {e}")));
            let name = Path::new(p)
                .file_name()
                .map_or(p.clone(), |n| n.to_string_lossy().into_owned());
            Trace::parse(&name, &text, lead, &geo).unwrap_or_else(|e| die(e))
        })
        .collect();
    print!("{}", report(&geo, &traces, k).unwrap_or_else(|e| die(e)));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One layer of `e` experts, unit 1 MiB.
    fn geo(e: u32) -> Geo {
        Geo {
            unit: vec![1 << 20],
            n_experts: e,
            classes: vec![(1 << 20, vec![0])],
        }
    }

    /// Decode computes, one routed id each, step = index.
    fn trace(ids: &[u16]) -> Trace {
        Trace {
            name: "t".into(),
            computes: ids
                .iter()
                .enumerate()
                .map(|(s, &id)| Compute {
                    decode: true,
                    step: s as u64,
                    layers: vec![(0, vec![id])],
                    predicted: Vec::new(),
                })
                .collect(),
            top_k: None,
        }
    }

    fn cfg(slots: u64, k: u32, ram: Policy) -> Config {
        Config {
            pool_bytes: slots << 20,
            k,
            ram,
            vram: Policy::Lru,
            exclusive: false,
            promote: 0,
            global_vram: false,
            prefill_ref: true,
        }
    }

    #[test]
    fn pinned_keys_are_never_victims() {
        let g = geo(5);
        // Key 0 is profiled hot but never routed; a scan of 1..4 through a
        // 3-slot pool evicts it unless it is pinned.
        let seq = [1u16, 2, 3, 4, 1, 2, 3, 4];
        let mut counts = vec![0u64; 5];
        counts[0] = 9;
        let run = |pin: f64| {
            let mut sim = Sim::new(&g, cfg(3, 1, Policy::Clock));
            if pin > 0.0 {
                sim.pin(&counts, pin);
            } else {
                sim.prefill(&counts);
            }
            sim.run(&trace(&seq)).unwrap();
            sim.ram.resident(0)
        };
        assert!(!run(0.0), "unpinned pre-fill is evicted by the scan");
        assert!(run(0.34), "pinned key survives the scan");
    }

    #[test]
    fn prefill_ref_off_leaves_prefill_hits_unreferenced() {
        let g = geo(3);
        // VRAM K = 1, RAM 2 slots: 0, 1 fill RAM; the prefill re-routes 0
        // (VRAM miss, RAM hit); admitting 2 then evicts 0 unless that hit
        // set its CLOCK bit.
        let mut t = trace(&[0, 1, 0, 2]);
        t.computes[2].decode = false;
        let kept = |prefill_ref: bool| {
            let mut sim = Sim::new(
                &g,
                Config {
                    prefill_ref,
                    ..cfg(2, 1, Policy::Clock)
                },
            );
            sim.run(&t).unwrap();
            sim.ram.resident(0)
        };
        assert!(kept(true));
        assert!(!kept(false));
    }

    #[test]
    fn lfu_keeps_the_most_used_key_where_lru_drops_it() {
        let g = geo(3);
        // 0 is hot early; a scan of 1, 2 would evict it under LRU.
        let seq = [0u16, 0, 0, 1, 2, 0];
        let lfu = replay(
            &g,
            &[trace(&seq)],
            cfg(
                2,
                1,
                Policy::Freq {
                    half_life: u32::MAX,
                },
            ),
            false,
            None,
        )
        .unwrap();
        let lru = replay(&g, &[trace(&seq)], cfg(2, 1, Policy::Lru), false, None).unwrap();
        assert!(
            lfu.ram_hits > lru.ram_hits,
            "{} vs {}",
            lfu.ram_hits,
            lru.ram_hits
        );
    }

    #[test]
    fn parse_groups_computes_and_phases() {
        let g = Geo {
            unit: vec![1, 1],
            n_experts: 8,
            classes: vec![(1, vec![0, 1])],
        };
        let text = "# golem-routing-trace v1 model=x layers=2 top_k=2\n\
                    0\tpf\t3\t1,2\n0\tpf\t4\t3,4\n0\tpf\t3\t5,6\n0\tpf\t4\t7,0\n\
                    0\tdc\t3\t1,2\n0\tdc\t4\t1,2\n1\tdc\t3\t1,2\n";
        let t = Trace::parse("t", text, 3, &g).unwrap();
        let shape: Vec<(bool, u64, usize)> = t
            .computes
            .iter()
            .map(|c| (c.decode, c.step, c.layers.len()))
            .collect();
        assert_eq!(
            shape,
            [(false, 0, 2), (false, 0, 2), (true, 0, 2), (true, 1, 1)]
        );
        assert_eq!(
            t.computes[1].layers[1],
            (1, vec![0, 7]),
            "sorted, deduplicated set"
        );
        assert!(Trace::parse("t", "# golem-routing-trace v1\n0\tdc\t9\t1\n", 3, &g).is_err());
        assert!(Trace::parse("t", "# golem-routing-trace v1\n0\tdc\t3\t8\n", 3, &g).is_err());
    }

    #[test]
    fn a_pool_holding_every_key_reads_each_key_from_disk_once() {
        let g = geo(16);
        let ids: Vec<u16> = (0..64).map(|i| (i * 7 % 16) as u16).collect();
        let st = replay(&g, &[trace(&ids)], cfg(32, 1, Policy::Clock), false, None).unwrap();
        assert_eq!(st.disk_bytes, 16 << 20);
        assert_eq!(st.bytes, 64 << 20);
    }

    #[test]
    fn both_evict_the_unreused_key_when_the_reused_one_is_oldest_filled() {
        // Pool 3, K = 1 (VRAM holds only the last id). A B C A D B: at D the
        // pool is {A, B, C} with A reused. LRU stamps A=4, B=2, C=3 → evicts
        // B; CLOCK's hand clears A's reference bit, then evicts B. B misses.
        let g = geo(8);
        let seq = [0, 1, 2, 0, 3, 1];
        for p in [Policy::Lru, Policy::Clock] {
            let st = replay(&g, &[trace(&seq)], cfg(3, 1, p), false, None).unwrap();
            assert_eq!(st.disk_bytes, 5 << 20, "{p:?}");
        }
    }

    #[test]
    fn clock_and_lru_diverge_where_first_fills_carry_no_reference_bit() {
        // Pool 3, K = 1. A B A C D: both evict B at D (A reused). At F, LRU
        // evicts A (stamps A=3, C=4, D=5) but CLOCK's hand sits on C, filled
        // once with no reference bit → evicts C. The final C: LRU hit, CLOCK miss.
        let g = geo(8);
        let seq = [0, 1, 0, 2, 3, 5, 2];
        let lru = replay(&g, &[trace(&seq)], cfg(3, 1, Policy::Lru), false, None).unwrap();
        let clock = replay(&g, &[trace(&seq)], cfg(3, 1, Policy::Clock), false, None).unwrap();
        assert_eq!(lru.disk_bytes, 5 << 20, "LRU keeps C");
        assert_eq!(clock.disk_bytes, 6 << 20, "CLOCK evicted C");
    }

    #[test]
    fn a_pool_at_the_in_flight_floor_never_runs_out_of_victims() {
        // Two layers in one class, top-k 2 per layer, K = 2: floor = 2·2 + 1 = 5
        // slots. One compute routes 4 distinct units; all 4 must coexist.
        let g = Geo {
            unit: vec![1 << 20, 1 << 20],
            n_experts: 8,
            classes: vec![(1 << 20, vec![0, 1])],
        };
        let c = |s, a: [u16; 2], b: [u16; 2]| Compute {
            decode: true,
            step: s,
            layers: vec![(0, a.to_vec()), (1, b.to_vec())],
            predicted: Vec::new(),
        };
        let t = Trace {
            name: "t".into(),
            computes: vec![
                c(0, [0, 1], [0, 1]),
                c(1, [2, 3], [2, 3]),
                c(2, [4, 5], [4, 5]),
            ],
            top_k: None,
        };
        let st = replay(&g, &[t], cfg(5, 2, Policy::Clock), false, None).unwrap();
        assert_eq!(st.disk_bytes, 12 << 20);
        assert!(Sim::below_floor(&g, 4 << 20, 2), "4 slots < floor 5");
        assert!(!Sim::below_floor(&g, 5 << 20, 2));
    }

    #[test]
    fn exclusive_residency_adds_vram_capacity_to_the_pool() {
        // 4 RAM slots + K = 2 VRAM: a 6-key cycle fits only exclusively.
        let g = geo(8);
        let ids: Vec<u16> = (0..60).map(|i| (i % 6) as u16).collect();
        let t = [trace(&ids)];
        let base = cfg(4, 2, Policy::Lru);
        let inc = replay(&g, &t, base, false, None).unwrap();
        let exc = replay(
            &g,
            &t,
            Config {
                exclusive: true,
                ..base
            },
            false,
            None,
        )
        .unwrap();
        assert_eq!(exc.disk_bytes, 6 << 20, "only compulsory misses");
        assert!(
            inc.disk_bytes > 30 << 20,
            "inclusive LRU thrashes a 6-cycle in 4 slots"
        );
    }

    #[test]
    fn profile_prefill_turns_first_touches_into_hits() {
        let g = geo(8);
        let t = [trace(&[0, 1, 2, 0, 1, 2]), trace(&[0, 1, 2, 0, 1, 2])];
        let none = replay(&g, &t, cfg(4, 1, Policy::Clock), false, None).unwrap();
        let loo = |i: usize| counts(&g, &t, i);
        let prof = replay(&g, &t, cfg(4, 1, Policy::Clock), false, Some(&loo)).unwrap();
        assert_eq!(none.disk_bytes, 6 << 20);
        assert_eq!(prof.disk_bytes, 0);
    }

    #[test]
    fn promotion_brings_the_hot_key_back_before_its_next_use() {
        // K = 1, pool 3. A A A A (A scores ≈ 4), B takes the only VRAM slot;
        // between steps A (≈ 4) beats B (≈ 1) by the margin and is promoted,
        // so the final A is a VRAM hit instead of a RAM hit.
        let g = geo(8);
        let seq = [0, 0, 0, 0, 1, 0];
        let base = Config {
            vram: Policy::Decay { min_dwell: 0 },
            ..cfg(3, 1, Policy::Clock)
        };
        let off = replay(&g, &[trace(&seq)], base, false, None).unwrap();
        let on = replay(
            &g,
            &[trace(&seq)],
            Config { promote: 8, ..base },
            false,
            None,
        )
        .unwrap();
        assert_eq!(on.vram_hits, off.vram_hits + 1);
        assert_eq!(on.promoted, 1);
        assert_eq!(
            on.disk_bytes, off.disk_bytes,
            "promotion moves no disk bytes"
        );
    }

    #[test]
    fn decayed_score_keeps_the_frequent_key() {
        // Pool 3, K = 1. A is hit four times, then B, C, D stream once each.
        // At D, LRU evicts A (oldest stamp); decay (D = 0) evicts B (score ≈ 1
        // vs A's ≈ 4), so the final A is a RAM hit.
        let g = geo(8);
        let seq = [0, 0, 0, 0, 1, 2, 3, 0];
        let decay = replay(
            &g,
            &[trace(&seq)],
            cfg(3, 1, Policy::Decay { min_dwell: 0 }),
            false,
            None,
        )
        .unwrap();
        let lru = replay(&g, &[trace(&seq)], cfg(3, 1, Policy::Lru), false, None).unwrap();
        assert_eq!(decay.disk_bytes, 4 << 20, "A survives B, C, D");
        assert_eq!(lru.disk_bytes, 5 << 20, "LRU evicted A at D");
    }

    #[test]
    fn pr_rows_attach_to_their_decode_step_in_score_order() {
        let g = Geo {
            unit: vec![1, 1],
            n_experts: 8,
            classes: vec![(1, vec![0, 1])],
        };
        let text = "# golem-routing-trace v1 model=x layers=2 top_k=1\n\
                    0\tpf\t3\t1\n0\tdc\t3\t1\n0\tpr\t4\t5,2,7\n0\tdc\t4\t2\n\
                    1\tpr\t3\t6,1\n1\tdc\t3\t6\n";
        let t = Trace::parse("t", text, 3, &g).unwrap();
        let shape: Vec<(bool, u64, usize, usize)> = t
            .computes
            .iter()
            .map(|c| (c.decode, c.step, c.layers.len(), c.predicted.len()))
            .collect();
        assert_eq!(shape, [(false, 0, 1, 0), (true, 0, 2, 1), (true, 1, 1, 1)]);
        assert_eq!(t.computes[1].predicted[0], (1, vec![5, 2, 7]), "unsorted");
        assert_eq!(t.top_k, Some(1));
    }

    #[test]
    fn a_pr_row_before_its_steps_first_dc_row_waits_for_that_compute() {
        // Lead 0: blk 0 is the first expert layer, so no prev_blk reset may
        // split the step.
        let g = Geo {
            unit: vec![1, 1],
            n_experts: 8,
            classes: vec![(1, vec![0, 1])],
        };
        let text = "# golem-routing-trace v1 model=x layers=2 top_k=1\n\
                    0\tpr\t1\t3\n0\tdc\t0\t1\n0\tdc\t1\t3\n";
        let t = Trace::parse("t", text, 0, &g).unwrap();
        assert_eq!(t.computes.len(), 1);
        assert_eq!(t.computes[0].layers.len(), 2);
        assert_eq!(t.computes[0].predicted, [(1, vec![3])]);
    }

    #[test]
    fn prediction_scores_only_would_be_disk_misses() {
        // Top-1. Step 0 routes 0 (cold, unpredicted). Step 1 predicts [1, 2, 0]
        // and routes 1: 0 is resident, so only 1 is a disk miss; margin 0 names
        // 1 (caught, no waste), margin 2 also names 2 (cold, unrouted: waste)
        // and 0 (resident: no waste).
        let g = geo(8);
        let mut t = trace(&[0, 1]);
        t.computes[1].predicted = vec![(0, vec![1, 2, 0])];
        let st = replay(&g, &[t], cfg(4, 1, Policy::Clock), false, None).unwrap();
        assert_eq!((st.pred_disk_units, st.pred_disk_bytes), (1, 1 << 20));
        assert_eq!(st.pred_caught, [1, 1, 1]);
        assert_eq!(st.pred_wasted_bytes, [0, 1 << 20, 1 << 20]);
        assert_eq!(st.pred_unmatched, 0);
    }

    #[test]
    fn lookahead_records_score_as_one_union_and_cut_at_top_k() {
        // top_k 1 from the header, two records for one layer naming 2 twice:
        // margin 0 names {1, 2} (2 wasted once); a stray record for layer 3
        // never runs and is counted unmatched.
        let g = geo(8);
        let mut t = trace(&[0, 1]);
        t.top_k = Some(1);
        t.computes[1].predicted = vec![(0, vec![1, 2]), (0, vec![2, 3]), (3, vec![4])];
        let st = replay(&g, &[t], cfg(4, 1, Policy::Clock), false, None).unwrap();
        assert_eq!(st.pred_caught[0], 1);
        assert_eq!(st.pred_wasted_bytes[0], 1 << 20, "2 counted once");
        assert_eq!(st.pred_wasted_bytes[1], 2 << 20, "margin 2 adds 3");
        assert_eq!(st.pred_unmatched, 1);
    }

    #[test]
    fn a_global_arena_lends_an_idle_layer_its_slots() {
        // Two layers in one class, K = 1 each. Layer 1 is routed once, then
        // layer 0 alternates two experts: the shared 2-slot arena gives layer
        // 1's idle slot to layer 0; per-layer K thrashes.
        let g = Geo {
            unit: vec![1 << 20, 1 << 20],
            n_experts: 8,
            classes: vec![(1 << 20, vec![0, 1])],
        };
        let mut computes = vec![Compute {
            decode: true,
            step: 0,
            layers: vec![(1, vec![7])],
            predicted: Vec::new(),
        }];
        computes.extend((1..13).map(|s| Compute {
            decode: true,
            step: s,
            layers: vec![(0, vec![(s % 2) as u16])],
            predicted: Vec::new(),
        }));
        let t = [Trace {
            name: "t".into(),
            computes,
            top_k: None,
        }];
        let base = cfg(8, 1, Policy::Clock);
        let per_layer = replay(&g, &t, base, false, None).unwrap();
        let arena = replay(
            &g,
            &t,
            Config {
                global_vram: true,
                ..base
            },
            false,
            None,
        )
        .unwrap();
        assert_eq!(per_layer.vram_hits, 0, "K = 1 thrashes a 2-cycle");
        assert_eq!(arena.vram_hits, 10, "two first touches, then all hits");
    }
}
