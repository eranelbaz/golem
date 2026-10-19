//! Per-machine scheduling profiles (#81): the hardware/model fingerprint,
//! the tunable knobs, the sweep that picks them, and the profile file
//! (`$XDG_CONFIG_HOME/golem/tuning/<fingerprint>.json`). Quality knobs
//! (quant, top-k, sampling, draft, router caching) have no field in
//! [`Knobs`], so no sweep can trade output quality for speed.

use std::error::Error;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Hex sha256 over the canonical [`FingerprintInput`] JSON.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fingerprint(pub String);

/// What a profile is valid for; any change re-keys it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FingerprintInput {
    /// Physical cores, SMT siblings, L3 (CCD) domains, NUMA nodes.
    pub cpu: String,
    /// Per GPU: name, arch, VRAM bytes.
    pub gpu: Vec<String>,
    /// Per shard: file name, size, mtime (Unix seconds).
    pub model_shards: Vec<(String, u64, u64)>,
    /// `CARGO_PKG_VERSION` + vendor pin hash.
    pub engine: String,
}

/// Pure hash of `i`.
pub fn fingerprint_of(i: &FingerprintInput) -> Result<Fingerprint, serde_json::Error> {
    let json = serde_json::to_vec(i)?;
    Ok(Fingerprint(
        Sha256::digest(&json)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect(),
    ))
}

/// Fingerprint of this machine + `model` + this build.
pub fn fingerprint(model: &Path) -> Result<Fingerprint, Box<dyn Error>> {
    Ok(fingerprint_of(&probe(model)?)?)
}

fn read(p: impl AsRef<Path>) -> String {
    std::fs::read_to_string(p)
        .map(|s| s.trim().to_owned())
        .unwrap_or_default()
}

/// Reads the fingerprint inputs from sysfs, the model shards and the build.
pub fn probe(model: &Path) -> Result<FingerprintInput, Box<dyn Error>> {
    let sys = sysinfo::System::new_all();
    let phys = sys.physical_core_count().unwrap_or(0);
    let logical = std::thread::available_parallelism().map_or(0, |n| n.get());
    let mut l3: Vec<String> = Vec::new();
    for c in 0..logical {
        let d = format!("/sys/devices/system/cpu/cpu{c}/cache/index3");
        let dom = format!(
            "{}@{}",
            read(format!("{d}/size")),
            read(format!("{d}/shared_cpu_list"))
        );
        if !l3.contains(&dom) {
            l3.push(dom);
        }
    }
    let cpu = format!(
        "physical={phys} smt={} l3=[{}] numa={}",
        logical.checked_div(phys).unwrap_or(0),
        l3.join(" "),
        read("/sys/devices/system/node/online")
    );
    let mut gpu = Vec::new();
    if let Ok(rd) = std::fs::read_dir("/sys/class/kfd/kfd/topology/nodes") {
        let mut nodes: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
        nodes.sort();
        for n in nodes {
            let props = read(n.join("properties"));
            let prop = |k: &str| {
                props
                    .lines()
                    .find_map(|l| l.strip_prefix(k)?.strip_prefix(' '))
                    .unwrap_or("0")
                    .to_owned()
            };
            if prop("simd_count") == "0" {
                continue; // CPU agent
            }
            let vram = read(n.join("mem_banks/0/properties"))
                .lines()
                .find_map(|l| l.strip_prefix("size_in_bytes ").map(str::to_owned))
                .unwrap_or_default();
            gpu.push(format!(
                "{} gfx={} vram={vram}",
                read(n.join("name")),
                prop("gfx_target_version")
            ));
        }
    }
    let model_shards = golem_gguf::shard_stamps(model)?;
    const VENDOR_SHA: &str = include_str!("../../../vendor/VENDOR_SHA");
    let pin = VENDOR_SHA
        .lines()
        .find_map(|l| l.strip_prefix("content_sha256 = "))
        .unwrap_or("unknown");
    Ok(FingerprintInput {
        cpu,
        gpu,
        model_shards,
        engine: format!("{} pin={pin}", env!("CARGO_PKG_VERSION")),
    })
}

/// Scheduling knobs a profile may set. `Default` (all zero/false) is
/// today's behaviour: 0 means "the engine default" for every numeric knob.
#[derive(Debug, Default, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Knobs {
    /// CPU threadpool; 0 = R-15 default.
    pub threads: u32,
    /// Interleave the RAM pool across NUMA nodes.
    pub numa_interleave: bool,
    /// Store reads must go `O_DIRECT`.
    pub o_direct: bool,
    /// Store reader threads per drain; 0 = derived per drive.
    pub io_threads: usize,
    /// Derived reader threads per drive; 0 = 1.
    pub io_depth: u32,
    /// Pooled H2D on a dedicated copy stream.
    pub async_h2d: bool,
    /// Share of the planner's device-expert-cache allowance (1.0, 0.75,
    /// 0.5); 0 = 1.0.
    pub vram_experts_frac: f32,
    /// Decode-only CPU/GPU hybrid expert math.
    #[serde(default)]
    pub cpu_expert_math: bool,
    /// Share of slab misses sent to the CPU (1.0, 0.7, 0.5); 0 = 0.7.
    #[serde(default)]
    pub cpu_expert_frac: f32,
}

/// One sweep dimension: the values coordinate descent tries for one knob.
#[derive(Debug, Clone)]
pub enum Axis {
    /// [`Knobs::threads`].
    Threads(Vec<u32>),
    /// [`Knobs::numa_interleave`].
    NumaInterleave(Vec<bool>),
    /// [`Knobs::o_direct`].
    ODirect(Vec<bool>),
    /// [`Knobs::io_threads`].
    IoThreads(Vec<usize>),
    /// [`Knobs::io_depth`].
    IoDepth(Vec<u32>),
    /// [`Knobs::vram_experts_frac`].
    VramExpertsFrac(Vec<f32>),
    /// [`Knobs::async_h2d`].
    AsyncH2d(Vec<bool>),
    /// [`Knobs::cpu_expert_math`].
    CpuExpertMath(Vec<bool>),
    /// [`Knobs::cpu_expert_frac`].
    CpuExpertFrac(Vec<f32>),
}

impl Axis {
    /// `base` with this axis set to each of its values.
    pub fn variants(&self, base: &Knobs) -> Vec<Knobs> {
        fn each<T: Copy>(v: &[T], set: impl Fn(&mut Knobs, T), base: &Knobs) -> Vec<Knobs> {
            v.iter()
                .map(|&x| {
                    let mut k = *base;
                    set(&mut k, x);
                    k
                })
                .collect()
        }
        match self {
            Axis::Threads(v) => each(v, |k, x| k.threads = x, base),
            Axis::NumaInterleave(v) => each(v, |k, x| k.numa_interleave = x, base),
            Axis::ODirect(v) => each(v, |k, x| k.o_direct = x, base),
            Axis::IoThreads(v) => each(v, |k, x| k.io_threads = x, base),
            Axis::IoDepth(v) => each(v, |k, x| k.io_depth = x, base),
            Axis::VramExpertsFrac(v) => each(v, |k, x| k.vram_experts_frac = x, base),
            Axis::AsyncH2d(v) => each(v, |k, x| k.async_h2d = x, base),
            Axis::CpuExpertMath(v) => each(v, |k, x| k.cpu_expert_math = x, base),
            Axis::CpuExpertFrac(v) => each(v, |k, x| k.cpu_expert_frac = x, base),
        }
    }
}

/// Warm steady-state measurement of one configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    /// Median decode tokens/s.
    pub tok_s: f64,
    /// RAM-pool hit rate over the measured turns (0..1).
    pub hit_rate: f64,
    /// Time to first token, median.
    pub ttft_p50_ms: f64,
    /// Time to first token, p99.
    pub ttft_p99_ms: f64,
    /// Greedy `(token, top1 − top2 logit gap)` per generation step, one
    /// row per prompt of the fixed set (§12.3 parity, gated by [`greedy_matches`]).
    #[serde(default)]
    pub greedy: Vec<Vec<(i32, f32)>>,
}

/// Largest top-1/top-2 logit gap at which two settings may pick different
/// greedy tokens: the §12.3 "< 0.2" near-tie bucket. Prefill chunk shape
/// (K / top_k tokens, #113; snapshot delta prefill, #84) reorders GPU
/// reductions → logits shift by ≤ 0.47 on the target; measured divergences
/// sat at gaps 0.10–0.15.
pub const NEAR_TIE_GAP: f32 = 0.2;

/// Greedy output held: per prompt, the token streams agree up to their
/// first difference, and there both sides were near-ties (what follows a
/// near-tie flip is unchecked). An empty baseline row never matches.
pub fn greedy_matches(c: &Metrics, base: &Metrics) -> bool {
    !base.greedy.is_empty()
        && base.greedy.iter().all(|b| !b.is_empty())
        && c.greedy.len() == base.greedy.len()
        && c.greedy.iter().zip(&base.greedy).all(|(c, b)| {
            match c.iter().zip(b).find(|(x, y)| x.0 != y.0) {
                Some((x, y)) => x.1 < NEAR_TIE_GAP && y.1 < NEAR_TIE_GAP,
                None => c.len() == b.len(),
            }
        })
}

/// A saved sweep result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Profile {
    /// [`Fingerprint`] hex.
    pub fingerprint: String,
    /// Free VRAM when the sweep started.
    pub measured_free_vram: u64,
    /// Available RAM when the sweep started.
    pub measured_free_ram: u64,
    /// The winner; `None` = nothing beat the baseline (a baseline-only
    /// record, never loaded by `run`/`serve`).
    pub knobs: Option<Knobs>,
    /// Winner's confirmation metrics (baseline's when no winner).
    pub metrics: Metrics,
    /// Baseline metrics.
    pub baseline: Metrics,
    /// Every measured candidate; `Err` = disqualified (reason).
    pub candidates: Vec<(Knobs, Result<Metrics, String>)>,
}

impl Profile {
    /// Baseline-only record with zeroed metrics.
    #[cfg(test)]
    #[allow(dead_code)] // used by tests/optimize.rs, which compiles this file
    pub fn test_default() -> Self {
        let m = Metrics {
            tok_s: 0.0,
            hit_rate: 0.0,
            ttft_p50_ms: 0.0,
            ttft_p99_ms: 0.0,
            greedy: Vec::new(),
        };
        Self {
            fingerprint: String::new(),
            measured_free_vram: 0,
            measured_free_ram: 0,
            knobs: None,
            metrics: m.clone(),
            baseline: m,
            candidates: Vec::new(),
        }
    }
}

/// Reading noise tolerated between the sweep's free-memory snapshot and
/// now (≈ 1% of the target's 24 GiB VRAM).
pub const SLACK: u64 = 256 << 20;

/// A profile measured with more free memory than exists now (beyond
/// [`SLACK`]) would plan a pool or device cache that no longer fits.
/// `free_vram` `None` (no sysfs reading) skips the VRAM comparison rather
/// than counting as 0 free.
pub fn applicable(p: &Profile, free_vram: Option<u64>, free_ram: u64) -> bool {
    free_vram.is_none_or(|v| v + SLACK >= p.measured_free_vram)
        && free_ram + SLACK >= p.measured_free_ram
}

/// `(free, total)` VRAM of the first amdgpu card from sysfs — no HIP
/// runtime in the asking process, so its own context never skews the
/// reading. `None` without one.
pub fn sysfs_vram() -> Option<(u64, u64)> {
    for e in std::fs::read_dir("/sys/class/drm").ok()?.flatten() {
        let d = e.path().join("device");
        let num = |f: &str| read(d.join(f)).parse::<u64>().ok();
        if let (Some(total), Some(used)) = (num("mem_info_vram_total"), num("mem_info_vram_used")) {
            return Some((total.saturating_sub(used), total));
        }
    }
    None
}

/// `$XDG_CONFIG_HOME/golem/tuning` (`~/.config` when unset).
pub fn tuning_dir() -> Option<PathBuf> {
    crate::config::config_home(&|k| std::env::var(k).ok()).map(|d| d.join("tuning"))
}

/// The profile file for `fp`.
pub fn profile_path(fp: &Fingerprint) -> Option<PathBuf> {
    tuning_dir().map(|d| d.join(format!("{}.json", fp.0)))
}

/// The saved profile for `fp` when it carries a winner. An unreadable or
/// corrupt file is warned about and ignored.
pub fn load_profile(fp: &Fingerprint) -> Option<Profile> {
    let path = profile_path(fp)?;
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => {
            eprintln!("golem: warning: optimize profile {}: {e}", path.display());
            return None;
        }
    };
    let p: Profile = match serde_json::from_slice(&bytes) {
        Ok(p) => p,
        Err(e) => {
            eprintln!(
                "golem: warning: optimize profile {} corrupt, ignored: {e}",
                path.display()
            );
            return None;
        }
    };
    (p.fingerprint == fp.0 && p.knobs.is_some()).then_some(p)
}

/// Writes `p` atomically (tmp + rename).
pub fn save_profile(p: &Profile) -> Result<PathBuf, Box<dyn Error>> {
    let path = profile_path(&Fingerprint(p.fingerprint.clone()))
        .ok_or("no config dir: set XDG_CONFIG_HOME or HOME")?;
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d)?;
    }
    golem_engine::atomic::write_atomic(&path, &serde_json::to_string_pretty(p)?)?;
    Ok(path)
}

/// A `measure` error that ends the descent (time budget spent, or memory
/// not returned between candidates) instead of disqualifying one candidate.
#[derive(Debug)]
pub struct StopSweep(pub String);

impl std::fmt::Display for StopSweep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for StopSweep {}

/// Win rule: +3% tok/s and hit rate within 0.5 pp of the paired incumbent
/// (hit rate shifts across the first loads, so a fixed baseline drifts);
/// TTFT and greedy output (up to near-ties) held against the baseline so small regressions
/// cannot compound across coordinate steps.
fn wins(c: &Metrics, incumbent: &Metrics, base: &Metrics) -> bool {
    c.tok_s >= incumbent.tok_s * 1.03
        && c.hit_rate >= incumbent.hit_rate - 0.005
        && c.ttft_p50_ms <= base.ttft_p50_ms * 1.2
        && c.ttft_p99_ms <= base.ttft_p99_ms * 1.2
        && greedy_matches(c, base)
}

/// Bounded coordinate descent (max 2 passes) from `base` over `axes`. Each
/// candidate is measured paired with the incumbent in randomized order; the
/// winner is confirmed by a reverse-order rerun (winner, then baseline). A
/// failed `measure` disqualifies that candidate; a [`StopSweep`] ends the
/// descent with the best so far. `None` = nothing beat the baseline.
#[allow(clippy::type_complexity)] // #81 interface, spelled out by design
pub fn sweep(
    base: &Knobs,
    axes: &[Axis],
    measure: &mut dyn FnMut(&Knobs) -> Result<Metrics, Box<dyn Error>>,
) -> Result<Option<(Knobs, Metrics)>, Box<dyn Error>> {
    let baseline = measure(base)?;
    let mut best = (*base, baseline.clone());
    let mut tried = vec![*base];
    let mut coin = rng_seed();
    let mut stopped = false;
    // Hit rate given up across accepted steps; only confirm bounds it, so an
    // unconfirmed winner must keep it within one step's 0.5 pp.
    let mut drift = 0.0;
    'passes: for _ in 0..2 {
        let mut moved = false;
        for axis in axes {
            let start = best.0;
            let axis_drift = drift;
            for cand in axis.variants(&start) {
                if tried.contains(&cand) {
                    continue;
                }
                tried.push(cand);
                coin ^= coin << 13;
                coin ^= coin >> 7;
                coin ^= coin << 17;
                let pair = if coin & 1 == 0 {
                    measure(&cand).and_then(|c| Ok((c, measure(&start)?)))
                } else {
                    measure(&start).and_then(|i| Ok((measure(&cand)?, i)))
                };
                match pair {
                    Ok((c, inc)) => {
                        // Beat both this axis's best so far and a fresh
                        // incumbent run.
                        let bar = if best.0 == start { &inc } else { &best.1 };
                        if wins(&c, bar, &baseline) && wins(&c, &inc, &baseline) {
                            drift = axis_drift + (inc.hit_rate - c.hit_rate).max(0.0);
                            best = (cand, c);
                        }
                    }
                    Err(e) if e.is::<StopSweep>() => {
                        eprintln!("golem optimize: stopping early: {e}");
                        stopped = true;
                        break 'passes;
                    }
                    Err(e) => eprintln!("golem optimize: {cand:?} disqualified: {e}"),
                }
            }
            moved |= best.0 != start;
        }
        if !moved {
            break;
        }
    }
    if best.0 == *base {
        return Ok(None);
    }
    let unconfirmed = |why: &dyn std::fmt::Display| {
        if drift > 0.005 {
            eprintln!(
                "golem optimize: unconfirmed winner dropped ({why}; hit rate −{:.1} pp)",
                drift * 100.0
            );
            None
        } else {
            eprintln!("golem optimize: winner kept unconfirmed ({why})");
            Some(best.clone())
        }
    };
    if stopped {
        return Ok(unconfirmed(&"sweep stopped early"));
    }
    let confirm = measure(&best.0).and_then(|w| Ok((w, measure(base)?)));
    match confirm {
        Ok((w, b)) if wins(&w, &b, &baseline) => Ok(Some((best.0, w))),
        Ok(_) => {
            eprintln!("golem optimize: winner not confirmed by the reverse-order rerun");
            Ok(None)
        }
        Err(e) if e.is::<StopSweep>() => Ok(unconfirmed(&e)),
        Err(e) => {
            eprintln!("golem optimize: confirmation failed: {e}");
            Ok(None)
        }
    }
}

fn rng_seed() -> u64 {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    (t ^ u64::from(std::process::id())) | 1
}
