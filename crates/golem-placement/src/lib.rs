//! Three-tier expert placement policy: disk, RAM, VRAM (tech design §6.1–6.4).
//!
//! Pure logic — no FFI, no threads. Residency changes are decided here
//! (orderings consumed by the P2/P3 patches) and reported back via
//! [`Policy::apply_transition`].

pub mod journal;
pub mod pool;

pub use journal::{Journal, JournalHeader, JournalRecord};

use std::fmt;

/// Residency tier of an expert (§6.1). Bits are independent per tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    /// Weight bytes live only on disk (cold).
    Disk,
    /// Page-in'd into the mlock'd host pool.
    Ram,
    /// Copied into the device expert cache.
    Vram,
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Tier::Disk => "disk",
            Tier::Ram => "ram",
            Tier::Vram => "vram",
        })
    }
}

/// A cap cannot fit its tier budget (§6.4) — maps to
/// `golem_plan_infeasible` at the boundary; carries the exact numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct Infeasible {
    /// Tier whose cap/budget pair failed.
    pub tier: Tier,
    /// Bytes the cap requires (`u64::MAX` on `checked_mul` overflow).
    pub required_bytes: u64,
    /// The failing budget.
    pub budget_bytes: u64,
    /// Per-expert bytes supplied to [`Policy::new`].
    pub max_bytes_per_expert: u64,
}

impl fmt::Display for Infeasible {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "golem_plan_infeasible: {} cap requires {} bytes ({} experts × {} B) but budget is {}",
            self.tier,
            self.required_bytes,
            self.required_bytes / self.max_bytes_per_expert,
            self.max_bytes_per_expert,
            self.budget_bytes
        )
    }
}

impl std::error::Error for Infeasible {}

/// §6.2 default score half-life H, in steps.
pub const HALF_LIFE_STEPS: f64 = 512.0;

/// §6.2 decay of a score over `dt` steps: e^(−dt/H).
pub fn decay_factor(dt: u64, half_life: f64) -> f64 {
    (-(dt as f64) / half_life).exp()
}

/// Policy configuration (§6.1–6.4). Defaults are the design numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct PlacementConfig {
    /// Number of expert layers.
    pub layers: u32,
    /// Experts per layer.
    pub experts_per_layer: u32,
    /// Score half-life in steps, H (§6.2); decay = e^(−1/H).
    pub half_life_steps: f64,
    /// Minimum dwell in a tier before any transition, D_min (§6.3).
    pub min_dwell_steps: u64,
    /// VRAM expert budget, in experts (count cap).
    pub vram_experts_max: usize,
    /// Host pool capacity, in experts.
    pub ram_slots: usize,
}

impl Default for PlacementConfig {
    fn default() -> Self {
        Self {
            layers: 0,
            experts_per_layer: 0,
            half_life_steps: HALF_LIFE_STEPS,
            min_dwell_steps: 256,
            vram_experts_max: 0,
            ram_slots: 0,
        }
    }
}

/// Flat expert key: `layer * experts_per_layer + expert` (§6.1).
///
/// `None` when `(layer, expert)` is out of model range.
pub fn key(layers: u32, experts_per_layer: u32, layer: u32, expert: u32) -> Option<usize> {
    if layer >= layers || expert >= experts_per_layer {
        return None;
    }
    Some(layer as usize * experts_per_layer as usize + expert as usize)
}

const TIER_DISK: u8 = 1;
const TIER_RAM: u8 = 2;
const TIER_VRAM: u8 = 4;

#[derive(Debug, Clone)]
struct ExpertState {
    tiers: u8,
    score: f64,
    last_score_step: u64,
    last_transition_step: u64,
}

impl Default for ExpertState {
    fn default() -> Self {
        Self {
            tiers: TIER_DISK,
            score: 0.0,
            last_score_step: 0,
            last_transition_step: 0,
        }
    }
}

/// Placement policy over a flat expert table (§6.1): scores (§6.2), transitions
/// with hysteresis + min-dwell (§6.3), budgets (§6.4).
#[derive(Debug)]
pub struct Policy {
    cfg: PlacementConfig,
    states: Vec<ExpertState>,
    journal: Option<Journal>,
    /// Step of the last record created (pending or pushed); recording resumes
    /// strictly after this step on re-attach.
    journal_step: Option<u64>,
    /// Record for the current step; pushed to the journal when a later step's
    /// record is created or on [`Policy::detach_journal`].
    pending: Option<JournalRecord>,
    /// Per-pass scratch reused across [`Policy::pass`] calls: the decode
    /// hot path must not allocate the rank/flag buffers per pass.
    scratch_rank: Vec<(f64, usize)>,
    scratch_flags: Vec<bool>, // [0..n) = in_order, [n..2n) = promoted
}

impl Policy {
    /// Feasibility check at load time (§6.4, caps only): rejects caps that
    /// cannot fit their byte budgets at `max_bytes_per_expert` bytes each.
    ///
    /// # Panics
    /// Panics if `half_life_steps <= 0` or non-finite (would produce NaN
    /// scores).
    pub fn new(
        cfg: PlacementConfig,
        max_bytes_per_expert: u64,
        vram_budget_bytes: u64,
        ram_budget_bytes: u64,
    ) -> Result<Self, Infeasible> {
        let vram_req = (cfg.vram_experts_max as u64).saturating_mul(max_bytes_per_expert);
        if vram_req > vram_budget_bytes {
            return Err(Infeasible {
                tier: Tier::Vram,
                required_bytes: vram_req,
                budget_bytes: vram_budget_bytes,
                max_bytes_per_expert,
            });
        }
        let ram_req = (cfg.ram_slots as u64).saturating_mul(max_bytes_per_expert);
        if ram_req > ram_budget_bytes {
            return Err(Infeasible {
                tier: Tier::Ram,
                required_bytes: ram_req,
                budget_bytes: ram_budget_bytes,
                max_bytes_per_expert,
            });
        }
        assert!(
            cfg.half_life_steps > 0.0 && cfg.half_life_steps.is_finite(),
            "half_life_steps must be finite and > 0"
        );
        let n = cfg.layers as usize * cfg.experts_per_layer as usize;
        Ok(Self {
            cfg,
            states: vec![ExpertState::default(); n],
            journal: None,
            journal_step: None,
            pending: None,
            scratch_rank: Vec::with_capacity(n),
            scratch_flags: vec![false; 2 * n],
        })
    }

    /// The configuration this policy was built with.
    pub fn config(&self) -> &PlacementConfig {
        &self.cfg
    }

    /// Number of tracked experts (flat state count).
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// True when there are no tracked experts.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Class-scope helper so callers mutating `self.scratch_rank` (e.g.
    /// `pass`) can use it without a `&self` borrow conflict.
    fn decay_factor(from: u64, to: u64, half_life: f64) -> f64 {
        decay_factor(from.abs_diff(to), half_life)
    }

    /// Record an access at `step` (§6.2): lazy decay since the expert's last
    /// score touch, then `+1`.
    ///
    /// # Panics
    /// Panics if `key` is out of range (use [`key`] to build it).
    pub fn on_access(&mut self, key: usize, step: u64) {
        debug_assert!(
            step >= self.states[key].last_score_step,
            "steps must be monotonic"
        );
        let df = Self::decay_factor(
            self.states[key].last_score_step,
            step,
            self.cfg.half_life_steps,
        );
        let st = &mut self.states[key];
        st.score = st.score * df + 1.0;
        st.last_score_step = step;
        self.record_access(key, step);
    }

    /// Enable journal recording (§11), replacing any prior journal.
    /// Recording starts at the next step strictly greater than the last
    /// recorded step.
    pub fn attach_journal(&mut self, journal: Journal) {
        if self.journal.is_some() {
            self.flush_pending();
        }
        self.journal = Some(journal);
    }

    /// Flush the pending record (if any) and return the journal for
    /// dump-on-shutdown; `None` when not attached.
    pub fn detach_journal(&mut self) -> Option<Journal> {
        self.flush_pending();
        self.journal.take()
    }

    fn record_access(&mut self, key: usize, step: u64) {
        if self.journal.is_none() {
            return;
        }
        if self.start_record(step)
            && let Some(p) = self.pending.as_mut()
        {
            p.accesses.push(key);
        }
    }

    fn record_transition(&mut self, key: usize, tier: Tier, present: bool, step: u64) {
        if self.journal.is_none() {
            return;
        }
        if self.start_record(step)
            && let Some(p) = self.pending.as_mut()
        {
            p.transitions.push((key, tier, present));
        }
    }

    /// Create the pending record for `step`, pushing (or dropping, when
    /// empty) any record from an earlier step. Steps at or before
    /// `journal_step` are skipped, so re-attached journals record only
    /// strictly later steps. Returns whether the pending record now
    /// belongs to `step` — a skipped step (regressed or already recorded)
    /// returns false and the caller must not attribute the event to it.
    fn start_record(&mut self, step: u64) -> bool {
        if self.pending.as_ref().is_some_and(|p| p.step == step) {
            return true;
        }
        if self.journal_step.is_some_and(|s| step <= s) {
            return false;
        }
        self.flush_pending();
        self.journal_step = Some(step);
        self.pending = Some(JournalRecord {
            step,
            accesses: Vec::new(),
            transitions: Vec::new(),
        });
        true
    }

    /// Push the pending record when non-empty; drop it otherwise (records
    /// exist only for steps with at least one access or transition).
    fn flush_pending(&mut self) {
        if let Some(p) = self.pending.take()
            && (!p.accesses.is_empty() || !p.transitions.is_empty())
            && let Some(j) = self.journal.as_mut()
        {
            j.push(p);
        }
    }

    /// Effective score at `step` (§6.2), refreshed in place.
    ///
    /// # Panics
    /// Panics if `key` is out of range (use [`key`] to build it).
    pub fn score(&mut self, key: usize, step: u64) -> f64 {
        debug_assert!(
            step >= self.states[key].last_score_step,
            "steps must be monotonic"
        );
        let df = Self::decay_factor(
            self.states[key].last_score_step,
            step,
            self.cfg.half_life_steps,
        );
        let st = &mut self.states[key];
        st.score *= df;
        st.last_score_step = step;
        st.score
    }

    /// Report a residency change executed by P2/P3 (§6.3). Flips the tier bit
    /// and restarts the dwell clock.
    ///
    /// # Panics
    /// Panics if `key` is out of range (use [`key`] to build it).
    pub fn apply_transition(&mut self, key: usize, tier: Tier, present: bool, step: u64) {
        debug_assert!(
            step >= self.states[key].last_transition_step,
            "steps must be monotonic"
        );
        let bit = match tier {
            Tier::Disk => TIER_DISK,
            Tier::Ram => TIER_RAM,
            Tier::Vram => TIER_VRAM,
        };
        let st = &mut self.states[key];
        let bit_set = st.tiers & bit != 0;
        if bit_set == present {
            return; // no-op: dwell clock keeps running
        }
        if present {
            st.tiers |= bit;
        } else {
            st.tiers &= !bit;
        }
        st.last_transition_step = step;
        self.record_transition(key, tier, present, step);
    }

    /// Residency of `key` in `tier`.
    ///
    /// # Panics
    /// Panics if `key` is out of range (use [`key`] to build it).
    pub fn residency(&self, key: usize, tier: Tier) -> bool {
        let bit = match tier {
            Tier::Disk => TIER_DISK,
            Tier::Ram => TIER_RAM,
            Tier::Vram => TIER_VRAM,
        };
        self.states[key].tiers & bit != 0
    }

    /// Policy pass (§6.3): refresh scores, order experts, produce the
    /// promotion/pinning order for P2 and the eviction candidate lists.
    pub fn pass(&mut self, step: u64) -> PolicyOutput {
        let vram_cap = self.cfg.vram_experts_max;
        let min_dwell = self.cfg.min_dwell_steps;
        let half_life = self.cfg.half_life_steps;
        let n = self.states.len();
        self.scratch_rank.clear();
        let rank = &mut self.scratch_rank;
        for k in 0..n {
            let last = self.states[k].last_score_step;
            let d = Self::decay_factor(last, step, half_life);
            let st = &mut self.states[k];
            st.score *= d;
            st.last_score_step = step;
            rank.push((st.score, k));
        }
        rank.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));

        let resident = |k: usize, bit: u8| self.states[k].tiers & bit != 0;
        let dwell_ok =
            |k: usize| step.saturating_sub(self.states[k].last_transition_step) > min_dwell;

        // Promotion is a two-phase relative-swap rule (§6.3 as amended):
        // the order is seeded with current VRAM residents first (best-first,
        // up to the cap), then a dwell-expired RAM candidate fills any
        // remaining room and, at a full cap, displaces the weakest seeded
        // resident only when its score beats that resident by ≥ 20% (and
        // the resident's min-dwell has expired). Seeding residents before
        // considering challengers is what makes the margin real hysteresis
        // against incumbents — a single rank-sorted iteration can never
        // reach the swap branch because every earlier entry scored at
        // least as high. Absolute thresholds cannot separate hot from cold
        // here: the steady-state score of an average expert is p·H ≈ 14 at
        // H = 512, an order of magnitude above any fixed θ.
        const SWAP_MARGIN: f64 = 1.2;
        let mut vram_order: Vec<PolicyEntry> = Vec::with_capacity(vram_cap);
        let (in_order, promoted) = self.scratch_flags.split_at_mut(n);
        in_order.fill(false);
        promoted.fill(false);
        // Phase 1: incumbents keep their seats.
        for &(score, k) in rank.iter() {
            if vram_order.len() == vram_cap {
                break;
            }
            if resident(k, TIER_VRAM) {
                in_order[k] = true;
                vram_order.push(PolicyEntry { key: k, score });
            }
        }
        // Phase 2: challengers. The margin is relative (`score >= 1.2·incumbent`),
        // so a 0.0-score incumbent is displaced only through the dwell
        // tie-break (`dwell_ok(weakest.key)`): the margin is vacuous there
        // (0 ≥ 1.2·0) and min-dwell is what bounds the churn. The swap
        // transiently leaves a displaced resident in VRAM past the cap
        // until its eviction lands — bounded by D_min, and the real P2
        // slab (K slots) enforces the hard budget at promotion time.
        for &(score, k) in rank.iter() {
            if vram_cap == 0 || in_order[k] || !resident(k, TIER_RAM) || !dwell_ok(k) {
                continue;
            }
            if vram_order.len() < vram_cap {
                in_order[k] = true;
                vram_order.push(PolicyEntry { key: k, score });
                continue;
            }
            // The weakest seat, not the last-pushed one: after a swap the
            // challenger sits at the tail. Ties pick the higher key (rank order).
            let (wi, weakest) = vram_order
                .iter()
                .copied()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.score.total_cmp(&b.score).then(b.key.cmp(&a.key)))
                .expect("cap > 0 with a full order is non-empty");
            // A never-accessed challenger (score 0) never swaps: 0 ≥ 1.2·0
            // would otherwise churn zero-score seats. A 0.0-score incumbent
            // yields to any positive challenger once its dwell expires.
            if score > 0.0 && score >= SWAP_MARGIN * weakest.score && dwell_ok(weakest.key) {
                in_order[weakest.key] = false;
                in_order[k] = true;
                vram_order[wi] = PolicyEntry { key: k, score };
            } else {
                // Challengers arrive best-first and the weakest seat only
                // strengthens on a swap: once one is rejected, all later are.
                break;
            }
        }
        vram_order.sort_by(|a, b| b.score.total_cmp(&a.score).then(a.key.cmp(&b.key)));
        // in_order's pass is done; the same scratch buffer serves `promoted`.
        for e in &vram_order {
            if !resident(e.key, TIER_VRAM) {
                promoted[e.key] = true;
            }
        }
        // Eviction candidates: VRAM residents that fell out of the order
        // (displaced by a swap or pushed past the cap), dwell-gated,
        // weakest first.
        let mut vram_evict_order: Vec<PolicyEntry> = rank
            .iter()
            .filter(|&&(_, k)| resident(k, TIER_VRAM) && !in_order[k] && dwell_ok(k))
            .map(|&(_, k)| PolicyEntry {
                key: k,
                score: self.states[k].score,
            })
            .collect();
        vram_evict_order.reverse();

        let mut ram_evict_order: Vec<PolicyEntry> = Vec::new();
        for &(_, k) in rank.iter() {
            if !resident(k, TIER_RAM) || resident(k, TIER_VRAM) {
                continue;
            }
            // promoted into VRAM this pass — P2 owns it, P3 must not evict it
            if promoted[k] {
                continue;
            }
            // min-dwell gates RAM eviction too
            if !dwell_ok(k) {
                continue;
            }
            ram_evict_order.push(PolicyEntry {
                key: k,
                score: self.states[k].score,
            });
        }
        ram_evict_order.reverse();
        self.start_record(step);
        PolicyOutput {
            vram_order,
            vram_evict_order,
            ram_evict_order,
        }
    }

    /// Seed scores from a previous run's journal (§6.3 warm-up). Each
    /// recorded access is replayed at its original virtual step `r.step`
    /// (no gap) through the §6.2 lazy-decay score function, then every
    /// key is swept once to the common horizon `journal.last_step() +
    /// gap_steps` — the gap is applied exactly once, at the sweep. Net
    /// decay of an access at journal step `s` is `e^(−(T+gap−s)/H)`
    /// where `T` is the journal's last step, so the most recent access
    /// (`s = T`) decays by `e^(−gap/H)` and older accesses decay
    /// strictly more: recency is preserved, and within-run inter-access
    /// decays remain the true `s2 − s1`. Raw accesses are replayed
    /// without per-step dedup, mirroring the recorder. Returns
    /// [`SeedMismatch`]
    /// when the journal's header doesn't match this policy's
    /// `PlacementConfig` — same flat key count but different topology
    /// (e.g. 2×128 vs 1×256), or different half_life — silently seeding
    /// the wrong experts is worse than a cold start. Also refuses a
    /// journal containing a key outside this policy's flat key range:
    /// a well-formed but corrupt file would otherwise panic on replay. On Ok, ALL
    /// `last_score_step` values are swept to 0 (scores decayed into the
    /// sweep): the policy is left in a state where `on_access(k, 0)` is
    /// immediately legal — no debug_assert trip, no caller ordering
    /// requirement. Residency/dwell are NOT seeded — tiers start cold.
    pub fn seed_from_journal(
        &mut self,
        header: &JournalHeader,
        journal: &Journal,
        gap_steps: u64,
    ) -> Result<(), SeedMismatch> {
        if header.layers != self.cfg.layers
            || header.experts_per_layer != self.cfg.experts_per_layer
            || header.half_life_steps != self.cfg.half_life_steps
        {
            return Err(SeedMismatch {
                journal: header.clone(),
                policy_layers: self.cfg.layers,
                policy_experts: self.cfg.experts_per_layer,
                policy_half_life: self.cfg.half_life_steps,
            });
        }
        // An empty journal still honors the sweep contract: decay every
        // key into step 0 (decay_factor is magnitude-based, so this is
        // e^(−s/H) for a key last touched at s) rather than early-returning
        // with nonzero `last_score_step` stamps.
        // Pre-validate before any mutation: a mid-replay Err would leave
        // partially-seeded scores with nonzero `last_score_step` stamps,
        // tripping `warm_plan`'s debug_assert. `SeedMismatch` doubles as
        // the refusal type here — all its fields equal this policy's own
        // config when the trigger is a key range, not the header.
        let n = self.states.len();
        if journal.iter().any(|r| r.accesses.iter().any(|&k| k >= n)) {
            return Err(SeedMismatch {
                journal: header.clone(),
                policy_layers: self.cfg.layers,
                policy_experts: self.cfg.experts_per_layer,
                policy_half_life: self.cfg.half_life_steps,
            });
        }
        let horizon = journal.last_step().map_or(0, |t| t + gap_steps);
        for r in journal.iter() {
            let vstep = r.step;
            for &k in &r.accesses {
                let df = Self::decay_factor(
                    self.states[k].last_score_step,
                    vstep,
                    self.cfg.half_life_steps,
                );
                let st = &mut self.states[k];
                st.score = st.score * df + 1.0;
                st.last_score_step = vstep;
            }
        }
        // Finalization sweep: apply the gap exactly once by decaying
        // every key to the common horizon `T + gap_steps` (never through
        // public `score()`/`on_access`, whose monotonic debug_asserts
        // would trip), then stamp `last_score_step = 0`. Keys never
        // replayed have score 0, so the sweep is a no-op for them.
        for k in 0..self.states.len() {
            let df = Self::decay_factor(
                self.states[k].last_score_step,
                horizon,
                self.cfg.half_life_steps,
            );
            let st = &mut self.states[k];
            st.score *= df;
            st.last_score_step = 0;
        }
        Ok(())
    }

    /// Warm-start plan (§6.3): after [`Policy::seed_from_journal`], the
    /// ordered flat keys to page-in and promote BEFORE the first request.
    /// Computes its own top-N ranking BY SCORE — deliberately bypassing the
    /// `pass` promotion gates (residency, dwell, margin): a freshly-seeded
    /// policy is fully cold (every key [`Tier::Disk`]), and `pass()`
    /// promotion requires RAM residency, so `pass(step).vram_order` is
    /// empty by construction regardless of score and MUST NOT be the
    /// source. Ties break by ascending key index (matching `pass()`'s
    /// ordering). Returns `(vram, ram)`: `vram` = top `vram_experts_max`
    /// by seeded score, best-first; `ram` = the next best keys that fit
    /// the rest of the pool (`ram_slots − vram`). Does NOT touch residency — the
    /// pool executes page-in + promote and reports back via
    /// [`Policy::apply_transition`] (existing P2/P3 contract).
    ///
    /// # Panics
    /// Panics under the monotonic-step debug_assert if any
    /// `last_score_step` is nonzero — call only on a policy still at step
    /// 0, i.e. immediately after `seed_from_journal`.
    pub fn warm_plan(&mut self) -> (Vec<usize>, Vec<usize>) {
        let cfg = self.cfg.clone();
        let mut rank: Vec<(f64, usize)> = (0..self.states.len())
            .map(|k| (self.score(k, 0), k))
            .collect();
        rank.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        let vram_end = cfg.vram_experts_max.min(rank.len());
        let vram = rank[..vram_end].iter().map(|&(_, k)| k).collect();
        let ram = rank[vram_end..]
            .iter()
            .take(cfg.ram_slots.saturating_sub(vram_end))
            .map(|&(_, k)| k)
            .collect();
        (vram, ram)
    }
}

/// Journal/policy fingerprint disagreement from
/// [`Policy::seed_from_journal`] — refusing to seed is safer than warming
/// the wrong experts.
#[derive(Debug, Clone, PartialEq)]
pub struct SeedMismatch {
    /// Header carried by the journal that was offered.
    pub journal: JournalHeader,
    /// This policy's layers.
    pub policy_layers: u32,
    /// This policy's experts per layer.
    pub policy_experts: u32,
    /// This policy's score half-life (§6.2).
    pub policy_half_life: f64,
}

/// One entry in a policy ordering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolicyEntry {
    /// Flat expert key.
    pub key: usize,
    /// Effective score at the pass.
    pub score: f64,
}

/// Orderings a policy pass produces (§6.3). P2 consumes `vram_order`
/// (best-first) and `vram_evict_order` (worst-first); P3 consumes
/// `ram_evict_order` (worst-first) under pool pressure.
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyOutput {
    /// Prefer-resident VRAM ordering, best-first, truncated to `vram_experts_max`.
    pub vram_order: Vec<PolicyEntry>,
    /// VRAM eviction candidates, worst-first.
    pub vram_evict_order: Vec<PolicyEntry>,
    /// RAM eviction candidates (pool pressure), worst-first.
    pub ram_evict_order: Vec<PolicyEntry>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy1() -> Policy {
        let cfg = PlacementConfig {
            layers: 1,
            experts_per_layer: 8,
            vram_experts_max: 8,
            ram_slots: 8,
            ..Default::default()
        };
        Policy::new(cfg, 1, u64::MAX, u64::MAX).unwrap()
    }

    fn seeded() -> Policy {
        let cfg = PlacementConfig {
            layers: 1,
            experts_per_layer: 8,
            vram_experts_max: 4,
            ram_slots: 8,
            ..Default::default()
        };
        Policy::new(cfg, 1, u64::MAX, u64::MAX).unwrap()
    }

    #[test]
    fn key_indexes_flat_and_rejects_out_of_range() {
        assert_eq!(key(2, 4, 1, 3), Some(7));
        assert_eq!(key(2, 4, 2, 0), None);
        assert_eq!(key(2, 4, 0, 4), None);
    }

    #[test]
    fn defaults_match_design_numbers() {
        let c = PlacementConfig::default();
        assert_eq!((c.half_life_steps, c.min_dwell_steps), (512.0, 256));
    }

    #[test]
    fn new_rejects_vram_cap_that_exceeds_budget() {
        let cfg = PlacementConfig {
            layers: 1,
            experts_per_layer: 8,
            vram_experts_max: 4,
            ..Default::default()
        };
        let err = Policy::new(cfg, 1000, 3000, u64::MAX).unwrap_err();
        assert_eq!(
            err,
            Infeasible {
                tier: Tier::Vram,
                required_bytes: 4000,
                budget_bytes: 3000,
                max_bytes_per_expert: 1000
            }
        );
        assert!(err.to_string().contains("4000") && err.to_string().contains("3000"));
    }

    #[test]
    fn new_rejects_ram_cap_that_exceeds_budget() {
        let cfg = PlacementConfig {
            layers: 1,
            experts_per_layer: 8,
            ram_slots: 4,
            ..Default::default()
        };
        let err = Policy::new(cfg, 1000, u64::MAX, 3000).unwrap_err();
        assert_eq!(
            err,
            Infeasible {
                tier: Tier::Ram,
                required_bytes: 4000,
                budget_bytes: 3000,
                max_bytes_per_expert: 1000
            }
        );
    }

    #[test]
    fn new_accepts_feasible_config() {
        let cfg = PlacementConfig {
            layers: 1,
            experts_per_layer: 8,
            vram_experts_max: 8,
            ram_slots: 6,
            ..Default::default()
        };
        let p = Policy::new(cfg, 1000, 8000, 8000).unwrap();
        assert_eq!(p.len(), 8);
    }

    #[test]
    fn repeated_access_accumulates_and_idle_decays() {
        let mut p = policy1();
        for s in 0..100 {
            p.on_access(0, s);
        }
        let hot = p.score(0, 100);
        let cold = p.score(1, 100);
        assert!(hot > cold, "hot {hot} cold {cold}");
        p.on_access(1, 100);
        let decayed = p.score(1, 512);
        assert!(decayed < 0.5, "{decayed}");
    }

    #[test]
    fn lazy_decay_matches_eager_reference() {
        let decay = (-1.0_f64 / 512.0).exp();
        let mut eager = [0.0_f64; 2];
        let mut p = policy1();
        let mut xor = 0x1234_5678_u64;
        let mut step = 0_u64;
        for _ in 0..2000 {
            for e in eager.iter_mut() {
                *e *= decay;
            }
            xor ^= xor << 13;
            xor ^= xor >> 7;
            xor ^= xor << 17;
            let k = (xor % 2) as usize;
            eager[k] += 1.0;
            p.on_access(k, step);
            step += 1;
        }
        for (k, e) in eager.iter().enumerate() {
            assert!(
                (p.score(k, step - 1) - e).abs() <= 1e-9 * e.abs().max(1.0),
                "k={k}"
            );
        }
    }

    #[test]
    fn access_ordering_by_recency_and_frequency() {
        let mut p = policy1();
        for s in 0..50 {
            p.on_access(3, s);
        }
        p.on_access(5, 0);
        assert!(p.score(3, 50) > p.score(5, 50));
    }

    #[test]
    fn transition_updates_residency_and_dwell() {
        let mut p = seeded();
        p.apply_transition(0, Tier::Ram, true, 10);
        assert!(p.residency(0, Tier::Ram) && !p.residency(0, Tier::Vram));
        for s in [11, 12, 13, 14, 15] {
            p.on_access(0, s);
        }
        let out = p.pass(100); // dwell = 90 < 256
        assert!(
            out.vram_order.iter().all(|e| e.key != 0),
            "min-dwell must block promotion: {out:?}"
        );
        p.apply_transition(0, Tier::Vram, true, 300);
        assert!(p.residency(0, Tier::Vram));
    }

    #[test]
    fn swap_margin_is_real_hysteresis_against_incumbents() {
        // Cap 1 fully occupied by a VRAM resident: the swap branch must be
        // reachable through the full-cap path — a 1.1× challenger is
        // rejected, a 1.3× one swaps after the incumbent's min-dwell. A
        // huge half-life makes decay ≈ 1, so scores are access counts.
        let mut p = Policy::new(
            PlacementConfig {
                layers: 1,
                experts_per_layer: 8,
                vram_experts_max: 1,
                min_dwell_steps: 10,
                half_life_steps: 1.0e12,
                ..Default::default()
            },
            1,
            u64::MAX,
            u64::MAX,
        )
        .unwrap();
        // Resident: score 100 at the pass step.
        p.apply_transition(0, Tier::Ram, true, 0);
        p.apply_transition(0, Tier::Vram, true, 1);
        for s in 0..100 {
            p.on_access(0, s);
        }
        // Challenger, RAM-resident, score 110 < 1.2 × 100: rejected. Access
        // steps must stay above the first pass (pass sweeps all stamps).
        p.apply_transition(1, Tier::Ram, true, 0);
        for s in 1000..1110 {
            p.on_access(1, s);
        }
        let out = p.pass(1000);
        assert_eq!(out.vram_order[0].key, 0, "1.1× challenger must not swap");
        assert!(out.vram_evict_order.iter().all(|e| e.key != 0));
        // Challenger, score 130 ≥ 1.2 × 100, incumbent dwell expired: swaps.
        p.apply_transition(2, Tier::Ram, true, 0);
        for s in 2000..2130 {
            p.on_access(2, s);
        }
        let out = p.pass(3000);
        assert_eq!(out.vram_order[0].key, 2, "1.3× challenger must swap");
        assert!(out.vram_evict_order.iter().any(|e| e.key == 0));
    }

    #[test]
    fn each_challenger_faces_the_weakest_remaining_incumbent() {
        // Cap 2, incumbents 100 and 200, challengers 300 and 250. 300 takes
        // the 100 seat; 250 must then face the 200 incumbent (≥ 1.2×), not
        // the just-seated 300 challenger.
        let mut p = Policy::new(
            PlacementConfig {
                layers: 1,
                experts_per_layer: 8,
                vram_experts_max: 2,
                min_dwell_steps: 10,
                half_life_steps: 1.0e12,
                ..Default::default()
            },
            1,
            u64::MAX,
            u64::MAX,
        )
        .unwrap();
        for (k, n) in [(0usize, 100u64), (1, 200)] {
            p.apply_transition(k, Tier::Ram, true, 0);
            p.apply_transition(k, Tier::Vram, true, 1);
            for s in 0..n {
                p.on_access(k, s);
            }
        }
        for (k, n) in [(2usize, 300u64), (3, 250)] {
            p.apply_transition(k, Tier::Ram, true, 0);
            for s in 0..n {
                p.on_access(k, 300 + s);
            }
        }
        let out = p.pass(1000);
        let order: Vec<_> = out.vram_order.iter().map(|e| e.key).collect();
        assert_eq!(order, [2, 3], "both challengers swap, best-first");
        let evict: Vec<_> = out.vram_evict_order.iter().map(|e| e.key).collect();
        assert_eq!(evict, [0, 1], "both incumbents displaced, worst-first");
    }

    #[test]
    fn all_zero_scores_never_swap() {
        // Cap 1 held by a 0.0-score resident, challenger also 0.0: 0 ≥ 1.2·0
        // must not displace it even with the dwell long expired.
        let mut p = Policy::new(
            PlacementConfig {
                layers: 1,
                experts_per_layer: 8,
                vram_experts_max: 1,
                min_dwell_steps: 10,
                half_life_steps: 1.0e12,
                ..Default::default()
            },
            1,
            u64::MAX,
            u64::MAX,
        )
        .unwrap();
        p.apply_transition(0, Tier::Ram, true, 0);
        p.apply_transition(0, Tier::Vram, true, 1);
        p.apply_transition(1, Tier::Ram, true, 0);
        let out = p.pass(1000);
        assert_eq!(out.vram_order.len(), 1);
        assert_eq!(out.vram_order[0].key, 0, "zero challenger must not swap");
    }

    #[test]
    fn zero_score_incumbent_leaves_via_dwell_tie_break() {
        // Cap 1 held by a never-accessed (0.0-score) VRAM resident. Its
        // margin comparison is vacuous (0 ≥ 1.2·0), so displacement goes
        // through the min-dwell tie-break: no swap while its dwell runs,
        // swap once expired.
        let mut p = Policy::new(
            PlacementConfig {
                layers: 1,
                experts_per_layer: 8,
                vram_experts_max: 1,
                min_dwell_steps: 10,
                half_life_steps: 1.0e12,
                ..Default::default()
            },
            1,
            u64::MAX,
            u64::MAX,
        )
        .unwrap();
        p.apply_transition(0, Tier::Ram, true, 0);
        p.apply_transition(0, Tier::Vram, true, 1);
        // Challenger: RAM-resident, score 1.
        p.apply_transition(1, Tier::Ram, true, 0);
        p.on_access(1, 1000);
        let out = p.pass(5);
        assert_eq!(out.vram_order[0].key, 0, "incumbent dwell has not expired");
        let out = p.pass(1000);
        assert_eq!(
            out.vram_order[0].key, 1,
            "0.0-score incumbent displaced after dwell"
        );
        assert!(out.vram_evict_order.iter().any(|e| e.key == 0));
    }

    #[test]
    fn zero_vram_cap_yields_empty_order_no_panic() {
        let mut p = Policy::new(
            PlacementConfig {
                layers: 1,
                experts_per_layer: 8,
                vram_experts_max: 0,
                ram_slots: 8,
                ..Default::default()
            },
            1,
            u64::MAX,
            u64::MAX,
        )
        .unwrap();
        p.apply_transition(0, Tier::Ram, true, 0);
        p.apply_transition(0, Tier::Vram, true, 1);
        let out = p.pass(10_000);
        assert!(out.vram_order.is_empty());
        // A resident nobody keeps must be an eviction candidate, not a panic.
        assert_eq!(out.vram_evict_order.first().map(|e| e.key), Some(0));
    }

    #[test]
    fn promotion_requires_room_or_margin_and_dwell() {
        // Under the cap: any dwell-expired RAM candidate promotes.
        let mut p = seeded();
        p.apply_transition(2, Tier::Ram, true, 0);
        for s in 0..300 {
            if s % 2 == 0 {
                p.on_access(2, s);
            }
        }
        let out = p.pass(300);
        assert!(out.vram_order.iter().any(|e| e.key == 2));
        // Within min-dwell: no promotion.
        p.apply_transition(3, Tier::Ram, true, 100);
        let out = p.pass(301);
        assert!(out.vram_order.iter().all(|e| e.key != 3));
    }

    #[test]
    fn borderline_candidate_does_not_displace_resident() {
        // Cap 1, resident key 0; key 1 touches 9 of every 10 steps — its
        // score tracks ~0.9× key 0's, under the 1.2× swap margin, so the
        // pair cannot oscillate once min-dwell has expired.
        let mut p = Policy::new(
            PlacementConfig {
                layers: 1,
                experts_per_layer: 8,
                vram_experts_max: 1,
                min_dwell_steps: 10,
                ..Default::default()
            },
            1,
            u64::MAX,
            u64::MAX,
        )
        .unwrap();
        p.apply_transition(0, Tier::Ram, true, 0);
        p.apply_transition(0, Tier::Vram, true, 1);
        p.apply_transition(1, Tier::Ram, true, 0);
        for step in 100..400 {
            p.on_access(0, step);
            if step % 10 != 0 {
                p.on_access(1, step);
            }
            let out = p.pass(step);
            assert_eq!(out.vram_order[0].key, 0, "step {step}: no oscillation");
            assert!(
                out.vram_evict_order.iter().all(|e| e.key != 0),
                "step {step}: resident not displaced"
            );
        }
        // A decisive hot streak clears the margin and takes the slot.
        for step in 400..1000 {
            p.on_access(1, step);
        }
        let out = p.pass(1000);
        assert_eq!(out.vram_order[0].key, 1, "margin swap fired");
        assert!(out.vram_evict_order.iter().any(|e| e.key == 0));
    }

    #[test]
    fn eviction_candidates_respect_hysteresis_and_cap() {
        let mut p = seeded();
        for k in 0..5 {
            p.apply_transition(k, Tier::Ram, true, 0);
            p.apply_transition(k, Tier::Vram, true, 1);
            for s in (0..(5 - k) * 400).map(|s| s as u64) {
                p.on_access(k, s);
            }
        }
        let out = p.pass(100_000);
        assert!(
            out.vram_evict_order.iter().any(|e| e.key == 4),
            "coldest beyond-cut resident must be evictable: {out:?}"
        );
        assert_eq!(
            out.vram_evict_order.first().map(|e| e.key),
            Some(4),
            "coldest beyond-cut resident is evicted first"
        );
        assert!(out.vram_order.len() <= 4, "budget cap respected");
        assert_eq!(out.vram_order[0].key, 0, "best-first ordering");
    }

    #[test]
    fn beyond_cut_eviction_ignores_hysteresis() {
        let mut p = seeded();
        for k in 0..4 {
            p.apply_transition(k, Tier::Ram, true, 0);
            p.apply_transition(k, Tier::Vram, true, 1);
            for s in 0..10 {
                p.on_access(k, 9990 + s);
            }
        }
        p.apply_transition(6, Tier::Ram, true, 0);
        p.apply_transition(6, Tier::Vram, true, 1);
        p.on_access(6, 9900);
        let out = p.pass(10_000);
        assert!(
            out.vram_evict_order.iter().any(|e| e.key == 6),
            "budget cap is hard — warm resident beyond the cut is evictable: {out:?}"
        );
        assert!(out.vram_evict_order.iter().all(|e| e.key != 0));
        assert_eq!(out.vram_order.len(), 4);
        assert_eq!(out.vram_order[0].key, 0);
    }

    #[test]
    fn residents_within_cap_are_not_eviction_candidates() {
        // Budget slack: every VRAM resident is in the order, so none is a
        // demotion candidate — placement changes only via the swap margin.
        let cfg = PlacementConfig {
            layers: 1,
            experts_per_layer: 8,
            vram_experts_max: 8,
            ..Default::default()
        };
        let mut p = Policy::new(cfg, 1, u64::MAX, u64::MAX).unwrap();
        for k in [0_usize, 1, 2] {
            p.apply_transition(k, Tier::Ram, true, 0);
            p.apply_transition(k, Tier::Vram, true, 1);
        }
        p.on_access(0, 0);
        p.on_access(1, 0);
        p.on_access(2, 9900);
        let out = p.pass(10_000);
        assert!(
            out.vram_evict_order.is_empty(),
            "no resident is displaced when the cap has slack: {out:?}"
        );
        assert_eq!(out.vram_order.len(), 3);
    }

    #[test]
    fn ram_eviction_order_excludes_vram_resident_and_promoted() {
        // RAM residents beyond the VRAM cap (not in the order, so not
        // promoted) are the P3 eviction supply, weakest first.
        let mut p = Policy::new(
            PlacementConfig {
                layers: 1,
                experts_per_layer: 8,
                vram_experts_max: 2,
                ram_slots: 8,
                ..Default::default()
            },
            1,
            u64::MAX,
            u64::MAX,
        )
        .unwrap();
        for k in [1_usize, 2, 3, 4] {
            p.apply_transition(k, Tier::Ram, true, 0);
        }
        p.apply_transition(1, Tier::Vram, true, 1);
        // Scored: a 0.0-score VRAM incumbent would now be displacable via
        // the dwell tie-break, and this test is about RAM eviction supply.
        p.on_access(1, 9998);
        p.on_access(2, 9999); // hottest RAM resident → promoted, not evictable
        let out = p.pass(10_000);
        assert!(
            out.ram_evict_order.iter().all(|e| e.key != 1),
            "vram-resident excluded"
        );
        assert!(
            out.ram_evict_order.iter().all(|e| e.key != 2),
            "promoted excluded"
        );
        assert!(
            out.ram_evict_order.iter().any(|e| e.key == 3)
                && out.ram_evict_order.iter().any(|e| e.key == 4),
            "cap-overflow ram residents are the eviction supply: {out:?}"
        );
        assert_eq!(out.ram_evict_order[0].key, 4, "weakest first");
    }

    #[test]
    fn promote_candidate_not_offered_for_ram_eviction() {
        let mut p = policy1();
        p.apply_transition(0, Tier::Ram, true, 0);
        for s in 0..300 {
            p.on_access(0, s);
        }
        let out = p.pass(300);
        assert!(
            out.vram_order.iter().any(|e| e.key == 0),
            "candidate promotes"
        );
        assert!(
            out.ram_evict_order.iter().all(|e| e.key != 0),
            "same key must not be in both orderings: {out:?}"
        );
    }

    #[test]
    fn dwell_expired_ram_resident_is_promoted() {
        let cfg = PlacementConfig {
            layers: 1,
            experts_per_layer: 8,
            vram_experts_max: 8,
            ..Default::default()
        };
        let mut p = Policy::new(cfg, 1, u64::MAX, u64::MAX).unwrap();
        p.apply_transition(0, Tier::Ram, true, 0);
        let out = p.pass(10_000);
        assert!(
            out.vram_order.iter().any(|e| e.key == 0),
            "promoted (in the VRAM order, so not P3-evictable): {out:?}"
        );
        assert!(out.ram_evict_order.iter().all(|e| e.key != 0));
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "monotonic")]
    fn backward_step_panics_in_debug() {
        let mut p = policy1();
        p.on_access(0, 10);
        p.on_access(0, 5);
    }

    #[test]
    fn auto_records_scripted_sequence() {
        let mut p = policy1();
        p.attach_journal(Journal::new(journal::JOURNAL_CAP));
        p.on_access(0, 0);
        p.on_access(1, 0);
        p.pass(0);
        p.apply_transition(2, Tier::Vram, true, 1);
        p.pass(1);
        let j = p.detach_journal().unwrap();
        let recs: Vec<_> = j.iter().collect();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].step, 0);
        assert_eq!(recs[0].accesses, vec![0, 1]);
        assert!(recs[0].transitions.is_empty());
        assert_eq!(recs[1].step, 1);
        assert!(recs[1].accesses.is_empty());
        assert_eq!(recs[1].transitions, vec![(2, Tier::Vram, true)]);
    }

    #[test]
    fn detach_flushes_open_record() {
        let mut p = policy1();
        p.attach_journal(Journal::new(journal::JOURNAL_CAP));
        p.on_access(3, 2);
        let j = p.detach_journal().unwrap();
        assert_eq!(j.len(), 1);
        assert_eq!(j.iter().next().unwrap().step, 2);
        assert_eq!(j.iter().next().unwrap().accesses, vec![3]);
    }

    #[test]
    fn pass_twice_replaces_not_duplicates() {
        let mut p = policy1();
        p.attach_journal(Journal::new(journal::JOURNAL_CAP));
        p.on_access(5, 0);
        p.pass(0);
        p.pass(0);
        let j = p.detach_journal().unwrap();
        assert_eq!(j.len(), 1);
        let rec = j.iter().next().unwrap();
        assert_eq!(rec.step, 0);
        assert_eq!(rec.accesses, vec![5]);
    }

    #[test]
    fn reattach_starts_at_next_step() {
        let mut p = policy1();
        p.attach_journal(Journal::new(journal::JOURNAL_CAP));
        p.on_access(0, 1);
        assert_eq!(p.detach_journal().unwrap().len(), 1);
        p.attach_journal(Journal::new(journal::JOURNAL_CAP));
        p.on_access(1, 1); // at journal_step: not recorded
        p.on_access(2, 2); // strictly greater: recorded
        let j = p.detach_journal().unwrap();
        assert_eq!(j.len(), 1);
        let rec = j.iter().next().unwrap();
        assert_eq!(rec.step, 2);
        assert_eq!(rec.accesses, vec![2]);
    }

    #[test]
    fn detached_policy_records_nothing() {
        let mut p = policy1();
        p.on_access(0, 0);
        p.pass(0);
        p.apply_transition(1, Tier::Ram, true, 1);
        assert!(p.detach_journal().is_none());

        p.attach_journal(Journal::new(journal::JOURNAL_CAP));
        p.on_access(2, 2);
        let j = p.detach_journal().unwrap();
        p.on_access(3, 3);
        p.pass(3);
        assert_eq!(j.len(), 1);
        assert_eq!(j.iter().next().unwrap().step, 2);
        assert!(p.detach_journal().is_none());
    }

    #[test]
    fn noop_transitions_not_recorded() {
        let mut p = policy1();
        p.attach_journal(Journal::new(journal::JOURNAL_CAP));
        p.apply_transition(0, Tier::Disk, true, 0); // already resident: no-op
        p.apply_transition(1, Tier::Vram, false, 1); // already absent: no-op
        assert!(p.detach_journal().unwrap().is_empty());
    }

    #[test]
    fn empty_intermediate_steps_produce_no_records() {
        let mut p = policy1();
        p.attach_journal(Journal::new(journal::JOURNAL_CAP));
        p.on_access(0, 0);
        p.pass(1); // empty step 1 created
        p.pass(2); // empty step 1 dropped
        p.on_access(1, 2);
        p.pass(3); // pushes the step-2 record
        let j = p.detach_journal().unwrap();
        let recs: Vec<_> = j.iter().collect();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].step, 0); // non-empty: kept
        assert_eq!(recs[0].accesses, vec![0]);
        assert_eq!(recs[1].step, 2); // empty step 1 dropped
        assert_eq!(recs[1].accesses, vec![1]);
    }

    #[test]
    fn seed_from_journal_replays_and_sweeps_preserving_recency() {
        let mut p = Policy::new(
            PlacementConfig {
                layers: 1,
                experts_per_layer: 8,
                vram_experts_max: 8,
                ram_slots: 8,
                half_life_steps: 512.0,
                ..Default::default()
            },
            1,
            u64::MAX,
            u64::MAX,
        )
        .unwrap();
        let mut j = Journal::new(journal::JOURNAL_CAP);
        j.push(JournalRecord {
            step: 3,
            accesses: vec![1],
            transitions: vec![],
        });
        j.push(JournalRecord {
            step: 1000,
            accesses: vec![0],
            transitions: vec![],
        });
        let header = journal::JournalHeader {
            layers: 1,
            experts_per_layer: 8,
            half_life_steps: 512.0,
        };
        p.seed_from_journal(&header, &j, 0).unwrap();
        // Recency-preserving sweep to horizon T+gap = 1000: the newest
        // access (A@1000 = T) is undecayed, older B@3 keeps only
        // e^(−997/512).
        let a = p.score(0, 0);
        assert!((a - 1.0).abs() <= 1e-9, "score(A,0)={a}, expected 1.0");
        let b = p.score(1, 0);
        let expected = (-997.0f64 / 512.0).exp();
        assert!(
            (b - expected).abs() <= expected * 1e-9,
            "score(B,0)={b}, expected e^(-997/512)={expected}"
        );
        // Recency order restored: the most recent access outranks the old one.
        assert!(a > b, "A@1000 ({a}) must outrank B@3 ({b})");
        // Key absent from the journal: exact zero, no blanket bump.
        assert_eq!(p.score(2, 0), 0.0);
        // Sweep contract: on_access from step 0 must be immediately legal.
        p.on_access(0, 0);
    }

    #[test]
    fn seed_from_journal_applies_gap_once_at_sweep() {
        let mut p = Policy::new(
            PlacementConfig {
                layers: 1,
                experts_per_layer: 8,
                vram_experts_max: 8,
                ram_slots: 8,
                half_life_steps: 512.0,
                ..Default::default()
            },
            1,
            u64::MAX,
            u64::MAX,
        )
        .unwrap();
        let mut j = Journal::new(journal::JOURNAL_CAP);
        j.push(JournalRecord {
            step: 3,
            accesses: vec![1],
            transitions: vec![],
        });
        j.push(JournalRecord {
            step: 1000,
            accesses: vec![0],
            transitions: vec![],
        });
        let header = journal::JournalHeader {
            layers: 1,
            experts_per_layer: 8,
            half_life_steps: 512.0,
        };
        p.seed_from_journal(&header, &j, 512).unwrap();
        // Gap applied exactly once at the sweep to T+gap = 1512: newest
        // access A@1000 decays by e^(−512/512), older B@3 by
        // e^(−(997+512)/512). Replay at r.step (no gap) keeps
        // inter-access decay at the true 997.
        let a = p.score(0, 0);
        let ea = (-512.0f64 / 512.0).exp();
        assert!(
            (a - ea).abs() <= ea * 1e-9,
            "score(A,0)={a}, expected e^(-1)={ea}"
        );
        let b = p.score(1, 0);
        let eb = (-(997.0f64 + 512.0) / 512.0).exp();
        assert!(
            (b - eb).abs() <= eb * 1e-9,
            "score(B,0)={b}, expected e^(-1509/512)={eb}"
        );
        assert!(a > b, "A@1000 ({a}) must outrank B@3 ({b})");
        // Sweep contract: on_access from step 0 must be immediately legal.
        p.on_access(0, 0);
    }

    #[test]
    fn regressed_step_not_attributed_to_pending_record() {
        let mut p = policy1();
        p.attach_journal(Journal::new(journal::JOURNAL_CAP));
        p.on_access(0, 10); // per-key asserts pass for both keys
        p.on_access(1, 5); // regressed step: must not land in step-10 record
        p.on_access(2, 10); // back to the open step: recorded
        let j = p.detach_journal().unwrap();
        let recs: Vec<_> = j.iter().collect();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].step, 10);
        assert_eq!(recs[0].accesses, vec![0, 2]);
    }

    #[test]
    fn seed_from_journal_rejects_out_of_range_key() {
        let mut p = policy1();
        let mut j = Journal::new(journal::JOURNAL_CAP);
        j.push(JournalRecord {
            step: 3,
            accesses: vec![8], // flat key count is 8 (1×8): out of range
            transitions: vec![],
        });
        let header = journal::JournalHeader {
            layers: 1,
            experts_per_layer: 8,
            half_life_steps: 512.0,
        };
        assert!(p.seed_from_journal(&header, &j, 0).is_err());
        // Boundary key is fine.
        j.clear();
        j.push(JournalRecord {
            step: 3,
            accesses: vec![7],
            transitions: vec![],
        });
        assert!(p.seed_from_journal(&header, &j, 0).is_ok());
    }

    #[test]
    fn seed_from_journal_empty_sweeps_to_zero() {
        let mut p = policy1();
        p.on_access(0, 100); // policy saw steps before seeding
        let j = Journal::new(journal::JOURNAL_CAP);
        let header = journal::JournalHeader {
            layers: 1,
            experts_per_layer: 8,
            half_life_steps: 512.0,
        };
        p.seed_from_journal(&header, &j, 0).unwrap();
        let expected = (-100.0f64 / 512.0).exp();
        let s = p.score(0, 0);
        assert!((s - expected).abs() <= expected * 1e-9);
        p.on_access(1, 0); // sweep contract holds
    }

    #[test]
    fn seed_from_journal_rejects_mismatched_header() {
        let mut p = policy1();
        let j = Journal::new(journal::JOURNAL_CAP);
        let wrong_topology = journal::JournalHeader {
            layers: 2,
            experts_per_layer: 128,
            half_life_steps: 512.0,
        };
        let err = p.seed_from_journal(&wrong_topology, &j, 0).unwrap_err();
        assert_eq!(err.journal, wrong_topology);
        assert_eq!(err.policy_layers, 1);
        assert_eq!(err.policy_experts, 8);
        assert_eq!(err.policy_half_life, 512.0);
        let wrong_half_life = journal::JournalHeader {
            layers: 1,
            experts_per_layer: 8,
            half_life_steps: 100.0,
        };
        assert!(p.seed_from_journal(&wrong_half_life, &j, 0).is_err());
    }
}
