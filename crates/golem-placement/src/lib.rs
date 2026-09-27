//! Three-tier expert placement policy: disk, RAM, VRAM (tech design §6.1–6.4).
//!
//! Pure logic — no FFI, no I/O, no threads. Residency changes are decided here
//! (orderings consumed by the P2/P3 patches) and reported back via
//! [`Policy::apply_transition`].

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

/// A floor set cannot fit its tier budget (§6.4) — maps to
/// `golem_plan_infeasible` at the boundary; carries the exact numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct Infeasible {
    /// Tier whose floor/budget pair failed.
    pub tier: Tier,
    /// Bytes the floor set requires (`u64::MAX` on `checked_mul` overflow, 0 on a count rejection).
    pub required_bytes: u64,
    /// The failing budget.
    pub budget_bytes: u64,
    /// Per-expert bytes supplied to [`Policy::new`]; 0 when rejected on counts.
    pub max_bytes_per_expert: u64,
}

impl fmt::Display for Infeasible {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.max_bytes_per_expert == 0 {
            return write!(
                f,
                "golem_plan_infeasible: {} floor count exceeds capacity",
                self.tier
            );
        }
        write!(
            f,
            "golem_plan_infeasible: {} floor requires {} bytes ({} experts × {} B) but budget is {}",
            self.tier,
            self.required_bytes,
            self.required_bytes / self.max_bytes_per_expert,
            self.max_bytes_per_expert,
            self.budget_bytes
        )
    }
}

impl std::error::Error for Infeasible {}

/// Policy configuration (§6.1–6.4). Defaults are the design numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct PlacementConfig {
    /// Number of expert layers.
    pub layers: u32,
    /// Experts per layer.
    pub experts_per_layer: u32,
    /// Score half-life in steps, H (§6.2); decay = e^(−1/H).
    pub half_life_steps: f64,
    /// Promotion threshold θ_hi (§6.3).
    pub theta_hi: f64,
    /// Demotion threshold θ_lo (§6.3).
    pub theta_lo: f64,
    /// Minimum dwell in a tier before any transition, D_min (§6.3).
    pub min_dwell_steps: u64,
    /// VRAM expert budget, in experts (count cap).
    pub vram_experts_max: usize,
    /// Host pool capacity, in experts.
    pub ram_slots: usize,
    /// VRAM floor: top-N vram residents pinned against eviction.
    pub vram_experts_min: usize,
    /// RAM floor: top-N ram residents pinned against eviction.
    pub ram_experts_min: usize,
}

impl Default for PlacementConfig {
    fn default() -> Self {
        Self {
            layers: 0,
            experts_per_layer: 0,
            half_life_steps: 512.0,
            theta_hi: 1.0,
            theta_lo: 0.4,
            min_dwell_steps: 256,
            vram_experts_max: 0,
            ram_slots: 0,
            vram_experts_min: 0,
            ram_experts_min: 0,
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
    Some((layer * experts_per_layer + expert) as usize)
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
/// with hysteresis + min-dwell + floors (§6.3), budgets (§6.4).
#[derive(Debug)]
pub struct Policy {
    cfg: PlacementConfig,
    states: Vec<ExpertState>,
}

impl Policy {
    /// Feasibility check at load time (§6.4): rejects floor sets that cannot
    /// fit their byte budgets, and count-inconsistent floor/cap pairs.
    /// Byte checks run first; count rejections use the
    /// `max_bytes_per_expert = 0` sentinel.
    pub fn new(
        cfg: PlacementConfig,
        max_bytes_per_expert: u64,
        vram_budget_bytes: u64,
        ram_budget_bytes: u64,
    ) -> Result<Self, Infeasible> {
        let vram_req = (cfg.vram_experts_min as u64)
            .checked_mul(max_bytes_per_expert)
            .unwrap_or(u64::MAX);
        if vram_req > vram_budget_bytes {
            return Err(Infeasible {
                tier: Tier::Vram,
                required_bytes: vram_req,
                budget_bytes: vram_budget_bytes,
                max_bytes_per_expert,
            });
        }
        let ram_req = (cfg.ram_experts_min as u64)
            .checked_mul(max_bytes_per_expert)
            .unwrap_or(u64::MAX);
        if ram_req > ram_budget_bytes {
            return Err(Infeasible {
                tier: Tier::Ram,
                required_bytes: ram_req,
                budget_bytes: ram_budget_bytes,
                max_bytes_per_expert,
            });
        }
        if cfg.vram_experts_min > cfg.vram_experts_max || cfg.ram_experts_min > cfg.ram_slots {
            return Err(Infeasible {
                tier: if cfg.vram_experts_min > cfg.vram_experts_max {
                    Tier::Vram
                } else {
                    Tier::Ram
                },
                required_bytes: 0,
                budget_bytes: 0,
                max_bytes_per_expert: 0,
            });
        }
        let n = cfg.layers as usize * cfg.experts_per_layer as usize;
        Ok(Self {
            cfg,
            states: vec![ExpertState::default(); n],
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

    fn decay_factor(&self, from: u64, to: u64) -> f64 {
        let dt = from.max(to) - from.min(to);
        (-(dt as f64) / self.cfg.half_life_steps).exp()
    }

    /// Record an access at `step` (§6.2): lazy decay since the expert's last
    /// score touch, then `+1`.
    pub fn on_access(&mut self, key: usize, step: u64) {
        let df = self.decay_factor(self.states[key].last_score_step, step);
        let st = &mut self.states[key];
        st.score = st.score * df + 1.0;
        st.last_score_step = step;
    }

    /// Effective score at `step` (§6.2), refreshed in place.
    pub fn score(&mut self, key: usize, step: u64) -> f64 {
        let df = self.decay_factor(self.states[key].last_score_step, step);
        let st = &mut self.states[key];
        st.score *= df;
        st.last_score_step = step;
        st.score
    }

    /// Report a residency change executed by P2/P3 (§6.3). Flips the tier bit
    /// and restarts the dwell clock.
    pub fn apply_transition(&mut self, key: usize, tier: Tier, present: bool, step: u64) {
        let bit = match tier {
            Tier::Disk => TIER_DISK,
            Tier::Ram => TIER_RAM,
            Tier::Vram => TIER_VRAM,
        };
        let st = &mut self.states[key];
        if present {
            st.tiers |= bit;
        } else {
            st.tiers &= !bit;
        }
        st.last_transition_step = step;
    }

    /// Residency of `key` in `tier`.
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
        let cfg = self.cfg.clone();
        let n = self.states.len();
        let mut rank: Vec<(f64, usize)> = Vec::with_capacity(n);
        for k in 0..n {
            let df = self.decay_factor(self.states[k].last_score_step, step);
            let st = &mut self.states[k];
            st.score *= df;
            st.last_score_step = step;
            rank.push((st.score, k));
        }
        rank.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));

        let resident = |k: usize, bit: u8| self.states[k].tiers & bit != 0;
        let dwell_ok = |k: usize| step.saturating_sub(self.states[k].last_transition_step) > cfg.min_dwell_steps;

        let mut vram_order: Vec<PolicyEntry> = Vec::with_capacity(cfg.vram_experts_max);
        for &(_, k) in &rank {
            if vram_order.len() == cfg.vram_experts_max {
                break;
            }
            let promoted = !resident(k, TIER_VRAM)
                && resident(k, TIER_RAM)
                && self.states[k].score >= cfg.theta_hi
                && dwell_ok(k);
            if resident(k, TIER_VRAM) || promoted {
                vram_order.push(PolicyEntry {
                    key: k,
                    score: self.states[k].score,
                    floor_pinned: false,
                });
            }
        }
        let mut vram_floor = 0;
        for e in &mut vram_order {
            if vram_floor == cfg.vram_experts_min {
                break;
            }
            e.floor_pinned = true;
            vram_floor += 1;
        }

        let in_order: Vec<usize> = vram_order.iter().map(|e| e.key).collect();
        let vram_floor_keys: Vec<usize> = vram_order
            .iter()
            .filter(|e| e.floor_pinned)
            .map(|e| e.key)
            .collect();
        // Evictable = floor-unpinned vram resident that either missed the truncated
        // order (beyond-cut — hard cap, dwell-free) or crossed below θ_lo with dwell.
        let mut vram_evict_order: Vec<PolicyEntry> = rank
            .iter()
            .filter(|&&(s, k)| {
                resident(k, TIER_VRAM)
                    && !vram_floor_keys.contains(&k)
                    && (!in_order.contains(&k) || (s < cfg.theta_lo && dwell_ok(k)))
            })
            .map(|&(s, k)| PolicyEntry {
                key: k,
                score: s,
                floor_pinned: false,
            })
            .collect();
        vram_evict_order.reverse();

        let mut ram_evict_order: Vec<PolicyEntry> = Vec::new();
        let mut ram_floor = 0usize;
        for &(_, k) in &rank {
            if !resident(k, TIER_RAM) || resident(k, TIER_VRAM) {
                continue;
            }
            if ram_floor < cfg.ram_experts_min {
                ram_floor += 1;
                continue;
            }
            ram_evict_order.push(PolicyEntry {
                key: k,
                score: self.states[k].score,
                floor_pinned: false,
            });
        }
        ram_evict_order.reverse();
        PolicyOutput {
            vram_order,
            vram_evict_order,
            ram_evict_order,
        }
    }
}

/// One entry in a policy ordering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PolicyEntry {
    /// Flat expert key.
    pub key: usize,
    /// Effective score at the pass.
    pub score: f64,
    /// Pinned by the tier floor — the executor must not evict.
    pub floor_pinned: bool,
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
            vram_experts_min: 1,
            ram_slots: 8,
            ram_experts_min: 1,
            ..Default::default()
        };
        Policy::new(cfg, 1, u64::MAX, u64::MAX).unwrap()
    }

    fn seeded() -> Policy {
        let cfg = PlacementConfig {
            layers: 1,
            experts_per_layer: 8,
            vram_experts_max: 4,
            vram_experts_min: 1,
            ram_slots: 8,
            ram_experts_min: 1,
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
        assert_eq!(
            (c.half_life_steps, c.theta_hi, c.theta_lo, c.min_dwell_steps),
            (512.0, 1.0, 0.4, 256)
        );
    }

    #[test]
    fn new_rejects_vram_floor_that_exceeds_budget() {
        let cfg = PlacementConfig {
            layers: 1,
            experts_per_layer: 8,
            vram_experts_max: 8,
            vram_experts_min: 4,
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
    fn new_rejects_ram_floor_that_exceeds_budget() {
        let cfg = PlacementConfig {
            layers: 1,
            experts_per_layer: 8,
            ram_experts_min: 4,
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
    fn new_rejects_bad_counts() {
        let a = PlacementConfig {
            layers: 1,
            experts_per_layer: 8,
            ram_experts_min: 3,
            ram_slots: 2,
            ..Default::default()
        };
        assert!(Policy::new(a, 1, u64::MAX, 1 << 40).is_err());
        let b = PlacementConfig {
            layers: 1,
            experts_per_layer: 8,
            vram_experts_min: 5,
            vram_experts_max: 4,
            ..Default::default()
        };
        let err = Policy::new(b, 1, 1 << 40, 1 << 40).unwrap_err();
        assert_eq!(err.max_bytes_per_expert, 0);
        assert!(err.to_string().contains("count"));
    }

    #[test]
    fn new_accepts_feasible_config() {
        let cfg = PlacementConfig {
            layers: 1,
            experts_per_layer: 8,
            vram_experts_max: 8,
            vram_experts_min: 2,
            ram_slots: 6,
            ram_experts_min: 2,
            ..Default::default()
        };
        let p = Policy::new(cfg, 1000, 4000, 8000).unwrap();
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
        p.on_access(1, 0);
        let decayed = p.score(1, 512);
        assert!(decayed < 1.0 / std::f64::consts::E + 1e-9, "{decayed}");
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
    fn promotion_requires_theta_hi_and_dwell() {
        let mut p = seeded();
        p.apply_transition(2, Tier::Ram, true, 0);
        for s in 0..300 {
            if s % 2 == 0 {
                p.on_access(2, s);
            }
        }
        let out = p.pass(300);
        assert!(out.vram_order.iter().any(|e| e.key == 2));
        p.apply_transition(3, Tier::Ram, true, 0);
        let out = p.pass(301);
        assert!(out.vram_order.iter().all(|e| e.key != 3));
    }

    #[test]
    fn eviction_candidates_respect_hysteresis_floor_and_cap() {
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
        assert!(
            out.vram_evict_order.iter().all(|e| e.key != 0),
            "floor-pinned top resident is never evicted"
        );
        assert!(out.vram_order.len() <= 4, "budget cap respected");
        assert_eq!(out.vram_order[0].key, 0, "best-first ordering");
        assert!(out.vram_order[0].floor_pinned);
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
    fn demotion_requires_theta_lo_when_budget_has_slack() {
        let cfg = PlacementConfig {
            layers: 1,
            experts_per_layer: 8,
            vram_experts_max: 8,
            vram_experts_min: 0,
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
            out.vram_evict_order.iter().any(|e| e.key == 0)
                && out.vram_evict_order.iter().any(|e| e.key == 1),
            "sub-θ_lo residents evictable: {out:?}"
        );
        assert!(
            out.vram_evict_order.iter().all(|e| e.key != 2),
            "θ_lo..θ_hi band is protected when budget has slack"
        );
    }

    #[test]
    fn ram_eviction_order_excludes_vram_resident_and_floor() {
        let mut p = seeded();
        p.apply_transition(1, Tier::Ram, true, 0);
        p.apply_transition(2, Tier::Ram, true, 0);
        p.apply_transition(1, Tier::Vram, true, 1);
        p.on_access(2, 9999); // top ram score → ram floor pin
        let out = p.pass(10_000);
        assert!(out.ram_evict_order.iter().all(|e| e.key != 1), "vram-resident excluded");
        assert!(
            out.ram_evict_order.iter().all(|e| e.key != 2),
            "ram floor-pinned excluded"
        );
        p.apply_transition(7, Tier::Ram, true, 0);
        let out = p.pass(10_001);
        assert!(
            out.ram_evict_order.iter().any(|e| e.key == 7),
            "cold unpinned ram expert evictable"
        );
    }
}
