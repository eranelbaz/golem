//! Randomized-trace acceptance properties of the placement policy:
//! hottest keys resident · no thrash under oscillation · budget never exceeded.
//!
//! Deterministic xorshift PRNG, no `rand` dependency. The executor applies
//! `pass()` output exactly as P2/P3 would; `misbehave = true` skips VRAM
//! evictions so the property harness proves it detects violations.

use golem_placement::{PlacementConfig, Policy, PolicyOutput, Tier};

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

const NUM_KEYS: usize = 256; // layers 2 × experts 128
const HOT: usize = 16;
const STEPS: u64 = 20_000;
const WARMUP: u64 = 512;
const NEVER: u64 = u64::MAX;

fn fixture() -> Policy {
    let cfg = PlacementConfig {
        layers: 2,
        experts_per_layer: 128,
        vram_experts_max: 32,
        ram_slots: 160, // headroom: 2 hot rotations + uniform tail ≈ 122 — dwell-gated page-ins need eviction supply
        ..Default::default()
    };
    Policy::new(cfg, 1, u64::MAX, u64::MAX).unwrap()
}

/// Executor stand-in for P2/P3: page-in accessed keys to RAM, evict from RAM
/// along `ram_evict_order` while over `ram_slots`, promote along `vram_order`,
/// evict VRAM along `vram_evict_order` (skipped when misbehaving).
struct Executor {
    misbehave: bool,
    last_promote: [u64; NUM_KEYS],
}

impl Executor {
    fn apply(
        &mut self,
        p: &mut Policy,
        accessed: usize,
        step: u64,
        out: &PolicyOutput,
        violations: &mut Vec<String>,
    ) {
        if !p.residency(accessed, Tier::Ram) {
            p.apply_transition(accessed, Tier::Ram, true, step);
        }
        let mut ram = (0..p.len()).filter(|&k| p.residency(k, Tier::Ram)).count();
        // consume order: evict before promote — promoting into a full device
        // budget is the failure mode P2 must avoid (the pool consumes in this order)
        for e in out.ram_evict_order.iter() {
            if ram <= p.config().ram_slots {
                break;
            }
            if p.residency(e.key, Tier::Ram) {
                p.apply_transition(e.key, Tier::Ram, false, step);
                ram -= 1;
            }
        }
        let mut vram = (0..p.len()).filter(|&k| p.residency(k, Tier::Vram)).count();
        let mut victims: Vec<usize> = out
            .vram_evict_order
            .iter()
            .filter(|e| p.residency(e.key, Tier::Vram))
            .map(|e| e.key)
            .collect();
        for e in out.vram_order.iter() {
            if !p.residency(e.key, Tier::Vram) {
                // evict-before-promote: a real P2 cache cannot promote into a
                // full device budget; when the policy offers no victim
                // (min-dwell), the promotion is deferred, not over-cap.
                while vram >= p.config().vram_experts_max && !self.misbehave {
                    // injected fault: promote without evicting (over-cap)
                    let Some(victim) = victims.pop() else {
                        break;
                    };
                    p.apply_transition(victim, Tier::Vram, false, step);
                    vram -= 1;
                }
                if vram >= p.config().vram_experts_max && !self.misbehave {
                    continue; // no victim offered: defer the promotion
                }
                if step > WARMUP
                    && self.last_promote[e.key] != NEVER
                    && step.saturating_sub(self.last_promote[e.key]) <= p.config().min_dwell_steps
                {
                    violations.push(format!(
                        "step {step}: key {} promoted twice within D_min (prev at {})",
                        e.key, self.last_promote[e.key]
                    ));
                }
                p.apply_transition(e.key, Tier::Vram, true, step);
                self.last_promote[e.key] = step;
                vram += 1;
            }
        }
    }
}

fn run_trace(seed: u64, misbehave: bool) -> (Vec<String>, usize, usize) {
    assert_ne!(seed, 0, "xorshift(0) is degenerate");
    let mut p = fixture();
    let mut xor = XorShift(seed);
    let mut ex = Executor {
        misbehave,
        last_promote: [NEVER; NUM_KEYS],
    };
    let mut violations: Vec<String> = Vec::new();
    let mut max_vram: usize = 0;
    let mut max_ram: usize = 0;
    let mut over_cap_run: u64 = 0;
    let mut prev_top: Option<Vec<usize>> = None;
    // hot-set base offset advances by 1 every 3·D_min steps
    let rotation_period = 3 * p.config().min_dwell_steps;

    for step in 0..STEPS {
        let key = if xor.next() % 100 < 60 {
            (xor.next() as usize) % HOT
        } else {
            (xor.next() as usize) % NUM_KEYS
        } + (step / rotation_period) as usize * HOT;
        let key = key % NUM_KEYS;
        p.on_access(key, step);
        let out = p.pass(step);
        // property 1: the top of the previous pass's VRAM order is still
        // resident (no thrash at the head of the ranking)
        if step > WARMUP {
            if let Some(top) = &prev_top {
                for &k in top {
                    if !p.residency(k, Tier::Vram) {
                        violations.push(format!("step {step}: vram top key {k} lost residency"));
                    }
                }
            }
        }
        prev_top = Some(
            out.vram_order
                .iter()
                .map(|e| e.key)
                .filter(|&k| p.residency(k, Tier::Vram))
                .take(8)
                .collect(),
        );
        ex.apply(&mut p, key, step, &out, &mut violations);
        // property 3: budgets never exceeded
        let vram = (0..p.len()).filter(|&k| p.residency(k, Tier::Vram)).count();
        let ram = (0..p.len()).filter(|&k| p.residency(k, Tier::Ram)).count();
        max_vram = max_vram.max(vram);
        max_ram = max_ram.max(ram);
        // min-dwell makes cap corrections lag promotions (a displaced young
        // resident is not evictable until D_min expires), so over-cap bursts
        // are legal; only a burst persisting ≥ D_min steps is a violation.
        if vram > p.config().vram_experts_max {
            over_cap_run += 1;
            if over_cap_run >= p.config().min_dwell_steps {
                violations.push(format!(
                    "step {step}: vram residents {vram} > cap {} for {over_cap_run} steps",
                    p.config().vram_experts_max
                ));
            }
        } else {
            over_cap_run = 0;
        }
        if ram > p.config().ram_slots {
            violations.push(format!(
                "step {step}: ram residents {ram} > slots {}",
                p.config().ram_slots
            ));
        }
    }
    (violations, max_vram, max_ram)
}

#[test]
fn vram_top_no_thrash_budget_held_across_seeds() {
    for i in 0..5 {
        let seed = 0x9E3779B97F4A7C15 ^ i;
        let (violations, max_vram, max_ram) = run_trace(seed, false);
        assert!(violations.is_empty(), "seed {seed:#x}: {violations:?}");
        assert!(max_vram <= 32 && max_ram <= 160);
    }
}

#[test]
fn executor_catches_injected_budget_violation() {
    // Injected fault: promote without evicting — the harness must catch the
    // resulting over-cap residency.
    let (violations, _, _) = run_trace(0x9E3779B97F4A7C15, true);
    assert!(
        violations.iter().any(|v| v.contains("vram residents")),
        "misbehaving executor must trip the budget property, got {violations:?}"
    );
}
