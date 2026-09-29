//! Journal warm-start seeding (§6.3, M2.5 task 3): replay a recorded trace
//! into a fresh policy with a step gap and check the seeded ranking against
//! an independent replay of the same trace. The xorshift trace generator is
//! duplicated from `tests/properties.rs` (its helpers are private to that
//! binary).

use golem_placement::{
    Journal, JournalHeader, PlacementConfig, Policy, Tier, journal::JOURNAL_CAP,
};

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
const HOT: usize = 32;
const STEPS: u64 = 2048;
const GAP: u64 = 512;

/// Brief fixture (authoritative): vram floor+top (64) + ram floor (32) = 96
/// pre-promoted slots ≥ HOT (32) — the whole hot set plus slack fits by
/// construction.
fn fixture() -> Policy {
    let cfg = PlacementConfig {
        layers: 2,
        experts_per_layer: 128,
        vram_experts_max: 64,
        vram_experts_min: 8,
        ram_slots: 128,
        ram_experts_min: 32,
        ..Default::default()
    };
    Policy::new(cfg, 1, u64::MAX, u64::MAX).unwrap()
}

#[test]
fn seeded_ranking_matches_trace_hot_set() {
    let mut p1 = fixture();
    p1.attach_journal(Journal::new(JOURNAL_CAP));
    let mut xor = XorShift(0x9E3779B97F4A7C15);
    let mut counts = [0usize; NUM_KEYS];
    let mut trace = Vec::with_capacity(STEPS as usize);
    for step in 0..STEPS {
        let key = if xor.next() % 100 < 60 {
            (xor.next() as usize) % HOT
        } else {
            (xor.next() as usize) % NUM_KEYS
        };
        p1.on_access(key, step);
        counts[key] += 1;
        trace.push((step, key));
    }
    let journal = p1.detach_journal().unwrap();
    let header = JournalHeader {
        layers: 2,
        experts_per_layer: 128,
        half_life_steps: 512.0,
    };

    let mut p2 = fixture();
    p2.seed_from_journal(&header, &journal, GAP).unwrap();

    // Never-accessed keys: exact zero after seeding.
    let never: Vec<usize> = (0..NUM_KEYS).filter(|&k| counts[k] == 0).collect();
    assert!(!never.is_empty(), "fixture must leave keys untouched");
    for &k in &never {
        assert_eq!(p2.score(k, 0), 0.0, "key {k} never accessed but scored");
    }
    // Every hot-set key outscores every never-accessed key.
    for (h, &hits) in counts.iter().enumerate().take(HOT) {
        assert!(hits > 0, "hot key {h} never hit by the trace");
        for &n in &never {
            assert!(
                p2.score(h, 0) > p2.score(n, 0),
                "hot key {h} does not outrank never-accessed key {n}"
            );
        }
    }

    // Top-16 by seeded score vs top-16 by an independent replay of the
    // same trace (§6.3 math redone here, not via Policy): ≥ 14 overlap.
    // Replay at the original step (no gap); the gap is applied exactly
    // once at the sweep to the common horizon T+gap, so recent accesses
    // dominate — hit counts are NOT the comparator (exact set equality
    // is flaky on a stochastic trace). Net decay of an access at step s
    // is e^(−(T+gap−s)/H), matching seed_from_journal exactly.
    let mut expected = [0f64; NUM_KEYS];
    let mut last_touch = [0u64; NUM_KEYS];
    let h = header.half_life_steps;
    for &(step, key) in &trace {
        expected[key] = expected[key] * (-((step - last_touch[key]) as f64) / h).exp() + 1.0;
        last_touch[key] = step;
    }
    let horizon = (STEPS - 1) + GAP; // journal.last_step() + gap_steps
    for k in 0..NUM_KEYS {
        expected[k] *= (-((horizon - last_touch[k]) as f64) / h).exp();
    }
    let mut by_score: Vec<usize> = (0..NUM_KEYS).collect();
    by_score.sort_by(|&a, &b| p2.score(b, 0).total_cmp(&p2.score(a, 0)));
    let top_scored: Vec<usize> = by_score.into_iter().take(16).collect();
    let mut by_expected: Vec<usize> = (0..NUM_KEYS).collect();
    by_expected.sort_by(|&a, &b| expected[b].total_cmp(&expected[a]).then(a.cmp(&b)));
    let top_expected: std::collections::HashSet<usize> = by_expected.into_iter().take(16).collect();
    let overlap = top_scored
        .iter()
        .filter(|k| top_expected.contains(k))
        .count();
    assert!(
        overlap >= 14,
        "top-16 overlap {overlap}/16 below threshold: by score {top_scored:?}, independent {top_expected:?}"
    );
}

/// The window both runs are counted over (same window or the comparison
/// is meaningless).
const EARLY_WINDOW: usize = 256;

/// Local executor (P3 stand-in): replays a trace against a policy and
/// counts slow paths — accesses to keys resident in neither VRAM nor RAM
/// (residency changes only via `apply_transition`, so a check right after
/// `on_access` is the demand-page-in decision). Slow paths are tagged by
/// key category: hot (< HOT) vs tail.
struct Executor {
    policy: Policy,
    hot_slow: usize,
    tail_slow: usize,
}

impl Executor {
    fn new(policy: Policy) -> Self {
        Self {
            policy,
            hot_slow: 0,
            tail_slow: 0,
        }
    }

    fn run(&mut self, trace: &[(u64, usize)]) {
        for &(step, key) in trace {
            self.policy.on_access(key, step);
            let resident =
                self.policy.residency(key, Tier::Vram) || self.policy.residency(key, Tier::Ram);
            if !resident {
                if key < HOT {
                    self.hot_slow += 1;
                } else {
                    self.tail_slow += 1;
                }
            }
        }
    }

    fn total_slow(&self) -> usize {
        self.hot_slow + self.tail_slow
    }
}

/// Issue #23 acceptance: on a replayed trace, journal seeding + warm plan
/// must cut early P3 slow paths ≥25% (measured ≥4× here), never slow-path
/// on a pre-promoted hot key, and never be worse than cold anywhere in
/// the window. Both runs are counted over the SAME window 0..256.
#[test]
fn warm_start_reduces_early_slow_paths() {
    let (journal, trace) = gen_trace(0x9E3779B97F4A7C15);

    // WARM: fresh policy, seed_from_journal at the standard gap, apply the
    // warm plan via the executor at step 0 (P2 page-in + promote).
    let mut warm_policy = seeded_policy(&journal);
    let (vram, ram) = warm_policy.warm_plan();
    for &k in &vram {
        warm_policy.apply_transition(k, Tier::Vram, true, 0);
    }
    for &k in &ram {
        warm_policy.apply_transition(k, Tier::Ram, true, 0);
    }
    let mut warm = Executor::new(warm_policy);
    warm.run(&trace[..EARLY_WINDOW]);

    // COLD: same trace, same disk-cold start, same window.
    let mut cold = Executor::new(fixture());
    cold.run(&trace[..EARLY_WINDOW]);

    let (warm_early, cold_early) = (warm.total_slow(), cold.total_slow());
    // Derived bound (first-touch arithmetic, moves with the fixture): at
    // step 0 nothing is resident, so cold slow paths ≈ distinct
    // first-touches in 0..256 ≈ 32 hot + ~82 tail ≈ 114; warm ≈ tail keys
    // outside the top-96 seeded set ≈ 18.
    assert!(
        warm_early * 4 <= cold_early * 3,
        "warm ({warm_early}) must be ≤75% of cold ({cold_early})"
    );
    assert_eq!(
        warm.hot_slow, 0,
        "pre-promoted hot keys must never slow-path"
    );
    assert!(
        warm_early <= cold_early,
        "warm must not be worse anywhere in the window"
    );
}

/// Record a deterministic trace into a journal (same generator as the test
/// above) and return the journal plus the full (step, key) trace.
fn gen_trace(seed: u64) -> (Journal, Vec<(u64, usize)>) {
    let mut p = fixture();
    p.attach_journal(Journal::new(JOURNAL_CAP));
    let mut xor = XorShift(seed);
    let mut trace = Vec::with_capacity(STEPS as usize);
    for step in 0..STEPS {
        let key = if xor.next() % 100 < 60 {
            (xor.next() as usize) % HOT
        } else {
            (xor.next() as usize) % NUM_KEYS
        };
        p.on_access(key, step);
        trace.push((step, key));
    }
    (p.detach_journal().unwrap(), trace)
}

/// Fresh policy seeded from `journal` at the standard gap.
fn seeded_policy(journal: &Journal) -> Policy {
    let header = JournalHeader {
        layers: 2,
        experts_per_layer: 128,
        half_life_steps: 512.0,
    };
    let mut p = fixture();
    p.seed_from_journal(&header, journal, GAP).unwrap();
    p
}

/// Independent §6.2 replay of `trace` (not via `Policy`): net seeded score
/// per key, gap applied once at the sweep to horizon T+gap.
fn independent_scores(trace: &[(u64, usize)]) -> [f64; NUM_KEYS] {
    let mut expected = [0f64; NUM_KEYS];
    let mut last_touch = [0u64; NUM_KEYS];
    let h = 512.0; // fixture half_life
    for &(step, key) in trace {
        expected[key] = expected[key] * (-((step - last_touch[key]) as f64) / h).exp() + 1.0;
        last_touch[key] = step;
    }
    let horizon = (STEPS - 1) + GAP;
    for k in 0..NUM_KEYS {
        expected[k] *= (-((horizon - last_touch[k]) as f64) / h).exp();
    }
    expected
}

#[test]
fn warm_plan_slices_seeded_ranking_into_floors() {
    let (journal, trace) = gen_trace(0x9E3779B97F4A7C15);
    let mut p = seeded_policy(&journal);
    let (vram, ram) = p.warm_plan();
    let cfg = p.config().clone();

    // Exact shape: vram_experts_max VRAM keys, ram_experts_min RAM keys.
    assert_eq!(vram.len(), cfg.vram_experts_max);
    assert_eq!(ram.len(), cfg.ram_experts_min);
    for &k in &ram {
        assert!(!vram.contains(&k), "key {k} appears in both floors");
    }

    // Both lists follow pass()'s total order: score desc, ties by
    // ascending key index; ram continues where vram stops (next-best).
    let mut rank_of = |key: usize| (p.score(key, 0), key);
    let mut check_order = |ks: &[usize]| {
        for pair in ks.windows(2) {
            let (a, b) = (rank_of(pair[0]), rank_of(pair[1]));
            assert!(
                a.0 > b.0 || (a.0 == b.0 && a.1 < b.1),
                "keys {} and {} out of order",
                pair[0],
                pair[1]
            );
        }
    };
    check_order(&vram);
    check_order(&ram);
    let last_vram = rank_of(vram[cfg.vram_experts_max - 1]);
    let first_ram = rank_of(ram[0]);
    assert!(
        last_vram.0 > first_ram.0 || (last_vram.0 == first_ram.0 && last_vram.1 < first_ram.1),
        "ram floor starts before the vram ranking ends"
    );

    // The plan's VRAM list tracks an independent §6.2 replay of the same
    // trace: near-total overlap on the top vram_experts_max.
    let expected = independent_scores(&trace);
    let mut by_expected: Vec<usize> = (0..NUM_KEYS).collect();
    by_expected.sort_by(|&a, &b| expected[b].total_cmp(&expected[a]).then(a.cmp(&b)));
    let top_expected: std::collections::HashSet<usize> =
        by_expected.into_iter().take(cfg.vram_experts_max).collect();
    let overlap = vram.iter().filter(|k| top_expected.contains(k)).count();
    assert!(
        overlap >= cfg.vram_experts_max - 2,
        "warm plan vram overlap {overlap}/{} with independent replay",
        cfg.vram_experts_max
    );
}

#[test]
fn warm_plan_pre_promotion_zeroes_demand_pageins() {
    let (journal, trace) = gen_trace(0x9E3779B97F4A7C15);
    let window = 256usize;

    // Warm run: seed, apply the plan at step 0 (pool executes page-in +
    // promote and reports back via apply_transition), replay the window.
    let mut warm = seeded_policy(&journal);
    let (vram, ram) = warm.warm_plan();
    let mut promoted = std::collections::HashSet::new();
    for &k in &vram {
        warm.apply_transition(k, Tier::Vram, true, 0);
        promoted.insert(k);
    }
    for &k in &ram {
        warm.apply_transition(k, Tier::Ram, true, 0);
        promoted.insert(k);
    }
    let resident = |p: &Policy, k: usize| p.residency(k, Tier::Vram) || p.residency(k, Tier::Ram);
    let mut warm_demand = 0usize;
    for &(step, key) in trace.iter().take(window) {
        warm.on_access(key, step);
        if promoted.contains(&key) && !resident(&warm, key) {
            warm_demand += 1;
        }
    }

    // Cold run: same trace from the same disk-cold start, no seeding.
    let mut cold = fixture();
    let mut cold_demand = 0usize;
    for &(step, key) in trace.iter().take(window) {
        cold.on_access(key, step);
        if promoted.contains(&key) && !resident(&cold, key) {
            cold_demand += 1;
        }
    }

    assert_eq!(
        warm_demand, 0,
        "pre-promoted keys must never demand-page-in"
    );
    assert!(
        cold_demand >= HOT,
        "cold run page-ins {cold_demand} below hot-set size {HOT} in the same window"
    );
}
