//! Journal warm-start seeding (§6.3, M2.5 task 3): replay a recorded trace
//! into a fresh policy with a step gap and check the seeded ranking against
//! an independent replay of the same trace. The xorshift trace generator is
//! duplicated from `tests/properties.rs` (its helpers are private to that
//! binary).

use golem_placement::{Journal, JournalHeader, PlacementConfig, Policy, journal::JOURNAL_CAP};

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
const STEPS: u64 = 2048;
const GAP: u64 = 128;

fn fixture() -> Policy {
    let cfg = PlacementConfig {
        layers: 2,
        experts_per_layer: 128,
        vram_experts_max: 32,
        vram_experts_min: 8,
        ram_slots: 160,
        ram_experts_min: 8,
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
