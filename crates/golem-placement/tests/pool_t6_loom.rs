//! T6 (§12.2): model-check the pool's lock discipline over (access, page-in,
//! evict, copy) interleavings. Provenance assertions run INSIDE each spawned
//! thread — post-join checks can only catch final-state, not transient torn
//! reads. 1 unit slot / 2 experts forces evict-vs-page-in races with a
//! minimal loom state space. Runs in BOTH modes: std executes one pass, loom
//! explores the interleavings (no `#[ignore]`).

use golem_placement::pool::primitives::arc_dyn;
use golem_placement::pool::primitives::{self, Arc};
use golem_placement::pool::{
    Arena, ExpertKind, ExpertPool, P3Stats, PolicySnapshot, PoolError, PoolGeometry, SlotClass,
    SnapshotHandle, StepClaims, UnitLayout,
};

use golem_placement::pool::primitives::Ordering;
use std::io;

/// Source writing plane-identifying bytes: slot provenance is checkable per
/// plane (`0x10 × (kind + 1) + expert`).
struct ProvenanceSource;

fn plane_byte(kind: ExpertKind, expert: u32) -> u8 {
    0x10 * (kind.ordinal() as u8 + 1) + expert as u8
}

impl golem_placement::pool::ExpertSource for ProvenanceSource {
    fn read_expert(
        &self,
        _layer: u32,
        kind: ExpertKind,
        expert: u32,
        into: &mut [u8],
        head: usize,
    ) -> io::Result<()> {
        into[head..].fill(plane_byte(kind, expert));
        Ok(())
    }

    fn plane(&self, _layer: u32, _kind: ExpertKind, _expert: u32) -> Option<(u64, u64)> {
        None
    }
}

/// 1 unit slot (3 planes × 8 B) over 2 experts of one layer.
fn loom_pool() -> Arc<ExpertPool> {
    let plane = 8u64;
    let arena = Arena::new(&[SlotClass {
        slot_size: 3 * plane as usize,
        slots: 1,
    }])
    .unwrap();
    let geometry = PoolGeometry {
        layers: vec![UnitLayout::new([plane; 3], [None; 3]).unwrap()],
        n_experts: 2,
        expert_bytes: 0,
    };
    ExpertPool::new(
        arena,
        arc_dyn(std::sync::Arc::new(ProvenanceSource)),
        Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::empty()))),
        Arc::new(P3Stats::new()),
        geometry,
        vec![0; 1],
    )
}

/// Runs one `ensure_range` and counts the copy if it ran. `Busy` is the only
/// legal failure: the 1-slot pool can be fully leased by the other thread. A
/// source error or a swallowed result would make the model vacuous.
fn ensure_counted(pool: &ExpertPool, expert: u32, copies: &Counter, expect: ExpertKind) {
    let mut copy = |bytes: &[u8]| {
        assert!(
            bytes.iter().all(|&b| b == plane_byte(expect, expert)),
            "torn slot under lease"
        );
        copies.fetch_add(1, Ordering::Relaxed);
    };
    match pool.ensure_range(true, 0, ExpertKind::Gate, expert, expert, &mut copy) {
        Ok(()) | Err(PoolError::Busy) => {}
        Err(e) => panic!("unexpected pool error: {e}"),
    }
}

type Counter = Arc<primitives::AtomicUsize>;

fn counter() -> Counter {
    Arc::new(primitives::AtomicUsize::new(0))
}

#[test]
fn t6_access_pagein_evict_copy_no_torn_reads() {
    primitives::model(|| {
        let pool = loom_pool();
        let copies = counter();
        let (p2, c2) = (pool.clone(), copies.clone());
        let pagein = primitives::spawn(move || ensure_counted(&p2, 1, &c2, ExpertKind::Gate));
        ensure_counted(&pool, 0, &copies, ExpertKind::Gate);
        pagein.join().unwrap();
        assert!(
            copies.load(Ordering::Relaxed) >= 1,
            "no copy ran: the model checked nothing"
        );
        // Post-join: residency ⊆ slot provenance (final-state sanity only).
        for e in 0..2u32 {
            if pool.is_resident(0, e) {
                let mut step = StepClaims::new(pool.clone());
                step.ensure_range_leased(0, ExpertKind::Gate, e, e, &mut [std::ptr::null()])
                    .unwrap();
                assert_eq!(step.leases()[0].bytes()[0], plane_byte(ExpertKind::Gate, e));
            }
        }
    });
}

/// Same-unit concurrent claims: exactly one thread fills; the other must park
/// on the fill-completion gate and then read FULLY written bytes — no torn
/// mid-fill reads under lease. One unit, one slot: neither claim can be Busy.
#[test]
fn t6_same_key_concurrent_claim_no_torn_fill() {
    primitives::model(|| {
        let pool = loom_pool();
        let copies = counter();
        let (p2, c2) = (pool.clone(), copies.clone());
        let second = primitives::spawn(move || ensure_counted(&p2, 1, &c2, ExpertKind::Gate));
        ensure_counted(&pool, 1, &copies, ExpertKind::Gate);
        second.join().unwrap();
        assert_eq!(
            copies.load(Ordering::Relaxed),
            2,
            "both claimers of one key must copy"
        );
    });
}

#[test]
fn t6_leased_expert_never_evicted() {
    primitives::model(|| {
        let pool = loom_pool();
        let copies = counter();
        let (p2, c2) = (pool.clone(), copies.clone());
        let evictor = primitives::spawn(move || ensure_counted(&p2, 1, &c2, ExpertKind::Gate));
        ensure_counted(&pool, 0, &copies, ExpertKind::Gate);
        evictor.join().unwrap();
        assert!(
            copies.load(Ordering::Relaxed) >= 1,
            "no copy ran: the model checked nothing"
        );
    });
}

/// Gate and Down of one expert claimed concurrently: one fill, both planes
/// read their own bytes; nobody observes a half-filled unit. Both steps'
/// leases are held across the join, so the second claimer must ride the first fill (an
/// evict-and-refill between the two would be legal only after a release).
#[test]
fn t6_two_planes_of_one_unit_one_fill() {
    primitives::model(|| {
        let pool = loom_pool();
        let p2 = pool.clone();
        let t = primitives::spawn(move || {
            let mut step = StepClaims::new(p2);
            step.ensure_range_leased(0, ExpertKind::Down, 1, 1, &mut [std::ptr::null()])
                .unwrap();
            step
        });
        let mut mine = StepClaims::new(pool.clone());
        mine.ensure_range_leased(0, ExpertKind::Gate, 1, 1, &mut [std::ptr::null()])
            .unwrap();
        let theirs = t.join().unwrap();
        // One slot serves the unit; both planes carry their own fill bytes.
        let slots: Vec<u32> = mine
            .leases()
            .iter()
            .chain(theirs.leases())
            .map(|l| l.slot())
            .collect();
        assert!(slots.windows(2).all(|w| w[0] == w[1]), "one slot per unit");
        let bytes = mine.leases()[0].bytes();
        assert!(
            bytes[..8]
                .iter()
                .all(|&b| b == plane_byte(ExpertKind::Gate, 1))
        );
        assert!(
            bytes[16..]
                .iter()
                .all(|&b| b == plane_byte(ExpertKind::Down, 1))
        );
        assert_eq!(
            pool.stats.demand_slow_pageins.load(Ordering::Relaxed),
            1,
            "the Down claim must ride the gate fill, not page the unit again"
        );
        drop(mine);
        drop(theirs);
    });
}

/// A step claim pins the unit for all its planes: a concurrent evictor on
/// the 1-slot pool can never take the slot between the Gate claim and the
/// Down reuse, and the Down plane reads its own fill bytes.
#[test]
fn t6_step_claim_pins_the_unit_across_planes() {
    primitives::model(|| {
        let pool = loom_pool();
        let copies = counter();
        let (p2, c2) = (pool.clone(), copies.clone());
        let evictor = primitives::spawn(move || ensure_counted(&p2, 1, &c2, ExpertKind::Gate));
        let mut step = StepClaims::new(pool.clone());
        let mut ptr = [std::ptr::null()];
        match step.ensure_range_leased(0, ExpertKind::Gate, 0, 0, &mut ptr) {
            Ok(()) => {
                step.ensure_range_leased(0, ExpertKind::Down, 0, 0, &mut ptr)
                    .unwrap();
                let bytes = step.leases()[0].bytes();
                assert!(
                    bytes[16..]
                        .iter()
                        .all(|&b| b == plane_byte(ExpertKind::Down, 0))
                );
                assert_eq!(ptr[0], bytes[16..].as_ptr());
                assert_eq!(pool.stats.demand_claims.load(Ordering::Relaxed), 1);
            }
            // The evictor leased the only slot first.
            Err(PoolError::Busy) => assert!(step.leases().is_empty()),
            Err(e) => panic!("unexpected pool error: {e}"),
        }
        step.next_step();
        evictor.join().unwrap();
    });
}
