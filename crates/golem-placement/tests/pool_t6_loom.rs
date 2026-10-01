//! T6 (§12.2): model-check the pool's lock discipline over (access, page-in,
//! evict, copy) interleavings. Provenance assertions run INSIDE each spawned
//! thread — post-join checks can only catch final-state, not transient torn
//! reads. 1 unit slot / 2 experts forces evict-vs-page-in races with a
//! minimal loom state space. Runs in BOTH modes: std executes one pass, loom
//! explores the interleavings (no `#[ignore]`).

use golem_placement::pool::primitives::arc_dyn;
use golem_placement::pool::primitives::{self, Arc};
use golem_placement::pool::{
    Arena, ExpertKind, ExpertPool, P3Stats, PolicySnapshot, PoolGeometry, SlotClass,
    SnapshotHandle, UnitLayout,
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
    ) -> io::Result<()> {
        into.fill(plane_byte(kind, expert));
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
        layers: vec![UnitLayout {
            stride: [plane; 3],
            offset: [0, plane, 2 * plane],
        }],
        n_experts: 2,
        expert_bytes: 0,
    };
    ExpertPool::new(
        arena,
        arc_dyn(std::sync::Arc::new(ProvenanceSource)),
        Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::test_empty()))),
        Arc::new(P3Stats::new()),
        geometry,
        vec![0; 1],
    )
}

#[test]
fn t6_access_pagein_evict_copy_no_torn_reads() {
    primitives::model(|| {
        let pool = loom_pool();
        let p2 = pool.clone();
        let pagein = primitives::spawn(move || {
            let mut copy = |bytes: &[u8]| {
                assert!(
                    bytes.iter().all(|&b| b == plane_byte(ExpertKind::Gate, 1)),
                    "torn slot under lease"
                )
            };
            let _ = p2.ensure_range(true, 0, ExpertKind::Gate, 1, 1, &mut copy);
        });
        let mut copy = |bytes: &[u8]| {
            assert!(
                bytes.iter().all(|&b| b == plane_byte(ExpertKind::Gate, 0)),
                "torn slot under lease"
            )
        };
        let _ = pool.ensure_range(true, 0, ExpertKind::Gate, 0, 0, &mut copy);
        pagein.join().unwrap();
        // Post-join: residency ⊆ slot provenance (final-state sanity only).
        for e in 0..2u32 {
            if pool.residency.is_resident(0, e) {
                let lease = pool.lease_for_test(0, e);
                assert_eq!(
                    pool.slot_bytes_under(&lease)[0],
                    plane_byte(ExpertKind::Gate, e)
                );
            }
        }
    });
}

/// Same-unit concurrent claims: exactly one thread fills; the other must park
/// on the fill-completion gate and then read FULLY written bytes — no torn
/// mid-fill reads under lease.
#[test]
fn t6_same_key_concurrent_claim_no_torn_fill() {
    primitives::model(|| {
        let pool = loom_pool();
        let p2 = pool.clone();
        let second = primitives::spawn(move || {
            let mut copy = |bytes: &[u8]| {
                assert!(
                    bytes.iter().all(|&b| b == plane_byte(ExpertKind::Gate, 1)),
                    "torn mid-fill read"
                )
            };
            let _ = p2.ensure_range(true, 0, ExpertKind::Gate, 1, 1, &mut copy);
        });
        let mut copy = |bytes: &[u8]| {
            assert!(
                bytes.iter().all(|&b| b == plane_byte(ExpertKind::Gate, 1)),
                "torn mid-fill read"
            )
        };
        let _ = pool.ensure_range(true, 0, ExpertKind::Gate, 1, 1, &mut copy);
        second.join().unwrap();
    });
}

#[test]
fn t6_leased_expert_never_evicted() {
    primitives::model(|| {
        let pool = loom_pool();
        let p2 = pool.clone();
        let evictor = primitives::spawn(move || {
            let mut copy =
                |bytes: &[u8]| assert!(bytes.iter().all(|&b| b == plane_byte(ExpertKind::Gate, 1)));
            let _ = p2.ensure_range(true, 0, ExpertKind::Gate, 1, 1, &mut copy);
        });
        let mut copy = |bytes: &[u8]| {
            assert!(
                bytes.iter().all(|&b| b == plane_byte(ExpertKind::Gate, 0)),
                "lease violated: slot refilled during copy"
            )
        };
        let _ = pool.ensure_range(true, 0, ExpertKind::Gate, 0, 0, &mut copy);
        evictor.join().unwrap();
    });
}

/// Gate and Down of one expert claimed concurrently: one fill, both planes
/// read their own bytes; nobody observes a half-filled unit. Both leases are
/// held across the join, so the second claimer must ride the first fill (an
/// evict-and-refill between the two would be legal only after a release).
#[test]
fn t6_two_planes_of_one_unit_one_fill() {
    primitives::model(|| {
        let pool = loom_pool();
        let p2 = pool.clone();
        let t = primitives::spawn(move || {
            let (_, leases) = p2
                .ensure_range_leased(true, 0, ExpertKind::Down, 1, 1)
                .unwrap();
            leases
        });
        let (_, mine) = pool
            .ensure_range_leased(true, 0, ExpertKind::Gate, 1, 1)
            .unwrap();
        let theirs = t.join().unwrap();
        // One slot serves the unit; both planes carry their own fill bytes.
        let slots: Vec<u32> = mine.iter().chain(theirs.iter()).map(|l| l.slot()).collect();
        assert!(slots.windows(2).all(|w| w[0] == w[1]), "one slot per unit");
        let bytes = pool.arena.slot(slots[0], &mine[0]);
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
