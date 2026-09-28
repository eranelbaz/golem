//! T6 (§12.2): model-check the pool's lock discipline over (access, page-in,
//! evict, copy) interleavings. Provenance assertions run INSIDE each spawned
//! thread — post-join checks can only catch final-state, not transient torn
//! reads. 1 slot / 2 experts / 1 kind forces evict-vs-page-in races with a
//! minimal loom state space. Runs in BOTH modes: std executes one pass, loom
//! explores the interleavings (no `#[ignore]`).

use golem_placement::pool::primitives::arc_dyn;
use golem_placement::pool::primitives::{self, Arc};
use golem_placement::pool::{
    Arena, ExpertKind, ExpertPool, P3Stats, PolicySnapshot, SnapshotHandle,
};

use std::io;

/// Source writing expert-identifying bytes: slot provenance is checkable.
struct ProvenanceSource;

impl golem_placement::pool::ExpertSource for ProvenanceSource {
    fn read_expert(
        &self,
        _layer: u32,
        _kind: ExpertKind,
        expert: u32,
        into: &mut [u8],
    ) -> io::Result<()> {
        into.fill(expert as u8);
        Ok(())
    }

    fn plane(&self, _layer: u32, _kind: ExpertKind, _expert: u32) -> Option<(u64, u64)> {
        None
    }
}

/// 1-slot pool over 2 experts of one kind.
fn loom_pool() -> Arc<ExpertPool> {
    let slot_size = 8;
    let arena = Arena::new(slot_size, slot_size).unwrap();
    ExpertPool::new(
        arena,
        arc_dyn(std::sync::Arc::new(ProvenanceSource)),
        Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::test_empty()))),
        Arc::new(P3Stats::new()),
        1,
        2,
    )
}

#[test]
fn t6_access_pagein_evict_copy_no_torn_reads() {
    primitives::model(|| {
        let pool = loom_pool();
        let p2 = pool.clone();
        let pagein = primitives::spawn(move || {
            let mut copy = |bytes: &[u8]| assert_eq!(bytes[0], 1, "torn slot under lease");
            let _ = p2.ensure_range(true, 0, ExpertKind::Gate, 1, 1, &mut copy);
        });
        let mut copy = |bytes: &[u8]| assert_eq!(bytes[0], 0, "torn slot under lease");
        let _ = pool.ensure_range(true, 0, ExpertKind::Gate, 0, 0, &mut copy);
        pagein.join().unwrap();
        // Post-join: residency ⊆ slot provenance (final-state sanity only).
        for e in 0..2u32 {
            if pool.residency.is_resident(0, ExpertKind::Gate, e) {
                let lease = pool.lease_for_test(0, ExpertKind::Gate, e);
                assert_eq!(pool.slot_bytes_under(&lease)[0], e as u8);
            }
        }
    });
}

/// Same-key concurrent claims: exactly one thread fills; the other must park
/// on the fill-completion gate and then read FULLY written bytes — no torn
/// mid-fill reads under lease.
#[test]
fn t6_same_key_concurrent_claim_no_torn_fill() {
    primitives::model(|| {
        let pool = loom_pool();
        let p2 = pool.clone();
        let second = primitives::spawn(move || {
            let mut copy =
                |bytes: &[u8]| assert!(bytes.iter().all(|&b| b == 1), "torn mid-fill read");
            let _ = p2.ensure_range(true, 0, ExpertKind::Gate, 1, 1, &mut copy);
        });
        let mut copy = |bytes: &[u8]| assert!(bytes.iter().all(|&b| b == 1), "torn mid-fill read");
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
            let mut copy = |bytes: &[u8]| assert_eq!(bytes[0], 1);
            let _ = p2.ensure_range(true, 0, ExpertKind::Gate, 1, 1, &mut copy);
        });
        let mut copy =
            |bytes: &[u8]| assert_eq!(bytes[0], 0, "lease violated: slot refilled during copy");
        let _ = pool.ensure_range(true, 0, ExpertKind::Gate, 0, 0, &mut copy);
        evictor.join().unwrap();
    });
}
