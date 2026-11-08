//! T6 (§12.2): model-check the pool's lock discipline over (access, page-in,
//! evict, copy) interleavings. Provenance assertions run INSIDE each spawned
//! thread — post-join checks can only catch final-state, not transient torn
//! reads. 1 unit slot / 2 experts forces evict-vs-page-in races with a
//! minimal loom state space. Runs in BOTH modes: std executes one pass, loom
//! explores the interleavings (no `#[ignore]`).

use golem_placement::pool::primitives::arc_dyn;
use golem_placement::pool::primitives::{self, Arc};
use golem_placement::pool::{
    Arena, DownJobRunner, ExpertKind, ExpertPool, P3Stats, PolicySnapshot, PoolError, PoolGeometry,
    SlotClass, SnapshotHandle, StepClaims, UnitLayout,
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
    loom_pool_slots(1)
}

fn loom_pool_slots(slots: u32) -> Arc<ExpertPool> {
    let plane = 8u64;
    let arena = Arena::new(&[SlotClass {
        slot_size: 3 * plane as usize,
        slots,
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

/// The ABI v5 batched miss-list path (`ensure_batch` + `end_batch_step`)
/// interleaved with a thread-local `StepClaims` claimer: the pool-owned
/// batch table must ride the same shard discipline (one fill per unit,
/// provenance-correct plane pointers), and `end_batch_step` must release
/// its leases so the other thread's claims never deadlock on the 1-slot
/// pool. `Busy` stays the only legal failure for the concurrent claimer.
#[test]
fn t6_ensure_batch_interleaves_with_range_claims() {
    primitives::model(|| {
        let pool = loom_pool();
        let p2 = pool.clone();
        let batch = primitives::spawn(move || {
            let mut out = [std::ptr::null(); 2];
            let misses = [(ExpertKind::Gate, 1u32), (ExpertKind::Down, 1u32)];
            let r = p2.ensure_batch(0, &misses, &mut out);
            if r.is_ok() {
                p2.end_batch_step();
            }
            r
        });
        let mut step = StepClaims::new(pool.clone());
        let mut ptr = [std::ptr::null()];
        match step.ensure_range_leased(0, ExpertKind::Gate, 0, 0, &mut ptr) {
            Ok(()) | Err(PoolError::Busy) => {}
            Err(e) => panic!("unexpected pool error: {e}"),
        }
        let batched = batch.join().unwrap();
        match batched {
            Ok(()) | Err(PoolError::Busy) => {}
            Err(e) => panic!("unexpected pool error: {e}"),
        }
        step.next_step();
    });
}

/// One `ensure_batch` holding fill claims on both units of a 2-slot layer
/// races a step claim on one of them: the shared unit gets one fill and one
/// slot, and every lease is gone once both sides end their step.
#[test]
fn t6_ensure_batch_fills_race_a_claim_on_one_unit() {
    primitives::model(|| {
        let pool = loom_pool_slots(2);
        let p2 = pool.clone();
        let batch = primitives::spawn(move || {
            let mut out = [std::ptr::null(); 3];
            let misses = [
                (ExpertKind::Gate, 0u32),
                (ExpertKind::Down, 1u32),
                (ExpertKind::Up, 0u32),
            ];
            p2.ensure_batch(0, &misses, &mut out).unwrap();
            out[1] as usize - 16
        });
        let mut step = StepClaims::new(pool.clone());
        let mut ptr = [std::ptr::null()];
        step.ensure_range_leased(0, ExpertKind::Gate, 1, 1, &mut ptr)
            .unwrap();
        let bytes = step.leases()[0].bytes();
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
        let batch_slot_base = batch.join().unwrap();
        assert_eq!(
            bytes.as_ptr() as usize,
            batch_slot_base,
            "one slot per unit"
        );
        let s = &pool.stats;
        assert_eq!(
            s.demand_slow_pageins.load(Ordering::Relaxed),
            2,
            "one fill per unit"
        );
        step.next_step();
        pool.end_batch_step();
        for e in 0..2u32 {
            assert!(pool.is_resident(0, e));
            assert_eq!(pool.outstanding_leases(0, e), 0, "lease leaked");
        }
    });
}

/// Demand and prefetch race for the same unit (P17 + §6.5 counters): at most
/// one fill happens, the demand copy never reads torn bytes, and the
/// prefetch counts a hit exactly when its fill served the demand claim. A
/// prefetch that finds the only slot mid-fill under the demand claim skips
/// with `Busy` (it never waits or retries).
#[test]
fn t6_demand_and_prefetch_same_key_one_fill_hit_iff_prefetch_filled() {
    primitives::model(|| {
        let pool = loom_pool();
        let copies = counter();
        let p2 = pool.clone();
        let prefetch = primitives::spawn(move || match p2.ensure_unit(false, 0, 0) {
            Ok(()) | Err(PoolError::Busy) => {}
            Err(e) => panic!("unexpected pool error: {e}"),
        });
        ensure_counted(&pool, 0, &copies, ExpertKind::Gate);
        prefetch.join().unwrap();
        assert_eq!(copies.load(Ordering::Relaxed), 1, "the demand copy ran");
        let s = &pool.stats;
        let (demand, pre) = (
            s.demand_slow_pageins
                .load(std::sync::atomic::Ordering::Relaxed),
            s.prefetch_pageins
                .load(std::sync::atomic::Ordering::Relaxed),
        );
        assert_eq!(demand + pre, 1, "one fill per unit");
        assert_eq!(
            s.prefetch_hits.load(std::sync::atomic::Ordering::Relaxed),
            pre,
            "a prefetch fill is always claimed by the demand that follows it"
        );
    });
}

/// CPU-path resident take + batch release race a demand claim on the other
/// unit of the 1-slot layer: a taken unit is never evicted under its lease
/// (its planes keep their provenance until `release_all`), and once released
/// the demand claim can proceed (`Busy` is the only legal failure).
#[test]
fn t6_resident_take_and_release_race_a_demand_claim() {
    primitives::model(|| {
        let pool = loom_pool();
        pool.ensure_unit(true, 0, 0).unwrap();
        let p2 = pool.clone();
        let taker = primitives::spawn(move || {
            let mut batch = golem_placement::pool::LeaseBatch::new(&p2, 2);
            let (mut ids, mut ptrs) = ([-1i32; 2], [std::ptr::null(); 6]);
            let n = p2.lease_resident_take(0, &[0, 1], 1.0, &mut ids, &mut ptrs, &mut batch);
            for &e in &ids[..n] {
                assert!(p2.outstanding_leases(0, e as u32) >= 1);
                assert!(p2.is_resident(0, e as u32), "leased unit evicted");
            }
            batch.release_all();
        });
        let copies = counter();
        ensure_counted(&pool, 1, &copies, ExpertKind::Gate);
        taker.join().unwrap();
    });
}

fn _lease_batch_is_send() {
    fn send<T: Send>() {}
    send::<golem_placement::pool::LeaseBatch>();
}

/// 2 unit slots over 2 experts of one layer, role-major page-in with a
/// one-unit split cap.
fn role_major_pool(
    source: std::sync::Arc<dyn golem_placement::pool::ExpertSource>,
) -> Arc<ExpertPool> {
    let plane = 8u64;
    let arena = Arena::new(&[SlotClass {
        slot_size: 3 * plane as usize,
        slots: 2,
    }])
    .unwrap();
    ExpertPool::try_new_role_major(
        arena,
        arc_dyn(source),
        Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::empty()))),
        Arc::new(P3Stats::new()),
        PoolGeometry {
            layers: vec![UnitLayout::new([plane; 3], [None; 3]).unwrap()],
            n_experts: 2,
            expert_bytes: 0,
        },
        vec![0],
        1,
    )
    .unwrap()
}

/// Role-major page-in (§3.4 P3), one job: the decode thread splits unit 1
/// (gate handed out, down queued), the reader body runs that single job on
/// its own thread, a concurrent claimer on the same key gets `Busy` or parks
/// on the fill gate, and the decode thread joins before releasing. Every
/// interleaving: the claimer never reads a half-written unit, the unit ends
/// resident in one slot, and no lease leaks.
#[test]
fn t6_role_major_down_join_vs_concurrent_claim() {
    primitives::model(|| {
        let pool = role_major_pool(std::sync::Arc::new(ProvenanceSource));
        let runner = DownJobRunner::attach(&pool).unwrap();
        let mut out = [std::ptr::null()];
        pool.ensure_batch(0, &[(ExpertKind::Gate, 1)], &mut out)
            .unwrap();
        assert_eq!(pool.stats.split_units.load(Ordering::Relaxed), 1);
        let reader = primitives::spawn(move || assert!(runner.run_one()));
        let p = pool.clone();
        let claimer = primitives::spawn(move || {
            let mut copy = |down: &[u8]| {
                assert!(
                    down.iter().all(|&b| b == plane_byte(ExpertKind::Down, 1)),
                    "claimer read a down plane still being written"
                );
            };
            match p.ensure_range(false, 0, ExpertKind::Down, 1, 1, &mut copy) {
                Ok(()) | Err(PoolError::Busy) => {}
                Err(e) => panic!("unexpected pool error: {e}"),
            }
        });
        pool.join_split().unwrap();
        pool.end_batch_step();
        reader.join().unwrap();
        claimer.join().unwrap();
        assert!(pool.is_resident(0, 1));
        assert_eq!(pool.outstanding_leases(0, 1), 0, "lease leaked");
    });
}

/// [`ProvenanceSource`] whose down-plane reads fail.
struct DownFailSource;

impl golem_placement::pool::ExpertSource for DownFailSource {
    fn read_expert(
        &self,
        layer: u32,
        kind: ExpertKind,
        expert: u32,
        into: &mut [u8],
        head: usize,
    ) -> io::Result<()> {
        if kind == ExpertKind::Down {
            return Err(io::Error::other("down read failed"));
        }
        ProvenanceSource.read_expert(layer, kind, expert, into, head)
    }

    fn plane(&self, _layer: u32, _kind: ExpertKind, _expert: u32) -> Option<(u64, u64)> {
        None
    }
}

/// Role-major page-in (§3.4 P3), failing down read: the decode thread
/// splits unit 1, the reader's down read fails, and a concurrent prefetch
/// claim on the key gets `Busy`, parks on the fill gate, or (after the
/// cancel) fills the unit itself and hits the same failure. The decode
/// thread's idle settle poisons and cancels the claim, which wakes the
/// parked claimer. Every interleaving: the claimer returns, the key ends
/// non-resident and no lease leaks.
#[test]
fn t6_role_major_failed_down_settle_wakes_claimer() {
    primitives::model(|| {
        let pool = role_major_pool(std::sync::Arc::new(DownFailSource));
        let runner = DownJobRunner::attach(&pool).unwrap();
        let mut out = [std::ptr::null()];
        pool.ensure_batch(0, &[(ExpertKind::Gate, 1)], &mut out)
            .unwrap();
        let reader = primitives::spawn(move || assert!(runner.run_one()));
        let p = pool.clone();
        let claimer = primitives::spawn(move || match p.ensure_unit(false, 0, 1) {
            Err(PoolError::Busy | PoolError::Source { .. }) => {}
            other => panic!("claim on a failed unit: {other:?}"),
        });
        reader.join().unwrap();
        pool.settle_split().unwrap();
        claimer.join().unwrap();
        pool.end_batch_step();
        assert!(!pool.is_resident(0, 1), "no ghost residency");
        assert_eq!(pool.outstanding_leases(0, 1), 0, "lease leaked");
        assert_eq!(pool.fill(), 0.0, "slot back on the free list");
    });
}
