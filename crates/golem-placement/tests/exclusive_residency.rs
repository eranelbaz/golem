//! Exclusive RAM/VRAM residency (design §6.3 O10): pool occupancy shows no
//! expert in both tiers. Promotion (an H2D-staged batch, released via
//! [`ExpertPool::end_batch_step_exclusive`]) frees the pool slot and marks
//! the unit VRAM-present; the eviction D2H path ([`ExpertPool::claim_d2h`] +
//! [`ExpertPool::commit_d2h`]) restores exactly one RAM copy.

// cfg(loom) is set via RUSTFLAGS by the T6 leg; see primitives.rs.
#![allow(unexpected_cfgs)]

use golem_gguf::ExpertKind;
use golem_placement::pool::primitives::{Arc, arc_dyn};
use golem_placement::pool::{
    Arena, ExpertPool, P3Stats, PolicySnapshot, PoolGeometry, SlotClass, SnapshotHandle, UnitLayout,
};

fn plane_len() -> u64 {
    (16 * 32 * 4) as u64
}

struct ByteSource;

impl golem_placement::pool::ExpertSource for ByteSource {
    fn read_expert(
        &self,
        layer: u32,
        _kind: ExpertKind,
        expert: u32,
        into: &mut [u8],
        head: usize,
    ) -> std::io::Result<()> {
        if layer != 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::NotFound, "layer"));
        }
        into[head..].fill(0xAA + expert as u8);
        Ok(())
    }

    fn plane(&self, _layer: u32, _kind: ExpertKind, _expert: u32) -> Option<(u64, u64)> {
        Some((0, plane_len()))
    }
}

fn pool(slots: u32) -> Arc<ExpertPool> {
    let plane = plane_len();
    ExpertPool::new(
        Arena::new(&[SlotClass {
            slot_size: 3 * plane as usize,
            slots,
        }])
        .expect("mlock may fail on low RLIMIT_MEMLOCK"),
        arc_dyn(std::sync::Arc::new(ByteSource)),
        Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::empty()))),
        Arc::new(P3Stats::new()),
        PoolGeometry {
            layers: vec![UnitLayout::new([plane; 3], [None; 3]).unwrap()],
            n_experts: 4,
            expert_bytes: 0,
        },
        vec![0; 1],
    )
}

#[test]
#[cfg(not(loom))]
fn promoted_unit_leaves_the_pool_tier() {
    let pool = pool(2);
    let mut buf = vec![0u8; plane_len() as usize];
    // Stage expert 0 H2D (the demand ensure path, as one compute does).
    pool.ensure_batch(
        0,
        &[(ExpertKind::Gate, 0)],
        &mut [buf.as_mut_ptr() as *const u8],
    )
    .unwrap();
    assert!(pool.is_resident(0, 0) && !pool.is_vram(0, 0));
    // Promotion lands: release + mark. Occupancy: VRAM only.
    pool.end_batch_step_exclusive();
    assert!(!pool.is_resident(0, 0), "exclusive: pool slot released");
    assert!(pool.is_vram(0, 0), "VRAM presence recorded");
}

#[test]
#[cfg(not(loom))]
fn d2h_restores_exactly_one_ram_copy() {
    let pool = pool(2);
    let mut buf = vec![0u8; plane_len() as usize];
    pool.ensure_batch(
        0,
        &[(ExpertKind::Gate, 1)],
        &mut [buf.as_mut_ptr() as *const u8],
    )
    .unwrap();
    pool.end_batch_step_exclusive();
    assert!(!pool.is_resident(0, 1) && pool.is_vram(0, 1));
    // The slab slot is evicted; the pool takes the D2H.
    pool.clear_vram(0, 1);
    let dst @ [gate, up, down] = match pool.claim_d2h(0, 1, 0.0) {
        golem_placement::pool::D2hIncoming::Dest(dst) => dst,
        _other => panic!("expected a D2H destination"),
    };
    // The "substrate": land the copy, then commit (copy_done ordering).
    let plane = plane_len() as usize;
    // SAFETY: the three destinations are the claimed slot's plane regions,
    // each `plane_len()` bytes (the pool's own layout), returned exclusively
    // to this caller by claim_d2h.
    #[allow(unsafe_code)]
    unsafe {
        std::ptr::copy_nonoverlapping(buf.as_ptr(), gate, plane);
        std::ptr::copy_nonoverlapping(buf.as_ptr(), up, plane);
        std::ptr::copy_nonoverlapping(buf.as_ptr(), down, plane);
    }
    let _ = dst;
    pool.commit_d2h();
    assert!(pool.is_resident(0, 1), "D2H restored the RAM copy");
    assert!(!pool.is_vram(0, 1), "eviction cleared VRAM presence");
    // Occupancy invariant: the unit lives in exactly one tier.
    assert!(pool.is_resident(0, 1) ^ pool.is_vram(0, 1));
}

#[test]
#[cfg(not(loom))]
fn d2h_resident_arm_releases_the_probe_lease() {
    let pool = pool(2);
    let mut buf = vec![0u8; plane_len() as usize];
    // Expert 2 becomes pool-resident and stays (never promoted).
    pool.ensure_batch(
        0,
        &[(ExpertKind::Gate, 2)],
        &mut [buf.as_mut_ptr() as *const u8],
    )
    .unwrap();
    pool.end_batch_step();
    assert!(pool.is_resident(0, 2));
    // A D2H for a key whose RAM copy is current must report Resident AND
    // leave zero outstanding leases (the probe lease is released, not leaked
    // — a leaked lease would pin the slot against release_resident forever).
    assert!(matches!(
        pool.claim_d2h(0, 2, 0.0),
        golem_placement::pool::D2hIncoming::Resident
    ));
    assert_eq!(
        pool.outstanding_leases(0, 2),
        0,
        "claim_d2h Resident leaked the probe's lease"
    );
}

#[test]
#[cfg(not(loom))]
fn device_cache_reset_drops_the_staged_batch_without_releasing_slots() {
    let pool = pool(2);
    let mut buf = vec![0u8; plane_len() as usize];
    // Stage a batch whose copy_done has NOT run yet (leases still held).
    pool.ensure_batch(
        0,
        &[(ExpertKind::Gate, 3)],
        &mut [buf.as_mut_ptr() as *const u8],
    )
    .unwrap();
    assert!(pool.is_resident(0, 3) && !pool.is_vram(0, 3));
    // Between-turns device cache reset: the staged VRAM copies are gone.
    pool.reset_exclusive_batch();
    pool.clear_all_vram();
    // The next compute's copy_done must release nothing and set no VRAM bit:
    // the unit stays exactly once in the pool tier (its RAM copy is current).
    pool.end_batch_step_exclusive();
    assert!(pool.is_resident(0, 3), "RAM copy must survive the reset");
    assert!(!pool.is_vram(0, 3), "no stale VRAM bit after reset");
    assert!(pool.is_resident(0, 3) ^ pool.is_vram(0, 3));
}

#[test]
#[cfg(not(loom))]
fn score_floor_drops_into_a_full_hot_pool() {
    let pool = pool(1); // one slot: the resident victim must score colder
    let scores = vec![5.0f32, 0.0, 0.0, 0.0];
    pool.publish_snapshot(Arc::new(PolicySnapshot::scored(scores, 4)));
    let mut buf = vec![0u8; plane_len() as usize];
    // Expert 0 (score 5.0) fills the only slot.
    pool.ensure_batch(
        0,
        &[(ExpertKind::Gate, 0)],
        &mut [buf.as_mut_ptr() as *const u8],
    )
    .unwrap();
    pool.end_batch_step();
    // An incoming eviction with score 0.0 must not displace the hotter key.
    assert!(matches!(
        pool.claim_d2h(0, 1, 0.0),
        golem_placement::pool::D2hIncoming::Dropped
    ));
    assert!(pool.is_resident(0, 0), "hot victim kept");
}
