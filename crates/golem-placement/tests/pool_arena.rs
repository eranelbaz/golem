//! mlock slot arena acceptance (§5.4): locked pages, disjoint slots. Fill and
//! read go through the residency capability tokens (§6.6): `&mut` needs
//! a `FillClaim`, shared reads need a `Lease` — neither is constructible
//! outside the residency map / pager, so the round trip runs through
//! [`ExpertPool`].

// cfg(loom) is set via RUSTFLAGS by the T6 leg; see primitives.rs.
#![allow(unexpected_cfgs)]

use golem_gguf::ExpertKind;
use golem_placement::pool::primitives::{Arc, arc_dyn};
use golem_placement::pool::{
    Arena, ExpertPool, P3Stats, PolicySnapshot, SlotClass, SnapshotHandle,
};

/// Deterministic 1-byte-per-expert source: expert `e` of layer 0 reads as
/// `0xAA + e`; other layers fail (out of scope).
struct ByteSource;

fn plane_len() -> u64 {
    (16 * 32 * 4) as u64
}

impl golem_placement::pool::ExpertSource for ByteSource {
    fn read_expert(
        &self,
        layer: u32,
        _kind: ExpertKind,
        expert: u32,
        into: &mut [u8],
    ) -> std::io::Result<()> {
        if layer != 0 {
            return Err(std::io::Error::new(std::io::ErrorKind::NotFound, "layer"));
        }
        into.fill(0xAA + expert as u8);
        Ok(())
    }

    fn plane(&self, _layer: u32, _kind: ExpertKind, _expert: u32) -> Option<(u64, u64)> {
        Some((0, plane_len()))
    }
}

// Sequential unit test over loom-primitive pool types: needs no exploration
// (the t6 model covers the interleavings) and cannot run outside `model()`.
#[test]
#[cfg(not(loom))]
fn arena_locks_and_slots_are_disjoint() {
    let slot_size = 16 * 32 * 4; // n_ff_exp * n_embd * F32 (tiny-moe plane)
    let pool = ExpertPool::new(
        Arena::new(&[SlotClass {
            slot_size,
            slots: 4,
        }])
        .expect("mlock may fail on low RLIMIT_MEMLOCK"),
        arc_dyn(std::sync::Arc::new(ByteSource)),
        Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::test_empty()))),
        Arc::new(P3Stats::new()),
        1, // n_layers
        2, // n_experts
    );

    // Page in experts 0 and 1 of layer 0; each fills its own slot.
    let mut buf = vec![0u8; plane_len() as usize];
    pool.ensure_range(true, 0, ExpertKind::Gate, 0, 0, &mut |s| {
        buf.copy_from_slice(s)
    })
    .unwrap();
    assert!(buf.iter().all(|&b| b == 0xAA), "expert 0 bytes");
    pool.ensure_range(true, 0, ExpertKind::Gate, 1, 1, &mut |s| {
        buf.copy_from_slice(s)
    })
    .unwrap();
    assert!(
        buf.iter().all(|&b| b == 0xAB),
        "expert 1 bytes (0xAB): {:02X?}",
        &buf[..8]
    );

    // Warm re-read: expert 0 still reads 0xAA — slot 1's fill did not clobber
    // it (disjointness through the capability-token API).
    pool.ensure_range(true, 0, ExpertKind::Gate, 0, 0, &mut |s| {
        buf.copy_from_slice(s)
    })
    .unwrap();
    assert!(buf.iter().all(|&b| b == 0xAA), "expert 0 bytes intact");
}

#[test]
fn arena_rejects_zero() {
    assert!(Arena::new(&[]).is_err());
    assert!(
        Arena::new(&[SlotClass {
            slot_size: 0,
            slots: 1
        }])
        .is_err()
    );
    assert!(
        Arena::new(&[SlotClass {
            slot_size: 4096,
            slots: 0
        }])
        .is_err()
    );
}

#[test]
#[cfg(not(loom))]
fn class_regions_are_disjoint_and_sized_per_class() {
    let a = Arena::new(&[
        SlotClass {
            slot_size: 4096,
            slots: 3,
        },
        SlotClass {
            slot_size: 6144,
            slots: 2,
        },
    ])
    .expect("mlock may fail on low RLIMIT_MEMLOCK");
    assert_eq!(a.slot_count(), 5);
    assert_eq!(a.class_range(0), 0..3);
    assert_eq!(a.class_range(1), 3..5);
    assert_eq!(a.as_ptr_range().1, 3 * 4096 + 2 * 6144);
    assert_eq!((a.slot_size_of(2), a.slot_size_of(3)), (4096, 6144));
    assert_eq!((a.class_of(2), a.class_of(3)), (0, 1));
}
