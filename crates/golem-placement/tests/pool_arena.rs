//! mlock slot arena acceptance (§5.4): locked pages, disjoint slots. Fill and
//! read go through the residency capability tokens (review H-1): `&mut` needs
//! a `FillClaim`, shared reads need a `Lease` — neither is constructible
//! outside the residency map / pager, so the round trip runs through
//! [`ExpertPool`].

use golem_gguf::ExpertKind;
use golem_placement::pool::primitives::{Arc, arc_dyn};
use golem_placement::pool::{Arena, ExpertPool, P3Stats, PolicySnapshot, SnapshotHandle};

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
        Arena::new(slot_size * 4, slot_size).expect("mlock may fail on low RLIMIT_MEMLOCK"),
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
    assert!(Arena::new(0, 4096).is_err());
}
