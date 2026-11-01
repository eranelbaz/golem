//! ABI v17: the batched `ensure_host_resident` trampoline reports the
//! RAM-resident hits through `hits_ready` before the blocking page-in of the
//! misses, and books the post-hand-off span into the per-layer ensure disk
//! wait. Drives the engine-registered trampolines directly over the tiny-MoE
//! fixture on the CPU backend.
#![allow(unsafe_code)]
mod common;

use common::expert_tensor as xt;
use common::tiny_moe_path;
use golem_engine::config::EngineConfig;
use golem_engine::p3_glue;
use std::ffi::c_void;

/// What `record_hits` saw: calls and non-null (hit) pointers per call.
#[derive(Default)]
struct Seen {
    calls: usize,
    non_null: usize,
}

/// Six-entry miss list (expert 3 then 5, three planes each), matching the
/// ensure below.
const N_MISS: usize = 6;

extern "C" fn record_hits(ud: *mut c_void, out_ptrs: *const *const c_void) {
    // SAFETY: `ud` is the test's `&mut Seen`, live across the ensure call;
    // the trampoline passes the `N_MISS`-long out array.
    let (seen, ptrs) = unsafe {
        (
            &mut *ud.cast::<Seen>(),
            std::slice::from_raw_parts(out_ptrs, N_MISS),
        )
    };
    seen.calls += 1;
    seen.non_null += ptrs.iter().filter(|p| !p.is_null()).count();
}

#[test]
fn ensure_reports_ram_hits_before_disk_misses() {
    let hooks = p3_glue::enable(&EngineConfig {
        model_path: tiny_moe_path(),
        ram_experts_gib: 1,
        device_cache_experts: 4,
        ..EngineConfig::default()
    })
    .expect("enable");
    let payload = hooks.payload_ptr();
    let (_ctx, gate) = xt::gate_tensor();

    // Warm expert 3 (no callback: the v16 path); expert 5 stays cold.
    let mut planes: [*const u8; 3] = [std::ptr::null(); 3];
    let rc = p3_glue::ensure_host_resident(
        payload,
        gate.cast(),
        [1, 2, 3].as_ptr(),
        [3, 3, 3].as_ptr(),
        3,
        planes.as_mut_ptr().cast(),
        None,
        std::ptr::null_mut(),
    );
    assert_eq!(rc, 0, "warm-up ensure paged in");
    p3_glue::copy_done(payload);

    let before = hooks.ensure_disk_wait_ns(xt::POOL_LAYER);
    let mut seen = Seen::default();
    let mut out: [*const u8; N_MISS] = [std::ptr::null(); N_MISS];
    let rc = p3_glue::ensure_host_resident(
        payload,
        gate.cast(),
        [1, 2, 3, 1, 2, 3].as_ptr(),
        [3, 3, 3, 5, 5, 5].as_ptr(),
        N_MISS as i32,
        out.as_mut_ptr().cast(),
        Some(record_hits),
        std::ptr::from_mut(&mut seen).cast(),
    );
    assert_eq!(rc, 0, "ensure paged in");
    p3_glue::copy_done(payload);

    assert_eq!(seen.calls, 1, "hits_ready runs exactly once");
    assert_eq!(seen.non_null, 3, "hits_ready sees only expert 3's planes");
    assert!(
        out.iter().all(|p| !p.is_null()),
        "all planes filled on return"
    );
    let after = hooks.ensure_disk_wait_ns(xt::POOL_LAYER);
    assert!(
        after > before,
        "a miss books ensure disk wait ({before} → {after} ns)"
    );
}
