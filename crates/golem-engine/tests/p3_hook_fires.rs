//! P3 acceptance, ABI v5: the batched `ensure_host_resident` trampoline
//! claims each routed unit once per step, hands out per-plane pool pointers
//! and holds the leases until `copy_done`; the merged routing readback books
//! recency lock-free; `SessionHooks` is fully owned per session (two
//! sequential enable/drop cycles each decode; two cycles run concurrently on
//! separate threads). Drives the engine-registered trampolines directly over
//! the tiny-MoE fixture on the CPU backend — no GPU split needed.
#![allow(unsafe_code)]
mod common;

use common::expert_tensor as xt;
use common::tiny_moe_path;
use golem_engine::config::EngineConfig;
use golem_engine::p3_glue;
use golem_llama_sys::raw;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Pooled config for tiny-moe on the CPU backend: explicit K (no VRAM query
/// exists off HIP), 1 GiB pool. A failed precondition (RLIMIT_MEMLOCK below
/// the pool) FAILS — an early return reads as a pass (AGENTS.md Evidence).
fn pooled_config() -> EngineConfig {
    EngineConfig {
        model_path: tiny_moe_path(),
        ram_experts_gib: 1,
        device_cache_experts: 4,
        ..EngineConfig::default()
    }
}

/// One simulated decode step through the engine-registered trampolines:
/// routing readback for blk.1 with routed ids [3, 5], then the batched
/// ensure for the three planes of expert 3, then copy_done. `payload` is
/// [`SessionHooks::payload_ptr`]; `gate` the synthetic gate tensor.
fn drive_step(payload: *mut libc::c_void, gate: *mut raw::ggml_tensor) -> [*const u8; 3] {
    let ids: [i32; 2] = [3, 5];
    p3_glue::routing(
        payload,
        xt::BLK,
        std::ptr::null(),
        ids.as_ptr(),
        2,
        xt::N_EXPERTS,
    );
    let mut out: [*const u8; 3] = [std::ptr::null(); 3];
    let kinds = [1, 2, 3]; // gate, up, down
    let miss_ids = [3, 3, 3];
    let rc = p3_glue::ensure_host_resident(
        payload,
        gate.cast(),
        kinds.as_ptr(),
        miss_ids.as_ptr(),
        3,
        out.as_mut_ptr().cast(),
    );
    assert_eq!(rc, 0, "ensure paged in");
    p3_glue::copy_done(payload);
    out
}

#[test]
fn ensure_pages_once_claims_once_and_counts_calls() {
    let hooks = p3_glue::enable(&pooled_config()).expect("enable");
    let payload = hooks.payload_ptr();
    let (_ctx, gate) = xt::gate_tensor();

    let out = drive_step(payload, gate);
    assert!(
        out.iter().all(|p| !p.is_null()),
        "all three plane pointers must be filled: {out:?}"
    );
    assert_ne!(out[0], out[1], "distinct planes live at distinct offsets");

    // Claim-once: three plane misses of ONE unit → one demand claim, one
    // unit page-in; the routing readback and copy_done each fired once.
    let stats = hooks.state_stats();
    assert_eq!(stats.demand_claims.load(Ordering::Relaxed), 1);
    assert_eq!(stats.demand_slow_pageins.load(Ordering::Relaxed), 1);
    assert_eq!(hooks.routing_calls(), 1);
    assert_eq!(hooks.ensure_calls(), 1);
    assert_eq!(hooks.copy_done_calls(), 1);

    // A second identical step: units resident → fast hits, no new page-ins;
    // the copy_done above released the leases so the re-claim succeeds.
    // SAFETY: as above.
    let _ = drive_step(payload, gate);
    assert_eq!(stats.demand_slow_pageins.load(Ordering::Relaxed), 1);
    assert_eq!(hooks.ensure_calls(), 2);
}

#[test]
fn two_sequential_enable_drop_cycles_each_decode() {
    for cycle in 0..2 {
        let hooks = p3_glue::enable(&pooled_config())
            .unwrap_or_else(|e| panic!("cycle {cycle}: enable must succeed, got {e}"));
        let payload = hooks.payload_ptr();
        let (_ctx, gate) = xt::gate_tensor();
        let out = drive_step(payload, gate);
        assert!(
            out.iter().all(|p| !p.is_null()),
            "cycle {cycle}: pooled decode through the fresh ctx"
        );
        drop(hooks);
        // The drop must fully unwind (unregister-join → ctx free): the next
        // cycle's enable + ctx create prove it — golem_ctx_free aborts if
        // hooks were still registered.
    }
    assert!(!p3_glue::hooks_registered(), "no hooks outlive the cycles");
}

#[test]
fn two_sessionhooks_run_their_hook_paths_concurrently() {
    // Two direct enables on separate threads, each driving its own ctx +
    // pool state concurrently: the payload state is per-session (no
    // process-global hook state exists), so both full hook paths pass.
    let failures = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for t in 0..2 {
            let failures = &failures;
            s.spawn(move || {
                let hooks = match p3_glue::enable(&pooled_config()) {
                    Ok(h) => h,
                    Err(e) => {
                        eprintln!("thread {t}: enable failed: {e}");
                        failures.fetch_add(1, Ordering::SeqCst);
                        return;
                    }
                };
                let payload = hooks.payload_ptr();
                let (_ctx, gate) = xt::gate_tensor();
                for _ in 0..8 {
                    let out = drive_step(payload, gate);
                    if out.iter().any(|p| p.is_null()) {
                        failures.fetch_add(1, Ordering::SeqCst);
                        return;
                    }
                }
                drop(hooks);
            });
        }
    });
    assert_eq!(failures.load(Ordering::SeqCst), 0);
}

/// A counting allocator that only counts while the current thread opts in,
/// so background threads (order publisher, prefetcher) don't pollute the
/// steady-state zero-allocation assertion.
static COUNTING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static COUNT: AtomicUsize = AtomicUsize::new(0);

struct CountingAlloc;
// SAFETY: forwards to System; the counter is only an observation.
unsafe impl std::alloc::GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            COUNT.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: delegated to the system allocator with the same layout.
        unsafe { std::alloc::System.alloc(layout) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        // SAFETY: delegated with the layout the allocation came in with.
        unsafe { std::alloc::System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: CountingAlloc = CountingAlloc;

#[test]
fn trampolines_allocate_nothing_in_steady_state() {
    let hooks = p3_glue::enable(&pooled_config()).expect("enable");
    let payload = hooks.payload_ptr();
    let (_ctx, gate) = xt::gate_tensor();
    // Warm: first sight registers (allocates), first page-ins fill slots.
    let _ = drive_step(payload, gate);

    COUNT.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    for _ in 0..200 {
        let _ = drive_step(payload, gate);
    }
    COUNTING.store(false, Ordering::Relaxed);
    let allocs = COUNT.load(Ordering::Relaxed);
    assert_eq!(
        allocs, 0,
        "batched ensure + routing + copy_done must not allocate in steady state"
    );
}
