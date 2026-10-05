#![allow(unsafe_code)]
//! P2/P4 binding contract: device cache budget setter + stats block.
//!
//! DRIFT vs. task brief: bindgen renders `golem_device_cache_stats` OPAQUE
//! (`_unused: [u8; 0]`) — the header only forward-declares it; the member
//! definition lives in ggml-backend.cpp with `std::atomic<uint64_t>` members
//! (bytes_copied, bytes_skipped, evictions — repr(C)-contiguous, 3×u64,
//! lock-free 8-byte per the C++ TU comment). Field access by name is
//! impossible; counters are read through the returned pointer as
//! `[AtomicU64; 3]` instead of `AtomicU64::from_ptr(&(*p).evictions)`.

use golem_llama_sys::raw;
use std::sync::atomic::{AtomicU64, Ordering};

#[test]
fn golem_p2_device_cache_stats_roundtrip() {
    // Opaque to bindgen (zero-sized marker struct), not a laid-out record.
    assert_eq!(std::mem::size_of::<raw::golem_device_cache_stats>(), 0);

    // SAFETY: FFI call, no invariants (<= 0 = unset budget).
    unsafe { raw::golem_set_device_cache_experts(0) };
    let p = unsafe { raw::golem_get_device_cache_stats() };
    assert!(!p.is_null());

    // Layout contract guaranteed by the C++ definition: 3 contiguous
    // lock-free u64 counters (bytes_copied, bytes_skipped, evictions).
    let stats = unsafe { &*(p.cast::<[AtomicU64; 3]>()) };
    // Fresh process: counters at zero (nextest isolates per-test processes;
    // no decode ran in this one).
    assert_eq!(stats[0].load(Ordering::Relaxed), 0); // bytes_copied
    assert_eq!(stats[1].load(Ordering::Relaxed), 0); // bytes_skipped
    assert_eq!(stats[2].load(Ordering::Relaxed), 0); // evictions
}

#[test]
fn golem_hooks_abi_version_matches_header() {
    assert_eq!(
        golem_llama_sys::hooks_abi_version(),
        raw::GOLEM_HOOKS_ABI_VERSION as i32
    );
}

#[test]
fn golem_p1_expert_placement_stats_roundtrip() {
    // Zero asserts below need a fresh process (no graph built yet).
    assert_eq!(
        std::env::var("NEXTEST_EXECUTION_MODE").as_deref(),
        Ok("process-per-test"),
        "run under cargo nextest: this test needs its own process"
    );
    assert_eq!(std::mem::size_of::<raw::golem_expert_placement_stats>(), 0);
    // SAFETY: FFI getter, no invariants; returns a process-lifetime static, never null.
    let p = unsafe { raw::golem_get_expert_placement_stats() };
    assert!(!p.is_null());
    // SAFETY: layout contract from the C++ definition (opaque to bindgen): 2
    // contiguous lock-free u64 counters, on_gpu then on_cpu; process-lifetime static.
    let stats = unsafe { &*(p.cast::<[AtomicU64; 2]>()) };
    assert_eq!(stats[0].load(Ordering::Relaxed), 0);
    assert_eq!(stats[1].load(Ordering::Relaxed), 0);
}
