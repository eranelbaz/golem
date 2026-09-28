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

    // SAFETY: FFI call, no invariants (0 = full capacity).
    unsafe { raw::golem_set_device_cache_budget(0) };
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
