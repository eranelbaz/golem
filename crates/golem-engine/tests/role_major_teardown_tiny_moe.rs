//! Role-major page-in (§3.4 P3) teardown through the engine-registered
//! trampolines over the tiny-MoE pool on the CPU backend: a gate-only miss
//! splits the unit, its down read fails, a `Read` prefetcher parks on the
//! poisoned key, and dropping the hooks (`SessionHooks::drop`: settle, then
//! reader stop, then prefetch stop) completes within a deadline.
#![allow(unsafe_code)]
mod common;

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use common::{expert_tensor as xt, tiny_moe_path};
use golem_engine::config::{EngineConfig, PrefetchMode};
use golem_engine::p3_glue;

const DEADLINE: Duration = Duration::from_secs(10);
/// Kind code of the gate plane in the miss list (golem-hooks.h).
const GATE: i32 = 1;
const EXPERT: i32 = 3;

#[test]
fn failed_down_read_then_drop_completes_within_deadline() {
    let hooks = p3_glue::enable(&EngineConfig {
        model_path: tiny_moe_path(),
        ram_experts_gib: 1,
        device_cache_experts: 4,
        prefetch: PrefetchMode::Read,
        role_major_pagein: true,
        ..EngineConfig::default()
    })
    .expect("pooled role-major enable");
    hooks.set_scoring(true); // Session::new arms it after the self-check
    let payload = hooks.payload_ptr();
    let (_ctx, gate) = xt::gate_tensor();
    let stats = hooks.state_stats();

    // Gate-only miss: the unit splits, its down read queues and fails.
    hooks.fail_next_down_read_for_test();
    let mut out: [*const u8; 1] = [std::ptr::null()];
    let rc = p3_glue::ensure_host_resident(
        payload,
        gate.cast(),
        [GATE].as_ptr(),
        [EXPERT].as_ptr(),
        1,
        out.as_mut_ptr().cast(),
        None,
        std::ptr::null_mut(),
    );
    assert_eq!(rc, 0, "gate/up handed out before the down read");
    assert!(!out[0].is_null());
    assert_eq!(stats.split_units.load(Ordering::Relaxed), 1, "unit split");

    // The prefetcher reads the predicted key and parks on its fill gate.
    p3_glue::predict(payload, xt::BLK - 1, xt::BLK, [EXPERT].as_ptr(), 1);

    let dropped = std::sync::mpsc::channel();
    let t0 = Instant::now();
    std::thread::spawn(move || {
        drop(hooks);
        let _ = dropped.0.send(());
    });
    dropped
        .1
        .recv_timeout(DEADLINE)
        .expect("SessionHooks drop hung on the poisoned unit");
    assert!(t0.elapsed() < DEADLINE);
    assert_eq!(
        stats.source_errors.load(Ordering::Relaxed),
        1,
        "the failed down read counted once"
    );
}
