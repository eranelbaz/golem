//! P3 acceptance: counters move across a cold → warm drive of the
//! engine-registered trampolines (the cold step pages the routed unit in;
//! the warm step hits the RAM pool and the claim stays one-per-unit), and a
//! pooled `Session` still refuses a second live session with
//! `EngineError::PoolInit`. Drives the trampolines directly over the
//! tiny-MoE fixture on the CPU backend; the per-token FFI-call arithmetic on
//! HIP lives in the Task 8 evidence leg.
#![allow(unsafe_code)]
mod common;

use common::expert_tensor as xt;
use common::tiny_moe_path;
use golem_engine::config::EngineConfig;
use golem_engine::error::EngineError;
use golem_engine::p3_glue::{self, SessionHooks};
use golem_engine::session::{ChatTurn, Session};
use golem_llama_sys::raw;
use std::sync::atomic::Ordering;

fn pooled_config() -> EngineConfig {
    EngineConfig {
        model_path: tiny_moe_path(),
        ram_experts_gib: 1,
        device_cache_experts: 4,
        ..EngineConfig::default()
    }
}

fn turn(prompt: &str) -> [ChatTurn<'_>; 1] {
    [ChatTurn {
        role: "user",
        content: prompt,
    }]
}

fn drive_step(hooks: &SessionHooks, gate: *mut raw::ggml_tensor) {
    let ids: [i32; 2] = [3, 5];
    let mut out: [*const u8; 3] = [std::ptr::null(); 3];
    let kinds = [1, 2, 3];
    let miss_ids = [3, 3, 3];
    let payload = hooks.payload_ptr();
    p3_glue::routing(
        payload,
        xt::BLK,
        std::ptr::null(),
        ids.as_ptr(),
        2,
        xt::N_EXPERTS,
    );
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
    assert!(out.iter().all(|p| !p.is_null()), "all planes filled");
}

#[test]
fn p3_counters_move_cold_to_warm_and_live_session_is_refused() {
    let hooks = p3_glue::enable(&pooled_config()).expect("enable");
    let (_ctx, gate) = xt::gate_tensor();
    let stats = hooks.state_stats();

    // Cold: the routed unit (expert 3, all three planes) pages in exactly
    // once; the batched miss list claims it ONCE per step despite three
    // plane misses (§6.6 claim-once).
    drive_step(&hooks, gate);
    assert_eq!(
        stats.demand_slow_pageins.load(Ordering::Relaxed),
        1,
        "cold step pages the unit in exactly once"
    );
    assert_eq!(stats.demand_claims.load(Ordering::Relaxed), 1);

    // Warm: the unit is resident — no page-in, and the ABI v5 call shape
    // stays one routing + one batched ensure + one copy_done per step.
    drive_step(&hooks, gate);
    assert_eq!(
        stats.demand_slow_pageins.load(Ordering::Relaxed),
        1,
        "warm step must not page experts in"
    );
    assert_eq!(hooks.routing_calls(), 2);
    assert_eq!(hooks.ensure_calls(), 2);
    assert_eq!(hooks.copy_done_calls(), 2);
    drop(hooks);

    // Session-level: a pooled session constructs, drops, and a second
    // sequential one constructs again — per-session hooks, no
    // one-model-per-process refusal (the sequential-cycle proof; the
    // parallelSessions leg is the HIP H-10 test).
    let mut on = Session::new(&pooled_config()).expect("first pooled session");
    assert!(on.p3_stats().is_some());
    let _ = on.chat_turn(&turn("Hello"), &mut |_| {}).unwrap();
    drop(on);

    // While a session IS live, a second pooled construction is refused: a
    // pool enabled under a live session would re-route its decodes.
    let live = Session::new(&pooled_config()).expect("live session");
    match Session::new(&pooled_config()) {
        Err(EngineError::PoolInit(msg)) => {
            assert!(
                msg.contains("session already exists"),
                "unexpected refusal: {msg}"
            );
        }
        Err(e) => panic!("second session must be refused as PoolInit, got: {e}"),
        Ok(_) => panic!("a second pooled session while one is live must be refused"),
    }
    drop(live);

    let again = Session::new(&pooled_config()).expect("sequential pooled session works");
    drop(again);
}
