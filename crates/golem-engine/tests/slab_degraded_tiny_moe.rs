//! A device-slab allocation that fails after load (ABI v12) fails the turn,
//! never the process: the engine resets the cache, halves K (≥ top-k) and
//! the next turn serves with every expert op still on the GPU (I2). On the
//! target, VRAM taken by another process after load is the trigger; here the
//! substrate's injection seam forces it. Engine-registered hooks (pooled
//! session); HIP only — CPU builds never reach the slab path.
#![cfg(feature = "hip-tests")]
#![allow(unsafe_code)] // FFI test seam (slab failure injection, cache reset)
mod common;

use golem_engine::error::EngineError;
use golem_engine::session::{ChatTurn, Session, expert_placement};
use golem_engine::slot::TurnRequest;
use xtask::tiny_moe::dims::{N_EXPERT, N_EXPERT_USED};

#[test]
fn slab_alloc_failure_degrades_and_keeps_serving() {
    let mut cfg = common::offload::offload_config();
    cfg.ram_experts_gib = 1;
    cfg.device_cache_experts = N_EXPERT;
    cfg.n_predict = 8;
    let mut s = Session::new(&cfg).expect("pooled offload session");
    let msgs = [ChatTurn {
        role: "user",
        content: "hello",
    }];
    let ctx = s.p3_ctx_ptr().expect("pooled ⇒ live ctx");
    // SAFETY: FFI on the session's live ctx between turns: dropping the
    // slabs forces a reallocation on the next compute, which then fails.
    unsafe {
        golem_llama_sys::raw::golem_reset_device_cache(ctx);
        golem_llama_sys::raw::golem_inject_slab_alloc_failure(ctx, 1);
    }
    let err = s
        .turn(&TurnRequest::new(&msgs), &mut |_| {})
        .expect_err("injected slab failure must fail the turn");
    let k = (N_EXPERT / 2).max(N_EXPERT_USED);
    assert!(
        matches!(err, EngineError::DeviceCacheDegraded(got) if got == k),
        "{err}"
    );
    assert_eq!(s.p3_hooks().unwrap().device_cache_experts, k);
    // SAFETY: plain counter read on the live ctx.
    assert_eq!(
        unsafe { golem_llama_sys::raw::golem_slab_alloc_failures(ctx) },
        1
    );
    let before = expert_placement();
    let out = s
        .turn(&TurnRequest::new(&msgs), &mut |_| {})
        .expect("degraded session keeps serving");
    assert!(out.completion_tokens > 0, "no tokens after degrade");
    let after = expert_placement();
    assert_eq!(after.on_cpu, before.on_cpu, "expert math left the GPU");
}
