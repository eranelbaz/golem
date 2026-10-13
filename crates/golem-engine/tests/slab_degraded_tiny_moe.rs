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
    degrades_and_keeps_serving(false);
}

/// Same with pooled H2D on the dedicated copy stream (async_h2d): the
/// failed compute drains the copy stream before the cache reset frees slabs.
#[test]
fn slab_alloc_failure_degrades_and_keeps_serving_async_h2d() {
    degrades_and_keeps_serving(true);
}

fn degrades_and_keeps_serving(async_h2d: bool) {
    let mut cfg = common::offload::offload_config();
    cfg.async_h2d = async_h2d;
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

#[test]
fn slab_alloc_failure_at_top_k_floor_is_fatal() {
    let mut cfg = common::offload::offload_config();
    cfg.ram_experts_gib = 1;
    cfg.device_cache_experts = N_EXPERT_USED;
    cfg.n_predict = 8;
    let mut s = Session::new(&cfg).expect("pooled offload session");
    assert_eq!(s.p3_hooks().unwrap().device_cache_experts, N_EXPERT_USED);
    let msgs = [ChatTurn {
        role: "user",
        content: "hello",
    }];
    let ctx = s.p3_ctx_ptr().expect("pooled ⇒ live ctx");
    // SAFETY: as above — between turns, on the session's live ctx.
    unsafe {
        golem_llama_sys::raw::golem_reset_device_cache(ctx);
        golem_llama_sys::raw::golem_inject_slab_alloc_failure(ctx, 1);
    }
    let err = s
        .turn(&TurnRequest::new(&msgs), &mut |_| {})
        .expect_err("injected slab failure must fail the turn");
    assert!(matches!(err, EngineError::BackendUnavailable(_)), "{err}");
}

/// Batched (`parallel 2`): the degradation reaches the scheduler's
/// `take_degraded` (→ health Degraded on the server); the sampled slot fails
/// with `DeviceCacheDegraded`, the greedy one replays and finishes.
#[test]
fn batched_slab_alloc_failure_reports_degraded() {
    use golem_engine::{SamplingParams, Scheduler, SlotEvent, SlotJob};
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    let mut cfg = common::offload::offload_config();
    cfg.ram_experts_gib = 1;
    cfg.device_cache_experts = N_EXPERT;
    cfg.n_predict = 16;
    cfg.parallel = 2;
    let mut sch = Scheduler::new(Session::new(&cfg).expect("pooled session")).unwrap();
    let msgs = [ChatTurn {
        role: "user",
        content: "hello",
    }];
    for temperature in [0.0, 1.0] {
        let r = sch.session().render(&TurnRequest::new(&msgs)).unwrap();
        let job = SlotJob {
            tokens: sch.tokenize(&r.prompt).unwrap(),
            sampling: SamplingParams {
                temperature,
                ..SamplingParams::default()
            },
            max_tokens: Some(16),
            stop: Vec::new(),
            cancel: Arc::new(AtomicBool::new(false)),
            rendered: Some(r),
        };
        sch.admit(job).ok().expect("free slot");
    }
    let mut texts = [0usize; 2];
    let mut done: [Option<Result<(), EngineError>>; 2] = [None, None];
    let mut injected = false;
    let mut degraded = None;
    for _ in 0..1000 {
        if !injected && texts.iter().all(|&n| n > 0) {
            let ctx = sch.session().p3_ctx_ptr().expect("pooled ⇒ live ctx");
            // SAFETY: between steps, on the session's live ctx.
            unsafe {
                golem_llama_sys::raw::golem_reset_device_cache(ctx);
                golem_llama_sys::raw::golem_inject_slab_alloc_failure(ctx, 1);
            }
            injected = true;
        }
        let active = sch
            .step(&mut |i, e| match e {
                SlotEvent::Text(_) => texts[i] += 1,
                SlotEvent::Done(_) => done[i] = Some(Ok(())),
                SlotEvent::Failed(e) => done[i] = Some(Err(e)),
            })
            .expect("step");
        degraded = degraded.or(sch.take_degraded());
        if !active {
            break;
        }
    }
    assert!(injected, "slots never both generated");
    let k = (N_EXPERT / 2).max(N_EXPERT_USED);
    assert_eq!(degraded, Some(k));
    assert!(matches!(done[0], Some(Ok(()))), "{:?}", done[0]);
    assert!(
        matches!(done[1], Some(Err(EngineError::DeviceCacheDegraded(got))) if got == k),
        "{:?}",
        done[1]
    );
}
