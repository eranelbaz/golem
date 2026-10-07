//! P2 acceptance, reworked to the ABI v5 routing semantics: the merged
//! routing readback books recency lock-free, the eviction order reaches C
//! through the per-ctx triple buffers (`golem_policy_order`), the
//! `copy_done` flush publishes synchronously, the background publisher
//! catches dirty layers, and the epoch gate makes pushes skip (never block)
//! when no buffer is free. Drives the engine-registered trampolines over the
//! tiny-MoE fixture on the CPU backend.
#![allow(unsafe_code)]
mod common;

use common::expert_tensor as xt;
use common::tiny_moe_path;
use golem_engine::config::{EngineConfig, VramOrder};
use golem_engine::p3_glue::{self, SessionHooks};
use golem_llama_sys::raw;
use std::time::{Duration, Instant};

fn pooled_config() -> EngineConfig {
    EngineConfig {
        model_path: tiny_moe_path(),
        ram_experts_gib: 1,
        device_cache_experts: 4,
        // These legs assert publisher mechanics with recency expectations.
        vram_order: VramOrder::Recency,
        ..EngineConfig::default()
    }
}

/// One routing readback for blk.1 with `ids` at the current step.
fn route(hooks: &SessionHooks, ids: &[i32]) {
    p3_glue::routing(
        hooks.payload_ptr(),
        xt::BLK,
        std::ptr::null(),
        ids.as_ptr(),
        ids.len() as i64,
        xt::N_EXPERTS,
    );
}

/// `copy_done` with the ctx attached so the flush can push orders.
fn copy_done(hooks: &SessionHooks) {
    p3_glue::copy_done(hooks.payload_ptr())
}

/// The current evict-first order C would consume for the gate tensor.
///
/// # Safety
/// `gate` must be a live ggml tensor with an expert name.
unsafe fn policy_order(hooks: &SessionHooks, gate: *mut raw::ggml_tensor) -> Option<Vec<i32>> {
    // SAFETY: live ctx + live tensor with a golem-recognized expert name.
    let p = unsafe { raw::golem_policy_order(hooks.ctx_ptr(), gate.cast(), xt::N_EXPERTS) };
    if p.is_null() {
        return None;
    }
    // SAFETY: the acquired triple-buffer slot holds n_experts entries and is
    // stable until this layer's next publish (same contract C relies on).
    Some(unsafe { std::slice::from_raw_parts(p, xt::N_EXPERTS as usize) }.to_vec())
}

#[test]
fn routing_semantics_flush_publisher_and_epoch_gate() {
    let hooks = p3_glue::enable(&pooled_config()).expect("enable");
    let (_ctx, gate) = xt::gate_tensor();

    // Nothing pushed yet → C consumes the identity order fallback (NULL).
    // SAFETY: live tensor.
    assert!(
        unsafe { policy_order(&hooks, gate) }.is_none(),
        "order must be NULL before the first push (identity fallback in C)"
    );

    // Step 1 routes expert 2, step 2 routes expert 4: recency puts 4 after 2.
    route(&hooks, &[2]);
    hooks.advance_decode_step();
    route(&hooks, &[4]);
    copy_done(&hooks); // bounded flush; the publisher may also win the claim

    // SAFETY: live tensor.
    let order = wait_order(&hooks, gate);
    assert_eq!(order.len(), 8);
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..8).collect::<Vec<i32>>(), "permutation");
    assert_eq!(order[7], 4, "most recent expert evicted last: {order:?}");
    assert_eq!(order[6], 2, "earlier-step expert next-to-last: {order:?}");

    // Publisher catches a later dirty layer without any flush. Each probe
    // advances the epoch (the sched does this per step entry): reader pins
    // from an older epoch must never block a publisher slot.
    hooks.advance_decode_step();
    route(&hooks, &[6]);
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        // SAFETY: FFI on a live ctx.
        unsafe { raw::golem_policy_order_epoch_advance(hooks.ctx_ptr()) };
        // SAFETY: live tensor (publishes the reader epoch on acquire).
        let order = unsafe { policy_order(&hooks, gate) }.expect("order already published");
        if order[7] == 6 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "background publisher never published the new recency: {order:?}"
        );
        std::thread::sleep(Duration::from_micros(200));
    }

    // Epoch gate: acquire (pin the current slot at a FIXED epoch), publish,
    // acquire again... after three pins no slot is free and the push SKIPS
    // (returns 0) instead of blocking or clobbering a reader-held buffer.
    for _ in 0..6 {
        hooks.advance_decode_step();
        route(&hooks, &[7]);
        copy_done(&hooks);
        // SAFETY: live tensor (pins the current slot at the fixed epoch).
        let _ = unsafe { policy_order(&hooks, gate) };
    }
    assert!(
        hooks.order_pushes_failed() > 0,
        "exhausted triple buffers must skip, not clobber a reader-held slot"
    );

    // The gate clears once the reader epoch advances (C bumps it per step
    // entry; the seam exists for non-sched drivers).
    // SAFETY: FFI on a live ctx.
    unsafe { raw::golem_policy_order_epoch_advance(hooks.ctx_ptr()) };
    hooks.advance_decode_step();
    route(&hooks, &[1]);
    copy_done(&hooks);
    let order = wait_order(&hooks, gate);
    assert_eq!(order[7], 1, "recency updated: {order:?}");
    assert_eq!(
        hooks.routing_calls(),
        10,
        "one routing call per ids tensor per step"
    );
}

/// Waits (bounded) for the layer's order to be published and returns it:
/// the dirty claim may be won by the publisher thread microseconds before
/// the `copy_done` flush, so publication can land one poll later.
fn wait_order(hooks: &SessionHooks, gate: *mut raw::ggml_tensor) -> Vec<i32> {
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        // SAFETY: live ctx + live tensor.
        if let Some(order) = unsafe { policy_order(hooks, gate) } {
            return order;
        }
        assert!(
            Instant::now() < deadline,
            "order never published after the flush"
        );
        std::thread::sleep(Duration::from_micros(100));
    }
}
