#![allow(unsafe_code, missing_docs)]
//! P1 regression: expert metadata queries over a synthetic tensor, via the
//! linked substrate. Unset-hook invariant: metadata fns are pure; register
//! with a NULL ctx (null-guarded, returns a NULL handle) + a NULL-handle
//! unregister restores the unset state.
#[allow(dead_code)] // fixture helpers shared with other suites go unused here
mod common;

use common::*;
use golem_llama_sys::raw::*;

#[test]
fn expert_metadata_derived_from_tensor_dims() {
    unsafe {
        let ctx = test_ggml_ctx();
        let t = new_tensor_3d(&ctx, test_type(), 64, 32, 8, "blk.0.ffn_gate_exps.weight");
        assert_eq!(golem_tensor_expert_count(t.cast()), 8);
        assert_eq!(
            golem_tensor_expert_stride_bytes(t.cast()),
            (64 * 32 * type_size()) as i32
        );
        let named = new_tensor_3d(&ctx, test_type(), 64, 32, 8, "ffn_gate_exps");
        assert_eq!(golem_tensor_expert_count(named.cast()), 8);
        assert_eq!(
            golem_tensor_expert_count(
                new_tensor_3d(&ctx, test_type(), 64, 32, 8, "blk.0.ffn_up_exps.weight").cast()
            ),
            8
        );
        assert_eq!(
            golem_tensor_expert_count(
                new_tensor_3d(&ctx, test_type(), 64, 32, 8, "blk.0.ffn_down_exps.weight").cast()
            ),
            8
        );
        let dense = new_tensor_3d(&ctx, test_type(), 64, 32, 8, "ffn_down");
        assert_eq!(golem_tensor_expert_count(dense.cast()), 0);
        // Loader-created .scale variants are not expert planes (one shared matcher, §6.6).
        let scale = new_tensor_3d(&ctx, test_type(), 64, 32, 8, "blk.0.ffn_gate_exps.scale");
        assert_eq!(golem_tensor_expert_count(scale.cast()), 0);
        assert_eq!(golem_tensor_expert_stride_bytes(dense.cast()), 0);
    }
}

#[test]
fn two_dim_tensor_with_exps_suffix_is_not_expert() {
    unsafe {
        let ctx = test_ggml_ctx();
        let t = new_tensor_2d(&ctx, test_type(), 64, 32, "blk.0.ffn_gate_exps");
        assert_eq!(golem_tensor_expert_count(t.cast()), 0);
        assert_eq!(golem_tensor_expert_stride_bytes(t.cast()), 0);
    }
}

#[test]
fn register_null_ctx_smoke() {
    unsafe {
        // ABI v7: register takes the ctx first; a NULL ctx is null-guarded
        // and returns a NULL handle, whose unregister is a no-op.
        let h = golem_register_hooks(
            std::ptr::null_mut(),
            None,
            None,
            None,
            None,
            std::ptr::null_mut(),
        );
        assert!(h.is_null(), "NULL ctx must not produce a handle");
        golem_unregister_hooks(h);
        // No assertion beyond "does not crash + links": the unset state is
        // byte-identity-proven by the parity dump gate.
    }
}

#[test]
fn ctx_lifecycle_roundtrip() {
    let ctx = golem_llama_sys::GolemCtx::create(2, 8).expect("ctx create");
    // SAFETY: FFI query with a live ctx.
    assert_eq!(unsafe { golem_ctx_generation(ctx.as_ptr()) }, 1);
    assert_eq!(
        unsafe { golem_any_ctx_hooked() },
        0,
        "creating a ctx does not hook anything"
    );
    // Drop frees (no hooks registered, no scheds attached — free is legal).
    drop(ctx);
}

#[test]
fn hooked_guard_unregisters_and_recovers_payload() {
    extern "C" fn nop(
        _payload: *mut std::ffi::c_void,
        _a: *const std::ffi::c_void,
        _b: *const i32,
        _c: *const i32,
        _n: i32,
        _o: *mut *const std::ffi::c_void,
    ) {
    }
    extern "C" fn nop2(
        _payload: *mut std::ffi::c_void,
        _blk: i32,
        _t: *const std::ffi::c_void,
        _ids: *const i32,
        _n: i64,
        _e: i32,
    ) {
    }
    extern "C" fn nop3(_payload: *mut std::ffi::c_void) {}
    let ctx = golem_llama_sys::GolemCtx::create(1, 4).expect("ctx create");
    let ptr = ctx.as_ptr();
    let hooked = ctx.register(
        golem_llama_sys::HookFns {
            ensure: Some(nop),
            routing: Some(nop2),
            copy_done: Some(nop3),
            evict_batch: None,
        },
        std::sync::Arc::new([0u64; 4]),
    );
    assert_eq!(hooked.ctx_ptr(), ptr);
    // SAFETY: FFI query with a live ctx.
    assert_eq!(unsafe { golem_any_ctx_hooked() }, 1, "registered ⇒ hooked");
    // SAFETY: FFI query with a live ctx.
    let gen_ = unsafe { golem_ctx_generation(ptr) };
    assert!(gen_ >= 2, "register bumps the generation: {gen_}");
    // Drop unregisters (joins) and recovers the payload; the ctx is then
    // freed only when the last Arc drops — golem_ctx_free would abort if the
    // hooks were still registered, so a clean exit is the proof.
    drop(hooked);
    assert_eq!(
        unsafe { golem_any_ctx_hooked() },
        0,
        "dropping the guard unregisters"
    );
    // `ctx` was consumed by register: dropping the guard frees the ctx too
    // (last Arc) — a clean exit here is also the ctx_free-after-unregister proof.
}
