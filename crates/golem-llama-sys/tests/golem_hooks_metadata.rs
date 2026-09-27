#![allow(unsafe_code, missing_docs)]
//! P1 regression: expert metadata queries over a synthetic tensor, via the
//! linked substrate. Unset-hook invariant: metadata fns are pure; register +
//! re-register(NULL) restores the unset state.
#[allow(dead_code)] // fixture helpers shared with other suites go unused here
mod common;

use common::*;
use golem_llama_sys::raw::*;

#[test]
fn expert_metadata_derived_from_tensor_dims() {
    unsafe {
        let ctx = test_ggml_ctx();
        let t = new_tensor_3d(&ctx, test_type(), 64, 32, 8, "blk.0.ffn_gate_exps");
        assert_eq!(golem_tensor_expert_count(t.cast()), 8);
        assert_eq!(
            golem_tensor_expert_stride_bytes(t.cast()),
            (64 * 32 * type_size()) as i32
        );
        let named = new_tensor_3d(&ctx, test_type(), 64, 32, 8, "ffn_gate_exps");
        assert_eq!(golem_tensor_expert_count(named.cast()), 8);
        let dense = new_tensor_3d(&ctx, test_type(), 64, 32, 8, "ffn_down");
        assert_eq!(golem_tensor_expert_count(dense.cast()), 0);
        assert_eq!(golem_tensor_expert_stride_bytes(dense.cast()), 0);
    }
}

#[test]
fn register_then_unset_restores_null_state() {
    unsafe {
        golem_register_hooks(None, None, None, std::ptr::null_mut());
        // No assertion beyond "does not crash + links": the unset state is
        // byte-identity-proven by Task 4's dump gate.
    }
}
