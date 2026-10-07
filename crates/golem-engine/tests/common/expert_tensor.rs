//! Synthetic expert-weights tensor for the ABI v5 trampoline tests: a real
//! ggml tensor named like the tiny-moe gate plane, whose expert-0 plane
//! bytes are byte-identical to the fixture GGUF (the glue's first-sight
//! cross-check preads and compares them).
#![allow(dead_code, unsafe_code)]
use super::tiny_moe_path;
use golem_gguf::ExpertKind;
use golem_llama_sys::raw;
use golem_placement::pool::{ExpertSource, GgufPreadSource};
use xtask::tiny_moe::dims;

pub struct TestGgmlCtx(*mut raw::ggml_context);
impl Drop for TestGgmlCtx {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: matches the ggml_init allocation.
            unsafe { raw::ggml_free(self.0) };
        }
    }
}

/// Expert-pool geometry of the tiny-moe fixture: one expert layer (blk.1
/// after the dense lead of 1), 8 experts, gate plane `n_embd × n_ff_exp`
/// f32 (2048 B stride).
pub const BLK: i32 = dims::N_LAYER_DENSE_LEAD as i32; // 1
pub const POOL_LAYER: u32 = 0;
pub const N_EXPERTS: i32 = dims::N_EXPERT as i32;
pub const N_EMBD: i64 = dims::N_EMBD as i64;
pub const N_FF_EXP: i64 = dims::N_FF_EXP as i64;

/// The tensor pointer stays valid while the returned ctx is alive.
///
/// # Panics
/// On fixture/source failures — a broken precondition, never a skip.
pub fn gate_tensor() -> (TestGgmlCtx, *mut raw::ggml_tensor) {
    // SAFETY: plain ggml scratch allocation; freed by the guard.
    let ctx = TestGgmlCtx(unsafe {
        raw::ggml_init(raw::ggml_init_params {
            mem_size: 4 * 1024 * 1024,
            mem_buffer: std::ptr::null_mut(),
            no_alloc: false,
        })
    });
    assert!(!ctx.0.is_null());
    // SAFETY: `ctx` outlives the tensor; the name is shorter than GGML_MAX_NAME.
    let t = unsafe {
        let c_name = std::ffi::CString::new("blk.1.ffn_gate_exps.weight").unwrap();
        let t = raw::ggml_new_tensor_3d(
            ctx.0,
            raw::ggml_type::GGML_TYPE_F32,
            N_EMBD,
            N_FF_EXP,
            N_EXPERTS as i64,
        );
        assert!(!t.is_null());
        raw::ggml_set_name(t, c_name.as_ptr());
        t
    };
    // Fill expert 0's gate plane from the fixture so the glue's first-sight
    // byte cross-check (pread vs host bytes) passes.
    let source = GgufPreadSource::open(&tiny_moe_path(), dims::N_LAYER_DENSE_LEAD)
        .expect("open fixture GGUF");
    let (_, plane_stride) = source
        .plane(POOL_LAYER, ExpertKind::Gate, 0)
        .expect("gate plane");
    // SAFETY: `t.data` is an in-context (no_alloc = false) buffer of
    // ne0·ne1·ne2 f32 elements.
    let host = unsafe {
        std::slice::from_raw_parts_mut(
            (*t).data as *mut u8,
            (N_EMBD * N_FF_EXP * N_EXPERTS as i64 * 4) as usize,
        )
    };
    source
        .read_expert(
            POOL_LAYER,
            ExpertKind::Gate,
            0,
            &mut host[..plane_stride as usize],
            0,
        )
        .expect("pread gate plane");
    (ctx, t)
}
