//! Acceptance (board M1.1): load a tiny GGUF via the pinned substrate and
//! greedy-decode deterministically. Unsafe by definition — this is the FFI crate.
#![allow(unsafe_code, missing_docs)]

mod common;

use golem_llama_sys::raw::*;
use golem_llama_sys::{Backend, substrate_backend};

/// Drop guards — declared so that reverse drop order frees contexts BEFORE the model
/// (llama_context's destructor dereferences the model; freeing the model first is UB).
struct ModelGuard(*mut llama_model);
impl Drop for ModelGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { llama_model_free(self.0) };
        }
    }
}
struct Ctx(*mut llama_context);
impl Drop for Ctx {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { llama_free(self.0) };
        }
    }
}

#[test]
fn greedy_decode_is_deterministic() {
    let path = common::fixture();
    let cpath = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
    unsafe {
        let model = ModelGuard(llama_model_load_from_file(
            cpath.as_ptr(),
            llama_model_default_params(),
        ));
        assert!(
            !model.0.is_null(),
            "llama_model_load_from_file failed for {path:?}"
        );

        let mut cparams = llama_context_default_params();
        cparams.n_ctx = 64;
        cparams.n_batch = 64; // default 2048 wastes hundreds of MB scratch at this vocab
        cparams.n_threads = 4; // pin explicitly — determinism must not ride on GGML_DEFAULT_N_THREADS
        cparams.n_threads_batch = 4;
        cparams.no_perf = true;

        let _ctx_a = Ctx(llama_init_from_model(model.0, cparams));
        let _ctx_b = Ctx(llama_init_from_model(model.0, cparams));
        assert!(
            !_ctx_a.0.is_null() && !_ctx_b.0.is_null(),
            "llama_init_from_model failed"
        );

        let vocab = llama_model_get_vocab(model.0);
        let n_vocab = llama_vocab_n_tokens(vocab);
        let bos = llama_token_bos(vocab); // vocab-taking API at this pin

        let run = |ctx: *mut llama_context| -> (llama_token, Vec<f32>) {
            let mut tok = bos;
            let batch = llama_batch_get_one(&mut tok, 1);
            assert_eq!(llama_decode(ctx, batch), 0, "llama_decode failed");
            let p = llama_get_logits_ith(ctx, 0);
            assert!(
                !p.is_null(),
                "no logits at i=0 (Release path returns nullptr on failure)"
            );
            let logits = std::slice::from_raw_parts(p, n_vocab as usize);
            let argmax = (0..n_vocab as isize)
                .max_by(|a, b| logits[*a as usize].total_cmp(&logits[*b as usize]))
                .unwrap() as llama_token;
            (argmax, logits.to_vec())
        };

        let (t1, l1) = run(_ctx_a.0);
        let (t2, l2) = run(_ctx_b.0);
        // guards drop here: contexts freed first, model last (reverse declaration order)

        assert!(
            !l1.iter().any(|f| f.is_nan()),
            "NaN logits — poisoned build, not nondeterminism"
        );
        if substrate_backend() == Backend::Cpu {
            assert_eq!(t1, t2, "same input must give same greedy token");
            assert_eq!(
                l1, l2,
                "logits must be bit-identical for identical input/batch"
            );
        } else {
            // HIP build: determinism is deliberately NOT asserted here — HIP
            // reductions may reorder near-tied top-2 logits (known substrate
            // nondeterminism, tech design §7.4). The hardware leg's job is
            // that the HIP substrate links, loads, decodes, and executes on
            // the GPU (proved by the init log: ggml_hip_init / offloaded
            // layers to GPU); the recorded outputs are evidence, not contract.
            eprintln!(
                "hip decode run: t1={t1} t2={t2} logits_bit_identical={}",
                l1 == l2
            );
        }
        assert!(
            (0..n_vocab as i32).contains(&t1),
            "argmax out of vocab: {t1}"
        );
    }
}
