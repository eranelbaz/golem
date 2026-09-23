#![allow(unsafe_code, missing_docs)]

#[allow(dead_code)] // conformance helpers shared with ci_key/pin_gate_conformance
mod common;

use golem_llama_sys::raw::*;
use golem_llama_sys::{Backend, substrate_backend};

const _BACKEND_GUARDS_EXHAUSTIVE: () = match Backend::Cpu {
    Backend::Cpu => (),
    Backend::Hip => (),
};

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

/// Field drop order (top-to-bottom) frees contexts before the model their destructors dereference.
struct Session {
    ctx_a: Ctx,
    ctx_b: Ctx,
    model: ModelGuard,
}

impl Session {
    fn load() -> Session {
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
            cparams.n_threads = 4;
            cparams.n_threads_batch = 4;
            cparams.no_perf = true;

            let ctx_a = Ctx(llama_init_from_model(model.0, cparams));
            let ctx_b = Ctx(llama_init_from_model(model.0, cparams));
            assert!(
                !ctx_a.0.is_null() && !ctx_b.0.is_null(),
                "llama_init_from_model failed"
            );

            Session {
                ctx_a,
                ctx_b,
                model,
            }
        }
    }

    fn run(&self, ctx: *mut llama_context) -> (llama_token, Vec<f32>) {
        unsafe {
            let vocab = llama_model_get_vocab(self.model.0);
            let n_vocab = llama_vocab_n_tokens(vocab);
            let bos = llama_token_bos(vocab);

            let mut tok = bos;
            let batch = llama_batch_get_one(&mut tok, 1);
            assert_eq!(llama_decode(ctx, batch), 0, "llama_decode failed");
            let p = llama_get_logits_ith(ctx, 0);
            assert!(!p.is_null(), "no logits at i=0");
            let logits = std::slice::from_raw_parts(p, n_vocab as usize);
            let argmax = (0..n_vocab as isize)
                .max_by(|a, b| logits[*a as usize].total_cmp(&logits[*b as usize]))
                .unwrap() as llama_token;
            (argmax, logits.to_vec())
        }
    }
}

fn two_context_greedy_decode() -> [(llama_token, Vec<f32>); 2] {
    let session = Session::load();
    [session.run(session.ctx_a.0), session.run(session.ctx_b.0)]
}

fn assert_decode_sane(token: llama_token, logits: &[f32]) {
    assert!(
        logits.iter().all(|f| f.is_finite()),
        "non-finite (NaN/inf) logits — poisoned build, not nondeterminism"
    );
    assert!(
        (0..logits.len() as i32).contains(&token),
        "argmax out of vocab: {token}"
    );
}

fn skip_backend_not_built() {
    eprintln!(
        "SKIPPED (GOLEM_LLAMA_SYS_BACKEND={:?}, substrate={:?}): zero assertions executed",
        std::env::var("GOLEM_LLAMA_SYS_BACKEND").as_deref(),
        substrate_backend()
    );
}

#[test]
fn greedy_decode_is_deterministic() {
    if substrate_backend() != Backend::Cpu {
        skip_backend_not_built();
        return;
    }
    let [(t1, l1), (t2, l2)] = two_context_greedy_decode();
    assert_decode_sane(t1, &l1);
    assert_decode_sane(t2, &l2);
    assert_eq!(t1, t2, "same input must give same greedy token");
    assert_eq!(
        l1, l2,
        "logits must be bit-identical for identical input/batch"
    );
}

#[test]
fn hip_decode_runs() {
    if substrate_backend() != Backend::Hip {
        skip_backend_not_built();
        return;
    }
    let [(t1, l1), (t2, l2)] = two_context_greedy_decode();
    assert_decode_sane(t1, &l1);
    assert_decode_sane(t2, &l2);
    if t1 != t2 {
        eprintln!(
            "note: hip greedy token differs across contexts (near-tie reorder): t1={t1} t2={t2}"
        );
    }
}
