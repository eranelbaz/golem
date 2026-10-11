//! The glm5next hybrid fixture loads, decodes deterministically, and carries
//! a recurrent-state term in its per-sequence state (§2.5).
#![allow(unsafe_code)]
mod common;

use golem_engine::session::{ChatTurn, Session};
use golem_llama_sys::raw;
use xtask::tiny_hybrid::dims;

#[test]
fn chat_is_deterministic_across_sessions() {
    let mut cfg = common::tiny_hybrid_config();
    cfg.n_predict = 4;
    let turns = [ChatTurn {
        role: "user",
        content: "hello",
    }];
    let run = || {
        let mut s = Session::new(&cfg).unwrap();
        let mut out = String::new();
        let n = s.chat_turn(&turns, &mut |p| out.push_str(p)).unwrap();
        (n, out)
    };
    let a = run();
    assert!(a.0 > 0, "greedy chat produced no tokens");
    assert_eq!(a, run(), "greedy decode must be deterministic");
}

#[test]
fn seq_state_exceeds_one_token_of_kv() {
    let cfg = common::tiny_hybrid_config();
    // Session::new initialises the backend once per process.
    drop(Session::new(&cfg).unwrap());
    let path = std::ffi::CString::new(cfg.model_path.to_str().unwrap()).unwrap();
    // SAFETY: path is a NUL-terminated CString; default params per llama.h.
    let model = unsafe {
        raw::llama_model_load_from_file(path.as_ptr(), raw::llama_model_default_params())
    };
    assert!(!model.is_null());
    // SAFETY: model is live; default context params with a small n_ctx.
    let ctx = unsafe {
        let mut p = raw::llama_context_default_params();
        p.n_ctx = cfg.n_ctx;
        raw::llama_init_from_model(model, p)
    };
    assert!(!ctx.is_null());
    let mut tok: raw::llama_token = 1;
    // SAFETY: ctx is live; the batch borrows `tok`, alive across the call.
    let (rc, size) = unsafe {
        let rc = raw::llama_decode(ctx, raw::llama_batch_get_one(&mut tok, 1));
        raw::llama_synchronize(ctx);
        (rc, raw::llama_state_seq_get_size(ctx, 0))
    };
    // SAFETY: ctx freed before the model it borrows; each exactly once.
    unsafe {
        raw::llama_free(ctx);
        raw::llama_model_free(model);
    }
    assert_eq!(rc, 0, "1-token prefill failed");

    // Upper bound on one token's attention KV: f16 K + V latents plus the
    // indexer's key/gate/pooled heads, per attention layer.
    let attn_layers = dims::HEAD_COUNT_KV.iter().filter(|&&h| h > 0).count();
    let kv_bytes = attn_layers
        * 2
        * (2 * (dims::KV_LORA_RANK as usize) + 3 * dims::INDEXER_KEY_LENGTH as usize);
    assert!(
        size > kv_bytes,
        "seq state {size} B has no recurrent term over {kv_bytes} B of KV"
    );
}
