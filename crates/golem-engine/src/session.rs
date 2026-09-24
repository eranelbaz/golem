//! Model + context lifecycle (tech design §7): backend init, load with
//! fail-fast sanity checks, RAII teardown, and the §4.3 decode self-check probe.
#![allow(unsafe_code)]

use std::ffi::{CStr, CString};
use std::sync::Once;

use golem_llama_sys::raw;
use golem_llama_sys::{Backend, substrate_backend};

use crate::config::EngineConfig;
use crate::error::EngineError;
use crate::gguf;
use crate::sanity;

/// Owned `llama_model` handle; `Drop` calls `llama_model_free`.
pub(crate) struct Model {
    ptr: *mut raw::llama_model,
}

impl Model {
    pub(crate) fn as_ptr(&self) -> *mut raw::llama_model {
        self.ptr
    }

    /// Embedded chat template, `None` when the model ships none.
    #[allow(dead_code)] // consumed by Task 5's chat_turn
    pub(crate) fn chat_template(&self) -> Option<String> {
        // SAFETY: self.ptr is a live llama_model from llama_model_load_from_file;
        // null name selects the default template (vendor/llama.cpp/include/llama.h).
        let p = unsafe { raw::llama_model_chat_template(self.ptr, std::ptr::null()) };
        if p.is_null() {
            None
        } else {
            // SAFETY: llama.cpp returns a pointer to a NUL-terminated static buffer.
            Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
        }
    }

    /// Vocabulary handle for tokenization and logits sizing; lifetime tied to the model.
    pub(crate) fn vocab(&self) -> *const raw::llama_vocab {
        // SAFETY: self.ptr is a live llama_model; the vocab outlives the model.
        unsafe { raw::llama_model_get_vocab(self.ptr) }
    }
}

impl Drop for Model {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr came from llama_model_load_from_file and is freed exactly once here.
            unsafe { raw::llama_model_free(self.ptr) };
        }
    }
}

/// Owned `llama_context` handle; `Drop` calls `llama_free`.
pub(crate) struct Context {
    ptr: *mut raw::llama_context,
}

impl Context {
    pub(crate) fn as_ptr(&self) -> *mut raw::llama_context {
        self.ptr
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr came from llama_init_from_model and is freed exactly once here,
            // while the model it dereferences is still alive (see LoadedModel field order).
            unsafe { raw::llama_free(self.ptr) };
        }
    }
}

/// RAII guard around `llama_batch_init`/`llama_batch_free`, reused by Task 5's chat_turn.
pub(crate) struct Batch {
    inner: raw::llama_batch,
}

impl Batch {
    /// `llama_batch_init(n_tokens, embd, n_seq_max)`; contents are uninitialized
    /// (llama.cpp mallocs), so every slot used must be written before decode.
    pub(crate) fn init(n_tokens: i32, embd: i32, n_seq_max: i32) -> Self {
        // SAFETY: plain allocation with by-value struct return; freed in Drop via llama_batch_free.
        Self {
            inner: unsafe { raw::llama_batch_init(n_tokens, embd, n_seq_max) },
        }
    }
}

impl Drop for Batch {
    fn drop(&mut self) {
        // SAFETY: inner came from llama_batch_init and is freed exactly once here.
        unsafe { raw::llama_batch_free(self.inner) };
    }
}

/// A loaded model plus a ready context, usable for `chat_turn` (Task 5).
pub struct LoadedModel {
    // Load-bearing field order: fields drop top-to-bottom, and the context must be
    // freed (llama_free) *before* the model it dereferences (llama_model_free) —
    // swapping these is a use-after-free at teardown. Same pattern as
    // golem-llama-sys/tests/smoke.rs::Session.
    ctx: Context,
    model: Model,
    arch: String,
}

impl LoadedModel {
    /// Architecture string from the GGUF metadata (e.g. `"glm4moe"`).
    pub fn metadata_arch(&self) -> &str {
        &self.arch
    }

    pub(crate) fn ctx(&self) -> &Context {
        &self.ctx
    }

    pub(crate) fn model(&self) -> &Model {
        &self.model
    }
}

/// Loads the model at `config.model_path`, creates a context with `config`-derived
/// params, and runs the decode self-check probe before handing the handle back.
pub fn load(config: &EngineConfig) -> Result<LoadedModel, EngineError> {
    init_backend();

    // Fail fast, fail loud: refuse-to-start on a §4.3 violation before weights touch memory.
    let metadata = gguf::read_metadata(&config.model_path)?;
    sanity::check(&metadata)?;

    let arch = metadata
        .kv_str("general.architecture")
        .unwrap_or_default()
        .to_owned();

    // OsStr bytes (not to_str) so every valid Linux path round-trips; interior NUL
    // cannot occur in a path, so CString::new only fails on the empty default path.
    use std::os::unix::ffi::OsStrExt;
    let cpath = CString::new(config.model_path.as_os_str().as_bytes())
        .map_err(|_| EngineError::ModelLoad(config.model_path.clone()))?;

    let model = {
        // SAFETY: cpath is a valid NUL-terminated CString; params are llama.cpp defaults;
        // FFI contract per vendor/llama.cpp/include/llama.h. llama.cpp prints its own
        // failure reason to stderr; we pass the path through ModelLoad.
        let ptr = unsafe {
            raw::llama_model_load_from_file(cpath.as_ptr(), raw::llama_model_default_params())
        };
        if ptr.is_null() {
            return Err(EngineError::ModelLoad(config.model_path.clone()));
        }
        Model { ptr }
    };

    let n_batch = config.batch_tokens.min(config.n_ctx);
    // SAFETY: returns llama.cpp's default context params by value.
    let mut params = unsafe { raw::llama_context_default_params() };
    params.n_ctx = config.n_ctx;
    params.n_batch = n_batch;
    params.n_ubatch = n_batch;
    params.n_threads = config.threads as i32;
    params.n_threads_batch = config.threads as i32;
    params.no_perf = true;
    params.embeddings = false;

    let ctx = {
        // SAFETY: model is a live llama_model handle; params are default-derived per llama.h.
        let ptr = unsafe { raw::llama_init_from_model(model.as_ptr(), params) };
        if ptr.is_null() {
            return Err(EngineError::ContextCreate);
        }
        Context { ptr }
    };

    let loaded = LoadedModel { ctx, model, arch };
    decode_self_check(loaded.ctx(), loaded.model().vocab())?;
    Ok(loaded)
}

/// `llama_backend_init` exactly once per process, however many models get loaded.
fn init_backend() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // SAFETY: global one-time llama.cpp backend initialization; no invariants to uphold.
        unsafe { raw::llama_backend_init() };
    });
}

/// The lightweight §4.3-item-4 probe: one decode must produce finite logits, and —
/// on CPU only, where reduction order is bit-deterministic — a cleared-memory
/// re-decode of the identical batch must agree on the argmax. On HIP the equality
/// check would false-fail on GPU reduction nondeterminism, so it is CPU-gated.
fn decode_self_check(ctx: &Context, vocab: *const raw::llama_vocab) -> Result<(), EngineError> {
    // SAFETY: vocab is a live llama_vocab owned by the loaded model.
    let bos = unsafe { raw::llama_vocab_bos(vocab) };
    let token = if bos == raw::LLAMA_TOKEN_NULL { 0 } else { bos };

    let mut batch = Batch::init(1, 0, 1);
    {
        // SAFETY: Batch::init(1, ..) allocated one slot per array; llama_batch_init
        // mallocs (no zeroing), so n_seq_id/seq_id must be written explicitly or
        // llama.cpp validates garbage. Single write per slot, before any decode.
        // Mutate in place (&mut) — llama_batch is a plain copyable struct.
        let b = &mut batch.inner;
        unsafe {
            b.n_tokens = 1;
            *b.token.add(0) = token;
            *b.pos.add(0) = 0;
            *b.n_seq_id.add(0) = 1;
            **b.seq_id.add(0) = 0;
            *b.logits.add(0) = 1;
        }
    }

    // SAFETY: ctx is a live llama_context; batch arrays are allocated and fully written.
    let code = unsafe { raw::llama_decode(ctx.as_ptr(), batch.inner) };
    if code != 0 {
        return Err(EngineError::Decode(code));
    }

    // SAFETY: vocab is live; n_vocab is the loaded vocab size (non-negative).
    let n_vocab = unsafe { raw::llama_vocab_n_tokens(vocab) };
    let n = usize::try_from(n_vocab)
        .map_err(|_| EngineError::InvalidModel(format!("vocab size {n_vocab} is negative")))?;

    // SAFETY: the decode reported success with one output token, so logits for i=0
    // are n_vocab valid floats (llama.h contract).
    let logits_ptr = unsafe { raw::llama_get_logits_ith(ctx.as_ptr(), 0) };
    if logits_ptr.is_null() {
        // No EngineError code can represent this without lying about a decode return
        // code; it is a broken FFI invariant, not a model rejection.
        panic!("llama_get_logits_ith returned null after a successful decode");
    }
    // SAFETY: logits_ptr is non-null and valid for n floats per llama.h.
    let logits = unsafe { std::slice::from_raw_parts(logits_ptr, n) };
    if !logits.iter().all(|v| v.is_finite()) {
        return Err(EngineError::UnsupportedModel(
            "decode self-check produced non-finite logits".to_owned(),
        ));
    }
    let argmax = argmax_of(logits);

    // Determinism smoke: CPU reduction order is bit-deterministic, so the identical
    // batch over a cleared KV cache must pick the same token.
    if substrate_backend() == Backend::Cpu {
        // SAFETY: llama_get_memory returns the context's live memory handle; clearing
        // an idle context between decodes is the documented llama_memory_clear use.
        unsafe { raw::llama_memory_clear(raw::llama_get_memory(ctx.as_ptr()), true) };

        // SAFETY: same invariants as the first decode; batch arrays are still alive.
        let code = unsafe { raw::llama_decode(ctx.as_ptr(), batch.inner) };
        if code != 0 {
            return Err(EngineError::Decode(code));
        }

        // SAFETY: successful decode with one output token; n floats valid.
        let logits_ptr = unsafe { raw::llama_get_logits_ith(ctx.as_ptr(), 0) };
        if logits_ptr.is_null() {
            panic!("llama_get_logits_ith returned null after a successful re-decode");
        }
        // SAFETY: logits_ptr is non-null and valid for n floats per llama.h.
        let logits = unsafe { std::slice::from_raw_parts(logits_ptr, n) };
        if argmax_of(logits) != argmax {
            return Err(EngineError::UnsupportedModel(
                "decode self-check: argmax changed across identical re-decode".to_owned(),
            ));
        }
    }

    Ok(())
}

fn argmax_of(logits: &[f32]) -> usize {
    (0..logits.len())
        .max_by(|a, b| logits[*a].total_cmp(&logits[*b]))
        .unwrap_or(0)
}
