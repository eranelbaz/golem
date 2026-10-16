//! Model + context lifecycle (tech design §7): backend init, load with
//! fail-fast sanity checks, RAII teardown, the §4.3 decode self-check probe,
//! chat-template resolution (model metadata, else the chatml fallback), and
//! the per-turn entry points (`Session::turn`, driving `slot::step`).
#![allow(unsafe_code)]

use std::ffi::{CStr, CString, c_char, c_void};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Once};

use golem_llama_sys::raw;
use golem_llama_sys::{Backend, substrate_backend};
use golem_placement::pool::P3Stats;

use crate::chat::{ChatTemplates, Rendered};
use crate::config::EngineConfig;
use crate::error::EngineError;
use crate::gguf;
use crate::p3_glue;
use crate::parity::{self, ParityStep};
use crate::sanity;
use crate::slot::{self, FinishReason, SamplingParams, SlotState, TurnOutcome, TurnRequest};
use crate::snapshot::{SnapshotCache, SnapshotStats};

/// Owned `llama_model` handle; `Drop` calls `llama_model_free`.
pub(crate) struct Model {
    ptr: *mut raw::llama_model,
}

impl Model {
    pub(crate) fn as_ptr(&self) -> *mut raw::llama_model {
        self.ptr
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

/// RAII guard around `llama_batch_init`/`llama_batch_free`, shared by
/// `Session::chat_turn`.
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

/// A loaded model plus a ready context, usable for `Session::chat_turn`.
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

/// Serializes session construction so the pooled-concurrent refusal cannot
/// race a concurrent `new` that has not yet counted (shared by [`Session::new`]
/// and [`Session::new_with_test_ctx`]).
fn construction_lock() -> std::sync::MutexGuard<'static, ()> {
    static CONSTRUCT: std::sync::Mutex<()> = std::sync::Mutex::new(());
    CONSTRUCT.lock().unwrap_or_else(|e| e.into_inner())
}

/// Loads the model at `config.model_path`, creates a context with
/// `config`-derived params, and runs the decode self-check probe before
/// handing the handle back. No per-process hook state remains: each model
/// carries its own `golem_ctx` (ABI v5), so sequential models in one process
/// are legal — a pooled session attaches its ctx via [`load_with_hooks`].
pub fn load(config: &EngineConfig) -> Result<LoadedModel, EngineError> {
    load_with_hooks(config, None)
}

/// [`load`] with a pooled session's `golem_ctx` attached to the model
/// immediately after load and BEFORE `llama_init_from_model` — so the
/// context's first `sched_reserve` sees the hooks (`routing_fn != NULL`)
/// and sizes no full-size expert-plane `input_cpy` (cpy shrink).
pub(crate) fn load_with_hooks(
    config: &EngineConfig,
    attach: Option<raw::golem_ctx_t>,
) -> Result<LoadedModel, EngineError> {
    init_backend(forces_expert_offload(config));

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
        // SAFETY: pure handle lookup — returns llama.cpp's process-global CPU
        // buffer type (never null, no invariants). The loader compares override
        // entries against exactly this pointer to route CPU-pinned tensors
        // through its extra-buffer-type selection (llama-model-loader.cpp), so
        // this is the handle that reproduces the `-ncmoe` semantics.
        let cpu_buft = unsafe { raw::ggml_backend_cpu_buffer_type() };
        // `_moe_patterns` is load-bearing: it owns the CStrings the override
        // array points into and must stay bound until the load call returns.
        let (_moe_patterns, moe_overrides) = moe_buft_overrides(config.cpu_moe_layers, cpu_buft);
        // SAFETY: returns llama.cpp's default model params by value.
        let mut params = unsafe { raw::llama_model_default_params() };
        params.n_gpu_layers = n_gpu_layers_i32(config.n_gpu_layers);
        // AGENTS.md hot path: expert tensors must never be pre-faulted into the page
        // cache (186 GiB of MAP_POPULATE livelocked the kernel and flushed
        // everything else). The pool pages them from the GGUF itself, so
        // mark them TENSOR_READ_LAZY in the arch (glm5next.cpp) and turn the
        // loader's on-demand mode on: non-expert weights (≈8.4 GiB) load
        // normally, expert ranges are mapped without populate. Pool-off loads
        // keep the loader default: nothing would page the lazy experts in.
        if config.ram_experts_gib > 0 {
            params.lazy_mode = raw::llama_lazy_mode::LLAMA_LAZY_MODE_ON;
        }
        if !moe_overrides.is_empty() {
            params.tensor_buft_overrides = moe_overrides.as_ptr();
        }
        // SAFETY: cpath is a valid NUL-terminated CString; params are the
        // defaults amended from the config; `_moe_patterns` outlives this call
        // (same block), keeping every pattern pointer in the NULL-terminated
        // `tensor_buft_overrides` array valid while llama.cpp reads it. FFI
        // contract per vendor/llama.cpp/include/llama.h. llama.cpp prints its
        // own failure reason to stderr; we pass the path through ModelLoad.
        let ptr = unsafe { raw::llama_model_load_from_file(cpath.as_ptr(), params) };
        if ptr.is_null() {
            return Err(EngineError::ModelLoad(config.model_path.clone()));
        }
        Model { ptr }
    };
    // Per-model hook attach (ABI v5 §3.4): the ctx rides the llama_model so
    // every sched `sched_reserve` creates (including re-reserves) re-applies
    // it. Must precede `llama_init_from_model` — its self-check decode is
    // already a hooked decode.
    if let Some(ctx) = attach {
        // SAFETY: model is live and has no ctx yet; ctx is a live golem_ctx
        // owned by the caller's SessionHooks (outlives this model).
        unsafe { raw::llama_golem_model_attach(model.as_ptr(), ctx) };
    }

    let n_batch = config.batch_tokens.min(config.n_ctx);
    // Prefill streaming (§7.6): the context's ubatch (and with it the
    // compute buffer `ws`, resident for decode too) is sized for the
    // streamed prefill shape. n_batch covers the larger of the two chunk
    // shapes so both the streamed and per-token paths stay one-ubatch.
    let n_ubatch = if config.prefill_stream {
        let ub = if config.prefill_ubatch == 0 {
            2048
        } else {
            config.prefill_ubatch
        };
        ub.min(config.n_ctx)
    } else {
        n_batch
    };
    // SAFETY: returns llama.cpp's default context params by value.
    let mut params = unsafe { raw::llama_context_default_params() };
    // One seq per slot. llama.cpp splits n_ctx across sequences
    // (n_ctx_seq = n_ctx / n_seq_max, llama-context.cpp), so each slot gets
    // the full configured window only at n_ctx × parallel; per-seq KV
    // (kv_unified = false) keeps one slot's cache from starving another.
    params.n_ctx = config.n_ctx.saturating_mul(config.parallel.max(1));
    params.n_seq_max = config.parallel.max(1);
    params.kv_unified = false;
    params.n_batch = n_batch.max(n_ubatch);
    params.n_ubatch = n_ubatch;
    params.n_threads = config.threads as i32;
    params.n_threads_batch = config.threads as i32;
    params.no_perf = true;
    params.embeddings = false;
    if config.flash_attn_on {
        params.flash_attn_type = raw::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }

    let ctx = {
        // SAFETY: model is a live llama_model handle; params are default-derived per llama.h.
        let ptr = unsafe { raw::llama_init_from_model(model.as_ptr(), params) };
        if ptr.is_null() {
            return Err(EngineError::ContextCreate);
        }
        Context { ptr }
    };

    let loaded = LoadedModel { ctx, model, arch };
    if let Err(e) = decode_self_check(loaded.ctx(), loaded.model().vocab()) {
        if let Some(ctx) = attach {
            // SAFETY: the self-check decode is synchronized; the backend
            // is alive until `loaded` drops, so the slabs and copy stream
            // it bound are freed before `golem_ctx_free` asserts on them.
            unsafe { raw::golem_reset_device_cache(ctx) };
        }
        return Err(e);
    }
    Ok(loaded)
}

/// u32 config value → the `i32` `n_gpu_layers` the FFI takes. Values beyond
/// `i32::MAX` saturate to llama.h's "all layers" sentinel (-1): a caller asking
/// for more than 2³¹ layers plainly wants the whole model offloaded.
fn n_gpu_layers_i32(n_gpu_layers: u32) -> i32 {
    i32::try_from(n_gpu_layers).unwrap_or(-1)
}

/// Builds the `llama_model_params::tensor_buft_overrides` array implementing
/// `cpu_moe_layers = Some(n)`: the public-API equivalent of the vendor's
/// `-ncmoe n` (common/common.h `llm_add_n_cpu_ffn_overrides` applying the
/// `LLM_FFN_EXPS_REGEX` pattern per block — a common/arg.cpp path libllama
/// callers cannot reach). One entry per layer block pins that block's expert
/// tensors to `cpu_buft`; the NULL-`pattern` terminator is what the loader
/// scans for (llama-model-loader.cpp). `None`/`Some(0)` → empty: the vendor
/// maps an empty overrides vector back to `tensor_buft_overrides = NULL`
/// (common/common.cpp). Returns the backing CStrings — which must outlive the
/// `llama_model_load_from_file` call, since the array points into them — and
/// the terminated array.
fn moe_buft_overrides(
    cpu_moe_layers: Option<u32>,
    cpu_buft: raw::ggml_backend_buffer_type_t,
) -> (Vec<CString>, Vec<raw::llama_model_tensor_buft_override>) {
    let Some(n) = cpu_moe_layers.filter(|n| *n > 0) else {
        return (Vec::new(), Vec::new());
    };
    let mut patterns = Vec::with_capacity(n as usize);
    let mut overrides = Vec::with_capacity(n as usize + 1);
    for i in 0..n {
        let pattern = CString::new(format!(r"blk\.{i}\.ffn_(up|down|gate|gate_up)_(ch|)exps"))
            .expect("pattern cannot contain NUL");
        overrides.push(raw::llama_model_tensor_buft_override {
            pattern: pattern.as_ptr(),
            buft: cpu_buft,
        });
        patterns.push(pattern);
    }
    overrides.push(raw::llama_model_tensor_buft_override {
        pattern: std::ptr::null(),
        buft: std::ptr::null_mut(),
    });
    (patterns, overrides)
}

/// Whether this config puts expert weights on the host (pool on, or experts
/// pinned to CPU buffers) while the engine is expected to run their
/// `MUL_MAT_ID` on the GPU (design §3.4, I2). Flagless runs keep upstream's
/// op-offload threshold untouched.
fn forces_expert_offload(config: &EngineConfig) -> bool {
    config.ram_experts_gib > 0 || config.cpu_moe_layers.is_some_and(|n| n > 0)
}

/// `llama_backend_init` exactly once per process, however many models get
/// loaded. The first load decides the op-offload threshold: it is read once at
/// device init, so a later session cannot change it (a pooled session after a
/// flagless one is caught at runtime by `check_experts_on_gpu`).
fn init_backend(force_expert_offload: bool) {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        if force_expert_offload {
            // Upstream keeps ops on host-resident weights on the CPU below
            // batch 32, and MUL_MAT_ID's batch is ne[2] = tokens (1 during
            // decode), so force the threshold to 1. Must precede
            // llama_backend_init (read at device init).
            // SAFETY: FFI call into the vendored substrate; no invariants to uphold.
            unsafe { raw::golem_set_expert_offload_min_batch(1) };
        }
        // SAFETY: FFI call into the vendored substrate; `llama_log` is a
        // 'static fn that only reads `text` for the call's duration and is
        // thread-safe (tracing + a thread-local, no borrowed state); llama.cpp
        // may invoke it from any thread. It passes only valid ggml_log_level
        // discriminants, so the non_exhaustive Rust enum stays defined.
        unsafe { raw::llama_log_set(Some(llama_log), std::ptr::null_mut()) };
        // SAFETY: global one-time llama.cpp backend initialization; no invariants to uphold.
        unsafe { raw::llama_backend_init() };
    });
}

thread_local! {
    /// Level of this thread's last non-CONT record; CONT lines inherit it.
    static LAST_LOG_LEVEL: std::cell::Cell<raw::ggml_log_level> =
        const { std::cell::Cell::new(raw::ggml_log_level::GGML_LOG_LEVEL_INFO) };
}

/// Routes llama.cpp/ggml logs into `tracing` so `RUST_LOG` filters them; the
/// upstream default callback prints every level, DEBUG included, to stderr.
/// With no subscriber installed, WARN/ERROR still reach stderr.
///
/// # Panics
/// A panic inside the tracing subscriber propagates across this `extern "C"`
/// boundary and aborts; subscriber code must not panic.
unsafe extern "C" fn llama_log(
    level: raw::ggml_log_level,
    text: *const c_char,
    _user_data: *mut c_void,
) {
    use raw::ggml_log_level as lvl;
    let level = if level == lvl::GGML_LOG_LEVEL_CONT {
        LAST_LOG_LEVEL.get()
    } else {
        LAST_LOG_LEVEL.set(level);
        level
    };
    if text.is_null()
        || (level == lvl::GGML_LOG_LEVEL_DEBUG
            && !tracing::enabled!(target: "llama", tracing::Level::DEBUG))
    {
        return;
    }
    // SAFETY: llama.cpp passes a NUL-terminated buffer valid for this call.
    let msg = unsafe { CStr::from_ptr(text) }.to_string_lossy();
    if !tracing::dispatcher::has_been_set() {
        if level == lvl::GGML_LOG_LEVEL_WARN || level == lvl::GGML_LOG_LEVEL_ERROR {
            eprint!("{msg}");
        }
        return;
    }
    let msg = msg.trim_end();
    if msg.is_empty() {
        return;
    }
    match level {
        lvl::GGML_LOG_LEVEL_DEBUG => tracing::debug!(target: "llama", "{msg}"),
        lvl::GGML_LOG_LEVEL_WARN => tracing::warn!(target: "llama", "{msg}"),
        lvl::GGML_LOG_LEVEL_ERROR => tracing::error!(target: "llama", "{msg}"),
        _ => tracing::info!(target: "llama", "{msg}"),
    }
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
        // Synchronize first: the failed decode may have launched a compute whose
        // expert-copy leases release only at the next compute (§2.4).
        // SAFETY: ctx is live.
        unsafe { raw::llama_synchronize(ctx.as_ptr()) };
        return Err(EngineError::Decode(code));
    }
    // The lease-release contract: llama_decode is async w.r.t. the GPU —
    // synchronize before returning so the expert-copy leases released at the
    // next compute can never outlive an in-flight DMA. Sampling reads logits
    // through getters that synchronize, but decode paths without sampling
    // (probes, future draft loops) must not rely on that.
    // SAFETY: ctx is live.
    unsafe { raw::llama_synchronize(ctx.as_ptr()) };

    // SAFETY: vocab is live; n_vocab is the loaded vocab size (non-negative).
    let n_vocab = unsafe { raw::llama_vocab_n_tokens(vocab) };
    let n = usize::try_from(n_vocab)
        .map_err(|_| EngineError::InvalidModel(format!("vocab size {n_vocab} is negative")))?;

    // SAFETY: the decode reported success with one output token, so logits for i=0
    // are n_vocab valid floats (llama.h contract).
    let logits_ptr = unsafe { raw::llama_get_logits_ith(ctx.as_ptr(), 0) };
    if logits_ptr.is_null() {
        // No decode return code can represent this without lying; it is a
        // broken FFI invariant, not a model rejection — surface it as a value.
        return Err(EngineError::LogitsUnavailable);
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
            // Synchronize first: see the first decode's error path above.
            // SAFETY: ctx is live.
            unsafe { raw::llama_synchronize(ctx.as_ptr()) };
            return Err(EngineError::Decode(code));
        }
        // synchronize before the logits getter and the next compute's lease
        // release (the first decode of decode_self_check establishes this).
        // SAFETY: ctx is live.
        unsafe { raw::llama_synchronize(ctx.as_ptr()) };

        // SAFETY: successful decode with one output token; n floats valid.
        let logits_ptr = unsafe { raw::llama_get_logits_ith(ctx.as_ptr(), 0) };
        if logits_ptr.is_null() {
            return Err(EngineError::LogitsUnavailable);
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

/// One message of a chat conversation.
pub struct ChatTurn<'a> {
    /// Speaker role the chat template understands (`"system"`, `"user"`,
    /// `"assistant"`, ...).
    pub role: &'a str,
    /// Message body text.
    pub content: &'a str,
}

/// Builds the session's chat templates and validates them with a 1-message
/// probe render.
fn probe_templates(
    loaded: &LoadedModel,
    config: &EngineConfig,
) -> Result<ChatTemplates, EngineError> {
    let over = config.chat_template.as_deref();
    let chat = ChatTemplates::from_model(loaded.model().as_ptr(), over)?;
    let probe = chat.render(
        r#"[{"role":"user","content":"ping"}]"#,
        None,
        None,
        false,
        None,
    )?;
    if probe.prompt.is_empty() {
        return Err(EngineError::TemplateUnavailable);
    }
    Ok(chat)
}

/// Renders `req`'s transcript (raw JSON, else its `ChatTurn`s) and tools.
fn render(chat: &ChatTemplates, req: &TurnRequest<'_>) -> Result<Rendered, EngineError> {
    let built;
    let messages = match req.messages_json {
        Some(m) => m,
        None => {
            built = serde_json::Value::Array(
                req.messages
                    .iter()
                    .map(|m| serde_json::json!({"role": m.role, "content": m.content}))
                    .collect(),
            )
            .to_string();
            &built
        }
    };
    chat.render(
        messages,
        req.tools_json,
        req.tool_choice,
        req.reasoning_inline,
        req.reasoning_effort,
    )
}

/// A loaded model bound to its resolved chat template, driving its decode
/// slots; a turn restores the previous prompt's state snapshot when enabled
/// (§7.1), else re-prefills the full transcript.
/// Field order is load-bearing: `slots` (sampler chains) drop BEFORE the
/// llama ctx inside `inner`.
pub struct Session {
    slots: Vec<SlotState>,
    inner: SessionInner,
}

/// Everything but the slots. Field order is load-bearing: `loaded`
/// (llama_free) drops BEFORE `p3` (unregister already done in
/// `Session::drop`; ctx free + pool drop here) — the §3.4 teardown order.
pub(crate) struct SessionInner {
    loaded: LoadedModel,
    config: EngineConfig,
    chat: ChatTemplates,
    last_prompt_tokens: Vec<raw::llama_token>,
    /// P3 pool state when the RAM expert pool is enabled (nonzero
    /// `ram_experts_gib`); `None` keeps flagless runs pristine (hooks unset,
    /// T4). The session owns its hooks: dropping the session unregisters
    /// (C joins in-flight steps) and frees the per-model ctx — so the field
    /// must drop AFTER `loaded` (declaration order below) keeps the model
    /// alive while the hooks still reference it.
    p3: Option<p3_glue::SessionHooks>,
    /// §11 per-step recorder when `config.timing_csv` is set; flushed by the
    /// caller (`write_timing`) at turn end and reused across turns.
    /// Fixed-capacity, preallocated.
    timing: Option<crate::timing::TimingRecorder>,
    /// Set once a pooled offloaded turn's prefill passed
    /// [`check_experts_on_gpu`] and [`check_pool_engaged`] and its first
    /// decode step passed [`check_experts_on_gpu`].
    pool_verified: bool,
    /// Prompt-prefill state snapshots (§7.1); `None` when disabled.
    snapshots: Option<SnapshotCache>,
    /// Test seam: the turn's cancel flag is set after this many prefill
    /// chunks decoded (0 = off).
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) test_cancel_after_chunks: u32,
    /// The one `llama_batch` every step fills (capacity `llama_n_batch`),
    /// allocated at load: no per-turn or per-step batch allocation.
    batch: Batch,
    batch_cap: usize,
    /// Aggregate cancel read by the context's abort callback (registered
    /// once at construction, cleared in `Session::drop`).
    pub(crate) abort: Arc<AbortSet>,
    /// Round-robin start slot for the next step (fairness across slots).
    pub(crate) rr: usize,
    /// Decode steps since the last streamed chunk (`slot::STREAM_EVERY`).
    pub(crate) decode_since_stream: u32,
    /// K of the last batched slab degradation, until the scheduler reports it.
    pub(crate) degraded: Option<u32>,
    /// Test seam: the n-th generation step (1-based, counted from when it is
    /// set) fails with `Decode(1)` before reaching `llama_decode` (0 = off).
    test_fail_gen_step: u32,
}

impl Drop for Session {
    fn drop(&mut self) {
        // §3.4 teardown order: detach the model's ctx → unregister-join →
        // `llama_free` (field drop below: the sched destructor decrements the
        // ctx attach refcount) → `golem_ctx_free` (inside SessionHooks::drop,
        // aborts while the refcount ≠ 0) → pool/prefetch drop. `p3` is
        // declared after `loaded`, so its drop runs after `llama_free`.
        if let Some(_p3) = self.inner.p3.as_ref() {
            // SAFETY: the model is live until `loaded` drops below; NULL
            // detaches so no later context can reach the freed ctx.
            unsafe {
                raw::llama_golem_model_attach(
                    self.inner.loaded.model().as_ptr(),
                    std::ptr::null_mut(),
                )
            };
        }
        if let Some(p3) = self.inner.p3.as_mut() {
            // Reclaim the device slab set while the substrate backend is
            // still alive — freeing those buffers after `llama_free` would be
            // a use-after-free (the ctx_free dc-empty assert enforces it).
            // Decode is idle between turns, so no slab entry is in flight.
            let ctx = p3.live_ctx_ptr();
            if !ctx.is_null() {
                // SAFETY: FFI on the session's live ctx; clears + frees the
                // device-cache entries only.
                unsafe { raw::golem_reset_device_cache(ctx) };
                // The slab bytes are gone: drop the exclusive-residency
                // VRAM-presence bits so the prefetcher serves those units
                // from disk again.
                p3.device_cache_reset();
            }
            // Unregister + join BEFORE `llama_free`: no in-flight C callback
            // may touch the pool while the context tears down.
            p3.unregister();
        }
        // SAFETY: ctx is live; the set the callback points at is still owned
        // by `inner` and dropped after this.
        unsafe {
            raw::llama_set_abort_callback(
                self.inner.loaded.ctx().as_ptr(),
                None,
                std::ptr::null_mut(),
            )
        };
        // Pairs with session_opened in `new`; keeps the live-session counter
        // exact.
        p3_glue::session_closed();
    }
}

impl Session {
    /// Loads per `config`, resolves the chat template (model-embedded, else
    /// [`FALLBACK_CHAT_TEMPLATE`]), and validates it with a 1-message probe
    /// apply: apply failure → `TemplateApply(code)`; empty/unusable →
    /// `TemplateUnavailable`. A pooled config's hooks are created BEFORE the
    /// model loads and attach to the model inside `load_with_hooks`.
    pub fn new(config: &EngineConfig) -> Result<Self, EngineError> {
        // Serializes construction so the pooled-concurrent refusal cannot
        // race a concurrent `new` that has not yet counted.
        let _construct = construction_lock();
        // A second CONCURRENT pool-enabled session is refused before any
        // model load (VRAM arithmetic in the refusal). Sequential sessions —
        // pooled or not — are legal: the hooks are per-model (ABI v5).
        // Streaming stages from the RAM expert pool (§7.6); without one it
        // would silently fall back to the per-token path.
        if config.parallel == 0 {
            return Err(EngineError::InvalidConfig("parallel must be >= 1".into()));
        }
        if config.prefill_stream && config.ram_experts_gib == 0 {
            return Err(EngineError::InvalidConfig(
                "prefill_stream requires the RAM expert pool (ram_experts_gib > 0)".into(),
            ));
        }
        if config.ram_experts_gib > 0 {
            p3_glue::refuse_if_pool_session_live()?;
        }
        // The copy stream is a HIP stream of the split backend's context; the
        // CPU substrate has none, and without the RAM pool there is no pooled
        // H2D to move onto it.
        if config.async_h2d && substrate_backend() == Backend::Cpu {
            return Err(EngineError::InvalidConfig(
                "async_h2d: no dedicated H2D copy stream in this substrate".into(),
            ));
        }
        if config.async_h2d && config.ram_experts_gib == 0 {
            return Err(EngineError::InvalidConfig(
                "async_h2d: needs the RAM expert pool (ram_experts_gib > 0)".into(),
            ));
        }
        if config.o_direct && (config.manifest_path.is_none() || config.ram_experts_gib == 0) {
            return Err(EngineError::InvalidConfig(
                "o_direct needs the store-backed pool (a manifest and ram_experts_gib > 0)".into(),
            ));
        }
        if !(config.vram_experts_frac > 0.0 && config.vram_experts_frac <= 1.0) {
            return Err(EngineError::InvalidConfig(format!(
                "vram_experts_frac {} outside (0, 1]",
                config.vram_experts_frac
            )));
        }
        if let Some(set) = &config.decode_cpuset {
            match golem_store::sys::pin_current_thread(set) {
                Ok(()) => {}
                Err(e) if config.pin_auto => tracing::warn!("decode thread pinning: {e}"),
                Err(e) => return Err(e.into()),
            }
        }
        match &config.manifest_path {
            Some(m) => {
                for w in crate::verify::verify_model(&config.model_path, m, config.full_verify)? {
                    tracing::warn!("{w}");
                }
            }
            None => tracing::warn!("no manifest: start-time model verification (R-8) skipped"),
        }
        // One copy stream per hook ctx: a second device backend would hit the
        // C-side FFI abort in golem_cs_begin, and the load's self-check decode
        // already binds the copy stream. `init_backend` first: the registry
        // must not be touched before it sets the op-offload threshold.
        if config.async_h2d {
            init_backend(forces_expert_offload(config));
            if golem_llama_sys::gpu_device_count() > 1 {
                return Err(EngineError::InvalidConfig(
                    "async_h2d: supports a single GPU device".into(),
                ));
            }
        }
        // P3 (§3.4): build the RAM expert pool + create the per-model ctx
        // BEFORE the model loads — the ctx must attach to the llama_model
        // before `llama_init_from_model` so its first `sched_reserve` sees
        // the hooks (cpy shrink). If the load fails, the guards drop cleanly.
        let report = load_plan(config)?;
        let clamped;
        let config = match report.as_ref().and_then(|(_, k)| *k) {
            Some(k) => {
                clamped = EngineConfig {
                    device_cache_experts: k,
                    ..config.clone()
                };
                &clamped
            }
            None => config,
        };
        let report = report.map(|(r, _)| r);
        let mut p3 = if config.ram_experts_gib > 0 {
            let mut p = p3_glue::enable(config)?;
            if let Some(r) = &report {
                p.cap_device_cache(r.term("device_expert_cache"));
            }
            Some(p)
        } else {
            None
        };
        let loaded = match load_with_hooks(config, p3.as_ref().map(|p| p.ctx_ptr())) {
            Ok(loaded) => loaded,
            Err(e) => {
                drop(p3);
                return Err(e);
            }
        };
        // Derived K needs the resident weights, KV and compute buffer
        // (ruling R3).
        let rest = p3
            .as_mut()
            .map_or(Ok(()), |p| p.fit_device_cache())
            .and_then(|()| Ok((probe_templates(&loaded, config)?, snapshot_cache(config)?)));
        let (chat, snapshots) = match rest {
            Ok(r) => r,
            Err(e) => {
                if let Some(p) = p3.as_mut() {
                    // SAFETY: decode is idle and the backend lives until
                    // `loaded` drops: free what the self-check bound before
                    // `golem_ctx_free` asserts on it.
                    unsafe { raw::golem_reset_device_cache(p.ctx_ptr()) };
                    p.device_cache_reset();
                }
                return Err(e);
            }
        };
        let (batch, batch_cap, abort) = shared_batch(&loaded, config);

        Ok({
            p3_glue::session_opened();
            Session {
                slots: (0..config.parallel)
                    .map(|i| SlotState::new(i as i32, config.n_predict))
                    .collect(),
                inner: SessionInner {
                    loaded,
                    config: config.clone(),
                    chat,
                    last_prompt_tokens: Vec::new(),
                    p3,
                    timing: config
                        .timing_csv
                        .as_ref()
                        .map(|_| crate::timing::TimingRecorder::new(config.n_predict as usize + 1)),
                    pool_verified: false,
                    snapshots,
                    #[cfg(any(test, feature = "test-support"))]
                    test_cancel_after_chunks: 0,
                    batch,
                    batch_cap,
                    abort,
                    rr: 0,
                    decode_since_stream: 0,
                    degraded: None,
                    test_fail_gen_step: 0,
                },
            }
        })
    }

    /// Test seam (hip-tests): [`Session::new`] with a caller-owned
    /// `golem_ctx` attached to the model BEFORE `llama_init_from_model` —
    /// the caller registers its own hook set on that ctx (ABI v5) and keeps
    /// the registration alive for the session's lifetime. Pool stays off.
    #[cfg(feature = "hip-tests")]
    pub fn new_with_test_ctx(
        config: &EngineConfig,
        ctx: raw::golem_ctx_t,
    ) -> Result<Self, EngineError> {
        let _guard = construction_lock();
        let loaded = load_with_hooks(config, Some(ctx))?;
        let chat = probe_templates(&loaded, config)?;
        let (batch, batch_cap, abort) = shared_batch(&loaded, config);
        Ok({
            p3_glue::session_opened();
            Session {
                slots: (0..config.parallel)
                    .map(|i| SlotState::new(i as i32, config.n_predict))
                    .collect(),
                inner: SessionInner {
                    loaded,
                    config: config.clone(),
                    chat,
                    last_prompt_tokens: Vec::new(),
                    p3: None,
                    timing: config
                        .timing_csv
                        .as_ref()
                        .map(|_| crate::timing::TimingRecorder::new(config.n_predict as usize + 1)),
                    pool_verified: false,
                    snapshots: None,
                    #[cfg(any(test, feature = "test-support"))]
                    test_cancel_after_chunks: 0,
                    batch,
                    batch_cap,
                    abort,
                    rr: 0,
                    decode_since_stream: 0,
                    degraded: None,
                    test_fail_gen_step: 0,
                },
            }
        })
    }

    /// The P3 slow-path counters when the RAM expert pool is enabled for this
    /// session; `None` when the pool is off (the default). Demand and prefetch
    /// page-ins are counted separately (design §6.6).
    pub fn p3_stats(&self) -> Option<&P3Stats> {
        self.inner.p3.as_ref().map(|p| p.stats.as_ref())
    }

    /// The session's hook guards (test/observability seam: stream counters,
    /// trampoline call counts, prefill-stream arming). `None` when the pool
    /// is off.
    pub fn p3_hooks(&self) -> Option<&crate::p3_glue::SessionHooks> {
        self.inner.p3.as_ref()
    }

    /// Test seam: mutable hooks (between turns only).
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn p3_hooks_mut(&mut self) -> Option<&mut crate::p3_glue::SessionHooks> {
        self.inner.p3.as_mut()
    }

    /// The session's per-model hook context pointer (ABI v5), for test seams
    /// that address per-ctx substrate state (the P4 budget setter/reset).
    /// `None` when the pool is off.
    #[cfg_attr(not(feature = "hip-tests"), allow(dead_code))]
    pub fn p3_ctx_ptr(&self) -> Option<raw::golem_ctx_t> {
        self.inner.p3.as_ref().map(|p| p.ctx_ptr())
    }

    /// The tokenized prompt of the most recent `chat_turn` (post-render,
    /// post-special-tokens).
    pub fn last_prompt_tokens(&self) -> &[i32] {
        &self.inner.last_prompt_tokens
    }

    /// True when the session's config requests GPU offload (`n_gpu_layers > 0`):
    /// the dump-header hook parity legs record to prove a "hip"-labeled run is
    /// really attempting device execution (mis-labeled-leg guard). This is the
    /// config's intent, not llama.cpp's runtime placement — a CPU-only
    /// substrate reports `true` with zero devices to offload onto.
    pub fn offloaded(&self) -> bool {
        self.inner.offloaded()
    }

    /// §6.5 prefetch/prediction counters; `None` without the RAM expert pool.
    pub fn prefetch_stats(&self) -> Option<crate::p3_glue::PrefetchStats> {
        self.inner.p3.as_ref().map(|p| p.prefetch_stats())
    }

    /// Resolved P2 slot count per layer (explicit or derived from free VRAM);
    /// `None` without the RAM expert pool.
    pub fn device_cache_experts(&self) -> Option<u32> {
        self.inner.p3.as_ref().map(|p| p.device_cache_experts)
    }

    /// Device bytes the expert cache holds right now: `(slabs + remapped ids,
    /// prefill-stream slabs)`, summed from C's live buffers (§3.4, §7.6);
    /// `None` unless this session drives the pooled GPU split.
    pub fn device_cache_resident_bytes(&self) -> Option<(u64, u64)> {
        if !self.offloaded() {
            return None;
        }
        let p = self.inner.p3.as_ref()?;
        let (mut slab, mut stream) = (0u64, 0u64);
        // SAFETY: live ctx while p3 holds the hooks; `Session` is !Sync, so
        // `&self` means no compute is in flight (C reads stream slabs
        // unlocked); the out pointers are valid locals.
        unsafe { raw::golem_device_cache_resident_bytes(p.ctx_ptr(), &mut slab, &mut stream) };
        Some((slab, stream))
    }

    /// Whether prefill streaming (§7.6) survived the device-cache fit; the
    /// fit drops it when the stream slabs and a K ≥ top-k cache do not both fit.
    pub fn prefill_streaming(&self) -> bool {
        self.inner.p3.as_ref().is_some_and(|p| p.stream.is_some())
    }

    /// P4 device expert-copy cache counters (§3.4): bytes copied into / skipped
    /// by the persistent device cache, evictions, and the derived skip ratio.
    /// `None` unless this session drives the pooled GPU split — the only P2
    /// call-site (hooks registered + offload requested).
    pub fn device_cache_stats(&self) -> Option<DeviceCacheStats> {
        if !self.offloaded() || self.inner.p3.is_none() {
            return None;
        }
        // SAFETY: FFI getter with this session's live ctx; the returned
        // pointer is ctx-owned (alive while the hooks are registered).
        let block = unsafe {
            raw::golem_get_device_cache_stats(self.inner.p3.as_ref().expect("p3 present").ctx_ptr())
        };
        assert!(
            !block.is_null(),
            "golem_get_device_cache_stats returned null"
        );
        // SAFETY: layout contract from the C++ definition (bindgen renders
        // the struct opaque): 5 contiguous lock-free u64 counters in member
        // order bytes_copied, bytes_skipped, evictions, cache_experts_copied,
        // stream_experts_copied (pinned by static_assert/offsetof there).
        let counters = unsafe { &*block.cast::<[std::sync::atomic::AtomicU64; 5]>() };
        let [
            bytes_copied,
            bytes_skipped,
            evictions,
            cache_copies,
            stream_copies,
        ] = counters
            .each_ref()
            .map(|c| c.load(std::sync::atomic::Ordering::Relaxed));
        Some(DeviceCacheStats {
            bytes_copied,
            bytes_skipped,
            evictions,
            cache_copies,
            stream_copies,
            skip_ratio: skip_ratio(bytes_copied, bytes_skipped),
        })
    }

    /// Expert `MUL_MAT_ID` nodes the scheduler placed on a CPU backend
    /// (process-cumulative, I2: must stay 0 on HIP builds). Process-global
    /// like [`expert_placement`].
    pub fn expert_mul_mat_id_off_gpu(&self) -> u64 {
        expert_placement().on_cpu
    }

    /// Expert `MUL_MAT_ID` nodes the scheduler placed on a GPU backend
    /// (process-cumulative, process-global like [`expert_placement`]).
    pub fn expert_mul_mat_id_on_gpu(&self) -> u64 {
        expert_placement().on_gpu
    }

    /// Locked RAM: expert pool (mlocked + GPU-registered) plus the
    /// snapshot arena.
    pub fn locked_ram_bytes(&self) -> u64 {
        let pool = if self.inner.p3.is_some() {
            self.inner.config.ram_experts_gib << 30
        } else {
            0
        };
        pool + self
            .inner
            .snapshots
            .as_ref()
            .map_or(0, SnapshotCache::arena_bytes)
    }

    /// Fraction of RAM-pool slots in use; `None` without the pool.
    pub fn pool_fill(&self) -> Option<f64> {
        self.inner.p3.as_ref().map(|p| p.pool_fill())
    }

    /// Store-backed source health; `None` without a manifest-backed pool.
    pub fn storage_health(&self) -> Option<golem_store::InstanceHealth> {
        self.inner.p3.as_ref().and_then(|p| p.storage_health())
    }

    /// Test seam: marks every store replica unreadable.
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn fail_storage_for_test(&self) {
        if let Some(p) = &self.inner.p3 {
            p.fail_storage_for_test();
        }
    }

    /// Test seam: the next page-in marks every replica unreadable before
    /// its reads, so the in-flight read fails.
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn fail_storage_in_read_for_test(&self) {
        if let Some(p) = &self.inner.p3 {
            p.fail_storage_in_read_for_test();
        }
    }

    /// Test seam: the stream reader kills every replica right before its
    /// next read, so a streamed-prefill read fails.
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn fail_storage_in_stream_read_for_test(&self) {
        if let Some(p) = &self.inner.p3 {
            p.fail_storage_in_stream_read_for_test();
        }
    }

    /// Flushes the §11 per-step rows into `w` without destroying the
    /// recorder — every turn records (a `take`-style drain would leave turns
    /// 2+ empty). No-op when the timing CSV is disabled.
    pub fn write_timing(&mut self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        match self.inner.timing.as_mut() {
            Some(t) => t.flush(w),
            None => Ok(()),
        }
    }

    /// Flushes this turn's routing-trace records (§11 pattern) into `w`:
    /// the `# golem-routing-trace v1` header on first call, then one
    /// `step\tphase\tlayer\tid,id,...` line per routing readback since the
    /// last flush; the ring clears for the next turn. No-op when the routing
    /// trace is disabled (config) or the pool is off (no routing readbacks).
    pub fn write_routing_trace(&mut self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        match self.inner.p3.as_ref() {
            Some(p3) => p3.write_routing_trace(w),
            None => Ok(()),
        }
    }

    /// Drops the buffered timing rows and routing-trace records of a turn
    /// that failed, keeping the trace header state (a sink flush would
    /// swallow the header).
    pub fn discard_turn_records(&mut self) {
        if let Some(t) = self.inner.timing.as_mut() {
            let _ = t.flush(&mut std::io::sink());
        }
        if let Some(p3) = self.inner.p3.as_ref() {
            p3.drain_routing_trace();
        }
    }

    /// Runs one chat turn greedily with the session-wide `n_predict`
    /// (a [`TurnRequest`] with default sampling and no `max_tokens`). Piece
    /// bytes stream through `emit` as they complete UTF-8 characters
    /// (verbatim — no space stripping); buffered residue is flushed through
    /// `emit` even when the turn fails. Returns tokens generated.
    pub fn chat_turn(
        &mut self,
        messages: &[ChatTurn<'_>],
        emit: &mut dyn FnMut(&str),
    ) -> Result<u32, EngineError> {
        Ok(self
            .turn(&TurnRequest::new(messages), emit)?
            .completion_tokens)
    }

    /// Runs one chat turn exactly like [`Session::chat_turn`] (same
    /// `config.n_predict`, same greedy sampler chain) but discards text output
    /// and instead records one [`ParityStep`] per `llama_sampler_sample` call
    /// into `steps` (cleared first): the sampler's token choice plus the two
    /// largest logits of the row it was sampled from. `position` is the
    /// 0-based generation-step index, not `llama_pos`; a trailing EOG sample is
    /// recorded but not counted in the returned token count. Returns tokens
    /// generated.
    pub fn chat_turn_parity(
        &mut self,
        messages: &[ChatTurn<'_>],
        steps: &mut Vec<ParityStep>,
    ) -> Result<u32, EngineError> {
        steps.clear();
        self.slots[0].parity = Some(std::mem::take(steps));
        let out = self.turn(&TurnRequest::new(messages), &mut |_| {});
        *steps = self.slots[0].parity.take().unwrap_or_default();
        Ok(out?.completion_tokens)
    }

    /// Snapshot restores attempted (a prefix hit); 0 when snapshots are off.
    pub fn snapshot_hits(&self) -> u64 {
        self.snapshot_stats().map_or(0, |s| s.hits)
    }

    /// Snapshot cache counters; `None` when snapshots are off.
    pub fn snapshot_stats(&self) -> Option<SnapshotStats> {
        self.inner.snapshots.as_ref().map(SnapshotCache::stats)
    }

    /// Test seam: truncates every stored snapshot so its next restore is
    /// rejected by `llama_state_seq_set_data`.
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn corrupt_snapshots_for_test(&mut self) {
        if let Some(c) = self.inner.snapshots.as_mut() {
            c.truncate_all_for_test();
        }
    }

    /// Renders `req` through the chat layer; pass the result back via
    /// [`TurnRequest::rendered`] to keep its [`crate::OutputParser`] for
    /// parsing the generated text.
    pub fn render(&self, req: &TurnRequest<'_>) -> Result<Rendered, EngineError> {
        render(&self.inner.chat, req)
    }

    /// Test seam: every later turn sets its own cancel flag right after
    /// prefill chunk `n` decodes (0 disables).
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn cancel_after_prefill_chunks_for_test(&mut self, n: u32) {
        self.inner.test_cancel_after_chunks = n;
    }

    /// Test seam: highest position held for `seq` in the KV/recurrent memory,
    /// `-1` when the sequence is empty.
    #[doc(hidden)]
    pub fn seq_pos_max(&self, seq: i32) -> i32 {
        // SAFETY: ctx is live for `&self`; `&self` excludes a concurrent turn,
        // so no decode is in flight; the query only reads memory metadata.
        unsafe {
            raw::llama_memory_seq_pos_max(
                raw::llama_get_memory(self.inner.loaded.ctx().as_ptr()),
                seq,
            )
        }
    }

    /// Runs one turn on slot 0: renders `req.messages`, drops the slot's
    /// sequence, restores the longest snapshotted prompt prefix (§7.1) and
    /// prefills the rest in chunks, then samples per
    /// `req.sampling` until EOG, a stop string, `max_tokens` (default
    /// `config.n_predict`, clamped to the context left after the prompt),
    /// `n_ctx_seq - 1` positions, or cancel. Every `Err`/`Cancelled` exit
    /// drops the sequence (`llama_memory_seq_rm`): recurrent state cannot be
    /// rewound (§2.5).
    pub fn turn(
        &mut self,
        req: &TurnRequest<'_>,
        emit: &mut dyn FnMut(&str),
    ) -> Result<TurnOutcome, EngineError> {
        let mut res = self.drive(req, emit);
        // Every decode of a live session (prefill chunks, generation steps)
        // runs inside `drive`, so this is the one slab-failure recovery site;
        // load's self-check decode fails the load instead.
        if let Err(e) = res {
            res = Err(self.inner.recover_slab(e));
        }
        if res.is_err() {
            self.slots[0].flush_text(emit);
            self.slots[0].abort();
        }
        res
    }

    fn drive(
        &mut self,
        req: &TurnRequest<'_>,
        emit: &mut dyn FnMut(&str),
    ) -> Result<TurnOutcome, EngineError> {
        let guard = self.inner.begin(&mut self.slots[0], req)?;
        let finish = loop {
            if let Some(f) = self.slots[0].finished() {
                break f;
            }
            slot::step(
                Core {
                    session: &mut self.inner,
                    slots: &mut self.slots[..1],
                },
                &mut |_, t| emit(t),
                false,
            )?;
        };
        if finish == FinishReason::Cancelled {
            drop(guard);
            tracing::info!(slot = self.slots[0].seq, "cancel: sequence removed");
        } else {
            guard.disarm();
        }
        match self.slots[0].take_result() {
            Some(r) => r,
            None => unreachable!("loop exits on Done"),
        }
    }

    /// Test seam: the `n`-th generation batch from now (1-based) fails as if
    /// `llama_decode` returned 1, without running it (0 disables).
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn fail_generation_step_for_test(&mut self, n: u32) {
        self.inner.test_fail_gen_step = n;
    }

    /// `llama_n_ctx_seq`: the context window each slot gets.
    pub fn n_ctx_seq(&self) -> u32 {
        self.inner.n_ctx_seq()
    }

    pub(crate) fn parts(&mut self) -> (&mut SessionInner, &mut [SlotState]) {
        (&mut self.inner, &mut self.slots)
    }
}

/// The session state `slot::step` drives, paired with the slots it advances;
/// built per call, no allocation.
pub(crate) struct Core<'s> {
    pub(crate) session: &'s mut SessionInner,
    pub(crate) slots: &'s mut [SlotState],
}

/// Per-turn FFI state of one slot: tokenized prompt, prefill progress and the
/// I2 runtime-proof window. The token batch itself is the session's
/// (`SessionInner::batch`), shared by every slot.
pub(crate) struct TurnFfi {
    pub(crate) prompt: Vec<raw::llama_token>,
    /// Prompt tokens already in the sequence (restored prefix + prefilled).
    pub(crate) n_past: usize,
    /// Prefill chunks decoded this turn (test seam counter).
    pub(crate) chunks: u32,
    pub(crate) chunk_cap: usize,
    pub(crate) n_ubatch: usize,
    pub(crate) streaming: bool,
    n_vocab: usize,
    pub(crate) verify: bool,
    placement_before: ExpertPlacement,
    traffic_before: u64,
    /// Re-prefill of `prompt ++ generated` after a failed batch: no snapshot.
    pub(crate) replay: bool,
}

/// Aggregate cancel for the session's `llama_set_abort_callback`: a step's
/// compute aborts only when every sequence in its batch is cancelled. Slots
/// are fixed per batch, so this is a preallocated pointer array, refilled
/// per step with no allocation; all fields atomic (C reads it mid-decode).
pub(crate) struct AbortSet {
    flags: Box<[std::sync::atomic::AtomicPtr<AtomicBool>]>,
    len: std::sync::atomic::AtomicUsize,
}

impl AbortSet {
    fn new(n: usize) -> Self {
        Self {
            flags: (0..n)
                .map(|_| std::sync::atomic::AtomicPtr::new(std::ptr::null_mut()))
                .collect(),
            len: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Arms the set with this step's flags; any participant without a flag
    /// can never be cancelled, so the set stays empty (never aborts).
    /// Callers keep the `Arc`s alive (slot-owned) until `clear`.
    pub(crate) fn arm<'a>(&self, flags: impl Iterator<Item = Option<&'a Arc<AtomicBool>>>) {
        let mut n = 0;
        for f in flags {
            let Some(f) = f else {
                self.len.store(0, Ordering::Release);
                return;
            };
            if n < self.flags.len() {
                self.flags[n].store(Arc::as_ptr(f).cast_mut(), Ordering::Relaxed);
                n += 1;
            }
        }
        self.len.store(n, Ordering::Release);
    }

    pub(crate) fn clear(&self) {
        self.len.store(0, Ordering::Release);
    }

    pub(crate) fn all_cancelled(&self) -> bool {
        let n = self.len.load(Ordering::Acquire);
        n > 0
            && self.flags[..n].iter().all(|p| {
                // SAFETY: armed pointers come from `Arc`s the slots hold
                // until `clear` (called after the decode returns).
                unsafe { &*p.load(Ordering::Relaxed) }.load(Ordering::Relaxed)
            })
    }
}

/// `ggml_abort_callback` trampoline: `data` is the session's [`AbortSet`].
unsafe extern "C" fn abort_requested(data: *mut std::ffi::c_void) -> bool {
    // SAFETY: `data` is `Arc::as_ptr` of `SessionInner::abort`, which
    // outlives the context (the callback is cleared in `Session::drop`).
    unsafe { &*(data as *const AbortSet) }.all_cancelled()
}

/// Drops sequence `seq` from the KV/recurrent memory on drop unless disarmed.
/// Holds the raw handle (both fields `Copy`) so it never borrows the session.
pub(crate) struct SeqResetGuard {
    mem: raw::llama_memory_t,
    seq: raw::llama_seq_id,
    armed: bool,
}

impl SeqResetGuard {
    pub(crate) fn disarm(mut self) {
        self.armed = false;
    }
}

impl Drop for SeqResetGuard {
    fn drop(&mut self) {
        if self.armed {
            // SAFETY: mem is the live context's memory handle; the guard is a
            // local of `Session::drive`, which holds `&mut Session` (ctx alive)
            // for the guard's whole lifetime; no decode is in flight (every
            // decode path synchronizes before returning).
            unsafe { raw::llama_memory_seq_rm(self.mem, self.seq, -1, -1) };
        }
    }
}

impl SessionInner {
    /// Renders + tokenizes `req` and arms slot 0 via [`SessionInner::arm`].
    fn begin(
        &mut self,
        slot: &mut SlotState,
        req: &TurnRequest<'_>,
    ) -> Result<SeqResetGuard, EngineError> {
        let own;
        let rendered = match req.rendered {
            Some(r) => r,
            None => {
                own = render(&self.chat, req)?;
                &own
            }
        };
        let prompt = self.tokenize(&rendered.prompt)?;
        self.arm(
            slot,
            prompt,
            &req.sampling,
            Some(rendered),
            req.stop,
            req.max_tokens,
            req.cancel.clone(),
        )
    }

    /// Tokenizes a rendered prompt. add_special = true: the chat layer strips
    /// a leading BOS text when the vocab adds BOS (common/chat.cpp), and GLM's
    /// `[gMASK]<sop>` is template text with no vocab BOS, so this yields at
    /// most one BOS.
    pub(crate) fn tokenize(&self, text: &str) -> Result<Vec<raw::llama_token>, EngineError> {
        // SAFETY: vocab is live; text is valid for its byte length.
        let prompt = unsafe { tokenize_prompt(self.loaded.model().vocab(), text, true, true)? };
        if prompt.is_empty() {
            return Err(EngineError::Tokenize(0));
        }
        Ok(prompt)
    }

    /// Drops the slot's sequence (never the whole memory — other slots live
    /// there), restores a snapshot prefix (the cache is shared by all slots),
    /// applies the per-sequence context guard, builds the slot's sampler
    /// (with `rendered`'s tool-call grammar) and arms it for `step`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn arm(
        &mut self,
        slot: &mut SlotState,
        prompt: Vec<raw::llama_token>,
        sampling: &SamplingParams,
        rendered: Option<&Rendered>,
        stop: &[String],
        max_tokens: Option<u32>,
        cancel: Option<Arc<AtomicBool>>,
    ) -> Result<SeqResetGuard, EngineError> {
        let ctx = self.loaded.ctx();
        // SAFETY: ctx is live; no decode is in flight between steps.
        let mem = unsafe { raw::llama_get_memory(ctx.as_ptr()) };
        // SAFETY: as above; removes only this slot's sequence.
        unsafe { raw::llama_memory_seq_rm(mem, slot.seq, -1, -1) };
        let guard = SeqResetGuard {
            mem,
            seq: slot.seq,
            armed: true,
        };
        self.last_prompt_tokens.clone_from(&prompt);
        let restored = self.restore_prefix(mem, slot.seq, &prompt);
        let prompt_len = prompt.len() as u32;
        let n_ctx_seq = self.n_ctx_seq();
        if prompt_len >= n_ctx_seq {
            return Err(EngineError::ContextOverflow {
                prompt: prompt_len,
                n_ctx: n_ctx_seq,
            });
        }
        let limit = max_tokens
            .unwrap_or(self.config.n_predict)
            .min(n_ctx_seq - prompt_len);

        let chunk_cap = prefill_chunk_cap(
            self.config.batch_tokens as usize,
            n_ctx_seq as usize,
            self.p3.as_ref().map(|p| (p.prefill_k(), p.top_k)),
        );
        if chunk_cap == 0 {
            // n_ctx <= 1 cannot fit even one prompt token — llama_decode would
            // return 1; fail with that same code instead of chunking by zero.
            return Err(EngineError::Decode(1));
        }
        // One llama_decode must carry exactly one ubatch (the C lease-release
        // contract, §2.4).
        let n_ubatch = self.n_ubatch();
        // Prefill streaming (§7.6): whole-plane ubatches above the
        // --prefill-stream-min tail threshold; the per-token cap below it.
        let streaming = self.p3.as_ref().is_some_and(|p| p.stream_enabled());
        let vocab = self.loaded.model().vocab();
        // Parity recording and penalties need the vocab size (validated
        // non-negative by the load-time decode self-check probe).
        // SAFETY: vocab is live.
        let n_vocab = unsafe { raw::llama_vocab_n_tokens(vocab) };
        let n_vocab_us = usize::try_from(n_vocab)
            .map_err(|_| EngineError::InvalidModel("vocab size is negative".to_owned()))?;

        let grammar = match rendered {
            Some(r) if !r.grammar.is_empty() => r.parser.grammar_sampler(vocab)?,
            _ => None,
        };
        let armed = slot::Armed {
            sampler: Sampler::from_params(sampling, n_vocab, grammar)?,
            ffi: TurnFfi {
                prompt,
                n_past: restored,
                chunks: 0,
                chunk_cap: chunk_cap.min(n_ubatch),
                n_ubatch,
                streaming,
                n_vocab: n_vocab_us,
                verify: false,
                placement_before: ExpertPlacement::default(),
                traffic_before: 0,
                replay: false,
            },
        };
        slot.start(stop, cancel, sampling, prompt_len, limit, armed);
        Ok(guard)
    }

    /// After a sequence was dropped: restore the longest snapshotted prefix
    /// of `prompt` into `seq` (0 on miss).
    pub(crate) fn restore_into(
        &mut self,
        seq: raw::llama_seq_id,
        prompt: &[raw::llama_token],
    ) -> usize {
        // SAFETY: ctx is live and idle.
        let mem = unsafe { raw::llama_get_memory(self.loaded.ctx().as_ptr()) };
        self.restore_prefix(mem, seq, prompt)
    }

    /// Restores the longest snapshotted strict prefix of `prompt` into `seq`
    /// (§7.1); returns how many prompt tokens it covers (0 on miss). A
    /// rejected restore drops the seq and the entry and falls back to a full
    /// prefill — a delta is never decoded on top of a failed restore.
    fn restore_prefix(
        &mut self,
        mem: raw::llama_memory_t,
        seq: raw::llama_seq_id,
        prompt: &[raw::llama_token],
    ) -> usize {
        let Some(cache) = self.snapshots.as_mut() else {
            return 0;
        };
        let Some((n, bytes)) = cache.longest_prefix(prompt) else {
            return 0;
        };
        let ctx = self.loaded.ctx().as_ptr();
        // SAFETY: ctx is live and idle; `bytes` is a live arena range of
        // exactly `bytes.len()` bytes produced by `llama_state_seq_get_data`
        // on this model + context shape.
        let read = unsafe { raw::llama_state_seq_set_data(ctx, bytes.as_ptr(), bytes.len(), seq) };
        if read != 0 && read == bytes.len() {
            return n;
        }
        tracing::warn!(
            read,
            expected = bytes.len(),
            "snapshot restore rejected; full prefill"
        );
        // SAFETY: as in `begin`; clears whatever the failed read left.
        unsafe { raw::llama_memory_seq_rm(mem, seq, -1, -1) };
        cache.invalidate(n, prompt);
        0
    }

    /// Snapshots `seq` right after a clean prompt prefill, keyed by the
    /// prompt tokens (§7.1: the next turn's render shares exactly this
    /// prefix). Once per turn, off the per-token path.
    pub(crate) fn snapshot_prompt(&mut self, ffi: &TurnFfi, seq: raw::llama_seq_id) {
        let Some(cache) = self.snapshots.as_mut() else {
            return;
        };
        let ctx = self.loaded.ctx().as_ptr();
        // SAFETY: ctx is live; the prefill synchronized before returning.
        let size = unsafe { raw::llama_state_seq_get_size(ctx, seq) };
        let Some(dst) = cache.reserve(&ffi.prompt, size) else {
            return;
        };
        // SAFETY: `dst` is an exclusive arena range of exactly `size` bytes.
        let wrote = unsafe { raw::llama_state_seq_get_data(ctx, dst.as_mut_ptr(), size, seq) };
        if wrote == size {
            cache.commit();
        } else {
            cache.abort_short_write();
        }
    }

    /// Context window per sequence as llama.cpp sized it (padded/trimmed —
    /// never recomputed from config).
    pub(crate) fn n_ctx_seq(&self) -> u32 {
        // SAFETY: ctx is live.
        unsafe { raw::llama_n_ctx_seq(self.loaded.ctx().as_ptr()) }
    }

    /// Opens a slot's prefill (its first chunk): arms the I2 runtime-proof
    /// window and drops stale routing-trace records from a failed prior turn.
    pub(crate) fn prefill_begin(&self, ffi: &mut TurnFfi) {
        // I2 runtime proof window: this turn's prefill and first batch-1 decode
        // (the op-offload threshold is per ubatch, so both shapes are checked).
        ffi.verify = !self.pool_verified
            && self.p3.is_some()
            && self.offloaded()
            && substrate_backend() == Backend::Hip;
        ffi.placement_before = expert_placement();
        ffi.traffic_before = self.p3.as_ref().map_or(0, prefill_traffic);
        if let Some(p3) = self.p3.as_ref() {
            p3.drain_routing_trace();
        }
    }

    /// Closes a slot's prefill (after its final chunk): the I2 checks.
    pub(crate) fn prefill_end(&self, ffi: &TurnFfi) -> Result<(), EngineError> {
        if ffi.verify {
            check_experts_on_gpu(ffi.placement_before, expert_placement())?;
            if let Some(p3) = &self.p3 {
                check_pool_engaged(ffi.traffic_before, prefill_traffic(p3))?;
            }
        }
        Ok(())
    }

    /// After a slot's first generation step: the decode half of the I2 proof.
    pub(crate) fn verify_decode(&mut self, ffi: &TurnFfi) -> Result<(), EngineError> {
        if ffi.verify && !self.pool_verified {
            check_experts_on_gpu(ffi.placement_before, expert_placement())?;
            self.pool_verified = true;
        }
        Ok(())
    }

    /// Next chunk length for a slot's prefill and whether it streams (§7.6),
    /// given `n_gen` generation tokens already in the batch. Stream chunks
    /// are whole ubatches and always run alone; a routed-only chunk shares
    /// the ubatch with the generation tokens and, pooled, the routed-set
    /// budget `chunk_cap` (= K / top_k tokens) too.
    pub(crate) fn next_chunk(&self, ffi: &TurnFfi, n_gen: usize) -> (usize, bool) {
        let remaining = ffi.prompt.len() - ffi.n_past;
        let p3 = self.p3.as_ref();
        if ffi.streaming
            && p3.is_some_and(|p| p.stream_enabled())
            && remaining >= self.config.prefill_stream_min as usize
        {
            return (remaining.min(ffi.n_ubatch), true);
        }
        // Live K: a mid-turn degrade or mode switch must not leave a chunk
        // routing more experts than the slab holds.
        let cap = p3.map_or(ffi.chunk_cap, |p| {
            ffi.chunk_cap
                .min((p.prefill_k() / p.top_k.max(1)).max(1) as usize)
        });
        let room = cap
            .saturating_sub(n_gen)
            .min(ffi.n_ubatch - n_gen.min(ffi.n_ubatch));
        (remaining.min(room), false)
    }

    pub(crate) fn batch_clear(&mut self) {
        self.batch.inner.n_tokens = 0;
    }

    /// Appends one token to the shared batch; returns its logits row.
    pub(crate) fn batch_push(
        &mut self,
        tok: raw::llama_token,
        pos: usize,
        seq: raw::llama_seq_id,
        logits: bool,
    ) -> Result<i32, EngineError> {
        let i = self.batch.inner.n_tokens as usize;
        if i >= self.batch_cap {
            return Err(EngineError::InvalidConfig(format!(
                "step overflows the {}-token batch",
                self.batch_cap
            )));
        }
        // SAFETY: i < batch_cap slots allocated by Batch::init; the slot is
        // fully written before any decode observes it.
        unsafe {
            fill_slot(
                &mut self.batch.inner,
                i,
                tok,
                pos as raw::llama_pos,
                seq,
                logits,
            )
        };
        self.batch.inner.n_tokens += 1;
        Ok(i as i32)
    }

    /// Prompt tokens of a slot's next chunk into the batch; logits only on
    /// the prompt's last token (it bootstraps generation). Returns that row.
    pub(crate) fn push_chunk(
        &mut self,
        ffi: &TurnFfi,
        seq: raw::llama_seq_id,
        len: usize,
    ) -> Result<i32, EngineError> {
        let end = ffi.n_past + len;
        let mut row = -1;
        for p in ffi.n_past..end {
            row = self.batch_push(ffi.prompt[p], p, seq, p + 1 == ffi.prompt.len())?;
        }
        Ok(row)
    }

    /// One `llama_decode` of the filled batch (one ubatch, the lease-release
    /// contract §2.4). `stream`: the batch is one streamed prefill chunk,
    /// alone (§7.6), whose source abort points at its slot's cancel flag
    /// (`Some(None)`: a streamed chunk without one). `gen` generation
    /// tokens make it a policy step with a §11 timing row (`step` 0-based);
    /// a pure prefill batch is not (§6.2: it routes nearly every expert).
    /// `Ok(false)` when the abort callback aborted the compute (every
    /// participant cancelled).
    pub(crate) fn run_batch(
        &mut self,
        stream: Option<Option<&Arc<AtomicBool>>>,
        n_gen: u32,
        step: u64,
        t0: std::time::Instant,
    ) -> Result<bool, EngineError> {
        let n = self.batch.inner.n_tokens as usize;
        let n_ubatch = self.n_ubatch();
        if n == 0 || n > n_ubatch {
            return Err(EngineError::InvalidConfig(format!(
                "step of {n} tokens is not one {n_ubatch}-token ubatch"
            )));
        }
        if n_gen > 0 && self.test_fail_gen_step > 0 {
            self.test_fail_gen_step -= 1;
            if self.test_fail_gen_step == 0 {
                return Err(EngineError::Decode(1));
            }
        }
        let _scope = (n_gen == 0).then(p3_glue::PrefillScope::enter);
        // Stream chunks arm the C flag + read-ahead and pause the prefetcher;
        // the guard disarms/resumes on every exit path (§7.6.4).
        struct StreamGuard<'a>(&'a p3_glue::SessionHooks);
        impl Drop for StreamGuard<'_> {
            fn drop(&mut self) {
                self.0.set_prefill_stream(false);
                // llama_decode returned: no H2D in flight.
                self.0.release_stream();
            }
        }
        let (_stream_guard, _stream_abort) = match (stream, self.p3.as_ref()) {
            (Some(cancel), Some(p3)) => {
                p3.set_prefill_stream(true);
                (
                    Some(StreamGuard(p3)),
                    cancel.map(|c| p3.stream_abort_guard(c)),
                )
            }
            _ => (None, None),
        };
        let ctx = self.loaded.ctx();
        // SAFETY: ctx is a live context; batch slots 0..n_tokens fully written.
        let code = unsafe { raw::llama_decode(ctx.as_ptr(), self.batch.inner) };
        if code != 0 {
            // Synchronize first: the failed decode may have launched a compute
            // whose expert-copy leases release only at the next compute.
            // SAFETY: ctx is live.
            unsafe { raw::llama_synchronize(ctx.as_ptr()) };
            // 2 = aborted; a cancel only if the predicate both the abort
            // callback and the stream source fire on still holds (the set
            // is cleared only after this returns).
            return if code == 2 && self.abort.all_cancelled() {
                Ok(false)
            } else {
                Err(EngineError::Decode(code))
            };
        }
        if n_gen > 0 {
            p3_glue::advance_decode_step();
        }
        // The lease-release contract: synchronize before the next compute.
        // SAFETY: ctx is live.
        unsafe { raw::llama_synchronize(ctx.as_ptr()) };
        if let Some(t) = self.timing.as_mut() {
            // §11 step boundary: snapshot (and reset) the P3/store
            // accumulators into this step's row; a prefill-only decode's
            // are discarded, not charged to the next generation row. H2D is
            // measured only on the async_h2d copy stream; policy stays
            // unmeasured.
            let io = p3_glue::take_step_io_stats();
            if n_gen == 0 {
                return Ok(true);
            }
            let ns2ms = |ns: u64| ns as f64 / 1_000_000.0;
            t.row(crate::timing::StepTiming {
                step,
                tokens: n_gen,
                wall_ms: t0.elapsed().as_secs_f64() * 1_000.0,
                p3_wait_ms: ns2ms(io.p3_wait_ns),
                disk_worker_ms: io.disk_ns.map_or(f64::NAN, ns2ms),
                disk_bytes: io.disk_bytes.unwrap_or(0),
                h2d_ms: io.h2d.map_or(f64::NAN, |h| h.0),
                h2d_bytes: io.h2d.map_or(0, |h| h.1),
                policy_ms: f64::NAN,
            });
        }
        Ok(true)
    }

    fn n_ubatch(&self) -> usize {
        // SAFETY: ctx is a live context; n_ubatch is a load-time constant (§4.2).
        unsafe { raw::llama_n_ubatch(self.loaded.ctx().as_ptr()) as usize }
    }

    /// Drops `seq` from the KV/recurrent memory (cancel, failed batch).
    pub(crate) fn seq_rm(&self, seq: raw::llama_seq_id) {
        // SAFETY: ctx is live and idle (every decode synchronizes before
        // returning); removes only this sequence.
        unsafe {
            raw::llama_memory_seq_rm(
                raw::llama_get_memory(self.loaded.ctx().as_ptr()),
                seq,
                -1,
                -1,
            )
        };
    }

    /// Whether the store-backed source reports `FatalIo` (§5.2).
    pub(crate) fn storage_lost(&self) -> bool {
        self.p3
            .as_ref()
            .and_then(|p| p.storage_health())
            .is_some_and(|h| h == golem_store::InstanceHealth::FatalIo)
    }

    /// Sizes the device cache for streamed prefill or decode (#135).
    pub(crate) fn set_vram_mode(&mut self, stream: bool) {
        if let Some(p) = self.p3.as_mut() {
            p.set_vram_mode(stream);
        }
    }

    /// Slab-failure recovery (§2.4): a `Decode(-2)` shrinks the device
    /// cache; maps to `DeviceCacheDegraded` when K changed.
    pub(crate) fn recover_slab(&mut self, e: EngineError) -> EngineError {
        if let EngineError::Decode(-2) = e
            && let Some(p) = self.p3.as_mut()
        {
            return match p.shrink_after_slab_failure() {
                Ok(Some(k)) => EngineError::DeviceCacheDegraded(k),
                Ok(None) => e,
                Err(e) => e,
            };
        }
        e
    }

    /// Resets the per-step accumulators right before the first generation
    /// decode, so prefill activity and stale rows never bleed into step 0.
    pub(crate) fn start_decode(&mut self) -> Result<(), EngineError> {
        if let Some(t) = self.timing.as_mut() {
            t.flush(&mut std::io::sink())?; // stale rows from a failed prior turn never leak in
        }
        let _ = p3_glue::take_step_io_stats();
        Ok(())
    }

    /// Samples from logits row `idx`, recording a [`ParityStep`] at
    /// `position` when `parity` is set.
    pub(crate) fn sample(
        &self,
        sampler: &Sampler,
        ffi: &TurnFfi,
        idx: i32,
        parity: Option<&mut Vec<ParityStep>>,
        position: u32,
    ) -> raw::llama_token {
        let ctx = self.loaded.ctx();
        // SAFETY: sampler chain and ctx are live; the preceding successful
        // decode emitted logits for row idx.
        let token = unsafe { raw::llama_sampler_sample(sampler.as_ptr(), ctx.as_ptr(), idx) };
        if let Some(steps) = parity {
            // Pure read of the same row — no decode behavior change.
            // SAFETY: row idx holds n_vocab logits (as above).
            let (top1, top2) = unsafe { parity::top2_logits_ith(ctx.as_ptr(), idx, ffi.n_vocab) };
            steps.push(ParityStep {
                position,
                token,
                top1,
                top2,
            });
        }
        token
    }

    pub(crate) fn is_eog(&self, tok: raw::llama_token) -> bool {
        // SAFETY: vocab is live; tok is a valid token id from the sampler.
        unsafe { raw::llama_vocab_is_eog(self.loaded.model().vocab(), tok) }
    }

    /// Writes `tok`'s piece into `out` (cleared first): no allocation once
    /// `out` holds the longest piece.
    pub(crate) fn piece_into(
        &self,
        tok: raw::llama_token,
        out: &mut Vec<u8>,
    ) -> Result<(), EngineError> {
        // SAFETY: vocab is live; tok is a valid token id from the sampler.
        unsafe { token_piece_into(self.loaded.model().vocab(), tok, out) }
    }

    fn offloaded(&self) -> bool {
        self.config.n_gpu_layers > 0
    }
}

/// Owned sampler chain; `Drop` calls `llama_sampler_free`, which frees the
/// samplers added to the chain.
pub(crate) struct Sampler {
    ptr: *mut raw::llama_sampler,
}

impl Sampler {
    /// Chain per `p`: penalties (when either is nonzero — applied in greedy
    /// mode too), the request's tool-call `grammar` (owned sampler, moved
    /// into the chain), then greedy for `temperature <= 0`, else
    /// top_k (if > 0) → top_p → temp → dist(seed).
    pub(crate) fn from_params(
        p: &SamplingParams,
        n_vocab: i32,
        grammar: Option<*mut raw::llama_sampler>,
    ) -> Result<Self, EngineError> {
        // SAFETY: returns llama.cpp's default sampler-chain params by value.
        let params = unsafe { raw::llama_sampler_chain_default_params() };
        // SAFETY: params are defaults; freed exactly once in Drop via llama_sampler_free.
        let chain = unsafe { raw::llama_sampler_chain_init(params) };
        if chain.is_null() {
            if let Some(g) = grammar {
                // SAFETY: owned, never added to a chain.
                unsafe { raw::llama_sampler_free(g) };
            }
            return Err(EngineError::SamplerInit);
        }
        let s = Sampler { ptr: chain };
        let add = |smpl: *mut raw::llama_sampler| {
            // SAFETY: chain is a live sampler chain; smpl ownership transfers
            // to the chain (freed with it).
            unsafe { raw::llama_sampler_chain_add(chain, smpl) }
        };
        if p.frequency_penalty != 0.0 || p.presence_penalty != 0.0 {
            // SAFETY: plain constructor; ownership passes to the chain.
            add(unsafe {
                raw::llama_sampler_init_penalties(
                    n_vocab,
                    64,
                    1.0,
                    p.frequency_penalty,
                    p.presence_penalty,
                )
            });
        }
        if let Some(g) = grammar {
            add(g);
        }
        if p.temperature <= 0.0 {
            // SAFETY: plain constructor; ownership passes to the chain.
            add(unsafe { raw::llama_sampler_init_greedy() });
            return Ok(s);
        }
        if p.top_k > 0 {
            let k = i32::try_from(p.top_k).unwrap_or(i32::MAX);
            // SAFETY: plain constructor; ownership passes to the chain.
            add(unsafe { raw::llama_sampler_init_top_k(k) });
        }
        // SAFETY: plain constructors; ownership passes to the chain.
        add(unsafe { raw::llama_sampler_init_top_p(p.top_p, 1) });
        // SAFETY: as above.
        add(unsafe { raw::llama_sampler_init_temp(p.temperature) });
        // SAFETY: as above.
        add(unsafe { raw::llama_sampler_init_dist(p.seed) });
        Ok(s)
    }

    fn as_ptr(&self) -> *mut raw::llama_sampler {
        self.ptr
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: ptr came from llama_sampler_chain_init and is freed exactly once here.
            unsafe { raw::llama_sampler_free(self.ptr) };
        }
    }
}

/// Appends `bytes` to the pending byte buffer and emits — into `out` — the
/// longest valid UTF-8 prefix; an incomplete trailing sequence stays pending
/// until more bytes arrive (`std::str::from_utf8`: `valid_up_to` for the flush
/// length, `error_len == None` = keep the tail). Invalid bytes surface as
/// U+FFFD (`from_utf8_lossy` semantics). Byte-buffered because a
/// byte-fallback vocab legitimately splits multi-byte characters across pieces;
/// per-piece lossy conversion would corrupt them.
pub(crate) fn emit_valid_prefix(pending: &mut Vec<u8>, out: &mut String, bytes: &[u8]) {
    pending.extend_from_slice(bytes);
    loop {
        match std::str::from_utf8(pending) {
            Ok(valid) => {
                out.push_str(valid);
                pending.clear();
                break;
            }
            Err(e) => {
                let ok = e.valid_up_to();
                // Guaranteed valid: from_utf8 error positions are char boundaries.
                out.push_str(std::str::from_utf8(&pending[..ok]).unwrap());
                pending.drain(..ok);
                match e.error_len() {
                    None => break, // incomplete sequence — wait for more bytes
                    Some(bad) => {
                        out.push('\u{FFFD}');
                        pending.drain(..bad);
                    }
                }
            }
        }
    }
}

/// Turn-end flush: complete any character pending on a real boundary, then
/// lossily drain the residue (a multi-byte character truncated mid-sequence
/// can never complete).
pub(crate) fn flush_utf8_tail(pending: &mut Vec<u8>, out: &mut String) {
    emit_valid_prefix(pending, out, &[]);
    if !pending.is_empty() {
        out.push_str(&String::from_utf8_lossy(pending));
        pending.clear();
    }
}

/// Writes one slot of the malloc'd batch buffer: token, position, single
/// sequence id `seq`, and the logits flag. llama_batch_init mallocs and does not zero, so
/// every used slot must be written before decode.
///
/// # Safety
/// `i` must be below the `n_tokens` the `Batch` was initialized with.
unsafe fn fill_slot(
    b: &mut raw::llama_batch,
    i: usize,
    token: i32,
    pos: raw::llama_pos,
    seq: raw::llama_seq_id,
    wants_logits: bool,
) {
    // SAFETY: caller guarantees i is below the allocated n_tokens; each field of
    // slot i is written once, before any decode observes it.
    unsafe {
        *b.token.add(i) = token;
        *b.pos.add(i) = pos;
        *b.n_seq_id.add(i) = 1;
        **b.seq_id.add(i) = seq;
        *b.logits.add(i) = if wants_logits { 1 } else { 0 };
    }
}

/// Tokenizes `text` (two-pass: null-buffer sizing call, whose negative return is
/// the negated token count needed per the llama.cpp tokenize contract — not an
/// error; the real call's negative return maps to `Tokenize`).
///
/// # Safety
/// `vocab` must be a live llama_vocab owned by the loaded model.
unsafe fn tokenize_prompt(
    vocab: *const raw::llama_vocab,
    text: &str,
    add_special: bool,
    parse_special: bool,
) -> Result<Vec<raw::llama_token>, EngineError> {
    // SAFETY: text is valid for text.len() bytes; the null/0 call only sizes.
    let needed = unsafe {
        raw::llama_tokenize(
            vocab,
            text.as_ptr().cast(),
            text.len() as i32,
            std::ptr::null_mut(),
            0,
            add_special,
            parse_special,
        )
    };
    let n = needed.unsigned_abs() as usize;
    let mut tokens = vec![0 as raw::llama_token; n];
    if n > 0 {
        // SAFETY: tokens holds exactly the count the sizing call demanded.
        let written = unsafe {
            raw::llama_tokenize(
                vocab,
                text.as_ptr().cast(),
                text.len() as i32,
                tokens.as_mut_ptr(),
                n as i32,
                add_special,
                parse_special,
            )
        };
        if written < 0 {
            return Err(EngineError::Tokenize(written));
        }
        // Deterministically == n; truncate defensively anyway.
        tokens.truncate(written as usize);
    }
    Ok(tokens)
}

/// Detokenizes one token into `out`'s capacity (the `lstrip = 0, special =
/// false` call pinned for the emitter — special tokens render as empty
/// pieces and are skipped). A too-small buffer grows to the demanded size
/// and retries once.
///
/// # Safety
/// `vocab` must be a live llama_vocab owned by the loaded model; `token` must be
/// a valid token id.
unsafe fn token_piece_into(
    vocab: *const raw::llama_vocab,
    token: raw::llama_token,
    out: &mut Vec<u8>,
) -> Result<(), EngineError> {
    out.clear();
    for _ in 0..2 {
        let cap = i32::try_from(out.capacity()).unwrap_or(i32::MAX);
        // SAFETY: llama writes at most `cap` bytes into out's spare capacity.
        let n = unsafe {
            raw::llama_token_to_piece(vocab, token, out.as_mut_ptr().cast(), cap, 0, false)
        };
        if n >= 0 {
            // SAFETY: llama initialized the first n ≤ cap bytes.
            unsafe { out.set_len(n as usize) };
            return Ok(());
        }
        out.reserve_exact(n.unsigned_abs() as usize);
    }
    Err(EngineError::Detokenize(token))
}

/// Expert copies the pool has served (demand page-ins + fast hits), cumulative.
/// Pool copies plus streamed planes: streamed planes are expert copies served
/// on the GPU split too (`stream_source` fires only from that copy loop, §7.6).
fn prefill_traffic(p: &p3_glue::SessionHooks) -> u64 {
    pool_traffic(&p.stats) + p.stream_stats().map_or(0, |s| s.planes)
}

fn pool_traffic(stats: &P3Stats) -> u64 {
    use std::sync::atomic::Ordering::Relaxed;
    stats.demand_slow_pageins.load(Relaxed) + stats.fast_hits.load(Relaxed)
}

/// Runtime proof of design I2 for a pooled offloaded run: between the
/// `before` and `now` traffic snapshots (pool copies plus streamed planes),
/// the GPU split's expert-copy path must have served copies
/// (`ensure_host_resident` and `stream_source` fire only from there). Zero
/// traffic means the expert `MUL_MAT_ID` ran somewhere else — silently on
/// the CPU, or the experts never left the device — and decoding on would be
/// a silent violation.
fn check_pool_engaged(before: u64, now: u64) -> Result<(), EngineError> {
    if now == before {
        return Err(EngineError::PoolInit(
            "the expert pool served no copies during this pooled prefill: expert \
             MUL_MAT_ID is not running through the GPU split (CPU fallback, or experts \
             already device-resident); refusing to decode silently off-pool"
                .into(),
        ));
    }
    Ok(())
}

/// Expert `MUL_MAT_ID` nodes the scheduler assigned per backend type,
/// cumulative for the process (counted once per graph build, P1
/// `golem_get_expert_placement_stats`). Two independent loads, not a
/// consistent snapshot: compare deltas over a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExpertPlacement {
    /// Expert `MUL_MAT_ID` nodes assigned to a GPU backend (cumulative).
    pub on_gpu: u64,
    /// Expert `MUL_MAT_ID` nodes assigned to a CPU backend (cumulative).
    pub on_cpu: u64,
}

/// Reads the P1 scheduler placement counters (see [`ExpertPlacement`]).
pub fn expert_placement() -> ExpertPlacement {
    // SAFETY: FFI getter; returns a process-lifetime static, never null.
    let block = unsafe { raw::golem_get_expert_placement_stats() };
    assert!(
        !block.is_null(),
        "golem_get_expert_placement_stats returned null"
    );
    // SAFETY: layout contract from the C++ definition (bindgen renders the
    // struct opaque): 2 contiguous lock-free u64 counters, on_gpu then on_cpu.
    let counters = unsafe { &*block.cast::<[std::sync::atomic::AtomicU64; 2]>() };
    let load = |i: usize| counters[i].load(std::sync::atomic::Ordering::Relaxed);
    ExpertPlacement {
        on_gpu: load(0),
        on_cpu: load(1),
    }
}

/// Runtime proof of design I2 over the graphs built between `before` and
/// `after`: every expert `MUL_MAT_ID` on the GPU, and at least one seen (a
/// zero count would prove nothing). A future opt-in CPU expert-math mode
/// (§1 O1) would report the CPU count here instead of refusing; the counters are
/// process-global, so that mode also needs per-session counts if strict and
/// opted-in sessions can share a process. The window covers graph builds:
/// a reused graph keeps the placement of the build that produced it. HIP
/// substrate only (the gate at the call site).
fn check_experts_on_gpu(
    before: ExpertPlacement,
    after: ExpertPlacement,
) -> Result<(), EngineError> {
    let on_gpu = after.on_gpu - before.on_gpu;
    let on_cpu = after.on_cpu - before.on_cpu;
    if on_cpu != 0 || on_gpu == 0 {
        return Err(EngineError::BackendUnavailable(format!(
            "the scheduler placed {on_cpu} expert MUL_MAT_ID node(s) on the CPU and {on_gpu} \
             on the GPU since this pooled turn began (0 and 0 = no expert graph built); every \
             expert op must run on the GPU \
             (design I2). Inspect split causes with GGML_SCHED_DEBUG=2 (1.off/1.wgt); a \
             flagless session loaded first in this process pins the op-offload threshold \
             at upstream's 32. The counts are process-global: another session decoding in \
             this process during the turn is included in them"
        )));
    }
    Ok(())
}

/// P4 device expert-copy cache counters (§3.4), read when the RAM expert pool
/// is enabled and the config requests GPU offload (the split scheduler is the
/// only P2 call-site).
#[derive(Debug, Clone, PartialEq)]
pub struct DeviceCacheStats {
    /// Expert plane bytes the split scheduler copied into the device cache.
    pub bytes_copied: u64,
    /// Expert plane bytes served from the device cache (no copy).
    pub bytes_skipped: u64,
    /// Cached experts evicted under the soft budget.
    pub evictions: u64,
    /// Expert planes copied host→device into the device cache.
    pub cache_copies: u64,
    /// Expert planes copied host→device by streamed prefill.
    pub stream_copies: u64,
    /// `bytes_skipped / (bytes_skipped + bytes_copied)`; 0.0 with no traffic.
    pub skip_ratio: f64,
}

/// `skipped / (skipped + copied)`; 0.0 when there is no traffic at all.
fn skip_ratio(bytes_copied: u64, bytes_skipped: u64) -> f64 {
    let total = bytes_copied.saturating_add(bytes_skipped);
    if total == 0 {
        0.0
    } else {
        bytes_skipped as f64 / total as f64
    }
}

/// The §10 plan inputs for this config (`ws` = fitted `WS_MEASURED`). The
/// context ubatch mirrors `load_with_hooks`; an explicit K prices its slab
/// as `(K + 1) × Σ_layers unit bytes` — target 9 × 644,022,272 B = 5.4 GiB
/// at K = 8.
fn plan_input(config: &EngineConfig, m: &golem_plan::ModelFacts) -> golem_plan::PlanInput {
    let ubatch = if config.prefill_stream {
        match config.prefill_ubatch {
            0 => 2048,
            u => u,
        }
    } else {
        config.batch_tokens
    };
    golem_plan::PlanInput {
        ctx: config.n_ctx,
        parallel: config.parallel,
        prefill_ubatch: ubatch.min(config.n_ctx),
        prefill_stream: config.prefill_stream,
        vram_experts: m.device_cache_bytes(u64::from(config.device_cache_experts)),
        experts: if config.ram_experts_gib == 0 {
            golem_plan::ExpertResidency::Resident(resident_expert_bytes(config, m))
        } else {
            golem_plan::ExpertResidency::Pooled {
                ram_experts: config.ram_experts_gib << 30,
            }
        },
        snapshot_ram: config.snapshot_ram_bytes.unwrap_or(0),
        ws: golem_plan::WS_MEASURED,
    }
}

/// Pool off: routed-expert bytes of the top `n_gpu_layers` expert layers not
/// pinned below `cpu_moe_layers`, at the mean per-layer size. Target, all
/// offloaded: 185.5 GB ≫ 24 GiB; all pinned: 0. `cpu_moe_layers` counts
/// blocks from 0, so dense-lead blocks pin nothing; each missed layer would
/// under-count 288 × 3 × ~5.1 MB ≈ 4.4 GB. Upper bound: assumes every
/// offloaded block (top `n_gpu_layers`) carries experts.
fn resident_expert_bytes(config: &EngineConfig, m: &golem_plan::ModelFacts) -> u64 {
    let layers = m.expert_layers;
    let pinned = config
        .cpu_moe_layers
        .unwrap_or(0)
        .saturating_sub(m.first_expert_block);
    let gpu = config.n_gpu_layers.min(layers.saturating_sub(pinned));
    m.expert_bytes_total / u64::from(layers.max(1)) * u64::from(gpu)
}

/// Load-time §10 VRAM feasibility (§6.4), same `golem_plan::plan` as
/// `golem plan`, against the VRAM free before the load. RAM admission is
/// the CLI's `fit_ram` over the same `ram_terms`, so RAM is not re-judged
/// here. An explicit K whose slab overruns what the other terms leave is
/// clamped to the largest K the plan fits (WARN), returned as `Some(k)`;
/// an explicit K is not refit after load, so a short estimate falls back to
/// `shrink_after_slab_failure`. Below top-k there is no
/// usable K: `InvalidConfig` naming the max. Otherwise Err = `PlanInfeasible`
/// with the failing inequality.
fn check_plan(
    config: &EngineConfig,
    m: &golem_plan::ModelFacts,
    vram_free: u64,
) -> Result<(golem_plan::PlanReport, Option<u32>), EngineError> {
    let mach = golem_plan::Machine {
        vram_total: vram_free,
        ram_total: u64::MAX,
        ram_free: u64::MAX,
        rocm_overhead: golem_plan::ROCM_OVERHEAD,
    };
    let mut input = plan_input(config, m);
    let mut clamped = None;
    let k = u64::from(config.device_cache_experts);
    if matches!(input.experts, golem_plan::ExpertResidency::Pooled { .. }) {
        let max = golem_plan::clamp_device_cache_k(m, &mach, &input, k).map_err(|max| {
            EngineError::InvalidConfig(format!(
                "device_cache_experts {k}: at most {max} fit in {:.1} GiB free VRAM \
                 (below expert_used_count {}); lower --ctx or --parallel",
                vram_free as f64 / f64::from(1u32 << 30),
                m.top_k
            ))
        })?;
        if max < k {
            tracing::warn!(
                requested = k,
                clamped = max,
                "device_cache_experts exceeds free VRAM; clamped"
            );
            input.vram_experts = m.device_cache_bytes(max);
            // max < k ≤ u32::MAX.
            clamped = Some(max as u32);
        }
    }
    let r = golem_plan::plan_at_frac(m, &mach, &input, config.vram_experts_frac);
    if r.feasible {
        Ok((r, clamped))
    } else {
        Err(EngineError::PlanInfeasible(Box::new(r)))
    }
}

/// `(free, total)` device VRAM on a GPU build (`hipMemGetInfo`); `None`
/// on CPU builds or when the runtime reports an error.
pub fn device_vram() -> Option<(u64, u64)> {
    #[cfg(golem_backend_hip)]
    return golem_llama_sys::pinned::free_device_mem().map(|(f, t)| (f as u64, t as u64));
    #[cfg(not(golem_backend_hip))]
    None
}

/// Runs [`check_plan`] for a pooled or offloaded session on a GPU build;
/// `None` for a CPU-only session or when no VRAM reading exists.
fn load_plan(
    config: &EngineConfig,
) -> Result<Option<(golem_plan::PlanReport, Option<u32>)>, EngineError> {
    if config.ram_experts_gib == 0 && config.n_gpu_layers == 0 {
        return Ok(None);
    }
    let Some((free, _)) = device_vram() else {
        return Ok(None);
    };
    let facts = match config.model_facts {
        Some(f) => f,
        None => golem_plan::ModelFacts::from_gguf(&config.model_path)
            .map_err(|e| EngineError::InvalidModel(format!("plan: {e}")))?,
    };
    check_plan(config, &facts, free).map(Some)
}

/// Builds the snapshot cache (§7.1) per `config.snapshot_ram_bytes`.
/// Floor = `parallel × snapshot_bytes(n_ctx)`: one full-context snapshot per
/// slot — below it long chats silently lose reuse, so smaller budgets are
/// refused. Target: 145.56 MiB + 19,712 B × 100k ≈ 2.0 GiB at parallel 1.
fn snapshot_cache(config: &EngineConfig) -> Result<Option<SnapshotCache>, EngineError> {
    if config.snapshot_ram_bytes == Some(0) {
        return Ok(None);
    }
    let facts = match config.model_facts {
        Some(f) => f,
        None => golem_plan::ModelFacts::from_gguf(&config.model_path)
            .map_err(|e| EngineError::InvalidModel(format!("snapshot sizing: {e}")))?,
    };
    let (floor, default) = golem_plan::snapshot_arena(&facts, config.parallel, config.n_ctx);
    let bytes = match config.snapshot_ram_bytes {
        None => default,
        Some(n) if n < floor => {
            return Err(EngineError::InvalidConfig(format!(
                "snapshot_ram_bytes {n} < floor {floor} (parallel {} × one {}-token snapshot)",
                config.parallel, config.n_ctx
            )));
        }
        Some(n) => n,
    };
    let len = usize::try_from(bytes)
        .map_err(|_| EngineError::InvalidConfig("snapshot_ram_bytes > usize".into()))?;
    Ok(Some(SnapshotCache::new(
        p3_glue::PinnedArena::new(len)?,
        config.n_ctx as usize,
    )))
}

/// The session's one batch: `llama_n_batch` token slots (≥ every chunk and
/// every decode step, both capped at one ubatch), one seq id per token.
fn shared_batch(loaded: &LoadedModel, config: &EngineConfig) -> (Batch, usize, Arc<AbortSet>) {
    // SAFETY: ctx is live.
    let n = (unsafe { raw::llama_n_batch(loaded.ctx().as_ptr()) } as usize)
        .max(config.parallel as usize);
    let abort = Arc::new(AbortSet::new(config.parallel as usize));
    // SAFETY: ctx is live and idle; the set outlives the context's use of
    // it (`Session::drop` clears the callback before `abort` drops).
    unsafe {
        raw::llama_set_abort_callback(
            loaded.ctx().as_ptr(),
            Some(abort_requested),
            Arc::as_ptr(&abort) as *mut std::ffi::c_void,
        )
    };
    (Batch::init(n as i32, 0, 1), n, abort)
}

/// Prefill chunk length in tokens. Base cap `min(batch_tokens, n_ctx - 1)`:
/// one KV slot stays free so the first generated token always has a position
/// to decode into. With the expert pool on (§2.4), also capped at
/// `max(1, K / top_k)`: one chunk routes at most `cap × top_k` distinct
/// experts per layer and the device slab has K slots — a larger routed set
/// aborts mid-decode. A chunk ≤ n_ubatch is exactly one ubatch, so one
/// `llama_decode` stays one ubatch and the context needs no re-creation;
/// `n_batch`/`n_ubatch` are deliberately untouched.
fn prefill_chunk_cap(batch_tokens: usize, n_ctx: usize, pool: Option<(u32, u32)>) -> usize {
    let mut cap = batch_tokens.min(n_ctx.saturating_sub(1));
    if let Some((k, top_k)) = pool {
        cap = cap.min((k as usize / top_k.max(1) as usize).max(1));
    }
    cap
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::Ordering;

    fn glm() -> golem_plan::ModelFacts {
        golem_plan::ModelFacts {
            t_ne: 8_966_188_280,
            shexp: 1_123_024_896,
            n_expert: 288,
            top_k: 8,
            expert_layers: 42,
            first_expert_block: 0,
            max_plane_stride: 6_881_280,
            expert_bytes_total: 185_478_414_336,
            kv_per_token: 19_712,
            rs_bytes: 152_633_344,
            sparse_attn_layers: 11,
            dense_below_ctx: 2_052,
        }
    }

    /// Target, 24 GiB free: 100k ctx + streamed 512 ubatch (fitted ws 0.79
    /// GiB) + explicit K = 8 (9 × 644 MB = 5.4 GiB) is 8.35 + 1.84 + 0.14 +
    /// 0.79 + 5.4 + 3.69 + 2.64 ≈ 22.9 GiB → fits; K = 12 (7.8 GiB) →
    /// 25.3 GiB, clamped to the ≈ 6.55 GiB left: ⌊6.55 / 0.6⌋ − 1 = 9.
    #[test]
    fn load_check_clamps_an_overcommitted_k() {
        let mut cfg = EngineConfig {
            n_ctx: 100_000,
            ram_experts_gib: 57,
            prefill_stream: true,
            // ubatch 512 keeps the fitted ws (810 MiB) from eating the
            // explicit K = 8 cache at this ctx.
            prefill_ubatch: 512,
            device_cache_experts: 8,
            ..EngineConfig::default()
        };
        let r = check_plan(&cfg, &glm(), 24 << 30).unwrap().0;
        assert_eq!(r.term("device_expert_cache"), 9 * 644_022_272);
        assert_eq!(r.term("stream_slabs"), 3_963_617_280);
        cfg.device_cache_experts = 12;
        let (r, k) = check_plan(&cfg, &glm(), 24 << 30).unwrap();
        assert_eq!(k, Some(9));
        assert_eq!(r.term("device_expert_cache"), 10 * 644_022_272);
        assert!(r.vram_headroom >= 0);
        // Derived K: the cache term is whatever the other terms leave.
        cfg.device_cache_experts = 0;
        let r = check_plan(&cfg, &glm(), 24 << 30).unwrap().0;
        assert_eq!(r.vram_headroom, 0);
        // ctx 100k at ubatch 512: fitted ws 0.79 GiB; 24 GiB − (8.35 t_ne +
        // 1.84 kv + 0.14 rs + 0.79 ws + 3.69 slabs + 2.64 ROCm) ≈ 6.55 GiB.
        assert!(r.term("device_expert_cache") > 5 << 30);
        // Half of that 6.55 GiB cache holds K = 4 < top_k 8.
        cfg.vram_experts_frac = 0.5;
        let e = check_plan(&cfg, &glm(), 24 << 30).unwrap_err();
        assert!(e.to_string().contains("device_expert_cache "), "{e}");
    }

    /// Issue #102, real target: `--device-cache-experts 512` at ctx 262,144
    /// on 23.9 GiB free priced 513 × 644 MB ≈ 307 GiB + 4.8 GiB KV. Now the
    /// other terms are 8.35 t_ne + 4.81 kv + 0.14 rs + 1.57 ws + 2.64 ROCm =
    /// 18,803,322,759 B (17.5 GiB), leaving 6.4 GiB → ⌊6.86 / 0.644⌋ − 1 =
    /// K = 9. The top-k boundary: others + 9 slots fits K = 8 exactly; one
    /// byte less leaves K = 7 < 8, refused naming the max. At 12 GiB free
    /// less than one slab slot (644 MB) is left: max 0. Never the raw
    /// 328.5 GiB inequality.
    #[test]
    fn explicit_k_512_clamps_or_names_the_max() {
        let cfg = EngineConfig {
            n_ctx: 262_144,
            ram_experts_gib: 57,
            device_cache_experts: 512,
            ..EngineConfig::default()
        };
        let (r, k) = check_plan(&cfg, &glm(), (239 << 30) / 10).unwrap();
        assert_eq!(k, Some(9));
        assert_eq!(r.term("device_expert_cache"), 10 * 644_022_272);
        let at_top_k = 18_803_322_759 + 9 * 644_022_272;
        assert_eq!(check_plan(&cfg, &glm(), at_top_k).unwrap().1, Some(8));
        let e = check_plan(&cfg, &glm(), at_top_k - 1).unwrap_err();
        assert!(e.to_string().contains("at most 7 fit"), "{e}");
        let e = check_plan(&cfg, &glm(), 12 << 30).unwrap_err();
        let msg = e.to_string();
        assert!(matches!(e, EngineError::InvalidConfig(_)), "{msg}");
        assert!(
            msg.contains("device_cache_experts 512: at most 0 fit"),
            "{msg}"
        );
    }

    /// Pool off, 4 layers × 1 GiB experts vs 4 GiB free: all offloaded
    /// layers' experts alone exceed VRAM; pinning every MoE layer to the CPU
    /// leaves only the non-expert terms.
    #[test]
    fn pool_off_load_check_counts_resident_experts() {
        let m = golem_plan::ModelFacts {
            t_ne: 256 << 20,
            shexp: 0,
            n_expert: 8,
            top_k: 2,
            expert_layers: 4,
            first_expert_block: 0,
            max_plane_stride: 1 << 20,
            expert_bytes_total: 4 << 30,
            kv_per_token: 1024,
            rs_bytes: 0,
            sparse_attn_layers: 4,
            dense_below_ctx: 0,
        };
        let mut cfg = EngineConfig {
            n_ctx: 4096,
            n_gpu_layers: 99,
            ..EngineConfig::default()
        };
        let e = check_plan(&cfg, &m, 4 << 30).unwrap_err();
        assert!(
            matches!(&e, EngineError::PlanInfeasible(r) if r.term("resident_experts") == 4 << 30),
            "{e}"
        );
        cfg.cpu_moe_layers = Some(4);
        let r = check_plan(&cfg, &m, 4 << 30).unwrap().0;
        assert_eq!(r.term("resident_experts"), 0);
        assert_eq!(r.term("ram_pool"), 0);
    }

    #[test]
    fn resident_experts_skip_dense_lead() {
        let m = golem_plan::ModelFacts {
            first_expert_block: 1,
            ..glm()
        };
        let cfg = EngineConfig {
            n_gpu_layers: 99,
            cpu_moe_layers: Some(2),
            ..EngineConfig::default()
        };
        let per_layer = m.expert_bytes_total / u64::from(m.expert_layers);
        assert_eq!(
            resident_expert_bytes(&cfg, &m),
            per_layer * u64::from(m.expert_layers - 1)
        );
    }

    #[test]
    fn skip_ratio_math() {
        assert_eq!(skip_ratio(0, 0), 0.0, "no traffic ⇒ 0.0, not NaN");
        assert_eq!(skip_ratio(3, 0), 0.0, "all copied ⇒ 0.0");
        assert_eq!(skip_ratio(0, 5), 1.0, "all skipped ⇒ 1.0");
        assert_eq!(skip_ratio(1, 3), 0.75);
    }

    #[test]
    fn expert_placement_check_requires_gpu_and_rejects_any_cpu() {
        let p = |on_gpu, on_cpu| ExpertPlacement { on_gpu, on_cpu };
        let before = p(10, 3);
        check_experts_on_gpu(before, p(16, 3)).unwrap();
        let err = check_experts_on_gpu(before, p(16, 4)).unwrap_err();
        assert!(matches!(err, EngineError::BackendUnavailable(_)), "{err}");
        let err = check_experts_on_gpu(before, before).unwrap_err();
        assert!(matches!(err, EngineError::BackendUnavailable(_)), "{err}");
    }

    #[test]
    fn pool_engaged_check_counts_only_traffic_since_the_snapshot() {
        let stats = P3Stats::new();
        let err = check_pool_engaged(0, pool_traffic(&stats)).unwrap_err();
        assert!(matches!(err, EngineError::PoolInit(_)), "{err}");
        stats.fast_hits.store(1, Ordering::Relaxed);
        check_pool_engaged(0, pool_traffic(&stats)).unwrap();
        let before = pool_traffic(&stats);
        let err = check_pool_engaged(before, pool_traffic(&stats)).unwrap_err();
        assert!(matches!(err, EngineError::PoolInit(_)), "{err}");
        stats.demand_slow_pageins.store(1, Ordering::Relaxed);
        check_pool_engaged(before, pool_traffic(&stats)).unwrap();
    }

    #[test]
    fn expert_offload_is_forced_only_for_pool_or_cpu_moe_runs() {
        let flagless = EngineConfig::default();
        assert!(!forces_expert_offload(&flagless));
        let cpu_moe_zero = EngineConfig {
            cpu_moe_layers: Some(0),
            ..EngineConfig::default()
        };
        assert!(!forces_expert_offload(&cpu_moe_zero));
        let cpu_moe = EngineConfig {
            cpu_moe_layers: Some(2),
            ..EngineConfig::default()
        };
        assert!(forces_expert_offload(&cpu_moe));
        let pooled = EngineConfig {
            ram_experts_gib: 1,
            ..EngineConfig::default()
        };
        assert!(forces_expert_offload(&pooled));
    }

    /// One rendering path (§8.1): the shim's no-tools render of the tiny-moe
    /// fallback (chatml) equals the old `llama_chat_apply_template` render.
    #[test]
    fn shim_no_tools_render_matches_legacy_template_apply() {
        let cfg = EngineConfig {
            model_path: xtask::tiny_moe::cached_fixture(),
            ..EngineConfig::default()
        };
        let s = Session::new(&cfg).expect("tiny-moe loads");
        let turns = [
            ChatTurn {
                role: "system",
                content: "be brief",
            },
            ChatTurn {
                role: "user",
                content: "hello",
            },
            ChatTurn {
                role: "assistant",
                content: "hi there",
            },
            ChatTurn {
                role: "user",
                content: "again",
            },
        ];
        let shim = s.render(&TurnRequest::new(&turns)).unwrap().prompt;
        let legacy = legacy_apply_template(c"chatml", &turns, true);
        assert_eq!(shim, legacy);
    }

    /// The pre-buffer two-pass detokenize (sizing call, exact alloc).
    fn piece_two_pass(vocab: *const raw::llama_vocab, tok: raw::llama_token) -> Vec<u8> {
        // SAFETY: null/0 only sizes; vocab is live.
        let n = unsafe { raw::llama_token_to_piece(vocab, tok, std::ptr::null_mut(), 0, 0, false) }
            .unsigned_abs() as usize;
        let mut buf = vec![0u8; n];
        // SAFETY: buf holds the n bytes the sizing call demanded.
        let w = unsafe {
            raw::llama_token_to_piece(vocab, tok, buf.as_mut_ptr().cast(), n as i32, 0, false)
        };
        buf.truncate(w.max(0) as usize);
        buf
    }

    #[test]
    fn piece_into_matches_two_pass_over_vocab() {
        let cfg = EngineConfig {
            model_path: xtask::tiny_moe::cached_fixture(),
            ..EngineConfig::default()
        };
        let s = Session::new(&cfg).expect("tiny-moe loads");
        let vocab = s.inner.loaded.model().vocab();
        // SAFETY: vocab is live.
        let n_vocab = unsafe { raw::llama_vocab_n_tokens(vocab) };
        assert!(n_vocab > 0);
        let mut reserved = Vec::with_capacity(crate::slot::PIECE_RESERVE);
        for tok in 0..n_vocab {
            let want = piece_two_pass(vocab, tok);
            s.inner.piece_into(tok, &mut reserved).unwrap();
            assert_eq!(reserved, want, "token {tok}");
            let mut empty = Vec::new();
            s.inner.piece_into(tok, &mut empty).unwrap();
            assert_eq!(empty, want, "token {tok} via the regrow path");
        }
    }

    #[test]
    fn slot_piece_buffer_capacity_stable_across_decode() {
        let cfg = EngineConfig {
            model_path: xtask::tiny_moe::cached_fixture(),
            ..EngineConfig::default()
        };
        let mut s = Session::new(&cfg).expect("tiny-moe loads");
        let msgs = [ChatTurn {
            role: "user",
            content: "count to ten",
        }];
        let turn = |s: &mut Session, n| {
            let out = s
                .turn(&TurnRequest::new(&msgs).max_tokens(Some(n)), &mut |_| {})
                .expect("turn");
            assert!(out.completion_tokens > 1, "precondition: decode steps ran");
        };
        turn(&mut s, 4);
        let warm = s.slots[0].piece_capacity();
        turn(&mut s, 32);
        assert_eq!(s.slots[0].piece_capacity(), warm);
    }

    #[test]
    fn moe_buft_overrides_build_n_entries_plus_terminator() {
        let (patterns, overrides) = moe_buft_overrides(Some(3), std::ptr::null_mut());
        assert_eq!(patterns.len(), 3);
        assert_eq!(overrides.len(), 4, "3 entries + NULL terminator");
        for (i, (pat, entry)) in patterns.iter().zip(overrides.iter()).enumerate() {
            assert_eq!(
                pat.to_str().unwrap(),
                format!(r"blk\.{i}\.ffn_(up|down|gate|gate_up)_(ch|)exps"),
                "pattern must match the vendor's llm_ffn_block_regex(i, LLM_FFN_EXPS_REGEX)"
            );
            assert_eq!(
                entry.pattern,
                pat.as_ptr(),
                "entry points at its own CString"
            );
            assert!(entry.buft.is_null(), "the test passes a null buffer type");
        }
        assert!(overrides[3].pattern.is_null(), "terminator pattern");
        assert!(overrides[3].buft.is_null(), "terminator buft");
    }

    #[test]
    fn moe_buft_overrides_default_is_empty() {
        for setting in [None, Some(0)] {
            let (patterns, overrides) = moe_buft_overrides(setting, std::ptr::null_mut());
            assert!(patterns.is_empty());
            assert!(
                overrides.is_empty(),
                "{setting:?}: vendor maps an empty overrides vector to NULL"
            );
        }
    }

    #[test]
    fn utf8_emitter_splits_only_on_char_boundaries() {
        let mut pending = Vec::new();
        let mut out = String::new();
        emit_valid_prefix(&mut pending, &mut out, &[0xE4, 0xBD]); // incomplete 3-byte char
        assert!(out.is_empty());
        emit_valid_prefix(&mut pending, &mut out, &[0xA0]);
        assert_eq!(out, "\u{4f60}"); // 你 — assembled across two pieces
    }

    #[test]
    fn prefill_chunk_cap_is_max_one_k_over_top_k_when_pooled() {
        // The cap is the §2.4 routed-set bound: one chunk routes at most
        // cap × top_k distinct experts per layer against K slab slots.
        // Target shape: K = 10, top-k = 8 → 1 token per chunk (the real-model
        // repro); K = 64, top-k = 8 → 8.
        assert_eq!(prefill_chunk_cap(512, 4096, Some((10, 8))), 1);
        assert_eq!(prefill_chunk_cap(512, 4096, Some((64, 8))), 8);
        // Base cap (pool off): min(batch_tokens, n_ctx - 1).
        assert_eq!(prefill_chunk_cap(512, 4096, None), 512);
        assert_eq!(prefill_chunk_cap(512, 64, None), 63);
        // Pool cap never raises the base cap, and never yields 0.
        assert_eq!(prefill_chunk_cap(4, 4096, Some((64, 8))), 4);
        assert_eq!(prefill_chunk_cap(512, 4096, Some((0, 8))), 1);
        // tiny-moe cannot trigger the slab abort this cap prevents: it has
        // n_expert 8 and its K clamps to 8, so even a full-batch chunk of
        // 8 tokens at top-k ≤ 8 routes at most 8 ≤ K experts per layer.
        assert_eq!(prefill_chunk_cap(8, 4096, Some((8, 8))), 1);
    }

    /// Concurrent-only guard: a pool-off session no longer blocks a pooled
    /// one (per-model ctx, ABI v5 — hooks only route THEIR model's decodes),
    /// but a second CONCURRENT pool-enabled session is refused with the slab
    /// VRAM arithmetic, and enable succeeds again once it drops.
    /// CPU-testable: a pool-off tiny-moe session loads without hooks, and a
    /// pooled enable with an explicit K works on CPU (see
    /// `multi_chunk_prefill_advances_the_policy_step_zero_times`). Success
    /// after drop leaves no live hooks (nextest runs each test in its own
    /// process, so no other test observes residual state).
    #[test]
    fn pooled_enable_concurrent_only_refusal_with_vram_arithmetic() {
        let model = xtask::tiny_moe::cached_fixture();
        let pooled = EngineConfig {
            model_path: model.clone(),
            ram_experts_gib: 1,
            device_cache_experts: 8, // tiny-moe: K=8 ≥ top_k=2
            ..EngineConfig::default()
        };
        let plain = EngineConfig {
            model_path: model,
            ..EngineConfig::default()
        };
        // A live POOL-OFF session no longer refuses a pooled enable: the
        // hooks attach to the pooled model only.
        let plain_session = Session::new(&plain).expect("pool-off session loads");
        let hooks = p3_glue::enable(&pooled).expect("pooled enable alongside a pool-off session");
        drop(plain_session);

        // Concurrent second POOL-ON enable: refused, naming the slab VRAM.
        let err = match p3_glue::enable(&pooled) {
            Err(e) => e,
            Ok(_) => panic!("concurrent second pool enable must be refused"),
        };
        assert!(
            err.to_string().contains("7.3 GB") && err.to_string().contains("24 GiB"),
            "refusal must carry the slab VRAM arithmetic: {err}"
        );
        drop(hooks);
        let _again = p3_glue::enable(&pooled).expect("enable succeeds once the pool dropped");
    }

    /// Pool-on construction must be refused while another POOL-ON session is
    /// live BEFORE any model load: the model path here does not exist, so a
    /// load attempt would surface a GGUF error instead of the concurrent-pool
    /// refusal. A pool-off live session no longer refuses (per-model hooks).
    #[test]
    fn pooled_new_is_refused_before_load_while_a_pooled_session_is_live() {
        let pooled = EngineConfig {
            model_path: xtask::tiny_moe::cached_fixture(),
            ram_experts_gib: 1,
            device_cache_experts: 8,
            ..EngineConfig::default()
        };
        let _live = Session::new(&pooled).expect("pooled session loads");
        let pooled_missing_model = EngineConfig {
            model_path: "/nonexistent/never-loaded.gguf".into(),
            ram_experts_gib: 1,
            device_cache_experts: 8,
            ..EngineConfig::default()
        };
        let err = match Session::new(&pooled_missing_model) {
            Err(e) => e,
            Ok(_) => panic!("concurrent pool-on session must be refused"),
        };
        assert!(
            matches!(&err, EngineError::PoolInit(m) if m.contains("two device slab sets")),
            "must be the concurrent-pool refusal, not a load error: {err}"
        );
    }

    /// §6.2: prefill chunks are excluded from policy-step scoring — a prefill
    /// ubatch routes nearly every expert and would flatten the scores. A
    /// multi-chunk prefill must leave the step counter untouched. `n_predict = 0` makes the turn
    /// prefill-only, isolating the prefill path from generation decodes.
    /// Pool-enabled, so the chunk cap comes from the real pooled geometry
    /// (CPU needs the explicit `device_cache_experts`; no VRAM query to
    /// derive it). Preconditions fail loudly — no skip (evidence rule).
    #[test]
    fn multi_chunk_prefill_advances_the_policy_step_zero_times() {
        let cfg = EngineConfig {
            model_path: xtask::tiny_moe::cached_fixture(),
            ram_experts_gib: 1,
            device_cache_experts: 8, // tiny-moe: K=8, top_k=2 → cap 4.
            n_predict: 0,
            ..EngineConfig::default()
        };
        let mut s = Session::new(&cfg).expect("pooled CPU session loads tiny-moe");
        let (k, top_k) = s
            .inner
            .p3
            .as_ref()
            .map(|p| (p.device_cache_experts, p.top_k))
            .expect("pool enabled for this session");
        let cap = prefill_chunk_cap(
            cfg.batch_tokens as usize,
            cfg.n_ctx as usize,
            Some((k, top_k)),
        );
        let turns = [ChatTurn {
            role: "user",
            content: "one two three four five six seven eight nine ten eleven twelve",
        }];
        let mut out = String::new();
        s.chat_turn(&turns, &mut |piece| out.push_str(piece))
            .expect("prefill-only turn succeeds");
        let tokens = s.last_prompt_tokens().len();
        assert!(
            tokens >= 2 * cap,
            "precondition: prompt must span ≥2 prefill chunks (cap {cap}, tokens {tokens})"
        );
        assert!(out.is_empty(), "n_predict = 0 generates nothing");
        assert_eq!(
            p3_glue::decode_step_for_test(),
            0,
            "multi-chunk prefill must not advance the policy step (§6.2)"
        );
    }

    /// Applies the chat template: sizing call with a null buffer (documented pattern,
    /// returns the total byte count), then the real call into a zeroed `count + 1`
    /// buffer (+1 because `strncpy` does not NUL-terminate an exact-size render).
    ///
    /// The pre-shim render path, kept as the no-tools parity oracle.
    fn legacy_apply_template(tmpl: &CStr, turns: &[ChatTurn<'_>], add_ass: bool) -> String {
        let owned: Vec<(CString, CString)> = turns
            .iter()
            .map(|t| {
                (
                    CString::new(t.role).unwrap(),
                    CString::new(t.content).unwrap(),
                )
            })
            .collect();
        // Pointers into `owned`; they outlive both FFI calls below.
        let chat: Vec<raw::llama_chat_message> = owned
            .iter()
            .map(|(r, c)| raw::llama_chat_message {
                role: r.as_ptr(),
                content: c.as_ptr(),
            })
            .collect();
        let n_msg = chat.len();
        // SAFETY: tmpl and chat point to live NUL-terminated data; null/0 buffer is
        // the documented sizing call (returns the total number of bytes needed,
        // negative on detection/apply failure).
        let n = unsafe {
            raw::llama_chat_apply_template(
                tmpl.as_ptr(),
                chat.as_ptr(),
                n_msg,
                add_ass,
                std::ptr::null_mut(),
                0,
            )
        };
        assert!(n >= 0, "llama_chat_apply_template sizing failed: {n}");
        let mut buf = vec![0u8; usize::try_from(n).unwrap_or(0) + 1];
        // SAFETY: buf is n+1 zeroed bytes and length n matches the sizing result.
        let written = unsafe {
            raw::llama_chat_apply_template(
                tmpl.as_ptr(),
                chat.as_ptr(),
                n_msg,
                add_ass,
                buf.as_mut_ptr().cast(),
                n,
            )
        };
        assert!(written >= 0, "llama_chat_apply_template failed: {written}");
        // SAFETY: buf is NUL-terminated (zeroed +1 byte; at most `written` render bytes).
        unsafe { CStr::from_ptr(buf.as_ptr().cast()) }
            .to_string_lossy()
            .into_owned()
    }
}
