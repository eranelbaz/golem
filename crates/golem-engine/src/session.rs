//! Model + context lifecycle (tech design §7): backend init, load with
//! fail-fast sanity checks, RAII teardown, the §4.3 decode self-check probe,
//! chat-template resolution (model metadata, else the chatml fallback), and
//! the single-slot greedy decode loop (`Session::chat_turn`).
#![allow(unsafe_code)]

use std::ffi::{CStr, CString};
use std::sync::Once;

use golem_llama_sys::raw;
use golem_llama_sys::{Backend, substrate_backend};
use golem_placement::pool::P3Stats;

use crate::config::EngineConfig;
use crate::error::EngineError;
use crate::gguf;
use crate::p3_glue;
use crate::parity::{self, ParityStep};
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
    pub(crate) fn chat_template(&self) -> Option<String> {
        // SAFETY: self.ptr is a live llama_model from llama_model_load_from_file;
        // null name selects the default template (vendor/llama.cpp/include/llama.h).
        let p = unsafe { raw::llama_model_chat_template(self.ptr, std::ptr::null()) };
        if p.is_null() {
            None
        } else {
            // SAFETY: llama.cpp returns a pointer to a NUL-terminated buffer
            // owned by the model, valid for the model's lifetime.
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
    params.n_ctx = config.n_ctx;
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
    decode_self_check(loaded.ctx(), loaded.model().vocab())?;
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

/// Chatml template copied from the `llm_chat_apply_template` CHATML branch of
/// `vendor/llama.cpp/src/llama-chat.cpp` (the `ss <<` sequence rendered as the
/// canonical jinja-style string; same literal as the vendor's own
/// test-chat-template.cpp chatml case). llama.cpp detects chatml by the
/// `<|im_start|>` substring and renders from its hardcoded branch, so this
/// string is a detection key, never a parsed template.
pub(crate) const FALLBACK_CHAT_TEMPLATE: &str = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}";

/// One message of a chat conversation.
pub struct ChatTurn<'a> {
    /// Speaker role the chat template understands (`"system"`, `"user"`,
    /// `"assistant"`, ...).
    pub role: &'a str,
    /// Message body text.
    pub content: &'a str,
}

/// CString pair backing one `raw::llama_chat_message` for the duration of a
/// template-apply call.
struct ChatMessageFFI {
    role: CString,
    content: CString,
}

fn ffi_messages(messages: &[ChatTurn<'_>]) -> Result<Vec<ChatMessageFFI>, EngineError> {
    messages
        .iter()
        .map(|m| {
            Ok(ChatMessageFFI {
                role: CString::new(m.role).map_err(|_| {
                    EngineError::InvalidModel("chat message role contains NUL".into())
                })?,
                content: CString::new(m.content).map_err(|_| {
                    EngineError::InvalidModel("chat message content contains NUL".into())
                })?,
            })
        })
        .collect()
}

/// A loaded model bound to its resolved chat template, driving one decode slot
/// (stateless per turn — every `chat_turn` re-prefills the full transcript).
/// Field order is load-bearing: `loaded` (llama_free) drops BEFORE `p3`
/// (unregister already done in `Drop`; ctx free + pool drop here) — the
/// §3.4 teardown order.
pub struct Session {
    loaded: LoadedModel,
    config: EngineConfig,
    template: CString,
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
}

impl Drop for Session {
    fn drop(&mut self) {
        // §3.4 teardown order: detach the model's ctx → unregister-join →
        // `llama_free` (field drop below: the sched destructor decrements the
        // ctx attach refcount) → `golem_ctx_free` (inside SessionHooks::drop,
        // aborts while the refcount ≠ 0) → pool/prefetch drop. `p3` is
        // declared after `loaded`, so its drop runs after `llama_free`.
        if let Some(_p3) = self.p3.as_ref() {
            // SAFETY: the model is live until `loaded` drops below; NULL
            // detaches so no later context can reach the freed ctx.
            unsafe {
                raw::llama_golem_model_attach(self.loaded.model().as_ptr(), std::ptr::null_mut())
            };
        }
        if let Some(p3) = self.p3.as_mut() {
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
        if config.prefill_stream && config.ram_experts_gib == 0 {
            return Err(EngineError::InvalidConfig(
                "prefill_stream requires the RAM expert pool (ram_experts_gib > 0)".into(),
            ));
        }
        if config.ram_experts_gib > 0 {
            p3_glue::refuse_if_pool_session_live()?;
        }
        // P3 (§3.4): build the RAM expert pool + create the per-model ctx
        // BEFORE the model loads — the ctx must attach to the llama_model
        // before `llama_init_from_model` so its first `sched_reserve` sees
        // the hooks (cpy shrink). If the load fails, the guards drop cleanly.
        let mut p3 = if config.ram_experts_gib > 0 {
            Some(p3_glue::enable(config)?)
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
        // (ruling R3). On error the slabs are already freed by the fit.
        if let Some(p) = p3.as_mut() {
            p.fit_device_cache()?;
        }
        let template_src = loaded
            .model()
            .chat_template()
            .unwrap_or_else(|| FALLBACK_CHAT_TEMPLATE.to_owned());
        // Interior NUL would truncate the template at the FFI boundary: unusable.
        let template = CString::new(template_src).map_err(|_| EngineError::TemplateUnavailable)?;

        let probe = ffi_messages(&[ChatTurn {
            role: "user",
            content: "ping",
        }])?;
        // SAFETY: template is a live CString; probe CStrings outlive the call.
        if unsafe { apply_template(&template, &probe, true) }?.is_empty() {
            return Err(EngineError::TemplateUnavailable);
        }

        // P3 guards came from `enable` above; nothing to build here.

        Ok({
            p3_glue::session_opened();
            Session {
                loaded,
                config: config.clone(),
                template,
                last_prompt_tokens: Vec::new(),
                p3,
                timing: config
                    .timing_csv
                    .as_ref()
                    .map(|_| crate::timing::TimingRecorder::new(config.n_predict as usize + 1)),
                pool_verified: false,
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
        let template_src = loaded
            .model()
            .chat_template()
            .unwrap_or_else(|| FALLBACK_CHAT_TEMPLATE.to_owned());
        let template = CString::new(template_src).map_err(|_| EngineError::TemplateUnavailable)?;
        let probe = ffi_messages(&[ChatTurn {
            role: "user",
            content: "ping",
        }])?;
        // SAFETY: template is a live CString; probe CStrings outlive the call.
        if unsafe { apply_template(&template, &probe, true) }?.is_empty() {
            return Err(EngineError::TemplateUnavailable);
        }
        Ok({
            p3_glue::session_opened();
            Session {
                loaded,
                config: config.clone(),
                template,
                last_prompt_tokens: Vec::new(),
                p3: None,
                timing: config
                    .timing_csv
                    .as_ref()
                    .map(|_| crate::timing::TimingRecorder::new(config.n_predict as usize + 1)),
                pool_verified: false,
            }
        })
    }

    /// The P3 slow-path counters when the RAM expert pool is enabled for this
    /// session; `None` when the pool is off (the default). Demand and prefetch
    /// page-ins are counted separately (design §6.6).
    pub fn p3_stats(&self) -> Option<&P3Stats> {
        self.p3.as_ref().map(|p| p.stats.as_ref())
    }

    /// The session's hook guards (test/observability seam: stream counters,
    /// trampoline call counts, prefill-stream arming). `None` when the pool
    /// is off.
    pub fn p3_hooks(&self) -> Option<&crate::p3_glue::SessionHooks> {
        self.p3.as_ref()
    }

    /// The session's per-model hook context pointer (ABI v5), for test seams
    /// that address per-ctx substrate state (the P4 budget setter/reset).
    /// `None` when the pool is off.
    #[cfg_attr(not(feature = "hip-tests"), allow(dead_code))]
    pub fn p3_ctx_ptr(&self) -> Option<raw::golem_ctx_t> {
        self.p3.as_ref().map(|p| p.ctx_ptr())
    }

    /// The tokenized prompt of the most recent `chat_turn` (post-render,
    /// post-special-tokens).
    pub fn last_prompt_tokens(&self) -> &[i32] {
        &self.last_prompt_tokens
    }

    /// True when the session's config requests GPU offload (`n_gpu_layers > 0`):
    /// the dump-header hook parity legs record to prove a "hip"-labeled run is
    /// really attempting device execution (mis-labeled-leg guard). This is the
    /// config's intent, not llama.cpp's runtime placement — a CPU-only
    /// substrate reports `true` with zero devices to offload onto.
    pub fn offloaded(&self) -> bool {
        self.config.n_gpu_layers > 0
    }

    /// §6.5 prefetch/prediction counters; `None` without the RAM expert pool.
    pub fn prefetch_stats(&self) -> Option<crate::p3_glue::PrefetchStats> {
        self.p3.as_ref().map(|p| p.prefetch_stats())
    }

    /// Resolved P2 slot count per layer (explicit or derived from free VRAM);
    /// `None` without the RAM expert pool.
    pub fn device_cache_experts(&self) -> Option<u32> {
        self.p3.as_ref().map(|p| p.device_cache_experts)
    }

    /// P4 device expert-copy cache counters (§3.4): bytes copied into / skipped
    /// by the persistent device cache, evictions, and the derived skip ratio.
    /// `None` unless this session drives the pooled GPU split — the only P2
    /// call-site (hooks registered + offload requested).
    pub fn device_cache_stats(&self) -> Option<DeviceCacheStats> {
        if !self.offloaded() || self.p3.is_none() {
            return None;
        }
        // SAFETY: FFI getter with this session's live ctx; the returned
        // pointer is ctx-owned (alive while the hooks are registered).
        let block = unsafe {
            raw::golem_get_device_cache_stats(self.p3.as_ref().expect("p3 present").ctx_ptr())
        };
        assert!(
            !block.is_null(),
            "golem_get_device_cache_stats returned null"
        );
        // SAFETY: layout contract from the C++ definition (bindgen renders
        // the struct opaque): 3 contiguous lock-free u64 counters in member
        // order bytes_copied, bytes_skipped, evictions.
        let counters = unsafe { &*block.cast::<[std::sync::atomic::AtomicU64; 3]>() };
        let load = |i: usize| counters[i].load(std::sync::atomic::Ordering::Relaxed);
        Some(DeviceCacheStats::new(load(0), load(1), load(2)))
    }

    /// Flushes the §11 per-step rows into `w` without destroying the
    /// recorder — every turn records (a `take`-style drain would leave turns
    /// 2+ empty). No-op when the timing CSV is disabled.
    pub fn write_timing(&mut self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        match self.timing.as_mut() {
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
        match self.p3.as_ref() {
            Some(p3) => p3.write_routing_trace(w),
            None => Ok(()),
        }
    }

    /// Runs one chat turn: renders `messages` with the session template,
    /// clears the KV cache, re-prefills the full transcript in chunks, then
    /// greedily generates until EOG, `n_ctx - 1` positions, or `n_predict`
    /// tokens (`n_predict == 0` samples and generates nothing: the turn is
    /// prefill-only and returns 0). Piece bytes stream through `emit` as they
    /// complete UTF-8 characters (verbatim — no space stripping); buffered
    /// residue is flushed through `emit` even when the turn fails. Returns
    /// tokens generated.
    pub fn chat_turn(
        &mut self,
        messages: &[ChatTurn<'_>],
        emit: &mut dyn FnMut(&str),
    ) -> Result<u32, EngineError> {
        let mut pending = Vec::new();
        match self.chat_turn_inner(&mut pending, messages, emit, None) {
            Ok(n) => Ok(n),
            Err(e) => {
                let mut tail = String::new();
                flush_utf8_tail(&mut pending, &mut tail);
                if !tail.is_empty() {
                    emit(&tail);
                }
                Err(e)
            }
        }
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
        let mut pending = Vec::new();
        let mut sink = |_: &str| {};
        self.chat_turn_inner(&mut pending, messages, &mut sink, Some(steps))
    }

    fn chat_turn_inner(
        &mut self,
        pending: &mut Vec<u8>,
        messages: &[ChatTurn<'_>],
        emit: &mut dyn FnMut(&str),
        mut steps: Option<&mut Vec<ParityStep>>,
    ) -> Result<u32, EngineError> {
        let msgs = ffi_messages(messages)?;
        // SAFETY: template and msgs CStrings are alive for the duration of the call.
        let rendered = unsafe { apply_template(&self.template, &msgs, true)? };

        // Stateless per turn: full-transcript re-prefill.
        let ctx = self.loaded.ctx();
        // SAFETY: llama_get_memory returns the context's live memory handle; the
        // context is idle between turns, the documented llama_memory_clear use.
        unsafe { raw::llama_memory_clear(raw::llama_get_memory(ctx.as_ptr()), true) };

        let vocab = self.loaded.model().vocab();
        // add_special = true: the fallback chatml renders no BOS text, so this
        // yields exactly one BOS.
        // SAFETY: vocab is live; rendered is valid for its byte length.
        let prompt = unsafe { tokenize_prompt(vocab, &rendered, true, true)? };
        if prompt.is_empty() {
            return Err(EngineError::Tokenize(0));
        }
        self.last_prompt_tokens = prompt.clone();
        let prompt_len = prompt.len();

        let n_ctx = self.config.n_ctx as usize;
        let chunk_cap = prefill_chunk_cap(
            self.config.batch_tokens as usize,
            n_ctx,
            self.p3.as_ref().map(|p| (p.device_cache_experts, p.top_k)),
        );
        if chunk_cap == 0 {
            // n_ctx <= 1 cannot fit even one prompt token — llama_decode would
            // return 1; fail with that same code instead of chunking by zero.
            return Err(EngineError::Decode(1));
        }
        // One llama_decode must carry exactly one ubatch (the C lease-release
        // contract, §2.4): n_ubatch is a load-time constant (§4.2), fetched once.
        // SAFETY: ctx is a live context.
        let n_ubatch = unsafe { raw::llama_n_ubatch(ctx.as_ptr()) } as usize;
        // Prefill streaming (§7.6): whole-plane ubatches above the
        // --prefill-stream-min tail threshold; the per-token cap below it.
        let streaming = self.p3.as_ref().is_some_and(|p| p.stream_enabled());
        let batch_cap = chunk_cap.max(if streaming { n_ubatch } else { 0 });

        let mut batch = Batch::init(batch_cap as i32, 0, 1);
        // §6.2: prefill routes nearly every expert; keep it out of recency.
        // I2 runtime proof window: this turn's prefill and first batch-1 decode
        // (the op-offload threshold is per ubatch, so both shapes are checked).
        let verify = !self.pool_verified
            && self.p3.is_some()
            && self.offloaded()
            && substrate_backend() == Backend::Hip;
        let placement_before = expert_placement();
        // Streamed planes are expert copies served on the GPU split too
        // (stream_source fires only from that copy loop, §7.6).
        let traffic_of = |p: &p3_glue::SessionHooks| {
            pool_traffic(&p.stats) + p.stream_stats().map_or(0, |s| s.planes)
        };
        let traffic_before = self.p3.as_ref().map_or(0, traffic_of);
        if let Some(p3) = self.p3.as_ref() {
            // Stale-record guard: a failed prior turn's routing records are
            // dropped (the header survives for the real file flush) before
            // this turn's prefill records (phase `pf`) start.
            p3.drain_routing_trace();
        }
        let prefill = p3_glue::PrefillScope::enter();
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
        let _stream_guard = self.p3.as_ref().map(StreamGuard);
        let stream_min = self.config.prefill_stream_min as usize;
        let mut n_past: usize = 0;
        let mut last_chunk_len = 0;
        while n_past < prompt_len {
            let remaining = prompt_len - n_past;
            let stream_chunk = streaming && remaining >= stream_min;
            let len = if stream_chunk {
                remaining.min(n_ubatch)
            } else {
                remaining.min(chunk_cap)
            };
            if let Some(p3) = &self.p3 {
                p3.set_prefill_stream(stream_chunk);
            }
            let is_final_chunk = n_past + len == prompt_len;
            {
                // SAFETY: slots 0..len are within the chunk_cap-sized allocation;
                // every used slot is fully written before decode (llama_batch_init
                // mallocs and does not zero).
                let b = &mut batch.inner;
                for (i, &t) in prompt[n_past..n_past + len].iter().enumerate() {
                    // Logits only on the final token of the final chunk: that
                    // sample bootstraps generation; intermediate tokens never need it.
                    // SAFETY: i < len <= batch_cap slots allocated by Batch::init.
                    unsafe {
                        fill_slot(
                            b,
                            i,
                            t,
                            (n_past + i) as raw::llama_pos,
                            is_final_chunk && i + 1 == len,
                        );
                    }
                }
                b.n_tokens = len as i32;
            }
            debug_assert!(
                len <= n_ubatch,
                "prefill chunk must be exactly one ubatch (lease-release contract, §2.4)"
            );
            // SAFETY: ctx is a live context; batch slots 0..n_tokens fully written.
            let code = unsafe { raw::llama_decode(ctx.as_ptr(), batch.inner) };
            if code != 0 {
                // 1 = no KV slot left (ctx too small for the prompt); the rest fatal alike.
                // Synchronize first: the failed decode may have launched a compute
                // whose expert-copy leases release only at the next compute.
                // SAFETY: ctx is live.
                unsafe { raw::llama_synchronize(ctx.as_ptr()) };
                return Err(EngineError::Decode(code));
            }
            // §6.2: prefill chunks are NOT policy steps — a prefill ubatch
            // routes nearly every expert and would flatten the scores, so
            // only the generation decodes below advance the step.
            // synchronize (see decode helper above).
            // SAFETY: ctx is live.
            unsafe { raw::llama_synchronize(ctx.as_ptr()) };
            n_past += len;
            last_chunk_len = len;
        }
        drop(prefill);
        // Disarm before the generation decodes (the guard is the backstop).
        if let Some(p3) = &self.p3 {
            p3.set_prefill_stream(false);
        }

        if verify {
            check_experts_on_gpu(placement_before, expert_placement())?;
            if let Some(p3) = &self.p3 {
                check_pool_engaged(traffic_before, traffic_of(p3))?;
            }
        }

        if self.config.n_predict == 0 {
            return Ok(0);
        }

        // Parity recording needs the logits row length, which is the vocab size
        // (validated non-negative by the load-time decode self-check probe).
        // SAFETY: vocab is live.
        let n_vocab = usize::try_from(unsafe { raw::llama_vocab_n_tokens(vocab) })
            .map_err(|_| EngineError::InvalidModel("vocab size is negative".to_owned()))?;

        // Greedy sampler chain; the chain owns the greedy sampler and frees it.
        let sampler = Sampler::greedy();

        // SAFETY: sampler chain and ctx are live; the final decode emitted logits
        // for slot last_chunk_len - 1.
        let mut tok = unsafe {
            raw::llama_sampler_sample(sampler.as_ptr(), ctx.as_ptr(), (last_chunk_len - 1) as i32)
        };
        if let Some(steps) = steps.as_mut() {
            // Generation step 0: the bootstrap sample off the final prompt-token
            // logits. The token is the sampler's output; fetch the same row's
            // top-2 alongside (pure read — no decode behavior change).
            // SAFETY: the successful final-chunk decode emitted logits for slot
            // last_chunk_len - 1; n_vocab is that row's length.
            let (top1, top2) = unsafe {
                parity::top2_logits_ith(ctx.as_ptr(), (last_chunk_len - 1) as i32, n_vocab)
            };
            steps.push(ParityStep {
                position: 0,
                token: tok,
                top1,
                top2,
            });
        }

        let mut generated: u32 = 0;
        let mut scratch = String::new();
        if let Some(t) = self.timing.as_mut() {
            t.flush(&mut std::io::sink())?; // stale rows from a failed prior turn never leak in
        }
        // Prefill activity must not bleed into step 0's row: take-and-discard
        // the P3/store accumulators right before the decode loop (also
        // prevents an EOG-exited turn from bleeding into the next one).
        let _ = p3_glue::take_step_io_stats();
        loop {
            // §11: step wall time starts at the loop head (emit + decode).
            let step_t0 = std::time::Instant::now();
            // SAFETY: vocab is live; tok is a valid token id from the sampler.
            if unsafe { raw::llama_vocab_is_eog(vocab, tok) } {
                // EOG is never decoded, so it has no row: every decode step
                // (0-based, numbered below) already wrote its row before the
                // follow-up sample. The pre-loop accumulator reset above
                // keeps this exit from bleeding into the next turn.
                break;
            }
            // SAFETY: vocab is live; tok is a valid token id from the sampler.
            let piece = unsafe { token_piece(vocab, tok) };
            emit_valid_prefix(pending, &mut scratch, &piece);
            if !scratch.is_empty() {
                emit(&scratch);
                scratch.clear();
            }
            generated += 1;
            // One KV slot stays free (chunk_cap ≤ min(batch_tokens, n_ctx - 1));
            // never decode at pos >= n_ctx - 1.
            if n_past >= n_ctx - 1 || generated >= self.config.n_predict {
                break;
            }
            {
                let b = &mut batch.inner;
                // SAFETY: slot 0 exists (chunk_cap >= 1); fully written before decode.
                unsafe { fill_slot(b, 0, tok, n_past as raw::llama_pos, true) };
                b.n_tokens = 1;
            }
            debug_assert!(
                batch.inner.n_tokens as usize <= n_ubatch,
                "one decode step = one ubatch (lease-release contract, §2.4)"
            );
            // SAFETY: same invariants as the prefill decode.
            let code = unsafe { raw::llama_decode(ctx.as_ptr(), batch.inner) };
            if code != 0 {
                // Synchronize first: see the prefill decode's error path.
                // SAFETY: ctx is live.
                unsafe { raw::llama_synchronize(ctx.as_ptr()) };
                return Err(EngineError::Decode(code));
            }
            p3_glue::advance_decode_step();
            // synchronize (see decode helper above).
            // SAFETY: ctx is live.
            unsafe { raw::llama_synchronize(ctx.as_ptr()) };
            n_past += 1;
            if let Some(t) = self.timing.as_mut() {
                // §11 step boundary: snapshot (and reset) the P3/store
                // accumulators into this step's row. H2D and policy stay
                // unmeasured (no fork event pair / no P2 instrumentation).
                let io = p3_glue::take_step_io_stats();
                let ns2ms = |ns: u64| ns as f64 / 1_000_000.0;
                t.row(crate::timing::StepTiming {
                    // 0-based: `generated` was incremented for this step's
                    // emit above, matching the StepTiming::step doc.
                    step: generated as u64 - 1,
                    tokens: 1,
                    wall_ms: step_t0.elapsed().as_secs_f64() * 1_000.0,
                    p3_wait_ms: ns2ms(io.p3_wait_ns),
                    disk_worker_ms: io.disk_ns.map_or(f64::NAN, ns2ms),
                    disk_bytes: io.disk_bytes.unwrap_or(0),
                    h2d_ms: f64::NAN,
                    h2d_bytes: 0,
                    policy_ms: f64::NAN,
                });
            }
            if verify && !self.pool_verified {
                check_experts_on_gpu(placement_before, expert_placement())?;
                self.pool_verified = true;
            }
            // SAFETY: sampler chain and ctx are live; the decode emitted logits for slot 0.
            tok = unsafe { raw::llama_sampler_sample(sampler.as_ptr(), ctx.as_ptr(), 0) };
            if let Some(steps) = steps.as_mut() {
                // `generated` counts prior non-EOG samples, which is exactly the
                // 0-based index of the sample just taken.
                // SAFETY: the successful single-token decode emitted logits for slot 0.
                let (top1, top2) = unsafe { parity::top2_logits_ith(ctx.as_ptr(), 0, n_vocab) };
                steps.push(ParityStep {
                    position: generated,
                    token: tok,
                    top1,
                    top2,
                });
            }
        }
        // Turn end: flush whatever bytes remain buffered (n_predict can cut a
        // multi-byte character in half — lossy-drain that residue).
        flush_utf8_tail(pending, &mut scratch);
        if !scratch.is_empty() {
            emit(&scratch);
        }
        Ok(generated)
    }
}

/// Owned greedy sampler chain; `Drop` calls `llama_sampler_free`, which frees
/// the samplers added to the chain.
struct Sampler {
    ptr: *mut raw::llama_sampler,
}

impl Sampler {
    /// `llama_sampler_chain_init(llama_sampler_chain_default_params())` + greedy.
    fn greedy() -> Self {
        // SAFETY: returns llama.cpp's default sampler-chain params by value.
        let params = unsafe { raw::llama_sampler_chain_default_params() };
        // SAFETY: params are defaults; freed exactly once in Drop via llama_sampler_free.
        let chain = unsafe { raw::llama_sampler_chain_init(params) };
        if chain.is_null() {
            // Broken FFI invariant, not a model rejection.
            panic!("llama_sampler_chain_init returned null");
        }
        // SAFETY: chain is a live sampler chain; init_greedy ownership transfers
        // to the chain (freed with it).
        unsafe { raw::llama_sampler_chain_add(chain, raw::llama_sampler_init_greedy()) };
        Sampler { ptr: chain }
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
fn emit_valid_prefix(pending: &mut Vec<u8>, out: &mut String, bytes: &[u8]) {
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
fn flush_utf8_tail(pending: &mut Vec<u8>, out: &mut String) {
    emit_valid_prefix(pending, out, &[]);
    if !pending.is_empty() {
        out.push_str(&String::from_utf8_lossy(pending));
        pending.clear();
    }
}

/// Writes one slot of the malloc'd batch buffer: token, position, single-sequence
/// id 0, and the logits flag. llama_batch_init mallocs and does not zero, so
/// every used slot must be written before decode.
///
/// # Safety
/// `i` must be below the `n_tokens` the `Batch` was initialized with.
unsafe fn fill_slot(
    b: &mut raw::llama_batch,
    i: usize,
    token: i32,
    pos: raw::llama_pos,
    wants_logits: bool,
) {
    // SAFETY: caller guarantees i is below the allocated n_tokens; each field of
    // slot i is written once, before any decode observes it.
    unsafe {
        *b.token.add(i) = token;
        *b.pos.add(i) = pos;
        *b.n_seq_id.add(i) = 1;
        **b.seq_id.add(i) = 0;
        *b.logits.add(i) = if wants_logits { 1 } else { 0 };
    }
}

/// Applies the chat template: sizing call with a null buffer (documented pattern,
/// returns the total byte count), then the real call into a zeroed `count + 1`
/// buffer (+1 because `strncpy` does not NUL-terminate an exact-size render).
///
/// # Safety
/// `tmpl` must be a valid NUL-terminated CString, and every `ChatMessageFFI` in
/// `msgs` must stay alive for the duration of the call.
unsafe fn apply_template(
    tmpl: &CStr,
    msgs: &[ChatMessageFFI],
    add_ass: bool,
) -> Result<String, EngineError> {
    // Pointers into msgs' CStrings; they outlive both FFI calls below.
    let chat: Vec<raw::llama_chat_message> = msgs
        .iter()
        .map(|m| raw::llama_chat_message {
            role: m.role.as_ptr(),
            content: m.content.as_ptr(),
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
    if n < 0 {
        return Err(EngineError::TemplateApply(n));
    }
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
    if written < 0 {
        return Err(EngineError::TemplateApply(written));
    }
    // SAFETY: buf is NUL-terminated (zeroed +1 byte; at most `written` render bytes).
    let s = unsafe { CStr::from_ptr(buf.as_ptr().cast()) }
        .to_string_lossy()
        .into_owned();
    Ok(s)
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

/// Detokenizes one token to its raw piece bytes (two-pass sizing like
/// [`tokenize_prompt`]; the `lstrip = 0, special = false` call pinned for the
/// emitter — special tokens render as empty pieces and are skipped).
///
/// # Safety
/// `vocab` must be a live llama_vocab owned by the loaded model; `token` must be
/// a valid token id.
unsafe fn token_piece(vocab: *const raw::llama_vocab, token: raw::llama_token) -> Vec<u8> {
    // SAFETY: null/0 buffer only sizes; nothing is written through the null pointer.
    let needed =
        unsafe { raw::llama_token_to_piece(vocab, token, std::ptr::null_mut(), 0, 0, false) };
    let n = needed.unsigned_abs() as usize;
    if n == 0 {
        return Vec::new();
    }
    let mut buf = vec![0u8; n];
    // SAFETY: buf holds exactly the byte count the sizing call demanded.
    let written = unsafe {
        raw::llama_token_to_piece(vocab, token, buf.as_mut_ptr().cast(), n as i32, 0, false)
    };
    if written < 0 {
        // Unreachable: same token, buffer sized from the sizing call.
        return Vec::new();
    }
    buf.truncate(written as usize);
    buf
}

/// Expert copies the pool has served (demand page-ins + fast hits), cumulative.
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// `bytes_skipped / (bytes_skipped + bytes_copied)`; 0.0 with no traffic.
    pub skip_ratio: f64,
}

impl DeviceCacheStats {
    fn new(bytes_copied: u64, bytes_skipped: u64, evictions: u64) -> Self {
        Self {
            bytes_copied,
            bytes_skipped,
            evictions,
            skip_ratio: skip_ratio(bytes_copied, bytes_skipped),
        }
    }
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

    #[test]
    fn skip_ratio_math() {
        assert_eq!(skip_ratio(0, 0), 0.0, "no traffic ⇒ 0.0, not NaN");
        assert_eq!(skip_ratio(3, 0), 0.0, "all copied ⇒ 0.0");
        assert_eq!(skip_ratio(0, 5), 1.0, "all skipped ⇒ 1.0");
        assert_eq!(skip_ratio(1, 3), 0.75);
        let stats = DeviceCacheStats::new(1, 3, 7);
        assert_eq!(stats.bytes_copied, 1);
        assert_eq!(stats.bytes_skipped, 3);
        assert_eq!(stats.evictions, 7);
        assert_eq!(stats.skip_ratio, 0.75);
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

    #[test]
    fn fallback_template_contains_chatml_marker() {
        // llama.cpp detects chatml by the <|im_start|> substring (llama-chat.cpp
        // llm_chat_detect_template); the fallback must carry it.
        assert!(FALLBACK_CHAT_TEMPLATE.contains("<|im_start|>"));
        assert!(FALLBACK_CHAT_TEMPLATE.contains("<|im_end|>"));
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
        let (k, top_k) =
            s.p3.as_ref()
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
}
