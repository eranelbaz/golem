//! Model + context lifecycle (tech design §7): backend init, load with
//! fail-fast sanity checks, RAII teardown, the §4.3 decode self-check probe,
//! chat-template resolution (model metadata, else the chatml fallback), and
//! the single-slot greedy decode loop (`Session::chat_turn`).
#![allow(unsafe_code)]

use std::cell::Cell;
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
                role: CString::new(m.role)
                    .map_err(|_| EngineError::InvalidModel("chat message role contains NUL".into()))?,
                content: CString::new(m.content).map_err(|_| {
                    EngineError::InvalidModel("chat message content contains NUL".into())
                })?,
            })
        })
        .collect()
}

/// A loaded model bound to its resolved chat template, driving one decode slot
/// (M1.3: stateless per turn — every `chat_turn` re-prefills the full transcript).
pub struct Session {
    loaded: LoadedModel,
    config: EngineConfig,
    template: CString,
    last_prompt_tokens: Vec<raw::llama_token>,
}

impl Session {
    /// Loads per `config`, resolves the chat template (model-embedded, else
    /// [`FALLBACK_CHAT_TEMPLATE`]), and validates it with a 1-message probe apply:
    /// apply failure → `TemplateApply(code)`; empty/unusable → `TemplateUnavailable`.
    pub fn new(config: &EngineConfig) -> Result<Self, EngineError> {
        let loaded = load(config)?;
        let template_src = loaded
            .model()
            .chat_template()
            .unwrap_or_else(|| FALLBACK_CHAT_TEMPLATE.to_owned());
        // Interior NUL would truncate the template at the FFI boundary: unusable.
        let template =
            CString::new(template_src).map_err(|_| EngineError::TemplateUnavailable)?;

        let probe = ffi_messages(&[ChatTurn {
            role: "user",
            content: "ping",
        }])?;
        // SAFETY: template is a live CString; probe CStrings outlive the call.
        if unsafe { apply_template(&template, &probe, true) }?.is_empty() {
            return Err(EngineError::TemplateUnavailable);
        }

        Ok(Session {
            loaded,
            config: config.clone(),
            template,
            last_prompt_tokens: Vec::new(),
        })
    }

    /// The tokenized prompt of the most recent `chat_turn` (post-render,
    /// post-special-tokens).
    pub fn last_prompt_tokens(&self) -> &[i32] {
        &self.last_prompt_tokens
    }

    /// Runs one chat turn: renders `messages` with the session template,
    /// clears the KV cache, re-prefills the full transcript in chunks, then
    /// greedily generates until EOG, `n_ctx - 1` positions, or `n_predict`
    /// tokens. Piece bytes stream through `emit` as they complete UTF-8
    /// characters (verbatim — no space stripping). Returns tokens generated.
    pub fn chat_turn(
        &mut self,
        messages: &[ChatTurn<'_>],
        emit: &mut dyn FnMut(&str),
    ) -> Result<u32, EngineError> {
        let msgs = ffi_messages(messages)?;
        // SAFETY: template and msgs CStrings are alive for the duration of the call.
        let rendered = unsafe { apply_template(&self.template, &msgs, true)? };

        // Full-transcript re-prefill: every turn is stateless w.r.t. prior turns.
        let ctx = self.loaded.ctx();
        // SAFETY: llama_get_memory returns the context's live memory handle; the
        // context is idle between turns, the documented llama_memory_clear use.
        unsafe { raw::llama_memory_clear(raw::llama_get_memory(ctx.as_ptr()), true) };

        let vocab = self.loaded.model().vocab();
        // add_special = true: the fallback chatml renders no BOS text, so this
        // yields exactly one BOS (templates embedding BOS are an M2.x concern).
        // SAFETY: vocab is live; rendered is valid for its byte length.
        let prompt = unsafe { tokenize_prompt(vocab, &rendered, true, true)? };
        if prompt.is_empty() {
            // Nothing to prefill; llama_decode would run on an empty batch.
            return Err(EngineError::Tokenize(0));
        }
        self.last_prompt_tokens = prompt.clone();
        let prompt_len = prompt.len();

        let n_ctx = self.config.n_ctx as usize;
        // Chunks of min(batch_tokens, n_ctx - 1): one KV slot stays free so the
        // first generated token always has a position to decode into.
        let chunk_cap = (self.config.batch_tokens as usize).min(n_ctx.saturating_sub(1));
        if chunk_cap == 0 {
            // n_ctx <= 1 cannot fit even one prompt token — llama_decode would
            // return 1; fail with that same code instead of chunking by zero.
            return Err(EngineError::Decode(1));
        }

        // One batch per turn: reused for prefill chunks (n_tokens reset per chunk)
        // and generation steps (n_tokens = 1); freed by Drop.
        let mut batch = Batch::init(chunk_cap as i32, 0, 1);
        let mut n_past: usize = 0;
        let mut last_chunk_len = 0;
        for chunk in prompt.chunks(chunk_cap) {
            let len = chunk.len();
            let is_final_chunk = n_past + len == prompt_len;
            {
                // SAFETY: slots 0..len are within the chunk_cap-sized allocation;
                // every used slot is fully written before decode (llama_batch_init
                // mallocs and does not zero — Task 4 probe precedent).
                let b = &mut batch.inner;
                for (i, &t) in chunk.iter().enumerate() {
                    // Logits only on the final token of the final chunk: that
                    // sample bootstraps generation; intermediate tokens never need it.
                    // SAFETY: i < len <= chunk_cap slots allocated by Batch::init.
                    unsafe {
                        fill_slot(b, i, t, (n_past + i) as raw::llama_pos, is_final_chunk && i + 1 == len);
                    }
                }
                b.n_tokens = len as i32;
            }
            // SAFETY: ctx is a live context; batch slots 0..n_tokens fully written.
            let code = unsafe { raw::llama_decode(ctx.as_ptr(), batch.inner) };
            if code != 0 {
                // 1 = no KV slot left (ctx too small for the prompt); the rest fatal alike.
                return Err(EngineError::Decode(code));
            }
            n_past += len;
            last_chunk_len = len;
        }

        // Greedy sampler chain; the chain owns the greedy sampler and frees it.
        let sampler = Sampler::greedy();

        // Bootstrap: sample from the prefill's final-token logits.
        // SAFETY: sampler chain and ctx are live; the final decode emitted logits
        // for slot last_chunk_len - 1.
        let mut tok =
            unsafe { raw::llama_sampler_sample(sampler.as_ptr(), ctx.as_ptr(), (last_chunk_len - 1) as i32) };

        let mut generated: u32 = 0;
        let mut scratch = String::new();
        loop {
            // SAFETY: vocab is live; tok is a valid token id from the sampler.
            if unsafe { raw::llama_vocab_is_eog(vocab, tok) } {
                break;
            }
            // SAFETY: vocab is live; tok is a valid token id from the sampler.
            let piece = unsafe { token_piece(vocab, tok) };
            emit_valid_prefix(&mut scratch, &piece);
            if !scratch.is_empty() {
                emit(&scratch);
                scratch.clear();
            }
            generated += 1;
            // One KV slot stays free (chunk_cap = min(batch_tokens, n_ctx - 1));
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
            // SAFETY: same invariants as the prefill decode.
            let code = unsafe { raw::llama_decode(ctx.as_ptr(), batch.inner) };
            if code != 0 {
                return Err(EngineError::Decode(code));
            }
            n_past += 1;
            // SAFETY: sampler chain and ctx are live; the decode emitted logits for slot 0.
            tok = unsafe { raw::llama_sampler_sample(sampler.as_ptr(), ctx.as_ptr(), 0) };
        }
        // Turn end: flush whatever bytes remain buffered (n_predict can cut a
        // multi-byte character in half — lossy-drain that residue).
        flush_utf8_tail(&mut scratch);
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

// Bytes of a not-yet-complete UTF-8 character, carried across `emit_valid_prefix`
// calls. Sound for the engine: `Session` is !`Send` (raw pointers), so a turn's
// calls are sequential on one thread, and each turn flushes the buffer empty
// before returning (`flush_utf8_tail`).
thread_local! {
    static PENDING_UTF8: Cell<Vec<u8>> = const { Cell::new(Vec::new()) };
}

/// Appends `bytes` to the pending byte buffer and emits — into `out` — the
/// longest valid UTF-8 prefix; an incomplete trailing sequence stays pending
/// until more bytes arrive (`std::str::from_utf8`: `valid_up_to` for the flush
/// length, `error_len == None` = keep the tail). Invalid bytes surface as
/// U+FFFD (`from_utf8_lossy` semantics). Byte-buffered because Task 0's
/// byte-fallback vocab legitimately splits multi-byte characters across pieces;
/// per-piece lossy conversion would corrupt them.
fn emit_valid_prefix(out: &mut String, bytes: &[u8]) {
    PENDING_UTF8.with(|cell| {
        let mut pending = cell.take();
        pending.extend_from_slice(bytes);
        loop {
            match std::str::from_utf8(&pending) {
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
        cell.set(pending);
    });
}

/// Turn-end flush: complete any character pending on a real boundary, then
/// lossily drain the residue (a multi-byte character truncated mid-sequence
/// can never complete).
fn flush_utf8_tail(out: &mut String) {
    emit_valid_prefix(out, &[]);
    PENDING_UTF8.with(|cell| {
        let rest = cell.take();
        if !rest.is_empty() {
            out.push_str(&String::from_utf8_lossy(&rest));
        }
    });
}

/// Writes one slot of the malloc'd batch buffer: token, position, single-sequence
/// id 0, and the logits flag. llama_batch_init mallocs and does not zero, so
/// every used slot must be written before decode (Task 4 probe precedent).
///
/// # Safety
/// `i` must be below the `n_tokens` the `Batch` was initialized with.
unsafe fn fill_slot(b: &mut raw::llama_batch, i: usize, token: i32, pos: raw::llama_pos, wants_logits: bool) {
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
    let n_msg = chat.len() as raw::size_t;
    // SAFETY: tmpl and chat point to live NUL-terminated data; null/0 buffer is
    // the documented sizing call (returns the total number of bytes needed,
    // negative on detection/apply failure).
    let n = unsafe {
        raw::llama_chat_apply_template(tmpl.as_ptr(), chat.as_ptr(), n_msg, add_ass, std::ptr::null_mut(), 0)
    };
    if n < 0 {
        return Err(EngineError::TemplateApply(n));
    }
    let mut buf = vec![0u8; usize::try_from(n).unwrap_or(0) + 1];
    // SAFETY: buf is n+1 zeroed bytes and length n matches the sizing result.
    let written = unsafe {
        raw::llama_chat_apply_template(tmpl.as_ptr(), chat.as_ptr(), n_msg, add_ass, buf.as_mut_ptr().cast(), n)
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
    let needed = unsafe {
        raw::llama_token_to_piece(vocab, token, std::ptr::null_mut(), 0, 0, false)
    };
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fallback_template_contains_chatml_marker() {
        // llama.cpp detects chatml by the <|im_start|> substring (llama-chat.cpp
        // llm_chat_detect_template); the fallback must carry it.
        assert!(FALLBACK_CHAT_TEMPLATE.contains("<|im_start|>"));
        assert!(FALLBACK_CHAT_TEMPLATE.contains("<|im_end|>"));
    }

    #[test]
    fn utf8_emitter_splits_only_on_char_boundaries() {
        let mut out = String::new();
        emit_valid_prefix(&mut out, &[0xE4, 0xBD]); // incomplete 3-byte char
        assert!(out.is_empty());
        emit_valid_prefix(&mut out, &[0xA0]);
        assert_eq!(out, "\u{4f60}"); // 你 — assembled across two pieces
    }
}
