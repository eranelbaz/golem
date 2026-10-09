//! Engine error type mapping llama.cpp failures to loud, specific messages.

use std::io;
use std::path::PathBuf;

/// Errors raised by the engine across model loading, session setup, and decode.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// `EngineConfig` value out of its documented range.
    #[error("invalid engine config: {0}")]
    InvalidConfig(String),
    /// GGUF file malformed at the byte level (magic, header, tensor table).
    #[error("invalid GGUF model: {0}")]
    InvalidModel(String),
    /// GGUF parses but violates a load-time check (tech design §4.3 subset).
    #[error("unsupported GGUF model: {0}")]
    UnsupportedModel(String),
    /// llama_model_load_from_file returned null.
    #[error("llama model load failed: {0}")]
    ModelLoad(PathBuf),
    /// llama_init_from_model returned null.
    #[error("llama_init_from_model returned null")]
    ContextCreate,
    /// No chat template available and no fallback applies.
    #[error("no chat template available and no fallback applies")]
    TemplateUnavailable,
    /// llama_chat_apply_template returned a negative code.
    #[error("llama_chat_apply_template failed with code {0}")]
    TemplateApply(i32),
    /// The chat layer (common/chat.h shim) rejected a render or parse.
    #[error("chat template: {0}")]
    ChatTemplate(String),
    /// llama_tokenize returned a negative code.
    #[error("llama_tokenize failed with code {0}")]
    Tokenize(i32),
    /// llama_decode returned nonzero (positive: no KV slot / aborted; negative: fatal).
    #[error("llama_decode failed with code {0}")]
    Decode(i32),
    /// Prompt does not fit the per-sequence context (`llama_n_ctx_seq`).
    #[error("prompt is {prompt} tokens; context holds {n_ctx}")]
    ContextOverflow {
        /// Prompt length in tokens.
        prompt: u32,
        /// `llama_n_ctx_seq` of the live context.
        n_ctx: u32,
    },
    /// llama_sampler_chain_init returned null.
    #[error("llama_sampler_chain_init returned null")]
    SamplerInit,
    /// llama_get_logits_ith returned null after a successful decode — the
    /// substrate broke the logits-output FFI contract.
    #[error("llama_get_logits_ith returned null after a successful decode")]
    LogitsUnavailable,
    /// T1 op-parity leg: raw ggml failure — `ggml_init` returned null,
    /// a tensor allocation failed, or the graph compute reported a
    /// non-success [`ggml_status`].
    #[error("ggml op-parity leg failed: {0}")]
    OpCompute(String),
    /// Filesystem failure.
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    /// RAM expert pool (P3, §5.4) construction failed — arena mapping/mlock,
    /// expert-tensor geometry, or a second pool-enabled session in one process.
    #[error("RAM expert pool: {0}")]
    PoolInit(String),
    /// Expert math would run off the GPU (design §2.3 I2, §8.4
    /// `golem_backend_unavailable`): refused rather than a silent CPU fallback.
    #[error("GPU backend unavailable for expert math: {0}")]
    BackendUnavailable(String),
    /// The §10 memory model refuses the configuration at load (§6.4).
    #[error("golem_plan_infeasible: {}", .0.violation().unwrap_or_default())]
    PlanInfeasible(Box<golem_plan::PlanReport>),
    /// A device-cache slab allocation failed after load (`llama_decode`
    /// −2): the turn fails, the cache is reset and shrunk to the carried K,
    /// and the session keeps serving degraded.
    #[error("device expert cache allocation failed; shrunk to K = {0} experts per layer")]
    DeviceCacheDegraded(u32),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_is_loud_and_specific() {
        let e = EngineError::InvalidModel("no magic".into());
        assert_eq!(e.to_string(), "invalid GGUF model: no magic");
        let e = EngineError::Decode(-1);
        assert_eq!(e.to_string(), "llama_decode failed with code -1");
        let e = EngineError::ModelLoad(PathBuf::from("/tmp/x.gguf"));
        assert_eq!(e.to_string(), "llama model load failed: /tmp/x.gguf");
    }

    #[test]
    fn error_is_std_error() {
        fn assert_error<T: std::error::Error>(_: &T) {}
        assert_error(&EngineError::ContextCreate);
    }
}
