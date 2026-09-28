//! Engine error type mapping llama.cpp failures to loud, specific messages.

use std::io;
use std::path::PathBuf;

/// Errors raised by the engine across model loading, session setup, and decode.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
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
    /// llama_tokenize returned a negative code.
    #[error("llama_tokenize failed with code {0}")]
    Tokenize(i32),
    /// llama_decode returned nonzero (positive: no KV slot / aborted; negative: fatal).
    #[error("llama_decode failed with code {0}")]
    Decode(i32),
    /// T1 op-parity leg (M1.4): raw ggml failure — `ggml_init` returned null,
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
