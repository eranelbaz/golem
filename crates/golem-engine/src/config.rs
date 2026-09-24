//! Engine configuration (tech design §7.2 defaults).

use std::path::PathBuf;

/// Knobs for constructing a llama.cpp context and running the decode loop.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Path to the GGUF model file; empty until a model is selected.
    pub model_path: PathBuf,
    /// Context size (KV cache slots).
    pub n_ctx: u32,
    /// Maximum tokens per llama_decode batch.
    pub batch_tokens: u32,
    /// Maximum tokens to generate per completion.
    pub n_predict: u32,
    /// Decode threads (tech design §7.2: default 1).
    pub threads: u32,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            n_ctx: 256,
            batch_tokens: 512,
            n_predict: 64,
            threads: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_design() {
        let c = EngineConfig::default();
        assert_eq!(c.n_ctx, 256);
        assert_eq!(c.batch_tokens, 512);
        assert_eq!(c.n_predict, 64);
        assert_eq!(c.threads, 1); // §7.2: decode-threads default 1
        assert_eq!(c.model_path, PathBuf::new());
    }
}
