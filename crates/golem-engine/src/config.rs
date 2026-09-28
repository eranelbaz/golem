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
    /// Model layers to offload to the GPU (`llama_model_params::n_gpu_layers`,
    /// llama.h): 0 keeps the whole model on CPU — the CI-leg default.
    pub n_gpu_layers: u32,
    /// Force Flash Attention (`LLAMA_FLASH_ATTN_TYPE_ENABLED` on the context
    /// params); `false` leaves llama.cpp's `AUTO` default untouched.
    pub flash_attn_on: bool,
    /// Expert (MoE) FFN weights of the first N layers pinned to CPU buffers via
    /// `tensor_buft_overrides` — the public-API equivalent of the vendor's
    /// `-ncmoe N` flag (a common/arg.cpp path unreachable through libllama at
    /// this pin). `None` keeps the llama.cpp default (no overrides).
    pub cpu_moe_layers: Option<u32>,
    /// RAM expert pool size in GiB (P3, §5.4). **0 = off** (the default): no
    /// pool is built, hooks stay unset, decode is byte-identical upstream
    /// (T4 invariant). The CLI computes a suggestion but always passes an
    /// explicit whole-GiB value.
    pub ram_experts_gib: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            n_ctx: 256,
            batch_tokens: 512,
            n_predict: 64,
            threads: 1,
            n_gpu_layers: 0,
            flash_attn_on: false,
            cpu_moe_layers: None,
            ram_experts_gib: 0,
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
        assert_eq!(c.threads, 1);
        assert_eq!(c.model_path, PathBuf::new());
        assert_eq!(
            c.n_gpu_layers, 0,
            "default offloads nothing (CI leg unchanged)"
        );
        assert!(!c.flash_attn_on, "default leaves flash attention at AUTO");
        assert_eq!(c.cpu_moe_layers, None, "default writes no buft overrides");
        assert_eq!(c.ram_experts_gib, 0, "RAM expert pool defaults off (T4)");
    }
}
