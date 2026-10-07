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
    /// Decode threads (§7.2 as amended: logical CPUs / 2, min 1 — CPU-side
    /// ops still run on these: sampling, the non-offloaded tail, tokenizer,
    /// ids readback).
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
    /// Per-(layer, kind) device expert-slab size, in experts (§3.4 as amended
    /// for the slot-indirected cache). **0 (the default) = derive from free
    /// VRAM at pool enable** (half of free, bounded; never unlimited —
    /// unbounded slabs OOM the 24 GiB card on the real model). A routed set
    /// larger than K aborts loudly, so a K below the model's top-k is rejected at enable.
    pub device_cache_experts: u32,
    /// Store reader threads per drain (§5.4 `--io-threads`); `None` derives
    /// one thread per physical drive hosting an eligible replica.
    pub io_threads: Option<usize>,
    /// Model manifest (§4.2) for the store-backed expert source: when set,
    /// the P3 slow path reads expert planes from the manifest's replica set
    /// through the shared chunk queue; `None` (the default) reads the single
    /// GGUF replica via pread.
    pub manifest_path: Option<PathBuf>,
    /// §11 per-step timing CSV destination: when set, each generation step's
    /// timings are recorded and flushed at turn end; `None` (the default)
    /// records nothing.
    pub timing_csv: Option<PathBuf>,
    /// Routing-trace destination (§11 timing-CSV pattern): when set, every
    /// routing readback is appended to a preallocated per-step ring and
    /// flushed at turn end as `step\tphase\tlayer\tid,id,...` under a
    /// `# golem-routing-trace v1` header; `None` (the default) traces
    /// nothing — the trampoline's only cost is one `Option` check.
    pub routing_trace: Option<PathBuf>,
    /// P2 evict-first order (§6.2). The policy replay
    /// (`docs/baselines/m3.7-policy-replay.md`) puts the decayed score
    /// ahead of recency on real traces.
    pub vram_order: VramOrder,
    /// §6.5 next-layer prediction (P5): `Off` (the default) builds no
    /// prediction nodes and runs no prefetch thread.
    pub prefetch: PrefetchMode,
    /// Layers ahead each MoE layer predicts (1 = the next layer only).
    pub prefetch_lookahead: u32,
}

/// What a P5 next-layer prediction drives (§6.5, `--prefetch`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrefetchMode {
    /// No prediction.
    #[default]
    Off,
    /// Predicted experts are shielded from RAM-pool eviction; no extra I/O.
    Protect,
    /// Predicted experts missing from the pool are read at prefetch priority.
    Read,
}

/// How the P2 slab picks its victim among a layer's non-routed slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VramOrder {
    /// Least recently routed first.
    Recency,
    /// Lowest §6.2 decayed score first.
    #[default]
    Score,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            n_ctx: 256,
            batch_tokens: 512,
            n_predict: 64,
            threads: std::thread::available_parallelism()
                .map(|n| (n.get() / 2).max(1) as u32)
                .unwrap_or(1),
            n_gpu_layers: 0,
            flash_attn_on: false,
            cpu_moe_layers: None,
            ram_experts_gib: 0,
            device_cache_experts: 0,
            io_threads: None,
            manifest_path: None,
            timing_csv: None,
            routing_trace: None,
            vram_order: VramOrder::default(),
            prefetch: PrefetchMode::default(),
            prefetch_lookahead: 1,
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
        assert!(c.threads >= 1, "threadpool never empty");
        assert_eq!(c.model_path, PathBuf::new());
        assert_eq!(
            c.n_gpu_layers, 0,
            "default offloads nothing (CI leg unchanged)"
        );
        assert!(!c.flash_attn_on, "default leaves flash attention at AUTO");
        assert_eq!(c.cpu_moe_layers, None, "default writes no buft overrides");
        assert_eq!(c.ram_experts_gib, 0, "RAM expert pool defaults off (T4)");
        assert_eq!(
            c.device_cache_experts, 0,
            "0 = unset: the engine derives a bounded K from free VRAM at enable"
        );
        assert_eq!(
            c.io_threads, None,
            "io threads default to the derived per-drive count"
        );
    }
}
