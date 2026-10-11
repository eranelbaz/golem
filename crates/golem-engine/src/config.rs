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
    /// Maximum tokens to generate per completion (a request's `max_tokens`
    /// overrides it).
    pub n_predict: u32,
    /// Decode slots (concurrent sequences). Only 1 is accepted until the
    /// multi-slot scheduler lands.
    pub parallel: u32,
    /// CPU threadpool size (§7.2 as amended by R-15: physical cores / 2,
    /// min 1 — CPU-side ops still run on these: sampling, the non-offloaded
    /// tail, tokenizer, ids readback). Target: 16 physical cores → 8.
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
    /// Derived store reader threads per physical drive (§5.2, D-12: 1–2);
    /// 0 = 1. Ignored when `io_threads` is set.
    pub io_depth: u32,
    /// Interleave the RAM expert pool's pages across every NUMA node
    /// (`MPOL_INTERLEAVE` while the arena faults in); `false` keeps the
    /// first-touch policy. A no-op on one node.
    pub numa_interleave: bool,
    /// Store reads must go `O_DIRECT` (§5.4): a replica without a direct
    /// handle, or a direct read failure, fails the read instead of falling
    /// back to buffered + `DONTNEED`. Needs a manifest. `false` keeps the
    /// probe-and-fall-back behaviour.
    pub o_direct: bool,
    /// Pooled expert H2D on a dedicated copy stream overlapping compute.
    /// Not implemented by the vendored scheduler: `true` is refused at load.
    pub async_h2d: bool,
    /// Share of the §10 planner's `device_expert_cache` allowance the
    /// derived K may use (1.0 = all of it). No effect with an explicit K.
    pub vram_experts_frac: f32,
    /// Model manifest (§4.2) for the store-backed expert source: when set,
    /// the P3 slow path reads expert planes from the manifest's replica set
    /// through the shared chunk queue; `None` (the default) reads the single
    /// GGUF replica via pread.
    pub manifest_path: Option<PathBuf>,
    /// §11 per-step timing CSV destination: when set, each generation step's
    /// timings are recorded and flushed at turn end; `None` (the default)
    /// records nothing.
    pub timing_csv: Option<PathBuf>,
    /// Jinja chat template overriding the model's own (`None`: model
    /// metadata, else chatml).
    pub chat_template: Option<String>,
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
    /// RAM/VRAM expert residency (§6.3 O10): `Inclusive` (the default) keeps
    /// an expert's pool slot while its VRAM copy exists; `Exclusive` releases
    /// the pool slot once the H2D lands and D2Hs evictions back into the
    /// pool. Exclusive stays opt-in until the A/B proves it on decode.
    pub residency: Residency,
    /// Prefill layer streaming (§7.6, `--prefill-stream`): prompt chunks of
    /// `prefill_ubatch` tokens stream each expert plane through
    /// double-buffered device slabs instead of paging misses into the pool.
    /// Off by default; auto-disabled at enable when the VRAM budget cannot
    /// fit both the stream slabs and a K ≥ top-k (a warning, never an error).
    pub prefill_stream: bool,
    /// Streamed prefill ubatch (`--prefill-ubatch`); `n_ubatch` of the
    /// context when `prefill_stream` is on. 0 = 2048 (the §7.6 default).
    pub prefill_ubatch: u32,
    /// Minimum prompt-tail length that still streams (`--prefill-stream-min`):
    /// below it the chunk takes the per-token path (streaming a whole-plane
    /// ubatch costs ≥ one disk pass regardless of chunk size).
    pub prefill_stream_min: u32,
    /// Routed-share switch (`--prefill-stream-routed-max`): above this share
    /// of the ubatch's routed experts a plane streams whole instead of
    /// routed-only (BMoE's 0.85; Golem measures its own crossover).
    pub prefill_stream_routed_max: f32,
    /// State-snapshot arena bytes (§7.1): `Some(0)` disables snapshots (and
    /// the floor rule); `None` = planner default `max(2 GiB, 2 × floor)`;
    /// any other value must be ≥ the floor `parallel × snapshot_bytes(n_ctx)`
    /// (one full-context snapshot per slot) or the load is refused.
    pub snapshot_ram_bytes: Option<u64>,
    /// Header facts the caller already parsed (snapshot floor sizing);
    /// `None` = the engine reads them from `model_path`.
    pub model_facts: Option<golem_plan::ModelFacts>,
    /// R-15: CPUs the store read queue and prefetch threads pin themselves
    /// to at spawn (the non-V-cache CCD); `None` leaves them unpinned.
    pub io_cpuset: Option<golem_store::sys::CpuSet>,
    /// R-15: CPUs the thread calling `Session::new` pins itself to before
    /// anything loads (the V-cache CCD), so ggml's threadpool and every
    /// load-time thread inherit it; `None` leaves affinity untouched.
    pub decode_cpuset: Option<golem_store::sys::CpuSet>,
    /// The cpusets came from `--pin auto` detection: a decode pin failure
    /// warns and continues instead of failing the load.
    pub pin_auto: bool,
    /// R-8 start check: full SHA-256 of every shard instead of the sampled
    /// windows (`--full-verify`). The check runs only with a manifest.
    pub full_verify: bool,
}

/// R-15 default CPU threadpool: physical cores / 2 (min 1), logical / 2
/// when the physical count is unknown.
pub fn default_threads() -> u32 {
    let phys = sysinfo::System::new().physical_core_count();
    let n = phys.unwrap_or_else(|| std::thread::available_parallelism().map_or(2, |n| n.get()));
    (n / 2).max(1) as u32
}

/// RAM/VRAM expert residency policy (§6.3 O10, `--residency`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Residency {
    /// An expert may live in both tiers (today's behaviour).
    #[default]
    Inclusive,
    /// At most one tier holds an expert: promotion releases the pool slot,
    /// VRAM eviction copies back into the pool or drops.
    Exclusive,
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
            parallel: 1,
            threads: default_threads(),
            n_gpu_layers: 0,
            flash_attn_on: false,
            cpu_moe_layers: None,
            ram_experts_gib: 0,
            device_cache_experts: 0,
            io_threads: None,
            io_depth: 0,
            numa_interleave: false,
            o_direct: false,
            async_h2d: false,
            vram_experts_frac: 1.0,
            manifest_path: None,
            timing_csv: None,
            routing_trace: None,
            vram_order: VramOrder::default(),
            prefetch: PrefetchMode::default(),
            prefetch_lookahead: 1,
            residency: Residency::default(),
            prefill_stream: false,
            prefill_ubatch: 2048,
            prefill_stream_min: 256,
            prefill_stream_routed_max: 0.85,
            // Off by default: only the CLI asks for the planner-sized arena.
            snapshot_ram_bytes: Some(0),
            model_facts: None,
            chat_template: None,
            io_cpuset: None,
            decode_cpuset: None,
            pin_auto: false,
            full_verify: false,
        }
    }
}

impl std::str::FromStr for Residency {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "inclusive" => Ok(Self::Inclusive),
            "exclusive" => Ok(Self::Exclusive),
            _ => Err(format!("want inclusive or exclusive, got {s:?}")),
        }
    }
}

impl std::str::FromStr for PrefetchMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "off" => Ok(Self::Off),
            "protect" => Ok(Self::Protect),
            "read" => Ok(Self::Read),
            _ => Err(format!("want off, protect or read, got {s:?}")),
        }
    }
}

impl std::str::FromStr for VramOrder {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "score" => Ok(Self::Score),
            "recency" => Ok(Self::Recency),
            _ => Err(format!("want score or recency, got {s:?}")),
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
        if let Some(p) = sysinfo::System::new().physical_core_count() {
            assert_eq!(c.threads, (p / 2).max(1) as u32, "R-15: physical / 2");
        }
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
