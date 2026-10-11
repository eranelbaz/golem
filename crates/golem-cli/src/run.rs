//! `golem run` (tech design §9): greedy decode of a local GGUF model, either
//! one-shot via `--prompt` or as a stdin multi-turn chat REPL.
// The RLIMIT_MEMLOCK preflight is the one sanctioned unsafe call-site here.
#![allow(unsafe_code)]

use std::error::Error;
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use golem_engine::ParityStep;
use golem_engine::atomic::write_atomic;
use golem_engine::config::{EngineConfig, PrefetchMode, Residency, VramOrder};
use golem_engine::gguf::read_metadata;
use golem_engine::session::{ChatTurn, Session};
use golem_engine::timing::TimingRecorder;
use golem_llama_sys::{Backend, substrate_backend};
use golem_placement::pool::{GgufPreadSource, StoreExpertSource};
use golem_store::ChunkQueue;
use serde::Serialize;
use sha2::{Digest, Sha256};

/// Arguments of the `run` subcommand (tech design §9). Engine flags resolve
/// through §9.4 precedence ([`crate::config`]).
#[derive(Debug, clap::Args)]
pub struct RunArgs {
    /// Path to the GGUF model file (first shard of a split).
    /// When omitted, `golem run` prints the full flag list and exits.
    pub model: Option<PathBuf>,
    /// One-shot prompt text; when omitted, the stdin chat REPL runs instead.
    #[arg(long)]
    pub prompt: Option<String>,
    /// Write a T2 parity dump (JSONL: line-1 header, then one record per
    /// generation step) to this file instead of streaming the completion;
    /// requires `--prompt`. Model identity in the dump is sha256 + byte size,
    /// never a path.
    #[arg(long, value_name = "FILE")]
    pub parity_dump: Option<PathBuf>,
    /// With no compatible instance running, start a detached `serve` and
    /// attach to it; it keeps running after this session ends.
    #[arg(long, conflicts_with = "new")]
    pub keep: bool,
    /// Load in-process even when a running instance could serve this.
    #[arg(long)]
    pub new: bool,
    /// Skip the O5 admission check against running instances and the RAM fit.
    #[arg(long)]
    pub allow_overcommit: bool,
    #[command(flatten)]
    pub engine: EngineArgs,
}

/// Engine flags shared by `run` and `serve`.
#[derive(Debug, clap::Args)]
pub struct EngineArgs {
    /// Maximum tokens to generate per completion (engine §7.2 default: 64).
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub n_predict: Option<u32>,
    /// Context size in KV cache slots (engine §7.2 default: 256).
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..))]
    pub ctx: Option<u32>,
    /// CPU threadpool size (R-15 default: physical cores / 2, min 1). The
    /// calibration matrix runs reference-vs-reference legs at differing
    /// thread counts (cpu-t16 vs cpu-t4), so the suite driver injects this
    /// flag per leg.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub threads: Option<u32>,
    /// 0 keeps the whole model on CPU (the CI-leg default); an omitted flag
    /// infers 99 on HIP builds (hardware-parity legs pass it explicitly), 0 on
    /// CPU. Without any GPU offload a "hip"-labeled binary would silently run
    /// on CPU.
    #[arg(long, value_parser = clap::value_parser!(u32))]
    pub n_gpu_layers: Option<u32>,
    /// Force Flash Attention on (`EngineConfig::flash_attn_on` —
    /// `LLAMA_FLASH_ATTN_TYPE_ENABLED`); without it llama.cpp's AUTO default
    /// applies. Hardware parity legs run `fa = on` on both backends.
    #[arg(long)]
    pub flash_attn: bool,
    /// Expert (MoE) FFN weights of the first N layers pinned to CPU buffers
    /// (`EngineConfig::cpu_moe_layers` — the public-API equivalent of the
    /// vendor's `-ncmoe N`; unset keeps the llama.cpp default). The hardware
    /// HIP leg combines this with `--n-gpu-layers` to reproduce `-ncmoe 45`.
    #[arg(long, value_parser = clap::value_parser!(u32))]
    pub cpu_moe_layers: Option<u32>,
    /// RAM expert pool size in GiB (P3, §5.4); 0/omitted keeps the pool off.
    /// Fractional GiB are floored to whole GiB (the engine config is whole-GiB).
    /// The pool is `mlock`ed: before loading anything, the preflight checks
    /// `RLIMIT_MEMLOCK` and available RAM, and fails with an actionable
    /// suggestion instead of failing mid-load. With an inferred manifest but
    /// no explicit size, the pool is sized to the preflight cap itself.
    #[arg(long, value_name = "GIB")]
    pub ram_experts_gib: Option<f64>,
    /// Per-layer device expert-slab size in experts
    /// (`EngineConfig::device_cache_experts`); omitted/0 derives it from free
    /// VRAM, which needs a GPU backend — CPU pooled runs must set it. Must be
    /// at least the model's top-k.
    #[arg(long, value_name = "EXPERTS")]
    pub device_cache_experts: Option<u32>,
    /// Store reader threads per drain (§5.4, `EngineConfig::io_threads`);
    /// omitted derives one thread per physical drive hosting an eligible
    /// replica (D-12: per-drive threads, never per-file).
    #[arg(long, value_name = "N")]
    pub io_threads: Option<usize>,
    /// Derived store reader threads per physical drive (D-12: 1–2; default
    /// 1); `--io-threads` overrides the derived count.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..=8))]
    pub io_depth: Option<u32>,
    /// Interleave the RAM expert pool across NUMA nodes (no-op on one node).
    #[arg(long)]
    pub numa_interleave: bool,
    /// Require `O_DIRECT` store reads: fail instead of falling back to
    /// buffered + DONTNEED. Needs a manifest.
    #[arg(long)]
    pub o_direct: bool,
    /// Pooled H2D on a dedicated copy stream (refused: the substrate has
    /// none yet).
    #[arg(long, hide = true)]
    pub async_h2d: bool,
    /// Share of the planner's device-expert-cache allowance a derived
    /// `--device-cache-experts` may use, in (0, 1] (default 1).
    #[arg(long, value_name = "FRAC")]
    pub vram_experts_frac: Option<f32>,
    /// Ignore the saved `golem optimize` profile for this model.
    #[arg(long)]
    pub no_optimize_profile: bool,
    /// Model manifest (§4.2): the P3 expert pool then reads expert planes
    /// from the manifest's replica set through the shared chunk queue
    /// (striped, CRC-checked) instead of the single GGUF replica. Omitted →
    /// `manifest.json` next to the model file, when present.
    #[arg(long, value_name = "PATH")]
    pub manifest: Option<PathBuf>,
    /// §11 per-step timing CSV destination: opened/truncated with a header
    /// (engine version, fork commit, resolved config), then one row per
    /// generation decode step, flushed at each turn end. H2D columns are
    /// `unmeasured` (no fork event pair on the copy stream yet).
    #[arg(long, value_name = "FILE")]
    pub timing_csv: Option<PathBuf>,
    /// Per-step routing-trace destination (design §2.3 step 4): a
    /// `# golem-routing-trace v1` header, then one `step\tphase\tlayer\tid,...`
    /// record per MoE layer per decode step, flushed at each turn end.
    /// Requires the RAM expert pool (`--ram-experts-gib`); consumes no
    /// per-step heap when unset.
    #[arg(long, value_name = "FILE")]
    pub routing_trace: Option<PathBuf>,
    /// P2 evict-first order among a layer's non-routed VRAM slots:
    /// `score` (§6.2 decayed score, the default) or `recency`.
    #[arg(long, value_name = "ORDER")]
    pub vram_order: Option<VramOrder>,
    /// §6.5 next-layer routing prediction (P5): `off` (default), `protect`
    /// (shield predicted experts from RAM eviction) or `read` (page predicted
    /// experts in at prefetch priority). Needs the RAM expert pool.
    #[arg(long, value_name = "MODE")]
    pub prefetch: Option<PrefetchMode>,
    /// Layers ahead each MoE layer predicts for `--prefetch` (default 1).
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..=4))]
    pub prefetch_lookahead: Option<u32>,
    /// RAM/VRAM expert residency (design §6.3 O10): `inclusive` (default)
    /// keeps an expert's pool slot while its VRAM copy exists; `exclusive`
    /// releases the pool slot once the H2D lands and copies evictions back
    /// into the pool (or drops them when a RAM victim scores higher). Needs
    /// the RAM expert pool. Exclusive is opt-in until the inclusive-vs-
    /// exclusive A/B proves it on decode tok/s.
    #[arg(long, value_name = "MODE")]
    pub residency: Option<Residency>,
    /// Prefill layer streaming (design §7.6): prompt chunks of
    /// `--prefill-ubatch` tokens stream each expert plane through
    /// double-buffered device slabs instead of paging misses into the pool.
    /// Needs the RAM expert pool. Auto-disabled with a warning when the
    /// stream slabs cannot fit alongside a K ≥ top-k device cache.
    #[arg(long)]
    pub prefill_stream: bool,
    /// Streamed prefill ubatch (`EngineConfig::prefill_ubatch`); the
    /// context's n_ubatch (and compute buffer) when `--prefill-stream` is
    /// on. Default 2048 (§7.6).
    #[arg(long, value_name = "TOKENS", value_parser = clap::value_parser!(u32).range(1..))]
    pub prefill_ubatch: Option<u32>,
    /// Minimum prompt-tail length that still streams (`EngineConfig::
    /// prefill_stream_min`); shorter tails take the per-token path.
    #[arg(long, value_name = "TOKENS", value_parser = clap::value_parser!(u32).range(1..))]
    pub prefill_stream_min: Option<u32>,
    /// Routed-share switch above which a streamed plane fills whole instead
    /// of routed-only (hidden measurement knob).
    #[arg(long, hide = true)]
    pub prefill_stream_routed_max: Option<f32>,
    /// State-snapshot arena in GiB (§7.1: turn N+1 restores turn N's prompt
    /// state and prefills only the delta). 0 disables snapshots; omitted =
    /// `max(2 GiB, 2 × floor)`, shrunk toward the floor (one full-context
    /// snapshot per slot) when RAM is short. Below the floor the load is
    /// refused (`golem_plan_infeasible`).
    #[arg(long, value_name = "GIB", value_parser = parse_gib)]
    pub snapshot_ram_gib: Option<f64>,
    /// R-15 CPU pinning: `auto` (default) pins the decode thread — and with
    /// it ggml's threadpool — to the V-cache CCD and the I/O threads to the
    /// other CCD; a no-op with one CCD. `off` leaves affinity alone.
    #[arg(long, value_name = "MODE")]
    pub pin: Option<Pin>,
    /// R-8: full SHA-256 of every shard at start instead of the sampled
    /// check (needs a manifest).
    #[arg(long)]
    pub full_verify: bool,
}

/// `--pin` values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pin {
    /// Pin when a V-cache CCD exists.
    Auto,
    /// Never pin.
    Off,
}

impl std::str::FromStr for Pin {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "auto" => Ok(Self::Auto),
            "off" => Ok(Self::Off),
            _ => Err(format!("want auto or off, got {s:?}")),
        }
    }
}

impl EngineArgs {
    /// Maps the CLI surface onto `EngineConfig` (flag name ↔ config field:
    /// `--ctx` → `n_ctx`, `--flash-attn` → `flash_attn_on`, the rest by
    /// name), each flag resolved flag > env `GOLEM_<FLAG>` > `golem.toml`
    /// > profile > engine default (§9.4).
    pub fn to_config(
        &self,
        model: PathBuf,
        l: &crate::config::Layers<'_>,
    ) -> Result<EngineConfig, crate::config::ConfigError> {
        let d = EngineConfig::default();
        let flag = |b: bool| b.then_some(true);
        let snapshot_ram_gib: Option<f64> = l.opt(self.snapshot_ram_gib, "snapshot_ram_gib")?;
        let pin = l.get(self.pin, "pin", Pin::Auto)?;
        let (decode_cpuset, io_cpuset) = match pin {
            Pin::Auto => golem_store::sys::detect_vcache_split()
                .map_or((None, None), |(v, rest)| (Some(v), Some(rest))),
            Pin::Off => (None, None),
        };
        Ok(EngineConfig {
            model_path: model,
            n_ctx: l.get(self.ctx, "ctx", d.n_ctx)?,
            n_predict: l.get(self.n_predict, "n_predict", d.n_predict)?,
            threads: l.get(self.threads, "threads", d.threads)?,
            n_gpu_layers: l.get(self.n_gpu_layers, "n_gpu_layers", 0)?,
            flash_attn_on: l.get(flag(self.flash_attn), "flash_attn", false)?,
            cpu_moe_layers: l.opt(self.cpu_moe_layers, "cpu_moe_layers")?,
            // Fractional GiB floor to the engine's whole-GiB unit; a value in
            // (0, 1) therefore disables the pool — the help text says so.
            ram_experts_gib: l
                .get(self.ram_experts_gib, "ram_experts_gib", 0.0f64)?
                .max(0.0) as u64,
            device_cache_experts: l.get(self.device_cache_experts, "device_cache_experts", 0)?,
            io_threads: l.opt(self.io_threads, "io_threads")?,
            io_depth: l.get(self.io_depth, "io_depth", d.io_depth)?,
            numa_interleave: l.get(flag(self.numa_interleave), "numa_interleave", false)?,
            o_direct: l.get(flag(self.o_direct), "o_direct", false)?,
            async_h2d: l.get(flag(self.async_h2d), "async_h2d", false)?,
            vram_experts_frac: l.get(
                self.vram_experts_frac,
                "vram_experts_frac",
                d.vram_experts_frac,
            )?,
            manifest_path: l.opt(self.manifest.clone(), "manifest")?,
            timing_csv: l.opt(self.timing_csv.clone(), "timing_csv")?,
            routing_trace: l.opt(self.routing_trace.clone(), "routing_trace")?,
            vram_order: l.get(self.vram_order, "vram_order", d.vram_order)?,
            prefetch: l.get(self.prefetch, "prefetch", d.prefetch)?,
            prefetch_lookahead: l.get(self.prefetch_lookahead, "prefetch_lookahead", 1)?,
            residency: l.get(self.residency, "residency", d.residency)?,
            prefill_stream: l.get(flag(self.prefill_stream), "prefill_stream", false)?,
            prefill_ubatch: l.get(self.prefill_ubatch, "prefill_ubatch", d.prefill_ubatch)?,
            prefill_stream_min: l.get(
                self.prefill_stream_min,
                "prefill_stream_min",
                d.prefill_stream_min,
            )?,
            prefill_stream_routed_max: l.get(
                self.prefill_stream_routed_max,
                "prefill_stream_routed_max",
                d.prefill_stream_routed_max,
            )?,
            snapshot_ram_bytes: snapshot_ram_gib.map(|g| (g * (1u64 << 30) as f64) as u64),
            full_verify: l.get(flag(self.full_verify), "full_verify", false)?,
            decode_cpuset,
            io_cpuset,
            pin_auto: pin == Pin::Auto,
            ..d
        })
    }
}

/// A configuration the §10 RAM arithmetic cannot fit; surfaced as
/// `golem_plan_infeasible`.
#[derive(Debug)]
pub struct PlanInfeasible(pub String);

impl std::fmt::Display for PlanInfeasible {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for PlanInfeasible {}

/// Resolves the engine flags for `model`: §9.4 precedence, §9 inference
/// (printed), flag cross-checks, the §10 RAM fit and the mlock preflight —
/// shared by `run` and `serve` (`parallel` = decode slots, sizing the
/// snapshot floor). Also returns the requested (pre-fit) snapshot bytes,
/// the registry's config identity.
pub fn engine_config(
    engine: &EngineArgs,
    model: &Path,
    parallel: u32,
    layers: &crate::config::Layers<'_>,
    ram_free: u64,
) -> Result<(EngineConfig, Option<u64>), Box<dyn Error>> {
    let mut config = engine.to_config(model.to_path_buf(), layers)?;
    let requested_snapshot = config.snapshot_ram_bytes;
    config.parallel = parallel;
    // §9 inference: explicit flags already won in `to_config`; fill the rest
    // from the model folder and the compiled backend.
    let inferred = infer_defaults(&mut config, engine.n_gpu_layers);
    if !inferred.is_empty() {
        eprintln!(
            "golem: inferred {}",
            inferred
                .iter()
                .map(|name| match *name {
                    "manifest" => format!(
                        "manifest={}",
                        config
                            .manifest_path
                            .as_deref()
                            .unwrap_or(Path::new(""))
                            .display()
                    ),
                    "n-gpu-layers" => format!("n-gpu-layers={}", config.n_gpu_layers),
                    _ => format!("ram-experts-gib={}", config.ram_experts_gib),
                })
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    // §3.4 P3: the store path only exists behind the RAM expert pool; a
    // --manifest without the pool would silently decode off the plain GGUF
    // path while the user believes replicas are serving experts.
    if config.prefetch != PrefetchMode::Off && config.ram_experts_gib == 0 {
        return Err(
            "--prefetch requires --ram-experts-gib > 0: predictions steer the expert pool".into(),
        );
    }
    // Prefill stream (§7.6) drives the pool's byte sources and the device
    // slabs; without the pool there is nothing to stream from.
    if config.prefill_stream && config.ram_experts_gib == 0 {
        return Err(
            "--prefill-stream requires --ram-experts-gib > 0: streamed planes lease \
             pool slots and stage misses in the pool's store-backed reader"
                .into(),
        );
    }
    if config.manifest_path.is_some() && config.ram_experts_gib == 0 {
        return Err(
            "--manifest requires --ram-experts-gib > 0: the expert pool \
             is the only consumer of the store-backed source"
                .into(),
        );
    }
    let facts = golem_plan::ModelFacts::from_gguf(model)?;
    let (snap, pool) = fit_ram(&facts, ram_free, &config)?;
    if pool != config.ram_experts_gib {
        eprintln!(
            "golem: RAM short: expert pool shrunk {} -> {pool} GiB to keep {snap} B of state snapshots",
            config.ram_experts_gib
        );
        config.ram_experts_gib = pool;
    }
    config.snapshot_ram_bytes = Some(snap);
    config.model_facts = Some(facts);
    preflight_mlock(
        model,
        config.manifest_path.as_deref(),
        config.ram_experts_gib,
        snap,
    )?;
    Ok((config, requested_snapshot))
}

/// §10 RAM admission, one rule: `ram_pool + snapshots + host_overhead +
/// staging_ring ≤ ram_free − 4 GiB`. Returns `(snapshot bytes, pool GiB)`.
/// Default snapshots `max(2 GiB, 2 × floor)` shrink to the floor first, then
/// the pool shrinks; below the floor the plan is infeasible (never clamped).
/// Target: 83 − 4 = 79 GiB; pool 63 + staging 3.7 + host 2 = 68.7 → 10.3 GiB
/// left ≥ default 4.0 GiB (floor 2.0 GiB at 100k ctx, parallel 1).
fn fit_ram(
    facts: &golem_plan::ModelFacts,
    ram_free: u64,
    config: &EngineConfig,
) -> Result<(u64, u64), PlanInfeasible> {
    const GIB: u64 = 1 << 30;
    let budget = ram_free.saturating_sub(4 * GIB);
    let input = golem_plan::PlanInput {
        ctx: config.n_ctx,
        parallel: config.parallel,
        prefill_stream: config.prefill_stream,
        ram_experts: config.ram_experts_gib * GIB,
        ..Default::default()
    };
    let mach = golem_plan::Machine {
        vram_total: 0,
        ram_total: 0,
        ram_free,
        rocm_overhead: 0,
    };
    // Everything but the pool and snapshots (host overhead, staging ring).
    let fixed: u64 = golem_plan::ram_terms(facts, &mach, &input)
        .iter()
        .filter(|t| t.name != "ram_pool" && t.name != "state_snapshots")
        .map(|t| t.bytes)
        .sum();
    let pool = config.ram_experts_gib * GIB;
    let (floor, default) = golem_plan::snapshot_arena(facts, config.parallel, config.n_ctx);
    let fits = |snap: u64, pool: u64| fixed + pool + snap <= budget;
    let want = match config.snapshot_ram_bytes {
        Some(n) if n > 0 && n < floor => {
            return Err(PlanInfeasible(format!(
                "--snapshot-ram-gib is {n} B, below the floor {floor} B (parallel {} × one \
                 {}-token snapshot); raise it or pass 0 to disable snapshots",
                config.parallel, config.n_ctx
            )));
        }
        Some(n) => n,
        None => default,
    };
    if fits(want, pool) {
        return Ok((want, config.ram_experts_gib));
    }
    // Only the planner default shrinks (toward the floor); an explicit size
    // is the user's call. Then the pool gives way.
    let snap = if config.snapshot_ram_bytes.is_none() {
        if fits(floor, pool) {
            return Ok((budget - fixed - pool, config.ram_experts_gib));
        }
        floor
    } else {
        want
    };
    let pool_room = budget.saturating_sub(fixed + snap) / GIB;
    if config.ram_experts_gib > 0 && pool_room >= 1 {
        return Ok((snap, pool_room.min(config.ram_experts_gib)));
    }
    let per_slot = budget.saturating_sub(fixed + pool) / u64::from(config.parallel.max(1));
    let max_ctx = per_slot.saturating_sub(facts.rs_bytes) / facts.kv_per_token.max(1);
    Err(PlanInfeasible(format!(
        "RAM: {snap} B of state snapshots + {pool} B expert pool + {fixed} B host/staging \
         exceed {budget} B (available {ram_free} B − 4 GiB headroom); largest --ctx that \
         fits: {max_ctx} (or shrink --ram-experts-gib)"
    )))
}

/// Finite, non-negative GiB (clap parser).
fn parse_gib(s: &str) -> Result<f64, String> {
    match s.parse::<f64>() {
        Ok(g) if g.is_finite() && g >= 0.0 => Ok(g),
        _ => Err(format!("expected a finite GiB value >= 0, got {s:?}")),
    }
}

/// Runs one greedy `run` invocation: loads the model, then either prints a
/// single completion (`--prompt`) or serves a multi-turn chat REPL
/// on stdin until EOF. Returns `Err` for load/init failures and one-shot
/// decode errors (process unusable); per-turn REPL errors are printed and the
/// loop continues.
pub fn run(args: &RunArgs) -> Result<(), Box<dyn Error>> {
    let model = match args.model.clone() {
        Some(m) => m,
        // No model: main() already printed the run help.
        None => return Ok(()),
    };
    let layers = crate::config::Layers::from_process()?
        .with_profile(&model, args.engine.no_optimize_profile);
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let ram_free = if args.allow_overcommit {
        u64::MAX
    } else {
        sys.available_memory()
    };
    let (config, requested_snapshot) = engine_config(&args.engine, &model, 1, &layers, ram_free)?;
    let eff = crate::serve::effective(&config, requested_snapshot)?;
    let reg = golem_registry::Registry::open()?;
    if !args.new && args.parity_dump.is_none() {
        // Same layers as `serve`: the key a `--keep` server was started
        // with, or the operator's configured key for an existing one.
        let api_key: Option<String> = layers.opt(None, "api_key")?;
        if let Some(i) = attachable(&reg, &eff)? {
            eprintln!("golem: attached to instance {} ({})", i.id, i.endpoint);
            return attach(
                &i,
                args.prompt.as_deref(),
                config.n_predict,
                api_key.as_deref(),
            );
        }
        if args.keep {
            let (id, endpoint) = start_detached()?;
            eprintln!("golem: started instance {id} ({endpoint})");
            let i = reg.find(&id)?;
            return attach(
                &i,
                args.prompt.as_deref(),
                config.n_predict,
                api_key.as_deref(),
            );
        }
    }
    let model_name = model
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let guard = crate::serve::register(
        &reg,
        crate::serve::new_id(),
        "run",
        model_name,
        &config,
        &eff,
        args.allow_overcommit,
    )?;
    // §11 CSV lifecycle: truncate + header here, rows appended per turn end.
    let mut timing_out = match &config.timing_csv {
        Some(path) => {
            let mut f = std::fs::File::create(path)?;
            TimingRecorder::header(
                &mut f,
                env!("CARGO_PKG_VERSION"),
                &fork_commit(),
                &resolved_config_line(&config),
            )?;
            f.flush()?;
            Some(f)
        }
        None => None,
    };
    // Routing trace (§2.3 step 4): same turn-end flush lifecycle as the CSV.
    let mut trace_out = match &config.routing_trace {
        Some(path) => Some(std::fs::File::create(path)?),
        None => None,
    };
    let mut session = Session::new(&config)?;
    guard.update(|i| i.health = "healthy".into());
    if let Some(k) = session.device_cache_experts() {
        eprintln!("golem: device cache {k} expert slots per layer");
    }

    let result = match (&args.prompt, &args.parity_dump) {
        (Some(prompt), Some(dump)) => {
            one_shot_parity(&mut session, &model, prompt, dump, config.n_predict)
        }
        (Some(prompt), None) => one_shot(&mut session, prompt, &mut timing_out, &mut trace_out),
        // The dump records exactly one prompt (header `prompt_tokens`), so a
        // multi-turn REPL dump would be ill-defined — refuse it up front.
        (None, Some(_)) => Err("--parity-dump requires --prompt".into()),
        (None, None) => repl(&mut session, &mut timing_out, &mut trace_out),
    };
    // §6.5 metrics for the whole run: stderr, and the timing CSV's trailer.
    if let Some(stats) = session
        .prefetch_stats()
        .filter(|s| s.mode != PrefetchMode::Off)
    {
        eprintln!("golem: {stats}");
        // A trailer write failure must not mask the run's own result.
        if let Some(Err(e)) = timing_out
            .as_mut()
            .map(|f| writeln!(f, "# {stats}").and_then(|()| f.flush()))
        {
            eprintln!("golem: timing CSV trailer: {e}");
        }
    }
    result
}

/// §9 defaults: infer what the model folder and the compiled backend already
/// decide — explicit flags always win (`to_config` ran first). Order matters:
/// the manifest is located before the pool is sized, so the size can derive
/// from the manifest's expert bytes. Returns the names of what was inferred.
fn infer_defaults(
    config: &mut EngineConfig,
    explicit_gpu_layers: Option<u32>,
) -> Vec<&'static str> {
    let mut inferred = Vec::new();
    if config.manifest_path.is_none()
        && let Some(dir) = config.model_path.parent()
    {
        let m = dir.join("manifest.json");
        if m.is_file() {
            config.manifest_path = Some(m);
            inferred.push("manifest");
        }
    }
    if explicit_gpu_layers.is_none()
        && config.n_gpu_layers == 0
        && substrate_backend() == Backend::Hip
    {
        config.n_gpu_layers = 99;
        inferred.push("n-gpu-layers");
    }
    if config.ram_experts_gib == 0 && config.manifest_path.is_some() {
        match suggest_pool_gib(&config.model_path, config.manifest_path.as_deref()) {
            Some(gib) => {
                config.ram_experts_gib = gib;
                inferred.push("ram-experts-gib");
            }
            // No workable pool size → the manifest would hard-error against
            // the pool requirement while the plain GGUF path would have
            // worked: drop the inferred manifest, keep the run going.
            None => {
                config.manifest_path = None;
                inferred.retain(|&name| name != "manifest");
            }
        }
    }
    inferred
}

/// Workable pool size for a manifest-bearing run: the preflight's own cap,
/// min(RLIMIT_MEMLOCK, 75% available RAM, expert bytes), floored to GiB.
/// `None` → no workable pool; the caller proceeds without one (the manifest
/// then stays unused and the run decodes off the plain GGUF path).
fn suggest_pool_gib(model: &Path, manifest: Option<&Path>) -> Option<u64> {
    let expert_bytes = match expert_plane_bytes(model, manifest) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("warning: cannot pre-scan expert tensors from {model:?}: {e}");
            return None;
        }
    };
    if expert_bytes == 0 {
        return None;
    }
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let mut cap = sys.available_memory() * 3 / 4;
    if let Some(limit) = memlock_limit() {
        cap = cap.min(limit);
    }
    let gib = cap.min(expert_bytes) / (1 << 30);
    (gib > 0).then_some(gib)
}

/// `RLIMIT_MEMLOCK` bytes, `None` = unlimited.
fn memlock_limit() -> Option<u64> {
    // SAFETY: plain getrlimit(2) with an out-pointer; no invariants.
    let mut rl = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    if unsafe { libc::getrlimit(libc::RLIMIT_MEMLOCK, &mut rl) } != 0 {
        return None; // cannot read the limit — let the arena backstop decide
    }
    (rl.rlim_cur != libc::RLIM_INFINITY).then_some(rl.rlim_cur)
}

/// mlock preflight, before any weights load: refuse to start when the expert
/// pool plus the snapshot arena cannot be locked (`RLIMIT_MEMLOCK` too low),
/// with an actionable suggestion (§5.4: mlock is a correctness feature —
/// never fall back). RAM fit is `fit_ram`'s single rule.
fn preflight_mlock(
    model: &Path,
    manifest: Option<&Path>,
    gib: u64,
    snapshot_bytes: u64,
) -> Result<(), Box<dyn Error>> {
    let want_bytes = gib
        .checked_mul(1 << 30)
        .and_then(|b| b.checked_add(snapshot_bytes))
        .ok_or("--ram-experts-gib overflows")?;
    if let Some(limit) = memlock_limit()
        && want_bytes > limit
    {
        // .max(1): a sub-GiB suggestion would print `--ram-experts-gib 0`,
        // which disables the pool — suggest the smallest workable pool.
        let suggest_gib = suggest_pool_gib(model, manifest).unwrap_or(1).max(1);
        return Err(format!(
            "--ram-experts-gib {gib} + {snapshot_bytes} B of state snapshots need \
                 {want_bytes} B locked but RLIMIT_MEMLOCK is {limit} B.\nRaise it: \
                 `ulimit -l unlimited` (or systemd `LimitMEMLOCK=infinity`), or size down, \
                 e.g. --ram-experts-gib {suggest_gib} / --snapshot-ram-gib 0"
        )
        .into());
    }
    Ok(())
}

/// Total pageable expert bytes: sum of `stride × n_experts` over the model's
/// expert planes (dense lead excluded), via the same GGUF walk the pool uses.
fn expert_plane_bytes(model: &Path, manifest: Option<&Path>) -> Result<u64, Box<dyn Error>> {
    // Store-backed run: the manifest's expert_index is the source of truth
    // (already cross-checked against the replica tensor tables at open).
    if let Some(mpath) = manifest {
        let queue = Arc::new(ChunkQueue::new(0, None));
        let src = StoreExpertSource::open(mpath, queue)?;
        return Ok(src.expert_bytes());
    }
    let metadata = read_metadata(model)?;
    let arch = metadata
        .kv_str("general.architecture")
        .unwrap_or_default()
        .to_owned();
    let dense_lead = u32::try_from(
        metadata
            .kv_u64(&format!("{arch}.leading_dense_block_count"))
            .unwrap_or(0),
    )?;
    let src = GgufPreadSource::open(model, dense_lead)?;
    Ok(src.geometry()?.map(|g| g.expert_bytes).unwrap_or(0))
}

/// Vendored llama.cpp fork commit for the CSV header comment: read from the
/// checkout's `vendor/VENDOR_SHA` (`commit=` line) so the header carries the
/// real fork; `GOLEM_FORK_COMMIT` env overrides (e.g. for out-of-tree builds
/// without `vendor/`).
fn fork_commit() -> String {
    if let Some(c) = option_env!("GOLEM_FORK_COMMIT") {
        return c.to_owned();
    }
    const VENDOR_SHA: &str = include_str!("../../../vendor/VENDOR_SHA");
    VENDOR_SHA
        .lines()
        .find_map(|l| l.strip_prefix("commit="))
        .unwrap_or("unknown")
        .trim()
        .to_owned()
}

fn one_shot(
    session: &mut Session,
    prompt: &str,
    timing_out: &mut Option<std::fs::File>,
    trace_out: &mut Option<std::fs::File>,
) -> Result<(), Box<dyn Error>> {
    session.chat_turn(
        &[ChatTurn {
            role: "user",
            content: prompt,
        }],
        &mut emit_stdout,
    )?;
    // Timing CSV is best-effort post-streaming: log, don't fail the run
    // (matches the REPL path).
    if let Err(e) = flush_timing(session, timing_out) {
        eprintln!("error: timing csv: {e}");
    }
    if let Err(e) = flush_trace(session, trace_out) {
        eprintln!("error: routing trace: {e}");
    }
    println!();
    Ok(())
}

/// §11: appends the turn's recorded rows to the CSV file (the recorder
/// survives the flush; it is reset by the engine at the next turn start).
fn flush_timing(
    session: &mut Session,
    timing_out: &mut Option<std::fs::File>,
) -> Result<(), Box<dyn Error>> {
    if let Some(f) = timing_out.as_mut() {
        session.write_timing(f)?;
        f.flush()?;
    }
    Ok(())
}

/// Flushes the turn's buffered routing-trace records (ring resets for the
/// next turn). Best-effort like the CSV.
fn flush_trace(
    session: &mut Session,
    trace_out: &mut Option<std::fs::File>,
) -> Result<(), Box<dyn Error>> {
    if let Some(f) = trace_out.as_mut() {
        session.write_routing_trace(f)?;
        f.flush()?;
    }
    Ok(())
}

/// One-line resolved-config string for the CSV header comment (§11).
fn resolved_config_line(config: &EngineConfig) -> String {
    format!(
        "ctx={} batch={} predict={} threads={} gpu_layers={} fa={} cpu_moe={:?} \
         ram_experts_gib={} device_cache_experts={} io_threads={:?} manifest={:?} \
         vram_order={:?} prefetch={:?} prefetch_lookahead={} model_size={}",
        config.n_ctx,
        config.batch_tokens,
        config.n_predict,
        config.threads,
        config.n_gpu_layers,
        config.flash_attn_on,
        config.cpu_moe_layers,
        config.ram_experts_gib,
        config.device_cache_experts,
        config.io_threads,
        config.manifest_path,
        config.vram_order,
        config.prefetch,
        config.prefetch_lookahead,
        std::fs::metadata(&config.model_path)
            .map(|m| m.len())
            .unwrap_or(0),
    )
}

/// `--prompt --parity-dump <FILE>`: one parity turn (same greedy decode as
/// `chat_turn`, per-step token + top-2 logits recorded, no text streamed),
/// then the JSONL dump written atomically.
fn one_shot_parity(
    session: &mut Session,
    model_path: &Path,
    prompt: &str,
    dump_path: &Path,
    n_predict: u32,
) -> Result<(), Box<dyn Error>> {
    let messages = [ChatTurn {
        role: "user",
        content: prompt,
    }];
    let mut steps = Vec::new();
    session.chat_turn_parity(&messages, &mut steps)?;
    let prompt_tokens = u64::try_from(session.last_prompt_tokens().len()).unwrap_or(u64::MAX);
    let offloaded = session.offloaded();

    // Hash the model file after the turn: the engine has already proven the
    // file loads, and multi-GiB hashing is worth skipping on failed runs.
    let (model_sha256, model_bytes) = model_identity(model_path)?;
    let header = ParityDumpHeader {
        format: "golem-parity-dump",
        version: 1,
        model_sha256: &model_sha256,
        model_bytes,
        backend: substrate_backend().as_str(),
        offload: offloaded,
        prompt_tokens,
        n_predict,
    };
    write_atomic(dump_path, &render_parity_dump(&header, &steps))?;
    eprintln!(
        "parity dump: {} steps, {prompt_tokens} prompt tokens -> {}",
        steps.len(),
        dump_path.display()
    );
    Ok(())
}

/// Line 1 of a parity dump. Field order is the serialization order and part of
/// the frozen dump contract (the header is deliberately not key-sorted).
/// `offload` records the session's offload intent (`n_gpu_layers > 0`) so a
/// "hip"-labeled leg that never asked for device execution is detectable
/// alongside the substrate `backend` field.
#[derive(Serialize)]
struct ParityDumpHeader<'a> {
    format: &'static str,
    version: u32,
    model_sha256: &'a str,
    model_bytes: u64,
    backend: &'a str,
    offload: bool,
    prompt_tokens: u64,
    n_predict: u32,
}

/// Renders the dump body: line-1 header, then one record per step, each a
/// compact single-line JSON object in the frozen field order (the record
/// type is `golem_engine::parity::ParityStep`, serialized in place — no
/// per-writer copy struct).
fn render_parity_dump(header: &ParityDumpHeader<'_>, steps: &[ParityStep]) -> String {
    let mut out = String::with_capacity(96 + 96 * steps.len());
    // Plain string/number fields — serialization cannot fail.
    out.push_str(&serde_json::to_string(header).expect("header serialization cannot fail"));
    out.push('\n');
    for step in steps {
        out.push_str(&serde_json::to_string(step).expect("step serialization cannot fail"));
        out.push('\n');
    }
    out
}

/// Chunk size for streaming the model file through sha256 (models are
/// multi-GiB; never hold one in memory).
const HASH_CHUNK_BYTES: usize = 1024 * 1024;

/// Streams `path` through sha256 in chunks; returns the lowercase hex digest
/// and byte size — the only model identity a dump carries (path-scrub policy).
fn model_identity(path: &Path) -> std::io::Result<(String, u64)> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; HASH_CHUNK_BYTES];
    let mut total: u64 = 0;
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        total += n as u64;
    }
    let hex: String = hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Ok((hex, total))
}

/// Stdin chat REPL. Each turn re-sends the whole conversation; the engine
/// restores the previous prompt's state snapshot (§7.1) and prefills only
/// the delta. `/reset` drops it, `/exit` ends the session.
fn repl(
    session: &mut Session,
    timing_out: &mut Option<std::fs::File>,
    trace_out: &mut Option<std::fs::File>,
) -> Result<(), Box<dyn Error>> {
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    // (role, content) in send order; owned because each turn borrows it all.
    let mut history: Vec<(&'static str, String)> = Vec::new();
    eprintln!("golem: multi-turn chat; /reset clears history, /exit quits");
    loop {
        eprint!("golem> ");
        let _ = std::io::stderr().flush();
        let line = match lines.next() {
            Some(Ok(line)) => line,
            Some(Err(e)) => return Err(e.into()),
            None => return Ok(()),
        };
        match line.trim() {
            "" => continue,
            "/exit" | "/quit" => return Ok(()),
            "/reset" => {
                history.clear();
                eprintln!("golem: history cleared");
                continue;
            }
            _ => {}
        }
        history.push(("user", line));
        let turns: Vec<ChatTurn<'_>> = history
            .iter()
            .map(|(role, content)| ChatTurn {
                role,
                content: content.as_str(),
            })
            .collect();
        let mut reply = String::new();
        let mut emit = |piece: &str| {
            reply.push_str(piece);
            emit_stdout(piece);
        };
        match session.chat_turn(&turns, &mut emit) {
            Ok(_) => {
                history.push(("assistant", reply));
                if let Err(e) = flush_timing(session, timing_out) {
                    eprintln!("error: timing csv: {e}");
                }
                if let Err(e) = flush_trace(session, trace_out) {
                    eprintln!("error: routing trace: {e}");
                }
                println!()
            }
            Err(e) => {
                // Drop the failed turn: replaying a prompt that already
                // overran the context would fail the same way every turn.
                history.pop();
                println!();
                eprintln!("error: {e}");
            }
        }
    }
}

/// A live serving instance this run can attach to (§9.4): `config_hash`
/// fast path, else the [`golem_registry::compatible`] check; a same-model
/// conflict is printed with the `--new` hint.
fn attachable(
    reg: &golem_registry::Registry,
    want: &golem_registry::EffectiveConfig,
) -> Result<Option<golem_registry::Instance>, Box<dyn Error>> {
    if let Some(i) = reg.find_compatible(&want.hash())? {
        return Ok(Some(i));
    }
    for i in reg.list_live()? {
        let Some(running) = &i.config else { continue };
        if i.kind != "serve" || !matches!(i.health.as_str(), "healthy" | "degraded") {
            continue;
        }
        match golem_registry::compatible(running, want) {
            Ok(()) => return Ok(Some(i)),
            Err(why) if running.model_path == want.model_path => eprintln!(
                "golem: instance {} serves this model but cannot attach: {why}; \
                 starting in-process (pass --new to skip this check)",
                i.id
            ),
            Err(_) => {}
        }
    }
    Ok(None)
}

/// `golem serve --detach` with this run's model and engine flags (run-only
/// flags dropped); returns `(id, endpoint)` from its ready line.
fn start_detached() -> Result<(String, String), Box<dyn Error>> {
    const RUN_ONLY: [&str; 3] = ["--keep", "--new", "--greedy"];
    const RUN_ONLY_VALUE: [&str; 2] = ["--prompt", "--parity-dump"];
    let mut argv = std::env::args().skip(1).peekable();
    let mut fwd = Vec::new();
    while let Some(a) = argv.next() {
        let name = a.split('=').next().unwrap_or_default();
        if fwd.is_empty() && a == "run" {
            fwd.extend(["serve".to_owned(), "--detach".to_owned()]);
        } else if RUN_ONLY_VALUE.contains(&name) {
            if !a.contains('=') {
                argv.next();
            }
        } else if !RUN_ONLY.contains(&name) {
            fwd.push(a);
        }
    }
    let out = std::process::Command::new(std::env::current_exe()?)
        .args(&fwd)
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::inherit())
        .output()?;
    if !out.status.success() {
        return Err("detached server failed to start".into());
    }
    let line = String::from_utf8_lossy(&out.stdout);
    let mut w = line.split_whitespace();
    match (w.next(), w.next()) {
        (Some(id), Some(ep)) => Ok((id.to_owned(), ep.to_owned())),
        _ => Err(format!("unexpected serve --detach output: {line}").into()),
    }
}

/// Chat client over a running instance: `--prompt` one-shot, else the stdin
/// REPL, streaming each reply from the SSE endpoint.
fn attach(
    i: &golem_registry::Instance,
    prompt: Option<&str>,
    n_predict: u32,
    api_key: Option<&str>,
) -> Result<(), Box<dyn Error>> {
    let addr = i
        .endpoint
        .strip_prefix("http://")
        .ok_or_else(|| format!("instance {} has no HTTP endpoint", i.id))?;
    let mut history: Vec<serde_json::Value> = Vec::new();
    let turn = |history: &mut Vec<serde_json::Value>, text: String| {
        history.push(serde_json::json!({"role": "user", "content": text}));
        let reply = stream_chat(addr, &i.model, history, n_predict, api_key)?;
        history.push(serde_json::json!({"role": "assistant", "content": reply}));
        println!();
        Ok::<_, Box<dyn Error>>(())
    };
    if let Some(p) = prompt {
        return turn(&mut history, p.to_owned());
    }
    eprintln!("golem: multi-turn chat; /reset clears history, /exit quits");
    for line in std::io::stdin().lock().lines() {
        let line = line?;
        match line.trim() {
            "" => continue,
            "/exit" | "/quit" => break,
            "/reset" => {
                history.clear();
                eprintln!("golem: history cleared");
            }
            _ => {
                if let Err(e) = turn(&mut history, line) {
                    history.pop();
                    eprintln!("error: {e}");
                }
            }
        }
    }
    Ok(())
}

/// One streamed chat completion; prints content deltas as they arrive and
/// returns the full reply. HTTP/1.0 keeps the body un-chunked and
/// close-delimited, so SSE lines arrive intact.
fn stream_chat(
    addr: &str,
    model: &str,
    messages: &[serde_json::Value],
    n_predict: u32,
    api_key: Option<&str>,
) -> Result<String, Box<dyn Error>> {
    // `run` is greedy; the server's OpenAI default is temperature 1.
    let body = serde_json::json!({
        "model": model, "messages": messages, "stream": true, "max_tokens": n_predict,
        "temperature": 0,
    })
    .to_string();
    let auth = api_key.map_or(String::new(), |k| format!("authorization: Bearer {k}\r\n"));
    let mut s = std::net::TcpStream::connect(addr)?;
    write!(
        s,
        "POST /v1/chat/completions HTTP/1.0\r\nhost: {addr}\r\ncontent-type: application/json\r\n\
         {auth}content-length: {}\r\n\r\n{body}",
        body.len()
    )?;
    let mut lines = std::io::BufReader::new(s).lines();
    let status = lines.next().transpose()?.unwrap_or_default();
    if !status.contains(" 200 ") {
        let rest: Vec<String> = lines.map_while(Result::ok).collect();
        return Err(rest.last().cloned().unwrap_or(status).into());
    }
    let mut reply = String::new();
    for line in lines {
        let line = line?;
        let Some(data) = line.strip_prefix("data: ") else {
            continue;
        };
        if data == "[DONE]" {
            break;
        }
        let v: serde_json::Value = serde_json::from_str(data)?;
        if v.get("error").is_some() {
            return Err(data.into());
        }
        if let Some(piece) = v["choices"][0]["delta"]["content"].as_str() {
            reply.push_str(piece);
            emit_stdout(piece);
        }
    }
    Ok(reply)
}

fn emit_stdout(piece: &str) {
    print!("{piece}");
    let _ = std::io::stdout().flush();
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::{Args as _, FromArgMatches as _};
    use golem_engine::ParityStep;

    const GIB: u64 = 1 << 30;

    /// Real target facts (golem-plan's GLM-5.3-Flash fixture numbers).
    fn glm() -> golem_plan::ModelFacts {
        golem_plan::ModelFacts {
            t_ne: 8 << 30,
            shexp: 1 << 30,
            n_expert: 288,
            top_k: 8,
            expert_layers: 42,
            max_plane_stride: 6_881_280,
            expert_bytes_total: 42 * 288 * 3 * 4_718_592,
            kv_per_token: 19_712,
            rs_bytes: 152_630_000,
            sparse_attn_layers: 11,
        }
    }

    fn cfg(ctx: u32, pool_gib: u64, snap: Option<u64>) -> EngineConfig {
        EngineConfig {
            n_ctx: ctx,
            ram_experts_gib: pool_gib,
            prefill_stream: true,
            snapshot_ram_bytes: snap,
            ..EngineConfig::default()
        }
    }

    #[test]
    fn fit_ram_real_target_keeps_default_snapshots() {
        // 83 GiB free − 4 = 79; pool 63 + staging 3.7 + host 2 = 68.7 → 10.3 left.
        let floor = golem_plan::snapshot_bytes(152_630_000, 19_712, 100_000);
        let (snap, pool) = fit_ram(&glm(), 83 * GIB, &cfg(100_000, 63, None)).unwrap();
        assert_eq!((snap, pool), ((2 * GIB).max(2 * floor), 63));
    }

    #[test]
    fn fit_ram_shrinks_snapshots_then_pool_and_refuses_below_floor() {
        let floor = golem_plan::snapshot_bytes(152_630_000, 19_712, 100_000);
        // 70 GiB free: 66 budget − 68.7 fixed+pool → default can't fit; pool gives way.
        let (snap, pool) = fit_ram(&glm(), 70 * GIB, &cfg(100_000, 63, None)).unwrap();
        assert_eq!(snap, floor);
        assert!((1..63).contains(&pool));
        // Explicit size below the floor: refused, never clamped.
        assert!(fit_ram(&glm(), 83 * GIB, &cfg(100_000, 63, Some(floor - 1))).is_err());
        // No pool to shrink and no room for the floor: infeasible, suggests a ctx.
        let e = fit_ram(&glm(), 8 * GIB, &cfg(100_000, 0, None)).unwrap_err();
        assert!(e.0.contains("largest --ctx"), "{}", e.0);
        // Disabled snapshots need no floor.
        assert_eq!(
            fit_ram(&glm(), 83 * GIB, &cfg(100_000, 63, Some(0))).unwrap(),
            (0, 63)
        );
    }

    /// Known digest of `b"hello parity\n"` (computed with python3 hashlib).
    const SHA_SMALL: &str = "aec52f3eb3180f83512a00206ccd475a49683fc37607961437ae5cdf9f0dd985";
    /// Known digest of 2.5 MiB (2621440 bytes) of 0xA5 — crosses the 1 MiB
    /// chunk boundary twice, pinning the streamed-read behavior.
    const SHA_CHUNKED: &str = "15012bb905af98147b23c43561f60131e7d820a874008fc5454ca85df7d7a067";

    fn test_header<'a>(
        sha: &'a str,
        bytes: u64,
        backend: &'a str,
        offload: bool,
    ) -> ParityDumpHeader<'a> {
        ParityDumpHeader {
            format: "golem-parity-dump",
            version: 1,
            model_sha256: sha,
            model_bytes: bytes,
            backend,
            offload,
            prompt_tokens: 7,
            n_predict: 64,
        }
    }

    fn test_steps() -> Vec<ParityStep> {
        vec![
            ParityStep {
                position: 0,
                token: 11,
                top1: 5.5,
                top2: 1.25,
            },
            ParityStep {
                position: 1,
                token: -3,
                top1: 0.0,
                top2: 0.0,
            },
            ParityStep {
                position: 2,
                token: 42,
                top1: -7.75,
                top2: -9.5,
            },
        ]
    }

    fn scratch_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "golem-cli-{}-{}-{tag}",
            std::process::id(),
            std::thread::current().name().unwrap_or("t")
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn parity_dump_is_valid_jsonl() {
        let body = render_parity_dump(&test_header(SHA_SMALL, 1234, "cpu", false), &test_steps());
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 4, "1 header + 1 line per step");

        let h: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
        assert_eq!(h["format"], "golem-parity-dump");
        assert_eq!(h["version"], 1);
        assert_eq!(h["model_sha256"], SHA_SMALL);
        assert_eq!(h["model_bytes"], 1234);
        assert_eq!(h["backend"], "cpu");
        assert_eq!(h["offload"], false);
        assert_eq!(h["prompt_tokens"], 7);
        assert_eq!(h["n_predict"], 64);

        let expected = [
            (0u64, 11i64, 5.5f64, 1.25f64),
            (1, -3, 0.0, 0.0),
            (2, 42, -7.75, -9.5),
        ];
        for (line, &(position, token, top1, top2)) in lines[1..].iter().zip(&expected) {
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            assert_eq!(v["position"], position);
            assert_eq!(v["token"], token);
            assert_eq!(v["top1"], top1);
            assert_eq!(v["top2"], top2);
        }
    }

    #[test]
    fn parity_dump_field_order_is_frozen() {
        // The dump contract fixes byte-level field order (diffable legs,
        // deterministic artifacts); serde struct order = serialization order.
        let body = render_parity_dump(&test_header(SHA_SMALL, 1234, "cpu", false), &test_steps());

        let header = body.lines().next().unwrap();
        assert!(header.starts_with("{\"format\":\"golem-parity-dump\",\"version\":1,"));
        let idx = |key: &str, line: &str| line.find(key).unwrap();
        let (sha, bytes) = (
            idx("\"model_sha256\"", header),
            idx("\"model_bytes\"", header),
        );
        let (backend, offload, prompt, npredict) = (
            idx("\"backend\"", header),
            idx("\"offload\"", header),
            idx("\"prompt_tokens\"", header),
            idx("\"n_predict\"", header),
        );
        assert!(
            sha < bytes
                && bytes < backend
                && backend < offload
                && offload < prompt
                && prompt < npredict
        );

        // Step records are also key-sorted: position < token < top1 < top2.
        let step = body.lines().nth(1).unwrap();
        assert!(step.starts_with("{\"position\":"));
        let (p, t, t1, t2) = (
            idx("\"position\"", step),
            idx("\"token\"", step),
            idx("\"top1\"", step),
            idx("\"top2\"", step),
        );
        assert!(p < t && t < t1 && t1 < t2);
    }

    #[test]
    fn parity_dump_header_records_backend() {
        for backend in ["cpu", "hip"] {
            let body = render_parity_dump(&test_header(SHA_SMALL, 1, backend, false), &[]);
            let header = body.lines().next().unwrap();
            assert!(header.contains(&format!("\"backend\":\"{backend}\"")));
        }
    }

    #[test]
    fn parity_dump_header_records_offload_flag() {
        for offload in [false, true] {
            let body = render_parity_dump(&test_header(SHA_SMALL, 1, "cpu", offload), &[]);
            let header = body.lines().next().unwrap();
            assert!(header.contains(&format!("\"offload\":{offload}")));
        }
    }

    #[test]
    fn parity_dump_records_hash_and_bytes_never_paths() {
        // Path-scrub policy (session-8): model identity is sha256 + byte size,
        // never a machine-local path.
        let dir = scratch_dir("scrub");
        let model = dir.join("secret-model-name.gguf");
        std::fs::write(&model, b"weights").unwrap();

        let (sha, bytes) = model_identity(&model).unwrap();
        let body = render_parity_dump(&test_header(&sha, bytes, "cpu", false), &test_steps());
        assert!(!body.contains("secret-model-name"));
        assert!(!body.contains(".gguf"));
        assert!(!body.contains(dir.to_str().unwrap()));
        let header = body.lines().next().unwrap();
        assert!(header.contains(&format!("\"model_sha256\":\"{sha}\"")));
        assert!(header.contains("\"model_bytes\":7"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn model_identity_matches_known_digest_and_size() {
        let dir = scratch_dir("small");
        let model = dir.join("model.bin");
        std::fs::write(&model, b"hello parity\n").unwrap();
        let (sha, bytes) = model_identity(&model).unwrap();
        assert_eq!(sha, SHA_SMALL);
        assert_eq!(bytes, 13);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn model_identity_streams_beyond_one_chunk() {
        let dir = scratch_dir("chunked");
        let model = dir.join("model.bin");
        std::fs::write(&model, vec![0xA5u8; 2 * 1024 * 1024 + 512 * 1024]).unwrap();
        let (sha, bytes) = model_identity(&model).unwrap();
        assert_eq!(sha, SHA_CHUNKED);
        assert_eq!(bytes, 2_621_440);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    /// Parses a `run` flag tail through the real clap definition (augment the
    /// subcommand's args onto a bare command), so the tests pin the exact
    /// flag spellings users and the suite driver rely on.
    fn parse_run_args(argv: &[&str]) -> RunArgs {
        let matches = RunArgs::augment_args(clap::Command::new("run"))
            .try_get_matches_from(argv)
            .expect("test argv must parse");
        RunArgs::from_arg_matches(&matches).expect("matches fit RunArgs")
    }

    #[test]
    fn infer_defaults_discovers_manifest_and_keeps_explicit_flags() {
        let dir = scratch_dir("infer");
        let model = dir.join("model.gguf");
        std::fs::write(&model, b"not a gguf").unwrap();
        std::fs::write(dir.join("manifest.json"), b"{}").unwrap();

        // Bare args: manifest found next to the model, then rolled back — a
        // junk manifest carries no expert bytes, so no pool size can be
        // inferred and keeping the manifest would hard-error the run.
        let mut config = parse_run_args(&["run", "model.gguf"])
            .engine
            .to_config("model.gguf".into(), &crate::config::Layers::none())
            .unwrap();
        config.model_path = model.clone();
        let inferred = infer_defaults(&mut config, None);
        assert_eq!(
            config.manifest_path, None,
            "unusable manifest → plain GGUF path, no hard error"
        );
        assert_eq!(
            config.n_gpu_layers,
            if substrate_backend() == Backend::Hip {
                99
            } else {
                0
            },
            "HIP builds infer full offload, CPU builds stay on CPU"
        );
        assert_eq!(config.ram_experts_gib, 0, "no expert bytes → no pool");
        assert!(
            inferred == ["n-gpu-layers"] || inferred.is_empty(),
            "only the backend offload may remain inferred: {inferred:?}"
        );

        // Explicit `--n-gpu-layers 0` keeps the run CPU-only even on HIP
        // builds — the CPU-leg repro on GPU hardware must stay reachable.
        let mut config = parse_run_args(&["run", "--n-gpu-layers", "0", "model.gguf"])
            .engine
            .to_config("model.gguf".into(), &crate::config::Layers::none())
            .unwrap();
        config.model_path = model.clone();
        infer_defaults(&mut config, Some(0));
        assert_eq!(config.n_gpu_layers, 0, "explicit 0 must not be re-inferred");

        // Explicit flags win: no manifest file in the folder, no inference.
        let mut config = parse_run_args(&[
            "run",
            "--n-gpu-layers",
            "7",
            "--ram-experts-gib",
            "3",
            "--manifest",
            "/elsewhere/manifest.json",
            "model.gguf",
        ])
        .engine
        .to_config("model.gguf".into(), &crate::config::Layers::none())
        .unwrap();
        config.model_path = model;
        infer_defaults(&mut config, Some(7));
        assert_eq!(config.n_gpu_layers, 7);
        assert_eq!(config.ram_experts_gib, 3);
        assert_eq!(
            config.manifest_path,
            Some(PathBuf::from("/elsewhere/manifest.json"))
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn run_args_default_to_the_cpu_only_config() {
        let config = parse_run_args(&["run", "model.gguf"])
            .engine
            .to_config("model.gguf".into(), &crate::config::Layers::none())
            .unwrap();
        assert_eq!(config.model_path, PathBuf::from("model.gguf"));
        assert_eq!(config.n_ctx, 256);
        assert_eq!(config.n_predict, 64);
        // No --threads flag → the engine default (logical CPUs / 2, min 1),
        // never the old hardcoded 1 that overrode it.
        assert_eq!(
            config.threads,
            EngineConfig::default().threads,
            "no flag → engine default threads"
        );
        assert_eq!(config.n_gpu_layers, 0, "default offloads nothing");
        assert!(
            !config.flash_attn_on,
            "default leaves flash attention at AUTO"
        );
        assert_eq!(
            config.cpu_moe_layers, None,
            "default writes no buft overrides"
        );
        assert_eq!(config.ram_experts_gib, 0, "pool off by default");
    }

    #[test]
    fn run_args_map_ram_experts_gib_flag_into_config() {
        let config = parse_run_args(&["run", "m.gguf", "--ram-experts-gib", "2"])
            .engine
            .to_config("model.gguf".into(), &crate::config::Layers::none())
            .unwrap();
        assert_eq!(config.ram_experts_gib, 2);
        // Fractional GiB floor to the engine's whole-GiB unit.
        let config = parse_run_args(&["run", "m.gguf", "--ram-experts-gib", "2.75"])
            .engine
            .to_config("model.gguf".into(), &crate::config::Layers::none())
            .unwrap();
        assert_eq!(config.ram_experts_gib, 2);
    }

    #[test]
    fn run_args_map_device_cache_experts_flag_into_config() {
        let config = parse_run_args(&["run", "m.gguf"])
            .engine
            .to_config("model.gguf".into(), &crate::config::Layers::none())
            .unwrap();
        assert_eq!(config.device_cache_experts, 0, "unset = derive from VRAM");
        let config = parse_run_args(&["run", "m.gguf", "--device-cache-experts", "9"])
            .engine
            .to_config("model.gguf".into(), &crate::config::Layers::none())
            .unwrap();
        assert_eq!(config.device_cache_experts, 9);
    }

    #[test]
    fn run_args_map_vram_order_flag_into_config() {
        let config = parse_run_args(&["run", "m.gguf"])
            .engine
            .to_config("model.gguf".into(), &crate::config::Layers::none())
            .unwrap();
        assert_eq!(config.vram_order, VramOrder::Score);
        let config = parse_run_args(&["run", "m.gguf", "--vram-order", "recency"])
            .engine
            .to_config("model.gguf".into(), &crate::config::Layers::none())
            .unwrap();
        assert_eq!(config.vram_order, VramOrder::Recency);
        let bad = RunArgs::augment_args(clap::Command::new("run")).try_get_matches_from([
            "run",
            "m.gguf",
            "--vram-order",
            "lfu",
        ]);
        assert!(bad.is_err(), "unknown order is a parse error");
    }

    #[test]
    fn run_args_map_prefetch_flags_into_config() {
        let config = parse_run_args(&["run", "m.gguf"])
            .engine
            .to_config("model.gguf".into(), &crate::config::Layers::none())
            .unwrap();
        assert_eq!(
            (config.prefetch, config.prefetch_lookahead),
            (PrefetchMode::Off, 1)
        );
        let config = parse_run_args(&[
            "run",
            "m.gguf",
            "--prefetch",
            "read",
            "--prefetch-lookahead",
            "2",
        ])
        .engine
        .to_config("model.gguf".into(), &crate::config::Layers::none())
        .unwrap();
        assert_eq!(
            (config.prefetch, config.prefetch_lookahead),
            (PrefetchMode::Read, 2)
        );
        for bad in [["--prefetch", "on"], ["--prefetch-lookahead", "0"]] {
            let argv = ["run", "m.gguf", bad[0], bad[1]];
            let r = RunArgs::augment_args(clap::Command::new("run")).try_get_matches_from(argv);
            assert!(r.is_err(), "{bad:?} must not parse");
        }
    }

    #[test]
    fn run_args_map_io_threads_flag_into_config() {
        let config = parse_run_args(&["run", "m.gguf"])
            .engine
            .to_config("model.gguf".into(), &crate::config::Layers::none())
            .unwrap();
        assert_eq!(config.io_threads, None, "unset = derived per-drive count");
        let config = parse_run_args(&["run", "m.gguf", "--io-threads", "4"])
            .engine
            .to_config("model.gguf".into(), &crate::config::Layers::none())
            .unwrap();
        assert_eq!(config.io_threads, Some(4));
    }

    #[test]
    fn preflight_refuses_an_impossible_pool_with_err_not_exit() {
        // An overflowing lock request must return Err (a process::exit would
        // kill this test); the RAM fit itself is `fit_ram`'s.
        let err = preflight_mlock(Path::new("/nonexistent.gguf"), None, u64::MAX, 1).unwrap_err();
        assert!(err.to_string().contains("--ram-experts-gib"), "{err}");
    }

    #[test]
    fn run_args_map_offload_and_threads_flags_into_config() {
        // The hardware HIP leg config: n_gpu_layers=99, fa=ON, -ncmoe 45, t16.
        let args = parse_run_args(&[
            "run",
            "model.gguf",
            "--threads",
            "16",
            "--n-gpu-layers",
            "99",
            "--flash-attn",
            "--cpu-moe-layers",
            "45",
        ]);
        let config = args
            .engine
            .to_config("model.gguf".into(), &crate::config::Layers::none())
            .unwrap();
        assert_eq!(config.n_predict, 64, "n_predict default untouched");
        assert_eq!(config.threads, 16);
        assert_eq!(config.n_gpu_layers, 99);
        assert!(config.flash_attn_on);
        assert_eq!(config.cpu_moe_layers, Some(45));
    }

    #[test]
    fn run_args_flags_are_independent() {
        // Each knob maps on its own — e.g. the t16-vs-t4 calibration legs
        // change only threads, the mislabeled-leg probe changes only layers.
        let config = parse_run_args(&["run", "m.gguf", "--threads", "4"])
            .engine
            .to_config("model.gguf".into(), &crate::config::Layers::none())
            .unwrap();
        assert_eq!(config.threads, 4);
        assert_eq!(config.n_gpu_layers, 0);
        assert!(!config.flash_attn_on);
        assert_eq!(config.cpu_moe_layers, None);

        let config = parse_run_args(&["run", "m.gguf", "--n-gpu-layers", "7"])
            .engine
            .to_config("model.gguf".into(), &crate::config::Layers::none())
            .unwrap();
        assert_eq!(config.threads, EngineConfig::default().threads);
        assert_eq!(config.n_gpu_layers, 7);
        assert!(!config.flash_attn_on);
        assert_eq!(config.cpu_moe_layers, None);

        let config = parse_run_args(&["run", "m.gguf", "--flash-attn"])
            .engine
            .to_config("model.gguf".into(), &crate::config::Layers::none())
            .unwrap();
        assert!(config.flash_attn_on);
        assert_eq!(config.n_gpu_layers, 0);
        assert_eq!(config.cpu_moe_layers, None);
    }

    #[test]
    fn run_args_reject_zero_threads_and_layers_boundary_values() {
        // `--threads 0` would make llama_decode run with no threads.
        let err = RunArgs::augment_args(clap::Command::new("run"))
            .try_get_matches_from(["run", "m.gguf", "--threads", "0"])
            .expect_err("threads must be >= 1");
        assert!(err.kind() == clap::error::ErrorKind::ValueValidation);
    }
}
