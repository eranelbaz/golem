//! `golem plan` (tech design §10): the per-tier memory table, the four
//! governing numbers and a fit verdict, from the GGUF header alone — no
//! model load. Infeasible → exit 2 with a `golem_plan_infeasible` envelope.

use std::error::Error;
use std::path::PathBuf;

use crate::run::Gib;
use golem_api::{ErrorCode, ErrorEnvelope};
use golem_plan::{Machine, ModelFacts, PlanInput, PlanReport, Term, WS_MEASURED, total};

/// Disk read bandwidth across both drives, measured in M3.3 (4.73 GB/s,
/// docs/baselines/m3.3-store-bench.md); prices the cold-expert latency.
const DISK_BPS: f64 = 4.73e9;

const GIB: f64 = (1u64 << 30) as f64;

/// Arguments of the `plan` subcommand.
#[derive(Debug, clap::Args)]
pub struct PlanArgs {
    /// Path to the GGUF model file (first shard of a split); else
    /// `GOLEM_MODEL`, else `model` in `golem.toml`, else the `[models.*]`
    /// entry named by `--model-name`.
    pub model: Option<PathBuf>,
    /// `[models.<NAME>]` entry of `golem.toml` to plan when no `model` is
    /// set; flag > `GOLEM_MODEL_NAME` > the first entry.
    #[arg(long, value_name = "NAME")]
    pub model_name: Option<String>,
    /// Context length per sequence; resolved like `serve --ctx` (flag > env >
    /// `[models.*]` > golem.toml > engine default 256).
    #[arg(long, value_parser = clap::value_parser!(u32).range(2..))]
    pub ctx: Option<u32>,
    /// Concurrent sequences [default: 1].
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub parallel: Option<u32>,
    /// Streamed prefill ubatch; prefill streaming (§7.6) is assumed on.
    /// Resolved like `serve --prefill-ubatch` [default: 2048].
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub prefill_ubatch: Option<u32>,
    /// Device expert-cache cap; omitted = `--device-cache-experts`, else what
    /// the other VRAM terms leave.
    #[arg(long, value_name = "GIB", conflicts_with = "device_cache_experts")]
    pub vram_experts_gib: Option<f64>,
    /// Pinned per-layer device expert-cache size, priced as `serve` clamps
    /// it; resolved like `serve --device-cache-experts` (0 = derived).
    #[arg(long, value_name = "EXPERTS")]
    pub device_cache_experts: Option<u32>,
    /// Fraction (0, 1] of the derived device expert-cache allowance; same knob
    /// as `run --vram-experts-frac` [default: 1].
    #[arg(long, value_parser = frac, conflicts_with = "vram_experts_gib")]
    pub vram_experts_frac: Option<f32>,
    /// VRAM held back for other GPU clients (compositor, browsers); resolved
    /// like `serve`, else max(1.5 GiB, 6 % of device VRAM).
    #[arg(long, value_name = "GIB", value_parser = crate::run::parse_gib)]
    pub vram_reserve_gib: Option<f64>,
    /// Host expert-pool cap, floored to whole GiB as `serve` does (so [0, 1)
    /// is rejected); resolved like `serve`, else min(0.75 · free RAM, Σ
    /// expert bytes).
    #[arg(long, value_name = "GIB", value_parser = crate::run::parse_gib)]
    pub ram_experts_gib: Option<f64>,
    /// State-snapshot arena; resolved like `serve`, else the §7.1 default
    /// max(2 GiB, 2 × floor).
    #[arg(long, value_name = "GIB", value_parser = crate::run::parse_gib)]
    pub snapshot_ram_gib: Option<f64>,
    /// Device VRAM; omitted = the HIP device's total (HIP builds), else 24.
    #[arg(long, value_name = "GIB")]
    pub vram_gib: Option<f64>,
    /// K cache element type the KV term is priced at; resolved like
    /// `run --cache-type-k` (flag > env > golem.toml > f16).
    #[arg(long, value_name = "TYPE")]
    pub cache_type_k: Option<golem_plan::KvCacheType>,
    /// V cache element type the KV term is priced at (as `--cache-type-k`).
    #[arg(long, value_name = "TYPE")]
    pub cache_type_v: Option<golem_plan::KvCacheType>,
    /// VRAM victim ring slots priced as a VRAM term (as `run --victim-ring`;
    /// flag > golem.toml `victim_ring_slots` > 0 = off).
    #[arg(long, value_name = "N")]
    pub victim_ring: Option<u32>,
    /// Ignore the saved `golem optimize` profile for this model (as `serve`).
    #[arg(long)]
    pub no_optimize_profile: bool,
    /// Machine-readable report on stdout.
    #[arg(long)]
    pub json: bool,
}

fn frac(s: &str) -> Result<f32, String> {
    let f: f32 = s.parse().map_err(|e| format!("{s}: {e}"))?;
    if f > 0.0 && f <= 1.0 {
        Ok(f)
    } else {
        Err(format!("{s} outside (0, 1]"))
    }
}

fn bytes(gib: Option<f64>) -> u64 {
    gib.map_or(0, |g| (g * GIB) as u64)
}

/// The four §10 governing numbers.
#[derive(Debug, serde::Serialize)]
struct Governing {
    vram_headroom: i64,
    ram_pool: u64,
    kv_per_token: u64,
    /// One cold expert-layer (3 planes at the max stride) read from disk.
    cold_expert_ms: f64,
}

fn governing(m: &ModelFacts, r: &PlanReport) -> Governing {
    Governing {
        vram_headroom: r.vram_headroom,
        ram_pool: r.term("ram_pool"),
        kv_per_token: m.kv_per_token,
        cold_expert_ms: 3.0 * m.max_plane_stride as f64 / DISK_BPS * 1e3,
    }
}

/// `golem_plan_infeasible` with the failing inequality and, as detail, the
/// per-tier overrun in bytes.
pub fn infeasible_envelope(r: &PlanReport) -> ErrorEnvelope {
    let over = |h: i64| u64::try_from(-h).unwrap_or(0);
    ErrorEnvelope::with_detail(
        ErrorCode::PlanInfeasible,
        r.violation().unwrap_or_default(),
        Some(serde_json::json!({
            "vram_overrun": over(r.vram_headroom),
            "ram_overrun": over(r.ram_headroom),
            "largest_fitting_ubatch": r.largest_fitting_ubatch,
            "device_expert_cache": r.term("device_expert_cache"),
            "min_device_expert_cache": r.min_device_expert_cache,
        })),
    )
}

fn table(tier: &str, terms: &[Term], cap: u64) -> String {
    let mut s = format!("{tier}\n");
    for t in terms {
        let name = t.name.replace('.', "  └ ");
        s += &format!("  {name:<22} {:>10.2} GiB\n", t.bytes as f64 / GIB);
    }
    s += &format!(
        "  {:<22} {:>10.2} GiB of {:.2} GiB\n",
        "total",
        total(terms) as f64 / GIB,
        cap as f64 / GIB
    );
    s
}

/// This machine; `vram_gib` overrides the device total (HIP builds read it,
/// else 24 GiB).
pub fn machine(vram_gib: Option<f64>) -> Machine {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    Machine {
        vram_total: vram_gib
            .map(|g| bytes(Some(g)))
            .or_else(|| golem_engine::session::device_vram().map(|(_, t)| t))
            .unwrap_or(24 << 30),
        ram_total: sys.total_memory(),
        ram_free: sys.available_memory(),
        rocm_overhead: golem_plan::ROCM_OVERHEAD,
    }
}

/// Runs `golem plan`; returns the process exit code (0 fits, 2 infeasible).
pub fn plan(args: &PlanArgs) -> Result<i32, Box<dyn Error>> {
    let base = crate::config::Layers::from_process()?;
    let cache_flags = (args.cache_type_k, args.cache_type_v);
    let start = crate::serve::start_model(
        &base,
        args.model.clone(),
        args.model_name
            .clone()
            .or_else(|| (base.env)("GOLEM_MODEL_NAME")),
        args.no_optimize_profile,
        cache_flags,
    )?;
    let (model, l) = (start.path, start.layers);
    let d = golem_engine::config::EngineConfig::default();
    let (k, v) = l.cache_types(cache_flags)?;
    let m = ModelFacts::from_gguf_kv(&model, k, v)?;
    let mach = machine(args.vram_gib);
    let ctx = l.get(args.ctx, "ctx", d.n_ctx)?;
    let parallel = l.get(args.parallel, "parallel", 1)?;
    let snapshot_ram = match l.opt(args.snapshot_ram_gib.map(Gib), "snapshot_ram_gib")? {
        Some(Gib(g)) => bytes(Some(g)),
        None => golem_plan::snapshot_arena(&m, parallel, ctx).1,
    };
    let ram_experts = l.opt(args.ram_experts_gib.map(Gib), "ram_experts_gib")?;
    // `serve` floors to whole GiB, so [0, 1) runs with no pool, which the
    // assumed prefill stream needs; `Pooled { 0 }` would price the default.
    if let Some(Gib(g)) = ram_experts
        && g < 1.0
    {
        return Err(format!(
            "ram_experts_gib {g} (--ram-experts-gib / GOLEM_RAM_EXPERTS_GIB / \
             golem.toml) floors to 0 GiB: no expert pool, but plan assumes \
             prefill streaming, which needs one; use >= 1 or leave it unset"
        )
        .into());
    }
    let mut input = PlanInput {
        ctx,
        parallel,
        prefill_ubatch: l
            .get(args.prefill_ubatch, "prefill_ubatch", d.prefill_ubatch)?
            .min(ctx),
        prefill_stream: true,
        vram_experts: bytes(args.vram_experts_gib),
        experts: golem_plan::ExpertResidency::Pooled {
            ram_experts: bytes(ram_experts.map(|Gib(g)| g.floor())),
        },
        snapshot_ram,
        vram_reserve: l
            .opt(args.vram_reserve_gib.map(Gib), "vram_reserve_gib")?
            .map_or_else(
                || golem_plan::default_vram_reserve(mach.vram_total),
                |Gib(g)| bytes(Some(g)),
            ),
        victim_ring_slots: l.get(args.victim_ring, "victim_ring_slots", 0)?,
        ws: WS_MEASURED,
    };
    if args.vram_experts_gib.is_none() {
        let k = l.get(args.device_cache_experts, "device_cache_experts", 0)?;
        input.vram_experts = crate::serve::device_cache_k_bytes(k, &m, &mach, &input);
    }
    let frac = l.get(
        args.vram_experts_frac,
        "vram_experts_frac",
        d.vram_experts_frac,
    )?;
    let r = golem_plan::plan_at_frac(&m, &mach, &input, frac);
    let g = governing(&m, &r);
    if args.json {
        println!(
            "{}",
            serde_json::json!({
                "model": m, "machine": mach, "input": input,
                "report": r, "governing": g,
                "verdict": if r.feasible { "fits" } else { "infeasible" },
            })
        );
    } else {
        print!("{}", table("VRAM", &r.vram, mach.vram_total));
        print!("{}", table("RAM", &r.ram, mach.ram_free));
        println!(
            "VRAM headroom {:.2} GiB | RAM pool {:.2} GiB | KV {} B/token | cold expert ≈ {:.1} ms",
            g.vram_headroom as f64 / GIB,
            g.ram_pool as f64 / GIB,
            g.kv_per_token,
            g.cold_expert_ms
        );
        println!(
            "ws: fitted on gfx1100 ({} MiB floor, {} + {}·ctx B/ubatch token, \
             ≥ {}·ctx; {} below ctx {})",
            WS_MEASURED.base >> 20,
            WS_MEASURED.per_ubatch_token,
            WS_MEASURED.per_ubatch_ctx_token,
            WS_MEASURED.score_per_ubatch_ctx_token,
            WS_MEASURED.dense_per_ubatch_token,
            m.dense_below_ctx
        );
        println!(
            "verdict: {}",
            if r.feasible { "fits" } else { "infeasible" }
        );
    }
    if r.feasible {
        Ok(0)
    } else {
        eprintln!("{}", infeasible_envelope(&r));
        Ok(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_names_the_inequality_and_overrun() {
        let r = PlanReport {
            vram: vec![Term {
                name: "t_ne",
                bytes: 25 << 30,
            }],
            ram: vec![],
            vram_headroom: -(1 << 30),
            ram_headroom: 0,
            min_device_expert_cache: 0,
            feasible: false,
            largest_fitting_ubatch: Some(2048),
        };
        let e = infeasible_envelope(&r).to_string();
        assert!(e.contains("golem_plan_infeasible"), "{e}");
        assert!(e.contains("VRAM 25.0 GiB > 24.0 GiB"), "{e}");
        assert!(
            e.contains("largest --prefill-ubatch that fits: 2048"),
            "{e}"
        );
        assert!(e.contains("\"vram_overrun\":1073741824"), "{e}");
        assert!(e.contains("\"min_device_expert_cache\":0"), "{e}");
    }
}
