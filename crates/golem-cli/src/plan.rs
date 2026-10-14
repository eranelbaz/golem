//! `golem plan` (tech design §10): the per-tier memory table, the four
//! governing numbers and a fit verdict, from the GGUF header alone — no
//! model load. Infeasible → exit 2 with a `golem_plan_infeasible` envelope.

use std::error::Error;
use std::path::PathBuf;

use golem_api::{ErrorCode, ErrorEnvelope};
use golem_plan::{Machine, ModelFacts, PlanInput, PlanReport, Term, WS_MEASURED, total};

/// Disk read bandwidth across both drives, measured in M3.3 (4.73 GB/s,
/// docs/baselines/m3.3-store-bench.md); prices the cold-expert latency.
const DISK_BPS: f64 = 4.73e9;

const GIB: f64 = (1u64 << 30) as f64;

/// Arguments of the `plan` subcommand.
#[derive(Debug, clap::Args)]
pub struct PlanArgs {
    /// Path to the GGUF model file (first shard of a split).
    pub model: PathBuf,
    /// Context length per sequence.
    #[arg(long, default_value_t = 4096, value_parser = clap::value_parser!(u32).range(2..))]
    pub ctx: u32,
    /// Concurrent sequences.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
    pub parallel: u32,
    /// Streamed prefill ubatch; prefill streaming (§7.6) is assumed on.
    #[arg(long, default_value_t = 2048, value_parser = clap::value_parser!(u32).range(1..))]
    pub prefill_ubatch: u32,
    /// Device expert-cache cap; omitted = what the other VRAM terms leave.
    #[arg(long, value_name = "GIB")]
    pub vram_experts_gib: Option<f64>,
    /// Fraction (0, 1] of the derived device expert-cache allowance; same knob
    /// as `run --vram-experts-frac`.
    #[arg(long, default_value_t = 1.0, value_parser = frac, conflicts_with = "vram_experts_gib")]
    pub vram_experts_frac: f32,
    /// Host expert-pool cap; omitted = min(0.75 · free RAM, Σ expert bytes).
    #[arg(long, value_name = "GIB")]
    pub ram_experts_gib: Option<f64>,
    /// State-snapshot arena; omitted = the §7.1 default max(2 GiB, 2 × floor).
    #[arg(long, value_name = "GIB")]
    pub snapshot_ram_gib: Option<f64>,
    /// Device VRAM; omitted = the HIP device's total (HIP builds), else 24.
    #[arg(long, value_name = "GIB")]
    pub vram_gib: Option<f64>,
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
    let m = ModelFacts::from_gguf(&args.model)?;
    let mach = machine(args.vram_gib);
    let snapshot_ram = match args.snapshot_ram_gib {
        Some(g) => bytes(Some(g)),
        None => golem_plan::snapshot_arena(&m, args.parallel, args.ctx).1,
    };
    let input = PlanInput {
        ctx: args.ctx,
        parallel: args.parallel,
        prefill_ubatch: args.prefill_ubatch.min(args.ctx),
        prefill_stream: true,
        vram_experts: bytes(args.vram_experts_gib),
        experts: golem_plan::ExpertResidency::Pooled {
            ram_experts: bytes(args.ram_experts_gib),
        },
        snapshot_ram,
        ws: WS_MEASURED,
    };
    let r = golem_plan::plan_at_frac(&m, &mach, &input, args.vram_experts_frac);
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
            "ws: fitted on gfx1100 ({} MiB base, {} B/ctx token, {} B/ubatch token; {} below ctx {})",
            WS_MEASURED.base >> 20,
            WS_MEASURED.per_ctx_token,
            WS_MEASURED.per_ubatch_token,
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
