//! `golem optimize` (#81): sweeps the scheduling [`Knobs`] for this machine
//! and model, one child process per measurement, and saves the winner as
//! the profile layer `run`/`serve` load (§9.4).
//!
//! The parent holds an exclusive `optimize` registry entry for the whole
//! sweep. That entry reserves the planned bytes once; children
//! (`--measure-one`) register nothing (a second reservation would double
//! count: ≈ 72.7 GiB planned RAM twice > 93 GiB on the target and refuse
//! every candidate) and are listed in the entry's `children` so `stop`
//! reaches them.

use std::error::Error;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use golem_engine::config::EngineConfig;
use golem_engine::session::{ChatTurn, Session};
use golem_registry::Registry;
use serde::{Deserialize, Serialize};

use crate::profile::{self, Axis, Knobs, Metrics, Profile, StopSweep};
use crate::run::{EngineArgs, engine_config};

const MIB: u64 = 1 << 20;
/// Free VRAM / locked RAM must return this close to the pre-sweep reading
/// after a child exits, before the next one starts.
const RETURN_SLACK: u64 = profile::SLACK;
const RETURN_WAIT: Duration = Duration::from_secs(30);
/// Total free-VRAM drift tolerated across the sweep before it stops.
const DRIFT_BUDGET: u64 = 1 << 30;
/// Measured runs per candidate (median).
const RUNS: usize = 5;
/// Warm-up stops once the RAM-pool hit rate moves less than 1 pp per turn.
const STEADY_PP: f64 = 0.01;
const MAX_WARMUP: usize = 10;
/// Fixed greedy prompt set, one turn each per run.
const PROMPTS: [&str; 2] = [
    "Explain in two sentences why the sky is blue.",
    "Write a haiku about a lighthouse in winter.",
];

/// Arguments of the `optimize` subcommand.
#[derive(Debug, clap::Args)]
pub struct OptimizeArgs {
    /// Path to the GGUF model file (first shard of a split).
    #[arg(long)]
    pub model: PathBuf,
    /// Wall-clock bound; coordinate descent stops early with the best so far.
    #[arg(long, default_value_t = 120)]
    pub max_minutes: u64,
    /// Override the swept axes: `knob=v1,v2;knob=...` over threads,
    /// numa_interleave, o_direct, io_threads, io_depth, vram_experts_frac,
    /// async_h2d.
    #[arg(long)]
    pub axes: Option<String>,
    /// Internal: measure one knob set (JSON) and print its metrics.
    #[arg(long, hide = true, requires = "parent_entry")]
    pub measure_one: Option<String>,
    /// Internal: the parent's registry entry id.
    #[arg(long, hide = true)]
    pub parent_entry: Option<String>,
    /// Test seam (debug builds only): admit against a machine whose RAM/VRAM
    /// hold the planned bytes once but not twice.
    #[cfg(debug_assertions)]
    #[arg(long, hide = true)]
    pub test_tight_budget: bool,
    #[command(flatten)]
    pub engine: EngineArgs,
}

/// A child's report: warm steady state (gated) and cold (reported).
#[derive(Debug, Serialize, Deserialize)]
struct Measured {
    warm: Metrics,
    cold: Metrics,
    load_s: f64,
}

/// Runs `golem optimize`.
pub fn optimize(args: &OptimizeArgs) -> Result<(), Box<dyn Error>> {
    if let Some(json) = &args.measure_one {
        let knobs: Knobs = serde_json::from_str(json)?;
        let m = measure_one(args, &knobs)?;
        println!("{}", serde_json::to_string(&m)?);
        return Ok(());
    }
    // Baseline = today's resolved config: env and file apply, no profile.
    let layers = crate::config::Layers::from_process()?;
    #[cfg_attr(not(debug_assertions), allow(unused_mut))]
    let mut mach = machine();
    let (config, requested_snapshot) = engine_config(
        &args.engine,
        &args.model,
        1,
        &layers,
        engine_ram_free(args, &mach),
    )?;
    let facts = config.model_facts.ok_or("model facts unresolved")?;
    let fp = profile::fingerprint(&args.model)?;
    let (free_vram, free_ram) = (profile::sysfs_vram().map_or(0, |(f, _)| f), mach.ram_free);
    let eff = crate::serve::effective(&config, requested_snapshot)?;
    let model_name = args
        .model
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let inst = crate::serve::instance(
        crate::serve::new_id(),
        "optimize",
        model_name,
        &config,
        &eff,
        &mach,
    )?;
    #[cfg(debug_assertions)]
    if args.test_tight_budget {
        mach.ram_total = inst.planned_ram + inst.planned_ram / 2;
        mach.ram_free = mach.ram_total + (4 << 30);
        if inst.planned_vram > 0 {
            mach.vram_total = inst.planned_vram + inst.planned_vram / 2;
        }
    }
    let reg = Registry::open()?;
    let guard = reg.admit_and_register(&inst, &mach, false)?;
    guard.update(|i| i.health = "healthy".into());
    let id = guard.instance().id;

    let base_plan = plan_for(&config, &facts, &mach, &Knobs::default());
    let mut axes = match &args.axes {
        Some(s) => parse_axes(s)?,
        None => default_axes(&config),
    };
    fold_defaults(&mut axes, &config);
    let base = Knobs::default();
    let n_cand = 1 + 2 * axes
        .iter()
        .map(|a| a.variants(&base).iter().filter(|k| **k != base).count())
        .sum::<usize>();
    println!(
        "golem optimize: {} axes, ≤ {n_cand} candidates per pass (2 passes max, paired with the \
         incumbent); expected wall clock: candidates × (load ≈ L s, measured on the first, + 6 turns), \
         bounded by --max-minutes {}",
        axes.len(),
        args.max_minutes
    );
    let mut mem0 = (free_vram, mlocked());
    let deadline = Instant::now() + Duration::from_secs(args.max_minutes * 60);
    // Longest measurement so far. After the baseline `sweep` measures in
    // pairs (candidate + incumbent), so a pair starts only if both fit.
    let mut longest = Duration::ZERO;
    let mut n_measured = 0usize;
    let mut candidates: Vec<(Knobs, Result<Metrics, String>)> = Vec::new();
    let mut first = true;
    let mut measure = |k: &Knobs| -> Result<Metrics, Box<dyn Error>> {
        let need = if n_measured % 2 == 1 {
            2 * longest
        } else {
            longest
        };
        if Instant::now() + need >= deadline {
            return Err(Box::new(StopSweep(format!(
                "--max-minutes {} reached",
                args.max_minutes
            ))));
        }
        let t0 = Instant::now();
        let r = plan_check(&config, &facts, &mach, k, &base_plan).and_then(|()| {
            println!("plan ok: {}", serde_json::to_string(k)?);
            run_child(&id, &guard, k)
        });
        let line = serde_json::to_string(k)?;
        match &r {
            Ok(m) => {
                println!(
                    "candidate {line}: warm {:.2} tok/s hit {:.1}% ttft p50 {:.0} ms p99 {:.0} ms | \
                     cold {:.2} tok/s ttft {:.0} ms | load {:.1} s",
                    m.warm.tok_s,
                    m.warm.hit_rate * 100.0,
                    m.warm.ttft_p50_ms,
                    m.warm.ttft_p99_ms,
                    m.cold.tok_s,
                    m.cold.ttft_p50_ms,
                    m.load_s
                );
                if first {
                    first = false;
                    let per = t0.elapsed().as_secs_f64();
                    println!(
                        "golem optimize: {per:.0} s per measurement → ≤ {n_cand} measurements ≈ \
                         {:.0} min per pass (bounded by --max-minutes {})",
                        per * n_cand as f64 / 60.0,
                        args.max_minutes
                    );
                }
            }
            Err(e) => println!("candidate {line}: disqualified: {e}"),
        }
        n_measured += 1;
        longest = longest.max(t0.elapsed());
        candidates.push((
            *k,
            r.as_ref()
                .map(|m| m.warm.clone())
                .map_err(|e| e.to_string()),
        ));
        mem0 = memory_returned(mem0)?;
        if free_vram.saturating_sub(mem0.0) > DRIFT_BUDGET {
            return Err(Box::new(StopSweep(format!(
                "free VRAM fell {} MiB during the sweep (budget {} MiB)",
                (free_vram - mem0.0) / MIB,
                DRIFT_BUDGET / MIB
            ))));
        }
        r.map(|m| m.warm)
    };
    let result = profile::sweep(&Knobs::default(), &axes, &mut measure)?;
    // Lowest free VRAM any candidate ran with: the profile applies only when
    // at least that much is free.
    let measured_vram = mem0.0.min(free_vram);
    let baseline = candidates
        .iter()
        .find_map(|(k, m)| (*k == Knobs::default()).then(|| m.clone().ok()).flatten())
        .ok_or("baseline measurement failed")?;
    let p = Profile {
        fingerprint: fp.0,
        measured_free_vram: measured_vram,
        measured_free_ram: free_ram,
        knobs: result.as_ref().map(|r| r.0),
        metrics: result.as_ref().map_or(&baseline, |r| &r.1).clone(),
        baseline: baseline.clone(),
        candidates,
    };
    // A no-winner run never replaces a saved winner for this fingerprint.
    let kept = if result.is_none() {
        profile::load_profile(&profile::Fingerprint(p.fingerprint.clone()))
            .and_then(|_| profile::profile_path(&profile::Fingerprint(p.fingerprint.clone())))
    } else {
        None
    };
    let path = match kept.clone() {
        Some(path) => path,
        None => profile::save_profile(&p)?,
    };
    match result {
        Some((k, m)) => println!(
            "winner {}: {:.2} tok/s vs baseline {:.2} ({:+.1}%); profile saved: {}",
            serde_json::to_string(&k)?,
            m.tok_s,
            baseline.tok_s,
            (m.tok_s / baseline.tok_s - 1.0) * 100.0,
            path.display()
        ),
        None if kept.is_some() => println!(
            "baseline {:.2} tok/s: no winner this run; existing profile kept: {}",
            baseline.tok_s,
            path.display()
        ),
        None => println!(
            "baseline {:.2} tok/s: no winner; baseline recorded (not applied by run/serve): {}",
            baseline.tok_s,
            path.display()
        ),
    }
    drop(guard);
    Ok(())
}

/// RAM reading for `engine_config`'s fit. Under the debug-only tight-budget
/// seam the fit must not judge the host: the seam's machine is defined by the
/// planned bytes and is enforced at admission, so the fit sees an
/// always-sufficient reading and the planned pool stays verbatim.
#[cfg_attr(not(debug_assertions), allow(unused_variables))]
fn engine_ram_free(args: &OptimizeArgs, mach: &golem_plan::Machine) -> u64 {
    #[cfg(debug_assertions)]
    if args.test_tight_budget {
        return 1 << 40;
    }
    mach.ram_free
}

/// This machine from sysfs/sysinfo — the parent never starts a HIP runtime.
fn machine() -> golem_plan::Machine {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    golem_plan::Machine {
        vram_total: profile::sysfs_vram().map_or(24 << 30, |(_, t)| t),
        ram_total: sys.total_memory(),
        ram_free: sys.available_memory(),
        rocm_overhead: golem_plan::ROCM_OVERHEAD,
    }
}

/// `Mlocked` from `/proc/meminfo`, bytes.
fn mlocked() -> u64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("Mlocked:"))
                .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<u64>().ok())
        })
        .map_or(0, |kb| kb * 1024)
}

/// Waits for the last child's locked RAM to come back (a child still
/// holding its pool would price the next candidate wrong, so the sweep
/// stops). Free VRAM is shared with the desktop and drifts once the child
/// has exited: a lower reading is warned about and becomes the new baseline.
fn memory_returned((vram0, locked0): (u64, u64)) -> Result<(u64, u64), Box<dyn Error>> {
    let t = Instant::now();
    loop {
        let vram = profile::sysfs_vram().map_or(vram0, |(f, _)| f);
        let locked = mlocked();
        let vram_back = vram + RETURN_SLACK >= vram0;
        if vram_back && locked <= locked0 + RETURN_SLACK {
            return Ok((vram0, locked0));
        }
        if t.elapsed() > RETURN_WAIT {
            if locked > locked0 + RETURN_SLACK {
                return Err(Box::new(StopSweep(format!(
                    "locked RAM not returned after the candidate: {} MiB (was {} MiB)",
                    locked / MIB,
                    locked0 / MIB
                ))));
            }
            eprintln!(
                "golem optimize: free VRAM {} MiB after the candidate (was {} MiB); another \
                 process is using VRAM, continuing from the lower reading",
                vram / MIB,
                vram0 / MIB
            );
            return Ok((vram, locked0));
        }
        std::thread::sleep(Duration::from_millis(200));
    }
}

/// The §10 plan with `k`'s device-cache share; `vram_experts` is the
/// (clamped) explicit K's slab, else the derived allowance at 1.0.
fn plan_for(
    config: &EngineConfig,
    facts: &golem_plan::ModelFacts,
    mach: &golem_plan::Machine,
    k: &Knobs,
) -> golem_plan::PlanReport {
    let mut input = golem_plan::PlanInput {
        ctx: config.n_ctx,
        parallel: config.parallel,
        prefill_ubatch: config.prefill_ubatch.min(config.n_ctx),
        prefill_stream: config.prefill_stream,
        vram_experts: 0,
        experts: golem_plan::ExpertResidency::Pooled {
            ram_experts: config.ram_experts_gib << 30,
        },
        snapshot_ram: config.snapshot_ram_bytes.unwrap_or(0),
        ws: golem_plan::WS_MEASURED,
    };
    input.vram_experts = crate::serve::explicit_cache_bytes(config, facts, mach, &input);
    let frac = if k.vram_experts_frac > 0.0 {
        k.vram_experts_frac
    } else {
        1.0
    };
    golem_plan::plan_at_frac(facts, mach, &input, frac)
}

/// A candidate must fit and plan no more than the baseline.
fn plan_check(
    config: &EngineConfig,
    facts: &golem_plan::ModelFacts,
    mach: &golem_plan::Machine,
    k: &Knobs,
    base: &golem_plan::PlanReport,
) -> Result<(), Box<dyn Error>> {
    if !(0.0..=1.0).contains(&k.vram_experts_frac) {
        return Err(format!(
            "vram_experts_frac {} outside [0, 1] (0 = default)",
            k.vram_experts_frac
        )
        .into());
    }
    let r = plan_for(config, facts, mach, k);
    if let Some(v) = r.violation() {
        return Err(format!("plan infeasible: {v}").into());
    }
    let (v, b) = (golem_plan::total(&r.vram), golem_plan::total(&base.vram));
    let (rv, rb) = (golem_plan::total(&r.ram), golem_plan::total(&base.ram));
    if v > b || rv > rb {
        return Err(format!("plan above baseline: VRAM {v} > {b} B or RAM {rv} > {rb} B").into());
    }
    Ok(())
}

/// One measurement in a child; its pid sits in the entry's `children` while
/// it runs.
fn run_child(
    parent: &str,
    guard: &golem_registry::Guard,
    k: &Knobs,
) -> Result<Measured, Box<dyn Error>> {
    let mut child = std::process::Command::new(std::env::current_exe()?)
        .args(std::env::args_os().skip(1))
        .args(["--measure-one", &serde_json::to_string(k)?])
        .args(["--parent-entry", parent])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()?;
    let pid = child.id();
    // An unregistered child would hold mlocked RAM outside the entry's
    // reservation and out of `stop`'s reach.
    let entry = match golem_registry::Child::of(pid) {
        Ok(c) => c,
        Err(e) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(format!("child {pid}: cannot record start time: {e}").into());
        }
    };
    guard.update(|i| i.children.push(entry));
    // The child checks ownership only after this byte: its pid is in the
    // entry first. Dropping stdin on error makes the child's read fail.
    let go = child.stdin.take().map(|mut s| s.write_all(b"g"));
    let out = child.wait_with_output();
    guard.update(|i| i.children.retain(|c| c.pid() != pid));
    let out = out?;
    go.ok_or("child stdin missing")??;
    let stdout = String::from_utf8_lossy(&out.stdout);
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        let tail: Vec<&str> = stderr.lines().rev().take(3).collect();
        return Err(format!(
            "child {}: {}",
            out.status,
            tail.into_iter().rev().collect::<Vec<_>>().join(" | ")
        )
        .into());
    }
    let last = stdout.lines().last().ok_or("child printed no metrics")?;
    Ok(serde_json::from_str(last)?)
}

/// `knob=v1,v2;knob=...`.
fn parse_axes(s: &str) -> Result<Vec<Axis>, Box<dyn Error>> {
    fn vals<T: std::str::FromStr>(v: &str) -> Result<Vec<T>, Box<dyn Error>>
    where
        T::Err: std::fmt::Display,
    {
        v.split(',')
            .map(|x| x.trim().parse().map_err(|e: T::Err| e.to_string().into()))
            .collect()
    }
    s.split(';')
        .filter(|a| !a.trim().is_empty())
        .map(|a| {
            let (k, v) = a
                .split_once('=')
                .ok_or_else(|| format!("axis {a:?}: want knob=v1,v2"))?;
            Ok(match k.trim() {
                "threads" => Axis::Threads(vals(v)?),
                "numa_interleave" => Axis::NumaInterleave(vals(v)?),
                "o_direct" => Axis::ODirect(vals(v)?),
                "io_threads" => Axis::IoThreads(vals(v)?),
                "io_depth" => Axis::IoDepth(vals(v)?),
                "async_h2d" => Axis::AsyncH2d(vals(v)?),
                "vram_experts_frac" => Axis::VramExpertsFrac(vals(v)?),
                other => return Err(format!("unknown knob {other:?}").into()),
            })
        })
        .collect()
}

/// The swept axes for this machine. NUMA only with > 1 node, `O_DIRECT`
/// and I/O only with a store-backed pool; `async_h2d` is opt-in via
/// `--axes` (no measured win yet). Target: 16 physical cores → threads {8, 12, 16}.
fn default_axes(config: &EngineConfig) -> Vec<Axis> {
    let phys = sysinfo::System::new().physical_core_count().unwrap_or(2) as u32;
    let mut threads = vec![(phys / 2).max(1), (phys * 3 / 4).max(1), phys.max(1)];
    threads.dedup();
    let mut axes = vec![Axis::Threads(threads)];
    let nodes = std::fs::read_to_string("/sys/devices/system/node/online")
        .ok()
        .and_then(|s| golem_store::sys::CpuSet::parse_list(&s))
        .map_or(1, |n| n.0.len());
    if nodes > 1 && config.ram_experts_gib > 0 {
        axes.push(Axis::NumaInterleave(vec![false, true]));
    }
    if config.manifest_path.is_some() && config.ram_experts_gib > 0 {
        axes.push(Axis::ODirect(vec![false, true]));
        // io_threads before io_depth: their gains overlap, so the larger one
        // must be tried from the base point first.
        axes.push(Axis::IoThreads(vec![0, 4, 8]));
        axes.push(Axis::IoDepth(vec![1, 2]));
    }
    if config.ram_experts_gib > 0 && config.device_cache_experts == 0 {
        axes.push(Axis::VramExpertsFrac(vec![1.0, 0.75, 0.5]));
    }
    axes
}

/// Rewrites axis values equal to the baseline's effective setting as the
/// [`Knobs`] default (0/false), so the baseline is never re-measured under
/// another spelling and the candidate count is exact.
fn fold_defaults(axes: &mut [Axis], config: &EngineConfig) {
    fn fold<T: PartialEq + Copy>(v: &mut Vec<T>, is_default: impl Fn(T) -> bool, zero: T) {
        for x in v.iter_mut() {
            if is_default(*x) {
                *x = zero;
            }
        }
        let mut seen = Vec::new();
        v.retain(|x| {
            let new = !seen.contains(x);
            seen.push(*x);
            new
        });
    }
    for a in axes {
        match a {
            Axis::Threads(v) => fold(v, |x| x == config.threads, 0),
            Axis::NumaInterleave(v) => fold(v, |x| x == config.numa_interleave, false),
            Axis::ODirect(v) => fold(v, |x| x == config.o_direct, false),
            Axis::IoThreads(v) => fold(v, |x| Some(x) == config.io_threads, 0),
            Axis::IoDepth(v) => fold(v, |x| x == config.io_depth.max(1), 0),
            Axis::VramExpertsFrac(v) => fold(v, |x| x == config.vram_experts_frac, 0.0),
            Axis::AsyncH2d(v) => fold(v, |x| x == config.async_h2d, false),
        }
    }
}

/// `k` over the resolved baseline config; 0/false knobs keep it.
fn apply(config: &mut EngineConfig, k: &Knobs) {
    if k.threads > 0 {
        config.threads = k.threads;
    }
    config.numa_interleave |= k.numa_interleave;
    config.o_direct |= k.o_direct;
    if k.io_threads > 0 {
        config.io_threads = Some(k.io_threads);
    }
    if k.io_depth > 0 {
        config.io_depth = k.io_depth;
    }
    config.async_h2d |= k.async_h2d;
    if k.vram_experts_frac > 0.0 {
        config.vram_experts_frac = k.vram_experts_frac;
    }
}

/// Evicts the model's shards from the page cache (cold measurement).
fn evict(model: &Path) -> Result<(), Box<dyn Error>> {
    for p in profile::shards(model)? {
        let f = std::fs::File::open(&p)?;
        let len = usize::try_from(f.metadata()?.len())?;
        golem_store::sys::drop_page_cache(&f, 0, len);
    }
    Ok(())
}

/// One prompt-set pass: per-turn TTFT and decode tok/s and
/// the pool hit rate over the pass.
struct Pass {
    ttft_ms: Vec<f64>,
    tok_s: f64,
    hit_rate: f64,
}

fn pass(s: &mut Session) -> Result<Pass, Box<dyn Error>> {
    let counters = |s: &Session| {
        s.p3_stats().map_or((0, 0), |p| {
            use std::sync::atomic::Ordering::Relaxed;
            (
                p.fast_hits.load(Relaxed),
                p.demand_slow_pageins.load(Relaxed),
            )
        })
    };
    let (h0, m0) = counters(s);
    let mut ttft_ms = Vec::new();
    let (mut tokens, mut decode_s) = (0u64, 0f64);
    for prompt in PROMPTS {
        let t0 = Instant::now();
        let mut first: Option<Instant> = None;
        let n = s.chat_turn(
            &[ChatTurn {
                role: "user",
                content: prompt,
            }],
            &mut |_| {
                first.get_or_insert_with(Instant::now);
            },
        )?;
        let end = Instant::now();
        let first = first.unwrap_or(end);
        ttft_ms.push((first - t0).as_secs_f64() * 1e3);
        tokens += u64::from(n.saturating_sub(1));
        decode_s += (end - first).as_secs_f64();
    }
    let (h1, m1) = counters(s);
    let (hits, misses) = (h1 - h0, m1 - m0);
    Ok(Pass {
        ttft_ms,
        tok_s: if decode_s > 0.0 {
            tokens as f64 / decode_s
        } else {
            0.0
        },
        hit_rate: if hits + misses == 0 {
            1.0
        } else {
            hits as f64 / (hits + misses) as f64
        },
    })
}

fn percentile(v: &mut [f64], p: f64) -> f64 {
    v.sort_by(f64::total_cmp);
    let i = ((p * v.len() as f64).ceil() as usize).clamp(1, v.len()) - 1;
    v[i]
}

fn metrics(passes: &[Pass]) -> Metrics {
    let mut tok: Vec<f64> = passes.iter().map(|p| p.tok_s).collect();
    let mut hit: Vec<f64> = passes.iter().map(|p| p.hit_rate).collect();
    let mut ttft: Vec<f64> = passes.iter().flat_map(|p| p.ttft_ms.clone()).collect();
    Metrics {
        tok_s: percentile(&mut tok, 0.5),
        hit_rate: percentile(&mut hit, 0.5),
        ttft_p50_ms: percentile(&mut ttft, 0.5),
        ttft_p99_ms: percentile(&mut ttft, 0.99),
        greedy: Vec::new(),
    }
}

/// Child: loads with `knobs`, measures cold (page cache evicted, fresh
/// session, first pass) then warm (warm-up until the hit rate is steady,
/// median of [`RUNS`] passes on the same session).
fn measure_one(args: &OptimizeArgs, knobs: &Knobs) -> Result<Measured, Box<dyn Error>> {
    let parent = args
        .parent_entry
        .as_deref()
        .ok_or("--parent-entry missing")?;
    // A parent that dies (even SIGKILLed) takes the child with it; a parent
    // that died before the prctl is caught by the pid check below.
    crate::sys::die_with_parent()?;
    // Wait for the parent's go byte: it is written after this pid joins the
    // entry's `children`.
    let mut go = [0u8; 1];
    std::io::Read::read_exact(&mut std::io::stdin(), &mut go)?;
    // The parent's entry is this child's only reservation.
    let entry = Registry::open()?.find(parent)?;
    let me = golem_registry::Child::of(std::process::id())?;
    let listed = entry.children.iter().any(|&c| match c {
        golem_registry::Child::Timed { .. } => c == me,
        golem_registry::Child::Bare(pid) => pid == me.pid(),
    });
    if entry.kind != "optimize" || entry.pid != crate::sys::parent_pid() || !listed {
        return Err(format!("entry `{parent}` does not own this measurement").into());
    }
    let layers = crate::config::Layers::from_process()?;
    let (mut config, _) = engine_config(
        &args.engine,
        &args.model,
        1,
        &layers,
        engine_ram_free(args, &machine()),
    )?;
    apply(&mut config, knobs);
    evict(&args.model)?;
    let t = Instant::now();
    let mut s = Session::new(&config)?;
    let load_s = t.elapsed().as_secs_f64();
    let cold = metrics(&[pass(&mut s)?]);
    let mut last = cold.hit_rate;
    for _ in 0..MAX_WARMUP {
        let h = pass(&mut s)?.hit_rate;
        let steady = (h - last).abs() < STEADY_PP;
        last = h;
        if steady {
            break;
        }
    }
    let passes = (0..RUNS)
        .map(|_| pass(&mut s))
        .collect::<Result<Vec<_>, _>>()?;
    let mut warm = metrics(&passes);
    let mut steps = Vec::new();
    for prompt in PROMPTS {
        s.chat_turn_parity(
            &[ChatTurn {
                role: "user",
                content: prompt,
            }],
            &mut steps,
        )?;
        warm.greedy
            .push(steps.iter().map(|p| (p.token, p.top1 - p.top2)).collect());
    }
    std::io::stdout().flush()?;
    Ok(Measured { warm, cold, load_s })
}
