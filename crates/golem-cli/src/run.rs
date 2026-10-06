//! `golem run` (tech design §9): greedy decode of a local GGUF model, either
//! one-shot via `--prompt` or as a stdin REPL of independent single turns.
// The RLIMIT_MEMLOCK preflight is the one sanctioned unsafe call-site here.
#![allow(unsafe_code)]

use std::error::Error;
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use golem_engine::ParityStep;
use golem_engine::atomic::write_atomic;
use golem_engine::config::EngineConfig;
use golem_engine::gguf::read_metadata;
use golem_engine::session::{ChatTurn, Session};
use golem_engine::timing::TimingRecorder;
use golem_llama_sys::substrate_backend;
use golem_placement::pool::{GgufPreadSource, StoreExpertSource};
use golem_store::ChunkQueue;
use serde::Serialize;
use sha2::{Digest, Sha256};

/// Arguments of the `run` subcommand (tech design §9 minimal surface; flags
/// only — §9.4 precedence rules are not implemented).
#[derive(Debug, clap::Args)]
pub struct RunArgs {
    /// Path to the GGUF model file.
    pub model: PathBuf,
    /// One-shot prompt text; when omitted, the stdin REPL runs instead.
    #[arg(long)]
    pub prompt: Option<String>,
    /// Maximum tokens to generate per completion (engine §7.2 default: 64).
    #[arg(long, default_value_t = 64, value_parser = clap::value_parser!(u32).range(1..))]
    pub n_predict: u32,
    /// Context size in KV cache slots (engine §7.2 default: 256).
    #[arg(long, default_value_t = 256, value_parser = clap::value_parser!(u32).range(2..))]
    pub ctx: u32,
    /// Decode threads (engine §7.2 default: logical CPUs / 2, min 1). The
    /// calibration matrix runs reference-vs-reference legs at differing
    /// thread counts (cpu-t16 vs cpu-t4), so the suite driver injects this
    /// flag per leg.
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
    pub threads: Option<u32>,
    /// Model layers to offload to the GPU (`EngineConfig::n_gpu_layers`):
    /// 0 keeps the whole model on CPU (the CI-leg default); the hardware HIP
    /// parity leg uses 99. Without this knob a "hip"-labeled binary would
    /// silently run on CPU.
    #[arg(long, default_value_t = 0)]
    pub n_gpu_layers: u32,
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
    /// suggestion instead of failing mid-load.
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
    /// Model manifest (§4.2): the P3 expert pool then reads expert planes
    /// from the manifest's replica set through the shared chunk queue
    /// (striped, CRC-checked) instead of the single GGUF replica.
    #[arg(long, value_name = "PATH")]
    pub manifest: Option<PathBuf>,
    /// Write a T2 parity dump (JSONL: line-1 header, then one record per
    /// generation step) to this file instead of streaming the completion;
    /// requires `--prompt`. Model identity in the dump is sha256 + byte size,
    /// never a path.
    #[arg(long, value_name = "FILE")]
    pub parity_dump: Option<PathBuf>,
    /// §11 per-step timing CSV destination: opened/truncated with a header
    /// (engine version, fork commit, resolved config), then one row per
    /// generation decode step, flushed at each turn end. H2D columns are
    /// `unmeasured` (no fork event pair on the copy stream yet).
    #[arg(long, value_name = "FILE")]
    pub timing_csv: Option<PathBuf>,
}

impl RunArgs {
    /// Maps the CLI surface onto `EngineConfig` (flag name ↔ config field):
    /// `--ctx` → `n_ctx`, `--threads` → `threads`, `--n-gpu-layers` →
    /// `n_gpu_layers`, `--flash-attn` → `flash_attn_on`, `--cpu-moe-layers`
    /// → `cpu_moe_layers`, `--device-cache-experts` → `device_cache_experts`.
    /// Everything else stays at the engine default.
    pub fn to_config(&self) -> EngineConfig {
        let default = EngineConfig::default();
        EngineConfig {
            model_path: self.model.clone(),
            n_ctx: self.ctx,
            n_predict: self.n_predict,
            threads: self.threads.unwrap_or(default.threads),
            n_gpu_layers: self.n_gpu_layers,
            flash_attn_on: self.flash_attn,
            cpu_moe_layers: self.cpu_moe_layers,
            // Fractional GiB floor to the engine's whole-GiB unit; a value in
            // (0, 1) therefore disables the pool — the help text says so.
            ram_experts_gib: self.ram_experts_gib.unwrap_or(0.0).max(0.0) as u64,
            device_cache_experts: self.device_cache_experts.unwrap_or(0),
            io_threads: self.io_threads,
            manifest_path: self.manifest.clone(),
            timing_csv: self.timing_csv.clone(),
            ..default
        }
    }
}

/// Runs one greedy `run` invocation: loads the model, then either prints a
/// single completion (`--prompt`) or serves per-line single-turn conversations
/// on stdin until EOF. Returns `Err` for load/init failures and one-shot
/// decode errors (process unusable); per-turn REPL errors are printed and the
/// loop continues.
pub fn run(args: &RunArgs) -> Result<(), Box<dyn Error>> {
    let config = args.to_config();
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
    // §3.4 P3: the store path only exists behind the RAM expert pool; a
    // --manifest without the pool would silently decode off the plain GGUF
    // path while the user believes replicas are serving experts.
    if args.manifest.is_some() && config.ram_experts_gib == 0 {
        return Err(
            "--manifest requires --ram-experts-gib > 0: the expert pool \
             is the only consumer of the store-backed source"
                .into(),
        );
    }
    if config.ram_experts_gib > 0 {
        preflight_ram_pool(
            &args.model,
            args.manifest.as_deref(),
            config.ram_experts_gib,
        )?;
    }
    let mut session = Session::new(&config)?;

    match (&args.prompt, &args.parity_dump) {
        (Some(prompt), Some(dump)) => {
            one_shot_parity(&mut session, &args.model, prompt, dump, args.n_predict)
        }
        (Some(prompt), None) => one_shot(&mut session, prompt, &mut timing_out),
        // The dump records exactly one prompt (header `prompt_tokens`), so a
        // multi-turn REPL dump would be ill-defined — refuse it up front.
        (None, Some(_)) => Err("--parity-dump requires --prompt".into()),
        (None, None) => repl(&mut session, &mut timing_out),
    }
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

/// P3 pool preflight, before any weights load: refuse to start when the
/// requested pool cannot be locked (`RLIMIT_MEMLOCK` too low) or would exceed
/// 75% of free RAM, each with an actionable `min(75% free RAM, expert bytes)`
/// suggestion (§5.4: mlock is a correctness feature — never fall back).
fn preflight_ram_pool(
    model: &Path,
    manifest: Option<&Path>,
    gib: u64,
) -> Result<(), Box<dyn Error>> {
    let want_bytes = gib
        .checked_mul(1 << 30)
        .ok_or("--ram-experts-gib overflows")?;

    // Expert bytes from the GGUF tensor table (header walk only).
    let expert_bytes = expert_plane_bytes(model, manifest).unwrap_or_else(|e| {
        eprintln!("warning: cannot pre-scan expert tensors from {model:?}: {e}");
        0
    });

    // Available-RAM suggestion: min(75% of available, what the experts actually need).
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let free = sys.available_memory();
    let suggest = (free * 3 / 4).min(if expert_bytes > 0 {
        expert_bytes
    } else {
        u64::MAX
    });
    // .max(1): a sub-GiB suggestion would print `--ram-experts-gib 0`, which
    // disables the pool entirely — suggest the smallest workable pool instead.
    let suggest_gib = (suggest / (1 << 30)).max(1);

    if let Some(limit) = memlock_limit() {
        if want_bytes > limit {
            return Err(format!(
                "--ram-experts-gib {gib} needs {want_bytes} B locked but RLIMIT_MEMLOCK \
                 is {limit} B.\nRaise it: `ulimit -l unlimited` (or systemd `LimitMEMLOCK=\
                 infinity`), or size the pool down, e.g. --ram-experts-gib {suggest_gib}"
            )
            .into());
        }
    }
    if want_bytes > free * 3 / 4 {
        return Err(format!(
            "--ram-experts-gib {gib} wants {want_bytes} B but only {free} B is available \
             (cap: 75% of available RAM).\nSuggested size: --ram-experts-gib {suggest_gib} \
             (expert bytes: {expert_bytes} B)"
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
        let queue = Arc::new(ChunkQueue::new(0));
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

/// One-line resolved-config string for the CSV header comment (§11).
fn resolved_config_line(config: &EngineConfig) -> String {
    format!(
        "ctx={} batch={} predict={} threads={} gpu_layers={} fa={} cpu_moe={:?} \
         ram_experts_gib={} device_cache_experts={} io_threads={:?} manifest={:?} \
         model_size={}",
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

fn repl(
    session: &mut Session,
    timing_out: &mut Option<std::fs::File>,
) -> Result<(), Box<dyn Error>> {
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        eprint!("golem> ");
        let _ = std::io::stderr().flush();
        let line = match lines.next() {
            Some(Ok(line)) => line,
            Some(Err(e)) => return Err(e.into()),
            None => return Ok(()),
        };
        let turn = [ChatTurn {
            role: "user",
            content: line.as_str(),
        }];
        match session.chat_turn(&turn, &mut emit_stdout) {
            Ok(_) => {
                if let Err(e) = flush_timing(session, timing_out) {
                    eprintln!("error: timing csv: {e}");
                }
                println!()
            }
            Err(e) => {
                println!();
                eprintln!("error: {e}");
            }
        }
    }
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
    fn run_args_default_to_the_cpu_only_config() {
        let config = parse_run_args(&["run", "model.gguf"]).to_config();
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
        let config = parse_run_args(&["run", "m.gguf", "--ram-experts-gib", "2"]).to_config();
        assert_eq!(config.ram_experts_gib, 2);
        // Fractional GiB floor to the engine's whole-GiB unit.
        let config = parse_run_args(&["run", "m.gguf", "--ram-experts-gib", "2.75"]).to_config();
        assert_eq!(config.ram_experts_gib, 2);
    }

    #[test]
    fn run_args_map_device_cache_experts_flag_into_config() {
        let config = parse_run_args(&["run", "m.gguf"]).to_config();
        assert_eq!(config.device_cache_experts, 0, "unset = derive from VRAM");
        let config = parse_run_args(&["run", "m.gguf", "--device-cache-experts", "9"]).to_config();
        assert_eq!(config.device_cache_experts, 9);
    }

    #[test]
    fn run_args_map_io_threads_flag_into_config() {
        let config = parse_run_args(&["run", "m.gguf"]).to_config();
        assert_eq!(config.io_threads, None, "unset = derived per-drive count");
        let config = parse_run_args(&["run", "m.gguf", "--io-threads", "4"]).to_config();
        assert_eq!(config.io_threads, Some(4));
    }

    #[test]
    fn preflight_refuses_an_impossible_pool_with_err_not_exit() {
        // 2^60 GiB-bytes exceeds both RLIMIT_MEMLOCK and available RAM; the
        // preflight must return Err (a process::exit would kill this test).
        let err = preflight_ram_pool(Path::new("/nonexistent.gguf"), None, 1 << 30).unwrap_err();
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
        assert_eq!(args.n_predict, 64, "n_predict default untouched");
        let config = args.to_config();
        assert_eq!(config.threads, 16);
        assert_eq!(config.n_gpu_layers, 99);
        assert!(config.flash_attn_on);
        assert_eq!(config.cpu_moe_layers, Some(45));
    }

    #[test]
    fn run_args_flags_are_independent() {
        // Each knob maps on its own — e.g. the t16-vs-t4 calibration legs
        // change only threads, the mislabeled-leg probe changes only layers.
        let config = parse_run_args(&["run", "m.gguf", "--threads", "4"]).to_config();
        assert_eq!(config.threads, 4);
        assert_eq!(config.n_gpu_layers, 0);
        assert!(!config.flash_attn_on);
        assert_eq!(config.cpu_moe_layers, None);

        let config = parse_run_args(&["run", "m.gguf", "--n-gpu-layers", "7"]).to_config();
        assert_eq!(config.threads, EngineConfig::default().threads);
        assert_eq!(config.n_gpu_layers, 7);
        assert!(!config.flash_attn_on);
        assert_eq!(config.cpu_moe_layers, None);

        let config = parse_run_args(&["run", "m.gguf", "--flash-attn"]).to_config();
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
