//! `golem run` (tech design §9): greedy decode of a local GGUF model, either
//! one-shot via `--prompt` or as a stdin REPL of independent single turns.

use std::error::Error;
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};

use golem_engine::ParityStep;
use golem_engine::atomic::write_atomic;
use golem_engine::config::EngineConfig;
use golem_engine::session::{ChatTurn, Session};
use golem_llama_sys::substrate_backend;
use serde::Serialize;
use sha2::{Digest, Sha256};

/// Arguments of the `run` subcommand (tech design §9 minimal surface; flags
/// only — §9.4 precedence rules are an M4.x concern).
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
    /// Decode threads (engine §7.2 default: 1). The M1.4 calibration matrix
    /// runs reference-vs-reference legs at differing thread counts
    /// (cpu-t16 vs cpu-t4), so the suite driver injects this flag per leg.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
    pub threads: u32,
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
    /// Write a T2 parity dump (JSONL: line-1 header, then one record per
    /// generation step) to this file instead of streaming the completion;
    /// requires `--prompt`. Model identity in the dump is sha256 + byte size,
    /// never a path.
    #[arg(long, value_name = "FILE")]
    pub parity_dump: Option<PathBuf>,
}

impl RunArgs {
    /// Maps the CLI surface onto `EngineConfig` (flag name ↔ config field):
    /// `--ctx` → `n_ctx`, `--threads` → `threads`, `--n-gpu-layers` →
    /// `n_gpu_layers`, `--flash-attn` → `flash_attn_on`, `--cpu-moe-layers`
    /// → `cpu_moe_layers`. Everything else stays at the engine default.
    pub fn to_config(&self) -> EngineConfig {
        EngineConfig {
            model_path: self.model.clone(),
            n_ctx: self.ctx,
            n_predict: self.n_predict,
            threads: self.threads,
            n_gpu_layers: self.n_gpu_layers,
            flash_attn_on: self.flash_attn,
            cpu_moe_layers: self.cpu_moe_layers,
            ..EngineConfig::default()
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
    let mut session = Session::new(&config)?;

    match (&args.prompt, &args.parity_dump) {
        (Some(prompt), Some(dump)) => {
            one_shot_parity(&mut session, &args.model, prompt, dump, args.n_predict)
        }
        (Some(prompt), None) => one_shot(&mut session, prompt),
        // The dump records exactly one prompt (header `prompt_tokens`), so a
        // multi-turn REPL dump would be ill-defined — refuse it up front.
        (None, Some(_)) => Err("--parity-dump requires --prompt".into()),
        (None, None) => repl(&mut session),
    }
}

fn one_shot(session: &mut Session, prompt: &str) -> Result<(), Box<dyn Error>> {
    session.chat_turn(
        &[ChatTurn {
            role: "user",
            content: prompt,
        }],
        &mut emit_stdout,
    )?;
    println!();
    Ok(())
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

/// One per-step record; also key-sorted (position < token < top1 < top2).
#[derive(Serialize)]
struct ParityDumpStep {
    position: u32,
    token: i32,
    top1: f32,
    top2: f32,
}

/// Renders the dump body: line-1 header, then one record per step, each a
/// compact single-line JSON object in the frozen field order.
fn render_parity_dump(header: &ParityDumpHeader<'_>, steps: &[ParityStep]) -> String {
    let mut out = String::with_capacity(96 + 96 * steps.len());
    // Plain string/number fields — serialization cannot fail.
    out.push_str(&serde_json::to_string(header).expect("header serialization cannot fail"));
    out.push('\n');
    for step in steps {
        let record = ParityDumpStep {
            position: step.position,
            token: step.token,
            top1: step.top1,
            top2: step.top2,
        };
        out.push_str(&serde_json::to_string(&record).expect("step serialization cannot fail"));
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

fn repl(session: &mut Session) -> Result<(), Box<dyn Error>> {
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
            Ok(_) => println!(),
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
        assert_eq!(config.threads, 1, "engine §7.2 default");
        assert_eq!(config.n_gpu_layers, 0, "default offloads nothing");
        assert!(
            !config.flash_attn_on,
            "default leaves flash attention at AUTO"
        );
        assert_eq!(
            config.cpu_moe_layers, None,
            "default writes no buft overrides"
        );
    }

    #[test]
    fn run_args_map_offload_and_threads_flags_into_config() {
        // The Task 7 hardware HIP leg: n_gpu_layers=99, fa=ON, -ncmoe 45, t16.
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
        assert_eq!(config.threads, 1);
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
