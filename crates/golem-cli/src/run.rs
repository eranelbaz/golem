//! `golem run` (tech design §9): greedy decode of a local GGUF model, either
//! one-shot via `--prompt` or as a stdin REPL of independent single turns.

use std::error::Error;
use std::io::{BufRead, Read, Write};
use std::path::{Path, PathBuf};

use golem_engine::ParityStep;
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
    /// Write a T2 parity dump (JSONL: line-1 header, then one record per
    /// generation step) to this file instead of streaming the completion;
    /// requires `--prompt`. Model identity in the dump is sha256 + byte size,
    /// never a path.
    #[arg(long, value_name = "FILE")]
    pub parity_dump: Option<PathBuf>,
}

/// Runs one greedy `run` invocation: loads the model, then either prints a
/// single completion (`--prompt`) or serves per-line single-turn conversations
/// on stdin until EOF. Returns `Err` for load/init failures and one-shot
/// decode errors (process unusable); per-turn REPL errors are printed and the
/// loop continues.
pub fn run(args: &RunArgs) -> Result<(), Box<dyn Error>> {
    let config = EngineConfig {
        model_path: args.model.clone(),
        n_ctx: args.ctx,
        n_predict: args.n_predict,
        ..EngineConfig::default()
    };
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

/// Writes `contents` to `path` atomically: temp file in the target's directory
/// (same filesystem, so rename cannot degrade to a copy), then rename over the
/// target — readers see the old or the new file, never a partial one.
fn write_atomic(path: &Path, contents: &str) -> std::io::Result<()> {
    let name = path.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "parity dump path has no file name",
        )
    })?;
    let tmp = path.with_file_name(format!("{}.tmp", name.to_string_lossy()));
    if let Err(e) = std::fs::File::create(&tmp).and_then(|mut f| f.write_all(contents.as_bytes())) {
        let _ = std::fs::remove_file(&tmp);
        return Err(e);
    }
    std::fs::rename(&tmp, path)
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

    #[test]
    fn atomic_write_replaces_target_without_tmp_leftovers() {
        let dir = scratch_dir("atomic");
        let target = dir.join("dump.jsonl");
        std::fs::write(&target, "STALE").unwrap();
        write_atomic(&target, "NEW-CONTENT").unwrap();
        assert_eq!(std::fs::read_to_string(&target).unwrap(), "NEW-CONTENT");
        assert_eq!(
            std::fs::read_dir(&dir).unwrap().count(),
            1,
            "no tmp residue"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
