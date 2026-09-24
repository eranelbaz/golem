//! `golem run` (tech design §9): greedy decode of a local GGUF model, either
//! one-shot via `--prompt` or as a stdin REPL of independent single turns.

use std::error::Error;
use std::io::{BufRead, Write};
use std::path::PathBuf;

use golem_engine::config::EngineConfig;
use golem_engine::session::{ChatTurn, Session};

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
}

/// Runs one greedy `run` invocation: loads the model, then either prints a
/// single completion (`--prompt`) or serves per-line single-turn conversations
/// on stdin until EOF. Returns `Err` only for load/init failures (process
/// unusable); per-turn REPL errors are printed and the loop continues.
pub fn run(args: &RunArgs) -> Result<(), Box<dyn Error>> {
    let config = EngineConfig {
        model_path: args.model.clone(),
        n_ctx: args.ctx,
        n_predict: args.n_predict,
        ..EngineConfig::default()
    };
    let mut session = Session::new(&config)?;

    match &args.prompt {
        Some(prompt) => one_shot(&mut session, prompt),
        None => repl(&mut session),
    }
}

/// One-shot mode: a single user turn, assistant output on stdout.
fn one_shot(session: &mut Session, prompt: &str) -> Result<(), Box<dyn Error>> {
    session.chat_turn(
        &[ChatTurn {
            role: "user",
            content: prompt,
        }],
        &mut emit_stdout,
    )?;
    // Terminate the streamed completion so stdout stays line-oriented.
    println!();
    Ok(())
}

/// REPL mode: each stdin line is an independent single-turn conversation.
/// The `golem> ` prompt goes to stderr so stdout stays parseable; per-turn
/// errors print and continue (one bad line must not kill the session).
fn repl(session: &mut Session) -> Result<(), Box<dyn Error>> {
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        eprint!("golem> ");
        let _ = std::io::stderr().flush();
        // Read failure (not EOF) makes every future read fail too: fatal.
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

/// Streams one decoded piece to stdout, flushing so streamed text is visible
/// before the next `golem> ` prompt is written.
fn emit_stdout(piece: &str) {
    print!("{piece}");
    let _ = std::io::stdout().flush();
}
