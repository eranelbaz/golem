//! `golem` CLI entry point (tech design §9): run.
//! Bare `golem` (no subcommand) prints the version.
//! Future: `golem import` (§5.5 store intake: hash + CRC + mirror a shard
//! set into `$GOLEM_DATA_DIR`; `golem_store::import` already implements it).

use clap::{Parser, Subcommand};

mod run;

#[derive(Parser)]
#[command(name = "golem", version, about = "golem CLI: run")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run a local GGUF model (greedy sampling); `--parity-dump
    /// <FILE>` emits a T2 parity dump (JSONL) instead of streaming text.
    Run {
        /// The only sampler for now; the flag is kept so later samplers slot in
        /// without a breaking change.
        #[arg(long, required = true)]
        greedy: bool,
        #[command(flatten)]
        args: run::RunArgs,
    },
}

fn main() {
    // Engine diagnostics (store health transitions, pool/stream warnings) go
    // to stderr; `RUST_LOG` overrides the warn default.
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .try_init();
    let cli = Cli::parse();
    let status = match cli.command {
        None => {
            println!("golem {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some(Command::Run { args, .. }) => run::run(&args),
    };
    if let Err(e) = status {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
