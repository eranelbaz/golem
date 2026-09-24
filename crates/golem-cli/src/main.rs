//! `golem` CLI entry point (tech design §9): run, serve, list, stop, plan, pull.
//! Bare `golem` (no subcommand) prints the version.

use clap::{Parser, Subcommand};

mod run;

#[derive(Parser)]
#[command(
    name = "golem",
    version,
    about = "golem CLI: run, serve, list, stop, plan, pull"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run a local GGUF model (greedy sampling this milestone); `--parity-dump
    /// <FILE>` emits a T2 parity dump (JSONL) instead of streaming text.
    Run {
        /// The only sampler this milestone; the flag is kept so later samplers
        /// slot in without a breaking change.
        #[arg(long, required = true)]
        greedy: bool,
        #[command(flatten)]
        args: run::RunArgs,
    },
}

fn main() {
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
