//! `golem` CLI entry point (tech design §9): run, serve, list, stop, plan, pull.
//!
//! Task M1.3 adds the first subcommand (`run`); the rest arrive with their
//! board tasks. Bare `golem` (no subcommand) prints the version.

use clap::{Parser, Subcommand};

mod run;

/// `golem` command-line entry point.
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

/// The `golem` subcommands (only `run` exists so far).
#[derive(Subcommand)]
enum Command {
    /// Run a local GGUF model (greedy sampling this milestone).
    Run {
        /// Use greedy sampling — the only sampler this milestone; the flag is
        /// kept so later samplers slot in without a breaking change.
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
