//! `golem` CLI entry point (tech design §9): run, serve, list, stop, plan, pull.
//! Bare `golem` (no subcommand) prints the version.

use clap::{Parser, Subcommand};
use std::path::PathBuf;

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
    /// Compute the T1 op-parity leg (M1.4): run `ggml_mul_mat_id` over
    /// synthetic expert tensors and write a checksum-per-expert-id JSONL dump.
    /// Fully synthetic — no model file is read.
    ParityOp {
        /// Where to write the op dump (JSONL, written atomically).
        #[arg(long, value_name = "FILE")]
        dump: PathBuf,
        /// Expert-weight encoding for the leg: `fp32` or the Q8_0-quantized
        /// variant (same weights, ggml-quantized; compute stays f32).
        #[arg(long, value_parser = ["fp32", "q8_0"], default_value = "fp32")]
        quant: String,
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
        Some(Command::ParityOp { dump, quant }) => {
            let quant = match quant.as_str() {
                "fp32" => golem_engine::Quant::Fp32,
                "q8_0" => golem_engine::Quant::Q8_0,
                other => unreachable!("clap value_parser restricts quant, got {other:?}"),
            };
            golem_engine::run_op_leg(&dump, quant)
                .map(|_| {
                    eprintln!(
                        "op dump: {} ids -> {}",
                        golem_engine::parity_op::OP_N_IDS,
                        dump.display()
                    );
                })
                .map_err(Into::into)
        }
    };
    if let Err(e) = status {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
