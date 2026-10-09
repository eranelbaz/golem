//! `golem` CLI entry point (tech design §9): run, serve.
//! Bare `golem` (no subcommand) prints the version.
//! Future: `golem import` (§5.5 store intake: hash + CRC + mirror a shard
//! set into `$GOLEM_DATA_DIR`; `golem_store::import` already implements it).

use clap::{Parser, Subcommand};

mod run;
mod serve;

#[derive(Parser)]
#[command(name = "golem", version, about = "golem CLI: run, serve")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run a local GGUF model (greedy sampling); `--parity-dump
    /// <FILE>` emits a T2 parity dump (JSONL) instead of streaming text.
    Run {
        /// The only sampler, on by default; the flag stays so later samplers
        /// slot in without a breaking change.
        #[arg(long, action = clap::ArgAction::Set, num_args = 0..=1,
            require_equals = true, default_missing_value = "true", default_value_t = true)]
        greedy: bool,
        #[command(flatten)]
        args: run::RunArgs,
    },
    /// Serve a local GGUF model over the OpenAI-compatible HTTP API until
    /// SIGINT/SIGTERM.
    Serve {
        #[command(flatten)]
        args: serve::ServeArgs,
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
        Some(Command::Run { args, .. }) => {
            // No model path → print the run help (every valid flag) instead
            // of clap's bare "required arguments" error.
            if args.model.is_none() {
                use clap::CommandFactory as _;
                let mut cmd = Cli::command();
                cmd.find_subcommand_mut("run")
                    .expect("run subcommand exists")
                    .print_help()
                    .expect("write help");
                std::process::exit(0);
            }
            run::run(&args)
        }
        Some(Command::Serve { args }) => serve::serve(&args),
    };
    if let Err(e) = status {
        let code = if e.is::<golem_server::ServeError>() {
            golem_api::ErrorCode::BackendUnavailable
        } else {
            golem_api::ErrorCode::InvalidRequest
        };
        eprintln!("{}", golem_api::ErrorEnvelope::new(code, e.to_string()));
        // The decode worker is leaked mid-`llama_decode`: exit now, skipping
        // pool teardown (the kernel reclaims it).
        if matches!(
            e.downcast_ref::<golem_server::ServeError>(),
            Some(golem_server::ServeError::ShutdownTimeout(_))
        ) {
            std::process::exit(3);
        }
        std::process::exit(1);
    }
}
