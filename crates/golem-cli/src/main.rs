//! `golem` CLI entry point (tech design §9): run, serve, plan, list, stop.
//! Bare `golem` (no subcommand) prints the version.
//! Future: `golem import` (§5.5 store intake: hash + CRC + mirror a shard
//! set into `$GOLEM_DATA_DIR`; `golem_store::import` already implements it).

use clap::{Parser, Subcommand};

mod list;
mod plan;
mod run;
mod serve;
mod stop;
mod sys;

use std::io::{BufRead, Write};
use std::time::Duration;

use golem_api::{ErrorCode, ErrorEnvelope};

#[derive(Parser)]
#[command(
    name = "golem",
    version,
    about = "golem CLI: run, serve, plan, list, stop"
)]
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
    /// Print the §10 memory plan (VRAM/RAM tables, governing numbers,
    /// verdict) from the GGUF header; exit 2 when it does not fit.
    Plan {
        #[command(flatten)]
        args: plan::PlanArgs,
    },
    /// List running instances (`--models`: stored model manifests).
    List {
        #[command(flatten)]
        args: list::ListArgs,
    },
    /// Stop a running instance by id or model name.
    Stop {
        #[command(flatten)]
        args: stop::StopArgs,
    },
}

fn init_tracing() {
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
}

fn main() {
    let cli = Cli::parse();
    // Detach forks before tracing, tokio or any other thread exists.
    if let Some(Command::Serve { args }) = &cli.command
        && args.detach
    {
        detach(args);
    }
    init_tracing();
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
        Some(Command::Serve { args }) => golem_registry::Registry::open()
            .map_err(Into::into)
            .and_then(|reg| serve::serve(&args, &reg, serve::new_id(), |_| {})),
        Some(Command::Plan { args }) => match plan::plan(&args) {
            Ok(0) => Ok(()),
            Ok(code) => std::process::exit(code),
            Err(e) => Err(e),
        },
        Some(Command::List { args }) => list::list(&args),
        Some(Command::Stop { args }) => stop::stop(&args),
    };
    if let Err(e) = status {
        eprintln!("{}", envelope(e.as_ref()));
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

/// The one error envelope for a CLI failure (§8.4).
fn envelope(e: &(dyn std::error::Error + 'static)) -> ErrorEnvelope {
    if let Some(env) = e.downcast_ref::<ErrorEnvelope>() {
        return env.clone();
    }
    let code = if e.is::<golem_server::ServeError>() {
        ErrorCode::BackendUnavailable
    } else if matches!(
        e.downcast_ref::<golem_registry::RegistryError>(),
        Some(golem_registry::RegistryError::InstanceNotFound(_))
    ) {
        ErrorCode::InstanceNotFound
    } else if let Some(golem_registry::RegistryError::Control(env)) = e.downcast_ref() {
        return env.clone();
    } else if e.is::<run::PlanInfeasible>()
        || matches!(
            e.downcast_ref::<golem_engine::error::EngineError>(),
            Some(golem_engine::error::EngineError::PlanInfeasible(_))
        )
    {
        ErrorCode::PlanInfeasible
    } else {
        ErrorCode::InvalidRequest
    };
    ErrorEnvelope::new(code, e.to_string())
}

/// `serve --detach`: the parent waits on a pipe for the grandchild's one
/// line (`ready <id> <endpoint>` or an `ErrorEnvelope`), prints `<id>
/// <endpoint>` and exits; the setsid'd grandchild returns and serves.
fn detach(args: &serve::ServeArgs) {
    let fail = |msg: String| -> ! {
        eprintln!("{}", ErrorEnvelope::new(ErrorCode::InvalidRequest, msg));
        std::process::exit(1)
    };
    let id = serve::new_id();
    let Some(log_path) = serve::log_path(&id) else {
        fail("no log dir: set GOLEM_DATA_DIR or HOME".into())
    };
    let log = log_path
        .parent()
        .map_or(Ok(()), std::fs::create_dir_all)
        .and_then(|()| std::fs::File::create(&log_path))
        .unwrap_or_else(|e| fail(format!("log {}: {e}", log_path.display())));
    let (rd, wr) = sys::pipe().unwrap_or_else(|e| fail(format!("pipe: {e}")));
    match sys::fork().unwrap_or_else(|e| fail(format!("fork: {e}"))) {
        sys::Fork::Parent => {
            drop(wr);
            wait_ready(rd, &log_path, Duration::from_secs(args.load_timeout))
        }
        sys::Fork::Child => {
            drop(rd);
            if sys::setsid().is_err() {
                sys::exit_now(1);
            }
            match sys::fork() {
                Ok(sys::Fork::Child) => {}
                Ok(sys::Fork::Parent) => sys::exit_now(0),
                Err(_) => sys::exit_now(1),
            }
            if sys::redirect_stdio(&log).is_err() {
                sys::exit_now(1);
            }
            drop(log);
            init_tracing();
            serve_detached(args, id, wr)
        }
    }
}

/// Grandchild: serves, reporting readiness (or the load error) on `wr`.
fn serve_detached(args: &serve::ServeArgs, id: String, wr: std::fs::File) -> ! {
    let pipe = std::sync::Mutex::new(Some(wr));
    let report = |line: String| {
        if let Some(mut w) = pipe.lock().unwrap_or_else(|p| p.into_inner()).take() {
            let _ = writeln!(w, "{line}");
        }
    };
    let reg = golem_registry::Registry::open().and_then(|r| {
        if serve::lingering() {
            Ok(r)
        } else {
            eprintln!(
                "warning: user linger is off; logind removes $XDG_RUNTIME_DIR at logout, \
                 registering under $XDG_STATE_HOME instead (`loginctl enable-linger` to avoid)"
            );
            golem_registry::Registry::open_fallback()
        }
    });
    let res = reg.map_err(Into::into).and_then(|reg| {
        serve::serve(args, &reg, id.clone(), |ep| {
            report(format!("ready {id} {ep}"))
        })
    });
    match res {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            let env = envelope(e.as_ref());
            eprintln!("{env}");
            report(env.to_string());
            std::process::exit(1)
        }
    }
}

/// Parent: one line from the grandchild, or EOF / timeout → failure with the
/// log tail.
fn wait_ready(rd: std::fs::File, log: &std::path::Path, timeout: Duration) -> ! {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut line = String::new();
        let _ = std::io::BufReader::new(rd).read_line(&mut line);
        let _ = tx.send(line);
    });
    let line = rx.recv_timeout(timeout).unwrap_or_default();
    if let Some(rest) = line.trim().strip_prefix("ready ") {
        println!("{rest}");
        std::process::exit(0);
    }
    let tail = std::fs::read_to_string(log).unwrap_or_default();
    let lines: Vec<&str> = tail.lines().collect();
    for l in &lines[lines.len().saturating_sub(20)..] {
        eprintln!("{l}");
    }
    match line.trim() {
        "" => eprintln!(
            "{}",
            ErrorEnvelope::new(
                ErrorCode::BackendUnavailable,
                format!(
                    "instance exited or timed out during load; log: {}",
                    log.display()
                ),
            )
        ),
        envelope => eprintln!("{envelope}"),
    }
    std::process::exit(1)
}
