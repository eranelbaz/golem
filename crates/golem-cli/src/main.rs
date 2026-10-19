//! `golem` CLI entry point (tech design §9): run, serve, plan, list, stop,
//! optimize.
//! Bare `golem` (no subcommand) prints the version.
//! Future: `golem import` (§5.5 store intake: hash + CRC + mirror a shard
//! set into `$GOLEM_DATA_DIR`; `golem_store::import` already implements it).

use clap::{Parser, Subcommand};

mod config;
mod list;
mod optimize;
mod plan;
mod profile;
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
    about = "golem CLI: run, serve, plan, list, stop, optimize"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run a local GGUF model (greedy sampling); `--parity-dump
    /// <FILE>` emits a T2 parity dump (JSONL) instead of streaming text.
    ///
    /// Ctrl-C during a chat reply cancels that turn and keeps the chat;
    /// Ctrl-C at the prompt, SIGTERM, or any signal during `--prompt` exits
    /// with 128 + signal (130 SIGINT, 143 SIGTERM). A second signal before
    /// the first is handled kills the process.
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
    /// Stop a running instance by id or model name (drains first).
    ///
    /// `serve` drains over its control socket; `run` gets SIGTERM and up to
    /// 60 s to tear down before SIGKILL (which may land mid-load); other
    /// kinds get 5 s.
    Stop {
        #[command(flatten)]
        args: stop::StopArgs,
    },
    /// Sweep the scheduling knobs on this machine and save the winner as
    /// the model's profile (loaded by run/serve below flags, env and file).
    Optimize {
        #[command(flatten)]
        args: optimize::OptimizeArgs,
    },
}

/// Default filter: the CLI's own config-file and optimize-profile lines are
/// INFO.
const CLI_LOG: &str = "warn,golem=info";

/// `serve` default filter: [`CLI_LOG`] plus request lines (method, URI,
/// status, tok/s) and cancels at INFO.
const SERVE_LOG: &str = "warn,golem=info,golem_server=info,tower_http=info,golem_engine::slot=info";

/// Engine diagnostics (store health transitions, pool/stream warnings) go
/// to stderr; `RUST_LOG` overrides `default`. `json`: one JSON object per
/// line (`serve` off a TTY or detached).
fn init_tracing(json: bool, default: &str) {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default));
    let b = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(filter);
    let _ = if json {
        b.json().try_init()
    } else {
        b.with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
            .try_init()
    };
}

/// `serve` logs JSON when detached or when stderr is not a TTY.
fn serve_json_logs(detach: bool) -> bool {
    detach || !std::io::IsTerminal::is_terminal(&std::io::stderr())
}

fn main() {
    let cli = Cli::parse();
    // Detach forks before tracing, tokio or any other thread exists.
    if let Some(Command::Serve { args }) = &cli.command
        && args.detach
    {
        detach(args);
    }
    match &cli.command {
        Some(Command::Serve { args }) => init_tracing(serve_json_logs(args.detach), SERVE_LOG),
        _ => init_tracing(false, CLI_LOG),
    }
    let status = match cli.command {
        None => {
            println!("golem {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        Some(Command::Run { args, .. }) => match run::run(&args) {
            // No model in any layer → print the run help (every valid flag)
            // instead of a bare "no model" error.
            Err(e) if e.is::<config::NoModel>() => {
                use clap::CommandFactory as _;
                let mut cmd = Cli::command();
                cmd.find_subcommand_mut("run")
                    .expect("run subcommand exists")
                    .print_help()
                    .expect("write help");
                std::process::exit(0);
            }
            r => r,
        },
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
        Some(Command::Optimize { args }) => optimize::optimize(&args),
    };
    if let Err(e) = status {
        // A signal is the user's choice, not a failure: no envelope, the
        // shell convention 128 + signal (130 SIGINT, 143 SIGTERM).
        if let Some(sys::Interrupted(sig)) = e.downcast_ref() {
            eprintln!("{e}");
            std::process::exit(128 + sig);
        }
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
    let code = if matches!(
        e.downcast_ref::<golem_server::ServeError>(),
        Some(golem_server::ServeError::StorageUnavailable)
    ) {
        ErrorCode::StorageUnavailable
    } else if e.is::<golem_server::ServeError>() {
        ErrorCode::BackendUnavailable
    } else if matches!(
        e.downcast_ref::<golem_registry::RegistryError>(),
        Some(golem_registry::RegistryError::InstanceNotFound(_))
    ) {
        ErrorCode::InstanceNotFound
    } else if let Some(golem_registry::RegistryError::Control(env)) = e.downcast_ref() {
        return env.clone();
    } else if e.is::<stop::StopTimeout>() {
        ErrorCode::StopTimeout
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

/// `serve --detach`: the parent waits on a pipe for the grandchild's
/// `pid <pid>` line, then one line (`ready <id> <endpoint>` or an
/// `ErrorEnvelope`), prints `<id> <endpoint>` and exits; the setsid'd
/// grandchild returns and serves.
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
            init_tracing(true, SERVE_LOG);
            serve_detached(args, id, wr)
        }
    }
}

/// Grandchild: serves, reporting readiness (or the load error) on `wr`.
fn serve_detached(args: &serve::ServeArgs, id: String, mut wr: std::fs::File) -> ! {
    // The parent kills us by this pid if the load outlives its timeout.
    if writeln!(wr, "pid {}", std::process::id()).is_err() {
        std::process::exit(1);
    }
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

/// Parent: the grandchild's pid, then one line, or EOF / timeout → failure
/// with the log tail. On timeout the grandchild is SIGKILLed and its entry
/// reaped, so no orphan outlives the failed start.
fn wait_ready(rd: std::fs::File, log: &std::path::Path, timeout: Duration) -> ! {
    let mut rd = std::io::BufReader::new(rd);
    let mut line = String::new();
    let _ = rd.read_line(&mut line);
    let pid = line
        .trim()
        .strip_prefix("pid ")
        .and_then(|p| p.parse::<u32>().ok());
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut line = String::new();
        let _ = rd.read_line(&mut line);
        let _ = tx.send(line);
    });
    let (line, timed_out) = match rx.recv_timeout(timeout) {
        Ok(l) => (l, false),
        Err(_) => (String::new(), true),
    };
    if let Some(rest) = line.trim().strip_prefix("ready ") {
        println!("{rest}");
        std::process::exit(0);
    }
    if let (true, Some(pid)) = (timed_out, pid) {
        let _ = sys::kill(pid);
        // Reap the killed grandchild's entry (owner dead once it is gone).
        let t = std::time::Instant::now();
        while golem_registry::pid_start_time(pid).is_ok() && t.elapsed() < Duration::from_secs(5) {
            std::thread::sleep(Duration::from_millis(20));
        }
        let _ = golem_registry::Registry::open().and_then(|r| r.list_live());
        eprintln!("golem: load timed out; killed instance pid {pid}");
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_maps_stop_timeout_and_storage_unavailable() {
        let e = stop::StopTimeout::for_test("i1");
        assert_eq!(envelope(&e).error.code, ErrorCode::StopTimeout);
        let e = golem_server::ServeError::StorageUnavailable;
        assert_eq!(envelope(&e).error.code, ErrorCode::StorageUnavailable);
        let e = golem_server::ServeError::WorkerDied("fatal");
        assert_eq!(envelope(&e).error.code, ErrorCode::BackendUnavailable);
    }
}
