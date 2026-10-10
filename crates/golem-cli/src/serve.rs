//! `golem serve` (tech design §8–§9): OpenAI-compatible HTTP server over one
//! loaded model.

use std::error::Error;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use golem_engine::config::EngineConfig;
use golem_engine::gguf::read_metadata;
use golem_llama_sys::{Backend, substrate_backend};
use golem_registry::{EffectiveConfig, Guard, Instance, Registry};
use golem_server::{ReasoningMode, ServeError, ServerConfig, spawn_worker};
use tokio::signal::unix::{SignalKind, signal};

use crate::run::{EngineArgs, engine_config};

/// Arguments of the `serve` subcommand.
#[derive(clap::Args)]
pub struct ServeArgs {
    /// Path to the GGUF model file (first shard of a split).
    pub model: PathBuf,
    /// Bind address [default: 127.0.0.1].
    #[arg(long)]
    pub host: Option<IpAddr>,
    /// Bind port; 0 picks a free one (printed as `listening on ADDR`)
    /// [default: 0].
    #[arg(long)]
    pub port: Option<u16>,
    /// Decode slots batched by the scheduler. Values > 1 are refused until
    /// the `parallel ∈ {2,4}` HW leg passes (`docs/baselines/m4.4-batching.md`,
    /// `m4.6-plan-t8.md`); `GOLEM_ALLOW_PARALLEL=1` lifts that for testing.
    /// [default: 1]
    #[arg(long, value_parser = clap::value_parser!(u32).range(1..=64))]
    pub parallel: Option<u32>,
    /// Model id served on the API; default GGUF `general.name`, else the
    /// file stem.
    #[arg(long)]
    pub model_name: Option<String>,
    /// Reasoning delivery: `separate` → `reasoning_content`, `inline` →
    /// left in `content` with its markers [default: separate].
    #[arg(long, value_enum)]
    pub reasoning: Option<Reasoning>,
    /// Fork into the background once loaded: prints `<id> <endpoint>` and
    /// exits; the server logs to `$GOLEM_DATA_DIR/logs/<id>.log`.
    #[arg(long)]
    pub detach: bool,
    /// `--detach`: seconds to wait for the model to load.
    #[arg(long, value_name = "SECS", default_value_t = 600)]
    pub load_timeout: u64,
    /// Skip the O5 admission check against running instances.
    #[arg(long)]
    pub allow_overcommit: bool,
    /// Require `Authorization: Bearer <KEY>` on `/v1/*` and `/metrics`
    /// (`/health` stays open). The flag is visible to every user in `ps`:
    /// prefer `GOLEM_API_KEY` or `api_key` in `golem.toml`.
    #[arg(long, value_name = "KEY")]
    pub api_key: Option<String>,
    #[command(flatten)]
    pub engine: EngineArgs,
}

/// Manual so the API key never reaches a log line.
impl std::fmt::Debug for ServeArgs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServeArgs")
            .field("model", &self.model)
            .field("host", &self.host)
            .field("port", &self.port)
            .field("parallel", &self.parallel)
            .field("model_name", &self.model_name)
            .field("reasoning", &self.reasoning)
            .field("detach", &self.detach)
            .field("load_timeout", &self.load_timeout)
            .field("allow_overcommit", &self.allow_overcommit)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("engine", &self.engine)
            .finish()
    }
}

/// `--reasoning` values.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum Reasoning {
    /// Reasoning stays in `content`.
    Inline,
    /// Reasoning goes to `reasoning_content`.
    Separate,
}

impl std::str::FromStr for Reasoning {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        <Self as clap::ValueEnum>::from_str(s, true)
    }
}

/// `serve` flags resolved through §9.4 layers (flag > env > file > profile
/// > default).
struct Listen {
    host: IpAddr,
    port: u16,
    parallel: u32,
    reasoning: Reasoning,
    model_name: Option<String>,
}

fn listen(
    args: &ServeArgs,
    layers: &crate::config::Layers<'_>,
) -> Result<Listen, crate::config::ConfigError> {
    Ok(Listen {
        host: layers.get(args.host, "host", IpAddr::from([127, 0, 0, 1]))?,
        port: layers.get(args.port, "port", 0)?,
        parallel: layers.get(args.parallel, "parallel", 1)?,
        reasoning: layers.get(args.reasoning, "reasoning", Reasoning::Separate)?,
        model_name: layers.opt(args.model_name.clone(), "model_name")?,
    })
}

/// The requested (pre-RAM-fit) configuration identity; `snapshot_ram_bytes`
/// is the resolved request, since the fitted arena tracks free RAM.
pub fn effective(
    config: &EngineConfig,
    snapshot_ram_bytes: Option<u64>,
) -> Result<EffectiveConfig, Box<dyn Error>> {
    let path = std::fs::canonicalize(&config.model_path)?;
    let meta = std::fs::metadata(&path)?;
    Ok(EffectiveConfig {
        model_size: meta.len(),
        model_mtime: meta
            .modified()?
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()),
        model_path: path,
        ctx: config.n_ctx,
        parallel: config.parallel,
        ram_experts_gib: config.ram_experts_gib,
        snapshot_ram_bytes,
    })
}

/// Admits and registers a `loading` entry for `config` (§10 O5): planned
/// VRAM is the §10 device total when layers are offloaded to a GPU, planned
/// RAM the pool + snapshots + staging ring + host overhead.
/// Target: pool 63 + snapshots 4 + staging 3.7 + host 2 = 72.7 GiB RAM, and
/// the whole 24 GiB device (the expert cache takes what the rest leaves).
pub fn register(
    reg: &Registry,
    id: String,
    kind: &str,
    model: String,
    config: &EngineConfig,
    eff: &EffectiveConfig,
    allow_overcommit: bool,
) -> Result<Guard, Box<dyn Error>> {
    let mach = crate::plan::machine(None);
    let inst = instance(id, kind, model, config, eff, &mach)?;
    Ok(reg.admit_and_register(&inst, &mach, allow_overcommit)?)
}

/// The `loading` registry entry [`register`] admits, priced on `mach`.
pub fn instance(
    id: String,
    kind: &str,
    model: String,
    config: &EngineConfig,
    eff: &EffectiveConfig,
    mach: &golem_plan::Machine,
) -> Result<Instance, Box<dyn Error>> {
    let mach = *mach;
    let facts = config.model_facts.ok_or("model facts unresolved")?;
    let input = golem_plan::PlanInput {
        ctx: config.n_ctx,
        parallel: config.parallel,
        prefill_ubatch: config.prefill_ubatch.min(config.n_ctx),
        prefill_stream: config.prefill_stream,
        ram_experts: config.ram_experts_gib << 30,
        snapshot_ram: config.snapshot_ram_bytes.unwrap_or(0),
        ws: golem_plan::WS_UNMEASURED,
        ..Default::default()
    };
    let planned_vram = if config.n_gpu_layers > 0 && substrate_backend() == Backend::Hip {
        golem_plan::total(&golem_plan::vram_terms(&facts, &mach, &input))
    } else {
        0
    };
    // `ram_terms` substitutes the default pool for 0; no pool means 0 here.
    let planned_ram = golem_plan::ram_terms(&facts, &mach, &input)
        .iter()
        .filter(|t| t.name != "ram_pool" || config.ram_experts_gib > 0)
        .map(|t| t.bytes)
        .sum();
    let pid = std::process::id();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    Ok(Instance {
        id,
        pid,
        pid_start_time: golem_registry::pid_start_time(pid)?,
        kind: kind.into(),
        endpoint: String::new(),
        model,
        config_hash: eff.hash(),
        started_at: now,
        planned_vram,
        planned_ram,
        health: "loading".into(),
        heartbeat_at: now,
        children: Vec::new(),
        config: Some(eff.clone()),
    })
}

/// Fresh instance id.
pub fn new_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..8].to_owned()
}

/// `--parallel > 1` is gated on the target HW leg (4 slots ≈ 30.7 unique
/// experts/layer/step vs 8 solo → up to ≈ 3.8× bytes/step on misses, so
/// aggregate tok/s may fall below solo); recommended only once measured.
fn check_parallel(parallel: u32, allowed: bool) -> Result<(), String> {
    if parallel > 1 && !allowed {
        return Err(format!(
            "--parallel {parallel}: batching > 1 awaits its HW leg \
             (docs/baselines/m4.4-batching.md); set GOLEM_ALLOW_PARALLEL=1 to override"
        ));
    }
    Ok(())
}

/// Loads the model, serves until SIGINT/SIGTERM or a control-socket `stop`,
/// then drains and tears down. The instance is registered in `reg` as `id`
/// for its whole life; `ready` gets the endpoint once it serves.
/// `Err(ServeError::ShutdownTimeout)` means the worker is still running:
/// the caller must exit the process without further teardown.
pub fn serve(
    args: &ServeArgs,
    reg: &Registry,
    id: String,
    ready: impl FnOnce(&str) + Send,
) -> Result<(), Box<dyn Error>> {
    let layers = crate::config::Layers::from_process()?
        .with_profile(&args.model, args.engine.no_optimize_profile);
    let l = listen(args, &layers)?;
    check_parallel(
        l.parallel,
        std::env::var("GOLEM_ALLOW_PARALLEL").as_deref() == Ok("1"),
    )?;
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let (config, requested_snapshot) = engine_config(
        &args.engine,
        &args.model,
        l.parallel,
        &layers,
        sys.available_memory(),
    )?;
    let api_key = layers.opt(args.api_key.clone(), "api_key")?;
    let model_name = match l.model_name {
        Some(n) => n,
        None => read_metadata(&args.model)?
            .kv_str("general.name")
            .map(str::to_owned)
            .unwrap_or_else(|| {
                args.model
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default()
            }),
    };
    let eff = effective(&config, requested_snapshot)?;
    let guard = Arc::new(register(
        reg,
        id,
        "serve",
        model_name.clone(),
        &config,
        &eff,
        args.allow_overcommit,
    )?);
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    // Installed before the worker exists so a failure here leaks nothing.
    let (mut term, mut int) = {
        let _rt = rt.enter();
        (
            signal(SignalKind::terminate())?,
            signal(SignalKind::interrupt())?,
        )
    };
    let worker = spawn_worker(config)?;
    let health = worker.health.clone();
    let cfg = ServerConfig {
        host: l.host,
        port: l.port,
        model_name,
        api_key,
        drain_grace: golem_server::DRAIN_GRACE,
        reasoning: match l.reasoning {
            Reasoning::Inline => ReasoningMode::Inline,
            Reasoning::Separate => ReasoningMode::Separate,
        },
    };
    rt.block_on(async {
        // SIGTERM keeps the §9.2 drain grace. Ctrl-C is interactive: it
        // cancels in-flight work at once, and a second Ctrl-C exits.
        let shutdown = async move {
            tokio::select! {
                _ = term.recv() => {}
                _ = int.recv() => {
                    health.begin_drain();
                    tokio::spawn(async move {
                        int.recv().await;
                        eprintln!("second interrupt: exiting without teardown");
                        crate::sys::exit_now(130);
                    });
                }
            }
        };
        golem_server::serve_registered(cfg, worker, shutdown, guard, |a| {
            ready(&format!("http://{a}"));
        })
        .await?;
        Ok::<_, ServeError>(())
    })?;
    Ok(())
}

/// `true` when logind keeps `$XDG_RUNTIME_DIR` past logout for this user.
pub fn lingering() -> bool {
    let user = std::env::var("USER").unwrap_or_default();
    std::process::Command::new("loginctl")
        .args(["show-user", &user, "-p", "Linger"])
        .output()
        .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).trim() == "Linger=yes")
}

/// Log file of a detached instance.
pub fn log_path(id: &str) -> Option<PathBuf> {
    let root = golem_store::data_dir()
        .or_else(|| std::env::var_os("HOME").map(|h| Path::new(&h).join(".local/share/golem")))?;
    Some(root.join("logs").join(format!("{id}.log")))
}

#[cfg(test)]
mod tests {
    #[test]
    fn parallel_above_one_needs_the_override() {
        assert!(super::check_parallel(1, false).is_ok());
        assert!(super::check_parallel(4, false).is_err());
        assert!(super::check_parallel(4, true).is_ok());
    }

    /// §9.4: `GOLEM_PORT` beats the file and the default; the flag beats it.
    #[test]
    fn port_precedence_flag_env_file_default() {
        #[derive(clap::Parser)]
        struct P {
            #[command(flatten)]
            a: super::ServeArgs,
        }
        let layers = |env: &'static dyn Fn(&str) -> Option<String>| crate::config::Layers {
            file: toml::from_str("port = 7000\nhost = \"0.0.0.0\"").unwrap(),
            profile: toml::Table::new(),
            env,
        };
        let env: &'static dyn Fn(&str) -> Option<String> =
            &|k| (k == "GOLEM_PORT").then(|| "8123".to_owned());
        let none: &'static dyn Fn(&str) -> Option<String> = &|_| None;
        let bare = <P as clap::Parser>::parse_from(["serve", "m.gguf"]).a;
        let flag = <P as clap::Parser>::parse_from(["serve", "m.gguf", "--port", "9"]).a;
        assert_eq!(super::listen(&bare, &layers(env)).unwrap().port, 8123);
        assert_eq!(super::listen(&flag, &layers(env)).unwrap().port, 9);
        let file = super::listen(&bare, &layers(none)).unwrap();
        assert_eq!((file.port, file.host.to_string()), (7000, "0.0.0.0".into()));
        let d = super::listen(&bare, &crate::config::Layers::none()).unwrap();
        assert_eq!((d.port, d.parallel), (0, 1));
    }

    #[test]
    fn debug_redacts_api_key() {
        #[derive(clap::Parser)]
        struct P {
            #[command(flatten)]
            a: super::ServeArgs,
        }
        let p = <P as clap::Parser>::parse_from(["serve", "m.gguf", "--api-key", "s3cret"]);
        let d = format!("{:?}", p.a);
        assert!(!d.contains("s3cret") && d.contains("<redacted>"), "{d}");
    }
}
