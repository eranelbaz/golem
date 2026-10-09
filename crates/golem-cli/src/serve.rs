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
#[derive(Debug, clap::Args)]
pub struct ServeArgs {
    /// Path to the GGUF model file (first shard of a split).
    pub model: PathBuf,
    /// Bind address.
    #[arg(long, default_value = "127.0.0.1")]
    pub host: IpAddr,
    /// Bind port; 0 picks a free one (printed as `listening on ADDR`).
    #[arg(long, default_value_t = 0)]
    pub port: u16,
    /// Decode slots; only 1 is supported.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..=1))]
    pub parallel: u32,
    /// Model id served on the API; default GGUF `general.name`, else the
    /// file stem.
    #[arg(long)]
    pub model_name: Option<String>,
    /// Reasoning delivery: `separate` → `reasoning_content`, `inline` →
    /// left in `content` with its markers.
    #[arg(long, value_enum, default_value_t = Reasoning::Separate)]
    pub reasoning: Reasoning,
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
    #[command(flatten)]
    pub engine: EngineArgs,
}

/// `--reasoning` values.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum Reasoning {
    /// Reasoning stays in `content`.
    Inline,
    /// Reasoning goes to `reasoning_content`.
    Separate,
}

/// The requested (pre-RAM-fit) configuration identity; `snapshot_ram_gib` is
/// the flag, since the resolved arena tracks free RAM.
pub fn effective(
    config: &EngineConfig,
    snapshot_ram_gib: Option<f64>,
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
        snapshot_ram_bytes: snapshot_ram_gib.map(|g| (g * (1u64 << 30) as f64) as u64),
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
    let inst = Instance {
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
    };
    Ok(reg.admit_and_register(&inst, &mach, allow_overcommit)?)
}

/// Fresh instance id.
pub fn new_id() -> String {
    uuid::Uuid::new_v4().simple().to_string()[..8].to_owned()
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
    let config = engine_config(&args.engine, &args.model, args.parallel)?;
    let model_name = match &args.model_name {
        Some(n) => n.clone(),
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
    let eff = effective(&config, args.engine.snapshot_ram_gib)?;
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
    let cfg = ServerConfig {
        host: args.host,
        port: args.port,
        model_name,
        api_key: None,
        reasoning: match args.reasoning {
            Reasoning::Inline => ReasoningMode::Inline,
            Reasoning::Separate => ReasoningMode::Separate,
        },
    };
    rt.block_on(async {
        let shutdown = async move {
            tokio::select! {
                _ = term.recv() => {}
                _ = int.recv() => {}
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
