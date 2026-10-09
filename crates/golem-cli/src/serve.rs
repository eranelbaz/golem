//! `golem serve` (tech design §8–§9): OpenAI-compatible HTTP server over one
//! loaded model.

use std::error::Error;
use std::net::IpAddr;
use std::path::PathBuf;

use golem_engine::gguf::read_metadata;
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

/// Loads the model, serves until SIGINT/SIGTERM, then drains and tears down.
/// `Err(ServeError::ShutdownTimeout)` means the worker is still running:
/// the caller must exit the process without further teardown.
pub fn serve(args: &ServeArgs) -> Result<(), Box<dyn Error>> {
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
        golem_server::serve(cfg, worker, shutdown).await?;
        Ok::<_, ServeError>(())
    })?;
    Ok(())
}
