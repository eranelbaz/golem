//! Instance process: HTTP server, slot scheduler, decode loop (tech design sections 2 and 7).

mod app;
mod worker;

use std::future::Future;
use std::io;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

pub use app::{AppState, router};
pub use worker::{
    Event, HealthSnapshot, HealthState, Job, ServeState, ShutdownTimeout, WorkerHandle,
    spawn_worker,
};

/// Listener and API identity of one instance.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Bind address; `127.0.0.1` by default (CLI).
    pub host: IpAddr,
    /// Bind port; 0 picks a free one.
    pub port: u16,
    /// Model id clients must send and `/v1/models` lists.
    pub model_name: String,
    /// When set, `/v1/*` requires `Authorization: Bearer <key>`.
    pub api_key: Option<String>,
}

/// Why [`serve`] failed.
#[derive(Debug, thiserror::Error)]
pub enum ServeError {
    /// Bind or accept failure.
    #[error("http server: {0}")]
    Io(#[from] io::Error),
    /// The worker did not join within the shutdown bound; it is leaked and
    /// the process must exit without further teardown.
    #[error(transparent)]
    ShutdownTimeout(#[from] ShutdownTimeout),
}

/// Bound on the worker join at shutdown: one non-abortable `llama_decode`
/// step on the target is ≪ 60 s even cold.
const WORKER_JOIN_TIMEOUT: Duration = Duration::from_secs(60);

/// Binds `cfg.host:cfg.port`, prints `listening on {addr}` to stderr, serves
/// until `shutdown` resolves, then drains (in-flight job cancelled, answered
/// `golem_busy`) and joins the worker. Returns after the worker joined, so
/// `Session::drop` (§3.4 teardown order) has completed.
pub async fn serve(
    cfg: ServerConfig,
    worker: WorkerHandle,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<SocketAddr, ServeError> {
    let listener = tokio::net::TcpListener::bind((cfg.host, cfg.port)).await?;
    let addr = listener.local_addr()?;
    // Unconditional: launchers (and the pytest) wait on this line, and the
    // CLI's default log filter is `warn`.
    eprintln!("listening on {addr}");
    serve_listener(listener, cfg, worker, shutdown).await?;
    Ok(addr)
}

async fn serve_listener(
    listener: tokio::net::TcpListener,
    cfg: ServerConfig,
    worker: WorkerHandle,
    shutdown: impl Future<Output = ()> + Send + 'static,
) -> Result<(), ServeError> {
    let health = worker.health.clone();
    let state = AppState {
        cfg: Arc::new(cfg),
        jobs: worker.jobs.clone(),
        health: health.clone(),
    };
    let drain = health.clone();
    // The router (and its jobs sender clones) drops when this returns, so the
    // worker's `recv` sees `Disconnected` once the handle's sender drops too.
    axum::serve(listener, router(state))
        .with_graceful_shutdown(async move {
            shutdown.await;
            drain.begin_drain();
        })
        .await?;
    let h = health.clone();
    let joined = tokio::task::spawn_blocking(move || {
        worker.shutdown(|| h.cancel_in_flight(), WORKER_JOIN_TIMEOUT)
    })
    .await
    .map_err(io::Error::other)?;
    if let Err(e) = joined {
        tracing::error!("{e}");
        health.set(ServeState::Fatal);
        return Err(e.into());
    }
    Ok(())
}

/// Fixture config, in-process routers and fault injection for tests.
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use super::*;
    use golem_engine::config::EngineConfig;

    pub use crate::worker::{
        inject_slow_decode, inject_worker_panic, inject_worker_panic_after_first_token,
    };

    /// Non-pooled tiny-moe: pooled sessions refuse to coexist under
    /// `cargo test`; pooled paths are covered by engine tests.
    pub fn test_engine_config() -> EngineConfig {
        EngineConfig {
            model_path: xtask::tiny_moe::cached_fixture(),
            n_ctx: 4096,
            ..Default::default()
        }
    }

    /// `ServerConfig` serving the fixture as `tiny-moe` on an ephemeral port.
    pub fn test_server_config() -> ServerConfig {
        ServerConfig {
            host: IpAddr::from([127, 0, 0, 1]),
            port: 0,
            model_name: "tiny-moe".into(),
            api_key: None,
        }
    }

    /// Router over a freshly spawned worker; the worker thread is detached.
    pub async fn router_for_test(cfg: EngineConfig) -> axum::Router {
        let w = tokio::task::spawn_blocking(move || spawn_worker(cfg))
            .await
            .expect("spawn_worker task")
            .expect("tiny-moe worker loads");
        router(AppState {
            cfg: Arc::new(test_server_config()),
            jobs: w.jobs.clone(),
            health: w.health.clone(),
        })
    }

    /// Real socket server: returns its address, a shutdown trigger and the
    /// task resolving to `serve`'s result.
    pub async fn serve_on_port0(
        cfg: EngineConfig,
    ) -> (
        SocketAddr,
        tokio::sync::oneshot::Sender<()>,
        tokio::task::JoinHandle<Result<(), ServeError>>,
    ) {
        let worker = tokio::task::spawn_blocking(move || spawn_worker(cfg))
            .await
            .expect("spawn_worker task")
            .expect("tiny-moe worker loads");
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("local addr");
        let (tx, rx) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(serve_listener(
            listener,
            test_server_config(),
            worker,
            async move {
                let _ = rx.await;
            },
        ));
        (addr, tx, task)
    }
}

#[cfg(any(test, feature = "test-support"))]
pub use test_support::{router_for_test, test_engine_config};
