//! Instance process: HTTP server, slot scheduler, decode loop (tech design sections 2 and 7).

mod app;
mod sse;
mod worker;

use std::future::Future;
use std::io;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

pub use app::{AppState, router};
pub use worker::{
    Event, HealthSnapshot, HealthState, Job, ReasoningMode, ServeState, ShutdownTimeout,
    WorkerHandle, spawn_worker,
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
    /// Reasoning delivery (`--reasoning`).
    pub reasoning: ReasoningMode,
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
    /// The decode worker died (panic or fatal engine error); torn down.
    #[error("decode worker died: {0}")]
    WorkerDied(&'static str),
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

/// Graceful `stop` drain bound (§9.2): the in-flight request may finish.
const STOP_DRAIN: Duration = Duration::from_secs(30);

/// [`serve`] for a registered instance: writes the bound endpoint and
/// health transitions into `guard`, and answers the control socket next to
/// the entry (`status`, `stop`) until shutdown. `ready` receives the bound
/// address once the socket is listening.
pub async fn serve_registered(
    cfg: ServerConfig,
    worker: WorkerHandle,
    shutdown: impl Future<Output = ()> + Send + 'static,
    guard: Arc<golem_registry::Guard>,
    ready: impl FnOnce(SocketAddr) + Send,
) -> Result<SocketAddr, ServeError> {
    let listener = tokio::net::TcpListener::bind((cfg.host, cfg.port)).await?;
    let addr = listener.local_addr()?;
    eprintln!("listening on {addr}");
    let sock = guard.sock_path();
    let _ = std::fs::remove_file(&sock);
    let control = tokio::net::UnixListener::bind(&sock)?;
    let health = worker.health.clone();
    guard.update(|i| {
        i.endpoint = format!("http://{addr}");
        i.health = health.get().as_str().into();
    });
    ready(addr);
    let (stop_tx, mut stop_rx) = tokio::sync::mpsc::channel::<()>(1);
    let ctl = tokio::spawn(control_loop(control, health.clone(), stop_tx));
    let (h, g) = (health.clone(), guard.clone());
    let sync = tokio::spawn(async move {
        loop {
            let s = h.get().as_str();
            if g.instance().health != s {
                g.update(|i| i.health = s.into());
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
    });
    let res = serve_listener(listener, cfg, worker, async move {
        tokio::select! {
            _ = shutdown => {}
            _ = stop_rx.recv() => {}
        }
    })
    .await;
    ctl.abort();
    sync.abort();
    res.map(|()| addr)
}

async fn control_loop(
    l: tokio::net::UnixListener,
    health: Arc<HealthState>,
    stop: tokio::sync::mpsc::Sender<()>,
) {
    while let Ok((s, _)) = l.accept().await {
        let Ok(s) = s
            .into_std()
            .and_then(|s| s.set_nonblocking(false).map(|()| s))
        else {
            continue;
        };
        let (h, st) = (health.clone(), stop.clone());
        tokio::task::spawn_blocking(move || control_conn(s, h, st));
    }
}

/// One control exchange; blocking std I/O on a blocking-pool thread (the
/// graceful stop waits out the drain).
fn control_conn(
    s: std::os::unix::net::UnixStream,
    health: Arc<HealthState>,
    stop: tokio::sync::mpsc::Sender<()>,
) {
    use golem_registry::{ControlRequest, StopMode};
    use std::io::{BufRead, Write};
    let mut line = String::new();
    let Ok(r) = s.try_clone() else { return };
    if std::io::BufReader::new(r).read_line(&mut line).is_err() {
        return;
    }
    let reply = match serde_json::from_str::<ControlRequest>(&line) {
        Ok(ControlRequest::Status) => {
            let snap = health.snapshot();
            serde_json::json!({
                "ok": true,
                "health": health.get().as_str(),
                "turns": snap.turns,
                "device_cache_skip_ratio": snap.device_cache_skip_ratio,
            })
        }
        Ok(ControlRequest::Stop { mode }) => {
            if mode == StopMode::Graceful {
                health.set(ServeState::Draining);
                let t = std::time::Instant::now();
                while health.busy() && t.elapsed() < STOP_DRAIN {
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
            let _ = stop.try_send(());
            serde_json::json!({ "ok": true })
        }
        Err(e) => serde_json::to_value(golem_api::ErrorEnvelope::new(
            golem_api::ErrorCode::InvalidRequest,
            format!("control request: {e}"),
        ))
        .unwrap_or_default(),
    };
    let _ = (&s).write_all(format!("{reply}\n").as_bytes());
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
    let served = axum::serve(listener, router(state))
        .with_graceful_shutdown(async move {
            tokio::select! {
                _ = shutdown => {}
                _ = worker_dead(&drain) => {}
            }
            drain.begin_drain();
        })
        .await;
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
    served?;
    match health.get() {
        s @ (ServeState::Fatal | ServeState::FatalIo) => Err(ServeError::WorkerDied(s.as_str())),
        _ => Ok(()),
    }
}

/// Resolves once the worker is fatal; polled since `fatal` is terminal and
/// 100 ms is noise against process teardown.
async fn worker_dead(health: &HealthState) {
    while !matches!(health.get(), ServeState::Fatal | ServeState::FatalIo) {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// Fixture config, in-process routers and fault injection for tests.
#[cfg(any(test, feature = "test-support"))]
pub mod test_support {
    use super::*;
    use golem_engine::config::EngineConfig;

    pub use crate::sse_support::{
        Logs, capture_logs, complete_once, drop_after_first_frame, normalize, sse_frames,
    };
    pub use crate::worker::{TestHooks, spawn_worker_with_hooks};

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
            reasoning: ReasoningMode::Separate,
        }
    }

    /// Router over a freshly spawned worker; the worker thread is detached.
    pub async fn router_for_test(cfg: EngineConfig) -> axum::Router {
        router_with_hooks(cfg, TestHooks::default()).await
    }

    /// [`router_for_test`] with fault injection on its worker.
    pub async fn router_with_hooks(cfg: EngineConfig, hooks: TestHooks) -> axum::Router {
        let w = tokio::task::spawn_blocking(move || spawn_worker_with_hooks(cfg, hooks))
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
        hooks: TestHooks,
    ) -> (
        SocketAddr,
        tokio::sync::oneshot::Sender<()>,
        tokio::task::JoinHandle<Result<(), ServeError>>,
    ) {
        let worker = tokio::task::spawn_blocking(move || spawn_worker_with_hooks(cfg, hooks))
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

/// Socket-level SSE helpers over one process-wide tiny-moe server (one
/// worker per test binary: each `Session` loads the fixture, and helper
/// calls in one test share it).
#[cfg(any(test, feature = "test-support"))]
pub mod sse_support {
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::{SocketAddr, TcpStream};
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::{Duration, Instant};

    use super::test_support::{TestHooks, test_engine_config, test_server_config};
    use super::*;

    /// 20 ms/piece: a 4096-token stream stays in flight for seconds, so a
    /// disconnect after the first frame lands mid-generation.
    const SLOW_MS: u64 = 20;

    fn server() -> SocketAddr {
        static ADDR: OnceLock<SocketAddr> = OnceLock::new();
        *ADDR.get_or_init(|| {
            let (tx, rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                let rt = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()
                    .expect("runtime");
                rt.block_on(async move {
                    let worker = crate::worker::spawn_worker_with_hooks(
                        test_engine_config(),
                        TestHooks::slow_decode(SLOW_MS),
                    )
                    .expect("tiny-moe worker loads");
                    let l = tokio::net::TcpListener::bind("127.0.0.1:0")
                        .await
                        .expect("bind");
                    tx.send(l.local_addr().expect("addr")).expect("addr send");
                    let _ = serve_listener(l, test_server_config(), worker, std::future::pending())
                        .await;
                });
            });
            rx.recv().expect("server started")
        })
    }

    fn post(body: &str) -> TcpStream {
        let mut s = TcpStream::connect(server()).expect("connect");
        write!(
            s,
            "POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\n\
             content-length: {}\r\nconnection: close\r\n\r\n{body}",
            body.len()
        )
        .expect("write");
        s
    }

    /// Posts `body` and returns the `data:` payloads of the SSE response, in
    /// order (keep-alive comments dropped). Retries a 503 (slot still busy
    /// from a previous cancelled request) for up to 10 s.
    pub async fn sse_frames(body: &str) -> Vec<String> {
        let body = body.to_owned();
        tokio::task::spawn_blocking(move || {
            let t = Instant::now();
            loop {
                let mut raw = String::new();
                post(&body).read_to_string(&mut raw).expect("read");
                if raw.starts_with("HTTP/1.1 503") && t.elapsed() < Duration::from_secs(10) {
                    std::thread::sleep(Duration::from_millis(50));
                    continue;
                }
                assert!(raw.starts_with("HTTP/1.1 200"), "{raw}");
                assert!(raw.contains("content-type: text/event-stream"), "{raw}");
                return dechunk(raw.split_once("\r\n\r\n").map_or("", |(_, b)| b))
                    .lines()
                    .filter_map(|l| l.strip_prefix("data: "))
                    .map(str::to_owned)
                    .collect();
            }
        })
        .await
        .expect("client task")
    }

    /// HTTP/1.1 chunked transfer decoding (the SSE body is chunked).
    fn dechunk(mut b: &str) -> String {
        let mut out = String::new();
        while let Some((len, rest)) = b.split_once("\r\n") {
            let Ok(n) = usize::from_str_radix(len.trim(), 16) else {
                break;
            };
            if n == 0 || rest.len() < n {
                break;
            }
            out.push_str(&rest[..n]);
            b = rest[n..].trim_start_matches("\r\n");
        }
        out
    }

    /// Golden form of `frames`: `id`/`created` replaced, one frame per line.
    pub fn normalize(frames: &[String]) -> String {
        frames
            .iter()
            .map(|f| match serde_json::from_str::<serde_json::Value>(f) {
                Ok(mut v) if v.is_object() => {
                    v["id"] = "ID".into();
                    v["created"] = 0.into();
                    v.to_string()
                }
                _ => f.clone(),
            })
            .map(|l| l + "\n")
            .collect()
    }

    /// Opens a streaming request, reads until the first generated frame
    /// (past the role chunk, which is sent before the worker starts), then
    /// drops the connection.
    pub async fn drop_after_first_frame(body: &str) {
        let body = body.to_owned();
        tokio::task::spawn_blocking(move || {
            let mut r = BufReader::new(post(&body));
            let mut line = String::new();
            while r.read_line(&mut line).expect("read") > 0 {
                if line.contains("data: ") && line.contains("\"content\"") {
                    return;
                }
                line.clear();
            }
            panic!("stream closed before the first frame");
        })
        .await
        .expect("client task");
    }

    /// One short non-streaming request; `Ok` on HTTP 200 (retrying a busy
    /// 503 for up to 10 s), else the raw response.
    pub async fn complete_once() -> Result<(), String> {
        tokio::task::spawn_blocking(|| {
            let body = r#"{"model":"tiny-moe","messages":[{"role":"user","content":"hi"}],"max_tokens":2}"#;
            let t = Instant::now();
            loop {
                let mut raw = String::new();
                post(body).read_to_string(&mut raw).map_err(|e| e.to_string())?;
                if raw.starts_with("HTTP/1.1 200") {
                    return Ok(());
                }
                if !raw.starts_with("HTTP/1.1 503") || t.elapsed() > Duration::from_secs(10) {
                    return Err(raw);
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        })
        .await
        .expect("client task")
    }

    /// Captured `tracing` output of every thread (process-global subscriber,
    /// installed once).
    #[derive(Clone)]
    pub struct Logs(Arc<Mutex<Vec<u8>>>);

    impl Logs {
        /// Polls until a captured line contains `needle` or `timeout` passes.
        pub fn wait_for(&self, needle: &str, timeout: Duration) -> bool {
            let t = Instant::now();
            loop {
                if String::from_utf8_lossy(&self.0.lock().unwrap_or_else(|p| p.into_inner()))
                    .contains(needle)
                {
                    return true;
                }
                if t.elapsed() > timeout {
                    return false;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
        }
    }

    struct LogWriter(Arc<Mutex<Vec<u8>>>);

    impl Write for LogWriter {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Installs (once) a global `info` subscriber writing into a buffer.
    pub fn capture_logs() -> Logs {
        static LOGS: OnceLock<Arc<Mutex<Vec<u8>>>> = OnceLock::new();
        let buf = LOGS
            .get_or_init(|| {
                let buf = Arc::new(Mutex::new(Vec::new()));
                let w = buf.clone();
                tracing_subscriber::fmt()
                    .with_max_level(tracing::Level::INFO)
                    .with_ansi(false)
                    .with_writer(move || LogWriter(w.clone()))
                    .init();
                buf
            })
            .clone();
        Logs(buf)
    }
}
