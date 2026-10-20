//! Instance process: HTTP server, slot scheduler, decode loop (tech design sections 2 and 7).

mod anthropic;
mod app;
mod metrics;
mod sse;
mod worker;

use std::future::Future;
use std::io;
use std::net::{IpAddr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

pub use app::{AppState, router};
pub use metrics::MetricsSnapshot;
pub use worker::{
    Event, HealthSnapshot, HealthState, Job, ReasoningMode, ServeState, ShutdownTimeout, TurnSinks,
    WorkerHandle, check_sinks, spawn_worker, spawn_worker_models,
};

/// Listener and API identity of one instance.
#[derive(Clone)]
pub struct ServerConfig {
    /// Bind address; `127.0.0.1` by default (CLI).
    pub host: IpAddr,
    /// Bind port; 0 picks a free one.
    pub port: u16,
    /// Model id clients must send and `/v1/models` lists.
    pub model_name: String,
    /// Further served models (§8.1), loaded on demand by the worker.
    pub models: Vec<String>,
    /// When set, `/v1/*` and `/metrics` require `Authorization: Bearer <key>`
    /// (`/health` stays open).
    pub api_key: Option<String>,
    /// Reasoning delivery (`--reasoning`).
    pub reasoning: ReasoningMode,
    /// Graceful-shutdown grace (§9.2): the in-flight request may finish
    /// within it before it is cancelled; 0 (`stop --now`) cancels at once.
    pub drain_grace: Duration,
}

/// Manual so the API key never reaches a log line.
impl std::fmt::Debug for ServerConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ServerConfig")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("model_name", &self.model_name)
            .field("models", &self.models)
            .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
            .field("reasoning", &self.reasoning)
            .field("drain_grace", &self.drain_grace)
            .finish()
    }
}

/// Default `drain_grace` (§9.2).
pub const DRAIN_GRACE: Duration = Duration::from_secs(30);

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
    /// Every replica of a shard became unreadable (`fatal-io`, §5.2).
    #[error("storage unavailable: every replica of a shard is unreadable")]
    StorageUnavailable,
}

/// Bound on the worker join at shutdown: one non-abortable `llama_decode`
/// step on the target is ≪ 60 s even cold.
const WORKER_JOIN_TIMEOUT: Duration = Duration::from_secs(60);

/// Bound on closing connections once the drain ended: a cancelled stream
/// stops within one plane (≈0.11 s cold) and only its last frames remain to
/// flush; a reader that stalls past 10 s is cut (30 s grace + 10 s + 60 s
/// join = 100 s, under `golem stop`'s 180 s exit wait).
const CLOSE_DEADLINE: Duration = Duration::from_secs(10);

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
    // Owner-only: the socket stops the instance.
    std::fs::set_permissions(&sock, std::os::unix::fs::PermissionsExt::from_mode(0o600))?;
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

/// One control exchange; blocking std I/O on a blocking-pool thread.
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
            // Ack at once; `serve_listener` runs the drain (grace, then
            // cancel) and `golem stop` follows it through the registry.
            if mode == StopMode::Now {
                health.begin_drain();
            } else {
                health.set(ServeState::Draining);
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
    let grace = cfg.drain_grace;
    let state = AppState {
        cfg: Arc::new(cfg),
        jobs: worker.jobs.clone(),
        health: health.clone(),
    };
    let drain = health.clone();
    // The router (and its jobs sender clones) drops when this returns, so the
    // worker's `recv` sees `Disconnected` once the handle's sender drops too.
    let served = serve_bounded(
        listener,
        router(state),
        async move {
            tokio::select! {
                _ = shutdown => {}
                _ = worker_dead(&drain) => {}
            }
            // New requests 503 from here; in-flight and queued ones get `grace`.
            drain.set(ServeState::Draining);
            let t = tokio::time::Instant::now();
            while drain.pending() && t.elapsed() < grace {
                tokio::time::sleep(Duration::from_millis(50)).await;
            }
            tracing::info!("draining → cancelling");
            drain.begin_drain();
        },
        CLOSE_DEADLINE,
    )
    .await;
    health.set(ServeState::Releasing);
    tracing::info!("cancelling → releasing");
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
        s @ ServeState::Fatal => Err(ServeError::WorkerDied(s.as_str())),
        ServeState::FatalIo => Err(ServeError::StorageUnavailable),
        _ => Ok(()),
    }
}

/// Serves `app` until `shutdown` resolves, then closes gracefully; a
/// connection still open `close_deadline` later (a client that stopped
/// reading a stream) is aborted, so shutdown never waits on a peer.
async fn serve_bounded(
    listener: tokio::net::TcpListener,
    app: axum::Router,
    shutdown: impl Future<Output = ()> + Send + 'static,
    close_deadline: Duration,
) -> io::Result<()> {
    use std::future::IntoFuture;
    let (kill, killed) = tokio::sync::watch::channel(false);
    let drained = Arc::new(tokio::sync::Notify::new());
    let signal = drained.clone();
    let served = axum::serve(KillableListener { listener, killed }, app)
        .with_graceful_shutdown(async move {
            shutdown.await;
            signal.notify_one();
        })
        .into_future();
    tokio::pin!(served);
    tokio::select! {
        r = &mut served => return r,
        _ = async {
            drained.notified().await;
            tokio::time::sleep(close_deadline).await;
        } => {}
    }
    tracing::warn!("connections still open {close_deadline:?} after drain; aborting them");
    let _ = kill.send(true);
    served.await
}

/// TCP listener whose connections fail with `ConnectionAborted` once the
/// kill flag is set (wakes a write blocked on a stalled peer too).
struct KillableListener {
    listener: tokio::net::TcpListener,
    killed: tokio::sync::watch::Receiver<bool>,
}

impl axum::serve::Listener for KillableListener {
    type Io = KillableIo;
    type Addr = SocketAddr;

    async fn accept(&mut self) -> (KillableIo, SocketAddr) {
        loop {
            match self.listener.accept().await {
                Ok((io, addr)) => {
                    let mut rx = self.killed.clone();
                    let kill = Box::pin(async move {
                        // A dropped sender means serve returned: never fire.
                        if rx.wait_for(|&k| k).await.is_err() {
                            std::future::pending::<()>().await;
                        }
                    });
                    return (KillableIo { io, kill }, addr);
                }
                // Transient (EMFILE, aborted handshake): back off, retry.
                Err(e) => {
                    tracing::warn!("accept: {e}");
                    tokio::time::sleep(Duration::from_millis(50)).await;
                }
            }
        }
    }

    fn local_addr(&self) -> io::Result<SocketAddr> {
        self.listener.local_addr()
    }
}

/// One accepted connection plus its kill signal (one boxed future per
/// connection, never per request byte).
struct KillableIo {
    io: tokio::net::TcpStream,
    kill: std::pin::Pin<Box<dyn Future<Output = ()> + Send>>,
}

impl KillableIo {
    fn killed(&mut self, cx: &mut std::task::Context<'_>) -> bool {
        self.kill.as_mut().poll(cx).is_ready()
    }
}

fn aborted() -> io::Error {
    io::Error::new(io::ErrorKind::ConnectionAborted, "shutdown close deadline")
}

impl tokio::io::AsyncRead for KillableIo {
    fn poll_read(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &mut tokio::io::ReadBuf<'_>,
    ) -> std::task::Poll<io::Result<()>> {
        if self.killed(cx) {
            return std::task::Poll::Ready(Err(aborted()));
        }
        std::pin::Pin::new(&mut self.io).poll_read(cx, buf)
    }
}

impl tokio::io::AsyncWrite for KillableIo {
    fn poll_write(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<io::Result<usize>> {
        if self.killed(cx) {
            return std::task::Poll::Ready(Err(aborted()));
        }
        std::pin::Pin::new(&mut self.io).poll_write(cx, buf)
    }

    fn poll_flush(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<io::Result<()>> {
        if self.killed(cx) {
            return std::task::Poll::Ready(Err(aborted()));
        }
        std::pin::Pin::new(&mut self.io).poll_flush(cx)
    }

    fn poll_shutdown(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<io::Result<()>> {
        std::pin::Pin::new(&mut self.io).poll_shutdown(cx)
    }
}

/// Resolves once the worker is fatal; polled since `fatal` is terminal and
/// 100 ms is noise against process teardown. `fatal-io` keeps serving
/// (503s and `/health`, §5.2).
async fn worker_dead(health: &HealthState) {
    while health.get() != ServeState::Fatal {
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
            models: Vec::new(),
            api_key: None,
            reasoning: ReasoningMode::Separate,
            // Shutdown tests exercise the cancel path; drain tests set it.
            drain_grace: Duration::ZERO,
        }
    }

    /// GET `path` on a fresh fixture router (all layers offloaded on HIP
    /// builds, so I2 placement is meaningful); the body as text.
    pub async fn get_text(path: &str) -> String {
        use tower::ServiceExt;
        let cfg = EngineConfig {
            n_gpu_layers: if cfg!(golem_backend_hip) { 99 } else { 0 },
            ..test_engine_config()
        };
        let r = router_for_test(cfg)
            .await
            .oneshot(
                axum::http::Request::get(path)
                    .body(axum::body::Body::empty())
                    .expect("request"),
            )
            .await
            .expect("response");
        String::from_utf8(
            axum::body::to_bytes(r.into_body(), 1 << 20)
                .await
                .expect("body")
                .to_vec(),
        )
        .expect("utf-8 body")
    }

    /// Router over a freshly spawned worker; the worker thread is detached.
    pub async fn router_for_test(cfg: EngineConfig) -> axum::Router {
        router_with_hooks(cfg, TestHooks::default()).await
    }

    /// [`router_for_test`] with fault injection on its worker.
    pub async fn router_with_hooks(cfg: EngineConfig, hooks: TestHooks) -> axum::Router {
        let w = tokio::task::spawn_blocking(move || {
            spawn_worker_with_hooks(cfg, hooks, TurnSinks::default())
        })
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
        serve_with(cfg, hooks, test_server_config()).await
    }

    /// [`serve_on_port0`] with an explicit `ServerConfig` (port ignored).
    pub async fn serve_with(
        cfg: EngineConfig,
        hooks: TestHooks,
        scfg: ServerConfig,
    ) -> (
        SocketAddr,
        tokio::sync::oneshot::Sender<()>,
        tokio::task::JoinHandle<Result<(), ServeError>>,
    ) {
        serve_with_sinks(cfg, hooks, scfg, TurnSinks::default()).await
    }

    /// [`serve_with`] recording each turn into `sinks`.
    pub async fn serve_with_sinks(
        cfg: EngineConfig,
        hooks: TestHooks,
        scfg: ServerConfig,
        sinks: TurnSinks,
    ) -> (
        SocketAddr,
        tokio::sync::oneshot::Sender<()>,
        tokio::task::JoinHandle<Result<(), ServeError>>,
    ) {
        let worker =
            tokio::task::spawn_blocking(move || spawn_worker_with_hooks(cfg, hooks, sinks))
                .await
                .expect("spawn_worker task")
                .expect("tiny-moe worker loads");
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind");
        let addr = listener.local_addr().expect("local addr");
        let (tx, rx) = tokio::sync::oneshot::channel();
        let task = tokio::spawn(serve_listener(listener, scfg, worker, async move {
            let _ = rx.await;
        }));
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
                        TurnSinks::default(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};

    #[test]
    fn debug_redacts_api_key() {
        let cfg = ServerConfig {
            api_key: Some("s3cret".into()),
            ..test_support::test_server_config()
        };
        let d = format!("{cfg:?}");
        assert!(!d.contains("s3cret") && d.contains("<redacted>"), "{d}");
    }

    /// A client that never reads an endless stream fills the socket
    /// buffers and parks the server's write; the drain must still end
    /// within the close deadline, and a normal in-flight request completes.
    #[tokio::test(flavor = "multi_thread")]
    async fn close_deadline_cuts_a_stalled_stream_reader() {
        use axum::routing::get;
        let chunk = axum::body::Bytes::from(vec![b'x'; 64 << 10]);
        let app = axum::Router::new()
            .route(
                "/ok",
                get(|| async {
                    tokio::time::sleep(Duration::from_millis(500)).await;
                    "ok"
                }),
            )
            .route(
                "/stream",
                get(move || async move {
                    let it = std::iter::repeat_with(move || {
                        Ok::<_, std::convert::Infallible>(chunk.clone())
                    });
                    axum::body::Body::from_stream(tokio_stream::iter(it))
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = tokio::sync::oneshot::channel::<()>();
        let deadline = Duration::from_secs(1);
        let task = tokio::spawn(serve_bounded(
            listener,
            app,
            async move {
                let _ = rx.await;
            },
            deadline,
        ));
        let req = move |path: &str| {
            let mut s = std::net::TcpStream::connect(addr).unwrap();
            write!(
                s,
                "GET {path} HTTP/1.1\r\nhost: x\r\nconnection: close\r\n\r\n"
            )
            .unwrap();
            s
        };
        let _stalled = req("/stream");
        tokio::time::sleep(Duration::from_millis(300)).await;
        let normal = tokio::task::spawn_blocking(move || {
            let mut out = String::new();
            req("/ok").read_to_string(&mut out).unwrap();
            out
        });
        tokio::time::sleep(Duration::from_millis(100)).await;
        let t = std::time::Instant::now();
        tx.send(()).unwrap();
        tokio::time::timeout(deadline + Duration::from_secs(5), task)
            .await
            .expect("serve returns despite the stalled reader")
            .unwrap()
            .unwrap();
        assert!(t.elapsed() >= deadline, "the stalled stream held the drain");
        let out = normal.await.unwrap();
        assert!(
            out.starts_with("HTTP/1.1 200") && out.ends_with("ok"),
            "{out}"
        );
    }
}
