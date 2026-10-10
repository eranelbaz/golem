//! Decode worker: owns the `!Send` `Session` (inside a `Scheduler`) on one
//! std thread and batches up to `parallel` jobs across its slots; jobs queue
//! on a `sync_channel(parallel − 1)`.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU8, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use golem_api::chat::{FunctionCall, Stop, ToolCall};
use golem_api::{ChatCompletionRequest, ErrorCode, ErrorEnvelope, MessageContent};
use golem_engine::config::EngineConfig;
use golem_engine::{
    ChatTurn, EngineError, FinishReason, OutputParser, Piece, ReasoningSplitter, SamplingParams,
    Scheduler, Session, SlotEvent, SlotJob, TurnOutcome,
};
use golem_llama_sys::raw::LLAMA_DEFAULT_SEED;

use crate::metrics::{MetricsSnapshot, code_index};

/// Per-job event channel depth: 256 pieces ≫ one SSE write, so a live reader
/// never fills it; at ≤ ~10 tok/s cold decode on the target that is ≥ 25 s
/// of slack.
pub(crate) const EVENT_CAPACITY: usize = 256;

/// One accepted request.
pub struct Job {
    /// Validated request.
    pub req: ChatCompletionRequest,
    /// Set by the client side (disconnect) or drain; polled per decode step.
    pub cancel: Arc<AtomicBool>,
    /// Text pieces, then exactly one terminal event.
    pub events: tokio::sync::mpsc::Sender<Event>,
    /// How reasoning-marker text is delivered.
    pub reasoning: ReasoningMode,
}

/// `--reasoning`: where text between the template's reasoning markers goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReasoningMode {
    /// Markers and reasoning stay in `content` (splitter bypassed).
    Inline,
    /// Reasoning goes to `reasoning_content`; markers are dropped.
    #[default]
    Separate,
}

/// Worker → handler stream for one job.
#[derive(Debug)]
pub enum Event {
    /// Generated answer text piece.
    Text(String),
    /// Generated reasoning piece (`separate` mode only).
    Reasoning(String),
    /// Tool-call delta: the first one per call carries `index`, `id`, `type`
    /// and the name; later ones append to `arguments`.
    ToolCall(ToolCall),
    /// Turn finished; sent after all post-turn work.
    Done(TurnOutcome),
    /// Turn failed; the envelope is the client-facing error.
    Failed(ErrorEnvelope),
}

/// Lifecycle state reported by `/health`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ServeState {
    /// Model loading.
    Loading = 0,
    /// Serving.
    Healthy = 1,
    /// Serving with a degraded tier.
    Degraded = 2,
    /// Shutting down; new requests get 503.
    Draining = 3,
    /// Worker dead; requests get 500.
    Fatal = 4,
    /// Every replica of a shard unreadable (§5.2): requests get 503
    /// `golem_storage_unavailable`; the process keeps answering `/health`.
    FatalIo = 5,
    /// Drained; tearing down (pool munlock, unregister).
    Releasing = 6,
}

impl ServeState {
    /// Wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Loading => "loading",
            Self::Healthy => "healthy",
            Self::Degraded => "degraded",
            Self::Draining => "draining",
            Self::Fatal => "fatal",
            Self::FatalIo => "fatal-io",
            Self::Releasing => "releasing",
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Loading,
            1 => Self::Healthy,
            2 => Self::Degraded,
            3 => Self::Draining,
            4 => Self::Fatal,
            5 => Self::FatalIo,
            _ => Self::Releasing,
        }
    }
}

/// Tier counters published by the worker at each turn end.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct HealthSnapshot {
    /// Turns completed since load.
    pub turns: u64,
    /// Device expert cache skip ratio (§3.4 P4); `None` without the pool.
    pub device_cache_skip_ratio: Option<f64>,
}

/// `/health` reports `degraded:stalled` past this gap between decode
/// progress marks while a job is active: one cold streamed default
/// `prefill_ubatch` on the target plus a 30 s margin.
// 2048 tok / 5.6 tok/s (docs/baselines/m3.6-prefill-stream.md) ≈ 365.7 s + 30 s = 395.7 s.
pub const STALL_MS: u64 = 2048 * 10_000 / 56 + 30_000;

/// Published every [`METRICS_EVERY`] steps and at turn end.
const METRICS_EVERY: u64 = 64;

/// Shared instance health; written by the worker and the drain path.
#[derive(Debug)]
pub struct HealthState {
    state: AtomicU8,
    snapshot: Mutex<HealthSnapshot>,
    /// Worker-published counters (`/metrics`, `/health` tiers).
    pub(crate) metrics: Mutex<MetricsSnapshot>,
    /// Per-[`ErrorCode`] responses, indexed by [`code_index`].
    pub(crate) errors: [AtomicU64; crate::metrics::N_CODES],
    /// Jobs handed to the worker and not yet dequeued.
    pub(crate) queued: AtomicI64,
    epoch: Instant,
    /// ms since `epoch` of the last `llama_decode` return or admission.
    last_progress: AtomicU64,
    /// Cancel flags of the jobs on the worker (one per busy slot).
    in_flight: Mutex<Vec<Arc<AtomicBool>>>,
    #[cfg(any(test, feature = "test-support"))]
    hooks: TestHooks,
}

impl HealthState {
    pub(crate) fn new() -> Self {
        Self {
            state: AtomicU8::new(ServeState::Loading as u8),
            snapshot: Mutex::new(HealthSnapshot::default()),
            metrics: Mutex::new(MetricsSnapshot::default()),
            errors: Default::default(),
            queued: AtomicI64::new(0),
            epoch: Instant::now(),
            last_progress: AtomicU64::new(0),
            in_flight: Mutex::new(Vec::new()),
            #[cfg(any(test, feature = "test-support"))]
            hooks: TestHooks::default(),
        }
    }

    /// Current state.
    pub fn get(&self) -> ServeState {
        ServeState::from_u8(self.state.load(Ordering::SeqCst))
    }

    /// Sets the state; `fatal*` is terminal.
    pub fn set(&self, s: ServeState) {
        let _ = self
            .state
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |cur| {
                (!matches!(
                    ServeState::from_u8(cur),
                    ServeState::Fatal | ServeState::FatalIo
                ))
                .then_some(s as u8)
            });
    }

    /// Counts one error response.
    pub(crate) fn count_error(&self, code: ErrorCode) {
        self.errors[code_index(code)].fetch_add(1, Ordering::Relaxed);
    }

    fn mark_progress(&self) {
        self.last_progress
            .store(self.epoch.elapsed().as_millis() as u64, Ordering::Relaxed);
    }

    /// A job is active and no decode progressed for [`STALL_MS`].
    pub fn stalled(&self) -> bool {
        self.stalled_at(self.epoch.elapsed().as_millis() as u64)
    }

    fn stalled_at(&self, now_ms: u64) -> bool {
        self.busy() && now_ms.saturating_sub(self.last_progress.load(Ordering::Relaxed)) > STALL_MS
    }

    /// Jobs on the worker.
    pub(crate) fn in_flight_count(&self) -> usize {
        self.in_flight
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .len()
    }

    /// Last published tier snapshot.
    pub fn snapshot(&self) -> HealthSnapshot {
        self.snapshot
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Draining → new requests 503; the in-flight job is cancelled (drain
    /// grace 0) and answered `golem_busy`.
    pub fn begin_drain(&self) {
        self.set(ServeState::Draining);
        self.cancel_in_flight();
    }

    /// Sets the cancel flag of every job currently on the worker.
    pub fn cancel_in_flight(&self) {
        for c in self
            .in_flight
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
        {
            c.store(true, Ordering::SeqCst);
        }
    }

    /// At least one job is on the worker.
    pub fn busy(&self) -> bool {
        !self
            .in_flight
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .is_empty()
    }

    /// Republishes the busy slots' flags; a job admitted while Draining is
    /// cancelled here, so `begin_drain` never misses one. Only when the job
    /// set changed (admission or finish), never per decode step.
    fn publish_in_flight(&self, flags: impl Iterator<Item = Arc<AtomicBool>>) {
        let mut v = self.in_flight.lock().unwrap_or_else(|p| p.into_inner());
        v.clear();
        v.extend(flags);
        if self.get() == ServeState::Draining {
            for c in v.iter() {
                c.store(true, Ordering::SeqCst);
            }
        }
    }
}

/// The worker did not exit within the shutdown bound.
#[derive(Debug, thiserror::Error)]
#[error("decode worker did not join within {0:?}; leaking it")]
pub struct ShutdownTimeout(pub Duration);

/// Owner side of the worker thread.
pub struct WorkerHandle {
    /// Job queue (`parallel − 1` deep): `try_send` fails → 503 `golem_busy`.
    pub jobs: SyncSender<Job>,
    /// Shared health.
    pub health: Arc<HealthState>,
    join: std::thread::JoinHandle<()>,
}

impl WorkerHandle {
    /// Drops this sender (the worker exits once every clone is gone), calls
    /// `in_flight_cancel`, and joins within `timeout`. On timeout the thread
    /// is detached, never joined.
    pub fn shutdown(
        self,
        in_flight_cancel: impl Fn(),
        timeout: Duration,
    ) -> Result<(), ShutdownTimeout> {
        drop(self.jobs);
        in_flight_cancel();
        let deadline = Instant::now() + timeout;
        while !self.join.is_finished() {
            if Instant::now() >= deadline {
                return Err(ShutdownTimeout(timeout));
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        // A panic was already caught inside the thread; join cannot fail.
        let _ = self.join.join();
        Ok(())
    }
}

/// Loads `cfg` on a new thread (the `Session` never leaves it) and returns
/// once the model is loaded, or the load error.
pub fn spawn_worker(cfg: EngineConfig) -> Result<WorkerHandle, EngineError> {
    spawn(cfg, HealthState::new())
}

/// [`spawn_worker`] with fault injection scoped to this worker.
#[cfg(any(test, feature = "test-support"))]
pub fn spawn_worker_with_hooks(
    cfg: EngineConfig,
    hooks: TestHooks,
) -> Result<WorkerHandle, EngineError> {
    spawn(
        cfg,
        HealthState {
            hooks,
            ..HealthState::new()
        },
    )
}

fn spawn(cfg: EngineConfig, health: HealthState) -> Result<WorkerHandle, EngineError> {
    let health = Arc::new(health);
    // parallel − 1 queued jobs: with every slot busy the handler's
    // `try_send` fills this, then 503s — never more queued than can start
    // once one slot frees (parallel 1 keeps the rendezvous).
    let (jobs, rx) = sync_channel::<Job>(cfg.parallel.saturating_sub(1) as usize);
    let (loaded_tx, loaded_rx) = sync_channel::<Result<(), EngineError>>(0);
    let h = health.clone();
    let join = std::thread::Builder::new()
        .name("golem-decode".into())
        .spawn(move || {
            let mut sch = match Session::new(&cfg).and_then(Scheduler::new) {
                Ok(s) => s,
                Err(e) => {
                    h.set(ServeState::Fatal);
                    let _ = loaded_tx.send(Err(e));
                    return;
                }
            };
            h.set(ServeState::Healthy);
            let _ = loaded_tx.send(Ok(()));
            run(&mut sch, &rx, &h);
        })?;
    match loaded_rx.recv() {
        Ok(Ok(())) => Ok(WorkerHandle { jobs, health, join }),
        Ok(Err(e)) => {
            let _ = join.join();
            Err(e)
        }
        Err(_) => {
            let _ = join.join();
            Err(EngineError::InvalidConfig(
                "decode worker died during load".into(),
            ))
        }
    }
}

/// One admitted job's per-slot delivery state.
struct SlotRun {
    job: Job,
    /// Reserved before any text so the terminal event is never dropped.
    permit: tokio::sync::mpsc::OwnedPermit<Event>,
    splitter: Option<ReasoningSplitter>,
    tools: Option<ToolStream>,
    /// Cancelled because the event channel was full (client too slow),
    /// not because the client left.
    slow_client: std::cell::Cell<bool>,
    admitted: Instant,
    /// First emitted piece (stop-string holdback can delay it).
    first_piece: Option<Instant>,
}

/// Non-blocking delivery; a full or closed channel cancels the job.
fn try_deliver(job: &Job, slow_client: &std::cell::Cell<bool>, ev: Event) {
    if let Err(e) = job.events.try_send(ev) {
        if matches!(e, tokio::sync::mpsc::error::TrySendError::Full(_)) {
            tracing::warn!("client too slow to drain events; cancelling");
            slow_client.set(true);
        }
        job.cancel.store(true, Ordering::SeqCst);
    }
}

impl SlotRun {
    #[cfg(any(test, feature = "test-support"))]
    fn send(&self, ev: Event) {
        try_deliver(&self.job, &self.slow_client, ev);
    }

    fn piece(&mut self, parser: Option<&OutputParser>, t: &str) {
        let (job, splitter, tools, slow) = (
            &self.job,
            &mut self.splitter,
            &mut self.tools,
            &self.slow_client,
        );
        let send = |ev: Event| try_deliver(job, slow, ev);
        if let (Some(ts), Some(p)) = (tools.as_mut(), parser) {
            ts.push(p, t, true, &send);
            return;
        }
        let mut send_piece = |p: Piece| {
            send(match p {
                Piece::Content(t) => Event::Text(t.into()),
                Piece::Reasoning(t) => Event::Reasoning(t.into()),
            })
        };
        match splitter {
            Some(s) => s.push(t, &mut send_piece),
            None => send_piece(Piece::Content(t)),
        }
    }
}

/// The decode loop: admits jobs into free slots (never dequeuing one it
/// cannot admit), steps every active slot through one batched decode, and
/// answers each finished slot on its own event stream.
fn run(sch: &mut Scheduler, rx: &Receiver<Job>, health: &HealthState) {
    let n = sch.free_slots();
    let mut runs: Vec<Option<SlotRun>> = (0..n).map(|_| None).collect();
    // Per-step event buffers, reused: no allocation per step.
    let mut pieces: Vec<(usize, String)> = Vec::with_capacity(n);
    let mut ended: Vec<(usize, Result<TurnOutcome, EngineError>)> = Vec::with_capacity(n);
    let mut changed = false;
    let mut slot_tokens = vec![0u64; n];
    let mut steps = 0u64;
    publish_metrics(sch, health, &slot_tokens);
    loop {
        // Block only when every slot is free; else poll between steps.
        let idle = runs.iter().all(Option::is_none);
        while sch.free_slots() > 0 {
            let job = if idle && runs.iter().all(Option::is_none) {
                match rx.recv() {
                    Ok(j) => j,
                    Err(_) => return,
                }
            } else {
                match rx.try_recv() {
                    Ok(j) => j,
                    Err(std::sync::mpsc::TryRecvError::Empty) => break,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        if runs.iter().all(Option::is_none) {
                            return;
                        }
                        break;
                    }
                }
            };
            health.queued.fetch_sub(1, Ordering::Relaxed);
            match admit(sch, &mut runs, job, health) {
                Ok(()) => {
                    changed = true;
                    health.mark_progress();
                }
                Err(true) => {
                    fail_all(&mut runs, health);
                    return;
                }
                Err(false) => {}
            }
        }
        if std::mem::take(&mut changed) {
            health.publish_in_flight(runs.iter().flatten().map(|r| r.job.cancel.clone()));
        }
        if runs.iter().all(Option::is_none) {
            continue;
        }
        if storage_lost(sch, &mut runs, health) {
            refuse_all(rx, health);
            return;
        }
        let stepped = catch_unwind(AssertUnwindSafe(|| {
            step(
                sch,
                &mut runs,
                &mut pieces,
                &mut ended,
                health,
                &mut slot_tokens,
            )
        }));
        health.mark_progress();
        steps += 1;
        if steps.is_multiple_of(METRICS_EVERY) {
            publish_metrics(sch, health, &slot_tokens);
        }
        // A step error (or panic) after `llama_decode` leaves every slot's
        // sequence in an unknown state, so the worker is fatal and every
        // slot fails with it; per-slot batch failures are recovered inside
        // `Scheduler::step` and arrive as `SlotEvent::Failed`.
        let fatal = match stepped {
            Ok(Ok(finished)) => {
                changed = finished;
                false
            }
            // A page-in failed mid-decode on a dead shard (ABI v13).
            Ok(Err(EngineError::StorageUnavailable)) if storage_lost(sch, &mut runs, health) => {
                refuse_all(rx, health);
                return;
            }
            Ok(Err(e)) => {
                tracing::error!("decode worker: {e}");
                true
            }
            Err(p) => {
                let msg = p
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| p.downcast_ref::<String>().cloned())
                    .unwrap_or_default();
                tracing::error!(
                    "{}",
                    ErrorEnvelope::new(
                        ErrorCode::BackendUnavailable,
                        format!("decode worker panicked: {msg}"),
                    )
                );
                true
            }
        };
        if fatal {
            fail_all(&mut runs, health);
            return;
        }
    }
}

/// §5.2 fatal I/O, checked once per step (a handful of atomic replica
/// health reads, never per expert): a shard with no readable replica fails
/// every in-flight job with 503 `golem_storage_unavailable` and moves the
/// instance to `fatal-io`.
fn storage_lost(sch: &mut Scheduler, runs: &mut [Option<SlotRun>], health: &HealthState) -> bool {
    if sch.session().storage_health() != Some(golem_store::InstanceHealth::FatalIo) {
        return false;
    }
    health.set(ServeState::FatalIo);
    health.publish_in_flight(std::iter::empty());
    for r in runs.iter_mut() {
        if let Some(run) = r.take() {
            run.permit.send(Event::Failed(storage_unavailable()));
        }
    }
    publish_metrics(sch, health, &[]);
    true
}

/// No further decode may touch the dead shard: answers every later job
/// until the handlers disconnect.
fn refuse_all(rx: &Receiver<Job>, health: &HealthState) {
    while let Ok(job) = rx.recv() {
        health.queued.fetch_sub(1, Ordering::Relaxed);
        let _ = job.events.try_send(Event::Failed(storage_unavailable()));
    }
}

fn storage_unavailable() -> ErrorEnvelope {
    ErrorEnvelope::new(
        ErrorCode::StorageUnavailable,
        "every replica of a model shard is unreadable",
    )
}

/// Copies the session's counters into the shared snapshot: one `try_lock`,
/// skipped on contention (the next cadence point republishes).
fn publish_metrics(sch: &mut Scheduler, health: &HealthState, slot_tokens: &[u64]) {
    let Ok(mut m) = health.metrics.try_lock() else {
        return;
    };
    m.refresh(sch.session(), slot_tokens);
}

/// Fatal worker: drops every in-flight job (its stream closes before a
/// terminal event → 500), then `Disconnected` for new ones.
fn fail_all(runs: &mut [Option<SlotRun>], health: &HealthState) {
    health.set(ServeState::Fatal);
    health.publish_in_flight(std::iter::empty());
    for r in runs.iter_mut() {
        *r = None;
    }
}

fn draining() -> ErrorEnvelope {
    ErrorEnvelope::new(ErrorCode::Busy, "server draining")
}

/// Renders, tokenizes and admits one job. `Err(true)`: the worker is no
/// longer usable; `Err(false)`: the job was answered and dropped.
fn admit(
    sch: &mut Scheduler,
    runs: &mut [Option<SlotRun>],
    job: Job,
    health: &HealthState,
) -> Result<(), bool> {
    // Dropped while queued (client left): never run.
    if job.cancel.load(Ordering::SeqCst) {
        return Err(false);
    }
    // `begin_drain` sets Draining then cancels the published flags; a job
    // dequeued after that is answered busy here.
    match health.get() {
        ServeState::Draining | ServeState::Releasing => {
            let _ = job.events.try_send(Event::Failed(draining()));
            return Err(false);
        }
        ServeState::FatalIo => {
            let _ = job.events.try_send(Event::Failed(storage_unavailable()));
            return Err(false);
        }
        _ => {}
    }
    #[cfg(any(test, feature = "test-support"))]
    if health
        .hooks
        .panic
        .compare_exchange(INJECT_AT_START, 0, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        tracing::error!("decode worker panicked: injected worker panic");
        return Err(true);
    }
    let Ok(permit) = job.events.clone().try_reserve_owned() else {
        return Err(false);
    };
    let req = &job.req;
    let texts: Vec<String> = req.messages.iter().map(message_text).collect();
    let turns: Vec<ChatTurn<'_>> = req
        .messages
        .iter()
        .zip(&texts)
        .map(|(m, t)| ChatTurn {
            role: &m.role,
            content: t,
        })
        .collect();
    let stop: Vec<String> = match &req.stop {
        None => Vec::new(),
        Some(Stop::One(s)) => vec![s.clone()],
        Some(Stop::Many(v)) => v.clone(),
    };
    // OpenAI defaults, not `SamplingParams::default()` (greedy).
    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(1.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k.unwrap_or(0),
        seed: req.seed.map_or(LLAMA_DEFAULT_SEED, |s| s as u32),
        frequency_penalty: req.frequency_penalty.unwrap_or(0.0),
        presence_penalty: req.presence_penalty.unwrap_or(0.0),
    };
    // The raw array keeps assistant `tool_calls` and `tool` messages.
    let messages_json = serde_json::to_string(&req.messages).unwrap_or_default();
    let tools_json = req
        .tools
        .as_ref()
        .map(|t| serde_json::to_string(t).unwrap_or_default());
    let tool_choice = req
        .tool_choice
        .as_ref()
        .map(|c| serde_json::to_string(c).unwrap_or_default());
    let base = golem_engine::TurnRequest::new(&turns)
        .messages_json(&messages_json)
        .tools(tools_json.as_deref(), tool_choice.as_deref())
        .reasoning_inline(job.reasoning == ReasoningMode::Inline)
        .reasoning_effort(req.reasoning_effort.as_deref());
    let tool_mode = tools_json.is_some() && req.tool_choice != Some("none".into());
    let invalid = |e: EngineError| {
        Event::Failed(ErrorEnvelope::new(ErrorCode::InvalidRequest, e.to_string()))
    };
    let rendered = match sch.session().render(&base) {
        Ok(r) => r,
        Err(e) => {
            permit.send(invalid(e));
            return Err(false);
        }
    };
    let tokens = match sch.tokenize(&rendered.prompt) {
        Ok(t) => t,
        Err(e) => {
            permit.send(invalid(e));
            return Err(false);
        }
    };
    let splitter = match job.reasoning {
        ReasoningMode::Separate if !tool_mode => sch.session().reasoning_splitter(),
        _ => None,
    };
    let slot_job = SlotJob {
        tokens,
        sampling,
        max_tokens: req.max_tokens,
        stop,
        cancel: job.cancel.clone(),
        // Only tool-mode renders carry a grammar the sampler must apply.
        rendered: tool_mode.then_some(rendered),
    };
    let Ok(id) = sch.admit(slot_job) else {
        unreachable!("admit runs only while a slot is free");
    };
    runs[id] = Some(SlotRun {
        job,
        permit,
        splitter,
        tools: tool_mode.then(ToolStream::new),
        slow_client: std::cell::Cell::new(false),
        admitted: Instant::now(),
        first_piece: None,
    });
    #[cfg(any(test, feature = "test-support"))]
    if health.hooks.fail_storage.swap(false, Ordering::SeqCst) {
        sch.session().fail_storage_for_test();
    }
    #[cfg(any(test, feature = "test-support"))]
    if health
        .hooks
        .fail_storage_in_read
        .swap(false, Ordering::SeqCst)
    {
        sch.session().fail_storage_in_read_for_test();
    }
    Ok(())
}

/// One scheduler step; delivers each slot's pieces and terminal events.
/// `Ok(true)` when a slot finished (the in-flight set changed).
fn step(
    sch: &mut Scheduler,
    runs: &mut [Option<SlotRun>],
    pieces: &mut Vec<(usize, String)>,
    ended: &mut Vec<(usize, Result<TurnOutcome, EngineError>)>,
    health: &HealthState,
    slot_tokens: &mut [u64],
) -> Result<bool, EngineError> {
    pieces.clear();
    ended.clear();
    // `parser` borrows `sch`, so text is buffered per step and delivered
    // after: ≤ one piece per slot per step (the Task 3 `String` budget).
    sch.step(&mut |i, e| match e {
        SlotEvent::Text(t) => pieces.push((i, t)),
        SlotEvent::Done(o) => ended.push((i, Ok(o))),
        SlotEvent::Failed(e) => ended.push((i, Err(e))),
    })?;
    // Greedy slots replay a degraded batch, so no `Failed` may carry it.
    if let Some(k) = sch.take_degraded() {
        tracing::warn!("{}", EngineError::DeviceCacheDegraded(k));
        if health.get() == ServeState::Healthy {
            health.set(ServeState::Degraded);
        }
    }
    for (i, t) in pieces.drain(..) {
        let Some(run) = runs[i].as_mut() else {
            continue;
        };
        slot_tokens[i] += 1;
        run.first_piece.get_or_insert_with(Instant::now);
        #[cfg(any(test, feature = "test-support"))]
        if health
            .hooks
            .panic
            .compare_exchange(INJECT_AFTER_TOKEN, 0, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            run.send(Event::Text(t));
            panic!("injected worker panic after first token");
        }
        #[cfg(any(test, feature = "test-support"))]
        std::thread::sleep(Duration::from_millis(health.hooks.slow_ms));
        run.piece(sch.parser(i), &t);
    }
    let finished = !ended.is_empty();
    for (i, res) in ended.drain(..) {
        let Some(run) = runs[i].take() else { continue };
        finish(sch, run, i, res, health);
    }
    if finished {
        publish_metrics(sch, health, slot_tokens);
    }
    Ok(finished)
}

/// Final flushes and the slot's one terminal event.
fn finish(
    sch: &mut Scheduler,
    mut run: SlotRun,
    id: usize,
    res: Result<TurnOutcome, EngineError>,
    health: &HealthState,
) {
    let SlotRun {
        job,
        splitter,
        tools,
        slow_client,
        ..
    } = &mut run;
    let send = |ev: Event| try_deliver(job, slow_client, ev);
    if let Some(s) = splitter {
        s.finish(&mut |p: Piece| {
            send(match p {
                Piece::Content(t) => Event::Text(t.into()),
                Piece::Reasoning(t) => Event::Reasoning(t.into()),
            })
        });
    }
    let called = match (tools.as_mut(), sch.parser(id)) {
        (Some(ts), Some(p)) => {
            ts.push(p, "", false, &send);
            ts.calls > 0
        }
        _ => false,
    };
    let res = res.map(|o| match o.finish {
        FinishReason::Stop if called => TurnOutcome {
            finish: FinishReason::ToolCalls,
            ..o
        },
        _ => o,
    });
    {
        let mut snap = health.snapshot.lock().unwrap_or_else(|p| p.into_inner());
        snap.turns += 1;
        snap.device_cache_skip_ratio = sch.session().device_cache_stats().map(|s| s.skip_ratio);
    }
    let event = match res {
        Ok(o) if o.finish == FinishReason::Cancelled && health.get() == ServeState::Draining => {
            Event::Failed(draining())
        }
        // A live client that fell behind gets an error, not a truncated
        // `Done`; a closed receiver (disconnect) keeps `Done`.
        Ok(o)
            if o.finish == FinishReason::Cancelled
                && run.slow_client.get()
                && !run.job.events.is_closed() =>
        {
            Event::Failed(ErrorEnvelope::new(
                ErrorCode::BackendUnavailable,
                "client too slow to drain events",
            ))
        }
        Ok(o) => Event::Done(o),
        Err(e @ EngineError::DeviceCacheDegraded(_)) => {
            tracing::warn!("{e}");
            if health.get() == ServeState::Healthy {
                health.set(ServeState::Degraded);
            }
            Event::Failed(ErrorEnvelope::new(
                ErrorCode::BackendUnavailable,
                e.to_string(),
            ))
        }
        Err(
            e @ (EngineError::ContextOverflow { .. }
            | EngineError::ChatTemplate(_)
            | EngineError::Tokenize(_)
            | EngineError::TemplateApply(_)
            | EngineError::TemplateUnavailable),
        ) => Event::Failed(ErrorEnvelope::new(ErrorCode::InvalidRequest, e.to_string())),
        // A failed batch fails only its unrecoverable slots (the session
        // stays usable): 500 golem_backend_unavailable.
        Err(e) => {
            let env = ErrorEnvelope::new(ErrorCode::BackendUnavailable, e.to_string());
            tracing::error!("{env}");
            Event::Failed(env)
        }
    };
    log_turn(&run, &event);
    let SlotRun { permit, .. } = run;
    permit.send(event);
}

/// One INFO line per finished request: status (499 = client left), finish,
/// and prefill / generation rates. Prefill time is admission → first piece,
/// so a snapshot-restored prompt or `--parallel` > 1 reads as latency, not
/// throughput.
fn log_turn(run: &SlotRun, event: &Event) {
    let end = Instant::now();
    let first = run.first_piece.unwrap_or(end);
    let prefill_s = first.duration_since(run.admitted).as_secs_f64();
    let gen_s = end.duration_since(first).as_secs_f64();
    let rate = |n: u32, s: f64| if s > 0.0 { f64::from(n) / s } else { 0.0 };
    match event {
        Event::Done(o) => tracing::info!(
            status = if run.job.events.is_closed() { 499 } else { 200 },
            finish = ?o.finish,
            prompt_tokens = o.prompt_tokens,
            completion_tokens = o.completion_tokens,
            prefill_s = format_args!("{prefill_s:.2}"),
            prefill_tok_s = format_args!("{:.1}", rate(o.prompt_tokens, prefill_s)),
            gen_tok_s = format_args!("{:.1}", rate(o.completion_tokens.saturating_sub(1), gen_s)),
            "chat completion"
        ),
        Event::Failed(e) => tracing::info!(
            status = crate::app::status_of(e.error.code).as_u16(),
            code = ?e.error.code,
            elapsed_s = format_args!("{:.2}", end.duration_since(run.admitted).as_secs_f64()),
            "chat completion failed: {}",
            e.error.message
        ),
        _ => {}
    }
}

/// Turns the growing generated text into OpenAI deltas by re-parsing it with
/// the request's chat-layer parser (`partial` until the turn ends) and
/// emitting what each parse added. Re-parsing is O(text) per piece — one
/// parse per token at ≤ ~10 tok/s on the target, off the per-expert path.
struct ToolStream {
    text: String,
    content: String,
    reasoning: String,
    /// Per announced call: arguments sent so far.
    args: Vec<String>,
    calls: usize,
}

impl ToolStream {
    fn new() -> Self {
        Self {
            text: String::new(),
            content: String::new(),
            reasoning: String::new(),
            args: Vec::new(),
            calls: 0,
        }
    }

    fn push(&mut self, parser: &OutputParser, piece: &str, partial: bool, send: &dyn Fn(Event)) {
        self.text.push_str(piece);
        let msg = match parser.parse(&self.text, partial) {
            Ok(msg) => msg,
            Err(_) if partial => return,
            // Unparseable final text is still the model's answer: emit what
            // was not yet sent as content.
            Err(_) => serde_json::json!({ "content": self.text }),
        };
        let field = |k: &str| msg[k].as_str().unwrap_or_default().to_owned();
        for (seen, now, ev) in [
            (
                &mut self.reasoning,
                field("reasoning_content"),
                Event::Reasoning as fn(String) -> Event,
            ),
            (&mut self.content, field("content"), Event::Text),
        ] {
            // A partial parse can only grow; anything else is a re-parse we
            // cannot retract, so it is held until the text settles. The final
            // parse flushes what is held past the longest common prefix.
            let sent = match now.strip_prefix(seen.as_str()) {
                Some(_) => seen.len(),
                None if partial => continue,
                None => common_prefix(seen, &now),
            };
            if now.len() > sent {
                send(ev(now[sent..].to_owned()));
                *seen = now;
            }
        }
        let calls = msg["tool_calls"].as_array().map_or(&[][..], Vec::as_slice);
        for (i, c) in calls.iter().enumerate() {
            let name = c["function"]["name"].as_str().unwrap_or_default();
            let args = c["function"]["arguments"].as_str().unwrap_or_default();
            // Announce a call once its name is complete: arguments started,
            // or the turn ended.
            if i == self.args.len() {
                if name.is_empty() || (partial && args.is_empty()) {
                    break;
                }
                send(Event::ToolCall(ToolCall {
                    index: Some(i as u32),
                    id: Some(format!("call_{}", uuid::Uuid::new_v4().simple())),
                    kind: Some("function".into()),
                    function: FunctionCall {
                        name: Some(name.into()),
                        arguments: args.into(),
                    },
                }));
                self.args.push(args.into());
                self.calls += 1;
                continue;
            }
            let seen = &mut self.args[i];
            if let Some(add) = args.strip_prefix(seen.as_str()).filter(|a| !a.is_empty()) {
                send(Event::ToolCall(ToolCall {
                    index: Some(i as u32),
                    id: None,
                    kind: None,
                    function: FunctionCall {
                        name: None,
                        arguments: add.into(),
                    },
                }));
                *seen = args.into();
            }
        }
    }
}

/// Byte length of the longest common prefix of `a` and `b`, on a char
/// boundary.
fn common_prefix(a: &str, b: &str) -> usize {
    a.char_indices()
        .zip(b.chars())
        .find(|((_, x), y)| x != y)
        .map_or(a.len().min(b.len()), |((i, _), _)| i)
}

fn message_text(m: &golem_api::ChatMessage) -> String {
    match &m.content {
        None => String::new(),
        Some(MessageContent::Text(s)) => s.clone(),
        Some(MessageContent::Parts(parts)) => {
            parts.iter().filter_map(|p| p.text.as_deref()).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GLM: &str = include_str!("../../golem-engine/tests/fixtures/glm-5.3-flash.jinja");

    fn job(content: &str, max_tokens: u32) -> (Job, tokio::sync::mpsc::Receiver<Event>) {
        let (tx, rx) = tokio::sync::mpsc::channel(EVENT_CAPACITY);
        let req: ChatCompletionRequest = serde_json::from_value(serde_json::json!({
            "model": "tiny-moe",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }))
        .unwrap();
        let job = Job {
            req,
            cancel: Arc::new(AtomicBool::new(false)),
            events: tx,
            reasoning: ReasoningMode::Inline,
        };
        (job, rx)
    }

    /// `--parallel 2`: a job sent while slot 0 generates starts (first piece)
    /// before slot 0's turn ends; a job whose client left while queued
    /// never runs.
    #[test]
    fn job_sent_while_slot_0_generates_starts_before_it_finishes() {
        let cfg = EngineConfig {
            parallel: 2,
            ..crate::test_engine_config()
        };
        let w = spawn_worker_with_hooks(cfg, TestHooks::slow_decode(20)).expect("worker");
        let (a, mut ra) = job("count to ten", 64);
        assert!(w.jobs.try_send(a).is_ok(), "slot 0");
        assert!(matches!(ra.blocking_recv(), Some(Event::Text(_))));
        let (gone, mut rg) = job("x", 4);
        gone.cancel.store(true, Ordering::SeqCst);
        assert!(w.jobs.try_send(gone).is_ok(), "queued");
        let (b, mut rb) = job("hello", 4);
        assert!(w.jobs.send(b).is_ok(), "slot 1");
        assert!(
            matches!(rb.blocking_recv(), Some(Event::Text(_))),
            "slot 1 started"
        );
        let a_done = std::iter::from_fn(|| ra.try_recv().ok())
            .any(|e| matches!(e, Event::Done(_) | Event::Failed(_)));
        assert!(!a_done, "slot 0 already finished when slot 1 started");
        assert!(rg.blocking_recv().is_none(), "cancelled queued job ran");
        w.shutdown(|| {}, Duration::from_secs(30)).ok();
    }

    /// A reader that never drains fills the channel: the job ends with
    /// `Failed(BackendUnavailable)` through the reserved permit, not `Done`.
    #[test]
    fn slow_client_gets_failed_not_done() {
        let w = spawn_worker(crate::test_engine_config()).expect("worker");
        let (mut j, _) = job("count to ten", 64);
        // Capacity 1: the reserved terminal permit takes the only slot, so
        // the first text piece finds the channel full.
        let (tx, mut rx) = tokio::sync::mpsc::channel(1);
        j.events = tx;
        assert!(w.jobs.send(j).is_ok());
        match rx.blocking_recv() {
            Some(Event::Failed(env)) => {
                assert_eq!(env.error.code, ErrorCode::BackendUnavailable);
                assert!(env.error.message.contains("too slow"), "{env}");
            }
            other => panic!("expected Failed, got {other:?}"),
        }
        w.shutdown(|| {}, Duration::from_secs(30)).ok();
    }

    #[test]
    fn stalled_only_with_an_active_job_past_the_bound() {
        let h = HealthState::new();
        assert!(!h.stalled_at(STALL_MS + 1), "idle is never stalled");
        h.publish_in_flight(std::iter::once(Arc::new(AtomicBool::new(false))));
        assert!(!h.stalled_at(STALL_MS));
        assert!(h.stalled_at(STALL_MS + 1));
        h.mark_progress();
        assert!(!h.stalled_at(STALL_MS));
    }

    /// A final parse failure (here: NUL in the text) still delivers the
    /// generated text as content instead of finishing empty.
    #[test]
    fn final_parse_failure_falls_back_to_raw_content() {
        let t = golem_engine::ChatTemplates::from_source(GLM).unwrap();
        let r = t
            .render(
                r#"[{"role":"user","content":"hi"}]"#,
                Some("[]"),
                None,
                false,
                None,
            )
            .unwrap();
        let out = std::cell::RefCell::new(String::new());
        let send = |ev: Event| match ev {
            Event::Text(t) => out.borrow_mut().push_str(&t),
            other => panic!("unexpected {other:?}"),
        };
        let mut ts = ToolStream::new();
        ts.push(&r.parser, "bad\0", true, &send);
        ts.push(&r.parser, "text", true, &send);
        assert!(out.borrow().is_empty(), "partial failures are held");
        ts.push(&r.parser, "", false, &send);
        assert_eq!(*out.borrow(), "bad\0text");
        assert_eq!(ts.calls, 0);
    }

    #[test]
    fn common_prefix_respects_char_boundaries() {
        assert_eq!(common_prefix("abé", "abè"), 2);
        assert_eq!(common_prefix("ab", "abc"), 2);
        assert_eq!(common_prefix("abc", "ab"), 2);
    }
}

#[cfg(any(test, feature = "test-support"))]
const INJECT_AT_START: u8 = 1;
#[cfg(any(test, feature = "test-support"))]
const INJECT_AFTER_TOKEN: u8 = 2;

/// Fault injection for one worker.
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug, Default)]
pub struct TestHooks {
    panic: AtomicU8,
    slow_ms: u64,
    fail_storage: AtomicBool,
    fail_storage_in_read: AtomicBool,
}

#[cfg(any(test, feature = "test-support"))]
impl TestHooks {
    /// Every emitted piece is delayed by `ms` (keeps a job in flight long
    /// enough to disconnect or shut down under it; tiny-moe EOGs within
    /// ~200 tokens).
    pub fn slow_decode(ms: u64) -> Self {
        Self {
            slow_ms: ms,
            ..Self::default()
        }
    }

    /// The first job panics before decoding.
    pub fn panic_at_start() -> Self {
        Self {
            panic: AtomicU8::new(INJECT_AT_START),
            ..Self::default()
        }
    }

    /// Every store replica is marked unreadable right after the first job
    /// is admitted (the m3.3 `mark_degraded` fault seam).
    pub fn fail_storage_after_admit() -> Self {
        Self {
            fail_storage: AtomicBool::new(true),
            ..Self::default()
        }
    }

    /// Every store replica dies inside the first job's first page-in, so
    /// the in-flight read itself fails (not the per-step pre-check).
    pub fn fail_storage_during_read() -> Self {
        Self {
            fail_storage_in_read: AtomicBool::new(true),
            ..Self::default()
        }
    }

    /// The first job panics right after emitting its first piece.
    pub fn panic_after_first_token() -> Self {
        Self {
            panic: AtomicU8::new(INJECT_AFTER_TOKEN),
            ..Self::default()
        }
    }
}
