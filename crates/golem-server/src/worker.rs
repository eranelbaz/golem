//! Decode worker: owns the `!Send` `Session` on one std thread and serves
//! jobs handed over a rendezvous channel (slot scheduler v0, `parallel` = 1).

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use golem_api::chat::{FunctionCall, Stop, ToolCall};
use golem_api::{ChatCompletionRequest, ErrorCode, ErrorEnvelope, MessageContent};
use golem_engine::config::EngineConfig;
use golem_engine::{
    ChatTurn, EngineError, FinishReason, OutputParser, Piece, SamplingParams, Session, TurnOutcome,
};
use golem_llama_sys::raw::LLAMA_DEFAULT_SEED;

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
    /// Worker dead on a storage error.
    FatalIo = 5,
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
        }
    }

    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Loading,
            1 => Self::Healthy,
            2 => Self::Degraded,
            3 => Self::Draining,
            4 => Self::Fatal,
            _ => Self::FatalIo,
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

/// Shared instance health; written by the worker and the drain path.
#[derive(Debug)]
pub struct HealthState {
    state: AtomicU8,
    snapshot: Mutex<HealthSnapshot>,
    in_flight: Mutex<Option<Arc<AtomicBool>>>,
    #[cfg(any(test, feature = "test-support"))]
    hooks: TestHooks,
}

impl HealthState {
    pub(crate) fn new() -> Self {
        Self {
            state: AtomicU8::new(ServeState::Loading as u8),
            snapshot: Mutex::new(HealthSnapshot::default()),
            in_flight: Mutex::new(None),
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
                (cur < ServeState::Fatal as u8).then_some(s as u8)
            });
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

    /// Sets the cancel flag of the job currently on the worker, if any.
    pub fn cancel_in_flight(&self) {
        if let Some(c) = &*self.in_flight.lock().unwrap_or_else(|p| p.into_inner()) {
            c.store(true, Ordering::SeqCst);
        }
    }

    fn set_in_flight(&self, c: Option<Arc<AtomicBool>>) {
        *self.in_flight.lock().unwrap_or_else(|p| p.into_inner()) = c;
    }
}

/// The worker did not exit within the shutdown bound.
#[derive(Debug, thiserror::Error)]
#[error("decode worker did not join within {0:?}; leaking it")]
pub struct ShutdownTimeout(pub Duration);

/// Owner side of the worker thread.
pub struct WorkerHandle {
    /// Rendezvous job queue: `try_send` succeeds only while the worker idles.
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
    if cfg.parallel != 1 {
        return Err(EngineError::InvalidConfig(format!(
            "parallel = {} unsupported: the v0 scheduler serves one slot",
            cfg.parallel
        )));
    }
    let health = Arc::new(health);
    let (jobs, rx) = sync_channel::<Job>(0);
    let (loaded_tx, loaded_rx) = sync_channel::<Result<(), EngineError>>(0);
    let h = health.clone();
    let join = std::thread::Builder::new()
        .name("golem-decode".into())
        .spawn(move || {
            let mut session = match Session::new(&cfg) {
                Ok(s) => s,
                Err(e) => {
                    h.set(ServeState::Fatal);
                    let _ = loaded_tx.send(Err(e));
                    return;
                }
            };
            h.set(ServeState::Healthy);
            let _ = loaded_tx.send(Ok(()));
            run(&mut session, &rx, &h);
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

fn run(session: &mut Session, rx: &Receiver<Job>, health: &HealthState) {
    while let Ok(job) = rx.recv() {
        // Published before the drain check: `begin_drain` sets Draining then
        // cancels, so either we see Draining or it sees this job.
        health.set_in_flight(Some(job.cancel.clone()));
        if health.get() == ServeState::Draining {
            health.set_in_flight(None);
            let _ = job.events.try_send(Event::Failed(draining()));
            continue;
        }
        let fatal = match catch_unwind(AssertUnwindSafe(|| run_job(session, &job, health))) {
            Ok(fatal) => fatal,
            Err(p) => {
                let msg = p
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| p.downcast_ref::<String>().cloned())
                    .unwrap_or_default();
                let env = ErrorEnvelope::new(
                    ErrorCode::BackendUnavailable,
                    format!("decode worker panicked: {msg}"),
                );
                tracing::error!("{env}");
                true
            }
        };
        health.set_in_flight(None);
        if fatal {
            health.set(ServeState::Fatal);
            // Dropping `job` here (and `rx` on return) closes the handler's
            // stream before a terminal event → 500, then `Disconnected`.
            return;
        }
    }
}

fn draining() -> ErrorEnvelope {
    ErrorEnvelope::new(ErrorCode::Busy, "server draining")
}

/// Runs one job; returns `true` when the session is no longer usable.
fn run_job(session: &mut Session, job: &Job, health: &HealthState) -> bool {
    #[cfg(any(test, feature = "test-support"))]
    if health
        .hooks
        .panic
        .compare_exchange(INJECT_AT_START, 0, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        panic!("injected worker panic");
    }
    // Reserved before any text so the terminal event is never dropped.
    let Ok(permit) = job.events.try_reserve() else {
        return false;
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
        .reasoning_inline(job.reasoning == ReasoningMode::Inline);
    let tool_mode = tools_json.is_some() && req.tool_choice != Some("none".into());
    let rendered = if tool_mode {
        match session.render(&base) {
            Ok(r) => Some(r),
            Err(e) => {
                permit.send(Event::Failed(ErrorEnvelope::new(
                    ErrorCode::InvalidRequest,
                    e.to_string(),
                )));
                return false;
            }
        }
    } else {
        None
    };
    let mut turn = base
        .sampling(sampling)
        .max_tokens(req.max_tokens)
        .stop(&stop)
        .cancel(Some(job.cancel.clone()));
    if let Some(r) = &rendered {
        turn = turn.rendered(r);
    }
    let cancel = &job.cancel;
    let events = &job.events;
    let mut splitter = match job.reasoning {
        ReasoningMode::Separate if rendered.is_none() => session.reasoning_splitter(),
        _ => None,
    };
    let mut tools = rendered.as_ref().map(|r| ToolStream::new(&r.parser));
    let send = |ev: Event| {
        if let Err(e) = events.try_send(ev) {
            if matches!(e, tokio::sync::mpsc::error::TrySendError::Full(_)) {
                tracing::warn!("client too slow to drain events; cancelling");
            }
            cancel.store(true, Ordering::SeqCst);
        }
    };
    let send = &send;
    let mut send_piece = |p: Piece| {
        send(match p {
            Piece::Content(t) => Event::Text(t.into()),
            Piece::Reasoning(t) => Event::Reasoning(t.into()),
        })
    };
    // One `String` per piece: ≤ ~10 tok/s cold decode on the target → ≤ 10
    // allocs/s, off the per-expert path.
    let res = session.turn(&turn, &mut |t| {
        #[cfg(any(test, feature = "test-support"))]
        if health
            .hooks
            .panic
            .compare_exchange(INJECT_AFTER_TOKEN, 0, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok()
        {
            let _ = events.try_send(Event::Text(t.into()));
            panic!("injected worker panic after first token");
        }
        #[cfg(any(test, feature = "test-support"))]
        std::thread::sleep(Duration::from_millis(health.hooks.slow_ms));
        if let Some(ts) = &mut tools {
            ts.push(t, true, send);
            return;
        }
        match &mut splitter {
            Some(s) => s.push(t, &mut send_piece),
            None => send_piece(Piece::Content(t)),
        }
    });
    if let Some(s) = &mut splitter {
        s.finish(&mut send_piece);
    }
    let called = match &mut tools {
        Some(ts) => {
            ts.push("", false, send);
            ts.calls > 0
        }
        None => false,
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
        snap.device_cache_skip_ratio = session.device_cache_stats().map(|s| s.skip_ratio);
    }
    let (event, fatal) = match res {
        Ok(o) if o.finish == FinishReason::Cancelled && health.get() == ServeState::Draining => {
            (Event::Failed(draining()), false)
        }
        Ok(o) => (Event::Done(o), false),
        Err(e) => {
            let fatal = !matches!(
                e,
                EngineError::ContextOverflow { .. }
                    | EngineError::ChatTemplate(_)
                    | EngineError::Tokenize(_)
                    | EngineError::TemplateApply(_)
                    | EngineError::TemplateUnavailable
            );
            let code = if fatal {
                ErrorCode::BackendUnavailable
            } else {
                ErrorCode::InvalidRequest
            };
            let env = ErrorEnvelope::new(code, e.to_string());
            if fatal {
                tracing::error!("{env}");
            }
            (Event::Failed(env), fatal)
        }
    };
    permit.send(event);
    fatal
}

/// Turns the growing generated text into OpenAI deltas by re-parsing it with
/// the request's chat-layer parser (`partial` until the turn ends) and
/// emitting what each parse added. Re-parsing is O(text) per piece — one
/// parse per token at ≤ ~10 tok/s on the target, off the per-expert path.
struct ToolStream<'p> {
    parser: &'p OutputParser,
    text: String,
    content: String,
    reasoning: String,
    /// Per announced call: arguments sent so far.
    args: Vec<String>,
    calls: usize,
}

impl<'p> ToolStream<'p> {
    fn new(parser: &'p OutputParser) -> Self {
        Self {
            parser,
            text: String::new(),
            content: String::new(),
            reasoning: String::new(),
            args: Vec::new(),
            calls: 0,
        }
    }

    fn push(&mut self, piece: &str, partial: bool, send: &dyn Fn(Event)) {
        self.text.push_str(piece);
        let msg = match self.parser.parse(&self.text, partial) {
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
            )
            .unwrap();
        let out = std::cell::RefCell::new(String::new());
        let send = |ev: Event| match ev {
            Event::Text(t) => out.borrow_mut().push_str(&t),
            other => panic!("unexpected {other:?}"),
        };
        let mut ts = ToolStream::new(&r.parser);
        ts.push("bad\0", true, &send);
        ts.push("text", true, &send);
        assert!(out.borrow().is_empty(), "partial failures are held");
        ts.push("", false, &send);
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

    /// The first job panics right after emitting its first piece.
    pub fn panic_after_first_token() -> Self {
        Self {
            panic: AtomicU8::new(INJECT_AFTER_TOKEN),
            ..Self::default()
        }
    }
}
