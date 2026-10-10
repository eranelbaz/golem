//! HTTP surface (tech design §8.1): routes, error envelope mapping, and the
//! handler side of the job hand-off.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::{Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use golem_api::chat::{Choice, FinishReason, ModelCard, ToolCall};
use golem_api::{
    ChatCompletionResponse, ChatMessage, ErrorCode, ErrorEnvelope, MessageContent, ModelList, Usage,
};
use golem_engine::FinishReason as EngineFinish;
use tower_http::trace::TraceLayer;

use crate::ServerConfig;
use crate::worker::{EVENT_CAPACITY, Event, HealthState, Job, ServeState};

/// Shared handler state.
#[derive(Clone)]
pub struct AppState {
    /// Listener and API identity.
    pub cfg: Arc<ServerConfig>,
    /// Rendezvous hand-off to the decode worker.
    pub jobs: SyncSender<Job>,
    /// Instance health.
    pub health: Arc<HealthState>,
}

/// A wait shorter than one turn's post-turn work on the target, so an agent
/// firing its next turn right after the previous response is not refused.
const ACCEPT_RETRY: Duration = Duration::from_millis(250);
const ACCEPT_STEP: Duration = Duration::from_millis(10);

/// Error response: status plus §8.4 envelope.
struct ApiError(StatusCode, ErrorEnvelope);

impl From<ErrorEnvelope> for ApiError {
    fn from(e: ErrorEnvelope) -> Self {
        let status = match e.error.code {
            ErrorCode::UnsupportedField
            | ErrorCode::InvalidRequest
            | ErrorCode::ModelMismatch
            | ErrorCode::MultimodalUnsupported => StatusCode::BAD_REQUEST,
            ErrorCode::NotFound => StatusCode::NOT_FOUND,
            ErrorCode::Busy | ErrorCode::StorageUnavailable => StatusCode::SERVICE_UNAVAILABLE,
            _ => StatusCode::INTERNAL_SERVER_ERROR,
        };
        Self(status, e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let busy = self.1.error.code == ErrorCode::Busy;
        let mut r = (self.0, Json(self.1)).into_response();
        if busy {
            r.headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
        }
        r
    }
}

fn api_err(code: ErrorCode, msg: impl Into<String>) -> ApiError {
    ErrorEnvelope::new(code, msg).into()
}

pub(crate) fn backend_gone_envelope() -> ErrorEnvelope {
    ErrorEnvelope::new(
        ErrorCode::BackendUnavailable,
        "decode worker is not running",
    )
}

fn backend_gone() -> ApiError {
    backend_gone_envelope().into()
}

pub(crate) fn finish_reason(f: EngineFinish) -> FinishReason {
    match f {
        EngineFinish::Length => FinishReason::Length,
        EngineFinish::ToolCalls => FinishReason::ToolCalls,
        EngineFinish::Stop | EngineFinish::Cancelled => FinishReason::Stop,
    }
}

/// Sets the job's cancel flag when dropped with the handler future or the
/// SSE body (client gone) — independent of events, which prefill does not
/// produce.
pub(crate) struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// `/v1/chat/completions`, `/v1/models`, `/metrics` (behind the API key
/// when set), `/health` (always open); everything else 404.
pub fn router(state: AppState) -> Router {
    let keyed = Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/models", get(models))
        .route("/metrics", get(metrics))
        .route_layer(middleware::from_fn_with_state(state.clone(), authorize));
    Router::new()
        .merge(keyed)
        .route("/health", get(health))
        .fallback(not_found)
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Compares in time independent of where the inputs differ.
fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    let n = a.len().max(b.len());
    let diff = (0..n).fold((a.len() ^ b.len()) as u8, |d, i| {
        d | (a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0))
    });
    std::hint::black_box(diff) == 0 && a.len() == b.len()
}

/// `--api-key`: a missing or wrong `Authorization: Bearer` is a 401
/// `golem_invalid_request` (built here, not through the 400 code map).
async fn authorize(State(st): State<AppState>, req: Request, next: Next) -> Response {
    if let Some(key) = &st.cfg.api_key {
        let got = req
            .headers()
            .get(header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.strip_prefix("Bearer "))
            .unwrap_or("");
        if !ct_eq(got.as_bytes(), key.as_bytes()) {
            st.health.count_error(ErrorCode::InvalidRequest);
            return ApiError(
                StatusCode::UNAUTHORIZED,
                ErrorEnvelope::new(ErrorCode::InvalidRequest, "missing or wrong API key"),
            )
            .into_response();
        }
    }
    next.run(req).await
}

async fn chat(
    State(st): State<AppState>,
    body: Result<Json<serde_json::Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let health = st.health.clone();
    match chat_inner(st, body).await {
        Ok(r) => r,
        Err(e) => {
            health.count_error(e.1.error.code);
            e.into_response()
        }
    }
}

async fn chat_inner(
    st: AppState,
    body: Result<Json<serde_json::Value>, axum::extract::rejection::JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(raw) = body.map_err(|e| api_err(ErrorCode::InvalidRequest, e.body_text()))?;
    let req = golem_api::policy::validate(&raw, &st.cfg.model_name)?;
    match st.health.get() {
        ServeState::Draining | ServeState::Releasing => {
            return Err(api_err(ErrorCode::Busy, "server draining"));
        }
        ServeState::FatalIo => {
            return Err(api_err(
                ErrorCode::StorageUnavailable,
                "every replica of a model shard is unreadable",
            ));
        }
        ServeState::Fatal => return Err(backend_gone()),
        _ => {}
    }
    let stream = req.stream == Some(true);
    let include_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);
    let cancel = Arc::new(AtomicBool::new(false));
    let guard = CancelOnDrop(cancel.clone());
    let (tx, mut rx) = tokio::sync::mpsc::channel(EVENT_CAPACITY);
    let mut job = Job {
        req,
        cancel,
        events: tx,
        reasoning: st.cfg.reasoning,
    };
    let deadline = Instant::now() + ACCEPT_RETRY;
    // Counted before the send: the worker's dequeue may run first.
    st.health.queued.fetch_add(1, Ordering::Relaxed);
    loop {
        match st.jobs.try_send(job) {
            Ok(()) => break,
            Err(TrySendError::Disconnected(_)) => {
                st.health.queued.fetch_sub(1, Ordering::Relaxed);
                return Err(backend_gone());
            }
            Err(TrySendError::Full(j)) => {
                if Instant::now() >= deadline {
                    st.health.queued.fetch_sub(1, Ordering::Relaxed);
                    return Err(api_err(ErrorCode::Busy, "all slots busy"));
                }
                job = j;
                tokio::time::sleep(ACCEPT_STEP).await;
            }
        }
    }
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    if stream {
        let meta = crate::sse::ChunkMeta {
            id,
            created: now_secs(),
            model: st.cfg.model_name.clone(),
            include_usage,
            health: st.health.clone(),
        };
        return Ok(crate::sse::response(meta, rx, guard).into_response());
    }
    let _guard = guard;
    let (mut text, mut reasoning) = (String::new(), String::new());
    let mut calls: Vec<ToolCall> = Vec::new();
    let outcome = loop {
        match rx.recv().await {
            Some(Event::Text(t)) => text.push_str(&t),
            Some(Event::Reasoning(t)) => reasoning.push_str(&t),
            Some(Event::ToolCall(d)) => merge_tool_call(&mut calls, d),
            Some(Event::Done(o)) => break o,
            Some(Event::Failed(e)) => return Err(e.into()),
            // Worker died mid-job: never a 200 with truncated content.
            None => return Err(backend_gone()),
        }
    };
    let finish_reason = finish_reason(outcome.finish);
    Ok(Json(ChatCompletionResponse {
        id,
        object: "chat.completion".into(),
        created: now_secs(),
        model: st.cfg.model_name.clone(),
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content: (calls.is_empty() || !text.is_empty())
                    .then_some(MessageContent::Text(text)),
                reasoning_content: (!reasoning.is_empty()).then_some(reasoning),
                tool_calls: (!calls.is_empty()).then_some(calls),
                tool_call_id: None,
                name: None,
            },
            finish_reason,
        }],
        usage: Usage {
            prompt_tokens: outcome.prompt_tokens,
            completion_tokens: outcome.completion_tokens,
            total_tokens: outcome.prompt_tokens + outcome.completion_tokens,
        },
    })
    .into_response())
}

/// Folds one streamed tool-call delta into the full calls (`index` selects
/// the call; `arguments` pieces append).
fn merge_tool_call(calls: &mut Vec<ToolCall>, d: ToolCall) {
    let i = d.index.unwrap_or(0) as usize;
    if i >= calls.len() {
        calls.push(ToolCall { index: None, ..d });
        return;
    }
    calls[i].function.arguments.push_str(&d.function.arguments);
}

async fn models(State(st): State<AppState>) -> Result<Json<ModelList>, ApiError> {
    Ok(Json(ModelList {
        object: "list".into(),
        data: vec![ModelCard {
            id: st.cfg.model_name.clone(),
            object: "model".into(),
            created: now_secs(),
            owned_by: "golem".into(),
        }],
    }))
}

/// §8.1/§11: `{state, tiers: {vram_fill, ram_fill}, degraded: [...]}`.
async fn health(State(st): State<AppState>) -> Response {
    let state = st.health.get();
    let snap = st.health.snapshot();
    let m = *st.health.metrics.lock().unwrap_or_else(|p| p.into_inner());
    let status = match state {
        ServeState::Healthy | ServeState::Degraded => StatusCode::OK,
        ServeState::Loading
        | ServeState::Draining
        | ServeState::Releasing
        | ServeState::FatalIo => StatusCode::SERVICE_UNAVAILABLE,
        ServeState::Fatal => StatusCode::INTERNAL_SERVER_ERROR,
    };
    let mut degraded = Vec::new();
    if state == ServeState::Degraded {
        degraded.push("device_cache");
    }
    if st.health.stalled() {
        degraded.push("stalled");
    }
    let body = serde_json::json!({
        "state": state.as_str(),
        "tiers": { "vram_fill": m.vram_fill(), "ram_fill": m.ram_fill },
        "degraded": degraded,
        "model": st.cfg.model_name,
        "turns": snap.turns,
        "device_cache_skip_ratio": snap.device_cache_skip_ratio,
    });
    (status, Json(body)).into_response()
}

async fn metrics(State(st): State<AppState>) -> Response {
    (
        [(
            header::CONTENT_TYPE,
            HeaderValue::from_static("text/plain; version=0.0.4"),
        )],
        crate::metrics::render(&st.health),
    )
        .into_response()
}

async fn not_found(State(st): State<AppState>) -> ApiError {
    st.health.count_error(ErrorCode::NotFound);
    api_err(ErrorCode::NotFound, "no such route")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ReasoningMode;
    use axum::body::Body;
    use golem_engine::TurnOutcome;
    use tower::ServiceExt;

    /// Router over a scripted worker: every job gets `events`, then Done.
    fn scripted(events: Vec<Event>) -> Router {
        let (jobs, rx) = std::sync::mpsc::sync_channel::<Job>(0);
        std::thread::spawn(move || {
            if let Ok(job) = rx.recv() {
                assert_eq!(job.reasoning, ReasoningMode::Separate);
                for e in events {
                    job.events.blocking_send(e).unwrap();
                }
                job.events
                    .blocking_send(Event::Done(TurnOutcome {
                        prompt_tokens: 3,
                        completion_tokens: 2,
                        finish: EngineFinish::Stop,
                    }))
                    .unwrap();
            }
        });
        let mut cfg = crate::test_support::test_server_config();
        cfg.reasoning = ReasoningMode::Separate;
        let health = Arc::new(HealthState::new());
        health.set(ServeState::Healthy);
        router(AppState {
            cfg: Arc::new(cfg),
            jobs,
            health,
        })
    }

    async fn post(app: Router, stream: bool) -> String {
        let body = serde_json::json!({"model":"tiny-moe","messages":[{"role":"user","content":"hi"}],"stream":stream});
        let r = app
            .oneshot(
                axum::http::Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
        String::from_utf8(
            axum::body::to_bytes(r.into_body(), 1 << 20)
                .await
                .unwrap()
                .to_vec(),
        )
        .unwrap()
    }

    fn split_events() -> Vec<Event> {
        vec![
            Event::Reasoning("plan".into()),
            Event::Text("answer".into()),
        ]
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn api_key_guards_v1_but_not_health() {
        let mut cfg = crate::test_support::test_server_config();
        cfg.api_key = Some("s3cret".into());
        let (jobs, _rx) = std::sync::mpsc::sync_channel::<Job>(0);
        let health = Arc::new(HealthState::new());
        health.set(ServeState::Healthy);
        let app = router(AppState {
            cfg: Arc::new(cfg),
            jobs,
            health,
        });
        let get = |path: &str, key: Option<&str>| {
            let mut r = axum::http::Request::get(path);
            if let Some(k) = key {
                r = r.header("authorization", format!("Bearer {k}"));
            }
            r.body(Body::empty()).unwrap()
        };
        let r = app.clone().oneshot(get("/v1/models", None)).await.unwrap();
        assert_eq!(r.status(), 401);
        let v: serde_json::Value =
            serde_json::from_slice(&axum::body::to_bytes(r.into_body(), 1 << 16).await.unwrap())
                .unwrap();
        assert_eq!(v["error"]["code"], "golem_invalid_request");
        let r = app
            .clone()
            .oneshot(get("/v1/models", Some("wrong")))
            .await
            .unwrap();
        assert_eq!(r.status(), 401);
        let r = app
            .clone()
            .oneshot(get("/v1/models", Some("s3cret")))
            .await
            .unwrap();
        assert_eq!(r.status(), 200);
        let r = app.clone().oneshot(get("/health", None)).await.unwrap();
        assert_eq!(r.status(), 200);
    }

    #[test]
    fn constant_time_eq_matches_only_equal_keys() {
        assert!(ct_eq(b"abc", b"abc"));
        assert!(!ct_eq(b"abc", b"abd"));
        assert!(!ct_eq(b"abc", b"abcd"));
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn separate_mode_fills_reasoning_content() {
        let v: serde_json::Value =
            serde_json::from_str(&post(scripted(split_events()), false).await).unwrap();
        let m = &v["choices"][0]["message"];
        assert_eq!(
            (&m["reasoning_content"], &m["content"]),
            (&"plan".into(), &"answer".into())
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn sse_carries_reasoning_content_deltas() {
        let body = post(scripted(split_events()), true).await;
        let data: Vec<serde_json::Value> = body
            .lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter(|l| *l != "[DONE]")
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(data[1]["choices"][0]["delta"]["reasoning_content"], "plan");
        assert_eq!(data[2]["choices"][0]["delta"]["content"], "answer");
        assert_eq!(data[3]["choices"][0]["finish_reason"], "stop");
        assert!(body.trim_end().ends_with("data: [DONE]"));
    }
}
