//! HTTP surface (tech design §8.1): routes, error envelope mapping, and the
//! handler side of the job hand-off.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
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

/// `/v1/chat/completions`, `/v1/models`, `/health`; everything else 404.
pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat))
        .route("/v1/models", get(models))
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

fn authorize(cfg: &ServerConfig, headers: &HeaderMap) -> Result<(), ApiError> {
    let Some(key) = &cfg.api_key else {
        return Ok(());
    };
    let got = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    if got == Some(key.as_str()) {
        Ok(())
    } else {
        Err(ApiError(
            StatusCode::UNAUTHORIZED,
            ErrorEnvelope::new(ErrorCode::InvalidRequest, "missing or wrong API key"),
        ))
    }
}

async fn chat(
    State(st): State<AppState>,
    headers: HeaderMap,
    body: Result<Json<serde_json::Value>, axum::extract::rejection::JsonRejection>,
) -> Result<Response, ApiError> {
    authorize(&st.cfg, &headers)?;
    let Json(raw) = body.map_err(|e| api_err(ErrorCode::InvalidRequest, e.body_text()))?;
    let req = golem_api::policy::validate(&raw, &st.cfg.model_name)?;
    match st.health.get() {
        ServeState::Draining => return Err(api_err(ErrorCode::Busy, "server draining")),
        ServeState::Fatal | ServeState::FatalIo => return Err(backend_gone()),
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
    loop {
        match st.jobs.try_send(job) {
            Ok(()) => break,
            Err(TrySendError::Disconnected(_)) => return Err(backend_gone()),
            Err(TrySendError::Full(j)) => {
                if Instant::now() >= deadline {
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

async fn models(
    State(st): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<ModelList>, ApiError> {
    authorize(&st.cfg, &headers)?;
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

async fn health(State(st): State<AppState>) -> Response {
    let state = st.health.get();
    let snap = st.health.snapshot();
    let status = match state {
        ServeState::Healthy | ServeState::Degraded => StatusCode::OK,
        ServeState::Loading | ServeState::Draining => StatusCode::SERVICE_UNAVAILABLE,
        ServeState::Fatal | ServeState::FatalIo => StatusCode::INTERNAL_SERVER_ERROR,
    };
    let body = serde_json::json!({
        "state": state.as_str(),
        "model": st.cfg.model_name,
        "turns": snap.turns,
        "device_cache_skip_ratio": snap.device_cache_skip_ratio,
    });
    (status, Json(body)).into_response()
}

async fn not_found() -> ApiError {
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
