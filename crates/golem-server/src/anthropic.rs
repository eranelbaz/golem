//! `POST /v1/messages` (design §8.6): the Anthropic request is translated
//! into the chat request, run through the same `policy::validate`, worker,
//! template and tool parser, and the result translated back.

use std::time::Duration;

use axum::Json;
use axum::extract::State;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::sse::{Event as SseEvent, KeepAlive};
use axum::response::{IntoResponse, Response};
use golem_api::anthropic::{error_body, error_type};
use golem_api::chat::ToolCall;
use golem_api::{ErrorCode, ErrorEnvelope};
use golem_engine::{FinishReason as EngineFinish, TurnOutcome};
use serde_json::{Value, json};

use crate::ReasoningMode;
use crate::app::{ApiError, AppState, Collected, backend_gone_envelope, collect, submit};
use crate::worker::{Event, HealthState};

/// Real `ping` events, not SSE comments: a cold agent prefill on the target
/// runs for minutes (≈ 14k tokens / 5.6 tok/s ≈ 42 min,
/// `docs/baselines/m3.6-prefill-stream.md`) with no content events.
const PING_EVERY: Duration = Duration::from_secs(10);

/// An [`ApiError`] rendered as the Anthropic envelope.
pub(crate) struct AnthropicError(pub ApiError);

impl IntoResponse for AnthropicError {
    fn into_response(self) -> Response {
        let ApiError(status, env) = self.0;
        let kind = match status {
            StatusCode::UNAUTHORIZED => "authentication_error",
            _ => error_type(env.error.code),
        };
        let mut r = (status, Json(error_body(&env, kind))).into_response();
        if env.error.code == ErrorCode::Busy {
            r.headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("1"));
        }
        r
    }
}

pub(crate) async fn messages(
    State(st): State<AppState>,
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Response {
    let health = st.health.clone();
    match inner(st, body).await {
        Ok(r) => r,
        Err(e) => {
            health.count_error(e.1.error.code);
            AnthropicError(e).into_response()
        }
    }
}

async fn inner(
    st: AppState,
    body: Result<Json<Value>, axum::extract::rejection::JsonRejection>,
) -> Result<Response, ApiError> {
    let Json(raw) = body.map_err(|e| {
        ApiError::from(ErrorEnvelope::new(ErrorCode::InvalidRequest, e.body_text()))
    })?;
    let chat = golem_api::anthropic::to_chat(&raw)?;
    let req = golem_api::policy::validate(&chat, &st.cfg.model_name)?;
    let stream = req.stream == Some(true);
    let stops = req.stop_list();
    // Reasoning always arrives as `thinking` blocks, never inline in `text`.
    let (mut rx, guard) = submit(&st, req, ReasoningMode::Separate).await?;
    let id = format!("msg_{}", uuid::Uuid::new_v4().simple());
    let model = st.cfg.model_name.clone();
    if stream {
        let start = json!({
            "type": "message_start",
            "message": {
                "id": id, "type": "message", "role": "assistant", "content": [],
                "model": model, "stop_reason": null, "stop_sequence": null,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        });
        let mut blocks = Blocks {
            stops,
            health: st.health.clone(),
            open: None,
            index: 0,
            args: String::new(),
            bad_args: None,
        };
        let ping = KeepAlive::new()
            .interval(PING_EVERY)
            .event(event("ping", &json!({"type": "ping"})));
        return Ok(crate::sse::stream(
            vec![event("message_start", &start)],
            rx,
            guard,
            ping,
            move |ev| blocks.frames(ev),
        )
        .into_response());
    }
    let _guard = guard;
    let Collected {
        text,
        reasoning,
        calls,
        outcome,
    } = collect(&mut rx).await?;
    let mut content = Vec::new();
    if !reasoning.is_empty() {
        content.push(json!({"type": "thinking", "thinking": reasoning, "signature": ""}));
    }
    if !text.is_empty() {
        content.push(json!({"type": "text", "text": text}));
    }
    for c in &calls {
        content.push(tool_use(c, input(&c.function.arguments)?));
    }
    let (reason, seq) = stop(&outcome, &stops);
    Ok(Json(json!({
        "id": id, "type": "message", "role": "assistant", "model": model,
        "content": content, "stop_reason": reason, "stop_sequence": seq,
        "usage": {"input_tokens": outcome.prompt_tokens, "output_tokens": outcome.completion_tokens},
    }))
    .into_response())
}

fn event(name: &str, data: &Value) -> SseEvent {
    SseEvent::default().event(name).data(data.to_string())
}

fn tool_use(c: &ToolCall, input: Value) -> Value {
    json!({
        "type": "tool_use",
        "id": c.id.clone().unwrap_or_default(),
        "name": c.function.name.clone().unwrap_or_default(),
        "input": input,
    })
}

/// Parsed tool arguments; unparseable ones are a server error, never `{}`.
fn input(args: &str) -> Result<Value, ApiError> {
    if args.trim().is_empty() {
        return Ok(json!({}));
    }
    serde_json::from_str(args).map_err(|e| {
        ApiError::from(ErrorEnvelope::new(
            ErrorCode::BackendUnavailable,
            format!("tool call arguments are not valid JSON: {e}"),
        ))
    })
}

/// `stop_reason` and the matched `stop_sequence`.
fn stop(o: &TurnOutcome, stops: &[String]) -> (&'static str, Value) {
    match o.finish {
        EngineFinish::Length => ("max_tokens", Value::Null),
        EngineFinish::ToolCalls => ("tool_use", Value::Null),
        EngineFinish::Stop | EngineFinish::Cancelled => {
            match o.stop_sequence.and_then(|i| stops.get(i as usize)) {
                Some(s) => ("stop_sequence", s.as_str().into()),
                None => ("end_turn", Value::Null),
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Thinking,
    Text,
    /// Carries the worker's tool-call index.
    Tool(u32),
}

/// Streaming content-block state: one open block at a time, indices in
/// order of opening.
struct Blocks {
    stops: Vec<String>,
    health: std::sync::Arc<HealthState>,
    open: Option<Kind>,
    index: u32,
    /// Arguments of the open `tool_use` block, checked when it closes.
    args: String,
    /// First tool call whose arguments did not parse.
    bad_args: Option<ErrorEnvelope>,
}

impl Blocks {
    /// Closes the open block unless it is `kind` (a new tool call always
    /// opens a new block); then opens `body` as the next block.
    fn ensure(&mut self, kind: Kind, body: Value, out: &mut Vec<SseEvent>) {
        if self.open == Some(kind) && !matches!(kind, Kind::Tool(_)) {
            return;
        }
        self.close(out);
        out.push(event(
            "content_block_start",
            &json!({"type": "content_block_start", "index": self.index, "content_block": body}),
        ));
        self.open = Some(kind);
    }

    fn close(&mut self, out: &mut Vec<SseEvent>) {
        if let Some(Kind::Tool(_)) = self.open
            && let Err(ApiError(_, e)) = input(&std::mem::take(&mut self.args))
        {
            self.bad_args.get_or_insert(e);
        }
        if self.open.take().is_some() {
            out.push(event(
                "content_block_stop",
                &json!({"type": "content_block_stop", "index": self.index}),
            ));
            self.index += 1;
        }
    }

    fn delta(&self, delta: Value) -> SseEvent {
        event(
            "content_block_delta",
            &json!({"type": "content_block_delta", "index": self.index, "delta": delta}),
        )
    }

    fn error(&self, e: &ErrorEnvelope) -> SseEvent {
        self.health.count_error(e.error.code);
        event("error", &error_body(e, error_type(e.error.code)))
    }

    fn frames(&mut self, ev: Option<Event>) -> Vec<SseEvent> {
        let mut out = Vec::new();
        match ev {
            Some(Event::Reasoning(t)) => {
                self.ensure(
                    Kind::Thinking,
                    json!({"type": "thinking", "thinking": "", "signature": ""}),
                    &mut out,
                );
                out.push(self.delta(json!({"type": "thinking_delta", "thinking": t})));
            }
            Some(Event::Text(t)) => {
                self.ensure(Kind::Text, json!({"type": "text", "text": ""}), &mut out);
                out.push(self.delta(json!({"type": "text_delta", "text": t})));
            }
            Some(Event::ToolCall(c)) => {
                let kind = Kind::Tool(c.index.unwrap_or(0));
                // The first delta of a call carries its name.
                if c.function.name.is_some() {
                    self.ensure(kind, tool_use(&c, json!({})), &mut out);
                }
                if self.open != Some(kind) {
                    tracing::warn!("tool-call arguments for a closed block dropped");
                    self.bad_args.get_or_insert(ErrorEnvelope::new(
                        ErrorCode::BackendUnavailable,
                        "tool call arguments arrived after its block closed",
                    ));
                } else if !c.function.arguments.is_empty() {
                    self.args.push_str(&c.function.arguments);
                    out.push(self.delta(
                        json!({"type": "input_json_delta", "partial_json": c.function.arguments}),
                    ));
                }
            }
            Some(Event::Done(o)) => {
                self.close(&mut out);
                // Never `tool_use` with input the client cannot parse.
                if let Some(e) = self.bad_args.take() {
                    out.push(self.error(&e));
                    return out;
                }
                let (reason, seq) = stop(&o, &self.stops);
                out.push(event(
                    "message_delta",
                    &json!({
                        "type": "message_delta",
                        "delta": {"stop_reason": reason, "stop_sequence": seq},
                        "usage": {"input_tokens": o.prompt_tokens, "output_tokens": o.completion_tokens},
                    }),
                ));
                out.push(event("message_stop", &json!({"type": "message_stop"})));
            }
            Some(Event::Failed(e)) => out.push(self.error(&e)),
            None => out.push(self.error(&backend_gone_envelope())),
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::app::tests::scripted_with;
    use axum::body::Body;
    use golem_api::chat::FunctionCall;
    use tower::ServiceExt;

    fn outcome(finish: EngineFinish, stop_sequence: Option<u32>) -> TurnOutcome {
        TurnOutcome {
            prompt_tokens: 3,
            completion_tokens: 2,
            finish,
            stop_sequence,
        }
    }

    fn call(name: Option<&str>, args: &str) -> Event {
        Event::ToolCall(ToolCall {
            index: Some(0),
            id: name.map(|_| "call_1".into()),
            kind: name.map(|_| "function".into()),
            function: FunctionCall {
                name: name.map(Into::into),
                arguments: args.into(),
            },
        })
    }

    fn call2(name: Option<&str>, args: &str) -> Event {
        let Event::ToolCall(mut c) = call(name, args) else {
            unreachable!()
        };
        c.index = Some(1);
        c.id = name.map(|_| "call_2".into());
        Event::ToolCall(c)
    }

    /// Named scripted cases: worker events, outcome, request extras.
    fn cases() -> Vec<(&'static str, Vec<Event>, TurnOutcome, Value)> {
        let stop = EngineFinish::Stop;
        vec![
            (
                "text",
                vec![Event::Text("hel".into()), Event::Text("lo".into())],
                outcome(stop, None),
                json!({}),
            ),
            (
                "thinking_text",
                vec![
                    Event::Reasoning("plan".into()),
                    Event::Text("answer".into()),
                ],
                outcome(stop, None),
                json!({"thinking": {"type": "enabled", "budget_tokens": 1024}}),
            ),
            (
                "tool_use",
                vec![
                    Event::Reasoning("call f".into()),
                    call(Some("f"), "{\"x\":"),
                    call(None, "1}"),
                ],
                outcome(EngineFinish::ToolCalls, None),
                json!({"tools": [{"name": "f", "input_schema": {"type": "object"}}], "tool_choice": {"type": "any"}}),
            ),
            (
                "stop_sequence",
                vec![Event::Text("a".into())],
                outcome(stop, Some(1)),
                json!({"stop_sequences": ["X", "Y"]}),
            ),
            (
                "max_tokens",
                vec![Event::Text("a".into())],
                outcome(EngineFinish::Length, None),
                json!({}),
            ),
            (
                "two_tools_then_text",
                vec![
                    call(Some("f"), "{}"),
                    call2(Some("g"), "{\"y\":2}"),
                    Event::Text("done".into()),
                ],
                outcome(EngineFinish::ToolCalls, None),
                json!({"tools": [{"name": "f", "input_schema": {"type": "object"}}, {"name": "g", "input_schema": {"type": "object"}}]}),
            ),
            (
                "text_then_error",
                vec![
                    Event::Text("par".into()),
                    Event::Failed(ErrorEnvelope::new(
                        ErrorCode::BackendUnavailable,
                        "decode worker is not running",
                    )),
                ],
                outcome(stop, None),
                json!({}),
            ),
            (
                "bad_tool_args",
                vec![
                    call(Some("f"), "{\"x\":"),
                    Event::Text("oops".into()),
                    call(None, "1}"),
                ],
                outcome(EngineFinish::ToolCalls, None),
                json!({"tools": [{"name": "f", "input_schema": {"type": "object"}}]}),
            ),
            (
                "empty_stop_string_index",
                vec![Event::Text("a".into())],
                outcome(stop, Some(1)),
                json!({"stop_sequences": ["", "Z"]}),
            ),
        ]
    }

    fn request(extra: &Value, stream: bool) -> axum::http::Request<Body> {
        let mut body = json!({
            "model": "tiny-moe", "max_tokens": 16, "stream": stream,
            "messages": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t0", "name": "f", "input": {}}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t0", "content": "r"}]},
            ],
        });
        for (k, v) in extra.as_object().into_iter().flatten() {
            body[k] = v.clone();
        }
        axum::http::Request::post("/v1/messages")
            .header("content-type", "application/json")
            .header("anthropic-version", "2023-06-01")
            .body(Body::from(body.to_string()))
            .unwrap()
    }

    async fn send(app: axum::Router, req: axum::http::Request<Body>) -> (StatusCode, String) {
        let r = app.oneshot(req).await.unwrap();
        let status = r.status();
        let b = axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap();
        (status, String::from_utf8(b.to_vec()).unwrap())
    }

    /// `id` normalised; streams one `event data` line per frame.
    fn normalize(body: &str, stream: bool) -> String {
        let fix = |mut v: Value| {
            for p in ["/id", "/message/id"] {
                if let Some(id) = v.pointer_mut(p) {
                    *id = "ID".into();
                }
            }
            v.to_string()
        };
        if !stream {
            return fix(serde_json::from_str(body).unwrap()) + "\n";
        }
        body.split("\n\n")
            .filter(|f| !f.trim().is_empty())
            .map(|f| {
                let ev = f
                    .lines()
                    .find_map(|l| l.strip_prefix("event: "))
                    .unwrap_or("");
                let data = f
                    .lines()
                    .find_map(|l| l.strip_prefix("data: "))
                    .unwrap_or("");
                format!("{ev} {}\n", fix(serde_json::from_str(data).unwrap()))
            })
            .collect()
    }

    fn golden(name: &str, got: &str) {
        let path = format!(
            "{}/tests/golden/anthropic/{name}",
            env!("CARGO_MANIFEST_DIR")
        );
        if std::env::var_os("GOLEM_BLESS").is_some() {
            std::fs::create_dir_all(std::path::Path::new(&path).parent().unwrap()).unwrap();
            std::fs::write(&path, got).unwrap();
        }
        let want = std::fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("golden {name} missing: run once with GOLEM_BLESS=1"));
        assert_eq!(got, want, "{name}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn scripted_turns_match_goldens_stream_and_not() {
        for (name, events, out, extra) in cases() {
            for stream in [false, true] {
                let app = scripted_with(events.clone(), out);
                let (status, body) = send(app, request(&extra, stream)).await;
                // Non-stream merges argument deltas by call index, so only
                // `text_then_error` fails there.
                let failing = name == "text_then_error";
                assert_eq!(status == 200, stream || !failing, "{name}: {status} {body}");
                let ext = if stream { "sse" } else { "json" };
                golden(&format!("{name}.{ext}"), &normalize(&body, stream));
            }
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn worker_failure_is_an_anthropic_error_stream_and_not() {
        for stream in [false, true] {
            let (jobs, rx) = std::sync::mpsc::sync_channel::<crate::Job>(0);
            std::thread::spawn(move || {
                if let Ok(job) = rx.recv() {
                    job.events
                        .blocking_send(Event::Failed(ErrorEnvelope::new(
                            ErrorCode::Busy,
                            "server draining",
                        )))
                        .unwrap();
                }
            });
            let health = std::sync::Arc::new(HealthState::new());
            health.set(crate::ServeState::Healthy);
            let app = crate::router(AppState {
                cfg: std::sync::Arc::new(crate::test_support::test_server_config()),
                jobs,
                health,
            });
            let (status, body) = send(app, request(&json!({}), stream)).await;
            let ext = if stream { "sse" } else { "json" };
            if !stream {
                assert_eq!(status, 503);
            }
            golden(&format!("error.{ext}"), &normalize(&body, stream));
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn validation_errors_use_the_anthropic_envelope() {
        let app = scripted_with(Vec::new(), outcome(EngineFinish::Stop, None));
        let (status, body) = send(app, request(&json!({"model": "other"}), false)).await;
        assert_eq!(status, 400);
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["type"], "error");
        assert_eq!(v["error"]["type"], "invalid_request_error");
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .starts_with("golem_model_mismatch: ")
        );
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn x_api_key_or_bearer_authorize_and_401_is_authentication_error() {
        let mut cfg = crate::test_support::test_server_config();
        cfg.api_key = Some("s3cret".into());
        let (jobs, _rx) = std::sync::mpsc::sync_channel::<crate::Job>(0);
        let health = std::sync::Arc::new(HealthState::new());
        health.set(crate::ServeState::Healthy);
        let app = crate::router(AppState {
            cfg: std::sync::Arc::new(cfg),
            jobs,
            health,
        });
        let get = |path: &str, hdr: Option<(&str, &str)>| {
            let mut r = axum::http::Request::get(path);
            if let Some((k, v)) = hdr {
                r = r.header(k, v);
            }
            r.body(Body::empty()).unwrap()
        };
        let (s, b) = send(app.clone(), get("/v1/messages/count_tokens", None)).await;
        assert_eq!(s, 401);
        let v: Value = serde_json::from_str(&b).unwrap();
        assert_eq!(v["error"]["type"], "authentication_error");
        let (s, _) = send(
            app.clone(),
            get("/v1/models", Some(("x-api-key", "s3cret"))),
        )
        .await;
        assert_eq!(s, 200);
        let (s, _) = send(app.clone(), get("/v1/models", Some(("x-api-key", "wrong")))).await;
        assert_eq!(s, 401);
        let (s, b) = send(
            app.clone(),
            get(
                "/v1/messages/count_tokens",
                Some(("authorization", "Bearer s3cret")),
            ),
        )
        .await;
        assert_eq!(s, 404);
        let v: Value = serde_json::from_str(&b).unwrap();
        assert_eq!(v["error"]["type"], "not_found_error");
        let (s, b) = send(app, get("/v1/models", None)).await;
        assert_eq!(s, 401);
        let v: Value = serde_json::from_str(&b).unwrap();
        assert_eq!(
            v["error"]["code"], "golem_invalid_request",
            "OpenAI routes keep §8.4"
        );
    }
}
