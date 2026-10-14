//! `/v1/messages` over a real tiny-moe worker (design §8.6): text, stream
//! framing, a forced tool round trip (Qwen2.5 template, as in
//! `tools_tiny_moe.rs`), and the official Anthropic Python SDK.
use axum::body::Body;
use golem_server::router_for_test;
use http::Request;
use serde_json::{Value, json};
use tower::ServiceExt;

async fn app() -> axum::Router {
    router_for_test(golem_engine::config::EngineConfig {
        chat_template: Some(include_str!("qwen2.5-tools.jinja").into()),
        ..golem_server::test_engine_config()
    })
    .await
}

async fn post(app: axum::Router, body: Value) -> (u16, String) {
    let r = app
        .oneshot(
            Request::post("/v1/messages")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let s = r.status().as_u16();
    let b = axum::body::to_bytes(r.into_body(), 1 << 22).await.unwrap();
    (s, String::from_utf8(b.to_vec()).unwrap())
}

fn tools_req(stream: bool) -> Value {
    json!({
        "model": "tiny-moe", "max_tokens": 256, "temperature": 0, "stream": stream,
        "messages": [{"role": "user", "content": "call f"}],
        "tools": [{"name": "f", "input_schema": {
            "type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}}],
        "tool_choice": {"type": "tool", "name": "f"},
    })
}

#[tokio::test(flavor = "multi_thread")]
async fn text_turn_is_a_message() {
    let (s, b) = post(
        app().await,
        json!({"model": "tiny-moe", "max_tokens": 4, "temperature": 0,
               "system": [{"type": "text", "text": "be brief"}],
               "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(s, 200, "{b}");
    let v: Value = serde_json::from_str(&b).unwrap();
    assert_eq!(
        (&v["type"], &v["role"]),
        (&json!("message"), &json!("assistant"))
    );
    assert!(v["id"].as_str().unwrap().starts_with("msg_"));
    assert_eq!(v["stop_reason"], "max_tokens", "{b}");
    assert_eq!(v["usage"]["output_tokens"], 4);
    assert!(v["usage"]["input_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn forced_tool_round_trip() {
    let app = app().await;
    let (s, b) = post(app.clone(), tools_req(false)).await;
    assert_eq!(s, 200, "{b}");
    let v: Value = serde_json::from_str(&b).unwrap();
    assert_eq!(v["stop_reason"], "tool_use", "{b}");
    let call = v["content"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["type"] == "tool_use")
        .unwrap_or_else(|| panic!("no tool_use block in {b}"));
    assert_eq!(call["name"], "f");
    assert!(call["input"]["x"].is_i64(), "{call}");
    // Feed the call and its result back: the next turn is accepted.
    let mut next = tools_req(false);
    next["tool_choice"] = json!({"type": "none"});
    next["max_tokens"] = 4.into();
    next["messages"] = json!([
        {"role": "user", "content": "call f"},
        {"role": "assistant", "content": v["content"]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call["id"], "content": "7"}]},
    ]);
    let (s, b) = post(app, next).await;
    assert_eq!(s, 200, "{b}");
}

#[tokio::test(flavor = "multi_thread")]
async fn forced_tool_streams_input_json_deltas() {
    let (s, b) = post(app().await, tools_req(true)).await;
    assert_eq!(s, 200, "{b}");
    let frames: Vec<(String, Value)> = b
        .split("\n\n")
        .filter(|f| !f.trim().is_empty())
        .map(|f| {
            let ev = f
                .lines()
                .find_map(|l| l.strip_prefix("event: "))
                .unwrap_or_default();
            let data = f
                .lines()
                .find_map(|l| l.strip_prefix("data: "))
                .unwrap_or_default();
            (ev.to_owned(), serde_json::from_str(data).unwrap())
        })
        .collect();
    assert_eq!(frames[0].0, "message_start");
    assert_eq!(frames.last().unwrap().0, "message_stop");
    for (ev, d) in &frames {
        assert_eq!(
            ev,
            d["type"].as_str().unwrap(),
            "event name matches data.type"
        );
    }
    let start = frames
        .iter()
        .find(|(_, d)| d["content_block"]["type"] == "tool_use")
        .unwrap_or_else(|| panic!("no tool_use block start in {b}"));
    let idx = &start.1["index"];
    let args: String = frames
        .iter()
        .filter(|(_, d)| &d["index"] == idx && d["delta"]["type"] == "input_json_delta")
        .filter_map(|(_, d)| d["delta"]["partial_json"].as_str())
        .collect();
    let args: Value = serde_json::from_str(&args).unwrap_or_else(|e| panic!("{args:?}: {e}\n{b}"));
    assert!(args["x"].is_i64(), "{args}");
    let delta = frames.iter().find(|(e, _)| e == "message_delta").unwrap();
    assert_eq!(delta.1["delta"]["stop_reason"], "tool_use");
}

/// Opt-in: the official Anthropic Python SDK, unmodified, against a real
/// `golem serve` socket. Needs `uv` and network for the first package fetch;
/// either missing is a failure, never a skip.
#[tokio::test(flavor = "multi_thread")]
#[ignore = "set GOLEM_ANTHROPIC_SDK=1 and run with --run-ignored"]
async fn anthropic_python_sdk_runs_unmodified() {
    assert!(
        std::env::var_os("GOLEM_ANTHROPIC_SDK").is_some(),
        "GOLEM_ANTHROPIC_SDK=1 is required for this leg"
    );
    let cfg = golem_engine::config::EngineConfig {
        chat_template: Some(include_str!("qwen2.5-tools.jinja").into()),
        ..golem_server::test_engine_config()
    };
    let (addr, stop, task) =
        golem_server::test_support::serve_on_port0(cfg, Default::default()).await;
    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/anthropic/sdk_check.py");
    let out = tokio::task::spawn_blocking(move || {
        std::process::Command::new("uv")
            .args(["run", "--quiet", "--with", "anthropic", "python", script])
            .arg(format!("http://{addr}"))
            .output()
            .expect("`uv` must be on PATH for the SDK leg")
    })
    .await
    .expect("SDK task");
    let _ = stop.send(());
    let _ = task.await;
    assert!(
        out.status.success(),
        "SDK check failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    for leg in ["text ok", "stream ok", "tools ok"] {
        assert!(stdout.contains(leg), "missing `{leg}`:\n{stdout}");
    }
}
