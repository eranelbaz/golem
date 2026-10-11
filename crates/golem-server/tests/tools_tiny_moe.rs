//! Tools / tool_choice over a real tiny-moe worker. tiny-moe has random
//! weights, so only grammar-forced named calls are deterministic; its chatml
//! fallback has no tool syntax (the chat layer emits no grammar for it), so
//! the worker serves the vendored Qwen2.5 template (Hermes `<tool_call>`
//! JSON). `auto` parsing is covered by golem-engine's canned-text tests on
//! the GLM template.
use axum::body::Body;
use golem_server::router_for_test;

async fn app() -> axum::Router {
    router_for_test(golem_engine::config::EngineConfig {
        chat_template: Some(include_str!("qwen2.5-tools.jinja").into()),
        ..golem_server::test_engine_config()
    })
    .await
}
use http::Request;
use tower::ServiceExt;

fn chat(body: serde_json::Value) -> Request<Body> {
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn body(r: axum::response::Response) -> String {
    String::from_utf8(
        axum::body::to_bytes(r.into_body(), 1 << 22)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap()
}

fn tools_req(choice: serde_json::Value, stream: bool) -> serde_json::Value {
    serde_json::json!({
        "model": "tiny-moe",
        "messages": [{"role":"user","content":"call f"}],
        "tools": [{"type":"function","function":{"name":"f","parameters":{
            "type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}}}],
        "tool_choice": choice,
        "max_tokens": 256,
        "temperature": 0,
        "stream": stream,
    })
}

fn named_f() -> serde_json::Value {
    serde_json::json!({"type":"function","function":{"name":"f"}})
}

#[tokio::test(flavor = "multi_thread")]
async fn named_tool_choice_forces_the_call() {
    let app = app().await;
    let r = app
        .oneshot(chat(tools_req(named_f(), false)))
        .await
        .unwrap();
    let status = r.status();
    let b = body(r).await;
    assert_eq!(status, 200, "{b}");
    let v: serde_json::Value = serde_json::from_str(&b).unwrap();
    let c = &v["choices"][0];
    assert_eq!(c["finish_reason"], "tool_calls", "{b}");
    let call = &c["message"]["tool_calls"][0];
    assert_eq!(call["type"], "function");
    assert_eq!(call["function"]["name"], "f");
    let args: serde_json::Value =
        serde_json::from_str(call["function"]["arguments"].as_str().unwrap()).expect("JSON args");
    assert!(args["x"].is_i64(), "{args}");
}

#[tokio::test(flavor = "multi_thread")]
async fn named_tool_choice_streams_tool_call_deltas() {
    let app = app().await;
    let r = app.oneshot(chat(tools_req(named_f(), true))).await.unwrap();
    assert_eq!(r.status(), 200);
    let b = body(r).await;
    let data: Vec<serde_json::Value> = b
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .filter(|l| *l != "[DONE]")
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let deltas: Vec<&serde_json::Value> = data
        .iter()
        .filter_map(|d| d["choices"][0]["delta"]["tool_calls"].get(0))
        .collect();
    assert!(!deltas.is_empty(), "no delta.tool_calls in {b}");
    assert_eq!(deltas[0]["function"]["name"], "f");
    assert_eq!(deltas[0]["index"], 0);
    // Random weights may emit further calls; call 0's pieces must form its JSON.
    let args: String = deltas
        .iter()
        .filter(|d| d["index"] == 0)
        .filter_map(|d| d["function"]["arguments"].as_str())
        .collect();
    let args: serde_json::Value = serde_json::from_str(&args)
        .unwrap_or_else(|e| panic!("concatenated JSON args {args:?}: {e}\n{b}"));
    assert!(args["x"].is_i64(), "{args}");
    let finish = data
        .iter()
        .find_map(|d| d["choices"][0]["finish_reason"].as_str());
    assert_eq!(finish, Some("tool_calls"));
    assert!(b.trim_end().ends_with("data: [DONE]"));
}

#[tokio::test(flavor = "multi_thread")]
async fn auto_tools_request_is_accepted() {
    let app = app().await;
    let mut req = tools_req("auto".into(), false);
    req["max_tokens"] = 4.into();
    let r = app.oneshot(chat(req)).await.unwrap();
    assert_eq!(r.status(), 200, "{}", body(r).await);
}

#[tokio::test(flavor = "multi_thread")]
async fn named_function_outside_tools_is_400() {
    let app = app().await;
    let choice = serde_json::json!({"type":"function","function":{"name":"g"}});
    let r = app.oneshot(chat(tools_req(choice, false))).await.unwrap();
    assert_eq!(r.status(), 400);
    let v: serde_json::Value = serde_json::from_str(&body(r).await).unwrap();
    assert_eq!(v["error"]["code"], "golem_invalid_request");
}
