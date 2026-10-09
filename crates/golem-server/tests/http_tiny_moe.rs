//! HTTP surface over a real tiny-moe worker (non-pooled: pooled sessions are
//! refused concurrently under `cargo test`).
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::time::{Duration, Instant};

use axum::body::Body;
use golem_server::ServeError;
use golem_server::router_for_test;
use golem_server::test_support::{TestHooks, router_with_hooks, serve_on_port0};
use http::Request;
use tower::ServiceExt;

fn chat(body: serde_json::Value) -> Request<Body> {
    Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap()
}

async fn json(r: axum::response::Response) -> serde_json::Value {
    serde_json::from_slice(&axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap()).unwrap()
}

fn hi(max_tokens: u32) -> serde_json::Value {
    serde_json::json!({"model":"tiny-moe","messages":[{"role":"user","content":"hi"}],"max_tokens":max_tokens,"temperature":0})
}

#[tokio::test(flavor = "multi_thread")]
async fn non_streaming_chat_models_health_and_404() {
    let app = router_for_test(golem_server::test_engine_config()).await;
    let r = app.clone().oneshot(chat(hi(4))).await.unwrap();
    assert_eq!(r.status(), 200);
    let v = json(r).await;
    assert_eq!(v["object"], "chat.completion");
    assert!(
        (1..=4).contains(&v["usage"]["completion_tokens"].as_u64().unwrap()),
        "random weights may EOG early"
    );
    let r = app
        .clone()
        .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(json(r).await["data"][0]["id"], "tiny-moe");
    let r = app
        .clone()
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    assert_eq!(json(r).await["state"], "healthy");
    let r = app
        .clone()
        .oneshot(Request::get("/nope").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(r.status(), 404);
    assert_eq!(json(r).await["error"]["code"], "golem_not_found");
}

#[tokio::test(flavor = "multi_thread")]
async fn unsupported_field_is_400_envelope() {
    let app = router_for_test(golem_server::test_engine_config()).await;
    let body = serde_json::json!({"model":"tiny-moe","messages":[],"logprobs":true});
    let r = app.clone().oneshot(chat(body)).await.unwrap();
    assert_eq!(r.status(), 400);
    assert_eq!(json(r).await["error"]["code"], "golem_unsupported_field");
    // Supported by policy but not wired yet: refused, never silently ignored.
    for extra in [
        serde_json::json!({"stream": true}),
        serde_json::json!({"tools": [{"type":"function","function":{"name":"f"}}]}),
    ] {
        let mut body = hi(4);
        body.as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        let r = app.clone().oneshot(chat(body)).await.unwrap();
        assert_eq!(r.status(), 400);
        assert_eq!(json(r).await["error"]["code"], "golem_unsupported_field");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn back_to_back_requests_are_both_accepted() {
    let app = router_for_test(golem_server::test_engine_config()).await;
    for _ in 0..2 {
        let r = app.clone().oneshot(chat(hi(4))).await.unwrap();
        assert_eq!(r.status(), 200);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn unset_top_p_still_samples() {
    let app = router_for_test(golem_server::test_engine_config()).await;
    let mut outs = Vec::new();
    for seed in [1, 2] {
        let body = serde_json::json!({"model":"tiny-moe","messages":[{"role":"user","content":"hi"}],
            "max_tokens":32,"temperature":0.8,"seed":seed});
        let r = app.clone().oneshot(chat(body)).await.unwrap();
        assert_eq!(r.status(), 200);
        outs.push(json(r).await["choices"][0]["message"]["content"].clone());
    }
    assert_ne!(outs[0], outs[1], "top_p unset must not collapse to greedy");
}

/// Writes a raw HTTP request; returns the open socket.
fn raw_post(addr: SocketAddr, body: &serde_json::Value) -> TcpStream {
    let body = body.to_string();
    let mut s = TcpStream::connect(addr).unwrap();
    write!(
        s,
        "POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\n\
         content-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    s
}

fn raw_response(mut s: TcpStream) -> String {
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn non_streaming_disconnect_frees_slot() {
    // 100 ms/piece: an uncancelled 4096-token job (tiny-moe EOGs after
    // ~200 pieces) would hold the slot ≫ 10 s.
    let (addr, stop, task) = serve_on_port0(
        golem_server::test_engine_config(),
        TestHooks::slow_decode(100),
    )
    .await;
    let resp = tokio::task::spawn_blocking(move || {
        let s = raw_post(addr, &hi(4096));
        std::thread::sleep(Duration::from_millis(200));
        drop(s);
        let t = Instant::now();
        // Busy until the worker observes the cancel; retry like a client
        // honouring `Retry-After`, within the 10 s bound.
        loop {
            let resp = raw_response(raw_post(addr, &hi(2)));
            if !resp.starts_with("HTTP/1.1 503") || t.elapsed() > Duration::from_secs(10) {
                break (resp, t.elapsed());
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    })
    .await
    .unwrap();
    assert!(resp.0.starts_with("HTTP/1.1 200"), "{}", resp.0);
    assert!(
        resp.1 < Duration::from_secs(10),
        "slot freed late: {:?}",
        resp.1
    );
    stop.send(()).unwrap();
    task.await.unwrap().unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_death_mid_job_is_500_not_truncated_200() {
    let app = router_with_hooks(
        golem_server::test_engine_config(),
        TestHooks::panic_after_first_token(),
    )
    .await;
    let r = app.clone().oneshot(chat(hi(16))).await.unwrap();
    assert_eq!(r.status(), 500);
    assert_eq!(json(r).await["error"]["code"], "golem_backend_unavailable");
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_panic_reports_fatal_not_busy() {
    let app = router_with_hooks(
        golem_server::test_engine_config(),
        TestHooks::panic_at_start(),
    )
    .await;
    let r = app.clone().oneshot(chat(hi(4))).await.unwrap();
    assert_eq!(r.status(), 500);
    let r = app.clone().oneshot(chat(hi(4))).await.unwrap();
    assert_eq!(r.status(), 500);
    assert_eq!(json(r).await["error"]["code"], "golem_backend_unavailable");
    let r = app
        .clone()
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(json(r).await["state"], "fatal");
}

#[tokio::test(flavor = "multi_thread")]
async fn shutdown_joins_worker_during_active_request() {
    let (addr, stop, task) = serve_on_port0(
        golem_server::test_engine_config(),
        TestHooks::slow_decode(100),
    )
    .await;
    assert_eq!(golem_engine::p3_glue::live_sessions(), 1);
    let client = tokio::task::spawn_blocking(move || raw_response(raw_post(addr, &hi(4096))));
    tokio::time::sleep(Duration::from_millis(300)).await;
    stop.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(30), task)
        .await
        .expect("serve returns within 30 s")
        .unwrap()
        .unwrap();
    assert_eq!(golem_engine::p3_glue::live_sessions(), 0);
    let resp = client.await.unwrap();
    assert!(
        resp.starts_with("HTTP/1.1 503") && resp.contains("server draining"),
        "{resp}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn worker_death_ends_serve_with_error() {
    let (addr, _stop, task) = serve_on_port0(
        golem_server::test_engine_config(),
        TestHooks::panic_at_start(),
    )
    .await;
    let resp = tokio::task::spawn_blocking(move || raw_response(raw_post(addr, &hi(4))))
        .await
        .unwrap();
    assert!(resp.starts_with("HTTP/1.1 500"), "{resp}");
    let res = tokio::time::timeout(Duration::from_secs(30), task)
        .await
        .expect("serve returns after worker death")
        .unwrap();
    assert!(
        matches!(res, Err(ServeError::WorkerDied("fatal"))),
        "{res:?}"
    );
    assert_eq!(golem_engine::p3_glue::live_sessions(), 0);
}
