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

fn hi_stream(max_tokens: u32) -> serde_json::Value {
    let mut b = hi(max_tokens);
    b["stream"] = true.into();
    b
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

/// Body-and-headers of a raw GET.
fn get(addr: SocketAddr, path: &str) -> String {
    let mut s = TcpStream::connect(addr).unwrap();
    write!(
        s,
        "GET {path} HTTP/1.1\r\nhost: x\r\nconnection: close\r\n\r\n"
    )
    .unwrap();
    raw_response(s)
}

fn raw_response(mut s: TcpStream) -> String {
    let mut out = String::new();
    s.read_to_string(&mut out).unwrap();
    out
}

/// Body of a chunked `raw_response`.
fn dechunk(resp: &str) -> String {
    let (_, mut rest) = resp.split_once("\r\n\r\n").expect(resp);
    let mut body = String::new();
    while let Some((size, tail)) = rest.split_once("\r\n") {
        let n = usize::from_str_radix(size, 16).expect(resp);
        if n == 0 {
            break;
        }
        body.push_str(&tail[..n]);
        rest = &tail[n + 2..];
    }
    body
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

#[tokio::test(flavor = "multi_thread")]
async fn streaming_worker_death_ends_with_error_frame_then_done() {
    let app = router_with_hooks(
        golem_server::test_engine_config(),
        TestHooks::panic_after_first_token(),
    )
    .await;
    let r = app.clone().oneshot(chat(hi_stream(16))).await.unwrap();
    assert_eq!(r.status(), 200, "headers precede the failure");
    let body = String::from_utf8(
        axum::body::to_bytes(r.into_body(), 1 << 20)
            .await
            .unwrap()
            .to_vec(),
    )
    .unwrap();
    assert_error_then_done(&body, "golem_backend_unavailable");
}

/// The stream tail is exactly one error frame then one `data: [DONE]`.
fn assert_error_then_done(body: &str, needle: &str) {
    assert_eq!(body.matches("[DONE]").count(), 1, "{body}");
    let head = body
        .strip_suffix("\n\ndata: [DONE]\n\n")
        .unwrap_or_else(|| panic!("no trailing [DONE]: {body}"));
    let last = &head[head.rfind("data: ").expect(body)..];
    assert!(
        last.starts_with("data: {\"error\"") && last.contains(needle),
        "{body}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn streaming_disconnect_frees_slot() {
    let (addr, stop, task) = serve_on_port0(
        golem_server::test_engine_config(),
        TestHooks::slow_decode(100),
    )
    .await;
    let resp = tokio::task::spawn_blocking(move || {
        let mut s = raw_post(addr, &hi_stream(4096));
        // Wait for the first SSE frame, then disconnect mid-generation.
        let mut buf = [0u8; 512];
        let mut seen = String::new();
        while !seen.contains("data: ") {
            let n = s.read(&mut buf).unwrap();
            assert!(n > 0, "stream closed early: {seen}");
            seen.push_str(&String::from_utf8_lossy(&buf[..n]));
        }
        drop(s);
        let t = Instant::now();
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
async fn streaming_shutdown_ends_with_draining_error_frame_then_done() {
    let (addr, stop, task) = serve_on_port0(
        golem_server::test_engine_config(),
        TestHooks::slow_decode(100),
    )
    .await;
    let client =
        tokio::task::spawn_blocking(move || raw_response(raw_post(addr, &hi_stream(4096))));
    tokio::time::sleep(Duration::from_millis(300)).await;
    stop.send(()).unwrap();
    tokio::time::timeout(Duration::from_secs(30), task)
        .await
        .expect("serve returns within 30 s")
        .unwrap()
        .unwrap();
    let resp = client.await.unwrap();
    assert!(resp.starts_with("HTTP/1.1 200"), "{resp}");
    assert_error_then_done(&dechunk(&resp), "server draining");
}

#[tokio::test(flavor = "multi_thread")]
async fn metrics_parse_as_prometheus_text() {
    let body = golem_server::test_support::get_text("/metrics").await;
    for line in body
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let (name, val) = line.rsplit_once(' ').expect(line);
        assert!(
            name.split('{')
                .next()
                .unwrap()
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_'),
            "{line}"
        );
        val.parse::<f64>().unwrap_or_else(|_| panic!("{line}"));
    }
    assert!(body.contains("golem_queue_depth "), "{body}");
    assert!(
        body.contains("golem_errors_total{code=\"golem_busy\"} "),
        "{body}"
    );
    // on HIP builds the series is emitted and must be 0; on CPU it is omitted (on_cpu is nonzero there)
    #[cfg(golem_backend_hip)]
    assert!(body.contains("golem_expert_mul_mat_id_off_gpu_total 0"));
    #[cfg(golem_backend_hip)]
    assert!(
        body.contains("golem_expert_mul_mat_id_total{device=\"cpu\"} 0"),
        "{body}"
    );
    #[cfg(not(golem_backend_hip))]
    assert!(!body.contains("golem_expert_mul_mat_id_off_gpu_total"));
    #[cfg(not(golem_backend_hip))]
    assert!(!body.contains("golem_expert_mul_mat_id_total"));
    // Non-pooled: no device cache, so no fake-zero copy series.
    assert!(!body.contains("golem_expert_h2d_copies_total"), "{body}");
}

#[tokio::test(flavor = "multi_thread")]
async fn health_reports_tiers_and_degraded_list() {
    let app = router_for_test(golem_server::test_engine_config()).await;
    let r = app
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let v = json(r).await;
    assert_eq!(v["state"], "healthy");
    assert!(
        v["tiers"].as_object().unwrap().contains_key("vram_fill"),
        "{v}"
    );
    assert!(
        v["tiers"].as_object().unwrap().contains_key("ram_fill"),
        "{v}"
    );
    assert_eq!(v["degraded"], serde_json::json!([]));
}

/// Graceful drain (§9.2): the in-flight stream finishes with `[DONE]`, a
/// request arriving during the drain gets 503, and serve returns < 35 s.
#[tokio::test(flavor = "multi_thread")]
async fn drain_finishes_in_flight_stream_and_refuses_new_requests() {
    let mut scfg = golem_server::test_support::test_server_config();
    scfg.drain_grace = Duration::from_secs(30);
    let (addr, stop, task) = golem_server::test_support::serve_with(
        golem_server::test_engine_config(),
        TestHooks::slow_decode(50),
        scfg,
    )
    .await;
    let t = Instant::now();
    let client = tokio::task::spawn_blocking(move || raw_response(raw_post(addr, &hi_stream(48))));
    tokio::time::sleep(Duration::from_millis(300)).await;
    stop.send(()).unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    let late = tokio::task::spawn_blocking(move || raw_response(raw_post(addr, &hi(2))))
        .await
        .unwrap();
    assert!(
        late.starts_with("HTTP/1.1 503") && late.contains("golem_busy"),
        "{late}"
    );
    let resp = client.await.unwrap();
    assert!(
        resp.starts_with("HTTP/1.1 200") && resp.contains("[DONE]"),
        "{resp}"
    );
    tokio::time::timeout(Duration::from_secs(35), task)
        .await
        .expect("serve returns within 35 s")
        .unwrap()
        .unwrap();
    assert!(t.elapsed() < Duration::from_secs(35));
}

/// `--parallel 2` drain: both in-flight streams and the queued one finish
/// with `[DONE]` within the grace; a new request gets 503.
#[tokio::test(flavor = "multi_thread")]
async fn drain_parallel_2_finishes_queued_stream() {
    let mut scfg = golem_server::test_support::test_server_config();
    scfg.drain_grace = Duration::from_secs(30);
    let (addr, stop, task) = golem_server::test_support::serve_with(
        golem_engine::config::EngineConfig {
            parallel: 2,
            ..golem_server::test_engine_config()
        },
        TestHooks::slow_decode(50),
        scfg,
    )
    .await;
    let clients: Vec<_> = (0..3)
        .map(|_| tokio::task::spawn_blocking(move || raw_response(raw_post(addr, &hi_stream(48)))))
        .collect();
    let t = Instant::now();
    while !get(addr, "/metrics").contains("golem_queue_depth 1\n") {
        assert!(
            t.elapsed() < Duration::from_secs(10),
            "third request never queued"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    stop.send(()).unwrap();
    tokio::time::sleep(Duration::from_millis(200)).await;
    let late = tokio::task::spawn_blocking(move || raw_response(raw_post(addr, &hi(2))))
        .await
        .unwrap();
    assert!(
        late.starts_with("HTTP/1.1 503") && late.contains("golem_busy"),
        "{late}"
    );
    for c in clients {
        let resp = c.await.unwrap();
        assert!(
            resp.starts_with("HTTP/1.1 200") && resp.contains("[DONE]") && !resp.contains("error"),
            "{resp}"
        );
    }
    tokio::time::timeout(Duration::from_secs(35), task)
        .await
        .expect("serve returns within 35 s")
        .unwrap()
        .unwrap();
}

/// After a finished request the idle worker reports not busy, so shutdown
/// does not sit out the whole drain grace.
#[tokio::test(flavor = "multi_thread")]
async fn shutdown_after_finished_request_skips_grace() {
    let mut scfg = golem_server::test_support::test_server_config();
    scfg.drain_grace = Duration::from_secs(30);
    let (addr, stop, task) = golem_server::test_support::serve_with(
        golem_server::test_engine_config(),
        TestHooks::default(),
        scfg,
    )
    .await;
    let resp = tokio::task::spawn_blocking(move || raw_response(raw_post(addr, &hi(4))))
        .await
        .unwrap();
    assert!(resp.starts_with("HTTP/1.1 200"), "{resp}");
    let t = Instant::now();
    stop.send(()).unwrap();
    task.await.unwrap().unwrap();
    assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
}

/// Separate mode on the GLM template (prompt opens `<think>`): generated
/// text streams as `reasoning_content`, and no marker leaks into `content`.
#[tokio::test(flavor = "multi_thread")]
async fn separate_mode_glm_streams_reasoning_without_markers() {
    let cfg = golem_engine::config::EngineConfig {
        chat_template: Some(
            include_str!("../../golem-engine/tests/fixtures/glm-5.3-flash.jinja").into(),
        ),
        ..golem_server::test_engine_config()
    };
    let app = router_with_hooks(cfg, TestHooks::default()).await;
    let r = app.oneshot(chat(hi_stream(16))).await.unwrap();
    assert_eq!(r.status(), 200);
    let body = axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap();
    let body = String::from_utf8(body.to_vec()).unwrap();
    let deltas: Vec<serde_json::Value> = body
        .split("\n\n")
        .filter_map(|f| f.strip_prefix("data: "))
        .filter(|d| *d != "[DONE]")
        .map(|d| {
            serde_json::from_str::<serde_json::Value>(d).unwrap()["choices"][0]["delta"].clone()
        })
        .collect();
    let field = |k: &str| -> String { deltas.iter().filter_map(|d| d[k].as_str()).collect() };
    assert!(!field("reasoning_content").is_empty(), "{body}");
    assert!(!field("content").contains("think>"), "{body}");
    assert!(body.ends_with("data: [DONE]\n\n"), "{body}");
}
