//! §8.1 several served models: a request naming the unloaded one swaps it
//! in; idle unload drops the model and the next request reloads it.

use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use golem_engine::config::EngineConfig;
use golem_server::test_support::{test_engine_config, test_server_config};
use golem_server::{AppState, ServerConfig, TurnSinks, router, spawn_worker_models};
use http::Request;
use serde_json::{Value, json};
use tower::ServiceExt;

async fn app(idle: Option<Duration>) -> (axum::Router, golem_server::WorkerHandle) {
    let hybrid = EngineConfig {
        model_path: xtask::tiny_hybrid::cached_fixture(),
        n_ctx: 4096,
        ..Default::default()
    };
    let w = tokio::task::spawn_blocking(move || {
        spawn_worker_models(
            vec![
                ("tiny-moe".into(), test_engine_config()),
                ("tiny-hybrid".into(), hybrid),
            ],
            idle,
            TurnSinks::default(),
        )
    })
    .await
    .unwrap()
    .expect("worker starts");
    let app = router(AppState {
        cfg: Arc::new(ServerConfig {
            models: vec!["tiny-hybrid".into()],
            ..test_server_config()
        }),
        jobs: w.jobs.clone(),
        health: w.health.clone(),
    });
    (app, w)
}

async fn ask(app: &axum::Router, model: &str) -> (u16, Value) {
    let body = json!({"model": model, "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4, "temperature": 0});
    let r = app
        .clone()
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("content-type", "application/json")
                .body(Body::from(body.to_string()))
                .unwrap(),
        )
        .await
        .unwrap();
    let s = r.status().as_u16();
    let b = axum::body::to_bytes(r.into_body(), 1 << 22).await.unwrap();
    (s, serde_json::from_slice(&b).unwrap())
}

#[tokio::test(flavor = "multi_thread")]
async fn request_for_the_other_model_swaps_it_in() {
    let (app, _w) = app(None).await;
    let (s, first) = ask(&app, "tiny-moe").await;
    assert_eq!(s, 200, "{first}");
    for m in ["tiny-hybrid", "tiny-moe"] {
        let (s, b) = ask(&app, m).await;
        assert_eq!(s, 200, "{b}");
        assert_eq!(b["model"], m);
    }
    let (_, again) = ask(&app, "tiny-moe").await;
    assert_eq!(
        again["choices"][0]["message"]["content"], first["choices"][0]["message"]["content"],
        "greedy output changed across a swap"
    );
    let r = app
        .clone()
        .oneshot(Request::get("/v1/models").body(Body::empty()).unwrap())
        .await
        .unwrap();
    let b: Value =
        serde_json::from_slice(&axum::body::to_bytes(r.into_body(), 1 << 16).await.unwrap())
            .unwrap();
    let ids: Vec<&str> = b["data"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|m| m["id"].as_str())
        .collect();
    assert_eq!(ids, ["tiny-moe", "tiny-hybrid"]);
    let (s, b) = ask(&app, "nope").await;
    assert_eq!(
        (s, b["error"]["code"].as_str()),
        (400, Some("golem_model_mismatch"))
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn idle_unload_starts_empty_and_reloads_on_request() {
    let (app, w) = app(Some(Duration::from_millis(300))).await;
    assert_eq!(w.health.get().as_str(), "idle");
    let (s, b) = ask(&app, "tiny-hybrid").await;
    assert_eq!((s, b["model"].as_str()), (200, Some("tiny-hybrid")), "{b}");
    assert_eq!(w.health.get().as_str(), "healthy");
    tokio::time::sleep(Duration::from_millis(900)).await;
    assert_eq!(w.health.get().as_str(), "idle", "model not unloaded");
    let (s, b) = ask(&app, "tiny-moe").await;
    assert_eq!((s, b["model"].as_str()), (200, Some("tiny-moe")), "{b}");
}
