//! §8.1 `GET /v1/golem/experts` over a pooled tiny-moe worker: a decode
//! lists each layer's routed ids once with a tier, a prefill-only request
//! reports `step_kind: "prefill"` (HIP: a CPU-only substrate publishes no
//! rows), and an idle unload leaves no stale model.
use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use golem_engine::config::EngineConfig;
use golem_server::test_support::{test_engine_config, test_server_config};
use golem_server::{AppState, ServerConfig, TurnSinks, router, spawn_worker_models};
use http::Request;
use serde_json::{Value, json};
use tower::ServiceExt;

async fn get(app: &axum::Router, path: &str) -> (u16, Value) {
    let r = app
        .clone()
        .oneshot(Request::get(path).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let s = r.status().as_u16();
    let b = axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap();
    (s, serde_json::from_slice(&b).unwrap())
}

async fn ask(app: &axum::Router, max_tokens: u32) -> Value {
    let body = json!({"model": "tiny-moe", "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": max_tokens, "temperature": 0});
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
    let s = r.status();
    let b = axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap();
    let v: Value = serde_json::from_slice(&b).unwrap();
    assert_eq!(s, 200, "{v}");
    v
}

#[tokio::test(flavor = "multi_thread")]
async fn experts_lists_decode_tiers_then_prefill_then_idle() {
    let cfg = EngineConfig {
        ram_experts_gib: 1,
        device_cache_experts: 4,
        ..test_engine_config()
    };
    let w = tokio::task::spawn_blocking(move || {
        spawn_worker_models(
            vec![("tiny-moe".into(), cfg)],
            Some(Duration::from_millis(500)),
            true,
            false,
            TurnSinks::default(),
        )
    })
    .await
    .unwrap()
    .expect("worker starts");
    let app = router(AppState {
        cfg: Arc::new(ServerConfig {
            models: vec!["tiny-moe".into()],
            ..test_server_config()
        }),
        jobs: w.jobs.clone(),
        health: w.health.clone(),
    });
    assert_eq!(
        get(&app, "/v1/golem/experts").await,
        (200, json!({"state": "idle"}))
    );

    let v = ask(&app, 4).await;
    assert!(
        v["usage"]["completion_tokens"].as_u64().unwrap() >= 2,
        "precondition: at least one generation decode step ran: {v}"
    );
    let (s, e) = get(&app, "/v1/golem/experts").await;
    assert_eq!(s, 200, "{e}");
    // A CPU-only substrate never copies expert ids to an offload backend, so
    // the routing readback (the only row writer) never fires.
    #[cfg(not(golem_backend_hip))]
    assert_eq!(e, json!({"state": "healthy"}));
    #[cfg(golem_backend_hip)]
    decode_rows_then_prefill(&app, &e).await;

    tokio::time::sleep(Duration::from_millis(1500)).await;
    assert_eq!(
        w.health.get().as_str(),
        "idle",
        "precondition: model unloaded"
    );
    assert_eq!(
        get(&app, "/v1/golem/experts").await,
        (200, json!({"state": "idle"})),
        "no stale model after unload"
    );
}

#[cfg(golem_backend_hip)]
async fn decode_rows_then_prefill(app: &axum::Router, e: &Value) {
    assert_eq!(e["model"], "tiny-moe", "{e}");
    assert_eq!(e["step_kind"], "decode", "{e}");
    assert!(e["step"].as_u64().is_some(), "{e}");
    let (n_layers, n_experts, top_k) = (
        e["n_layers"].as_u64().unwrap(),
        e["n_experts"].as_u64().unwrap(),
        e["top_k"].as_u64().unwrap(),
    );
    let layers = e["layers"].as_array().unwrap();
    assert_eq!(layers.len() as u64, n_layers);
    for (i, l) in layers.iter().enumerate() {
        assert_eq!(l["layer"], i as u64);
        assert!(l.get("open").is_none(), "row {i} closed: {l}");
        assert_eq!(l["step"], e["step"], "every layer ran this step: {l}");
        let routed = l["routed"].as_array().unwrap();
        assert!(
            !routed.is_empty() && routed.len() as u64 <= top_k,
            "decode routes 1..=top_k distinct ids: {l}"
        );
        let mut ids: Vec<u64> = routed.iter().map(|u| u["id"].as_u64().unwrap()).collect();
        assert!(ids.iter().all(|&id| id < n_experts));
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), routed.len(), "each id once: {l}");
        for u in routed {
            assert!(
                ["vram", "ram", "disk", "cpu"].contains(&u["tier"].as_str().unwrap()),
                "{u}"
            );
        }
        assert!(l["vram_resident"].as_u64().unwrap() <= 4, "≤ K slots: {l}");
        assert!(l["ram_resident"].as_u64().is_some());
        assert_eq!(l["truncated"], false);
    }

    // max_tokens 1: the token is sampled from the prefill logits, so no
    // generation decode runs and the rows stay as the last decode left them.
    let v = ask(app, 1).await;
    assert_eq!(v["usage"]["completion_tokens"], 1);
    let (_, p) = get(app, "/v1/golem/experts").await;
    assert_eq!(p["step_kind"], "prefill", "{p}");
    assert_eq!(p["step"], e["step"]);
    assert_eq!(
        p["layers"], e["layers"],
        "prefill leaves the rows untouched"
    );
}
