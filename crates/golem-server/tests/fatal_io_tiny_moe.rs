//! §5.2 fatal I/O: every replica of a shard unreadable → health `fatal-io`,
//! the in-flight request fails 503 `golem_storage_unavailable`, new requests
//! likewise; the process keeps serving `/health`. Pooled store-backed
//! tiny-moe over a hand-built manifest; replicas fault-injected through the
//! m3.3 `mark_degraded` seam.
use std::path::PathBuf;

use axum::body::Body;
use golem_server::test_support::{TestHooks, router_with_hooks};
use http::Request;
use tower::ServiceExt;

/// Copy of the tiny-moe fixture, zero-padded to a page multiple (the
/// store's `O_DIRECT` reads are page-aligned), plus its manifest (one
/// shard, replica = the copy's dir). Returns (model, manifest).
fn fixture() -> (PathBuf, PathBuf) {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "golem-fatal-io-{}-{}",
        std::process::id(),
        N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("tiny-moe-320.gguf");
    let mut bytes = std::fs::read(golem_server::test_engine_config().model_path).unwrap();
    bytes.resize(bytes.len().next_multiple_of(4096), 0);
    std::fs::write(&model, &bytes).unwrap();
    let mut f = std::fs::File::open(&model).unwrap();
    let entries = golem_gguf::tensor_entries(&mut f).unwrap();
    let index: Vec<serde_json::Value> = entries
        .iter()
        .filter(|e| e.expert_kind().is_some())
        .map(|e| {
            let stride = golem_gguf::ggml_row_bytes(e.ggml_type, e.dims[0]).unwrap() * e.dims[1];
            serde_json::json!({
                "tensor": e.name,
                "layer": golem_gguf::blk_of(&e.name).unwrap(),
                "experts": e.dims[2],
                "bytes_per_expert": stride,
                "offsets_bytes": (0..e.dims[2]).map(|i| i * stride).collect::<Vec<_>>(),
            })
        })
        .collect();
    let m = serde_json::json!({
        "name": "tiny-moe",
        "shards": [{
            "file": model.file_name().unwrap().to_string_lossy(),
            "bytes": std::fs::metadata(&model).unwrap().len(),
            "sha256": "0".repeat(64),
            "replicas": [model.parent().unwrap()],
        }],
        "expert_index": index,
    });
    let p = dir.join("manifest.json");
    std::fs::write(&p, m.to_string()).unwrap();
    (model, p)
}

async fn body_json(r: axum::response::Response) -> serde_json::Value {
    serde_json::from_slice(&axum::body::to_bytes(r.into_body(), 1 << 20).await.unwrap()).unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn all_replicas_unreadable_is_fatal_io_503() {
    fatal_io_503(TestHooks::fail_storage_after_admit()).await;
}

/// The replicas die inside the page-in itself (ABI v13 ensure failure):
/// the process survives and the request still gets the 503. Needs a GPU
/// split: the page-in runs on the miss-list path CPU builds never reach.
#[cfg(feature = "hip-tests")]
#[tokio::test(flavor = "multi_thread")]
async fn replicas_lost_during_read_is_fatal_io_503() {
    fatal_io_503(TestHooks::fail_storage_during_read()).await;
}

async fn fatal_io_503(hooks: TestHooks) {
    let (model, manifest) = fixture();
    let cfg = golem_engine::config::EngineConfig {
        model_path: model,
        ram_experts_gib: 1,
        device_cache_experts: 4,
        manifest_path: Some(manifest),
        ..golem_server::test_engine_config()
    };
    let app = router_with_hooks(cfg, hooks).await;
    let chat = || {
        Request::post("/v1/chat/completions")
            .header("content-type", "application/json")
            .body(Body::from(
                serde_json::json!({"model":"tiny-moe","messages":[{"role":"user","content":"hi"}],
                    "max_tokens":32,"temperature":0})
                .to_string(),
            ))
            .unwrap()
    };
    let r = app.clone().oneshot(chat()).await.unwrap();
    assert_eq!(r.status(), 503);
    assert_eq!(
        body_json(r).await["error"]["code"],
        "golem_storage_unavailable"
    );
    let r = app.clone().oneshot(chat()).await.unwrap();
    assert_eq!(r.status(), 503);
    assert_eq!(
        body_json(r).await["error"]["code"],
        "golem_storage_unavailable"
    );
    let r = app
        .oneshot(Request::get("/health").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(body_json(r).await["state"], "fatal-io");
}
