//! §6.5 P5 prediction sink (issue #51): drives the engine-registered
//! `predict` + `routing` trampolines over the tiny-MoE pool on the CPU
//! backend. The fixture has one expert layer, so its graph builds no
//! prediction node; the C delivery path is exercised on the real model
//! (prediction recall > 0 in `golem run --prefetch ...`).
mod common;

use std::time::{Duration, Instant};

use common::{expert_tensor as xt, tiny_moe_path};
use golem_engine::config::{EngineConfig, PrefetchMode};
use golem_engine::p3_glue::{self, SessionHooks};

fn hooks(mode: PrefetchMode) -> SessionHooks {
    p3_glue::enable(&EngineConfig {
        model_path: tiny_moe_path(),
        ram_experts_gib: 1,
        device_cache_experts: 4,
        prefetch: mode,
        ..EngineConfig::default()
    })
    .expect("pooled enable")
}

fn predict(h: &SessionHooks, ids: &[i32]) {
    p3_glue::predict(
        h.payload_ptr(),
        xt::BLK - 1,
        xt::BLK,
        ids.as_ptr(),
        ids.len() as i32,
    );
}

fn route(h: &SessionHooks, ids: &[i32]) {
    p3_glue::routing(
        h.payload_ptr(),
        xt::BLK,
        std::ptr::null(),
        ids.as_ptr(),
        ids.len() as i64,
        xt::N_EXPERTS,
    );
}

#[test]
fn protect_books_prediction_recall_on_decode_steps_only() {
    let h = hooks(PrefetchMode::Protect);
    predict(&h, &[1, 2, 3]);
    route(&h, &[2, 5]);
    let s = h.prefetch_stats();
    assert_eq!((s.predicted_hits, s.predicted_routed), (1, 2));
    assert_eq!(s.prefetch_pageins, 0, "protect reads nothing");
    // Prefill (scoring off, §6.2): predictions are ignored.
    h.set_scoring(false);
    predict(&h, &[5]);
    route(&h, &[5]);
    h.set_scoring(true);
    assert_eq!(h.prefetch_stats().predicted_routed, 2);
}

#[test]
fn read_pages_predicted_experts_in_and_counts_their_demand_use() {
    let h = hooks(PrefetchMode::Read);
    predict(&h, &[6, 7]);
    let deadline = Instant::now() + Duration::from_secs(5);
    while h.prefetch_stats().prefetch_pageins < 2 {
        assert!(
            Instant::now() < deadline,
            "prefetch thread never paged the predictions in"
        );
        std::thread::sleep(Duration::from_millis(2));
    }
    let s = h.prefetch_stats();
    assert_eq!(
        (s.mode, s.prefetch_hits, s.prefetch_wasted_bytes),
        (PrefetchMode::Read, 0, 0)
    );
}

#[test]
fn off_ignores_predictions() {
    let h = hooks(PrefetchMode::Off);
    predict(&h, &[1]);
    route(&h, &[1]);
    let s = h.prefetch_stats();
    assert_eq!((s.predicted_routed, s.prefetch_pageins), (0, 0));
}

#[test]
fn routing_trace_logs_predictions_as_pr_rows_verbatim() {
    let dir = std::env::temp_dir().join(format!("golem-pr-trace-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let h = p3_glue::enable(&EngineConfig {
        model_path: tiny_moe_path(),
        ram_experts_gib: 1,
        device_cache_experts: 4,
        prefetch: PrefetchMode::Protect,
        routing_trace: Some(dir.join("t.tsv")),
        ..EngineConfig::default()
    })
    .expect("pooled enable with trace");
    predict(&h, &[3, 1, 2]);
    route(&h, &[2, 5]);
    let mut out = Vec::new();
    h.write_routing_trace(&mut out).unwrap();
    let text = String::from_utf8(out).unwrap();
    let rows: Vec<&str> = text.lines().skip(1).collect();
    assert_eq!(
        rows,
        [
            format!("0\tpr\t{}\t3,1,2", xt::BLK),
            format!("0\tdc\t{}\t2,5", xt::BLK)
        ]
    );
    let _ = std::fs::remove_dir_all(&dir);
}
