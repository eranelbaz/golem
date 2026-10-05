//! Design I2: the engine (not the test) forces every expert MUL_MAT_ID onto
//! the GPU. This suite deliberately does NOT set `GGML_OP_OFFLOAD_MIN_BATCH`:
//! with host-resident experts at decode batch 1, the split-copy path (and so
//! the P3 hook) only fires when the op ran on the GPU. If it ran on the CPU,
//! the P3 counters stay at zero and this test fails — the silent-fallback
//! regime upstream defaults to.
//!
//! The P1 placement counters (`expert_placement`) additionally prove no
//! expert node was CPU-assigned; both tests need their own process
//! (`common::offload::require_process_per_test`).

#![cfg(feature = "hip-tests")]
mod common;

use common::offload::offload_config;
use golem_engine::error::EngineError;
use golem_engine::session::{ChatTurn, Session, expert_placement};

#[test]
fn expert_matmul_runs_on_gpu_without_env_override() {
    common::offload::require_process_per_test();
    let mut cfg = offload_config();
    cfg.ram_experts_gib = 1; // pool on ⇒ engine registers all three hooks
    let mut session = Session::new(&cfg).expect("pooled offload session");
    let before = expert_placement();
    let mut out = String::new();
    session
        .chat_turn(
            &[ChatTurn {
                role: "user",
                content: "Hello",
            }],
            &mut |piece| out.push_str(piece),
        )
        .expect("decode");
    assert!(!out.is_empty());

    let stats = session
        .p3_stats()
        .expect("pooled split ⇒ P3 counters present");
    let demand = stats
        .demand_slow_pageins
        .load(std::sync::atomic::Ordering::Relaxed)
        + stats.fast_hits.load(std::sync::atomic::Ordering::Relaxed);
    assert!(
        demand > 0,
        "no expert copies served ⇒ MUL_MAT_ID stayed on the CPU \
         (silent I2 violation): {stats:?}"
    );

    let after = expert_placement();
    assert!(
        after.on_cpu == before.on_cpu && after.on_gpu > before.on_gpu,
        "every expert MUL_MAT_ID must be GPU-assigned: {before:?} -> {after:?}"
    );
}

/// The offload threshold is read once at device init: a flagless session
/// first leaves it at upstream's 32, so the later pooled session's prefill
/// chunks (< 32 tokens) put every expert MUL_MAT_ID on the CPU. The engine
/// must refuse at the first pooled prefill.
#[test]
fn expert_matmul_on_cpu_is_refused() {
    common::offload::require_process_per_test();
    let flagless = golem_engine::config::EngineConfig {
        model_path: common::tiny_moe_path(),
        n_gpu_layers: 99,
        ..Default::default()
    };
    drop(Session::new(&flagless).expect("flagless session"));

    let mut cfg = offload_config();
    cfg.ram_experts_gib = 1;
    cfg.batch_tokens = 8; // every prefill chunk < upstream's batch-32 threshold
    let mut session = Session::new(&cfg).expect("pooled offload session");
    let err = session
        .chat_turn(
            &[ChatTurn {
                role: "user",
                content: "Hello",
            }],
            &mut |_| {},
        )
        .expect_err("CPU expert math must be refused");
    assert!(
        matches!(&err, EngineError::BackendUnavailable(m) if m.contains("on the CPU")),
        "{err}"
    );
}
