//! Review C-4: the engine (not the test) forces every expert MUL_MAT_ID onto
//! the GPU. This suite deliberately does NOT set `GGML_OP_OFFLOAD_MIN_BATCH`:
//! with host-resident experts at decode batch 1, the split-copy path (and so
//! the P3 hook) only fires when the op ran on the GPU. If it ran on the CPU,
//! the P3 counters stay at zero and this test fails — the silent-fallback
//! regime upstream defaults to.

#![cfg(feature = "hip-tests")]
mod common;

use common::offload::offload_config;
use golem_engine::session::{ChatTurn, Session};

#[test]
fn expert_matmul_runs_on_gpu_without_env_override() {
    let mut cfg = offload_config();
    cfg.ram_experts_gib = 1; // pool on ⇒ engine registers all three hooks
    let mut session = Session::new(&cfg).expect("pooled offload session");
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
         (silent I2 violation, review C-4): {stats:?}"
    );
}
