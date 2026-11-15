//! Streaming/placement smoke test for deepseek2 (GLM-4.7-Flash arch) on the
//! HIP substrate through the engine-registered hooks: lazy expert planes, a
//! device budget below one layer's expert planes (K + 1 = 3 slab slots of an
//! 8-expert layer), lookahead prediction nodes firing, and every expert
//! `MUL_MAT_ID` placed on the GPU. It does not observe tiers. Missing
//! preconditions fail.
//!
//! Real target: GLM-4.7-Flash, 46 MoE layers x 64 experts x 3 planes, ~17.5 GB
//! file vs 24 GiB VRAM; the fixture is 2 MoE layers x 8 experts x 3 planes.
#![cfg(feature = "hip-tests")]
mod common;

use common::offload::{offload_config, require_process_per_test};
use golem_engine::config::{EngineConfig, PrefetchMode};
use golem_engine::session::{ChatTurn, Session, expert_placement};
use xtask::tiny_deepseek2::dims::{N_EXPERT, N_EXPERT_USED};

const K: u32 = N_EXPERT_USED;

fn config() -> EngineConfig {
    EngineConfig {
        model_path: xtask::tiny_deepseek2::cached_fixture(),
        ram_experts_gib: 1,
        device_cache_experts: K,
        async_h2d: true,
        prefetch: PrefetchMode::Read,
        n_predict: 24,
        ignore_eos: true,
        ..offload_config()
    }
}

#[test]
fn deepseek2_streams_experts_with_lookahead_on_gpu() {
    require_process_per_test();
    let cfg = config();
    let before = expert_placement();
    let mut s = Session::new(&cfg).expect("pooled deepseek2 session");
    let mut out = String::new();
    s.chat_turn(
        &[ChatTurn {
            role: "user",
            content: "the quick brown fox",
        }],
        &mut |p| out.push_str(p),
    )
    .expect("decode");
    assert!(!out.trim().is_empty(), "decode produced no output");
    let after = expert_placement();
    assert!(
        after.on_cpu == before.on_cpu && after.on_gpu > before.on_gpu,
        "every expert MUL_MAT_ID must be GPU-assigned: {before:?} -> {after:?}"
    );

    let facts = golem_plan::ModelFacts::from_gguf(&cfg.model_path).expect("fixture facts");
    assert_eq!(facts.n_expert, N_EXPERT);
    let (slab_res, _) = s.device_cache_resident_bytes().expect("pool on");
    let per_slot = facts.slab_bytes_per_expert();
    let per_layer = |l| u64::from(N_EXPERT) * xtask::tiny_deepseek2::layer_expert_bytes(l);
    let one_layer = per_layer(1).min(per_layer(2));
    assert_eq!(
        per_slot,
        xtask::tiny_deepseek2::layer_expert_bytes(1) + xtask::tiny_deepseek2::layer_expert_bytes(2)
    );
    assert!(
        slab_res > 0
            && slab_res <= (u64::from(K) + 1) * per_slot + (1 << 20)
            && slab_res < one_layer,
        "device budget must stay below one layer's planes: slab {slab_res} B, \
         {per_slot} B per slot, {one_layer} B per layer"
    );

    let p = s.prefetch_stats().expect("prefetch on");
    // One edge (blk1 -> blk2) books K routed ids per single-token decode step;
    // n_predict tokens take at least n_predict - 1 decode steps after prefill.
    let min_routed = u64::from(K) * u64::from(cfg.n_predict - 1);
    assert!(
        p.predicted_routed >= min_routed,
        "lookahead prediction nodes must fire every decode step (>= {min_routed}): {p}"
    );
}
