//! Offload (GPU-split) config for P3/P2 tests. Only meaningful where a GPU
//! backend exists; callers gate on the `hip-tests` feature.
#![allow(dead_code, unsafe_code)] // compiled into every suite via `mod common`; used by hip-tests only
use super::tiny_moe_path;
use golem_engine::config::EngineConfig;

/// Offload config that actually engages the split-scheduler expert-copy path:
/// - `n_gpu_layers: 99` + `cpu_moe_layers: Some(99)` pins every expert plane
///   to a host (CPU) weights buffer while attention runs on the GPU. The
///   split scheduler then copies the routed experts per step — the P1/P2
///   hook call-site (`ggml_backend_buffer_is_host` branch). With plain full
///   offload the experts fit on the GPU and no copies happen.
/// - The engine itself forces the op-offload threshold to 1 (design I2):
///   tests must NOT set `GGML_OP_OFFLOAD_MIN_BATCH`; a HIP test that passes
///   without the env proves the engine sets it.
pub fn offload_config() -> EngineConfig {
    EngineConfig {
        model_path: tiny_moe_path(),
        n_gpu_layers: 99,
        cpu_moe_layers: Some(99),
        ..Default::default()
    }
}

/// Precondition for tests that depend on process-global substrate state
/// (op-offload threshold read once at device init, P1 counters): each test
/// must own its process. nextest's default mode does; plain `cargo test`
/// does not, and the test then fails here instead of passing by luck.
pub fn require_process_per_test() {
    assert_eq!(
        std::env::var("NEXTEST_EXECUTION_MODE").as_deref(),
        Ok("process-per-test"),
        "run under cargo nextest: this test needs its own process"
    );
}
