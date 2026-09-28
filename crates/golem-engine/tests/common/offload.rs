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
/// - `GGML_OP_OFFLOAD_MIN_BATCH=1` (read once at device init, hence set here)
///   lowers the op-offload threshold from the default 32: MUL_MAT_ID's batch
///   size is `ne[2]` (tokens), 1 during decode, so without this the op stays
///   on the CPU and the split copy path never runs at all.
///
/// Must be called before the first `Session::new` in the process (backend
/// init reads the env).
pub fn offload_config() -> EngineConfig {
    // SAFETY: single-threaded test setup; the value only feeds the GPU
    // backend's device init below.
    unsafe { std::env::set_var("GGML_OP_OFFLOAD_MIN_BATCH", "1") };
    EngineConfig {
        model_path: tiny_moe_path(),
        n_gpu_layers: 99,
        cpu_moe_layers: Some(99),
        ..Default::default()
    }
}
