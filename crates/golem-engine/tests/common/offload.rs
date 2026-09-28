//! Offload (GPU-split) config for P3 tests. Only meaningful where a GPU
//! backend exists; callers gate on the `hip-tests` feature.
#![allow(dead_code)] // compiled into every suite via `mod common`; used by hip-tests only
use super::tiny_moe_path;
use golem_engine::config::EngineConfig;

pub fn offload_config() -> EngineConfig {
    EngineConfig {
        model_path: tiny_moe_path(),
        n_gpu_layers: 99,     // offload everything; expert tensors ride cpu_moe
        ..Default::default()  // cpu_moe_layers stays None
    }
}
