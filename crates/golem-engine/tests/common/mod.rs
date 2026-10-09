//! Resolves the shared fixtures via `xtask::{tiny_moe, tiny_hybrid}::cached_fixture`.
#![allow(dead_code)]
use golem_engine::config::EngineConfig;

pub fn tiny_moe_path() -> std::path::PathBuf {
    xtask::tiny_moe::cached_fixture()
}

pub fn tiny_moe_config() -> EngineConfig {
    EngineConfig {
        model_path: tiny_moe_path(),
        ..Default::default()
    }
}

/// glm5next hybrid (KDA + sparse attention): carries recurrent state, unlike tiny-moe.
pub fn tiny_hybrid_config() -> EngineConfig {
    EngineConfig {
        model_path: xtask::tiny_hybrid::cached_fixture(),
        ..Default::default()
    }
}

pub mod expert_tensor;
pub mod offload;
