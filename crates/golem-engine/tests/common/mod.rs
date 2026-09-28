//! Resolves the shared tiny-moe fixture via `xtask::tiny_moe::cached_fixture`.
pub fn tiny_moe_path() -> std::path::PathBuf {
    xtask::tiny_moe::cached_fixture()
}

pub mod offload;
