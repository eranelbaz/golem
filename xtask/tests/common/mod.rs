//! Shared helpers for xtask integration tests.

use std::path::PathBuf;

pub struct Cleanup(pub PathBuf);

impl Drop for Cleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

pub fn temp_dir(tag: &str) -> Cleanup {
    let dir = std::env::temp_dir().join(format!("golem-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    Cleanup(dir)
}

/// llama.cpp commit from engine/llama-cpp.lock — the single pin source of truth.
pub fn pinned_commit() -> String {
    let lock = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../engine/llama-cpp.lock");
    let text =
        std::fs::read_to_string(&lock).unwrap_or_else(|e| panic!("read {}: {e}", lock.display()));
    text.lines()
        .find_map(|l| l.trim_start().strip_prefix("commit = "))
        .map(str::trim)
        .unwrap_or_else(|| panic!("no `commit = ` line in {}", lock.display()))
        .to_owned()
}
