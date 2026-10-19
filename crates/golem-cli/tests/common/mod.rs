//! Resolves the shared tiny-moe fixture via `xtask::tiny_moe::cached_fixture`,
//! and spawns `golem` against a private registry.
#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

pub fn tiny_moe_path() -> PathBuf {
    xtask::tiny_moe::cached_fixture()
}

/// Fresh dir for `XDG_RUNTIME_DIR`, `XDG_STATE_HOME`, `XDG_CONFIG_HOME` and
/// `GOLEM_DATA_DIR`, so no test reads or writes the user's real registry or
/// `golem.toml`.
pub fn tempdir_env() -> PathBuf {
    static N: AtomicU32 = AtomicU32::new(0);
    let d = std::env::temp_dir().join(format!(
        "golem-cli-{}-{}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = std::fs::remove_dir_all(&d);
    for sub in ["run", "state", "data"] {
        std::fs::create_dir_all(d.join(sub)).unwrap();
    }
    d
}

/// `golem` with `rt`'s private XDG/data dirs.
pub fn golem_cmd(rt: &std::path::Path) -> std::process::Command {
    let mut c = std::process::Command::new(env!("CARGO_BIN_EXE_golem"));
    c.env("XDG_RUNTIME_DIR", rt.join("run"))
        .env("XDG_STATE_HOME", rt.join("state"))
        .env("XDG_CONFIG_HOME", rt.join("config"))
        .env_remove("GOLEM_CONFIG_DIR")
        .env_remove("GOLEM_MODEL")
        .env("GOLEM_DATA_DIR", rt.join("data"));
    c
}

/// `golem` in a fresh private environment.
pub fn golem_isolated() -> std::process::Command {
    golem_cmd(&tempdir_env())
}
