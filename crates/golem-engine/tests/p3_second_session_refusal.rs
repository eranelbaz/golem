//! The second-session refusal must run BEFORE `load()` —
//! a different model's load decodes its self-check through the stale hook
//! set (tensor table keyed by recycled host addresses, host bytes diverging
//! from the old pool) and dies in `p3_fatal` instead of returning `Err`.
//! Runs only where a GPU split exists (feature `hip-tests`, HIP CI job); CPU
//! builds never enter the split-scheduler expert-copy path, so the pool would
//! idle.
#![cfg(feature = "hip-tests")]
mod common;

use std::path::PathBuf;

use golem_engine::config::EngineConfig;
use golem_engine::error::EngineError;
use golem_engine::session::Session;

/// Second model: same real tiny-moe shape, different weights (another seed) —
/// exactly the "second model in one process" shape the refusal exists for.
/// Cached in `.test-cache/` next to the primary fixture (never in git).
fn second_fixture() -> PathBuf {
    const SECOND_SEED: u64 = 0x6d_6f_6c_65_67; // "golem" reversed
    let cache = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(".test-cache");
    std::fs::create_dir_all(&cache).unwrap();
    let p = cache.join("tiny-moe-320-seed2.gguf");
    if !p.exists() {
        let tmp = cache.join(format!("tiny-moe-320-seed2-{}.tmp", std::process::id()));
        xtask::tiny_moe::generate(
            &mut std::io::BufWriter::new(std::fs::File::create(&tmp).unwrap()),
            SECOND_SEED,
        )
        .unwrap();
        std::fs::rename(&tmp, &p).unwrap();
    }
    p
}

#[test]
fn second_session_with_a_different_model_returns_err_before_load() {
    // Pool ON: registers the process-global hooks (leaked, never unregistered).
    let mut cfg = common::offload::offload_config();
    cfg.ram_experts_gib = 1;
    // Precondition fails loudly — no skip (AGENTS.md Evidence; memlock is
    // unlimited on the HIP CI machine).
    let _owner = Session::new(&cfg)
        .expect("pooled session must load (RLIMIT_MEMLOCK must allow the 1 GiB pool)");

    // A DIFFERENT fixture, pool ON: refused with Err(PoolInit) before the
    // model loads (a live session already exists).
    let other = EngineConfig {
        model_path: second_fixture(),
        ram_experts_gib: 1,
        ..EngineConfig::default()
    };
    match Session::new(&other) {
        Err(EngineError::PoolInit(msg)) => {
            assert!(
                msg.contains("session already exists"),
                "refusal must name the live session: {msg}"
            );
        }
        Err(e) => panic!("second session must be refused as PoolInit, got: {e}"),
        Ok(_) => panic!("second session in a hook-owning process must be refused"),
    }

    // Pool OFF with a different model: `load` itself refuses once the hooks
    // are registered — a load would decode through the stale hooks and abort
    // the process (p3_fatal) instead of returning.
    let other_off = EngineConfig {
        model_path: second_fixture(),
        ..EngineConfig::default()
    };
    match Session::new(&other_off) {
        Err(EngineError::PoolInit(msg)) => {
            assert!(
                msg.contains("expert hooks"),
                "refusal must name the hook ownership: {msg}"
            );
        }
        Err(e) => panic!("second session must be refused as PoolInit, got: {e}"),
        Ok(_) => panic!("second session in a hook-owning process must be refused"),
    }

    // Same-shape sanity: the second fixture really loads (in a fresh process
    // the hooks would be unset), so the refusal is about process state, not a
    // broken fixture. Probed via the GGUF metadata walk only — loading here is
    // impossible by construction (hooks are registered).
    let metadata = golem_engine::gguf::read_metadata(&second_fixture()).unwrap();
    assert_eq!(
        metadata.kv_u64("glm4moe.expert_count"),
        Some(8),
        "second fixture must be real tiny-moe shape"
    );
}
