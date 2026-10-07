//! Per-model hook lifecycle (ABI v5, H-10 session level): sequential pooled
//! sessions in one process are LEGAL — each model carries its own `golem_ctx`,
//! and a dropped session's teardown (unregister-join → ctx free) leaves no
//! live hook state for the next one. Only a second CONCURRENT pool-enabled
//! session is refused, before any model load, with the slab VRAM arithmetic
//! (two K=10 slab sets ≈ 2 × 7.3 GB on a 24 GiB card). CPU-testable: a pooled
//! session with an explicit K loads and decodes its self-check without a GPU
//! split (the trampolines are registered either way); the full two-model
//! *bytes* proof is `h10_two_models.rs` (HIP-gated — the split-scheduler
//! expert-copy path only runs where a GPU split exists).
mod common;

use std::path::PathBuf;

use common::tiny_moe_path;
use golem_engine::config::EngineConfig;
use golem_engine::error::EngineError;
use golem_engine::session::Session;

/// Second fixture: same real tiny-moe shape, different weights (another
/// seed) — exactly the "two sequential models in one process" shape H-10
/// asks for. Cached in `.test-cache/` next to the primary fixture.
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

fn pooled(path: &std::path::Path) -> EngineConfig {
    EngineConfig {
        model_path: path.into(),
        ram_experts_gib: 1,
        device_cache_experts: 8, // tiny-moe: K=8 ≥ top_k=2
        ..EngineConfig::default()
    }
}

/// H-10 (session level): two sequential pooled sessions, different models,
/// one process. The second must load and decode — a stale hook set would
/// abort in `p3_fatal` instead of returning `Ok`.
#[test]
fn two_sequential_pooled_sessions_load_and_decode() {
    let a = Session::new(&pooled(&tiny_moe_path()))
        .expect("first pooled session must load (RLIMIT_MEMLOCK must allow the 1 GiB pool)");
    drop(a);
    let b = Session::new(&pooled(&second_fixture()))
        .expect("second sequential pooled session must load: hooks are per-model");
    drop(b);
}

/// Concurrent-only refusal: while a pooled session is LIVE, a second pooled
/// `Session::new` is refused BEFORE any model load (the model path here does
/// not exist, so a load attempt would surface a GGUF error instead) and the
/// message carries the slab VRAM arithmetic.
#[test]
fn concurrent_second_pooled_session_refused_with_vram_arithmetic() {
    let _live = Session::new(&pooled(&tiny_moe_path()))
        .expect("pooled session must load (RLIMIT_MEMLOCK must allow the 1 GiB pool)");
    let other = EngineConfig {
        model_path: "/nonexistent/never-loaded.gguf".into(),
        ram_experts_gib: 1,
        device_cache_experts: 8,
        ..EngineConfig::default()
    };
    match Session::new(&other) {
        Err(EngineError::PoolInit(msg)) => {
            assert!(
                msg.contains("7.3 GB") && msg.contains("24 GiB"),
                "refusal must carry the slab VRAM arithmetic: {msg}"
            );
        }
        Err(e) => panic!("second session must be refused as PoolInit, got: {e}"),
        Ok(_) => panic!("concurrent second pooled session must be refused"),
    }
}

/// Same-shape sanity: the second fixture really is tiny-moe shape, so the
/// sequential test above proves lifecycle, not a broken fixture.
#[test]
fn second_fixture_is_real_tiny_moe_shape() {
    let metadata = golem_engine::gguf::read_metadata(&second_fixture()).unwrap();
    assert_eq!(
        metadata.kv_u64("glm4moe.expert_count"),
        Some(8),
        "second fixture must be real tiny-moe shape"
    );
}
