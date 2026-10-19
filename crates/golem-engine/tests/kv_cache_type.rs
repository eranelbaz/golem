//! `cache_type_k/v` reach llama's context: mixed types load on a non-MLA
//! model, a quantized cache loads on the MLA hybrid, and the snapshot floor
//! follows the priced type.
mod common;

use golem_engine::config::EngineConfig;
use golem_engine::session::{ChatTurn, Session};
use golem_plan::KvCacheType;

fn greedy(cfg: &EngineConfig) -> String {
    let mut s = Session::new(cfg).unwrap();
    let mut out = String::new();
    s.chat_turn(
        &[ChatTurn {
            role: "user",
            content: "the quick brown fox",
        }],
        &mut |p| out.push_str(p),
    )
    .unwrap();
    out
}

#[test]
fn mixed_types_load_on_tiny_moe() {
    let cfg = EngineConfig {
        cache_type_k: KvCacheType::F32,
        cache_type_v: KvCacheType::F16,
        n_predict: 8,
        ..common::tiny_moe_config()
    };
    assert!(!greedy(&cfg).is_empty());
}

#[test]
fn q8_0_loads_on_the_mla_hybrid_and_prices_its_snapshot_floor() {
    let base = common::tiny_hybrid_config();
    let q8 = EngineConfig {
        cache_type_k: KvCacheType::Q8_0,
        cache_type_v: KvCacheType::Q8_0,
        n_predict: 8,
        ..base.clone()
    };
    assert!(!greedy(&q8).is_empty());
    let facts = |c: &EngineConfig| {
        golem_plan::ModelFacts::from_gguf_kv(&c.model_path, c.cache_type_k, c.cache_type_v).unwrap()
    };
    let (f16, q8f) = (facts(&base), facts(&q8));
    // glm5next: K at 34 B per 32 elements, the indexer stays f16.
    use xtask::tiny_hybrid::dims;
    let attn = dims::HEAD_COUNT_KV.iter().filter(|&&h| h > 0).count() as u64;
    let k = u64::from(dims::KEY_LENGTH);
    let idx = 3 * u64::from(dims::INDEXER_KEY_LENGTH);
    assert_eq!(q8f.kv_per_token, attn * (k / 32 * 34 + idx * 2));
    assert!(q8f.kv_per_token < f16.kv_per_token);
    let floor = |f| golem_plan::snapshot_arena(f, 1, 4096).0;
    assert_eq!(
        floor(&f16) - floor(&q8f),
        (f16.kv_per_token - q8f.kv_per_token) * 4096
    );
}

#[test]
fn mixed_types_refused_on_the_mla_hybrid() {
    let cfg = EngineConfig {
        cache_type_k: KvCacheType::Q8_0,
        cache_type_v: KvCacheType::F16,
        ..common::tiny_hybrid_config()
    };
    assert!(Session::new(&cfg).is_err());
}
