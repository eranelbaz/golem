//! `ModelFacts::from_gguf` on the tiny fixtures, against sums recomputed from
//! the raw tensor table and the fixture's known dims.

use std::fs::File;
use std::path::Path;

use golem_gguf::{ggml_row_bytes, kv_int, read_kvs, shard_path, tensor_entries};
use golem_plan::ModelFacts;

fn table_bytes(path: &Path) -> u64 {
    tensor_entries(&mut File::open(path).unwrap())
        .unwrap()
        .iter()
        .map(|t| {
            ggml_row_bytes(t.ggml_type, t.dims[0]).unwrap()
                * t.dims[1..].iter().map(|&d| d.max(1)).product::<u64>()
        })
        .sum()
}

#[test]
fn tiny_moe_facts() {
    use xtask::tiny_moe::dims;
    let path = xtask::tiny_moe::cached_fixture();
    let m = ModelFacts::from_gguf(&path).unwrap();
    assert_eq!(m.n_expert, dims::N_EXPERT);
    assert_eq!(m.expert_layers, dims::N_LAYER - dims::N_LAYER_DENSE_LEAD);
    assert_eq!(m.t_ne + m.expert_bytes_total, table_bytes(&path));
    assert!(m.shexp <= m.t_ne);
    // glm4moe: every layer is full attention, f16 K and V.
    assert_eq!(m.sparse_attn_layers, dims::N_LAYER);
    assert_eq!(m.rs_bytes, 0);
    let per_layer = 2 * u64::from(dims::HEAD_DIM * dims::N_HEAD_KV) * 2;
    assert_eq!(m.kv_per_token, u64::from(m.sparse_attn_layers) * per_layer);
}

/// Real target: 11 MLA layers × (512 K + 3 × 128 indexer) × 2 B = 19,712 B/token;
/// 34 KDA layers × (3·3·64·128 + 128·128·64) × 4 B = 152,633,344 B = 145.56 MiB.
#[test]
fn tiny_hybrid_facts() {
    use xtask::tiny_hybrid::dims;
    let first = xtask::tiny_hybrid::cached_fixture();
    let m = ModelFacts::from_gguf(&first).unwrap();
    let count = kv_int(
        &read_kvs(&mut File::open(&first).unwrap()).unwrap(),
        "split.count",
    )
    .unwrap();
    let shards: u64 = (0..count)
        .map(|i| table_bytes(&shard_path(&first, i, count).unwrap()))
        .sum();
    assert_eq!(m.t_ne + m.expert_bytes_total, shards);
    assert_eq!(m.n_expert, dims::N_EXPERT);
    assert_eq!(m.expert_layers, dims::N_LAYER - dims::N_LAYER_DENSE_LEAD);
    assert!(m.shexp > 0 && m.shexp <= m.t_ne);

    let attn = dims::HEAD_COUNT_KV.iter().filter(|&&h| h > 0).count() as u32;
    assert_eq!(m.sparse_attn_layers, attn);
    // MLA is K-only; the pooling indexer caches 3 heads per cell; f16.
    let per_layer = u64::from(dims::KEY_LENGTH + 3 * dims::INDEXER_KEY_LENGTH) * 2;
    assert_eq!(m.kv_per_token, u64::from(attn) * per_layer);
    let d_inner = u64::from(dims::N_HEAD * dims::KDA_HEAD_DIM);
    let conv = 3 * u64::from(dims::SSM_CONV - 1) * d_inner;
    let state = u64::from(dims::KDA_HEAD_DIM) * d_inner;
    let recr = u64::from(dims::N_LAYER - attn);
    assert_eq!(m.rs_bytes, recr * (conv + state) * 4);
}
