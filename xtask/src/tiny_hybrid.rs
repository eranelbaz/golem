//! Synthetic 3-layer glm5next GGUF: two KDA (linear-attention, recurrent
//! state) layers and one DSA (sparse MLA) layer, MoE FFN on blk.1..=2, split
//! into 2 shards with a metadata-only first shard like the real model. Hparams
//! follow the vendored `test-llama-archs` glm5next fixture; n_embd = 256 so
//! the expert planes can be K-quants (Q4_K/Q5_K/Q6_K: 144/176/210 B rows).
//!
//! Not matched to GLM-5.3-Flash UD-Q4_K_XL (45 layers, 288 experts, 6 shards):
//! 8 experts × 36–53 KB planes keep generation and CPU decode fast.

use std::io::Write;
use std::path::PathBuf;

use crate::gguf::{self, GgufKv, GgufTensorMeta, GgufValue};
use crate::tiny_moe::{Xorshift, d, kv, repo_root, token_table};
use golem_gguf::{GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K};

pub mod dims {
    pub const N_VOCAB: u32 = 320;
    pub const N_EMBD: u32 = 256;
    /// KDA head count and MLA head count.
    pub const N_HEAD: u32 = 2;
    /// Per-layer `head_count_kv`: 0 marks a recurrent (KDA) layer.
    pub const HEAD_COUNT_KV: [i32; 3] = [0, 0, 1];
    pub const N_LAYER: u32 = 3;
    pub const N_LAYER_DENSE_LEAD: u32 = 1;
    pub const N_EXPERT: u32 = 8;
    pub const N_EXPERT_USED: u32 = 2;
    pub const N_EXPERT_SHARED: u32 = 1;
    pub const N_FF: u32 = 256;
    pub const N_FF_EXP: u32 = 256;
    pub const N_FF_SHEXP: u32 = N_FF_EXP * N_EXPERT_SHARED;
    pub const N_CTX: u32 = 256;
    pub const KDA_HEAD_DIM: u32 = 128;
    pub const SSM_CONV: u32 = 4;
    pub const Q_LORA_RANK: u32 = 64;
    pub const KV_LORA_RANK: u32 = 512;
    pub const KEY_LENGTH: u32 = 512;
    pub const VALUE_LENGTH: u32 = 512;
    pub const KEY_LENGTH_MLA: u32 = 192;
    pub const VALUE_LENGTH_MLA: u32 = 128;
    pub const INDEXER_HEADS: u32 = 64;
    pub const INDEXER_KEY_LENGTH: u32 = 128;
    pub const INDEXER_TOP_K: u32 = 8;
    pub const INDEXER_KPOOL: u32 = 4;
    pub const HC_MULT: u32 = 4;
}

pub const DEFAULT_SEED: u64 = crate::tiny_moe::DEFAULT_SEED;

const N_SHARDS: u16 = 2;
/// Expert plane types (gate, up, down), uniform across layers.
const EXPERT_TYPES: [u32; 3] = [GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K];

/// Generates the split fixture under `.test-cache/tiny-hybrid/` and returns
/// the first shard (the path llama.cpp loads). Safe to call concurrently.
pub fn cached_fixture() -> PathBuf {
    cached("tiny-hybrid", None)
}

/// [`cached_fixture`] with `swiglu_clamp_exp = 0.5`, so the routed-expert
/// SWIGLU_CLAMP path is exercised; a separate cache dir keeps the unclamped
/// fixture's outputs unchanged.
pub fn cached_fixture_clamped() -> PathBuf {
    cached("tiny-hybrid-clamped", Some(0.5))
}

fn cached(name: &str, clamp_exp: Option<f32>) -> PathBuf {
    let dir = repo_root().join(".test-cache").join(name);
    std::fs::create_dir_all(&dir).unwrap();
    let paths: Vec<PathBuf> = (1..=N_SHARDS)
        .map(|i| dir.join(format!("tiny-hybrid-{i:05}-of-{N_SHARDS:05}.gguf")))
        .collect();
    if !paths.iter().all(|p| p.exists()) {
        let tensors = tensors(DEFAULT_SEED);
        for (shard_no, path) in paths.iter().enumerate() {
            let shard: &[_] = if shard_no == 0 { &[] } else { &tensors };
            let tmp = path.with_extension(format!("gguf.{}.tmp", std::process::id()));
            let mut w = std::io::BufWriter::new(std::fs::File::create(&tmp).unwrap());
            gguf::write_gguf(
                &mut w,
                &kvs(shard_no as u16, tensors.len(), clamp_exp),
                shard,
            )
            .unwrap();
            w.flush().unwrap();
            // A concurrent generator may have won the rename; its bytes are identical.
            if let Err(e) = std::fs::rename(&tmp, path) {
                let _ = std::fs::remove_file(&tmp);
                assert!(path.exists(), "rename {}: {e}", path.display());
            }
        }
    }
    paths[0].clone()
}

fn kvs(shard_no: u16, n_tensors: usize, clamp_exp: Option<f32>) -> Vec<GgufKv> {
    let u = GgufValue::U32;
    let f = GgufValue::F32;
    let a = |k: &str| format!("glm5next.{k}");
    let toks = token_table();
    let mut out = vec![
        kv("general.architecture", GgufValue::Str("glm5next".into())),
        kv("general.name", GgufValue::Str("tiny-hybrid".into())),
        kv("general.file_type", u(0)),
        kv(&a("context_length"), u(dims::N_CTX)),
        kv(&a("embedding_length"), u(dims::N_EMBD)),
        kv(&a("block_count"), u(dims::N_LAYER)),
        kv(&a("feed_forward_length"), u(dims::N_FF)),
        kv(&a("expert_feed_forward_length"), u(dims::N_FF_EXP)),
        kv(&a("expert_shared_feed_forward_length"), u(dims::N_FF_SHEXP)),
        kv(&a("expert_count"), u(dims::N_EXPERT)),
        kv(&a("expert_used_count"), u(dims::N_EXPERT_USED)),
        kv(&a("expert_shared_count"), u(dims::N_EXPERT_SHARED)),
        kv(&a("leading_dense_block_count"), u(dims::N_LAYER_DENSE_LEAD)),
        kv(&a("expert_weights_scale"), f(1.0)),
        kv(&a("expert_weights_norm"), GgufValue::Bool(true)),
        kv(&a("expert_gating_func"), u(2)), // sigmoid
        kv(&a("attention.head_count"), u(dims::N_HEAD)),
        kv(
            &a("attention.head_count_kv"),
            GgufValue::ArrI32(dims::HEAD_COUNT_KV.to_vec()),
        ),
        kv(&a("attention.layer_norm_rms_epsilon"), f(1e-5)),
        kv(&a("attention.layer_norm_epsilon"), f(1e-6)),
        kv(&a("attention.q_lora_rank"), u(dims::Q_LORA_RANK)),
        kv(&a("attention.kv_lora_rank"), u(dims::KV_LORA_RANK)),
        kv(&a("attention.key_length"), u(dims::KEY_LENGTH)),
        kv(&a("attention.value_length"), u(dims::VALUE_LENGTH)),
        kv(&a("attention.key_length_mla"), u(dims::KEY_LENGTH_MLA)),
        kv(&a("attention.value_length_mla"), u(dims::VALUE_LENGTH_MLA)),
        kv(&a("rope.dimension_count"), u(0)), // nope-only MLA
        kv(&a("attention.indexer.head_count"), u(dims::INDEXER_HEADS)),
        kv(
            &a("attention.indexer.key_length"),
            u(dims::INDEXER_KEY_LENGTH),
        ),
        kv(&a("attention.indexer.top_k"), u(dims::INDEXER_TOP_K)),
        kv(&a("attention.indexer.kpool"), u(dims::INDEXER_KPOOL)),
        kv(&a("ssm.conv_kernel"), u(dims::SSM_CONV)),
        kv(&a("kda.head_dim"), u(dims::KDA_HEAD_DIM)),
        kv(&a("kda.gate_lower_bound"), f(-5.0)),
        kv(&a("hyper_connection.count"), u(dims::HC_MULT)),
        kv(&a("hyper_connection.sinkhorn_iterations"), u(2)),
        kv(&a("hyper_connection.epsilon"), f(1e-6)),
        kv("tokenizer.ggml.model", GgufValue::Str("llama".into())),
        kv(
            "tokenizer.ggml.tokens",
            GgufValue::ArrStr(toks.iter().map(|(t, _, _)| t.clone()).collect()),
        ),
        kv(
            "tokenizer.ggml.scores",
            GgufValue::ArrF32(toks.iter().map(|(_, _, s)| *s).collect()),
        ),
        kv(
            "tokenizer.ggml.token_type",
            GgufValue::ArrI32(toks.iter().map(|(_, t, _)| *t).collect()),
        ),
        kv("split.no", GgufValue::U16(shard_no)),
        kv("split.count", GgufValue::U16(N_SHARDS)),
        kv(
            "split.tensors.count",
            GgufValue::I32(i32::try_from(n_tensors).unwrap()),
        ),
    ];
    if let Some(c) = clamp_exp {
        out.push(kv(&a("swiglu_clamp_exp"), f(c)));
    }
    out
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fill {
    Ones,
    Zeros,
    Random,
}

/// f16 1e-3: the K-quant super-block scales, small so dequantized weights
/// stay near ±1.
const QUANT_SCALE_F16: [u8; 2] = 0x1419u16.to_le_bytes();

fn tensors(seed: u64) -> Vec<(GgufTensorMeta, Vec<u8>)> {
    use dims::*;
    let mut rng = Xorshift::new(seed);
    let mut out = Vec::new();
    let mut push = |name: String, dims: Vec<u64>, fill: Fill| {
        let n = dims.iter().product::<u64>() as usize;
        let data = (0..n)
            .flat_map(|_| {
                match fill {
                    Fill::Ones => 1.0f32,
                    Fill::Zeros => 0.0,
                    Fill::Random => rng.next_f32() * 0.1,
                }
                .to_le_bytes()
            })
            .collect();
        out.push((
            GgufTensorMeta {
                name,
                dims,
                ggml_type: GGML_TYPE_F32,
            },
            data,
        ));
    };
    let d_inner = KDA_HEAD_DIM * N_HEAD;
    let hc_mix = (2 + HC_MULT) * HC_MULT;

    push(
        "token_embd.weight".into(),
        d(&[N_EMBD, N_VOCAB]),
        Fill::Random,
    );
    push("output_norm.weight".into(), d(&[N_EMBD]), Fill::Ones);
    push("output.weight".into(), d(&[N_EMBD, N_VOCAB]), Fill::Random);
    for l in 0..N_LAYER {
        let mut t = |s: &str, dims: &[u32], fill| push(format!("blk.{l}.{s}"), d(dims), fill);
        t("attn_norm.weight", &[N_EMBD], Fill::Ones);
        t("ffn_norm.weight", &[N_EMBD], Fill::Ones);
        for p in ["attn", "ffn"] {
            t(
                &format!("hc_{p}_fn.weight"),
                &[HC_MULT * N_EMBD, hc_mix],
                Fill::Random,
            );
            t(&format!("hc_{p}_base.weight"), &[hc_mix], Fill::Random);
            t(&format!("hc_{p}_scale.weight"), &[3], Fill::Random);
        }
        if HEAD_COUNT_KV[l as usize] == 0 {
            for s in ["attn_q", "attn_k", "attn_v"] {
                t(&format!("{s}.weight"), &[N_EMBD, d_inner], Fill::Random);
            }
            for s in ["ssm_conv1d_q", "ssm_conv1d_k", "ssm_conv1d_v"] {
                t(
                    &format!("{s}.weight"),
                    &[SSM_CONV, 1, d_inner],
                    Fill::Random,
                );
            }
            t("ssm_f_a.weight", &[N_EMBD, KDA_HEAD_DIM], Fill::Random);
            t("ssm_f_b.weight", &[KDA_HEAD_DIM, d_inner], Fill::Random);
            t("ssm_g_a.weight", &[N_EMBD, KDA_HEAD_DIM], Fill::Random);
            t("ssm_g_b.weight", &[KDA_HEAD_DIM, d_inner], Fill::Random);
            t("ssm_beta.weight", &[N_EMBD, N_HEAD], Fill::Random);
            t("ssm_a", &[N_HEAD], Fill::Random);
            t("ssm_dt.bias", &[d_inner], Fill::Zeros);
            t("ssm_norm.weight", &[KDA_HEAD_DIM], Fill::Ones);
            t("attn_output.weight", &[d_inner, N_EMBD], Fill::Random);
        } else {
            t("attn_q_a.weight", &[N_EMBD, Q_LORA_RANK], Fill::Random);
            t("attn_q_a_norm.weight", &[Q_LORA_RANK], Fill::Ones);
            t(
                "attn_q_b.weight",
                &[Q_LORA_RANK, N_HEAD * KEY_LENGTH_MLA],
                Fill::Random,
            );
            t(
                "attn_kv_a_mqa.weight",
                &[N_EMBD, KV_LORA_RANK],
                Fill::Random,
            );
            t("attn_kv_a_norm.weight", &[KV_LORA_RANK], Fill::Ones);
            t(
                "attn_k_b.weight",
                &[KEY_LENGTH_MLA, KV_LORA_RANK, N_HEAD],
                Fill::Random,
            );
            t(
                "attn_v_b.weight",
                &[KV_LORA_RANK, VALUE_LENGTH_MLA, N_HEAD],
                Fill::Random,
            );
            t(
                "attn_output.weight",
                &[N_HEAD * VALUE_LENGTH_MLA, N_EMBD],
                Fill::Random,
            );
            t("indexer.k_norm.weight", &[INDEXER_KEY_LENGTH], Fill::Ones);
            t("indexer.k_norm.bias", &[INDEXER_KEY_LENGTH], Fill::Zeros);
            t(
                "indexer.proj.weight",
                &[N_EMBD, INDEXER_HEADS],
                Fill::Random,
            );
            t(
                "indexer.attn_k.weight",
                &[N_EMBD, INDEXER_KEY_LENGTH],
                Fill::Random,
            );
            t(
                "indexer.attn_q_b.weight",
                &[Q_LORA_RANK, INDEXER_HEADS * INDEXER_KEY_LENGTH],
                Fill::Random,
            );
            t(
                "indexer_compressor_gate.weight",
                &[N_EMBD, INDEXER_KEY_LENGTH],
                Fill::Random,
            );
            t(
                "indexer_compressor_ape.weight",
                &[INDEXER_KEY_LENGTH, INDEXER_KPOOL],
                Fill::Random,
            );
        }
        if l < N_LAYER_DENSE_LEAD {
            t("ffn_gate.weight", &[N_EMBD, N_FF], Fill::Random);
            t("ffn_up.weight", &[N_EMBD, N_FF], Fill::Random);
            t("ffn_down.weight", &[N_FF, N_EMBD], Fill::Random);
        } else {
            t("ffn_gate_inp.weight", &[N_EMBD, N_EXPERT], Fill::Random);
            t("exp_probs_b.bias", &[N_EXPERT], Fill::Zeros);
            t("ffn_gate_shexp.weight", &[N_EMBD, N_FF_SHEXP], Fill::Random);
            t("ffn_up_shexp.weight", &[N_EMBD, N_FF_SHEXP], Fill::Random);
            t("ffn_down_shexp.weight", &[N_FF_SHEXP, N_EMBD], Fill::Random);
        }
    }
    for l in N_LAYER_DENSE_LEAD..N_LAYER {
        for (kind, ggml_type) in ["gate", "up", "down"].into_iter().zip(EXPERT_TYPES) {
            let (ne0, ne1) = if kind == "down" {
                (N_FF_EXP, N_EMBD)
            } else {
                (N_EMBD, N_FF_EXP)
            };
            let dims = d(&[ne0, ne1, N_EXPERT]);
            out.push((
                GgufTensorMeta {
                    name: format!("blk.{l}.ffn_{kind}_exps.weight"),
                    dims: dims.clone(),
                    ggml_type,
                },
                quant_plane(&mut rng, ggml_type, &dims),
            ));
        }
    }
    out
}

/// Random K-quant blocks with every f16 super-block scale (`d`, and `dmin`
/// where present) pinned to [`QUANT_SCALE_F16`]: random scale bits would
/// include f16 Inf/NaN.
fn quant_plane(rng: &mut Xorshift, ggml_type: u32, dims: &[u64]) -> Vec<u8> {
    let (blck, size) = golem_gguf::ggml_type_block(ggml_type).unwrap();
    let n_blocks = dims.iter().product::<u64>() / blck;
    let scale_offsets: &[usize] = match ggml_type {
        GGML_TYPE_Q4_K | GGML_TYPE_Q5_K => &[0, 2],
        GGML_TYPE_Q6_K => &[208],
        _ => unreachable!(),
    };
    let mut out = Vec::with_capacity((n_blocks * size) as usize);
    for _ in 0..n_blocks {
        let mut block: Vec<u8> = (0..size)
            .map(|_| ((rng.next_f32() + 1.0) * 127.5) as u8)
            .collect();
        for &o in scale_offsets {
            block[o..o + 2].copy_from_slice(&QUANT_SCALE_F16);
        }
        out.extend(block);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shards() -> Vec<PathBuf> {
        let first = cached_fixture();
        (0..u64::from(N_SHARDS))
            .map(|no| golem_gguf::shard_path(&first, no, u64::from(N_SHARDS)).unwrap())
            .collect()
    }

    #[test]
    fn split_with_metadata_only_first_shard() {
        let paths = shards();
        for (no, p) in paths.iter().enumerate() {
            let mut f = std::fs::File::open(p).unwrap();
            let kvs = golem_gguf::read_kvs(&mut f).unwrap();
            assert_eq!(golem_gguf::kv_int(&kvs, "split.no"), Some(no as u64));
            assert_eq!(
                golem_gguf::kv_str(&kvs, "general.architecture"),
                Some("glm5next")
            );
            let n = golem_gguf::tensor_entries(&mut f).unwrap().len();
            assert_eq!(n == 0, no == 0, "{}", p.display());
        }
    }

    #[test]
    fn experts_are_matched_mixed_stride_k_quants() {
        let mut f = std::fs::File::open(&shards()[1]).unwrap();
        let entries = golem_gguf::tensor_entries(&mut f).unwrap();
        assert!(entries.iter().all(|e| e.name.ends_with(".weight")
            || e.name.ends_with(".bias")
            || e.name.ends_with("ssm_a")));
        let mut experts: Vec<_> = entries
            .iter()
            .filter(|e| crate::tiny_moe::is_expert_tensor(&e.name))
            .map(|e| {
                let row = golem_gguf::ggml_row_bytes(e.ggml_type, e.dims[0]).unwrap();
                (e.name.clone(), row)
            })
            .collect();
        experts.sort();
        assert_eq!(
            experts,
            [
                ("blk.1.ffn_down_exps.weight", 210),
                ("blk.1.ffn_gate_exps.weight", 144),
                ("blk.1.ffn_up_exps.weight", 176),
                ("blk.2.ffn_down_exps.weight", 210),
                ("blk.2.ffn_gate_exps.weight", 144),
                ("blk.2.ffn_up_exps.weight", 176),
            ]
            .map(|(n, r)| (n.to_owned(), r))
        );
    }

    #[test]
    fn has_recurrent_and_attention_layers() {
        let recr = dims::HEAD_COUNT_KV.iter().filter(|&&h| h == 0).count();
        assert!(recr >= 2 && recr < dims::N_LAYER as usize);
    }
}
