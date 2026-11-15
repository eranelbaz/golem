//! Synthetic 3-layer `deepseek2` (GLM-4.7-Flash shape) GGUF: single file, MLA
//! attention with a q-LoRA, one dense lead layer, MoE FFN on blk.1..=2 with
//! separate rank-3 `ffn_{gate,up,down}_exps.weight` planes, a shared expert and
//! sigmoid gating with a selection bias. n_embd = 256 so the expert planes can
//! be mixed-stride K-quants (Q4_K/Q5_K/Q6_K: 144/176/210 B rows).
//!
//! Quant types differ per layer, as in real GLM-4.7-Flash files: see
//! [`layer_expert_types`]. Row strides and per-expert slot bytes therefore
//! differ between blk.1 and blk.2 (two distinct slot classes).
//!
//! Not matched to GLM-4.7-Flash (47 layers, 64 experts top-4, 2048 embd):
//! 8 experts × 36–53 KB planes keep generation fast.

use std::path::PathBuf;

use crate::gguf::{GgufKv, GgufTensorMeta, GgufValue};
use crate::tiny_hybrid::{EXPERT_TYPES, quant_plane};
use crate::tiny_moe::{Fill, Xorshift, d, kv, push_f32, repo_root, token_table, write_gguf_atomic};

pub mod dims {
    pub const N_VOCAB: u32 = 320;
    pub const N_EMBD: u32 = 256;
    pub const N_HEAD: u32 = 2;
    pub const N_LAYER: u32 = 3;
    pub const N_LAYER_DENSE_LEAD: u32 = 1;
    pub const N_EXPERT: u32 = 8;
    pub const N_EXPERT_USED: u32 = 2;
    pub const N_EXPERT_SHARED: u32 = 1;
    pub const N_FF: u32 = 256;
    pub const N_FF_EXP: u32 = 256;
    pub const N_FF_SHEXP: u32 = N_FF_EXP * N_EXPERT_SHARED;
    pub const N_CTX: u32 = 256;
    pub const Q_LORA_RANK: u32 = 64;
    pub const KV_LORA_RANK: u32 = 64;
    pub const ROPE_DIM: u32 = 16;
    pub const QK_NOPE: u32 = 32;
    /// `attention.key_length_mla`: nope + rope per head.
    pub const KEY_LENGTH_MLA: u32 = QK_NOPE + ROPE_DIM;
    pub const VALUE_LENGTH_MLA: u32 = 32;
}

pub const DEFAULT_SEED: u64 = crate::tiny_moe::DEFAULT_SEED;

/// Generates the fixture under `.test-cache/tiny-deepseek2/` and returns its
/// path. Safe to call concurrently.
pub fn cached_fixture() -> PathBuf {
    let dir = repo_root().join(".test-cache").join("tiny-deepseek2");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tiny-deepseek2-mixed.gguf");
    if !path.exists() {
        let tensors = tensors(DEFAULT_SEED);
        write_gguf_atomic(&path, &kvs(), &tensors);
    }
    path
}

/// (gate, up, down) ggml types of MoE layer `l`: `EXPERT_TYPES` rotated by
/// `l`, with down reusing gate's type so the per-expert total differs per layer.
pub fn layer_expert_types(l: u32) -> [u32; 3] {
    let r = |i: usize| EXPERT_TYPES[(l as usize + i) % 3];
    [r(0), r(1), r(0)]
}

/// Bytes of one expert's gate+up+down planes in MoE layer `l`.
pub fn layer_expert_bytes(l: u32) -> u64 {
    layer_expert_types(l)
        .iter()
        .map(|&t| {
            golem_gguf::ggml_row_bytes(t, u64::from(dims::N_EMBD)).unwrap()
                * u64::from(dims::N_FF_EXP)
        })
        .sum()
}

fn kvs() -> Vec<GgufKv> {
    let u = GgufValue::U32;
    let f = GgufValue::F32;
    let a = |k: &str| format!("deepseek2.{k}");
    let toks = token_table();
    vec![
        kv("general.architecture", GgufValue::Str("deepseek2".into())),
        kv("general.name", GgufValue::Str("tiny-deepseek2".into())),
        kv("general.file_type", u(0)),
        kv(&a("context_length"), u(dims::N_CTX)),
        kv(&a("embedding_length"), u(dims::N_EMBD)),
        kv(&a("block_count"), u(dims::N_LAYER)),
        kv(&a("vocab_size"), u(dims::N_VOCAB)),
        kv(&a("feed_forward_length"), u(dims::N_FF)),
        kv(&a("expert_feed_forward_length"), u(dims::N_FF_EXP)),
        kv(&a("expert_count"), u(dims::N_EXPERT)),
        kv(&a("expert_used_count"), u(dims::N_EXPERT_USED)),
        kv(&a("expert_shared_count"), u(dims::N_EXPERT_SHARED)),
        kv(&a("leading_dense_block_count"), u(dims::N_LAYER_DENSE_LEAD)),
        kv(&a("expert_weights_scale"), f(1.8)),
        kv(&a("expert_weights_norm"), GgufValue::Bool(true)),
        kv(&a("expert_gating_func"), u(2)), // sigmoid
        kv(&a("attention.head_count"), u(dims::N_HEAD)),
        kv(&a("attention.head_count_kv"), u(1)),
        kv(&a("attention.layer_norm_rms_epsilon"), f(1e-5)),
        kv(&a("attention.q_lora_rank"), u(dims::Q_LORA_RANK)),
        kv(&a("attention.kv_lora_rank"), u(dims::KV_LORA_RANK)),
        kv(
            &a("attention.key_length"),
            u(dims::KV_LORA_RANK + dims::ROPE_DIM),
        ),
        kv(&a("attention.value_length"), u(dims::KV_LORA_RANK)),
        kv(&a("attention.key_length_mla"), u(dims::KEY_LENGTH_MLA)),
        kv(&a("attention.value_length_mla"), u(dims::VALUE_LENGTH_MLA)),
        kv(&a("rope.dimension_count"), u(dims::ROPE_DIM)),
        kv(&a("rope.freq_base"), f(10000.0)),
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
    ]
}

fn tensors(seed: u64) -> Vec<(GgufTensorMeta, Vec<u8>)> {
    use dims::*;
    let mut rng = Xorshift::new(seed);
    let mut out = Vec::new();
    let mut push =
        |name: String, dims: Vec<u64>, fill: Fill| push_f32(&mut out, &mut rng, name, dims, fill);

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
        t("attn_q_a.weight", &[N_EMBD, Q_LORA_RANK], Fill::Random);
        t("attn_q_a_norm.weight", &[Q_LORA_RANK], Fill::Ones);
        t(
            "attn_q_b.weight",
            &[Q_LORA_RANK, N_HEAD * KEY_LENGTH_MLA],
            Fill::Random,
        );
        t(
            "attn_kv_a_mqa.weight",
            &[N_EMBD, KV_LORA_RANK + ROPE_DIM],
            Fill::Random,
        );
        t("attn_kv_a_norm.weight", &[KV_LORA_RANK], Fill::Ones);
        t(
            "attn_k_b.weight",
            &[QK_NOPE, KV_LORA_RANK, N_HEAD],
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
        t("ffn_norm.weight", &[N_EMBD], Fill::Ones);
        if l < N_LAYER_DENSE_LEAD {
            t("ffn_gate.weight", &[N_EMBD, N_FF], Fill::Random);
            t("ffn_up.weight", &[N_EMBD, N_FF], Fill::Random);
            t("ffn_down.weight", &[N_FF, N_EMBD], Fill::Random);
        } else {
            t("ffn_gate_inp.weight", &[N_EMBD, N_EXPERT], Fill::Random);
            t("exp_probs_b.bias", &[N_EXPERT], Fill::Random);
            t("ffn_gate_shexp.weight", &[N_EMBD, N_FF_SHEXP], Fill::Random);
            t("ffn_up_shexp.weight", &[N_EMBD, N_FF_SHEXP], Fill::Random);
            t("ffn_down_shexp.weight", &[N_FF_SHEXP, N_EMBD], Fill::Random);
        }
    }
    for l in N_LAYER_DENSE_LEAD..N_LAYER {
        for (kind, ggml_type) in ["gate", "up", "down"]
            .into_iter()
            .zip(layer_expert_types(l))
        {
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

#[cfg(test)]
mod tests {
    use super::*;
    use golem_gguf::ExpertKind;

    fn entries() -> Vec<golem_gguf::TensorEntry> {
        let mut f = std::fs::File::open(cached_fixture()).unwrap();
        golem_gguf::tensor_entries(&mut f).unwrap()
    }

    #[test]
    fn single_file_deepseek2() {
        let mut f = std::fs::File::open(cached_fixture()).unwrap();
        let kvs = golem_gguf::read_kvs(&mut f).unwrap();
        assert_eq!(
            golem_gguf::kv_str(&kvs, "general.architecture"),
            Some("deepseek2")
        );
        assert_eq!(golem_gguf::kv_int(&kvs, "split.no"), None);
        assert!(!entries().is_empty());
    }

    #[test]
    fn slot_bytes_differ_across_moe_layers() {
        let (a, b) = (layer_expert_bytes(1), layer_expert_bytes(2));
        assert_eq!(a, (176 + 210 + 176) * 256);
        assert_eq!(b, (210 + 144 + 210) * 256);
        assert_ne!(a, b);
    }

    #[test]
    fn experts_are_separate_mixed_stride_k_quants() {
        let entries = entries();
        assert!(
            entries
                .iter()
                .all(|e| e.name.ends_with(".weight") || e.name.ends_with(".bias"))
        );
        let mut experts: Vec<_> = entries
            .iter()
            .filter(|e| crate::tiny_moe::is_expert_tensor(&e.name))
            .map(|e| {
                assert!(
                    ExpertKind::from_tensor_name(&e.name).is_some(),
                    "{}",
                    e.name
                );
                assert_eq!(e.dims.len(), 3, "{}", e.name);
                assert_eq!(e.dims[2], u64::from(dims::N_EXPERT));
                let row = golem_gguf::ggml_row_bytes(e.ggml_type, e.dims[0]).unwrap();
                (e.name.clone(), row)
            })
            .collect();
        experts.sort();
        assert_eq!(
            experts,
            [
                ("blk.1.ffn_down_exps.weight", 176),
                ("blk.1.ffn_gate_exps.weight", 176),
                ("blk.1.ffn_up_exps.weight", 210),
                ("blk.2.ffn_down_exps.weight", 210),
                ("blk.2.ffn_gate_exps.weight", 210),
                ("blk.2.ffn_up_exps.weight", 144),
            ]
            .map(|(n, r)| (n.to_owned(), r))
        );
        for e in entries.iter().filter(|e| {
            e.name.contains("shexp") || e.name.contains("gate_inp") || e.name.contains("probs")
        }) {
            assert_eq!(ExpertKind::from_tensor_name(&e.name), None, "{}", e.name);
        }
    }
}
