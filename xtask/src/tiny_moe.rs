//! Synthetic 2-layer / 8-expert glm4moe GGUF (design §12.1).

use std::io::Write;
use std::sync::OnceLock;

use regex::Regex;

use crate::gguf::{self, GGML_TYPE_F32, GgufKv, GgufTensorMeta, GgufValue};

pub mod dims {
    pub const N_VOCAB: u32 = 320;
    pub const N_EMBD: u32 = 32;
    pub const N_HEAD: u32 = 4;
    pub const N_HEAD_KV: u32 = 2;
    pub const HEAD_DIM: u32 = 8;
    pub const N_LAYER: u32 = 2;
    pub const N_LAYER_DENSE_LEAD: u32 = 1;
    pub const N_EXPERT: u32 = 8;
    pub const N_EXPERT_USED: u32 = 2;
    pub const N_EXPERT_SHARED: u32 = 0;
    pub const N_FF: u32 = 32;
    pub const N_FF_EXP: u32 = 16;
    pub const N_CTX: u32 = 256;
    pub const RMS_EPS: f32 = 1e-5;
    pub const ROPE_DIM: u32 = 8;
    pub const ROPE_FREQ_BASE: f32 = 10_000.0;
}

pub const DEFAULT_OUT: &str = "data/tiny-moe.gguf";
pub const DEFAULT_SEED: u64 = 0x67_6f_6c_65_6d; // "golem"

fn n_embd_k_gqa() -> u32 {
    dims::HEAD_DIM * dims::N_HEAD_KV
}

fn d(v: &[u32]) -> Vec<u64> {
    v.iter().map(|&x| u64::from(x)).collect()
}

/// splitmix64 premix keeps seed 0 valid and adjacent seeds distinct.
struct Xorshift(u64);

impl Xorshift {
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Xorshift((z ^ (z >> 31)) | 1)
    }

    fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let m = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32;
        m / (1u64 << 24) as f32 * 2.0 - 1.0
    }
}

pub fn tensor_names() -> Vec<String> {
    let mut names = vec![
        "token_embd.weight".to_owned(),
        "output_norm.weight".to_owned(),
        "output.weight".to_owned(),
    ];
    for l in 0..dims::N_LAYER {
        let blk = format!("blk.{l}.");
        names.extend(
            [
                "attn_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "post_attention_norm.weight",
            ]
            .map(|s| format!("{blk}{s}")),
        );
        if l < dims::N_LAYER_DENSE_LEAD {
            names.extend(
                ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"]
                    .map(|s| format!("{blk}{s}")),
            );
        } else {
            names.extend(
                [
                    "ffn_gate_inp.weight",
                    "exp_probs_b.bias",
                    "ffn_gate_exps.weight",
                    "ffn_up_exps.weight",
                    "ffn_down_exps.weight",
                ]
                .map(|s| format!("{blk}{s}")),
            );
        }
    }
    names
}

pub fn is_expert_tensor(name: &str) -> bool {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"ffn_(up|down|gate|gate_up)_(ch|)exps").unwrap())
        .is_match(name)
}

fn token_table() -> Vec<(String, i32, f32)> {
    let mut texts: Vec<String> = ["<unk>", "<s>", "</s>", "▁hello", "▁world", "<eoln>"]
        .iter()
        .map(|s| (*s).to_owned())
        .collect();
    for i in 0..58 {
        let letter = char::from(b'a' + (i % 26) as u8);
        texts.push(format!("▁{letter}{}", i / 26));
    }
    let types = [2, 3, 3].into_iter().chain(std::iter::repeat(1));
    let mut table: Vec<(String, i32, f32)> = texts
        .into_iter()
        .zip(types)
        .map(|(text, ttype)| (text, ttype, 0.0))
        .collect();
    // Token texts must be unique: llama.cpp SPM load hard-aborts (GGML_ASSERT
    // id_to_token.size() == token_to_id.size()) when token_to_id collapses a
    // duplicate, and byte_to_token resolves "<0xXX>" by text. The old id-5
    // "<0x0A>" duplicated the BYTE token at 64+10 and aborted every real load;
    // "<eoln>" keeps the slot newline-flavored without touching the byte block.
    for b in 0..=u8::MAX {
        table.push((format!("<0x{b:02X}>"), 6, 0.0));
    }
    table
}

pub fn generate(out: &mut impl Write, seed: u64) -> std::io::Result<()> {
    let mut rng = Xorshift::new(seed);
    let u = |v: u32| GgufValue::U32(v);
    let f = |v: f32| GgufValue::F32(v);
    let s = |v: &str| GgufValue::Str(v.to_owned());
    let toks = token_table();
    let kvs = vec![
        kv("general.architecture", s("glm4moe")),
        kv("general.name", s("tiny-moe")),
        kv("general.file_type", u(0)),
        kv("glm4moe.context_length", u(dims::N_CTX)),
        kv("glm4moe.embedding_length", u(dims::N_EMBD)),
        kv("glm4moe.block_count", u(dims::N_LAYER)),
        kv("glm4moe.feed_forward_length", u(dims::N_FF)),
        kv("glm4moe.expert_feed_forward_length", u(dims::N_FF_EXP)),
        kv("glm4moe.expert_count", u(dims::N_EXPERT)),
        kv("glm4moe.expert_used_count", u(dims::N_EXPERT_USED)),
        kv("glm4moe.expert_shared_count", u(dims::N_EXPERT_SHARED)),
        kv(
            "glm4moe.leading_dense_block_count",
            u(dims::N_LAYER_DENSE_LEAD),
        ),
        kv("glm4moe.attention.head_count", u(dims::N_HEAD)),
        kv("glm4moe.attention.head_count_kv", u(dims::N_HEAD_KV)),
        kv("glm4moe.attention.layer_norm_rms_epsilon", f(dims::RMS_EPS)),
        kv("glm4moe.attention.key_length", u(dims::HEAD_DIM)),
        kv("glm4moe.attention.value_length", u(dims::HEAD_DIM)),
        kv("glm4moe.rope.dimension_count", u(dims::ROPE_DIM)),
        kv("glm4moe.rope.freq_base", f(dims::ROPE_FREQ_BASE)),
        kv("tokenizer.ggml.model", s("llama")),
        kv(
            "tokenizer.ggml.tokens",
            GgufValue::ArrStr(toks.iter().map(|(t, _, _)| t.clone()).collect()),
        ),
        kv(
            "tokenizer.ggml.scores",
            GgufValue::ArrF32(toks.iter().map(|(_, _, score)| *score).collect()),
        ),
        kv(
            "tokenizer.ggml.token_type",
            GgufValue::ArrI32(toks.iter().map(|(_, ttype, _)| *ttype).collect()),
        ),
    ];

    let ne = u64::from(n_embd_k_gqa());
    let ne_full = u64::from(dims::N_EMBD);
    let nff = u64::from(dims::N_FF);
    let nffe = u64::from(dims::N_FF_EXP);
    let nx = u64::from(dims::N_EXPERT);

    let mut tensors: Vec<(GgufTensorMeta, Vec<u8>)> = Vec::new();
    let mut push =
        |tensors: &mut Vec<(GgufTensorMeta, Vec<u8>)>, name: &str, dims: Vec<u64>, fill: Fill| {
            let n = dims.iter().product::<u64>() as usize;
            let mut data = vec![0u8; n * 4];
            match fill {
                Fill::Ones | Fill::Zeros => {
                    let v = if fill == Fill::Ones { 1.0f32 } else { 0.0 };
                    for c in data.chunks_mut(4) {
                        c.copy_from_slice(&v.to_le_bytes());
                    }
                }
                Fill::Random => {
                    for c in data.chunks_mut(4) {
                        c.copy_from_slice(&rng.next_f32().to_le_bytes());
                    }
                }
            }
            tensors.push((
                GgufTensorMeta {
                    name: name.to_owned(),
                    dims,
                    ggml_type: GGML_TYPE_F32,
                },
                data,
            ));
        };

    push(
        &mut tensors,
        "token_embd.weight",
        d(&[32, dims::N_VOCAB]),
        Fill::Random,
    );
    push(&mut tensors, "output_norm.weight", d(&[32]), Fill::Ones);
    push(
        &mut tensors,
        "output.weight",
        d(&[32, dims::N_VOCAB]),
        Fill::Random,
    );
    for l in 0..dims::N_LAYER {
        let blk = format!("blk.{l}.");
        push(
            &mut tensors,
            &format!("{blk}attn_norm.weight"),
            vec![ne_full],
            Fill::Ones,
        );
        for (suffix, dims) in [
            ("attn_q.weight", vec![ne_full, ne_full]),
            ("attn_k.weight", vec![ne_full, ne]),
            ("attn_v.weight", vec![ne_full, ne]),
            ("attn_output.weight", vec![ne_full, ne_full]),
        ] {
            push(&mut tensors, &format!("{blk}{suffix}"), dims, Fill::Random);
        }
        push(
            &mut tensors,
            &format!("{blk}post_attention_norm.weight"),
            vec![ne_full],
            Fill::Ones,
        );
        if l < dims::N_LAYER_DENSE_LEAD {
            for suffix in ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"] {
                push(
                    &mut tensors,
                    &format!("{blk}{suffix}"),
                    vec![ne_full, nff],
                    Fill::Random,
                );
            }
        } else {
            push(
                &mut tensors,
                &format!("{blk}ffn_gate_inp.weight"),
                vec![ne_full, nx],
                Fill::Random,
            );
            push(
                &mut tensors,
                &format!("{blk}exp_probs_b.bias"),
                vec![nx],
                Fill::Zeros,
            );
            push(
                &mut tensors,
                &format!("{blk}ffn_gate_exps.weight"),
                vec![ne_full, nffe, nx],
                Fill::Random,
            );
            push(
                &mut tensors,
                &format!("{blk}ffn_up_exps.weight"),
                vec![ne_full, nffe, nx],
                Fill::Random,
            );
            push(
                &mut tensors,
                &format!("{blk}ffn_down_exps.weight"),
                vec![nffe, ne_full, nx],
                Fill::Random,
            );
        }
    }

    gguf::write_gguf(out, &kvs, &tensors)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fill {
    Ones,
    Zeros,
    Random,
}

fn kv(key: &str, value: GgufValue) -> GgufKv {
    GgufKv {
        key: key.to_owned(),
        value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gen_bytes() -> Vec<u8> {
        let mut buf = Vec::new();
        generate(&mut buf, DEFAULT_SEED).unwrap();
        buf
    }

    #[test]
    fn magic_and_shape_metadata() {
        let b = gen_bytes();
        assert_eq!(&b[0..4], b"GGUF");
        assert_eq!(u32::from_le_bytes(b[4..8].try_into().unwrap()), 3);
        let n_tensors = u64::from_le_bytes(b[8..16].try_into().unwrap());
        assert_eq!(n_tensors, 3 + 9 + 11);
        assert_eq!(b.len(), 179_040); // 64 + 256 byte-fallback tokens (+ vocab-shaped tensor growth)
    }

    #[test]
    fn byte_fallback_tokens_are_appended_byte_typed() {
        let toks = token_table();
        assert_eq!(toks.len(), 320);
        assert_eq!(toks[64].0, "<0x00>");
        assert_eq!(toks[255 + 64].0, "<0xFF>");
        for (i, (text, ttype, _)) in toks.iter().enumerate().skip(64) {
            assert_eq!(text, &format!("<0x{:02X}>", i - 64));
            assert_eq!(*ttype, 6);
        }
        assert_eq!(toks.iter().position(|t| t.0 == "▁hello"), Some(3));
    }

    #[test]
    fn expert_tensors_are_exactly_layer_1_exps() {
        let (_, tensors) = parse_header(&gen_bytes());
        let names: Vec<String> = tensors.iter().map(|(n, ..)| n.clone()).collect();
        assert_eq!(names, tensor_names());
        let mut found: Vec<String> = names.into_iter().filter(|n| is_expert_tensor(n)).collect();
        found.sort();
        assert_eq!(
            found,
            [
                "blk.1.ffn_down_exps.weight",
                "blk.1.ffn_gate_exps.weight",
                "blk.1.ffn_up_exps.weight",
            ]
            .map(String::from)
        );
    }

    #[test]
    fn deterministic_given_seed() {
        use sha2::{Digest, Sha256};
        let a = gen_bytes();
        let b = gen_bytes();
        let mut ha = Sha256::new();
        ha.update(&a);
        let mut hb = Sha256::new();
        hb.update(&b);
        assert_eq!(ha.finalize(), hb.finalize());
        let mut c = Vec::new();
        generate(&mut c, DEFAULT_SEED + 1).unwrap();
        assert_ne!(&a[..], &c[..]);
    }

    #[test]
    fn adjacent_seeds_diverge() {
        let mut a = Vec::new();
        generate(&mut a, 2).unwrap();
        let mut b = Vec::new();
        generate(&mut b, 3).unwrap();
        assert_ne!(&a[..], &b[..]);
    }

    #[test]
    fn kv_block_is_parseable_and_complete() {
        let b = gen_bytes();
        let (kvs, tensors) = parse_header(&b);
        let get = |k: &str| {
            kvs.iter()
                .find(|(key, _)| *key == k)
                .map(|(_, v)| v.clone())
        };
        assert_eq!(get("general.architecture").unwrap(), "glm4moe");
        assert_eq!(get("general.name").unwrap(), "tiny-moe");
        assert_eq!(get("general.file_type").unwrap(), KvVal::U32(0));
        assert_eq!(get("glm4moe.embedding_length").unwrap(), KvVal::U32(32));
        assert_eq!(get("glm4moe.block_count").unwrap(), KvVal::U32(2));
        assert_eq!(get("glm4moe.expert_count").unwrap(), KvVal::U32(8));
        assert_eq!(get("glm4moe.expert_used_count").unwrap(), KvVal::U32(2));
        assert_eq!(
            get("glm4moe.expert_feed_forward_length").unwrap(),
            KvVal::U32(16)
        );
        assert_eq!(get("glm4moe.feed_forward_length").unwrap(), KvVal::U32(32));
        assert_eq!(
            get("glm4moe.attention.layer_norm_rms_epsilon").unwrap(),
            KvVal::F32(1e-5)
        );
        assert_eq!(get("glm4moe.expert_shared_count").unwrap(), KvVal::U32(0));
        assert_eq!(
            get("glm4moe.leading_dense_block_count").unwrap(),
            KvVal::U32(1)
        );
        assert_eq!(get("glm4moe.attention.head_count").unwrap(), KvVal::U32(4));
        assert_eq!(
            get("glm4moe.attention.head_count_kv").unwrap(),
            KvVal::U32(2)
        );
        assert_eq!(get("glm4moe.attention.key_length").unwrap(), KvVal::U32(8));
        assert_eq!(
            get("glm4moe.attention.value_length").unwrap(),
            KvVal::U32(8)
        );
        assert_eq!(get("glm4moe.rope.dimension_count").unwrap(), KvVal::U32(8));
        assert_eq!(get("glm4moe.rope.freq_base").unwrap(), KvVal::F32(10_000.0));
        assert_eq!(get("glm4moe.context_length").unwrap(), KvVal::U32(256));
        assert_eq!(get("tokenizer.ggml.model").unwrap(), "llama");
        match get("tokenizer.ggml.tokens").unwrap() {
            KvVal::ArrStr(v) => assert_eq!(v.len(), 320),
            _ => panic!(),
        }
        assert_eq!(
            get("tokenizer.ggml.tokens").unwrap(),
            KvVal::ArrStr(token_table().into_iter().map(|(t, _, _)| t).collect())
        );
        assert_eq!(
            get("tokenizer.ggml.scores").unwrap(),
            KvVal::ArrF32(vec![0.0; 320])
        );
        let expected_types: Vec<i32> = [2, 3, 3]
            .into_iter()
            .chain(std::iter::repeat_n(1, 61))
            .chain(std::iter::repeat_n(6, 256))
            .collect();
        assert_eq!(
            get("tokenizer.ggml.token_type").unwrap(),
            KvVal::ArrI32(expected_types)
        );
        let names: Vec<_> = tensors.iter().map(|(n, ..)| n.clone()).collect();
        for expected in [
            "token_embd.weight",
            "output_norm.weight",
            "output.weight",
            "blk.0.attn_norm.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_output.weight",
            "blk.0.post_attention_norm.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "blk.1.attn_norm.weight",
            "blk.1.attn_q.weight",
            "blk.1.attn_k.weight",
            "blk.1.attn_v.weight",
            "blk.1.attn_output.weight",
            "blk.1.post_attention_norm.weight",
            "blk.1.ffn_gate_inp.weight",
            "blk.1.exp_probs_b.bias",
            "blk.1.ffn_gate_exps.weight",
            "blk.1.ffn_up_exps.weight",
            "blk.1.ffn_down_exps.weight",
        ] {
            assert!(names.contains(&expected.to_string()), "missing {expected}");
        }
        assert_eq!(names.len(), 23);
        let gate = tensors
            .iter()
            .find(|(n, ..)| n == "blk.1.ffn_gate_exps.weight")
            .unwrap();
        assert_eq!(gate.1, &[32, 16, 8]); // {n_embd, n_ff_exp, n_expert}
        let down = tensors
            .iter()
            .find(|(n, ..)| n == "blk.1.ffn_down_exps.weight")
            .unwrap();
        assert_eq!(down.1, &[16, 32, 8]); // {n_ff_exp, n_embd, n_expert}
        let tok = tensors
            .iter()
            .find(|(n, ..)| n == "token_embd.weight")
            .unwrap();
        assert_eq!(tok.1, &[32, 320]);
        let (_, _, _, bytes) = tensors
            .iter()
            .find(|(n, ..)| n == "output_norm.weight")
            .unwrap();
        let vals: Vec<f32> = bytes
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(vals, vec![1.0f32; 32]);
        assert!(tensors.iter().all(|(_, _, t, _)| *t == GGML_TYPE_F32));
    }

    type ParsedTensor = (String, Vec<u64>, u32, Vec<u8>);

    #[derive(Clone, Debug, PartialEq)]
    enum KvVal {
        U32(u32),
        F32(f32),
        Str(String),
        ArrI32(Vec<i32>),
        ArrF32(Vec<f32>),
        ArrStr(Vec<String>),
    }

    impl PartialEq<&str> for KvVal {
        fn eq(&self, other: &&str) -> bool {
            match self {
                KvVal::Str(s) => s == other,
                _ => false,
            }
        }
    }

    struct Cursor<'a> {
        b: &'a [u8],
        pos: usize,
    }

    impl<'a> Cursor<'a> {
        fn new(b: &'a [u8]) -> Self {
            Self { b, pos: 0 }
        }
        fn take(&mut self, n: usize) -> &'a [u8] {
            let s = &self.b[self.pos..self.pos + n];
            self.pos += n;
            s
        }
        fn u32(&mut self) -> u32 {
            u32::from_le_bytes(self.take(4).try_into().unwrap())
        }
        fn u64(&mut self) -> u64 {
            u64::from_le_bytes(self.take(8).try_into().unwrap())
        }
        fn str(&mut self) -> String {
            let n = self.u64() as usize;
            String::from_utf8(self.take(n).to_vec()).unwrap()
        }
    }

    fn parse_header(b: &[u8]) -> (Vec<(String, KvVal)>, Vec<ParsedTensor>) {
        let mut c = Cursor::new(b);
        assert_eq!(c.take(4), b"GGUF");
        assert_eq!(c.u32(), 3);
        let n_tensors = c.u64() as usize;
        let n_kv = c.u64() as usize;
        let mut kvs = Vec::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = c.str();
            let value = match c.u32() {
                4 => KvVal::U32(c.u32()),
                6 => KvVal::F32(f32::from_le_bytes(c.take(4).try_into().unwrap())),
                8 => KvVal::Str(c.str()),
                9 => {
                    let et = c.u32();
                    let n = c.u64() as usize;
                    match et {
                        5 => KvVal::ArrI32((0..n).map(|_| c.u32() as i32).collect()),
                        6 => KvVal::ArrF32(
                            (0..n)
                                .map(|_| f32::from_le_bytes(c.take(4).try_into().unwrap()))
                                .collect(),
                        ),
                        8 => KvVal::ArrStr((0..n).map(|_| c.str()).collect()),
                        t => panic!("unexpected array elem type {t}"),
                    }
                }
                t => panic!("unexpected kv type {t}"),
            };
            kvs.push((key, value));
        }
        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = c.str();
            let n_dims = c.u32() as usize;
            let dims: Vec<u64> = (0..n_dims).map(|_| c.u64()).collect();
            let ggml_type = c.u32();
            let offset = c.u64();
            let n: u64 = dims.iter().product();
            let start = 9408 + offset as usize; // header + KV + padded table end
            let end = start + n as usize * 4;
            assert!(end <= b.len(), "tensor {name} payload out of range");
            tensors.push((name, dims, ggml_type, b[start..end].to_vec()));
        }
        (kvs, tensors)
    }
}
