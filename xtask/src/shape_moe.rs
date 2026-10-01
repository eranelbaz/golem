//! Shape-realistic split-GGUF fixture (review §6 "Weakness 2"): 2 shards with
//! a metadata-only shard 1, `.weight` expert names, mixed Q4_K/Q5_K/Q6_K
//! expert planes of differing strides, 64 experts, and an MTP/NextN expert
//! layer that pool geometry must exclude (review R-2). Not loadable by the
//! substrate — this fixture exercises byte-source/preflight paths only.

use std::io::Write;
use std::path::PathBuf;

use crate::gguf::{self, GgufKv, GgufTensorMeta, GgufValue};
use golem_gguf::{GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K};

pub mod dims {
    pub const N_BLOCK: u32 = 5;
    pub const DENSE_LEAD: u32 = 1;
    pub const N_EXPERT: u32 = 64;
    pub const NEXTN: u32 = 1;
    /// Live expert layers: blk.1..=3, one per unit size class; blk.4 is MTP.
    pub const LIVE: [u32; 3] = [1, 2, 3];
    pub const NE0: u32 = 256;
    pub const NE1: u32 = 32;
}

const N_SHARDS: u64 = 2;

/// Single source of truth for the fixture: generates the split GGUF under
/// `.test-cache/shape-moe/` and returns the shard paths.
pub fn cached_fixture() -> Vec<PathBuf> {
    let dir = repo_root().join(".test-cache/shape-moe-v2");
    std::fs::create_dir_all(&dir).unwrap();
    let paths: Vec<PathBuf> = (1..=N_SHARDS)
        .map(|i| dir.join(format!("shape-moe-{:0>5}-of-{:0>5}.gguf", i, N_SHARDS)))
        .collect();
    if paths.iter().all(|p| p.exists()) {
        return paths;
    }
    let tensors = expert_tensors();
    for (shard_no, path) in paths.iter().enumerate() {
        let shard_tensors: &[(GgufTensorMeta, Vec<u8>)] = if shard_no == 0 {
            &[] // metadata-only first shard, like the real 6-shard model
        } else {
            &tensors
        };
        let tmp = path.with_extension("gguf.tmp");
        let f = std::fs::File::create(&tmp).unwrap();
        let mut w = std::io::BufWriter::new(f);
        write_shard(&mut w, u64::try_from(shard_no).unwrap(), shard_tensors).unwrap();
        w.flush().unwrap();
        std::fs::rename(&tmp, path).unwrap();
    }
    paths
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .canonicalize()
        .unwrap()
}

fn kv(key: &str, value: GgufValue) -> GgufKv {
    GgufKv {
        key: key.to_owned(),
        value,
    }
}

fn kvs(shard_no: u64) -> Vec<GgufKv> {
    let u = |v: u64| GgufValue::U32(u32::try_from(v).unwrap());
    vec![
        kv("general.architecture", GgufValue::Str("glm4moe".into())),
        kv("general.file_type", u(0)),
        kv("glm4moe.block_count", u(u64::from(dims::N_BLOCK))),
        kv("glm4moe.expert_count", u(u64::from(dims::N_EXPERT))),
        kv("glm4moe.expert_used_count", u(8)),
        kv(
            "glm4moe.leading_dense_block_count",
            u(u64::from(dims::DENSE_LEAD)),
        ),
        kv(
            "glm4moe.expert_feed_forward_length",
            u(u64::from(dims::NE1)),
        ),
        kv("glm4moe.nextn_predict_layers", u(u64::from(dims::NEXTN))),
        kv("split.no", u(shard_no)),
        kv("split.count", u(N_SHARDS)),
        kv("split.tensors.count", u(12)),
    ]
}

/// Per-block plane types: blk 1..=3 each land in a distinct expert-unit size
/// class; blk 4 is the MTP block (Q4/Q4/Q5 like blk 1).
fn expert_tensors() -> Vec<(GgufTensorMeta, Vec<u8>)> {
    let mut out = Vec::new();
    let types = [
        [GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K],
        [GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K],
        [GGML_TYPE_Q5_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K],
        [GGML_TYPE_Q4_K, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K],
    ];
    for (i, blk) in (dims::DENSE_LEAD..dims::N_BLOCK).enumerate() {
        for (kind, (name, ggml_type)) in [
            ("ffn_gate_exps", types[i][0]),
            ("ffn_up_exps", types[i][1]),
            ("ffn_down_exps", types[i][2]),
        ]
        .into_iter()
        .enumerate()
        {
            let (ne0, ne1) = (dims::NE0, dims::NE1);
            let row = golem_gguf::ggml_row_bytes(ggml_type, u64::from(ne0)).unwrap();
            let len = row * u64::from(ne1) * u64::from(dims::N_EXPERT);
            out.push((
                GgufTensorMeta {
                    name: format!("blk.{blk}.{name}.weight"),
                    dims: vec![u64::from(ne0), u64::from(ne1), u64::from(dims::N_EXPERT)],
                    ggml_type,
                },
                vec![fill_byte(blk, kind); len.try_into().unwrap()],
            ));
        }
    }
    out
}

/// Provenance byte: identifies the (block, plane kind) a read came from.
pub fn fill_byte(blk: u32, kind: usize) -> u8 {
    0x10 * blk as u8 + kind as u8 + 1
}

fn write_shard(
    w: &mut impl Write,
    shard_no: u64,
    tensors: &[(GgufTensorMeta, Vec<u8>)],
) -> std::io::Result<()> {
    gguf::write_gguf(w, &kvs(shard_no), tensors).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shards_are_split_and_first_is_metadata_only() {
        let paths = cached_fixture();
        assert_eq!(paths.len(), 2);
        for (no, p) in paths.iter().enumerate() {
            let mut f = std::fs::File::open(p).unwrap();
            let (_, n_tensors, n_kv) = golem_gguf::read_header(&mut f).unwrap();
            let kvs = golem_gguf::read_kvs(&mut f).unwrap();
            assert_eq!(
                golem_gguf::kv_int(&kvs, "split.no"),
                Some(no as u64),
                "{}",
                p.display()
            );
            assert_eq!(golem_gguf::kv_int(&kvs, "split.count"), Some(2));
            if no == 0 {
                assert_eq!(n_tensors, 0, "shard 1 must be metadata-only");
            } else {
                assert_eq!(n_tensors, 12, "4 expert blocks × 3 planes");
            }
            assert_eq!(n_kv, kvs.len() as u64);
        }
    }

    #[test]
    fn live_layers_span_three_unit_classes() {
        let paths = cached_fixture();
        let mut f = std::fs::File::open(&paths[1]).unwrap();
        let entries = golem_gguf::tensor_entries(&mut f).unwrap();
        let unit = |blk: u32| -> u64 {
            ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
                .iter()
                .map(|k| {
                    let e = entries
                        .iter()
                        .find(|e| e.name == format!("blk.{blk}.{k}.weight"))
                        .unwrap();
                    golem_gguf::ggml_row_bytes(e.ggml_type, e.dims[0]).unwrap() * e.dims[1]
                })
                .sum()
        };
        // Q4/Q4/Q5, Q4/Q4/Q6, Q5/Q5/Q6 rows at ne0 = 256 (144/176/210 B) × ne1 = 32.
        assert_eq!(dims::LIVE.map(unit), [14_848, 15_936, 17_984]);
        // MTP block 4 exists in the file and is excluded by the source walk.
        assert!(
            entries
                .iter()
                .any(|e| e.name == "blk.4.ffn_down_exps.weight")
        );
    }

    #[test]
    fn expert_planes_have_mixed_quant_strides() {
        let mut f = std::fs::File::open(&cached_fixture()[1]).unwrap();
        let entries = golem_gguf::tensor_entries(&mut f).unwrap();
        let strides: Vec<Vec<u64>> = dims::LIVE
            .iter()
            .map(|&blk| {
                entries
                    .iter()
                    .filter(|e| e.name.contains(&format!("blk.{blk}.")))
                    .map(|e| golem_gguf::ggml_row_bytes(e.ggml_type, e.dims[0]).unwrap())
                    .collect()
            })
            .collect();
        assert_eq!(
            strides,
            vec![
                vec![144, 144, 176],
                vec![144, 144, 210],
                vec![176, 176, 210],
            ],
            "Q4/Q4/Q5, Q4/Q4/Q6, Q5/Q5/Q6 rows at ne0=256"
        );
    }
}
