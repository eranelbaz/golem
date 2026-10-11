//! Model manifest types and validating loader (§4.2).

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::StoreError;

/// §4.2 manifest. Serde-renamed to the design's snake_case JSON keys verbatim.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Model name.
    pub name: String,
    /// Shards making up the model.
    pub shards: Vec<ShardEntry>,
    /// Source repo, passthrough.
    #[serde(default)]
    pub repo: Option<String>,
    /// Source revision, passthrough.
    #[serde(default)]
    pub revision: Option<String>,
    /// Where `revision` came from: `user`, `hf-cache`, or `unknown` (§5.5).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub revision_source: Option<String>,
    /// Quantization label, passthrough.
    #[serde(default)]
    pub quant: Option<String>,
    /// Architecture label, passthrough.
    #[serde(default)]
    pub arch: Option<String>,
    /// Expert-offset index for MoE tensors, passthrough.
    #[serde(default)]
    pub expert_index: Option<Vec<ExpertIndexEntry>>,
    /// Chat template digest, passthrough.
    #[serde(default)]
    pub chat_template_sha: Option<String>,
}

/// One shard: file name, size, digest, replica roots.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ShardEntry {
    /// Shard file name, relative to each replica root.
    pub file: String,
    /// Shard size in bytes.
    pub bytes: u64,
    /// Lowercase hex; `load_manifest` validates hex + length 64.
    pub sha256: String,
    /// Replica ROOT dirs; shard file lives at `replicas[i]/file`.
    pub replicas: Vec<PathBuf>,
    /// R-8 start-time check: SHA-256 of 16 × 1 MiB windows at `i·len/16`
    /// ([`crate::import::sampled_sha256`]). Absent in pre-R-8 manifests →
    /// size + mtime only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampled_sha256: Option<Vec<String>>,
    /// Shard mtime (Unix seconds) recorded at import; absent → not checked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mtime: Option<u64>,
}

/// Expert-offset index entry for MoE tensors.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExpertIndexEntry {
    /// Tensor name.
    pub tensor: String,
    /// Layer index.
    pub layer: u32,
    /// Expert count.
    pub experts: u32,
    /// Bytes per expert row.
    pub bytes_per_expert: u64,
    /// Byte offset of each expert within the tensor.
    pub offsets_bytes: Vec<u64>,
    /// CRC32C of each expert's bytes in this plane, index = expert id (§5.3, D-4).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crc32c: Option<Vec<u32>>,
}

/// Load and validate a manifest from `path`.
pub fn load_manifest(path: &Path) -> Result<ModelManifest, StoreError> {
    let path = path.to_path_buf();
    let text = std::fs::read_to_string(&path).map_err(|source| StoreError::Manifest {
        path: path.clone(),
        msg: format!("read: {source}"),
    })?;
    let m: ModelManifest = serde_json::from_str(&text).map_err(|e| StoreError::Manifest {
        path: path.clone(),
        msg: format!("parse: {e}"),
    })?;
    validate(&m, &path)?;
    Ok(m)
}

pub(crate) fn validate(m: &ModelManifest, path: &Path) -> Result<(), StoreError> {
    let err = |msg: String| StoreError::Manifest {
        path: path.to_path_buf(),
        msg,
    };
    if m.name.is_empty() {
        return Err(err("`name` must be non-empty".into()));
    }
    if m.shards.is_empty() {
        return Err(err("`shards` must contain at least one shard".into()));
    }
    for shard in &m.shards {
        if shard.bytes == 0 {
            return Err(err(format!("shard `{}`: `bytes` must be > 0", shard.file)));
        }
        if !is_sha256_hex(&shard.sha256) {
            return Err(err(format!(
                "shard `{}`: `sha256` must be 64 lowercase hex chars",
                shard.file
            )));
        }
        if shard.replicas.is_empty() {
            return Err(err(format!(
                "shard `{}`: `replicas` must be non-empty",
                shard.file
            )));
        }
    }
    for e in m.expert_index.iter().flatten() {
        let n = e.experts as usize;
        if e.offsets_bytes.len() != n {
            return Err(err(format!(
                "`{}`: offsets_bytes has {} entries, experts = {n}",
                e.tensor,
                e.offsets_bytes.len()
            )));
        }
        if let Some(c) = e.crc32c.as_ref().filter(|c| c.len() != n) {
            return Err(err(format!(
                "`{}`: crc32c has {} entries, experts = {n}",
                e.tensor,
                c.len()
            )));
        }
    }
    Ok(())
}

fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir_for(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("golem-store-manifest-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_tmp(dir: &Path, name: &str, body: &str) -> PathBuf {
        let p = dir.join(name);
        std::fs::write(&p, body).unwrap();
        p
    }

    const VALID: &str = r#"{
        "name": "tiny", "repo": "x/y", "revision": "abc", "quant": "Q8_0", "arch": "glm5next",
        "shards": [ { "file": "a.gguf", "bytes": 4096,
                      "sha256": "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
                      "replicas": ["/mnt/d0/golem", "/mnt/d1/golem"] } ],
        "expert_index": [ { "tensor": "blk.0.ffn_gate_exps", "layer": 0, "experts": 2,
                            "bytes_per_expert": 512, "offsets_bytes": [0, 512] } ],
        "chat_template_sha": "cafe"
    }"#;

    #[test]
    fn parses_full_shape() {
        let dir = std::env::temp_dir().join("golem-store-manifest-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = write_tmp(&dir, "ok.json", VALID);
        let m = load_manifest(&p).unwrap();
        assert_eq!(m.name, "tiny");
        assert_eq!(m.shards.len(), 1);
        assert_eq!(m.shards[0].replicas.len(), 2);
        assert_eq!(m.shards[0].file, "a.gguf");
    }

    #[test]
    fn rejects_bad_sha256() {
        let dir = std::env::temp_dir().join("golem-store-manifest-test");
        std::fs::create_dir_all(&dir).unwrap();
        let bad = VALID.replace(
            "00112233445566778899aabbccddeeff00112233445566778899aabbccddeeff",
            "nothex",
        );
        let p = write_tmp(&dir, "badsha.json", &bad);
        let err = load_manifest(&p).unwrap_err();
        assert!(err.to_string().contains("sha256"));
    }

    #[test]
    fn crc_len_must_match_experts() {
        let dir = tempdir_for("crc_len");
        let body = VALID.replace(
            r#""offsets_bytes""#,
            r#""crc32c": [1, 2, 3], "offsets_bytes""#,
        );
        let e = load_manifest(&write_tmp(&dir, "m.json", &body))
            .unwrap_err()
            .to_string();
        assert!(e.contains("crc32c"), "{e}");
    }

    #[test]
    fn offsets_len_must_match_experts() {
        let dir = tempdir_for("off_len");
        let m: ModelManifest = serde_json::from_str(VALID).unwrap();
        let mut m2 = m.clone();
        m2.expert_index.as_mut().unwrap()[0].offsets_bytes.pop();
        let body = serde_json::to_string(&m2).unwrap();
        let e = load_manifest(&write_tmp(&dir, "m.json", &body))
            .unwrap_err()
            .to_string();
        assert!(e.contains("offsets_bytes"), "{e}");
    }

    #[test]
    fn m03_manifest_without_new_fields_still_loads() {
        let dir = tempdir_for("legacy");
        let m = load_manifest(&write_tmp(&dir, "m.json", VALID)).unwrap();
        assert!(m.revision_source.is_none());
        assert!(m.expert_index.unwrap().iter().all(|e| e.crc32c.is_none()));
    }

    #[test]
    #[ignore = "requires GOLEM_M03_MANIFEST"]
    fn m03_real_manifest_loads() {
        let path = std::env::var("GOLEM_M03_MANIFEST").expect("set GOLEM_M03_MANIFEST");
        let m = load_manifest(Path::new(&path)).unwrap();
        assert_eq!(m.expert_index.unwrap().len(), 129);
    }

    #[test]
    fn rejects_empty_replicas() {
        let dir = std::env::temp_dir().join("golem-store-manifest-test");
        std::fs::create_dir_all(&dir).unwrap();
        let bad = VALID.replace(
            "\"replicas\": [\"/mnt/d0/golem\", \"/mnt/d1/golem\"]",
            "\"replicas\": []",
        );
        let p = write_tmp(&dir, "noreplicas.json", &bad);
        assert!(load_manifest(&p).is_err());
    }
}
