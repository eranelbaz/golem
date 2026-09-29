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

fn validate(m: &ModelManifest, path: &Path) -> Result<(), StoreError> {
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
    Ok(())
}

fn is_sha256_hex(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

#[cfg(test)]
mod tests {
    use super::*;

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
        "expert_index": [ { "tensor": "blk.0.ffn_gate_exps", "layer": 0, "experts": 8,
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
