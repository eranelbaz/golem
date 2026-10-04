//! `golem import` (§5.5): scan a local shard set, hash + CRC it in one pass,
//! clone verified full mirrors, build the §4.2 manifest.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use golem_gguf::{
    blk_of, data_section_offset, ggml_row_bytes, kv_int, kv_str, live_layers, read_kvs, shard_path,
    tensor_entries,
};

/// Import failure.
#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    /// Bad source layout or header content.
    #[error("{path}: {msg}")]
    Source {
        /// Offending path.
        path: PathBuf,
        /// What was wrong.
        msg: String,
    },
    /// GGUF parse failure.
    #[error("{path}: {source}")]
    Gguf {
        /// Shard path.
        path: PathBuf,
        /// Parse error.
        #[source]
        source: golem_gguf::Error,
    },
    /// I/O failure.
    #[error("{path}: {source}")]
    Io {
        /// File or dir path.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },
    /// A replica's re-hash disagrees with the source.
    #[error("replica {path}: sha256 {got} != source {want}")]
    ReplicaMismatch {
        /// Replica file.
        path: PathBuf,
        /// Replica digest.
        got: String,
        /// Source digest.
        want: String,
    },
}

fn source(path: impl Into<PathBuf>, msg: impl Into<String>) -> ImportError {
    ImportError::Source {
        path: path.into(),
        msg: msg.into(),
    }
}

/// One expert plane: absolute offset of expert 0 and per-expert stride.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaneSpan {
    /// Full tensor name, e.g. `blk.1.ffn_gate_exps.weight`.
    pub tensor: String,
    /// Block the plane belongs to (live blocks only; MTP never lands here).
    pub layer: u32,
    /// Index into [`SourceScan::shards`].
    pub shard: usize,
    /// Absolute file offset of expert 0's first byte.
    pub base: u64,
    /// Bytes per expert slice (`ggml_row_bytes(ne0) · ne1`).
    pub stride: u64,
    /// Expert count (`dims[2]`).
    pub experts: u32,
}

/// Result of walking the source headers.
#[derive(Debug, Clone)]
pub struct SourceScan {
    /// Directory the shards were scanned from.
    pub dir: PathBuf,
    /// Shard paths in split order (a single file when unsplit).
    pub shards: Vec<PathBuf>,
    /// `(len, mtime)` per shard at scan time; re-checked after hashing.
    pub stamps: Vec<(u64, SystemTime)>,
    /// `general.architecture` of shard 1.
    pub arch: String,
    /// sha256 hex of `tokenizer.chat_template` on shard 1, empty when absent.
    pub chat_template_sha: String,
    /// Live expert planes, sorted by `(layer, tensor)`.
    pub planes: Vec<PlaneSpan>,
}

fn open_kvs(path: &Path) -> Result<(std::fs::File, Vec<(String, golem_gguf::KvVal)>), ImportError> {
    let mut f = std::fs::File::open(path).map_err(|e| ImportError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let kvs = read_kvs(&mut f).map_err(|e| ImportError::Gguf {
        path: path.to_path_buf(),
        source: e,
    })?;
    Ok((f, kvs))
}

/// Walks `dir`'s GGUF shard set: discovers and validates the split layout,
/// gates on `general.architecture`, then records the absolute geometry of
/// every expert plane on a live layer. Every expert tensor (MTP included) is
/// bounds-checked against the file length *before* the live-layer filter, so
/// truncation inside an excluded block is still rejected.
///
/// # Errors
/// [`ImportError::Source`] for a bad layout, [`ImportError::Gguf`] for
/// unparseable headers, [`ImportError::Io`] for filesystem failures.
pub fn scan_source(dir: &Path) -> Result<SourceScan, ImportError> {
    let mut names: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(dir).map_err(|e| ImportError::Io {
        path: dir.to_path_buf(),
        source: e,
    })? {
        let entry = entry.map_err(|e| ImportError::Io {
            path: dir.to_path_buf(),
            source: e,
        })?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".gguf") {
            names.push(name);
        }
    }
    names.sort();
    if let Some(n) = names.iter().find(|n| n.starts_with("mmproj")) {
        return Err(source(
            dir.join(n),
            "mmproj projector shards are not an importable model",
        ));
    }
    let Some(first_name) = names.first().cloned() else {
        return Err(source(dir, "no .gguf files found"));
    };
    let first_path = dir.join(&first_name);
    let (_, first_kvs) = open_kvs(&first_path)?;

    // Shard-set shape: single file, or the exact -NNNNN-of-MMMMM family.
    let count = kv_int(&first_kvs, "split.count");
    let shards: Vec<PathBuf> = match count {
        None => {
            if names.len() != 1 {
                return Err(source(
                    &first_path,
                    format!("no split.count but {} .gguf files present", names.len()),
                ));
            }
            vec![first_path.clone()]
        }
        Some(count) => {
            if kv_int(&first_kvs, "split.no") != Some(0) {
                return Err(source(
                    &first_path,
                    format!(
                        "split.no {:?} != 0 (first sorted file must be shard 1)",
                        kv_int(&first_kvs, "split.no")
                    ),
                ));
            }
            let expected: Vec<PathBuf> = (0..count)
                .map(|i| shard_path(&first_path, i, count))
                .collect::<std::io::Result<_>>()
                .map_err(|e| ImportError::Io {
                    path: first_path.clone(),
                    source: e,
                })?;
            let listed: std::collections::HashSet<&str> =
                names.iter().map(String::as_str).collect();
            for p in &expected {
                let n = p.file_name().unwrap().to_string_lossy();
                if !listed.contains(n.as_ref()) {
                    return Err(source(&first_path, format!("missing shard {n}")));
                }
            }
            for n in &names {
                if !expected
                    .iter()
                    .any(|p| p.file_name().unwrap().to_string_lossy() == *n)
                {
                    return Err(source(dir.join(n), format!("unexpected file {n}")));
                }
            }
            expected
        }
    };

    // Shard-1 gates: arch, chat template, live range, tensor total.
    let arch = match kv_str(&first_kvs, "general.architecture") {
        Some(a) if a == "glm5next" => a.to_owned(),
        other => {
            return Err(source(
                &first_path,
                format!("general.architecture {other:?} != \"glm5next\""),
            ));
        }
    };
    let chat_template_sha = match kv_str(&first_kvs, "tokenizer.chat_template") {
        Some(t) => sha256_hex(t.as_bytes()),
        None => String::new(),
    };
    let live = live_layers(&first_kvs).map_err(|e| ImportError::Gguf {
        path: first_path.clone(),
        source: e,
    })?;
    let want_tensors = match count {
        None => None,
        Some(_) => Some(
            kv_int(&first_kvs, "split.tensors.count")
                .ok_or_else(|| source(&first_path, "split model missing split.tensors.count"))?,
        ),
    };

    // Per-shard walk: split identity, section offset, expert geometry.
    let mut stamps = Vec::with_capacity(shards.len());
    let mut planes: Vec<PlaneSpan> = Vec::new();
    let mut total_tensors = 0u64;
    for (i, path) in shards.iter().enumerate() {
        let (mut f, kvs) = open_kvs(path)?;
        if let Some(c) = kv_int(&kvs, "split.count") {
            if Some(c) != count {
                return Err(source(
                    path,
                    format!("split.count {c} != first shard's {count:?}"),
                ));
            }
            let no = kv_int(&kvs, "split.no")
                .ok_or_else(|| source(path, "split.count present but split.no missing"))?;
            if no != i as u64 {
                return Err(source(path, format!("split.no {no} != {i}")));
            }
        }
        let data_start = data_section_offset(&mut f).map_err(|e| ImportError::Gguf {
            path: path.clone(),
            source: e,
        })?;
        let entries = tensor_entries(&mut f).map_err(|e| ImportError::Gguf {
            path: path.clone(),
            source: e,
        })?;
        total_tensors += entries.len() as u64;
        let md = f.metadata().map_err(|e| ImportError::Io {
            path: path.clone(),
            source: e,
        })?;
        let mtime = md.modified().map_err(|e| ImportError::Io {
            path: path.clone(),
            source: e,
        })?;
        stamps.push((md.len(), mtime));

        for e in &entries {
            if e.expert_kind().is_none() {
                continue;
            }
            let blk = blk_of(&e.name).ok_or_else(|| {
                source(
                    path,
                    format!("expert tensor {} lacks a blk.<N> prefix", e.name),
                )
            })?;
            let stride = ggml_row_bytes(e.ggml_type, e.dims[0])
                .and_then(|row| row.checked_mul(e.dims[1]))
                .ok_or_else(|| {
                    source(
                        path,
                        format!(
                            "plane {}: no whole-block rows (type {}, ne0 {})",
                            e.name, e.ggml_type, e.dims[0]
                        ),
                    )
                })?;
            let experts = u32::try_from(e.dims[2]).map_err(|_| {
                source(
                    path,
                    format!("plane {}: expert count {} out of range", e.name, e.dims[2]),
                )
            })?;
            let base = data_start.checked_add(e.offset).ok_or_else(|| {
                source(path, format!("plane {}: data offset overflows u64", e.name))
            })?;
            let end =
                base.checked_add(u64::from(experts).checked_mul(stride).ok_or_else(|| {
                    source(path, format!("plane {}: size overflows u64", e.name))
                })?)
                .ok_or_else(|| source(path, format!("plane {}: end overflows u64", e.name)))?;
            if end > md.len() {
                return Err(source(
                    path,
                    format!("plane {} runs past end of shard", e.name),
                ));
            }
            // Bounds before filter: an excluded (MTP/dense) plane must be
            // whole too, so the truncation test above fires on it.
            if !live.contains(&blk) {
                continue;
            }
            planes.push(PlaneSpan {
                tensor: e.name.clone(),
                layer: blk,
                shard: i,
                base,
                stride,
                experts,
            });
        }
    }

    if let Some(want) = want_tensors {
        if total_tensors != want {
            return Err(source(
                &first_path,
                format!("split.tensors.count {want} != {total_tensors} tensors across shards"),
            ));
        }
    }
    planes.sort_by(|a, b| (&a.layer, &a.tensor).cmp(&(&b.layer, &b.tensor)));
    if let Some(dup) = planes.windows(2).find(|w| w[0].tensor == w[1].tensor) {
        return Err(source(
            &first_path,
            format!("duplicate tensor {}", dup[0].tensor),
        ));
    }
    if planes.len() != 3 * live.len() as usize {
        return Err(source(
            &first_path,
            format!(
                "expected 3 expert planes (gate/up/down) per live layer, got {} planes across {} live layers",
                planes.len(),
                live.len()
            ),
        ));
    }
    Ok(SourceScan {
        dir: dir.to_path_buf(),
        shards,
        stamps,
        arch,
        chat_template_sha,
        planes,
    })
}

/// Lowercase sha256 hex.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
