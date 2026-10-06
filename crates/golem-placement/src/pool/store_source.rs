//! Store-backed [`ExpertSource`] (§5.2–5.4): the manifest's replica set,
//! read through the shared [`ChunkQueue`] — one queued, per-drive drain per
//! expert plane instead of a blocking QD1 pread. Plane addressing comes from
//! the manifest's `expert_index` (`offsets_bytes` per expert), verified at
//! construction against the replica files' GGUF tensor tables (§5.3: sizes
//! re-checked on open).

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io;
use std::path::Path;
use std::sync::Arc;

use golem_gguf::{
    arch_kv_int, blk_of, data_section_offset, ggml_row_bytes, read_kvs, shard_path, tensor_entries,
};
use golem_store::{ChunkQueue, ReadJob, ReplicaRegistry, StoreError, load_manifest, plan_aligned};

use super::residency::ExpertKind;
use super::source::{DIRECT_ALIGN, ExpertSource, SourceError, plane_index};

/// Absolute per-expert addressing for one `(layer, kind)` plane, resolved
/// from the manifest entry plus the shard's GGUF tensor table.
struct StorePlane {
    /// Position of the owning shard in the manifest's `shards` vector.
    shard: usize,
    /// Absolute file offset of each expert's row (tensor base +
    /// `offsets_bytes[i]`).
    offsets: Vec<u64>,
    /// Bytes per expert row (GGUF row stride; equals the manifest's
    /// `bytes_per_expert`, checked at open).
    stride: u64,
    /// Expert count.
    n_experts: u32,
    /// §5.3 expected CRC32C per expert; `None` skips verification.
    crc: Option<Vec<u32>>,
}

/// Store-backed [`ExpertSource`] handle: the opened replica registry, the
/// shared queue, and the resolved plane table.
pub struct StoreExpertSource {
    queue: Arc<ChunkQueue>,
    registry: ReplicaRegistry,
    /// Indexed by the same flat index as [`super::source`] (`plane_index`).
    planes: Vec<Option<StorePlane>>,
    /// Whether §5.3 CRC verification is active (manifest carried `crc32c`).
    crc_enabled: bool,
}

impl fmt::Debug for StoreExpertSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StoreExpertSource")
            .field(
                "live_planes",
                &self.planes.iter().filter(|p| p.is_some()).count(),
            )
            .field("crc_enabled", &self.crc_enabled)
            .finish_non_exhaustive()
    }
}

/// Upper bound on a plausible `blk.N` index (same rule as the GGUF walk).
const MAX_BLOCKS: u32 = 1 << 16;

/// Upper bound on shard split count: bounded reservation, same defense as
/// the GGUF source (a bogus count fails on the first missing shard).
const MAX_SHARDS: u64 = 1 << 12;

thread_local! {
    /// Reused aligned covering-region scratch for planes whose slot region
    /// cannot host the queued read themselves (misaligned tensor base or an
    /// undersized plane buffer). Reserved to the largest plane once, then
    /// reused: steady-state page-in allocates nothing (§5.4 hot path).
    static SCRATCH: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

impl StoreExpertSource {
    /// Opens the manifest's replica set and resolves every `expert_index`
    /// entry to absolute per-expert file offsets. The manifest offsets are
    /// verified against the replica files' GGUF tensor tables and the shard
    /// sizes re-checked (`ReplicaRegistry::open`, §5.3); a missing
    /// `expert_index` is a construction error (fail fast, §4.3).
    ///
    /// # Errors
    /// Manifest load/parse failure, replica open failure, a tensor-table
    /// mismatch, or a manifest entry that cannot be tied to a live expert
    /// plane.
    pub fn open(manifest_path: &Path, queue: Arc<ChunkQueue>) -> Result<Self, SourceError> {
        let manifest = load_manifest(manifest_path)?;
        let index = manifest
            .expert_index
            .as_ref()
            .filter(|i| !i.is_empty())
            .ok_or_else(|| {
                SourceError::Store(StoreError::Invalid {
                    msg: format!(
                        "manifest {}: missing `expert_index`",
                        manifest_path.display()
                    ),
                })
            })?
            .clone();
        let registry = ReplicaRegistry::open(&manifest)?;

        // Shard-split arithmetic, same as GgufPreadSource derives from the
        // GGUF headers: shard i's absolute tensor base = data-section start
        // + table offset; expert i's row sits `offsets_bytes[i]` into the
        // tensor. Walk the FIRST replica only — replicas are byte-identical
        // by construction (§5.3 digests) and the registry verified sizes.
        let first = manifest.shards[0].replicas[0].join(&manifest.shards[0].file);
        let mut f0 = File::open(&first)?;
        let kvs = read_kvs(&mut f0)?;
        let split_count = match golem_gguf::kv_int(&kvs, "split.count") {
            None => 1,
            Some(n) if n == 0 || n > MAX_SHARDS => {
                return Err(SourceError::Store(StoreError::Invalid {
                    msg: format!("implausible split.count {n} in {}", first.display()),
                }));
            }
            Some(n) => n,
        };
        // Dense lead: pool layers are renumbered to exclude it, exactly like
        // the GGUF source — manifest layers are `blk.N` GGUF indices.
        let dense_lead = u32::try_from(arch_kv_int(&kvs, "leading_dense_block_count").unwrap_or(0))
            .map_err(|_| {
                SourceError::Store(StoreError::Invalid {
                    msg: "leading_dense_block_count > u32::MAX".into(),
                })
            })?;

        // tensor name → (shard, absolute base, row stride, n_experts).
        let mut tables: HashMap<String, (usize, u64, u64, u32)> = HashMap::new();
        for shard in 0..split_count {
            let shard_ix = shard as usize;
            let sp = shard_path(&first, shard, split_count)?;
            if split_count > 1 {
                // The manifest's shard list must be the split order; a
                // reordered manifest would map planes onto wrong files.
                let want = manifest
                    .shards
                    .get(shard_ix)
                    .ok_or_else(|| {
                        SourceError::Store(StoreError::Invalid {
                            msg: format!(
                                "manifest has {} shards but the GGUF split has {split_count}",
                                manifest.shards.len()
                            ),
                        })
                    })?
                    .file
                    .clone();
                if sp.file_name().and_then(|n| n.to_str()) != Some(want.as_str()) {
                    return Err(SourceError::Store(StoreError::Invalid {
                        msg: format!(
                            "shard {shard}: manifest file `{want}` != discovered `{}`",
                            sp.display()
                        ),
                    }));
                }
            }
            let mut f = File::open(&sp)?;
            let data_start = data_section_offset(&mut f)?;
            for e in tensor_entries(&mut f)? {
                let Some(kind) = e.expert_kind() else {
                    continue;
                };
                let _ = kind;
                let stride = ggml_row_bytes(e.ggml_type, e.dims[0])
                    .ok_or(SourceError::UnknownGgmlType {
                        name: e.name.clone(),
                        ggml_type: e.ggml_type,
                    })?
                    .checked_mul(e.dims[1])
                    .ok_or_else(|| SourceError::StrideOverflow {
                        name: e.name.clone(),
                    })?;
                let base = data_start.checked_add(e.offset).ok_or_else(|| {
                    SourceError::StrideOverflow {
                        name: e.name.clone(),
                    }
                })?;
                let n_experts =
                    u32::try_from(e.dims[2]).map_err(|_| SourceError::StrideOverflow {
                        name: e.name.clone(),
                    })?;
                tables.insert(e.name.clone(), (shard_ix, base, stride, n_experts));
            }
        }

        let mut planes: Vec<Option<StorePlane>> = Vec::new();
        let mut crc_enabled = false;
        for entry in &index {
            let kind = ExpertKind::from_tensor_name(&entry.tensor)
                .ok_or_else(|| SourceError::UnnamedBlock(entry.tensor.clone()))?;
            let blk = blk_of(&entry.tensor)
                .ok_or_else(|| SourceError::UnnamedBlock(entry.tensor.clone()))?;
            if blk >= MAX_BLOCKS {
                return Err(SourceError::BlockOutOfRange {
                    name: entry.tensor.clone(),
                });
            }
            if blk < dense_lead {
                return Err(SourceError::Store(StoreError::Invalid {
                    msg: format!(
                        "`{}`: blk.{blk} is inside the dense lead of {dense_lead}",
                        entry.tensor
                    ),
                }));
            }
            let layer = blk - dense_lead;
            let Some(&(shard, base, stride, n_experts)) = tables.get(&entry.tensor) else {
                return Err(SourceError::Store(StoreError::Invalid {
                    msg: format!("`{}`: not found in the replica tensor tables", entry.tensor),
                }));
            };
            if entry.bytes_per_expert != stride || entry.experts != n_experts {
                return Err(SourceError::Store(StoreError::Invalid {
                    msg: format!(
                        "`{}`: manifest says {} experts × {} B, tensor table says {n_experts} × {stride} B",
                        entry.tensor, entry.experts, entry.bytes_per_expert
                    ),
                }));
            }
            // §5.3: sizes re-checked on open — every expert row must lie
            // inside the registry-verified shard size.
            let file_len = manifest.shards[shard].bytes;
            let mut offsets = Vec::with_capacity(entry.offsets_bytes.len());
            for &o in &entry.offsets_bytes {
                let abs = base
                    .checked_add(o)
                    .ok_or_else(|| SourceError::StrideOverflow {
                        name: entry.tensor.clone(),
                    })?;
                if stride > file_len.saturating_sub(abs) {
                    return Err(SourceError::Store(StoreError::Invalid {
                        msg: format!(
                            "`{}`: expert row at {abs} exceeds shard size {file_len} (§5.3)",
                            entry.tensor
                        ),
                    }));
                }
                offsets.push(abs);
            }
            crc_enabled |= entry.crc32c.is_some();
            let ix = plane_index(layer, kind);
            if planes.len() <= ix {
                planes.resize_with(ix + 1, || None);
            }
            planes[ix] = Some(StorePlane {
                shard,
                offsets,
                stride,
                n_experts,
                crc: entry.crc32c.clone(),
            });
        }
        if !crc_enabled {
            // Once at startup, not once per read (§5.3 D-4 opt-out).
            tracing::info!(
                manifest = %manifest_path.display(),
                "manifest carries no crc32c digests; plane CRC verification is skipped"
            );
        }
        Ok(Self {
            queue,
            registry,
            planes,
            crc_enabled,
        })
    }

    /// Whether §5.3 CRC verification is active.
    pub fn crc_enabled(&self) -> bool {
        self.crc_enabled
    }

    /// Total pageable expert bytes (Σ `stride × n_experts` over live planes).
    pub fn expert_bytes(&self) -> u64 {
        self.planes
            .iter()
            .flatten()
            .map(|p| p.stride * u64::from(p.n_experts))
            .sum()
    }

    /// The opened replica registry, for health diagnostics.
    pub fn registry(&self) -> &ReplicaRegistry {
        &self.registry
    }
}

impl ExpertSource for StoreExpertSource {
    fn read_expert(
        &self,
        layer: u32,
        kind: ExpertKind,
        expert: u32,
        into: &mut [u8],
        head: usize,
    ) -> io::Result<()> {
        let Some(p) = self
            .planes
            .get(plane_index(layer, kind))
            .and_then(Option::as_ref)
        else {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("no expert plane layer {layer} kind {kind:?}"),
            ));
        };
        if expert >= p.n_experts {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("expert {expert} out of range (n={})", p.n_experts),
            ));
        }
        let stride = usize::try_from(p.stride).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "plane stride exceeds usize")
        })?;
        if into.len() < head.saturating_add(stride) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "buffer {} < head {head} + plane stride {stride}",
                    into.len()
                ),
            ));
        }
        let offset = p.offsets[expert as usize];
        let expected = p.crc.as_ref().map(|c| c[expert as usize]);
        // One queued drain per plane (§5.2): enqueue the aligned covering
        // chunks, wait — per-drive striping and replica failover happen
        // inside the drain, never a blocking QD1 read.
        let covering: usize = plan_aligned(offset, stride).iter().map(|c| c.1).sum();
        let logical_at = (offset % DIRECT_ALIGN) as usize;
        // Fast path: the plane buffer can host the aligned covering read
        // itself (aligned base + head scratch room) — bytes land in place,
        // no copy.
        if head == logical_at && into.len() >= covering {
            return self.drain(p.shard, offset, stride, expected, &mut into[..covering]);
        }
        SCRATCH.with(|s| -> io::Result<()> {
            let mut scratch = s.borrow_mut();
            scratch.resize(covering, 0);
            self.drain(p.shard, offset, stride, expected, &mut scratch)?;
            into[head..head + stride].copy_from_slice(&scratch[logical_at..logical_at + stride]);
            Ok(())
        })
    }

    fn plane(&self, layer: u32, kind: ExpertKind, expert: u32) -> Option<(u64, u64)> {
        let p = self.planes.get(plane_index(layer, kind))?.as_ref()?;
        Some((*p.offsets.get(expert as usize)?, p.stride))
    }
}

impl StoreExpertSource {
    /// Enqueues one plane's logical read and drains it through the shared
    /// queue, mapping [`StoreError`] onto [`io::Error`].
    fn drain(
        &self,
        shard: usize,
        offset: u64,
        stride: usize,
        expected: Option<u32>,
        dst: &mut [u8],
    ) -> io::Result<()> {
        self.queue
            .enqueue(ReadJob {
                shard_index: shard,
                offset,
                len: stride,
                expected_crc: expected.unwrap_or(0),
                verify_crc: expected.is_some(),
                prefetch: false,
                sources: &self.registry,
                dst,
            })
            .wait()
            .map_err(|e| match e {
                StoreError::Io { source, .. } => source,
                other @ StoreError::Unavailable { .. } => {
                    io::Error::new(io::ErrorKind::NotFound, other.to_string())
                }
                other @ StoreError::Crc { .. } => {
                    io::Error::new(io::ErrorKind::InvalidData, other.to_string())
                }
                other => io::Error::new(io::ErrorKind::InvalidInput, other.to_string()),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    const SHARD0: &str = "m-00001-of-00002.gguf";
    const SHARD1: &str = "m-00002-of-00002.gguf";
    const FILE_BYTES: u64 = 8192;
    /// f32 plane `dims = [4, 1, 2]` → 16 B per expert row, 2 experts.
    const STRIDE: u64 = 16;
    const N_EXPERTS: u32 = 2;

    /// Expert-row bytes for shard/expert: shard 0 rows are 0..16, 16..32;
    /// shard 1 rows start at 100.
    fn row_bytes(shard: usize, expert: u32) -> Vec<u8> {
        let base = if shard == 0 { 0u8 } else { 100 };
        ((expert * STRIDE as u32)..(expert * STRIDE as u32 + STRIDE as u32))
            .map(|i| base + i as u8)
            .collect()
    }

    fn write_shard(dir: &Path, name: &str, shard_no: u64, kvs: &[xtask::gguf::GgufKv]) -> u64 {
        let kind = match shard_no {
            0 => "ffn_gate_exps",
            _ => "ffn_up_exps",
        };
        let mut payload = row_bytes(shard_no as usize, 0);
        payload.extend(row_bytes(shard_no as usize, 1));
        let meta = xtask::gguf::GgufTensorMeta {
            name: format!("blk.1.{kind}.weight"),
            dims: vec![4, 1, u64::from(N_EXPERTS)],
            ggml_type: 0, // F32
        };
        let f = std::fs::File::create(dir.join(name)).unwrap();
        let data_start = xtask::gguf::write_gguf(&mut &f, kvs, &[(meta, payload)]).unwrap();
        drop(f);
        // Pad to FILE_BYTES so an aligned 4 KiB covering read never hits EOF.
        let len = dir.join(name).metadata().unwrap().len();
        std::fs::OpenOptions::new()
            .append(true)
            .open(dir.join(name))
            .unwrap()
            .write_all(&vec![0u8; (FILE_BYTES - len) as usize])
            .unwrap();
        data_start
    }

    use std::io::Write as _;

    /// Two-shard fixture: blk.1 gate in shard 0, up in shard 1; dense lead 1.
    /// Returns (manifest path, shard-0 data-section offset).
    fn fixture(tag: &str, crc: bool) -> (PathBuf, u64, u64) {
        let dir =
            std::env::temp_dir().join(format!("golem-store-src-{tag}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let kvs0 = vec![
            xtask::gguf::GgufKv {
                key: "general.architecture".into(),
                value: xtask::gguf::GgufValue::Str("glm4moe".into()),
            },
            xtask::gguf::GgufKv {
                key: "glm4moe.leading_dense_block_count".into(),
                value: xtask::gguf::GgufValue::U32(1),
            },
            xtask::gguf::GgufKv {
                key: "split.count".into(),
                value: xtask::gguf::GgufValue::U32(2),
            },
        ];
        let kvs1 = vec![xtask::gguf::GgufKv {
            key: "split.count".into(),
            value: xtask::gguf::GgufValue::U32(2),
        }];
        let d0 = write_shard(&dir, SHARD0, 0, &kvs0);
        let d1 = write_shard(&dir, SHARD1, 1, &kvs1);

        let crc_json = |shard: usize| -> String {
            if !crc {
                return String::new();
            }
            let c0 = crc32c::crc32c(&row_bytes(shard, 0));
            let c1 = crc32c::crc32c(&row_bytes(shard, 1));
            format!(r#", "crc32c": [{c0}, {c1}]"#)
        };
        let sha = "0".repeat(64);
        let rep = dir.display();
        let json = format!(
            r#"{{
                "name": "{tag}",
                "shards": [
                    {{"file": "{SHARD0}", "bytes": {FILE_BYTES}, "sha256": "{sha}", "replicas": ["{rep}"]}},
                    {{"file": "{SHARD1}", "bytes": {FILE_BYTES}, "sha256": "{sha}", "replicas": ["{rep}"]}}
                ],
                "expert_index": [
                    {{"tensor": "blk.1.ffn_gate_exps.weight", "layer": 1, "experts": {N_EXPERTS},
                      "bytes_per_expert": {STRIDE}, "offsets_bytes": [0, {STRIDE}]{}}},
                    {{"tensor": "blk.1.ffn_up_exps.weight", "layer": 1, "experts": {N_EXPERTS},
                      "bytes_per_expert": {STRIDE}, "offsets_bytes": [0, {STRIDE}]{}}}
                ]
            }}"#,
            crc_json(0),
            crc_json(1),
        );
        let path = dir.join("manifest.json");
        std::fs::write(&path, json).unwrap();
        (path, d0, d1)
    }

    #[test]
    fn maps_layer_kind_expert_to_manifest_shard_offset() {
        let (mpath, d0, d1) = fixture("map", true);
        let src = StoreExpertSource::open(&mpath, Arc::new(ChunkQueue::new(1))).unwrap();
        // plane(): entry offsets_bytes position + tensor base (dense lead 1 →
        // pool layer 0). Gate lives in shard 0 at tensor offset 0.
        let (off, stride) = src.plane(0, ExpertKind::Gate, 1).unwrap();
        assert_eq!(stride, STRIDE);
        assert_eq!(off, d0 + STRIDE, "tensor base + offsets_bytes[1]");
        let (uoff, _) = src.plane(0, ExpertKind::Up, 0).unwrap();
        assert_eq!(uoff, d1, "up plane is tensor 0 of the second shard file");

        // read_expert serves the fake bytes: expert rows written by the
        // fixture generator, read back through the shared queue.
        let mut buf = vec![0u8; STRIDE as usize];
        src.read_expert(0, ExpertKind::Gate, 1, &mut buf, 0)
            .unwrap();
        assert_eq!(buf, row_bytes(0, 1));
        src.read_expert(0, ExpertKind::Up, 0, &mut buf, 0).unwrap();
        assert_eq!(buf, row_bytes(1, 0));
    }

    #[test]
    fn missing_expert_index_fails_construction() {
        let dir =
            std::env::temp_dir().join(format!("golem-store-src-noindex-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let sha = "0".repeat(64);
        std::fs::write(dir.join("a.gguf"), vec![0u8; 8192]).unwrap();
        let json = format!(
            r#"{{"name": "t", "shards": [{{"file": "a.gguf", "bytes": 8192,
                  "sha256": "{sha}", "replicas": ["{}"]}}]}}"#,
            dir.display()
        );
        let p = dir.join("manifest.json");
        std::fs::write(&p, json).unwrap();
        let err = StoreExpertSource::open(&p, Arc::new(ChunkQueue::new(1))).unwrap_err();
        assert!(err.to_string().contains("expert_index"), "{err}");
    }

    #[test]
    fn absent_crc32c_skips_verification_and_logs_once() {
        let (mpath, _, _) = fixture("nocrc", false);
        let src = StoreExpertSource::open(&mpath, Arc::new(ChunkQueue::new(1))).unwrap();
        assert!(
            !src.crc_enabled(),
            "no crc32c in manifest → verification off"
        );
        // Reads still succeed with verification skipped.
        let mut buf = vec![0u8; STRIDE as usize];
        src.read_expert(0, ExpertKind::Gate, 0, &mut buf, 0)
            .unwrap();
        assert_eq!(buf, row_bytes(0, 0));

        let (mpath, _, _) = fixture("withcrc", true);
        let src = StoreExpertSource::open(&mpath, Arc::new(ChunkQueue::new(1))).unwrap();
        assert!(src.crc_enabled());
    }

    #[test]
    fn read_expert_returns_io_error_when_all_replicas_unavailable() {
        let (mpath, _, _) = fixture("unavail", true);
        let src = StoreExpertSource::open(&mpath, Arc::new(ChunkQueue::new(1))).unwrap();
        for s in src.registry().shards() {
            for r in &s.replicas {
                r.mark_degraded();
            }
        }
        let mut buf = vec![0u8; STRIDE as usize];
        let err = src
            .read_expert(0, ExpertKind::Gate, 0, &mut buf, 0)
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound, "{err}");
    }
}
