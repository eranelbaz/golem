//! Expert source seam (§5.4 byte source, v1): one GGUF replica path, `pread`
//! at per-plane file offsets derived from a tensor-table walk. Multi-replica
//! striping, CRC verification, and failover are M3.1/M3.3.

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use golem_gguf::{data_section_offset, ggml_row_bytes, tensor_entries};

use super::residency::ExpertKind;

/// Pool geometry derived from the expert plane table (§5.4): a uniform slot
/// size (the max plane stride — per-kind strides may diverge on quants), the
/// dense-lead-adjusted layer/expert counts, and total expert bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolGeometry {
    /// Max plane stride in bytes — the arena slot size.
    pub slot_size: u64,
    /// Expert layer count (dense lead excluded).
    pub n_layers: u32,
    /// Experts per layer (uniform; the loader enforces cross-tensor equality).
    pub n_experts: u32,
    /// Sum over planes of `stride × n_experts` — total pageable expert bytes.
    pub expert_bytes: u64,
}

/// Reads one expert plane from a byte source.
pub trait ExpertSource: Send + Sync {
    /// Read one expert plane into the first `stride` bytes of `into` (length
    /// ≥ that stride; slots may exceed the plane when per-kind strides
    /// diverge — §3.4). Blocking; bounded internal retries on EINTR/short
    /// reads; a 0-byte read is `UnexpectedEof`. `layer` is the
    /// DENSE-LEAD-ADJUSTED index (blk.1 is expert layer 0 in tiny-moe).
    ///
    /// # Errors
    /// I/O failure after the retry bound, unknown plane, or buffer/stride
    /// mismatch.
    fn read_expert(
        &self,
        layer: u32,
        kind: ExpertKind,
        expert: u32,
        into: &mut [u8],
    ) -> io::Result<()>;

    /// Absolute file offset + stride of one plane, for construction-time
    /// cross-checks against live tensors.
    fn plane(&self, layer: u32, kind: ExpertKind, expert: u32) -> Option<(u64, u64)>;
}

/// Source failures that are not plain I/O.
#[derive(Debug)]
pub enum SourceError {
    /// File/GGUF I/O failure.
    Io(io::Error),
    /// GGUF table parse failure.
    Gguf(golem_gguf::Error),
    /// Expert-tensor name could not be tied to a `blk.N.` prefix.
    UnnamedBlock(
        /// Offending tensor name.
        String,
    ),
    /// A ggml type without a block layout in M2.3's table.
    UnknownGgmlType {
        /// Offending tensor name.
        name: String,
        /// Unmapped ggml type tag.
        ggml_type: u32,
    },
    /// Plane byte size overflowed `u64`.
    StrideOverflow {
        /// Offending tensor name.
        name: String,
    },
}

impl fmt::Display for SourceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "expert source I/O: {e}"),
            Self::Gguf(e) => write!(f, "expert source GGUF: {e}"),
            Self::UnnamedBlock(name) => {
                write!(f, "expert tensor without blk.N prefix: {name}")
            }
            Self::UnknownGgmlType { name, ggml_type } => {
                write!(f, "tensor {name}: unknown ggml type {ggml_type}")
            }
            Self::StrideOverflow { name } => write!(f, "tensor {name}: plane stride overflow"),
        }
    }
}

impl std::error::Error for SourceError {}

impl From<io::Error> for SourceError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<golem_gguf::Error> for SourceError {
    fn from(e: golem_gguf::Error) -> Self {
        Self::Gguf(e)
    }
}

/// Per-(layer, kind) plane geometry: shard index, absolute file offset of
/// expert 0, the plane byte stride (from the GGUF row size — per-kind, never
/// a constant), and the expert count.
#[derive(Debug, Clone, Copy)]
struct Plane {
    shard: usize,
    base: u64,
    stride: u64,
    n_experts: u32,
}

/// One open GGUF replica set; `pread`-style reads at computed plane offsets
/// (the file cursors are never shared, so concurrent reads are safe).
#[derive(Debug)]
pub struct GgufPreadSource {
    files: Vec<File>,
    planes: HashMap<(u32, ExpertKind), Plane>,
}

/// `blk.<N>.` prefix parse.
fn blk_of(name: &str) -> Option<u32> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

impl GgufPreadSource {
    /// Aggregate pool geometry over the walked expert planes, or `None` when
    /// the model carries no expert tensors.
    pub fn geometry(&self) -> Option<PoolGeometry> {
        let mut slot_size = 0u64;
        let mut n_layers = 0u32;
        let mut n_experts = 0u32;
        let mut expert_bytes = 0u64;
        for (&(layer, _), p) in &self.planes {
            slot_size = slot_size.max(p.stride);
            n_layers = n_layers.max(layer + 1);
            n_experts = n_experts.max(p.n_experts);
            expert_bytes =
                expert_bytes.checked_add(p.stride.checked_mul(u64::from(p.n_experts))?)?;
        }
        (slot_size > 0).then_some(PoolGeometry {
            slot_size,
            n_layers,
            n_experts,
            expert_bytes,
        })
    }

    /// Opens the shard set rooted at `path` and walks every shard's GGUF
    /// tensor table (review C-5: split GGUFs, where the first shard can be
    /// metadata-only). Sibling shards are discovered by rewriting the
    /// `-NNNNN-of-MMMMM.gguf` suffix per `split.count`. Expert layers are
    /// numbered EXCLUDING the dense lead AND the trailing MTP/NextN blocks
    /// (review R-2): pool layer = gguf blk − `n_dense_lead`, live only for
    /// `n_dense_lead ≤ blk < block_count − nextn_predict_layers`.
    ///
    /// # Errors
    /// File open/parse failures, a missing/misnamed sibling shard, or an
    /// expert tensor whose stride math fails.
    pub fn open(path: &Path, n_dense_lead: u32) -> Result<Self, SourceError> {
        let kvs = golem_gguf::read_kvs(&mut File::open(path)?)?;
        let split_count = golem_gguf::kv_int(&kvs, "split.count").unwrap_or(1);
        let nextn = golem_gguf::arch_kv_int(&kvs, "nextn_predict_layers").unwrap_or(0);
        let block_count = golem_gguf::arch_kv_int(&kvs, "block_count");
        let mtp_from = block_count.map(|bc| {
            // A nextn value larger than block_count is a corrupt KV, not
            // "skip everything": name it.
            assert!(
                nextn <= bc,
                "nextn_predict_layers {nextn} > block_count {bc}"
            );
            bc - nextn
        });

        let mut files = Vec::with_capacity(split_count as usize);
        let mut planes = HashMap::new();
        for shard in 0..split_count {
            let shard_path = shard_path(path, shard, split_count)?;
            let mut file = File::open(&shard_path)?;
            let data_start = data_section_offset(&mut file)?;
            for e in tensor_entries(&mut file)? {
                let Some(kind) = ExpertKind::from_tensor_name(&e.name) else {
                    continue;
                };
                let Some(blk) = blk_of(&e.name) else {
                    return Err(SourceError::UnnamedBlock(e.name));
                };
                if blk < n_dense_lead {
                    continue; // dense lead carries no expert tensors
                }
                if let Some(mtp_from) = mtp_from {
                    if u64::from(blk) >= mtp_from {
                        continue; // MTP/NextN block: skipped at load (R-2)
                    }
                }
                let stride = ggml_row_bytes(e.ggml_type, e.dims[0])
                    .ok_or(SourceError::UnknownGgmlType {
                        name: e.name.clone(),
                        ggml_type: e.ggml_type,
                    })?
                    .checked_mul(e.dims[1])
                    .ok_or(SourceError::StrideOverflow {
                        name: e.name.clone(),
                    })?;
                planes.insert(
                    (blk - n_dense_lead, kind),
                    Plane {
                        shard: shard as usize,
                        base: data_start + e.offset,
                        stride,
                        n_experts: e.dims[2] as u32,
                    },
                );
            }
            files.push(file);
        }
        Ok(Self { files, planes })
    }
}

/// Sibling-shard path for shard `no` of `count`, rewriting the standard
/// `-NNNNN-of-MMMMM.gguf` suffix of `first`. Single-file models (`count ==
/// 1`) return `first` unchanged.
fn shard_path(first: &Path, no: u64, count: u64) -> Result<PathBuf, SourceError> {
    if count == 1 {
        return Ok(first.to_path_buf());
    }
    let name = first.file_name().and_then(|n| n.to_str()).ok_or_else(|| {
        SourceError::Io(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("non-UTF-8 model path {first:?}"),
        ))
    })?;
    let (stem, ext) = name.rsplit_once('.').unwrap_or((name, ""));
    let Some((prefix, counts)) = stem.rsplit_once("-of-") else {
        return Err(SourceError::Io(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "split.count = {count} but {name} does not end in -NNNNN-of-MMMMM.{ext:?}; \
                 cannot discover sibling shards"
            ),
        )));
    };
    let Some((base, idx)) = prefix.rsplit_once('-') else {
        return Err(SourceError::Io(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("cannot split shard index from {name}"),
        )));
    };
    if !idx.bytes().all(|b| b.is_ascii_digit()) || idx.is_empty() {
        return Err(SourceError::Io(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("shard index {idx:?} in {name} is not numeric"),
        )));
    }
    let width = idx.len();
    let ext_dot = if ext.is_empty() {
        String::new()
    } else {
        format!(".{ext}")
    };
    Ok(first.with_file_name(format!("{base}-{:0width$}-of-{counts}{ext_dot}", no + 1)))
}

/// Retry bound for EINTR / short reads (Global Constraints: bounded, then
/// surface the failure — the FFI layer aborts loudly).
const READ_RETRIES: usize = 64;

impl ExpertSource for GgufPreadSource {
    fn read_expert(
        &self,
        layer: u32,
        kind: ExpertKind,
        expert: u32,
        into: &mut [u8],
    ) -> io::Result<()> {
        let Some(p) = self.planes.get(&(layer, kind)) else {
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
        if into.len() < stride {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("buffer {} < plane stride {stride}", into.len()),
            ));
        }
        let plane = &mut into[..stride];
        let mut off = p.base + u64::from(expert) * p.stride;
        let mut done = 0usize;
        for _ in 0..READ_RETRIES {
            // Safe pread wrapper (FileExt::read_at): does not move the cursor,
            // so concurrent readers cannot disturb each other. Interrupted
            // reads retry within the same bounded loop as short reads.
            let n = match self.files[p.shard].read_at(&mut plane[done..], off) {
                Ok(n) => n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            };
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("pread at {off} returned 0"),
                ));
            }
            done += n;
            off += n as u64;
            if done == plane.len() {
                return Ok(());
            }
        }
        Err(io::Error::new(
            io::ErrorKind::Interrupted,
            format!("pread did not complete after {READ_RETRIES} retries"),
        ))
    }

    fn plane(&self, layer: u32, kind: ExpertKind, expert: u32) -> Option<(u64, u64)> {
        let p = self.planes.get(&(layer, kind))?;
        Some((p.base + u64::from(expert) * p.stride, p.stride))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> std::path::PathBuf {
        xtask::tiny_moe::cached_fixture()
    }

    /// Byte-exact expected plane bytes: replays xtask::tiny_moe's seeded
    /// (splitmix64-premix xorshift) f32 stream — the RNG and the per-tensor
    /// draw counts are replicated exactly, so this is a generator check, not a
    /// self-consistent re-read.
    struct FixtureRng(u64);

    impl FixtureRng {
        fn new(seed: u64) -> Self {
            let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            Self((z ^ (z >> 31)) | 1)
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

        /// f32 LE bytes after skipping `skip` draws.
        fn plane_bytes(&mut self, skip: usize, n: usize) -> Vec<u8> {
            for _ in 0..skip {
                self.next_f32();
            }
            let mut out = Vec::with_capacity(n * 4);
            for _ in 0..n {
                out.extend_from_slice(&self.next_f32().to_le_bytes());
            }
            out
        }
    }

    /// Random-fill draws before `blk.1.ffn_gate_exps` (Ones/Zeros tensors
    /// consume nothing). token_embd 32·320 + output 32·320; per block, attn
    /// = 32·32 + 32·16 + 32·16 + 32·32 (k/v use n_embd_k_gqa = 8·2 = 16)
    /// and blk.0 adds ffn 3·32·32; blk.1 adds gate_inp 32·8; exp_probs_b is
    /// Zeros. 20480 + 3072 + 3072 + 3072 + 256.
    const DRAWS_BEFORE_B1_GATE: usize = 29_952;
    const PLANE_ELEMS: usize = 16 * 32;

    #[test]
    fn gguf_source_reads_expert_plane_byte_exact() {
        let src = GgufPreadSource::open(&fixture(), 1).unwrap();
        let mut buf = vec![0u8; PLANE_ELEMS * 4];

        // gate plane, expert layer 0 (= blk.1), expert 3:
        src.read_expert(0, ExpertKind::Gate, 3, &mut buf).unwrap();
        let expected = FixtureRng::new(xtask::tiny_moe::DEFAULT_SEED)
            .plane_bytes(DRAWS_BEFORE_B1_GATE + 3 * PLANE_ELEMS, PLANE_ELEMS);
        assert_eq!(buf, expected, "plane bytes must match the generator");

        // down plane (different tensor, different file math):
        src.read_expert(0, ExpertKind::Down, 5, &mut buf).unwrap();
        let expected = FixtureRng::new(xtask::tiny_moe::DEFAULT_SEED).plane_bytes(
            DRAWS_BEFORE_B1_GATE + 8 * PLANE_ELEMS + 8 * PLANE_ELEMS + 5 * PLANE_ELEMS,
            PLANE_ELEMS,
        );
        assert_eq!(buf, expected);

        // blk.1 is the ONLY expert layer; out-of-range plane errors:
        assert!(src.read_expert(1, ExpertKind::Gate, 0, &mut buf).is_err());
        assert!(src.read_expert(0, ExpertKind::Gate, 8, &mut buf).is_err());
        // wrong buffer length:
        assert!(
            src.read_expert(0, ExpertKind::Gate, 0, &mut buf[..128])
                .is_err()
        );
        // oversized buffer is fine: the first `stride` bytes carry the plane,
        // the tail is untouched (slots may exceed diverging plane strides).
        let mut big = vec![0xAAu8; PLANE_ELEMS * 4 + 64];
        src.read_expert(0, ExpertKind::Gate, 3, &mut big).unwrap();
        assert_eq!(
            &big[..PLANE_ELEMS * 4],
            &FixtureRng::new(xtask::tiny_moe::DEFAULT_SEED)
                .plane_bytes(DRAWS_BEFORE_B1_GATE + 3 * PLANE_ELEMS, PLANE_ELEMS)[..]
        );
        assert!(big[PLANE_ELEMS * 4..].iter().all(|&b| b == 0xAA));
    }

    #[test]
    fn geometry_aggregates_the_expert_plane_table() {
        let src = GgufPreadSource::open(&fixture(), 1).unwrap();
        let g = src.geometry().unwrap();
        assert_eq!(g.n_layers, 1, "tiny-moe: blk.1 is the only expert layer");
        assert_eq!(g.n_experts, 8);
        assert_eq!(g.slot_size, (PLANE_ELEMS * 4) as u64);
        assert_eq!(
            g.expert_bytes,
            3 * 8 * (PLANE_ELEMS * 4) as u64,
            "3 kinds × 8 experts × plane stride"
        );
    }

    #[test]
    fn plane_offsets_are_absolute_and_strided() {
        let src = GgufPreadSource::open(&fixture(), 1).unwrap();
        let (o0, stride) = src.plane(0, ExpertKind::Gate, 0).unwrap();
        assert_eq!(stride, (PLANE_ELEMS * 4) as u64);
        let (o3, _) = src.plane(0, ExpertKind::Gate, 3).unwrap();
        assert_eq!(o3 - o0, 3 * stride);
        assert!(src.plane(1, ExpertKind::Up, 0).is_none());
    }

    #[test]
    fn split_source_walks_all_shards_and_excludes_mtp() {
        let paths = xtask::shape_moe::cached_fixture();
        let src = GgufPreadSource::open(&paths[0], 1).unwrap();
        let g = src.geometry().unwrap();
        // blk.2 is the MTP/NextN block (nextn_predict_layers = 1): excluded.
        assert_eq!(
            g.n_layers, 1,
            "MTP expert block must not join the pool (R-2)"
        );
        assert_eq!(g.n_experts, 64);
        let (base_g, s_gate) = src.plane(0, ExpertKind::Gate, 0).unwrap();
        let (_, s_up) = src.plane(0, ExpertKind::Up, 0).unwrap();
        let (_, s_down) = src.plane(0, ExpertKind::Down, 0).unwrap();
        assert_eq!(
            (s_gate, s_up, s_down),
            (144 * 32, 176 * 32, 210 * 32),
            "mixed K-quant rows × ne1 = 32"
        );
        assert_eq!(
            g.expert_bytes,
            64 * (144 + 176 + 210) * 32,
            "3 planes × 64 experts, per-kind strides"
        );
        // Metadata-only shard 1 holds no planes; all bytes live in shard 2.
        let mut buf = vec![0u8; 144 * 32];
        src.read_expert(0, ExpertKind::Gate, 63, &mut buf).unwrap();
        assert!(buf.iter().all(|&b| b == 0xA5), "read from the data shard");
        let _ = base_g;
    }

    #[test]
    fn quant_plane_stride_is_per_plane_not_per_element() {
        // No quant fixture exists (xtask emits F32 only), so the stride
        // contract is pinned through the ggml type table the source uses:
        // Q8_0 plane = ne0/32·34 per row × ne1 rows, NOT ne0·ne1·4.
        let ne0 = 32;
        let ne1 = 16;
        let stride = ggml_row_bytes(golem_gguf::GGML_TYPE_Q8_0, ne0)
            .unwrap()
            .checked_mul(ne1)
            .unwrap();
        assert_ne!(stride, ne0 * ne1 * 4, "per-element F32 stride");
        assert_eq!(stride, 34 * 16);
    }
}
