//! Expert source seam (§5.4 byte source, v1): one GGUF replica path, `pread`
//! at per-plane file offsets derived from a tensor-table walk. Expert bytes
//! never stay in the page cache: `O_DIRECT` straight into the slot where the
//! filesystem allows it, else a buffered read + `POSIX_FADV_DONTNEED`.
//! Multi-replica striping, CRC verification, and failover belong to
//! golem-store.

// `cfg(loom)` is set via RUSTFLAGS by the loom leg; declare it known here.
#![allow(unexpected_cfgs)]

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use golem_gguf::{data_section_offset, ggml_row_bytes, tensor_entries};

use super::residency::ExpertKind;

/// One expert layer's unit (§6.1): gate, up, down planes at fixed offsets
/// inside one slot — the unit of residency is `(layer, expert)`, and one
/// fill/lease/eviction covers all three planes.
///
/// Each plane owns a contiguous slot region of `region[k]` bytes; the plane
/// sits `head[k]` bytes in, so that `O_DIRECT` can read the block-aligned
/// file range covering the plane straight into the region (§5.4: no page
/// cache, no bounce). `head` and the region padding are scratch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnitLayout {
    /// Plane byte strides, indexed by `ExpertKind::ordinal()`.
    pub stride: [u64; 3],
    /// Plane offsets inside the slot (region start + head).
    pub offset: [u64; 3],
    /// Plane offset inside its region: the plane's file misalignment.
    pub head: [u64; 3],
    /// Region lengths, in plane order; their sum is the slot size.
    pub region: [u64; 3],
}

impl UnitLayout {
    /// Lays out planes of `stride` bytes. `direct[k] = Some(h)`: plane `k`
    /// can be read with `O_DIRECT` and sits `h` bytes past a block boundary
    /// in the file, so its region is `h + stride` rounded up to
    /// [`DIRECT_ALIGN`]; `None`: densely packed (`head = 0`, region = stride).
    /// `None` on overflow.
    pub fn new(stride: [u64; 3], direct: [Option<u64>; 3]) -> Option<Self> {
        let mut offset = [0; 3];
        let mut head = [0; 3];
        let mut region = [0; 3];
        let mut at = 0u64;
        for k in 0..3 {
            head[k] = direct[k].unwrap_or(0);
            offset[k] = at.checked_add(head[k])?;
            let align = if direct[k].is_some() { DIRECT_ALIGN } else { 1 };
            region[k] = head[k]
                .checked_add(stride[k])?
                .checked_next_multiple_of(align)?;
            at = at.checked_add(region[k])?;
        }
        Some(Self {
            stride,
            offset,
            head,
            region,
        })
    }

    /// Plane bytes (Σ stride) — what a unit costs on the device.
    pub fn bytes(&self) -> u64 {
        self.stride.iter().sum()
    }

    /// Slot bytes (Σ region) — what a unit costs in the RAM pool.
    pub fn slot_bytes(&self) -> u64 {
        self.region.iter().sum()
    }

    /// `(offset, stride)` of one plane inside the slot.
    pub fn plane(&self, kind: ExpertKind) -> (u64, u64) {
        (self.offset[kind.ordinal()], self.stride[kind.ordinal()])
    }
}

/// Pool geometry derived from the expert plane table (§5.4): per-layer unit
/// layouts (dense-lead-adjusted, MTP excluded), the dense-lead-adjusted
/// layer/expert counts, and total expert bytes. RAM slot sizing is per unit
/// class (§5 pool amendment), so no single slot size here — each class's
/// slot size is its own unit bytes (see [`Self::classes`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PoolGeometry {
    /// Per-layer unit layout, indexed by dense-lead-adjusted layer.
    pub layers: Vec<UnitLayout>,
    /// Experts per layer (uniform; the loader enforces cross-tensor equality).
    pub n_experts: u32,
    /// Sum over planes of `stride × n_experts` — total pageable expert bytes.
    pub expert_bytes: u64,
}

impl PoolGeometry {
    /// Expert layer count (dense lead excluded).
    pub fn n_layers(&self) -> u32 {
        self.layers.len() as u32
    }

    /// Largest slot in bytes (the largest class's slot size).
    pub fn max_unit(&self) -> u64 {
        self.layers
            .iter()
            .map(UnitLayout::slot_bytes)
            .max()
            .unwrap_or(0)
    }

    /// Distinct slot sizes ascending, with the layers of each — the per-class
    /// slab partition (§5 pool amendment).
    pub fn classes(&self) -> Vec<(u64, Vec<u32>)> {
        let mut by_size: BTreeMap<u64, Vec<u32>> = BTreeMap::new();
        for (layer, l) in self.layers.iter().enumerate() {
            by_size
                .entry(l.slot_bytes())
                .or_default()
                .push(layer as u32);
        }
        by_size.into_iter().collect()
    }
}

/// Reads one expert plane from a byte source.
pub trait ExpertSource: Send + Sync {
    /// Read one expert plane into `into[head..head + stride]` (`into` may be
    /// longer; every byte outside that range is scratch the source may
    /// overwrite — a direct read lands the whole aligned range covering the
    /// plane). Pass `head = 0` for a plain plane buffer. Blocking; bounded
    /// internal retries on EINTR/short
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
        head: usize,
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
    /// A ggml type without a block layout in the pool's type table.
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
    /// A live expert layer lacks one of the three planes, so it cannot form a
    /// residency unit (§6.1).
    IncompleteLayer {
        /// Dense-lead-adjusted layer missing a plane.
        layer: u32,
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
            Self::IncompleteLayer { layer } => {
                write!(f, "expert layer {layer} is missing one of gate/up/down")
            }
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

impl Plane {
    /// File misalignment shared by every expert of this plane, `None` when
    /// the stride is not block-aligned (each expert's misalignment would
    /// differ, so the plane always takes the buffered path). Real targets
    /// qualify: GLM-5.3-Flash expert planes are whole multiples of 4096 B.
    fn head(&self) -> Option<u64> {
        (self.stride % DIRECT_ALIGN == 0).then_some(self.base % DIRECT_ALIGN)
    }
}

/// `O_DIRECT` alignment for offset, length and buffer address: one page,
/// a multiple of every logical block size in use (512 / 4096).
pub const DIRECT_ALIGN: u64 = 4096;

/// One open GGUF replica set; `pread`-style reads at computed plane offsets
/// (the file cursors are never shared, so concurrent reads are safe).
#[derive(Debug)]
pub struct GgufPreadSource {
    files: Vec<File>,
    /// Per-shard `O_DIRECT` handle, `None` where the filesystem refused it.
    direct: Vec<Option<File>>,
    /// Cleared on the first direct-read failure of a shard (btrfs has
    /// returned EIO under concurrent `O_DIRECT`); that shard then stays
    /// buffered + `POSIX_FADV_DONTNEED`.
    direct_ok: Vec<AtomicBool>,
    direct_reads: AtomicU64,
    planes: HashMap<(u32, ExpertKind), Plane>,
}

/// `blk.<N>.` prefix parse.
fn blk_of(name: &str) -> Option<u32> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

/// Pure geometry over a walked plane table: one `UnitLayout` per layer in
/// `ExpertKind::ALL` order, every live layer complete (§6.1). Empty table →
/// an empty geometry, which `GgufPreadSource::geometry` surfaces as `None`.
fn geometry_of(planes: &HashMap<(u32, ExpertKind), Plane>) -> Result<PoolGeometry, SourceError> {
    let n_layers = planes.keys().map(|&(l, _)| l + 1).max().unwrap_or(0);
    let n_experts = planes.values().map(|p| p.n_experts).max().unwrap_or(0);
    let mut expert_bytes = 0u64;
    let mut layers = Vec::with_capacity(n_layers as usize);
    for layer in 0..n_layers {
        let mut stride = [0u64; 3];
        for (i, s) in stride.iter_mut().enumerate() {
            let kind = ExpertKind::ALL[i];
            let p = planes
                .get(&(layer, kind))
                .ok_or(SourceError::IncompleteLayer { layer })?;
            *s = p.stride;
            expert_bytes = expert_bytes
                .checked_add(p.stride.checked_mul(u64::from(p.n_experts)).ok_or(
                    SourceError::StrideOverflow {
                        name: tensor_name(layer, kind),
                    },
                )?)
                .ok_or(SourceError::StrideOverflow {
                    name: tensor_name(layer, kind),
                })?;
        }
        let head = ExpertKind::ALL.map(|k| planes[&(layer, k)].head());
        layers.push(
            UnitLayout::new(stride, head).ok_or(SourceError::StrideOverflow {
                name: tensor_name(layer, ExpertKind::Down),
            })?,
        );
    }
    Ok(PoolGeometry {
        layers,
        n_experts,
        expert_bytes,
    })
}

/// Tensor name of one expert plane, for error reports.
fn tensor_name(layer: u32, kind: ExpertKind) -> String {
    let stem = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"][kind.ordinal()];
    format!("blk.{layer}.{stem}.weight")
}

impl GgufPreadSource {
    /// Aggregate pool geometry over the walked expert planes: `Ok(None)` when
    /// the model carries no expert tensors, an error when a live layer cannot
    /// form a unit.
    pub fn geometry(&self) -> Result<Option<PoolGeometry>, SourceError> {
        if self.planes.is_empty() {
            return Ok(None);
        }
        geometry_of(&self.planes).map(Some)
    }

    /// Opens the shard set rooted at `path` and walks every shard's GGUF
    /// tensor table (split GGUFs, where the first shard can be
    /// metadata-only). Sibling shards are discovered by rewriting the
    /// `-NNNNN-of-MMMMM.gguf` suffix per `split.count`. Expert layers are
    /// numbered EXCLUDING the dense lead AND the trailing MTP/NextN blocks
    /// (§3.4): pool layer = gguf blk − `n_dense_lead`, live only for
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
        let mtp_from = block_count
            .map(|bc| {
                // A nextn value larger than block_count is a corrupt KV, not
                // "skip everything": name it (library rule: Result, not panic).
                if nextn > bc {
                    return Err(SourceError::Gguf(golem_gguf::Error(format!(
                        "nextn_predict_layers {nextn} > block_count {bc}"
                    ))));
                }
                Ok(bc - nextn)
            })
            .transpose()?;

        // Untrusted header value: bound the up-front reservation; a bogus
        // count fails on the first missing shard instead.
        let mut files = Vec::with_capacity((split_count as usize).min(64));
        let mut direct = Vec::with_capacity(files.capacity());
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
                        continue; // MTP/NextN block: skipped at load
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
            direct.push(
                OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(&shard_path)
                    .ok(),
            );
        }
        let direct_ok = direct.iter().map(|_| AtomicBool::new(true)).collect();
        Ok(Self {
            files,
            direct,
            direct_ok,
            direct_reads: AtomicU64::new(0),
            planes,
        })
    }

    /// Plane reads that bypassed the page cache via `O_DIRECT`.
    pub fn direct_reads(&self) -> u64 {
        self.direct_reads.load(Ordering::Relaxed)
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

/// Retry bound for EINTR / short reads (bounded, then
/// surface the failure — the FFI layer aborts loudly).
const READ_RETRIES: usize = 64;

impl GgufPreadSource {
    /// Direct (`O_DIRECT`) read of the block-aligned file range covering one
    /// plane into `into`, which starts `head` bytes before the plane. `None`
    /// when the shard/plane/buffer can't go direct; `Some(Err)` on an I/O
    /// failure (the caller disables direct reads for the shard and retries
    /// buffered).
    fn read_direct(
        &self,
        p: &Plane,
        start: u64,
        into: &mut [u8],
        head: usize,
    ) -> Option<io::Result<()>> {
        let file = self.direct[p.shard].as_ref()?;
        let plane_head = p.head()?;
        if !self.direct_ok[p.shard].load(Ordering::Relaxed) || head as u64 != plane_head {
            return None;
        }
        let file_start = start - plane_head;
        let len = (plane_head + p.stride).next_multiple_of(DIRECT_ALIGN);
        let buf = into.get_mut(..usize::try_from(len).ok()?)?;
        if buf.as_ptr() as u64 % DIRECT_ALIGN != 0 || file_start % DIRECT_ALIGN != 0 {
            return None;
        }
        let want = head + p.stride as usize; // bytes that must arrive
        Some(read_full(file, buf, file_start, want))
    }
}

/// Positioned read of `buf` at `off` until at least `want` bytes arrived
/// (a direct read may stop at EOF inside the aligned tail). Bounded retries
/// on EINTR / short reads; 0 bytes before `want` is `UnexpectedEof`.
fn read_full(file: &File, buf: &mut [u8], mut off: u64, want: usize) -> io::Result<()> {
    let mut done = 0usize;
    for _ in 0..READ_RETRIES {
        // Safe pread wrapper (FileExt::read_at): does not move the cursor,
        // so concurrent readers cannot disturb each other.
        let n = match file.read_at(&mut buf[done..], off) {
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
        if done >= want {
            return Ok(());
        }
    }
    Err(io::Error::new(
        io::ErrorKind::Interrupted,
        format!("pread did not complete after {READ_RETRIES} retries"),
    ))
}

impl ExpertSource for GgufPreadSource {
    fn read_expert(
        &self,
        layer: u32,
        kind: ExpertKind,
        expert: u32,
        into: &mut [u8],
        head: usize,
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
        if into.len() < head.saturating_add(stride) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "buffer {} < head {head} + plane stride {stride}",
                    into.len()
                ),
            ));
        }
        let start = p.base + u64::from(expert) * p.stride;
        match self.read_direct(p, start, into, head) {
            Some(Ok(())) => {
                self.direct_reads.fetch_add(1, Ordering::Relaxed);
                return Ok(());
            }
            Some(Err(e)) if self.direct_ok[p.shard].swap(false, Ordering::Relaxed) => {
                tracing::warn!(
                    shard = p.shard,
                    error = %e,
                    "O_DIRECT expert read failed; shard falls back to buffered reads"
                );
            }
            Some(Err(_)) | None => {}
        }
        let file = &self.files[p.shard];
        read_full(file, &mut into[head..head + stride], start, stride)?;
        super::arena::drop_page_cache(file, start, stride);
        Ok(())
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
        src.read_expert(0, ExpertKind::Gate, 3, &mut buf, 0)
            .unwrap();
        let expected = FixtureRng::new(xtask::tiny_moe::DEFAULT_SEED)
            .plane_bytes(DRAWS_BEFORE_B1_GATE + 3 * PLANE_ELEMS, PLANE_ELEMS);
        assert_eq!(buf, expected, "plane bytes must match the generator");

        // down plane (different tensor, different file math):
        src.read_expert(0, ExpertKind::Down, 5, &mut buf, 0)
            .unwrap();
        let expected = FixtureRng::new(xtask::tiny_moe::DEFAULT_SEED).plane_bytes(
            DRAWS_BEFORE_B1_GATE + 8 * PLANE_ELEMS + 8 * PLANE_ELEMS + 5 * PLANE_ELEMS,
            PLANE_ELEMS,
        );
        assert_eq!(buf, expected);

        // blk.1 is the ONLY expert layer; out-of-range plane errors:
        assert!(
            src.read_expert(1, ExpertKind::Gate, 0, &mut buf, 0)
                .is_err()
        );
        assert!(
            src.read_expert(0, ExpertKind::Gate, 8, &mut buf, 0)
                .is_err()
        );
        // wrong buffer length:
        assert!(
            src.read_expert(0, ExpertKind::Gate, 0, &mut buf[..128], 0)
                .is_err()
        );
        // oversized buffer is fine: the first `stride` bytes carry the plane,
        // the tail is untouched (slots may exceed diverging plane strides).
        let mut big = vec![0xAAu8; PLANE_ELEMS * 4 + 64];
        src.read_expert(0, ExpertKind::Gate, 3, &mut big, 0)
            .unwrap();
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
        let g = src.geometry().unwrap().unwrap();
        assert_eq!(g.n_layers(), 1, "tiny-moe: blk.1 is the only expert layer");
        assert_eq!(g.n_experts, 8);
        assert_eq!(g.max_unit(), 3 * (PLANE_ELEMS * 4) as u64);
        assert_eq!(g.layers[0].offset, [0, 2048, 4096]);
        assert_eq!(
            g.expert_bytes,
            3 * 8 * (PLANE_ELEMS * 4) as u64,
            "3 kinds × 8 experts × plane stride"
        );
    }

    #[test]
    fn geometry_packs_three_planes_per_unit_and_groups_classes() {
        let paths = xtask::shape_moe::cached_fixture();
        let g = GgufPreadSource::open(&paths[0], 1)
            .unwrap()
            .geometry()
            .unwrap()
            .unwrap();
        assert_eq!(g.n_layers(), 3, "blk 1..=3 live, blk 4 MTP excluded");
        assert_eq!(g.layers[0].offset, [0, 144 * 32, 2 * 144 * 32]);
        assert_eq!(
            g.layers.iter().map(UnitLayout::bytes).collect::<Vec<_>>(),
            [14_848, 15_936, 17_984]
        );
        assert_eq!(g.max_unit(), 17_984);
        assert_eq!(
            g.classes(),
            vec![(14_848, vec![0]), (15_936, vec![1]), (17_984, vec![2])]
        );
    }

    #[test]
    fn geometry_rejects_a_layer_missing_a_plane() {
        // Unit residency needs all three planes per live layer.
        let planes = [(0u32, ExpertKind::Gate), (0, ExpertKind::Up)];
        assert!(matches!(
            geometry_of(&planes_fixture(&planes)),
            Err(SourceError::IncompleteLayer { layer: 0 })
        ));
    }

    fn planes_fixture(spec: &[(u32, ExpertKind)]) -> HashMap<(u32, ExpertKind), Plane> {
        spec.iter()
            .map(|&(layer, kind)| {
                (
                    (layer, kind),
                    Plane {
                        shard: 0,
                        base: 0,
                        stride: 128,
                        n_experts: 8,
                    },
                )
            })
            .collect()
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
        let g = src.geometry().unwrap().unwrap();
        // blk.4 is the MTP/NextN block (nextn_predict_layers = 1): excluded.
        assert_eq!(g.n_layers(), 3, "MTP expert block must not join the pool");
        assert_eq!(g.n_experts, 64);
        let (_, s_gate) = src.plane(0, ExpertKind::Gate, 0).unwrap();
        let (_, s_up) = src.plane(0, ExpertKind::Up, 0).unwrap();
        let (_, s_down) = src.plane(0, ExpertKind::Down, 0).unwrap();
        assert_eq!(
            (s_gate, s_up, s_down),
            (144 * 32, 144 * 32, 176 * 32),
            "blk 1 unit class: Q4/Q4/Q5 rows × ne1 = 32"
        );
        assert_eq!(
            g.expert_bytes,
            64 * (14_848 + 15_936 + 17_984),
            "3 layers × 3 planes × 64 experts, per-unit bytes"
        );
        // Metadata-only shard 1 holds no planes; all bytes live in shard 2.
        let mut buf = vec![0u8; 144 * 32];
        src.read_expert(0, ExpertKind::Gate, 63, &mut buf, 0)
            .unwrap();
        assert!(
            buf.iter().all(|&b| b == xtask::shape_moe::fill_byte(1, 0)),
            "read from the data shard, gate provenance byte"
        );
    }

    /// Page-multiple F16 planes (2048·1 elems × 2 B = 4096 B) at an
    /// unaligned file offset: the direct path must land each plane at
    /// `head` inside its page-rounded region, byte-exact, and leave the
    /// buffered fallback byte-identical. `.test-cache` is on the repo
    /// filesystem; where it refuses O_DIRECT the test fails loudly rather
    /// than passing on the buffered path (evidence rule).
    #[test]
    #[cfg(not(loom))]
    fn direct_read_lands_plane_at_head_byte_exact() {
        use xtask::gguf::{GgufKv, GgufTensorMeta, GgufValue, write_gguf};
        const NE0: u64 = 2048;
        const N_EXP: u64 = 4;
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../.test-cache/direct-align.gguf");
        let byte = |k: usize, e: u64| (0x10 * (k as u64 + 1) + e) as u8;
        let tensors: Vec<_> = ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
            .iter()
            .enumerate()
            .map(|(k, stem)| {
                let data = (0..N_EXP)
                    .flat_map(|e| std::iter::repeat_n(byte(k, e), (NE0 * 2) as usize))
                    .collect();
                (
                    GgufTensorMeta {
                        name: format!("blk.0.{stem}.weight"),
                        dims: vec![NE0, 1, N_EXP],
                        ggml_type: golem_gguf::GGML_TYPE_F16,
                    },
                    data,
                )
            })
            .collect();
        let kvs = [GgufKv {
            key: "general.architecture".into(),
            value: GgufValue::Str("glm4moe".into()),
        }];
        write_gguf(&mut File::create(&path).unwrap(), &kvs, &tensors).unwrap();

        let src = GgufPreadSource::open(&path, 0).unwrap();
        assert!(src.direct[0].is_some(), "repo filesystem refused O_DIRECT");
        let g = src.geometry().unwrap().unwrap();
        let unit = g.layers[0];
        assert_ne!(unit.head[0], 0, "fixture must exercise an unaligned plane");
        assert!(unit.region.iter().all(|r| r % DIRECT_ALIGN == 0));

        // Production stack: page-aligned mlocked arena + pager page-in.
        use crate::pool::{Arena, ExpertPool, P3Stats, PolicySnapshot, SlotClass, SnapshotHandle};
        use std::sync::Arc;
        let src = Arc::new(src);
        let arena = Arena::new(&[SlotClass {
            slot_size: unit.slot_bytes() as usize,
            slots: 2,
        }])
        .unwrap();
        let pool = ExpertPool::new(
            arena,
            src.clone(),
            Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::test_empty()))),
            Arc::new(P3Stats::new()),
            g.clone(),
            vec![0],
        );
        for e in [0u32, 3] {
            for kind in ExpertKind::ALL {
                let k = kind.ordinal();
                let want = byte(k, u64::from(e));
                pool.ensure_range(true, 0, kind, e, e, &mut |plane| {
                    assert_eq!(plane.len() as u64, unit.stride[k]);
                    assert!(plane.iter().all(|&b| b == want), "plane {k} expert {e}");
                })
                .unwrap();
            }
        }
        assert_eq!(src.direct_reads(), 6, "every plane went direct");

        // Buffered path (head 0, plain buffer): byte-identical plane.
        let mut plain = vec![0u8; unit.stride[1] as usize];
        src.read_expert(0, ExpertKind::Up, 2, &mut plain, 0)
            .unwrap();
        assert!(plain.iter().all(|&b| b == byte(1, 2)));
        assert_eq!(
            src.direct_reads(),
            6,
            "head mismatch takes the buffered path"
        );
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
