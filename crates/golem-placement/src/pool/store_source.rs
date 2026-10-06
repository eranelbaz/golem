//! Store-backed [`ExpertSource`] (§5.2–5.4): the manifest's replica set,
//! read through the shared [`ChunkQueue`] — one queued, per-drive drain per
//! expert unit instead of a blocking QD1 pread. Plane addressing comes from
//! the manifest's `expert_index` (`offsets_bytes` per expert), verified at
//! construction against the replica files' GGUF tensor tables (§5.3: sizes
//! re-checked on open).

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

use golem_gguf::{
    arch_kv_int, blk_of, data_section_offset, ggml_row_bytes, read_kvs, shard_path, tensor_entries,
};
use golem_store::{
    ChunkQueue, InstanceHealth, ReadJob, ReplicaRegistry, StoreError, instance_health,
    load_manifest,
};

use super::residency::ExpertKind;
use super::source::{DIRECT_ALIGN, ExpertSource, MAX_BLOCKS, SourceError, plane_index};

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
    /// Per-expert aligned covering length: `align_up(offset + stride) -
    /// align_down(offset)` — the byte count the queue requires in `dst`
    /// (§5.4 O_DIRECT rule). Precomputed at open so the hot path never
    /// builds a plan `Vec` per routed expert.
    covering: Vec<u32>,
    /// `stride × n_experts`, checked at open (bound against overflow).
    total_bytes: u64,
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
    /// Last surfaced instance health (§5.2/§11, `InstanceHealth` ordinal):
    /// every `wait` outcome (success or failure) recomputes and `warn`s only
    /// on transition.
    last_health: AtomicU8,
    /// §5.3 checksum degradation, sticky until the store is re-opened: set
    /// on any CRC mismatch (even one a surviving replica later heals) and
    /// passed into every `instance_health` recomputation, so a later failure
    /// can never recompute a state below `DegradedChecksum`.
    checksum_degraded: AtomicBool,
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

/// Upper bound on shard split count: bounded reservation, same defense as
/// the GGUF source (a bogus count fails on the first missing shard).
const MAX_SHARDS: u64 = 1 << 12;

/// One plane's resolved read plan for a batched page-in.
struct Planned {
    /// Position of the shard in the manifest (resolved at planning; no
    /// post-drain re-lookup of the plane table).
    shard: usize,
    offset: u64,
    stride: usize,
    covering: usize,
    expected: Option<u32>,
    /// `Some(i)`: plane i reads in place; `None`: scratch window.
    in_place: Option<usize>,
    logical_at: usize,
}

thread_local! {
    /// Batched page-in planning buffers (plans, scratch-plane indices,
    /// scratch covering lengths), cleared and reused per call so the §5.4
    /// hot path allocates nothing steady-state.
    static PLAN: std::cell::RefCell<(Vec<Planned>, Vec<usize>, Vec<usize>)> = const {
        std::cell::RefCell::new((Vec::new(), Vec::new(), Vec::new()))
    };
}

thread_local! {
    /// Reused page-aligned covering-region scratch for planes whose slot
    /// region cannot host the queued read themselves (misaligned tensor base
    /// or an undersized plane buffer). Over-allocated by one page and sliced
    /// at the first page-aligned address (same trick as the §5.4 open-time
    /// O_DIRECT probe): the queue checks 4 KiB dst alignment, which a plain
    /// `Vec<u8>` allocation (align 1) would silently fail. Reserved to the
    /// largest plane at `open` on the opening thread; other threads allocate
    /// on first use, then never again — steady-state page-in allocates
    /// nothing (§5.4 hot path).
    static SCRATCH: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Page-aligned window of `need` bytes inside an over-allocated scratch
/// `Vec`: resizes to `need + DIRECT_ALIGN - 1` and slices at the first
/// 4 KiB-aligned address. Recomputed per call because a growth-triggered
/// realloc may move the allocation.
fn aligned_window(scratch: &mut Vec<u8>, need: usize) -> &mut [u8] {
    let page = DIRECT_ALIGN as usize;
    let raw = need + page - 1;
    if scratch.len() < raw {
        scratch.resize(raw, 0);
    }
    let skip = (page - scratch.as_ptr() as usize % page) % page;
    &mut scratch[skip..skip + need]
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
        // tensor. The first replica's tables are authoritative; all other
        // replicas are cross-checked against them below (§5.3).
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
        // Per-shard expert-tensor signatures (tensor-table offset, ggml
        // type, dims): the replica identity each shard's replicas must match
        // byte-for-byte (§5.3). Only expert tensors participate — attention,
        // norm, and embedding tensors may legitimately differ in count and
        // are not routed through this source.
        /// One shard's expert-tensor signature: name → (offset, ggml type, dims).
        type ShardSig = HashMap<String, (u64, u32, [u64; 3])>;
        let mut sig: Vec<ShardSig> = vec![HashMap::new(); split_count as usize];
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
                if e.expert_kind().is_none() {
                    continue;
                }
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
                sig[shard_ix].insert(e.name.clone(), (e.offset, e.ggml_type, e.dims));
            }
        }
        // Cross-check every remaining replica's tensor table against the
        // first: size and layout must match, or planes would resolve to
        // different bytes depending on which replica served the read (§5.3).
        for (s, shard) in manifest
            .shards
            .iter()
            .enumerate()
            .take(split_count as usize)
        {
            for (r, rep) in shard.replicas.iter().enumerate() {
                let rp = rep.join(&shard.file);
                let first_replica = s == 0 && r == 0;
                if first_replica || rp == shard_path(&first, s as u64, split_count)? {
                    continue; // already walked above
                }
                let mut f = File::open(&rp)?;
                let mut seen = 0usize;
                for e in tensor_entries(&mut f)? {
                    if e.expert_kind().is_none() {
                        continue; // non-expert tensors are not routed here
                    }
                    match sig[s].get(&e.name) {
                        Some(&(off, ty, dims))
                            if off == e.offset && ty == e.ggml_type && dims == e.dims => {}
                        _ => {
                            return Err(SourceError::Store(StoreError::Invalid {
                                msg: format!(
                                    "replica {} tensor `{}` differs from the first replica's shard-{s} tensor table (§5.3)",
                                    rp.display(),
                                    e.name
                                ),
                            }));
                        }
                    }
                    seen += 1;
                }
                if seen != sig[s].len() {
                    return Err(SourceError::Store(StoreError::Invalid {
                        msg: format!(
                            "replica {} has {seen} expert tensors, first replica's shard {s} has {} (§5.3)",
                            rp.display(),
                            sig[s].len()
                        ),
                    }));
                }
            }
        }

        let mut planes: Vec<Option<StorePlane>> = Vec::new();
        let mut crc_count = 0usize;
        let mut max_covering = 0usize;
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
            if entry.crc32c.is_some() {
                crc_count += 1;
            }
            // §5.4 covering length per expert, computed arithmetically (no
            // plan Vec, no per-expert allocation at open either).
            let mut covering = Vec::with_capacity(offsets.len());
            for &abs in &offsets {
                let start = abs - abs % DIRECT_ALIGN;
                let end = (abs + stride).div_ceil(DIRECT_ALIGN) * DIRECT_ALIGN;
                let cov =
                    usize::try_from(end - start).map_err(|_| SourceError::StrideOverflow {
                        name: entry.tensor.clone(),
                    })?;
                max_covering = max_covering.max(cov);
                covering.push(u32::try_from(cov).map_err(|_| SourceError::StrideOverflow {
                    name: entry.tensor.clone(),
                })?);
            }
            let total_bytes = stride.checked_mul(u64::from(n_experts)).ok_or_else(|| {
                SourceError::StrideOverflow {
                    name: entry.tensor.clone(),
                }
            })?;
            let ix = plane_index(layer, kind);
            if planes.len() <= ix {
                planes.resize_with(ix + 1, || None);
            }
            planes[ix] = Some(StorePlane {
                shard,
                offsets,
                stride,
                n_experts,
                covering,
                total_bytes,
                crc: entry.crc32c.clone(),
            });
        }
        // Reserve the aligned scratch to the largest covering region up
        // front (plus page slack) so the opening thread never reallocs.
        SCRATCH.with(|s| {
            s.borrow_mut()
                .reserve(max_covering + DIRECT_ALIGN as usize - 1)
        });
        let crc_enabled = crc_count > 0;
        if !crc_enabled {
            // Once at startup, not once per read (§5.3 D-4 opt-out).
            tracing::info!(
                manifest = %manifest_path.display(),
                "manifest carries no crc32c digests; plane CRC verification is skipped"
            );
        } else if crc_count < index.len() {
            // Partially digested manifest: only some planes are verified.
            tracing::warn!(
                manifest = %manifest_path.display(),
                "manifest carries crc32c for {crc_count} of {} planes; the rest skip verification",
                index.len()
            );
        }
        Ok(Self {
            queue,
            registry,
            planes,
            crc_enabled,
            last_health: AtomicU8::new(InstanceHealth::Healthy as u8),
            checksum_degraded: AtomicBool::new(false),
        })
    }

    /// Whether §5.3 CRC verification is active.
    pub fn crc_enabled(&self) -> bool {
        self.crc_enabled
    }

    /// Takes (and resets) the shared queue's cumulative disk time (ns) and
    /// byte counters (§11 per-step timing snapshots).
    pub fn take_disk_stats(&self) -> (u64, u64) {
        self.queue.take_disk_stats()
    }

    /// Total pageable expert bytes (Σ `stride × n_experts` over live planes;
    /// each product was checked at open).
    pub fn expert_bytes(&self) -> u64 {
        self.planes.iter().flatten().map(|p| p.total_bytes).sum()
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
        // inside the drain, never a blocking QD1 read. The covering length
        // was precomputed at open (`StorePlane::covering`); the plan `Vec`
        // that `ChunkQueue::enqueue` builds internally remains an accepted
        // per-read cost of the shared queue, not paid here.
        let covering = p.covering[expert as usize] as usize;
        let logical_at = (offset % DIRECT_ALIGN) as usize;
        // Fast path: the plane buffer can host the aligned covering read
        // itself (aligned base + head scratch room) — bytes land in place,
        // no copy.
        if head == logical_at
            && into.len() >= covering
            && into.as_ptr() as usize % DIRECT_ALIGN as usize == 0
        {
            return self.drain(p.shard, offset, stride, expected, &mut into[..covering]);
        }
        SCRATCH.with(|s| -> io::Result<()> {
            let mut scratch = s.borrow_mut();
            let buf = aligned_window(&mut scratch, covering);
            self.drain(p.shard, offset, stride, expected, buf)?;
            into[head..head + stride].copy_from_slice(&buf[logical_at..logical_at + stride]);
            Ok(())
        })
    }

    /// One queued drain for the whole unit (§5.2): every plane is resolved
    /// and validated up front, all covering reads are enqueued as one batch,
    /// and a single [`ReadBatch::wait`] drains them — per-drive striping,
    /// replica failover, and per-plane CRC verification happen inside that
    /// one drain. Planes that cannot host their aligned covering read in
    /// place (misaligned tensor base or undersized region) share the
    /// thread-local page-aligned scratch, split into per-plane windows; the
    /// in-place rule is unchanged from [`Self::read_expert`].
    fn read_expert_batched(
        &self,
        layer: u32,
        batch: &[(ExpertKind, u32)],
        targets: &mut [(&mut [u8], usize)],
    ) -> io::Result<()> {
        if batch.len() != targets.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("batch {} != targets {}", batch.len(), targets.len()),
            ));
        }
        // Resolve + validate every plane before enqueueing anything: a bad
        // plane must fail the unit before any bytes move (§3.4).
        let page = DIRECT_ALIGN as usize;
        PLAN.with(|pb| -> io::Result<()> {
            let (plans, scratch_planes, scratch_covs) = &mut *pb.borrow_mut();
            plans.clear();
            scratch_planes.clear();
            scratch_covs.clear();
            for (i, &(kind, expert)) in batch.iter().enumerate() {
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
                let (into, head) = &targets[i];
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
                let covering = p.covering[expert as usize] as usize;
                let logical_at = (offset % DIRECT_ALIGN) as usize;
                let in_place = (*head == logical_at
                    && into.len() >= covering
                    && into.as_ptr() as usize % page == 0)
                    .then_some(i);
                if in_place.is_none() {
                    scratch_planes.push(i);
                }
                plans.push(Planned {
                    shard: p.shard,
                    offset,
                    stride,
                    covering,
                    expected: p.crc.as_ref().map(|c| c[expert as usize]),
                    in_place,
                    logical_at,
                });
            }
            scratch_covs.extend(scratch_planes.iter().map(|&i| plans[i].covering));

            SCRATCH.with(|s| -> io::Result<()> {
                let mut scratch = s.borrow_mut();
                if !scratch_planes.is_empty() {
                    let need: usize = scratch_covs.iter().map(|c| c.div_ceil(page) * page).sum();
                    let raw = need + page - 1;
                    if scratch.len() < raw {
                        scratch.resize(raw, 0);
                    }
                }

                // Enqueue all planes first, drain once (§5.2). Scratch planes
                // split page-aligned windows off the scratch in plan order —
                // each window is rounded to a page multiple so the next stays
                // aligned — with no per-call window allocation.
                let degraded_before = self.queue.degraded_reads();
                let mut rb = self.queue.batch();
                let mut rest: &mut [u8] = scratch.as_mut_slice();
                let mut si = 0usize;
                for ((into, _), plan) in targets.iter_mut().zip(plans.iter()) {
                    let dst: &mut [u8] = match plan.in_place {
                        Some(_) => &mut into[..plan.covering],
                        None => {
                            let cov = scratch_covs[si];
                            si += 1;
                            let rounded = cov.div_ceil(page) * page;
                            let skip = (page - rest.as_ptr() as usize % page) % page;
                            let (head, tail) = rest.split_at_mut(skip + rounded);
                            rest = tail;
                            &mut head[skip..skip + cov]
                        }
                    };
                    rb.push(ReadJob {
                        shard_index: plan.shard,
                        offset: plan.offset,
                        len: plan.stride,
                        expected_crc: plan.expected.unwrap_or(0),
                        verify_crc: plan.expected.is_some(),
                        prefetch: false,
                        sources: &self.registry,
                        dst,
                    });
                }
                let result = rb.wait();
                self.note_wait_outcome(degraded_before, &result);
                result.map_err(|e| match e {
                    StoreError::Io { source, .. } => source,
                    other @ StoreError::Unavailable { .. } => {
                        io::Error::new(io::ErrorKind::NotFound, other.to_string())
                    }
                    other @ StoreError::Crc { .. } => {
                        io::Error::new(io::ErrorKind::InvalidData, other.to_string())
                    }
                    other => io::Error::new(io::ErrorKind::InvalidInput, other.to_string()),
                })?;

                // Scratch-served planes land in their window: copy the logical
                // row out (the only second copy, same rule as read_expert). The
                // drain did not move the scratch; re-walk it from the start.
                let mut rest: &mut [u8] = scratch.as_mut_slice();
                let mut si = 0usize;
                for ((into, head), plan) in targets.iter_mut().zip(plans.iter()) {
                    if plan.in_place.is_some() {
                        continue;
                    }
                    let cov = scratch_covs[si];
                    si += 1;
                    let rounded = cov.div_ceil(page) * page;
                    let skip = (page - rest.as_ptr() as usize % page) % page;
                    let (head_win, tail) = rest.split_at_mut(skip + rounded);
                    rest = tail;
                    let w = &head_win[skip..skip + cov];
                    into[*head..*head + plan.stride]
                        .copy_from_slice(&w[plan.logical_at..plan.logical_at + plan.stride]);
                }
                Ok(())
            })
        })
    }

    fn plane(&self, layer: u32, kind: ExpertKind, expert: u32) -> Option<(u64, u64)> {
        let p = self.planes.get(plane_index(layer, kind))?.as_ref()?;
        Some((*p.offsets.get(expert as usize)?, p.stride))
    }
}

impl StoreExpertSource {
    /// Recomputes instance health (§5.2/§11) and warns on transition. The
    /// §5.3 flag is sticky (`checksum_degraded`), so once a CRC mismatch has
    /// been recorded no later outcome can recompute a state below
    /// `DegradedChecksum`. Cheap: shard-wide atomic health reads, no I/O.
    fn note_health(&self) {
        let now = instance_health(
            &self.registry,
            self.checksum_degraded.load(Ordering::Relaxed),
        );
        let prev = self.last_health.swap(now as u8, Ordering::Relaxed);
        // Relaxed is sufficient: health only moves toward worse states, so a
        // lost race between concurrent waiters costs at most a duplicate
        // transition warn — accepted noise, never a state regression.
        if prev != now as u8 {
            // Invariant: `prev` was only ever written from `now as u8`, so
            // `from_u8` round-trips; the `unwrap_or` keeps a library panic
            // impossible if that invariant were ever broken.
            let from = InstanceHealth::from_u8(prev).unwrap_or(InstanceHealth::Healthy);
            tracing::warn!(
                from = ?from,
                to = ?now,
                "store instance health transition (§5.2/§11)"
            );
        }
    }

    /// Records one `wait` outcome (§5.2/§11). A CRC mismatch latches the
    /// sticky §5.3 flag; health is recomputed on failure AND whenever the
    /// queue reported replica degradation during the drain, so a
    /// degraded-with-survivor failover that succeeds still surfaces its
    /// `DegradedReplica`/`DegradedChecksum` transition.
    fn note_wait_outcome(&self, degraded_before: u64, result: &Result<(), StoreError>) {
        if matches!(result, Err(StoreError::Crc { .. })) {
            self.checksum_degraded.store(true, Ordering::Relaxed);
        }
        if result.is_err() || self.queue.degraded_reads() != degraded_before {
            self.note_health();
        }
    }

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
        let degraded_before = self.queue.degraded_reads();
        let result = self
            .queue
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
            .wait();
        self.note_wait_outcome(degraded_before, &result);
        result.map_err(|e| match e {
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
    use golem_store::ShardSources as _;
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

    /// Like `write_shard`, but writes into a replica directory and may add
    /// a non-expert tensor (`blk.1.attn_norm.weight`), which replica
    /// cross-checks must ignore (§5.3).
    fn write_replica_shard(
        dir: &Path,
        name: &str,
        shard_no: u64,
        kvs: &[xtask::gguf::GgufKv],
        extra_non_expert: bool,
    ) {
        let kind = match shard_no {
            0 => "ffn_gate_exps",
            _ => "ffn_up_exps",
        };
        let mut payload = row_bytes(shard_no as usize, 0);
        payload.extend(row_bytes(shard_no as usize, 1));
        let expert = xtask::gguf::GgufTensorMeta {
            name: format!("blk.1.{kind}.weight"),
            dims: vec![4, 1, u64::from(N_EXPERTS)],
            ggml_type: 0,
        };
        let mut tensors = vec![(expert, payload)];
        if extra_non_expert {
            tensors.push((
                xtask::gguf::GgufTensorMeta {
                    name: "blk.1.attn_norm.weight".into(),
                    dims: vec![4, 1, 1],
                    ggml_type: 0,
                },
                vec![7u8; 16],
            ));
        }
        let f = std::fs::File::create(dir.join(name)).unwrap();
        xtask::gguf::write_gguf(&mut &f, kvs, &tensors).unwrap();
        let len = dir.join(name).metadata().unwrap().len();
        std::fs::OpenOptions::new()
            .append(true)
            .open(dir.join(name))
            .unwrap()
            .write_all(&vec![0u8; (FILE_BYTES - len) as usize])
            .unwrap();
    }

    use std::io::Write as _;

    /// Two-shard fixture: blk.1 gate in shard 0, up in shard 1; dense lead 1.
    /// `crc`: `None` — no digests; `Some(true)` — correct digests;
    /// `Some(false)` — deliberately wrong digests (mismatch injection).
    /// Returns (manifest path, shard-0 data-section offset).
    fn fixture(tag: &str, crc: Option<bool>) -> (PathBuf, u64, u64) {
        fixture_in(tag, [crc, crc], 1)
    }

    /// `crc` per expert_index entry (gate/shard0, up/shard1); `reps`:
    /// replica dirs per shard (each written identically) so tests can
    /// exercise failover-with-survivor.
    fn fixture_in(tag: &str, crc: [Option<bool>; 2], reps: usize) -> (PathBuf, u64, u64) {
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
        // Additional replicas get identical copies in their own dirs.
        let mut rep_dirs = vec![dir.clone()];
        for r in 1..reps {
            let rd = dir.join(format!("rep{r}"));
            std::fs::create_dir_all(&rd).unwrap();
            write_shard(&rd, SHARD0, 0, &kvs0);
            write_shard(&rd, SHARD1, 1, &kvs1);
            rep_dirs.push(rd);
        }

        let crc_json = |shard: usize| -> String {
            let Some(good) = crc[shard] else {
                return String::new();
            };
            let mut c0 = crc32c::crc32c(&row_bytes(shard, 0));
            let mut c1 = crc32c::crc32c(&row_bytes(shard, 1));
            if !good {
                c0 = c0.wrapping_add(1);
                c1 = c1.wrapping_add(1);
            }
            format!(r#", "crc32c": [{c0}, {c1}]"#)
        };
        let sha = "0".repeat(64);
        let rep_list = rep_dirs
            .iter()
            .map(|d| format!("\"{}\"", d.display()))
            .collect::<Vec<_>>()
            .join(", ");
        let json = format!(
            r#"{{
                "name": "{tag}",
                "shards": [
                    {{"file": "{SHARD0}", "bytes": {FILE_BYTES}, "sha256": "{sha}", "replicas": [{rep_list}]}},
                    {{"file": "{SHARD1}", "bytes": {FILE_BYTES}, "sha256": "{sha}", "replicas": [{rep_list}]}}
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
        let (mpath, d0, d1) = fixture("map", Some(true));
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
    fn read_expert_batched_drains_planes_in_one_batch() {
        let (mpath, _, _) = fixture("batched", Some(true));
        let src = StoreExpertSource::open(&mpath, Arc::new(ChunkQueue::new(1))).unwrap();
        let batch = [(ExpertKind::Gate, 1u32), (ExpertKind::Up, 0u32)];
        let shard_of = |k: ExpertKind| if k == ExpertKind::Gate { 0 } else { 1 };

        // Page-aligned covering region for one plane (§5.4): the raw buffer,
        // the aligned-window skip, the covering length, the head.
        fn aligned_target(
            src: &StoreExpertSource,
            kind: ExpertKind,
            expert: u32,
        ) -> (Vec<u8>, usize, usize, usize) {
            let (off, stride) = src.plane(0, kind, expert).unwrap();
            let start = off - off % DIRECT_ALIGN;
            let end = (off + stride).div_ceil(DIRECT_ALIGN) * DIRECT_ALIGN;
            let covering = (end - start) as usize;
            let head = (off % DIRECT_ALIGN) as usize;
            let raw = vec![0u8; covering + DIRECT_ALIGN as usize];
            let skip = DIRECT_ALIGN as usize - raw.as_ptr() as usize % DIRECT_ALIGN as usize;
            (raw, skip, covering, head)
        }

        // Fast path: both planes land in place, one batched drain.
        let (mut raw0, skip0, cov0, head0) = aligned_target(&src, ExpertKind::Gate, 1);
        let (mut raw1, skip1, cov1, head1) = aligned_target(&src, ExpertKind::Up, 0);
        {
            let mut targets = [
                (&mut raw0[skip0..skip0 + cov0], head0),
                (&mut raw1[skip1..skip1 + cov1], head1),
            ];
            src.read_expert_batched(0, &batch, &mut targets).unwrap();
        }
        assert_eq!(
            &raw0[skip0 + head0..skip0 + head0 + STRIDE as usize],
            &row_bytes(shard_of(ExpertKind::Gate), 1)[..]
        );
        assert_eq!(
            &raw1[skip1 + head1..skip1 + head1 + STRIDE as usize],
            &row_bytes(shard_of(ExpertKind::Up), 0)[..]
        );

        // Scratch path: head 0 differs from the planes' logical offsets and
        // the buffers are undersized — bytes still land at head after the
        // shared-scratch copy (§5.4).
        let mut g = vec![0u8; STRIDE as usize];
        let mut u = vec![0u8; STRIDE as usize];
        {
            let mut targets = [(&mut g[..], 0usize), (&mut u[..], 0)];
            src.read_expert_batched(0, &batch, &mut targets).unwrap();
        }
        assert_eq!(g, row_bytes(shard_of(ExpertKind::Gate), 1));
        assert_eq!(u, row_bytes(shard_of(ExpertKind::Up), 0));
    }

    /// Cross-shard batch where one shard's replicas are all dead: the whole
    /// batched read fails (Unavailable from the dead shard) without hanging.
    #[test]
    fn read_expert_batched_fails_when_a_shards_replicas_die() {
        let (mpath, _, _) = fixture("batched-dead", Some(true));
        let src = StoreExpertSource::open(&mpath, Arc::new(ChunkQueue::new(1))).unwrap();
        // Kill shard 1's replicas (the up plane); gate (shard 0) is healthy.
        let replica_count = src.registry().slots(1).len();
        assert!(replica_count > 0);
        for r in 0..replica_count {
            src.registry().slot(1, r).mark_degraded();
        }
        let batch = [(ExpertKind::Gate, 1u32), (ExpertKind::Up, 0u32)];
        let mut g = vec![0u8; STRIDE as usize];
        let mut u = vec![0u8; STRIDE as usize];
        let mut targets = [(&mut g[..], 0usize), (&mut u[..], 0)];
        let err = src
            .read_expert_batched(0, &batch, &mut targets)
            .expect_err("dead shard fails the batched read");
        assert_eq!(err.kind(), io::ErrorKind::NotFound, "{err}");
    }

    #[test]
    fn absent_crc32c_skips_verification_and_logs_once() {
        let (mpath, _, _) = fixture("nocrc", None);
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

        // The flag is the observable hook for the once-at-open skip log:
        // absent digests → off, present digests → on.
        let (mpath, _, _) = fixture("withcrc", Some(true));
        let src = StoreExpertSource::open(&mpath, Arc::new(ChunkQueue::new(1))).unwrap();
        assert!(src.crc_enabled());
    }

    #[test]
    fn fast_in_place_path_when_buffer_hosts_covering_region() {
        let (mpath, _, _) = fixture("fastpath", Some(true));
        let src = StoreExpertSource::open(&mpath, Arc::new(ChunkQueue::new(1))).unwrap();
        for (kind, shard) in [(ExpertKind::Gate, 0usize), (ExpertKind::Up, 1)] {
            for expert in 0..2u32 {
                let (off, stride) = src.plane(0, kind, expert).unwrap();
                // The queue's aligned covering region, computed arithmetically.
                let start = off - off % DIRECT_ALIGN;
                let end = (off + stride).div_ceil(DIRECT_ALIGN) * DIRECT_ALIGN;
                let covering = (end - start) as usize;
                let head = (off % DIRECT_ALIGN) as usize;
                // Aligned, page-aligned head + adequately sized dst → bytes
                // land in place. Over-allocate and slice at a page-aligned
                // address so the fast path's dst-alignment guard holds.
                let mut raw = vec![0u8; covering + DIRECT_ALIGN as usize];
                let skip = DIRECT_ALIGN as usize - raw.as_ptr() as usize % DIRECT_ALIGN as usize;
                let buf = &mut raw[skip..skip + covering];
                src.read_expert(0, kind, expert, buf, head).unwrap();
                assert_eq!(
                    &buf[head..head + stride as usize],
                    &row_bytes(shard, expert)[..]
                );
            }
        }
    }

    #[test]
    fn replica_with_extra_non_expert_tensor_opens() {
        // Real GGUFs carry attention/norm/embedding tensors in every shard;
        // a genuine second replica differs there and must still open.
        let (mpath, _, _) = fixture("rep-nonexp", None);
        let dir = mpath.parent().unwrap();
        let rep2 = dir.join("replica2");
        std::fs::create_dir_all(&rep2).unwrap();
        let kvs0 = vec![xtask::gguf::GgufKv {
            key: "split.count".into(),
            value: xtask::gguf::GgufValue::U32(2),
        }];
        write_replica_shard(&rep2, SHARD0, 0, &kvs0, true);
        write_replica_shard(&rep2, SHARD1, 1, &kvs0, true);
        let json = std::fs::read_to_string(&mpath).unwrap().replace(
            &format!(r#"replicas": ["{}"]"#, dir.display()),
            &format!(r#"replicas": ["{}", "{}"]"#, dir.display(), rep2.display()),
        );
        std::fs::write(&mpath, json).unwrap();
        StoreExpertSource::open(&mpath, Arc::new(ChunkQueue::new(1))).unwrap();
    }

    #[test]
    fn wrong_shard_tensor_in_replica_fails_open() {
        // Shard 1's expert tensor must match shard 1's signature: a replica
        // carrying the gate tensor (shard 0's) in shard 1 is rejected.
        let (mpath, _, _) = fixture("rep-wrongshard", None);
        let dir = mpath.parent().unwrap();
        let rep2 = dir.join("replica2");
        std::fs::create_dir_all(&rep2).unwrap();
        let kvs0 = vec![xtask::gguf::GgufKv {
            key: "split.count".into(),
            value: xtask::gguf::GgufValue::U32(2),
        }];
        write_replica_shard(&rep2, SHARD0, 0, &kvs0, false);
        write_replica_shard(&rep2, SHARD1, 0, &kvs0, false); // gate layout in shard 1 file
        let json = std::fs::read_to_string(&mpath).unwrap().replace(
            &format!(r#"replicas": ["{}"]"#, dir.display()),
            &format!(r#"replicas": ["{}", "{}"]"#, dir.display(), rep2.display()),
        );
        std::fs::write(&mpath, json).unwrap();
        let err = StoreExpertSource::open(&mpath, Arc::new(ChunkQueue::new(1))).unwrap_err();
        assert!(err.to_string().contains("shard-1 tensor table"), "{err}");
    }

    #[test]
    fn manifest_tensor_table_mismatch_fails_open() {
        let (mpath, _, _) = fixture("mismatch", None);
        // Lie about bytes_per_expert: 15 ≠ the tensor table's 16 B stride.
        let json = std::fs::read_to_string(&mpath)
            .unwrap()
            .replace("\"bytes_per_expert\": 16", "\"bytes_per_expert\": 15");
        std::fs::write(&mpath, json).unwrap();
        let err = StoreExpertSource::open(&mpath, Arc::new(ChunkQueue::new(1))).unwrap_err();
        assert!(err.to_string().contains("tensor table says"), "{err}");
    }

    #[test]
    fn offset_past_shard_size_fails_open() {
        let (mpath, _, _) = fixture("overrun", None);
        // Second expert's row would extend past the registry-verified shard
        // size (§5.3): offsets [0, 8192] with 16 B rows in an 8192 B file.
        let json = std::fs::read_to_string(&mpath)
            .unwrap()
            .replace("\"offsets_bytes\": [0, 16]", "\"offsets_bytes\": [0, 8192]");
        std::fs::write(&mpath, json).unwrap();
        let err = StoreExpertSource::open(&mpath, Arc::new(ChunkQueue::new(1))).unwrap_err();
        assert!(err.to_string().contains("exceeds shard size"), "{err}");
    }

    #[test]
    fn crc_mismatch_surfaces_as_invalid_data_read_error() {
        let (mpath, _, _) = fixture("badcrc", Some(false));
        let src = StoreExpertSource::open(&mpath, Arc::new(ChunkQueue::new(1))).unwrap();
        assert!(src.crc_enabled());
        let mut buf = vec![0u8; 8192];
        let err = src
            .read_expert(0, ExpertKind::Gate, 0, &mut buf, 0)
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData, "{err}");
        // §5.3/§5.2: the CRC mismatch latches the sticky flag and drives the
        // aggregate state (single replica marked degraded → FatalIo wins
        // precedence, §11).
        assert!(src.checksum_degraded.load(Ordering::Relaxed));
        assert_eq!(
            src.last_health.load(Ordering::Relaxed),
            InstanceHealth::FatalIo as u8
        );
    }

    /// Truncates one replica's shard file below the covering region so its
    /// chunk read fails at EOF while the surviving replica still serves.
    fn truncate_replica(tag: &str, shard_file: &str) {
        let dir =
            std::env::temp_dir().join(format!("golem-store-src-{tag}-{}", std::process::id()));
        let f = std::fs::OpenOptions::new()
            .write(true)
            .open(dir.join(shard_file))
            .unwrap();
        f.set_len(1024).unwrap();
    }

    #[test]
    fn degraded_with_survivor_success_drives_degraded_replica() {
        // Failover success must still surface the health transition: the
        // failed chunk marks its replica Degraded mid-drain (§5.2) even
        // though the read completes via the survivor.
        let tag = "failover";
        let (mpath, _, _) = fixture_in(tag, [Some(true), Some(true)], 2);
        let src = StoreExpertSource::open(&mpath, Arc::new(ChunkQueue::new(1))).unwrap();
        truncate_replica(tag, SHARD0);
        let mut buf = vec![0u8; STRIDE as usize];
        src.read_expert(0, ExpertKind::Gate, 0, &mut buf, 0)
            .unwrap();
        assert!(!src.checksum_degraded.load(Ordering::Relaxed));
        assert_eq!(
            src.last_health.load(Ordering::Relaxed),
            InstanceHealth::DegradedReplica as u8
        );
    }

    #[test]
    fn checksum_flag_is_sticky_across_later_non_crc_outcomes() {
        // §5.3: degradation persists until re-open. After a CRC failure on
        // shard 0, a later degraded-with-survivor success on shard 1 (CRC
        // verified there and passing) must NOT recompute a state below
        // DegradedChecksum — the sticky flag feeds every recomputation.
        let tag = "sticky";
        let (mpath, _, _) = fixture_in(tag, [Some(false), Some(true)], 2);
        let src = StoreExpertSource::open(&mpath, Arc::new(ChunkQueue::new(1))).unwrap();
        let mut buf = vec![0u8; STRIDE as usize];
        let err = src
            .read_expert(0, ExpertKind::Gate, 0, &mut buf, 0)
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData, "{err}");
        assert!(src.checksum_degraded.load(Ordering::Relaxed));
        assert_eq!(
            src.last_health.load(Ordering::Relaxed),
            InstanceHealth::DegradedChecksum as u8
        );
        // Non-CRC failover on shard 1 (no digest → no verification):
        // succeeds, marks one replica Degraded, state stays DegradedChecksum.
        truncate_replica(tag, SHARD1);
        src.read_expert(0, ExpertKind::Up, 0, &mut buf, 0).unwrap();
        assert!(src.checksum_degraded.load(Ordering::Relaxed));
        assert_eq!(
            src.last_health.load(Ordering::Relaxed),
            InstanceHealth::DegradedChecksum as u8
        );
    }

    #[test]
    fn read_expert_returns_io_error_when_all_replicas_unavailable() {
        let (mpath, _, _) = fixture("unavail", Some(true));
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
        // Fully-dead shard → FatalIo surfaced on the failure path (§5.2).
        assert_eq!(
            src.last_health.load(Ordering::Relaxed),
            InstanceHealth::FatalIo as u8
        );
    }
}
