//! §5.2 shared chunk queue: one queue of chunk reads drained demand-first by
//! per-drive reader threads (D-12) from the queue's persistent crews
//! ([`crate::readers`]).
//!
//! Invariants:
//! - The global queue holds jobs **without** destination slices; slices live
//!   only drain-locally, carried by [`DrainJob`] and handed out as disjoint
//!   `&mut [u8]` regions via `split_at_mut`; the drain task borrowing them is
//!   lifetime-erased only inside `readers`, whose `run` returns after every
//!   worker released it (scope semantics, no raw pointer in this module).
//! - A worker that unwinds aborts its drain (`Shared::aborted`), so the other
//!   workers return instead of waiting for chunks nobody will settle.
//! - A failed chunk requeues onto the **same drain-local** work list with the
//!   failed replica's bit set in its exclude mask, so a healthy drive's thread
//!   retries it within the same `wait()`; the global queue only feeds new
//!   drains (a chunk parked there would deadlock the waiting drain).
//! - Demand chunks drain strictly before prefetch chunks.
//! - Health is read once per replica-choice decision, so a concurrent
//!   `mark_degraded` cannot flip the predicate mid-check.

use crate::StoreError;
use crate::engine::{ChunkSource, ShardSources};
use crate::replica::ReplicaHealth;
use crate::stripe::plan_aligned_into;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

/// Everything a drain needs to settle one logical read. Recycled through
/// [`DrainScratch::states`] once its drain released every clone.
#[derive(Default)]
pub(crate) struct ReadState {
    /// Position of the shard in the manifest.
    pub shard_index: usize,
    /// Aligned chunk plan over the covering region (`plan_aligned` output).
    pub chunks: Vec<(u64, usize)>,
    /// Logical window inside the covering region: the §5.3 CRC is verified
    /// over `dst[crc_at..crc_at + crc_len]` once every chunk has landed.
    pub crc_at: usize,
    pub crc_len: usize,
    /// §5.3 expected CRC32C over the logical window (one per plane read).
    pub expected_crc: u32,
    /// Set when the §5.3 checksum must be verified; callers that have no
    /// trusted per-plane digest (manifest `crc32c` absent) disable it.
    pub verify_crc: bool,
    /// Settled chunk results plus the first terminal error.
    pub progress: Mutex<Progress>,
    /// Replicas that successfully served at least one chunk of this read —
    /// the degradation target on a plane CRC mismatch (§5.3).
    pub served: AtomicU32,
}

/// Per-read drain progress shared between the waiting thread and the drain's
/// workers.
#[derive(Default)]
pub(crate) struct Progress {
    /// Set when one chunk is irrecoverable: the read's remaining chunks fail
    /// without further reads.
    pub failed: bool,
    /// First terminal error of the read.
    pub error: Option<StoreError>,
    /// Per chunk, the §5.3 CRC32C of its slice of the logical window and
    /// that slice's length, computed by the worker that landed it (hot in
    /// cache, in parallel across the drain) and combined in `finish_read`.
    pub chunk_crcs: Vec<(u32, usize)>,
}

impl ReadState {
    /// Records a settled chunk; `Err` fails the whole read. `crc` is the
    /// landed chunk's `(index, (crc, len))` window part.
    pub fn settle(&self, result: Result<(), StoreError>, crc: Option<(usize, (u32, usize))>) {
        let mut p = self.lock_progress();
        if let Some((i, part)) = crc {
            p.chunk_crcs[i] = part;
        }
        if let Err(e) = result {
            if p.error.is_none() {
                p.error = Some(e);
            }
            p.failed = true;
        }
    }

    /// First terminal error, if the read failed.
    pub fn take_error(&self) -> Option<StoreError> {
        self.lock_progress().error.take()
    }

    /// Poisoning can only follow a worker panic; recovering the guard keeps
    /// the drain's bookkeeping consistent.
    fn lock_progress(&self) -> MutexGuard<'_, Progress> {
        self.progress.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// A chunk read waiting in the global queue. Carries no destination slice:
/// slices do not outlive their drain.
struct Job {
    read: Arc<ReadState>,
    /// Owning batch id (one per [`ReadBatch`] or [`ReadHandle`]).
    batch: u64,
    /// Index of `read` within its batch's drain-local read list.
    read_index: u32,
    chunk_index: usize,
    /// Bit i set = replica i has already failed this chunk.
    exclude_mask: u32,
}

/// Drain-local job: the chunk plus the disjoint destination sub-slice it is
/// read directly into. Never leaves the drain that created it.
struct DrainJob<'a> {
    job: Job,
    dst: &'a mut [u8],
    /// Resolved once at list-build time: index into the drain's ctx Vec.
    shard: usize,
}

#[derive(Default)]
struct QueueInner {
    /// Reads the caller is (or will be) blocked on; drained first.
    demand: VecDeque<Job>,
    /// Read-ahead; strictly behind demand.
    prefetch: VecDeque<Job>,
    /// Set at queue shutdown: queued-but-undrained reads must fail their
    /// `wait()` instead of hanging.
    shutdown: bool,
}

/// Per-shard drain context buffers, indexed by shard; refilled per drain
/// without allocating once grown.
#[derive(Default)]
struct ShardBufs {
    slots: Vec<Arc<dyn ChunkSource>>,
    replica_drive: Vec<usize>,
    drives: usize,
    /// Filled by the current drain.
    used: bool,
}

/// Reusable per-drain buffers (§5.2 hot path: zero heap allocations per
/// drain once warm). Held by one batch or handle from creation to drop, so
/// concurrent drains never share one. Vecs whose elements borrow a drain's
/// destinations are stored empty under `'static` and re-typed with
/// [`recycle`].
#[derive(Default)]
struct DrainScratch {
    reads: Vec<BatchEntry<'static>>,
    /// Unique `ReadState`s ready for the next submit.
    states: Vec<Arc<ReadState>>,
    work: VecDeque<Job>,
    base: Vec<usize>,
    slots: Vec<Option<&'static mut [u8]>>,
    list: Vec<DrainJob<'static>>,
    ctxs: Vec<Option<ShardCtx<'static>>>,
    shards: Vec<ShardBufs>,
    names: Vec<&'static str>,
}

/// Moves `v`'s allocation into a vector of the same element type under
/// other lifetimes: an in-place `collect` keeps the buffer when the layouts
/// match (best-effort std specialization; a miss only re-allocates).
pub fn recycle<T, U>(mut v: Vec<T>) -> Vec<U> {
    debug_assert!(
        size_of::<T>() == size_of::<U>() && align_of::<T>() == align_of::<U>(),
        "recycle across layouts never reuses the buffer"
    );
    v.clear();
    v.into_iter().filter_map(|_| None).collect()
}

/// Derived reader threads per drive when unset (D-12). Two: GLM-5.3-Flash
/// UD-IQ3_XXS decode at a 63 GiB pool, 128 tokens, both drives, A/B in both
/// orders: 6.55–6.65 tok/s at 1 vs 6.74–6.77 at 2, cold and warm; 3
/// regresses in `read_bench`.
pub const DEFAULT_IO_DEPTH: usize = 2;

/// The derived reader threads per drive for a configured depth: 0 →
/// [`DEFAULT_IO_DEPTH`].
pub fn resolve_io_depth(depth: usize) -> usize {
    if depth == 0 { DEFAULT_IO_DEPTH } else { depth }
}

/// One shared queue of chunk reads (§5.2).
pub struct ChunkQueue {
    inner: Mutex<QueueInner>,
    /// Reader threads per drain; 0 = derive one thread per physical drive.
    /// An explicit override wins over the derived per-drive count (D-12
    /// bounds the default at 1–2 per drive; the operator's flag does not).
    /// An explicit value ignores `io_depth` entirely.
    io_threads: usize,
    /// Derived reader threads per drive (D-12: 1–2).
    io_depth: usize,
    /// Persistent reader crews (§5.2). R-15: every crew thread pins itself
    /// to `io_cpuset` once at spawn; the serial fallback (all crews busy)
    /// runs unpinned on the caller thread.
    readers: crate::readers::ReaderPool,
    /// Cumulative disk time (ns) and bytes served by this queue's drains
    /// (§11 timing): per-chunk atomics on the slow path are acceptable; the
    /// timing consumer takes (swap-to-zero) snapshots per decode step.
    disk_ns: AtomicU64,
    disk_bytes: AtomicU64,
    /// Count of in-drain replica degradation events (chunk read failure or
    /// §5.3 CRC mismatch), §5.2. Monotonic: callers diff this across a
    /// `wait()` to notice that a drain degraded a replica even when the
    /// read itself succeeded (failover-with-survivor), so health transitions
    /// surface on success paths, not only on read errors.
    degraded_reads: AtomicU64,
    /// Home drive of the next drain's worker 0 (§5.2): rotates per drain so
    /// single-chunk drains (one plane < `CHUNK`) alternate drives.
    next_home: AtomicUsize,
    /// Next batch id: each [`ReadBatch`] and [`ReadHandle`] takes one so its
    /// wait/drop extracts exactly its own jobs in one queue pass.
    next_batch: AtomicU64,
    /// Idle drain scratches: one per concurrently live batch/handle, taken at
    /// creation and returned at drop (never locked across a drain's I/O).
    scratch: Mutex<Vec<DrainScratch>>,
}

impl ChunkQueue {
    /// New queue; `io_threads` caps the drain's reader threads (0 = derive
    /// one per physical drive hosting an eligible replica, §5.2/D-12);
    /// `io_cpuset` pins every reader thread (R-15, off the V-cache CCD).
    /// A non-zero `io_threads` ignores the per-drive `io_depth`.
    pub fn new(io_threads: usize, io_cpuset: Option<crate::sys::CpuSet>) -> Self {
        Self {
            inner: Mutex::new(QueueInner::default()),
            io_threads,
            io_depth: DEFAULT_IO_DEPTH,
            readers: crate::readers::ReaderPool::new(io_cpuset),
            disk_ns: AtomicU64::new(0),
            disk_bytes: AtomicU64::new(0),
            degraded_reads: AtomicU64::new(0),
            next_home: AtomicUsize::new(0),
            next_batch: AtomicU64::new(0),
            scratch: Mutex::new(Vec::new()),
        }
    }

    /// Derived reader threads per drive (`--io-depth`, D-12); 0 keeps
    /// [`DEFAULT_IO_DEPTH`]. An explicit `io_threads` still wins.
    #[must_use]
    pub fn with_io_depth(mut self, depth: usize) -> Self {
        self.io_depth = resolve_io_depth(depth);
        self
    }

    /// Cumulative in-drain replica degradation events (§5.2): snapshot
    /// before/after a `wait()` to detect failover on success paths.
    pub fn degraded_reads(&self) -> u64 {
        self.degraded_reads.load(Ordering::Relaxed)
    }

    /// Drains run serially on their caller because all
    /// [`crate::readers::MAX_CREWS`] reader crews were busy (§5.2).
    pub fn serial_fallbacks(&self) -> u64 {
        self.readers.serial_fallbacks()
    }

    /// Drains the accumulated disk time (ns) and byte counters, resetting
    /// both to zero (§11 per-step snapshots: the caller owns the interval).
    pub fn take_disk_stats(&self) -> (u64, u64) {
        (
            self.disk_ns.swap(0, Ordering::Relaxed),
            self.disk_bytes.swap(0, Ordering::Relaxed),
        )
    }

    /// Queues a logical read without touching its destination; call
    /// [`ReadHandle::wait`] to drain it. Prefetch reads queue strictly behind
    /// demand reads.
    ///
    /// Lifetime: the returned handle owns the read's chunks via its `Arc`
    /// (the queue itself is cloned into the handle, so handles keep it alive
    /// past engine drop). Dropping a handle without `wait`ing it purges its
    /// jobs from the global queue; a read that is `wait`ed after queue shutdown
    /// fails instead of draining. There is no queue-side reaper.
    pub fn enqueue<'a>(self: &Arc<Self>, job: ReadJob<'a>) -> ReadHandle<'a> {
        let batch = self.next_batch.fetch_add(1, Ordering::Relaxed);
        let mut scratch = self.take_scratch();
        let (read, dst, sources) = self.submit(job, batch, 0, &mut scratch.states);
        ReadHandle {
            inner: Some(HandleInner {
                // Cloned Arc: the handle keeps the queue alive past engine drop.
                queue: Arc::clone(self),
                batch,
                read,
                dst,
                sources,
                scratch,
            }),
        }
    }

    /// Starts an empty batch (§5.2): push several logical reads — typically
    /// one expert unit's planes — then drain them all in a single
    /// [`ReadBatch::wait`] instead of one blocking drain per read.
    pub fn batch(self: &Arc<Self>) -> ReadBatch<'_> {
        let mut scratch = self.take_scratch();
        ReadBatch {
            queue: Arc::clone(self),
            id: self.next_batch.fetch_add(1, Ordering::Relaxed),
            reads: recycle(std::mem::take(&mut scratch.reads)),
            scratch,
        }
    }

    fn take_scratch(&self) -> DrainScratch {
        self.scratch
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pop()
            .unwrap_or_default()
    }

    fn put_scratch(&self, mut scratch: DrainScratch) {
        // A drain that unwound skipped its cleanup: never pool stale jobs or
        // another drain's replica handles.
        scratch.work.clear();
        for b in scratch.shards.iter_mut().filter(|b| b.used) {
            b.slots.clear();
            b.used = false;
        }
        self.scratch
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(scratch);
    }

    /// Shared enqueue path: builds the read state and queues its chunks.
    /// Returns the parts the caller's handle keeps (the destination slices
    /// never enter the global queue).
    fn submit<'a>(
        self: &Arc<Self>,
        job: ReadJob<'a>,
        batch: u64,
        read_index: u32,
        states: &mut Vec<Arc<ReadState>>,
    ) -> (Arc<ReadState>, &'a mut [u8], &'a dyn ShardSources) {
        let mut state = states.pop().unwrap_or_default();
        if Arc::get_mut(&mut state).is_none() {
            state = Arc::default();
        }
        let s = Arc::get_mut(&mut state).expect("recycled or fresh ReadState is unique");
        s.shard_index = job.shard_index;
        plan_aligned_into(job.offset, job.len, &mut s.chunks);
        // The logical window inside the covering region: head scratch is
        // `align_down(offset, 4096) - offset` bytes (§5.4).
        s.crc_at = (job.offset - align_down(job.offset)) as usize;
        s.crc_len = job.len;
        s.expected_crc = job.expected_crc;
        s.verify_crc = job.verify_crc;
        s.progress.clear_poison();
        let p = s.progress.get_mut().unwrap_or_else(PoisonError::into_inner);
        p.failed = false;
        p.error = None;
        p.chunk_crcs.clear();
        p.chunk_crcs.resize(s.chunks.len(), (0, 0));
        *s.served.get_mut() = 0;
        {
            let mut q = self.lock();
            let qu = if job.prefetch {
                &mut q.prefetch
            } else {
                &mut q.demand
            };
            for chunk_index in 0..state.chunks.len() {
                qu.push_back(Job {
                    read: Arc::clone(&state),
                    batch,
                    read_index,
                    chunk_index,
                    exclude_mask: 0,
                });
            }
        }
        (state, job.dst, job.sources)
    }

    /// Fails every queued-but-undrained read at its next `wait()`; already
    /// draining reads (which own their chunks) are unaffected.
    pub fn shutdown(&self) {
        self.lock().shutdown = true;
    }

    /// Poisoning can only follow a panic; the queue's bookkeeping stays
    /// readable.
    fn lock(&self) -> MutexGuard<'_, QueueInner> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// The aligned 4 KiB base of `off` (§5.4).
fn align_down(off: u64) -> u64 {
    off - off % 4096
}

/// One queued logical read: shard, logical range, per-plane expected CRC, and
/// the destination covering region. The global queue never holds `dst` — it
/// moves into the [`ReadHandle`] and is split drain-locally.
pub struct ReadJob<'a> {
    /// Position of the shard in the manifest.
    pub shard_index: usize,
    /// Absolute byte offset of the logical read.
    pub offset: u64,
    /// Length of the logical read in bytes.
    pub len: usize,
    /// §5.3 expected CRC32C over the logical range (one per plane read).
    pub expected_crc: u32,
    /// Verify [`ReadJob::expected_crc`] once the drain lands (`false` when
    /// the manifest carries no per-plane digest).
    pub verify_crc: bool,
    /// Prefetch reads drain strictly after all queued demand reads.
    pub prefetch: bool,
    /// Replica sources of the shard.
    pub sources: &'a dyn ShardSources,
    /// Destination covering region; must be exactly `plan_aligned(offset,
    /// len)` bytes long. The logical read lands at the head-scratch offset.
    pub dst: &'a mut [u8],
}

/// Handle to a queued read; [`ReadHandle::wait`] drains exactly this read's
/// chunks (demand-first within the drain) and blocks until they settle.
pub struct ReadHandle<'a> {
    // `None` once `wait` has taken ownership; `Drop` purges any remaining
    // queued jobs so an un-waited (or early-failed) handle never leaks its
    // `Arc<ReadState>` chunks in the global queue.
    inner: Option<HandleInner<'a>>,
}

struct HandleInner<'a> {
    // Arc, not a borrow of the enqueuing engine: handles may outlive the
    // engine (drop then fails queued-but-undrained reads at their wait()).
    queue: Arc<ChunkQueue>,
    batch: u64,
    read: Arc<ReadState>,
    dst: &'a mut [u8],
    sources: &'a dyn ShardSources,
    scratch: DrainScratch,
}

/// Moves every queued job of `batch` from the global queue into `work`
/// (demand first, queue order preserved). Called once per batch by `wait`
/// and by the `Drop` impls, so a batch that is never (or only partially)
/// drained cannot leak its chunks. One lock, one pass per deque, no
/// allocation beyond `work`'s own growth.
fn purge_batch(queue: &ChunkQueue, batch: u64, work: &mut VecDeque<Job>) {
    let mut q = queue.lock();
    let QueueInner {
        demand, prefetch, ..
    } = &mut *q;
    for dq in [demand, prefetch] {
        if dq.iter().all(|j| j.batch == batch) {
            work.append(dq);
            continue;
        }
        // Rotate once over the original length: other batches' jobs return
        // to the back in their original relative order.
        for _ in 0..dq.len() {
            let j = dq.pop_front().expect("rotation stays within len");
            if j.batch == batch {
                work.push_back(j);
            } else {
                dq.push_back(j);
            }
        }
    }
}

/// Discards every queued job of `batch` in place (cancel path: nothing to drain).
fn drop_batch(queue: &ChunkQueue, batch: u64) {
    let mut q = queue.lock();
    q.demand.retain(|j| j.batch != batch);
    q.prefetch.retain(|j| j.batch != batch);
}

impl Drop for ReadHandle<'_> {
    fn drop(&mut self) {
        if let Some(mut inner) = self.inner.take() {
            drop_batch(&inner.queue, inner.batch);
            inner.scratch.states.push(inner.read);
            inner.queue.put_scratch(inner.scratch);
        }
    }
}

impl ReadHandle<'_> {
    /// Drains this read's chunks to completion: collects its eligible chunks
    /// (demand first), splits the destination into disjoint sub-slices, and
    /// runs the drain's reader threads, each looping on the shared drain-local
    /// work list. A chunk whose read fails is requeued onto that same list
    /// with the failed replica excluded, so a healthy drive's thread retries
    /// it within this call.
    ///
    /// Fails with [`StoreError::Unavailable`] when a chunk's exclude mask
    /// covers every healthy replica of the shard (immediately when no
    /// eligible reader exists at all — a 1-replica failure never hangs), and
    /// with [`StoreError::Crc`] on a §5.3 checksum mismatch.
    pub fn wait(mut self) -> Result<(), StoreError> {
        let HandleInner {
            queue,
            batch,
            read,
            dst,
            sources,
            mut scratch,
        } = self.inner.take().expect("handle consumed at most once");
        let mut entries = [BatchEntry { read, dst, sources }];
        let result = drain_batch(&queue, batch, &mut entries, &mut scratch);
        let [BatchEntry { read, .. }] = entries;
        scratch.states.push(read);
        queue.put_scratch(scratch);
        result
    }
}

/// One drain over `entries` (§5.2): takes the batch's queued chunks, checks
/// every read, resolves the per-shard contexts, runs the reader threads and
/// settles every read. Every buffer comes from `scratch`.
fn drain_batch(
    queue: &ChunkQueue,
    batch: u64,
    entries: &mut [BatchEntry<'_>],
    scratch: &mut DrainScratch,
) -> Result<(), StoreError> {
    let DrainScratch {
        work,
        base,
        slots,
        list,
        ctxs,
        shards,
        names,
        ..
    } = scratch;
    // Take every read's queued chunks out of the global queue up front
    // (demand first) so all early returns below — and the Drop fallback —
    // leave no jobs behind. A failed chunk can then requeue drain-locally
    // instead of deadlocking on a queue nobody else drains.
    purge_batch(queue, batch, work);
    let mut names_v: Vec<&str> = recycle(std::mem::take(names));
    let mut ctxs_v: Vec<Option<ShardCtx<'_>>> = recycle(std::mem::take(ctxs));
    let result = 'run: {
        if queue.lock().shutdown {
            // Queue closed before this batch drained: fail, never hang (its
            // jobs were already purged above).
            let e = &entries[0];
            break 'run Err(StoreError::Unavailable {
                shard: e.sources.shard_name(e.read.shard_index).to_owned(),
            });
        }
        for e in entries.iter() {
            let (read, sources) = (&e.read, e.sources);
            let shard = read.shard_index;
            let replica_count = sources.replica_count(shard);
            // Exclude masks (and the §5.3 `served` set) are `u32` bitmasks
            // over replica indices: bound the shard rather than widen — real
            // topologies are ≤6 replicas, and a 33rd replica is a manifest
            // bug surfaced as `Err`, not a shift-overflow panic.
            if replica_count > 32 {
                break 'run Err(StoreError::Invalid {
                    msg: format!(
                        "shard {} has {replica_count} replicas; the u32 exclude mask supports at most 32",
                        sources.shard_name(shard)
                    ),
                });
            }
            // Destination contract (§5.4): exactly the aligned covering
            // region per read, chunk 0 at the region base. Wrong length is a
            // caller bug → `Err`, never a `split_at_mut` panic; also proves
            // the CRC window below is in bounds.
            let covering: usize = read.chunks.iter().map(|c| c.1).sum();
            if e.dst.len() != covering {
                break 'run Err(StoreError::Invalid {
                    msg: format!(
                        "destination is {} bytes but the aligned covering region is {covering} bytes",
                        e.dst.len(),
                    ),
                });
            }
            if shards.len() <= shard {
                shards.resize_with(shard + 1, ShardBufs::default);
            }
            if !shards[shard].used {
                shards[shard].fill(sources, shard, &mut names_v);
            }
            // Fail immediately when no eligible reader exists at all.
            if healthy_mask(sources, shard, replica_count) == 0 {
                let mut p = read.lock_progress();
                p.failed = true;
                p.error = Some(StoreError::Unavailable {
                    shard: sources.shard_name(shard).to_owned(),
                });
            }
        }
        // Per-shard drain context, built once per drain (§5.2 hot path: no
        // per-chunk replica-slot or drive lookups), indexed by shard index —
        // a flat Vec, no per-drain HashMap.
        let max_shard = entries
            .iter()
            .map(|e| e.read.shard_index)
            .max()
            .unwrap_or(0);
        ctxs_v.resize_with(max_shard + 1, || None);
        for e in entries.iter() {
            let shard = e.read.shard_index;
            if ctxs_v[shard].is_none() {
                let b = &shards[shard];
                ctxs_v[shard] = Some(ShardCtx {
                    shard,
                    slots: &b.slots,
                    replica_drive: &b.replica_drive,
                    drives: b.drives,
                    sources: e.sources,
                });
            }
        }
        if entries.iter().any(|e| !e.read.lock_progress().failed) {
            drain_reads(queue, &ctxs_v, entries, work, base, slots, list);
        }
        // Collect per read: settle EVERY read even when one already failed —
        // a CRC mismatch on read B must still mark its serving replicas
        // Degraded when read A failed with Unavailable — then return the
        // first error. (A batch-wide "stop after first failure" flag is
        // deliberately absent: it would leave B's chunks undrained and its
        // replicas un-degraded.)
        let mut first_err: Option<StoreError> = None;
        for e in entries.iter() {
            if let Err(err) = finish_read(queue, &e.read, e.dst, e.sources, e.read.shard_index) {
                first_err.get_or_insert(err);
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    };
    *ctxs = recycle(ctxs_v);
    *names = recycle(names_v);
    work.clear();
    // Release the replica handles: scratch outlives this drain.
    for b in shards.iter_mut().filter(|b| b.used) {
        b.slots.clear();
        b.used = false;
    }
    result
}

/// Post-drain collection for one read: first terminal error wins, else the
/// §5.3 CRC over the logical window; a mismatch degrades every replica that
/// served a chunk (a plane-granular CRC cannot attribute one replica, §5.3).
fn finish_read(
    queue: &ChunkQueue,
    read: &ReadState,
    dst: &[u8],
    sources: &dyn ShardSources,
    shard: usize,
) -> Result<(), StoreError> {
    if let Some(e) = read.take_error() {
        return Err(e);
    }
    if !read.verify_crc {
        return Ok(());
    }
    let replica_count = sources.replica_count(shard);
    let crc = read
        .lock_progress()
        .chunk_crcs
        .iter()
        .fold(0, |acc, &(c, len)| crc32c::crc32c_combine(acc, c, len));
    debug_assert_eq!(
        crc,
        crc32c::crc32c(&dst[read.crc_at..read.crc_at + read.crc_len])
    );
    if crc != read.expected_crc {
        // §5.3: treat as a corrupted replica, same path as EIO.
        let served = read.served.load(Ordering::Relaxed);
        for r in 0..replica_count {
            if served & (1 << r) != 0 {
                sources.slot(shard, r).mark_degraded();
            }
        }
        queue.degraded_reads.fetch_add(1, Ordering::Relaxed);
        tracing::warn!(
            shard = %sources.shard_name(shard),
            expected = read.expected_crc,
            got = crc,
            "plane crc mismatch; serving replicas marked Degraded"
        );
        return Err(StoreError::Crc {
            shard: sources.shard_name(shard).to_owned(),
        });
    }
    Ok(())
}

/// One logical read of a batch: its state plus the handle-owned parts the
/// global queue never sees.
struct BatchEntry<'a> {
    read: Arc<ReadState>,
    dst: &'a mut [u8],
    sources: &'a dyn ShardSources,
}

/// A batch of queued reads drained together (§5.2): typically one expert
/// unit's planes, enqueued first and drained once. Reads may span shards;
/// each read keeps its own [`ReadState`] (chunk plan, CRC window, progress).
/// Dropping the batch without `wait`ing purges every read's queued chunks.
pub struct ReadBatch<'a> {
    queue: Arc<ChunkQueue>,
    id: u64,
    reads: Vec<BatchEntry<'a>>,
    scratch: DrainScratch,
}

impl Drop for ReadBatch<'_> {
    fn drop(&mut self) {
        if !self.reads.is_empty() {
            drop_batch(&self.queue, self.id);
        }
        self.release_reads();
        self.queue.put_scratch(std::mem::take(&mut self.scratch));
    }
}

impl<'a> ReadBatch<'a> {
    /// Queues one more logical read into the batch (chunks enter the global
    /// queue immediately; the drain happens at [`ReadBatch::wait`]).
    pub fn push(&mut self, job: ReadJob<'a>) {
        // A batch is one layer's misses: ≤ 288 experts × 3 planes, far below u32::MAX.
        debug_assert!(self.reads.len() < u32::MAX as usize);
        let read_index = self.reads.len() as u32;
        let (read, dst, sources) =
            self.queue
                .submit(job, self.id, read_index, &mut self.scratch.states);
        self.reads.push(BatchEntry { read, dst, sources });
    }

    /// Drains every read in the batch in one pass: one drain-local work list
    /// over all reads' chunks, one set of per-drive reader threads. Each
    /// read's CRC is verified on landing; the first failing read fails the
    /// whole batch (the caller discards the unit and requeues).
    ///
    /// # Errors
    /// [`StoreError::Unavailable`] when a read's replicas are exhausted (or
    /// the queue was shut down), [`StoreError::Crc`] on a checksum mismatch.
    pub fn wait(mut self) -> Result<(), StoreError> {
        if self.reads.is_empty() {
            return Ok(());
        }
        let result = drain_batch(&self.queue, self.id, &mut self.reads, &mut self.scratch);
        // The drain purged every job: Drop has nothing left to cancel.
        self.release_reads();
        result
    }

    /// Returns the read states and the `reads` buffer to the scratch.
    fn release_reads(&mut self) {
        if self.reads.capacity() == 0 {
            return;
        }
        let mut reads = std::mem::take(&mut self.reads);
        self.scratch.states.extend(reads.drain(..).map(|e| e.read));
        self.scratch.reads = recycle(reads);
    }
}

/// Bit i set iff replica i of `shard` is Healthy (read once per decision).
fn healthy_mask(sources: &dyn ShardSources, shard: usize, replica_count: usize) -> u32 {
    (0..replica_count)
        .filter(|&r| sources.slot(shard, r).health() == ReplicaHealth::Healthy)
        .fold(0u32, |m, r| m | (1 << r))
}

/// Shared drain state: the drain-local work list (jobs carry their slices),
/// the count of unsettled chunks, and the abort flag a panicking worker sets.
struct Shared<'a> {
    list: VecDeque<DrainJob<'a>>,
    remaining: usize,
    aborted: bool,
}

/// Aborts the drain if a worker unwinds: its chunk never settles, so the
/// other workers must stop waiting on `remaining`.
struct AbortOnUnwind<'s, 'a>(&'s (Mutex<Shared<'a>>, Condvar));

impl Drop for AbortOnUnwind<'_, '_> {
    fn drop(&mut self) {
        if std::thread::panicking() {
            let (lock, cv) = self.0;
            lock.lock().unwrap_or_else(PoisonError::into_inner).aborted = true;
            cv.notify_all();
        }
    }
}

/// Per-drain context for one shard (§5.2): resolved once per drain — the hot
/// loop never touches sysfs nor allocates per chunk.
struct ShardCtx<'a> {
    /// Shard this context serves.
    shard: usize,
    /// The shard's replica slots (chunk readers).
    slots: &'a [Arc<dyn ChunkSource>],
    /// Physical-drive index per replica (drive-grouped striping, §5.2).
    replica_drive: &'a [usize],
    /// Distinct drives hosting the shard's replicas.
    drives: usize,
    /// Health marking on chunk failure / CRC degradation.
    sources: &'a dyn ShardSources,
}

impl ShardBufs {
    /// Resolves the shard's replica slots and drive map (one sysfs-free pass:
    /// device identity was cached at replica open, §5.1/§5.2).
    fn fill<'s>(&mut self, sources: &'s dyn ShardSources, shard: usize, names: &mut Vec<&'s str>) {
        self.slots.clear();
        sources.slots_into(shard, &mut self.slots);
        names.clear();
        self.replica_drive.clear();
        for r in 0..self.slots.len() {
            let name = sources.replica_drive(shard, r);
            let idx = match names.iter().position(|n| *n == name) {
                Some(i) => i,
                None => {
                    names.push(name);
                    names.len() - 1
                }
            };
            self.replica_drive.push(idx);
        }
        self.drives = names.len();
        self.used = true;
    }
}

/// Runs the drain to completion over a batch of reads: chunks from every
/// read (possibly spanning shards) share one drain-local work list and one
/// set of reader threads; each chunk's slice is the disjoint sub-slice of
/// its own read's destination. Readers come from the queue's crews.
/// `base`, `slots_buf` and `list_buf` are the drain scratch's reused Vecs.
fn drain_reads(
    queue: &ChunkQueue,
    ctxs: &[Option<ShardCtx<'_>>],
    reads: &mut [BatchEntry<'_>],
    work: &mut VecDeque<Job>,
    base: &mut Vec<usize>,
    slots_buf: &mut Vec<Option<&'static mut [u8]>>,
    list_buf: &mut Vec<DrainJob<'static>>,
) {
    // Chunk i of a read reads into the i-th disjoint sub-slice of that
    // read's destination (§5.2: disk→destination exactly once). Reborrowed
    // per call so the caller can re-read `dst` for CRC verification.
    // Flat slot list over every read's chunks; `base[i]` is read i's first
    // slot. The caller validated each `dst` covers exactly its chunk plan.
    let mut slots: Vec<Option<&mut [u8]>> = recycle(std::mem::take(slots_buf));
    base.clear();
    for e in reads.iter_mut() {
        base.push(slots.len());
        let mut rest = &mut *e.dst;
        for &(_, len) in &e.read.chunks {
            let (head, tail) = rest.split_at_mut(len);
            slots.push(Some(head));
            rest = tail;
        }
        debug_assert!(rest.is_empty());
    }
    let mut list: VecDeque<DrainJob<'_>> = recycle(std::mem::take(list_buf)).into();
    for job in work.drain(..) {
        let shard = job.read.shard_index;
        let slot = &mut slots[base[job.read_index as usize] + job.chunk_index];
        let dst = slot.take().expect("exactly one job per chunk");
        list.push_back(DrainJob { job, dst, shard });
    }
    debug_assert!(slots.iter().all(Option::is_none), "every chunk slot taken");
    *slots_buf = recycle(slots);
    let remaining = list.len();

    // Reader threads per drain (§5.2, D-12): one thread per physical drive
    // hosting an eligible replica by default — per-drive, not per-file. An
    // explicit `--io-threads` override REPLACES the derived count. The crew
    // grows to `full`; `workers` caps it at the pending chunk count (more
    // workers than chunks would idle).
    let drives = ctxs
        .iter()
        .filter_map(|c| c.as_ref())
        .map(|c| c.drives)
        .max()
        .unwrap_or(0);
    let full = if queue.io_threads > 0 {
        queue.io_threads
    } else {
        drives.max(1) * queue.io_depth
    };
    let workers = full.min(list.len().max(1));

    let first_home = queue.next_home.fetch_add(workers, Ordering::Relaxed);
    // Borrowed by the task, not Arc'd: `run` returns only after every worker
    // released it.
    let shared = (
        Mutex::new(Shared {
            list,
            remaining,
            aborted: false,
        }),
        Condvar::new(),
    );
    let task = |w: usize| {
        let shared = &shared;
        // Any unwind on this worker (pop loop or `run_chunk`) aborts the
        // drain, so the other workers stop waiting on `remaining`.
        let _abort = AbortOnUnwind(shared);
        // Home drive for striping (§5.2): workers are pinned round-robin
        // to drives, starting at the drain's rotating first home; at pop
        // time each worker self-selects the chunks its drive can serve,
        // so the faster drive consumes proportionally more chunks.
        let home = if drives == 0 {
            0
        } else {
            first_home.wrapping_add(w) % drives
        };
        loop {
            // Pop a job (any drive's thread may serve it — the slice
            // travels with the job), or exit when everything settled.
            let job = {
                let (lock, cv) = shared;
                let mut sh = lock.lock().unwrap_or_else(PoisonError::into_inner);
                loop {
                    if sh.aborted {
                        return;
                    }
                    // §5.2 self-selection: take the first job whose
                    // healthy replicas include this worker's home
                    // drive, but only if no queued job is exclusively
                    // served by another drive (that job is that
                    // drive's worker's to take — no starvation). With
                    // all-multi-eligible chunks this is the front
                    // job, so workers race on the list and the faster
                    // drive (shorter per-chunk latency) pops more.
                    // Fallback: front, so a job whose home worker is
                    // busy still drains; exhaustion semantics unchanged.
                    let home_first = sh.list.iter().position(|j| {
                        let Some(c) = &ctxs[j.shard] else {
                            return true;
                        };
                        let elig =
                            healthy_mask(c.sources, c.shard, c.slots.len()) & !j.job.exclude_mask;
                        (0..c.slots.len())
                            .any(|r| elig & (1 << r) != 0 && c.replica_drive[r] == home)
                    });
                    match home_first {
                        Some(i) => break sh.list.remove(i).unwrap(),
                        None => {
                            if let Some(j) = sh.list.pop_front() {
                                break j;
                            }
                        }
                    }
                    if sh.remaining == 0 {
                        return;
                    }
                    sh = cv.wait(sh).unwrap_or_else(PoisonError::into_inner);
                }
            };
            // Shard ctx resolved once per job at list-build time
            // (flat Vec index, §5.2 hot path).
            let ctx = ctxs[job.shard]
                .as_ref()
                .expect("ctx exists for every enqueued job's shard");
            run_chunk(queue, shared, ctx, home, job);
        }
    };
    queue.readers.run(full, workers, &task);
    let list = shared
        .0
        .into_inner()
        .unwrap_or_else(PoisonError::into_inner)
        .list;
    *list_buf = recycle(Vec::from(list));
}

/// Serves one chunk, preferring replicas on the worker's home drive (§5.2
/// striping: each worker self-selects the chunks its drive can serve; a
/// worker whose home drive is degraded falls back to any eligible replica).
/// On failure degrades the replica, requeues onto the same drain-local list
/// with the replica's bit set, and lets a healthy drive retry.
fn run_chunk<'a>(
    queue: &ChunkQueue,
    shared: &(Mutex<Shared<'a>>, Condvar),
    ctx: &ShardCtx<'_>,
    home: usize,
    mut job: DrainJob<'a>,
) {
    let read = &*job.job.read;
    let shard = ctx.shard;
    let slots = &ctx.slots;
    let sources = ctx.sources;
    let replica_drive = &ctx.replica_drive;
    let replica_count = slots.len();
    let (offset, _) = read.chunks[job.job.chunk_index];
    let mut mask = job.job.exclude_mask;
    // One decision per pop: a requeue returns and the retry happens on some
    // worker's next pop of the same drain-local list.
    {
        // Another chunk's terminal failure makes this one moot.
        if read.lock_progress().failed {
            return settle(
                shared,
                read,
                Err(StoreError::Unavailable {
                    shard: sources.shard_name(shard).to_owned(),
                }),
            );
        }
        // Health read once per decision: a concurrent mark_degraded on
        // another thread cannot flip this predicate mid-check. Home-drive
        // match first (§5.2 striping), then any eligible replica.
        let eligible = healthy_mask(sources, shard, replica_count) & !mask;
        let replica = (0..replica_count)
            .find(|&r| eligible & (1 << r) != 0 && replica_drive[r] == home)
            .or_else(|| (0..replica_count).find(|&r| eligible & (1 << r) != 0));
        let replica = match replica {
            Some(r) => r,
            None => {
                // Exclude mask covers every healthy replica: fail the read,
                // no hang (the 1-replica case lands here on requeue).
                return settle(
                    shared,
                    read,
                    Err(StoreError::Unavailable {
                        shard: sources.shard_name(shard).to_owned(),
                    }),
                );
            }
        };
        // Read directly into the destination sub-slice: expert bytes are
        // copied disk→destination exactly once (§5.2).
        let t0 = std::time::Instant::now();
        match slots[replica].read_chunk(offset, job.dst) {
            Ok(()) => {
                let ns = t0.elapsed().as_nanos() as u64;
                queue.disk_ns.fetch_add(ns, Ordering::Relaxed);
                queue
                    .disk_bytes
                    .fetch_add(job.dst.len() as u64, Ordering::Relaxed);
                read.served.fetch_or(1 << replica, Ordering::Relaxed);
                // §5.3 CRC of this chunk's slice of the logical window, on
                // the worker while the bytes are hot: a plane's hash runs
                // in parallel across the drain instead of serially after it.
                let crc = read.verify_crc.then(|| {
                    let at = (offset - read.chunks[0].0) as usize;
                    let lo = read.crc_at.max(at);
                    let hi = (read.crc_at + read.crc_len).min(at + job.dst.len());
                    let part = if lo < hi {
                        (crc32c::crc32c(&job.dst[lo - at..hi - at]), hi - lo)
                    } else {
                        (0, 0)
                    };
                    (job.job.chunk_index, part)
                });
                settle_with(shared, read, Ok(()), crc);
            }
            Err(e) => {
                sources.slot(shard, replica).mark_degraded();
                queue.degraded_reads.fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    shard = %sources.shard_name(shard),
                    replica,
                    offset,
                    len = job.dst.len(),
                    reason = %e,
                    "chunk read failed; replica marked Degraded, chunk requeued"
                );
                mask |= 1 << replica;
                job.job.exclude_mask = mask;
                // Same drain-local list — a healthy drive's thread retries
                // within this wait; the global queue is not involved.
                let (lock, cv) = shared;
                lock.lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .list
                    .push_back(job);
                cv.notify_all();
            }
        }
    }
}

/// Records a settled chunk and releases its share of the drain.
fn settle(shared: &(Mutex<Shared<'_>>, Condvar), read: &ReadState, result: Result<(), StoreError>) {
    settle_with(shared, read, result, None);
}

/// [`settle`] recording a landed chunk's CRC part under the same lock.
fn settle_with(
    shared: &(Mutex<Shared<'_>>, Condvar),
    read: &ReadState,
    result: Result<(), StoreError>,
    crc: Option<(usize, (u32, usize))>,
) {
    read.settle(result, crc);
    let (lock, cv) = shared;
    lock.lock()
        .unwrap_or_else(PoisonError::into_inner)
        .remaining -= 1;
    cv.notify_all();
}

// Every test here spawns reader crews outside `loom::model`.
#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use crate::replica::{ReplicaRegistry, ReplicaSlot};
    use crate::stripe::{CHUNK, plan_aligned};
    use std::io;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    const SHARD_BYTES: usize = 4096;
    const SHARD_FILE: &str = "s0.gguf";

    /// Deterministic fixture pattern over absolute byte index.
    fn pattern(off: usize, len: usize) -> Vec<u8> {
        (off..off + len)
            .map(|i| ((i * 7 + 3) % 251) as u8)
            .collect()
    }

    /// Fixture files over `replica_counts.len()` shards, shard `s` with
    /// `replica_counts[s]` replica dirs; returns (paths, registry).
    fn fixture_shards(replica_counts: &[usize], tag: &str) -> (Vec<PathBuf>, ReplicaRegistry) {
        let root = std::env::temp_dir().join(format!("golem-store-queue-{tag}"));
        let mut shards = Vec::new();
        let mut paths = Vec::new();
        for (s, &n) in replica_counts.iter().enumerate() {
            let mut replicas = Vec::with_capacity(n);
            for r in 0..n {
                let dir = root.join(format!("s{s}-r{r}"));
                std::fs::create_dir_all(&dir).unwrap();
                std::fs::write(dir.join(format!("s{s}.gguf")), pattern(0, SHARD_BYTES)).unwrap();
                replicas.push(dir.clone());
            }
            paths.extend(replicas.iter().cloned());
            shards.push(crate::manifest::ShardEntry {
                file: format!("s{s}.gguf"),
                bytes: SHARD_BYTES as u64,
                sha256: "0".repeat(64),
                replicas,
                ..Default::default()
            });
        }
        let manifest = crate::manifest::ModelManifest {
            name: tag.to_owned(),
            shards,
            ..Default::default()
        };
        (paths, ReplicaRegistry::open(&manifest).unwrap())
    }

    /// Fixture files over `n` replica dirs; returns (paths, registry).
    fn fixture(n: usize, tag: &str) -> (Vec<PathBuf>, ReplicaRegistry) {
        fixture_sized(n, tag, SHARD_BYTES)
    }

    /// [`fixture`] with a `bytes`-long shard.
    fn fixture_sized(n: usize, tag: &str, bytes: usize) -> (Vec<PathBuf>, ReplicaRegistry) {
        let root = std::env::temp_dir().join(format!("golem-store-queue-{tag}"));
        let mut replicas = Vec::with_capacity(n);
        for r in 0..n {
            let dir = root.join(format!("r{r}"));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(SHARD_FILE), pattern(0, bytes)).unwrap();
            replicas.push(dir);
        }
        let manifest = crate::manifest::ModelManifest {
            name: tag.to_owned(),
            shards: vec![crate::manifest::ShardEntry {
                file: SHARD_FILE.to_owned(),
                bytes: bytes as u64,
                sha256: "0".repeat(64),
                replicas: replicas.clone(),
                ..Default::default()
            }],
            ..Default::default()
        };
        (replicas, ReplicaRegistry::open(&manifest).unwrap())
    }

    /// Fails source: tracks peak concurrent `read_chunk` calls so tests can
    /// observe the actual worker count.
    #[derive(Default)]
    struct Concurrency {
        active: AtomicUsize,
        max: AtomicUsize,
    }

    struct ConcurrencyGuard<'a>(&'a Concurrency);

    impl Drop for ConcurrencyGuard<'_> {
        fn drop(&mut self) {
            self.0.active.fetch_sub(1, Ordering::SeqCst);
        }
    }

    /// Delegates to the real slot until the countdown hits 0, then fails with
    /// EIO; records every attempt (replica id, offset) in a shared log.
    /// `delay_ms` throttles each read (striping determinism).
    struct FailingSource {
        inner: Arc<ReplicaSlot>,
        fail_nth: AtomicUsize,
        id: usize,
        log: Arc<std::sync::Mutex<Vec<(usize, u64)>>>,
        delay_ms: u64,
        conc: Option<Arc<Concurrency>>,
        /// When set, each read spins (bounded) until two readers are
        /// concurrently inside `read_chunk` — a deterministic "≥2 workers
        /// ran" signal, robust to scheduler noise.
        wait_for_pair: Option<Arc<AtomicBool>>,
        gate: Option<Arc<Gate>>,
    }

    impl Concurrency {
        /// RAII enter; peak count observable via `max` after all reads settle.
        fn enter(self: &Arc<Self>) -> ConcurrencyGuard<'_> {
            let a = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max.fetch_max(a, Ordering::SeqCst);
            ConcurrencyGuard(self)
        }
    }

    impl ChunkSource for FailingSource {
        fn read_chunk(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
            let start = std::time::Instant::now();
            self.log.lock().unwrap().push((self.id, offset));
            if self.delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(self.delay_ms));
            }
            let _g = self.conc.as_ref().map(|c| c.enter());
            if let Some(flag) = &self.wait_for_pair {
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                while !flag.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
                    if let Some(c) = &self.conc
                        && c.active.load(Ordering::SeqCst) >= 2
                    {
                        flag.store(true, Ordering::SeqCst);
                        break;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
            if let Some(g) = &self.gate
                && let Some(w) = g.wait_until
            {
                while g.done.load(Ordering::SeqCst) < w {
                    assert!(
                        start.elapsed() < std::time::Duration::from_secs(30),
                        "gate deadline exceeded: fast replica stalled"
                    );
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
            }
            // usize::MAX = always fail (never decremented).
            if self.fail_nth.load(Ordering::SeqCst) == 0
                || self.fail_nth.load(Ordering::SeqCst) == usize::MAX
            {
                return Err(io::Error::from_raw_os_error(libc::EIO));
            }
            self.fail_nth.fetch_sub(1, Ordering::SeqCst);
            self.inner.read_at(offset, buf)?;
            if let Some(g) = &self.gate
                && g.bump
            {
                g.done.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        }
    }

    /// Real registry + per-replica FailingSources; health marking flows
    /// through `slot()`. `drives` overrides the cached per-replica device
    /// identity (empty = use the real one) so multi-drive striping is
    /// testable on one filesystem.
    struct TestSources {
        reg: ReplicaRegistry,
        /// Per-(shard, replica) failure budget; entry (s, r, n) fails the
        /// first n chunk reads of replica r of shard s (usize::MAX = always).
        fail: Vec<(usize, usize, usize)>,
        log: Arc<std::sync::Mutex<Vec<(usize, u64)>>>,
        drives: Vec<String>,
        /// Per-read throttle (0 = none); widens the pop window so multi-worker
        /// striping is deterministic.
        delay_ms: u64,
        /// Per-replica throttle override (index = replica, empty = use
        /// `delay_ms`); models drives of different speeds.
        delays_ms: Vec<u64>,
        /// Shared peak-concurrency tracker (None = untracked).
        conc: Option<Arc<Concurrency>>,
        /// Pair-achievement flag handed to every source (see
        /// [`FailingSource::wait_for_pair`]).
        wait_for_pair: Option<Arc<AtomicBool>>,
        /// Per-replica ordering gate (index = replica, empty = ungated);
        /// makes cross-drive interleavings deterministic (see [`Gate`]).
        gates: Vec<Option<Arc<Gate>>>,
    }

    /// Deterministic cross-replica ordering for multi-drive tests: one
    /// replica counts completed reads (`bump`), another blocks until the
    /// count reaches `wait_until`. Replaces wall-clock sleep assumptions
    /// that flake under parallel test load.
    struct Gate {
        done: Arc<AtomicUsize>,
        /// This replica increments `done` after each successful read.
        bump: bool,
        /// This replica blocks in `read_chunk` until `done` ≥ this count.
        wait_until: Option<usize>,
    }

    impl ShardSources for TestSources {
        fn slots(&self, shard_index: usize) -> Vec<Arc<dyn ChunkSource>> {
            self.reg.shards()[shard_index]
                .replicas
                .iter()
                .enumerate()
                .map(|(replica, slot)| {
                    // Every replica is logged; `fail` sets the per-(shard,
                    // replica) failure budget (default: effectively never
                    // fails).
                    let nth = self
                        .fail
                        .iter()
                        .find(|(s, r, _)| *s == shard_index && *r == replica)
                        .map(|(_, _, n)| *n)
                        .unwrap_or(1_000_000);
                    Arc::new(FailingSource {
                        inner: Arc::clone(slot),
                        fail_nth: AtomicUsize::new(nth),
                        id: replica,
                        log: Arc::clone(&self.log),
                        delay_ms: self
                            .delays_ms
                            .get(replica)
                            .copied()
                            .unwrap_or(self.delay_ms),
                        conc: self.conc.clone(),
                        wait_for_pair: self.wait_for_pair.clone(),
                        gate: self.gates.get(replica).cloned().flatten(),
                    }) as Arc<dyn ChunkSource>
                })
                .collect()
        }

        fn slot(&self, shard_index: usize, replica_index: usize) -> &ReplicaSlot {
            self.reg.slot(shard_index, replica_index)
        }

        fn shard_name(&self, shard_index: usize) -> &str {
            self.reg.shard_name(shard_index)
        }

        fn replica_drive(&self, shard_index: usize, replica_index: usize) -> &str {
            match self.drives.get(replica_index) {
                Some(d) => d,
                None => self.slot(shard_index, replica_index).drive.as_str(),
            }
        }
    }

    /// A full logical read of the fixture shard, drained by `queue`.
    fn read_whole<'a>(
        queue: &'a Arc<ChunkQueue>,
        sources: &'a TestSources,
        dst: &'a mut [u8],
    ) -> ReadHandle<'a> {
        queue.enqueue(ReadJob {
            shard_index: 0,
            offset: 0,
            len: SHARD_BYTES,
            expected_crc: crc32c::crc32c(&pattern(0, SHARD_BYTES)),
            verify_crc: true,
            prefetch: false,
            sources,
            dst,
        })
    }

    /// One batch, two reads (§5.2): both land in one wait, each into its own
    /// destination, each CRC-verified.
    #[test]
    fn batch_drains_two_reads_in_one_wait() {
        let (_paths, reg) = fixture(2, "batch");
        let sources = TestSources {
            reg,
            fail: vec![],
            log: Arc::new(std::sync::Mutex::new(Vec::new())),
            drives: Vec::new(),
            delay_ms: 0,
            delays_ms: Vec::new(),
            conc: None,
            gates: Vec::new(),
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(0, None));
        // Aligned covering regions (§5.4): both spans cover 0..4096; the
        // logical windows sit at the head-scratch offsets (0 and 1024).
        let mut d0 = vec![0u8; 4096];
        let mut d1 = vec![0u8; 4096];
        let mut batch = queue.batch();
        batch.push(ReadJob {
            shard_index: 0,
            offset: 0,
            len: 512,
            expected_crc: crc32c::crc32c(&pattern(0, 512)),
            verify_crc: true,
            prefetch: false,
            sources: &sources,
            dst: &mut d0,
        });
        batch.push(ReadJob {
            shard_index: 0,
            offset: 1024,
            len: 3072,
            expected_crc: crc32c::crc32c(&pattern(1024, 3072)),
            verify_crc: true,
            prefetch: false,
            sources: &sources,
            dst: &mut d1,
        });
        batch.wait().unwrap();
        assert_eq!(d0, pattern(0, 4096));
        assert_eq!(&d1[1024..], pattern(1024, 3072));
    }

    /// Panics on every read, or with a flag only on the first one (later
    /// reads succeed without touching `buf`).
    struct PanicSource(Option<Arc<AtomicBool>>);

    impl ChunkSource for PanicSource {
        fn read_chunk(&self, _offset: u64, _buf: &mut [u8]) -> io::Result<()> {
            if let Some(fired) = &self.0
                && fired.swap(true, Ordering::SeqCst)
            {
                return Ok(());
            }
            panic!("injected reader panic");
        }
    }

    struct PanicSources(ReplicaRegistry, Option<Arc<AtomicBool>>);

    impl ShardSources for PanicSources {
        fn slots(&self, _shard_index: usize) -> Vec<Arc<dyn ChunkSource>> {
            vec![Arc::new(PanicSource(self.1.clone()))]
        }
        fn slot(&self, shard_index: usize, replica_index: usize) -> &ReplicaSlot {
            self.0.slot(shard_index, replica_index)
        }
        fn shard_name(&self, shard_index: usize) -> &str {
            self.0.shard_name(shard_index)
        }
        fn replica_drive(&self, shard_index: usize, replica_index: usize) -> &str {
            self.0.replica_drive(shard_index, replica_index)
        }
    }

    /// A drain that unwinds returns its scratch to the pool; the next batch on
    /// that scratch must not see the dead drain's jobs or replica handles.
    #[test]
    fn batch_after_reader_panic_reuses_clean_scratch() {
        let (_p0, reg0) = fixture(1, "panic-a");
        let (_p1, reg1) = fixture(1, "panic-b");
        let panicky = PanicSources(reg0, None);
        let queue = Arc::new(ChunkQueue::new(1, None));
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut d = vec![0u8; 4096];
            let mut b = queue.batch();
            b.push(ReadJob {
                shard_index: 0,
                offset: 0,
                len: 4096,
                expected_crc: 0,
                verify_crc: false,
                prefetch: false,
                sources: &panicky,
                dst: &mut d,
            });
            let _ = b.wait();
        }));
        assert!(unwound.is_err(), "reader panic propagates to the caller");
        let mut d = vec![0u8; 4096];
        let mut b = queue.batch();
        b.push(ReadJob {
            shard_index: 0,
            offset: 0,
            len: 4096,
            expected_crc: crc32c::crc32c(&pattern(0, 4096)),
            verify_crc: true,
            prefetch: false,
            sources: &reg1,
            dst: &mut d,
        });
        b.wait().unwrap();
        assert_eq!(d, pattern(0, 4096));
    }

    /// Two workers, one panics mid-chunk: the other (whose reads succeed)
    /// must not wait forever on the never-settled chunk; the panic reaches
    /// the caller and the next
    /// drain on the same crews succeeds.
    #[test]
    fn two_worker_panic_aborts_drain_without_hang() {
        let bytes = crate::stripe::CHUNK * 4;
        let (_p0, reg0) = fixture(1, "panic2-a");
        let (_p1, reg1) = fixture(1, "panic2-b");
        // Only the first read panics: the surviving worker keeps serving
        // chunks until it observes the abort.
        let panicky = PanicSources(reg0, Some(Arc::new(AtomicBool::new(false))));
        let queue = Arc::new(ChunkQueue::new(2, None));
        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut d = vec![0u8; bytes];
            let mut b = queue.batch();
            b.push(ReadJob {
                shard_index: 0,
                offset: 0,
                len: bytes,
                expected_crc: 0,
                verify_crc: false,
                prefetch: false,
                sources: &panicky,
                dst: &mut d,
            });
            let _ = b.wait();
        }));
        assert!(unwound.is_err(), "worker panic propagates to the caller");
        assert_eq!(queue.readers.last_want(), 2);
        let mut d = vec![0u8; 4096];
        read_whole(&queue, &reg1_sources(reg1), &mut d)
            .wait()
            .unwrap();
        assert_eq!(d, pattern(0, 4096));
    }

    fn reg1_sources(reg: ReplicaRegistry) -> TestSources {
        TestSources {
            reg,
            fail: Vec::new(),
            log: Arc::new(std::sync::Mutex::new(Vec::new())),
            drives: Vec::new(),
            delay_ms: 0,
            delays_ms: Vec::new(),
            conc: None,
            wait_for_pair: None,
            gates: Vec::new(),
        }
    }

    /// Multi-chunk fixture over `drives.len()` replicas, one per fake drive.
    fn multi_drive(tag: &str, drives: &[&str], chunks: usize) -> (TestSources, usize) {
        let bytes = crate::stripe::CHUNK * chunks;
        let root = std::env::temp_dir().join(format!("golem-store-queue-{tag}"));
        let mut replicas = Vec::new();
        for r in 0..drives.len() {
            let dir = root.join(format!("r{r}"));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(SHARD_FILE), pattern(0, bytes)).unwrap();
            replicas.push(dir);
        }
        let manifest = crate::manifest::ModelManifest {
            name: tag.to_owned(),
            shards: vec![crate::manifest::ShardEntry {
                file: SHARD_FILE.to_owned(),
                bytes: bytes as u64,
                sha256: "0".repeat(64),
                replicas,
                ..Default::default()
            }],
            ..Default::default()
        };
        let mut s = reg1_sources(ReplicaRegistry::open(&manifest).unwrap());
        s.drives = drives.iter().map(|d| (*d).to_owned()).collect();
        (s, bytes)
    }

    fn drain_all(queue: &Arc<ChunkQueue>, sources: &TestSources, bytes: usize) {
        let mut dst = vec![0u8; bytes];
        queue
            .enqueue(ReadJob {
                shard_index: 0,
                offset: 0,
                len: bytes,
                expected_crc: crc32c::crc32c(&pattern(0, bytes)),
                verify_crc: true,
                prefetch: false,
                sources,
                dst: &mut dst,
            })
            .wait()
            .unwrap();
        assert_eq!(dst, pattern(0, bytes));
    }

    /// A crew first sized by a 1-drive drain grows for a later 2-drive drain
    /// (2 drives × depth 2 = 4 workers), with no hang.
    #[test]
    fn crew_grows_from_one_drive_to_two() {
        let queue = Arc::new(ChunkQueue::new(0, None));
        let (one, b1) = multi_drive("grow-1", &["dA"], 4);
        drain_all(&queue, &one, b1);
        assert_eq!(queue.readers.last_want(), 2);
        let (two, b2) = multi_drive("grow-2", &["dA", "dB"], 4);
        assert_eq!(queue.readers.max_crew_threads(), 2);
        drain_all(&queue, &two, b2);
        assert_eq!(queue.readers.last_want(), 4);
        assert_eq!(queue.readers.max_crew_threads(), 4);
        assert_eq!(queue.serial_fallbacks(), 0);
    }

    /// Drain A holds a crew (blocked in its read) while drain B (2 drives)
    /// runs on another; once both finish, a third 2-drive drain reuses A's
    /// small crew grown to 4 workers.
    #[test]
    fn held_crew_grows_on_reuse() {
        let queue = Arc::new(ChunkQueue::new(0, None));
        let (mut one, b1) = multi_drive("held-1", &["dA"], 1);
        let gate = Arc::new(Gate {
            done: Arc::new(AtomicUsize::new(0)),
            bump: false,
            wait_until: Some(1),
        });
        one.gates = vec![Some(Arc::clone(&gate))];
        let (two, b2) = multi_drive("held-2", &["dA", "dB"], 4);
        std::thread::scope(|s| {
            let a = s.spawn(|| drain_all(&queue, &one, b1));
            // A is parked in its gated read, holding one crew.
            while one.log.lock().unwrap().is_empty() {
                std::thread::yield_now();
            }
            drain_all(&queue, &two, b2);
            gate.done.store(1, Ordering::SeqCst);
            a.join().unwrap();
        });
        // Popping order is LIFO: A's crew (2 threads: 1 drive × depth 2)
        // is the top idle crew.
        assert_eq!(queue.readers.top_idle_crew_threads(), 2);
        drain_all(&queue, &two, b2);
        assert_eq!(queue.readers.last_want(), 4);
        assert_eq!(queue.readers.top_idle_crew_threads(), 4);
        assert_eq!(queue.serial_fallbacks(), 0);
    }

    /// Interleaved batches: one batch's wait extracts only its own jobs, the
    /// other's stay queued in order and drain correctly afterwards.
    #[test]
    fn interleaved_batches_drain_independently() {
        let (_paths, reg) = fixture(1, "batch-interleave");
        let sources = TestSources {
            reg,
            fail: vec![],
            log: Arc::new(std::sync::Mutex::new(Vec::new())),
            drives: Vec::new(),
            delay_ms: 0,
            delays_ms: Vec::new(),
            conc: None,
            gates: Vec::new(),
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(0, None));
        fn job<'a>(offset: u64, dst: &'a mut [u8], sources: &'a TestSources) -> ReadJob<'a> {
            ReadJob {
                shard_index: 0,
                offset,
                len: 4096,
                expected_crc: crc32c::crc32c(&pattern(offset as usize, 4096)),
                verify_crc: true,
                prefetch: false,
                sources,
                dst,
            }
        }
        let (mut a0, mut a1) = (vec![0u8; 4096], vec![0u8; 4096]);
        let (mut b0, mut b1) = (vec![0u8; 4096], vec![0u8; 4096]);
        let mut a = queue.batch();
        let mut b = queue.batch();
        a.push(job(0, &mut a0, &sources));
        b.push(job(0, &mut b0, &sources));
        a.push(job(0, &mut a1, &sources));
        b.push(job(0, &mut b1, &sources));
        assert_eq!(queued(&queue), (4, 0));
        a.wait().unwrap();
        {
            let q = queue.lock();
            let left: Vec<(u64, u32)> = q.demand.iter().map(|j| (j.batch, j.read_index)).collect();
            assert_eq!(
                left,
                vec![(b.id, 0), (b.id, 1)],
                "B's jobs stay queued in order"
            );
        }
        assert_eq!(a0, pattern(0, 4096));
        assert_eq!(a1, pattern(0, 4096));
        b.wait().unwrap();
        assert_eq!(b0, pattern(0, 4096));
        assert_eq!(b1, pattern(0, 4096));
        assert_eq!(queued(&queue), (0, 0));
    }

    /// One batched read's terminal failure fails the whole batch.
    #[test]
    fn batched_read_failure_fails_the_whole_batch() {
        let (_paths, reg) = fixture(2, "batch-fail");
        let sources = TestSources {
            reg,
            fail: vec![],
            log: Arc::new(std::sync::Mutex::new(Vec::new())),
            drives: Vec::new(),
            delay_ms: 0,
            delays_ms: Vec::new(),
            conc: None,
            gates: Vec::new(),
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(0, None));
        let mut good = vec![0u8; 4096];
        let mut bad = vec![0u8; 4096];
        let mut batch = queue.batch();
        batch.push(ReadJob {
            shard_index: 0,
            offset: 0,
            len: 512,
            expected_crc: crc32c::crc32c(&pattern(0, 512)),
            verify_crc: true,
            prefetch: false,
            sources: &sources,
            dst: &mut good,
        });
        // Wrong digest: lands, then fails CRC verification.
        batch.push(ReadJob {
            shard_index: 0,
            offset: 512,
            len: 512,
            expected_crc: 0xdead_beef,
            verify_crc: true,
            prefetch: false,
            sources: &sources,
            dst: &mut bad,
        });
        assert!(matches!(batch.wait(), Err(StoreError::Crc { .. })));
        assert_eq!(good, pattern(0, 4096));
    }

    /// Mixed batch failure: the first read's chunks fail at runtime (EIO
    /// from the source on every replica, then requeue-until-exhausted), the
    /// second read lands and fails CRC. finish_read still runs for EVERY
    /// read — the CRC read's serving replica is marked Degraded even though
    /// the first read already failed, and its untouched replica stays
    /// Healthy — and the FIRST read's error is returned.
    #[test]
    fn batch_settles_every_read_on_mixed_failure() {
        // Two shards: shard 0 for the runtime-failing read, shard 1 for the
        // CRC-failing read, so the failure injection can't shadow the CRC
        // read's replica choices.
        let (_paths, reg) = fixture_shards(&[2, 2], "batch-mixed");
        let sources = TestSources {
            reg,
            fail: vec![(0, 0, usize::MAX), (0, 1, usize::MAX)],
            log: Arc::new(std::sync::Mutex::new(Vec::new())),
            drives: Vec::new(),
            delay_ms: 0,
            delays_ms: Vec::new(),
            conc: None,
            gates: Vec::new(),
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(0, None));
        let mut dead = vec![0u8; 4096];
        let mut bad = vec![0u8; 4096];
        let mut batch = queue.batch();
        // Every chunk attempt of this read fails EIO at runtime; the chunk
        // requeues until every replica is excluded → Unavailable.
        batch.push(ReadJob {
            shard_index: 0,
            offset: 0,
            len: 512,
            expected_crc: crc32c::crc32c(&pattern(0, 512)),
            verify_crc: true,
            prefetch: false,
            sources: &sources,
            dst: &mut dead,
        });
        // Wrong digest on the second read: lands, then fails CRC.
        batch.push(ReadJob {
            shard_index: 1,
            offset: 512,
            len: 512,
            expected_crc: 0xdead_beef,
            verify_crc: true,
            prefetch: false,
            sources: &sources,
            dst: &mut bad,
        });
        // First read's error wins, but only after BOTH reads settled.
        assert!(matches!(batch.wait(), Err(StoreError::Unavailable { .. })));
        // Runtime failures degraded both shard-0 replicas; exactly one
        // shard-1 replica served the CRC read (one chunk) and is Degraded,
        // the other served nothing and is still Healthy.
        for r in 0..2 {
            assert_eq!(
                sources.slot(0, r).health(),
                crate::replica::ReplicaHealth::Degraded,
                "shard-0 replica {r} must be Degraded after runtime EIO"
            );
        }
        let degraded = (0..2)
            .filter(|&r| sources.slot(1, r).health() == crate::replica::ReplicaHealth::Degraded)
            .count();
        let healthy = (0..2)
            .filter(|&r| sources.slot(1, r).health() == crate::replica::ReplicaHealth::Healthy)
            .count();
        assert_eq!(degraded, 1, "exactly the CRC read's serving replica");
        assert_eq!(healthy, 1, "untouched shard-1 replica stays Healthy");
    }

    #[test]
    fn failed_chunk_requeues_and_lands_on_a_healthy_replica() {
        // 2 replicas, replica 0 fails its first read; all chunks complete
        // WITHIN ONE wait() — the failed chunk was retried by replica 1 in
        // the same drain.
        let (_paths, reg) = fixture(2, "requeue");
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sources = TestSources {
            reg,
            fail: vec![(0, 0, 0)],
            log,
            drives: Vec::new(),
            delay_ms: 0,
            delays_ms: Vec::new(),
            conc: None,
            gates: Vec::new(),
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(1, None));
        let mut dst = vec![0u8; SHARD_BYTES];
        read_whole(&queue, &sources, &mut dst).wait().unwrap();
        assert_eq!(dst, pattern(0, SHARD_BYTES));
        let reads = sources.log.lock().unwrap().clone();
        // Replica 0 attempted one chunk, failed, was degraded; every chunk
        // (including the requeued one) then came from replica 1 — all inside
        // the single wait() above, no second enqueue.
        assert_eq!(reads.iter().filter(|(r, _)| *r == 0).count(), 1);
        assert_eq!(
            reads.iter().filter(|(r, _)| *r == 1).count(),
            plan_aligned(0, SHARD_BYTES).len(),
            "replica 1 served the requeued chunk plus the rest"
        );
        assert_eq!(
            sources.slot(0, 0).health(),
            crate::replica::ReplicaHealth::Degraded
        );
        assert_eq!(
            sources.slot(0, 1).health(),
            crate::replica::ReplicaHealth::Healthy
        );
    }

    #[test]
    fn demand_chunks_drain_before_prefetch() {
        // 1 reader thread; the PREFETCH read is enqueued first, demand
        // second — demand must still be served strictly first.
        let (_paths, reg) = fixture(1, "demand-first");
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sources = TestSources {
            // Logging wrapper with an effectively inexhaustible budget.
            reg,
            fail: vec![(0, 0, 1_000_000)],
            log,
            drives: Vec::new(),
            delay_ms: 0,
            delays_ms: Vec::new(),
            conc: None,
            gates: Vec::new(),
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(1, None));
        let plan = plan_aligned(0, SHARD_BYTES);
        let mut demand_dst = vec![0u8; SHARD_BYTES];
        let mut prefetch_dst = vec![0u8; SHARD_BYTES];
        let demand = read_whole(&queue, &sources, &mut demand_dst);
        let prefetch = queue.enqueue(ReadJob {
            shard_index: 0,
            offset: 0,
            len: SHARD_BYTES,
            expected_crc: crc32c::crc32c(&pattern(0, SHARD_BYTES)),
            verify_crc: true,
            prefetch: true,
            sources: &sources,
            dst: &mut prefetch_dst,
        });
        // Prefetch is IN the queue first; draining the demand read must still
        // serve all of its chunks before any prefetch chunk — a FIFO-serving
        // queue would log the earlier-enqueued prefetch chunks first.
        demand.wait().unwrap();
        let demand_reads = sources.log.lock().unwrap().clone();
        let expected: Vec<(usize, u64)> = plan.iter().map(|c| (0, c.0)).collect();
        assert_eq!(
            demand_reads, expected,
            "demand chunks served first, in plan order"
        );
        // Then the queued prefetch read drains untouched.
        prefetch.wait().unwrap();
        assert_eq!(demand_dst, pattern(0, SHARD_BYTES));
        assert_eq!(prefetch_dst, pattern(0, SHARD_BYTES));
        assert_eq!(sources.log.lock().unwrap().len(), 2 * plan.len());
    }

    #[test]
    fn healthy_reads_stripe_across_drives() {
        // 2 replicas on 2 distinct drives, derived worker count (one per
        // drive), 2 chunks. Each worker prefers its home drive, so the chunks
        // land on different replicas — a lowest-eligible pick would serve
        // both chunks from replica 0 (no striping, §5.2).
        let bytes = crate::stripe::CHUNK + 8192; // 2 chunks: CHUNK + 8 KiB
        let root = std::env::temp_dir().join("golem-store-queue-stripe");
        let mut replicas = Vec::with_capacity(2);
        for r in 0..2 {
            let dir = root.join(format!("r{r}"));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(SHARD_FILE), pattern(0, bytes)).unwrap();
            replicas.push(dir);
        }
        let manifest = crate::manifest::ModelManifest {
            name: "stripe".to_owned(),
            shards: vec![crate::manifest::ShardEntry {
                file: SHARD_FILE.to_owned(),
                bytes: bytes as u64,
                sha256: "0".repeat(64),
                replicas,
                ..Default::default()
            }],
            ..Default::default()
        };
        let reg = ReplicaRegistry::open(&manifest).unwrap();
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sources = TestSources {
            reg,
            fail: Vec::new(),
            log,
            drives: vec!["drv0".to_owned(), "drv1".to_owned()],
            // Wide per-chunk window: worker 1 must pop while worker 0 still
            // serves its chunk, even under heavy CI load.
            delay_ms: 250,
            delays_ms: Vec::new(),
            conc: None,
            gates: Vec::new(),
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(0, None).with_io_depth(1)); // one worker per drive
        let mut dst = vec![0u8; bytes];
        queue
            .enqueue(ReadJob {
                shard_index: 0,
                offset: 0,
                len: bytes,
                expected_crc: crc32c::crc32c(&pattern(0, bytes)),
                verify_crc: true,
                prefetch: false,
                sources: &sources,
                dst: &mut dst,
            })
            .wait()
            .unwrap();
        assert_eq!(dst, pattern(0, bytes));
        let reads = sources.log.lock().unwrap().clone();
        assert_eq!(reads.len(), 2, "both chunks served once: {reads:?}");
        assert_ne!(
            reads[0].0, reads[1].0,
            "chunks served by different replicas (striped across drives)"
        );
    }

    #[test]
    fn single_chunk_drains_alternate_drives() {
        let (_paths, reg) = fixture(2, "alternate");
        let sources = TestSources {
            reg,
            fail: Vec::new(),
            log: Arc::new(std::sync::Mutex::new(Vec::new())),
            drives: vec!["drv0".to_owned(), "drv1".to_owned()],
            delay_ms: 0,
            delays_ms: Vec::new(),
            conc: None,
            gates: Vec::new(),
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(0, None));
        for _ in 0..4 {
            let mut dst = vec![0u8; SHARD_BYTES];
            read_whole(&queue, &sources, &mut dst).wait().unwrap();
        }
        let replicas: Vec<usize> = sources.log.lock().unwrap().iter().map(|e| e.0).collect();
        assert_eq!(replicas, [0, 1, 0, 1]);
    }

    #[test]
    fn last_plane_covering_read_past_eof_keeps_replicas_healthy() {
        // /tmp is tmpfs (no O_DIRECT): the fixture lives under target/.
        let bytes = crate::stripe::CHUNK + 5000;
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../../target/golem-store-queue-eof-{}",
            std::process::id()
        ));
        let mut replicas = Vec::with_capacity(2);
        for r in 0..2 {
            let dir = root.join(format!("r{r}"));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(SHARD_FILE), pattern(0, bytes)).unwrap();
            replicas.push(dir);
        }
        let manifest = crate::manifest::ModelManifest {
            name: "eof".to_owned(),
            shards: vec![crate::manifest::ShardEntry {
                file: SHARD_FILE.to_owned(),
                bytes: bytes as u64,
                sha256: "0".repeat(64),
                replicas,
                ..Default::default()
            }],
            ..Default::default()
        };
        let reg = ReplicaRegistry::open(&manifest).unwrap();
        let sources = TestSources {
            reg,
            fail: Vec::new(),
            log: Arc::new(std::sync::Mutex::new(Vec::new())),
            drives: vec!["drv0".to_owned(), "drv1".to_owned()],
            delay_ms: 0,
            delays_ms: Vec::new(),
            conc: None,
            gates: Vec::new(),
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(0, None));
        let (offset, len) = (100usize, bytes - 100);
        let plan = plan_aligned(offset as u64, len);
        assert!(plan.len() > 1, "multi-chunk plane");
        let covering: usize = plan.iter().map(|c| c.1).sum();
        assert!(covering > bytes, "covering read ends past EOF");
        // O_DIRECT needs a 4 KiB-aligned destination.
        let mut storage = vec![0u8; covering + 4096];
        let skip = (4096 - storage.as_ptr() as usize % 4096) % 4096;
        let dst = &mut storage[skip..skip + covering];
        queue
            .enqueue(ReadJob {
                shard_index: 0,
                offset: offset as u64,
                len,
                expected_crc: crc32c::crc32c(&pattern(offset, len)),
                verify_crc: true,
                prefetch: false,
                sources: &sources,
                dst,
            })
            .wait()
            .unwrap();
        assert_eq!(
            &storage[skip + offset..skip + offset + len],
            &pattern(offset, len)[..]
        );
        let mut direct = 0;
        for r in 0..2 {
            let slot = sources.slot(0, r);
            assert_eq!(slot.health(), ReplicaHealth::Healthy, "replica {r}");
            direct += slot.served_direct();
        }
        assert!(
            direct > 0,
            "O_DIRECT served the covering read (target/ must support O_DIRECT)"
        );
        assert_eq!(queue.degraded_reads(), 0);
        drop(sources);
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn faster_drive_self_selects_more_chunks() {
        // Two replicas on two drives; replica 0 (drive 0) is fast (10 ms/chunk),
        // replica 1 (drive 1) is slow (500 ms/chunk). 10 chunks, one worker per
        // drive: with self-selection (§5.2) both replicas are eligible for every
        // chunk, so the fast worker pops proportionally more.
        // Determinism: a slow-drive read blocks until the fast drive has
        // completed 8 chunk reads, so the slow worker can hold at most 2 chunks
        // regardless of thread scheduling under parallel test load — the fast
        // share is ≥80% by construction, not by wall-clock luck.
        let bytes = crate::stripe::CHUNK * 10;
        let root = std::env::temp_dir().join("golem-store-queue-self-select");
        let mut replicas = Vec::with_capacity(2);
        for r in 0..2 {
            let dir = root.join(format!("r{r}"));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(SHARD_FILE), pattern(0, bytes)).unwrap();
            replicas.push(dir);
        }
        let manifest = crate::manifest::ModelManifest {
            name: "self-select".to_owned(),
            shards: vec![crate::manifest::ShardEntry {
                file: SHARD_FILE.to_owned(),
                bytes: bytes as u64,
                sha256: "0".repeat(64),
                replicas,
                ..Default::default()
            }],
            ..Default::default()
        };
        let reg = ReplicaRegistry::open(&manifest).unwrap();
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let done = Arc::new(AtomicUsize::new(0));
        let gate = Arc::new(Gate {
            done: Arc::clone(&done),
            bump: false,
            wait_until: Some(8),
        });
        let fast_gate = Arc::new(Gate {
            done,
            bump: true,
            wait_until: None,
        });
        let sources = TestSources {
            reg,
            fail: Vec::new(),
            log,
            drives: vec!["fast0".to_owned(), "slow1".to_owned()],
            delay_ms: 0,
            delays_ms: vec![10, 500],
            conc: None,
            gates: vec![Some(fast_gate), Some(gate)],
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(0, None).with_io_depth(1)); // one worker per drive
        let mut dst = vec![0u8; bytes];
        queue
            .enqueue(ReadJob {
                shard_index: 0,
                offset: 0,
                len: bytes,
                expected_crc: crc32c::crc32c(&pattern(0, bytes)),
                verify_crc: true,
                prefetch: false,
                sources: &sources,
                dst: &mut dst,
            })
            .wait()
            .unwrap();
        assert_eq!(dst, pattern(0, bytes));
        let reads = sources.log.lock().unwrap().clone();
        assert_eq!(reads.len(), 10, "every chunk served once: {reads:?}");
        let fast = reads.iter().filter(|(r, _)| *r == 0).count();
        let slow = reads.iter().filter(|(r, _)| *r == 1).count();
        assert!(
            fast * 10 >= reads.len() * 6,
            "fast drive must self-select ≥60% of chunks (nominally ~90%): fast={fast} slow={slow}"
        );
        // Structurally beats a static 50/50 split: the gate pins the slow
        // drive to ≤2 of 10 chunks, so fast > slow is guaranteed, not timed.
        assert!(
            fast > slow,
            "fast must out-select slow: fast={fast} slow={slow}"
        );
    }

    #[test]
    fn wrong_dst_length_returns_invalid_not_panic() {
        // Destination shorter than the aligned covering region: `Err(Invalid)`
        // from the public API — never a `split_at_mut` panic.
        let (_paths, reg) = fixture(1, "dst-len");
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sources = TestSources {
            reg,
            fail: Vec::new(),
            log,
            drives: Vec::new(),
            delay_ms: 0,
            delays_ms: Vec::new(),
            conc: None,
            gates: Vec::new(),
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(1, None));
        let covering = plan_aligned(0, SHARD_BYTES)
            .iter()
            .map(|c| c.1)
            .sum::<usize>();
        let mut short = vec![0u8; covering - 1];
        let err = read_whole(&queue, &sources, &mut short).wait().unwrap_err();
        assert!(matches!(err, StoreError::Invalid { .. }), "got {err:?}");
        // Same for a too-long destination (CRC window would be inconsistent).
        let mut long = vec![0u8; covering + 1];
        let err = read_whole(&queue, &sources, &mut long).wait().unwrap_err();
        assert!(matches!(err, StoreError::Invalid { .. }), "got {err:?}");
    }

    #[test]
    fn io_threads_override_is_not_clamped_to_replica_count() {
        // `--io-threads 4` over 2 replicas on ONE real drive runs 4 workers,
        // not the derived 1-per-drive (nor 2-per-replica): the explicit flag
        // replaces the derived count (D-12 bounds only the default). Peak
        // concurrent reads must exceed the derived count of 1.
        let bytes = crate::stripe::CHUNK * 4; // 4 chunks, one per worker
        let root = std::env::temp_dir().join("golem-store-queue-io-override");
        let mut replicas = Vec::with_capacity(2);
        for r in 0..2 {
            let dir = root.join(format!("r{r}"));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(SHARD_FILE), pattern(0, bytes)).unwrap();
            replicas.push(dir);
        }
        let manifest = crate::manifest::ModelManifest {
            name: "io-override".to_owned(),
            shards: vec![crate::manifest::ShardEntry {
                file: SHARD_FILE.to_owned(),
                bytes: bytes as u64,
                sha256: "0".repeat(64),
                replicas,
                ..Default::default()
            }],
            ..Default::default()
        };
        let reg = ReplicaRegistry::open(&manifest).unwrap();
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let conc = Arc::new(Concurrency::default());
        let pair = Arc::new(AtomicBool::new(false));
        let sources = TestSources {
            reg,
            fail: Vec::new(),
            log,
            drives: Vec::new(),
            delay_ms: 0,
            delays_ms: Vec::new(),
            conc: Some(Arc::clone(&conc)),
            gates: Vec::new(),
            wait_for_pair: Some(Arc::clone(&pair)),
        };
        let queue = Arc::new(ChunkQueue::new(4, None));
        let mut dst = vec![0u8; bytes];
        queue
            .enqueue(ReadJob {
                shard_index: 0,
                offset: 0,
                len: bytes,
                expected_crc: crc32c::crc32c(&pattern(0, bytes)),
                verify_crc: true,
                prefetch: false,
                sources: &sources,
                dst: &mut dst,
            })
            .wait()
            .unwrap();
        assert_eq!(dst, pattern(0, bytes));
        assert!(
            pair.load(Ordering::SeqCst),
            "override ran >= 2 workers; derived 1-drive count is 1"
        );
        assert!(conc.max.load(Ordering::SeqCst) >= 2);
    }

    /// (demand, prefetch) queue depths, for purge assertions.
    fn queued(queue: &ChunkQueue) -> (usize, usize) {
        let q = queue.lock();
        (q.demand.len(), q.prefetch.len())
    }

    #[test]
    fn dropping_an_unwaited_handle_empties_the_queue() {
        // Dropping a handle without wait()ing purges ALL of its queued
        // chunks (demand and prefetch) — nothing leaks into later drains.
        let (_paths, reg) = fixture(1, "drop-purge");
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sources = TestSources {
            reg,
            fail: Vec::new(),
            log,
            drives: Vec::new(),
            delay_ms: 0,
            delays_ms: Vec::new(),
            conc: None,
            gates: Vec::new(),
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(1, None));
        let mut dst_a = vec![0u8; SHARD_BYTES];
        let mut dst_b = vec![0u8; SHARD_BYTES];
        let a = read_whole(&queue, &sources, &mut dst_a);
        let b = queue.enqueue(ReadJob {
            shard_index: 0,
            offset: 0,
            len: SHARD_BYTES,
            expected_crc: crc32c::crc32c(&pattern(0, SHARD_BYTES)),
            verify_crc: true,
            prefetch: true,
            sources: &sources,
            dst: &mut dst_b,
        });
        assert_eq!(
            queued(&queue),
            (
                plan_aligned(0, SHARD_BYTES).len(),
                plan_aligned(0, SHARD_BYTES).len()
            )
        );
        drop(a);
        drop(b);
        assert_eq!(queued(&queue), (0, 0), "un-waited handles leave no jobs");
    }

    #[test]
    fn early_return_wait_purges_queued_jobs() {
        // wait() that returns early (dst length mismatch) must still leave
        // the queue empty — the purge happens before any early-return path.
        let (_paths, reg) = fixture(1, "early-return-purge");
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sources = TestSources {
            reg,
            fail: Vec::new(),
            log,
            drives: Vec::new(),
            delay_ms: 0,
            delays_ms: Vec::new(),
            conc: None,
            gates: Vec::new(),
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(1, None));
        let mut short = vec![0u8; SHARD_BYTES - 1];
        let handle = read_whole(&queue, &sources, &mut short);
        let err = handle.wait().unwrap_err();
        assert!(matches!(err, StoreError::Invalid { .. }), "got {err:?}");
        assert_eq!(queued(&queue), (0, 0), "early-return wait purges its jobs");
    }

    #[test]
    fn dropping_handle_a_leaves_handle_b_drainable() {
        // Purge is per-read: dropping handle A removes only A's chunks; B's
        // jobs stay queued and drain to a correct result afterwards.
        let (_paths, reg) = fixture(1, "purge-isolation");
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sources = TestSources {
            reg,
            fail: Vec::new(),
            log,
            drives: Vec::new(),
            delay_ms: 0,
            delays_ms: Vec::new(),
            conc: None,
            gates: Vec::new(),
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(1, None));
        let chunks = plan_aligned(0, SHARD_BYTES).len();
        let mut dst_a = vec![0u8; SHARD_BYTES];
        let mut dst_b = vec![0u8; SHARD_BYTES];
        let a = read_whole(&queue, &sources, &mut dst_a);
        let b = read_whole(&queue, &sources, &mut dst_b);
        assert_eq!(queued(&queue), (2 * chunks, 0));
        drop(a);
        assert_eq!(queued(&queue), (chunks, 0), "only A's chunks purged");
        b.wait().unwrap();
        assert_eq!(dst_b, pattern(0, SHARD_BYTES));
        assert_eq!(queued(&queue), (0, 0));
    }

    #[test]
    fn more_than_32_replicas_is_invalid_not_shift_overflow() {
        // The u32 exclude mask bounds shards at 32 replicas; the 33rd is a
        // manifest bug surfaced as `Err(Invalid)`, not a `1 << r` panic.
        let bytes = SHARD_BYTES;
        let root = std::env::temp_dir().join("golem-store-queue-33");
        let mut replicas = Vec::with_capacity(33);
        for r in 0..33 {
            let dir = root.join(format!("r{r}"));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(SHARD_FILE), pattern(0, bytes)).unwrap();
            replicas.push(dir);
        }
        let manifest = crate::manifest::ModelManifest {
            name: "too-many".to_owned(),
            shards: vec![crate::manifest::ShardEntry {
                file: SHARD_FILE.to_owned(),
                bytes: bytes as u64,
                sha256: "0".repeat(64),
                replicas,
                ..Default::default()
            }],
            ..Default::default()
        };
        let reg = ReplicaRegistry::open(&manifest).unwrap();
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sources = TestSources {
            reg,
            fail: Vec::new(),
            log,
            drives: Vec::new(),
            delay_ms: 0,
            delays_ms: Vec::new(),
            conc: None,
            gates: Vec::new(),
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(1, None));
        let mut dst = vec![0u8; SHARD_BYTES];
        let err = read_whole(&queue, &sources, &mut dst).wait().unwrap_err();
        assert!(matches!(err, StoreError::Invalid { .. }), "got {err:?}");
        // The early return also purged this read's queued chunks.
        assert_eq!(queued(&queue), (0, 0));
    }

    #[test]
    fn single_replica_failure_fails_promptly_without_hanging() {
        // 1 replica, source always fails: wait() returns
        // Err(StoreError::Unavailable) promptly — a hang would trip the
        // nextest slow-timeout and show as a failure, not a CI stall.
        let (_paths, reg) = fixture(1, "single-fail");
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sources = TestSources {
            reg,
            fail: vec![(0, 0, usize::MAX)],
            log,
            drives: Vec::new(),
            delay_ms: 0,
            delays_ms: Vec::new(),
            conc: None,
            gates: Vec::new(),
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(1, None));
        let mut dst = vec![0u8; SHARD_BYTES];
        let result = read_whole(&queue, &sources, &mut dst).wait();
        match result {
            Err(StoreError::Unavailable { shard }) => assert_eq!(shard, SHARD_FILE),
            other => panic!("expected Unavailable, got {other:?}"),
        }
        assert_eq!(
            sources.slot(0, 0).health(),
            crate::replica::ReplicaHealth::Degraded
        );
    }

    #[test]
    fn four_replica_shard_survives_three_dead() {
        // T5 shape: 4 replicas, 3 always-fail → read still completes.
        let (_paths, reg) = fixture(4, "survive-three");
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sources = TestSources {
            reg,
            fail: vec![(0, 0, 0), (0, 1, 0), (0, 2, 0)],
            log,
            drives: Vec::new(),
            delay_ms: 0,
            delays_ms: Vec::new(),
            conc: None,
            gates: Vec::new(),
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(4, None));
        let mut dst = vec![0u8; SHARD_BYTES];
        read_whole(&queue, &sources, &mut dst).wait().unwrap();
        assert_eq!(dst, pattern(0, SHARD_BYTES));
        assert_eq!(
            sources.slot(0, 3).health(),
            crate::replica::ReplicaHealth::Healthy
        );
    }

    #[test]
    fn chunks_land_directly_in_the_destination_region() {
        // Region pre-filled 0xAA; after wait() the logical range equals the
        // fake data and scratch bytes (head padding) may differ but nothing
        // outside the &mut [u8] region changed (scope-borrowed, so enforced).
        let (_paths, reg) = fixture(1, "land-direct");
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sources = TestSources {
            reg,
            fail: Vec::new(),
            log,
            drives: Vec::new(),
            delay_ms: 0,
            delays_ms: Vec::new(),
            conc: None,
            gates: Vec::new(),
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(1, None));
        // Read [512, 512+2048): covering region is [0, 4096).
        let covering = plan_aligned(512, 2048).iter().map(|c| c.1).sum::<usize>();
        assert_eq!(covering, 4096);
        let mut dst = vec![0xAAu8; covering];
        queue
            .enqueue(ReadJob {
                shard_index: 0,
                offset: 512,
                len: 2048,
                expected_crc: crc32c::crc32c(&pattern(512, 2048)),
                verify_crc: true,
                prefetch: false,
                sources: &sources,
                dst: &mut dst,
            })
            .wait()
            .unwrap();
        assert_eq!(&dst[512..512 + 2048], &pattern(512, 2048)[..]);
    }

    #[test]
    fn crc_mismatch_marks_replica_degraded_and_errors() {
        // §5.3: flip bytes in both replicas' files → plane CRC mismatch →
        // Err(Crc) and the serving replicas are degraded.
        let (paths, reg) = fixture(2, "crc-mismatch");
        for p in &paths {
            let mut data = std::fs::read(p.join(SHARD_FILE)).unwrap();
            data[100] ^= 0xff;
            std::fs::write(p.join(SHARD_FILE), data).unwrap();
        }
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sources = TestSources {
            reg,
            fail: Vec::new(),
            log,
            drives: Vec::new(),
            delay_ms: 0,
            delays_ms: Vec::new(),
            conc: None,
            gates: Vec::new(),
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(2, None));
        let mut dst = vec![0u8; SHARD_BYTES];
        let result = read_whole(&queue, &sources, &mut dst).wait();
        match result {
            Err(StoreError::Crc { shard }) => assert_eq!(shard, SHARD_FILE),
            other => panic!("expected Crc, got {other:?}"),
        }
        // Only the replica that actually served chunks is degraded.
        assert_eq!(
            sources.slot(0, 0).health(),
            crate::replica::ReplicaHealth::Degraded
        );
        assert_eq!(
            sources.slot(0, 1).health(),
            crate::replica::ReplicaHealth::Healthy
        );
    }

    /// §5.3 over a multi-chunk read: an unaligned head, a full middle
    /// chunk and a partial tail. The per-chunk CRC parts fold to the window
    /// CRC, a flipped byte inside the window fails it, and bytes outside the
    /// window (head padding, past the end) are not hashed.
    #[test]
    fn multi_chunk_crc_detects_a_corrupt_middle_chunk_only_inside_the_window() {
        const OFF: usize = 1000;
        const LEN: usize = 2 * CHUNK + 5000;
        let bytes = 2 * CHUNK + 3 * 4096 + 777;
        let read = |tag: &str, flip: Option<usize>| {
            let (paths, reg) = fixture_sized(2, tag, bytes);
            if let Some(at) = flip {
                for p in &paths {
                    let mut data = std::fs::read(p.join(SHARD_FILE)).unwrap();
                    data[at] ^= 0xff;
                    std::fs::write(p.join(SHARD_FILE), data).unwrap();
                }
            }
            let sources = TestSources {
                reg,
                fail: Vec::new(),
                log: Arc::new(std::sync::Mutex::new(Vec::new())),
                drives: Vec::new(),
                delay_ms: 0,
                delays_ms: Vec::new(),
                conc: None,
                gates: Vec::new(),
                wait_for_pair: None,
            };
            let queue = Arc::new(ChunkQueue::new(2, None));
            let plan = plan_aligned(OFF as u64, LEN);
            assert_eq!(plan.len(), 3, "head, middle, tail");
            let mut dst = vec![0u8; plan.iter().map(|c| c.1).sum()];
            let result = queue
                .enqueue(ReadJob {
                    shard_index: 0,
                    offset: OFF as u64,
                    len: LEN,
                    expected_crc: crc32c::crc32c(&pattern(OFF, LEN)),
                    verify_crc: true,
                    prefetch: false,
                    sources: &sources,
                    dst: &mut dst,
                })
                .wait();
            (result, dst)
        };
        let (ok, dst) = read("crc-multi-clean", None);
        ok.unwrap();
        assert!(dst[OFF..OFF + LEN] == pattern(OFF, LEN)[..]);
        match read("crc-multi-middle", Some(CHUNK + 4096)).0 {
            Err(StoreError::Crc { .. }) => {}
            other => panic!("expected Crc, got {other:?}"),
        }
        read("crc-multi-head-pad", Some(OFF - 1)).0.unwrap();
        read("crc-multi-tail-pad", Some(OFF + LEN)).0.unwrap();
    }

    #[test]
    fn concurrent_enqueue_drain_stress_no_lost_wakeups() {
        // 4 reader threads × reads each, timeout guard; a lost wakeup would
        // hang a wait() and blow the deadline.
        let (_paths, reg) = fixture(2, "stress");
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sources = Arc::new(TestSources {
            reg,
            fail: Vec::new(),
            log,
            drives: Vec::new(),
            delay_ms: 0,
            delays_ms: Vec::new(),
            conc: None,
            gates: Vec::new(),
            wait_for_pair: None,
        });
        let queue = Arc::new(ChunkQueue::new(4, None));
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        const THREADS: usize = 4;
        const READS: usize = 16;
        std::thread::scope(|s| {
            for t in 0..THREADS {
                let queue = Arc::clone(&queue);
                let sources = Arc::clone(&sources);
                let tx = tx.clone();
                s.spawn(move || {
                    for i in 0..READS {
                        let mut dst = vec![0u8; SHARD_BYTES];
                        queue
                            .enqueue(ReadJob {
                                shard_index: 0,
                                offset: 0,
                                len: SHARD_BYTES,
                                expected_crc: crc32c::crc32c(&pattern(0, SHARD_BYTES)),
                                verify_crc: true,
                                prefetch: (t + i) % 2 == 1,
                                sources: sources.as_ref(),
                                dst: &mut dst,
                            })
                            .wait()
                            .unwrap();
                        assert_eq!(dst, pattern(0, SHARD_BYTES));
                    }
                    tx.send(()).unwrap();
                });
            }
            drop(tx);
        });
        // Watchdog: all THREADS completions must arrive within the deadline.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        for _ in 0..THREADS {
            let left = deadline
                .checked_duration_since(std::time::Instant::now())
                .unwrap_or_default();
            rx.recv_timeout(left)
                .expect("all drains completed before the deadline (no lost wakeups)");
        }
    }

    /// Zero-cost chunk source: the drain's pop/scan cost dominates.
    struct NullSource;

    impl ChunkSource for NullSource {
        fn read_chunk(&self, _offset: u64, _buf: &mut [u8]) -> io::Result<()> {
            Ok(())
        }
    }

    /// Two replicas on two fake drives, both served by [`NullSource`].
    struct NullSources(ReplicaRegistry);

    impl ShardSources for NullSources {
        fn slots(&self, _shard_index: usize) -> Vec<Arc<dyn ChunkSource>> {
            vec![Arc::new(NullSource), Arc::new(NullSource)]
        }
        fn slot(&self, shard_index: usize, replica_index: usize) -> &ReplicaSlot {
            self.0.slot(shard_index, replica_index)
        }
        fn shard_name(&self, shard_index: usize) -> &str {
            self.0.shard_name(shard_index)
        }
        fn replica_drive(&self, _shard_index: usize, replica_index: usize) -> &str {
            ["d0", "d1"][replica_index]
        }
    }

    /// Pop cost of the shared drain list (§5.2) under the worst scan: replica
    /// 0 Degraded, so every job is exclusive to drive 1 and drive-0 workers
    /// scan past all of it. One layer's misses = 288 experts × 3 planes = 864
    /// chunks; one UD-IQ3_XXS plane ≈ 4–6 MB at ~3 GB/s ≈ 1.5 ms per read, so
    /// per-drive queues are warranted only above 5 % of that = 75 µs/pop.
    #[test]
    #[ignore = "pop-cost bench; GOLEM_POP_BENCH=1 --ignored --nocapture"]
    fn pop_cost_with_one_replica_degraded() {
        assert_eq!(
            std::env::var("GOLEM_POP_BENCH").as_deref(),
            Ok("1"),
            "set GOLEM_POP_BENCH=1"
        );
        const JOBS: usize = 288 * 3;
        const LEN: usize = 4096;
        let (_paths, reg) = fixture(2, "pop-bench");
        let sources = NullSources(reg);
        sources.slot(0, 0).mark_degraded();
        let queue = Arc::new(ChunkQueue::new(0, None));
        let mut dsts = vec![vec![0u8; LEN]; JOBS];
        let start = std::time::Instant::now();
        let mut b = queue.batch();
        for dst in &mut dsts {
            b.push(ReadJob {
                shard_index: 0,
                offset: 0,
                len: LEN,
                expected_crc: 0,
                verify_crc: false,
                prefetch: false,
                sources: &sources,
                dst,
            });
        }
        b.wait().unwrap();
        let ns_per_pop = start.elapsed().as_nanos() / JOBS as u128;
        println!("pop_cost_with_one_replica_degraded: {ns_per_pop} ns/pop over {JOBS} chunks");
        assert!(
            ns_per_pop < 75_000,
            "{ns_per_pop} ns/pop ≥ 75 µs: per-drive queues warranted"
        );
    }
}
