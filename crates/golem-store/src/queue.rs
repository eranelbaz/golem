//! §5.2 shared chunk queue: one queue of chunk reads drained demand-first by
//! per-drive scoped reader threads (D-12).
//!
//! Invariants:
//! - The global queue holds jobs **without** destination slices; slices live
//!   only drain-locally, carried by [`DrainJob`] and handed out as disjoint
//!   `&mut [u8]` regions via `split_at_mut` inside `std::thread::scope` — no
//!   raw pointer ever crosses a thread and this crate stays `unsafe`-free.
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
use crate::stripe::plan_aligned;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};

/// Everything a drain needs to settle one logical read.
pub(crate) struct ReadState {
    /// Position of the shard in the manifest.
    pub shard_index: usize,
    /// Shard file name for errors.
    pub shard_name: String,
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
pub(crate) struct Progress {
    /// Set when one chunk is irrecoverable: the read's remaining chunks fail
    /// without further reads.
    pub failed: bool,
    /// First terminal error of the read.
    pub error: Option<StoreError>,
}

impl ReadState {
    /// Records a settled chunk; `Err` fails the whole read.
    pub fn settle(&self, result: Result<(), StoreError>) {
        let mut p = self.lock_progress();
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
    chunk_index: usize,
    /// Bit i set = replica i has already failed this chunk.
    exclude_mask: u32,
}

/// Drain-local job: the chunk plus the disjoint destination sub-slice it is
/// read directly into. Never leaves the drain that created it.
struct DrainJob<'a> {
    job: Job,
    dst: &'a mut [u8],
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

/// One shared queue of chunk reads (§5.2).
pub struct ChunkQueue {
    inner: Mutex<QueueInner>,
    /// Reader threads per drain; 0 = derive one thread per physical drive.
    /// An explicit override wins over the derived per-drive count (D-12
    /// bounds the default at 1–2 per drive; the operator's flag does not).
    io_threads: usize,
}

impl ChunkQueue {
    /// New queue; `io_threads` caps the drain's reader threads (0 = derive
    /// one per physical drive hosting an eligible replica, §5.2/D-12).
    pub fn new(io_threads: usize) -> Self {
        Self {
            inner: Mutex::new(QueueInner::default()),
            io_threads,
        }
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
        let (read, dst, sources) = self.submit(job);
        ReadHandle {
            inner: Some(HandleInner {
                // Cloned Arc: the handle keeps the queue alive past engine drop.
                queue: Arc::clone(self),
                read,
                dst,
                sources,
            }),
        }
    }

    /// Starts an empty batch (§5.2): push several logical reads — typically
    /// one expert unit's planes — then drain them all in a single
    /// [`ReadBatch::wait`] instead of one blocking drain per read.
    pub fn batch(self: &Arc<Self>) -> ReadBatch<'_> {
        ReadBatch {
            queue: Arc::clone(self),
            reads: Vec::new(),
        }
    }

    /// Shared enqueue path: builds the read state and queues its chunks.
    /// Returns the parts the caller's handle keeps (the destination slices
    /// never enter the global queue).
    fn submit<'a>(
        self: &Arc<Self>,
        job: ReadJob<'a>,
    ) -> (Arc<ReadState>, &'a mut [u8], &'a dyn ShardSources) {
        let chunks = plan_aligned(job.offset, job.len);
        let state = Arc::new(ReadState {
            shard_index: job.shard_index,
            shard_name: job.sources.shard_name(job.shard_index).to_owned(),
            // The logical window inside the covering region: head scratch is
            // `align_down(offset, 4096) - offset` bytes (§5.4).
            crc_at: (job.offset - align_down(job.offset)) as usize,
            crc_len: job.len,
            expected_crc: job.expected_crc,
            verify_crc: job.verify_crc,
            progress: Mutex::new(Progress {
                failed: false,
                error: None,
            }),
            served: AtomicU32::new(0),
            chunks,
        });
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
    read: Arc<ReadState>,
    dst: &'a mut [u8],
    sources: &'a dyn ShardSources,
}

/// Removes every queued job belonging to `read` from the global queue
/// (demand first). Called by `wait` and by `Drop for ReadHandle`, so a read
/// that is never (or only partially) drained cannot leak its chunks.
fn purge(queue: &ChunkQueue, read: &Arc<ReadState>) -> VecDeque<Job> {
    let mut work = VecDeque::new();
    let mut q = queue.lock();
    let QueueInner {
        demand, prefetch, ..
    } = &mut *q;
    // Single pass per deque: one lock acquisition, no O(n) `remove` per hit.
    for dq in [demand, prefetch] {
        let (mine, rest): (VecDeque<_>, VecDeque<_>) = std::mem::take(dq)
            .into_iter()
            .partition(|j| Arc::ptr_eq(&j.read, read));
        work.extend(mine);
        *dq = rest;
    }
    work
}

impl Drop for ReadHandle<'_> {
    fn drop(&mut self) {
        if let Some(inner) = self.inner.take() {
            let HandleInner { queue, read, .. } = inner;
            purge(&queue, &read);
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
            read,
            dst,
            sources,
        } = self.inner.take().expect("handle consumed at most once");
        let shard = read.shard_index;
        let replica_count = sources.slots(shard).len();
        // Take this read's chunks out of the global queue up front (demand
        // first) so every early-return path below — and the Drop fallback —
        // leave no jobs behind. A failed chunk can then requeue drain-locally
        // instead of deadlocking on a queue nobody else drains.
        let work = purge(&queue, &read);
        // Exclude masks (and the §5.3 `served` set) are `u32` bitmasks over
        // replica indices: bound the shard rather than widen — real
        // topologies are ≤6 replicas, and a 33rd replica is a manifest bug
        // surfaced as `Err`, not a shift-overflow panic.
        if replica_count > 32 {
            return Err(StoreError::Invalid {
                msg: format!(
                    "shard {} has {replica_count} replicas; the u32 exclude mask supports at most 32",
                    read.shard_name
                ),
            });
        }
        // Destination contract (§5.4): exactly the aligned covering region,
        // chunk 0 at the region base. Wrong length is a caller bug → `Err`,
        // never a `split_at_mut` panic; also proves the CRC window below is
        // in bounds.
        let covering: usize = read.chunks.iter().map(|c| c.1).sum();
        if dst.len() != covering {
            return Err(StoreError::Invalid {
                msg: format!(
                    "destination is {} bytes but the aligned covering region is {covering} bytes",
                    dst.len(),
                ),
            });
        }

        if queue.lock().shutdown {
            // Queue closed before this read drained: fail, never hang (its
            // jobs were already purged above).
            return Err(StoreError::Unavailable {
                shard: read.shard_name.clone(),
            });
        }
        debug_assert_eq!(work.len(), read.chunks.len());

        // Fail immediately when no eligible reader exists at all.
        if healthy_mask(sources, shard, replica_count) == 0 {
            let mut p = read.lock_progress();
            p.failed = true;
            p.error = Some(StoreError::Unavailable {
                shard: read.shard_name.clone(),
            });
        }

        let mut ctxs = HashMap::new();
        ctxs.insert(shard, ShardCtx::new(sources, shard));
        if read.lock_progress().error.is_none() {
            // Reborrow: the drain only needs the destination for its own
            // duration; CRC verification reads it again below.
            let mut reads = [(Arc::clone(&read), &mut *dst)];
            drain_reads(&queue, &ctxs, &mut reads, work);
        }

        // Collect: first terminal error wins; else verify the §5.3 plane CRC
        // over the logical window once every chunk has landed.
        finish_read(&read, dst, sources, shard)
    }
}

/// Post-drain collection for one read: first terminal error wins, else the
/// §5.3 CRC over the logical window; a mismatch degrades every replica that
/// served a chunk (a plane-granular CRC cannot attribute one replica, §5.3).
fn finish_read(
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
    let replica_count = sources.slots(shard).len();
    let crc = crc32c::crc32c(&dst[read.crc_at..read.crc_at + read.crc_len]);
    if crc != read.expected_crc {
        // §5.3: treat as a corrupted replica, same path as EIO.
        let served = read.served.load(Ordering::Relaxed);
        for r in 0..replica_count {
            if served & (1 << r) != 0 {
                sources.slot(shard, r).mark_degraded();
            }
        }
        tracing::warn!(
            shard = %read.shard_name,
            expected = read.expected_crc,
            got = crc,
            "plane crc mismatch; serving replicas marked Degraded"
        );
        return Err(StoreError::Crc {
            shard: read.shard_name.clone(),
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
    reads: Vec<BatchEntry<'a>>,
}

impl Drop for ReadBatch<'_> {
    fn drop(&mut self) {
        for e in &mut self.reads {
            purge(&self.queue, &e.read);
        }
    }
}

impl<'a> ReadBatch<'a> {
    /// Queues one more logical read into the batch (chunks enter the global
    /// queue immediately; the drain happens at [`ReadBatch::wait`]).
    pub fn push(&mut self, job: ReadJob<'a>) {
        let (read, dst, sources) = self.queue.submit(job);
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
        let entries = std::mem::take(&mut self.reads);
        if entries.is_empty() {
            return Ok(());
        }
        // Take every read's queued chunks out of the global queue up front
        // (demand first) so all early returns below — and the Drop fallback
        // — leave no jobs behind.
        let mut work: VecDeque<Job> = VecDeque::new();
        let mut reads: Vec<(Arc<ReadState>, &mut [u8], &dyn ShardSources)> = Vec::new();
        for BatchEntry { read, dst, sources } in entries {
            work.extend(purge(&self.queue, &read));
            reads.push((read, dst, sources));
        }
        if self.queue.lock().shutdown {
            return Err(StoreError::Unavailable {
                shard: reads[0].0.shard_name.clone(),
            });
        }

        // Per-shard drain context, built once per drain (§5.2 hot path: no
        // per-chunk replica-slot or drive lookups).
        let mut ctxs: HashMap<usize, ShardCtx<'_>> = HashMap::new();
        for (read, dst, sources) in &reads {
            let shard = read.shard_index;
            ctxs.entry(shard)
                .or_insert_with(|| ShardCtx::new(*sources, shard));
            let replica_count = sources.slots(shard).len();
            if replica_count > 32 {
                return Err(StoreError::Invalid {
                    msg: format!(
                        "shard {} has {replica_count} replicas; the u32 exclude mask supports at most 32",
                        read.shard_name
                    ),
                });
            }
            // Destination contract (§5.4): exactly the aligned covering
            // region per read; also proves the CRC window below is in bounds.
            let covering: usize = read.chunks.iter().map(|c| c.1).sum();
            if dst.len() != covering {
                return Err(StoreError::Invalid {
                    msg: format!(
                        "destination is {} bytes but the aligned covering region is {covering} bytes",
                        dst.len(),
                    ),
                });
            }
            // Fail immediately when no eligible reader exists at all.
            if healthy_mask(*sources, shard, replica_count) == 0 {
                let mut p = read.lock_progress();
                p.failed = true;
                p.error = Some(StoreError::Unavailable {
                    shard: read.shard_name.clone(),
                });
            }
        }
        {
            let mut slice: Vec<(Arc<ReadState>, &mut [u8])> = reads
                .iter_mut()
                .map(|(r, d, _)| (Arc::clone(r), &mut **d))
                .collect();
            drain_reads(&self.queue, &ctxs, &mut slice, work);
        }
        // Collect per read; the first failure fails the whole batch.
        for (read, dst, sources) in &reads {
            finish_read(read, dst, *sources, read.shard_index)?;
        }
        Ok(())
    }
}

/// Bit i set iff replica i of `shard` is Healthy (read once per decision).
fn healthy_mask(sources: &dyn ShardSources, shard: usize, replica_count: usize) -> u32 {
    (0..replica_count)
        .filter(|&r| sources.slot(shard, r).health() == ReplicaHealth::Healthy)
        .fold(0u32, |m, r| m | (1 << r))
}

/// Shared drain state: the drain-local work list (jobs carry their slices)
/// and the count of unsettled chunks.
struct Shared<'a> {
    list: VecDeque<DrainJob<'a>>,
    remaining: usize,
}

/// Per-drain context for one shard (§5.2): resolved once per drain — the hot
/// loop never touches sysfs nor allocates per chunk.
struct ShardCtx<'a> {
    /// Shard this context serves.
    shard: usize,
    /// The shard's replica slots (chunk readers).
    slots: Arc<[Arc<dyn ChunkSource>]>,
    /// Physical-drive index per replica (drive-grouped striping, §5.2).
    replica_drive: Vec<usize>,
    /// Distinct drives hosting the shard's replicas.
    drives: usize,
    /// Health marking on chunk failure / CRC degradation.
    sources: &'a dyn ShardSources,
}

impl<'a> ShardCtx<'a> {
    /// Resolves the shard's replica slots and drive map (one sysfs-free pass:
    /// device identity was cached at replica open, §5.1/§5.2).
    fn new(sources: &'a dyn ShardSources, shard: usize) -> Self {
        let slots: Arc<[Arc<dyn ChunkSource>]> = sources.slots(shard).into();
        let replica_count = slots.len();
        let mut names: Vec<&str> = Vec::with_capacity(replica_count);
        let mut replica_drive = Vec::with_capacity(replica_count);
        for r in 0..replica_count {
            let name = sources.replica_drive(shard, r);
            let idx = match names.iter().position(|n| *n == name) {
                Some(i) => i,
                None => {
                    names.push(name);
                    names.len() - 1
                }
            };
            replica_drive.push(idx);
        }
        ShardCtx {
            shard,
            slots,
            replica_drive,
            drives: names.len(),
            sources,
        }
    }
}

/// Runs the drain to completion over a batch of reads: chunks from every
/// read (possibly spanning shards) share one drain-local work list and one
/// set of reader threads; each chunk's slice is the disjoint sub-slice of
/// its own read's destination. Reader threads run inside `thread::scope`.
fn drain_reads(
    queue: &ChunkQueue,
    ctxs: &HashMap<usize, ShardCtx<'_>>,
    reads: &mut [(Arc<ReadState>, &mut [u8])],
    work: VecDeque<Job>,
) {
    // Chunk i of a read reads into the i-th disjoint sub-slice of that
    // read's destination (§5.2: disk→destination exactly once). Reborrowed
    // per call so the caller can re-read `dst` for CRC verification.
    type ReadSlices<'x> = Vec<(Arc<ReadState>, Vec<Option<&'x mut [u8]>>)>;
    let mut slices: ReadSlices<'_> = reads
        .iter_mut()
        .map(|(read, dst)| {
            let lens: Vec<usize> = read.chunks.iter().map(|c| c.1).collect();
            let parts: Vec<Option<&mut [u8]>> =
                split_disjoint(dst, &lens).into_iter().map(Some).collect();
            (Arc::clone(read), parts)
        })
        .collect();
    let mut list: VecDeque<DrainJob<'_>> = VecDeque::with_capacity(work.len());
    for job in work {
        let (_, parts) = slices
            .iter_mut()
            .find(|(r, _)| Arc::ptr_eq(r, &job.read))
            .expect("queued job belongs to a batched read");
        let dst = parts[job.chunk_index]
            .take()
            .expect("exactly one job per chunk");
        list.push_back(DrainJob { job, dst });
    }
    let remaining = list.len();

    // Reader threads per drain (§5.2, D-12): one thread per physical drive
    // hosting an eligible replica by default — per-drive, not per-file. An
    // explicit `--io-threads` override REPLACES the derived count, capped at
    // the pending chunk count: more workers than chunks would idle.
    let drives = ctxs.values().map(|c| c.drives).max().unwrap_or(0);
    let workers = (if queue.io_threads > 0 {
        queue.io_threads
    } else {
        drives.max(1)
    })
    .min(list.len().max(1));

    let shared = Arc::new((Mutex::new(Shared { list, remaining }), Condvar::new()));
    std::thread::scope(|s| {
        for w in 0..workers {
            let shared = Arc::clone(&shared);
            // Home drive for striping (§5.2): workers are pinned round-robin
            // to drives so healthy reads stripe across devices instead of
            // every worker contending for the lowest replica index. A fast
            // drive's worker pops more chunks — it self-selects.
            let home = if drives == 0 { 0 } else { w % drives };
            s.spawn(move || {
                loop {
                    // Pop a job (any drive's thread may serve it — the slice
                    // travels with the job), or exit when everything settled.
                    let job = {
                        let (lock, cv) = &*shared;
                        let mut sh = lock.lock().unwrap_or_else(PoisonError::into_inner);
                        loop {
                            if let Some(j) = sh.list.pop_front() {
                                break j;
                            }
                            if sh.remaining == 0 {
                                return;
                            }
                            sh = cv.wait(sh).unwrap_or_else(PoisonError::into_inner);
                        }
                    };
                    let ctx = &ctxs[&job.job.read.shard_index];
                    run_chunk(&shared, ctx, home, job);
                }
            });
        }
    });
}

/// Serves one chunk, preferring replicas on the worker's home drive (§5.2
/// striping: each worker self-selects the chunks its drive can serve; a
/// worker whose home drive is degraded falls back to any eligible replica).
/// On failure degrades the replica, requeues onto the same drain-local list
/// with the replica's bit set, and lets a healthy drive retry.
fn run_chunk<'a>(
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
                    shard: read.shard_name.clone(),
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
                        shard: read.shard_name.clone(),
                    }),
                );
            }
        };
        // Read directly into the destination sub-slice: expert bytes are
        // copied disk→destination exactly once (§5.2).
        match slots[replica].read_chunk(offset, job.dst) {
            Ok(()) => {
                read.served.fetch_or(1 << replica, Ordering::Relaxed);
                settle(shared, read, Ok(()));
            }
            Err(e) => {
                sources.slot(shard, replica).mark_degraded();
                tracing::warn!(
                    shard = %read.shard_name,
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
    read.settle(result);
    let (lock, cv) = shared;
    lock.lock()
        .unwrap_or_else(PoisonError::into_inner)
        .remaining -= 1;
    cv.notify_all();
}

/// Splits `whole` into consecutive disjoint sub-slices of `lens` lengths.
fn split_disjoint<'a>(mut whole: &'a mut [u8], lens: &[usize]) -> Vec<&'a mut [u8]> {
    // Precondition enforced by the public entry point (`wait` validates the
    // destination length): the lens sum to exactly `whole.len()`.
    debug_assert_eq!(lens.iter().sum::<usize>(), whole.len());
    let mut out = Vec::with_capacity(lens.len());
    for &l in lens {
        let (head, tail) = whole.split_at_mut(l);
        out.push(head);
        whole = tail;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::replica::{ReplicaRegistry, ReplicaSlot};
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

    /// Fixture files over `n` replica dirs; returns (paths, registry).
    fn fixture(n: usize, tag: &str) -> (Vec<PathBuf>, ReplicaRegistry) {
        let root = std::env::temp_dir().join(format!("golem-store-queue-{tag}"));
        let mut replicas = Vec::with_capacity(n);
        for r in 0..n {
            let dir = root.join(format!("r{r}"));
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(SHARD_FILE), pattern(0, SHARD_BYTES)).unwrap();
            replicas.push(dir);
        }
        let manifest = crate::manifest::ModelManifest {
            name: tag.to_owned(),
            shards: vec![crate::manifest::ShardEntry {
                file: SHARD_FILE.to_owned(),
                bytes: SHARD_BYTES as u64,
                sha256: "0".repeat(64),
                replicas: replicas.clone(),
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
            self.log.lock().unwrap().push((self.id, offset));
            if self.delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(self.delay_ms));
            }
            let _g = self.conc.as_ref().map(|c| c.enter());
            if let Some(flag) = &self.wait_for_pair {
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                while !flag.load(Ordering::SeqCst) && std::time::Instant::now() < deadline {
                    if let Some(c) = &self.conc {
                        if c.active.load(Ordering::SeqCst) >= 2 {
                            flag.store(true, Ordering::SeqCst);
                            break;
                        }
                    }
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
            self.inner.read_at(offset, buf)
        }
    }

    /// Real registry + per-replica FailingSources; health marking flows
    /// through `slot()`. `drives` overrides the cached per-replica device
    /// identity (empty = use the real one) so multi-drive striping is
    /// testable on one filesystem.
    struct TestSources {
        reg: ReplicaRegistry,
        fail: Vec<(usize, usize)>,
        log: Arc<std::sync::Mutex<Vec<(usize, u64)>>>,
        drives: Vec<String>,
        /// Per-read throttle (0 = none); widens the pop window so multi-worker
        /// striping is deterministic.
        delay_ms: u64,
        /// Shared peak-concurrency tracker (None = untracked).
        conc: Option<Arc<Concurrency>>,
        /// Pair-achievement flag handed to every source (see
        /// [`FailingSource::wait_for_pair`]).
        wait_for_pair: Option<Arc<AtomicBool>>,
    }

    impl ShardSources for TestSources {
        fn slots(&self, shard_index: usize) -> Vec<Arc<dyn ChunkSource>> {
            self.reg.shards()[shard_index]
                .replicas
                .iter()
                .enumerate()
                .map(|(replica, slot)| {
                    // Every replica is logged; `fail` sets the per-replica
                    // failure budget (default: effectively never fails).
                    let nth = self
                        .fail
                        .iter()
                        .find(|(r, _)| *r == replica)
                        .map(|(_, n)| *n)
                        .unwrap_or(1_000_000);
                    Arc::new(FailingSource {
                        inner: Arc::clone(slot),
                        fail_nth: AtomicUsize::new(nth),
                        id: replica,
                        log: Arc::clone(&self.log),
                        delay_ms: self.delay_ms,
                        conc: self.conc.clone(),
                        wait_for_pair: self.wait_for_pair.clone(),
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
            conc: None,
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(0));
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
            conc: None,
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(0));
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

    #[test]
    fn failed_chunk_requeues_and_lands_on_a_healthy_replica() {
        // 2 replicas, replica 0 fails its first read; all chunks complete
        // WITHIN ONE wait() — the failed chunk was retried by replica 1 in
        // the same drain.
        let (_paths, reg) = fixture(2, "requeue");
        let log = Arc::new(std::sync::Mutex::new(Vec::new()));
        let sources = TestSources {
            reg,
            fail: vec![(0, 0)],
            log,
            drives: Vec::new(),
            delay_ms: 0,
            conc: None,
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(1));
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
            fail: vec![(0, 1_000_000)],
            log,
            drives: Vec::new(),
            delay_ms: 0,
            conc: None,
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(1));
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
        let bytes = crate::stripe::CHUNK + 8192; // 2 chunks: 2 MiB + 8 KiB
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
            conc: None,
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(0)); // 0 = derived, one worker per drive
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
            conc: None,
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(1));
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
            conc: Some(Arc::clone(&conc)),
            wait_for_pair: Some(Arc::clone(&pair)),
        };
        let queue = Arc::new(ChunkQueue::new(4));
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
            conc: None,
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(1));
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
            conc: None,
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(1));
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
            conc: None,
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(1));
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
            conc: None,
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(1));
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
            fail: vec![(0, usize::MAX)],
            log,
            drives: Vec::new(),
            delay_ms: 0,
            conc: None,
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(1));
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
            fail: vec![(0, 0), (1, 0), (2, 0)],
            log,
            drives: Vec::new(),
            delay_ms: 0,
            conc: None,
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(4));
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
            conc: None,
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(1));
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
            conc: None,
            wait_for_pair: None,
        };
        let queue = Arc::new(ChunkQueue::new(2));
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
            conc: None,
            wait_for_pair: None,
        });
        let queue = Arc::new(ChunkQueue::new(4));
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
}
