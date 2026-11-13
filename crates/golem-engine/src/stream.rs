//! Prefill-stream byte sources (design §7.6, ABI v10): the pinned staging
//! ring, the reader threads, and the two stream trampolines C calls per
//! plane. Decode hot set isolation (§7.6.4): pooled hits lease a resident
//! slot WITHOUT scoring or paging in ([`ExpertPool::lease_resident`]);
//! misses are read into staging slots, never into pool slots — a streamed
//! ubatch changes no pool residency or score.
//!
//! Pipeline (§7.6.3): the routing readback of layer L WARMS its up/down
//! planes (routed-only) and layer L+1's gate plane (its ids are not known
//! before its router runs): whole under [`PrefillGate::Whole`]; under
//! [`PrefillGate::Predicted`] the previous ubatch's layer-L+1 routed set,
//! with L+1's own readback staging the misses into the same slot. The copy
//! loop's `stream_source` consumes the warm plane and parks on its
//! completion only when the disk is the bound. Slots
//! recycle through `stream_release(seq)` (C releases seq−2 after the slab's
//! event sync) and the `copy_done` backstop releases everything left.
#![allow(unsafe_code)]

use std::collections::VecDeque;
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use golem_placement::pool::{
    Arena, DIRECT_ALIGN, ExpertKind, ExpertPool, ExpertSource, Lease, SlotClass,
};
use golem_store::recycle;

use crate::config::PrefillGate;
use crate::p3_glue::p3_fatal;

use golem_plan::STAGING_SLOTS;

/// Reader threads: one per warm plane in flight (≤ 3 ahead of the copy
/// loop), so the next plane's drain starts while the previous one's tail
/// still runs and the drives never drain to zero between planes. Regions
/// are disjoint per expert and slots are counted per slot, so readers
/// complete in any order.
const READERS: usize = 3;

/// Cumulative stream counters (§7.6/§11 observability).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct StreamStats {
    /// Host→device expert bytes the copy loop streamed (sums C's copies of
    /// the pointers this module handed out).
    pub bytes: u64,
    /// Per-expert plane hits served from the pool.
    pub pool_hits: u64,
    /// Per-expert ranges served from staging.
    pub staged: u64,
    /// stream_source calls (one per streamed plane).
    pub planes: u64,
    /// Σ ids over successfully sourced planes (the ids C copies H2D).
    pub sourced_ids: u64,
    /// Planes currently holding pool leases or a staging slot (sourced and
    /// not yet released).
    pub held: u64,
    /// Bytes the stream readers read into staging (page-rounded regions,
    /// including warmed planes C never sourced); failed reads excluded.
    pub read_bytes: u64,
    /// Plane bytes of per-expert ranges served from the pool (a subset of
    /// `bytes`).
    pub pool_bytes: u64,
    /// Routed gate ids a predicted gate warm missed (resident or staged at
    /// the layer's own readback; 0 under [`PrefillGate::Whole`]).
    pub gate_missed: u64,
    /// Σ ns reader threads spent in plane reads.
    pub reader_read_ns: u64,
    /// Σ ns the decode thread (`warm` or `source`) waited for a free
    /// staging slot.
    pub slot_wait_ns: u64,
    /// Σ ns the copy loop waited in `source` for its plane's reads.
    pub source_wait_ns: u64,
    /// Store-wide drains (stream, decode and prefetch) run serially on the
    /// caller because every crew was busy (0 without a store).
    pub serial_fallbacks: u64,
}

/// One warmed plane's misses, read as one batched drain so the plane's
/// experts spread across every drive's reader threads (§5.2).
struct Task {
    layer: u32,
    kind: ExpertKind,
    /// Slot base; expert `e` lands at `plane_at[row + e]` past it.
    base: *mut u8,
    /// Staged ids, ascending.
    ids: Vec<u32>,
    slot: usize,
}

struct Hold {
    slot: usize,
    leases: Vec<Lease>,
}

/// A warmed plane: its staging slot (if it had misses) and the miss ids the
/// reader was told to fill. Created by `warm` at the routing readback,
/// consumed by `source` in the copy loop, dropped unreferenced by
/// `release_all` between stream chunks.
struct Pending {
    slot: usize,
    /// Slot base (valid when `slot != usize::MAX`).
    base: *mut u8,
    /// Staged (non-resident at warm time) ids, ascending.
    staged_ids: Vec<u32>,
    /// Every id warmed for this plane (routed set), ascending.
    warmed_ids: Vec<u32>,
    /// A predicted gate whose routed misses are not staged yet: `source`
    /// tops it up with C's ids, so a routed id is never handed stale bytes.
    needs_top_up: bool,
}

/// Reader outcome since the last `release_all`; the worst failure wins.
#[derive(Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ReadState {
    #[default]
    Ok,
    /// A read failed with the store already `FatalIo`: the turn fails.
    StorageLost,
    /// Any other read failure or a reader panic: a contract breach.
    Fatal,
}

/// A streamed plane cannot be sourced because the store lost every replica.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamError {
    /// A reader failure with the store's health already `FatalIo`.
    StorageLost,
}

#[derive(Default)]
struct Inner {
    free_slots: Vec<usize>,
    /// Outstanding reader tasks (plane batches) per slot; a slot is only
    /// handed out when its count is 0.
    pending_reads: [usize; STAGING_SLOTS as usize],
    queue: VecDeque<Task>,
    holds: BTreeMap<u64, Hold>,
    /// Keyed (pool layer, kind code − 1).
    warm: HashMap<(u32, u8), Pending>,
    /// Per layer, the last streamed ubatch's routed ids, ascending (the
    /// predicted gate set; empty = no history yet).
    last_routed: Vec<Vec<u32>>,
    read: ReadState,
    stop: bool,
}

/// The per-session stream state. Shared with C as part of the hook payload;
/// the decode thread drives `warm`/`source`/`release`, the reader threads
/// consumes `queue` — the `Inner` mutex + condvar are the only shared state
/// and the happens-before edges for the staging bytes.
pub struct StreamState {
    pool: Arc<ExpertPool>,
    source: Arc<dyn ExpertSource>,
    /// Whether the source's health is `FatalIo` (false for sources without
    /// a health, e.g. GGUF pread).
    storage_lost: Box<dyn Fn() -> bool + Send + Sync>,
    arena: Arc<Arena>,
    slot_bytes: usize,
    /// Plane byte length per layer × kind (flat, from the geometry).
    plane_lens: Vec<u32>,
    /// Staged plane offset inside a slot per layer × kind × expert (flat,
    /// see [`Self::row`]). Each expert owns a page-aligned region of
    /// `head + plane_len` rounded up to a page, its plane `head` = file
    /// offset mod page bytes in, so the covering read lands in place (§5.4)
    /// and no two regions share a page: a plane batch's targets are
    /// disjoint. `plane_at % DIRECT_ALIGN` is the expert's head.
    plane_at: Vec<usize>,
    n_layers: u32,
    n_experts: u32,
    gate: PrefillGate,
    /// C's routed-share switch: a predicted set above it warms whole.
    routed_max: f32,
    inner: Mutex<Inner>,
    cv: Condvar,
    /// Each reader holds an `Arc<Self>`, so `Drop` never runs while one
    /// lives: `stop` joins them, releasing the pool and arena with the session.
    readers: Mutex<Vec<std::thread::JoinHandle<()>>>,
    /// H2D expert bytes this module's pointers fed (the copy loop's copies).
    pub bytes: AtomicU64,
    /// Per-expert plane hits served from the pool.
    pub pool_hits: AtomicU64,
    /// Plane bytes of pool hits.
    pool_bytes: AtomicU64,
    /// Per-expert ranges served from staging.
    pub staged: AtomicU64,
    /// Bytes successfully read into staging.
    read_bytes: AtomicU64,
    /// Routed gate ids a predicted warm missed.
    gate_missed: AtomicU64,
    reader_read_ns: AtomicU64,
    slot_wait_ns: AtomicU64,
    source_wait_ns: AtomicU64,
    /// stream_source calls (one per streamed plane).
    pub planes: AtomicU64,
    /// Σ ids over successfully sourced planes.
    sourced_ids: AtomicU64,
    /// Mirror of `Inner::holds.len()`, stored under the lock at every
    /// insert/remove so `stats()` never takes it.
    held: AtomicU64,
    /// Test seam: runs on a reader thread right before its next read.
    #[cfg(any(test, feature = "test-support"))]
    test_before_read: Mutex<Option<Box<dyn FnOnce() + Send>>>,
}

// SAFETY: `Task.base`/`Pending.base` point into the mlocked arena the state
// owns (the `arena` field keeps it alive); no raw pointer dangles. All uses
// are serialized by the seq protocol: a slot is handed out only when its
// reads are done and its previous plane's H2D + compute completed under C's
// slab event sync.
unsafe impl Send for StreamState {}
unsafe impl Sync for StreamState {}

impl StreamState {
    /// Builds the staging ring (`STAGING_SLOTS` slots, mlocked) and spawns
    /// the reader threads. `slot_bytes` is a minimum: each slot grows to fit
    /// the page-aligned per-expert regions of the largest plane.
    ///
    /// # Errors
    /// Arena mmap/mlock failure (no unlocked fallback, §5.4).
    pub fn new(
        pool: Arc<ExpertPool>,
        source: Arc<dyn ExpertSource>,
        storage_lost: Box<dyn Fn() -> bool + Send + Sync>,
        slot_bytes: usize,
        gate: PrefillGate,
        routed_max: f32,
    ) -> Result<Arc<Self>, String> {
        let geo = &pool.geometry;
        let page = DIRECT_ALIGN as usize;
        let mut plane_lens = Vec::with_capacity(geo.layers.len() * 3);
        let mut plane_at = Vec::with_capacity(geo.layers.len() * 3 * geo.n_experts as usize);
        let mut need = 0usize;
        for (layer, unit) in geo.layers.iter().enumerate() {
            for kind in [ExpertKind::Gate, ExpertKind::Up, ExpertKind::Down] {
                let (_, stride) = unit.plane(kind);
                plane_lens.push(u32::try_from(stride).map_err(|_| "plane stride > u32")?);
                let mut at = 0usize;
                for e in 0..geo.n_experts {
                    let head = source
                        .plane(layer as u32, kind, e)
                        .map_or(unit.head[kind.ordinal()], |(off, _)| off % DIRECT_ALIGN)
                        as usize;
                    plane_at.push(at + head);
                    at += (head + stride as usize).next_multiple_of(page);
                }
                need = need.max(at);
            }
        }
        // Each region costs at most two pages over its plane. DeepSeek-V4.1-
        // Flash UD-IQ2_XXS: ≤ 384 × 8 KiB = 3 MiB per slot on 384 × 4.52 MB
        // = 1.74 GB, × 6 slots ≤ 18 MiB; GLM-5.3-Flash UD-IQ3_XXS (page-
        // multiple strides): 288 × 4 KiB = 1.1 MiB per slot.
        let slot_bytes = slot_bytes.max(need).next_multiple_of(page);
        let arena = Arena::new(&[SlotClass {
            slot_size: slot_bytes,
            slots: STAGING_SLOTS,
        }])
        .map_err(|e| e.to_string())?;
        let last_routed = (0..geo.layers.len())
            .map(|_| Vec::with_capacity(geo.n_experts as usize))
            .collect();
        let state = Arc::new(Self {
            n_experts: geo.n_experts,
            n_layers: geo.layers.len() as u32,
            gate,
            routed_max,
            pool,
            source,
            storage_lost,
            arena: Arc::new(arena),
            slot_bytes,
            plane_lens,
            plane_at,
            inner: Mutex::new(Inner {
                free_slots: (0..STAGING_SLOTS as usize).collect(),
                pending_reads: [0; STAGING_SLOTS as usize],
                queue: VecDeque::new(),
                holds: BTreeMap::new(),
                warm: HashMap::new(),
                last_routed,
                read: ReadState::Ok,
                stop: false,
            }),
            cv: Condvar::new(),
            readers: Mutex::new(Vec::with_capacity(READERS)),
            bytes: AtomicU64::new(0),
            pool_hits: AtomicU64::new(0),
            pool_bytes: AtomicU64::new(0),
            staged: AtomicU64::new(0),
            read_bytes: AtomicU64::new(0),
            gate_missed: AtomicU64::new(0),
            reader_read_ns: AtomicU64::new(0),
            slot_wait_ns: AtomicU64::new(0),
            source_wait_ns: AtomicU64::new(0),
            planes: AtomicU64::new(0),
            sourced_ids: AtomicU64::new(0),
            held: AtomicU64::new(0),
            #[cfg(any(test, feature = "test-support"))]
            test_before_read: Mutex::new(None),
        });
        state.spawn_readers();
        Ok(state)
    }

    /// The arena's extent, for `hipHostRegister` (§3.5: pinned staging is
    /// load-bearing for full-bandwidth H2D) and §10 RAM accounting.
    pub fn arena_ptr_range(&self) -> (*mut u8, usize) {
        self.arena.as_ptr_range()
    }

    /// Total staging bytes (`slot_bytes × STAGING_SLOTS`, §10 RAM term).
    pub fn staged_bytes(&self) -> usize {
        self.slot_bytes * STAGING_SLOTS as usize
    }

    fn spawn_readers(self: &Arc<Self>) {
        let mut readers = self.readers.lock().unwrap_or_else(|e| e.into_inner());
        for _ in 0..READERS {
            let state = self.clone();
            readers.push(
                std::thread::Builder::new()
                    .name("golem-stream-reader".into())
                    .spawn(move || state.reader_loop())
                    .expect("spawn stream reader"),
            );
        }
    }

    fn reader_loop(self: Arc<Self>) {
        let page = DIRECT_ALIGN as usize;
        let mut batch: Vec<(ExpertKind, u32)> = Vec::new();
        let mut targets_buf: Vec<(&mut [u8], usize)> = Vec::new();
        loop {
            let task = {
                let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                loop {
                    if inner.stop {
                        return;
                    }
                    if let Some(t) = inner.queue.pop_front() {
                        break t;
                    }
                    inner = self.cv.wait(inner).unwrap_or_else(|e| e.into_inner());
                }
            };
            // Read OUTSIDE the lock: the disk pass is the pipeline's bound.
            let plane_len = self.plane_len(task.layer, task.kind);
            let at = &self.plane_at[self.row(task.layer, task.kind)..];
            batch.clear();
            let mut targets: Vec<(&mut [u8], usize)> = recycle(std::mem::take(&mut targets_buf));
            for &id in &task.ids {
                let off = at[id as usize];
                let head = off % page;
                batch.push((task.kind, id));
                // SAFETY: the region `off - head .. + (head + plane_len)
                // rounded to a page` is inside the slot (`new` sized the slot
                // to the furthest region end) and no two experts' regions
                // overlap (`plane_at` lays them out ascending, page-aligned),
                // so the slices are disjoint. The slot is exclusively the
                // readers' until pending hits 0; C reads only after that.
                // Other readers write other slots or other experts' regions.
                let region = unsafe {
                    std::slice::from_raw_parts_mut(
                        task.base.add(off - head),
                        (head + plane_len).next_multiple_of(page),
                    )
                };
                targets.push((region, head));
            }
            let task_bytes: u64 = targets.iter().map(|(r, _)| r.len() as u64).sum();
            #[cfg(any(test, feature = "test-support"))]
            if let Some(f) = self
                .test_before_read
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take()
            {
                f();
            }
            // A panicking source must not strand pending_reads (every
            // waiter would hang): it counts as a fatal read. The store
            // degrades replicas inside the drain, so its health is final by
            // the time it returns.
            let reading = Instant::now();
            let read = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                match self
                    .source
                    .read_expert_batched(task.layer, &batch, &mut targets)
                {
                    Ok(()) => ReadState::Ok,
                    Err(_) if (self.storage_lost)() => ReadState::StorageLost,
                    Err(_) => ReadState::Fatal,
                }
            }))
            .unwrap_or(ReadState::Fatal);
            add_ns(&self.reader_read_ns, reading);
            if read == ReadState::Ok {
                self.read_bytes.fetch_add(task_bytes, Ordering::Relaxed);
            }
            targets_buf = recycle(targets);
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.read = inner.read.max(read);
            inner.pending_reads[task.slot] -= 1;
            self.cv.notify_all();
        }
    }

    fn plane_len(&self, layer: u32, kind: ExpertKind) -> usize {
        self.plane_lens[(layer as usize) * 3 + kind as usize] as usize
    }

    /// First `plane_at` index of `(layer, kind)`.
    fn row(&self, layer: u32, kind: ExpertKind) -> usize {
        ((layer as usize) * 3 + kind as usize) * self.n_experts as usize
    }

    /// The routing-readback read-ahead (decode thread, once per layer per
    /// streamed chunk): warms `layer`'s up/down planes routed-only and the
    /// NEXT layer's gate plane (its ids are unknown until its router runs):
    /// whole, or under [`PrefillGate::Predicted`] the previous ubatch's
    /// routed set for that layer (whole without history or above the
    /// routed-share switch), this layer's gate misses topped up first.
    /// Resident experts are skipped (the pool copy is authoritative and
    /// `source` will lease it); no lease is taken here — the prefetcher is
    /// paused for the chunk and no demand claims run, so nothing can evict
    /// between warm and source.
    /// A failed read stops the warm-up; the next `source` reports it.
    pub fn warm(&self, layer: u32, routed_ids: &[i32]) {
        if layer >= self.n_layers {
            return;
        }
        let mut routed: Vec<u32> = routed_ids
            .iter()
            .filter(|&&id| id >= 0 && (id as u32) < self.n_experts)
            .map(|&id| id as u32)
            .collect();
        let predicted = self.gate == PrefillGate::Predicted;
        let whole = || (0..self.n_experts).collect::<Vec<u32>>();
        let next = if !predicted {
            (layer + 1 < self.n_layers).then(whole)
        } else {
            routed.sort_unstable();
            routed.dedup();
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            let mine = &mut inner.last_routed[layer as usize];
            mine.clear();
            mine.extend_from_slice(&routed);
            (layer + 1 < self.n_layers).then(|| {
                let guess = &inner.last_routed[layer as usize + 1];
                if guess.is_empty() || guess.len() as f32 / self.n_experts as f32 > self.routed_max
                {
                    whole()
                } else {
                    guess.clone()
                }
            })
        };
        // Errors are dropped: `source` reports them and `release_all`
        // reclaims warmed planes. This layer's gate misses go first: its
        // gate is sourced before up/down.
        let _ = (if predicted {
            self.top_up_gate(layer, &routed)
        } else {
            Ok(())
        })
        .and_then(|()| self.warm_plane(layer, ExpertKind::Up, &routed))
        .and_then(|()| self.warm_plane(layer, ExpertKind::Down, &routed))
        .and_then(|()| match &next {
            Some(ids) => {
                self.warm_plane(layer + 1, ExpertKind::Gate, ids)?;
                if predicted && ids.len() < self.n_experts as usize {
                    let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(p) = inner.warm.get_mut(&(layer + 1, ExpertKind::Gate as u8)) {
                        p.needs_top_up = true;
                    }
                }
                Ok(())
            }
            None => Ok(()),
        });
    }

    /// Stages `layer`'s routed gate ids its predicted warm missed into the
    /// same slot (regions are per expert, so the second batch is disjoint
    /// from the first). With no predicted warm (layer 0, first ubatch) the
    /// gate is warmed routed-only now.
    fn top_up_gate(&self, layer: u32, routed: &[u32]) -> Result<(), StreamError> {
        let key = (layer, ExpertKind::Gate as u8);
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        check_read(inner.read)?;
        let Some(pending) = inner.warm.get(&key) else {
            drop(inner);
            return self.warm_plane(layer, ExpertKind::Gate, routed);
        };
        let missed: Vec<u32> = routed
            .iter()
            .copied()
            .filter(|id| pending.warmed_ids.binary_search(id).is_err())
            .collect();
        if missed.is_empty() {
            if let Some(p) = inner.warm.get_mut(&key) {
                p.needs_top_up = false;
            }
            return Ok(());
        }
        self.gate_missed
            .fetch_add(missed.len() as u64, Ordering::Relaxed);
        let staged: Vec<u32> = missed
            .iter()
            .copied()
            .filter(|&id| !self.pool.is_resident(layer, id))
            .collect();
        let (slot, base) = if staged.is_empty() || pending.slot != usize::MAX {
            (pending.slot, pending.base)
        } else {
            let (guard, slot, base) = self.take_slot(inner)?;
            inner = guard;
            (slot, base)
        };
        if !staged.is_empty() {
            inner.pending_reads[slot] += 1;
            inner.queue.push_back(Task {
                layer,
                kind: ExpertKind::Gate,
                base,
                ids: staged.clone(),
                slot,
            });
            self.cv.notify_all();
        }
        let pending = inner.warm.get_mut(&key).expect("checked above");
        pending.slot = slot;
        pending.base = base;
        pending.staged_ids.extend_from_slice(&staged);
        pending.staged_ids.sort_unstable();
        pending.warmed_ids.extend_from_slice(&missed);
        pending.warmed_ids.sort_unstable();
        pending.needs_top_up = false;
        Ok(())
    }

    /// Errors only before a staging slot is popped: nothing to give back.
    fn warm_plane(&self, layer: u32, kind: ExpertKind, ids: &[u32]) -> Result<(), StreamError> {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        check_read(inner.read)?;
        let mut ids = ids.to_vec();
        ids.sort_unstable();
        ids.dedup();
        let staged_ids: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|&id| !self.pool.is_resident(layer, id))
            .collect();
        let (slot, base) = if staged_ids.is_empty() {
            (usize::MAX, std::ptr::null_mut())
        } else {
            let (guard, slot, base) = self.take_slot(inner)?;
            inner = guard;
            inner.pending_reads[slot] += 1;
            inner.queue.push_back(Task {
                layer,
                kind,
                base,
                ids: staged_ids.clone(),
                slot,
            });
            // The condvar is shared with decode-thread waiters: wake every
            // waiter so a reader is never the one passed over.
            self.cv.notify_all();
            (slot, base)
        };
        inner.warm.insert(
            (layer, kind as u8),
            Pending {
                slot,
                base,
                staged_ids,
                warmed_ids: ids,
                needs_top_up: false,
            },
        );
        Ok(())
    }

    /// Waits for a free staging slot (bounded by the release protocol: C
    /// releases seq−2 before refilling a half, copy_done drops the tail
    /// between chunks).
    fn take_slot<'a>(
        &self,
        mut inner: std::sync::MutexGuard<'a, Inner>,
    ) -> Result<(std::sync::MutexGuard<'a, Inner>, usize, *mut u8), StreamError> {
        let start = Instant::now();
        loop {
            check_read(inner.read).inspect_err(|_| add_ns(&self.slot_wait_ns, start))?;
            if inner.stop {
                p3_fatal("stream slot after stop".into());
            }
            if let Some(s) = inner.free_slots.pop() {
                debug_assert_eq!(inner.pending_reads[s], 0);
                add_ns(&self.slot_wait_ns, start);
                // Slots are indexed 0..STAGING_SLOTS in the arena.
                let base = self.arena.slot_base(s as u32);
                return Ok((inner, s, base));
            }
            inner = self.cv.wait(inner).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// The `stream_source` hook body (decode thread, once per streamed
    /// plane): pooled hits lease without scoring; misses come from the warm
    /// slot (or are staged now when `warm` never ran — e.g. the first
    /// layer's gate) and block on the reader's completion.
    ///
    /// `blk` is the DENSE-LEAD-ADJUSTED pool layer (the glue subtracts the
    /// lead before calling).
    ///
    /// # Errors
    /// [`StreamError::StorageLost`] when a reader failure left the plane
    /// unfilled; a slot this plane took is held under `seq`, so C's
    /// `stream_release(seq)` reclaims it once its reads finish.
    pub fn source(
        &self,
        blk: i32,
        kind_code: i32,
        seq: u64,
        ids: &[i32],
        out: &mut [*const c_void],
    ) -> Result<(), StreamError> {
        let Some(kind) = kind_from_code(kind_code) else {
            p3_fatal(format!("stream_source: bad kind {kind_code}"));
        };
        if blk < 0 || blk as u32 >= self.n_layers || ids.len() != out.len() {
            p3_fatal(format!(
                "stream_source: blk {blk}, ids {} vs out {}",
                ids.len(),
                out.len()
            ));
        }
        let layer = blk as u32;
        let plane_len = self.plane_len(layer, kind);
        let row = self.row(layer, kind);
        self.planes.fetch_add(1, Ordering::Relaxed);

        // Fallback: no warm plane (first layer's gate, or C reordered the
        // ids) — stage on demand with the same path.
        {
            let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if !inner.warm.contains_key(&(layer, kind as u8)) {
                drop(inner);
                let ids: Vec<u32> = ids
                    .iter()
                    .filter(|&&id| id >= 0 && (id as u32) < self.n_experts)
                    .map(|&id| id as u32)
                    .collect();
                self.warm_plane(layer, kind, &ids)?;
            } else if inner.warm[&(layer, kind as u8)].needs_top_up {
                // The layer's readback never topped this gate up: C's ids
                // are its routed set (or whole, above the share switch).
                drop(inner);
                let ids: Vec<u32> = ids
                    .iter()
                    .filter(|&&id| id >= 0 && (id as u32) < self.n_experts)
                    .map(|&id| id as u32)
                    .collect();
                self.top_up_gate(layer, &ids)?;
            }
        }
        let mut pending = {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner
                .warm
                .remove(&(layer, kind as u8))
                .expect("warm plane inserted above")
        };

        // Wait for the plane's reads (the pipeline: warm enqueued them one
        // layer ahead, so this parks only when the disk is the bound — the
        // intended steady state).
        {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if pending.slot != usize::MAX {
                let start = Instant::now();
                // `read` is store-global on purpose: any plane's loss ends
                // this wait, not just this slot's reads.
                while inner.pending_reads[pending.slot] > 0 && inner.read == ReadState::Ok {
                    inner = self.cv.wait(inner).unwrap_or_else(|e| e.into_inner());
                }
                add_ns(&self.source_wait_ns, start);
            }
            if let Err(e) = check_read(inner.read) {
                debug_assert!(!inner.holds.contains_key(&seq));
                inner.holds.insert(
                    seq,
                    Hold {
                        slot: pending.slot,
                        leases: Vec::new(),
                    },
                );
                self.held.store(inner.holds.len() as u64, Ordering::Relaxed);
                return Err(e);
            }
        }

        // Fill the pointers: pooled hits lease now (held until release, no
        // score, no page-in — §7.6.4); staged ids point into the warm slot.
        // An id outside the warmed set (whole fill over a routed-only warm)
        // points at the slot's stale but finite bytes — `MUL_MAT_ID` never
        // reads unrouted ranges (§7.6.6). A warmed id that is now neither
        // resident nor staged lost residency after warm: fatal, never stale.
        let mut leases: Vec<Lease> = Vec::with_capacity(ids.len());
        for (i, &id) in ids.iter().enumerate() {
            if id < 0 || id as u32 >= self.n_experts {
                p3_fatal(format!(
                    "stream_source: expert id {id} outside 0..{}",
                    self.n_experts
                ));
            }
            let id = id as u32;
            let staged =
                pending.slot != usize::MAX && pending.staged_ids.binary_search(&id).is_ok();
            if let Some((ptr, lease)) = self.pool.lease_resident(layer, kind, id) {
                leases.push(lease);
                self.pool_hits.fetch_add(1, Ordering::Relaxed);
                self.pool_bytes
                    .fetch_add(plane_len as u64, Ordering::Relaxed);
                self.bytes.fetch_add(plane_len as u64, Ordering::Relaxed);
                out[i] = ptr as *const c_void;
            } else if staged {
                self.staged.fetch_add(1, Ordering::Relaxed);
                self.bytes.fetch_add(plane_len as u64, Ordering::Relaxed);
                // SAFETY: `plane_at` offsets lie inside the slot (`new`).
                out[i] =
                    unsafe { pending.base.add(self.plane_at[row + id as usize]) } as *const c_void;
            } else if pending.warmed_ids.binary_search(&id).is_err() {
                // A whole fill over a routed-only warm with no misses has no
                // slot yet: take one with no reads for the stale pointers.
                if pending.slot == usize::MAX {
                    let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
                    let (_inner, slot, base) = self.take_slot(inner)?;
                    pending.slot = slot;
                    pending.base = base;
                }
                // SAFETY: as above; the bytes are stale (never staged this
                // plane) but finite, and the id is unrouted.
                out[i] =
                    unsafe { pending.base.add(self.plane_at[row + id as usize]) } as *const c_void;
            } else {
                // A warmed id lost residency after warm (the pause protocol
                // forbids it): fatal, never stale.
                p3_fatal(format!(
                    "stream_source: expert {id} of layer {layer} neither resident nor staged"
                ));
            }
        }
        // SAFETY: the mutex critical sections above are the happens-before
        // edges for the staging bytes C is about to read.
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        debug_assert!(!inner.holds.contains_key(&seq));
        inner.holds.insert(
            seq,
            Hold {
                slot: pending.slot,
                leases,
            },
        );
        self.held.store(inner.holds.len() as u64, Ordering::Relaxed);
        self.sourced_ids
            .fetch_add(ids.len() as u64, Ordering::Relaxed);
        Ok(())
    }

    /// The `stream_release` hook body: releases seq and everything smaller —
    /// leases drop, staging slots recycle. C calls this after the slab's
    /// event sync, so the H2D reading these host bytes has completed; the
    /// `copy_done` backstop runs after the compute-entry drain for the same
    /// reason. A plane held after a failed read may still have reads in
    /// flight: its slot recycles only once they finish.
    pub fn release(&self, seq: u64) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let done: Vec<u64> = inner.holds.range(..=seq).map(|(&s, _)| s).collect();
        for s in done {
            if let Some(h) = inner.holds.remove(&s) {
                self.held.store(inner.holds.len() as u64, Ordering::Relaxed);
                if h.slot != usize::MAX {
                    while inner.pending_reads[h.slot] > 0 {
                        inner = self.cv.wait(inner).unwrap_or_else(|e| e.into_inner());
                    }
                    inner.free_slots.push(h.slot);
                }
                drop(h.leases);
            }
        }
        self.cv.notify_all();
    }

    /// Releases everything and drops unconsumed warm planes (the copy_done
    /// backstop between stream chunks). A dropped warm plane's reads may
    /// still be in flight: its slot is reclaimed only once they finish
    /// (bounded by one plane drain) so the bytes can never be overwritten
    /// mid-read.
    pub fn release_all(&self) {
        let seq = {
            let inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.holds.keys().copied().max().unwrap_or(0)
        };
        self.release(seq);
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let dropped: Vec<Pending> = inner.warm.drain().map(|(_, p)| p).collect();
        for p in dropped {
            if p.slot != usize::MAX {
                while inner.pending_reads[p.slot] > 0 {
                    inner = self.cv.wait(inner).unwrap_or_else(|e| e.into_inner());
                }
                inner.free_slots.push(p.slot);
            }
        }
        // Nothing is in flight any more. A fatal read is sticky: one on a
        // warmed plane never consumed still aborts here; only StorageLost
        // resets for the next turn.
        if inner.read == ReadState::Fatal {
            drop(inner);
            p3_fatal("stream reader failed on an unconsumed plane; aborting".into());
        }
        inner.read = ReadState::Ok;
        self.cv.notify_all();
    }

    /// Snapshot of the counters.
    pub fn stats(&self) -> StreamStats {
        StreamStats {
            bytes: self.bytes.load(Ordering::Relaxed),
            pool_hits: self.pool_hits.load(Ordering::Relaxed),
            staged: self.staged.load(Ordering::Relaxed),
            planes: self.planes.load(Ordering::Relaxed),
            sourced_ids: self.sourced_ids.load(Ordering::Relaxed),
            held: self.held.load(Ordering::Relaxed),
            read_bytes: self.read_bytes.load(Ordering::Relaxed),
            pool_bytes: self.pool_bytes.load(Ordering::Relaxed),
            gate_missed: self.gate_missed.load(Ordering::Relaxed),
            reader_read_ns: self.reader_read_ns.load(Ordering::Relaxed),
            slot_wait_ns: self.slot_wait_ns.load(Ordering::Relaxed),
            source_wait_ns: self.source_wait_ns.load(Ordering::Relaxed),
            serial_fallbacks: 0,
        }
    }

    /// Test seam: `f` runs on a reader thread before its next read.
    #[cfg(any(test, feature = "test-support"))]
    pub(crate) fn before_next_read_for_test(&self, f: Box<dyn FnOnce() + Send>) {
        *self
            .test_before_read
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(f);
    }

    /// Stops and joins the reader threads (session teardown; holds are empty
    /// once the hooks are unregistered and copy_done ran).
    pub fn stop(&self) {
        {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.stop = true;
            self.cv.notify_all();
        }
        let handles = std::mem::take(&mut *self.readers.lock().unwrap_or_else(|e| e.into_inner()));
        for h in handles {
            if h.thread().id() != std::thread::current().id() {
                let _ = h.join();
            }
        }
    }
}

fn add_ns(counter: &AtomicU64, since: Instant) {
    counter.fetch_add(
        u64::try_from(since.elapsed().as_nanos()).unwrap_or(u64::MAX),
        Ordering::Relaxed,
    );
}

fn check_read(read: ReadState) -> Result<(), StreamError> {
    match read {
        ReadState::Ok => Ok(()),
        ReadState::StorageLost => Err(StreamError::StorageLost),
        ReadState::Fatal => {
            p3_fatal("stream reader failed; aborting rather than decoding on garbage".into())
        }
    }
}

fn kind_from_code(code: i32) -> Option<ExpertKind> {
    match code {
        1 => Some(ExpertKind::Gate),
        2 => Some(ExpertKind::Up),
        3 => Some(ExpertKind::Down),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use golem_placement::pool::{
        P3Stats, PolicySnapshot, PoolGeometry, SnapshotHandle, UnitLayout,
    };
    use std::sync::atomic::AtomicBool;

    const STRIDE: u64 = 64;
    const N_EXPERTS: u32 = 4;

    /// Fails every read of `fail` (layer, expert), marking storage lost first
    /// like the store degrading its replicas inside `read_expert`.
    struct FailSource {
        fail: (u32, u32),
        lost: Arc<AtomicBool>,
    }

    impl ExpertSource for FailSource {
        fn read_expert(
            &self,
            layer: u32,
            _kind: ExpertKind,
            expert: u32,
            into: &mut [u8],
            head: usize,
        ) -> std::io::Result<()> {
            if (layer, expert) == self.fail {
                self.lost.store(true, Ordering::SeqCst);
                return Err(std::io::Error::other("replicas gone"));
            }
            into[head..].fill(expert as u8);
            Ok(())
        }

        fn plane(&self, _layer: u32, _kind: ExpertKind, _expert: u32) -> Option<(u64, u64)> {
            None
        }
    }

    fn state(fail: (u32, u32)) -> Arc<StreamState> {
        state_with(fail, PrefillGate::Whole)
    }

    fn state_with(fail: (u32, u32), gate: PrefillGate) -> Arc<StreamState> {
        let lost = Arc::new(AtomicBool::new(false));
        let source: Arc<dyn ExpertSource> = Arc::new(FailSource {
            fail,
            lost: lost.clone(),
        });
        let arena = Arena::new(&[SlotClass {
            slot_size: 3 * STRIDE as usize,
            slots: 4,
        }])
        .unwrap();
        let pool = ExpertPool::new(
            arena,
            source.clone(),
            Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::empty()))),
            Arc::new(P3Stats::new()),
            PoolGeometry {
                layers: vec![UnitLayout::new([STRIDE; 3], [None; 3]).unwrap(); 2],
                n_experts: N_EXPERTS,
                expert_bytes: 0,
            },
            vec![0; 2],
        );
        let storage_lost = Box::new(move || lost.load(Ordering::SeqCst));
        StreamState::new(
            pool,
            source,
            storage_lost,
            N_EXPERTS as usize * STRIDE as usize,
            gate,
            0.85,
        )
        .unwrap()
    }

    fn all_slots_free(s: &StreamState) -> bool {
        let inner = s.inner.lock().unwrap();
        inner.free_slots.len() == STAGING_SLOTS as usize && inner.holds.is_empty()
    }

    #[test]
    fn storage_loss_on_second_expert_fails_the_plane_and_release_reclaims_it() {
        let s = state((0, 1));
        let ids = [0, 1, 2, 3];
        let mut out = [std::ptr::null(); 4];
        assert_eq!(
            s.source(0, 1, 1, &ids, &mut out),
            Err(StreamError::StorageLost)
        );
        assert_eq!(s.stats().held, 1, "the failed plane's slot is held by seq");
        s.release(1);
        assert!(all_slots_free(&s));
        s.release_all();
        s.stop();
    }

    #[test]
    fn storage_loss_seen_by_warm_first_does_not_abort() {
        let s = state((0, 1));
        s.warm(0, &[1]);
        let mut out = [std::ptr::null(); 1];
        assert_eq!(
            s.source(0, 2, 1, &[1], &mut out),
            Err(StreamError::StorageLost)
        );
        // The read-ahead now observes the failure and returns quietly.
        s.warm(0, &[1]);
        assert_eq!(
            s.source(0, 3, 2, &[1], &mut out),
            Err(StreamError::StorageLost)
        );
        s.release(2);
        s.release_all();
        assert!(all_slots_free(&s));
        s.stop();
    }

    #[test]
    fn storage_loss_resets_at_the_boundary_for_the_next_turn() {
        let s = state((0, 1));
        let mut out = [std::ptr::null(); 1];
        assert_eq!(
            s.source(0, 1, 1, &[1], &mut out),
            Err(StreamError::StorageLost)
        );
        s.release_all();
        s.warm(0, &[2]);
        assert_eq!(s.source(0, 1, 2, &[2], &mut out), Ok(()));
        s.release_all();
        assert!(all_slots_free(&s));
        s.stop();
    }

    /// Bytes `id` of a sourced pointer: `FailSource` fills each staged
    /// plane with its expert id.
    fn staged_byte(p: *const c_void) -> u8 {
        // SAFETY: test-only; `p` points into the live staging arena.
        unsafe { *(p as *const u8) }
    }

    #[test]
    fn predicted_gate_warms_last_routing_and_tops_up_misses() {
        let s = state_with((9, 9), PrefillGate::Predicted);
        let mut out = [std::ptr::null(); 2];
        // First ubatch: no history, layer 1's gate warms whole. Regions:
        // gate0 {1} + up0/down0 {1} + gate1 whole (4) + up1/down1 {2} = 9.
        s.warm(0, &[1]);
        s.warm(1, &[2]);
        s.source(1, 1, 1, &[2], &mut out[..1]).unwrap();
        assert_eq!(staged_byte(out[0]), 2);
        s.release_all();
        let region = s.stats().read_bytes / 9;
        assert_eq!(s.stats().read_bytes, 9 * region);
        assert_eq!(s.stats().gate_missed, 0);

        // Same routing: layer 1's gate warms {2} only — 6 regions.
        s.warm(0, &[1]);
        s.warm(1, &[2]);
        s.release_all();
        assert_eq!(s.stats().read_bytes, 15 * region);

        // 3 is a miss, staged at layer 1's own readback into the same slot:
        // gate0 + up0/down0 + gate1 {2} + miss {3} + up1/down1 {2, 3} = 9.
        s.warm(0, &[1]);
        s.warm(1, &[2, 3]);
        s.source(1, 1, 2, &[2, 3], &mut out).unwrap();
        assert_eq!((staged_byte(out[0]), staged_byte(out[1])), (2, 3));
        assert_eq!(s.stats().gate_missed, 1);
        s.release_all();
        assert_eq!(s.stats().read_bytes, 24 * region);
        assert!(all_slots_free(&s));
        s.stop();
    }

    #[test]
    fn predicted_gate_without_its_readback_tops_up_in_source() {
        let s = state_with((9, 9), PrefillGate::Predicted);
        s.warm(0, &[0]);
        s.warm(1, &[2]);
        s.release_all();
        // Layer 1's readback never fires: source stages the miss itself.
        s.warm(0, &[0]);
        let mut out = [std::ptr::null(); 1];
        s.source(1, 1, 1, &[3], &mut out).unwrap();
        assert_eq!(staged_byte(out[0]), 3);
        assert_eq!(s.stats().gate_missed, 1);
        s.release_all();
        assert!(all_slots_free(&s));
        s.stop();
    }

    #[test]
    fn predicted_gate_over_the_share_switch_warms_whole() {
        let s = state_with((9, 9), PrefillGate::Predicted);
        s.warm(0, &[0]);
        s.warm(1, &[0, 1, 2, 3]);
        s.release_all();
        s.warm(0, &[0]);
        {
            let inner = s.inner.lock().unwrap();
            let p = &inner.warm[&(1, ExpertKind::Gate as u8)];
            assert_eq!(p.warmed_ids, vec![0, 1, 2, 3]);
        }
        s.release_all();
        s.stop();
    }

    #[test]
    fn stats_held_mirrors_holds_len() {
        let s = state((9, 9));
        let len = |s: &StreamState| s.inner.lock().unwrap().holds.len() as u64;
        let mut out = [std::ptr::null(); 1];
        for seq in 1..=3 {
            s.source(0, 1, seq, &[seq as i32], &mut out).unwrap();
            assert_eq!(s.stats().held, len(&s));
        }
        assert_eq!(s.stats().held, 3);
        s.release(2);
        assert_eq!((s.stats().held, len(&s)), (1, 1));
        s.release_all();
        assert_eq!((s.stats().held, len(&s)), (0, 0));
        s.stop();
    }

    #[test]
    fn whole_fill_over_all_resident_routed_warm_takes_a_slot() {
        let s = state((9, 9));
        s.pool.ensure_unit(true, 0, 0).unwrap();
        s.warm(0, &[0]);
        let mut out = [std::ptr::null(); N_EXPERTS as usize];
        s.source(0, 2, 1, &[0, 1, 2, 3], &mut out).unwrap();
        let (lo, len) = s.arena_ptr_range();
        let arena = lo as usize..lo as usize + len;
        assert!(out[1..].iter().all(|&p| arena.contains(&(p as usize))));
        assert_eq!((s.stats().pool_hits, s.stats().staged), (1, 0));
        assert_eq!(s.stats().pool_bytes, s.plane_len(0, ExpertKind::Up) as u64);
        assert_eq!(s.stats().held, 1);
        s.release(1);
        assert_eq!(s.stats().held, 0);
        s.release_all();
        assert!(all_slots_free(&s));
        s.stop();
    }

    /// Records each read's head and region length and writes what a file
    /// of back-to-back planes holds there: the previous expert's tail before
    /// the plane, the plane's id, then the next expert's head bytes.
    struct HeadSource(Mutex<Vec<(usize, usize)>>);

    impl ExpertSource for HeadSource {
        fn read_expert(
            &self,
            _layer: u32,
            _kind: ExpertKind,
            expert: u32,
            into: &mut [u8],
            head: usize,
        ) -> std::io::Result<()> {
            self.0.lock().unwrap().push((head, into.len()));
            let end = head + PAGE_STRIDE as usize;
            into[..head].fill(expert.wrapping_sub(1) as u8);
            into[head..end].fill(expert as u8);
            into[end..].fill((expert as u8).wrapping_add(1));
            Ok(())
        }

        fn plane(&self, _layer: u32, _kind: ExpertKind, _expert: u32) -> Option<(u64, u64)> {
            None
        }
    }

    const PAGE_STRIDE: u64 = DIRECT_ALIGN;

    /// End to end over the real store source (§5.4): a split-free GGUF whose
    /// expert tensors start off a page boundary with 4 KiB strides (the
    /// GLM-5.3-Flash shape: bases 32–4000 B past a page, page-multiple
    /// strides). A streamed plane lands byte-exact with zero scratch copies.
    #[test]
    fn store_backed_staging_reads_every_plane_in_place() {
        use golem_placement::pool::{GgufPreadSource, StoreExpertSource};
        use golem_store::ChunkQueue;
        use xtask::gguf::{GgufKv, GgufTensorMeta, GgufValue, write_gguf};

        const N: u64 = 4;
        // F32 rows of 1024 elems = 4096 B per expert.
        let stride = DIRECT_ALIGN;
        let byte = |k: usize, e: u64, i: u64| (k as u64 * 64 + e * 8 + i % 7) as u8;
        let dir = std::env::temp_dir().join(format!("golem-stream-store-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = "m.gguf";
        let kvs = vec![
            GgufKv {
                key: "general.architecture".into(),
                value: GgufValue::Str("glm4moe".into()),
            },
            GgufKv {
                key: "glm4moe.leading_dense_block_count".into(),
                value: GgufValue::U32(1),
            },
        ];
        // A small leading tensor pushes the expert bases off the page.
        let mut tensors = vec![(
            GgufTensorMeta {
                name: "blk.0.attn_norm.weight".into(),
                dims: vec![8, 1, 1],
                ggml_type: 0,
            },
            vec![0u8; 32],
        )];
        for (k, kind) in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"]
            .iter()
            .enumerate()
        {
            let payload = (0..N)
                .flat_map(|e| (0..stride).map(move |i| byte(k, e, i)))
                .collect();
            tensors.push((
                GgufTensorMeta {
                    name: format!("blk.1.{kind}.weight"),
                    dims: vec![stride / 4, 1, N],
                    ggml_type: 0,
                },
                payload,
            ));
        }
        let path = dir.join(file);
        let f = std::fs::File::create(&path).unwrap();
        write_gguf(&mut &f, &kvs, &tensors).unwrap();
        drop(f);
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.resize(bytes.len().next_multiple_of(4096) + 4096, 0);
        std::fs::write(&path, &bytes).unwrap();
        let mut f = std::fs::File::open(&path).unwrap();
        let index: Vec<serde_json::Value> = golem_gguf::tensor_entries(&mut f)
            .unwrap()
            .iter()
            .filter(|e| e.expert_kind().is_some())
            .map(|e| {
                serde_json::json!({
                    "tensor": e.name,
                    "layer": 1,
                    "experts": N,
                    "bytes_per_expert": stride,
                    "offsets_bytes": (0..N).map(|i| i * stride).collect::<Vec<_>>(),
                })
            })
            .collect();
        let manifest = dir.join("manifest.json");
        std::fs::write(
            &manifest,
            serde_json::json!({
                "name": "stream-store",
                "shards": [{"file": file, "bytes": bytes.len(),
                            "sha256": "0".repeat(64), "replicas": [dir]}],
                "expert_index": index,
            })
            .to_string(),
        )
        .unwrap();

        let geo = GgufPreadSource::open(&path, 1)
            .unwrap()
            .geometry()
            .unwrap()
            .unwrap();
        let layout = geo.layers[0];
        assert!(
            layout.head.iter().all(|&h| h != 0),
            "fixture must misalign every plane: {layout:?}"
        );
        let store = Arc::new(
            StoreExpertSource::open(&manifest, Arc::new(ChunkQueue::new(1, None))).unwrap(),
        );
        let source: Arc<dyn ExpertSource> = store.clone();
        let arena = Arena::new(&[SlotClass {
            slot_size: layout.slot_bytes() as usize,
            slots: 1,
        }])
        .unwrap();
        let pool = ExpertPool::new(
            arena,
            source.clone(),
            Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::empty()))),
            Arc::new(P3Stats::new()),
            geo,
            vec![0],
        );
        let s = StreamState::new(
            pool,
            source,
            Box::new(|| false),
            N as usize * stride as usize,
            PrefillGate::Whole,
            0.85,
        )
        .unwrap();
        let ids = [0, 1, 2, 3];
        for (k, code) in [1, 2, 3].into_iter().enumerate() {
            let mut out = [std::ptr::null(); N as usize];
            s.source(0, code, k as u64 + 1, &ids, &mut out).unwrap();
            for (e, &p) in out.iter().enumerate() {
                // SAFETY: one staged plane, held under seq k + 1.
                let plane = unsafe { std::slice::from_raw_parts(p as *const u8, stride as usize) };
                assert!(
                    plane
                        .iter()
                        .enumerate()
                        .all(|(i, &b)| b == byte(k, e as u64, i as u64)),
                    "kind {code} expert {e}"
                );
            }
        }
        assert_eq!(store.scratch_by_reason(), [0; 3], "a staged plane bounced");
        s.release_all();
        s.stop();
    }

    /// Staged planes sit at their file head so the store reads them in place
    /// (§5.4), each in its own page-aligned region: no two experts' covering
    /// reads share a page.
    #[test]
    fn staged_planes_sit_at_their_head_in_disjoint_regions() {
        const HEAD: u64 = 288;
        let source = Arc::new(HeadSource(Mutex::new(Vec::new())));
        let dyn_source: Arc<dyn ExpertSource> = source.clone();
        let layout = UnitLayout::new([PAGE_STRIDE; 3], [Some(HEAD); 3]).unwrap();
        let arena = Arena::new(&[SlotClass {
            slot_size: layout.slot_bytes() as usize,
            slots: 1,
        }])
        .unwrap();
        let pool = ExpertPool::new(
            arena,
            dyn_source.clone(),
            Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::empty()))),
            Arc::new(P3Stats::new()),
            PoolGeometry {
                layers: vec![layout],
                n_experts: N_EXPERTS,
                expert_bytes: 0,
            },
            vec![0],
        );
        let s = StreamState::new(
            pool,
            dyn_source,
            Box::new(|| false),
            N_EXPERTS as usize * PAGE_STRIDE as usize,
            PrefillGate::Whole,
            0.85,
        )
        .unwrap();
        let region = (HEAD + PAGE_STRIDE).next_multiple_of(DIRECT_ALIGN) as usize;
        assert_eq!(
            s.staged_bytes(),
            STAGING_SLOTS as usize * N_EXPERTS as usize * region
        );
        let ids = [0, 1, 2, 3];
        let mut out = [std::ptr::null(); N_EXPERTS as usize];
        s.source(0, 1, 1, &ids, &mut out).unwrap();
        assert_eq!(s.stats().read_bytes, (N_EXPERTS as usize * region) as u64);
        assert_eq!(s.stats().pool_bytes, 0);
        assert_eq!(
            *source.0.lock().unwrap(),
            vec![(HEAD as usize, region); N_EXPERTS as usize]
        );
        for (i, &p) in out.iter().enumerate() {
            assert_eq!(p as usize % DIRECT_ALIGN as usize, HEAD as usize);
            if i > 0 {
                assert_eq!(p as usize - out[i - 1] as usize, region);
            }
            // SAFETY: the pointer covers one staged plane, held under seq 1.
            let plane = unsafe { std::slice::from_raw_parts(p as *const u8, PAGE_STRIDE as usize) };
            assert!(plane.iter().all(|&b| b == i as u8), "expert {i}");
        }
        s.release_all();
        assert!(all_slots_free(&s));
        s.stop();
    }

    /// One prefill plane batch over the real store source spreads across
    /// both replicas' drives, byte-identical to the serial GGUF pread path.
    /// Fixture: a 2-shard split GGUF, Q4_K gate/up and Q6_K down planes of
    /// non-page strides (every expert's head differs, as in DeepSeek-V4.1-
    /// Flash UD-IQ2_XXS), one replica under the temp dir and one in
    /// `/dev/shm` (distinct `st_dev`s, so distinct drives to the queue).
    #[test]
    fn plane_batch_reads_from_both_drives_byte_identical_to_serial() {
        use golem_placement::pool::{GgufPreadSource, StoreExpertSource};
        use golem_store::ChunkQueue;
        use xtask::gguf::{GgufKv, GgufTensorMeta, GgufValue, write_gguf};

        const N: u64 = 16;
        const NE0: u64 = 2560;
        const ROWS: u64 = 211;
        // (shard, stem, ggml type, block bytes per 256 elems)
        let planes = [
            (0usize, "ffn_gate_exps", 12u32, 144u64),
            (0, "ffn_up_exps", 12, 144),
            (1, "ffn_down_exps", 14, 210),
        ];
        let byte = |k: usize, e: u64, i: u64| (k as u64 * 89 + e * 13 + i % 251) as u8;
        let tag = format!("golem-stream-2drv-{}", std::process::id());
        let dirs = [
            std::env::temp_dir().join(&tag),
            std::path::Path::new("/dev/shm").join(&tag),
        ];
        struct RmDirs<'a>(&'a [std::path::PathBuf]);
        impl Drop for RmDirs<'_> {
            fn drop(&mut self) {
                for d in self.0 {
                    let _ = std::fs::remove_dir_all(d);
                }
            }
        }
        let _cleanup = RmDirs(&dirs);
        assert!(
            std::path::Path::new("/dev/shm").is_dir(),
            "precondition: /dev/shm hosts the second replica"
        );
        let names = ["m-00001-of-00002.gguf", "m-00002-of-00002.gguf"];
        let mut lens = [0u64; 2];
        for shard in 0..2 {
            let mut kvs = vec![
                GgufKv {
                    key: "split.count".into(),
                    value: GgufValue::U32(2),
                },
                GgufKv {
                    key: "split.no".into(),
                    value: GgufValue::U32(shard as u32),
                },
            ];
            if shard == 0 {
                kvs.insert(
                    0,
                    GgufKv {
                        key: "general.architecture".into(),
                        value: GgufValue::Str("glm4moe".into()),
                    },
                );
                kvs.push(GgufKv {
                    key: "glm4moe.leading_dense_block_count".into(),
                    value: GgufValue::U32(1),
                });
            }
            // A small leading tensor pushes the expert bases off the page.
            let mut tensors = vec![(
                GgufTensorMeta {
                    name: format!("blk.0.attn_norm_{shard}.weight"),
                    dims: vec![8, 1, 1],
                    ggml_type: 0,
                },
                vec![0u8; 32],
            )];
            for (k, &(_, stem, ty, blk)) in planes.iter().enumerate().filter(|p| p.1.0 == shard) {
                let stride = NE0 / 256 * blk * ROWS;
                let payload = (0..N)
                    .flat_map(|e| (0..stride).map(move |i| byte(k, e, i)))
                    .collect();
                tensors.push((
                    GgufTensorMeta {
                        name: format!("blk.1.{stem}.weight"),
                        dims: vec![NE0, ROWS, N],
                        ggml_type: ty,
                    },
                    payload,
                ));
            }
            let mut buf = Vec::new();
            write_gguf(&mut buf, &kvs, &tensors).unwrap();
            buf.resize(buf.len().next_multiple_of(4096) + 4096, 0);
            lens[shard] = buf.len() as u64;
            for d in &dirs {
                std::fs::create_dir_all(d).unwrap();
                std::fs::write(d.join(names[shard]), &buf).unwrap();
            }
        }
        let first = dirs[0].join(names[0]);
        let gguf = GgufPreadSource::open(&first, 1).unwrap();
        let mut index = Vec::new();
        for (shard, name) in names.iter().enumerate() {
            let mut f = std::fs::File::open(dirs[0].join(name)).unwrap();
            for e in golem_gguf::tensor_entries(&mut f).unwrap() {
                if e.expert_kind().is_none() {
                    continue;
                }
                let stride = planes.iter().find(|p| e.name.contains(p.1)).unwrap();
                assert_eq!(stride.0, shard);
                let stride = NE0 / 256 * stride.3 * ROWS;
                index.push(serde_json::json!({
                    "tensor": e.name, "layer": 1, "experts": N, "bytes_per_expert": stride,
                    "offsets_bytes": (0..N).map(|i| i * stride).collect::<Vec<_>>(),
                }));
            }
        }
        let manifest = dirs[0].join("manifest.json");
        let shards: Vec<_> = (0..2)
            .map(|s| {
                serde_json::json!({"file": names[s], "bytes": lens[s],
                    "sha256": "0".repeat(64), "replicas": dirs})
            })
            .collect();
        std::fs::write(
            &manifest,
            serde_json::json!({"name": "stream-2drv", "shards": shards, "expert_index": index})
                .to_string(),
        )
        .unwrap();

        let geo = gguf.geometry().unwrap().unwrap();
        let store = Arc::new(
            StoreExpertSource::open(&manifest, Arc::new(ChunkQueue::new(0, None))).unwrap(),
        );
        let reg = store.registry();
        for shard in reg.shards() {
            assert_ne!(
                shard.replicas[0].drive, shard.replicas[1].drive,
                "precondition: replicas must sit on distinct devices"
            );
        }
        let mut heads = std::collections::BTreeSet::new();
        for kind in ExpertKind::ALL {
            for e in 0..N as u32 {
                heads.insert(store.plane(0, kind, e).unwrap().0 % DIRECT_ALIGN);
            }
        }
        assert!(
            heads.len() > 3 && !heads.contains(&0),
            "mixed heads: {heads:?}"
        );
        let source: Arc<dyn ExpertSource> = store.clone();
        let arena = Arena::new(&[SlotClass {
            slot_size: geo.layers[0].slot_bytes() as usize,
            slots: 1,
        }])
        .unwrap();
        let pool = ExpertPool::new(
            arena,
            source.clone(),
            Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::empty()))),
            Arc::new(P3Stats::new()),
            geo.clone(),
            vec![0],
        );
        let max_stride = geo.layers[0].stride.iter().copied().max().unwrap();
        let s = StreamState::new(
            pool,
            source,
            Box::new(|| false),
            N as usize * max_stride as usize,
            PrefillGate::Whole,
            0.85,
        )
        .unwrap();
        let served = |shard: usize| {
            reg.shards()[shard]
                .replicas
                .iter()
                .map(|r| r.served_chunks())
                .collect::<Vec<_>>()
        };
        let ids: Vec<i32> = (0..N as i32).collect();
        let total = || {
            (0..2)
                .map(served)
                .fold([0, 0], |a, s| [a[0] + s[0], a[1] + s[1]])
        };
        let before = total();
        for (k, code) in [1, 2, 3].into_iter().enumerate() {
            let kind = ExpertKind::ALL[k];
            let mut out = vec![std::ptr::null(); N as usize];
            s.source(0, code, k as u64 + 1, &ids, &mut out).unwrap();
            let stride = geo.layers[0].stride[k] as usize;
            let mut serial = vec![0u8; stride];
            for (e, &p) in out.iter().enumerate() {
                gguf.read_expert(0, kind, e as u32, &mut serial, 0).unwrap();
                // SAFETY: one staged plane, held under seq k + 1.
                let plane = unsafe { std::slice::from_raw_parts(p as *const u8, stride) };
                assert_eq!(plane, &serial[..], "kind {code} expert {e}");
                assert!(
                    serial
                        .iter()
                        .enumerate()
                        .all(|(i, &b)| b == byte(k, e as u64, i as u64))
                );
            }
        }
        // §5.2 striping across the planes; one small plane may be drained
        // entirely by the faster (tmpfs) replica's worker under self-selection.
        // Per-chunk striping is the store's `healthy_reads_stripe_across_drives`.
        let after = total();
        assert!(
            (0..2).all(|r| after[r] > before[r]),
            "the plane batches must use both drives: {before:?} → {after:?}"
        );
        assert_eq!(store.scratch_by_reason(), [0; 3], "a staged plane bounced");
        s.release_all();
        s.stop();
    }

    /// Blocks reads until `want` are in flight at once (or a timeout),
    /// recording the peak; later reads pass straight through.
    struct GateSource {
        live: Mutex<usize>,
        cv: Condvar,
        peak: AtomicU64,
        want: usize,
    }

    impl ExpertSource for GateSource {
        fn read_expert(
            &self,
            _layer: u32,
            _kind: ExpertKind,
            expert: u32,
            into: &mut [u8],
            head: usize,
        ) -> std::io::Result<()> {
            let mut live = self.live.lock().unwrap();
            *live += 1;
            self.peak.fetch_max(*live as u64, Ordering::SeqCst);
            self.cv.notify_all();
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            while self.peak.load(Ordering::SeqCst) < self.want as u64
                && std::time::Instant::now() < deadline
            {
                live = self
                    .cv
                    .wait_timeout(live, deadline - std::time::Instant::now())
                    .unwrap()
                    .0;
            }
            *live -= 1;
            drop(live);
            into[head..].fill(expert as u8);
            Ok(())
        }

        fn plane(&self, _layer: u32, _kind: ExpertKind, _expert: u32) -> Option<(u64, u64)> {
            None
        }
    }

    /// The warm set's planes drain concurrently, one per reader: a plane
    /// batch never waits for the previous plane's drain to finish.
    #[test]
    fn warm_planes_drain_concurrently() {
        let gate = Arc::new(GateSource {
            live: Mutex::new(0),
            cv: Condvar::new(),
            peak: AtomicU64::new(0),
            want: READERS,
        });
        let source: Arc<dyn ExpertSource> = gate.clone();
        let layout = UnitLayout::new([STRIDE; 3], [None; 3]).unwrap();
        let arena = Arena::new(&[SlotClass {
            slot_size: layout.slot_bytes() as usize,
            slots: 1,
        }])
        .unwrap();
        let pool = ExpertPool::new(
            arena,
            source.clone(),
            Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::empty()))),
            Arc::new(P3Stats::new()),
            PoolGeometry {
                layers: vec![layout; 2],
                n_experts: N_EXPERTS,
                expert_bytes: 0,
            },
            vec![0, 0],
        );
        let s = StreamState::new(
            pool,
            source,
            Box::new(|| false),
            N_EXPERTS as usize * STRIDE as usize,
            PrefillGate::Whole,
            0.85,
        )
        .unwrap();
        // Layer 0 up + down routed, layer 1 gate whole: three planes.
        s.warm(0, &[1]);
        let mut out = [std::ptr::null(); 1];
        s.source(0, 2, 1, &[1], &mut out).unwrap();
        assert_eq!(gate.peak.load(Ordering::SeqCst), READERS as u64);
        s.release_all();
        assert!(all_slots_free(&s));
        s.stop();
    }
}
