//! Prefill-stream byte sources (design §7.6, ABI v10): the pinned staging
//! ring, the reader thread, and the two stream trampolines C calls per
//! plane. Decode hot set isolation (§7.6.4): pooled hits lease a resident
//! slot WITHOUT scoring or paging in ([`ExpertPool::lease_resident`]);
//! misses are read into staging slots, never into pool slots — a streamed
//! ubatch changes no pool residency or score.
//!
//! Pipeline (§7.6.3): the routing readback of layer L WARMS its up/down
//! planes (routed-only) and layer L+1's whole gate plane (its ids are not
//! known before its router runs — the design keeps the gate whole, ruling
//! R2 defers prediction). The copy loop's `stream_source` consumes the warm
//! plane and parks on its completion only when the disk is the bound. Slots
//! recycle through `stream_release(seq)` (C releases seq−2 after the slab's
//! event sync) and the `copy_done` backstop releases everything left.
#![allow(unsafe_code)]

use std::collections::VecDeque;
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};

use golem_placement::pool::{Arena, ExpertKind, ExpertPool, ExpertSource, Lease, SlotClass};

use crate::p3_glue::p3_fatal;

/// Staging slots (§7.6.2: ≥ 2 max planes of RAM). Peak concurrent use is 5
/// (two sourced planes still held until their slab reuse + up to three warm
/// planes ahead of the copy loop), 6 leaves one spare.
const STAGING_SLOTS: u32 = 6;

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
    /// Planes currently holding pool leases or a staging slot (sourced and
    /// not yet released).
    pub held: u64,
}

struct Task {
    layer: u32,
    kind: ExpertKind,
    expert: u32,
    /// Staging destination: slot base + `expert × plane_len`.
    dst: *mut u8,
    plane_len: usize,
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
}

#[derive(Default)]
struct Inner {
    free_slots: Vec<usize>,
    /// Outstanding reader tasks per slot; a slot is only handed out when
    /// its count is 0.
    pending_reads: [usize; STAGING_SLOTS as usize],
    queue: VecDeque<Task>,
    holds: BTreeMap<u64, Hold>,
    /// Keyed (pool layer, kind code − 1).
    warm: HashMap<(u32, u8), Pending>,
    failed: bool,
    stop: bool,
}

/// The per-session stream state. Shared with C as part of the hook payload;
/// the decode thread drives `warm`/`source`/`release`, the reader thread
/// consumes `queue` — the `Inner` mutex + condvar are the only shared state
/// and the happens-before edges for the staging bytes.
pub struct StreamState {
    pool: Arc<ExpertPool>,
    source: Arc<dyn ExpertSource>,
    arena: Arc<Arena>,
    slot_bytes: usize,
    /// Plane byte length per layer × kind (flat, from the geometry).
    plane_lens: Vec<u32>,
    n_layers: u32,
    n_experts: u32,
    inner: Mutex<Inner>,
    cv: Condvar,
    /// The reader holds an `Arc<Self>`, so `Drop` never runs while it lives:
    /// `stop` joins it explicitly, releasing the pool and arena with the session.
    reader: Mutex<Option<std::thread::JoinHandle<()>>>,
    /// H2D expert bytes this module's pointers fed (the copy loop's copies).
    pub bytes: AtomicU64,
    /// Per-expert plane hits served from the pool.
    pub pool_hits: AtomicU64,
    /// Per-expert ranges served from staging.
    pub staged: AtomicU64,
    /// stream_source calls (one per streamed plane).
    pub planes: AtomicU64,
}

// SAFETY: `Task.dst`/`Pending.base` point into the mlocked arena the state
// owns (the `arena` field keeps it alive); no raw pointer dangles. All uses
// are serialized by the seq protocol: a slot is handed out only when its
// reads are done and its previous plane's H2D + compute completed under C's
// slab event sync.
unsafe impl Send for StreamState {}
unsafe impl Sync for StreamState {}

impl StreamState {
    /// Builds the staging ring (`slot_bytes × STAGING_SLOTS`, mlocked) and
    /// spawns the reader thread.
    ///
    /// # Errors
    /// Arena mmap/mlock failure (no unlocked fallback, §5.4).
    pub fn new(
        pool: Arc<ExpertPool>,
        source: Arc<dyn ExpertSource>,
        slot_bytes: usize,
    ) -> Result<Arc<Self>, String> {
        let arena = Arena::new(&[SlotClass {
            slot_size: slot_bytes,
            slots: STAGING_SLOTS,
        }])
        .map_err(|e| e.to_string())?;
        let geo = &pool.geometry;
        let mut plane_lens = Vec::with_capacity(geo.layers.len() * 3);
        for unit in &geo.layers {
            for kind in [ExpertKind::Gate, ExpertKind::Up, ExpertKind::Down] {
                let (_, stride) = unit.plane(kind);
                plane_lens.push(u32::try_from(stride).map_err(|_| "plane stride > u32")?);
            }
        }
        let state = Arc::new(Self {
            n_experts: geo.n_experts,
            n_layers: geo.layers.len() as u32,
            pool,
            source,
            arena: Arc::new(arena),
            slot_bytes,
            plane_lens,
            inner: Mutex::new(Inner {
                free_slots: (0..STAGING_SLOTS as usize).collect(),
                pending_reads: [0; STAGING_SLOTS as usize],
                queue: VecDeque::new(),
                holds: BTreeMap::new(),
                warm: HashMap::new(),
                failed: false,
                stop: false,
            }),
            cv: Condvar::new(),
            reader: Mutex::new(None),
            bytes: AtomicU64::new(0),
            pool_hits: AtomicU64::new(0),
            staged: AtomicU64::new(0),
            planes: AtomicU64::new(0),
        });
        state.spawn_reader();
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

    fn spawn_reader(self: &Arc<Self>) {
        let state = self.clone();
        let handle = std::thread::Builder::new()
            .name("golem-stream-reader".into())
            .spawn(move || state.reader_loop())
            .expect("spawn stream reader");
        *self.reader.lock().unwrap_or_else(|e| e.into_inner()) = Some(handle);
    }

    fn reader_loop(self: Arc<Self>) {
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
            // SAFETY: `dst..dst+plane_len` is inside the slot (slot_bytes =
            // n_experts × max stride ≥ plane_len); the task is exclusively
            // owned here and C reads the bytes only after pending hits 0.
            let plane = unsafe { std::slice::from_raw_parts_mut(task.dst, task.plane_len) };
            // A panicking source must not strand pending_reads (every
            // waiter would hang): it counts as a failed read.
            let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.source
                    .read_expert(task.layer, task.kind, task.expert, plane, 0)
                    .is_ok()
            }))
            .unwrap_or(false);
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            if !ok {
                inner.failed = true;
            }
            inner.pending_reads[task.slot] -= 1;
            self.cv.notify_all();
        }
    }

    /// The routing-readback read-ahead (decode thread, once per layer per
    /// streamed chunk): warms `layer`'s up/down planes routed-only and the
    /// NEXT layer's whole gate plane (its ids are unknown until its router
    /// runs — §7.6 keeps the gate whole). Resident experts are skipped (the
    /// pool copy is authoritative and `source` will lease it); no lease is
    /// taken here — the prefetcher is paused for the chunk and no demand
    /// claims run, so nothing can evict between warm and source.
    pub fn warm(&self, layer: u32, routed_ids: &[i32]) {
        if layer >= self.n_layers {
            return;
        }
        let routed: Vec<u32> = routed_ids
            .iter()
            .filter(|&&id| id >= 0 && (id as u32) < self.n_experts)
            .map(|&id| id as u32)
            .collect();
        // This layer's id-dependent planes, routed-only.
        self.warm_plane(layer, ExpertKind::Up, &routed);
        self.warm_plane(layer, ExpertKind::Down, &routed);
        // Next layer's gate, whole.
        if layer + 1 < self.n_layers {
            let whole: Vec<u32> = (0..self.n_experts).collect();
            self.warm_plane(layer + 1, ExpertKind::Gate, &whole);
        }
    }

    fn warm_plane(&self, layer: u32, kind: ExpertKind, ids: &[u32]) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if inner.failed {
            p3_fatal("stream reader failed; aborting rather than decoding on garbage".into());
        }
        let plane_len = self.plane_lens[(layer as usize) * 3 + kind as usize] as usize;
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
            // Wait for a free staging slot (bounded by the release protocol:
            // C releases seq−2 before refilling a half, copy_done drops the
            // tail between chunks).
            let (slot, base) = loop {
                if inner.failed {
                    p3_fatal(
                        "stream reader failed; aborting rather than decoding on garbage".into(),
                    );
                }
                if inner.stop {
                    p3_fatal("stream warm after stop".into());
                }
                if let Some(&s) = inner.free_slots.last() {
                    inner.free_slots.pop();
                    debug_assert_eq!(inner.pending_reads[s], 0);
                    // SAFETY: s < STAGING_SLOTS, arena alive.
                    // Slots are indexed 0..STAGING_SLOTS in the arena.
                    break (s, self.arena.slot_base(s as u32));
                }
                inner = self.cv.wait(inner).unwrap_or_else(|e| e.into_inner());
            };
            inner.pending_reads[slot] = staged_ids.len();
            for &id in &staged_ids {
                // SAFETY: id × plane_len stays inside the slot.
                inner.queue.push_back(Task {
                    layer,
                    kind,
                    expert: id,
                    dst: unsafe { base.add(id as usize * plane_len) },
                    plane_len,
                    slot,
                });
            }
            self.cv.notify_one();
            (slot, base)
        };
        inner.warm.insert(
            (layer, kind as u8),
            Pending {
                slot,
                base,
                staged_ids,
                warmed_ids: ids,
            },
        );
    }

    /// The `stream_source` hook body (decode thread, once per streamed
    /// plane): pooled hits lease without scoring; misses come from the warm
    /// slot (or are staged now when `warm` never ran — e.g. the first
    /// layer's gate) and block on the reader's completion.
    ///
    /// `blk` is the DENSE-LEAD-ADJUSTED pool layer (the glue subtracts the
    /// lead before calling).
    pub fn source(
        &self,
        blk: i32,
        kind_code: i32,
        seq: u64,
        ids: &[i32],
        out: &mut [*const c_void],
    ) {
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
        let plane_len = self.plane_lens[(layer as usize) * 3 + kind as usize] as usize;
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
                self.warm_plane(layer, kind, &ids);
            }
        }
        let pending = {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner
                .warm
                .remove(&(layer, kind as u8))
                .expect("warm plane inserted above")
        };
        // Wait for the plane's reads (the pipeline: warm enqueued them one
        // layer ahead, so this parks only when the disk is the bound — the
        // intended steady state).
        if pending.slot != usize::MAX {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            while inner.pending_reads[pending.slot] > 0 {
                if inner.failed {
                    p3_fatal(
                        "stream reader failed; aborting rather than decoding on garbage".into(),
                    );
                }
                inner = self.cv.wait(inner).unwrap_or_else(|e| e.into_inner());
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
            if id < 0 {
                p3_fatal(format!("stream_source: negative expert id {id}"));
            }
            let id = id as u32;
            let staged =
                pending.slot != usize::MAX && pending.staged_ids.binary_search(&id).is_ok();
            if let Some((ptr, lease)) = self.pool.lease_resident(layer, kind, id) {
                leases.push(lease);
                self.pool_hits.fetch_add(1, Ordering::Relaxed);
                self.bytes.fetch_add(plane_len as u64, Ordering::Relaxed);
                out[i] = ptr as *const c_void;
            } else if staged {
                self.staged.fetch_add(1, Ordering::Relaxed);
                self.bytes.fetch_add(plane_len as u64, Ordering::Relaxed);
                // SAFETY: id × plane_len inside the slot (slot_bytes bound).
                out[i] = unsafe { pending.base.add(id as usize * plane_len) } as *const c_void;
            } else if pending.slot != usize::MAX && pending.warmed_ids.binary_search(&id).is_err() {
                // SAFETY: as above; the bytes are stale (never staged this
                // plane) but finite, and the id is unrouted.
                out[i] = unsafe { pending.base.add(id as usize * plane_len) } as *const c_void;
            } else {
                // Warm had no slot (everything resident) but this id is no
                // longer resident: read it inline (never expected under the
                // pause protocol; correctness over speed).
                // SAFETY: slot-bounds argument does not apply — this is the
                // pool slot leased below? No: read into a stack buffer is
                // impossible (C keeps the pointer). Lease the resident slot
                // is the only option; if residency is truly gone, fatal.
                p3_fatal(format!(
                    "stream_source: expert {id} of layer {layer} neither resident nor staged"
                ));
            }
        }
        // SAFETY: the mutex critical sections above are the happens-before
        // edges for the staging bytes C is about to read.
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.holds.insert(
            seq,
            Hold {
                slot: pending.slot,
                leases,
            },
        );
    }

    /// The `stream_release` hook body: releases seq and everything smaller —
    /// leases drop, staging slots recycle. C calls this after the slab's
    /// event sync, so the H2D reading these host bytes has completed; the
    /// `copy_done` backstop runs after the compute-entry drain for the same
    /// reason.
    pub fn release(&self, seq: u64) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let done: Vec<u64> = inner.holds.range(..=seq).map(|(&s, _)| s).collect();
        for s in done {
            if let Some(h) = inner.holds.remove(&s) {
                if h.slot != usize::MAX {
                    debug_assert_eq!(inner.pending_reads[h.slot], 0);
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
    /// (bounded by one plane read) so the bytes can never be overwritten
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
        self.cv.notify_all();
    }

    /// Snapshot of the counters.
    pub fn stats(&self) -> StreamStats {
        StreamStats {
            bytes: self.bytes.load(Ordering::Relaxed),
            pool_hits: self.pool_hits.load(Ordering::Relaxed),
            staged: self.staged.load(Ordering::Relaxed),
            planes: self.planes.load(Ordering::Relaxed),
            held: self
                .inner
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .holds
                .len() as u64,
        }
    }

    /// Stops and joins the reader thread (session teardown; holds are empty
    /// once the hooks are unregistered and copy_done ran).
    pub fn stop(&self) {
        {
            let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
            inner.stop = true;
            self.cv.notify_all();
        }
        let handle = self.reader.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(h) = handle
            && h.thread().id() != std::thread::current().id()
        {
            let _ = h.join();
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
