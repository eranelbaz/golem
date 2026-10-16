//! P3 FFI glue (§3.4, ABI v5): per-session hook context, the batched
//! `ensure_host_resident` trampoline, the merged routing readback, and the
//! off-decode-thread eviction-order publisher.
//!
//! Hook lifecycle (per [`SessionHooks`], one per pooled session): `enable`
//! builds the RAM expert pool, creates a `golem_ctx` (owned by the sys-crate
//! RAII guards), registers the three hooks with an `Arc<HookState>` payload
//! and returns the guards. Dropping [`SessionHooks`] unregisters (the C side
//! joins in-flight steps) and frees the ctx — two sequential pooled sessions
//! in one process are legal; a concurrent second `enable` is refused (the
//! live-session guard plus the [`CURRENT`] reservation). The old
//! leaked-handle architecture is gone: no process-global hook state exists.
//!
//! Decode hot path (per AGENTS.md): the trampolines allocate nothing in
//! steady state — the miss list is staged into a preallocated `HookState`
//! scratch, the claim table lives in the pool (`ExpertPool::ensure_batch`),
//! and the routing readback touches only atomics. The eviction order is
//! rebuilt OFF the decode thread (one publisher thread) into the ctx's
//! triple-buffered `(blk, kind)` buffers via `golem_policy_order_push`;
//! `copy_done` runs a bounded non-blocking flush-if-dirty as the lag
//! fallback. C never blocks: when no triple-buffer slot is free the push is
//! skipped and the layer stays dirty for the next pass.
//!
//! Decode-thread-only fields: `HookState` is shared with the publisher
//! thread, so everything is atomic/mutex-protected; the decode thread is the
//! only writer of the step counter and the only caller of the trampolines
//! while hooks are registered (C invokes them from the single scheduler
//! compute path).
#![allow(unsafe_code)]

use golem_placement::pool::SlotClass;
use golem_placement::{HALF_LIFE_STEPS, decay_factor};
use std::collections::HashMap;
use std::ffi::{CStr, c_void};
#[cfg(test)]
use std::sync::atomic::AtomicI32;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};

use std::sync::{Arc, Mutex};

use crate::stream::StreamState;

use golem_gguf::blk_of;
use golem_llama_sys::raw;
use golem_llama_sys::{CtxRaw, GolemCtx, HookFns, Hooked};
use golem_placement::pool::{
    AccessFeed, D2hIncoming, ExpertKind, ExpertPool, ExpertSource, GgufPreadSource, P3Stats,
    PolicySnapshot, PredictionQueue, PrefetchHandle, SnapshotHandle, StoreExpertSource, WarmFeed,
    spawn_prefetch,
};
use golem_placement::scores::WarmShape;
use golem_store::ChunkQueue;

use crate::config::{EngineConfig, PrefetchMode, Residency, VramOrder};
use crate::error::EngineError;

/// One traced routing readback: fixed fields plus a range into the ring's
/// preallocated flat id buffer (§11 pattern — no per-id heap write).
struct TraceRec {
    step: u64,
    phase: TracePhase,
    layer: i32,
    start: usize,
    len: usize,
}

/// A trace record's phase column: routed ids of a prefill chunk (`pf`) or a
/// decode step (`dc`), or a P5 prediction for the layer (`pr`, ids in
/// descending router score, top k + margin).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum TracePhase {
    Prefill,
    Decode,
    Predict,
}

impl TracePhase {
    fn tag(self) -> &'static str {
        match self {
            Self::Prefill => "pf",
            Self::Decode => "dc",
            Self::Predict => "pr",
        }
    }
}

/// Preallocated per-step routing-trace ring (§11 timing-CSV pattern): the
/// routing readback appends fixed-size records into `records` and their ids
/// into the flat `ids` buffer, both sized at enable — capacity overflow
/// drops and counts, never reallocates. Flushed (and cleared) at turn end by
/// [`SessionHooks::write_routing_trace`]. With `trace: None` in
/// [`HookState`], the trampoline's only cost is one `Option` check.
struct TraceRing {
    /// Model path — the header's `model=<sha8>` is streamed lazily at first
    /// flush (multi-GiB hashing must not run at enable).
    model_path: std::path::PathBuf,
    header_written: bool,
    records: Vec<TraceRec>,
    ids: Vec<i32>,
    /// Fixed capacities (sized at enable); overflow drops and counts.
    rec_cap: usize,
    id_cap: usize,
    dropped: usize,
}

/// Routing-trace ring capacities for one turn: one record per (prefill
/// chunk or decode step, layer), `top_k` ids per token per layer; a turn's
/// tokens are bounded by `n_ctx + n_predict`. Pooled prefill chunks are
/// `max(1, K / top_k)` tokens (K = 19, top-k 8 → 2), so chunks are bounded
/// by tokens, not by `batch`. GLM at ctx 4096 + 512: 193.5k records × 32 B
/// ≈ 6.2 MB plus 1.55M ids × 4 B ≈ 6.2 MB; at ctx 131072: 5.5M records ≈
/// 177 MB plus 44M ids ≈ 177 MB (opt-in `--routing-trace` only). With prediction on,
/// each decode step adds `lookahead` records of `top_k + margin` ids per
/// layer: 512 steps × 42 layers × 12 ids ≈ 21.5k records, 258k ids ≈ 1 MB.
fn trace_caps(
    n_ctx: u32,
    n_predict: u32,
    n_layers: u32,
    top_k: u32,
    pred: (u32, u32),
) -> (usize, usize) {
    let (tokens, steps, layers) = (
        n_ctx as usize + n_predict as usize,
        n_predict as usize,
        n_layers as usize,
    );
    let (lookahead, pred_ids) = (pred.0 as usize, pred.1 as usize);
    let recs = (tokens + steps * lookahead) * layers;
    let ids = (tokens * top_k as usize + steps * lookahead * pred_ids) * layers;
    (recs, ids)
}

impl TraceRing {
    fn new(model_path: std::path::PathBuf, rec_cap: usize, id_cap: usize) -> Self {
        Self {
            model_path,
            header_written: false,
            records: Vec::with_capacity(rec_cap),
            ids: Vec::with_capacity(id_cap),
            rec_cap,
            id_cap,
            dropped: 0,
        }
    }

    /// Appends one routing readback. Pure push-into-capacity: no allocation
    /// on the decode hot path.
    fn record(&mut self, step: u64, phase: TracePhase, layer: i32, ids: &[i32]) {
        if self.records.len() >= self.rec_cap || self.ids.len() + ids.len() > self.id_cap {
            self.dropped += 1;
            return;
        }
        let start = self.ids.len();
        self.ids.extend_from_slice(ids);
        self.records.push(TraceRec {
            step,
            phase,
            layer,
            start,
            len: ids.len(),
        });
    }

    /// Record drop without a header write (pre-turn stale-record guard).
    fn clear_records(&mut self) {
        self.records.clear();
        self.ids.clear();
        self.dropped = 0;
    }

    /// Writes the `# golem-routing-trace v1` header once (hashing the model
    /// file on first use — the only expensive part, paid once per file), then
    /// one `step\tphase\tlayer\tid,id,...` line per record, then clears the
    /// ring for the next turn.
    fn flush(
        &mut self,
        w: &mut impl std::io::Write,
        n_layers: u32,
        top_k: u32,
    ) -> std::io::Result<()> {
        if !self.header_written {
            self.header_written = true;
            writeln!(
                w,
                "# golem-routing-trace v1 model={} layers={n_layers} top_k={top_k}",
                model_sha8(&self.model_path)
            )?;
        }
        for r in &self.records {
            write!(w, "{}\t{}\t{}\t", r.step, r.phase.tag(), r.layer)?;
            let ids = &self.ids[r.start..r.start + r.len];
            for (i, id) in ids.iter().enumerate() {
                if i > 0 {
                    write!(w, ",")?;
                }
                write!(w, "{id}")?;
            }
            writeln!(w)?;
        }
        if self.dropped > 0 {
            tracing::warn!(dropped = self.dropped, "routing trace ring overflow");
            self.dropped = 0;
        }
        self.records.clear();
        self.ids.clear();
        Ok(())
    }
}

/// First 8 hex chars of the model file's sha256, streamed (models are
/// multi-GiB; never held in memory).
fn model_sha8(path: &std::path::Path) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1024 * 1024];
    let mut file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(e) => return format!("open-error-{e}"),
    };
    use std::io::Read;
    loop {
        match file.read(&mut buf) {
            Ok(0) | Err(_) => break,
            Ok(n) => hasher.update(&buf[..n]),
        }
    }
    hasher
        .finalize()
        .iter()
        .take(4)
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// One registered expert tensor: everything the ensure trampoline needs to
/// translate a `(tensor, miss list)` callback into pool work.
#[derive(Clone, Copy)]
struct TensorEntry {
    /// Dense-lead-adjusted expert layer index (pool key space).
    layer: u32,
}

/// The per-session hook payload (the C `void *payload`). Contains claims,
/// scores and stats — no owned ctx pointers except the registration-keyed
/// `ctx_ptr` window below.
struct HookState {
    /// The demand pager (arena + residency + stats + batch claim table).
    pool: Arc<ExpertPool>,
    /// Expert byte source: the GGUF pread path or the store-backed reader
    /// (`--manifest`); the [`ExpertSource`] seam is what the registration
    /// cross-checks and the pool page-in both drive.
    source: Arc<dyn ExpertSource>,
    /// Concrete store-backed source when `config.manifest_path` was set:
    /// owns the shared queue's lifetime (§11 disk timing reads it).
    store: Option<Arc<StoreExpertSource>>,
    /// Dense-lead-adjusted expert layer count (from the GGUF walk at enable).
    n_layers: u32,
    /// Dense lead: expert tensors exist only from blk.<dense_lead> on; pool
    /// layers are renumbered to exclude it.
    dense_lead: u32,
    n_experts: u32,
    /// `golem_ctx_generation` at registration: every `last_entry` hit is
    /// tagged with it, so an address recycled across sessions can never hit
    /// a stale entry.
    generation: u64,
    /// The live ctx pointer while hooks are registered, else null. Set at
    /// enable; cleared in `SessionHooks::drop` AFTER the unregister-join (so
    /// no callback can read it) and BEFORE the ctx is freed. Read by the
    /// decode thread (copy_done flush) and the publisher thread.
    ctx_ptr: AtomicPtr<raw::golem_ctx>,
    /// tensor pointer → entry; filled on first callback per tensor (guarded
    /// by its own mutex, never held across `ensure_batch` — the residency
    /// shard lock stays leaf-most, §6.6).
    tensors: Mutex<HashMap<usize, TensorEntry>>,
    /// Last `entry_for` hit on the decode thread, tagged with the ctx
    /// generation. C stages a layer's experts back to back on one tensor, so
    /// this answers nearly every call without the table lock; the tag
    /// rejects a recycled address left over from a previous session's
    /// registration. Decode-thread-only writer; mutex for `&self`.
    last_entry: Mutex<Option<(usize, u64, TensorEntry)>>,
    /// `ensure_host_resident` invocations, for the per-1000 slow-path line.
    slow_calls: AtomicUsize,
    /// Cumulative ns spent inside the ensure trampoline's page-in section
    /// (§11 per-step timing; taken (swap-to-zero) per decode step).
    p3_wait_ns: AtomicU64,
    /// Test/counters observability: trampoline call counts (the ABI v5 call
    /// arithmetic: routing == n_offloaded_expert_layers, ensure ≤ that).
    routing_calls: AtomicUsize,
    ensure_calls: AtomicUsize,
    copy_done_calls: AtomicUsize,
    /// Order pushes skipped because no triple-buffer slot was free under the
    /// epoch gate (the publisher retries; `copy_done`'s flush too).
    order_pushes_failed: AtomicUsize,
    /// Per-call miss list scratch, preallocated (`3 × K` capacity): the
    /// trampoline stages the FFI arrays here — no per-step heap. Mutex for
    /// `&self`; decode-thread-only in practice.
    miss_scratch: Mutex<Vec<(ExpertKind, u32)>>,
    /// Scratch the `copy_done` flush sorts into, preallocated at enable
    /// (decode-thread-only writer; mutex for `&self`).
    flush_scratch: Mutex<OrderScratch>,
    /// §6.2 recency + §6.6 dirty tracking (shared with the publisher).
    policy: PolicyState,
    /// §11-style routing-trace ring when `config.routing_trace` is set; the
    /// routing readback appends (preallocated, no per-id heap), turn end
    /// flushes. `None` costs the trampoline one `Option` check.
    trace: Option<Mutex<TraceRing>>,
    /// Exclusive residency (§6.3 O10): registers `evict_dest_batch`, releases
    /// promoted pool slots and commits D2H fills at `copy_done`.
    exclusive: bool,
    /// Per-eviction scratch the `evict_dest_batch` hook fills (expert, floor)
    /// once per layer — no per-expert allocation on the eviction path.
    evict_scratch: Mutex<Vec<(u32, f32)>>,
    /// §6.5 P5 prediction state (mode `Off`: never touched).
    pred: PredState,
    /// Prefill-stream state (§7.6, ABI v10); `None` when streaming is off
    /// or the derivation dropped it (ruling R3).
    stream: Option<Arc<StreamState>>,
    /// Whether stream chunks are active (drives the routing readback's
    /// read-ahead warm); toggled per chunk by `Session`.
    stream_armed: AtomicBool,
    /// The active turn's cancel flag (null = no turn). Stored by
    /// [`StreamAbortGuard`] (`Release`), loaded per streamed plane by
    /// `stream_source` on the decode thread (`Acquire`, ABI v11).
    abort: AtomicPtr<AtomicBool>,
    /// Test seam: absolute stream plane count at which `stream_source` sets
    /// the turn's cancel flag (0 = off).
    #[cfg(any(test, feature = "test-support"))]
    test_cancel_plane: AtomicU64,
    /// Test seam: the next `stream_source` returns
    /// `GOLEM_STREAM_SRC_CANCEL` without touching any cancel flag.
    #[cfg(any(test, feature = "test-support"))]
    test_spurious_cancel: AtomicBool,
    /// Test seam: the next ensure call marks every replica unreadable
    /// before its page-in reads, so the read itself fails.
    #[cfg(any(test, feature = "test-support"))]
    test_fail_storage_in_read: AtomicBool,
}

/// One turn's registration of its cancel flag in `HookState.abort`.
pub(crate) struct StreamAbortGuard {
    state: Arc<HookState>,
    /// Keeps the pointee alive until the pointer is nulled.
    _cancel: Arc<AtomicBool>,
}

impl Drop for StreamAbortGuard {
    fn drop(&mut self) {
        self.state
            .abort
            .store(std::ptr::null_mut(), Ordering::Release);
    }
}

/// Experts predicted beyond the routed top-k (§6.5): 8 + 4 = 12 per layer.
/// `read` reads at most 4 × 15.2 MB ≈ 61 MB more per layer than a perfect
/// predictor; the predictor itself is one 4,096 × 288 GEMV per predicted
/// layer (≈1.2 M MACs), negligible next to the expert matmuls.
const PREDICT_MARGIN: u32 = 4;

/// §6.5 P5 next-layer predictions: per target layer, the union of the
/// predictions made since its last routing readback (one bitset row per
/// layer), the accuracy counters, and the `read` mode's queue. Written only
/// on the decode thread (predict and routing trampolines).
struct PredState {
    mode: PrefetchMode,
    bits: Vec<AtomicU64>,
    pending: Vec<AtomicBool>,
    words: usize,
    /// Decode-step routed ids of layers that had a prediction, and how
    /// many of them it named.
    routed: AtomicU64,
    hits: AtomicU64,
    queue: Option<Arc<PredictionQueue>>,
}

impl PredState {
    fn new(mode: PrefetchMode, n_layers: u32, n_experts: u32, queue_cap: usize) -> Self {
        let words = (n_experts as usize).div_ceil(64);
        let rows = if mode == PrefetchMode::Off {
            0
        } else {
            n_layers as usize
        };
        Self {
            mode,
            bits: (0..rows * words).map(|_| AtomicU64::new(0)).collect(),
            pending: (0..rows).map(|_| AtomicBool::new(false)).collect(),
            words,
            routed: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            queue: (mode == PrefetchMode::Read).then(|| Arc::new(PredictionQueue::new(queue_cap))),
        }
    }

    /// Adds one prediction for `layer` to its pending union; the returned
    /// iterator (must be driven) sets the bits and yields the ids not
    /// already predicted, so lookahead > 1 never queues an id twice.
    fn add<'a>(
        &'a self,
        layer: usize,
        ids: &'a [i32],
        n_experts: u32,
    ) -> impl Iterator<Item = u32> + 'a {
        let row = &self.bits[layer * self.words..][..self.words];
        if !self.pending[layer].swap(true, Ordering::Relaxed) {
            row.iter().for_each(|w| w.store(0, Ordering::Relaxed));
        }
        ids.iter()
            .filter(move |&&id| id >= 0 && (id as u32) < n_experts)
            .filter(move |&&id| {
                let bit = 1u64 << (id % 64);
                row[id as usize / 64].fetch_or(bit, Ordering::Relaxed) & bit == 0
            })
            .map(|&id| id as u32)
    }

    /// The pending union of `layer` as expert ids.
    fn predicted(&self, layer: usize) -> impl Iterator<Item = u32> + '_ {
        let row = &self.bits[layer * self.words..][..self.words];
        row.iter().enumerate().flat_map(|(w, word)| {
            let v = word.load(Ordering::Relaxed);
            (0..64u32)
                .filter(move |b| v >> b & 1 == 1)
                .map(move |b| w as u32 * 64 + b)
        })
    }

    /// Books `layer`'s routed ids against its pending prediction, then
    /// retires the prediction. Returns whether one was pending.
    fn score(&self, layer: usize, ids: &[i32]) -> bool {
        if self
            .pending
            .get(layer)
            .is_none_or(|p| !p.swap(false, Ordering::Relaxed))
        {
            return false;
        }
        let row = &self.bits[layer * self.words..][..self.words];
        let hits = ids
            .iter()
            .filter(|&&id| {
                id >= 0
                    && row
                        .get(id as usize / 64)
                        .is_some_and(|w| w.load(Ordering::Relaxed) >> (id % 64) & 1 == 1)
            })
            .count();
        self.routed.fetch_add(ids.len() as u64, Ordering::Relaxed);
        self.hits.fetch_add(hits as u64, Ordering::Relaxed);
        true
    }
}

/// §6.5 prefetch counters for one session (`Session::prefetch_stats`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefetchStats {
    /// Configured mode.
    pub mode: PrefetchMode,
    /// Routed ids of decode steps whose layer had a prediction.
    pub predicted_routed: u64,
    /// Of those, ids the prediction named (prediction recall).
    pub predicted_hits: u64,
    /// Units the prefetcher paged in.
    pub prefetch_pageins: u64,
    /// Prefetched units claimed on demand before eviction.
    pub prefetch_hits: u64,
    /// Bytes of prefetched units evicted unused.
    pub prefetch_wasted_bytes: u64,
    /// Times the prefetcher stood down for a demand page-in.
    pub prefetch_standdowns: u64,
}

impl std::fmt::Display for PrefetchStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let pct = |n: u64, d: u64| {
            if d == 0 {
                0.0
            } else {
                100.0 * n as f64 / d as f64
            }
        };
        write!(
            f,
            "prefetch mode={:?} prediction_recall={:.1}% ({}/{}) prefetch_pageins={} \
             prefetch_hit_rate={:.1}% prefetch_wasted_bytes={} prefetch_standdowns={}",
            self.mode,
            pct(self.predicted_hits, self.predicted_routed),
            self.predicted_hits,
            self.predicted_routed,
            self.prefetch_pageins,
            pct(self.prefetch_hits, self.prefetch_pageins),
            self.prefetch_wasted_bytes,
            self.prefetch_standdowns
        )
    }
}

/// §6.2 policy state: per-expert last-access steps and decayed scores
/// feeding the evict-first order, the decode-step counter, and the per-layer
/// dirty flags consumed by the order publisher.
struct PolicyState {
    /// Flat `layer × n_experts` last-access step + 1; 0 = never accessed.
    /// Atomics: the routing readback books one routed id per element on the
    /// decode hot path, which takes no lock.
    last_access: Vec<AtomicU64>,
    /// Flat §6.2 score as of the key's last booking, packed with that step
    /// in one word (`f32 bits << 32 | step + 1`, 0 = never) so the
    /// publisher never pairs a score with another step's age. The decode
    /// thread is the only writer. GLM: 12,096 keys × 8 B ≈ 97 KB.
    score: Vec<AtomicU64>,
    vram_order: VramOrder,
    /// Decode-step counter: one post-prefill trunk `llama_decode` = one step
    /// (§6.2: prefill chunks are excluded from scoring), bumped by
    /// [`advance_decode_step`] from `Session`'s generation decode.
    step: AtomicU64,
    /// Whether the routing readback books recency: false during prefill
    /// (§6.2: a prefill ubatch routes nearly every expert and would flatten
    /// the scores), true during generation decodes. Driven by `Session`
    /// through [`set_scoring`]; starts false so the load self-check decode,
    /// which is not user traffic, books nothing (`Session::new` arms it).
    scoring: AtomicBool,
    /// Per-layer dirty flags: the routing readback marks; the publisher and
    /// the `copy_done` flush swap-to-claim and publish.
    dirty: Vec<AtomicBool>,
    dirty_any: AtomicBool,
    n_layers: usize,
    n_experts: usize,
}

impl PolicyState {
    fn new(n_layers: u32, n_experts: u32, vram_order: VramOrder) -> Self {
        let keys = n_layers as usize * n_experts as usize;
        Self {
            last_access: (0..keys).map(|_| AtomicU64::new(0)).collect(),
            score: (0..keys).map(|_| AtomicU64::new(0)).collect(),
            vram_order,
            step: AtomicU64::new(0),
            scoring: AtomicBool::new(false),
            dirty: (0..n_layers as usize)
                .map(|_| AtomicBool::new(false))
                .collect(),
            dirty_any: AtomicBool::new(false),
            n_layers: n_layers as usize,
            n_experts: n_experts as usize,
        }
    }

    /// A state with scoring armed (unit tests drive generation steps).
    #[cfg(test)]
    fn scored(n_layers: u32, n_experts: u32, vram_order: VramOrder) -> Self {
        let p = Self::new(n_layers, n_experts, vram_order);
        p.scoring.store(true, Ordering::Relaxed);
        p
    }

    /// Books one step's routed ids of `layer` (the dense ids readback):
    /// every routed expert's recency moves to `step + 1` and its score to
    /// `decayed + 1`. Out-of-range and negative ids are rejected (never
    /// wrapped into the next layer's row). Lock-free: relaxed loads/stores
    /// per routed id; repeated ids in one readback book once per step.
    fn book_routed(&self, layer: usize, ids: &[i32]) {
        if !self.scoring.load(Ordering::Relaxed) {
            return;
        }
        let step = self.step.load(Ordering::Relaxed);
        let n = self.n_experts;
        let base = layer * n;
        for &id in ids {
            if id < 0 || id as usize >= n {
                continue;
            }
            let i = base + id as usize;
            if self.last_access[i].load(Ordering::Relaxed) == step + 1 {
                continue;
            }
            let s = (self.decayed(i, step) + 1.0) as f32;
            let word = u64::from(s.to_bits()) << 32 | (step + 1) & u64::from(u32::MAX);
            self.score[i].store(word, Ordering::Relaxed);
            self.last_access[i].store(step + 1, Ordering::Relaxed);
        }
    }

    /// §6.2 score of flat key `i` decayed to `step` (0 when never booked).
    fn decayed(&self, i: usize, step: u64) -> f64 {
        let word = self.score[i].load(Ordering::Relaxed);
        let Some(last) = (word & u64::from(u32::MAX)).checked_sub(1) else {
            return 0.0;
        };
        let age = (step & u64::from(u32::MAX)).saturating_sub(last);
        f64::from(f32::from_bits((word >> 32) as u32)) * decay_factor(age, HALF_LIFE_STEPS)
    }

    /// Marks `layer`'s order dirty for the publisher.
    fn mark_dirty(&self, layer: usize) {
        self.dirty[layer].store(true, Ordering::Relaxed);
        self.dirty_any.store(true, Ordering::Relaxed);
    }

    /// Every booked key's `(key, last_access, score)` words and the step,
    /// for the warm-start file. Relaxed loads: called between turns, when
    /// the decode thread (the only writer) is idle.
    fn warm_words(&self) -> (u64, Vec<(usize, u64, u64)>) {
        let words = self
            .last_access
            .iter()
            .zip(&self.score)
            .enumerate()
            .filter_map(|(i, (a, w))| {
                let a = a.load(Ordering::Relaxed);
                (a != 0).then(|| (i, a, w.load(Ordering::Relaxed)))
            })
            .collect();
        (self.step.load(Ordering::Relaxed), words)
    }

    /// Restores a validated warm state verbatim (§6.3 warm-up) and marks
    /// every layer dirty so the publisher pushes the seeded VRAM order.
    /// Before hook registration: no reader or writer runs yet.
    fn restore(&self, warm: &golem_placement::scores::WarmState) {
        for &(i, a, w) in &warm.words {
            self.last_access[i].store(a, Ordering::Relaxed);
            self.score[i].store(w, Ordering::Relaxed);
        }
        self.step.store(warm.step, Ordering::Relaxed);
        for layer in 0..self.n_layers {
            self.mark_dirty(layer);
        }
    }

    /// Booked keys ranked best-first by §6.2 score decayed to the current
    /// step; ties by ascending key.
    fn ranked_keys(&self) -> Vec<usize> {
        let step = self.step.load(Ordering::Relaxed);
        let mut rank: Vec<(f64, usize)> = (0..self.score.len())
            .filter(|&i| self.last_access[i].load(Ordering::Relaxed) != 0)
            .map(|i| (self.decayed(i, step), i))
            .collect();
        rank.sort_by(|a, b| b.0.total_cmp(&a.0).then(a.1.cmp(&b.1)));
        rank.into_iter().map(|(_, i)| i).collect()
    }

    /// Last-access step of flat index `idx`; `None` = never accessed.
    #[cfg(test)]
    fn access_step(&self, idx: usize) -> Option<u64> {
        self.last_access[idx].load(Ordering::Relaxed).checked_sub(1)
    }

    /// Builds the evict-first order for `layer` into `order`: never-accessed
    /// first, then ascending last-access step ([`VramOrder::Recency`]) or
    /// ascending decayed score ([`VramOrder::Score`]); ties by ascending
    /// index. Preallocated scratch; no allocation (runs on the publisher).
    fn build_order(&self, layer: usize, s: &mut OrderScratch) {
        let n = self.n_experts;
        debug_assert_eq!(s.order.len(), n);
        s.order.iter_mut().zip(0..).for_each(|(o, i)| *o = i);
        let base = layer * n;
        let row = &self.last_access[base..][..n];
        // Keys are snapshotted once per expert before sorting: the decode
        // thread writes these atomics mid-sort, and a comparator reading
        // them live is not a total order.
        let step = self.step.load(Ordering::Relaxed);
        for (k, (w, i)) in s.keys.iter_mut().zip(row.iter().zip(base..)) {
            let raw = w.load(Ordering::Relaxed);
            *k = match self.vram_order {
                VramOrder::Recency => raw,
                // Decayed scores are ≥ 0 and finite, so their f64 bits order
                // like the values; never-accessed (raw 0) keys 0, first.
                VramOrder::Score if raw == 0 => 0,
                VramOrder::Score => self.decayed(i, step).to_bits().max(1),
            };
        }
        let keys = &s.keys;
        s.order.sort_unstable_by_key(|&i| (keys[i as usize], i));
    }
}

/// Recency order builder over raw keys (pure; unit-tested): `keys` are the
/// raw `last_access` words (0 = never, else step + 1). Writes a permutation
/// of `0..keys.len()` by construction.
#[cfg(test)]
fn build_order(keys: &[u64], order: &mut [i32]) {
    debug_assert_eq!(keys.len(), order.len());
    order.iter_mut().zip(0..).for_each(|(o, i)| *o = i);
    order.sort_unstable_by_key(|&i| (keys[i as usize], i));
}

/// Permutation check for tests; C validates every published order
/// authoritatively (golem_slab_step), so the hot path doesn't repeat it.
#[cfg(test)]
fn is_permutation(order: &[i32], n: usize) -> bool {
    order.len() == n && order.iter().all(|&i| i >= 0 && (i as usize) < n) && {
        let mut seen = vec![false; n];
        order
            .iter()
            .all(|&i| !std::mem::replace(&mut seen[i as usize], true))
    }
}

/// Preallocated `build_order` buffers: the published order and one snapshot
/// key per expert.
struct OrderScratch {
    order: Vec<i32>,
    keys: Vec<u64>,
}

impl OrderScratch {
    fn new(n: usize) -> Self {
        Self {
            order: vec![0; n],
            keys: vec![0; n],
        }
    }
}

/// The one pool-enabled session's payload state, process-wide. Set by a
/// successful [`enable`], cleared by `SessionHooks::drop` — the sequential
/// enable/drop cycle is legal, a concurrent second enable is refused.
static CURRENT: Mutex<Vec<Arc<HookState>>> = Mutex::new(Vec::new());

/// Process-wide count of live [`crate::session::Session`]s. Off-then-on
/// guard: once ANY session exists, [`enable`] refuses — a pool registered
/// under a live session would retroactively route that session's decodes
/// through the new hooks (the first session's next decode would page from a
/// pool built for a different model).
static LIVE_SESSIONS: AtomicUsize = AtomicUsize::new(0);

/// Err while a pool-enabled session's hooks are live: a second concurrent
/// pool would build a second slab set — at the real target (K = 10) each is
/// 10 slots × 42 layers × 3 planes × ≈5.8 MB ≈ 7.3 GB, twice on a 24 GiB
/// card. Sequential pooled sessions are legal (per-model ctx, ABI v5).
/// Checked by `Session::new` BEFORE `load` (so a refusal costs no model
/// load) and again by [`enable`].
pub(crate) fn refuse_if_pool_session_live() -> Result<(), EngineError> {
    if hooks_registered() {
        return Err(EngineError::PoolInit(
            "a pool-enabled session already exists in this process; a concurrent second \
             pool would hold two device slab sets (at the target: K=10 × 42 layers × \
             3 planes × ≈5.8 MB ≈ 7.3 GB each — 14.6 GB on a 24 GiB card) — drop \
             the first session before opening another pooled one"
                .into(),
        ));
    }
    Ok(())
}

/// Marks a successfully constructed session (called by `Session::new`).
pub(crate) fn session_opened() {
    LIVE_SESSIONS.fetch_add(1, Ordering::SeqCst);
}

/// Live `Session` count in this process (teardown observability: a server
/// shutdown must leave 0).
pub fn live_sessions() -> usize {
    LIVE_SESSIONS.load(Ordering::SeqCst)
}

/// Marks a dropped session (called by `Session::drop`).
pub(crate) fn session_closed() {
    let prev = LIVE_SESSIONS.fetch_sub(1, Ordering::SeqCst);
    debug_assert!(prev > 0, "session_closed without a matching session_opened");
}

/// Bumps the policy decode step: `Session` calls this after every successful
/// trunk decode (§6.2: prefill chunks are excluded from scoring — a prefill
/// ubatch routes nearly every expert and would flatten the scores). No-op
/// while hooks are unregistered.
pub(crate) fn advance_decode_step() {
    for h in current_states() {
        h.policy.step.fetch_add(1, Ordering::Relaxed);
        // A prediction whose layer never reported routing this step is
        // spent: drop it and its protect shield before the next step.
        for (layer, p) in h.pred.pending.iter().enumerate() {
            if p.swap(false, Ordering::Relaxed) && h.pred.mode == PrefetchMode::Protect {
                h.pool.protect(layer as u32, std::iter::empty());
            }
        }
    }
}

/// Per-step I/O timing snapshot (§11): P3 wait is always measured once the
/// hooks exist; disk time/bytes only when the store-backed source is live.
#[derive(Default)]
pub(crate) struct StepIoStats {
    pub p3_wait_ns: u64,
    pub disk_ns: Option<u64>,
    pub disk_bytes: Option<u64>,
    /// Copy-stream H2D (ms, bytes); `None` unless async_h2d measured it.
    pub h2d: Option<(f64, u64)>,
}

fn current_states() -> Vec<Arc<HookState>> {
    CURRENT.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Takes (swap-to-zero) the timing accumulators at a decode step boundary.
/// No-op while hooks are unregistered: unmeasured fields come back `None`.
pub(crate) fn take_step_io_stats() -> StepIoStats {
    let mut out = StepIoStats::default();
    for h in current_states() {
        out.p3_wait_ns += h.p3_wait_ns.swap(0, Ordering::Relaxed);
        if let Some(s) = &h.store {
            let (ns, bytes) = s.take_disk_stats();
            out.disk_ns = Some(out.disk_ns.unwrap_or(0) + ns);
            out.disk_bytes = Some(out.disk_bytes.unwrap_or(0) + bytes);
        }
        let ctx = h.ctx_ptr.load(Ordering::SeqCst);
        if !ctx.is_null() {
            let (mut ms, mut bytes) = (0.0, 0);
            // SAFETY: FFI on the live ctx between computes (the decode thread
            // synchronized); the cs_* fields are decode-thread-only and it
            // writes only the two out-params.
            if unsafe { raw::golem_take_h2d_timing(ctx, &mut ms, &mut bytes) } == 1 {
                let (m0, b0) = out.h2d.unwrap_or((0.0, 0));
                out.h2d = Some((m0 + ms, b0 + bytes));
            }
        }
    }
    out
}

/// Switches recency booking on for generation decodes and off for prefill
/// (§6.2). No-op while hooks are unregistered.
pub(crate) fn set_scoring(on: bool) {
    for h in current_states() {
        h.policy.scoring.store(on, Ordering::Relaxed);
    }
}

/// Scoring off until dropped, so every prefill exit path (errors included)
/// re-arms it.
pub(crate) struct PrefillScope;

impl PrefillScope {
    pub(crate) fn enter() -> Self {
        set_scoring(false);
        PrefillScope
    }
}

impl Drop for PrefillScope {
    fn drop(&mut self) {
        set_scoring(true);
    }
}

/// Test seam: the live policy decode step (0 while hooks are unregistered).
#[cfg(test)]
pub(crate) fn decode_step_for_test() -> u64 {
    current_states()
        .first()
        .map_or(0, |h| h.policy.step.load(Ordering::Relaxed))
}

/// Whether a pooled session's hooks are registered (a second pooled session
/// in the process is refused while one is live).
pub fn hooks_registered() -> bool {
    !CURRENT.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
}

/// The per-session hook guards: owns the registration (unregister-join on
/// drop), the order publisher, the prefetch thread and the pool. Drop order
/// (the ABI v5 contract): unregister-join → clear the ctx window + stop the
/// publisher → ctx free (inside the sys RAII guard).
pub struct SessionHooks {
    /// HIP registrations of the pool arena and the stream staging ring.
    /// Declared first: unregistered before any field can drop the last
    /// arena owner, so the range is never unmapped while registered (a
    /// stale registration on a reused VA faults the next session's H2D).
    #[cfg(golem_backend_hip)]
    _pinned: Vec<golem_llama_sys::pinned::HostRegistration>,
    /// §6.5 prefetch thread (`--prefetch read` only). Stopped before the
    /// hooks are torn down so its pool feed cannot outlive the pool state.
    prefetch: Option<PrefetchHandle>,
    /// §6.3 warm-up pre-fill thread (restored warm state only); exits once
    /// its one-shot feed is drained. Stopped/paused beside `prefetch`.
    warm: Option<PrefetchHandle>,
    /// Pre-fill inputs held from enable until [`Self::start_warm_fill`]
    /// (K is final only after the post-load fit).
    warm_fill: Option<WarmFill>,
    /// Warm-start file and the shape it is keyed by; `None` = disabled.
    warm_file: Option<(std::path::PathBuf, WarmShape)>,
    /// Units the pre-fill was handed (test seam).
    #[cfg(any(test, feature = "test-support"))]
    warm_planned: Vec<(u32, u32)>,
    /// The eviction-order publisher (background thread).
    publisher: PublisherHandle,
    /// The live hook registration; also owns the ctx (`Arc` inside the
    /// guard). Taken first in `drop`.
    hooked: Option<Hooked<HookState>>,
    /// The payload state (same Arc the C callbacks receive).
    state: Arc<HookState>,
    /// Pool counters for [`crate::session::Session::p3_stats`] (the same Arc
    /// the pool holds).
    pub(crate) stats: Arc<P3Stats>,
    /// `golem_ctx` keep-alive: a clone of the ctx Arc that outlives the
    /// registration. Declared LAST so it drops after publisher/prefetch/state
    /// — the §3.4 drop order ends with `golem_ctx_free` (which aborts while
    /// any sched attach refcount remains) and then the pool.
    ctx_keep: Arc<CtxRaw>,
    /// Device slab slots per layer (§3.4): explicit, or derived by
    /// [`Self::fit_device_cache`].
    pub device_cache_experts: u32,
    /// Pending derived-K fit; `None` once fitted or with an explicit K.
    fit: Option<DeviceFit>,
    /// `golem_slab_alloc_failures` already handled.
    slab_failures_seen: u64,
    /// Model routed top-k (`expert_used_count`).
    pub top_k: u32,
    /// Prefill-stream state (§7.6); `None` when disabled.
    pub(crate) stream: Option<Arc<StreamState>>,
    /// The C-side stream config stashed at enable (n_experts, max stride,
    /// routed_max): re-passed on every arm/disarm (the setter takes all).
    stream_cfg: Option<(i32, i32, f32)>,
    /// K while stream slabs may be live, and K for decode with them freed
    /// (#135); equal unless the derived fit found room for more.
    k_stream: u32,
    k_decode: u32,
    /// Whether the device cache is sized for streamed prefill (`k_stream`).
    vram_stream: bool,
    /// Both stream slabs' bytes (0 without streaming).
    stream_slab_vram: u64,
    /// Device bytes of one K slot across the model (§3.4). Read only on
    /// the HIP path (`refit_stream_k`'s VRAM query).
    #[cfg_attr(not(golem_backend_hip), allow(dead_code))]
    slab_bytes_per_expert: u64,
}

impl SessionHooks {
    /// The raw ctx pointer while hooks are registered (the per-ctx P4 stats
    /// getter). Panics after drop — callers hold the session.
    pub fn ctx_ptr(&self) -> raw::golem_ctx_t {
        self.hooked.as_ref().expect("hooks dropped").ctx_ptr()
    }

    /// The payload pointer C receives: drive the trampolines directly with
    /// it (test and measurement harnesses; the C scheduler does the same).
    pub fn payload_ptr(&self) -> *mut c_void {
        Arc::as_ptr(&self.state) as *mut c_void
    }

    /// Bumps this session's policy decode step (the `Session` decode path
    /// calls the free function; direct-`enable` harnesses call this).
    pub fn advance_decode_step(&self) {
        self.state.policy.step.fetch_add(1, Ordering::Relaxed);
    }

    /// Fraction of RAM-pool slots holding a unit.
    pub fn pool_fill(&self) -> f64 {
        self.state.pool.fill()
    }

    /// Store-backed source health (§5.2/§11); `None` on the GGUF pread path.
    pub fn storage_health(&self) -> Option<golem_store::InstanceHealth> {
        self.state.store.as_ref().map(|s| s.health())
    }

    /// Test seam: marks every replica of every shard unreadable.
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn fail_storage_for_test(&self) {
        self.state.fail_storage();
    }

    /// Test seam: the next `ensure_host_resident` kills every replica
    /// inside its page-in, so the in-flight read fails.
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn fail_storage_in_read_for_test(&self) {
        self.state
            .test_fail_storage_in_read
            .store(true, Ordering::Relaxed);
    }

    /// Test seam: the stream reader kills every replica right before its
    /// next read.
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn fail_storage_in_stream_read_for_test(&self) {
        if let Some(s) = &self.state.stream {
            let state = self.state.clone();
            s.before_next_read_for_test(Box::new(move || state.fail_storage()));
        }
    }

    /// §6.5 prefetch and prediction counters.
    pub fn prefetch_stats(&self) -> PrefetchStats {
        let p = &self.state.pred;
        let s = &self.state.pool.stats;
        PrefetchStats {
            mode: p.mode,
            predicted_routed: p.routed.load(Ordering::Relaxed),
            predicted_hits: p.hits.load(Ordering::Relaxed),
            prefetch_pageins: s.prefetch_pageins.load(Ordering::Relaxed),
            prefetch_hits: s.prefetch_hits.load(Ordering::Relaxed),
            prefetch_wasted_bytes: s.prefetch_wasted_bytes.load(Ordering::Relaxed),
            prefetch_standdowns: s.prefetch_standdowns.load(Ordering::Relaxed),
        }
    }

    /// Test/counters observability: trampoline call counts.
    pub fn routing_calls(&self) -> usize {
        self.state.routing_calls.load(Ordering::Relaxed)
    }
    /// Batched ensure calls so far.
    pub fn ensure_calls(&self) -> usize {
        self.state.ensure_calls.load(Ordering::Relaxed)
    }
    /// copy_done calls so far.
    pub fn copy_done_calls(&self) -> usize {
        self.state.copy_done_calls.load(Ordering::Relaxed)
    }
    /// Order pushes skipped because no triple-buffer slot was free under the
    /// epoch gate (retried by publisher/flush).
    pub fn order_pushes_failed(&self) -> usize {
        self.state.order_pushes_failed.load(Ordering::Relaxed)
    }
    /// The pool's counters (test/observability seam).
    pub fn state_stats(&self) -> Arc<P3Stats> {
        self.stats.clone()
    }

    /// Switches recency booking (§6.2) on this session's own policy only;
    /// the free [`set_scoring`] covers every live session (prefill scope).
    pub fn set_scoring(&self, on: bool) {
        self.state.policy.scoring.store(on, Ordering::Relaxed);
    }

    /// Derived-K fit (ruling R3), once the model, KV and compute buffer are
    /// resident: frees the load's top-k slab, then K = (free − stream slabs
    /// − headroom) / per-expert slab bytes − 1, capped at the pre-load
    /// bound. With streaming on and that K below top-k, streaming
    /// auto-disables with a warning and K is refit without the slabs.
    /// No-op for an explicit K or after the first call.
    pub(crate) fn fit_device_cache(&mut self) -> Result<(), EngineError> {
        let Some(fit) = self.fit.take() else {
            return Ok(());
        };
        let ctx = self.ctx_ptr();
        // SAFETY: FFI on the live ctx between computes (no slab in flight).
        unsafe { raw::golem_reset_device_cache(ctx) };
        self.device_cache_reset();
        #[cfg(golem_backend_hip)]
        let free = golem_llama_sys::pinned::free_device_mem().map(|(f, _)| f as u64);
        #[cfg(not(golem_backend_hip))]
        let free: Option<u64> = None;
        let Some(free) = free else {
            // No post-load reading: keep the top-k slab set at enable.
            tracing::warn!(
                k = self.device_cache_experts,
                "VRAM query failed after load; device cache stays at top-k"
            );
            return Ok(());
        };
        let k_of = |budget: u64| {
            k_from_free_vram(budget, fit.plan_cap, fit.slab_bytes_per_expert, self.top_k)
                .map(|k| k.min(fit.k_bound))
        };
        let k = match k_of(free.saturating_sub(fit.stream_slab_vram)) {
            Ok(k) => {
                // Decode runs with the stream slabs freed: the §10 cap
                // already netted them out, so it gives them back too.
                // Exclusive residency keeps one K: a reset drops VRAM-only
                // units to disk (§6.3 O10).
                self.k_decode = if self.stream.is_some() && !self.state.exclusive {
                    k_from_free_vram(
                        free,
                        fit.plan_cap.saturating_add(fit.stream_slab_vram),
                        fit.slab_bytes_per_expert,
                        self.top_k,
                    )
                    .map_or(k, |d| d.min(fit.k_bound).max(k))
                } else {
                    k
                };
                k
            }
            Err(e) if self.stream.is_some() => {
                tracing::warn!(
                    stream_slab_vram = fit.stream_slab_vram,
                    error = %e,
                    staging_ring_bytes = self.stream.as_ref().map_or(0, |s| s.staged_bytes()),
                    "prefill-stream disabled: the stream slabs do not fit alongside a \
                     K ≥ top-k device cache; continuing without streaming (the pinned \
                     staging ring stays allocated for the session)"
                );
                self.stream = None;
                self.stream_cfg = None;
                let k = k_of(free)?;
                self.k_decode = k;
                k
            }
            Err(e) => return Err(e),
        };
        // SAFETY: FFI setter on the live ctx between computes.
        unsafe { raw::golem_set_device_cache_experts(ctx, k as i32) };
        self.device_cache_experts = k;
        self.k_stream = k;
        self.vram_stream = true;
        tracing::info!(
            k_stream = k,
            k_decode = self.k_decode,
            "device cache fitted"
        );
        Ok(())
    }

    /// Sizes the device cache for streamed prefill (`k_stream`, stream slabs
    /// may allocate) or decode (`k_decode`, stream slabs freed). A K change
    /// reallocates every C slab entry anyway (`st.K != K`), so the switch is
    /// a full reset: it frees the stream slabs and the old-K slabs before
    /// the new ones allocate, so peak VRAM never holds both sizes. Entering
    /// stream mode re-checks free VRAM against the slabs, whose allocation
    /// aborts on failure; when they no longer fit, streaming is dropped for
    /// the session. Call between computes only.
    pub(crate) fn set_vram_mode(&mut self, stream: bool) {
        if stream == self.vram_stream {
            return;
        }
        self.vram_stream = stream;
        let k = vram_mode_k(stream, self.k_stream, self.k_decode);
        if k == self.device_cache_experts {
            return;
        }
        let ctx = self.ctx_ptr();
        for p in self.prefetch.iter().chain(&self.warm) {
            p.pause();
        }
        // SAFETY: FFI on the live ctx between computes (every decode
        // synchronizes before returning; no slab in flight).
        unsafe { raw::golem_reset_device_cache(ctx) };
        self.device_cache_reset();
        if stream {
            match self.refit_stream_k() {
                Some(k) => self.k_stream = self.k_stream.min(k),
                None => {
                    tracing::warn!(
                        stream_slab_vram = self.stream_slab_vram,
                        "prefill-stream disabled: free VRAM no longer fits the stream \
                         slabs beside a K ≥ top-k cache; continuing without streaming"
                    );
                    self.stream = None;
                    self.stream_cfg = None;
                    self.k_stream = self.k_decode;
                }
            }
        }
        let k = vram_mode_k(stream, self.k_stream, self.k_decode);
        // SAFETY: FFI setter on the live ctx between computes.
        unsafe { raw::golem_set_device_cache_experts(ctx, k as i32) };
        tracing::debug!(
            from = self.device_cache_experts,
            to = k,
            stream,
            "device cache resized"
        );
        self.device_cache_experts = k;
        for p in self.prefetch.iter().chain(&self.warm) {
            p.resume();
        }
    }

    /// K the free VRAM (after a reset) supports beside both stream slabs,
    /// whose allocation aborts on failure while a K slab's fails soft;
    /// `None` below top-k. Without a VRAM query the startup fit stands.
    fn refit_stream_k(&self) -> Option<u32> {
        #[cfg(golem_backend_hip)]
        if let Some((free, _)) = golem_llama_sys::pinned::free_device_mem() {
            return k_from_free_vram(
                (free as u64).saturating_sub(self.stream_slab_vram),
                u64::MAX,
                self.slab_bytes_per_expert,
                self.top_k,
            )
            .ok();
        }
        Some(self.k_stream)
    }

    /// Largest K a prefill chunk may run under (`chunk_cap` = K / top-k):
    /// a routed-only chunk can run in either mode.
    pub(crate) fn prefill_k(&self) -> u32 {
        self.k_stream.min(self.k_decode)
    }

    /// Test seam, between turns only: overrides both Ks (an explicit-K
    /// session never splits them) and puts the cache in stream mode at
    /// `k_stream`.
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn test_set_ks(&mut self, k_stream: u32, k_decode: u32) {
        self.k_stream = k_stream;
        self.k_decode = k_decode;
        self.vram_stream = true;
        let ctx = self.ctx_ptr();
        // SAFETY: FFI on the live ctx between computes.
        unsafe { raw::golem_reset_device_cache(ctx) };
        self.device_cache_reset();
        // SAFETY: FFI setter on the live ctx between computes.
        unsafe { raw::golem_set_device_cache_experts(ctx, k_stream as i32) };
        self.device_cache_experts = k_stream;
    }

    /// Caps the pending derived-K fit at the §10 plan's `device_expert_cache`
    /// bytes. No-op for an explicit K.
    pub(crate) fn cap_device_cache(&mut self, bytes: u64) {
        if let Some(f) = self.fit.as_mut() {
            f.plan_cap = bytes;
        }
    }

    /// After a `llama_decode` −2: when C reports a new device-slab allocation
    /// failure (ABI v12), frees every slab, halves K (never below top-k) and
    /// returns it; `None` when the −2 was not a slab failure. The next compute
    /// reallocates the smaller slabs; the session serves degraded. A failure
    /// with K already at top-k cannot shrink further → `BackendUnavailable`
    /// (fatal: one routed layer no longer fits on the device).
    pub(crate) fn shrink_after_slab_failure(&mut self) -> Result<Option<u32>, EngineError> {
        let ctx = self.ctx_ptr();
        // SAFETY: FFI counter read on the live ctx.
        let failures = unsafe { raw::golem_slab_alloc_failures(ctx) };
        if failures == self.slab_failures_seen {
            return Ok(None);
        }
        self.slab_failures_seen = failures;
        // SAFETY: FFI on the live ctx between computes (the failed decode
        // was synchronized; no slab in flight).
        unsafe { raw::golem_reset_device_cache(ctx) };
        self.device_cache_reset();
        if self.device_cache_experts <= self.top_k {
            return Err(EngineError::BackendUnavailable(format!(
                "device expert-cache slab allocation failed at the K = top-k = {} floor",
                self.top_k
            )));
        }
        let k = (self.device_cache_experts / 2).max(self.top_k);
        // SAFETY: FFI setter on the live ctx between computes.
        unsafe { raw::golem_set_device_cache_experts(ctx, k as i32) };
        tracing::warn!(
            from = self.device_cache_experts,
            to = k,
            "device expert-cache slab allocation failed (VRAM taken after load); \
             cache reset and shrunk, serving degraded"
        );
        self.device_cache_experts = k;
        self.k_stream = self.k_stream.min(k);
        self.k_decode = self.k_decode.min(k);
        Ok(Some(k))
    }

    /// Whether prefill streaming is enabled for this session.
    pub fn stream_enabled(&self) -> bool {
        self.stream.is_some()
    }

    /// Arms/disarms stream chunks (§7.6): flips the C flag, the routing
    /// readback's read-ahead, and pauses the prefetcher for the chunk
    /// (stream chunks must not contend for the disk queue or pool slots).
    /// No-op when streaming is disabled.
    pub fn set_prefill_stream(&self, on: bool) {
        let Some(_s) = &self.stream else {
            return;
        };
        self.state.stream_armed.store(on, Ordering::Relaxed);
        for p in self.prefetch.iter().chain(&self.warm) {
            if on {
                p.pause();
            } else {
                p.resume();
            }
        }
        // SAFETY: FFI flag setter on the live ctx; not concurrent with a
        // compute (called between llama_decode calls).
        if let Some((n_experts, max_stride, routed_max)) = self.stream_cfg {
            // SAFETY: FFI config setter on the live ctx.
            unsafe {
                raw::golem_set_prefill_stream(
                    self.ctx_ptr(),
                    on as i32,
                    n_experts,
                    max_stride,
                    routed_max,
                )
            };
        }
    }

    /// Points the stream-source abort check at `cancel` for one turn; the
    /// guard nulls it on drop. Owns clones of both Arcs, so it borrows
    /// nothing from the session.
    pub(crate) fn stream_abort_guard(&self, cancel: &Arc<AtomicBool>) -> StreamAbortGuard {
        self.state
            .abort
            .store(Arc::as_ptr(cancel).cast_mut(), Ordering::Release);
        StreamAbortGuard {
            state: self.state.clone(),
            _cancel: cancel.clone(),
        }
    }

    /// Test seam: sets the active turn's cancel flag from inside the
    /// `n`-th `stream_source` call (1-based, counted from now), so the
    /// abort fires mid-ubatch; `0` disables.
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn cancel_at_stream_plane_for_test(&self, n: u64) {
        let base = self.stream_stats().map_or(0, |s| s.planes);
        let at = if n == 0 { 0 } else { base + n };
        self.state.test_cancel_plane.store(at, Ordering::Relaxed);
    }

    /// Test seam: the next `stream_source` call returns
    /// `GOLEM_STREAM_SRC_CANCEL` with no cancel flag set.
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn spurious_stream_cancel_for_test(&self) {
        self.state
            .test_spurious_cancel
            .store(true, Ordering::Relaxed);
    }

    /// Drops every staging slot and pool lease the stream chunks still
    /// hold (a failed decode never reaches copy_done's backstop). Call only
    /// with no compute in flight.
    pub(crate) fn release_stream(&self) {
        if let Some(s) = &self.stream {
            s.release_all();
        }
    }

    /// Leases the pool's batched step claims still hold (0 between turns).
    pub fn pool_batch_leases(&self) -> usize {
        self.state.pool.batch_leases()
    }

    /// Cumulative stream counters (test/observability seam).
    pub fn stream_stats(&self) -> Option<crate::stream::StreamStats> {
        self.stream.as_ref().map(|s| s.stats())
    }

    /// The substrate's device cache was reset between turns (session.rs
    /// calls `golem_reset_device_cache`): the VRAM copies are gone, so the
    /// exclusive-residency presence bits must go too. The staged batch's
    /// leases are dropped WITHOUT releasing slots first: otherwise the next
    /// `copy_done`'s end_batch_step_exclusive would release the RAM copies
    /// of units whose slab entries no longer exist and set stale VRAM bits —
    /// a unit in neither tier.
    pub fn device_cache_reset(&self) {
        self.state.pool.reset_exclusive_batch();
        self.state.pool.clear_all_vram();
    }

    /// Stage 1 of the teardown order: unregister + join (drops `Hooked`,
    /// recovering the payload Arc). `Session` calls this explicitly BEFORE
    /// `llama_free`; the remaining teardown (publisher/prefetch stop, then
    /// `golem_ctx_free`, then the pool) rides [`Drop for SessionHooks`].
    /// The live ctx pointer while hooks are registered (or before the Drop
    /// clears it): the Session teardown's slab-reclamation seam —
    /// `golem_reset_device_cache` must run while the substrate backend is
    /// still alive, i.e. before `llama_free`.
    pub(crate) fn live_ctx_ptr(&self) -> raw::golem_ctx_t {
        self.state.ctx_ptr.load(Ordering::SeqCst)
    }

    /// Unregisters the hooks and joins in-flight callbacks (idempotent; the
    /// §3.4 Session teardown calls this before `llama_free`, the Drop runs it
    /// as a backstop).
    pub fn unregister(&mut self) {
        self.hooked = None;
        CURRENT
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .retain(|s| !Arc::ptr_eq(s, &self.state));
    }

    /// Starts the §6.3 warm-up pre-fill once K is final (after
    /// [`Self::fit_device_cache`]): the restored keys, best-first, truncated
    /// per slot class to `slots − (layers_c × k + 1)` with k = min(max(K
    /// stream, K decode), n_experts) — `class_layout`'s floor — so demand
    /// keeps its in-flight headroom and the CLOCK sweep never evicts an
    /// earlier, hotter warm unit. Background thread; no request waits on
    /// it. No-op without a restored warm state or after the first call.
    /// GLM, 59 GiB pool: ≈ 3,900 units ≈ 59 GiB / 4.7 GB/s ≈ 13 s of reads.
    pub(crate) fn start_warm_fill(&mut self) {
        let Some(fill) = self.warm_fill.take() else {
            return;
        };
        let n = self.state.n_experts;
        let k = self.k_stream.max(self.k_decode).min(n);
        let mut budget: Vec<u64> = fill
            .classes
            .iter()
            .map(|&(slots, layers)| slots.saturating_sub(layers * u64::from(k) + 1))
            .collect();
        let units: Vec<(u32, u32)> = self
            .state
            .policy
            .ranked_keys()
            .into_iter()
            .filter_map(|i| {
                let (layer, expert) = ((i / n as usize) as u32, (i % n as usize) as u32);
                let b = &mut budget[usize::from(fill.class_of_layer[layer as usize])];
                (*b > 0).then(|| {
                    *b -= 1;
                    (layer, expert)
                })
            })
            .collect();
        tracing::info!(units = units.len(), k, "warm-start pool pre-fill started");
        if units.is_empty() {
            return;
        }
        #[cfg(any(test, feature = "test-support"))]
        {
            self.warm_planned = units.clone();
        }
        self.warm = Some(spawn_prefetch(
            self.state.pool.clone(),
            Arc::new(WarmFeed::new(units)) as Arc<dyn AccessFeed>,
            fill.io_cpuset,
        ));
    }

    /// Writes the §6.2 policy words and step to the warm-start file
    /// (atomic replace; parent created). Call between turns only. No-op
    /// when no warm-start file is configured.
    pub fn save_warm_state(&self) -> std::io::Result<()> {
        let Some((path, shape)) = &self.warm_file else {
            return Ok(());
        };
        let (step, words) = self.state.policy.warm_words();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        crate::atomic::write_atomic(
            path,
            &golem_placement::scores::format_scores(shape, step, &words),
        )
    }

    /// Test seam: the live policy step and every booked key's words.
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn warm_words_for_test(&self) -> (u64, Vec<(usize, u64, u64)>) {
        self.state.policy.warm_words()
    }

    /// Test seam: true once the pre-fill thread has exited (feed drained),
    /// or when none was started.
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn warm_fill_done(&self) -> bool {
        self.warm.as_ref().is_none_or(PrefetchHandle::finished)
    }

    /// Test seam: the `(layer, expert)` units the pre-fill was handed.
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn warm_planned_for_test(&self) -> &[(u32, u32)] {
        &self.warm_planned
    }

    /// Test seam: whether `(layer, expert)` is pool-resident.
    #[cfg(any(test, feature = "test-support"))]
    #[doc(hidden)]
    pub fn is_resident_for_test(&self, layer: u32, expert: u32) -> bool {
        self.state.pool.is_resident(layer, expert)
    }

    /// Flushes this turn's routing-trace records into `w` (header on first
    /// call, then one line per record; the ring clears for the next turn).
    /// No-op when the routing trace is disabled.
    pub fn write_routing_trace(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        let Some(trace) = &self.state.trace else {
            return Ok(());
        };
        let mut t = trace.lock().unwrap_or_else(|e| e.into_inner());
        t.flush(w, self.state.n_layers, self.top_k)
    }

    /// Drops buffered records without touching the header (the pre-turn
    /// stale-record guard — a sink flush would swallow the header write).
    pub fn drain_routing_trace(&self) {
        if let Some(trace) = &self.state.trace {
            trace
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clear_records();
        }
    }
}

impl Drop for SessionHooks {
    fn drop(&mut self) {
        // 1. Unregister + join (skipped if `unregister` already ran — the
        //    §3.4 Session order calls it before `llama_free`): no C callback
        //    can touch the payload after this returns (and the payload Arc
        //    is recovered by the guard). The CURRENT window goes with it:
        //    clearing before the ctx free makes a subsequent enable() see a
        //    clean slate.
        self.unregister();
        // 2. Stop the publisher thread FIRST (it reads the ctx pointer
        //    window), then clear the pointer (post-join: no reader left).
        self.publisher.stop_and_join();
        self.state
            .ctx_ptr
            .store(std::ptr::null_mut(), Ordering::SeqCst);
        if let Some(s) = &self.stream {
            s.stop();
        }
        for p in self.prefetch.iter().chain(&self.warm) {
            p.stop();
        }
        // 3. `ctx_keep` (last field) drops now — golem_ctx_free aborts while
        //    a sched attach refcount remains, and `llama_free` (which ran
        //    earlier in the Session order) has already run the sched
        //    destructors. The pool/prefetch state drops with `state`.
        debug_assert_eq!(
            Arc::strong_count(&self.ctx_keep),
            1,
            "the ctx Arc must be exclusive before golem_ctx_free"
        );
    }
}

/// Handle to the order publisher thread. `stop_and_join` is explicit (drop
/// order inside [`SessionHooks`]); `Drop` also signals + joins as backstop.
struct PublisherHandle {
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl PublisherHandle {
    fn stop_and_join(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for PublisherHandle {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

/// Spawns the order publisher: wakes on a bounded poll, sorts every dirty
/// layer's recency row into preallocated scratch and pushes all three kinds
/// into the ctx's triple buffers. Never blocks the decode thread: when no
/// buffer slot is free under the epoch gate the push is skipped and the
/// layer stays dirty (retried next pass).
fn spawn_order_publisher(state: Arc<HookState>, interval: std::time::Duration) -> PublisherHandle {
    let stop = Arc::new(AtomicBool::new(false));
    let thread = {
        let stop = stop.clone();
        std::thread::Builder::new()
            .name("golem-order-publisher".into())
            .spawn(move || {
                // Off-hot-path scratch: allocated once per thread, never in
                // the publish loop.
                let mut order = OrderScratch::new(state.n_experts as usize);
                loop {
                    if stop.load(Ordering::Relaxed) {
                        return;
                    }
                    if state.policy.dirty_any.load(Ordering::Relaxed) {
                        publish_dirty(&state, usize::MAX, &mut order);
                    }
                    std::thread::park_timeout(interval);
                }
            })
            .expect("spawn order publisher")
    };
    PublisherHandle {
        stop,
        thread: Some(thread),
    }
}

/// Publishes up to `limit` dirty layers into the ctx's triple buffers.
/// Returns how many layers were published. Each layer is claimed with a
/// swap-to-false (the publisher and the `copy_done` flush are the two
/// competitors; exactly one wins a given dirty bit). A failed push (no free
/// slot under the epoch gate) re-marks the layer dirty and counts a skip —
/// never blocks, never overwrites a buffer a reader may still hold.
fn publish_dirty(state: &HookState, limit: usize, scratch: &mut OrderScratch) -> usize {
    // SAFETY: the ctx pointer is non-null only while hooks are registered;
    // it is cleared after the unregister-join, and this runs either on the
    // decode thread inside a registered hook (copy_done flush) or on the
    // publisher thread, which is stopped before the pointer is cleared.
    let ctx = state.ctx_ptr.load(Ordering::SeqCst);
    if ctx.is_null() {
        return 0;
    }
    let ctx: raw::golem_ctx_t = ctx;
    let n = state.n_experts as usize;
    let mut published = 0;
    for layer in 0..state.policy.n_layers {
        if published >= limit {
            break;
        }
        if !state.policy.dirty[layer].swap(false, Ordering::Relaxed) {
            continue;
        }
        state.policy.build_order(layer, scratch);
        let blk = layer as i32 + state.dense_lead as i32;
        let mut all_pushed = true;
        for kind in 1..=3 {
            // SAFETY: `ctx` is live (see above); `scratch.order` holds n
            // initialized i32s.
            let ok = unsafe {
                raw::golem_policy_order_push(ctx, blk, kind, scratch.order.as_ptr(), n as i32)
            };
            if ok == 0 {
                all_pushed = false;
                state.order_pushes_failed.fetch_add(1, Ordering::Relaxed);
            }
        }
        if all_pushed {
            published += 1;
        } else {
            state.policy.mark_dirty(layer);
        }
    }
    if published > 0 {
        state.policy.dirty_any.store(
            state.policy.dirty.iter().any(|d| d.load(Ordering::Relaxed)),
            Ordering::Relaxed,
        );
    }
    published
}

/// Builds the RAM expert pool for `config` (which must set
/// `ram_experts_gib > 0`), creates the per-model `golem_ctx` (ABI v5) and
/// registers the session's hooks. Returns the guards: dropping them
/// unregisters (join) and frees the ctx.
///
/// # Errors
/// GGUF/geometry failures, arena mmap/mlock failures (no silent unlocked
/// fallback, §5.4), or a concurrent second pool-enabled construction (one
/// pool-enabled session at a time: two slab sets at K=10 ≈ 10 × 42 × 3 ×
/// 5.8 MB ≈ 7.3 GB each on a 24 GiB card).
pub fn enable(config: &EngineConfig) -> Result<SessionHooks, EngineError> {
    debug_assert!(config.ram_experts_gib > 0, "enable() with pool disabled");
    // Refuse BEFORE allocating anything: a pool enabled while any session is
    // live would silently re-route that session's subsequent decodes through
    // the new hooks (they fire per step regardless of which session owns
    // them).
    refuse_if_pool_session_live()?;
    // CURRENT's lock held across construction: concurrent enables serialize
    // (each on its own ctx; the multi-ctx capability the ABI v5 surface is
    // built for) and never race a half-built state into the registry.
    let result = enable_inner(config);
    if let Ok(hooks) = &result {
        CURRENT
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .push(hooks.state.clone());
    }
    result
}

/// The header constant and the compiled substrate's reported ABI version must
/// agree exactly; a mismatch means the pinned substrate must be rebuilt.
/// Unit tests inject a mismatched header here to prove enable_inner checks
/// the ABI before touching the model path; 0 means "use the real constant".
#[cfg(test)]
static TEST_ABI_HEADER: AtomicI32 = AtomicI32::new(0);

fn abi_header_for_check() -> i32 {
    #[cfg(test)]
    {
        let injected = TEST_ABI_HEADER.load(Ordering::Relaxed);
        if injected != 0 {
            return injected;
        }
    }
    raw::GOLEM_HOOKS_ABI_VERSION as i32
}

fn check_abi(header: i32, compiled: i32) -> Result<(), EngineError> {
    if header != compiled {
        return Err(EngineError::PoolInit(format!(
            "golem hooks ABI mismatch: header GOLEM_HOOKS_ABI_VERSION={header} vs compiled substrate {compiled}; rebuild the pinned substrate (scripts/build-substrate.sh)"
        )));
    }
    Ok(())
}

/// `check_abi` narrows the header constant to `i32` to compare against the
/// getter's `i32` return; fail compilation if it ever outgrows that.
const _: () = assert!(
    (raw::GOLEM_HOOKS_ABI_VERSION as i64) >= i32::MIN as i64
        && (raw::GOLEM_HOOKS_ABI_VERSION as i64) <= i32::MAX as i64
);

fn enable_inner(config: &EngineConfig) -> Result<SessionHooks, EngineError> {
    // The compiled substrate's hook surface must match the header the bindings
    // were generated from — checked before any source open or allocation, so a
    // stale build fails fast instead of silently decoding with a drifted ABI.
    check_abi(abi_header_for_check(), golem_llama_sys::hooks_abi_version())?;
    if config.prefetch != PrefetchMode::Off && !(1..=4).contains(&config.prefetch_lookahead) {
        return Err(EngineError::InvalidConfig(format!(
            "prefetch_lookahead {} outside 1..=4",
            config.prefetch_lookahead
        )));
    }

    // Pool geometry from the GGUF file itself (header + tensor table only).
    let metadata = crate::gguf::read_metadata(&config.model_path)?;
    let arch = metadata
        .kv_str("general.architecture")
        .unwrap_or_default()
        .to_owned();
    // Dense lead (e.g. glm4moe.leading_dense_block_count = 1): expert tensors
    // exist only from blk.<dense_lead> on, and pool layers are renumbered to
    // exclude the lead (same mapping as GgufPreadSource::open).
    let dense_lead = metadata
        .kv_u64(&format!("{arch}.leading_dense_block_count"))
        .unwrap_or(0);
    let dense_lead = u32::try_from(dense_lead).map_err(|_| {
        EngineError::PoolInit(format!("leading_dense_block_count {dense_lead} < 0"))
    })?;

    // Pool geometry from the GGUF walk (header + tensor table only) in both
    // modes: the manifest path cross-checks its own offsets against the same
    // tables at open, so a divergence here is a construction error, not a
    // silent geometry mismatch.
    let geo = GgufPreadSource::open(&config.model_path, dense_lead)
        .map_err(|e| EngineError::PoolInit(e.to_string()))?
        .geometry()
        .map_err(|e| EngineError::PoolInit(e.to_string()))?
        .ok_or_else(|| EngineError::PoolInit("model has no expert tensors".into()))?;

    // Expert byte source: the store-backed reader when a manifest is
    // configured (§5.2 — the shared queue drives the P3 slow path; the
    // queue lives in the session's source Arc), else the single-replica GGUF
    // pread path.
    let (source, store): (Arc<dyn ExpertSource>, Option<Arc<StoreExpertSource>>) =
        match &config.manifest_path {
            Some(manifest_path) => {
                let queue = Arc::new(
                    ChunkQueue::new(config.io_threads.unwrap_or(0), config.io_cpuset.clone())
                        .with_io_depth(config.io_depth as usize),
                );
                let store = Arc::new(
                    StoreExpertSource::open(manifest_path, queue)
                        .map_err(|e| EngineError::PoolInit(e.to_string()))?,
                );
                if config.o_direct {
                    for r in store.registry().shards().iter().flat_map(|s| &s.replicas) {
                        r.require_direct()
                            .map_err(|e| EngineError::PoolInit(format!("o_direct: {e}")))?;
                    }
                }
                let source: Arc<dyn ExpertSource> = store.clone();
                (source, Some(store))
            }
            None => {
                let source = Arc::new(
                    GgufPreadSource::open(&config.model_path, dense_lead)
                        .map_err(|e| EngineError::PoolInit(e.to_string()))?,
                );
                (source, None)
            }
        };

    let n_layers = geo.n_layers();
    let pool_bytes = config
        .ram_experts_gib
        .checked_mul(1 << 30)
        .ok_or_else(|| EngineError::PoolInit("ram_experts_gib overflows bytes".into()))?;
    // Device-slab pricing input: total unit bytes across the model — one
    // slab slot costs Σ_layers unit bytes (§3.4).
    let slab = slab_bytes_per_expert(&geo);
    let n_expert_used = metadata
        .kv_u64(&format!("{arch}.expert_used_count"))
        .ok_or_else(|| {
            EngineError::PoolInit(format!("{arch}.expert_used_count missing from the GGUF"))
        })?;
    let n_expert_used = u32::try_from(n_expert_used)
        .map_err(|_| EngineError::PoolInit("expert_used_count > u32::MAX".into()))?;
    // Prefill-stream slab pricing (§7.6.1): two device slabs, each
    // n_experts × max plane stride. Priced BEFORE the K derivation so K
    // comes off the exact remaining budget (user ruling R3).
    let max_plane_stride = geo
        .layers
        .iter()
        .flat_map(|l| [ExpertKind::Gate, ExpertKind::Up, ExpertKind::Down].map(|k| l.plane(k).1))
        .max()
        .unwrap_or(0);
    let stream_slab_vram = if config.prefill_stream {
        2u64 * u64::from(geo.n_experts) * max_plane_stride
    } else {
        0
    };
    // The device-cache budget is a per-layer expert count, never a
    // byte cap on whichever tensor registers first.
    // It is never "unlimited": device allocations are bounded by the VRAM
    // budget, never proportional to expert count (unbounded slabs are ≈184 GB
    // on GLM-5.3-Flash — 126 live tensors × 288 experts × ≈5 MB — against a
    // 24 GiB card). An explicit config value wins; otherwise K derives from
    // the free VRAM at enable time against the exact budget (R3: free −
    // stream slabs − headroom; the measured `free` already nets out the
    // non-expert weights, KV and the compute buffer — they were allocated
    // at load). With streaming requested but K < top_k under the slab
    // budget, streaming auto-disables with a warning (never fails enable);
    // K below top-k without streaming is still a hard error (a routed set
    // larger than K aborts mid-decode).
    // `k` is the explicit K, or (derived) an upper bound priced off the
    // free VRAM before the model loads; [`SessionHooks::fit_device_cache`]
    // sets the real K once the weights, KV and compute buffer are resident.
    let (k, derived_k) = derive_device_cache_experts(config, slab, n_expert_used)?;
    let stream_enabled = config.prefill_stream;
    // Per-class RAM layout (§5.4): each unit-size class gets slots
    // proportional to its share of total unit bytes; remainder bytes are
    // trimmed by integer division.
    let classes = geo.classes();
    // C clamps the slab to n_experts, so that is the per-layer staging bound.
    let layout = class_layout(&classes, pool_bytes, k.min(geo.n_experts))?;
    let new_arena = || golem_placement::pool::Arena::new(&layout);
    let arena = if config.numa_interleave {
        golem_store::sys::with_numa_interleave(new_arena)
            .map_err(|e| EngineError::PoolInit(format!("numa_interleave: {e}")))?
    } else {
        new_arena()
    }
    .map_err(|e| EngineError::PoolInit(e.to_string()))?;
    // Layer → class index, aligned with `layout`/`classes` ordering.
    let mut class_of_layer = vec![0u8; n_layers as usize];
    for (c, (_, layers)) in classes.iter().enumerate() {
        let c = u8::try_from(c)
            .map_err(|_| EngineError::PoolInit("more than 256 unit-size classes".into()))?;
        for &l in layers {
            class_of_layer[l as usize] = c;
        }
    }

    #[cfg(golem_backend_hip)]
    let pool_range = arena.as_ptr_range();

    // Empty policy snapshot (nothing pinned/protected) until a
    // placement-backed snapshot builder feeds the pool (§6.6).
    let snapshots = Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::empty())));
    let pool = ExpertPool::try_new(
        arena,
        source.clone(),
        snapshots,
        Arc::new(P3Stats::new()),
        geo.clone(),
        class_of_layer.clone(),
    )
    .map_err(|e| EngineError::PoolInit(e.to_string()))?;

    // Prefill-stream state (§7.6, ruling R3 kept streaming on): the pinned
    // staging ring + reader thread (GPU-registered with the pool arena below).
    let stream = if stream_enabled {
        let slot_bytes = geo.n_experts as usize * max_plane_stride as usize;
        let health = store.clone();
        let storage_lost = Box::new(move || {
            health
                .as_ref()
                .is_some_and(|s| s.health() == golem_store::InstanceHealth::FatalIo)
        });
        let st = StreamState::new(pool.clone(), source.clone(), storage_lost, slot_bytes)
            .map_err(|e| EngineError::PoolInit(format!("stream staging ring: {e}")))?;
        Some(st)
    } else {
        None
    };

    // Design §3.5: mlock alone leaves the H2D copy pageable — driver bounce
    // buffers, ~half of PCIe bandwidth, not truly async, and PCIe is the
    // §2.4 bottleneck. A failed registration is a startup failure, never a
    // silent downgrade. Declared after `pool`/`stream` so an early return
    // unregisters before either arena unmaps.
    #[cfg(golem_backend_hip)]
    let pinned = {
        let mut ranges = vec![("expert pool", pool_range)];
        if let Some(st) = &stream {
            ranges.push(("prefill-stream staging ring", st.arena_ptr_range()));
        }
        let mut regs = Vec::with_capacity(ranges.len());
        for (what, (ptr, len)) in ranges {
            // SAFETY: a live, page-aligned, solely-owned mlocked mapping of
            // exactly `len` bytes, kept mapped by `pool`/`stream` while the
            // registration lives; no slot has been handed to the device yet.
            // The registration drops with `SessionHooks`; its drop
            // synchronizes the device before unregistering, so a drop
            // without a prior `llama_free` cannot race in-flight DMA.
            let reg = unsafe { golem_llama_sys::pinned::HostRegistration::new(ptr, len) }.map_err(
                |r| {
                    EngineError::PoolInit(format!(
                        "hipHostRegister failed on the {len}-byte {what} (HIP result {r})"
                    ))
                },
            )?;
            regs.push(reg);
            tracing::info!(gib = len as f64 / (1 << 30) as f64, "{what} GPU-pinned");
        }
        regs
    };

    // Per-model hook context (ABI v5). n_layers covers the full blk range —
    // C indexes its order buffers by the raw blk number, and the dense lead
    // has no expert tensors but does not shift blk numbering.
    let ctx = GolemCtx::create((dense_lead + n_layers) as i32, geo.n_experts as i32)
        .ok_or_else(|| EngineError::PoolInit("golem_ctx_create failed".into()))?;
    // SAFETY: FFI setter on a live ctx; bounded per-layer slab (never
    // unlimited, never proportional to total experts — AGENTS.md).
    // A derived K starts at top-k (the smallest legal slab) for the load's
    // self-check decode; `fit_device_cache` raises it.
    let k_start = if derived_k { n_expert_used.min(k) } else { k };
    unsafe { raw::golem_set_device_cache_experts(ctx.as_ptr(), k_start as i32) };
    // SAFETY: FFI config setter on a live ctx, no compute in flight.
    if config.async_h2d && unsafe { raw::golem_set_async_h2d(ctx.as_ptr(), 1) } != 0 {
        return Err(EngineError::InvalidConfig(
            "async_h2d: GGML_CUDA_GRAPH_OPT=1 owns the copy-stream index".into(),
        ));
    }
    // Stash the stream config C-side (slabs allocate lazily on the first
    // streamed split's backend); chunks arm/disarm via SessionHooks.
    let mut stream_cfg = None;
    if stream.is_some() {
        // SAFETY: FFI config setter on a live ctx, no compute in flight.
        unsafe {
            raw::golem_set_prefill_stream(
                ctx.as_ptr(),
                0,
                geo.n_experts as i32,
                max_plane_stride as i32,
                config.prefill_stream_routed_max,
            )
        };
        stream_cfg = Some((
            geo.n_experts as i32,
            max_plane_stride as i32,
            config.prefill_stream_routed_max,
        ));
    }

    // P2 state: per-expert recency and §6.2 decayed score; the evict-first
    // order sorts by `config.vram_order`.
    let policy = PolicyState::new(n_layers, geo.n_experts, config.vram_order);
    let warm_file = config.warm_state_dir.as_ref().and_then(|dir| {
        warm_shape(config, n_layers, geo.n_experts).map(|s| {
            let name = golem_placement::scores::scores_file_name(&s.model);
            (dir.join(name), s)
        })
    });
    let warm_fill = warm_file
        .as_ref()
        .is_some_and(|(path, shape)| restore_warm(&policy, path, shape))
        .then(|| WarmFill {
            classes: classes
                .iter()
                .zip(&layout)
                .map(|((_, layers), c)| (u64::from(c.slots), layers.len() as u64))
                .collect(),
            class_of_layer,
            io_cpuset: config.io_cpuset.clone(),
        });
    // Routing-trace ring (§11 pattern), preallocated at enable — overflow
    // drops and counts instead of reallocating on the hot path.
    let trace = config.routing_trace.as_ref().map(|_| {
        let pred = if config.prefetch == PrefetchMode::Off {
            (0, 0)
        } else {
            (config.prefetch_lookahead, n_expert_used + PREDICT_MARGIN)
        };
        let (rec_cap, id_cap) = trace_caps(
            config.n_ctx,
            config.n_predict,
            n_layers,
            n_expert_used,
            pred,
        );
        // Model file (hashed into the header), not the trace output path.
        Mutex::new(TraceRing::new(config.model_path.clone(), rec_cap, id_cap))
    });
    let state = Arc::new(HookState {
        p3_wait_ns: AtomicU64::new(0),
        slow_calls: AtomicUsize::new(0),
        routing_calls: AtomicUsize::new(0),
        ensure_calls: AtomicUsize::new(0),
        copy_done_calls: AtomicUsize::new(0),
        order_pushes_failed: AtomicUsize::new(0),
        tensors: Mutex::new(HashMap::new()),
        last_entry: Mutex::new(None),
        ctx_ptr: AtomicPtr::new(ctx.as_ptr()),
        generation: {
            // SAFETY: FFI query on a live ctx.
            unsafe { raw::golem_ctx_generation(ctx.as_ptr()) }
        },
        miss_scratch: Mutex::new(Vec::with_capacity(3 * k.min(geo.n_experts) as usize)),
        evict_scratch: Mutex::new(Vec::with_capacity(geo.n_experts as usize)),
        flush_scratch: Mutex::new(OrderScratch::new(geo.n_experts as usize)),
        exclusive: config.residency == Residency::Exclusive,
        policy,
        n_layers,
        dense_lead,
        n_experts: geo.n_experts,
        pool,
        source,
        store,
        trace,
        pred: PredState::new(
            config.prefetch,
            n_layers,
            geo.n_experts,
            // Two steps of every layer's predictions; older ones are stale.
            2 * (config.prefetch_lookahead * (n_expert_used + PREDICT_MARGIN)) as usize
                * n_layers as usize,
        ),
        stream: stream.clone(),
        stream_armed: AtomicBool::new(false),
        abort: AtomicPtr::new(std::ptr::null_mut()),
        #[cfg(any(test, feature = "test-support"))]
        test_cancel_plane: AtomicU64::new(0),
        #[cfg(any(test, feature = "test-support"))]
        test_spurious_cancel: AtomicBool::new(false),
        #[cfg(any(test, feature = "test-support"))]
        test_fail_storage_in_read: AtomicBool::new(false),
    });
    if config.prefetch != PrefetchMode::Off {
        // SAFETY: FFI setter on the live ctx, before the first graph build;
        // `predict` is a valid C ABI fn receiving the registered payload.
        unsafe {
            raw::golem_set_predict(
                ctx.as_ptr(),
                config.prefetch_lookahead as i32,
                PREDICT_MARGIN as i32,
                Some(predict),
            )
        };
    }
    let prefetch = state.pred.queue.clone().map(|q| {
        spawn_prefetch(
            state.pool.clone(),
            q as Arc<dyn AccessFeed>,
            config.io_cpuset.clone(),
        )
    });
    // Order publisher: 250 µs poll — the order may lag ≥ 1 step by design;
    // the copy_done flush bounds the worst case without blocking.
    let publisher = spawn_order_publisher(state.clone(), std::time::Duration::from_micros(250));

    // SAFETY: `state` is shared with C as `Arc::into_raw` and recovered by
    // the guard's drop after the C unregister-join returns — no in-flight
    // callback can race the recovery. All three fn pointers are valid C
    // ABIs; the C side's n_copies == 1 abort is the defense-in-depth (the
    // engine config cannot express pipeline parallelism or multi-GPU to
    // pre-reject here). The ctx is consumed into the guard: the registration
    // borrow is encoded by moving, so the ctx always outlives the hooks.
    let exclusive = state.exclusive;
    let hooked = ctx.register(
        HookFns {
            ensure: Some(ensure_host_resident),
            routing: Some(routing),
            copy_done: Some(copy_done),
            evict_batch: exclusive.then_some(evict_dest_batch),
            evict_abandon: exclusive.then_some(evict_abandon),
            stream_source: state.stream.as_ref().map(|_| stream_source as _),
            stream_release: state.stream.as_ref().map(|_| stream_release as _),
        },
        state.clone(),
    );
    // The ctx keep-alive: the registration owns one Arc; this clone orders
    // the free AFTER `llama_free` in the §3.4 Session teardown.
    let hooked_ctx_keep = hooked.ctx_arc();

    tracing::info!(
        layers = n_layers,
        experts = geo.n_experts,
        classes = ?layout,
        gib = config.ram_experts_gib,
        "P3 RAM expert pool registered (per-model ctx, hooks set)"
    );
    Ok(SessionHooks {
        #[cfg(golem_backend_hip)]
        _pinned: pinned,
        prefetch,
        warm: None,
        warm_fill,
        warm_file,
        #[cfg(any(test, feature = "test-support"))]
        warm_planned: Vec::new(),
        publisher,
        hooked: Some(hooked),
        stats: state.pool.stats.clone(),
        state,
        ctx_keep: hooked_ctx_keep,
        device_cache_experts: k_start,
        slab_failures_seen: 0,
        fit: derived_k.then_some(DeviceFit {
            k_bound: k,
            slab_bytes_per_expert: slab,
            stream_slab_vram,
            plan_cap: u64::MAX,
        }),
        top_k: n_expert_used,
        stream: stream.clone(),
        stream_cfg,
        k_stream: k_start,
        k_decode: k_start,
        vram_stream: true,
        stream_slab_vram,
        slab_bytes_per_expert: slab,
    })
}

/// Pre-fill inputs captured at enable: per class `(slots, layers)`, the
/// layer → class map, and the I/O cpuset the thread pins to.
struct WarmFill {
    classes: Vec<(u64, u64)>,
    class_of_layer: Vec<u8>,
    io_cpuset: Option<golem_store::sys::CpuSet>,
}

/// The shape a warm-start file is keyed by — the one fingerprint of the
/// model, also naming the file; `None` (warned, warm start off) when the
/// model cannot be fingerprinted.
fn warm_shape(config: &EngineConfig, n_layers: u32, n_experts: u32) -> Option<WarmShape> {
    match golem_placement::scores::model_fingerprint(&config.model_path) {
        Ok(model) => Some(WarmShape {
            layers: n_layers,
            experts: n_experts,
            half_life: HALF_LIFE_STEPS,
            model,
        }),
        Err(e) => {
            tracing::warn!(error = %e, "warm start off: cannot fingerprint the model");
            None
        }
    }
}

/// Loads and restores `path` into `policy`; false (cold start) when the
/// file is absent, or — with a warning — unreadable, corrupt or recorded
/// for another shape or model.
fn restore_warm(policy: &PolicyState, path: &std::path::Path, shape: &WarmShape) -> bool {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            tracing::info!(path = %path.display(), "no warm-start scores; starting cold");
            return false;
        }
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "warm-start scores unreadable; starting cold");
            return false;
        }
    };
    match golem_placement::scores::parse_scores(&text, shape) {
        Ok(warm) => {
            policy.restore(&warm);
            tracing::info!(
                path = %path.display(),
                step = warm.step,
                keys = warm.words.len(),
                "warm-start scores restored"
            );
            true
        }
        Err(e) => {
            tracing::warn!(path = %path.display(), error = %e, "warm-start scores refused; starting cold");
            false
        }
    }
}

/// The device-cache K of a VRAM mode (#135).
fn vram_mode_k(stream: bool, k_stream: u32, k_decode: u32) -> u32 {
    if stream { k_stream } else { k_decode }
}

/// Per-layer device-slab expert count for this pool, and whether it is
/// derived. Explicit `device_cache_experts > 0` wins as-is. Otherwise the
/// result is an upper bound from the free VRAM before the model loads (the
/// weights, KV and compute buffer are not resident yet); the session refits
/// it after context creation. Never unlimited, never zero: a routed set
/// larger than K aborts, so K = 0 could not decode at all.
fn derive_device_cache_experts(
    config: &EngineConfig,
    #[cfg_attr(not(golem_backend_hip), allow(unused_variables))] slab_bytes_per_expert: u64,
    n_expert_used: u32,
) -> Result<(u32, bool), EngineError> {
    if config.device_cache_experts > 0 {
        let k = config.device_cache_experts;
        if k < n_expert_used {
            return Err(EngineError::PoolInit(format!(
                "device_cache_experts {k} < expert_used_count {n_expert_used} — a routed \
                 set larger than K aborts mid-decode; raise the explicit K"
            )));
        }
        // The C ABI takes an i32 and reads any value ≤ 0 as "full slab";
        // a wrapped cast would silently become the unbounded allocation.
        if i32::try_from(k).is_err() {
            return Err(EngineError::PoolInit(format!(
                "device_cache_experts {k} exceeds i32::MAX"
            )));
        }
        return Ok((k, false));
    }
    #[cfg(golem_backend_hip)]
    if let Some((free, _total)) = golem_llama_sys::pinned::free_device_mem() {
        return Ok((
            k_from_free_vram(free as u64, u64::MAX, slab_bytes_per_expert, n_expert_used)?,
            true,
        ));
    }
    Err(EngineError::PoolInit(
        "device_cache_experts unset and no VRAM query available to derive a bounded \
         slab budget — set device_cache_experts explicitly"
            .into(),
    ))
}

/// Inputs of the post-load derived-K fit (ruling R3).
struct DeviceFit {
    /// Pre-load bound; also the pool layout's in-flight bound.
    k_bound: u32,
    slab_bytes_per_expert: u64,
    /// Both prefill-stream slabs (0 without streaming); they allocate
    /// lazily, so the post-load free VRAM does not net them out yet.
    stream_slab_vram: u64,
    /// The §10 plan's `device_expert_cache` bytes ([`SessionHooks::cap_device_cache`]);
    /// `u64::MAX` = uncapped.
    plan_cap: u64,
}

/// Device bytes one slab slot costs across the model: Σ_layers unit bytes —
/// one routed expert touches one unit in EVERY expert layer (§3.4).
fn slab_bytes_per_expert(geo: &golem_placement::pool::PoolGeometry) -> u64 {
    geo.layers.iter().map(|l| l.bytes()).sum()
}

/// Per-class slot counts: class c gets ⌊pool_bytes × share_c / unit_c⌋
/// slots, share_c = layers_c × unit_c / Σ_layers unit. Err when a class
/// falls below layers_c × k + 1, k = min(K, n_experts): P2 stages at most k
/// routed experts per layer (more aborts in C, design §2.4 prefill), and one
/// step's leases live until the next compute's copy_done, so one step pins
/// at most that many units; the extra slot is prefetch headroom so a
/// prefetch fill never finds every slot leased (`Busy`).
fn class_layout(
    classes: &[(u64, Vec<u32>)],
    pool_bytes: u64,
    k: u32,
) -> Result<Vec<SlotClass>, EngineError> {
    let mut out = Vec::with_capacity(classes.len());
    for (&(unit, ref layers), slots) in classes
        .iter()
        .zip(golem_placement::pool::class_slots(classes, pool_bytes))
    {
        let slots = u32::try_from(slots)
            .map_err(|_| EngineError::PoolInit("slot count exceeds u32::MAX".into()))?;
        let floor = layers.len() as u64 * u64::from(k) + 1;
        if u64::from(slots) < floor {
            return Err(EngineError::PoolInit(format!(
                "RAM pool of {pool_bytes} B too small for the unit-size {unit} B class \
                 ({} layers): {slots} slots < {floor} (in flight: {} layers × \
                 device_cache_experts {k}, +1 prefetch headroom); raise ram_experts_gib or lower \
                 device_cache_experts",
                layers.len(),
                layers.len()
            )));
        }
        out.push(SlotClass {
            slot_size: usize::try_from(unit)
                .map_err(|_| EngineError::PoolInit("slot size > usize".into()))?,
            slots,
        });
    }
    Ok(out)
}

/// Derived K from free VRAM against the exact budget (ruling R3):
/// `⌊min(free − headroom, plan_cap) / slab⌋ − 1`, so K never exceeds the
/// §10 plan's `device_expert_cache` term nor what is actually free — the
/// stream slabs, KV, `ws` and the non-expert
/// weights were all subtracted (or already allocated) before this runs;
/// `VRAM_HEADROOM_BYTES` (1 GiB) covers allocator fragmentation and driver
/// drift. Rejected below top-k — a routed set larger than K aborts
/// mid-decode, it is never floored. `golem_plan::plan` refuses a
/// `device_expert_cache` below top-k by the same [`golem_plan::device_cache_k`];
/// the headroom is not in that minimum because free VRAM here still holds the
/// plan's `rocm_overhead` term (2.83 GB > 1 GiB), so `plan_cap` binds first.
#[cfg_attr(not(golem_backend_hip), allow(dead_code))]
const VRAM_HEADROOM_BYTES: u64 = 1 << 30;

#[cfg_attr(not(golem_backend_hip), allow(dead_code))]
fn k_from_free_vram(
    free: u64,
    plan_cap: u64,
    slab_bytes_per_expert: u64,
    n_expert_used: u32,
) -> Result<u32, EngineError> {
    let k = golem_plan::device_cache_k(
        free.saturating_sub(VRAM_HEADROOM_BYTES).min(plan_cap),
        slab_bytes_per_expert,
    )
    .clamp(1, i32::MAX as u64);
    if k < u64::from(n_expert_used) {
        return Err(EngineError::PoolInit(format!(
            "free VRAM {free} B (plan cap {plan_cap} B) supports a slab of only {k} experts per layer at \
             {slab_bytes_per_expert} B per routed expert (Σ unit bytes), below \
             expert_used_count {n_expert_used} — even single-token decode would abort; \
             free VRAM or lower the resident set"
        )));
    }
    Ok(k as u32)
}

/// Miss-list kind code from the ABI (1=gate 2=up 3=down) → plane.
fn kind_from_code(code: i32) -> Option<ExpertKind> {
    match code {
        1 => Some(ExpertKind::Gate),
        2 => Some(ExpertKind::Up),
        3 => Some(ExpertKind::Down),
        _ => None,
    }
}

/// The state-snapshot byte store (§7.1): one mlocked [`golem_placement::pool::Arena`],
/// GPU-registered on HIP builds (§3.5 — pageable H2D is a bug).
pub struct PinnedArena {
    /// Declared before `mem`: fields drop in order, so the range is
    /// unregistered before the arena unmaps. `None` only for heap test arenas.
    #[cfg(golem_backend_hip)]
    _reg: Option<golem_llama_sys::pinned::HostRegistration>,
    mem: PinnedMem,
}

enum PinnedMem {
    Locked(golem_placement::pool::Arena),
    #[cfg(any(test, feature = "test-support"))]
    Heap(Vec<u8>),
}

impl PinnedArena {
    /// Maps, locks and (HIP) registers `len` bytes; any failure refuses the
    /// load rather than degrading to pageable memory.
    pub fn new(len: usize) -> Result<Self, EngineError> {
        let arena = golem_placement::pool::Arena::new_bytes(len)
            .map_err(|e| EngineError::PoolInit(format!("snapshot arena: {e}")))?;
        #[cfg(golem_backend_hip)]
        let reg = {
            let (ptr, len) = arena.as_ptr_range();
            // SAFETY: the arena is a live, page-aligned, solely-owned mapping
            // of exactly `len` bytes, not yet registered; `_reg` drops before
            // `mem` unmaps, and its drop synchronizes the device first.
            unsafe { golem_llama_sys::pinned::HostRegistration::new(ptr, len) }.map_err(|r| {
                EngineError::PoolInit(format!(
                    "hipHostRegister failed on the {len}-byte snapshot arena (HIP result {r})"
                ))
            })?
        };
        Ok(Self {
            #[cfg(golem_backend_hip)]
            _reg: Some(reg),
            mem: PinnedMem::Locked(arena),
        })
    }

    /// Unpinned heap bytes for cache-logic unit tests.
    #[cfg(any(test, feature = "test-support"))]
    pub fn heap_for_test(len: usize) -> Self {
        Self {
            #[cfg(golem_backend_hip)]
            _reg: None,
            mem: PinnedMem::Heap(vec![0; len]),
        }
    }

    /// Usable bytes.
    pub fn len(&self) -> usize {
        match &self.mem {
            PinnedMem::Locked(a) => a.as_ptr_range().1,
            #[cfg(any(test, feature = "test-support"))]
            PinnedMem::Heap(v) => v.len(),
        }
    }

    /// True when the arena holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Shared view of the whole arena.
    pub fn bytes(&self) -> &[u8] {
        match &self.mem {
            PinnedMem::Locked(a) => {
                let (ptr, len) = a.as_ptr_range();
                // SAFETY: the arena owns `[ptr, ptr+len)` for its lifetime and
                // hands out no other view; `&self` excludes `bytes_mut`.
                unsafe { std::slice::from_raw_parts(ptr, len) }
            }
            #[cfg(any(test, feature = "test-support"))]
            PinnedMem::Heap(v) => v,
        }
    }

    /// Exclusive view of the whole arena.
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        match &mut self.mem {
            PinnedMem::Locked(a) => {
                let (ptr, len) = a.as_ptr_range();
                // SAFETY: as `bytes`; `&mut self` makes this the only view.
                unsafe { std::slice::from_raw_parts_mut(ptr, len) }
            }
            #[cfg(any(test, feature = "test-support"))]
            PinnedMem::Heap(v) => v,
        }
    }
}

/// Loud terminal state for FFI-adjacent contract violations (corrupt-weight
/// decode is worse than a dead process — documented, deliberate). Storage
/// loss is not one: it fails the compute through the ABI v13 ensure code.
pub(crate) fn p3_fatal(context: String) -> ! {
    // stderr is the one channel always available in an abort path (tracing
    // may have no subscriber installed).
    eprintln!("golem p3 fatal: {context}");
    tracing::error!("{context}");
    std::process::abort()
}

/// Runs an FFI callback body, turning a panic (poisoned lock, failed assert)
/// into a `p3_fatal` with context: unwinding out of `extern "C"` would abort
/// without saying why.
fn guarded<R>(what: &str, body: impl FnOnce() -> R) -> R {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
        Ok(r) => r,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("non-string panic payload");
            p3_fatal(format!("{what} panicked: {msg}"))
        }
    }
}

/// Recovers the payload reference from the C `void *`.
///
/// # Safety
/// `payload` was installed by [`enable`] as `Arc::into_raw(HookState)` and
/// the callback runs strictly inside the registration's lifetime (the C
/// unregister-join precedes the recovery).
unsafe fn state_of<'a>(payload: *mut c_void) -> &'a HookState {
    unsafe { &*(payload as *const HookState) }
}

/// The P1 `ensure_host_resident` trampoline, ABI v5 batched form: ONE call
/// per (layer, step, split-segment) with the whole miss list across ALL
/// planes; the pool holds a lease on each returned slot until `copy_done`.
/// Runs on the substrate decode thread; touches only the pool's own
/// synchronization — no substrate state, no locks ordered with C code.
///
/// # Contract
/// A null tensor, a negative `n_miss`, or an out-of-range
/// `(kind, id)` is a contract violation → logged + `abort()`. A source read
/// failure (every replica unreadable, §5.2) returns 1 (ABI v13): C fails the
/// compute (`llama_decode` −3) and the store health reads `FatalIo`.
// The C ABI fixes the signature; the pointer contract is documented above
// and enforced by the guarded body.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn ensure_host_resident(
    payload: *mut c_void,
    tensor: *const c_void,
    miss_kinds: *const i32,
    miss_ids: *const i32,
    n_miss: i32,
    out_ptrs: *mut *const c_void,
) -> i32 {
    guarded("ensure_host_resident", || {
        // SAFETY: see `state_of`.
        let h = unsafe { state_of(payload) };
        let Some(entry) = h.entry_for(tensor) else {
            return 0; // non-expert tensor: upstream behavior
        };
        if n_miss <= 0 {
            return 0; // nothing missed this segment
        }
        if miss_kinds.is_null() || miss_ids.is_null() {
            p3_fatal("P3: ensure_host_resident called with null miss list".into());
        }
        if out_ptrs.is_null() {
            p3_fatal("P3: ensure_host_resident called with null out_ptrs".into());
        }
        let n = n_miss as usize;
        // SAFETY: C allocates `n` entries in each array and passes them here
        // (ggml-backend.cpp golem miss hoisting pre-pass).
        let (kinds, ids) = unsafe {
            (
                std::slice::from_raw_parts(miss_kinds, n),
                std::slice::from_raw_parts(miss_ids, n),
            )
        };
        // SAFETY: C allocates `n` writable out-pointer slots (copy_experts_to
        // plan distribution); `*const c_void` and `*const u8` share layout.
        let out: &mut [*const u8] = unsafe { std::slice::from_raw_parts_mut(out_ptrs.cast(), n) };

        let mut scratch = h.miss_scratch.lock().unwrap_or_else(|e| e.into_inner());
        scratch.clear();
        for (&kind, &id) in kinds.iter().zip(ids) {
            let Some(kind) = kind_from_code(kind) else {
                p3_fatal(format!(
                    "P3 miss-list contract violation: kind {kind} not in 1..=3"
                ));
            };
            if id < 0 || id as u32 >= h.n_experts {
                p3_fatal(format!(
                    "P3 miss-list contract violation: id {id} vs n_experts {}",
                    h.n_experts
                ));
            }
            scratch.push((kind, id as u32));
        }

        // The pool IS the H2D source. Claim + page-in each unit once per
        // compute and hand the slot's plane pointer to C; the lease stays
        // held until `copy_done`, so eviction cannot recycle a slot mid-DMA.
        // §11: a few `Instant::now()` per layer is the sanctioned hot-path cost.
        // The scratch mutex stays held across the call: the decode thread is
        // its only contender (stated invariant), so this serializes nothing.
        #[cfg(any(test, feature = "test-support"))]
        if h.test_fail_storage_in_read.swap(false, Ordering::Relaxed) {
            h.fail_storage();
        }
        let t0 = std::time::Instant::now();
        let paged = h.pool.ensure_batch(entry.layer, &scratch, out);
        drop(scratch);
        h.p3_wait_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        h.ensure_calls.fetch_add(1, Ordering::Relaxed);
        match paged {
            Ok(()) => {}
            // Storage loss is an operational state, not a contract breach:
            // fail the compute; the caller reads the store health.
            Err(e @ golem_placement::pool::PoolError::Source { .. }) => {
                tracing::error!(layer = entry.layer, n_miss = n, "P3 page-in failed: {e}");
                return 1;
            }
            Err(e) => p3_fatal(format!(
                "P3 page-in failed (layer {}, n_miss {n}): {e}; \
                 aborting rather than decoding on garbage",
                entry.layer
            )),
        }

        // Counters visibility: one info line per 1000 slow paths (real metrics
        // are P4's).
        if (h.slow_calls.fetch_add(1, Ordering::Relaxed) + 1) % 1000 == 0 {
            tracing::info!(
                calls = h.slow_calls.load(Ordering::Relaxed),
                demand_pageins = h.pool.stats.demand_slow_pageins.load(Ordering::Relaxed),
                fast_hits = h.pool.stats.fast_hits.load(Ordering::Relaxed),
                evictions = h.pool.stats.evictions.load(Ordering::Relaxed),
                "P3 slow-path progress"
            );
        }
        0
    })
}

/// Ends the previous compute's step: releases the batched leases and runs
/// the bounded non-blocking order flush. Runs at the START of a scheduler
/// compute, after C drained the backends, so the previous compute's DMAs are
/// complete. One call per compute — never per expert.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn copy_done(payload: *mut c_void) {
    guarded("copy_done", || {
        // SAFETY: see `state_of`.
        let h = unsafe { state_of(payload) };
        h.copy_done_calls.fetch_add(1, Ordering::Relaxed);
        if h.exclusive {
            // Exclusive residency (O10): the eviction D2Hs issued last
            // compute landed (C drained its backends above) — open their
            // fill gates; then release the pool slots of everything staged
            // H2D last compute and mark those units VRAM-present.
            h.pool.commit_d2h();
            h.pool.end_batch_step_exclusive();
            h.publish_scored_snapshot();
        } else {
            // The device consumed the step's pointers (C synchronized its
            // backends before this hook): the slots are safe to recycle.
            h.pool.end_batch_step();
        }
        // Prefill-stream backstop: release whatever the per-plane
        // stream_release calls left (the last ≤ 2 planes of the chunk) and
        // drop unconsumed warm planes (a chunk may end below the streaming
        // minimum). C drained its backends above, so H2D reads of staging
        // slots and pool leases are complete.
        if let Some(s) = &h.stream {
            s.release_all();
        }
        // Bounded flush-if-dirty: if the publisher has not caught up (it
        // polls at 250 µs), publish up to 8 dirty layers here. push never
        // blocks — a full triple-buffer set leaves the layer dirty for the
        // publisher's next pass.
        if h.policy.dirty_any.load(Ordering::Relaxed) {
            let mut scratch = h.flush_scratch.lock().unwrap_or_else(|e| e.into_inner());
            publish_dirty(h, 8, &mut scratch);
        }
    });
}

/// The prefill-stream byte-source hook (ABI v11, design §7.6): one call per
/// streamed plane from the copy loop; translates the raw blk into the
/// pool's dense-lead-adjusted layer and forwards to the stream state.
/// Returns `GOLEM_STREAM_SRC_CANCEL` when the active turn's cancel flag is
/// set, so a cancel lands within one plane of a streamed ubatch, and
/// `GOLEM_STREAM_SRC_STORAGE_LOST` when the store lost the plane's bytes.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn stream_source(
    payload: *mut c_void,
    blk: i32,
    kind: i32,
    seq: u64,
    ids: *const i32,
    n_ids: i32,
    out_ptrs: *mut *const c_void,
) -> i32 {
    guarded("stream_source", || {
        // SAFETY: see `state_of`.
        let h = unsafe { state_of(payload) };
        let flag = h.abort.load(Ordering::Acquire);
        // SAFETY: non-null only while a `StreamAbortGuard` holds the
        // `Arc<AtomicBool>` it points into (nulled in its `Drop`).
        let cancel = (!flag.is_null()).then(|| unsafe { &*flag });
        #[cfg(any(test, feature = "test-support"))]
        if let (Some(c), Some(s)) = (cancel, &h.stream) {
            let at = h.test_cancel_plane.load(Ordering::Relaxed);
            if at != 0 && s.stats().planes + 1 >= at {
                c.store(true, Ordering::Relaxed);
            }
        }
        #[cfg(any(test, feature = "test-support"))]
        if h.test_spurious_cancel.swap(false, Ordering::Relaxed) {
            return raw::GOLEM_STREAM_SRC_CANCEL as i32;
        }
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            return raw::GOLEM_STREAM_SRC_CANCEL as i32;
        }
        let Some(s) = &h.stream else {
            p3_fatal("stream_source without stream state (streaming not enabled)".into());
        };
        if blk < h.dense_lead as i32 || blk >= (h.dense_lead + h.n_layers) as i32 {
            p3_fatal(format!(
                "stream_source: blk {blk} outside the expert blk range [{}, {})",
                h.dense_lead,
                h.dense_lead + h.n_layers
            ));
        }
        // SAFETY: C passes `n_ids` ids and preallocates the matching out array.
        let (ids, out) = unsafe {
            (
                std::slice::from_raw_parts(ids, n_ids.max(0) as usize),
                std::slice::from_raw_parts_mut(out_ptrs, n_ids.max(0) as usize),
            )
        };
        if s.source(blk - h.dense_lead as i32, kind, seq, ids, out)
            .is_err()
        {
            return raw::GOLEM_STREAM_SRC_STORAGE_LOST as i32;
        }
        // Re-checked after the (possibly blocking) fill: C's abort path
        // releases this seq too, so the plane's leases never leak.
        if cancel.is_some_and(|c| c.load(Ordering::Relaxed)) {
            raw::GOLEM_STREAM_SRC_CANCEL as i32
        } else {
            raw::GOLEM_STREAM_SRC_OK as i32
        }
    })
}

/// The prefill-stream release hook (ABI v10): C drops the seq−2 plane's
/// leases and staging slot after the slab's event sync; the copy_done
/// backstop covers the tail.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn stream_release(payload: *mut c_void, seq: u64) {
    guarded("stream_release", || {
        // SAFETY: see `state_of`.
        let h = unsafe { state_of(payload) };
        if let Some(s) = &h.stream {
            s.release(seq);
        }
    });
}

/// The exclusive-residency eviction hook (ABI v8, §6.3 O10): the substrate
/// evicted fully-valid slab slots of one layer; decide per expert whether its
/// bytes move into the pool (D2H on the split backend's stream — the SAME
/// stream the H2D refills use, issued first so stream order lands the D2H
/// before the refill; ≈ 15.2 MB / 25 GB/s ≈ 0.6 ms per evicted expert,
/// measured decode impact pending A/B) or drop. Runs on the decode thread,
/// ONCE per layer per compute with the whole eviction list — one entry
/// lookup, one snapshot source and one pending-D2H lock per batch, never per
/// expert — BEFORE that compute's `ensure`: so a unit evicted and copied
/// back is already pool-resident when its next slab miss is served (it is
/// never re-read from disk while a VRAM copy could exist).
///
/// # Contract
/// A negative `n_evicts`, or an out-of-range expert id, is a contract
/// violation → logged + `abort()`. `out_ptrs` holds 3 × `n_evicts` slots
/// (kind-1 order per expert); `keep[i]` is written per expert (0 = drop).
// The C ABI fixes the signature; the pointer contract is enforced by the
// guarded body.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn evict_dest_batch(
    payload: *mut c_void,
    tensor: *const c_void,
    experts: *const i32,
    n_evicts: i32,
    out_ptrs: *const *mut c_void,
    keep: *mut i32,
) {
    guarded("evict_dest_batch", || {
        // SAFETY: see `state_of`.
        let h = unsafe { state_of(payload) };
        let Some(entry) = h.entry_for(tensor) else {
            // Non-expert tensor: nothing to place — drop every eviction.
            if !keep.is_null() && n_evicts > 0 {
                // SAFETY: C allocates `n_evicts` writable keep flags.
                let keep = unsafe { std::slice::from_raw_parts_mut(keep, n_evicts as usize) };
                keep.fill(0);
            }
            return;
        };
        if n_evicts <= 0 {
            return; // nothing evicted this layer
        }
        if experts.is_null() || out_ptrs.is_null() || keep.is_null() {
            p3_fatal("P3: evict_dest_batch called with null argument".into());
        }
        let n = n_evicts as usize;
        // SAFETY: C allocates `n` entries in each array and `3n` writable
        // out slots (ggml-backend.cpp golem evict batch).
        let (experts, keep) = unsafe {
            (
                std::slice::from_raw_parts(experts, n),
                std::slice::from_raw_parts_mut(keep, n),
            )
        };
        // SAFETY: C allocates `3n` writable out-pointer slots; `*const
        // c_void` and `*const u8` share layout.
        // SAFETY (cont.): the slots are writable despite the const element
        // (C writes through `void *const *` the same way).
        let out: &mut [*mut c_void] =
            unsafe { std::slice::from_raw_parts_mut(out_ptrs.cast_mut().cast(), 3 * n) };
        let step = h.policy.step.load(Ordering::Relaxed);
        let mut scratch = h.evict_scratch.lock().unwrap_or_else(|e| e.into_inner());
        scratch.clear();
        for &id in experts.iter() {
            if id < 0 || id as u32 >= h.n_experts {
                p3_fatal(format!(
                    "P3 evict_dest_batch contract violation: id {id} vs n_experts {}",
                    h.n_experts
                ));
            }
            let expert = id as u32;
            // The VRAM copy is gone the moment the substrate reports the
            // slot evicted — clear the presence bit before any later ensure.
            h.pool.clear_vram(entry.layer, expert);
            // The incoming copy's score floor: displace only strictly colder
            // RAM keys (the exclusive counterpart of the swap margin, O10).
            let floor = h.policy.decayed(
                entry.layer as usize * h.n_experts as usize + expert as usize,
                step,
            ) as f32;
            scratch.push((expert, floor));
        }
        // The batch itself: one policy snapshot + one pending-D2H lock per
        // layer (inside the pool), never per expert.
        let mut dst: Vec<D2hIncoming> = (0..n).map(|_| D2hIncoming::Dropped).collect();
        let (ids, floors): (Vec<u32>, Vec<f32>) = scratch.iter().copied().unzip();
        drop(scratch);
        h.pool.claim_d2h_batch(entry.layer, &ids, &floors, &mut dst);
        for (i, d) in dst.iter().enumerate() {
            match d {
                D2hIncoming::Dest(planes) => {
                    keep[i] = 1;
                    for (k, src) in planes.iter().enumerate() {
                        out[3 * i + k] = *src as *const c_void as *mut c_void;
                    }
                }
                D2hIncoming::Resident | D2hIncoming::Dropped => keep[i] = 0,
            }
        }
    })
}

/// The abandoned-D2H hook (ABI v9, §6.3 O10): the substrate reports parked
/// eviction D2Hs whose plane copies were not all issued by compute end. The
/// claims must be cancelled — the slot returns to the free list, the fill
/// gate reopens, and the key stays non-resident (its VRAM bit was cleared at
/// eviction time, so the unit is re-read from disk later) — never committed
/// with unwritten plane bytes. Runs on the decode thread after the substrate
/// synchronized its backends, so any issued copy has landed before the slot
/// is recycled.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn evict_abandon(payload: *mut c_void, blk: i32, experts: *const i32, n: i32) {
    guarded("evict_abandon", || {
        // SAFETY: see `state_of`.
        let h = unsafe { state_of(payload) };
        if experts.is_null() || n < 0 {
            p3_fatal("P3: evict_abandon called with null/negative argument".into());
        }
        if blk < h.dense_lead as i32 || blk >= (h.dense_lead + h.n_layers) as i32 {
            p3_fatal(format!(
                "P3: evict_abandon blk {blk} outside the expert blk range [{}, {})",
                h.dense_lead,
                h.dense_lead + h.n_layers
            ));
        }
        let layer = (blk - h.dense_lead as i32) as u32;
        // SAFETY: C allocates `n` expert ids it parked this compute.
        let ids = unsafe { std::slice::from_raw_parts(experts, n as usize) };
        for &expert in ids {
            h.pool.abandon_d2h(layer, expert as u32);
        }
    })
}

/// The P5 next-layer prediction trampoline (ABI v6, §6.5): fired after
/// `src_blk`'s routing readback with the top (k + margin) experts predicted
/// for `target_blk`, decode ubatches only. Adds them to the target layer's
/// pending prediction and acts per mode: `protect` shields the union from
/// RAM eviction, `read` queues it for the prefetch thread. One shard lock or
/// one queue lock per call — never per expert — plus the trace lock when
/// `--routing-trace` is on.
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn predict(
    payload: *mut c_void,
    _src_blk: i32,
    target_blk: i32,
    ids: *const i32,
    n_ids: i32,
) {
    guarded("predict", || {
        // SAFETY: see `state_of`.
        let h = unsafe { state_of(payload) };
        let lead = h.dense_lead as i32;
        if h.pred.mode == PrefetchMode::Off
            || !h.policy.scoring.load(Ordering::Relaxed)
            || ids.is_null()
            || n_ids <= 0
            || target_blk < lead
            || target_blk >= lead + h.n_layers as i32
        {
            return;
        }
        let layer = (target_blk - lead) as usize;
        // SAFETY: C passes `n_ids` ids it read this step.
        let ids = unsafe { std::slice::from_raw_parts(ids, n_ids as usize) };
        if let Some(trace) = &h.trace {
            trace.lock().unwrap_or_else(|e| e.into_inner()).record(
                h.policy.step.load(Ordering::Relaxed),
                TracePhase::Predict,
                target_blk,
                ids,
            );
        }
        let new = h.pred.add(layer, ids, h.n_experts);
        match (&h.pred.mode, &h.pred.queue) {
            (PrefetchMode::Protect, _) => {
                new.for_each(drop);
                h.pool.protect(layer as u32, h.pred.predicted(layer));
            }
            (PrefetchMode::Read, Some(q)) => q.push(layer as u32, new),
            _ => new.for_each(drop),
        }
    })
}

/// The v5 merged routing readback trampoline: fired ONCE per changed ids
/// tensor per step with the densely gathered ids. Replaces the per-tensor
/// `policy_query` + per-expert `notify_access` fan-out: score booking,
/// access atomics and dirty-order marking happen in this single lock-free
/// call. Keyed by `blk` (the layer number — fused gate_up layouts have no
/// separate gate tensor; gate/up/down share ids).
#[allow(clippy::not_unsafe_ptr_arg_deref)]
pub extern "C" fn routing(
    payload: *mut c_void,
    blk: i32,
    _tensor: *const c_void,
    ids_dense: *const i32,
    n_ids: i64,
    n_experts: i32,
) {
    guarded("routing", || {
        // SAFETY: see `state_of`.
        let h = unsafe { state_of(payload) };
        h.routing_calls.fetch_add(1, Ordering::Relaxed);
        if n_experts != h.n_experts as i32 {
            p3_fatal(format!(
                "P2: routing n_experts {n_experts} vs pool geometry {}",
                h.n_experts
            ));
        }
        if blk < h.dense_lead as i32 || blk >= (h.dense_lead + h.n_layers) as i32 {
            p3_fatal(format!(
                "P2: routing blk {blk} outside the expert blk range [{}, {})",
                h.dense_lead,
                h.dense_lead + h.n_layers
            ));
        }
        if ids_dense.is_null() || n_ids <= 0 {
            return; // nothing routed (defensive; C reads the ids first)
        }
        let layer = (blk - h.dense_lead as i32) as usize;
        // SAFETY: C passes `n_ids` gathered ids it read this step.
        let ids = unsafe { std::slice::from_raw_parts(ids_dense, n_ids as usize) };
        h.policy.book_routed(layer, ids);
        h.policy.mark_dirty(layer);
        // The layer ran: its prediction is spent, so `protect` releases
        // the shield (a stale one would pin slots through prefill).
        if h.pred.score(layer, ids) && h.pred.mode == PrefetchMode::Protect {
            h.pool.protect(layer as u32, std::iter::empty());
        }
        if let Some(trace) = &h.trace {
            let mut t = trace.lock().unwrap_or_else(|e| e.into_inner());
            let phase = if h.policy.scoring.load(Ordering::Relaxed) {
                TracePhase::Decode
            } else {
                TracePhase::Prefill
            };
            t.record(h.policy.step.load(Ordering::Relaxed), phase, blk, ids);
        }
        // Prefill-stream read-ahead (§7.6.3): this layer's ids are known —
        // warm its up/down planes and the next layer's whole gate plane.
        if h.stream_armed.load(Ordering::Relaxed)
            && let Some(s) = &h.stream
        {
            s.warm(layer as u32, ids);
        }
    })
}

impl HookState {
    /// Marks every replica of every store shard unreadable (fault seam).
    #[cfg(any(test, feature = "test-support"))]
    fn fail_storage(&self) {
        if let Some(s) = &self.store {
            for sh in s.registry().shards() {
                for r in &sh.replicas {
                    r.mark_degraded();
                }
            }
        }
    }

    /// Publishes the scored policy snapshot for the next compute's D2H floor
    /// decisions (exclusive residency, O10): every key's §6.2 decayed score
    /// at the current step, built into one fresh snapshot. Once per decode
    /// step (in `copy_done`), never per expert — the table-sized Vec is the
    /// only allocation, and the previous snapshot stays alive until the
    /// next step consumes it (an in-place scratch would need a third
    /// buffer; not worth it at one alloc per step).
    fn publish_scored_snapshot(&self) {
        let step = self.policy.step.load(Ordering::Relaxed);
        let n = self.n_experts as usize;
        let keys = self.n_layers as usize * n;
        let mut scores = Vec::with_capacity(keys);
        for i in 0..keys {
            scores.push(self.policy.decayed(i, step) as f32);
        }
        let mut snap = PolicySnapshot::empty();
        snap.replace_scores(scores);
        self.pool.publish_snapshot(Arc::new(snap));
    }

    /// Registered entry for `tensor`, registering it on first sight.
    ///
    /// First sight = the first ensure callback for this tensor, which runs
    /// while the loader-populated weights buffer is still pristine — exactly
    /// the window the load-time cross-checks need.
    fn entry_for(&self, tensor: *const c_void) -> Option<TensorEntry> {
        if tensor.is_null() {
            return None;
        }
        let key = tensor as usize;
        {
            let last = self.last_entry.lock().unwrap_or_else(|e| e.into_inner());
            // The generation tag rejects a recycled address left over from a
            // previous session's registration.
            if let Some((k, tag, entry)) = last.as_ref()
                && *k == key
                && *tag == self.generation
            {
                return Some(*entry);
            }
        }
        if let Some(&entry) = self
            .tensors
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&key)
        {
            *self.last_entry.lock().unwrap_or_else(|e| e.into_inner()) =
                Some((key, self.generation, entry));
            return Some(entry);
        }
        // First sight: identify, cross-check, register.
        // SAFETY: `tensor` is a live ggml_tensor owned by the substrate for
        // the callback's duration (ggml-backend.cpp passes the expert weights
        // tensor it is about to read); reading name/ne/nb/data is what the P1
        // metadata helpers below do through the same pointer.
        let t = unsafe { &*(tensor as *const raw::ggml_tensor) };
        // SAFETY: `t.name` is a NUL-terminated char array inside the live
        // tensor (ggml sets it at tensor creation).
        let name = unsafe { CStr::from_ptr(t.name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let Some(kind) = ExpertKind::from_tensor_name(&name) else {
            return None; // not an expert plane: no-op (upstream behavior)
        };
        let Some(blk) = blk_of(&name) else {
            p3_fatal(format!("P3: expert tensor without blk.N prefix: {name}"));
        };
        if blk < self.dense_lead {
            p3_fatal(format!(
                "P3: expert tensor {name} inside the dense lead — GGUF walk and \
                 substrate disagree"
            ));
        }
        let layer = blk - self.dense_lead;
        if layer >= self.n_layers {
            p3_fatal(format!(
                "P3: expert tensor {name} maps to layer {layer} ≥ {} expert layers",
                self.n_layers
            ));
        }

        // SAFETY: `tensor` is the live, non-null ggml tensor checked above;
        // both queries only read its name/ne/nb.
        let n_experts = unsafe { raw::golem_tensor_expert_count(tensor) };
        // SAFETY: as above.
        let stride = unsafe { raw::golem_tensor_expert_stride_bytes(tensor) };
        if n_experts <= 0 || stride <= 0 {
            p3_fatal(format!(
                "P3: expert metadata queries returned ({n_experts}, {stride}) for {name}"
            ));
        }
        let stride = stride as usize;

        // Cross-check 1: source plane geometry must agree per expert tensor.
        let Some((_, src_stride)) = self.source.plane(layer, kind, 0) else {
            p3_fatal(format!(
                "P3: substrate tensor {name} (layer {layer}, {kind:?}) has no GGUF plane"
            ));
        };
        if src_stride != stride as u64 {
            p3_fatal(format!(
                "P3: stride mismatch for {name}: substrate {stride} B vs GGUF {src_stride} B"
            ));
        }
        if n_experts != self.n_experts as i32 {
            p3_fatal(format!(
                "P3: expert count mismatch for {name}: substrate {n_experts} vs GGUF {}",
                self.n_experts
            ));
        }

        // Cross-check 2: one pread byte-compared against the loader-populated
        // host bytes (pristine at first sight — the pool has not written this
        // tensor yet). This is the load-time proof that plane addressing is
        // right; wrong it can never ship.
        let mut disk = vec![0u8; stride];
        if let Err(e) = self.source.read_expert(layer, kind, 0, &mut disk, 0) {
            p3_fatal(format!("P3: verification pread failed for {name}: {e}"));
        }
        if t.data.is_null() {
            p3_fatal(format!("P3: expert tensor {name} has no host data pointer"));
        }
        // SAFETY: the weights buffer holds n_experts·stride fully populated
        // bytes from `data` (loader-owned host weights); `t.data` is non-null
        // (checked above).
        let host = unsafe { std::slice::from_raw_parts(t.data as *const u8, stride) };
        if host != disk.as_slice() {
            p3_fatal(format!(
                "P3: host bytes diverge from GGUF plane bytes for {name} — \
                 refusing to pool this model"
            ));
        }

        let entry = TensorEntry { layer };
        self.tensors
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(key, entry);
        *self.last_entry.lock().unwrap_or_else(|e| e.into_inner()) =
            Some((key, self.generation, entry));
        tracing::debug!(tensor = %name, layer, ?kind, stride, "P3 tensor registered");
        Some(entry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn trace_caps_cover_every_layer_of_every_token() {
        // GLM: 1024-token ctx in 1-token chunks + 32 decode steps × 42 layers.
        let (recs, ids) = trace_caps(1024, 32, 42, 8, (0, 0));
        assert_eq!(recs, 1056 * 42);
        assert_eq!(ids, 1056 * 42 * 8);
        // Lookahead 1 at top 12 adds one 12-id record per decode step per layer.
        let (recs, ids) = trace_caps(1024, 32, 42, 8, (1, 12));
        assert_eq!(recs, (1056 + 32) * 42);
        assert_eq!(ids, (1056 * 8 + 32 * 12) * 42);
    }

    #[test]
    fn abi_mismatch_is_poolinit_with_both_versions() {
        let e = check_abi(1, 2).unwrap_err();
        let msg = match e {
            EngineError::PoolInit(m) => m,
            other => panic!("want PoolInit, got {other:?}"),
        };
        assert!(
            msg.contains("GOLEM_HOOKS_ABI_VERSION=1") && msg.contains("compiled substrate 2"),
            "must name both versions, got: {msg}"
        );
        assert!(check_abi(3, 3).is_ok());
    }

    #[test]
    fn abi_check_runs_and_passes_against_pinned_substrate() {
        check_abi(
            raw::GOLEM_HOOKS_ABI_VERSION as i32,
            golem_llama_sys::hooks_abi_version(),
        )
        .unwrap_or_else(|e| panic!("pinned substrate ABI mismatch: {e}"));
    }

    #[test]
    fn enable_path_runs_abi_check_before_open() {
        // Inject a header that cannot match the compiled substrate. If the
        // check_abi call in enable_inner were deleted (or moved after the
        // model open), this test would see the model-open Io failure instead
        // of the ABI mismatch — so the PoolInit/ABI assertion is the proof
        // that the check runs, and runs strictly before read_metadata.
        TEST_ABI_HEADER.store(raw::GOLEM_HOOKS_ABI_VERSION as i32 + 1, Ordering::Relaxed);
        let config = EngineConfig {
            model_path: PathBuf::from("/nonexistent/golem-abi-test.gguf"),
            ram_experts_gib: 16,
            ..EngineConfig::default()
        };
        let err = match enable(&config) {
            Err(e) => e,
            Ok(_) => panic!("enable must fail on an ABI mismatch"),
        };
        TEST_ABI_HEADER.store(0, Ordering::Relaxed);
        match &err {
            EngineError::PoolInit(m) => {
                assert!(
                    m.contains("ABI mismatch"),
                    "expected ABI mismatch, got: {m}"
                );
            }
            other => panic!("expected the ABI mismatch, got: {other:?}"),
        }
    }

    fn built(keys: &[u64]) -> Vec<i32> {
        let mut order = vec![0; keys.len()];
        build_order(keys, &mut order);
        order
    }

    #[test]
    fn device_cache_experts_explicit_wins_and_default_is_never_unbounded() {
        let c = EngineConfig {
            device_cache_experts: 9,
            ..EngineConfig::default()
        };
        assert_eq!(
            derive_device_cache_experts(&c, 42 * 5_777_000, 8).unwrap(),
            (9, false)
        );
        // Explicit K below the model's top-k must fail at enable, not abort
        // on the first decode (a routed set larger than K aborts).
        let c_small = EngineConfig {
            device_cache_experts: 7,
            ..EngineConfig::default()
        };
        let err = derive_device_cache_experts(&c_small, 42 * 5_777_000, 8).unwrap_err();
        assert!(err.to_string().contains("expert_used_count"), "{err}");
        // Default (0): derived, bounded, never 0. On the CPU backend there is
        // no VRAM query, so derivation is a loud error, not an unbounded cap.
        #[cfg(not(golem_backend_hip))]
        assert!(
            derive_device_cache_experts(&EngineConfig::default(), 42 * 5_777_000, 8).is_err(),
            "no VRAM query: derivation must fail loudly"
        );
        // An explicit K that would wrap the i32 ABI is refused, not passed
        // through as "full slab".
        let c_huge = EngineConfig {
            device_cache_experts: u32::MAX,
            ..EngineConfig::default()
        };
        let err = derive_device_cache_experts(&c_huge, 42 * 5_777_000, 8).unwrap_err();
        assert!(err.to_string().contains("i32::MAX"), "{err}");
    }

    const Q4: u64 = 4_718_592;
    const Q5: u64 = 5_767_168;
    const Q6: u64 = 6_881_280;

    fn glm_classes() -> Vec<(u64, Vec<u32>)> {
        let std: Vec<u32> = (0..42).filter(|l| ![8, 9, 41].contains(l)).collect(); // blk 11/12/44 − 3
        vec![
            (Q4 + Q4 + Q5, std),
            (Q4 + Q4 + Q6, vec![9, 41]),
            (Q5 + Q5 + Q6, vec![8]),
        ]
    }

    #[test]
    fn class_layout_splits_the_pool_by_byte_share() {
        let l = class_layout(&glm_classes(), 65 << 30, 9).unwrap();
        assert_eq!(
            l.iter().map(|c| c.slots).collect::<Vec<_>>(),
            [4_226, 216, 108]
        );
        // Floor: 1 GiB cannot hold 39 × 9 std units in flight.
        let err = class_layout(&glm_classes(), 1 << 30, 9).unwrap_err();
        assert!(err.to_string().contains("device_cache_experts 9"), "{err}");
        // The floor is priced by K, not top-k, and keeps one slot of prefetch
        // headroom per class (layers × K + 1): 9 × Σ unit gives slots
        // [351, 18, 9] = [39, 2, 1] × 9 — fits K = 8 (floor [313, 17, 9]),
        // refused at K = 9 (floor [352, 19, 10]).
        let pool = 9u64 * 644_022_272 + (1 << 20);
        assert!(class_layout(&glm_classes(), pool, 8).is_ok());
        assert!(class_layout(&glm_classes(), pool, 9).is_err());
        // Exactly layers × K slots ([312, 16, 8] at 8 × Σ unit) leaves no
        // prefetch headroom: refused.
        let tight = 8u64 * 644_022_272 + (1 << 20);
        let err = class_layout(&glm_classes(), tight, 8).unwrap_err();
        assert!(err.to_string().contains("prefetch headroom"), "{err}");
    }

    #[test]
    fn derived_k_prices_a_slab_slot_by_per_layer_unit_bytes() {
        let geo_bytes: u64 = glm_classes()
            .iter()
            .map(|(u, ls)| u * ls.len() as u64)
            .sum();
        assert_eq!(geo_bytes, 644_022_272);
        // Ruling R3 (exact budget): K = ⌊(free − 1 GiB headroom) / 644 MB⌋ − 1.
        // (2 × 6.657 GB − 1 GiB) / 644 MB = 19.0 → K = 18.
        assert_eq!(
            k_from_free_vram(2 * 6_657_000_000, u64::MAX, geo_bytes, 8).unwrap(),
            18
        );
        assert!(k_from_free_vram(1 << 30, u64::MAX, geo_bytes, 8).is_err());
        // Huge free VRAM stays within the i32 ABI (≤ 0 reads as "full slab" in C).
        assert_eq!(
            k_from_free_vram(u64::MAX, u64::MAX, geo_bytes, 8).unwrap(),
            i32::MAX as u32
        );
        // The §10 plan caps K below the free VRAM: a 6 GiB device_expert_cache
        // term is ⌊6 GiB / 644 MB⌋ − 1 = 9, though 13.3 GB free allows 18.
        assert_eq!(
            k_from_free_vram(2 * 6_657_000_000, 6 << 30, geo_bytes, 8).unwrap(),
            9
        );
    }

    #[test]
    fn decode_k_reclaims_the_stream_slabs_under_a_binding_plan_cap() {
        // IQ3_XXS: slabs 2,566,914,048 B, slot ≈ 381 MB; the §10 cap already
        // netted the slabs out and binds before free VRAM.
        let (slab, stream) = (381_000_000u64, 2_566_914_048u64);
        let (free, cap) = (12_000_000_000u64, 4_000_000_000u64);
        let k_stream = k_from_free_vram(free - stream, cap, slab, 8).unwrap();
        let k_decode = k_from_free_vram(free, cap + stream, slab, 8).unwrap();
        // ⌊4.0 GB / 381 MB⌋ − 1 = 9; ⌊6.567 GB / 381 MB⌋ − 1 = 16.
        assert_eq!((k_stream, k_decode), (9, 16));
        assert_eq!(vram_mode_k(true, k_stream, k_decode), 9);
        assert_eq!(vram_mode_k(false, k_stream, k_decode), 16);
        // UD-Q4_K_XL target: slabs 3,963,617,280 B / 644,022,272 B per slot
        // → 6 more decode slots per layer.
        let k = |free, cap| k_from_free_vram(free, cap, 644_022_272, 8).unwrap();
        assert_eq!(
            k(20 << 30, 8 << 30) + 6,
            k(20 << 30, (8 << 30) + 3_963_617_280)
        );
    }

    #[test]
    fn routing_books_recency_for_whole_layer_and_marks_dirty() {
        let p = PolicyState::scored(2, 4, VramOrder::Recency);
        p.step.store(5, Ordering::Relaxed);
        p.book_routed(1, &[2, 2, 7, -1]); // duplicates idempotent; out-of-range rejected
        p.mark_dirty(1); // the trampoline does this right after booking
        let booked: Vec<usize> = (0..8).filter(|&i| p.access_step(i).is_some()).collect();
        assert_eq!(booked, [6], "expert 2 of layer 1 booked; 7 and −1 rejected");
        assert!(p.dirty[1].load(Ordering::Relaxed) && p.dirty_any.load(Ordering::Relaxed));
        assert!(!p.dirty[0].load(Ordering::Relaxed), "other layer untouched");
        // Scoring off (prefill, §6.2): books nothing (layer 0's row stays
        // untouched; layer 1's earlier stamp is from the scored leg above).
        p.scoring.store(false, Ordering::Relaxed);
        p.step.store(9, Ordering::Relaxed);
        p.book_routed(0, &[0, 1, 2, 3]);
        assert!(
            (0..4).all(|i| p.access_step(i).is_none()),
            "prefill readbacks must not stamp recency (§6.2)"
        );
    }

    #[test]
    fn out_of_range_id_never_spills_into_the_next_layer() {
        // id == n_experts would have passed a loose bounds guard and booked
        // layer 1's row 0.
        let p = PolicyState::scored(2, 4, VramOrder::Recency);
        p.step.store(3, Ordering::Relaxed);
        p.book_routed(0, &[4, -1, 100]);
        assert!(
            (0..8).all(|i| p.access_step(i).is_none()),
            "out-of-range ids are rejected, not wrapped into the next layer"
        );
        p.book_routed(0, &[3]);
        assert_eq!(p.access_step(3), Some(3), "max valid id still books");
    }

    #[test]
    fn order_built_once_per_call_and_content_is_correct() {
        let p = PolicyState::scored(2, 6, VramOrder::Recency);
        p.step.store(7, Ordering::Relaxed);
        p.book_routed(0, &[3]);
        p.step.store(9, Ordering::Relaxed);
        p.book_routed(1, &[5]);
        let mut s = OrderScratch::new(6);
        p.build_order(0, &mut s);
        assert_eq!(
            s.order,
            vec![0, 1, 2, 4, 5, 3],
            "expert 3 accessed → evicted last"
        );
        p.build_order(1, &mut s);
        assert_eq!(*s.order.last().unwrap(), 5);
        assert!(is_permutation(&s.order, 6));
    }

    #[test]
    fn score_order_evicts_the_once_used_before_the_frequent_older_expert() {
        // Expert 1 routed at steps 0..4, expert 2 once at step 5; 0 and 3
        // never. Recency evicts the older 1 first; the §6.2 decayed score
        // (≈ 4 vs ≈ 1) evicts 2 first. Never-accessed lead both orders.
        let book = |order| {
            let p = PolicyState::scored(1, 4, order);
            for s in 0..4 {
                p.step.store(s, Ordering::Relaxed);
                p.book_routed(0, &[1]);
            }
            p.step.store(5, Ordering::Relaxed);
            p.book_routed(0, &[2]);
            let mut s = OrderScratch::new(4);
            p.build_order(0, &mut s);
            s.order
        };
        assert_eq!(book(VramOrder::Recency), [0, 3, 1, 2]);
        assert_eq!(book(VramOrder::Score), [0, 3, 2, 1]);
    }

    #[test]
    fn order_builder_recency_sorts_never_accessed_first() {
        // Raw last_access words: 0 = never, else step + 1.
        let hist = [6, 0, 3, 0, 3];
        assert_eq!(
            built(&hist),
            vec![1, 3, 2, 4, 0],
            "never-accessed first (ascending index), then ascending step"
        );
    }

    #[test]
    fn published_order_is_identity_for_fresh_experts_and_a_permutation_otherwise() {
        let p = PolicyState::scored(1, 6, VramOrder::Recency);
        let mut s = OrderScratch::new(6);
        p.build_order(0, &mut s);
        assert_eq!(s.order, vec![0, 1, 2, 3, 4, 5], "no accesses → identity");
        p.step.store(4, Ordering::Relaxed);
        p.book_routed(0, &[4]);
        p.build_order(0, &mut s);
        assert!(is_permutation(&s.order, 6));
        assert_eq!(*s.order.last().unwrap(), 4);
    }

    #[test]
    fn permutation_checker_accepts_and_rejects() {
        assert!(is_permutation(&[1, 0, 2], 3));
        assert!(!is_permutation(&[1, 1, 2], 3), "duplicate");
        assert!(!is_permutation(&[0, 1], 3), "short");
        assert!(!is_permutation(&[0, 3, 2], 3), "out of range");
        assert!(!is_permutation(&[0, -1, 2], 3), "negative");
    }

    #[test]
    fn prediction_recall_counts_routed_ids_the_union_named_and_retires_it() {
        let p = PredState::new(PrefetchMode::Protect, 2, 70, 16);
        let new: Vec<_> = p.add(1, &[3, 65, 99, -1], 70).collect(); // 99, −1 out of range
        assert_eq!(new, [3, 65]);
        let new: Vec<_> = p.add(1, &[4, 3], 70).collect();
        assert_eq!(new, [4], "union: 3 already predicted");
        assert_eq!(p.predicted(1).collect::<Vec<_>>(), [3, 4, 65]);
        p.score(1, &[3, 65, 7, 8]);
        assert_eq!(
            (
                p.hits.load(Ordering::Relaxed),
                p.routed.load(Ordering::Relaxed)
            ),
            (2, 4)
        );
        p.score(1, &[3]); // retired: no prediction pending
        p.score(0, &[3]); // never predicted
        assert_eq!(p.routed.load(Ordering::Relaxed), 4);
        p.add(1, &[9], 70).for_each(drop); // a new step's prediction replaces the old union
        assert_eq!(p.predicted(1).collect::<Vec<_>>(), [9]);
    }
}
