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
use std::collections::HashMap;
use std::ffi::{CStr, c_void};
#[cfg(test)]
use std::sync::atomic::AtomicI32;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU64, AtomicUsize, Ordering};

use std::sync::{Arc, Mutex};

use golem_gguf::blk_of;
use golem_llama_sys::raw;
use golem_llama_sys::{CtxRaw, GolemCtx, HookFns, Hooked};
use golem_placement::pool::{
    AccessFeed, ExpertKind, ExpertPool, ExpertSource, GgufPreadSource, P3Stats, PREFETCH_K,
    PolicySnapshot, PrefetchHandle, SnapshotHandle, StoreExpertSource, spawn_prefetch,
};
use golem_store::ChunkQueue;

use crate::config::EngineConfig;
use crate::error::EngineError;

/// One traced routing readback: fixed fields plus a range into the ring's
/// preallocated flat id buffer (§11 pattern — no per-id heap write).
struct TraceRec {
    step: u64,
    prefill: bool,
    layer: i32,
    start: usize,
    len: usize,
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
    fn record(&mut self, step: u64, prefill: bool, layer: i32, ids: &[i32]) {
        if self.records.len() >= self.rec_cap || self.ids.len() + ids.len() > self.id_cap {
            self.dropped += 1;
            return;
        }
        let start = self.ids.len();
        self.ids.extend_from_slice(ids);
        self.records.push(TraceRec {
            step,
            prefill,
            layer,
            start,
            len: ids.len(),
        });
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
            write!(
                w,
                "{}\t{}\t{}\t",
                r.step,
                if r.prefill { "pf" } else { "dc" },
                r.layer
            )?;
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
    flush_scratch: Mutex<Vec<i32>>,
    /// §6.2 recency + §6.6 dirty tracking (shared with the publisher).
    policy: PolicyState,
    /// §11-style routing-trace ring when `config.routing_trace` is set; the
    /// routing readback appends (preallocated, no per-id heap), turn end
    /// flushes. `None` costs the trampoline one `Option` check.
    trace: Option<Mutex<TraceRing>>,
}

/// §6.2 policy state: per-expert last-access steps feeding the evict-first
/// order, the decode-step counter, and the per-layer dirty flags consumed by
/// the order publisher.
struct PolicyState {
    /// Flat `layer × n_experts` last-access step + 1; 0 = never accessed.
    /// Atomics: the routing readback books one routed id per element on the
    /// decode hot path, which takes no lock.
    last_access: Vec<AtomicU64>,
    /// Decode-step counter: one post-prefill trunk `llama_decode` = one step
    /// (§6.2: prefill chunks are excluded from scoring), bumped by
    /// [`advance_decode_step`] from `Session`'s generation decode.
    step: AtomicU64,
    /// Whether the routing readback books recency: false during prefill
    /// (§6.2: a prefill ubatch routes nearly every expert and would flatten
    /// the scores), true during generation decodes. Driven by `Session`
    /// through [`set_scoring`].
    scoring: AtomicBool,
    /// Per-layer dirty flags: the routing readback marks; the publisher and
    /// the `copy_done` flush swap-to-claim and publish.
    dirty: Vec<AtomicBool>,
    dirty_any: AtomicBool,
    n_layers: usize,
    n_experts: usize,
}

impl PolicyState {
    fn new(n_layers: u32, n_experts: u32) -> Self {
        Self {
            last_access: (0..n_layers as usize * n_experts as usize)
                .map(|_| AtomicU64::new(0))
                .collect(),
            step: AtomicU64::new(0),
            scoring: AtomicBool::new(true),
            dirty: (0..n_layers as usize)
                .map(|_| AtomicBool::new(false))
                .collect(),
            dirty_any: AtomicBool::new(false),
            n_layers: n_layers as usize,
            n_experts: n_experts as usize,
        }
    }

    /// Books one step's routed ids of `layer` (the dense ids readback):
    /// every routed expert's recency moves to `step + 1`. Out-of-range and
    /// negative ids are rejected (never wrapped into the next layer's row).
    /// Lock-free: one relaxed store per routed id.
    fn book_routed(&self, layer: usize, ids: &[i32]) {
        if !self.scoring.load(Ordering::Relaxed) {
            return;
        }
        let step = self.step.load(Ordering::Relaxed);
        let n = self.n_experts;
        let row = &self.last_access[layer * n..][..n];
        for &id in ids {
            if id >= 0 && (id as usize) < n {
                row[id as usize].store(step + 1, Ordering::Relaxed);
            }
        }
    }

    /// Marks `layer`'s order dirty for the publisher.
    fn mark_dirty(&self, layer: usize) {
        self.dirty[layer].store(true, Ordering::Relaxed);
        self.dirty_any.store(true, Ordering::Relaxed);
    }

    /// Last-access step of flat index `idx`; `None` = never accessed.
    fn access_step(&self, idx: usize) -> Option<u64> {
        self.last_access[idx].load(Ordering::Relaxed).checked_sub(1)
    }

    /// §6.5 prefetch feed: union of accessed `(layer, expert)` units from
    /// the last `k` steps. Layer-major, expert-ascending order.
    fn recent_union(&self, k: usize) -> Vec<(u32, u32)> {
        let cutoff = self.step.load(Ordering::Relaxed).saturating_sub(k as u64);
        let n = self.n_experts;
        let mut out = Vec::new();
        for layer in 0..self.n_layers {
            for e in 0..n {
                if self
                    .access_step(layer * n + e)
                    .is_some_and(|step| step > cutoff)
                {
                    out.push((layer as u32, e as u32));
                }
            }
        }
        out
    }

    /// Builds the evict-first order for `layer` into `order` (recency sort:
    /// never-accessed first, then ascending last-access step; ties by
    /// ascending index). Preallocated `order`; no allocation.
    fn build_order(&self, layer: usize, order: &mut [i32]) {
        let n = self.n_experts;
        debug_assert_eq!(order.len(), n);
        order.iter_mut().zip(0..).for_each(|(o, i)| *o = i);
        let row = &self.last_access[layer * n..][..n];
        order.sort_unstable_by_key(|&i| (row[i as usize].load(Ordering::Relaxed), i));
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
    }
}

/// Per-step I/O timing snapshot (§11): P3 wait is always measured once the
/// hooks exist; disk time/bytes only when the store-backed source is live.
pub(crate) struct StepIoStats {
    pub p3_wait_ns: u64,
    pub disk_ns: Option<u64>,
    pub disk_bytes: Option<u64>,
}

fn current_states() -> Vec<Arc<HookState>> {
    CURRENT.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Takes (swap-to-zero) the timing accumulators at a decode step boundary.
/// No-op while hooks are unregistered: unmeasured fields come back `None`.
pub(crate) fn take_step_io_stats() -> StepIoStats {
    let states = current_states();
    if states.is_empty() {
        return StepIoStats {
            p3_wait_ns: 0,
            disk_ns: None,
            disk_bytes: None,
        };
    }
    let mut out = StepIoStats {
        p3_wait_ns: 0,
        disk_ns: None,
        disk_bytes: None,
    };
    for h in states {
        out.p3_wait_ns += h.p3_wait_ns.swap(0, Ordering::Relaxed);
        if let Some(s) = &h.store {
            let (ns, bytes) = s.take_disk_stats();
            out.disk_ns = Some(out.disk_ns.unwrap_or(0) + ns);
            out.disk_bytes = Some(out.disk_bytes.unwrap_or(0) + bytes);
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
    /// §6.5 prefetch thread. Stopped before the hooks are torn down so its
    /// pool feed cannot outlive the pool state.
    prefetch: PrefetchHandle,
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
    /// Device slab slots per layer, as priced at enable (§3.4).
    pub device_cache_experts: u32,
    /// Model routed top-k (`expert_used_count`).
    pub top_k: u32,
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

    /// Switches recency booking (§6.2) — harness seam mirroring the free
    /// [`set_scoring`] used by `Session`.
    pub fn set_scoring(&self, on: bool) {
        set_scoring(on);
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
        self.prefetch.stop();
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
                let mut order: Vec<i32> = vec![0; state.n_experts as usize];
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
fn publish_dirty(state: &HookState, limit: usize, order_scratch: &mut [i32]) -> usize {
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
        state.policy.build_order(layer, order_scratch);
        let blk = layer as i32 + state.dense_lead as i32;
        let mut all_pushed = true;
        for kind in 1..=3 {
            // SAFETY: `ctx` is live (see above); `order_scratch` holds n
            // initialized i32s.
            let ok = unsafe {
                raw::golem_policy_order_push(ctx, blk, kind, order_scratch.as_ptr(), n as i32)
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
                let queue = Arc::new(ChunkQueue::new(config.io_threads.unwrap_or(0)));
                let store = Arc::new(
                    StoreExpertSource::open(manifest_path, queue)
                        .map_err(|e| EngineError::PoolInit(e.to_string()))?,
                );
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
    // The device-cache budget is a per-layer expert count, never a
    // byte cap on whichever tensor registers first.
    // It is never "unlimited": device allocations are bounded by the VRAM
    // budget, never proportional to expert count (unbounded slabs are ≈184 GB
    // on GLM-5.3-Flash — 126 live tensors × 288 experts × ≈5 MB — against a
    // 24 GiB card). An explicit config value wins; otherwise K derives from
    // the free VRAM at enable time, with half held back as growth headroom
    // (K = 9 on the target). K is rejected below the model's top-k: a routed
    // set larger than K aborts mid-decode, so a smaller K must fail at enable.
    let k = derive_device_cache_experts(config, slab, n_expert_used)?;
    // Per-class RAM layout (§5.4): each unit-size class gets slots
    // proportional to its share of total unit bytes; remainder bytes are
    // trimmed by integer division.
    let classes = geo.classes();
    // C clamps the slab to n_experts, so that is the per-layer staging bound.
    let layout = class_layout(&classes, pool_bytes, k.min(geo.n_experts))?;
    let arena = golem_placement::pool::Arena::new(&layout)
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

    // Design §3.5: mlock alone leaves the H2D copy pageable —
    // driver bounce buffers, ~half of PCIe bandwidth, not truly async, and
    // PCIe is the §2.4 bottleneck. Register the arena with the HIP runtime;
    // a failed registration is a startup failure, never a silent downgrade.
    #[cfg(golem_backend_hip)]
    {
        let (ptr, len) = arena.as_ptr_range();
        // SAFETY: the arena is a live, page-aligned, solely-owned mapping of
        // exactly `len` bytes (mmap + mlock above); registration happens
        // before any slot is handed out and unregister rides the pool's drop
        // (the session teardown unregisters the hooks first, so no callback
        // can touch a deregistered arena).
        if let Err(r) = unsafe { golem_llama_sys::pinned::hip_host_register(ptr, len) } {
            return Err(EngineError::PoolInit(format!(
                "hipHostRegister failed on the {len}-byte expert pool (HIP result {r})"
            )));
        }
        tracing::info!(
            gib = len as f64 / (1 << 30) as f64,
            "expert pool GPU-pinned"
        );
    }

    // Empty policy snapshot (nothing pinned/protected) until a
    // placement-backed snapshot builder feeds the pool (§6.6).
    let snapshots = Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::empty())));
    let pool = ExpertPool::try_new(
        arena,
        source.clone(),
        snapshots,
        Arc::new(P3Stats::new()),
        geo.clone(),
        class_of_layer,
    )
    .map_err(|e| EngineError::PoolInit(e.to_string()))?;

    // Per-model hook context (ABI v5). n_layers covers the full blk range —
    // C indexes its order buffers by the raw blk number, and the dense lead
    // has no expert tensors but does not shift blk numbering.
    let ctx = GolemCtx::create((dense_lead + n_layers) as i32, geo.n_experts as i32)
        .ok_or_else(|| EngineError::PoolInit("golem_ctx_create failed".into()))?;
    // SAFETY: FFI setter on a live ctx; bounded per-layer slab (never
    // unlimited, never proportional to total experts — AGENTS.md).
    unsafe { raw::golem_set_device_cache_experts(ctx.as_ptr(), k as i32) };

    // P2 state: per-expert recency (never-accessed first, then ascending
    // last-access step) — pure recency order.
    let policy = PolicyState::new(n_layers, geo.n_experts);
    // Routing-trace ring (§11 pattern), preallocated at enable: decode steps
    // are bounded by n_predict, prefill ids by chunk_tokens × top_k —
    // overflow drops and counts instead of reallocating on the hot path.
    let trace = config.routing_trace.as_ref().map(|p| {
        let rec_cap = config.n_predict as usize + 256; // + prefill-chunk headroom
        let prefill_ids = (config.batch_tokens as usize).max(1) * n_expert_used as usize;
        Mutex::new(TraceRing::new(
            p.clone(),
            rec_cap,
            rec_cap.saturating_mul(prefill_ids),
        ))
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
        flush_scratch: Mutex::new(vec![0; geo.n_experts as usize]),
        policy,
        n_layers,
        dense_lead,
        n_experts: geo.n_experts,
        pool,
        source,
        store,
        trace,
    });

    let prefetch = spawn_prefetch(
        state.pool.clone(),
        Arc::new(PolicyFeed(state.clone())),
        PREFETCH_K,
    );
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
    let hooked = ctx.register(
        HookFns {
            ensure: Some(ensure_host_resident),
            routing: Some(routing),
            copy_done: Some(copy_done),
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
        prefetch,
        publisher,
        hooked: Some(hooked),
        stats: state.pool.stats.clone(),
        state,
        ctx_keep: hooked_ctx_keep,
        device_cache_experts: k,
        top_k: n_expert_used,
    })
}

/// Per-layer device-slab expert count for this pool. Explicit
/// `device_cache_experts > 0` wins; otherwise derived from free VRAM. Never
/// unlimited, never zero: a routed set larger than K aborts, so K = 0 could
/// not decode at all.
fn derive_device_cache_experts(
    #[cfg_attr(not(golem_backend_hip), allow(unused_variables))] config: &EngineConfig,
    #[cfg_attr(not(golem_backend_hip), allow(unused_variables))] slab_bytes_per_expert: u64,
    n_expert_used: u32,
) -> Result<u32, EngineError> {
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
        return Ok(k);
    }
    #[cfg(golem_backend_hip)]
    if let Some((free, _total)) = golem_llama_sys::pinned::free_device_mem() {
        return k_from_free_vram(free as u64, slab_bytes_per_expert, n_expert_used);
    }
    Err(EngineError::PoolInit(
        "device_cache_experts unset and no VRAM query available to derive a bounded \
         slab budget — set device_cache_experts explicitly"
            .into(),
    ))
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
    let total_unit_bytes: u64 = classes
        .iter()
        .map(|&(unit, ref ls)| unit.saturating_mul(ls.len() as u64))
        .sum();
    let mut out = Vec::with_capacity(classes.len());
    for &(unit, ref layers) in classes {
        // slots = ⌊pool_bytes × share / unit⌋ = ⌊pool_bytes × layers / Σ unit⌋
        // (the unit cancels); u128 because pool_bytes × layers can exceed u64.
        let slots =
            u128::from(pool_bytes) * u128::from(layers.len() as u64) / u128::from(total_unit_bytes);
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

/// Derived K from free VRAM: half is slab budget, the rest KV + compute-buffer
/// growth headroom; `− 1` is per-slab headroom. Rejected below top-k — a
/// routed set larger than K aborts mid-decode, it is never floored.
#[cfg_attr(not(golem_backend_hip), allow(dead_code))]
fn k_from_free_vram(
    free: u64,
    slab_bytes_per_expert: u64,
    n_expert_used: u32,
) -> Result<u32, EngineError> {
    let k = (free / 2) / slab_bytes_per_expert.max(1);
    let k = k.saturating_sub(1).clamp(1, i32::MAX as u64);
    if k < u64::from(n_expert_used) {
        return Err(EngineError::PoolInit(format!(
            "free VRAM {free} B supports a slab of only {k} experts per layer at \
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

/// Loud terminal state for FFI-adjacent contract violations (the void ABI
/// leaves no non-silent failure path; corrupt-weight
/// decode is worse than a dead process — documented, deliberate).
fn p3_fatal(context: String) -> ! {
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
/// `(kind, id)` is a contract violation → logged + `abort()`.
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
) {
    guarded("ensure_host_resident", || {
        // SAFETY: see `state_of`.
        let h = unsafe { state_of(payload) };
        let Some(entry) = h.entry_for(tensor) else {
            return; // non-expert tensor: upstream behavior
        };
        if n_miss <= 0 {
            return; // nothing missed this segment
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
        let t0 = std::time::Instant::now();
        let paged = h.pool.ensure_batch(entry.layer, &scratch, out);
        drop(scratch);
        h.p3_wait_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        h.ensure_calls.fetch_add(1, Ordering::Relaxed);
        if let Err(e) = paged {
            p3_fatal(format!(
                "P3 page-in failed (layer {}, n_miss {n}): {e}; \
                 aborting rather than decoding on garbage",
                entry.layer
            ));
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
        // The device consumed the step's pointers (C synchronized its
        // backends before this hook): the slots are safe to recycle.
        h.pool.end_batch_step();
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

/// §6.5 prefetch feed backed by the P2 access map: the routing readback
/// books per-expert recency; the prefetcher unions the last `k` steps.
struct PolicyFeed(Arc<HookState>);

impl AccessFeed for PolicyFeed {
    fn take_union(&self, k: usize) -> Vec<(u32, u32)> {
        self.0.policy.recent_union(k)
    }
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
        if let Some(trace) = &h.trace {
            let mut t = trace.lock().unwrap_or_else(|e| e.into_inner());
            t.record(
                h.policy.step.load(Ordering::Relaxed),
                !h.policy.scoring.load(Ordering::Relaxed),
                blk,
                ids,
            );
        }
    })
}

impl HookState {
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
            9
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
        // 6.657 GB slab budget / 644 MB = 10.34 → K = 9. Pricing by 42 × 3 × max
        // stride (867 MB) gave 7.68 → K = 6 < top-k: enable refused the real model.
        assert_eq!(
            k_from_free_vram(2 * 6_657_000_000, geo_bytes, 8).unwrap(),
            9
        );
        assert!(k_from_free_vram(2 * 6_657_000_000, 42 * 3 * Q6, 8).is_err());
        // Huge free VRAM stays within the i32 ABI (≤ 0 reads as "full slab" in C).
        assert_eq!(
            k_from_free_vram(u64::MAX, geo_bytes, 8).unwrap(),
            i32::MAX as u32
        );
    }

    #[test]
    fn routing_books_recency_for_whole_layer_and_marks_dirty() {
        let p = PolicyState::new(2, 4);
        p.step.store(5, Ordering::Relaxed); // recent_union keeps steps > step − k
        p.book_routed(1, &[2, 2, 7, -1]); // duplicates idempotent; out-of-range rejected
        p.mark_dirty(1); // the trampoline does this right after booking
        assert_eq!(
            p.recent_union(3),
            vec![(1, 2)],
            "expert 2 of layer 1 booked; 7 and −1 rejected"
        );
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
        let p = PolicyState::new(2, 4);
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
        let p = PolicyState::new(2, 6);
        p.step.store(7, Ordering::Relaxed);
        p.book_routed(0, &[3]);
        p.step.store(9, Ordering::Relaxed);
        p.book_routed(1, &[5]);
        let mut order = vec![0i32; 6];
        p.build_order(0, &mut order);
        assert_eq!(
            order,
            vec![0, 1, 2, 4, 5, 3],
            "expert 3 accessed → evicted last"
        );
        p.build_order(1, &mut order);
        assert_eq!(*order.last().unwrap(), 5);
        assert!(is_permutation(&order, 6));
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
        let p = PolicyState::new(1, 6);
        let mut order = vec![0i32; 6];
        p.build_order(0, &mut order);
        assert_eq!(order, vec![0, 1, 2, 3, 4, 5], "no accesses → identity");
        p.step.store(4, Ordering::Relaxed);
        p.book_routed(0, &[4]);
        p.build_order(0, &mut order);
        assert!(is_permutation(&order, 6));
        assert_eq!(*order.last().unwrap(), 4);
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
    fn recent_union_bounds_and_cutoff() {
        let p = PolicyState::new(1, 4);
        p.step.store(9, Ordering::Relaxed);
        p.book_routed(0, &[2]);
        assert_eq!(p.recent_union(1), vec![(0, 2)]);
        assert!(
            p.recent_union(0).is_empty(),
            "k = 0 excludes everything (cutoff = current step)"
        );
    }
}
