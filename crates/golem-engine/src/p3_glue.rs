//! P3 FFI glue (§3.4): process-lifetime hook registration, the
//! `ensure_host_resident` trampoline, and the per-tensor registration table.
//!
//! Hook lifecycle: registered exactly ONCE per process
//! via [`HANDLE`]; the pool handle is `Box::leak`ed and hooks are never
//! unregistered — the C side stores the fn pointers in plain non-atomic
//! statics, so any teardown could race a mid-flight callback (UB). The
//! registration table fills lazily on the first callback per expert tensor
//! (the P1 ABI exposes expert metadata per tensor but no model-tensor
//! enumeration, so data pointers are only knowable when the substrate hands
//! the tensor to the callback). `enable` runs after `load()` returns, so the
//! first callbacks fire on the first decode — a GPU split registers every
//! expert tensor on that step, and any cross-check mismatch aborts loudly
//! there (the void FFI ABI has no error return).
//!
//! Registration shape:
//! - *Lazy registration:* a load-time tensor walk is not expressible
//!   through the P1 ABI (no enumeration API); the table is built at first
//!   callback per tensor instead. The load-time cross-checks (source plane
//!   stride == substrate stride per expert tensor; one pread byte-compared
//!   against the loader-populated host bytes) run at that registration, where
//!   the host buffer is guaranteed untouched by the pool — mismatch is a loud
//!   abort (the void FFI ABI has no error return).
//! - *Pool pointers, not host bytes:* `ensure_host_resident` claims each
//!   `(layer, expert)` unit of `first..=last` once per compute through the
//!   thread-local `StepClaims` (the unit's other planes reuse that lease),
//!   writes the slot's plane pointer into `out_ptrs`, and C copies H2D from
//!   there. The leases are released by the next compute's `copy_done`.
#![allow(unsafe_code)]

use golem_placement::pool::SlotClass;
use std::collections::HashMap;
use std::ffi::{CStr, c_void};
#[cfg(test)]
use std::sync::atomic::AtomicI32;
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use golem_gguf::blk_of;
use golem_llama_sys::raw;
use golem_placement::pool::{
    AccessFeed, ExpertKind, ExpertPool, ExpertSource, GgufPreadSource, P3Stats, PREFETCH_K,
    PolicySnapshot, PrefetchHandle, SnapshotHandle, StepClaims, StoreExpertSource, spawn_prefetch,
};
use golem_store::ChunkQueue;

use crate::config::EngineConfig;
use crate::error::EngineError;

/// One registered expert tensor: everything the trampoline needs to translate
/// a `(tensor, first_id, last_id)` callback into pool work.
#[derive(Clone, Copy)]
struct TensorEntry {
    /// Dense-lead-adjusted expert layer index (pool key space).
    layer: u32,
    /// Which of the layer's three planes.
    kind: ExpertKind,
    /// Expert count (`ne[2]` via `golem_tensor_expert_count`).
    n_experts: u32,
}

thread_local! {
    /// Last `entry_for` hit on this thread. C stages a layer's experts back
    /// to back on one tensor, so this answers nearly every per-expert call
    /// without the table lock. Keyed by raw tensor address: it stays valid
    /// only while one model owns the process's hooks — `session::load`
    /// refuses a second model once the hooks are registered (a recycled
    /// address would hit a stale entry).
    static LAST_ENTRY: std::cell::Cell<Option<(usize, TensorEntry)>> =
        const { std::cell::Cell::new(None) };

    /// The decode thread's claim table: `ensure_host_resident` claims each
    /// unit once per compute here (other planes reuse the lease) and
    /// `copy_done` ends the step at the start of the NEXT compute on this
    /// same thread (both fire from `ggml_backend_sched_compute_splits`).
    /// Thread-local: no lock per routed expert beyond one shard lock per
    /// unit, none held across page-in I/O; allocated on first use, reused
    /// across steps. The engine synchronizes after every decode and
    /// `copy_done` drains the backends, so the prior step's DMAs are complete.
    static STEP_CLAIMS: std::cell::RefCell<Option<StepClaims>> =
        const { std::cell::RefCell::new(None) };
}

/// The leaked process-lifetime P3 state. See [`HANDLE`].
struct PoolHandle {
    /// Dense-lead-adjusted expert layer count (from the GGUF walk at `enable`).
    n_layers: u32,
    /// Dense lead: expert tensors exist only from blk.<dense_lead> on; pool
    /// layers are renumbered to exclude it.
    dense_lead: u32,
    /// §6.5 prefetch thread. Declaration order is the drop contract: the
    /// handle is stopped and joined BEFORE the pool `Arc` drops. (The engine leaks
    /// the whole handle — process lifetime, hooks never unregistered — so
    /// neither field actually drops; the order keeps the invariant structural.)
    // Never read directly: its whole job is Drop (stop + join).
    #[allow(dead_code)]
    prefetch: PrefetchHandle,
    /// Expert byte source: the GGUF pread path or the store-backed reader
    /// (`--manifest`); the [`ExpertSource`] seam is what the registration
    /// cross-checks and the pool page-in both drive.
    source: Arc<dyn ExpertSource>,
    /// Concrete store-backed source when `config.manifest_path` was set:
    /// its `Arc<ChunkQueue>` owns the shared queue's lifetime inside the
    /// leaked handle (never read directly).
    #[allow(dead_code)]
    store: Option<Arc<StoreExpertSource>>,
    /// The demand pager (arena + residency + stats).
    pool: Arc<ExpertPool>,
    /// Aggregate expert geometry from the GGUF walk.
    n_experts: u32,
    /// tensor pointer → entry; filled on first callback per tensor (guarded
    /// by its own mutex, never held across `ensure_range` — the residency
    /// shard lock stays leaf-most, §6.6).
    tensors: Mutex<HashMap<usize, TensorEntry>>,
    /// `ensure_host_resident` invocations, for the per-1000 slow-path line.
    slow_calls: AtomicUsize,
    /// P2/P4 policy state (order building + access bookkeeping), leaked
    /// alongside the handle and shared with the prefetch feed by reference.
    policy: &'static PolicyQueryState,
}

/// Process-lifetime registration: set once, read forever, never torn down.
static HANDLE: OnceLock<&'static PoolHandle> = OnceLock::new();

/// Build-in-progress reservation: set by the first `enable` BEFORE any
/// resource is allocated, so a concurrent second `enable` fails fast instead
/// of racing [`HANDLE`] and leaking a built arena + prefetch thread.
static ENABLING: AtomicBool = AtomicBool::new(false);

/// Process-wide count of live [`crate::session::Session`]s. Off-then-on
/// guard: once ANY session exists, [`enable`] refuses — a pool registered
/// under a live session would retroactively route that session's decodes
/// through the new hooks (the first session's next decode would page from a
/// pool built for a different model).
static LIVE_SESSIONS: AtomicUsize = AtomicUsize::new(0);

/// Err while any session is live: a pool enabled under a live session would
/// retroactively route that session's decodes through the new hooks. Checked
/// by `Session::new` BEFORE `load` (so a refusal costs no model load) and
/// again by [`enable`].
pub(crate) fn refuse_if_session_live() -> Result<(), EngineError> {
    if LIVE_SESSIONS.load(Ordering::SeqCst) > 0 {
        return Err(EngineError::PoolInit(
            "a session already exists in this process; enabling the expert pool \
             now would re-route its decodes through the new hooks — enable the \
             pool at construction of the FIRST session"
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

/// `last_queried` sentinel: no policy_query yet.
const NO_LAYER: u32 = u32::MAX;

/// P2 policy state (§3.4): per-expert last-access steps feeding the
/// evict-first order, plus the double-buffered order slices returned to C.
///
/// Attribution: `notify_access` carries only the expert id, so accesses are
/// booked under the layer of the most recent `policy_query`. Exact for the
/// current C call-site shape — C emits notify per routed expert inside the
/// same policy_query-guarded block, single decode thread; multiple contexts
/// would need tensor identity passed explicitly.
struct PolicyQueryState {
    /// Flat `layer × n_experts` last-access step + 1; 0 = never accessed.
    /// Atomics: `notify_access` fires per routed expert on the decode hot
    /// path, which takes no lock. One row per layer, shared by all three of
    /// its tensors — the same order is returned for gate/up/down, which the
    /// shared per-layer slot table requires (§3.4).
    last_access: Vec<AtomicU64>,
    /// Decode-step counter: one post-prefill trunk `llama_decode` = one step
    /// (§6.2: prefill chunks are excluded from scoring), bumped by
    /// [`advance_decode_step`] from `Session`'s generation decode. Rust drives
    /// decode, so Rust owns the step.
    step: AtomicU64,
    /// Per-layer order slots (index `n_layers` holds the identity order for
    /// non-expert tensors, in its own slot so it can never poison a layer
    /// entry). Pointer lifetime: a published pointer is stable until the
    /// SAME layer's order is rebuilt at the next step — interleaved
    /// publishes of other layers (or the identity) never clobber it. C
    /// consumes the order within one copy step: the callback calls
    /// policy_query and feeds the pointer to golem_slab_step before any
    /// further publish (ggml-backend.cpp compute_splits, single decode
    /// thread, n_copies == 1 enforced at callback entry), so the next-step
    /// rebuild never races a live consumer.
    order_bufs: Vec<Mutex<OrderBuf>>,
    /// Pointer table into `order_bufs`, read without a lock on cache hits;
    /// written under the slot mutex at build.
    order_ptrs: Vec<AtomicPtr<i32>>,
    /// Per-layer decode step of the last build: a rebuild happens only when
    /// the step advanced since that layer's last publish. Plain atomics,
    /// single decode thread; per-layer so interleaved publishes of other
    /// layers within one step stay cache hits.
    cached_step: Vec<AtomicU64>,
    /// Rebuild counter (test observability: one build per (layer, step)).
    order_builds: AtomicUsize,
    /// Last layer passed to policy_query; notify attribution key.
    /// [`NO_LAYER`] until the first query.
    last_queried: AtomicU32,
    /// Whether `notify_access` books recency: false during prefill (§6.2: a
    /// prefill ubatch routes nearly every expert and would flatten the
    /// scores), true during generation decodes. Driven by `Session` through
    /// [`set_scoring`].
    scoring: AtomicBool,
    n_layers: u32,
    n_experts: u32,
}

impl PolicyQueryState {
    fn new(n_layers: u32, n_experts: u32) -> Self {
        Self {
            last_access: (0..n_layers as usize * n_experts as usize)
                .map(|_| AtomicU64::new(0))
                .collect(),
            step: AtomicU64::new(0),
            order_bufs: (0..=n_layers)
                .map(|_| Mutex::new(OrderBuf::new(n_experts)))
                .collect(),
            order_ptrs: (0..=n_layers)
                .map(|_| AtomicPtr::new(std::ptr::null_mut()))
                .collect(),
            cached_step: (0..=n_layers).map(|_| AtomicU64::new(u64::MAX)).collect(),
            order_builds: AtomicUsize::new(0),
            last_queried: AtomicU32::new(NO_LAYER),
            scoring: AtomicBool::new(true),
            n_layers,
            n_experts,
        }
    }

    /// Returns the evict-first order for `layer` (shared by all three of its
    /// tensors), rebuilt at most once per (layer, decode step) and cached in
    /// the layer's slot; `None` = identity order for non-expert tensors,
    /// built into its own slot and never cached under the layer key. Hot
    /// path: cache hits are two relaxed atomic loads + a pointer load — no
    /// lock, no allocation (slots sized at enable). Accepted staleness:
    /// accesses recorded by `note_access` between a step's plane queries are
    /// not reflected until the next step's rebuild (LRU heuristic quality,
    /// not correctness).
    fn publish_order(&self, layer: Option<u32>) -> *const i32 {
        let slot = match layer {
            Some(layer) => {
                debug_assert!(
                    layer < self.n_layers,
                    "order requested outside the registered geometry"
                );
                let step = self.step.load(Ordering::Relaxed);
                let last = self.cached_step[layer as usize].load(Ordering::Relaxed);
                if last == step {
                    return self.order_ptrs[layer as usize]
                        .load(Ordering::Relaxed)
                        .cast_const();
                }
                let mut buf = self.order_bufs[layer as usize].lock().unwrap();
                let OrderBuf { keys, order } = &mut *buf;
                let lo = layer as usize * self.n_experts as usize;
                for (k, a) in keys.iter_mut().zip(&self.last_access[lo..]) {
                    *k = a.load(Ordering::Relaxed);
                }
                build_order(keys, order);
                self.order_builds.fetch_add(1, Ordering::Relaxed);
                self.order_ptrs[layer as usize]
                    .store(order.as_ptr() as *mut i32, Ordering::Relaxed);
                self.cached_step[layer as usize].store(step, Ordering::Relaxed);
                return order.as_ptr();
            }
            None => self.n_layers as usize,
        };
        let mut buf = self.order_bufs[slot].lock().unwrap();
        let order = &mut buf.order;
        order.iter_mut().zip(0..).for_each(|(o, i)| *o = i);
        self.order_ptrs[slot].store(order.as_ptr() as *mut i32, Ordering::Relaxed);
        order.as_ptr()
    }

    /// Test observability: total order rebuilds so far.
    #[cfg(test)]
    fn order_builds(&self) -> usize {
        self.order_builds.load(Ordering::Relaxed)
    }

    /// Last-access step of flat index `idx`; `None` = never accessed.
    fn access_step(&self, idx: usize) -> Option<u64> {
        self.last_access[idx].load(Ordering::Relaxed).checked_sub(1)
    }

    /// Record one routed-expert access under the last queried (layer, kind)
    /// at the current decode step.
    fn note_access(&self, expert_id: i64) {
        if !self.scoring.load(Ordering::Relaxed) {
            return;
        }
        let step = self.step.load(Ordering::Relaxed);
        let layer = self.last_queried.load(Ordering::Relaxed);
        if layer == NO_LAYER {
            return; // defensive: notify before any query — nothing to attribute
        }
        // Out-of-range ids are never routed by C (compute_splits masks ids to
        // [0, n_expert) before notify_access); reject before the usize cast
        // would wrap negatives and before id == n_experts could spill into the
        // next layer's row.
        if expert_id < 0 || expert_id as u64 >= self.n_experts as u64 {
            return;
        }
        let e = expert_id as usize;
        let idx = layer as usize * self.n_experts as usize + e;
        self.last_access[idx].store(step + 1, Ordering::Relaxed);
    }

    /// §6.5 prefetch feed: union of accessed `(layer, expert)` units from
    /// the last `k` steps. Layer-major, expert-ascending order.
    fn recent_union(&self, k: usize) -> Vec<(u32, u32)> {
        let cutoff = self.step.load(Ordering::Relaxed).saturating_sub(k as u64);
        let n = self.n_experts as usize;
        let mut out = Vec::new();
        for layer in 0..self.n_layers as usize {
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
}

/// One order slot handed to C plus its sort-key scratch, both
/// `n_experts` long for the process lifetime.
struct OrderBuf {
    keys: Vec<u64>,
    order: Vec<i32>,
}

impl OrderBuf {
    fn new(n_experts: u32) -> Self {
        Self {
            keys: vec![0; n_experts as usize],
            order: vec![0; n_experts as usize],
        }
    }
}

/// Recency order builder (pure; unit-tested): `keys` are the raw
/// `last_access` words (0 = never, else step + 1), so ascending key puts
/// never-accessed experts first, then ascending last-access step; ties break
/// by ascending index. Writes a permutation of `0..keys.len()` into `order`
/// by construction; no allocation (unstable sort, unique keys).
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

/// Bumps the policy decode step: `Session` calls this after every successful
/// trunk decode (§6.2: prefill chunks are excluded from scoring — a prefill
/// ubatch routes nearly every expert and would flatten the scores). No-op
/// while hooks are unregistered.
pub(crate) fn advance_decode_step() {
    if let Some(h) = HANDLE.get() {
        h.policy.step.fetch_add(1, Ordering::Relaxed);
    }
}

/// Switches recency booking on for generation decodes and off for prefill
/// (§6.2). No-op while hooks are unregistered.
pub(crate) fn set_scoring(on: bool) {
    if let Some(h) = HANDLE.get() {
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
    HANDLE
        .get()
        .map_or(0, |h| h.policy.step.load(Ordering::Relaxed))
}

/// Whether the process-global hooks are registered (a second model in the
/// process is refused: the hooks are never unregistered).
pub(crate) fn hooks_registered() -> bool {
    HANDLE.get().is_some()
}

/// P3 pool facts the session needs past enable (§2.4): a prefill chunk's
/// routed set is at most `chunk_tokens × top_k` distinct experts per layer
/// and must fit the `device_cache_experts`-slot slab, so the chunk is capped
/// at `max(1, device_cache_experts / top_k)` tokens.
pub(crate) struct PoolEnable {
    /// Pool counters for [`crate::session::Session::p3_stats`].
    pub stats: Arc<P3Stats>,
    /// Device slab slots per layer, as priced at enable (§3.4).
    pub device_cache_experts: u32,
    /// Model routed top-k (`expert_used_count`).
    pub top_k: u32,
}

/// Builds the RAM expert pool for `config` (which must set
/// `ram_experts_gib > 0`) and registers the P3 hooks for the process
/// lifetime. Returns the pool's counters plus the slab/routing geometry the
/// session needs to bound a prefill chunk (§2.4).
///
/// # Errors
/// GGUF/geometry failures, arena mmap/mlock failures (no silent unlocked
/// fallback, §5.4), or a second pool-enabled session in one process (one
/// pool-enabled session per process: the hooks are process-global).
pub(crate) fn enable(config: &EngineConfig) -> Result<PoolEnable, EngineError> {
    debug_assert!(config.ram_experts_gib > 0, "enable() with pool disabled");
    // Refuse BEFORE allocating anything: a pool enabled while any session is
    // live would silently re-route that session's subsequent decodes through
    // the new hooks (process-global, never unregistered).
    refuse_if_session_live()?;
    if ENABLING
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err(EngineError::PoolInit(
            "concurrent pool-enabled session construction".into(),
        ));
    }
    let result = enable_inner(config);
    // Reservation released either way: on success HANDLE now refuses a
    // second enable with the accurate one-pool-per-process error; on failure
    // a later enable() may retry from scratch.
    ENABLING.store(false, Ordering::SeqCst);
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

fn enable_inner(config: &EngineConfig) -> Result<PoolEnable, EngineError> {
    // The compiled substrate's hook surface must match the header the bindings
    // were generated from — checked before any source open or allocation, so a
    // stale build fails fast instead of silently decoding with a drifted ABI.
    // Deliberately before the HANDLE short-circuit: a stale substrate reports
    // the ABI error even on a second enable.
    check_abi(abi_header_for_check(), golem_llama_sys::hooks_abi_version())?;
    if HANDLE.get().is_some() {
        return Err(EngineError::PoolInit(
            "a pool-enabled session already exists (one per process: hooks are process-global)"
                .into(),
        ));
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
    // queue lives in the leaked handle via the store source's Arc), else
    // the single-replica GGUF pread path (byte-identical to pre-manifest
    // behavior).
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
    // the free VRAM at enable time (the model's non-expert weights are
    // already resident, so what is free is what the slabs + KV + compute
    // buffers must share), with half held back as growth headroom:
    // K = free/2 / Σ_layers unit bytes − 1 = 9 on the target
    // (Σ unit bytes = 644,022,272 across 3 stride classes; ≈6.657 GB usable
    // / 644 MB ≈ 10.34, minus the − 1 headroom). Pricing by 42 × 3 × max
    // stride instead gave K = 6 < top-k, refusing the real model at enable.
    // K is rejected below the model's top-k (expert_used_count): a routed
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
        // before any slot is handed out and unregister rides the process
        // teardown (the pool is leaked by design, hooks never unregistered).
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

    // P2 state: per-expert recency (never-accessed first, then
    // ascending last-access step) — pure recency order.
    let policy: &'static PolicyQueryState =
        Box::leak(Box::new(PolicyQueryState::new(n_layers, geo.n_experts)));
    let prefetch = spawn_prefetch(pool.clone(), Arc::new(PolicyFeed(policy)), PREFETCH_K);

    let handle: &'static PoolHandle = Box::leak(Box::new(PoolHandle {
        n_layers: geo.n_layers(),
        dense_lead,
        prefetch,
        source,
        store,
        pool: pool.clone(),
        n_experts: geo.n_experts,
        tensors: Mutex::new(HashMap::new()),
        slow_calls: AtomicUsize::new(0),
        policy,
    }));
    // CAS reservation above ⇒ no concurrent setter and `get()` was None.
    assert!(
        HANDLE.set(handle).is_ok(),
        "HANDLE must be unset after winning the enable reservation"
    );

    // Fresh device-cache state before the hooks engage it (the map may hold
    // entries from an earlier in-process model — the invalidation seam).
    // SAFETY: FFI call, no invariants.
    unsafe { raw::golem_reset_device_cache() };
    // SAFETY: FFI setter, no invariants.
    unsafe {
        raw::golem_set_device_cache_experts(k as i32);
    }

    // SAFETY: `handle` is leaked ('static) and never mutated after this, so
    // the context pointer outlives every callback; all four are valid fn
    // pointers (P3 ensure_host_resident / P2 policy_query / P4 notify_access /
    // copy_done; the C side's n_copies == 1 abort is the defense-in-depth —
    // the engine config cannot express pipeline parallelism or multi-GPU to
    // pre-reject here).
    unsafe {
        raw::golem_register_hooks(
            Some(ensure_host_resident),
            Some(policy_query),
            Some(notify_access),
            Some(copy_done),
            handle as *const PoolHandle as *mut c_void,
        );
    }

    tracing::info!(
        layers = n_layers,
        experts = geo.n_experts,
        classes = ?layout,
        gib = config.ram_experts_gib,
        "P3 RAM expert pool registered (hooks set for process lifetime)"
    );
    Ok(PoolEnable {
        stats: pool.stats.clone(),
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

/// The P1 `ensure_host_resident` trampoline. Runs on the substrate decode
/// thread; touches only the pool's own synchronization — no substrate state,
/// no locks ordered with C code (the concurrency test model covers it).
///
/// # Contract
/// No input clamping: an out-of-range or inverted `[first_id, last_id]` is a
/// contract violation → logged + `abort()`. Non-expert
/// tensors return immediately (null-guard semantics upstream of the pool).
pub(crate) extern "C" fn ensure_host_resident(
    ctx: *mut c_void,
    tensor: *const c_void,
    first_id: i32,
    last_id: i32,
    out_ptrs: *mut *const c_void,
) {
    guarded("ensure_host_resident", || {
        ensure_host_resident_inner(ctx, tensor, first_id, last_id, out_ptrs)
    })
}

fn ensure_host_resident_inner(
    ctx: *mut c_void,
    tensor: *const c_void,
    first_id: i32,
    last_id: i32,
    out_ptrs: *mut *const c_void,
) {
    // SAFETY: ctx was installed by `enable` as a leaked `&'static PoolHandle`
    // and is never mutated afterwards; it outlives every callback.
    let h = unsafe { &*(ctx as *const PoolHandle) };
    let Some(entry) = h.entry_for(tensor) else {
        return; // non-expert tensor: upstream behavior
    };
    if first_id < 0 || first_id > last_id || last_id >= entry.n_experts as i32 {
        p3_fatal(format!(
            "P3 range contract violation: [{first_id}, {last_id}] vs n_experts {}",
            entry.n_experts
        ));
    }
    let first = first_id as u32;
    let last = last_id as u32;
    let n = (last - first + 1) as usize;
    if out_ptrs.is_null() {
        p3_fatal("P3: ensure_host_resident called with null out_ptrs".into());
    }
    // SAFETY: C allocates `n` entries and passes the array here
    // (copy_experts_to); `*const c_void` and `*const u8` share layout.
    let out: &mut [*const u8] = unsafe { std::slice::from_raw_parts_mut(out_ptrs.cast(), n) };

    // The pool IS the H2D source. Claim + page-in each unit once per compute
    // and hand the slot's plane pointer to C; the lease stays held until
    // `copy_done`, so eviction cannot recycle a slot mid-DMA.
    let paged = STEP_CLAIMS.with_borrow_mut(|step| {
        step.get_or_insert_with(|| StepClaims::new(h.pool.clone()))
            .ensure_range_leased(entry.layer, entry.kind, first, last, out)
    });
    if let Err(e) = paged {
        p3_fatal(format!(
            "P3 page-in failed (layer {}, {kind:?}, experts [{first}, {last}]): {e}; \
             aborting rather than decoding on garbage",
            entry.layer,
            kind = entry.kind
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
}

/// Ends the previous compute's step: releases its host-pool leases and
/// refreshes the policy snapshot. Runs at the START of a scheduler compute,
/// after C drained the backends, so the previous compute's DMAs are complete.
extern "C" fn copy_done(_ctx: *mut c_void) {
    guarded("copy_done", || {
        STEP_CLAIMS.with_borrow_mut(|step| {
            if let Some(step) = step {
                step.next_step();
            }
        });
    });
}

/// §6.5 prefetch feed backed by the P2 access map: `notify_access` (P4)
/// records per-expert recency; the prefetcher unions the last `k` steps.
struct PolicyFeed(&'static PolicyQueryState);

impl AccessFeed for PolicyFeed {
    fn take_union(&self, k: usize) -> Vec<(u32, u32)> {
        self.0.recent_union(k)
    }
}

/// The P2 `policy_query` trampoline: returns the evict-first expert order
/// for `tensor` as a permutation of `0..n_experts`, valid until the next
/// call (frozen-buffer contract). First expert-tensor sight flows through
/// the full `entry_for` validate-and-cross-check path — per copy step
/// (verified: `ggml_backend_sched_compute_splits`, slab branch) C calls
/// policy_query, then per-slot ensure_host_resident, then notify_access —
/// so the first expert tensor ever seen arrives HERE, already registered
/// before any ensure_host_resident cross-check can abort on first sight.
extern "C" fn policy_query(ctx: *mut c_void, tensor: *const c_void, n_experts: i32) -> *const i32 {
    guarded("policy_query", || {
        policy_query_inner(ctx, tensor, n_experts)
    })
}

fn policy_query_inner(ctx: *mut c_void, tensor: *const c_void, n_experts: i32) -> *const i32 {
    // SAFETY: ctx was installed by `enable` as a leaked `&'static PoolHandle`
    // and is never mutated afterwards; it outlives every callback.
    let h = unsafe { &*(ctx as *const PoolHandle) };
    let Some(entry) = h.entry_for(tensor) else {
        // Defensive: C queries only expert tensors; anything else gets a
        // legal identity permutation (invariant 4: any permutation is valid).
        return h.policy.publish_order(None);
    };
    if n_experts != entry.n_experts as i32 {
        p3_fatal(format!(
            "P2: policy_query n_experts {n_experts} vs registered {} for layer {}",
            entry.n_experts, entry.layer
        ));
    }
    h.policy.last_queried.store(entry.layer, Ordering::Relaxed);
    h.policy.publish_order(Some(entry.layer))
}

/// The P4 `notify_access` trampoline: one routed expert of the tensor most
/// recently passed to [`policy_query`] — call-site adjacency guarantees the
/// attribution for the single decode thread (see [`PolicyQueryState`]).
extern "C" fn notify_access(ctx: *mut c_void, expert_id: i64) {
    // SAFETY: see policy_query.
    let h = unsafe { &*(ctx as *const PoolHandle) };
    guarded("notify_access", || h.policy.note_access(expert_id));
}

impl PoolHandle {
    /// Registered entry for `tensor`, registering it on first sight.
    ///
    /// First sight = the first callback for this tensor: `policy_query`
    /// when P2 is active (it precedes `ensure_host_resident` in
    /// `ggml_backend_sched_compute_splits`), else `ensure_host_resident`.
    /// Either way the callback runs while the loader-populated weights
    /// buffer is still pristine — exactly the window the load-time
    /// cross-checks need.
    fn entry_for(&self, tensor: *const c_void) -> Option<TensorEntry> {
        if tensor.is_null() {
            return None;
        }
        let key = tensor as usize;
        if let Some((k, entry)) = LAST_ENTRY.get()
            && k == key
        {
            return Some(entry);
        }
        if let Some(&entry) = self.tensors.lock().unwrap().get(&key) {
            LAST_ENTRY.set(Some((key, entry)));
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

        let entry = TensorEntry {
            layer,
            kind,
            n_experts: n_experts as u32,
        };
        self.tensors.lock().unwrap().insert(key, entry);
        LAST_ENTRY.set(Some((key, entry)));
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

    impl PolicyQueryState {
        fn order_for_test(&self, layer: u32) -> Vec<i32> {
            let p = self.publish_order(Some(layer));
            // SAFETY: the published buffer holds n_experts entries until the next publish.
            unsafe { std::slice::from_raw_parts(p, self.n_experts as usize) }.to_vec()
        }
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
    fn policy_order_and_access_are_shared_by_the_layers_three_tensors() {
        let p = PolicyQueryState::new(2, 4);
        p.step.store(5, Ordering::Relaxed); // recent_union keeps steps > step − k
        p.last_queried.store(1, Ordering::Relaxed); // last query: layer 1 (any kind)
        p.note_access(2);
        assert_eq!(
            p.order_for_test(1),
            vec![0, 1, 3, 2],
            "expert 2 accessed → evicted last"
        );
        assert_eq!(p.recent_union(3), vec![(1, 2)]);
    }

    #[test]
    fn blk_prefix_parses_ggml_and_gguf_names() {
        assert_eq!(blk_of("blk.1.ffn_gate_exps"), Some(1));
        assert_eq!(blk_of("blk.12.ffn_up_exps.weight"), Some(12));
        assert_eq!(blk_of("output.weight"), None);
        assert_eq!(blk_of("blk.x.ffn_gate_exps"), None);
    }

    #[test]
    fn order_built_once_per_layer_per_step_and_content_is_correct() {
        let pq = PolicyQueryState::new(2, 6);
        // distinct access patterns per layer so the orders differ
        pq.last_queried.store(0, Ordering::Relaxed);
        pq.note_access(3);
        let o0_a = pq.order_for_test(0);
        pq.last_queried.store(1, Ordering::Relaxed);
        pq.note_access(5);
        let o1 = pq.order_for_test(1); // interleaved layer, same step
        let o0_b = pq.order_for_test(0); // cache hit for layer 0, same step
        assert_eq!(
            o0_a, o0_b,
            "cache hit must return layer 0's order, not the last-built buffer"
        );
        assert_ne!(o0_a, o1);
        assert_eq!(pq.order_builds(), 2, "one build per (layer, step)");
        // None (identity) must invalidate/never poison the layer cache
        let _ = pq.publish_order(None);
        let o0_c = pq.order_for_test(0);
        assert_eq!(
            o0_a, o0_c,
            "identity publish must not corrupt the layer-0 cache"
        );
        pq.step.fetch_add(1, Ordering::Relaxed);
        let _ = pq.order_for_test(0);
        assert_eq!(pq.order_builds(), 3, "new step must rebuild");
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
    fn published_order_is_identity_for_non_expert_and_a_permutation_otherwise() {
        let pq = PolicyQueryState::new(2, 6);
        let id = pq.publish_order(None);
        // SAFETY: the published buffer holds n_experts entries until the next publish.
        assert_eq!(
            unsafe { std::slice::from_raw_parts(id, 6) },
            &[0, 1, 2, 3, 4, 5]
        );
        pq.last_queried.store(1, Ordering::Relaxed);
        pq.note_access(4);
        let o = pq.order_for_test(1);
        assert!(is_permutation(&o, 6));
        assert_eq!(*o.last().unwrap(), 4);
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
    fn notify_books_recency_at_the_current_decode_step() {
        let pq = PolicyQueryState::new(2, 4);
        // Notify before any query: defensive ignore.
        pq.note_access(0);
        assert!((0..8).all(|i| pq.access_step(i).is_none()));

        assert_eq!(pq.order_for_test(0), vec![0, 1, 2, 3]);
        pq.last_queried.store(0, Ordering::Relaxed);
        pq.step.store(7, Ordering::Relaxed);
        for id in 0..4 {
            pq.note_access(id);
        }
        assert_eq!(
            pq.step.load(Ordering::Relaxed),
            7,
            "notify never moves the step — Session's decode paths own it"
        );
        assert!((0..4).all(|i| pq.access_step(i) == Some(7)));

        // A later step's notify rewrites recency and feeds the prefetcher.
        pq.step.store(9, Ordering::Relaxed);
        pq.note_access(2);
        assert_eq!(pq.access_step(2), Some(9));
        assert_eq!(
            pq.recent_union(1),
            vec![(0, 2)],
            "only the newest step is in the last-1 union"
        );
        assert!(
            pq.recent_union(0).is_empty(),
            "k = 0 excludes everything (cutoff = current step)"
        );
    }

    #[test]
    fn prefill_routed_experts_are_not_booked_as_recent() {
        let pq = PolicyQueryState::new(2, 4);
        pq.last_queried.store(0, Ordering::Relaxed);
        pq.step.store(5, Ordering::Relaxed);
        pq.scoring.store(false, Ordering::Relaxed);
        for id in 0..4 {
            pq.note_access(id);
        }
        assert!(
            (0..8).all(|i| pq.access_step(i).is_none()),
            "prefill accesses must not stamp recency (§6.2)"
        );
        pq.scoring.store(true, Ordering::Relaxed);
        pq.note_access(1);
        assert_eq!(pq.access_step(1), Some(5));
    }

    #[test]
    fn out_of_range_id_never_spills_into_the_next_layer() {
        let pq = PolicyQueryState::new(2, 4);
        pq.last_queried.store(0, Ordering::Relaxed);
        pq.step.store(3, Ordering::Relaxed);
        // id == n_experts would have passed the old idx < map.len() guard and
        // booked layer 1's row 0.
        pq.note_access(4);
        pq.note_access(-1);
        pq.note_access(100);
        assert!(
            (0..8).all(|i| pq.access_step(i).is_none()),
            "out-of-range ids are rejected, not wrapped into the next layer"
        );
        pq.note_access(3);
        assert_eq!(pq.access_step(3), Some(3), "max valid id still books");
    }
}
