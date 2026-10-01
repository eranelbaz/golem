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
//! the tensor to the callback). On a GPU split these first callbacks fire
//! during `load()`'s decode self-check, so registration cross-checks still
//! fail the load loudly.
//!
//! Registration shape:
//! - *Lazy registration:* a load-time tensor walk is not expressible
//!   through the P1 ABI (no enumeration API); the table is built at first
//!   callback per tensor instead. The load-time cross-checks (source plane
//!   stride == substrate stride per expert tensor; one pread byte-compared
//!   against the loader-populated host bytes) run at that registration, where
//!   the host buffer is guaranteed untouched by the pool — mismatch is a loud
//!   abort (the void FFI ABI has no error return).
//! - *Single-expert `ensure_range` calls:* the pager's copy closure fires
//!   once per key with slot bytes only; the glue loops `first..=last` itself
//!   so each copy knows its exact destination offset (plane `e` at
//!   `data + e·stride`), with no reliance on intra-range call ordering.
#![allow(unsafe_code)]

use std::collections::HashMap;
use std::ffi::{CStr, c_void};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use golem_llama_sys::raw;
use golem_placement::pool::{
    AccessFeed, ExpertKind, ExpertPool, ExpertSource, GgufPreadSource, P3Stats, PREFETCH_K,
    PolicySnapshot, PrefetchHandle, SnapshotHandle, spawn_prefetch,
};

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

/// The leaked process-lifetime P3 state. See [`HANDLE`].
struct PoolHandle {
    /// Dense-lead-adjusted expert layer count (from the GGUF walk at `enable`).
    n_layers: u32,
    /// Dense lead: expert tensors exist only from blk.<dense_lead> on; pool
    /// layers are renumbered to exclude it.
    dense_lead: u32,
    /// §6.5 prefetch thread. Declaration order is the drop contract: the
    /// handle is stopped and joined BEFORE the pool `Arc` drops. (M2.3 leaks
    /// the whole handle — process lifetime, hooks never unregistered — so
    /// neither field actually drops; the order keeps the invariant structural
    /// when M4's server re-homes registration to a global.)
    // Never read directly: its whole job is Drop (stop + join).
    #[allow(dead_code)]
    prefetch: PrefetchHandle,
    /// GGUF-side expert source, kept for registration-time cross-checks.
    source: Arc<GgufPreadSource>,
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
    /// Leases held from the previous scheduler compute: filled
    /// by `ensure_host_resident`, drained by `copy_done` at the start of the
    /// NEXT compute — the engine synchronizes after every decode
    /// (session.rs synchronizes after every decode), so the prior step's DMAs
    /// are complete.
    in_flight_leases: Mutex<Vec<golem_placement::pool::Lease>>,
}

// SAFETY: `data` pointers point into the substrate's host weights buffer,
// which is process-lifetime by hook-lifecycle decree (leaked handle, hooks
// never unregistered ⇒ the model is never freed while a callback can run);
// entries are only added through the owning mutex. Every other field is
// plain owned data, so `PoolHandle` crossing threads upholds aliasing rules.
unsafe impl Send for PoolHandle {}
unsafe impl Sync for PoolHandle {}

/// Process-lifetime registration: set once, read forever, never torn down.
static HANDLE: OnceLock<&'static PoolHandle> = OnceLock::new();

/// Build-in-progress reservation: set by the first `enable` BEFORE any
/// resource is allocated, so a concurrent second `enable` fails fast instead
/// of racing [`HANDLE`] and leaking a built arena + prefetch thread.
static ENABLING: AtomicBool = AtomicBool::new(false);

/// P2 policy state (§3.4): per-expert last-access steps feeding the
/// evict-first order, plus the double-buffered order slices returned to C.
///
/// Attribution: `notify_access` carries only the expert id, so accesses are
/// booked under the (layer, kind) of the most recent `policy_query`. Exact
/// for the current C call-site shape — C emits notify per routed expert
/// inside the same policy_query-guarded block, single decode thread — M4's
/// multi-context re-homing must pass tensor identity explicitly.
type AccessMap = HashMap<(u32, ExpertKind), Vec<Option<u64>>>;

struct PolicyQueryState {
    /// (layer, kind) → per-expert last-access step; `None` = never accessed.
    last_access: Mutex<AccessMap>,
    /// Decode-step counter: one `llama_decode` call = one step (§6.2 as
    /// amended — prefill chunks count as one step each), bumped by
    /// [`advance_decode_step`] from `Session`'s decode paths. Rust drives
    /// decode, so Rust owns the step — the wall-clock turn heuristic this
    /// replaced collapsed a whole chat turn into one step.
    step: AtomicU64,
    /// Double-buffered order slices: the slice returned to C stays valid
    /// until the next policy_query call (frozen-buffer contract,
    /// golem-hooks.h; C consumes it within the same copy step).
    bufs: [Mutex<Vec<i32>>; 2],
    /// Index of the buffer currently exposed to C (`bufs` element).
    exposed: AtomicUsize,
    /// Last (layer, kind) passed to policy_query; notify attribution key.
    last_queried: Mutex<Option<(u32, ExpertKind)>>,
    /// `GOLEM_TEST_SHUFFLE_ORDER`, read ONCE at enable; reverses the recency
    /// order to prove consumers are order-agnostic (any permutation is legal,
    /// invariant 4; tests/t3_cache_invariance.rs is the only setter).
    shuffle: bool,
    n_layers: u32,
    n_experts: u32,
}

impl PolicyQueryState {
    fn new(n_layers: u32, n_experts: u32, shuffle: bool) -> Self {
        Self {
            last_access: Mutex::new(HashMap::new()),
            step: AtomicU64::new(0),
            bufs: [Mutex::new(Vec::new()), Mutex::new(Vec::new())],
            exposed: AtomicUsize::new(0),
            last_queried: Mutex::new(None),
            shuffle,
            n_layers,
            n_experts,
        }
    }

    /// Publish `order` in the buffer NOT currently exposed to C and return
    /// its pointer (valid until the next publish, per the frozen-buffer
    /// contract; C consumes within the same copy step).
    fn store_order(&self, order: Vec<i32>) -> *const i32 {
        let next = 1 - self.exposed.load(Ordering::Relaxed);
        *self.bufs[next].lock().unwrap() = order;
        self.exposed.store(next, Ordering::Relaxed);
        self.bufs[next].lock().unwrap().as_ptr()
    }

    /// Evict-first order for one (layer, kind): never-accessed experts
    /// first, then ascending last-access step. First sight of a tensor seeds
    /// its map entry with all-`None` (never accessed).
    fn order_for(&self, layer: u32, kind: ExpertKind) -> Vec<i32> {
        debug_assert!(
            layer < self.n_layers && self.n_experts > 0,
            "order requested outside the registered geometry"
        );
        let hist = {
            let mut map = self.last_access.lock().unwrap();
            map.entry((layer, kind))
                .or_insert_with(|| vec![None; self.n_experts as usize])
                .clone()
        };
        build_order(&hist, self.n_experts as usize, self.shuffle)
    }

    /// Record one routed-expert access under the last queried (layer, kind)
    /// at the current decode step.
    fn note_access(&self, expert_id: i64) {
        let step = self.step.load(Ordering::Relaxed);
        let Some((layer, kind)) = *self.last_queried.lock().unwrap() else {
            return; // defensive: notify before any query — nothing to attribute
        };
        if let Some(hist) = self.last_access.lock().unwrap().get_mut(&(layer, kind)) {
            let e = expert_id as usize;
            if e < hist.len() {
                hist[e] = Some(step);
            }
        }
    }

    /// §6.5 prefetch feed: union of accessed `(layer, expert)` units from the
    /// last `k` steps. The three tensors of a layer book the same routed
    /// experts, so duplicates collapse through a flat per-layer bitset —
    /// no hash set on the feed path.
    fn recent_union(&self, k: usize) -> Vec<(u32, u32)> {
        let cutoff = self.step.load(Ordering::Relaxed).saturating_sub(k as u64);
        let words = self.n_experts.div_ceil(64) as usize;
        let map = self.last_access.lock().unwrap();
        let mut out = Vec::new();
        // Flat per-layer bitset over all layers: the three tensors of a layer
        // book the same routed experts, so duplicates collapse to distinct
        // experts only — no hash set on the feed path.
        let mut seen = vec![0u64; self.n_layers as usize * words];
        for (&(layer, _), hist) in map.iter() {
            let row = &mut seen[layer as usize * words..(layer as usize + 1) * words];
            for (e, s) in hist.iter().enumerate() {
                if s.is_some_and(|step| step > cutoff) && row[e / 64] & (1 << (e % 64)) == 0 {
                    row[e / 64] |= 1 << (e % 64);
                    out.push((layer, e as u32));
                }
            }
        }
        out
    }
}

/// Recency order builder (pure; unit-tested): never-accessed experts first
/// (ascending index), then ascending last-access step. `shuffle` reverses
/// the result — any permutation is legal to return (invariant 4); the knob
/// exists to prove consumers don't depend on the ordering.
///
/// # Panics
/// Malformed input (map length ≠ expert count) — belt-and-suspenders: the
/// map is engine-built with exact length, so a mismatch is an internal
/// contract break.
fn build_order(last_access: &[Option<u64>], n_experts: usize, shuffle: bool) -> Vec<i32> {
    assert_eq!(
        last_access.len(),
        n_experts,
        "malformed access map: {} entries for {n_experts} experts",
        last_access.len()
    );
    let mut idx: Vec<usize> = (0..n_experts).collect();
    idx.sort_by_key(|&i| (last_access[i].is_some(), last_access[i].unwrap_or(0)));
    let order: Vec<i32> = idx.into_iter().map(|i| i as i32).collect();
    if shuffle {
        order.into_iter().rev().collect()
    } else {
        order
    }
}

/// Rust-side permutation check (belt-and-suspenders; C validates the order
/// authoritatively per the placement policy).
fn is_permutation(order: &[i32], n: usize) -> bool {
    order.len() == n && order.iter().all(|&i| i >= 0 && (i as usize) < n) && {
        let mut seen = vec![false; n];
        order
            .iter()
            .all(|&i| !std::mem::replace(&mut seen[i as usize], true))
    }
}

/// Bumps the policy decode step: `Session` calls this after every successful
/// `llama_decode` (§6.2 as amended: a step is one decode call, prefill chunks
/// included). No-op while hooks are unregistered (self-check decodes).
pub(crate) fn advance_decode_step() {
    if let Some(h) = HANDLE.get() {
        h.policy.step.fetch_add(1, Ordering::Relaxed);
    }
}

/// Whether the process-global hooks are registered (a second pool-enabled
/// session in one process is refused; M4's global registration re-homes it).
pub(crate) fn hooks_registered() -> bool {
    HANDLE.get().is_some()
}

/// Builds the RAM expert pool for `config` (which must set
/// `ram_experts_gib > 0`) and registers the P3 hooks for the process
/// lifetime. Returns the pool's counters for [`crate::session::Session::
/// p3_stats`].
///
/// # Errors
/// GGUF/geometry failures, arena mmap/mlock failures (no silent unlocked
/// fallback, §5.4), or a second pool-enabled session in one process (M2.3
/// serves exactly one pool-enabled session per process).
pub(crate) fn enable(config: &EngineConfig) -> Result<Arc<P3Stats>, EngineError> {
    debug_assert!(config.ram_experts_gib > 0, "enable() with pool disabled");
    if ENABLING
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err(EngineError::PoolInit(
            "concurrent pool-enabled session construction".into(),
        ));
    }
    let result = enable_inner(config);
    if result.is_err() {
        // Reservation released: a later enable() may retry from scratch.
        ENABLING.store(false, Ordering::SeqCst);
    }
    result
}

fn enable_inner(config: &EngineConfig) -> Result<Arc<P3Stats>, EngineError> {
    if HANDLE.get().is_some() {
        return Err(EngineError::PoolInit(
            "a pool-enabled session already exists (one per process in M2.3)".into(),
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

    let source = Arc::new(
        GgufPreadSource::open(&config.model_path, dense_lead)
            .map_err(|e| EngineError::PoolInit(e.to_string()))?,
    );
    let geo = source
        .geometry()
        .map_err(|e| EngineError::PoolInit(e.to_string()))?
        .ok_or_else(|| EngineError::PoolInit("model has no expert tensors".into()))?;

    // RAM slot size = the max unit size (all three planes of the largest
    // layer); per-class slabs arrive with the multi-class layout.
    let slot_size = usize::try_from(geo.max_unit())
        .map_err(|_| EngineError::PoolInit("slot size > usize".into()))?;
    let n_layers = geo.n_layers();
    let pool_bytes = config
        .ram_experts_gib
        .checked_mul(1 << 30)
        .ok_or_else(|| EngineError::PoolInit("ram_experts_gib overflows bytes".into()))?;
    let slots = pool_bytes / slot_size as u64;
    let slots = u32::try_from(slots)
        .map_err(|_| EngineError::PoolInit("slot count exceeds u32::MAX".into()))?;
    if slots < n_layers {
        return Err(EngineError::PoolInit(format!(
            "pool of {slots} slots (ram_experts_gib = {}) cannot cover {n_layers} expert layers \
             (one slot per layer minimum; plane stride {slot_size} B)",
            config.ram_experts_gib
        )));
    }
    // Remainder bytes are trimmed by integer division: pool_bytes is a
    // multiple of slot_size again after this. Single class: every slot is
    // one max-unit slot until the per-class layout lands (§5 amendment).
    let arena =
        golem_placement::pool::Arena::new(&[golem_placement::pool::SlotClass { slot_size, slots }])
            .map_err(|e| EngineError::PoolInit(e.to_string()))?;

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

    // M2.3 policy state: empty snapshot (nothing pinned/protected) until the
    // the placement-backed snapshot builder lands with the engine wiring.
    let snapshots = Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::test_empty())));
    let pool = ExpertPool::new(
        arena,
        source.clone(),
        snapshots,
        Arc::new(P3Stats::new()),
        geo.clone(),
        vec![0; geo.n_layers() as usize],
    );

    // P2 state: per-expert recency (never-accessed first, then
    // ascending last-access step) — pure recency order. The shuffle knob is
    // read exactly once per process here.
    let shuffle = std::env::var("GOLEM_TEST_SHUFFLE_ORDER").is_ok_and(|v| v != "0");
    let policy: &'static PolicyQueryState = Box::leak(Box::new(PolicyQueryState::new(
        n_layers,
        geo.n_experts,
        shuffle,
    )));
    let prefetch = spawn_prefetch(pool.clone(), Arc::new(PolicyFeed(policy)), PREFETCH_K);

    let handle: &'static PoolHandle = Box::leak(Box::new(PoolHandle {
        n_layers: geo.n_layers(),
        dense_lead,
        prefetch,
        source,
        pool: pool.clone(),
        n_experts: geo.n_experts,
        tensors: Mutex::new(HashMap::new()),
        in_flight_leases: Mutex::new(Vec::new()),
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
    // The device-cache budget is a per-(layer, kind) expert count, never a
    // byte cap on whichever tensor registers first.
    // It is never "unlimited": device allocations are bounded by the VRAM
    // budget, never proportional to expert count (unbounded slabs are ≈184 GB
    // on GLM-5.3-Flash — 126 live tensors × 288 experts × ≈5 MB — against a
    // 24 GiB card). An explicit config value wins; otherwise K derives from
    // the free VRAM at enable time (the model's non-expert weights are
    // already resident, so what is free is what the slabs + KV + compute
    // buffers must share), with half held back as growth headroom:
    // K = free/2 / (n_layers × 3 kinds × slot_size) − 1 = 8 on the target
    // (≈6.2 GiB usable / (42 × 3 × 5.77 MB) ≈ 9.15, minus the − 1 headroom).
    // K is floored at the model's top-k (expert_used_count): a routed set
    // larger than K aborts mid-decode, so a smaller K must fail at enable.
    let n_expert_used = metadata
        .kv_u64(&format!("{arch}.expert_used_count"))
        .ok_or_else(|| {
            EngineError::PoolInit(format!("{arch}.expert_used_count missing from the GGUF"))
        })?;
    let n_expert_used = u32::try_from(n_expert_used)
        .map_err(|_| EngineError::PoolInit("expert_used_count > u32::MAX".into()))?;
    let k = derive_device_cache_experts(config, n_layers, slot_size as u64, n_expert_used)?;
    // SAFETY: FFI setter, no invariants.
    unsafe {
        raw::golem_set_device_cache_experts(k as i32);
    }

    // SAFETY: `handle` is leaked ('static) and never mutated after this, so
    // the context pointer outlives every callback; all three are valid fn
    // pointers (P2 policy_query / P4 notify_access; the C
    // side's n_copies == 1 abort is the defense-in-depth — M2.3's config
    // cannot express pipeline parallelism or multi-GPU to pre-reject here).
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
        slot_size,
        slots,
        gib = config.ram_experts_gib,
        "P3 RAM expert pool registered (hooks set for process lifetime)"
    );
    Ok(pool.stats.clone())
}

/// Per-(layer, kind) device-slab expert count for this pool. Explicit
/// `device_cache_experts > 0` wins; otherwise derived from free VRAM. Never
/// unlimited, never zero: a routed set larger than K aborts, so K = 0 could
/// not decode at all.
fn derive_device_cache_experts(
    #[cfg_attr(not(golem_backend_hip), allow(unused_variables))] config: &EngineConfig,
    #[cfg_attr(not(golem_backend_hip), allow(unused_variables))] n_layers: u32,
    #[cfg_attr(not(golem_backend_hip), allow(unused_variables))] slot_size: u64,
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
        return k_from_free_vram(free as u64, n_layers, slot_size, n_expert_used);
    }
    Err(EngineError::PoolInit(
        "device_cache_experts unset and no VRAM query available to derive a bounded \
         slab budget — set device_cache_experts explicitly"
            .into(),
    ))
}

/// Derived K from free VRAM: half is slab budget, the rest KV + compute-buffer
/// growth headroom; `− 1` is per-slab headroom. Floored at top-k.
#[cfg_attr(not(golem_backend_hip), allow(dead_code))]
fn k_from_free_vram(
    free: u64,
    n_layers: u32,
    slot_size: u64,
    n_expert_used: u32,
) -> Result<u32, EngineError> {
    let slab_bytes_per_expert = u64::from(n_layers)
        .saturating_mul(3)
        .saturating_mul(slot_size);
    let k = (free / 2) / slab_bytes_per_expert.max(1);
    let k = k.saturating_sub(1).clamp(1, i32::MAX as u64);
    if k < u64::from(n_expert_used) {
        return Err(EngineError::PoolInit(format!(
            "free VRAM supports a slab of only {k} experts per (layer, kind), below \
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
    // SAFETY: C allocates `n` entries and passes the array here (copy_experts_to).
    let out: &mut [*const c_void] = unsafe {
        assert!(!out_ptrs.is_null());
        std::slice::from_raw_parts_mut(out_ptrs, n)
    };

    // The pool IS the H2D source. Claim + page-in each expert and
    // hand the slot pointer to C; the lease stays held (in `in_flight_leases`)
    // until `copy_done`, so eviction cannot recycle a slot mid-DMA.
    let (ptrs, leases) =
        match h
            .pool
            .ensure_range_leased(true, entry.layer, entry.kind, first, last)
        {
            Ok(r) => r,
            Err(e) => p3_fatal(format!(
                "P3 page-in failed (layer {}, {kind:?}, experts [{first}, {last}]): {e}; \
             aborting rather than decoding on garbage",
                entry.layer,
                kind = entry.kind
            )),
        };
    for (slot, dst) in ptrs.into_iter().zip(out.iter_mut()) {
        *dst = slot as *const c_void;
    }
    h.in_flight_leases.lock().unwrap().extend(leases);

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

/// Releases the host-pool leases of the previous compute. Runs
/// at the START of a scheduler compute; the engine synchronizes after
/// every decode, so its DMAs are complete.
extern "C" fn copy_done(ctx: *mut c_void) {
    // SAFETY: ctx was installed by `enable` as a leaked `&'static PoolHandle`.
    let h = unsafe { &*(ctx as *const PoolHandle) };
    h.in_flight_leases.lock().unwrap().clear(); // Lease::drop releases each
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
/// the full `entry_for` validate-and-cross-check path — within one copy step
/// C calls policy_query BEFORE ensure_host_resident, so the first expert
/// tensor ever seen arrives HERE first; no abort on first sight.
extern "C" fn policy_query(ctx: *mut c_void, tensor: *const c_void, n_experts: i32) -> *const i32 {
    // SAFETY: ctx was installed by `enable` as a leaked `&'static PoolHandle`
    // and is never mutated afterwards; it outlives every callback.
    let h = unsafe { &*(ctx as *const PoolHandle) };
    let Some(entry) = h.entry_for(tensor) else {
        // Defensive: C queries only expert tensors; anything else gets a
        // legal identity permutation (invariant 4: any permutation is valid).
        return h.policy.store_order((0..n_experts.max(0)).collect());
    };
    if n_experts != entry.n_experts as i32 {
        p3_fatal(format!(
            "P2: policy_query n_experts {n_experts} vs registered {} for layer {}",
            entry.n_experts, entry.layer
        ));
    }
    *h.policy.last_queried.lock().unwrap() = Some((entry.layer, entry.kind));
    let order = h.policy.order_for(entry.layer, entry.kind);
    // Belt-and-suspenders: C validates the permutation authoritatively
    // and aborts on corruption; failing here yields a
    // Rust-side message instead.
    if !is_permutation(&order, entry.n_experts as usize) {
        p3_fatal(format!(
            "P2: order builder produced a non-permutation for layer {}, {kind:?}: {order:?}",
            entry.layer,
            kind = entry.kind
        ));
    }
    h.policy.store_order(order)
}

/// The P4 `notify_access` trampoline: one routed expert of the tensor most
/// recently passed to [`policy_query`] — call-site adjacency guarantees the
/// attribution for M2.3's single decode thread (see [`PolicyQueryState`]).
extern "C" fn notify_access(ctx: *mut c_void, expert_id: i64) {
    // SAFETY: see policy_query.
    let h = unsafe { &*(ctx as *const PoolHandle) };
    h.policy.note_access(expert_id);
}

impl PoolHandle {
    /// Registered entry for `tensor`, registering it on first sight.
    ///
    /// First sight = the substrate's first `ensure_host_resident` call for
    /// this tensor, which precedes any host-byte read and happens while the
    /// loader-populated weights buffer is still pristine — exactly the window
    /// the load-time cross-checks need.
    fn entry_for(&self, tensor: *const c_void) -> Option<TensorEntry> {
        if tensor.is_null() {
            return None;
        }
        let key = tensor as usize;
        if let Some(entry) = self.tensors.lock().unwrap().get(&key) {
            return Some(*entry);
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

        let n_experts = unsafe { raw::golem_tensor_expert_count(tensor) };
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
        if let Err(e) = self.source.read_expert(layer, kind, 0, &mut disk) {
            p3_fatal(format!("P3: verification pread failed for {name}: {e}"));
        }
        // SAFETY: the weights buffer holds n_experts·stride fully populated
        // bytes from `data` (loader-owned host weights), and
        // `t.data` is non-null for a loaded weights tensor.
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
        tracing::debug!(tensor = %name, layer, ?kind, stride, "P3 tensor registered");
        Some(entry)
    }
}

/// `blk.<N>.` prefix parse (shared shape with the pool source's own walker).
fn blk_of(name: &str) -> Option<u32> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_cache_experts_explicit_wins_and_default_is_never_unbounded() {
        let c = EngineConfig {
            device_cache_experts: 9,
            ..EngineConfig::default()
        };
        assert_eq!(
            derive_device_cache_experts(&c, 42, 5_777_000, 8).unwrap(),
            9
        );
        // Explicit K below the model's top-k must fail at enable, not abort
        // on the first decode (a routed set larger than K aborts).
        let c_small = EngineConfig {
            device_cache_experts: 7,
            ..EngineConfig::default()
        };
        let err = derive_device_cache_experts(&c_small, 42, 5_777_000, 8).unwrap_err();
        assert!(err.to_string().contains("expert_used_count"), "{err}");
        // Default (0): derived, bounded, never 0. On the CPU backend there is
        // no VRAM query, so derivation is a loud error, not an unbounded cap.
        #[cfg(not(golem_backend_hip))]
        assert!(
            derive_device_cache_experts(&EngineConfig::default(), 42, 5_777_000, 8).is_err(),
            "no VRAM query: derivation must fail loudly"
        );
        // An explicit K that would wrap the i32 ABI is refused, not passed
        // through as "full slab".
        let c_huge = EngineConfig {
            device_cache_experts: u32::MAX,
            ..EngineConfig::default()
        };
        let err = derive_device_cache_experts(&c_huge, 42, 5_777_000, 8).unwrap_err();
        assert!(err.to_string().contains("i32::MAX"), "{err}");
    }

    #[test]
    fn derived_k_matches_target_arithmetic_and_floors_at_top_k() {
        // Target: ≈12.4 GiB free → 6.2 GiB slab budget / (42 × 3 × 5.77 MB)
        // ≈ 9.15 → 9 − 1 = 8, exactly top-k.
        let free = 2 * 6_657_000_000u64;
        assert_eq!(k_from_free_vram(free, 42, 5_777_000, 8).unwrap(), 8);
        // Any less free VRAM drops K below top-k: refused at enable.
        let err = k_from_free_vram(free - 2 * 727_902_000, 42, 5_777_000, 8).unwrap_err();
        assert!(err.to_string().contains("expert_used_count 8"), "{err}");
        // Huge free VRAM stays within the i32 ABI.
        assert_eq!(
            k_from_free_vram(u64::MAX, 1, 1, 8).unwrap(),
            i32::MAX as u32
        );
    }

    #[test]
    fn blk_prefix_parses_ggml_and_gguf_names() {
        assert_eq!(blk_of("blk.1.ffn_gate_exps"), Some(1));
        assert_eq!(blk_of("blk.12.ffn_up_exps.weight"), Some(12));
        assert_eq!(blk_of("output.weight"), None);
        assert_eq!(blk_of("blk.x.ffn_gate_exps"), None);
    }

    #[test]
    fn order_builder_recency_sorts_never_accessed_first() {
        let hist = vec![Some(5), None, Some(2), None, Some(2)];
        assert_eq!(
            build_order(&hist, 5, false),
            vec![1, 3, 2, 4, 0],
            "never-accessed first (ascending index), then ascending step"
        );
    }

    #[test]
    fn order_builder_shuffle_changes_order_but_stays_a_permutation() {
        let hist: Vec<Option<u64>> = (0..8u64).map(Some).collect();
        assert_eq!(build_order(&hist, 8, false), (0..8).collect::<Vec<i32>>());
        let shuffled = build_order(&hist, 8, true);
        assert_ne!(shuffled, build_order(&hist, 8, false));
        assert!(is_permutation(&shuffled, 8));
    }

    #[test]
    #[should_panic(expected = "malformed access map")]
    fn order_builder_rejects_malformed_map() {
        build_order(&[Some(1), None], 3, false);
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
        let pq = PolicyQueryState::new(2, 4, false);
        // Notify before any query: defensive ignore.
        pq.note_access(0);
        assert!(pq.last_access.lock().unwrap().is_empty());

        assert_eq!(pq.order_for(0, ExpertKind::Gate), vec![0, 1, 2, 3]);
        *pq.last_queried.lock().unwrap() = Some((0, ExpertKind::Gate));
        pq.step.store(7, Ordering::Relaxed);
        for id in 0..4 {
            pq.note_access(id);
        }
        assert_eq!(
            pq.step.load(Ordering::Relaxed),
            7,
            "notify never moves the step — Session's decode paths own it"
        );
        assert_eq!(
            pq.last_access.lock().unwrap()[&(0, ExpertKind::Gate)],
            vec![Some(7); 4]
        );

        // A later step's notify rewrites recency and feeds the prefetcher.
        pq.step.store(9, Ordering::Relaxed);
        pq.note_access(2);
        assert_eq!(
            pq.last_access.lock().unwrap()[&(0, ExpertKind::Gate)][2],
            Some(9)
        );
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
}
