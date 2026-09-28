//! P3 FFI glue (§3.4): process-lifetime hook registration, the
//! `ensure_host_resident` trampoline, and the per-tensor registration table.
//!
//! Hook lifecycle (Global Constraints): registered exactly ONCE per process
//! via [`HANDLE`]; the pool handle is `Box::leak`ed and hooks are never
//! unregistered — the C side stores the fn pointers in plain non-atomic
//! statics, so any teardown could race a mid-flight callback (UB). The
//! registration table fills lazily on the first callback per expert tensor
//! (the frozen P1 ABI exposes expert metadata per tensor but no model-tensor
//! enumeration, so data pointers are only knowable when the substrate hands
//! the tensor to the callback). On a GPU split these first callbacks fire
//! during `load()`'s decode self-check, so registration cross-checks still
//! fail the load loudly.
//!
//! Deviations from the plan sketch (protocol step 7):
//! - *Lazy registration:* the plan's load-time tensor walk is not expressible
//!   through the frozen ABI (no enumeration API); the table is built at first
//!   callback per tensor instead. The load-time cross-checks (source plane
//!   stride == substrate stride per expert tensor; one pread byte-compared
//!   against the loader-populated host bytes) run at that registration, where
//!   the host buffer is guaranteed untouched by the pool — mismatch is a loud
//!   abort (the void FFI ABI has no error return; Global Constraints).
//! - *Single-expert `ensure_range` calls:* the pager's copy closure fires
//!   once per key with slot bytes only; the glue loops `first..=last` itself
//!   so each copy knows its exact destination offset (plane `e` at
//!   `data + e·stride`), with no reliance on intra-range call ordering.
#![allow(unsafe_code)]

use std::collections::HashMap;
use std::ffi::{CStr, c_void};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use golem_llama_sys::raw;
use golem_placement::pool::{
    ExpertKind, ExpertPool, ExpertSource, GgufPreadSource, P3Stats, PolicySnapshot, SnapshotHandle,
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
    /// Bytes per expert plane — `golem_tensor_expert_stride_bytes` (`nb[2]`,
    /// authoritative; never recomputed, never hard-coded per kind).
    stride: usize,
    /// Expert count (`ne[2]` via `golem_tensor_expert_count`).
    n_experts: u32,
    /// Plane 0 of the tensor's host weights buffer. Stable for the process
    /// lifetime: hooks are never unregistered, so the substrate's weights
    /// buffer (and this pointer) outlives every callback by construction.
    data: *mut u8,
}

/// The leaked process-lifetime P3 state. See [`HANDLE`].
struct PoolHandle {
    /// Dense-lead-adjusted expert layer count (from the GGUF walk at `enable`).
    n_layers: u32,
    /// Dense lead: expert tensors exist only from blk.<dense_lead> on; pool
    /// layers are renumbered to exclude it.
    dense_lead: u32,
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

/// Builds the RAM expert pool for `config` (which must set
/// `ram_experts_gib > 0`) and registers the P3 hooks for the process
/// lifetime. Returns the pool's counters for [`crate::session::Session::
/// p3_stats`].
///
/// # Errors
/// GGUF/geometry failures, arena mmap/mlock failures (no silent unlocked
/// fallback, §5.4), or a second pool-enabled session in one process (M2.3
/// serves exactly one; multi-session arrives with M4's global registration).
pub(crate) fn enable(config: &EngineConfig) -> Result<Arc<P3Stats>, EngineError> {
    debug_assert!(config.ram_experts_gib > 0, "enable() with pool disabled");
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
        .ok_or_else(|| EngineError::PoolInit("model has no expert tensors".into()))?;

    let slot_size = usize::try_from(geo.slot_size)
        .map_err(|_| EngineError::PoolInit("slot size > usize".into()))?;
    let pool_bytes = config
        .ram_experts_gib
        .checked_mul(1 << 30)
        .ok_or_else(|| EngineError::PoolInit("ram_experts_gib overflows bytes".into()))?;
    let slots = pool_bytes / geo.slot_size;
    let slots = u32::try_from(slots)
        .map_err(|_| EngineError::PoolInit("slot count exceeds u32::MAX".into()))?;
    if slots < geo.n_layers {
        return Err(EngineError::PoolInit(format!(
            "pool of {slots} slots (ram_experts_gib = {}) cannot cover {} expert layers \
             (one slot per layer minimum; plane stride {slot_size} B)",
            config.ram_experts_gib, geo.n_layers
        )));
    }
    // Remainder bytes are trimmed by integer division: pool_bytes is a
    // multiple of slot_size again after this.
    let arena = golem_placement::pool::Arena::new(slots as usize * slot_size, slot_size)
        .map_err(|e| EngineError::PoolInit(e.to_string()))?;

    // M2.3 policy state: empty snapshot (nothing pinned/protected) until the
    // M2.2-backed builder lands with M2.4's snapshot wiring.
    let snapshots = Arc::new(SnapshotHandle::new(Arc::new(PolicySnapshot::test_empty())));
    let pool = ExpertPool::new(
        arena,
        source.clone(),
        snapshots,
        Arc::new(P3Stats::new()),
        geo.n_layers,
        geo.n_experts,
    );

    let handle: &'static PoolHandle = Box::leak(Box::new(PoolHandle {
        n_layers: geo.n_layers,
        dense_lead,
        source,
        pool: pool.clone(),
        n_experts: geo.n_experts,
        tensors: Mutex::new(HashMap::new()),
        slow_calls: AtomicUsize::new(0),
    }));
    if HANDLE.set(handle).is_err() {
        // Raced a concurrent enable(); the leaked arena is inert garbage the
        // process carries — loud failure beats silent double registration.
        return Err(EngineError::PoolInit(
            "concurrent pool-enabled session construction".into(),
        ));
    }

    // SAFETY: `handle` is leaked ('static) and never mutated after this, so
    // the context pointer outlives every callback; `Some(ensure_host_resident)`
    // is a valid fn pointer; policy_query/notify_access stay NULL (P2/P4 own
    // their call-sites — M2.3 invents no packing contract).
    unsafe {
        raw::golem_register_hooks(
            Some(ensure_host_resident),
            None,
            None,
            handle as *const PoolHandle as *mut c_void,
        );
    }

    tracing::info!(
        layers = geo.n_layers,
        experts = geo.n_experts,
        slot_size,
        slots,
        gib = config.ram_experts_gib,
        "P3 RAM expert pool registered (hooks set for process lifetime)"
    );
    Ok(pool.stats.clone())
}

/// Loud terminal state for FFI-adjacent contract violations (Global
/// Constraints: the void ABI leaves no non-silent failure path; corrupt-weight
/// decode is worse than a dead process — documented, deliberate).
fn p3_fatal(context: String) -> ! {
    tracing::error!("{context}");
    std::process::abort()
}

/// The P1 `ensure_host_resident` trampoline. Runs on the substrate decode
/// thread; touches only the pool's own synchronization — no substrate state,
/// no locks ordered with C code (T6-reviewed contract).
///
/// # Contract
/// No input clamping: an out-of-range or inverted `[first_id, last_id]` is a
/// contract violation → logged + `abort()` (Global Constraints). Non-expert
/// tensors return immediately (null-guard semantics upstream of the pool).
extern "C" fn ensure_host_resident(
    ctx: *mut c_void,
    tensor: *const c_void,
    first_id: i32,
    last_id: i32,
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

    for expert in first..=last {
        // Plane `e` lives at data + e·stride (plane 0 at +0; the registration
        // cross-check proved the source agrees byte-for-byte).
        let stride = entry.stride;
        // SAFETY: `add` is in-bounds: expert < n_experts (checked above) and
        // the buffer holds n_experts·stride bytes (registration-time
        // cross-check vs the source's GGUF geometry walk).
        let dst = unsafe { entry.data.add(expert as usize * stride) };
        let result = h.pool.ensure_range(
            true,
            entry.layer,
            entry.kind,
            expert,
            expert,
            &mut |slot: &[u8]| {
                // SAFETY: copy under the lease (the slot cannot be refilled while
                // the lease is held, §6.6); `dst` is the tensor's process-lifetime
                // host buffer, and the C call-site reads it only after this
                // callback returns; ranges are exactly one plane apart.
                unsafe {
                    std::ptr::copy_nonoverlapping(slot.as_ptr(), dst, stride);
                }
            },
        );
        if let Err(e) = result {
            p3_fatal(format!(
                "P3 page-in failed (layer {}, {kind:?}, expert {expert}): {e}; \
                 aborting rather than decoding on garbage",
                entry.layer,
                kind = entry.kind
            ));
        }
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
        // bytes from `data` (deviation 1: loader-owned host weights), and
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
            stride,
            n_experts: n_experts as u32,
            data: t.data.cast::<u8>(),
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
    fn blk_prefix_parses_ggml_and_gguf_names() {
        assert_eq!(blk_of("blk.1.ffn_gate_exps"), Some(1));
        assert_eq!(blk_of("blk.12.ffn_up_exps.weight"), Some(12));
        assert_eq!(blk_of("output.weight"), None);
        assert_eq!(blk_of("blk.x.ffn_gate_exps"), None);
    }
}
