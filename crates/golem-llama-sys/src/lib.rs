#![allow(unsafe_code, missing_docs)]
include!(concat!(env!("OUT_DIR"), "/backend_info.rs"));

#[path = "../pin_gate.rs"]
pub mod pin_gate;

#[path = "../backend_probe.rs"]
pub mod backend_probe;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Hip,
}

impl Backend {
    pub const fn as_str(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Hip => "hip",
        }
    }
}

pub fn substrate_backend() -> Backend {
    BACKEND
}

pub fn backend_env_input() -> &'static str {
    BACKEND_INPUT
}

pub fn pin_gate_skipped() -> bool {
    option_env!("GOLEM_LLAMA_PIN_GATE_SKIPPED") == Some("1")
}

/// ABI revision of the compiled substrate's hook surface; must equal
/// [`raw::GOLEM_HOOKS_ABI_VERSION`] (the header the bindings were generated from).
pub fn hooks_abi_version() -> i32 {
    // SAFETY: pure function, no invariants.
    unsafe { raw::golem_hooks_abi_version() }
}

// The RAII surface below is written against ABI v5 (per-model `golem_ctx`,
// handle-returning register, batched ensure, triple-buffered order push).
// A header drift past v5 must fail this build, not silently re-shape the
// guards.
const _: () = assert!(raw::GOLEM_HOOKS_ABI_VERSION == 5);

/// Per-model hook context (ABI v5): owns a `golem_ctx_t` from
/// `golem_ctx_create` and frees it on drop. `golem_ctx_free` aborts while
/// hooks are registered or scheds are attached — the C-side guard that makes
/// the drop order (unregister-join → llama_free → ctx free) enforceable.
/// Shareable via `Arc` so the order publisher and the decode thread can hold
/// the live pointer; the free runs only when the last reference drops.
pub struct GolemCtx {
    inner: std::sync::Arc<CtxRaw>,
}

/// The context Arc: cloned by [`Hooked::ctx_arc`] when the owner needs to
/// free the ctx strictly after other teardown steps (the §3.4 Session drop
/// order: unregister-join → `llama_free` → `golem_ctx_free`).
pub struct CtxRaw(raw::golem_ctx_t);
impl Drop for CtxRaw {
    fn drop(&mut self) {
        // Aborts (documented C behavior) if hooks are still registered or a
        // sched is still attached: that is a drop-order bug in the owner.
        // SAFETY: `self.0` came from golem_ctx_create and has no other owner.
        unsafe { raw::golem_ctx_free(self.0) }
    }
}

// SAFETY: golem_ctx_t is a C++ heap object the C side never exposes to Rust
// mutation; all its fields are atomics/mutex-protected. The handle only
// crosses FFI boundaries as an opaque pointer.
unsafe impl Send for CtxRaw {}
// SAFETY: as above — the C side synchronizes all access internally.
unsafe impl Sync for CtxRaw {}

impl GolemCtx {
    /// `golem_ctx_create`: `None` on a null return (C allocation failure).
    pub fn create(n_layers: i32, n_experts: i32) -> Option<Self> {
        // SAFETY: plain allocation call, no invariants.
        let ptr = unsafe { raw::golem_ctx_create(n_layers, n_experts) };
        if ptr.is_null() {
            None
        } else {
            Some(Self {
                inner: std::sync::Arc::new(CtxRaw(ptr)),
            })
        }
    }

    /// The raw context pointer (attach seams, order push). Valid while any
    /// `Arc` clone of this context lives.
    pub fn as_ptr(&self) -> raw::golem_ctx_t {
        self.inner.0
    }

    /// Registers the hooks and consumes the context: the returned
    /// [`Hooked`] guard keeps the context alive (via `Arc`) and its `Drop`
    /// unregisters (the C side joins in-flight callbacks) and then recovers
    /// the `payload` Arc. The borrow of the ctx by the registration is
    /// encoded by MOVING the ctx into the guard — no self-referential alias.
    ///
    /// `payload` is passed to C as `Arc::into_raw`; [`Hooked::drop`] recovers
    /// it with `Arc::from_raw` only after the join returns, so no in-flight
    /// callback can race the recovery (the C unregister joins in-flight
    /// steps before returning).
    pub fn register<T: Send>(self, hooks: HookFns, payload: std::sync::Arc<T>) -> Hooked<T> {
        let ptr = self.inner.0;
        let payload_ptr = std::sync::Arc::into_raw(payload);
        // SAFETY: `ptr` is a live ctx (owned by `self.inner`, moved into the
        // guard below); all fn pointers are valid C ABIs or None; the payload
        // pointer stays owned by the guard until after the unregister-join.
        let handle = unsafe {
            raw::golem_register_hooks(
                ptr,
                hooks.ensure,
                hooks.routing,
                hooks.copy_done,
                payload_ptr as *mut T as *mut std::ffi::c_void,
            )
        };
        Hooked {
            handle,
            payload: payload_ptr,
            ctx: self.inner,
        }
    }
}

/// The three hook fn pointers for [`GolemCtx::register`].
#[derive(Clone, Copy, Default)]
pub struct HookFns {
    pub ensure: raw::golem_ensure_host_resident_fn,
    pub routing: raw::golem_routing_fn,
    pub copy_done: raw::golem_copy_done_fn,
}

/// A live hook registration: `Drop` unregisters (C joins in-flight
/// callbacks) and recovers the payload Arc. Keeping the context `Arc` alive
/// is what enforces unregister-before-free by construction.
pub struct Hooked<T> {
    handle: raw::golem_hook_handle_t,
    payload: *const T, // Arc::into_raw — recovered exactly once, on drop
    ctx: std::sync::Arc<CtxRaw>,
}

// SAFETY: the raw payload pointer is exclusively owned (Arc::into_raw
// contract) and the C handle is only consumed by golem_unregister_hooks on
// this thread; the ctx Arc clone is Send+Sync.
unsafe impl<T: Send> Send for Hooked<T> {}
// SAFETY: drop is the only mutation and `&self` methods do not exist.
unsafe impl<T: Send> Sync for Hooked<T> {}

impl<T> Drop for Hooked<T> {
    fn drop(&mut self) {
        // SAFETY: the handle came from golem_register_hooks and has not been
        // unregistered yet (drop runs exactly once). The C join guarantees
        // no callback runs after this returns — the precondition for
        // recovering the payload Arc below.
        unsafe { raw::golem_unregister_hooks(self.handle) };
        // SAFETY: `self.payload` came from Arc::into_raw and every C callback
        // has finished (join above), so this is the unique recovery.
        drop(unsafe { std::sync::Arc::from_raw(self.payload.cast::<T>() as *mut T) });
    }
}

impl<T> Hooked<T> {
    /// The raw context pointer this registration is attached to (test seam
    /// and the order-publisher path).
    pub fn ctx_ptr(&self) -> raw::golem_ctx_t {
        self.ctx.0
    }

    /// A keep-alive clone of the owned ctx Arc: dropping it runs
    /// `golem_ctx_free` (which aborts while any sched is still attached).
    /// The registration's own Arc keeps the ctx alive through the
    /// unregister-join; the clone lets the owner order the free AFTER
    /// `llama_free` (the sched destructor decrements the attach refcount).
    pub fn ctx_arc(&self) -> std::sync::Arc<CtxRaw> {
        std::sync::Arc::clone(&self.ctx)
    }
}

pub mod raw {
    // bindgen bitfield accessors trip `unnecessary_transmutes`
    #![allow(
        non_camel_case_types,
        non_snake_case,
        non_upper_case_globals,
        improper_ctypes,
        unnecessary_transmutes,
        clippy::all
    )]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

/// GPU-pinned host memory (§3.5): register/deregister an mlock'd host
/// range with the HIP runtime so `hipMemcpyAsync` from it is truly async and
/// runs at full PCIe bandwidth instead of through driver bounce buffers.
/// Only available when the substrate backend is HIP; callers gate on
/// [`substrate_backend`]. `hipHostRegisterPortable` semantics: the pinned-ness
/// is visible to all HIP contexts in the process.
#[cfg(golem_backend_hip)]
pub mod pinned {
    // hipHostRegisterPortable = 0x01 (hip_runtime_api.h; registration takes
    // no unregister flag — hipHostUnregister takes none).
    const HIP_HOST_REGISTER_PORTABLE: std::ffi::c_uint = 0x01;

    unsafe extern "C" {
        fn hipHostRegister(
            host_ptr: *mut std::ffi::c_void,
            size_bytes: usize,
            flags: std::ffi::c_uint,
        ) -> std::ffi::c_int;
        fn hipHostUnregister(host_ptr: *mut std::ffi::c_void) -> std::ffi::c_int;
    }

    /// Registers `[ptr, ptr+len)` as pinned. Errors surfaced verbatim (HIP
    /// result codes); a failed registration must abort the caller's startup —
    /// silently-degraded PCIe bandwidth is exactly what §3.5 bans.
    ///
    /// # Safety
    /// `ptr` must be a live, page-aligned host allocation of exactly `len`
    /// bytes that is not already registered.
    pub unsafe fn hip_host_register(ptr: *mut u8, len: usize) -> Result<(), std::ffi::c_int> {
        // SAFETY: caller upholds the live-unregistered-allocation invariant.
        let r = unsafe { hipHostRegister(ptr as *mut _, len, HIP_HOST_REGISTER_PORTABLE) };
        if r == 0 { Ok(()) } else { Err(r) }
    }

    /// # Safety
    /// `ptr` must be the exact pointer (and live allocation) passed to
    /// [`hip_host_register`], with all consumers of the pinned-ness finished.
    pub unsafe fn hip_host_unregister(ptr: *mut u8) -> Result<(), std::ffi::c_int> {
        // SAFETY: caller upholds the exact-pointer/no-live-consumer invariant.
        let r = unsafe { hipHostUnregister(ptr as *mut _) };
        if r == 0 { Ok(()) } else { Err(r) }
    }

    /// `(free, total)` bytes of the current HIP device (`hipMemGetInfo`);
    /// the substrate's device must be the process's current one. `None` when the HIP
    /// runtime reports an error — callers must then refuse to derive a device
    /// budget rather than guess.
    pub fn free_device_mem() -> Option<(usize, usize)> {
        unsafe extern "C" {
            fn hipMemGetInfo(free: *mut usize, total: *mut usize) -> std::ffi::c_int;
        }
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: plain out-parameters, no invariants.
        let r = unsafe { hipMemGetInfo(&mut free, &mut total) };
        if r == 0 { Some((free, total)) } else { None }
    }
}
