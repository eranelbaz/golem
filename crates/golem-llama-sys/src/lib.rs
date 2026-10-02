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

    /// `(free, total)` device bytes (`hipMemGetInfo`). `None` when the HIP
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
