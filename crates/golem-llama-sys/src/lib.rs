//! Raw bindings to the vendored llama.cpp fork.
#![allow(unsafe_code, missing_docs)]
include!(concat!(env!("OUT_DIR"), "/backend_info.rs"));

/// Included (via `#[path]`) by `build.rs` too, so the build gate and this
/// export share one implementation; conformance-tested against the canonical
/// shell recipe in `tests/pin_gate_conformance.rs`.
#[path = "../pin_gate.rs"]
pub mod pin_gate;

/// Compute backend the vendored substrate was built with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// CPU reference build (CI, parity reference).
    Cpu,
    /// HIP/ROCm build for gfx1100 (hardware).
    Hip,
}

/// Backend this build of `golem-llama-sys` carries.
pub fn substrate_backend() -> Backend {
    BACKEND
}

/// Raw `GOLEM_LLAMA_SYS_BACKEND` input build.rs resolved (auto|cpu|hip).
pub fn backend_env_input() -> &'static str {
    BACKEND_INPUT
}

/// Whether this build was made with the `GOLEM_LLAMA_SYS_SKIP_PIN_GATE=1`
/// escape hatch — i.e. the vendored-substrate integrity gate never ran.
/// Assert `false` for shipped builds.
pub fn pin_gate_skipped() -> bool {
    option_env!("GOLEM_LLAMA_PIN_GATE_SKIPPED") == Some("1")
}

/// Raw generated bindings over `llama.h` + `headers/golem-hooks.h`.
///
/// `golem_register_hooks` is declared in the header but deliberately
/// undefined in the substrate at this pin — taking its address fails at
/// link time until then.
pub mod raw {
    // Generated code (bindgen); `unnecessary_transmutes` is warn-by-default
    // in rustc (not in clippy::all) and bindgen's bitfield accessors trip it.
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
