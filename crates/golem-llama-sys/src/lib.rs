//! Raw bindings to the vendored llama.cpp fork.
#![allow(unsafe_code, missing_docs)]
include!(concat!(env!("OUT_DIR"), "/backend_info.rs"));

/// Shared with `build.rs` via `#[path]`; conformance-tested against the
/// shell recipe in `tests/pin_gate_conformance.rs`.
#[path = "../pin_gate.rs"]
pub mod pin_gate;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Hip,
}

pub fn substrate_backend() -> Backend {
    BACKEND
}

pub fn backend_env_input() -> &'static str {
    BACKEND_INPUT
}

/// True if the gate never ran (`GOLEM_LLAMA_SYS_SKIP_PIN_GATE=1`);
/// assert `false` for shipped builds.
pub fn pin_gate_skipped() -> bool {
    option_env!("GOLEM_LLAMA_PIN_GATE_SKIPPED") == Some("1")
}

/// Raw generated bindings over `llama.h` + `headers/golem-hooks.h`.
/// `golem_register_hooks` is declared in the header but undefined in the
/// substrate at this pin — referencing it fails at link time.
pub mod raw {
    // bindgen output; its bitfield accessors trip rustc's `unnecessary_transmutes`.
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
