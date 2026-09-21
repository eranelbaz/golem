//! Raw bindings to the vendored llama.cpp fork (tech design section 3).
#![allow(unsafe_code, missing_docs)] // FFI crate: the one sanctioned unsafe island (workspace lints stay intact for all other crates)
include!(concat!(env!("OUT_DIR"), "/backend_info.rs"));

/// Compute backend the vendored substrate was built with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// CPU reference build (CI, parity reference).
    Cpu,
    /// HIP/ROCm build for gfx1100 (hardware).
    Hip,
}

/// Backend this build of `golem-llama-sys` carries (resolved by `build.rs`).
pub fn substrate_backend() -> Backend {
    BACKEND
}

/// Raw `GOLEM_LLAMA_SYS_BACKEND` input build.rs resolved (auto|cpu|hip).
pub fn backend_env_input() -> &'static str {
    BACKEND_INPUT
}
