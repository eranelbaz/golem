//! Raw bindings to the vendored llama.cpp fork (tech design section 3).
#![allow(unsafe_code, missing_docs)] // FFI crate: the one sanctioned unsafe island (workspace lints stay intact for all other crates)
include!(concat!(env!("OUT_DIR"), "/backend_info.rs"));

/// ADR 0001 vendor pin gate — the canonical `cs()` content-hash recipe, ported
/// byte-faithfully to Rust (`pin_gate.rs` at the crate root, `#[path]`-included
/// here AND by `build.rs`, so the build gate and this export share one source).
/// `tests/pin_gate_conformance.rs` byte-compares it against the shell recipe.
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

/// Backend this build of `golem-llama-sys` carries (resolved by `build.rs`).
pub fn substrate_backend() -> Backend {
    BACKEND
}

/// Raw `GOLEM_LLAMA_SYS_BACKEND` input build.rs resolved (auto|cpu|hip).
pub fn backend_env_input() -> &'static str {
    BACKEND_INPUT
}

/// Whether this build was produced with `GOLEM_LLAMA_SYS_SKIP_PIN_GATE=1`
/// (the ADR 0001 escape hatch). Always `false` for properly built artifacts —
/// assert this in CI/shipping checks; a `true` here means the
/// vendored-substrate integrity gate never ran for this build.
pub fn pin_gate_skipped() -> bool {
    // Stamped by build.rs via `cargo:rustc-env` (after the gate ran).
    option_env!("GOLEM_LLAMA_PIN_GATE_SKIPPED") == Some("1")
}

/// Raw generated bindings over `llama.h` + the `golem-hooks.h` ABI stub
/// (bindgen in `build.rs`, step (e); allowlisted to `llama_*`/`golem_*`
/// symbols and `LLAMA_*` vars). Hook typedefs come through as
/// `Option<unsafe extern "C" fn …>`; enum values are accessed via their
/// `::default()`/associated consts under the non_exhaustive Rust style.
///
/// NOTE: `golem_register_hooks` is declared in `headers/golem-hooks.h` but
/// deliberately undefined in the substrate at this pin (the P1 patch lands
/// in M2.1) — linking a caller that takes its address fails at link time
/// until then. All other `golem_*` surface is type/typedef only.
pub mod raw {
    // Generated code (bindgen, see build.rs): style + clippy exemptions are
    // scoped to this module only. `clippy::all` silences clippy over the
    // generated items; `unnecessary_transmutes` is the rustc-upstreamed lint
    // (toolchain 1.98: warn-by-default, NOT in clippy::all) that bindgen's
    // bitfield accessors trip.
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
