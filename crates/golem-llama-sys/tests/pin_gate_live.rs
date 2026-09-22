//! Live pin-gate hard gate. `build.rs` only runs the gate when cargo re-runs
//! the build script — and rerun detection is mtime-based, so a content change
//! that preserves mtimes could skip the gate. This test runs `verify_pin` on
//! the REAL vendored tree on every `cargo test` / CI run, independent of
//! cargo's rerun heuristics: a drifted tree fails CI even when the build
//! script was never re-invoked. It also asserts the gate-provenance flag so a
//! hatch-skipped build can never hide.

#![cfg(unix)] // the gate recipe is a Unix contract

use golem_llama_sys::pin_gate;

fn repo_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn vendored_tree_matches_recorded_pin() {
    let root = repo_root();
    let vendor = root.join("vendor/llama.cpp");
    let vendor_sha = root.join("vendor/VENDOR_SHA");
    if !vendor.is_dir() {
        panic!(
            "{} absent — vendor tree not materialized in this checkout",
            vendor.display()
        );
    }
    pin_gate::verify_pin(&vendor, &vendor_sha)
        .expect("vendored tree drifted (or VENDOR_SHA malformed) — re-vendor per ADR 0001");
}

#[test]
fn build_was_not_made_with_the_pin_gate_skipped() {
    assert!(
        !golem_llama_sys::pin_gate_skipped(),
        "this build was produced with GOLEM_LLAMA_SYS_SKIP_PIN_GATE=1 — \
         rebuild without the hatch before shipping"
    );
}
