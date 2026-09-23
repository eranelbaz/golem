//! Guards cargo's mtime rerun hole: gate runs on every `cargo test`.

#![cfg(unix)]

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
        .expect("vendored tree drifted (or VENDOR_SHA malformed) — re-vendor per scripts/build-substrate.sh");
}

#[test]
fn build_was_not_made_with_the_pin_gate_skipped() {
    assert!(
        !golem_llama_sys::pin_gate_skipped(),
        "this build was produced with GOLEM_LLAMA_SYS_SKIP_PIN_GATE=1 — \
         rebuild without the hatch before shipping"
    );
}
