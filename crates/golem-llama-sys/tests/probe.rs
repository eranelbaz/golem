//! Temporary: proves the substrate links and exports real symbols. Deleted in Task 3.
#![allow(unsafe_code)]

// References the lib crate so its build-script `-l` directives (carried in rlib
// metadata) participate in this test binary's link: rustc only emits an extern
// crate's native libs when the crate is actually referenced. The directive set
// itself is proven by `backend_detect.rs` linking too — this test additionally
// resolves a real substrate symbol below.
#[test]
fn backend_crate_resolves() {
    match golem_llama_sys::backend_env_input() {
        "cpu" | "hip" | "auto" => {}
        other => panic!("build.rs recorded unexpected env input {other:?}"),
    }
}

unsafe extern "C" {
    fn llama_max_devices() -> u64; // header says size_t; with size_t_is_usize(false) generated bindings use u64 — match that convention (ABI-identical to usize on LP64); link-only probe, returns a hardcoded constant
}
#[test]
fn links_substrate() {
    let n = unsafe { llama_max_devices() };
    assert!(n > 0, "substrate linked but returned nonsense: {n}");
}
