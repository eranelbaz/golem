//! The golem-hooks ABI lives in the vendored tree (P1); bindgen must resolve it there.
#[test]
fn hooks_header_resolves_from_vendor_tree() {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let vendored =
        std::path::Path::new(manifest).join("../../vendor/llama.cpp/ggml/include/golem-hooks.h");
    assert!(
        vendored.exists(),
        "expected vendor header at {}",
        vendored.display()
    );
    let text = std::fs::read_to_string(&vendored).expect("read vendor header");
    assert!(
        text.contains("golem_tensor_expert_count"),
        "metadata API missing"
    );
    assert!(
        text.contains("golem_register_hooks"),
        "hook registration missing"
    );
    assert!(
        text.contains("#define GOLEM_HOOKS_ABI_VERSION 14"),
        "hook ABI must be v14 (v12 slab alloc failure + v13 ensure failure + v14 resident bytes)"
    );
    for sym in [
        "golem_evict_abandon_fn",
        "golem_ctx_create",
        "golem_ctx_free",
        "golem_routing_fn",
        "golem_policy_order_push",
        "llama_golem_model_attach",
    ] {
        assert!(text.contains(sym), "v5 surface missing: {sym}");
    }
}
