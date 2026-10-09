#![allow(unsafe_code, missing_docs)]

use golem_llama_sys::raw;

#[test]
fn engine_surface_symbols_exist() {
    let fns: &[(&str, *const ())] = &[
        (
            "llama_model_load_from_file",
            raw::llama_model_load_from_file as *const (),
        ),
        (
            "llama_init_from_model",
            raw::llama_init_from_model as *const (),
        ),
        ("llama_decode", raw::llama_decode as *const ()),
        ("llama_batch_get_one", raw::llama_batch_get_one as *const ()),
        (
            "llama_sampler_chain_init",
            raw::llama_sampler_chain_init as *const (),
        ),
        ("llama_token_eos", raw::llama_token_eos as *const ()),
    ];
    for (name, f) in fns {
        assert!(!f.is_null(), "engine surface symbol {name} unresolved");
    }
}

fn assert_thin_fn_ptr<T>() {
    assert_eq!(
        std::mem::size_of::<T>(),
        std::mem::size_of::<usize>(),
        "hook typedef must stay a thin C fn pointer (Option niche), not a fat type"
    );
    assert!(
        !std::mem::needs_drop::<T>(),
        "hook typedef must not carry a drop path"
    );
}

#[test]
fn hook_abi_types_generated() {
    assert_thin_fn_ptr::<raw::golem_ensure_host_resident_fn>();
    assert_thin_fn_ptr::<raw::golem_routing_fn>();
    assert_thin_fn_ptr::<raw::golem_copy_done_fn>();
    assert_thin_fn_ptr::<raw::golem_predict_fn>();
    assert_thin_fn_ptr::<raw::golem_evict_abandon_fn>();
    // ABI v7: the per-model ctx surface, the P5 predict seam and the
    // exclusive-residency evict_batch hook are generated, and the header
    // constant the bindings came from matches the compiled substrate.
    // ABI v9 adds the abandoned-D2H hook (parked evictions whose plane
    // copies were not all issued must be reported, not committed).
    assert_eq!(raw::GOLEM_HOOKS_ABI_VERSION, 12);
    assert_eq!(golem_llama_sys::hooks_abi_version(), 12);
    let fns: &[(&str, *const ())] = &[
        ("golem_set_predict", raw::golem_set_predict as *const ()),
        (
            "golem_predict_lookahead",
            raw::golem_predict_lookahead as *const (),
        ),
        ("golem_ctx_create", raw::golem_ctx_create as *const ()),
        ("golem_ctx_free", raw::golem_ctx_free as *const ()),
        (
            "golem_register_hooks",
            raw::golem_register_hooks as *const (),
        ),
        (
            "golem_unregister_hooks",
            raw::golem_unregister_hooks as *const (),
        ),
        ("golem_policy_order", raw::golem_policy_order as *const ()),
        (
            "golem_policy_order_push",
            raw::golem_policy_order_push as *const (),
        ),
        (
            "golem_ctx_generation",
            raw::golem_ctx_generation as *const (),
        ),
        (
            "llama_golem_model_attach",
            raw::llama_golem_model_attach as *const (),
        ),
        (
            "llama_golem_context_detach",
            raw::llama_golem_context_detach as *const (),
        ),
    ];
    for (name, f) in fns {
        assert!(!f.is_null(), "hook surface symbol {name} unresolved");
    }
}

#[test]
fn batched_ensure_signature() {
    // The batched ensure carries the miss list (kinds + ids) and writes the
    // pool pointers — the exact v5 shape.
    extern "C" fn f(
        _payload: *mut std::ffi::c_void,
        _tensor: *const std::ffi::c_void,
        _kinds: *const i32,
        _ids: *const i32,
        _n: i32,
        _out: *mut *const std::ffi::c_void,
    ) {
    }
    let g: raw::golem_ensure_host_resident_fn = Some(f);
    assert_eq!(std::mem::size_of::<raw::golem_ensure_host_resident_fn>(), 8);
    let _ = g;
}
