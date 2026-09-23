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
    assert_thin_fn_ptr::<raw::golem_policy_query_fn>();
    assert_thin_fn_ptr::<raw::golem_notify_access_fn>();
}
