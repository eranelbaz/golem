//! Bindings must expose the exact symbols the engine layer needs.
#![allow(unsafe_code)]

use golem_llama_sys::raw;

#[test]
fn engine_surface_symbols_exist() {
    // Taking addresses of the generated externs fails at *link time* if any
    // is absent from the substrate — no runtime assert is needed.
    let fns: &[*const ()] = &[
        raw::llama_model_load_from_file as *const (),
        raw::llama_init_from_model as *const (),
        raw::llama_decode as *const (),
        raw::llama_batch_get_one as *const (),
        raw::llama_sampler_chain_init as *const (),
        raw::llama_token_eos as *const (),
    ];
    let _ = fns.len();
}

#[test]
fn hook_abi_types_generated() {
    // Bindgen wraps the hook typedef as Option<unsafe extern "C" fn …>, so
    // the annotation must be the bare alias (not Option<Option<…>>).
    let _never_called: raw::golem_ensure_host_resident_fn = None;
    let _ = _never_called;
}
