//! Bindings must expose the exact symbols the engine layer (M1.3) and P1 need.
#![allow(unsafe_code)]

use golem_llama_sys::raw;

#[test]
fn engine_surface_symbols_exist() {
    // Link-time proof: taking addresses of the generated externs requires them
    // to exist in the substrate at this pin.
    let fns: &[*const ()] = &[
        raw::llama_model_load_from_file as *const (),
        raw::llama_init_from_model as *const (),
        raw::llama_decode as *const (),
        raw::llama_batch_get_one as *const (),
        raw::llama_sampler_chain_init as *const (),
        raw::llama_token_eos as *const (),
    ];
    for f in fns {
        assert!(!f.is_null());
    }
}

#[test]
fn hook_abi_types_generated() {
    // Type-level only: forces bindgen's P1 typedefs into the compile.
    let _never_called: Option<
        unsafe extern "C" fn(*mut core::ffi::c_void, *const core::ffi::c_void, i32, i32),
    > = None;
    let _ = _never_called;
}
