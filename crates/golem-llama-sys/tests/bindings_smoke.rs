//! Bindings must expose the exact symbols the engine layer (M1.3) and P1 need.
#![allow(unsafe_code)]

use golem_llama_sys::raw;

#[test]
fn engine_surface_symbols_exist() {
    // Link-time proof: taking addresses of the generated externs requires
    // them to exist in the substrate at this pin — the failure mode is a
    // *link error*, so no runtime assert is possible (or needed) here.
    let fns: &[*const ()] = &[
        raw::llama_model_load_from_file as *const (),
        raw::llama_init_from_model as *const (),
        raw::llama_decode as *const (),
        raw::llama_batch_get_one as *const (),
        raw::llama_sampler_chain_init as *const (),
        raw::llama_token_eos as *const (),
    ];
    let _ = fns.len(); // touching the array keeps every cast intentional
}

#[test]
fn hook_abi_types_generated() {
    // Type-level only: names bindgen's generated P1 typedef so it must exist
    // in `raw` at this pin. Bindgen emits the typedef already wrapped as
    // Option<unsafe extern "C" fn …>, so the annotation is the bare alias —
    // an extra Option would be Option<Option<…>> and misstate the ABI.
    let _never_called: raw::golem_ensure_host_resident_fn = None;
    let _ = _never_called;
}
