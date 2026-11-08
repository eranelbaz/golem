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
    assert_eq!(raw::GOLEM_HOOKS_ABI_VERSION, 22);
    assert_eq!(golem_llama_sys::hooks_abi_version(), 22);
    // v20: device-side landed wait; a NULL ctx is refused / reports off.
    // SAFETY: NULL ctx is the documented refusal path of each entry point.
    unsafe {
        assert_eq!(
            raw::golem_set_device_wait(std::ptr::null_mut(), std::ptr::null_mut(), 1, 8, 0),
            -1
        );
        assert_eq!(raw::golem_device_wait_mode(std::ptr::null_mut()), 0);
        assert_eq!(raw::golem_device_wait_serve(std::ptr::null_mut(), 0, 1), -1);
        let mut req: *const u32 = std::ptr::null();
        let mut n = 0i32;
        assert_eq!(
            raw::golem_device_wait_ring(std::ptr::null_mut(), &mut req, &mut n),
            0
        );
        assert!(req.is_null() && n == 0);
    }
    // v19: host-sync timing of the MoE ids readbacks; a NULL ctx is refused.
    let (mut s, mut g, mut gm, mut n) = (0.0, 0.0, 0.0, 0u64);
    // SAFETY: a NULL ctx is the documented refusal path; the out-params are
    // live locals and untouched on refusal.
    let rc = unsafe {
        raw::golem_take_sync_gap_stats(std::ptr::null_mut(), &mut s, &mut g, &mut gm, &mut n)
    };
    assert_eq!((rc, n), (0, 0));
    // v18: VRAM victim ring setter + stats.
    let _: unsafe extern "C" fn(raw::golem_ctx_t, i32, i32) -> i32 = raw::golem_set_victim_ring;
    let _: unsafe extern "C" fn(raw::golem_ctx_t, *mut u64, *mut u64, *mut f64, *mut u64) -> i32 =
        raw::golem_take_victim_ring_stats;
    let (mut u, mut f, mut ms, mut b) = (0u64, 0u64, 0.0f64, 0u64);
    // SAFETY: NULL ctx is the documented refusal; the out-params are live
    // locals and untouched on refusal.
    let rc = unsafe {
        raw::golem_take_victim_ring_stats(std::ptr::null_mut(), &mut u, &mut f, &mut ms, &mut b)
    };
    assert_eq!((rc, u, f, b), (0, 0, 0, 0));
    // v17: ensure reports RAM hits early through `hits_ready`.
    assert_thin_fn_ptr::<raw::golem_hits_ready_fn>();
    // v16: CPU expert math surface.
    assert_thin_fn_ptr::<raw::golem_cpu_lease_fn>();
    // v15: stream source return codes.
    assert_eq!(
        (
            raw::GOLEM_STREAM_SRC_OK,
            raw::GOLEM_STREAM_SRC_CANCEL,
            raw::GOLEM_STREAM_SRC_STORAGE_LOST
        ),
        (0, 1, 2)
    );
    let fns: &[(&str, *const ())] = &[
        ("golem_set_predict", raw::golem_set_predict as *const ()),
        (
            "golem_predict_lookahead",
            raw::golem_predict_lookahead as *const (),
        ),
        ("golem_ctx_create", raw::golem_ctx_create as *const ()),
        ("golem_ctx_free", raw::golem_ctx_free as *const ()),
        (
            "golem_set_cpu_expert_math",
            raw::golem_set_cpu_expert_math as *const (),
        ),
        (
            "golem_take_cpu_expert_stats",
            raw::golem_take_cpu_expert_stats as *const (),
        ),
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
    // pool pointers — the v5 shape; v13 returns a failure code; v17 adds the
    // `hits_ready` callback + its user data.
    extern "C" fn f(
        _payload: *mut std::ffi::c_void,
        _tensor: *const std::ffi::c_void,
        _kinds: *const i32,
        _ids: *const i32,
        _n: i32,
        _out: *mut *const std::ffi::c_void,
        _hits_ready: raw::golem_hits_ready_fn,
        _hits_ud: *mut std::ffi::c_void,
    ) -> i32 {
        0
    }
    let g: raw::golem_ensure_host_resident_fn = Some(f);
    assert_eq!(std::mem::size_of::<raw::golem_ensure_host_resident_fn>(), 8);
    let _ = g;
}
