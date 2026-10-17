//! CPU expert math leases (design §1 O1, ABI v16): the engine-registered
//! `cpu_lease` hook leases only pool-resident routed units, one lease per
//! unit, and the next `copy_done` releases them all. Drives the trampolines
//! directly over the tiny-MoE fixture on the CPU backend.
#![allow(unsafe_code)]
mod common;

use common::expert_tensor as xt;
use common::tiny_moe_path;
use golem_engine::config::EngineConfig;
use golem_engine::p3_glue;
use std::ffi::c_void;

#[test]
fn cpu_lease_takes_resident_units_and_copy_done_releases_them() {
    let hooks = p3_glue::enable(&EngineConfig {
        model_path: tiny_moe_path(),
        ram_experts_gib: 1,
        device_cache_experts: 4,
        ..EngineConfig::default()
    })
    .expect("enable");
    let payload = hooks.payload_ptr();
    let (_ctx, gate) = xt::gate_tensor();

    // Page expert 3 in through the demand path; 5 stays cold.
    let (kinds, miss_ids) = ([1, 2, 3], [3, 3, 3]);
    let mut planes: [*const u8; 3] = [std::ptr::null(); 3];
    let rc = p3_glue::ensure_host_resident(
        payload,
        gate.cast(),
        kinds.as_ptr(),
        miss_ids.as_ptr(),
        3,
        planes.as_mut_ptr().cast(),
    );
    assert_eq!(rc, 0, "ensure paged in");
    p3_glue::copy_done(payload);

    let ids: [i32; 2] = [5, 3];
    let mut out_ids = [-1i32; 2];
    let mut out_ptrs: [*const c_void; 6] = [std::ptr::null(); 6];
    let taken = p3_glue::cpu_lease(
        payload,
        xt::BLK,
        ids.as_ptr(),
        2,
        1.0,
        out_ids.as_mut_ptr(),
        out_ptrs.as_mut_ptr(),
    );
    assert_eq!(taken, 1, "only the resident unit is taken");
    assert_eq!(out_ids[0], 3);
    assert!(out_ptrs[..3].iter().all(|p| !p.is_null()), "{out_ptrs:?}");
    assert_eq!(hooks.cpu_leases(), 1);

    // Outside the expert blk range: nothing taken, nothing leased.
    let none = p3_glue::cpu_lease(
        payload,
        xt::BLK - 1,
        ids.as_ptr(),
        2,
        1.0,
        out_ids.as_mut_ptr(),
        out_ptrs.as_mut_ptr(),
    );
    assert_eq!(none, 0);
    assert_eq!(hooks.cpu_leases(), 1);

    p3_glue::copy_done(payload);
    assert_eq!(hooks.cpu_leases(), 0, "copy_done releases the CPU leases");
}
