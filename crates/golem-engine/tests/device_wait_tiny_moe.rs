//! Device-side landed wait (ABI v19), through the engine-registered
//! hooks on the GPU split: the decode MoE layers are served off the decode
//! thread (no ids readback), greedy output is byte-identical to the host-sync
//! path in both the stream-wait mode and the wait-kernel fallback, every
//! expert MUL_MAT_ID stays on the GPU, and a storage loss fails the compute
//! instead of hanging.
#![cfg(feature = "hip-tests")]
#![allow(unsafe_code)] // FFI test seam (slab failure injection, cache reset)
mod common;

use common::offload::{offload_config, require_process_per_test};
use golem_engine::config::{DeviceWait, EngineConfig, PrefetchMode};
use golem_engine::session::{ChatTurn, Session, expert_placement};
use golem_engine::slot::TurnRequest;
use xtask::tiny_moe::dims::{N_EXPERT, N_EXPERT_USED};

const PROMPT: &str = "the quick brown fox";

fn config(device_wait: DeviceWait) -> EngineConfig {
    EngineConfig {
        ram_experts_gib: 1,
        // Below the expert count: slots are recycled, so the slab step's
        // eviction and re-copy run on the device-wait path too.
        device_cache_experts: 4,
        async_h2d: true,
        device_wait,
        n_predict: 24,
        ..offload_config()
    }
}

fn greedy(session: &mut Session) -> String {
    let mut out = String::new();
    session
        .chat_turn(
            &[ChatTurn {
                role: "user",
                content: PROMPT,
            }],
            &mut |p| out.push_str(p),
        )
        .expect("decode");
    out
}

/// One session's output and device-wait counters (mode, served, computes).
/// Also checks the fixture's device budget: K = 4 slots of an 8-expert
/// layer, so each plane's slab ((K + 1) experts) is smaller than one expert
/// tensor (8 experts) and nothing is device-resident whole (design §12.2).
fn run(cfg: &EngineConfig) -> (String, i32, u64, u64) {
    let mut s = Session::new(cfg).expect("pooled offload session");
    let out = greedy(&mut s);
    let facts = golem_plan::ModelFacts::from_gguf(&cfg.model_path).expect("fixture facts");
    let (slab_res, _) = s.device_cache_resident_bytes().expect("pool on");
    let per_slot = facts.slab_bytes_per_expert();
    let k = cfg.device_cache_experts as u64;
    assert!(
        (k + 1) < u64::from(facts.n_expert)
            && slab_res > 0
            && slab_res <= (k + 1) * per_slot + (1 << 20)
            && slab_res < facts.expert_bytes_total,
        "device budget must stay below one expert tensor per plane: slab {slab_res} B, \
         {} B per slot, {} B of expert tensors",
        per_slot,
        facts.expert_bytes_total
    );
    let h = s.p3_hooks().expect("pool on");
    let (served, computes) = h.device_wait_stats();
    (out, h.device_wait_mode(), served, computes)
}

#[test]
fn device_wait_matches_host_sync_and_skips_the_ids_readback() {
    require_process_per_test();
    let before = expert_placement();
    let (off, mode_off, served_off, _) = run(&config(DeviceWait::Off));
    assert!(!off.is_empty());
    assert_eq!((mode_off, served_off), (0, 0), "off: no device-wait serve");

    for (dw, want_mode) in [(DeviceWait::On, 1), (DeviceWait::Kernel, 2)] {
        let (out, mode, served, computes) = run(&config(dw));
        assert_eq!(
            mode, want_mode,
            "{dw:?}: gfx1100 supports stream wait-value; kernel is forced"
        );
        assert_eq!(
            out, off,
            "{dw:?}: greedy output must match the host-sync path"
        );
        assert!(
            served > 0 && computes > 0,
            "{dw:?}: decode MoE layers must be served off the decode thread \
             (served {served}, computes without an ids readback {computes})"
        );
    }
    // Same order reversed: on after off after kernel stays identical.
    let (again, ..) = run(&config(DeviceWait::Off));
    assert_eq!(again, off, "host-sync output is stable across sessions");

    let after = expert_placement();
    assert!(
        after.on_cpu == before.on_cpu && after.on_gpu > before.on_gpu,
        "every expert MUL_MAT_ID must be GPU-assigned: {before:?} -> {after:?}"
    );
}

/// P5 predictions ride the ring (the predict hook runs in the serve): the
/// glm5next tiny-hybrid fixture builds prediction nodes (tiny-moe's glm4moe
/// graph does not); output is unchanged and predictions are still scored.
#[test]
fn device_wait_with_prefetch_read_keeps_output_and_predictions() {
    require_process_per_test();
    let with = |dw| EngineConfig {
        model_path: xtask::tiny_hybrid::cached_fixture(),
        device_cache_experts: xtask::tiny_hybrid::dims::N_EXPERT_USED * 2,
        prefetch: PrefetchMode::Read,
        ..config(dw)
    };
    let mut s = Session::new(&with(DeviceWait::Off)).expect("session");
    let off = greedy(&mut s);
    let p_off = s.prefetch_stats().expect("prefetch on");
    assert!(
        p_off.predicted_routed > 0,
        "precondition: the fixture predicts: {p_off}"
    );
    drop(s);
    let mut s = Session::new(&with(DeviceWait::On)).expect("session");
    let out = greedy(&mut s);
    assert_eq!(out, off);
    let p = s.prefetch_stats().expect("prefetch on");
    let (served, _) = s.p3_hooks().unwrap().device_wait_stats();
    assert!(served > 0, "decode MoE layers served off-thread");
    assert_eq!(
        (p.predicted_routed, p.predicted_hits),
        (p_off.predicted_routed, p_off.predicted_hits),
        "the served predictions score exactly like the host-sync ones"
    );
}

/// The tightest legal device budget (K = top-k, every step recycles the
/// whole slab) still serves off-thread with unchanged output. A budget below
/// top-k is unservable by construction (`golem_dw_setup` returns 0 and the
/// host-sync path's slab step aborts, as without device wait).
#[test]
fn device_budget_at_top_k_serves_with_unchanged_output() {
    require_process_per_test();
    let tight = |dw| EngineConfig {
        device_cache_experts: N_EXPERT_USED,
        ..config(dw)
    };
    let (off, ..) = run(&tight(DeviceWait::Off));
    let (on, mode, served, _) = run(&tight(DeviceWait::On));
    assert_eq!(on, off);
    assert_eq!(mode, 1);
    assert!(served > 0, "K == top-k still serves");
}

/// A page-in failure inside a served layer fails the compute (llama_decode
/// -3, ABI v13) instead of hanging the device wait: the serve lands the words,
/// the decode thread fails the step, and no lease outlives the turn. (The
/// session's next turn is refused by the pre-existing prefill self-check on
/// both paths, so recovery is not part of this contract.)
#[test]
fn page_in_failure_in_a_served_layer_fails_the_turn() {
    require_process_per_test();
    let mut s = Session::new(&config(DeviceWait::On)).expect("session");
    s.p3_hooks().unwrap().fail_decode_page_in_for_test();
    let mut out = String::new();
    let r = s.chat_turn(
        &[ChatTurn {
            role: "user",
            content: PROMPT,
        }],
        &mut |p| out.push_str(p),
    );
    assert!(
        matches!(r, Err(golem_engine::EngineError::Decode(-3))),
        "a failed served page-in must fail the step with -3, got {r:?}"
    );
    let (served, _) = s.p3_hooks().unwrap().device_wait_stats();
    assert!(served > 0, "the failure came from a served layer");
    assert_eq!(s.p3_hooks().unwrap().pool_batch_leases(), 0);
}

/// A slab reallocation mid-compute (K lowered after a failed device
/// allocation) with earlier layers already published: the setup quiesces
/// before the reallocating prepare instead of holding the slot-table lock
/// across a device sync that waits on an unserved layer (no 120 s abort),
/// and the degraded session keeps serving off-thread with output unchanged.
#[test]
fn slab_realloc_after_degrade_serves_without_deadlock() {
    require_process_per_test();
    let cfg = EngineConfig {
        device_cache_experts: N_EXPERT,
        ..config(DeviceWait::On)
    };
    let msgs = [ChatTurn {
        role: "user",
        content: PROMPT,
    }];
    let mut s = Session::new(&cfg).expect("session");
    let ctx = s.p3_ctx_ptr().expect("pooled ⇒ live ctx");
    // SAFETY: FFI on the session's live ctx between turns: the next compute
    // reallocates every slab and the injected failure degrades K.
    unsafe {
        golem_llama_sys::raw::golem_reset_device_cache(ctx);
        golem_llama_sys::raw::golem_inject_slab_alloc_failure(ctx, 1);
    }
    let err = s
        .turn(&TurnRequest::new(&msgs), &mut |_| {})
        .expect_err("injected slab failure fails the turn");
    assert!(
        matches!(err, golem_engine::EngineError::DeviceCacheDegraded(_)),
        "{err}"
    );
    let (served_before, _) = s.p3_hooks().unwrap().device_wait_stats();
    let out = s
        .turn(&TurnRequest::new(&msgs), &mut |_| {})
        .expect("degraded session keeps serving");
    assert!(out.completion_tokens > 0);
    let (served, _) = s.p3_hooks().unwrap().device_wait_stats();
    assert!(
        served > served_before,
        "the reallocated layers are served off-thread"
    );
}

/// Device wait needs the copy stream: without async_h2d it is refused at load.
#[test]
fn device_wait_without_async_h2d_is_refused() {
    let cfg = EngineConfig {
        async_h2d: false,
        ..config(DeviceWait::On)
    };
    let e = Session::new(&cfg).err().expect("refused").to_string();
    assert!(e.contains("device_wait"), "{e}");
}
