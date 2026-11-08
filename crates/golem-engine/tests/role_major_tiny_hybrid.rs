//! Role-major expert page-in (§3.4 P3) on the HIP substrate through the
//! engine-registered hooks and the scheduler path (tiny-hybrid: 2 MoE layers
//! × 8 experts, top-2, K = top-k slab slots, 1 GiB pool, prefetch off). The
//! prefill streams (§7.6: stream chunks never touch the pool), so decode
//! starts on a cold pool and its misses are read from disk.
//!
//! Criteria: greedy tokens with `role_major_pagein` equal those without it
//! over `STEPS` decode steps, and the flag-on run split at least one unit
//! (`split_units` > 0 — the path ran). The flag-off run never splits.
#![cfg(feature = "hip-tests")]
mod common;

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use golem_engine::config::EngineConfig;
use golem_engine::{ChatTurn, SamplingParams, Scheduler, Session, SlotEvent, SlotJob, TurnRequest};

const STEPS: u32 = 32;
const K: u32 = xtask::tiny_hybrid::dims::N_EXPERT_USED;

fn config(role_major_pagein: bool) -> EngineConfig {
    let off = common::offload::offload_config();
    EngineConfig {
        model_path: xtask::tiny_hybrid::cached_fixture(),
        n_gpu_layers: off.n_gpu_layers,
        cpu_moe_layers: off.cpu_moe_layers,
        ram_experts_gib: 1,
        device_cache_experts: K,
        n_predict: STEPS,
        ignore_eos: true,
        prefill_stream: true,
        prefill_ubatch: 64,
        prefill_stream_min: 1,
        n_ctx: 512,
        role_major_pagein,
        ..EngineConfig::default()
    }
}

/// One greedy turn through the scheduler; returns the argmax token of every
/// sampled logits row and the session's `split_units`.
fn run(cfg: &EngineConfig) -> (Vec<usize>, u64) {
    let mut sch = Scheduler::new(Session::new(cfg).expect("pooled hip session")).unwrap();
    assert!(
        sch.session().prefill_streaming(),
        "precondition: streamed prefill leaves the pool cold"
    );
    sch.session().record_logits_for_test();
    let msg = [ChatTurn {
        role: "user",
        content: "one two three four five six seven eight",
    }];
    let rendered = sch
        .session()
        .render(&TurnRequest::new(&msg))
        .expect("render");
    let tokens = sch.tokenize(&rendered.prompt).expect("tokenize");
    sch.admit(SlotJob {
        tokens,
        sampling: SamplingParams::default(),
        max_tokens: Some(STEPS),
        stop: Vec::new(),
        cancel: Arc::new(AtomicBool::new(false)),
        rendered: Some(rendered),
    })
    .ok()
    .expect("free slot");
    let mut done = None;
    loop {
        let mut failed = None;
        let active = sch
            .step(&mut |_, e| match e {
                SlotEvent::Failed(e) => failed = Some(e.to_string()),
                SlotEvent::Done(o) => done = Some(o.completion_tokens),
                SlotEvent::Text(_) => {}
            })
            .expect("step");
        assert!(failed.is_none(), "{failed:?}");
        if !active {
            break;
        }
    }
    assert_eq!(done, Some(STEPS), "precondition: {STEPS} tokens");
    let s = sch.session();
    assert!(s.expert_mul_mat_id_on_gpu() > 0, "expert MUL_MAT_ID on GPU");
    assert_eq!(
        s.expert_mul_mat_id_off_gpu(),
        0,
        "no silent CPU expert math"
    );
    let n_vocab = xtask::tiny_hybrid::dims::N_VOCAB as usize;
    let argmax = s
        .take_logits_for_test()
        .chunks(n_vocab)
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| i)
                .unwrap()
        })
        .collect();
    let split = s
        .p3_stats()
        .expect("pooled")
        .split_units
        .load(Ordering::Relaxed);
    (argmax, split)
}

#[test]
fn role_major_matches_single_phase_and_splits() {
    common::offload::require_process_per_test();
    let (off, off_split) = run(&config(false));
    assert_eq!(off.len(), STEPS as usize, "one row per sampled token");
    assert_eq!(off_split, 0, "flag off never splits");
    let (on, on_split) = run(&config(true));
    assert_eq!(on, off, "greedy tokens differ with role-major page-in");
    assert!(on_split > 0, "role-major path never split a unit");
}
