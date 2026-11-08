//! §11 expert tiers on the HIP substrate through the engine-registered hooks
//! (tiny-hybrid: 2 MoE layers × 8 experts, top-2). K = top-k slab slots,
//! prefetch off, inclusive residency, a 1 GiB pool that never evicts
//! (2 × 8 units ≈ 2.4 MB). The tier leg streams its prefill (§7.6: stream
//! chunks never touch the pool), so the first decode steps find a cold pool
//! and must read from disk.
//!
//! Each tier's forcing condition is checked against residency read before
//! the step, and every tier must be observed within `STEPS` decode steps
//! (a missed precondition FAILS, it never skips):
//! - `disk`: the unit was not pool-resident before the step;
//! - `ram`: it was pool-resident and its slab slot was not valid;
//! - `vram`: its slab slot (all three planes) was valid before the step,
//!   so no plane missed and ensure never saw it; pool-resident (inclusive);
//! - `cpu`: CPU expert math leased it (separate session, stats > 0).
//!
//! Slot validity is read through the substrate's test getter before and
//! after each step. The slab step assigns and validates every routed
//! unit's slot before its H2D, and only routed units with a missing plane
//! reach ensure, so: `vram` ⇔ valid before the step; `ram`/`disk` ⇒ not
//! valid before; every slab-served unit is valid after.
//!
//! Every closed row also has `vram_resident` ≤ K, `ram_resident` equal to
//! Σ`is_resident` over the layer, and Σ`vram_resident` > 0 exactly when
//! the session holds slab bytes.
#![cfg(feature = "hip-tests")]
mod common;

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use golem_engine::config::EngineConfig;
use golem_engine::p3_glue::{ExpertTiers, Tier};
use golem_engine::{ChatTurn, SamplingParams, Scheduler, Session, SlotEvent, SlotJob, TurnRequest};

const STEPS: u32 = 48;
const N_LAYERS: u32 =
    xtask::tiny_hybrid::dims::N_LAYER - xtask::tiny_hybrid::dims::N_LAYER_DENSE_LEAD;
const N_EXPERT: u32 = xtask::tiny_hybrid::dims::N_EXPERT;
const K: u32 = xtask::tiny_hybrid::dims::N_EXPERT_USED;

fn config(cpu_expert_math: bool) -> EngineConfig {
    let off = common::offload::offload_config();
    EngineConfig {
        model_path: xtask::tiny_hybrid::cached_fixture(),
        n_gpu_layers: off.n_gpu_layers,
        cpu_moe_layers: off.cpu_moe_layers,
        ram_experts_gib: 1,
        device_cache_experts: K,
        n_predict: STEPS + 1,
        ignore_eos: true,
        cpu_expert_math,
        cpu_expert_frac: 1.0,
        // CPU expert math leases only pool-resident units: its leg keeps
        // the per-token prefill that fills the pool.
        prefill_stream: !cpu_expert_math,
        prefill_ubatch: 64,
        prefill_stream_min: 1,
        n_ctx: 512,
        ..EngineConfig::default()
    }
}

fn resident(sch: &mut Scheduler) -> Vec<Vec<bool>> {
    let h = sch.session().p3_hooks().expect("pooled session");
    (0..N_LAYERS)
        .map(|l| {
            (0..N_EXPERT)
                .map(|e| h.is_resident_for_test(l, e))
                .collect()
        })
        .collect()
}

/// Per layer, per expert: the substrate's slot-valid getter (1 valid).
fn slot_valid(sch: &mut Scheduler) -> Vec<Vec<i32>> {
    let h = sch.session().p3_hooks().expect("pooled session");
    (0..N_LAYERS)
        .map(|l| (0..N_EXPERT).map(|e| h.slot_valid_for_test(l, e)).collect())
        .collect()
}

/// Residency read around one step: pool bits and slab slot validity.
struct Around {
    pool: Vec<Vec<bool>>,
    slot: Vec<Vec<i32>>,
}

#[derive(Default)]
struct Seen {
    vram: u32,
    ram: u32,
    disk: u32,
    cpu: u32,
    decode_steps: u32,
}

/// Admits a short chat prompt and steps the scheduler to completion; after
/// each decode step checks every row against the residency read before it.
fn drive(cfg: &EngineConfig) -> (Seen, Scheduler) {
    let mut sch = Scheduler::new(Session::new(cfg).expect("pooled hip session")).unwrap();
    assert_eq!(
        sch.session().prefill_streaming(),
        cfg.prefill_stream,
        "precondition: prefill streaming as configured"
    );
    let view = sch.session().expert_tier_view().expect("pooled view");
    assert!(view.snapshot().is_none(), "no decode step yet");
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
        max_tokens: Some(STEPS + 1),
        stop: Vec::new(),
        cancel: Arc::new(AtomicBool::new(false)),
        rendered: Some(rendered),
    })
    .ok()
    .expect("free slot");
    let mut seen = Seen::default();
    let mut last_step = None;
    loop {
        let before = Around {
            pool: resident(&mut sch),
            slot: slot_valid(&mut sch),
        };
        let mut failed = None;
        let active = sch
            .step(&mut |_, e| {
                if let SlotEvent::Failed(e) = e {
                    failed = Some(e.to_string());
                }
            })
            .expect("step");
        assert!(failed.is_none(), "{failed:?}");
        let after = Around {
            pool: resident(&mut sch),
            slot: slot_valid(&mut sch),
        };
        let (slab_bytes, _) = sch
            .session()
            .device_cache_resident_bytes()
            .expect("pooled GPU split");
        let Some(t): Option<ExpertTiers> = view.snapshot() else {
            assert!(active, "no decode step ran");
            continue;
        };
        if t.step_kind == "decode" && t.step != last_step {
            last_step = t.step;
            seen.decode_steps += 1;
            check(
                &t,
                &before,
                &after,
                slab_bytes,
                cfg.cpu_expert_math,
                &mut seen,
            );
        }
        if !active {
            break;
        }
    }
    (seen, sch)
}

fn check(
    t: &ExpertTiers,
    before: &Around,
    after: &Around,
    slab_bytes: u64,
    cpu: bool,
    seen: &mut Seen,
) {
    assert_eq!((t.n_layers, t.n_experts, t.top_k), (N_LAYERS, N_EXPERT, K));
    let vram_sum: u32 = t.layers.iter().filter_map(|l| l.vram_resident).sum();
    assert_eq!(
        slab_bytes > 0,
        vram_sum > 0,
        "slab bytes {slab_bytes} vs Σvram_resident {vram_sum}"
    );
    for l in &t.layers {
        let li = l.layer as usize;
        assert!(!l.open, "closed after the step: {l:?}");
        assert_eq!(l.step, t.step, "every layer ran this step: {l:?}");
        assert!(
            !l.routed.is_empty() && l.routed.len() <= K as usize,
            "{l:?}"
        );
        let vram = l.vram_resident.expect("device slab reading");
        assert!(vram <= K, "vram_resident {vram} > K {K}");
        let oracle = after.pool[li].iter().filter(|&&r| r).count() as u32;
        let valid_now = after.slot[li].iter().filter(|&&v| v == 1).count() as u32;
        assert_eq!(vram, valid_now, "vram_resident vs Σ slot_valid: {l:?}");
        assert_eq!(l.ram_resident, oracle, "ram_resident vs Σ is_resident");
        for u in &l.routed {
            let was = before.pool[li][u.id as usize];
            let slot_was = before.slot[li][u.id as usize];
            let slot_now = after.slot[li][u.id as usize];
            assert!(slot_was >= 0 && slot_now >= 0, "slot getter refused");
            if u.tier != Tier::Cpu {
                assert_eq!(slot_now, 1, "slab unit {} not valid after: {l:?}", u.id);
            }
            match u.tier {
                Tier::Disk => {
                    assert!(!was, "disk unit {} was pool-resident: {l:?}", u.id);
                    assert_eq!(slot_was, 0, "disk unit {} had a valid slot: {l:?}", u.id);
                    seen.disk += 1;
                }
                Tier::Ram => {
                    assert!(was, "ram unit {} was not pool-resident: {l:?}", u.id);
                    assert_eq!(slot_was, 0, "ram unit {} had a valid slot: {l:?}", u.id);
                    seen.ram += 1;
                }
                Tier::Vram => {
                    assert!(was, "slab unit {} not in the inclusive pool: {l:?}", u.id);
                    assert_eq!(slot_was, 1, "vram unit {} had no valid slot: {l:?}", u.id);
                    seen.vram += 1;
                }
                Tier::Cpu => {
                    assert!(cpu, "cpu tier without CPU expert math: {l:?}");
                    assert!(was, "cpu unit {} was not pool-resident: {l:?}", u.id);
                    seen.cpu += 1;
                }
            }
            if !was {
                assert_eq!(
                    u.tier,
                    Tier::Disk,
                    "a unit absent from the pool was read: {l:?}"
                );
            }
        }
    }
}

#[test]
fn tiers_follow_forced_residency() {
    common::offload::require_process_per_test();
    let (seen, mut sch) = drive(&config(false));
    let s = sch.session();
    assert!(
        s.expert_mul_mat_id_on_gpu() > 0,
        "expert MUL_MAT_ID ran on the GPU"
    );
    assert_eq!(
        s.expert_mul_mat_id_off_gpu(),
        0,
        "no silent CPU expert math"
    );
    assert!(
        seen.decode_steps >= STEPS - 1,
        "decode steps: {}",
        seen.decode_steps
    );
    assert!(
        seen.disk > 0,
        "precondition: no decode step routed a never-paged unit"
    );
    assert!(seen.ram > 0, "precondition: no pool-resident slab miss");
    assert!(seen.vram > 0, "precondition: no slab hit in {STEPS} steps");
    assert_eq!(seen.cpu, 0);
}

#[test]
fn cpu_leased_units_read_cpu() {
    common::offload::require_process_per_test();
    let (seen, mut sch) = drive(&config(true));
    let stats = sch.session().cpu_expert_stats();
    assert!(
        stats.experts > 0,
        "precondition: CPU expert math ran: {stats:?}"
    );
    assert!(seen.cpu > 0, "no leased unit read cpu");
}
