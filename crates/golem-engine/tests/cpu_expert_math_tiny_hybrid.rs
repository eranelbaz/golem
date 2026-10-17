//! Decode-only CPU expert math (design §1 O1) on the HIP substrate, through
//! the engine-registered hooks over the tiny-hybrid fixture (2 MoE layers ×
//! 8 experts × 3 K-quant planes). K = 2
//! (= n_expert_used) slab experts per layer, so routing over 8 experts misses
//! the slab on most steps; the 1 GiB pool holds all 2 × 8 × 3 planes
//! (≈ 2.4 MB), so after one warm-up turn every miss is RAM-resident.
//!
//! With K == top-k the engine caps prefill chunks at 1 token, so the
//! per-token path also serves those chunks (identical one-token MoE math).
//!
//! Criterion: the CPU kernel's exactness vs the ggml CPU reference (clamp
//! included) is the vendor test golem-cpu-expert; here the session has no
//! forced-token API and the recurrent state compounds activation-quantization
//! drift even under an identical token prefix (0.9986 at step 2), so the
//! step-0 logits must have cosine ≥ 0.999 and the first `FORCED` greedy
//! tokens must be equal wherever the off-run's top-1 margin exceeds 1e-2
//! (a near-tie may flip). The clamped
//! fixture (swiglu_clamp_exp 0.5) gets a finite-logits leg only.
//! RLIMIT_MEMLOCK below the pool FAILS at `Session::new` (AGENTS.md).
#![cfg(feature = "hip-tests")]
#![allow(unsafe_code)]
mod common;

use golem_engine::config::EngineConfig;
use golem_engine::session::{ChatTurn, Session, expert_placement};
use golem_engine::timing::CpuExpertStats;

const STEPS: u32 = 32;
/// Leading steps compared between off and on (no forced-token API).
const FORCED: usize = 4;
const MSG: &[ChatTurn<'static>] = &[ChatTurn {
    role: "user",
    content: "one two three four five six seven eight",
}];

fn config(cpu_expert_math: bool, cpu_expert_frac: f32) -> EngineConfig {
    config_for(
        xtask::tiny_hybrid::cached_fixture(),
        cpu_expert_math,
        cpu_expert_frac,
    )
}

fn config_for(
    model_path: std::path::PathBuf,
    cpu_expert_math: bool,
    cpu_expert_frac: f32,
) -> EngineConfig {
    let off = common::offload::offload_config();
    EngineConfig {
        model_path,
        n_gpu_layers: off.n_gpu_layers,
        cpu_moe_layers: off.cpu_moe_layers,
        ram_experts_gib: 1,
        device_cache_experts: xtask::tiny_hybrid::dims::N_EXPERT_USED,
        n_predict: STEPS,
        cpu_expert_math,
        cpu_expert_frac,
        ..EngineConfig::default()
    }
}

/// One warm-up turn (fills the pool), then a recorded turn: its tokens, its
/// flat logits rows, and the CPU expert stats of the recorded turn alone.
fn run(cfg: &EngineConfig) -> (Vec<i32>, Vec<f32>, CpuExpertStats, Session) {
    let mut s = Session::new(cfg).expect("pooled hip session (RLIMIT_MEMLOCK ≥ 1 GiB)");
    let mut steps = Vec::new();
    s.chat_turn_parity(MSG, &mut steps).expect("warm-up turn");
    let warm = s.cpu_expert_stats();
    s.record_logits_for_test();
    let n = s.chat_turn_parity(MSG, &mut steps).expect("recorded turn");
    assert_eq!(n, STEPS, "precondition: {STEPS} tokens without EOG");
    let logits = s.take_logits_for_test();
    let total = s.cpu_expert_stats();
    let stats = CpuExpertStats {
        experts: total.experts - warm.experts,
        bytes: total.bytes - warm.bytes,
        padded: total.padded - warm.padded,
        ms: total.ms - warm.ms,
        join_wait_ms: total.join_wait_ms - warm.join_wait_ms,
    };
    let tokens = steps.iter().map(|p| p.token).collect();
    (tokens, logits, stats, s)
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let n = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    dot / (n(a) * n(b))
}

fn top1_margin(row: &[f32]) -> f32 {
    let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &x in row {
        if x > a {
            (a, b) = (x, a);
        } else if x > b {
            b = x;
        }
    }
    a - b
}

#[test]
fn cpu_expert_math_matches_gpu_output() {
    common::offload::require_process_per_test();
    let n_vocab = xtask::tiny_hybrid::dims::N_VOCAB as usize;
    let before = expert_placement();
    let (off_tok, off_logits, off_stats, off_s) = run(&config(false, 1.0));
    assert_eq!(
        off_s.cpu_expert_stats(),
        CpuExpertStats::default(),
        "flag-off run reported CPU expert math"
    );
    assert_eq!(off_stats, CpuExpertStats::default());
    drop(off_s);
    for frac in [1.0, 0.7] {
        let (on_tok, on_logits, stats, s) = run(&config(true, frac));
        assert_eq!(on_tok.len(), off_tok.len());
        assert_eq!(on_logits.len(), off_logits.len());
        assert_eq!(off_logits.len(), off_tok.len() * n_vocab);
        let mut min_cos = f64::INFINITY;
        let mut checked = 0;
        for (i, (a, b)) in off_logits
            .chunks(n_vocab)
            .zip(on_logits.chunks(n_vocab))
            .enumerate()
            .take(FORCED)
        {
            if on_tok[..i] != off_tok[..i] {
                break; // prefix diverged at a near-tie
            }
            if i == 0 {
                let c = cosine(a, b);
                min_cos = c;
                // Q8_K (CPU) vs q8_1 (GPU MMVQ) activation quantization, measured 0.99981 at step 0.
                assert!(c >= 0.999, "frac {frac} step 0: logit cosine {c}");
            }
            if top1_margin(a) > 1e-2 {
                checked += 1;
                assert_eq!(on_tok[i], off_tok[i], "frac {frac} step {i}: token differs");
            }
        }
        assert!(
            checked > 0,
            "precondition: no compared step had a top-1 margin > 1e-2"
        );
        assert!(stats.experts > 0, "frac {frac}: no expert ran on the CPU");
        assert!(stats.bytes > 0, "frac {frac}: no CPU expert bytes");
        assert_eq!(
            stats.padded,
            3 * stats.experts,
            "frac {frac}: padded planes"
        );
        println!(
            "frac {frac}: cpu experts/token {:.2}, bytes {}, step-0 cosine {min_cos:.7}, \
             tokens checked {checked}/{FORCED}",
            stats.experts as f64 / f64::from(STEPS),
            stats.bytes
        );
        drop(s);
    }
    let after = expert_placement();
    assert_eq!(
        after.on_cpu, before.on_cpu,
        "expert MUL_MAT_ID placed on the CPU (I2)"
    );
}

/// The routed-expert SWIGLU_CLAMP path end to end: the clamped fixture
/// decodes with CPU expert math to finite logits.
#[test]
fn cpu_expert_math_clamped_logits_finite() {
    common::offload::require_process_per_test();
    let cfg = config_for(xtask::tiny_hybrid::cached_fixture_clamped(), true, 1.0);
    let (_, logits, stats, _s) = run(&cfg);
    assert!(!logits.is_empty());
    assert!(logits.iter().all(|x| x.is_finite()), "non-finite logits");
    assert!(stats.experts > 0, "no expert ran on the CPU");
}

/// A device-cache reset after a flag-on turn drops the CPU worker's leases,
/// and the next turn still decodes with CPU expert math.
#[test]
fn reset_after_cpu_expert_turn_releases_leases() {
    common::offload::require_process_per_test();
    let (_, _, stats, mut s) = run(&config(true, 1.0));
    assert!(stats.experts > 0, "precondition: CPU expert math ran");
    let ctx = s.p3_hooks().expect("pooled session has hooks").ctx_ptr();
    assert!(!ctx.is_null());
    // SAFETY: decode is idle between turns and the backend is alive (the
    // Session teardown sequence, session.rs Drop).
    unsafe { golem_llama_sys::raw::golem_reset_device_cache(ctx) };
    let hooks = s.p3_hooks().unwrap();
    hooks.device_cache_reset();
    assert_eq!(hooks.cpu_leases(), 0, "reset left CPU leases");
    let before = s.cpu_expert_stats().experts;
    let mut steps = Vec::new();
    let n = s
        .chat_turn_parity(MSG, &mut steps)
        .expect("turn after reset");
    assert_eq!(n, STEPS);
    assert!(
        s.cpu_expert_stats().experts > before,
        "no CPU expert math after reset"
    );
}
