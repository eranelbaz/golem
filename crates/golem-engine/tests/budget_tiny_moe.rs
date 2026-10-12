//! Device expert-cache budget (P4, §3.4) on tiny-moe with an explicit
//! `K < n_expert` budget (review O6/Q5): a budgeted session must evict and
//! re-copy routed experts when routing breadth exceeds the slab, every expert
//! op must stay on the GPU while doing so (design I2), and a K = n_expert
//! session must serve a warm turn with zero evictions and zero re-copies.
//! Cross-budget output parity (K = 4 vs K = 8) needs a second HIP process
//! (one pool-enabled session per process), which lives in golem-cli's
//! `pooled_parity` (`device_cache_budget_k4_matches_k8`); this file covers
//! the counter/eviction behavior, where the counters are readable.
//! Runs only where a GPU split exists (feature `hip-tests`, HIP CI job); CPU
//! builds never enter the split-scheduler expert-copy path, so the pool
//! would idle.
#![cfg(feature = "hip-tests")]
#![allow(unsafe_code)] // FFI seams (reset/counters), same gate as t3_cache_invariance
mod common;

use golem_engine::parity::ParityStep;
use golem_engine::session::{ChatTurn, Session, expert_placement};
use xtask::tiny_moe::dims::N_EXPERT;

const N_PREDICT: u32 = 32;

/// Prompts chosen to route a broad expert set per layer: the eviction
/// assertion below is the breadth verification — greedy decode over these
/// distinct prompts routes more than `K = 4` distinct experts per touched
/// layer, and a routing that did not would fail `evictions > 0` instead of
/// passing vacuously.
const PROMPTS: [&str; 4] = [
    "The quick brown fox",
    "numbers 1 2 3 4 5 6 7 8 9",
    "MoE routing probe: which experts fire?",
    "a completely different subject line",
];

fn config(k: u32) -> golem_engine::config::EngineConfig {
    let mut cfg = common::offload::offload_config();
    cfg.ram_experts_gib = 1; // pooled ⇒ engine-registered hooks (the only P2 call-site)
    cfg.device_cache_experts = k;
    cfg.n_predict = N_PREDICT;
    cfg
}

fn greedy_tokens(session: &mut Session, prompt: &str) -> Vec<i32> {
    let turns = [ChatTurn {
        role: "user",
        content: prompt,
    }];
    let mut steps: Vec<ParityStep> = Vec::new();
    let n = session.chat_turn_parity(&turns, &mut steps).unwrap();
    assert!(n > 0, "greedy chat generated no tokens");
    steps
        .into_iter()
        .take(usize::try_from(n).unwrap())
        .map(|s| s.token)
        .collect()
}

#[test]
fn budget_k4_evicts_and_stays_on_gpu() {
    // One pool-enabled session per process; a failed precondition must FAIL,
    // never skip (AGENTS.md Evidence).
    let mut session = Session::new(&config(4)).expect("pooled offload session with K = 4");

    // Warm the cache: the session's own decode self-check plus this turn fill
    // the K = 4 slab with a first routed set.
    let _ = greedy_tokens(&mut session, PROMPTS[0]);

    // Measured window: clear the cached entries (counters stay cumulative —
    // all assertions are on deltas; see t3_cache_invariance for the seam).
    //
    // SAFETY: FFI with the session's live ctx (ABI v5: the device cache is
    // per-ctx); clears device-cache entries only.
    unsafe {
        golem_llama_sys::raw::golem_reset_device_cache(
            session.p3_ctx_ptr().expect("pooled session ⇒ live ctx"),
        )
    };
    let before = session
        .device_cache_stats()
        .expect("pooled split ⇒ P4 stats");
    let placement_before = expert_placement();

    let mut tokens = Vec::new();
    for p in PROMPTS {
        tokens.push(greedy_tokens(&mut session, p));
    }

    let after = session.device_cache_stats().unwrap();
    let (slab_res, _) = session.device_cache_resident_bytes().unwrap();
    assert!(slab_res > 0, "decode left no resident device slab bytes");
    let copied = after.bytes_copied - before.bytes_copied;
    let evictions = after.evictions - before.evictions;
    let skipped = after.bytes_skipped - before.bytes_skipped;
    assert!(
        copied > 0,
        "post-reset window must re-copy routed experts (the cache never \
         engaged): copied={copied} evictions={evictions} skipped={skipped}"
    );
    assert!(
        evictions > 0,
        "K = 4 < n_expert = {N_EXPERT} with routing breadth across {} varied \
         prompts must evict slab entries; zero evictions means the routing \
         never exceeded the budget (precondition unmet): \
         copied={copied} evictions={evictions} skipped={skipped}",
        PROMPTS.len()
    );

    // Design I2: every expert MUL_MAT_ID in the window ran on the GPU, even
    // while evicting (counters are process-global; this process only decodes
    // this session).
    let placement_after = expert_placement();
    let on_gpu = placement_after.on_gpu - placement_before.on_gpu;
    let on_cpu = placement_after.on_cpu - placement_before.on_cpu;
    assert!(
        on_cpu == 0 && on_gpu > 0,
        "expert ops must stay on the GPU under budget pressure: \
         on_gpu={on_gpu} on_cpu={on_cpu}"
    );

    // Decode must stay greedy-stable across the measured window (each prompt
    // yields tokens; content itself is not asserted here — cross-budget
    // parity is the pooled_parity leg).
    assert!(
        tokens.iter().all(|t| !t.is_empty()),
        "every measured turn must generate tokens"
    );
}

#[test]
fn budget_k8_warm_turn_has_zero_window_counters() {
    // K = n_expert: every routed pair fits, so a warm turn must neither
    // evict nor re-copy — served entirely from the device cache (the same
    // assertion shape as p3_counters' warm leg, at the full budget).
    let mut session = Session::new(&config(8)).expect("pooled offload session with K = 8");
    let _ = greedy_tokens(&mut session, PROMPTS[0]);

    let before = session
        .device_cache_stats()
        .expect("pooled split ⇒ P4 stats");
    let _ = greedy_tokens(&mut session, PROMPTS[0]);
    let after = session.device_cache_stats().unwrap();

    assert_eq!(
        after.evictions - before.evictions,
        0,
        "K = n_expert warm turn must not evict: {before:?} → {after:?}"
    );
    assert_eq!(
        after.bytes_copied - before.bytes_copied,
        0,
        "K = n_expert warm turn must not re-copy: {before:?} → {after:?}"
    );
    assert!(
        after.bytes_skipped > before.bytes_skipped,
        "K = n_expert warm turn must be served from the device cache: \
         {before:?} → {after:?}"
    );
}
