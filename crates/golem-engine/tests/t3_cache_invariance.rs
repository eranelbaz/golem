//! T3 board acceptance (§12.2 / spec I1): the persistent device expert-copy
//! cache must be decode-invariant — identical greedy token IDs across four
//! device-cache states for the same fixed greedy prompt — and hot expert
//! traffic must skip most copies on a warm cache (skip-ratio metric).
//! Runs only where a GPU split exists (feature `hip-tests`, HIP CI job);
//! CPU builds never enter the split-scheduler expert-copy path, so this file
//! is `#[cfg]`-skipped there (same gate as `p3_hook_fires.rs`).
//!
//! Pool OFF, one session: the C counters, `golem_reset_device_cache`, and
//! `golem_set_device_cache_budget` are process-global statics — they tick
//! whenever the P2 path runs, which here is driven by test-registered hooks
//! (the `p2_policy_query` seam: policy_query + ensure_host_resident both
//! set). The engine-registered-hook path (RAM pool) needs `ram_experts_gib`
//! ≥ 1 GiB of mlock, which boxes with `RLIMIT_MEMLOCK` below 1 GiB cannot
//! map (the HIP CI job runs it); reading the same statics through
//! `Session::device_cache_stats` is therefore gated behind the pool and the
//! counters are read here via the documented raw layout instead
//! (`[AtomicU64; 3]`, member order copied/skipped/evictions — pinned by
//! `golem_p2_bindings`). One session per the engine's own invariant: device
//! caches are invalidated at every model enable (`p3_glue`'s
//! `golem_reset_device_cache` seam), and this test re-invalidates per state
//! through the same FFI.
//!
//! Four states, each measured with `chat_turn_parity` (same decode path as
//! `chat_turn`, records per-step token IDs):
//! - (b) cold start — runs FIRST: `Session::new`'s decode self-check warms
//!   the cache, so `golem_reset_device_cache()` is called after session
//!   build; the measured turn must copy on first touch and then reuse its
//!   own copies (a state that only skips would be warm, a state that only
//!   copies never engaged the cache).
//! - (a) fully pre-promoted — unlimited budget, 4 warmup turns cache every
//!   routed expert; the measured turn's delta skip_ratio must exceed 0.5.
//! - (c) randomized eviction stress — budget = 3 experts' bytes with the
//!   test-only per-call order shuffle (`GOLEM_TEST_SHUFFLE_ORDER=1`, read
//!   ONCE at setup; this file is its only setter); the measured turn must
//!   evict.
//! - (d) low budget — 1 expert's stride per tensor, the smallest
//!   non-degenerate budget.
//!
//! The budget static is process-global: safe under nextest's per-test
//! process isolation; a plain `cargo test` run would cross-contaminate —
//! do not use it for this file. The stride is the tiny-moe expert-plane
//! stride (`n_embd * n_ff_exp * sizeof(f32)` — the value
//! `golem_tensor_expert_stride_bytes` reports for every F32 `ffn_*_exps`
//! plane of this fixture: gate/up are `[n_embd, n_ff_exp, n_expert]`, down
//! is `[n_ff_exp, n_embd, n_expert]`).
#![cfg(feature = "hip-tests")]
#![allow(unsafe_code)]
mod common;

use std::sync::atomic::{AtomicU64, Ordering};

use golem_engine::config::EngineConfig;
use golem_engine::parity::ParityStep;
use golem_engine::session::{ChatTurn, Session};
use xtask::tiny_moe::dims::{N_EMBD, N_FF_EXP};

const PROMPT: &str = "T3 cache-invariance probe";
const N_PREDICT: u32 = 32;
const WARMUP_TURNS: usize = 4;

fn expert_stride_bytes() -> u64 {
    u64::from(N_EMBD) * u64::from(N_FF_EXP) * std::mem::size_of::<f32>() as u64
}

/// Offloaded pool-off config: every expert plane on host, attention on GPU —
/// the split-scheduler expert-copy path engages (see `common::offload`).
fn config() -> EngineConfig {
    let mut cfg = common::offload::offload_config();
    cfg.n_predict = N_PREDICT;
    cfg.ram_experts_gib = 0; // pool off ⇒ no mlock; this test registers its own hooks
    cfg
}

/// The P4 device-cache counters in their raw C layout: a process-lifetime
/// static of 3 contiguous lock-free u64s, member order
/// copied/skipped/evictions (layout pinned by `golem_p2_bindings`; the C++
/// definition is the authority). Cumulative across the whole process.
fn raw_counters() -> [u64; 3] {
    // SAFETY: FFI getter; the returned pointer is a process-lifetime static
    // (never null, never freed), and the layout contract is the one
    // documented on `Session::device_cache_stats`.
    let block = unsafe { golem_llama_sys::raw::golem_get_device_cache_stats() };
    assert!(
        !block.is_null(),
        "golem_get_device_cache_stats returned null"
    );
    // SAFETY: same layout contract: 3 contiguous AtomicU64 in member order
    // bytes_copied, bytes_skipped, evictions.
    let counters = unsafe { &*block.cast::<[AtomicU64; 3]>() };
    [
        counters[0].load(Ordering::Relaxed),
        counters[1].load(Ordering::Relaxed),
        counters[2].load(Ordering::Relaxed),
    ]
}

struct Snapshot {
    copied: u64,
    skipped: u64,
    evictions: u64,
}

impl Snapshot {
    fn now() -> Self {
        let [copied, skipped, evictions] = raw_counters();
        Self {
            copied,
            skipped,
            evictions,
        }
    }

    fn copied_delta(&self, after: &Self) -> u64 {
        after.copied - self.copied
    }

    fn skipped_delta(&self, after: &Self) -> u64 {
        after.skipped - self.skipped
    }

    fn evictions_delta(&self, after: &Self) -> u64 {
        after.evictions - self.evictions
    }

    /// Delta skip ratio: skipped bytes / (skipped + copied) over the span.
    fn skip_ratio_delta(&self, after: &Self) -> f64 {
        let copied = self.copied_delta(after);
        let skipped = self.skipped_delta(after);
        let total = copied + skipped;
        if total == 0 {
            0.0
        } else {
            skipped as f64 / total as f64
        }
    }
}

/// Registers this test's P2 hooks: identity/shuffled policy order (any
/// permutation is legal — invariant 4) and pass-through pager seams.
/// Process-lifetime registration is fine: nextest isolates per-test
/// processes, so these hooks only ever see this test's decode.
fn register_p2_hooks(shuffle: bool) {
    use std::sync::atomic::AtomicUsize;

    static SHUFFLE: AtomicUsize = AtomicUsize::new(0);
    static CALL: AtomicUsize = AtomicUsize::new(0);
    SHUFFLE.store(shuffle as usize, Ordering::Relaxed);

    extern "C" fn on_ensure(
        _ctx: *mut libc::c_void,
        _tensor: *const libc::c_void,
        _first: i32,
        _last: i32,
    ) {
    }
    unsafe extern "C" fn on_query(
        _ctx: *mut libc::c_void,
        _tensor: *const libc::c_void,
        n: i32,
    ) -> *const i32 {
        let n = n as usize;
        let mut order: Vec<i32> = (0..n as i32).collect();
        if SHUFFLE.load(Ordering::Relaxed) == 1 {
            // rand-free LCG over the call counter: a fresh permutation per
            // query, so the C eviction walk hits varying victims.
            let mut lcg = (CALL.fetch_add(1, Ordering::Relaxed) as u64)
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            for i in (1..n).rev() {
                lcg = lcg
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let j = (lcg >> 33) as usize % (i + 1);
                order.swap(i, j);
            }
        }
        // contract: valid until the next call — leak-per-test is fine.
        Box::into_raw(order.into_boxed_slice()) as *const i32
    }
    extern "C" fn on_notify(_ctx: *mut libc::c_void, _expert_id: i64) {}

    // SAFETY: valid fn pointers; process-lifetime registration (no
    // unregister exists) is fine for a test process.
    unsafe {
        golem_llama_sys::raw::golem_register_hooks(
            Some(on_ensure),
            Some(on_query),
            Some(on_notify),
            std::ptr::null_mut(),
        );
    }
}

fn set_budget(experts: u64) {
    // SAFETY: FFI setter, no invariants (0 = full capacity).
    unsafe { golem_llama_sys::raw::golem_set_device_cache_budget(experts * expert_stride_bytes()) };
}

fn reset_cache() {
    // SAFETY: FFI, no invariants (clears cache entries; counters stay
    // cumulative, so all assertions are on counter deltas).
    unsafe { golem_llama_sys::raw::golem_reset_device_cache() };
}

/// One greedy turn, decoded through the parity recorder; returns the
/// generated token IDs (the trailing EOG sample, if recorded, is dropped).
fn greedy_tokens(session: &mut Session) -> Vec<i32> {
    let turns = [ChatTurn {
        role: "user",
        content: PROMPT,
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

fn warm_chat(session: &mut Session) {
    let turns = [ChatTurn {
        role: "user",
        content: PROMPT,
    }];
    session.chat_turn(&turns, &mut |_piece: &str| {}).unwrap();
}

#[test]
fn t3_four_cache_states_decode_identically() {
    // Explicit info filter: the plain fmt builder's default drops info!
    // events under this feature set, and the metric must land in the CI log.
    let _ = tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_new("info").unwrap())
        .try_init();

    // The shuffle knob is read ONCE, here, before any decode (same
    // read-once discipline as the engine policy state; this file is the
    // knob's only setter). It only permutes the expert order C receives;
    // any permutation is legal (invariant 4), so it must not change decode
    // output — that is exactly what states (a)-(d) then prove end to end.
    // SAFETY: single-threaded test setup, before any thread is spawned.
    unsafe { std::env::set_var("GOLEM_TEST_SHUFFLE_ORDER", "1") };
    register_p2_hooks(true);

    // Pool is off ⇒ no mlock failure path (that guard lives in the pooled tests).
    let mut session = Session::new(&config()).unwrap();

    // --- (b) cold start — runs FIRST (self-check warmup must not count).
    set_budget(0); // full capacity
    reset_cache();
    let cold0 = Snapshot::now();
    let tokens_cold = greedy_tokens(&mut session);
    let cold1 = Snapshot::now();
    let cold_copied = cold0.copied_delta(&cold1);
    let cold_skipped = cold0.skipped_delta(&cold1);
    assert!(
        cold_copied > 0,
        "cold turn must copy routed experts into the device cache \
         (a no-copy 'cold' turn means the reset did not clear): \
         copied={cold_copied} skipped={cold_skipped}"
    );
    // Hardware-refined coldness invariant (root-caused 2026-09-28, HIP):
    // after a reset the FIRST touch of every routed expert must copy, and
    // later execs of the same turn re-routing those experts must skip — the
    // persistent cache's entire purpose (§3.4). A literal `skipped == 0`
    // is unreachable for any multi-exec turn (32 decode steps + prefill
    // re-route the same hot experts); a warm state mislabeled cold shows
    // copied == 0 and is caught by the assert above.
    assert!(
        cold_skipped > 0,
        "cold turn must reuse its own first-touch copies within the turn \
         (the cache never engaged): copied={cold_copied} skipped={cold_skipped}"
    );

    // --- (a) fully pre-promoted: warm every routed expert, then measure.
    set_budget(0);
    for _ in 0..WARMUP_TURNS {
        warm_chat(&mut session);
    }
    let warm0 = Snapshot::now();
    let tokens_hot = greedy_tokens(&mut session);
    let warm1 = Snapshot::now();
    let ratio = warm0.skip_ratio_delta(&warm1);
    tracing::info!("T3 state (a) measured-turn delta skip_ratio = {ratio:.4}");
    assert!(
        ratio > 0.5,
        "warm cache must skip most expert copies (hot experts dominate \
         tiny-moe top-k routing); observed delta skip_ratio {ratio:.4} \
         (copied={} skipped={})",
        warm0.copied_delta(&warm1),
        warm0.skipped_delta(&warm1)
    );

    // --- (c) randomized eviction stress: 1-expert budget, shuffled order.
    // (On this substrate the upstream split heuristic engages the expert-copy
    // branch on only the first few execs of a turn, so the stress budget must
    // be the tightest non-degenerate one: with budget = 1 expert, the second
    // visit's eviction walk provably clears a non-routed valid expert (the
    // prefill and first decode route different pairs). The shuffle still
    // varies the victim order.)
    set_budget(1);
    reset_cache(); // the stress must start cold
    let stress0 = Snapshot::now();
    let tokens_stress = greedy_tokens(&mut session);
    let stress1 = Snapshot::now();
    let (d_cop, d_ski, d_evi) = (
        stress0.copied_delta(&stress1),
        stress0.skipped_delta(&stress1),
        stress0.evictions_delta(&stress1),
    );
    assert!(
        d_evi > 0,
        "1-expert budget with shuffled order must evict within one turn \
         (an eviction stress that never evicts proves nothing): \
         copied={d_cop} skipped={d_ski} evicted={d_evi}"
    );

    // --- (d) low budget: 1 expert's stride per tensor.
    set_budget(1);
    reset_cache();
    let tokens_low = greedy_tokens(&mut session);

    // Reset the process-global budget so nothing downstream of this test
    // inherits the tiny budget.
    set_budget(0);

    // Decode invariance across all four cache states (spec I1); each
    // assertion names the diverging pair and prints both vectors.
    assert_eq!(
        tokens_cold, tokens_hot,
        "state (b) cold vs state (a) pre-promoted diverged — the device \
         cache changed decode output"
    );
    assert_eq!(
        tokens_cold, tokens_stress,
        "state (b) cold vs state (c) eviction stress diverged — eviction \
         or the padding rule corrupts decode"
    );
    assert_eq!(
        tokens_cold, tokens_low,
        "state (b) cold vs state (d) low budget diverged — a one-expert \
         budget corrupts decode"
    );
}
