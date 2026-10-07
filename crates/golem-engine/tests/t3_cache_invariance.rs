//! T3 board acceptance (§12.2 / spec I1): the persistent device expert-copy
//! cache must be decode-invariant — identical greedy token IDs across four
//! device-cache states for the same fixed greedy prompt — and hot expert
//! traffic must skip most copies on a warm cache (skip-ratio metric).
//! Runs only where a GPU split exists (feature `hip-tests`, HIP CI job);
//! CPU builds never enter the split-scheduler expert-copy path, so this file
//! is `#[cfg]`-skipped there (same gate as `p3_hook_fires.rs`).
//!
//! Pool OFF, one session: the C counters, `golem_reset_device_cache`, and
//! `golem_set_device_cache_experts` is a process-global static — it ticks
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
//! `chat_turn`, records per-step token IDs). The budget is a per-layer expert
//! COUNT (`golem_set_device_cache_experts`); K = 2 is the smallest budget that
//! holds one routed set (tiny-moe top-k = 2):
//! - (b) cold start — runs FIRST: `Session::new`'s decode self-check warms
//!   the cache, so `golem_reset_device_cache()` is called after session
//!   build; the measured turn must copy on first touch and then reuse its
//!   own copies (a state that only skips would be warm, a state that only
//!   copies never engaged the cache). K = 8 = n_expert, identity order.
//! - (a) fully pre-promoted — K = 8, 4 warmup turns cache every routed
//!   expert; the measured turn's delta skip_ratio must exceed 0.5.
//! - (c) low budget, K = 2, reset cold, per-call SHUFFLED policy order (any
//!   permutation is legal, invariant 4).
//! - (d) low budget, K = 2, reset cold, identity order.
//!
//! Evictions are not asserted: the synthetic router routes the same pair for
//! every input, so no non-routed expert ever occupies a slot. The eviction
//! walk is covered by the vendored ctest (tests/golem-p2-cache.cpp).
//!
//! The budget static is process-global: safe under nextest's per-test
//! process isolation; a plain `cargo test` run would cross-contaminate —
//! do not use it for this file.
#![cfg(feature = "hip-tests")]
#![allow(unsafe_code)]
mod common;

use std::sync::atomic::{AtomicU64, Ordering};

use golem_engine::config::EngineConfig;
use golem_engine::parity::ParityStep;
use golem_engine::session::{ChatTurn, Session};

const PROMPT: &str = "T3 cache-invariance probe";
const N_PREDICT: u32 = 32;
const WARMUP_TURNS: usize = 4;

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
    let block = unsafe { golem_llama_sys::raw::golem_get_device_cache_stats(test_ctx()) };
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
}

impl Snapshot {
    fn now() -> Self {
        let [copied, skipped, _evictions] = raw_counters();
        Self { copied, skipped }
    }

    fn copied_delta(&self, after: &Self) -> u64 {
        after.copied - self.copied
    }

    fn skipped_delta(&self, after: &Self) -> u64 {
        after.skipped - self.skipped
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

/// Whether the routing order would be shuffled. ABI v5 note: the per-tensor
/// `policy_query` callback is GONE — the eviction order is consumed from the
/// ctx's triple buffers via `golem_policy_order`, and this test never pushes
/// (C falls back to the identity order + a one-time warning). The toggle is
/// kept so the state-(c)/(d) legs both run; the shuffle-permutation seam now
/// lives in the vendor ctest (`golem-p2-cache.cpp` order push cases).
static SHUFFLE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn set_shuffle(on: bool) {
    SHUFFLE.store(on, Ordering::Relaxed);
}

/// Registers this test's P2 hooks: identity/shuffled policy order (any
/// permutation is legal — invariant 4; see [`set_shuffle`]) and pass-through
/// pager seams. Process-lifetime registration is fine: nextest isolates
/// per-test processes, so these hooks only ever see this test's decode.
fn register_p2_hooks() {
    extern "C" fn on_ensure(
        _payload: *mut libc::c_void,
        tensor: *const libc::c_void,
        _miss_kinds: *const i32,
        miss_ids: *const i32,
        n_miss: i32,
        ptrs: *mut *const libc::c_void,
    ) {
        // ABI v2: the substrate copies from the returned pointers, so serve
        // the tensor's REAL per-expert planes — the same bytes the unpooled
        // upstream path reads (`input->data + id·nb[2]`). One shared static
        // here would hand every expert zero bytes, and the token asserts
        // below could never detect a misplaced plane.
        // SAFETY: `tensor` is the substrate's expert tensor, valid for the
        // call; only `data` is read, and the per-expert stride comes from
        // the substrate's own metadata query (nb[2], see golem-hooks.h).
        let base = unsafe { (*tensor.cast::<golem_llama_sys::raw::ggml_tensor>()).data };
        let stride =
            unsafe { golem_llama_sys::raw::golem_tensor_expert_stride_bytes(tensor) } as usize;
        // ABI v5: the miss list is flat across all planes (kind, id) pairs;
        // every miss of this test reads the tensor's REAL per-expert plane.
        for i in 0..n_miss as usize {
            // SAFETY: C passes `n_miss` entries in each list and `n_miss`
            // writable out slots (ggml-backend.cpp miss hoisting).
            let id = unsafe { *miss_ids.add(i) };
            unsafe { *ptrs.add(i) = base.byte_add(id as usize * stride).cast() };
        }
    }
    extern "C" fn on_routing(
        _payload: *mut libc::c_void,
        _blk: i32,
        _tensor: *const libc::c_void,
        _ids: *const i32,
        _n: i64,
        _n_experts: i32,
    ) {
    }

    // ABI v5: hooks register on a per-model ctx. This test drives the
    // substrate through a session whose engine hooks would normally own the
    // ctx; the test-owned ctx here keeps the seam working until session
    // lifecycle wiring attaches the model (routing presence is what turns
    // the C slab path on). The guards leak for the process lifetime:
    // nextest isolates per-test processes.
    use std::sync::OnceLock;
    static REGISTERED: OnceLock<()> = OnceLock::new();
    if REGISTERED.set(()).is_err() {
        return; // one registration per process
    }
    let ctx = golem_llama_sys::GolemCtx::create(4, 8).expect("test ctx");
    TEST_CTX
        .set(ctx.as_ptr() as usize)
        .expect("one test ctx per process");
    // The guard is intentionally leaked (process lifetime; nextest isolates
    // per-test processes): the substrate keeps calling the registered hooks
    // for every decode after this function returns, and the ctx must outlive
    // the Session built on it by `new_with_test_ctx`.
    let hooked = ctx.register(
        golem_llama_sys::HookFns {
            evict_batch: None,
            ensure: Some(on_ensure),
            routing: Some(on_routing),
            copy_done: None,
        },
        std::sync::Arc::new(()),
    );
    std::mem::forget(hooked);
}

/// A Session carrying the test-owned ctx on its model: `new_with_test_ctx`
/// attaches BEFORE `llama_init_from_model`, so every `sched_reserve`
/// (including re-reserves) sees the hooks and the slab path engages.
fn session_with_test_hooks() -> Session {
    Session::new_with_test_ctx(&config(), test_ctx()).expect("session with test-owned ctx")
}

/// The test-owned ctx pointer (set by `register_p2_hooks`). Held as `usize`:
/// the C-side ctx is internally synchronized (ABI v5 atomics + dc mutex) and
/// outlives the leak-for-process-lifetime guard.
static TEST_CTX: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

fn set_budget(experts: i32) {
    // SAFETY: FFI setter on the test ctx (ABI v5: per-ctx budget; 0 =
    // unlimited). The budget is a per-tensor expert COUNT — the slab is
    // sized from it.
    unsafe { golem_llama_sys::raw::golem_set_device_cache_experts(test_ctx(), experts) };
}

fn reset_cache() {
    // SAFETY: FFI on the test ctx (clears cache entries; counters stay
    // cumulative, so all assertions are on counter deltas).
    unsafe { golem_llama_sys::raw::golem_reset_device_cache(test_ctx()) };
}

fn test_ctx() -> golem_llama_sys::raw::golem_ctx_t {
    *TEST_CTX.get().expect("register_p2_hooks ran first") as golem_llama_sys::raw::golem_ctx_t
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

    register_p2_hooks();

    // The pooled path aborts on an unset (K = 0) budget, and the session's
    // load-time self-check already decodes through the hooks above.
    set_budget(8);

    // Pool is off ⇒ no mlock failure path (that guard lives in the pooled tests).
    let mut session = session_with_test_hooks();

    // --- (b) cold start — runs FIRST (self-check warmup must not count).
    set_budget(8); // K = n_expert: every routed pair fits
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
    set_budget(8);
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

    // --- (c) low budget (K = routed pair), shuffled policy order. Any
    // permutation is legal (invariant 4), so it must not change decode output.
    set_budget(2);
    set_shuffle(true);
    reset_cache(); // must start cold
    let tokens_stress = greedy_tokens(&mut session);

    // --- (d) low budget, identity order.
    set_shuffle(false);
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
        "state (b) cold vs state (c) shuffled low budget diverged — the policy \
         order or the padding rule corrupts decode"
    );
    assert_eq!(
        tokens_cold, tokens_low,
        "state (b) cold vs state (d) low budget diverged — a K = 2 \
         budget corrupts decode"
    );
}
