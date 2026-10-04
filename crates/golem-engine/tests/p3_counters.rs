//! P3 acceptance: counters move across a cold → warm decode (cold turn pages
//! experts in; warm turn pages nothing and is served by the device cache), and
//! a second session in the hook-owning process is refused with
//! `EngineError::PoolInit`. Cross-session output determinism is NOT asserted
//! here (not a GPU guarantee; the cross-process pool-on/off comparison lives
//! in golem-cli's `pooled_parity`).
//! Runs only where a GPU split exists (feature `hip-tests`, HIP CI job); CPU
//! builds never enter the split-scheduler expert-copy path, so the pool would
//! idle.
#![cfg(feature = "hip-tests")]
mod common;

use std::sync::atomic::Ordering;

use golem_engine::error::EngineError;
use golem_engine::session::{ChatTurn, Session};
use xtask::tiny_moe::dims;

fn turn(prompt: &str) -> [ChatTurn<'_>; 1] {
    [ChatTurn {
        role: "user",
        content: prompt,
    }]
}

fn chat(session: &mut Session, prompt: &str) -> String {
    let mut out = String::new();
    session
        .chat_turn(&turn(prompt), &mut |piece| out.push_str(piece))
        .unwrap();
    out
}

#[test]
fn p3_counters_move_cold_to_warm_and_second_session_is_refused() {
    let mut cfg = common::offload::offload_config();
    cfg.ram_experts_gib = 1;
    // Pool ON (1 GiB pool for tiny-moe's 3·8 planes of 2048 B). A failed
    // precondition (RLIMIT_MEMLOCK below the pool) FAILS — an early return
    // reads as a pass to the harness (AGENTS.md Evidence).
    let mut on = Session::new(&cfg).expect("session with pooled offload");
    let _ = chat(&mut on, "Hello");

    let stats = on.p3_stats().expect("pool enabled ⇒ stats present");
    assert!(
        stats.demand_slow_pageins.load(Ordering::Relaxed) > 0,
        "cold decode pages experts in"
    );

    // §6.6 claim-once: the slab copies all three planes of a unit in the same
    // compute (P2 validity bits move together), so ≤ 1 claim per unit per
    // compute means claims ≤ plane copies / 3. Every tiny-moe plane is
    // N_EMBD · N_FF_EXP f32 (gate/up [embd, ff], down [ff, embd]).
    let plane_bytes = u64::from(dims::N_EMBD * dims::N_FF_EXP) * 4;
    let plane_copies = on.device_cache_stats().unwrap().bytes_copied / plane_bytes;
    let claims = stats.demand_claims.load(Ordering::Relaxed);
    assert!(
        claims > 0 && 3 * claims <= plane_copies,
        "one shard-lock claim per unit per compute: {claims} claims for {plane_copies} plane copies"
    );

    // Warm decode: the P2 device cache serves the second turn (skips rise)
    // and the P3 pool pages nothing in — hot experts are resident in both
    // tiers, so neither the DMA nor the RAM pool is touched.
    let slow_before = on
        .p3_stats()
        .unwrap()
        .demand_slow_pageins
        .load(Ordering::Relaxed);
    let dc_before = on.device_cache_stats().expect("pooled split ⇒ P4 stats");
    let _ = chat(&mut on, "Hello");
    let slow_after = on
        .p3_stats()
        .unwrap()
        .demand_slow_pageins
        .load(Ordering::Relaxed);
    let dc_after = on.device_cache_stats().unwrap();
    assert_eq!(
        slow_after, slow_before,
        "warm turn must not page experts in from RAM-disk"
    );
    assert!(
        dc_after.bytes_skipped > dc_before.bytes_skipped,
        "warm turn must be served from the device cache: {dc_before:?} → {dc_after:?}"
    );

    // A second session in this process — pooled or not — must be refused:
    // the hooks are process-global and would serve the old model's pool
    // bytes to a new model's recycled tensor addresses.
    match Session::new(&common::offload::offload_config()) {
        Err(EngineError::PoolInit(msg)) => {
            assert!(msg.contains("expert hooks"), "unexpected refusal: {msg}");
        }
        Err(e) => panic!("second session must be refused as PoolInit, got: {e}"),
        Ok(_) => panic!("a second session in a hook-owning process must be refused"),
    }
}
