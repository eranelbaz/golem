//! Prefill-stream acceptance on tiny-moe (design §7.6, CPU/GPU substrate):
//! streaming a whole-prompt ubatch produces the same greedy output as the
//! per-token path, moves the stream counters, and leaves the pool untouched
//! (no pool page-ins from stream chunks — §7.6.4). A failed precondition
//! FAILS; nothing is silently skipped. One pooled session at a time (the
//! engine refuses a concurrent second pool), so the phases run sequentially
//! with explicit drops. Needs a GPU split: the CPU substrate has no
//! host-to-device expert copy to stream.

#![cfg(feature = "hip-tests")]
#![allow(unsafe_code)] // FFI counter read, same gate as budget_tiny_moe

mod common;

use golem_engine::config::EngineConfig;
use golem_engine::session::{ChatTurn, Session};
use golem_engine::stream::StreamStats;
use std::sync::atomic::Ordering;
use xtask::tiny_moe::dims::N_EXPERT;

fn config(stream: bool, n_predict: u32) -> EngineConfig {
    EngineConfig {
        model_path: common::tiny_moe_path(),
        ram_experts_gib: 1,
        device_cache_experts: 4,
        prefill_stream: stream,
        prefill_ubatch: 64,
        // The templated prompt is ~131 tokens: three stream chunks at 64,
        // the last one a short tail (prefill_stream_min 1 streams it too).
        prefill_stream_min: 1,
        n_predict,
        ..EngineConfig::default()
    }
}

const TURNS: [ChatTurn<'static>; 1] = [ChatTurn {
    role: "user",
    content: "counting one two three four five six seven eight nine ten",
}];

/// `stream_experts_copied` straight from the ctx block: this config is not
/// `offloaded()`, so `Session::device_cache_stats` reports `None`.
fn stream_copies(session: &Session) -> u64 {
    // SAFETY: FFI getter with the session's live ctx; ctx-owned block.
    let block = unsafe {
        golem_llama_sys::raw::golem_get_device_cache_stats(session.p3_ctx_ptr().expect("pooled"))
    };
    assert!(!block.is_null());
    // SAFETY: layout documented on `Session::device_cache_stats`: 5
    // contiguous AtomicU64, stream_experts_copied last.
    let c = unsafe { &*block.cast::<[std::sync::atomic::AtomicU64; 5]>() };
    c[4].load(Ordering::Relaxed)
}

#[test]
fn streamed_prefill_matches_per_token_and_spares_the_pool() {
    let run = |stream: bool| {
        let mut session = Session::new(&config(stream, 8)).expect("session");
        let mut out = String::new();
        let n = session
            .chat_turn(&TURNS, &mut |piece| out.push_str(piece))
            .expect("turn");
        let stats = session.p3_hooks().expect("pooled session").stream_stats();
        let dc = stream_copies(&session);
        (n, out, stats, dc)
    };
    let (n_stream, out_stream, stats, dc) = run(true);
    let (n_token, out_token, _, dc_token) = run(false);
    assert_eq!(dc_token, 0, "per-token path streamed");

    assert_eq!(n_stream, n_token, "same generation length");
    assert_eq!(out_stream, out_token, "greedy output must match");
    assert!(!out_stream.is_empty(), "expected non-empty output");

    let stats: StreamStats = stats.expect("stream stats");
    assert!(
        stats.planes > 0,
        "stream_source must fire for streamed planes"
    );
    assert!(stats.bytes > 0, "streamed bytes must be counted");
    // C copies one range per sourced id: every pooled/staged id plus any
    // unrouted stale id of a whole fill, at most n_expert per plane.
    assert!(
        dc >= stats.pool_hits + stats.staged && dc <= stats.planes * u64::from(N_EXPERT),
        "stream copies {dc} vs {stats:?}"
    );
    // Pool traffic (claims, paged bytes) of a turn with one generated token
    // is the same with and without streaming: stream chunks never claim or
    // page into the pool (§7.6.4).
    let pool_traffic = |stream: bool| {
        let mut session = Session::new(&config(stream, 1)).expect("session");
        session.chat_turn(&TURNS, &mut |_| {}).expect("turn");
        let st = session.p3_stats().expect("pooled session");
        (
            st.demand_claims.load(Ordering::Relaxed),
            st.bytes_paged.load(Ordering::Relaxed),
        )
    };
    assert_eq!(
        pool_traffic(true),
        pool_traffic(false),
        "stream chunks touched the pool"
    );

    // A tail shorter than prefill_stream_min takes the per-token path: no
    // plane streams, output unchanged.
    let mut session = Session::new(&EngineConfig {
        prefill_stream_min: u32::MAX,
        ..config(true, 8)
    })
    .expect("session");
    let mut out = String::new();
    session
        .chat_turn(&TURNS, &mut |piece| out.push_str(piece))
        .expect("turn");
    let gated = session.p3_hooks().expect("pooled session").stream_stats();
    assert_eq!(
        gated.expect("stream stats").planes,
        0,
        "gated tail streamed"
    );
    assert_eq!(out, out_token, "gated output must match");
}
