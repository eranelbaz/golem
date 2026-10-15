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
    // C copies exactly the id list it hands each successful stream_source.
    assert_eq!(dc, stats.sourced_ids, "stream copies vs {stats:?}");
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

/// #135: after a streamed prefill, decode runs at `k_decode` with the
/// stream slabs freed; the next turn re-shrinks to `k_stream` and streams
/// again. Greedy output matches the per-token path on both turns.
#[test]
fn decode_reclaims_the_stream_slabs_and_the_next_turn_streams_again() {
    let resident = |session: &Session| {
        let (mut slab, mut stream) = (0u64, 0u64);
        // SAFETY: FFI getter on the session's live ctx between turns; the
        // out pointers are valid locals.
        unsafe {
            golem_llama_sys::raw::golem_device_cache_resident_bytes(
                session.p3_ctx_ptr().expect("pooled"),
                &mut slab,
                &mut stream,
            )
        };
        (slab, stream)
    };
    let mut token = Session::new(&config(false, 8)).expect("session");
    let mut out_token = String::new();
    token
        .chat_turn(&TURNS, &mut |p| out_token.push_str(p))
        .expect("turn");
    drop(token);

    // Snapshots off: turn 2 must re-prefill (and stream), not restore.
    let mut session = Session::new(&EngineConfig {
        snapshot_ram_bytes: Some(0),
        ..config(true, 8)
    })
    .expect("session");
    // tiny-moe top_k 2: stream K 2, decode K 4.
    session.p3_hooks_mut().expect("pooled").test_set_ks(2, 4);
    for turn in 0..2 {
        let planes_before = session
            .p3_hooks()
            .and_then(|p| p.stream_stats())
            .expect("stream stats")
            .planes;
        let mut out = String::new();
        session
            .chat_turn(&TURNS, &mut |p| out.push_str(p))
            .expect("turn");
        let planes = session
            .p3_hooks()
            .and_then(|p| p.stream_stats())
            .expect("stream stats")
            .planes;
        assert!(planes > planes_before, "turn {turn} did not stream");
        assert_eq!(session.device_cache_experts(), Some(4), "turn {turn} K");
        let (slab, stream) = resident(&session);
        assert_eq!(stream, 0, "turn {turn}: stream slabs live during decode");
        assert!(slab > 0, "turn {turn}: decode slabs missing");
        assert_eq!(out, out_token, "turn {turn} output");
    }
}
