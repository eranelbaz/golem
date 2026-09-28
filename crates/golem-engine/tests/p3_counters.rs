//! P3 acceptance (Task 5): counters move across cold→warm decode, and pool-on
//! output matches pool-off byte-for-byte (§7.4 batch=1 determinism — the real
//! correctness gate: catches wrong plane addressing, kind mix-ups, and torn
//! copies that counters-only tests cannot). Runs only where a GPU split exists
//! (feature `hip-tests`, HIP CI job); CPU builds never enter the
//! split-scheduler expert-copy path, so the pool would idle.
#![cfg(feature = "hip-tests")]
mod common;

use std::sync::atomic::Ordering;

use golem_engine::session::{ChatTurn, Session};

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
fn p3_counters_move_and_output_matches_pool_off() {
    // Pool ON (1 GiB pool for tiny-moe's 3·8 planes of 2048 B):
    let mut cfg = common::offload::offload_config();
    cfg.ram_experts_gib = 1;
    let mut on = Session::new(&cfg).unwrap();
    let out_on = chat(&mut on, "Hello");

    let stats = on.p3_stats().expect("pool enabled ⇒ stats present");
    assert!(
        stats.demand_slow_pageins.load(Ordering::Relaxed) > 0,
        "cold decode pages experts in"
    );

    // Warm decode: fast hits rise (same session, experts now resident).
    let before = on.p3_stats().unwrap().fast_hits.load(Ordering::Relaxed);
    let _ = chat(&mut on, "Hello");
    let after = on.p3_stats().unwrap().fast_hits.load(Ordering::Relaxed);
    assert!(after > before, "second turn must be all fast hits");

    // Correctness reference: pool-on output == pool-off output byte-for-byte.
    let off_cfg = common::offload::offload_config();
    let mut off = Session::new(&off_cfg).unwrap();
    let out_off = chat(&mut off, "Hello");
    assert_eq!(out_on, out_off, "pool-on decode diverged from pool-off");
}
