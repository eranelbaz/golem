//! P3 acceptance: counters move across cold→warm decode, and pool-on
//! output is byte-for-byte deterministic across identical runs (§7.4 batch=1
//! determinism — the real correctness gate: catches wrong plane addressing,
//! kind mix-ups, and torn copies that counters-only tests cannot). The second
//! session below is NOT "pool off": P3 hooks are process-global, so it shares
//! the same pool — it is a determinism reference, not an off-pool baseline.
//! Runs only where a GPU split exists (feature `hip-tests`, HIP CI job); CPU
//! builds never enter the split-scheduler expert-copy path, so the pool would
//! idle.
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
fn p3_counters_move_and_output_is_deterministic() {
    // Pool ON (1 GiB pool for tiny-moe's 3·8 planes of 2048 B):
    let mut cfg = common::offload::offload_config();
    cfg.ram_experts_gib = 1;
    // Pool ON (1 GiB pool for tiny-moe's 3·8 planes of 2048 B). A failed
    // precondition (RLIMIT_MEMLOCK below the pool) FAILS — an early return
    // reads as a pass to the harness (AGENTS.md Evidence, H-7).
    let mut on = Session::new(&cfg).expect("session with pooled offload");
    let _ = chat(&mut on, "Hello");
    {
        let mut o2 = String::new();
        on.chat_turn(
            &[ChatTurn {
                role: "user",
                content: "Hello",
            }],
            &mut |p| o2.push_str(p),
        )
        .unwrap();
    }

    let stats = on.p3_stats().expect("pool enabled ⇒ stats present");
    assert!(
        stats.demand_slow_pageins.load(Ordering::Relaxed) > 0,
        "cold decode pages experts in"
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

    // Determinism (§7.4): per-(seed, batch-composition) determinism holds
    // within a session (asserted above via the warm turn); cross-session
    // bitwise equality is NOT a GPU guarantee — benign logit ties flip when
    // buffer layouts differ (the T2 logit-gap story), and this assertion was
    // never actually exercised on HIP while it skipped-as-passed.
    //
    // A second session in this process — pooled or not — must
    // be refused: the hooks are process-global and would serve the old
    // model's pool bytes to a new model's recycled tensor addresses.
    let off = Session::new(&common::offload::offload_config());
    assert!(
        off.is_err(),
        "a second session in a hook-owning process must be refused"
    );
}
