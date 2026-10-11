//! Cancel mid-prefill: the whole sequence is dropped (never a partial
//! rewind — §2.5, the hybrid fixture's recurrent state would be corrupted by
//! one), nothing stays leased, and the next turn on the same session equals
//! the same turn on a fresh session byte for byte (greedy).
mod common;
use golem_engine::config::EngineConfig;
use golem_engine::session::expert_placement;
use golem_engine::{ChatTurn, FinishReason, Session, TurnRequest};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

const LONG: &[ChatTurn<'static>] = &[ChatTurn {
    role: "user",
    content: "one two three four five six seven eight nine ten eleven twelve thirteen \
              fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty-one",
}];
const NEXT: &[ChatTurn<'static>] = &[ChatTurn {
    role: "user",
    content: "count to ten",
}];

fn greedy(s: &mut Session, msgs: &[ChatTurn<'_>]) -> String {
    let mut out = String::new();
    s.turn(&TurnRequest::new(msgs).max_tokens(Some(16)), &mut |t| {
        out.push_str(t)
    })
    .expect("turn");
    out
}

/// `arm` sets the session's cancel trigger for the next turn.
fn cancel_then_next_matches_fresh(cfg: &EngineConfig, arm: impl Fn(&mut Session)) {
    let mut s = Session::new(cfg).expect("session");
    let before = expert_placement();
    arm(&mut s);
    let cancel = Arc::new(AtomicBool::new(false));
    let out = s
        .turn(&TurnRequest::new(LONG).cancel(Some(cancel)), &mut |_| {})
        .expect("cancelled turn is Ok");
    assert_eq!(out.finish, FinishReason::Cancelled);
    assert_eq!(out.completion_tokens, 0, "cancelled during prefill");
    assert_eq!(s.seq_pos_max(0), -1, "cancel must drop the whole sequence");
    if let Some(h) = s.p3_hooks() {
        assert_eq!(h.pool_batch_leases(), 0, "pool leases left after cancel");
        if let Some(st) = h.stream_stats() {
            assert_eq!(st.held, 0, "stream staging/leases left after cancel");
        }
        assert_eq!(
            expert_placement().on_cpu,
            before.on_cpu,
            "expert MUL_MAT_ID ran on the CPU"
        );
    }
    s.cancel_after_prefill_chunks_for_test(0);
    if let Some(h) = s.p3_hooks() {
        h.cancel_at_stream_plane_for_test(0);
    }
    let next = greedy(&mut s, NEXT);
    drop(s);
    let fresh = greedy(&mut Session::new(cfg).expect("fresh session"), NEXT);
    assert!(!fresh.is_empty(), "precondition: NEXT must generate text");
    assert_eq!(
        next, fresh,
        "turn after cancel differs from a fresh session"
    );
}

fn chunked(mut cfg: EngineConfig) -> EngineConfig {
    // Several prefill chunks so the cancel lands mid-prompt.
    cfg.batch_tokens = 16;
    cfg.n_ctx = 512;
    cfg
}

#[test]
fn cancel_mid_prefill_then_next_turn_matches_fresh_session() {
    cancel_then_next_matches_fresh(&chunked(common::tiny_moe_config()), |s| {
        s.cancel_after_prefill_chunks_for_test(1)
    });
}

#[test]
fn cancel_mid_prefill_hybrid_then_next_turn_matches_fresh_session() {
    cancel_then_next_matches_fresh(&chunked(common::tiny_hybrid_config()), |s| {
        s.cancel_after_prefill_chunks_for_test(1)
    });
}

#[cfg(feature = "hip-tests")]
fn streamed() -> EngineConfig {
    EngineConfig {
        model_path: common::tiny_moe_path(),
        ram_experts_gib: 1,
        device_cache_experts: 4,
        prefill_stream: true,
        prefill_ubatch: 64,
        prefill_stream_min: 1,
        n_ctx: 512,
        ..EngineConfig::default()
    }
}

/// Pooled + streamed, cancel between stream chunks (Golem's own
/// before-decode check).
#[cfg(feature = "hip-tests")]
#[test]
fn cancel_mid_streamed_prefill_between_chunks() {
    cancel_then_next_matches_fresh(&streamed(), |s| s.cancel_after_prefill_chunks_for_test(1));
}

/// Pooled + streamed, cancel inside a streamed ubatch: `stream_source`
/// returns 1 at plane 3 → `llama_decode` == 2 → full `seq_rm` (ABI v11).
#[cfg(feature = "hip-tests")]
#[test]
fn cancel_inside_streamed_ubatch_aborts_within_one_plane() {
    let cfg = streamed();
    let mut s = Session::new(&cfg).expect("session");
    let planes0 = s.p3_hooks().unwrap().stream_stats().unwrap().planes;
    s.p3_hooks().unwrap().cancel_at_stream_plane_for_test(3);
    let out = s
        .turn(
            &TurnRequest::new(LONG).cancel(Some(Arc::new(AtomicBool::new(false)))),
            &mut |_| {},
        )
        .expect("cancelled turn is Ok");
    assert_eq!(out.finish, FinishReason::Cancelled);
    let st = s.p3_hooks().unwrap().stream_stats().unwrap();
    assert_eq!(
        st.planes - planes0,
        2,
        "abort must land at plane 3 (2 planes sourced)"
    );
    drop(s);
    cancel_then_next_matches_fresh(&cfg, |s| {
        s.p3_hooks().unwrap().cancel_at_stream_plane_for_test(3)
    });
}

/// Real-target leg (GOLEM_HW=1, `GOLEM_HW_MODEL` = first UD-Q4_K_XL shard):
/// cancel inside a streamed 2048-token prefill ubatch. Budget: one plane
/// ≈ 13.4 s / (42 × 3) ≈ 0.11 s cold, so cancel → return must be ≲ one
/// plane plus the drain. Prints `HW cancel_to_return_ms=…`.
#[cfg(feature = "hip-tests")]
#[test]
#[ignore = "real model; GOLEM_HW=1 GOLEM_HW_MODEL=<shard 1> --run-ignored"]
fn hw_cancel_inside_streamed_prefill() {
    use std::sync::atomic::Ordering;
    use std::time::Instant;
    assert_eq!(
        std::env::var("GOLEM_HW").as_deref(),
        Ok("1"),
        "set GOLEM_HW=1"
    );
    let model: std::path::PathBuf = std::env::var("GOLEM_HW_MODEL")
        .expect("GOLEM_HW_MODEL must name shard 1")
        .into();
    let cfg = EngineConfig {
        model_path: model.clone(),
        n_ctx: 4096,
        n_predict: 8,
        n_gpu_layers: 99,
        flash_attn_on: true,
        ram_experts_gib: 57,
        manifest_path: Some(model.with_file_name("manifest.json")),
        prefill_stream: true,
        ..EngineConfig::default()
    };
    // ≈ 2048+ prompt tokens: one full streamed ubatch.
    let long = "the quick brown fox jumps over the lazy dog ".repeat(230);
    let msgs = [ChatTurn {
        role: "user",
        content: &long,
    }];
    let mut s = Session::new(&cfg).expect("real-model session");
    let before = expert_placement();
    let cancel = Arc::new(AtomicBool::new(false));
    let c2 = cancel.clone();
    let planes0 = s.p3_hooks().unwrap().stream_stats().unwrap().planes;
    // Cancel 3 s into the turn: well inside the cold streamed ubatch.
    let fired = Arc::new(std::sync::Mutex::new(None::<Instant>));
    let f2 = fired.clone();
    let t = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_secs(3));
        *f2.lock().unwrap() = Some(Instant::now());
        c2.store(true, Ordering::SeqCst);
    });
    let out = s
        .turn(&TurnRequest::new(&msgs).cancel(Some(cancel)), &mut |_| {})
        .expect("cancelled turn is Ok");
    let returned = Instant::now();
    t.join().unwrap();
    let fired = fired
        .lock()
        .unwrap()
        .expect("cancel fired before the prefill ended");
    let st = s.p3_hooks().unwrap().stream_stats().unwrap();
    println!(
        "HW cancel_to_return_ms={:.0} planes_streamed={} prompt_tokens={}",
        (returned - fired).as_secs_f64() * 1e3,
        st.planes - planes0,
        s.last_prompt_tokens().len()
    );
    assert_eq!(out.finish, FinishReason::Cancelled);
    assert!(
        s.last_prompt_tokens().len() >= 2048,
        "prompt must fill one ubatch"
    );
    assert_eq!(s.seq_pos_max(0), -1);
    assert_eq!(s.p3_hooks().unwrap().pool_batch_leases(), 0);
    assert_eq!(st.held, 0);
    assert_eq!(expert_placement().on_cpu, before.on_cpu, "CPU expert math");
    let short = [ChatTurn {
        role: "user",
        content: "Name three prime numbers.",
    }];
    let mut next = String::new();
    s.turn(&TurnRequest::new(&short), &mut |t| next.push_str(t))
        .expect("next turn");
    drop(s);
    let mut fresh = String::new();
    Session::new(&cfg)
        .expect("fresh session")
        .turn(&TurnRequest::new(&short), &mut |t| fresh.push_str(t))
        .expect("fresh turn");
    println!("HW next={next:?} fresh={fresh:?}");
    assert_eq!(
        next, fresh,
        "turn after cancel differs from a fresh session"
    );
}
