//! Per-turn slot API: sampling, max_tokens, cancel, stop, context guard.
mod common;
use golem_engine::config::EngineConfig;
use golem_engine::{ChatTurn, FinishReason, SamplingParams, Session, TurnRequest};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

fn session() -> Session {
    Session::new(&common::tiny_moe_config()).expect("tiny-moe session")
}
const MSG: &[ChatTurn<'static>] = &[ChatTurn {
    role: "user",
    content: "count to ten",
}];

#[test]
fn default_turn_equals_chat_turn_greedy() {
    let mut s = session();
    let mut a = String::new();
    s.chat_turn(MSG, &mut |t| a.push_str(t)).unwrap();
    let mut b = String::new();
    let out = s
        .turn(&TurnRequest::new(MSG), &mut |t| b.push_str(t))
        .unwrap();
    assert_eq!(a, b);
    assert!(out.prompt_tokens > 0);
}

#[test]
fn max_tokens_yields_length() {
    let full = session().turn(&TurnRequest::new(MSG), &mut |_| {}).unwrap();
    assert!(
        full.completion_tokens > 3,
        "precondition: greedy MSG must not EOG within 3 tokens; pick another MSG"
    );
    let out = session()
        .turn(&TurnRequest::new(MSG).max_tokens(Some(3)), &mut |_| {})
        .unwrap();
    assert_eq!(
        (out.completion_tokens, out.finish),
        (3, FinishReason::Length)
    );
}

#[test]
fn seeded_sampling_is_reproducible() {
    let p = SamplingParams {
        temperature: 0.8,
        top_p: 0.9,
        top_k: 40,
        seed: 7,
        ..Default::default()
    };
    let run = || {
        let mut o = String::new();
        session()
            .turn(
                &TurnRequest::new(MSG).sampling(p).max_tokens(Some(16)),
                &mut |t| o.push_str(t),
            )
            .unwrap();
        o
    };
    assert_eq!(run(), run());
}

#[test]
fn preset_cancel_stops_before_first_token() {
    let c = Arc::new(AtomicBool::new(true));
    let out = session()
        .turn(
            &TurnRequest::new(MSG).max_tokens(Some(16)).cancel(Some(c)),
            &mut |_| {},
        )
        .unwrap();
    assert_eq!(
        (out.completion_tokens, out.finish),
        (0, FinishReason::Cancelled)
    );
}

/// Recurrent state cannot be rewound (§2.5): a turn cancelled mid-decode on
/// the hybrid fixture must leave sequence 0 empty.
#[test]
fn cancelled_turn_leaves_no_state_on_hybrid() {
    let mut s = Session::new(&common::tiny_hybrid_config()).expect("tiny-hybrid session");
    let c = Arc::new(AtomicBool::new(false));
    let flag = c.clone();
    let mut pieces = 0;
    let out = s
        .turn(&TurnRequest::new(MSG).cancel(Some(c)), &mut |_| {
            pieces += 1;
            if pieces >= 2 {
                flag.store(true, Ordering::Relaxed);
            }
        })
        .unwrap();
    assert_eq!(out.finish, FinishReason::Cancelled);
    assert!(
        out.completion_tokens >= 2,
        "precondition: cancel must land mid-decode, got {} tokens",
        out.completion_tokens
    );
    assert_eq!(s.seq_pos_max(0), -1, "sequence 0 left state after cancel");
}

#[test]
fn prompt_over_ctx_is_typed_error() {
    let long = "word ".repeat(10_000);
    let msg = [ChatTurn {
        role: "user",
        content: &long,
    }];
    let e = session()
        .turn(&TurnRequest::new(&msg), &mut |_| {})
        .unwrap_err();
    assert!(matches!(
        e,
        golem_engine::EngineError::ContextOverflow { .. }
    ));
}

#[test]
fn parallel_above_one_is_rejected_until_scheduler() {
    let cfg = EngineConfig {
        parallel: 2,
        ..common::tiny_moe_config()
    };
    assert!(matches!(
        Session::new(&cfg),
        Err(golem_engine::EngineError::InvalidConfig(_))
    ));
}
