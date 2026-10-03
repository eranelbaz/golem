//! Acceptance: a chat completes on tiny-moe in CI (CPU substrate).
//! Uses the reserved ▁hello / ▁world vocab tokens as the smoke prompt.
mod common;

use golem_engine::config::EngineConfig;
use golem_engine::session::{ChatTurn, Session};

#[test]
fn chat_completes_on_tiny_moe() {
    let cfg = EngineConfig {
        model_path: common::tiny_moe_path(),
        n_predict: 8,
        ..Default::default()
    };
    let mut session = Session::new(&cfg).unwrap();

    let mut out = String::new();
    let turns = [ChatTurn {
        role: "user",
        content: "hello",
    }];
    let n = session
        .chat_turn(&turns, &mut |piece| out.push_str(piece))
        .unwrap();

    // If EOG samples immediately (empty output), rotate the smoke prompt across
    // the reserved tokens — do not weaken the assertion.
    assert!(n > 0 && !out.is_empty(), "greedy chat produced no output");
    assert!(n <= 8, "generated {n} tokens, n_predict = 8");

    let bos = session.last_prompt_tokens()[0];
    assert_eq!(
        session
            .last_prompt_tokens()
            .iter()
            .filter(|&&t| t == bos)
            .count(),
        1,
        "exactly one BOS expected (add_special=true + BOS-less fallback template)"
    );

    let mut session2 = Session::new(&cfg).unwrap();
    let mut out2 = String::new();
    let n2 = session2
        .chat_turn(&turns, &mut |p| out2.push_str(p))
        .unwrap();
    assert_eq!((n, out), (n2, out2), "greedy decode must be deterministic");
}

#[test]
fn prompt_longer_than_ctx_is_rejected_loudly() {
    let cfg = EngineConfig {
        model_path: common::tiny_moe_path(),
        n_ctx: 32,
        n_predict: 4,
        ..Default::default()
    };
    let mut session = Session::new(&cfg).unwrap();
    let long = "hello ".repeat(200);
    let turns = [ChatTurn {
        role: "user",
        content: long.as_str(),
    }];
    let r = session.chat_turn(&turns, &mut |_| {});
    assert!(
        r.is_err(),
        "oversized prompt must fail loudly, not truncate silently"
    );
}

#[test]
fn zero_n_predict_prefills_without_generating() {
    let cfg = EngineConfig {
        model_path: common::tiny_moe_path(),
        n_predict: 0,
        ..Default::default()
    };
    let mut session = Session::new(&cfg).unwrap();
    let mut out = String::new();
    let turns = [ChatTurn {
        role: "user",
        content: "hello",
    }];
    let n = session
        .chat_turn(&turns, &mut |piece| out.push_str(piece))
        .unwrap();
    assert_eq!(n, 0, "n_predict = 0 must generate no tokens");
    assert!(out.is_empty(), "n_predict = 0 must not emit pieces");
    assert!(
        !session.last_prompt_tokens().is_empty(),
        "prefill still tokenizes the prompt"
    );
}
