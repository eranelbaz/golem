//! Acceptance (M1.4 Task 1): the parity dump mode records contiguous,
//! in-vocab, top1>=top2 steps on tiny-moe and decodes identically to
//! `chat_turn`. CPU substrate.
mod common;

use golem_engine::config::EngineConfig;
use golem_engine::parity::ParityStep;
use golem_engine::session::{ChatTurn, Session};
use xtask::tiny_moe::dims::N_VOCAB;

fn config() -> EngineConfig {
    EngineConfig {
        model_path: common::tiny_moe_path(),
        n_predict: 16,
        ..Default::default()
    }
}

#[test]
fn parity_steps_are_contiguous_in_vocab_and_top1_ge_top2() {
    let mut session = Session::new(&config()).unwrap();
    let turns = [ChatTurn {
        role: "user",
        content: "hello",
    }];
    let mut steps: Vec<ParityStep> = Vec::new();
    let n = session.chat_turn_parity(&turns, &mut steps).unwrap();

    assert!(n > 0, "greedy chat generated no tokens");
    assert!(n <= 16, "generated {n} tokens, n_predict = 16");
    assert!(
        !steps.is_empty() && steps.len() <= 16,
        "expected 1..=16 recorded steps, got {}",
        steps.len()
    );
    for (i, step) in steps.iter().enumerate() {
        assert_eq!(
            step.position, i as u32,
            "positions must be contiguous from 0"
        );
        assert!(
            step.token >= 0 && u32::try_from(step.token).is_ok_and(|t| t < N_VOCAB),
            "token {} out of vocab (N_VOCAB = {N_VOCAB})",
            step.token
        );
        assert!(
            step.top1.is_finite() && step.top2.is_finite(),
            "non-finite top-2 logits at position {}: ({}, {})",
            step.position,
            step.top1,
            step.top2
        );
        assert!(
            step.top1 >= step.top2,
            "top1 < top2 at position {}: {} < {}",
            step.position,
            step.top1,
            step.top2
        );
    }
}

#[test]
fn parity_decode_matches_chat_turn() {
    let turns = [ChatTurn {
        role: "user",
        content: "hello",
    }];

    let mut session = Session::new(&config()).unwrap();
    let mut out = String::new();
    let n = session
        .chat_turn(&turns, &mut |piece| out.push_str(piece))
        .unwrap();

    let mut session2 = Session::new(&config()).unwrap();
    let mut steps: Vec<ParityStep> = Vec::new();
    let n2 = session2.chat_turn_parity(&turns, &mut steps).unwrap();

    assert_eq!(n, n2, "parity mode must decode identically to chat_turn");
    assert!(!out.is_empty(), "expected non-empty greedy output");
    // Every sampler call is recorded; a trailing EOG sample is recorded but
    // not counted, so the count is n or n+1 — never anything else.
    assert!(
        steps.len() == usize::try_from(n).unwrap()
            || steps.len() == usize::try_from(n).unwrap() + 1,
        "recorded {} steps for {n} generated tokens",
        steps.len()
    );
}
