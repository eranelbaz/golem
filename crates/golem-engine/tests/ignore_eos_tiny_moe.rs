//! `ignore_eos`: the sampler chain never emits an end-of-generation token,
//! so a greedy turn that would stop early runs to its token limit. CPU
//! substrate, tiny-moe with `<eoln>` as its EOS token (greedy emits it as the
//! second token of the "hello" turn).
mod common;

use golem_engine::FinishReason;
use golem_engine::TurnRequest;
use golem_engine::config::EngineConfig;
use golem_engine::session::{ChatTurn, Session};

const N: u32 = 8;

fn config(ignore_eos: bool) -> EngineConfig {
    EngineConfig {
        model_path: xtask::tiny_moe::cached_eos_fixture(),
        n_predict: N,
        ignore_eos,
        ..Default::default()
    }
}

const TURN: [ChatTurn<'static>; 1] = [ChatTurn {
    role: "user",
    content: "hello",
}];

#[test]
fn ignore_eos_runs_to_the_token_limit() {
    // The EOS variant tokenizes exactly like the default fixture, so the
    // turn is the one whose greedy stream contains `<eoln>`.
    let mut base = Session::new(&EngineConfig {
        n_predict: N,
        ..common::tiny_moe_config()
    })
    .unwrap();
    base.chat_turn(&TURN, &mut |_| {}).unwrap();

    // Precondition (fails, never skips): without the flag the turn stops
    // on EOG before N tokens.
    let mut s = Session::new(&config(false)).unwrap();
    let out = s.turn(&TurnRequest::new(&TURN), &mut |_| {}).unwrap();
    assert_eq!(
        s.last_prompt_tokens(),
        base.last_prompt_tokens(),
        "the EOS variant must tokenize the prompt identically"
    );
    assert_eq!(out.finish, FinishReason::Stop, "precondition: {out:?}");
    assert!(out.completion_tokens < N, "precondition: {out:?}");

    let mut s = Session::new(&config(true)).unwrap();
    let out = s.turn(&TurnRequest::new(&TURN), &mut |_| {}).unwrap();
    assert_eq!(out.completion_tokens, N, "{out:?}");
    assert_eq!(out.finish, FinishReason::Length, "{out:?}");

    // Parity mode builds the same chain: no step samples the EOS token.
    let mut steps = Vec::new();
    let n = s.chat_turn_parity(&TURN, &mut steps).unwrap();
    assert_eq!(n, N);
    let eos = xtask::tiny_moe::EARLY_EOS_TOKEN as i32;
    assert!(steps.iter().all(|p| p.token != eos), "EOS sampled");
}
