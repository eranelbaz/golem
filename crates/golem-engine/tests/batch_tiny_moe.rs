//! Continuous batching (`Scheduler`, `--parallel`): batched greedy output equals
//! solo, per-seq state isolation (tiny-hybrid recurrent state), cancel,
//! failed-batch recovery, tool grammar through the scheduler, and the
//! streamed-ubatch invariant.
mod common;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use golem_engine::config::EngineConfig;
use golem_engine::{
    ChatTurn, FinishReason, SamplingParams, Scheduler, Session, SlotEvent, SlotJob, TurnRequest,
};

fn turns(p: &str) -> [ChatTurn<'_>; 1] {
    [ChatTurn {
        role: "user",
        content: p,
    }]
}

fn solo(cfg: &EngineConfig, p: &str, n: u32) -> String {
    let mut s = Session::new(cfg).expect("solo session");
    let mut out = String::new();
    s.turn(&TurnRequest::new(&turns(p)).max_tokens(Some(n)), &mut |t| {
        out.push_str(t)
    })
    .expect("solo turn");
    out
}

fn job(sch: &mut Scheduler, p: &str, n: u32, sampling: SamplingParams) -> SlotJob {
    let t = turns(p);
    let rendered = sch.session().render(&TurnRequest::new(&t)).expect("render");
    SlotJob {
        tokens: sch.tokenize(&rendered.prompt).expect("tokenize"),
        sampling,
        max_tokens: Some(n),
        stop: Vec::new(),
        cancel: Arc::new(AtomicBool::new(false)),
        rendered: Some(rendered),
    }
}

struct Run {
    text: Vec<String>,
    done: Vec<Option<Result<FinishReason, String>>>,
}

fn drive(sch: &mut Scheduler, n: usize, mut hook: impl FnMut(&mut Scheduler, usize)) -> Run {
    let mut r = Run {
        text: vec![String::new(); n],
        done: (0..n).map(|_| None).collect(),
    };
    let mut steps = 0;
    loop {
        hook(sch, steps);
        let active = sch
            .step(&mut |i, e| match e {
                SlotEvent::Text(t) => r.text[i].push_str(&t),
                SlotEvent::Done(o) => r.done[i] = Some(Ok(o.finish)),
                SlotEvent::Failed(e) => r.done[i] = Some(Err(e.to_string())),
            })
            .expect("step");
        steps += 1;
        assert!(steps < 10_000, "scheduler did not converge");
        if !active {
            return r;
        }
    }
}

fn parallel(cfg: EngineConfig, p: u32) -> EngineConfig {
    EngineConfig { parallel: p, ..cfg }
}

fn batched_equals_solo(cfg: EngineConfig) {
    let prompts = ["a", "bb", "count", "hello there"];
    let want: Vec<String> = prompts.iter().map(|p| solo(&cfg, p, 12)).collect();
    let mut sch = Scheduler::new(Session::new(&parallel(cfg, 4)).expect("session")).unwrap();
    for p in prompts {
        let j = job(&mut sch, p, 12, SamplingParams::default());
        sch.admit(j).ok().expect("free slot");
    }
    let got = drive(&mut sch, 4, |_, _| {});
    for (i, (s, t)) in want.iter().zip(&got.text).enumerate() {
        assert_eq!(
            s, t,
            "slot {i}: batch>1 delta — if this is reduction order, record it in docs/baselines/m4.4-batching.md and switch this to the T2 tolerance check"
        );
    }
}

#[test]
fn four_concurrent_slots_match_their_solo_greedy_output() {
    batched_equals_solo(common::tiny_moe_config());
}

#[test]
fn four_concurrent_slots_match_solo_on_tiny_hybrid() {
    // Per-seq recurrent state must not leak across slots.
    batched_equals_solo(common::tiny_hybrid_config());
}

#[test]
fn parallel_four_gives_each_slot_the_full_context() {
    let cfg = EngineConfig {
        n_ctx: 512,
        ..common::tiny_moe_config()
    };
    let s = Session::new(&parallel(cfg, 4)).expect("session");
    assert!(s.n_ctx_seq() >= 512, "n_ctx_seq {}", s.n_ctx_seq());
}

#[test]
fn admit_beyond_parallel_hands_the_job_back() {
    let mut sch =
        Scheduler::new(Session::new(&parallel(common::tiny_moe_config(), 2)).unwrap()).unwrap();
    for _ in 0..2 {
        let j = job(&mut sch, "a", 4, SamplingParams::default());
        assert!(sch.admit(j).is_ok());
    }
    let j = job(&mut sch, "a", 4, SamplingParams::default());
    assert!(sch.admit(j).is_err(), "third job must not fit two slots");
}

#[test]
fn joiner_starts_before_running_slot_finishes() {
    let cfg = common::tiny_moe_config();
    let mut sch = Scheduler::new(Session::new(&parallel(cfg, 2)).unwrap()).unwrap();
    let j = job(&mut sch, "count", 24, SamplingParams::default());
    sch.admit(j).ok().unwrap();
    let mut first_text_of_1 = None;
    let mut done_0 = None;
    let mut steps = 0;
    loop {
        if steps == 3 {
            let j = job(&mut sch, "hello there", 4, SamplingParams::default());
            sch.admit(j).ok().expect("free slot");
        }
        let active = sch
            .step(&mut |i, e| match (i, e) {
                (1, SlotEvent::Text(_)) if first_text_of_1.is_none() => {
                    first_text_of_1 = Some(steps)
                }
                (0, SlotEvent::Done(_)) => done_0 = Some(steps),
                _ => {}
            })
            .unwrap();
        steps += 1;
        if !active {
            break;
        }
    }
    let (a, b) = (
        first_text_of_1.expect("slot 1 emitted"),
        done_0.expect("slot 0 done"),
    );
    assert!(a < b, "joiner first text at step {a}, slot 0 done at {b}");
}

#[test]
fn cancel_one_slot_others_match_solo() {
    let cfg = common::tiny_moe_config();
    let prompts = ["a", "count", "hello there"];
    let want: Vec<String> = prompts.iter().map(|p| solo(&cfg, p, 12)).collect();
    let mut sch = Scheduler::new(Session::new(&parallel(cfg, 3)).unwrap()).unwrap();
    let mut cancel1 = None;
    for (i, p) in prompts.iter().enumerate() {
        let j = job(&mut sch, p, 12, SamplingParams::default());
        if i == 1 {
            cancel1 = Some(j.cancel.clone());
        }
        sch.admit(j).ok().unwrap();
    }
    let c = cancel1.unwrap();
    // Slot 0 prefills first (round-robin); cancel slot 1 while it is still
    // mid-prefill.
    let got = drive(&mut sch, 3, |_, step| {
        if step == 1 {
            c.store(true, Ordering::Relaxed);
        }
    });
    assert_eq!(got.done[1], Some(Ok(FinishReason::Cancelled)));
    assert_eq!(got.text[0], want[0]);
    assert_eq!(got.text[2], want[2]);
}

#[test]
fn failed_multi_seq_decode_resets_all_and_recovers() {
    let cfg = common::tiny_moe_config();
    let want = [solo(&cfg, "count", 12), solo(&cfg, "hello there", 12)];
    let mut sch = Scheduler::new(Session::new(&parallel(cfg, 3)).unwrap()).unwrap();
    for p in ["count", "hello there"] {
        let j = job(&mut sch, p, 12, SamplingParams::default());
        sch.admit(j).ok().unwrap();
    }
    let sampled = SamplingParams {
        temperature: 0.8,
        seed: 7,
        ..Default::default()
    };
    let j = job(&mut sch, "a", 12, sampled);
    sch.admit(j).ok().unwrap();
    sch.session().fail_generation_step_for_test(4);
    let got = drive(&mut sch, 3, |_, _| {});
    assert_eq!(got.text[0], want[0], "no duplicated or lost text");
    assert_eq!(got.text[1], want[1], "no duplicated or lost text");
    assert!(
        matches!(&got.done[2], Some(Err(m)) if m.contains("cannot be resumed")),
        "{:?}",
        got.done[2]
    );
}

#[test]
fn forced_tool_call_and_reasoning_survive_scheduler() {
    let tools = r#"[{"type":"function","function":{"name":"f","parameters":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"]}}}]"#;
    let choice = r#"{"type":"function","function":{"name":"f"}}"#;
    // tiny-moe's own template ignores tools; the server tool tests' one renders them.
    let cfg = EngineConfig {
        chat_template: Some(include_str!("../../golem-server/tests/qwen2.5-tools.jinja").into()),
        // The tools prompt is ~830 tokens.
        n_ctx: 4096,
        ..common::tiny_moe_config()
    };
    let mut sch = Scheduler::new(Session::new(&parallel(cfg, 2)).unwrap()).unwrap();
    let t = turns("call f");
    let req = TurnRequest::new(&t).tools(Some(tools), Some(choice));
    let rendered = sch.session().render(&req).expect("render");
    let j = SlotJob {
        tokens: sch.tokenize(&rendered.prompt).unwrap(),
        sampling: SamplingParams::default(),
        max_tokens: Some(256),
        stop: Vec::new(),
        cancel: Arc::new(AtomicBool::new(false)),
        rendered: Some(rendered),
    };
    sch.admit(j).ok().unwrap();
    let other = job(&mut sch, "a", 8, SamplingParams::default());
    sch.admit(other).ok().unwrap();
    let mut text = String::new();
    let mut finish = None;
    loop {
        let active = sch
            .step(&mut |i, e| match (i, e) {
                (0, SlotEvent::Text(t)) => text.push_str(&t),
                (0, SlotEvent::Done(o)) => finish = Some(o.finish),
                (0, SlotEvent::Failed(e)) => panic!("slot 0 failed: {e}"),
                _ => {}
            })
            .unwrap();
        if !active {
            break;
        }
    }
    assert!(finish.is_some(), "slot 0 never finished");
    let msg = sch
        .parser(0)
        .expect("parser")
        .parse(&text, false)
        .expect("parse");
    assert_eq!(msg["tool_calls"][0]["function"]["name"], "f", "{msg}");
}

fn stream_cfg(min: u32) -> EngineConfig {
    EngineConfig {
        ram_experts_gib: 1,
        device_cache_experts: 4,
        prefill_stream: true,
        prefill_ubatch: 64,
        prefill_stream_min: min,
        parallel: 2,
        ..common::tiny_moe_config()
    }
}

#[test]
fn streamed_chunk_never_mixed_with_decode_tokens() {
    let mut sch = Scheduler::new(Session::new(&stream_cfg(1)).unwrap()).unwrap();
    let j = job(&mut sch, "count", 24, SamplingParams::default());
    sch.admit(j).ok().unwrap();
    let mut joined = false;
    let mut steps = 0;
    loop {
        if steps == 4 && !joined {
            let j = job(&mut sch, "hello there", 4, SamplingParams::default());
            sch.admit(j).ok().unwrap();
            joined = true;
        }
        let planes0 = sch
            .session()
            .p3_hooks()
            .unwrap()
            .stream_stats()
            .unwrap()
            .planes;
        let mut texts = 0;
        let active = sch
            .step(&mut |_, e| {
                if let SlotEvent::Text(_) = e {
                    texts += 1
                }
            })
            .unwrap();
        let streamed = sch
            .session()
            .p3_hooks()
            .unwrap()
            .stream_stats()
            .unwrap()
            .planes
            > planes0;
        assert!(
            !(streamed && texts > 0),
            "step {steps}: streamed ubatch carried {texts} decode tokens"
        );
        steps += 1;
        if !active {
            break;
        }
    }
}

#[cfg(feature = "hip-tests")]
#[test]
fn long_joiner_chunk_takes_stream_path_short_takes_routed_only() {
    for (min, want_stream) in [(1, true), (u32::MAX, false)] {
        let mut sch = Scheduler::new(Session::new(&stream_cfg(min)).unwrap()).unwrap();
        let j = job(&mut sch, "count", 4, SamplingParams::default());
        sch.admit(j).ok().unwrap();
        drive(&mut sch, 2, |_, _| {});
        let hooks = sch.session().p3_hooks().unwrap();
        let (planes0, routed0) = (
            hooks.stream_stats().unwrap().planes,
            sch.session()
                .p3_stats()
                .unwrap()
                .fast_hits
                .load(Ordering::Relaxed)
                + sch
                    .session()
                    .p3_stats()
                    .unwrap()
                    .demand_slow_pageins
                    .load(Ordering::Relaxed),
        );
        let j = job(&mut sch, "hello there", 4, SamplingParams::default());
        sch.admit(j).ok().unwrap();
        drive(&mut sch, 2, |_, _| {});
        let planes = sch
            .session()
            .p3_hooks()
            .unwrap()
            .stream_stats()
            .unwrap()
            .planes;
        let p3 = sch.session().p3_stats().unwrap();
        let routed =
            p3.fast_hits.load(Ordering::Relaxed) + p3.demand_slow_pageins.load(Ordering::Relaxed);
        if want_stream {
            assert!(
                planes > planes0,
                "joiner ≥ prefill_stream_min did not stream"
            );
        } else {
            assert_eq!(planes, planes0, "short joiner streamed");
            assert!(routed > routed0, "routed-only counters did not move");
        }
    }
}
