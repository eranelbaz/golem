//! State snapshots (§7.1, R-4): a turn that restores the previous prompt's
//! snapshot and prefills only the delta decodes byte-identically to a full
//! re-prefill. Runs on tiny-moe (KV only) AND tiny-hybrid (recurrent state —
//! the case §2.5 is about); a failed precondition fails.

mod common;

use golem_engine::config::EngineConfig;
use golem_engine::{ChatTurn, Session};

fn floor(cfg: &EngineConfig) -> u64 {
    let f = golem_plan::ModelFacts::from_gguf(&cfg.model_path).unwrap();
    golem_plan::snapshot_bytes(f.rs_bytes, f.kv_per_token, u64::from(cfg.n_ctx))
}

fn two_turns(cfg: &EngineConfig, snap_bytes: u64, corrupt: bool) -> (String, Session) {
    let mut cfg = cfg.clone();
    cfg.snapshot_ram_bytes = Some(snap_bytes);
    cfg.n_predict = 16;
    let mut s = Session::new(&cfg).unwrap();
    let mut a1 = String::new();
    s.chat_turn(
        &[ChatTurn {
            role: "user",
            content: "hello",
        }],
        &mut |t| a1.push_str(t),
    )
    .unwrap();
    if corrupt {
        s.corrupt_snapshots_for_test();
    }
    let mut a2 = String::new();
    s.chat_turn(
        &[
            ChatTurn {
                role: "user",
                content: "hello",
            },
            ChatTurn {
                role: "assistant",
                content: &a1,
            },
            ChatTurn {
                role: "user",
                content: "and again",
            },
        ],
        &mut |t| a2.push_str(t),
    )
    .unwrap();
    (a2, s)
}

#[test]
fn restored_turn_is_byte_identical_to_full_reprefill() {
    for cfg in [common::tiny_moe_config(), common::tiny_hybrid_config()] {
        let (with, s) = two_turns(&cfg, 4 * floor(&cfg), false);
        let (without, _) = two_turns(&cfg, 0, false);
        assert_eq!(s.snapshot_hits(), 1, "second turn must restore");
        assert!(!without.is_empty(), "expected non-empty output");
        assert_eq!(with, without, "{:?}", cfg.model_path);
    }
}

#[test]
fn corrupted_snapshot_falls_back_to_full_prefill() {
    for cfg in [common::tiny_moe_config(), common::tiny_hybrid_config()] {
        let (with, s) = two_turns(&cfg, 4 * floor(&cfg), true);
        let (without, _) = two_turns(&cfg, 0, false);
        assert_eq!(s.snapshot_stats().unwrap().restore_failures, 1);
        assert_eq!(with, without, "{:?}", cfg.model_path);
    }
}

#[test]
fn snapshot_budget_below_floor_is_refused() {
    for cfg in [common::tiny_moe_config(), common::tiny_hybrid_config()] {
        let mut c = cfg.clone();
        c.snapshot_ram_bytes = Some(floor(&cfg) - 1);
        assert!(Session::new(&c).is_err(), "{:?}", cfg.model_path);
    }
}

#[test]
fn cancelled_prefill_commits_no_snapshot() {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    for cfg in [common::tiny_moe_config(), common::tiny_hybrid_config()] {
        let mut c = cfg.clone();
        c.snapshot_ram_bytes = Some(4 * floor(&cfg));
        let mut s = Session::new(&c).unwrap();
        let msgs = [ChatTurn {
            role: "user",
            content: "hello",
        }];
        let req =
            golem_engine::TurnRequest::new(&msgs).cancel(Some(Arc::new(AtomicBool::new(true))));
        s.turn(&req, &mut |_| {}).unwrap();
        assert_eq!(
            s.snapshot_stats().unwrap().used_bytes,
            0,
            "{:?}",
            c.model_path
        );
        assert_eq!(s.seq_pos_max(0), -1, "cancel drops the sequence");
    }
}

/// Pooled + streamed prefill: the restored turn's short delta takes the
/// per-token routed-only path, not whole streamed planes.
#[cfg(feature = "hip-tests")]
#[test]
fn short_delta_takes_routed_only_fill() {
    let cfg = EngineConfig {
        model_path: common::tiny_moe_path(),
        ram_experts_gib: 1,
        device_cache_experts: 4,
        prefill_stream: true,
        prefill_ubatch: 64,
        prefill_stream_min: 32,
        n_ctx: 1024,
        ..EngineConfig::default()
    };
    let snap = 4 * floor(&cfg);
    let mut c = cfg.clone();
    c.snapshot_ram_bytes = Some(snap);
    c.n_predict = 4;
    let mut s = Session::new(&c).unwrap();
    let long = "counting one two three four five six seven eight nine ten ".repeat(4);
    let mut a1 = String::new();
    s.chat_turn(
        &[ChatTurn {
            role: "user",
            content: &long,
        }],
        &mut |t| a1.push_str(t),
    )
    .unwrap();
    let planes_after_t1 = s.p3_hooks().unwrap().stream_stats().unwrap().planes;
    assert!(planes_after_t1 > 0, "turn 1 must stream");
    s.chat_turn(
        &[
            ChatTurn {
                role: "user",
                content: &long,
            },
            ChatTurn {
                role: "assistant",
                content: &a1,
            },
            ChatTurn {
                role: "user",
                content: "ok",
            },
        ],
        &mut |_| {},
    )
    .unwrap();
    assert_eq!(s.snapshot_hits(), 1, "second turn must restore");
    assert_eq!(
        s.p3_hooks().unwrap().stream_stats().unwrap().planes,
        planes_after_t1,
        "short delta must not stream whole planes"
    );
}

/// Real-target leg (GOLEM_HW=1, `GOLEM_HW_MODEL` = shard 1 of a GLM-5.3-Flash quant):
/// 5-turn chat, per-turn TTFT with snapshots off (`0`) vs on (2 GiB), each
/// session cold (fresh pool) then warm (same session, second pass);
/// `GOLEM_HW_ORDER=on,off` flips the A/B order. Prints one line per turn.
#[cfg(feature = "hip-tests")]
#[test]
#[ignore = "real model; GOLEM_HW=1 GOLEM_HW_MODEL=<shard 1> --run-ignored"]
fn hw_snapshot_turn_ttft() {
    assert_eq!(
        std::env::var("GOLEM_HW").as_deref(),
        Ok("1"),
        "set GOLEM_HW=1"
    );
    let model: std::path::PathBuf = std::env::var("GOLEM_HW_MODEL")
        .expect("GOLEM_HW_MODEL must name shard 1")
        .into();
    let order = std::env::var("GOLEM_HW_ORDER").unwrap_or_else(|_| "off,on".into());
    let users = [
        "Name three prime numbers.",
        "Now add them together.",
        "Is the sum prime?",
        "Explain why in one sentence.",
        "Thanks. Summarize our chat.",
    ];
    let mut outputs: Vec<Vec<String>> = Vec::new();
    for mode in order.split(',') {
        let cfg = EngineConfig {
            model_path: model.clone(),
            n_ctx: 4096,
            n_predict: 24,
            n_gpu_layers: 99,
            flash_attn_on: true,
            ram_experts_gib: 57,
            manifest_path: Some(model.with_file_name("manifest.json")),
            snapshot_ram_bytes: Some(if mode == "on" { 2 << 30 } else { 0 }),
            ..EngineConfig::default()
        };
        let mut s = Session::new(&cfg).expect("real-model session");
        for pass in ["cold", "warm"] {
            let mut hist: Vec<(&str, String)> = Vec::new();
            let mut answers = Vec::new();
            for (turn, u) in users.iter().enumerate() {
                hist.push(("user", (*u).to_owned()));
                let msgs: Vec<ChatTurn<'_>> = hist
                    .iter()
                    .map(|(r, c)| ChatTurn {
                        role: r,
                        content: c,
                    })
                    .collect();
                let t0 = std::time::Instant::now();
                let mut ttft = None;
                let mut a = String::new();
                s.chat_turn(&msgs, &mut |p| {
                    ttft.get_or_insert_with(|| t0.elapsed());
                    a.push_str(p);
                })
                .expect("turn");
                println!(
                    "HW mode={mode} pass={pass} turn={} prompt_tokens={} ttft_ms={:.0} hits={}",
                    turn + 1,
                    s.last_prompt_tokens().len(),
                    ttft.unwrap_or_else(|| t0.elapsed()).as_secs_f64() * 1e3,
                    s.snapshot_hits()
                );
                hist.push(("assistant", a.clone()));
                answers.push(a);
            }
            outputs.push(answers);
        }
        if mode == "on" {
            assert!(
                s.snapshot_hits() >= 8,
                "turns 2..5 of both passes must restore"
            );
        }
    }
    println!("HW outputs_equal_cold={}", outputs[0] == outputs[2]);
}
