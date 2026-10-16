//! `serve --timing-csv/--routing-trace` (§11, §2.3 step 4): the worker
//! appends each finished turn's rows to the files; a failed turn leaves
//! none. Pooled tiny-moe through a real socket server.
use std::io::{Read, Write};
use std::net::SocketAddr;

use golem_server::TurnSinks;
use golem_server::test_support::{TestHooks, serve_with_sinks, test_server_config};

fn post(addr: SocketAddr, body: serde_json::Value) -> (u16, serde_json::Value) {
    let body = body.to_string();
    let mut s = std::net::TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(std::time::Duration::from_secs(60)))
        .unwrap();
    write!(
        s,
        "POST /v1/chat/completions HTTP/1.1\r\nhost: x\r\ncontent-type: application/json\r\n\
         content-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut r = String::new();
    s.read_to_string(&mut r).unwrap();
    let status = r[9..12].parse().unwrap();
    let json = r.split_once("\r\n\r\n").map_or("", |(_, b)| b);
    (status, serde_json::from_str(json).unwrap_or_default())
}

/// Sampled (seeded): a greedy slot would replay the failed batch instead
/// of failing.
fn chat(content: &str) -> serde_json::Value {
    serde_json::json!({"model":"tiny-moe","messages":[{"role":"user","content":content}],
        "max_tokens":8,"temperature":1.0,"seed":7})
}

#[tokio::test(flavor = "multi_thread")]
async fn two_turns_one_header_no_rows_from_a_failed_turn() {
    assert_eq!(
        std::env::var("NEXTEST_EXECUTION_MODE").as_deref(),
        Ok("process-per-test"),
        "run under cargo nextest: a pooled session needs its own process"
    );
    let dir = std::env::temp_dir().join(format!("golem-turn-sinks-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let (csv, trace) = (dir.join("t.csv"), dir.join("t.tsv"));
    let cfg = golem_engine::config::EngineConfig {
        ram_experts_gib: 1,
        device_cache_experts: 4,
        timing_csv: Some(csv.clone()),
        routing_trace: Some(trace.clone()),
        ..golem_server::test_engine_config()
    };
    let model = cfg.model_path.clone();
    // The CLI writes the CSV header; the worker only appends rows.
    let mut timing = std::fs::File::create(&csv).unwrap();
    golem_engine::timing::TimingRecorder::header(&mut timing, "test", "test", "").unwrap();
    let sinks = TurnSinks {
        timing: Some(timing),
        trace: Some(std::fs::File::create(&trace).unwrap()),
    };
    let (addr, stop, task) = serve_with_sinks(
        cfg,
        TestHooks::fail_generation_step(2),
        test_server_config(),
        sinks,
    )
    .await;
    let tokens = tokio::task::spawn_blocking(move || {
        let (status, _) = post(addr, chat("first"));
        assert_eq!(status, 500, "the first turn fails at its 2nd step");
        [chat("second"), chat("third")].map(|b| {
            let (status, v) = post(addr, b);
            assert_eq!(status, 200, "{v}");
            v["usage"]["completion_tokens"].as_u64().unwrap()
        })
    })
    .await
    .unwrap();
    stop.send(()).unwrap();
    task.await.unwrap().unwrap();

    let csv = std::fs::read_to_string(&csv).unwrap();
    assert_eq!(csv.lines().filter(|l| l.starts_with("step,")).count(), 1);
    let steps: Vec<u64> = csv
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("step,"))
        .map(|l| l.split(',').next().unwrap().parse().unwrap())
        .collect();
    // Two contiguous 0.. runs, one per good turn; the token sampled from the
    // prefill logits has no decode row.
    let restart = steps.iter().rposition(|&s| s == 0).unwrap();
    let (a, b) = steps.split_at(restart);
    assert!(
        a.iter().copied().eq(0..a.len() as u64) && b.iter().copied().eq(0..b.len() as u64),
        "{csv}"
    );
    assert_eq!(
        [a.len() as u64, b.len() as u64],
        tokens.map(|t| t - 1),
        "{csv}"
    );

    let text = std::fs::read_to_string(&trace).unwrap();
    assert_eq!(text.matches("# golem-routing-trace v1").count(), 1);
    let phase_steps = |phase: &str| {
        text.lines()
            .filter_map(|l| {
                let mut f = l.split('\t');
                let step = f.next()?;
                (f.next()? == phase).then_some(step)
            })
            .collect::<std::collections::BTreeSet<_>>()
    };
    // One prefill per good turn: neither the failed turn's records nor the
    // load self-check's.
    assert_eq!(phase_steps("pf").len(), 2, "{text}");
    let first = text.lines().nth(1).unwrap_or_default();
    assert_eq!(first.split('\t').nth(1), Some("pf"), "{text}");
    assert!(phase_steps("dc").len() >= steps.len(), "{text}");
    let geo = xtask::replay::Geo::from_gguf(&model).unwrap();
    let mut f = std::fs::File::open(&model).unwrap();
    let lead = golem_gguf::live_layers(&golem_gguf::read_kvs(&mut f).unwrap())
        .unwrap()
        .start;
    xtask::replay::Trace::parse("serve", &text, lead, &geo).expect("replay parses it");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn recording_is_refused_with_parallel_above_one() {
    assert!(golem_server::check_sinks(1, true).is_ok());
    assert!(golem_server::check_sinks(2, false).is_ok());
    assert!(golem_server::check_sinks(2, true).is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn a_sink_without_its_recorder_is_refused() {
    let p = std::env::temp_dir().join(format!("golem-sink-{}.tsv", std::process::id()));
    let sinks = TurnSinks {
        trace: Some(std::fs::File::create(&p).unwrap()),
        ..TurnSinks::default()
    };
    let _ = std::fs::remove_file(&p);
    let r = tokio::task::spawn_blocking(move || {
        golem_server::spawn_worker(golem_server::test_engine_config(), sinks)
    })
    .await
    .unwrap();
    assert!(r.is_err());
}
