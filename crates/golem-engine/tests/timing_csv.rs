//! §11 per-step timing CSV format tests (pure formatting; no GPU, no model).

mod common;

use golem_engine::timing::{CpuExpertStats, StepTiming, TimingRecorder, VictimRingStats};

#[test]
fn csv_header_carries_version_fork_commit_and_config() {
    let mut buf = Vec::new();
    TimingRecorder::header(
        &mut buf,
        "0.1.0",
        "deadbeef",
        "ctx=4096 replicas=2 io-threads=2",
    )
    .unwrap();
    let s = String::from_utf8(buf).unwrap();
    assert!(
        s.contains("engine_version=0.1.0")
            && s.contains("fork_commit=deadbeef")
            && s.contains("ctx=4096")
    );
    assert!(
        s.lines().nth(1).unwrap().ends_with(
            ",policy_ms,compute_ms,cpu_expert_math_experts,cpu_expert_math_bytes,\
             cpu_expert_math_ms,cpu_expert_join_wait_ms,ring_d2h_ms,ring_d2h_bytes,ring_units,ring_full,\
             ids_readbacks,ids_sync_ms,ids_gap_ms,ids_gap_max_ms"
        ),
        "{s}"
    );
}

/// CPU expert math on: the four columns are numbers, and `compute_ms` does
/// not subtract the worker's ms (it overlaps the GPU).
#[test]
fn cpu_expert_columns_measured_and_not_subtracted_from_compute() {
    let mut r = TimingRecorder::new(64);
    r.row(StepTiming {
        step: 0,
        tokens: 1,
        wall_ms: 100.0,
        p3_wait_ms: 60.0,
        disk_wait_ms: f64::NAN,
        disk_worker_ms: f64::NAN,
        disk_bytes: 0,
        h2d_ms: f64::NAN,
        h2d_bytes: 0,
        policy_ms: f64::NAN,
        sync_gap: None,
        cpu_expert: Some(CpuExpertStats {
            experts: 12,
            bytes: 69_206_016,
            ms: 25.0,
            join_wait_ms: 3.5,
            padded: 36,
        }),
        victim_ring: None,
    });
    let mut out = Vec::new();
    r.flush(&mut out).unwrap();
    let line = std::str::from_utf8(&out).unwrap().lines().last().unwrap();
    assert!(
        line.ends_with(",40.0,12,69206016,25.0,3.5,unmeasured,unmeasured,unmeasured,unmeasured,unmeasured,unmeasured,unmeasured,unmeasured"),
        "row: {line}"
    );
}

#[test]
fn rows_write_unmeasured_for_unmeasured_columns_never_zero() {
    let mut r = TimingRecorder::new(64);
    r.row(StepTiming {
        step: 1,
        tokens: 1,
        wall_ms: 100.0,
        p3_wait_ms: 60.0,
        disk_wait_ms: f64::NAN,
        disk_worker_ms: 55.0,
        disk_bytes: 121_634_816,
        h2d_ms: f64::NAN,
        h2d_bytes: 0,
        policy_ms: f64::NAN,
        sync_gap: None,
        cpu_expert: None,
        victim_ring: None,
    });
    let mut out = Vec::new();
    r.flush(&mut out).unwrap();
    let line = std::str::from_utf8(&out)
        .unwrap()
        .lines()
        .last()
        .unwrap()
        .to_string();
    assert!(line.contains("unmeasured"));
    assert!(
        line.ends_with(
            ",unmeasured,unmeasured,unmeasured,unmeasured,unmeasured,unmeasured,unmeasured,unmeasured,unmeasured,unmeasured,unmeasured,unmeasured"
        ),
        "cpu_expert, ring and ids columns off: {line}"
    );
    assert!(!line.contains(",0,")); // no silent zero for unmeasured
    assert!(line.contains("55") && line.contains("121634816"));
}

/// Victim ring armed: the four ring columns are numbers, and the D2H
/// stream's ms is not subtracted from `compute_ms` (it is off the refill
/// path). 16 planes × 6,881,280 B = 110,100,480 B.
#[test]
fn ring_columns_measured_and_not_subtracted_from_compute() {
    let mut r = TimingRecorder::new(64);
    r.row(StepTiming {
        step: 0,
        tokens: 1,
        wall_ms: 100.0,
        p3_wait_ms: 60.0,
        disk_wait_ms: f64::NAN,
        disk_worker_ms: f64::NAN,
        disk_bytes: 0,
        h2d_ms: f64::NAN,
        h2d_bytes: 0,
        policy_ms: f64::NAN,
        sync_gap: None,
        cpu_expert: None,
        victim_ring: Some(VictimRingStats {
            units: 16,
            full: 1,
            d2h_ms: 4.5,
            d2h_bytes: 110_100_480,
        }),
    });
    let mut out = Vec::new();
    r.flush(&mut out).unwrap();
    let line = std::str::from_utf8(&out).unwrap().lines().last().unwrap();
    assert!(
        line.ends_with(",40.0,unmeasured,unmeasured,unmeasured,unmeasured,4.5,110100480,16,1,unmeasured,unmeasured,unmeasured,unmeasured"),
        "row: {line}"
    );
}

#[test]
fn compute_is_wall_minus_p3_h2d_policy_not_disk_worker() {
    let mut r = TimingRecorder::new(64);
    // 100 − (60 + 4.5) = 35.5: the disk worker runs inside the p3_wait
    // window (parallel) and is deliberately not subtracted.
    r.row(StepTiming {
        step: 0,
        tokens: 1,
        wall_ms: 100.0,
        p3_wait_ms: 60.0,
        disk_wait_ms: f64::NAN,
        disk_worker_ms: f64::NAN,
        disk_bytes: 0,
        h2d_ms: f64::NAN,
        h2d_bytes: 0,
        policy_ms: 4.5,
        sync_gap: None,
        cpu_expert: None,
        victim_ring: None,
    });
    let mut out = Vec::new();
    r.flush(&mut out).unwrap();
    let line = std::str::from_utf8(&out).unwrap().lines().last().unwrap();
    assert!(line.contains("35.5"), "row: {line}");
}

/// A real session turn's CSV ends with the prefill trailer; on a fresh
/// session (no restored prefix) it counts every prompt token.
#[test]
fn turn_csv_ends_with_prefill_trailer() {
    use golem_engine::config::EngineConfig;
    use golem_engine::session::{ChatTurn, Session};
    let cfg = EngineConfig {
        n_predict: 4,
        timing_csv: Some(
            std::env::temp_dir().join(format!("golem-unused-pf-{}.csv", std::process::id())),
        ),
        ..common::tiny_moe_config()
    };
    let mut s = Session::new(&cfg).unwrap();
    s.chat_turn(
        &[ChatTurn {
            role: "user",
            content: "hello",
        }],
        &mut |_| {},
    )
    .unwrap();
    let mut out = Vec::new();
    s.write_timing(&mut out).unwrap();
    let csv = String::from_utf8(out).unwrap();
    let last = csv.lines().last().unwrap();
    let rest = last.strip_prefix("# prefill_ms=").expect(&csv);
    let (ms, n) = rest.split_once(" prompt_tokens=").expect(last);
    assert!(ms.parse::<f64>().unwrap() >= 0.0, "{last}");
    let n: usize = n.parse().unwrap();
    assert!(n > 0, "{last}");
    assert_eq!(n, s.last_prompt_tokens().len(), "{last}");
}

/// HIP, pooled offload: `h2d_ms` is a measured number with async_h2d (the
/// copy-stream event pairs) and `unmeasured` without.
#[cfg(feature = "hip-tests")]
#[test]
fn h2d_columns_measured_only_with_async_h2d() {
    use golem_engine::config::EngineConfig;
    use golem_engine::session::{ChatTurn, Session};
    let rows = |async_h2d: bool| {
        let off = common::offload::offload_config();
        let cfg = EngineConfig {
            ram_experts_gib: 1,
            device_cache_experts: 4,
            n_predict: 4,
            async_h2d,
            timing_csv: Some(
                std::env::temp_dir().join(format!("golem-unused-{}.csv", std::process::id())),
            ),
            ..off
        };
        let mut s = Session::new(&cfg).unwrap();
        s.chat_turn(
            &[ChatTurn {
                role: "user",
                content: "hello",
            }],
            &mut |_| {},
        )
        .unwrap();
        let mut out = Vec::new();
        s.write_timing(&mut out).unwrap();
        String::from_utf8(out).unwrap()
    };
    // columns: step,tokens,wall_ms,p3_wait_ms,disk_wait_ms,disk_worker_ms,disk_bytes,h2d_ms,...
    let col = |csv: &str, i: usize| -> Vec<String> {
        csv.lines()
            .filter(|l| !l.starts_with('#') && !l.starts_with("step"))
            .map(|l| l.split(',').nth(i).unwrap().to_owned())
            .collect()
    };
    let csv_on = rows(true);
    let on = col(&csv_on, 7);
    assert!(!on.is_empty());
    assert!(on.iter().all(|v| v.parse::<f64>().is_ok()), "{on:?}");
    // The copy stream actually carried bytes (not just a zero reading).
    let bytes: u64 = col(&csv_on, 8)
        .iter()
        .map(|v| v.parse::<u64>().unwrap())
        .sum();
    assert!(bytes > 0, "{csv_on}");
    let off = col(&rows(false), 7);
    assert!(!off.is_empty());
    assert!(off.iter().all(|v| v == "unmeasured"), "{off:?}");
}

/// HIP, pooled offload through the engine-registered hooks: `policy_ms` and
/// the four ids-readback columns are measured numbers on every generation
/// row, and each row counts at least one readback per decode step.
#[cfg(feature = "hip-tests")]
#[test]
fn policy_and_ids_readback_columns_measured_with_hooks() {
    use golem_engine::config::EngineConfig;
    use golem_engine::session::{ChatTurn, Session};
    use xtask::tiny_moe::dims;
    let off = common::offload::offload_config();
    let csv_path = std::env::temp_dir().join(format!("golem-unused-p-{}.csv", std::process::id()));
    let cfg = EngineConfig {
        ram_experts_gib: 1,
        device_cache_experts: 4,
        n_predict: 4,
        timing_csv: Some(csv_path.clone()),
        ..off
    };
    let mut s = Session::new(&cfg).unwrap();
    s.chat_turn(
        &[ChatTurn {
            role: "user",
            content: "hello",
        }],
        &mut |_| {},
    )
    .unwrap();
    let mut out = Vec::new();
    s.write_timing(&mut out).unwrap();
    let csv = String::from_utf8(out).unwrap();
    // Rows only; the column names come from the recorder's own header.
    let mut hdr = Vec::new();
    golem_engine::timing::TimingRecorder::header(&mut hdr, "", "", "").unwrap();
    let hdr = String::from_utf8(hdr).unwrap();
    let header: Vec<&str> = hdr.lines().nth(1).unwrap().split(',').collect();
    let idx = |name: &str| header.iter().position(|h| *h == name).unwrap();
    let rows: Vec<Vec<&str>> = csv
        .lines()
        .filter(|l| !l.starts_with('#') && !l.starts_with("step"))
        .map(|l| l.split(',').collect())
        .collect();
    assert!(!rows.is_empty(), "{csv}");
    for r in &rows {
        for c in ["policy_ms", "ids_sync_ms", "ids_gap_ms", "ids_gap_max_ms"] {
            assert!(r[idx(c)].parse::<f64>().is_ok(), "{c}: {csv}");
        }
        let f = |c: &str| r[idx(c)].parse::<f64>().unwrap();
        assert!(f("ids_gap_max_ms") <= f("ids_gap_ms"), "max <= sum: {csv}");
        assert!(f("policy_ms") <= f("wall_ms"), "policy within wall: {csv}");
        // One readback per hooked MoE layer per decoded token.
        let moe_layers = u64::from(dims::N_LAYER - dims::N_LAYER_DENSE_LEAD);
        let tokens: u64 = r[idx("tokens")].parse().unwrap();
        let n: u64 = r[idx("ids_readbacks")].parse().unwrap();
        assert_eq!(
            n,
            moe_layers * tokens,
            "one readback per MoE layer per step: {csv}"
        );
    }
    let _ = std::fs::remove_file(&csv_path);
}
