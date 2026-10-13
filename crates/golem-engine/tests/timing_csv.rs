//! §11 per-step timing CSV format tests (pure formatting; no GPU, no model).

#[cfg(feature = "hip-tests")]
mod common;

use golem_engine::timing::{StepTiming, TimingRecorder};

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
}

#[test]
fn rows_write_unmeasured_for_unmeasured_columns_never_zero() {
    let mut r = TimingRecorder::new(64);
    r.row(StepTiming {
        step: 1,
        tokens: 1,
        wall_ms: 100.0,
        p3_wait_ms: 60.0,
        disk_worker_ms: 55.0,
        disk_bytes: 121_634_816,
        h2d_ms: f64::NAN,
        h2d_bytes: 0,
        policy_ms: f64::NAN,
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
    assert!(!line.contains(",0,")); // no silent zero for unmeasured
    assert!(line.contains("55") && line.contains("121634816"));
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
        disk_worker_ms: f64::NAN,
        disk_bytes: 0,
        h2d_ms: f64::NAN,
        h2d_bytes: 0,
        policy_ms: 4.5,
    });
    let mut out = Vec::new();
    r.flush(&mut out).unwrap();
    let line = std::str::from_utf8(&out).unwrap().lines().last().unwrap();
    assert!(line.contains("35.5"), "row: {line}");
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
            timing_csv: Some(std::env::temp_dir().join("golem-unused.csv")),
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
    // columns: step,tokens,wall_ms,p3_wait_ms,disk_worker_ms,disk_bytes,h2d_ms,...
    let col = |csv: &str, i: usize| -> Vec<String> {
        csv.lines()
            .filter(|l| !l.starts_with('#') && !l.starts_with("step"))
            .map(|l| l.split(',').nth(i).unwrap().to_owned())
            .collect()
    };
    let csv_on = rows(true);
    let on = col(&csv_on, 6);
    assert!(!on.is_empty());
    assert!(on.iter().all(|v| v.parse::<f64>().is_ok()), "{on:?}");
    // The copy stream actually carried bytes (not just a zero reading).
    let bytes: u64 = col(&csv_on, 7)
        .iter()
        .map(|v| v.parse::<u64>().unwrap())
        .sum();
    assert!(bytes > 0, "{csv_on}");
    let off = col(&rows(false), 6);
    assert!(!off.is_empty());
    assert!(off.iter().all(|v| v == "unmeasured"), "{off:?}");
}
