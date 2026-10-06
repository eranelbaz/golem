//! §11 per-step timing CSV format tests (pure formatting; no GPU, no model).

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
        disk_ms: 55.0,
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
fn compute_is_wall_minus_measured_parts() {
    let mut r = TimingRecorder::new(64);
    // 100 − (60 + 4.5) = 35.5: only measured columns are subtracted.
    r.row(StepTiming {
        step: 0,
        tokens: 1,
        wall_ms: 100.0,
        p3_wait_ms: 60.0,
        disk_ms: f64::NAN,
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
