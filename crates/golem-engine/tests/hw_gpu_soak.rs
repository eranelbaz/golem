//! HW soak: back-to-back turns on the real model while the compositor keeps
//! rendering, then the kernel log since the start must hold no amdgpu ring
//! timeout, GPU reset or VRAM-exhaustion line. Linux-only: reads
//! `journalctl -k`. Prints `HW soak turns=… secs=…`.
#![cfg(all(feature = "hip-tests", target_os = "linux"))]
use golem_engine::config::EngineConfig;
use golem_engine::{ChatTurn, Session, TurnRequest};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const GPU_FAULTS: &[&str] = &[
    "amdgpu_job_timedout",
    "GPU reset",
    "VRAM is lost",
    "MES failed",
    "Not enough memory for command submission",
    "Failed to pin framebuffer",
];

fn is_fault(l: &str) -> bool {
    let gpu = l.contains("amdgpu")
        && ((l.contains("ring ") && l.contains("timeout"))
            || GPU_FAULTS.iter().any(|f| l.contains(f)));
    gpu || l.contains("Out of memory") || l.contains("oom-kill")
}

#[test]
#[ignore = "real model; GOLEM_HW=1 GOLEM_HW_MODEL=<shard 1> [GOLEM_HW_SOAK_SECS=3600] --run-ignored"]
fn hw_soak_no_gpu_reset() {
    assert_eq!(
        std::env::var("GOLEM_HW").as_deref(),
        Ok("1"),
        "set GOLEM_HW=1"
    );
    let model: std::path::PathBuf = std::env::var("GOLEM_HW_MODEL")
        .expect("GOLEM_HW_MODEL must name shard 1")
        .into();
    let secs: u64 = std::env::var("GOLEM_HW_SOAK_SECS")
        .map(|s| s.parse().expect("GOLEM_HW_SOAK_SECS"))
        .unwrap_or(3600);
    let since = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let cfg = EngineConfig {
        model_path: model.clone(),
        n_ctx: 4096,
        n_gpu_layers: 99,
        flash_attn_on: true,
        ram_experts_gib: 57,
        manifest_path: Some(model.with_file_name("manifest.json")),
        prefill_stream: true,
        ..EngineConfig::default()
    };
    let long = "the quick brown fox jumps over the lazy dog ".repeat(230);
    let msgs = [ChatTurn {
        role: "user",
        content: &long,
    }];
    let mut s = Session::new(&cfg).expect("real-model session");
    let t = Instant::now();
    let mut turns = 0u64;
    let mut failed = None;
    while t.elapsed() < Duration::from_secs(secs) {
        if let Err(e) = s.turn(&TurnRequest::new(&msgs).max_tokens(Some(256)), &mut |_| {}) {
            failed = Some(e);
            break;
        }
        turns += 1;
    }
    drop(s);
    println!("HW soak turns={turns} secs={}", t.elapsed().as_secs());
    let out = std::process::Command::new("journalctl")
        .args(["-k", "--no-pager", "-o", "short-iso", "--since"])
        .arg(format!("@{since}"))
        .output()
        .expect("journalctl");
    assert!(
        out.status.success(),
        "journalctl: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let log = String::from_utf8_lossy(&out.stdout);
    let probe = std::process::Command::new("journalctl")
        .args(["-k", "--no-pager", "-n1", "-q"])
        .output()
        .expect("journalctl");
    assert!(
        !probe.stdout.is_empty(),
        "kernel journal unreadable: join systemd-journal, adm or wheel"
    );
    let hits: Vec<&str> = log.lines().filter(|l| is_fault(l)).collect();
    assert!(
        hits.is_empty(),
        "GPU faults during soak:\n{}",
        hits.join("\n")
    );
    assert!(failed.is_none(), "turn {turns} failed: {failed:?}");
}
