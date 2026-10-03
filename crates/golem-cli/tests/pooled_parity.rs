//! The RAM expert pool (P3) must not change decode output —
//! the same greedy prompt through the CLI binary is byte-identical with the
//! pool off (`--ram-experts-gib 0`) and on (`--ram-experts-gib 1`). The
//! P2/P3 hooks are process-global, so the comparison needs a second PROCESS
//! (`p3_counters` dropped its in-process comparison for exactly that
//! reason); integration tests here see the built `golem` binary via
//! `CARGO_BIN_EXE_golem`, which `golem-engine` tests cannot resolve.
//! Runs only where a GPU split exists (feature `hip-tests`, HIP CI job);
//! on CPU builds the split-scheduler expert-copy path never engages and the
//! pool is inert, so the comparison would prove nothing.
#![cfg(feature = "hip-tests")]
mod common;

fn run_greedy(ram_experts_gib: &str) -> std::process::Output {
    let model = common::tiny_moe_path();
    std::process::Command::new(env!("CARGO_BIN_EXE_golem"))
        .args([
            "run",
            "--greedy",
            "--n-predict",
            "16",
            "--n-gpu-layers",
            "99",
            "--cpu-moe-layers",
            "99",
            "--ram-experts-gib",
            ram_experts_gib,
        ])
        .arg(&model)
        .args(["--prompt", "Hello"])
        .output()
        .unwrap()
}

#[test]
fn pooled_and_unpooled_output_are_byte_identical() {
    // On a CPU substrate the split-scheduler path never engages and the
    // comparison would pass vacuously: fail loudly instead.
    assert_eq!(
        golem_llama_sys::substrate_backend(),
        golem_llama_sys::Backend::Hip,
        "hip-tests requires a HIP-built substrate (GOLEM_LLAMA_SYS_BACKEND=hip)"
    );
    let off = run_greedy("0");
    let on = run_greedy("1");
    assert!(
        off.status.success(),
        "pool-off run failed: {}",
        String::from_utf8_lossy(&off.stderr)
    );
    assert!(
        on.status.success(),
        "pool-on run failed: {}",
        String::from_utf8_lossy(&on.stderr)
    );
    assert!(!off.stdout.is_empty(), "pool-off run produced no output");
    assert_eq!(
        off.stdout,
        on.stdout,
        "the RAM expert pool must not change greedy output (§7.4): \
         off={:?} on={:?}",
        String::from_utf8_lossy(&off.stdout),
        String::from_utf8_lossy(&on.stdout)
    );
}
