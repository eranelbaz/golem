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
    run_greedy_args(ram_experts_gib, &[])
}

fn run_greedy_args(ram_experts_gib: &str, extra: &[&str]) -> std::process::Output {
    let model = common::tiny_moe_path();
    common::golem_isolated()
        .args([
            "run",
            "--greedy",
            "--n-predict",
            "32",
            "--n-gpu-layers",
            "99",
            "--cpu-moe-layers",
            "99",
            "--ram-experts-gib",
            ram_experts_gib,
        ])
        .args(extra)
        .arg(&model)
        .args([
            "--prompt",
            // Routing breadth: long and lexically varied so greedy decode
            // plausibly routes more than K = 4 distinct experts per touched
            // layer (tiny-moe top-k = 2) — a short prompt could leave the
            // K = 4 budget unexercised and the comparison vacuous. The
            // eviction/counter behavior itself is asserted in golem-engine's
            // `budget_tiny_moe`, where the counters are readable.
            "MoE routing probe: the quick brown fox jumps over the lazy dog \
while numbers 1 2 3 4 5 6 7 8 9 march past a completely different \
subject line",
        ])
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

/// Exclusive residency (O10) output parity: with a budgeted device cache
/// (K = 4 < tiny-moe n_expert = 8, so slab evictions + the eviction D2H path
/// engage), `--residency exclusive` must decode the same greedy text as the
/// inclusive baseline — evicted slab bytes must reach the pool for all three
/// planes (the multi-segment D2H deferral is covered exactly by the vendor
/// ctest golem-p2-evict.cpp; this leg covers the end-to-end CLI mode).
#[test]
fn exclusive_residency_output_matches_inclusive() {
    assert_eq!(
        golem_llama_sys::substrate_backend(),
        golem_llama_sys::Backend::Hip,
        "hip-tests requires a HIP-built substrate (GOLEM_LLAMA_SYS_BACKEND=hip)"
    );
    let excl = run_greedy_args(
        "1",
        &["--device-cache-experts", "4", "--residency", "exclusive"],
    );
    let incl = run_greedy_args(
        "1",
        &["--device-cache-experts", "4", "--residency", "inclusive"],
    );
    assert!(
        excl.status.success(),
        "exclusive run failed: {}",
        String::from_utf8_lossy(&excl.stderr)
    );
    assert!(
        incl.status.success(),
        "inclusive run failed: {}",
        String::from_utf8_lossy(&incl.stderr)
    );
    assert!(!excl.stdout.is_empty(), "exclusive run produced no output");
    assert_eq!(
        incl.stdout,
        excl.stdout,
        "exclusive residency must not change greedy output vs inclusive: \
         incl={:?} excl={:?}",
        String::from_utf8_lossy(&incl.stdout),
        String::from_utf8_lossy(&excl.stdout)
    );
}

/// Cross-budget output parity (review O6/Q5): a device cache budgeted to
/// K = 4 experts (< tiny-moe n_expert = 8, so the slab evicts under routing
/// breadth) must produce the same greedy output as the K = 8 (K ≥ n_expert)
/// baseline. The eviction/counter behavior itself is asserted in
/// golem-engine's `budget_tiny_moe`; this needs one HIP process per budget,
/// which only the CLI subprocesses provide (one pool-enabled session per
/// process). Cost: two more HIP sessions per run.
#[test]
fn device_cache_budget_k4_matches_k8() {
    assert_eq!(
        golem_llama_sys::substrate_backend(),
        golem_llama_sys::Backend::Hip,
        "hip-tests requires a HIP-built substrate (GOLEM_LLAMA_SYS_BACKEND=hip)"
    );
    let k4 = run_greedy_args("1", &["--device-cache-experts", "4"]);
    let k8 = run_greedy_args("1", &["--device-cache-experts", "8"]);
    assert!(
        k4.status.success(),
        "K = 4 run failed: {}",
        String::from_utf8_lossy(&k4.stderr)
    );
    assert!(
        k8.status.success(),
        "K = 8 run failed: {}",
        String::from_utf8_lossy(&k8.stderr)
    );
    assert!(!k4.stdout.is_empty(), "K = 4 run produced no output");
    assert_eq!(
        k4.stdout,
        k8.stdout,
        "a budgeted device cache (K = 4 < n_expert) must not change greedy \
         output vs the K = 8 baseline: k4={:?} k8={:?}",
        String::from_utf8_lossy(&k4.stdout),
        String::from_utf8_lossy(&k8.stdout)
    );
}
