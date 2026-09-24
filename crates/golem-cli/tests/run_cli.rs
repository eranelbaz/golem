//! End-to-end: `golem run --greedy tiny-moe.gguf --prompt hello` completes.
//! Shares the tiny-moe fixture via the same .test-cache convention.
mod common; // path resolution copy: engine tests/ modules aren't importable across crates

#[test]
fn run_greedy_one_shot_completes() {
    let model = common::tiny_moe_path();
    let bin = env!("CARGO_BIN_EXE_golem");
    let out = std::process::Command::new(bin)
        .args(["run", "--greedy"])
        .arg(&model)
        .args(["--prompt", "hello", "--n-predict", "8"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!out.stdout.is_empty(), "no assistant output on stdout");
}

#[test]
fn run_missing_model_errors_with_code_1() {
    let bin = env!("CARGO_BIN_EXE_golem");
    let out = std::process::Command::new(bin)
        .args(["run", "--greedy", "/nonexistent/model.gguf"])
        .args(["--prompt", "hi"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("error:"));
}
