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

/// The REPL is multi-turn: each prompt replays the accumulated history, so
/// turn 2's prefill is strictly longer than turn 1's. `/reset` drops it and
/// `/exit` ends the session before EOF.
#[test]
fn repl_accumulates_history_until_reset() {
    use std::io::Write as _;
    let model = common::tiny_moe_path();
    let bin = env!("CARGO_BIN_EXE_golem");
    let mut child = std::process::Command::new(bin)
        .args(["run", "--greedy"])
        .arg(&model)
        .args(["--n-predict", "4", "--timing-csv", "/dev/stdout"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"first\nsecond\n/reset\nthird\n/exit\n")
        .unwrap();
    let out = child.wait_with_output().unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stderr: {stderr}");
    assert!(stderr.contains("history cleared"), "stderr: {stderr}");
    // Four prompts reached the model (3 turns + none for /reset or /exit):
    // one "golem> " per read, plus the final prompt before /exit returns.
    assert_eq!(stderr.matches("golem> ").count(), 5, "stderr: {stderr}");
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
