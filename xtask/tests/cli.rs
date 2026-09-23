//! Integration tests for the `gen-tiny-moe` CLI subcommand.

use std::process::Command;

/// Removes the temp dir on drop — cleanup runs even when an assert fails.
struct Cleanup(std::path::PathBuf);

impl Drop for Cleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0); // temp hygiene: best-effort is fine
    }
}

#[test]
fn gen_tiny_moe_writes_deterministic_file() {
    let dir = std::env::temp_dir().join(format!("golem-m12-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let _cleanup = Cleanup(dir.clone());
    let out = dir.join("tiny-moe.gguf");
    let st = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["gen-tiny-moe", "--out", out.to_str().unwrap()])
        .status()
        .unwrap();
    assert!(st.success());
    let first = std::fs::read(&out).unwrap();
    let st = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["gen-tiny-moe", "--out", out.to_str().unwrap()])
        .status()
        .unwrap();
    assert!(st.success());
    assert_eq!(
        std::fs::read(&out).unwrap(),
        first,
        "output must be byte-deterministic"
    );
    let meta = std::fs::metadata(&out).unwrap();
    assert!(
        meta.len() > 100_000 && meta.len() < 1_000_000,
        "unexpected size {}",
        meta.len()
    );
}

#[test]
fn unknown_command_still_exits_2() {
    let st = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .arg("nope")
        .status()
        .unwrap();
    assert_eq!(st.code(), Some(2));
}

#[test]
fn bad_flag_exits_2() {
    let cases: [&[&str]; 2] = [
        &["gen-tiny-moe", "--bogus"],
        &["gen-tiny-moe", "--seed", "zzz"],
    ];
    for args in cases {
        let st = Command::new(env!("CARGO_BIN_EXE_xtask"))
            .args(args)
            .status()
            .unwrap();
        assert_eq!(
            st.code(),
            Some(2),
            "bad args {args:?} must exit 2, not panic"
        );
    }
}
