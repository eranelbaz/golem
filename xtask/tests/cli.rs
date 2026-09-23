//! Integration tests for the `gen-tiny-moe` CLI subcommand.

use std::process::Command;

#[allow(dead_code)]
mod common;

use common::temp_dir;

#[test]
fn gen_tiny_moe_writes_deterministic_file() {
    let _cleanup = temp_dir("m12");
    let dir = _cleanup.0.clone();
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
