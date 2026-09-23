//! Gated acceptance: the generated tiny-moe GGUF loads in the pinned substrate.

use std::process::Command;

const PIN: &str = "86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9";

/// Removes the temp dir on drop — cleanup runs even when an assert fails.
struct Cleanup(std::path::PathBuf);

impl Drop for Cleanup {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0); // temp hygiene: best-effort is fine
    }
}

fn build_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../vendor/llama.cpp/build")
}

/// Env override wins; a set-but-missing path is fatal (no silent fallback).
/// Otherwise: the canonical vendored build dir, if built.
fn substrate_cli() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("GOLEM_SUBSTRATE_LLAMA_CLI") {
        let p = std::path::PathBuf::from(p);
        assert!(
            p.is_file(),
            "GOLEM_SUBSTRATE_LLAMA_CLI={p:?} is not a file — build the substrate or unset the var"
        );
        return Some(p);
    }
    let rel = build_dir().join("bin/llama-cli");
    rel.is_file().then_some(rel)
}

/// Pin stamp proves the canonical vendored build came from the pinned tree.
/// Mismatch is fatal; an absent stamp (pre-stamp build) only notes and proceeds.
fn check_pin_stamp() {
    let stamp = build_dir().join("substrate-pin.txt");
    match std::fs::read_to_string(&stamp) {
        Ok(s) => {
            let got = s.trim();
            assert!(
                got == PIN,
                "substrate-pin.txt records {got:?}, pinned tree is {PIN} — rebuild via scripts/build-substrate.sh"
            );
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!(
                "note: {} absent — build predates the pin stamp; proceeding unverified",
                stamp.display()
            );
        }
        Err(e) => panic!("read {}: {e}", stamp.display()),
    }
}

#[test]
fn generated_file_loads_in_pinned_substrate() {
    let Some(cli) = substrate_cli() else {
        if std::env::var("GOLEM_REQUIRE_SUBSTRATE").is_ok_and(|v| v == "1") {
            panic!(
                "GOLEM_REQUIRE_SUBSTRATE=1 but no substrate llama-cli — run scripts/build-substrate.sh or set GOLEM_SUBSTRATE_LLAMA_CLI"
            );
        }
        eprintln!(
            "skipping: substrate llama-cli not built (set GOLEM_SUBSTRATE_LLAMA_CLI to enable; GOLEM_REQUIRE_SUBSTRATE=1 makes absence fatal)"
        );
        eprintln!(
            "SKIP IS NOT ACCEPTANCE EVIDENCE — acceptance requires a real run against the pinned substrate"
        );
        return;
    };
    // Override binary: the vendored pin stamp says nothing about it — skip the
    // stamp check and put provenance on the operator (a stale vendored build's
    // stamp failure would otherwise accuse the wrong binary).
    if std::env::var_os("GOLEM_SUBSTRATE_LLAMA_CLI").is_some() {
        eprintln!(
            "note: GOLEM_SUBSTRATE_LLAMA_CLI set — vendored pin stamp not checked; \
             provenance of the override binary is the operator's responsibility"
        );
    } else {
        check_pin_stamp();
    }
    let dir = std::env::temp_dir().join(format!("golem-m12-load-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let _cleanup = Cleanup(dir.clone());
    let model = dir.join("tiny-moe.gguf");
    let st = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .args(["gen-tiny-moe", "--out", model.to_str().unwrap()])
        .status()
        .unwrap();
    assert!(st.success());
    let out = Command::new(&cli)
        .args([
            "-m",
            model.to_str().unwrap(),
            "-p",
            "hello",
            "-n",
            "8",
            "--temp",
            "0",
            "-c",
            "64",
            "--no-display-prompt",
            "--simple-io",
            "-st",
            "-v",
            "--no-warmup",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "llama-cli failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("glm4moe"), "arch not reported:\n{stderr}");
    assert!(
        regex::Regex::new(r"n_expert\s*=\s*8")
            .unwrap()
            .is_match(&stderr),
        "expert count not reported:\n{stderr}"
    );
}
