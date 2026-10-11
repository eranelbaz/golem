//! Instance lifecycle on tiny-moe: `serve --detach` → `list` → `stop`,
//! `run` attach / `--keep`, and a detached load failure.
mod common;

use std::io::Write;
use std::path::Path;
use std::process::{Output, Stdio};

fn golem(rt: &Path, args: &[&str]) -> Output {
    common::golem_cmd(rt).args(args).output().unwrap()
}

fn tiny_moe() -> String {
    common::tiny_moe_path().to_string_lossy().into_owned()
}

fn ids(rt: &Path) -> Vec<String> {
    let out = golem(rt, &["list"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .skip(1)
        .filter(|l| !l.contains("unregistered"))
        .filter_map(|l| l.split_whitespace().next().map(str::to_owned))
        .collect()
}

#[test]
fn detach_list_stop_round_trip() {
    let rt = common::tempdir_env();
    let m = tiny_moe();
    let out = golem(
        &rt,
        &["serve", &m, "--detach", "--port", "0", "--allow-overcommit"],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let id = String::from_utf8(out.stdout)
        .unwrap()
        .split_whitespace()
        .next()
        .unwrap()
        .to_owned();
    let list = golem(&rt, &["list"]);
    assert!(String::from_utf8_lossy(&list.stdout).contains(&id));
    let sock = ["run/golem/instances", "state/golem/instances"]
        .iter()
        .map(|d| rt.join(d).join(format!("{id}.sock")))
        .find(|p| p.exists())
        .expect("control socket");
    use std::os::unix::fs::PermissionsExt;
    let mode = std::fs::metadata(&sock).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);
    let st = golem(&rt, &["stop", &id]);
    assert!(
        st.status.success(),
        "{}",
        String::from_utf8_lossy(&st.stderr)
    );
    assert!(!String::from_utf8_lossy(&golem(&rt, &["list"]).stdout).contains(&id));
    let e = golem(&rt, &["stop", &id]);
    assert!(!e.status.success());
    assert!(String::from_utf8_lossy(&e.stderr).contains("golem_instance_not_found"));
}

#[test]
fn run_attaches_to_compatible_instance_and_keep_leaves_server() {
    let rt = common::tempdir_env();
    let m = tiny_moe();
    let out = golem(
        &rt,
        &[
            "serve",
            &m,
            "--detach",
            "--ctx",
            "4096",
            "--allow-overcommit",
        ],
    );
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let id = String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .unwrap()
        .to_owned();

    // Smaller ctx fits inside the running instance → attach, no new entry.
    let mut child = common::golem_cmd(&rt)
        .args([
            "run",
            &m,
            "--ctx",
            "2048",
            "--n-predict",
            "4",
            "--allow-overcommit",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"hi\n/exit\n")
        .unwrap();
    let out = child.wait_with_output().unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{err}");
    assert!(err.contains(&format!("attached to instance {id}")), "{err}");
    assert!(!out.stdout.is_empty(), "no reply: {err}");
    assert_eq!(ids(&rt), std::slice::from_ref(&id));
    assert!(golem(&rt, &["stop", &id]).status.success());

    // Nothing running: --keep starts a detached server that outlives run.
    let out = golem(
        &rt,
        &[
            "run",
            &m,
            "--keep",
            "--prompt",
            "hi",
            "--n-predict",
            "2",
            "--allow-overcommit",
        ],
    );
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{err}");
    assert!(
        err.contains("attached to instance") || err.contains("started instance"),
        "{err}"
    );
    let left = ids(&rt);
    assert_eq!(left.len(), 1, "{left:?}");
    assert!(golem(&rt, &["stop", &left[0]]).status.success());
    assert!(ids(&rt).is_empty());
}

/// A keyed instance: `run` resolves `GOLEM_API_KEY` like `serve` and sends
/// it as a bearer token, so the attach succeeds instead of a 401.
#[test]
fn run_attaches_to_keyed_instance() {
    let rt = common::tempdir_env();
    let m = tiny_moe();
    let out = common::golem_cmd(&rt)
        .env("GOLEM_API_KEY", "k3y")
        .args(["serve", &m, "--detach", "--allow-overcommit"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let id = String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .unwrap()
        .to_owned();
    let out = common::golem_cmd(&rt)
        .env("GOLEM_API_KEY", "k3y")
        .args([
            "run",
            &m,
            "--prompt",
            "hi",
            "--n-predict",
            "2",
            "--allow-overcommit",
        ])
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(golem(&rt, &["stop", &id]).status.success());
    assert!(out.status.success(), "{err}");
    assert!(err.contains(&format!("attached to instance {id}")), "{err}");
    assert!(!out.stdout.is_empty(), "no reply: {err}");
}

#[test]
fn detach_reports_child_load_failure() {
    let rt = common::tempdir_env();
    let out = golem(&rt, &["serve", "/nonexistent.gguf", "--detach"]);
    assert!(!out.status.success()); // returns, does not hang
    assert!(String::from_utf8_lossy(&out.stderr).contains("error"));
    assert!(ids(&rt).is_empty());
}

#[test]
fn detach_load_timeout_kills_grandchild() {
    let rt = common::tempdir_env();
    let m = tiny_moe();
    let out = golem(
        &rt,
        &[
            "serve",
            &m,
            "--detach",
            "--port",
            "0",
            "--load-timeout",
            "0",
        ],
    );
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    let pid: u32 = err
        .split("killed instance pid ")
        .nth(1)
        .and_then(|s| s.split_whitespace().next())
        .and_then(|p| p.parse().ok())
        .unwrap_or_else(|| panic!("{err}"));
    assert!(
        golem_registry::pid_start_time(pid).is_err(),
        "pid {pid} survived"
    );
    assert!(ids(&rt).is_empty());
}
