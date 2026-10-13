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
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    let unkeyed = golem(
        &rt,
        &[
            "run",
            &m,
            "--prompt",
            "hi",
            "--n-predict",
            "2",
            "--allow-overcommit",
        ],
    );
    assert!(golem(&rt, &["stop", &id]).status.success());
    assert!(out.status.success(), "{err}");
    assert!(err.contains(&format!("attached to instance {id}")), "{err}");
    assert!(!out.stdout.is_empty(), "no reply: {err}");
    let e2 = String::from_utf8_lossy(&unkeyed.stderr);
    assert!(!unkeyed.status.success(), "{e2}");
    assert!(
        e2.contains(&format!("instance {id} requires an API key")) && e2.contains("GOLEM_API_KEY"),
        "{e2}"
    );
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

fn wait_for_ids(rt: &Path) -> Vec<String> {
    let t = std::time::Instant::now();
    loop {
        let v = ids(rt);
        if !v.is_empty() {
            return v;
        }
        assert!(t.elapsed().as_secs() < 60, "no registry entry within 60 s");
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

/// #91: `golem stop` on an in-process `run` blocked on idle stdin: the
/// SIGTERM handler wakes the REPL, which returns through the normal drops.
#[test]
fn stop_in_process_run_with_idle_stdin_exits_normally() {
    use std::os::unix::process::ExitStatusExt;
    let rt = common::tempdir_env();
    let mut child = common::golem_cmd(&rt)
        .args(["run", &tiny_moe(), "--new", "--allow-overcommit"])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let _stdin = child.stdin.take();
    let id = wait_for_ids(&rt).remove(0);
    let st = golem(&rt, &["stop", &id]);
    if !st.status.success() {
        let _ = child.kill();
    }
    let out = child.wait_with_output().unwrap();
    assert!(
        st.status.success(),
        "{}",
        String::from_utf8_lossy(&st.stderr)
    );
    assert!(
        out.status.signal().is_none(),
        "killed by signal {:?}: {}",
        out.status.signal(),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(ids(&rt).is_empty());
}

/// Two lines piped at once are both answered while stdin stays open: the
/// reader serves buffered complete lines before polling stdin again.
#[test]
fn run_repl_answers_two_lines_piped_at_once() {
    use std::io::BufRead;
    let rt = common::tempdir_env();
    let mut child = common::golem_cmd(&rt)
        .args([
            "run",
            &tiny_moe(),
            "--new",
            "--n-predict",
            "2",
            "--allow-overcommit",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    stdin.write_all(b"hi\nhello\n").unwrap();
    let stdout = child.stdout.take().unwrap();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        for l in std::io::BufReader::new(stdout).lines() {
            let _ = tx.send(l);
        }
    });
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    let mut replies = 0;
    while replies < 2 {
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        match rx.recv_timeout(left) {
            Ok(_) => replies += 1,
            Err(_) => break,
        }
    }
    drop(stdin);
    if replies < 2 {
        let _ = child.kill();
    }
    let status = child.wait().unwrap();
    assert_eq!(replies, 2, "second piped line not answered");
    assert!(status.success());
}

/// A registered `serve` entry for `pid` with no control socket.
fn fake_serve(rt: &Path, id: &str, pid: u32, heartbeat_age: u64) {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let i = golem_registry::Instance {
        id: id.into(),
        pid,
        pid_start_time: golem_registry::pid_start_time(pid).unwrap(),
        kind: "serve".into(),
        endpoint: String::new(),
        model: "m".into(),
        config_hash: "h".into(),
        started_at: now,
        planned_vram: 0,
        planned_ram: 0,
        health: "healthy".into(),
        heartbeat_at: now - heartbeat_age,
        children: vec![],
        config: None,
    };
    let d = rt.join("run/golem/instances");
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(
        d.join(format!("{id}.json")),
        serde_json::to_vec(&i).unwrap(),
    )
    .unwrap();
}

/// #91: a SIGSTOPped instance with a stale heartbeat and no socket is
/// killed by a plain `stop`; a fresh heartbeat with a dead socket is not.
#[test]
fn stop_kills_hung_instance_but_not_live_one() {
    let rt = common::tempdir_env();
    let mut hung = std::process::Command::new("sleep")
        .arg("120")
        .spawn()
        .unwrap();
    let mut live = std::process::Command::new("sleep")
        .arg("120")
        .spawn()
        .unwrap();
    let stopped = std::process::Command::new("kill")
        .args(["-STOP", &hung.id().to_string()])
        .status()
        .unwrap();
    assert!(stopped.success());
    fake_serve(&rt, "hung", hung.id(), 60);
    fake_serve(&rt, "live", live.id(), 0);
    let st = golem(&rt, &["stop", "live"]);
    let live_alive = live.try_wait().unwrap().is_none();
    let sh = golem(&rt, &["stop", "hung"]);
    let hung_status = hung.try_wait().unwrap();
    let _ = (live.kill(), hung.kill());
    let _ = (live.wait(), hung.wait());
    assert!(!st.status.success());
    assert!(live_alive, "live instance killed");
    let err = String::from_utf8_lossy(&sh.stderr);
    assert!(sh.status.success(), "{err}");
    assert!(err.contains("golem: hung is hung; killing"), "{err}");
    assert!(hung_status.is_some(), "hung instance survived");
}
