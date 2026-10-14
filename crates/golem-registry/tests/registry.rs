//! Registry: reaping, O5 admission, flock atomicity.
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use golem_registry::{Instance, Registry, admit};

fn inst(id: &str, pid: u32, hb: u64) -> Instance {
    Instance {
        id: id.into(),
        pid,
        pid_start_time: golem_registry::pid_start_time(pid).unwrap_or(0),
        kind: "serve".into(),
        endpoint: "http://127.0.0.1:1".into(),
        model: "m".into(),
        config_hash: "h".into(),
        started_at: 0,
        children: vec![],
        planned_vram: 20 << 30,
        planned_ram: 60 << 30,
        health: "healthy".into(),
        heartbeat_at: hb,
        config: None,
    }
}

/// Kills and reaps a test-spawned process even when an assert fails first.
struct Reaped(std::process::Child);

impl Drop for Reaped {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

impl std::ops::Deref for Reaped {
    type Target = std::process::Child;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Reaped {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// SIGKILLs a grandchild on drop while `(pid, start time)` still names it.
struct KillPid(u32, u64);

impl Drop for KillPid {
    fn drop(&mut self) {
        if golem_registry::pid_start_time(self.0).is_ok_and(|t| t == self.1) {
            let _ = std::process::Command::new("kill")
                .args(["-KILL", &self.0.to_string()])
                .status();
        }
    }
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

fn tempfile_dir() -> PathBuf {
    static N: AtomicU32 = AtomicU32::new(0);
    let d = std::env::temp_dir()
        .join(format!(
            "golem-reg-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ))
        .join("instances");
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn machine() -> golem_plan::Machine {
    golem_plan::Machine {
        vram_total: 24 << 30,
        ram_total: 93 << 30,
        ram_free: 90 << 30,
        rocm_overhead: 0,
    }
}

fn write(d: &std::path::Path, i: &Instance) {
    std::fs::write(
        d.join(format!("{}.json", i.id)),
        serde_json::to_vec(i).unwrap(),
    )
    .unwrap();
}

#[test]
fn stale_entries_reaped_live_kept() {
    let d = tempfile_dir();
    let r = Registry::at(d.clone(), d.parent().unwrap().join("admit.lock"));
    let _g = r
        .register(&inst("live", std::process::id(), now()))
        .unwrap();
    write(&d, &inst("dead", 999_999, now()));
    let ids: Vec<_> = r.list_live().unwrap().into_iter().map(|i| i.id).collect();
    assert_eq!(ids, ["live"]);
    assert!(!d.join("dead.json").exists());
}

#[test]
fn guard_drop_removes_entry_and_heartbeat_advances() {
    let d = tempfile_dir();
    let r = Registry::at(d.clone(), d.parent().unwrap().join("admit.lock"));
    let g = r.register(&inst("g", std::process::id(), 0)).unwrap();
    g.update(|i| i.health = "degraded".into());
    let i = r.find("g").unwrap();
    assert_eq!(i.health, "degraded");
    assert!(now() - i.heartbeat_at <= 1);
    drop(g);
    assert!(!d.join("g.json").exists());
    assert!(matches!(
        r.find("g"),
        Err(golem_registry::RegistryError::InstanceNotFound(_))
    ));
}

#[test]
fn second_instance_blocked_unless_overcommit() {
    let m = machine();
    let running = [inst("a", 1, now())]; // 20 GiB VRAM, 60 GiB RAM
    let e = admit(&running, 20 << 30, 60 << 30, &m, false).unwrap_err();
    assert_eq!(e.error.code, golem_api::ErrorCode::PlanInfeasible);
    assert!(
        e.error.message.contains("instance `a`"),
        "{}",
        e.error.message
    );
    assert!(admit(&running, 20 << 30, 60 << 30, &m, true).is_ok());
    // RAM-only overrun: VRAM 20+2 ≤ 24 fits, RAM 60+40 > 93 does not
    assert!(admit(&running, 2 << 30, 40 << 30, &m, false).is_err());
    assert!(admit(&running, 2 << 30, 30 << 30, &m, false).is_ok());
}

#[test]
fn concurrent_admits_cannot_both_pass() {
    let base = tempfile_dir();
    let lock = base.parent().unwrap().join("admit.lock");
    // One thread per dir: the shared lock, not the dir, serializes them.
    let dirs = [base.clone(), base.parent().unwrap().join("fallback")];
    let mut new = inst("x", std::process::id(), now());
    new.health = "loading".into();
    let results: Vec<_> = std::thread::scope(|s| {
        let hs: Vec<_> = dirs
            .iter()
            .enumerate()
            .map(|(n, d)| {
                let r = Registry::at(d.clone(), lock.clone()).with_read_dir(dirs[1 - n].clone());
                let mut i = new.clone();
                i.id = format!("t{n}");
                s.spawn(move || r.admit_and_register(&i, &machine(), false))
            })
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let ok = results.iter().filter(|r| r.is_ok()).count();
    assert_eq!(ok, 1);
    let e = results.iter().find_map(|r| r.as_ref().err()).unwrap();
    assert_eq!(e.error.code, golem_api::ErrorCode::PlanInfeasible);
    assert!(e.error.message.contains("loading"), "{}", e.error.message);
}

#[test]
fn loading_entry_with_old_heartbeat_is_not_reaped() {
    let d = tempfile_dir();
    let r = Registry::at(d.clone(), d.parent().unwrap().join("admit.lock"));
    let mut i = inst("ld", std::process::id(), now() - 60);
    i.health = "loading".into();
    write(&d, &i);
    let live = r.list_live().unwrap();
    assert_eq!(live.len(), 1);
    assert!(admit(&live, 20 << 30, 1 << 30, &machine(), false).is_err());
}

#[test]
fn reused_pid_with_other_start_time_is_reaped() {
    let d = tempfile_dir();
    let r = Registry::at(d.clone(), d.parent().unwrap().join("admit.lock"));
    let mut i = inst("reuse", std::process::id(), now());
    i.pid_start_time = 1;
    write(&d, &i);
    std::fs::write(d.join("reuse.sock"), b"").unwrap();
    assert!(r.list_live().unwrap().is_empty());
    assert!(!d.join("reuse.json").exists() && !d.join("reuse.sock").exists());
}

#[test]
fn compatible_requires_same_model_and_smaller_budgets() {
    use golem_registry::{EffectiveConfig, compatible};
    let run = EffectiveConfig {
        model_path: "/m.gguf".into(),
        model_size: 1,
        model_mtime: 2,
        ctx: 4096,
        parallel: 1,
        ram_experts_gib: 8,
        snapshot_ram_bytes: None,
    };
    let mut want = run.clone();
    want.ctx = 2048;
    assert!(compatible(&run, &want).is_ok());
    want.ctx = 8192;
    assert!(compatible(&run, &want).unwrap_err().contains("--ctx 8192"));
    want.ctx = 2048;
    want.model_mtime = 3;
    assert!(compatible(&run, &want).is_err());
}

#[test]
fn live_owner_with_old_heartbeat_and_dead_socket_is_kept() {
    let d = tempfile_dir();
    let r = Registry::at(d.clone(), d.parent().unwrap().join("admit.lock"));
    write(&d, &inst("hung", std::process::id(), now() - 60));
    std::fs::write(d.join("hung.sock"), b"").unwrap();
    assert_eq!(r.list_live().unwrap().len(), 1);
    assert!(d.join("hung.sock").exists());
}

#[test]
fn registry_dirs_and_entries_are_owner_only() {
    use std::os::unix::fs::PermissionsExt;
    let base = tempfile_dir().parent().unwrap().join("fresh");
    let d = base.join("instances");
    let r = Registry::at(d.clone(), base.join("lock/admit.lock"));
    let _g = r.register(&inst("p", std::process::id(), now())).unwrap();
    r.list_live().unwrap();
    let mode = |p: PathBuf| std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode(d.clone()), 0o700);
    assert_eq!(mode(base.join("lock")), 0o700);
    assert_eq!(mode(d.join("p.json")), 0o600);
    assert_eq!(mode(base.join("lock/admit.lock")), 0o600);
}

#[test]
fn find_by_ambiguous_model_name_lists_ids() {
    let d = tempfile_dir();
    let r = Registry::at(d.clone(), d.parent().unwrap().join("admit.lock"));
    write(&d, &inst("a1", std::process::id(), now()));
    write(&d, &inst("a2", std::process::id(), now()));
    let e = r.find("m").unwrap_err().to_string();
    assert!(e.contains("a1") && e.contains("a2"), "{e}");
    assert_eq!(r.find("a2").unwrap().id, "a2");
}

#[test]
fn signal_refuses_reused_pid() {
    let mut i = inst("x", std::process::id(), now());
    i.pid_start_time = 1;
    assert!(matches!(
        golem_registry::signal(&i, 0),
        Err(golem_registry::RegistryError::InstanceNotFound(_))
    ));
}

#[test]
fn signal_skips_child_with_reused_pid() {
    let mut owner = Reaped(
        std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .unwrap(),
    );
    let mut child = Reaped(
        std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .unwrap(),
    );
    let mut i = inst("x", owner.id(), now());
    i.children = vec![golem_registry::Child::Timed {
        pid: child.id(),
        start_time: 1,
    }];
    golem_registry::signal(&i, libc::SIGKILL).unwrap();
    owner.wait().unwrap();
    assert!(
        child.try_wait().unwrap().is_none(),
        "reused child pid killed"
    );
    child.kill().unwrap();
    child.wait().unwrap();
}

#[test]
fn bare_pid_children_still_load() {
    let d = tempfile_dir();
    let r = Registry::at(d.clone(), d.parent().unwrap().join("admit.lock"));
    let mut v = serde_json::to_value(inst("old", std::process::id(), now())).unwrap();
    v["children"] = serde_json::json!([4242]);
    std::fs::write(d.join("old.json"), v.to_string()).unwrap();
    let live = r.list_live().unwrap();
    assert_eq!(live[0].children, [golem_registry::Child::Bare(4242)]);
    golem_registry::signal(&live[0], 0).unwrap();
}

/// #81: an optimize sweep is exclusive in both directions, and
/// `allow_overcommit` bypasses neither rule.
#[test]
fn optimize_registration_is_exclusive() {
    let d = tempfile_dir();
    let r = Registry::at(d.clone(), d.parent().unwrap().join("admit.lock"));
    let small = |id: &str, kind: &str| Instance {
        kind: kind.into(),
        planned_vram: 0,
        planned_ram: 0,
        ..inst(id, std::process::id(), now())
    };
    let g = r
        .admit_and_register(&small("s", "serve"), &machine(), true)
        .unwrap();
    let e = r
        .admit_and_register(&small("o", "optimize"), &machine(), true)
        .unwrap_err();
    assert!(
        e.error.message.contains("instance `s`"),
        "{}",
        e.error.message
    );
    drop(g);
    let opt = r
        .admit_and_register(&small("o", "optimize"), &machine(), false)
        .unwrap();
    for kind in ["serve", "run", "optimize"] {
        let e = r
            .admit_and_register(&small("x", kind), &machine(), true)
            .unwrap_err();
        assert!(e.error.message.contains("optimize"), "{}", e.error.message);
    }
    drop(opt);
    assert!(
        r.admit_and_register(&small("s2", "serve"), &machine(), false)
            .is_ok()
    );
}

/// A legacy bare child pid is signalled only while the owner is its parent:
/// one that another process spawned survives the owner's SIGKILL.
#[test]
fn bare_child_of_another_parent_is_not_signalled() {
    let mut owner = Reaped(
        std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .unwrap(),
    );
    let mut stranger = Reaped(
        std::process::Command::new("sleep")
            .arg("60")
            .spawn()
            .unwrap(),
    );
    let mut i = inst("x", owner.id(), now());
    i.children = vec![golem_registry::Child::Bare(stranger.id())];
    golem_registry::signal(&i, libc::SIGKILL).unwrap();
    owner.wait().unwrap();
    assert!(
        stranger.try_wait().unwrap().is_none(),
        "foreign bare pid killed"
    );
    stranger.kill().unwrap();
    stranger.wait().unwrap();
}

/// A legacy bare child pid the owner spawned is signalled with it.
#[test]
fn bare_child_of_owner_is_signalled() {
    use std::io::BufRead;
    let mut owner = Reaped(
        std::process::Command::new("sh")
            .args(["-c", "sleep 60 & echo $!; wait"])
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap(),
    );
    let mut line = String::new();
    std::io::BufReader::new(owner.stdout.take().unwrap())
        .read_line(&mut line)
        .unwrap();
    let child: u32 = line.trim().parse().unwrap();
    let _child = KillPid(child, golem_registry::pid_start_time(child).unwrap());
    let mut i = inst("x", owner.id(), now());
    i.children = vec![golem_registry::Child::Bare(child)];
    golem_registry::signal(&i, libc::SIGKILL).unwrap();
    owner.wait().unwrap();
    let t = std::time::Instant::now();
    while golem_registry::pid_start_time(child).is_ok() {
        assert!(t.elapsed().as_secs() < 5, "owned bare child survived");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
}
