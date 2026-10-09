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
    write(&d, &inst("stale", std::process::id(), now() - 10));
    let ids: Vec<_> = r.list_live().unwrap().into_iter().map(|i| i.id).collect();
    assert_eq!(ids, ["live"]);
    assert!(!d.join("dead.json").exists() && !d.join("stale.json").exists());
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
