//! `golem import` end-to-end on the split-GGUF shape fixture: the manifest is
//! published next to the source and every replica, re-import is idempotent,
//! `--sync` follows `golem.toml`, and a manifest with a missing replica
//! refuses to serve instead of falling back to one disk.
mod common;

use std::path::{Path, PathBuf};
use std::process::Output;

fn shard_names() -> Vec<String> {
    xtask::shape_moe::cached_fixture()
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
        .collect()
}

/// An isolated `rt` with `src` (a copy of the fixture) and two replica dirs.
fn layout(tag: &str) -> (PathBuf, PathBuf, PathBuf, PathBuf) {
    let rt = common::tempdir_env().join(tag);
    let (src, r1, r2) = (rt.join("src"), rt.join("r1"), rt.join("r2"));
    for d in [&src, &r1, &r2] {
        std::fs::create_dir_all(d).unwrap();
    }
    for s in xtask::shape_moe::cached_fixture() {
        std::fs::copy(&s, src.join(s.file_name().unwrap())).unwrap();
    }
    (rt, src, r1, r2)
}

fn import(rt: &Path, args: &[&str]) -> Output {
    common::golem_cmd(rt)
        .arg("import")
        .args(args)
        .output()
        .unwrap()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// The manifest's replica roots, canonicalized, in shard order.
fn roots(path: &Path) -> Vec<PathBuf> {
    let m = golem_store::load_manifest(path).unwrap();
    m.shards[0].replicas.clone()
}

#[test]
fn import_publishes_a_manifest_in_the_source_and_every_replica() {
    let (rt, src, r1, r2) = layout("publish");
    let out = import(
        &rt,
        &[
            src.to_str().unwrap(),
            r1.to_str().unwrap(),
            r2.to_str().unwrap(),
            "--name",
            "shape",
            "--quant",
            "Q4_K",
        ],
    );
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert!(
        stderr(&out).contains("share a physical device"),
        "one same-device warning per pair: {}",
        stderr(&out)
    );
    let want: Vec<PathBuf> = [&src, &r1, &r2]
        .iter()
        .map(|d| d.canonicalize().unwrap())
        .collect();
    for d in [&src, &r1, &r2] {
        let p = d.join("manifest.json");
        let m = golem_store::load_manifest(&p).unwrap();
        assert_eq!((m.name.as_str(), m.shards.len()), ("shape", 2));
        assert_eq!(m.quant.as_deref(), Some("Q4_K"));
        assert_eq!(roots(&p), want, "every copy lists the same replica set");
    }
    for s in shard_names() {
        assert_eq!(
            std::fs::read(src.join(&s)).unwrap(),
            std::fs::read(r2.join(&s)).unwrap()
        );
    }
    std::fs::remove_dir_all(&rt).unwrap();
}

#[test]
fn reimport_is_idempotent() {
    let (rt, src, r1, _) = layout("idempotent");
    let args = [src.to_str().unwrap(), r1.to_str().unwrap()];
    let first = import(&rt, &args);
    assert!(first.status.success(), "stderr: {}", stderr(&first));
    assert!(
        stdout(&first).contains("0 replica shard(s) kept"),
        "a first import clones every shard: {}",
        stdout(&first)
    );
    let second = import(&rt, &args);
    assert!(second.status.success(), "stderr: {}", stderr(&second));
    // 2 shards × 1 replica already match by size, mtime and the R-8 windows.
    assert!(
        stdout(&second).contains("2 replica shard(s) kept"),
        "idempotent re-import skips them: {}",
        stdout(&second)
    );
    std::fs::remove_dir_all(&rt).unwrap();
}

#[test]
fn sync_flag_follows_golem_toml_and_reclones_a_missing_replica() {
    let (rt, src, r1, _) = layout("sync");
    let out = import(&rt, &[src.to_str().unwrap(), r1.to_str().unwrap()]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let model = src.join(&shard_names()[0]);
    let cfg = rt.join("config/golem");
    std::fs::create_dir_all(&cfg).unwrap();
    std::fs::write(
        cfg.join("golem.toml"),
        format!(
            "[models.shape]\npath = {:?}\nreplicas = [{:?}]\n",
            model, r1
        ),
    )
    .unwrap();

    let up = common::golem_cmd(&rt)
        .args(["import", "--sync", "shape"])
        .output()
        .unwrap();
    assert!(up.status.success(), "stderr: {}", stderr(&up));
    assert!(
        stdout(&up).contains("up to date"),
        "nothing to sync: {}",
        stdout(&up)
    );

    let gone = r1.join(&shard_names()[1]);
    std::fs::remove_file(&gone).unwrap();
    let out = common::golem_cmd(&rt)
        .args(["import", "--sync", "shape"])
        .output()
        .unwrap();
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    assert_eq!(
        std::fs::read(&gone).unwrap(),
        std::fs::read(src.join(&shard_names()[1])).unwrap(),
        "the missing replica shard is cloned back"
    );

    // Bare `--sync` targets the top-level `model` + `replicas`.
    std::fs::write(
        cfg.join("golem.toml"),
        format!("model = {:?}\nreplicas = [{:?}]\n", model, r1),
    )
    .unwrap();
    let up = common::golem_cmd(&rt)
        .args(["import", "--sync"])
        .output()
        .unwrap();
    assert!(up.status.success(), "stderr: {}", stderr(&up));
    assert!(stdout(&up).contains("up to date"), "{}", stdout(&up));

    // `--sync` and paths are mutually exclusive.
    let both = common::golem_cmd(&rt)
        .args(["import", src.to_str().unwrap(), "--sync"])
        .output()
        .unwrap();
    assert_eq!(
        both.status.code(),
        Some(1),
        "paths + --sync must be refused"
    );
    std::fs::remove_dir_all(&rt).unwrap();
}

#[test]
fn serve_refuses_a_manifest_with_a_missing_replica() {
    let (rt, src, r1, _) = layout("refuse");
    let out = import(&rt, &[src.to_str().unwrap(), r1.to_str().unwrap()]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let gone = r1.join(&shard_names()[1]);
    std::fs::remove_file(&gone).unwrap();

    let run = common::golem_cmd(&rt)
        .args([
            "run",
            "--n-gpu-layers",
            "0",
            "--n-predict",
            "1",
            "--prompt",
            "hi",
        ])
        .arg(src.join(&shard_names()[0]))
        .output()
        .unwrap();
    assert_eq!(run.status.code(), Some(1), "must refuse, not serve");
    let e = stderr(&run);
    assert!(
        e.contains(gone.to_str().unwrap()) && e.contains("--sync"),
        "names the missing replica and the fix: {e}"
    );
    std::fs::remove_dir_all(&rt).unwrap();
}
