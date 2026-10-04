//! `golem import` end-to-end: the CLI writes a manifest `load_manifest`
//! accepts, guards overwrites without `--force`, and requires
//! `GOLEM_DATA_DIR`.
use std::process::Command;

fn golem() -> Command {
    Command::new(env!("CARGO_BIN_EXE_golem"))
}

#[test]
fn import_cli_writes_loadable_manifest_and_guards_overwrite() {
    let data = std::env::temp_dir().join(format!("golem-cli-import-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&data);
    let src = xtask::shape_moe::cached_fixture()[0]
        .parent()
        .unwrap()
        .to_path_buf();
    let run = |force: bool| {
        let mut c = golem();
        c.env("GOLEM_DATA_DIR", &data)
            .args(["import", src.to_str().unwrap(), "--name", "shape"]);
        if force {
            c.arg("--force");
        }
        c.status().unwrap()
    };
    assert!(run(false).success());
    golem_store::load_manifest(&data.join("models/shape/manifest.json")).unwrap();
    assert!(!run(false).success(), "existing manifest without --force");
    assert!(run(true).success());
    let st = golem()
        .env_remove("GOLEM_DATA_DIR")
        .args(["import", src.to_str().unwrap()])
        .status()
        .unwrap();
    assert!(!st.success(), "GOLEM_DATA_DIR unset");
    let _ = std::fs::remove_dir_all(&data);
}
