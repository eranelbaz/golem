//! `golem plan` end to end on tiny-moe: header-only, no model load.
mod common;

fn plan(extra: &[&str]) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_golem"))
        .arg("plan")
        .arg(common::tiny_moe_path())
        .args(extra)
        .output()
        .unwrap()
}

#[test]
fn json_report_fits_on_a_roomy_device() {
    let out = plan(&["--json", "--vram-gib", "24", "--ctx", "256"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stderr}");
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["verdict"], "fits");
    let names: Vec<&str> = v["report"]["vram"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    for n in [
        "t_ne",
        "kv",
        "recurrent_state",
        "ws",
        "device_expert_cache",
        "stream_slabs",
        "rocm_overhead",
    ] {
        assert!(names.contains(&n), "{n} missing from {names:?}");
    }
    assert!(v["governing"]["kv_per_token"].as_u64().unwrap() > 0);
}

/// 1 GiB of VRAM cannot hold the 2.64 GiB ROCm overhead alone: exit 2 with
/// the failing inequality in a `golem_plan_infeasible` envelope.
#[test]
fn infeasible_exits_2_with_the_inequality() {
    let out = plan(&["--vram-gib", "1"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("golem_plan_infeasible"), "{stderr}");
    assert!(stderr.contains("> 1.0 GiB"), "{stderr}");
    assert!(String::from_utf8_lossy(&out.stdout).contains("verdict: infeasible"));
}

/// §9.4: with no positional, the model comes from `golem.toml` in
/// `$XDG_CONFIG_HOME/golem`; with none anywhere the error names all three.
#[test]
fn model_resolves_from_default_config_file() {
    let rt = common::tempdir_env();
    let run = || {
        common::golem_cmd(&rt)
            .args(["plan", "--json", "--vram-gib", "24", "--ctx", "256"])
            .output()
            .unwrap()
    };
    let out = run();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("GOLEM_MODEL") && err.contains("golem.toml"),
        "{err}"
    );
    let dir = rt.join("config/golem");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("golem.toml"),
        format!(
            "model = {:?}\n",
            common::tiny_moe_path().display().to_string()
        ),
    )
    .unwrap();
    let out = run();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
