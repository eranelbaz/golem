//! `golem plan` end to end on tiny-moe: header-only, no model load.
mod common;

fn plan(extra: &[&str]) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_golem"))
        .arg("plan")
        .env("XDG_CONFIG_HOME", common::tempdir_env())
        .env_remove("GOLEM_CONFIG_DIR")
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

/// tiny-moe rows are 8 elements per head: f32 doubles the f16 KV term; a
/// block-32 q8_0 cannot hold them and is refused before any arithmetic.
#[test]
fn cache_types_price_the_kv_term() {
    let kv = |extra: &[&str]| {
        let out = plan(&[&["--json", "--vram-gib", "24", "--ctx", "256"], extra].concat());
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        v["governing"]["kv_per_token"].as_u64().unwrap()
    };
    let f16 = kv(&[]);
    assert_eq!(
        kv(&["--cache-type-k", "f32", "--cache-type-v", "f32"]),
        2 * f16
    );
    assert_eq!(kv(&["--cache-type-k", "f32"]), f16 * 3 / 2);
    let out = plan(&["--cache-type-k", "q8_0"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("not a multiple of the type's block"));
}

/// No `model`: `--model-name` picks a `[models.*]` entry whose keys override the
/// shared ones, as `serve` resolves them; flags still win.
#[test]
fn model_name_selects_a_models_table_and_its_overrides() {
    let rt = common::tempdir_env();
    let dir = rt.join("config/golem");
    std::fs::create_dir_all(&dir).unwrap();
    let tiny = common::tiny_moe_path().display().to_string();
    std::fs::write(
        dir.join("golem.toml"),
        format!(
            "ctx = 512\nprefill_ubatch = 64\n\
             [models.a]\npath = \"/nonexistent.gguf\"\n\
             [models.b]\npath = {tiny:?}\nctx = 384\nvram_reserve_gib = 2\n"
        ),
    )
    .unwrap();
    let run = |extra: &[&str]| {
        common::golem_cmd(&rt)
            .args([
                "plan",
                "--json",
                "--vram-gib",
                "24",
                "--no-optimize-profile",
            ])
            .args(extra)
            .output()
            .unwrap()
    };
    let out = run(&["--model-name", "b"]);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{err}");
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["input"]["ctx"], 384);
    assert_eq!(v["input"]["prefill_ubatch"], 64);
    assert_eq!(v["input"]["vram_reserve"], 2u64 << 30);
    let out = run(&["--model-name", "b", "--ctx", "256"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["input"]["ctx"], 256);
    let out = run(&["--model-name", "zzz"]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("names no [models.*] entry"));
    std::fs::write(
        dir.join("golem.toml"),
        format!("[models.b]\npath = {tiny:?}\nmodel_name = \"x\"\n"),
    )
    .unwrap();
    let out = run(&[]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("models.b.model_name"));
}

/// `serve` floors `ram_experts_gib` to whole GiB, so [0, 1) means no pool;
/// plan refuses it from any layer rather than pricing the default pool, and
/// prices the floored pool otherwise.
#[test]
fn sub_gib_ram_experts_is_refused() {
    for g in ["0", "0.5"] {
        let out = plan(&["--json", "--vram-gib", "24", "--ram-experts-gib", g]);
        assert!(!out.status.success(), "{g}");
        assert!(String::from_utf8_lossy(&out.stderr).contains("floors to 0 GiB"));
    }
    let rt = common::tempdir_env();
    let dir = rt.join("config/golem");
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("golem.toml"), "ram_experts_gib = 0.5\n").unwrap();
    let out = common::golem_cmd(&rt)
        .args(["plan", "--json", "--vram-gib", "24"])
        .arg(common::tiny_moe_path())
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("floors to 0 GiB"));
    let out = plan(&["--json", "--vram-gib", "24", "--ram-experts-gib", "1.5"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stderr}");
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["input"]["experts"]["Pooled"]["ram_experts"], 1u64 << 30);
}
