//! `golem optimize` (#81): sweep gates, profile applicability, fingerprint,
//! §9.4 profile layer, and the end-to-end fixture sweep. The pure modules
//! are compiled in from the binary's sources (a bin crate has no lib to
//! link).
#[path = "../src/config.rs"]
#[allow(dead_code)]
mod config;
#[path = "../src/profile.rs"]
#[allow(dead_code)]
mod profile;

mod common;

use profile::{Axis, FingerprintInput, Knobs, Metrics, Profile, applicable, fingerprint_of, sweep};

fn m(tok_s: f64) -> Metrics {
    Metrics {
        tok_s,
        hit_rate: 0.9,
        ttft_p50_ms: 100.0,
        ttft_p99_ms: 200.0,
        output_sha256: [0; 32],
    }
}

#[test]
fn coordinate_descent_finds_separable_optimum() {
    let base = Knobs {
        threads: 4,
        o_direct: false,
        ..Knobs::default()
    };
    let axes = [
        Axis::Threads(vec![4, 8, 16]),
        Axis::ODirect(vec![false, true]),
    ];
    let (best, _) = sweep(&base, &axes, &mut |k| {
        Ok(m(
            10.0 - (k.threads as f64 - 8.0).abs() + if k.o_direct { 1.0 } else { 0.0 }
        ))
    })
    .unwrap()
    .unwrap();
    assert_eq!((best.threads, best.o_direct), (8, true));
}

#[test]
fn gates_reject_drift_small_gain_and_ttft_regression() {
    let base = Knobs {
        threads: 4,
        ..Knobs::default()
    };
    let axes = [Axis::Threads(vec![4, 8])];
    let r = |f: &dyn Fn(&Knobs) -> Metrics| sweep(&base, &axes, &mut |k| Ok(f(k))).unwrap();
    assert!(
        r(&|k| if k.threads == 8 { m(10.2) } else { m(10.0) }).is_none(),
        "+2% < +3% gate"
    );
    assert!(
        r(&|k| if k.threads == 8 {
            Metrics {
                output_sha256: [1; 32],
                ..m(20.0)
            }
        } else {
            m(10.0)
        })
        .is_none(),
        "greedy drift"
    );
    assert!(
        r(&|k| if k.threads == 8 {
            Metrics {
                ttft_p99_ms: 260.0,
                ..m(20.0)
            }
        } else {
            m(10.0)
        })
        .is_none(),
        "p99 +30%"
    );
    assert!(
        r(&|k| if k.threads == 8 {
            Metrics {
                hit_rate: 0.89,
                ..m(20.0)
            }
        } else {
            m(10.0)
        })
        .is_none(),
        "hit rate -1 pp"
    );
}

/// #81: gates are against the baseline, so two +11% p99 steps (each within
/// +20% of the incumbent) cannot compound past +20% of the baseline.
#[test]
fn ttft_gate_is_against_baseline_not_incumbent() {
    let base = Knobs {
        threads: 4,
        o_direct: false,
        ..Knobs::default()
    };
    let axes = [Axis::Threads(vec![4, 8]), Axis::ODirect(vec![false, true])];
    let (best, _) = sweep(&base, &axes, &mut |k| {
        let steps = u32::from(k.threads == 8) + u32::from(k.o_direct);
        Ok(Metrics {
            ttft_p99_ms: 200.0 * 1.11f64.powi(steps as i32),
            ..m(10.0 * 1.1f64.powi(steps as i32))
        })
    })
    .unwrap()
    .unwrap();
    assert_eq!(
        (best.threads, best.o_direct),
        (8, false),
        "second step's p99 is +23% of baseline"
    );
}

#[test]
fn failing_candidate_is_skipped_not_fatal() {
    let base = Knobs {
        o_direct: false,
        ..Knobs::default()
    };
    let axes = [Axis::ODirect(vec![false, true])];
    let r = sweep(&base, &axes, &mut |k| {
        if k.o_direct {
            Err("EIO".into())
        } else {
            Ok(m(10.0))
        }
    })
    .unwrap();
    assert!(r.is_none()); // baseline kept, sweep completed
}

#[test]
fn profile_ignored_when_less_free_memory_now() {
    let p = Profile {
        measured_free_vram: 20 << 30,
        measured_free_ram: 80 << 30,
        ..Profile::test_default()
    };
    assert!(applicable(&p, 20 << 30, 80 << 30));
    assert!(!applicable(&p, 18 << 30, 80 << 30));
    assert!(!applicable(&p, 20 << 30, 78 << 30));
}

#[test]
fn fingerprint_changes_with_any_component() {
    let model = common::tiny_moe_path();
    let real = profile::probe(&model).unwrap();
    assert_eq!(
        profile::fingerprint(&model).unwrap().0,
        fingerprint_of(&real).0,
        "fingerprint(path) = fingerprint_of(probe(path))"
    );
    let base = FingerprintInput {
        cpu: "cores=8".into(),
        gpu: vec!["gfx1100 24G".into()],
        model_shards: vec![("m.gguf".into(), 100, 7)],
        engine: "0.1 pin=abc".into(),
    };
    let variants = [
        FingerprintInput {
            cpu: "cores=16".into(),
            ..base.clone()
        },
        FingerprintInput {
            gpu: vec!["gfx1101 16G".into()],
            ..base.clone()
        },
        FingerprintInput {
            model_shards: vec![("m.gguf".into(), 100, 8)],
            ..base.clone()
        },
        FingerprintInput {
            engine: "0.2 pin=abc".into(),
            ..base.clone()
        },
    ];
    let mut seen = vec![fingerprint_of(&base).0];
    for v in &variants {
        let f = fingerprint_of(v).0;
        assert_eq!(f.len(), 64, "hex sha256");
        assert!(!seen.contains(&f), "{f} repeats");
        seen.push(f);
    }
    // Model mtime through the real probe: a temp copy, touched forward.
    let dir = common::tempdir_env();
    let copy = dir.join("tiny-moe-320.gguf");
    std::fs::copy(&model, &copy).unwrap();
    let a = profile::fingerprint(&copy).unwrap().0;
    let f = std::fs::File::options().write(true).open(&copy).unwrap();
    f.set_modified(std::time::SystemTime::now() + std::time::Duration::from_secs(3600))
        .unwrap();
    assert_ne!(a, profile::fingerprint(&copy).unwrap().0, "mtime");
}

#[test]
fn explicit_flag_beats_profile() {
    let knobs = Knobs {
        threads: 12,
        ..Knobs::default()
    };
    let with = |env: &'static dyn Fn(&str) -> Option<String>, k: Option<&Knobs>| config::Layers {
        file: toml::Table::new(),
        profile: config::profile_table(k),
        env,
    };
    let none: &'static dyn Fn(&str) -> Option<String> = &|_| None;
    let six: &'static dyn Fn(&str) -> Option<String> =
        &|k| (k == "GOLEM_THREADS").then(|| "6".to_owned());
    assert_eq!(
        with(none, Some(&knobs))
            .get(Some(4u32), "threads", 2)
            .unwrap(),
        4
    );
    assert_eq!(
        with(six, Some(&knobs)).get(None, "threads", 2u32).unwrap(),
        6
    );
    assert_eq!(
        with(none, Some(&knobs)).get(None, "threads", 2u32).unwrap(),
        12
    );
    // --no-optimize-profile: no profile layer.
    assert_eq!(with(none, None).get(None, "threads", 2u32).unwrap(), 2);
    // A file value still beats the profile.
    let l = config::Layers {
        file: toml::from_str("threads = 9").unwrap(),
        ..with(none, Some(&knobs))
    };
    assert_eq!(l.get(None, "threads", 2u32).unwrap(), 9);
    // Unset knobs (0) leave lower layers alone.
    assert_eq!(
        with(none, Some(&knobs))
            .get(None, "io_depth", 1u32)
            .unwrap(),
        1
    );
}

/// Split, quantized mixed-stride fixture (tiny-hybrid: Q4_K/Q5_K/Q6_K
/// expert planes across 2 shards) with a device budget of top-k = 2 experts
/// per layer, below one expert tensor (8 experts): a 2-candidate sweep runs
/// to completion under a machine budget that admits the parent's
/// reservation once but would refuse it twice (children register nothing).
#[test]
fn optimize_on_fixture_with_budget_below_one_expert_tensor() {
    let rt = common::tempdir_env();
    let model = xtask::tiny_hybrid::cached_fixture();
    let out = common::golem_cmd(&rt)
        .env("XDG_CONFIG_HOME", rt.join("config"))
        .args(["optimize", "--model"])
        .arg(&model)
        .args([
            "--ram-experts-gib",
            "1",
            "--device-cache-experts",
            "2",
            "--n-predict",
            "4",
            "--ctx",
            "256",
            "--snapshot-ram-gib",
            "0",
            "--axes",
            "threads=1,2",
            "--test-tight-budget",
        ])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "stdout:\n{stdout}\nstderr:\n{stderr}");
    let measured = stdout
        .lines()
        .filter(|l| l.starts_with("candidate "))
        .count();
    assert!(measured >= 2, "two candidates measured:\n{stdout}");
    assert!(
        stdout.contains("plan ok"),
        "every candidate plan-checked:\n{stdout}"
    );
    assert!(
        !stdout.contains("disqualified"),
        "no candidate refused (double count):\n{stdout}\n{stderr}"
    );
    let fp = profile::fingerprint(&model).unwrap().0;
    let rec = rt.join(format!("config/golem/tuning/{fp}.json"));
    assert!(
        stdout.contains("profile saved") || stdout.contains("baseline recorded, no profile saved"),
        "{stdout}"
    );
    let p: Profile = serde_json::from_slice(&std::fs::read(&rec).unwrap()).unwrap();
    assert_eq!(p.fingerprint, fp);
    assert_eq!(p.candidates.len(), measured);
    // Nothing left registered once optimize exits.
    let list = common::golem_cmd(&rt).arg("list").output().unwrap();
    assert!(
        !String::from_utf8_lossy(&list.stdout).contains("optimize"),
        "entry released"
    );
}

/// HW leg (#81 Step 4): the real sweep on GLM-5.3-Flash. Opt-in only:
/// `GOLEM_HW=1 GOLEM_HW_MODEL=<shard 1> cargo nextest run --run-ignored
/// only -E 'test(optimize_real_model)'`; results → docs/baselines/m4.9-optimize.md.
#[test]
#[ignore = "HW: loads the real model; needs GOLEM_HW=1 and GOLEM_HW_MODEL"]
fn optimize_real_model() {
    assert_eq!(
        std::env::var("GOLEM_HW").as_deref(),
        Ok("1"),
        "set GOLEM_HW=1"
    );
    let model = std::env::var("GOLEM_HW_MODEL").expect("GOLEM_HW_MODEL must name shard 1");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_golem"))
        .args(["optimize", "--model", &model])
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    println!("{stdout}");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(stdout.contains("winner") || stdout.contains("baseline recorded"));
}
