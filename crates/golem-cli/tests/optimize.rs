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

use profile::{
    Axis, FingerprintInput, Inapplicable, Knobs, Metrics, Profile, StopSweep, applicable,
    fingerprint_of, greedy_matches, sweep,
};

fn m(tok_s: f64) -> Metrics {
    Metrics {
        tok_s,
        hit_rate: 0.9,
        ttft_p50_ms: 100.0,
        ttft_p99_ms: 200.0,
        greedy: vec![vec![(1, 5.0), (2, 5.0)]],
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
                greedy: vec![vec![(1, 5.0), (3, 5.0)]],
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
                greedy: vec![vec![(1, 5.0), (3, 0.1)]],
                ..m(20.0)
            }
        } else {
            Metrics {
                greedy: vec![vec![(1, 5.0), (2, 0.15)]],
                ..m(10.0)
            }
        })
        .is_some(),
        "near-tie flip (#113) is not drift"
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

/// The first baseline run's hit rate is higher than every later run;
/// a real win at the incumbent's hit rate must still be selected and confirmed.
#[test]
fn hit_rate_gate_is_against_incumbent_not_first_baseline() {
    let base = Knobs {
        io_threads: 2,
        ..Knobs::default()
    };
    let axes = [Axis::IoThreads(vec![2, 4])];
    let mut first = true;
    let (best, _) = sweep(&base, &axes, &mut |k| {
        let hit_rate = if std::mem::take(&mut first) {
            0.906
        } else {
            0.9
        };
        Ok(Metrics {
            hit_rate,
            ..m(if k.io_threads == 4 { 11.0 } else { 10.0 })
        })
    })
    .unwrap()
    .unwrap();
    assert_eq!(best.io_threads, 4);
}

/// Two accepted steps each give up 0.4 pp of hit rate; with confirm cut off
/// by a stop, the 0.8 pp total exceeds one step's bound and nothing is kept.
#[test]
fn unconfirmed_winner_bounds_compounded_hit_rate_drift() {
    let axes = [Axis::IoThreads(vec![4]), Axis::IoDepth(vec![2])];
    let rate = |k: &Knobs| {
        0.9 - 0.004 * f64::from(u8::from(k.io_threads == 4) + u8::from(k.io_depth == 2))
    };
    let tok = |k: &Knobs| {
        10.0 * 1.1f64.powi(i32::from(
            u8::from(k.io_threads == 4) + u8::from(k.io_depth == 2),
        ))
    };
    let run = |stop_at: usize| {
        let mut n = 0;
        let base = Knobs {
            io_depth: 1,
            ..Knobs::default()
        };
        sweep(&base, &axes, &mut |k| {
            n += 1;
            if n > stop_at {
                return Err(Box::new(StopSweep("budget".into())));
            }
            Ok(Metrics {
                hit_rate: rate(k),
                ..m(tok(k))
            })
        })
        .unwrap()
    };
    // Baseline + 2 pairs: io_threads=4 then io_depth=2 accepted, stop before confirm.
    assert_eq!(run(5), None);
    // Baseline + 1 pair: one 0.4 pp step stays within bound and is kept.
    assert_eq!(run(3).map(|b| b.0.io_threads), Some(4));
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
    use golem_plan::KvCacheType::F16;
    let f16 = golem_plan::kv_cache_types(F16, F16);
    let p = Profile {
        measured_free_vram: 20 << 30,
        measured_free_ram: 80 << 30,
        ..Profile::test_default()
    };
    assert!(applicable(&p, &f16, Some(20 << 30), 80 << 30).is_ok());
    assert!(applicable(&p, &f16, Some(18 << 30), 80 << 30).is_err());
    assert!(applicable(&p, &f16, Some(20 << 30), 78 << 30).is_err());
    // Reading noise within the slack still applies.
    let noise = (20 << 30) - profile::SLACK;
    assert!(applicable(&p, &f16, Some(noise), (80 << 30) - profile::SLACK).is_ok());
    assert!(applicable(&p, &f16, Some(noise - 1), 80 << 30).is_err());
    // Unknown VRAM (no sysfs) is not 0 free; RAM still decides.
    assert!(applicable(&p, &f16, None, 80 << 30).is_ok());
    assert!(applicable(&p, &f16, None, 78 << 30).is_err());
}

/// #161: a profile tuned under one cache type must not be applied to a run
/// using another — the types price `kv_per_token` (GLM-5.3-Flash at ctx
/// 131,072: f16 19,712 B/token = 2.41 GiB vs q8_0 14,432 B = 1.76 GiB), so
/// the knobs were tuned against another free-VRAM figure.
#[test]
fn profile_skipped_when_cache_types_differ() {
    use golem_plan::KvCacheType::{F16, Q8_0};
    let (f16, q8) = (
        golem_plan::kv_cache_types(F16, F16),
        golem_plan::kv_cache_types(Q8_0, Q8_0),
    );
    let (vram, ram) = (Some(20 << 30), 80 << 30);
    let p = Profile {
        cache_types: f16.clone(),
        measured_free_vram: 20 << 30,
        measured_free_ram: 80 << 30,
        ..Profile::test_default()
    };
    // An f16 profile does not apply to a q8_0 run, and says so.
    assert_eq!(
        applicable(&p, &q8, vram, ram).unwrap_err(),
        Inapplicable::CacheTypes {
            measured: f16.clone(),
            run: q8.clone(),
        }
    );
    assert!(applicable(&p, &f16, vram, ram).is_ok());
    // A q8_0 profile does not apply to an f16 run either.
    let q = Profile {
        cache_types: q8.clone(),
        ..p.clone()
    };
    assert_eq!(
        applicable(&q, &f16, vram, ram).unwrap_err(),
        Inapplicable::CacheTypes {
            measured: q8.clone(),
            run: f16.clone(),
        }
    );
    assert!(applicable(&q, &q8, vram, ram).is_ok());
    // Memory is reported as itself, not as a type mismatch.
    assert_eq!(
        applicable(&p, &f16, Some(1 << 30), ram).unwrap_err(),
        Inapplicable::Memory
    );
    let why = Inapplicable::CacheTypes {
        measured: f16.clone(),
        run: q8.clone(),
    }
    .to_string();
    assert!(why.contains(&f16) && why.contains(&q8), "{why}");
    assert!(Inapplicable::Memory.to_string().contains("free memory"));
}

/// #161 back-compat: profile files written before the field existed ran
/// llama's f16 default, so they load as `f16/f16` and still apply to an f16
/// run.
#[test]
fn old_profile_file_loads_as_f16() {
    use golem_plan::KvCacheType::{F16, Q8_0};
    let f16 = golem_plan::kv_cache_types(F16, F16);
    let q8 = golem_plan::kv_cache_types(Q8_0, Q8_0);
    let p = Profile {
        cache_types: q8.clone(),
        ..Profile::test_default()
    };
    let mut json = serde_json::to_value(&p).unwrap();
    json.as_object_mut().unwrap().remove("cache_types");
    let old: Profile = serde_json::from_value(json).unwrap();
    assert_eq!(old.cache_types, f16);
    assert!(applicable(&old, &f16, Some(20 << 30), 80 << 30).is_ok());
    // An explicitly recorded pair survives the file round trip.
    let stored: Profile = serde_json::from_str(&serde_json::to_string(&p).unwrap()).unwrap();
    assert_eq!(stored.cache_types, q8);
    assert!(applicable(&stored, &f16, Some(20 << 30), 80 << 30).is_err());
}

/// #161 acceptance: a profile written under f16 is not applied to a q8_0
/// `run` (nor the reverse), the info log names the reason, and the same
/// profile applies to a run resolving the types it was measured under.
#[test]
fn run_skips_a_profile_written_under_other_cache_types() {
    use golem_plan::KvCacheType::{F16, Q8_0};
    let rt = common::tempdir_env();
    let model = xtask::tiny_hybrid::cached_fixture();
    let f16 = golem_plan::kv_cache_types(F16, F16);
    let q8 = golem_plan::kv_cache_types(Q8_0, Q8_0);
    let mut p = Profile {
        fingerprint: profile::fingerprint(&model).unwrap().0,
        cache_types: f16.clone(),
        knobs: Some(Knobs {
            threads: 1,
            ..Knobs::default()
        }),
        ..Profile::test_default()
    };
    let dir = rt.join("config/golem/tuning");
    std::fs::create_dir_all(&dir).unwrap();
    let rec = dir.join(format!("{}.json", p.fingerprint));
    std::fs::write(&rec, serde_json::to_vec(&p).unwrap()).unwrap();
    let short = p.fingerprint[..12].to_owned();
    let run = |k: &str, v: &str| {
        let out = common::golem_cmd(&rt)
            .env("GOLEM_MODEL", &model)
            .args([
                "run",
                "--prompt",
                "hi",
                "--n-predict",
                "2",
                "--ctx",
                "256",
                "--cache-type-k",
                k,
                "--cache-type-v",
                v,
            ])
            .output()
            .unwrap();
        let log = String::from_utf8_lossy(&out.stderr).into_owned();
        assert!(out.status.success(), "{log}");
        log
    };
    assert!(
        run("f16", "f16").contains(&format!(
            "optimize profile {short} applied (KV cache {f16})"
        )),
        "f16 profile applies to an f16 run"
    );
    assert!(
        run("q8_0", "q8_0").contains(&format!(
            "optimize profile {short} ignored: measured under KV cache {f16}, this run uses {q8}"
        )),
        "f16 profile is refused by a q8_0 run"
    );
    // Same record under q8_0: refused by an f16 run, applied by a q8_0 one.
    p.cache_types = q8.clone();
    std::fs::write(&rec, serde_json::to_vec(&p).unwrap()).unwrap();
    assert!(
        run("f16", "f16").contains(&format!(
            "optimize profile {short} ignored: measured under KV cache {q8}, this run uses {f16}"
        )),
        "q8_0 profile is refused by an f16 run"
    );
    assert!(
        run("q8_0", "q8_0").contains(&format!("optimize profile {short} applied (KV cache {q8})")),
        "q8_0 profile applies to a q8_0 run"
    );
}

#[test]
fn fingerprint_changes_with_any_component() {
    let model = common::tiny_moe_path();
    let real = profile::probe(&model).unwrap();
    assert_eq!(
        profile::fingerprint(&model).unwrap().0,
        fingerprint_of(&real).unwrap().0,
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
    let mut seen = vec![fingerprint_of(&base).unwrap().0];
    for v in &variants {
        let f = fingerprint_of(v).unwrap().0;
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
        path: None,
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
/// No `--model`: resolved from `GOLEM_MODEL` like `run` (#163).
#[test]
fn optimize_on_fixture_with_budget_below_one_expert_tensor() {
    let rt = common::tempdir_env();
    let model = xtask::tiny_hybrid::cached_fixture();
    let out = common::golem_cmd(&rt)
        .env("XDG_CONFIG_HOME", rt.join("config"))
        .env("GOLEM_MODEL", &model)
        .arg("optimize")
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
            // #161: optimize measures under the requested types (the child
            // inherits argv) and records them.
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q8_0",
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
        stdout.contains("profile saved") || stdout.contains("baseline recorded"),
        "{stdout}"
    );
    let p: Profile = serde_json::from_slice(&std::fs::read(&rec).unwrap()).unwrap();
    assert_eq!(p.fingerprint, fp);
    // #161: the writer records the types the sweep ran under; serde's default
    // would hide a missing field, so read the raw JSON.
    let raw: serde_json::Value = serde_json::from_slice(&std::fs::read(&rec).unwrap()).unwrap();
    assert_eq!(raw["cache_types"], "q8_0/q8_0");
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

/// `stop` on an `optimize` entry reaches the `--measure-one` child listed
/// in its `children`, not only the parent.
#[test]
fn stop_optimize_kills_measuring_child() {
    let rt = common::tempdir_env();
    let mut parent = common::golem_cmd(&rt)
        .env("XDG_CONFIG_HOME", rt.join("config"))
        .args(["optimize", "--model"])
        .arg(common::tiny_moe_path())
        .args(["--n-predict", "100000", "--ctx", "4096"])
        .args(["--snapshot-ram-gib", "0", "--axes", "threads=1,2"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let dir = rt.join("run/golem/instances");
    let t = std::time::Instant::now();
    let (id, child) = loop {
        let found = std::fs::read_dir(&dir).ok().and_then(|rd| {
            rd.filter_map(Result::ok)
                .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
                .find_map(|e| {
                    let v: serde_json::Value =
                        serde_json::from_slice(&std::fs::read(e.path()).ok()?).ok()?;
                    let c = v["children"].as_array()?.first()?["pid"].as_u64()?;
                    Some((v["id"].as_str()?.to_owned(), c))
                })
        });
        if let Some(f) = found {
            break f;
        }
        assert!(t.elapsed().as_secs() < 60, "no measuring child within 60 s");
        std::thread::sleep(std::time::Duration::from_millis(20));
    };
    let st = common::golem_cmd(&rt).args(["stop", &id]).output().unwrap();
    assert!(
        st.status.success(),
        "{}",
        String::from_utf8_lossy(&st.stderr)
    );
    let _ = parent.wait();
    let t = std::time::Instant::now();
    let alive = || {
        std::fs::read_to_string(format!("/proc/{child}/stat")).is_ok_and(|s| {
            s.rsplit_once(')')
                .is_some_and(|(_, r)| !r.trim_start().starts_with('Z'))
        })
    };
    while alive() && t.elapsed().as_secs() < 10 {
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert!(!alive(), "measuring child {child} survived stop");
}

/// #84/#113: a flip is accepted only where both sides were near-ties.
#[test]
fn greedy_matches_only_near_tie_flips() {
    let g = |v: Vec<(i32, f32)>| Metrics {
        greedy: vec![v],
        ..m(1.0)
    };
    let base = g(vec![(1, 5.0), (2, 0.1), (4, 5.0)]);
    assert!(greedy_matches(&base, &base));
    assert!(greedy_matches(&g(vec![(1, 5.0), (3, 0.15)]), &base));
    assert!(!greedy_matches(&g(vec![(1, 5.0), (3, 0.3)]), &base));
    assert!(!greedy_matches(&g(vec![(9, 0.1)]), &g(vec![(1, 5.0)])));
    assert!(!greedy_matches(&g(vec![(1, 5.0)]), &base), "prefix only");
    assert!(!greedy_matches(&g(vec![]), &g(vec![])), "empty baseline");
    let old = r#"{"tok_s":1,"hit_rate":1,"ttft_p50_ms":1,"ttft_p99_ms":1,"output_sha256":[0]}"#;
    assert!(
        serde_json::from_str::<Metrics>(old)
            .unwrap()
            .greedy
            .is_empty()
    );
}
