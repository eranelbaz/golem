//! `golem optimize` knobs (#81): each one at its default decodes exactly
//! today's greedy output, and its non-default setting either keeps the
//! output (scheduling-only knobs) or is refused at load.
mod common;

use std::path::PathBuf;

use golem_engine::config::EngineConfig;
use golem_engine::session::{ChatTurn, Session};

/// Page-padded tiny-moe copy plus a one-replica manifest (store-backed pool).
fn manifest_fixture() -> (PathBuf, PathBuf) {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let dir = std::env::temp_dir().join(format!(
        "golem-knobs-{}-{}",
        std::process::id(),
        N.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let model = dir.join("tiny-moe-320.gguf");
    let mut bytes = std::fs::read(common::tiny_moe_path()).unwrap();
    bytes.resize(bytes.len().next_multiple_of(4096), 0);
    std::fs::write(&model, &bytes).unwrap();
    let mut f = std::fs::File::open(&model).unwrap();
    let index: Vec<serde_json::Value> = golem_gguf::tensor_entries(&mut f)
        .unwrap()
        .iter()
        .filter(|e| e.expert_kind().is_some())
        .map(|e| {
            let stride = golem_gguf::ggml_row_bytes(e.ggml_type, e.dims[0]).unwrap() * e.dims[1];
            serde_json::json!({
                "tensor": e.name,
                "layer": golem_gguf::blk_of(&e.name).unwrap(),
                "experts": e.dims[2],
                "bytes_per_expert": stride,
                "offsets_bytes": (0..e.dims[2]).map(|i| i * stride).collect::<Vec<_>>(),
            })
        })
        .collect();
    let m = serde_json::json!({
        "name": "tiny-moe",
        "shards": [{
            "file": "tiny-moe-320.gguf",
            "bytes": bytes.len(),
            "sha256": "0".repeat(64),
            "replicas": [dir],
        }],
        "expert_index": index,
    });
    let p = dir.join("manifest.json");
    std::fs::write(&p, m.to_string()).unwrap();
    (model, p)
}

fn pooled() -> EngineConfig {
    let (model, manifest) = manifest_fixture();
    EngineConfig {
        model_path: model,
        manifest_path: Some(manifest),
        ram_experts_gib: 1,
        device_cache_experts: 4,
        n_predict: 8,
        ..EngineConfig::default()
    }
}

fn greedy(cfg: &EngineConfig) -> Result<String, golem_engine::EngineError> {
    let mut s = Session::new(cfg)?;
    let mut out = String::new();
    s.chat_turn(
        &[ChatTurn {
            role: "user",
            content: "the quick brown fox",
        }],
        &mut |p| out.push_str(p),
    )?;
    Ok(out)
}

fn assert_default_unchanged(set: impl Fn(&mut EngineConfig)) -> (EngineConfig, String) {
    let base = pooled();
    let want = greedy(&base).unwrap();
    assert!(!want.is_empty());
    let mut at_default = base.clone();
    set(&mut at_default);
    assert_eq!(greedy(&at_default).unwrap(), want, "default changed output");
    (base, want)
}

#[test]
fn io_depth_default_one_and_two_keep_greedy_output() {
    let d = EngineConfig::default().io_depth;
    let (base, want) = assert_default_unchanged(|c| c.io_depth = d);
    for io_depth in [1, 2] {
        let c = EngineConfig {
            io_depth,
            ..base.clone()
        };
        assert_eq!(greedy(&c).unwrap(), want, "io_depth {io_depth}");
    }
}

#[test]
fn numa_interleave_default_and_on_keep_greedy_output() {
    let d = EngineConfig::default().numa_interleave;
    assert!(!d, "first-touch is today's behaviour");
    let (base, want) = assert_default_unchanged(|c| c.numa_interleave = d);
    let on = EngineConfig {
        numa_interleave: true,
        ..base
    };
    assert_eq!(greedy(&on).unwrap(), want);
}

#[test]
fn o_direct_default_and_required_keep_greedy_output() {
    let d = EngineConfig::default().o_direct;
    assert!(!d, "probe-and-fall-back is today's behaviour");
    let (base, want) = assert_default_unchanged(|c| c.o_direct = d);
    // The repo filesystem serves O_DIRECT (the placement source tests fail
    // loudly where it does not), so requiring it changes nothing.
    let on = EngineConfig {
        o_direct: true,
        ..base
    };
    assert_eq!(greedy(&on).unwrap(), want);
}

#[test]
fn o_direct_without_manifest_is_refused() {
    let cfg = EngineConfig {
        manifest_path: None,
        o_direct: true,
        ..pooled()
    };
    let e = greedy(&cfg).unwrap_err().to_string();
    assert!(e.contains("o_direct"), "{e}");
}

/// CPU substrate: no copy stream exists, so `async_h2d` stays refused.
/// (An auto-detected HIP substrate accepts it with unchanged output.)
#[cfg(not(feature = "hip-tests"))]
#[test]
fn async_h2d_default_keeps_output_and_on_is_refused() {
    let d = EngineConfig::default().async_h2d;
    assert!(!d, "the scheduler's in-stream H2D is today's behaviour");
    let (base, _) = assert_default_unchanged(|c| c.async_h2d = d);
    let on = EngineConfig {
        async_h2d: true,
        ..base
    };
    let r = greedy(&on);
    if golem_llama_sys::substrate_backend() == golem_llama_sys::Backend::Cpu {
        let e = r.unwrap_err().to_string();
        assert!(e.contains("async_h2d"), "{e}");
    } else {
        // HIP substrate without the offload fixture: the copy stream exists
        // but nothing is pooled onto the device, so output is unchanged.
        let off = EngineConfig {
            async_h2d: false,
            ..on
        };
        assert_eq!(r.expect("async_h2d on HIP"), greedy(&off).unwrap());
    }
}

/// HIP: async_h2d without the RAM expert pool has no pooled H2D to move.
#[cfg(feature = "hip-tests")]
#[test]
fn async_h2d_without_pool_is_refused() {
    let cfg = EngineConfig {
        async_h2d: true,
        ram_experts_gib: 0,
        ..pooled()
    };
    let e = greedy(&cfg).unwrap_err().to_string();
    assert!(e.contains("ram_experts_gib > 0"), "{e}");
}

/// CPU substrate: every expert already runs on the CPU, so
/// `cpu_expert_math` is refused; its default keeps today's output.
#[cfg(not(feature = "hip-tests"))]
#[test]
fn cpu_expert_math_default_keeps_output_and_on_is_refused_on_cpu() {
    let d = EngineConfig::default().cpu_expert_math;
    assert!(!d, "GPU expert math only is today's behaviour");
    let (base, _) = assert_default_unchanged(|c| c.cpu_expert_math = d);
    let on = EngineConfig {
        cpu_expert_math: true,
        ..base
    };
    if golem_llama_sys::substrate_backend() == golem_llama_sys::Backend::Cpu {
        let e = greedy(&on).unwrap_err().to_string();
        assert!(e.contains("cpu_expert_math"), "{e}");
    }
}

/// HIP: CPU expert math reads RAM-pool slots; without the pool it is refused.
#[cfg(feature = "hip-tests")]
#[test]
fn cpu_expert_math_without_pool_is_refused() {
    let cfg = EngineConfig {
        cpu_expert_math: true,
        ram_experts_gib: 0,
        ..pooled()
    };
    let e = greedy(&cfg).unwrap_err().to_string();
    assert!(
        e.contains("cpu_expert_math") && e.contains("ram_experts_gib > 0"),
        "{e}"
    );
}

/// HIP: CPU expert math joins on the device per block; a partial offload
/// would leave a block's accumulator on the CPU, so it is refused.
#[cfg(feature = "hip-tests")]
#[test]
fn cpu_expert_math_partial_offload_is_refused() {
    let cfg = EngineConfig {
        cpu_expert_math: true,
        n_gpu_layers: 1,
        ..pooled()
    };
    let e = greedy(&cfg).unwrap_err().to_string();
    assert!(
        e.contains("cpu_expert_math") && e.contains("every layer offloaded"),
        "{e}"
    );
}

/// `cpu_expert_frac` outside (0, 1] and zero worker threads are refused on
/// any substrate; with the flag off neither is checked.
#[test]
fn cpu_expert_frac_and_threads_out_of_range_are_refused() {
    for frac in [0.0, -0.5, 1.5, f32::NAN] {
        let cfg = EngineConfig {
            cpu_expert_math: true,
            cpu_expert_frac: frac,
            ..pooled()
        };
        let e = greedy(&cfg).unwrap_err().to_string();
        assert!(e.contains("cpu_expert_frac"), "{frac}: {e}");
    }
    let cfg = EngineConfig {
        cpu_expert_math: true,
        cpu_expert_threads: 0,
        ..pooled()
    };
    let e = greedy(&cfg).unwrap_err().to_string();
    assert!(e.contains("cpu_expert_threads"), "{e}");
    let off = EngineConfig {
        cpu_expert_frac: 0.0,
        cpu_expert_threads: 0,
        ..pooled()
    };
    greedy(&off).expect("flag off ignores frac and threads");
}

/// HIP: GGML_CUDA_GRAPH_OPT=1 owns the copy-stream index. The workspace
/// denies `unsafe` (`env::set_var`), so the test re-runs itself in a child
/// process with the variable set; nothing else sees it.
#[cfg(feature = "hip-tests")]
#[test]
fn async_h2d_with_graph_opt_is_refused() {
    const NAME: &str = "async_h2d_with_graph_opt_is_refused";
    if std::env::var_os("GGML_CUDA_GRAPH_OPT").is_none() {
        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .args([NAME, "--exact"])
            .env("GGML_CUDA_GRAPH_OPT", "1")
            .output()
            .expect("spawn child test");
        let stdout = String::from_utf8_lossy(&out.stdout);
        // A filter that matched nothing also exits 0: require the run.
        assert!(
            out.status.success() && stdout.contains("1 passed"),
            "child: {stdout}{}",
            String::from_utf8_lossy(&out.stderr)
        );
        return;
    }
    let off = common::offload::offload_config();
    let cfg = EngineConfig {
        n_gpu_layers: off.n_gpu_layers,
        cpu_moe_layers: off.cpu_moe_layers,
        async_h2d: true,
        ..pooled()
    };
    let e = greedy(&cfg).unwrap_err().to_string();
    assert!(e.contains("GGML_CUDA_GRAPH_OPT"), "{e}");
}

/// HIP: pooled H2D on the dedicated copy stream decodes bit-identically to
/// the in-stream H2D across decode, device-cache eviction (K = 4 < 8
/// experts, two turns reusing evicted slots), streamed prefill, cancel and
/// storage loss mid-stream, with every expert op on the GPU.
#[cfg(feature = "hip-tests")]
#[test]
fn async_h2d_on_matches_off_output() {
    use golem_engine::session::expert_placement;
    use golem_engine::{FinishReason, TurnRequest};
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    const LONG: &[ChatTurn<'static>] = &[ChatTurn {
        role: "user",
        content: "one two three four five six seven eight nine ten eleven twelve thirteen \
                  fourteen fifteen sixteen seventeen eighteen nineteen twenty",
    }];
    const NEXT: &[ChatTurn<'static>] = &[ChatTurn {
        role: "user",
        content: "count to ten",
    }];
    let gpu = |async_h2d: bool| {
        let off = common::offload::offload_config();
        EngineConfig {
            n_gpu_layers: off.n_gpu_layers,
            cpu_moe_layers: off.cpu_moe_layers,
            async_h2d,
            ..pooled()
        }
    };
    let streamed = |async_h2d: bool| EngineConfig {
        prefill_stream: true,
        prefill_ubatch: 64,
        prefill_stream_min: 1,
        n_ctx: 512,
        ..gpu(async_h2d)
    };
    let turn = |s: &mut Session, msgs: &[ChatTurn<'_>]| {
        let mut out = String::new();
        s.turn(&TurnRequest::new(msgs).max_tokens(Some(16)), &mut |t| {
            out.push_str(t)
        })
        .expect("turn");
        out
    };
    let two_turns = |cfg: &EngineConfig| {
        let mut s = Session::new(cfg).expect("session");
        let a = turn(&mut s, LONG);
        let b = turn(&mut s, NEXT);
        let copies = s.device_cache_stats().map_or(0, |d| d.cache_copies);
        (a, b, copies)
    };
    let before = expert_placement();

    // Decode + eviction, plain and streamed.
    for mk in [&gpu as &dyn Fn(bool) -> EngineConfig, &streamed] {
        let (a0, b0, _) = two_turns(&mk(false));
        let (a1, b1, copies) = two_turns(&mk(true));
        assert!(!a0.is_empty() && !b0.is_empty());
        assert_eq!((a1, b1), (a0, b0), "async_h2d changed output");
        assert!(copies > 0, "no H2D on the copy stream");
    }
    let reference = turn(&mut Session::new(&streamed(false)).unwrap(), NEXT);

    for _ in 0..10 {
        // Cancel inside a streamed ubatch, then a turn on the same session.
        let mut s = Session::new(&streamed(true)).expect("session");
        s.p3_hooks().unwrap().cancel_at_stream_plane_for_test(3);
        let out = s
            .turn(
                &TurnRequest::new(LONG).cancel(Some(Arc::new(AtomicBool::new(false)))),
                &mut |_| {},
            )
            .expect("cancelled turn is Ok");
        assert_eq!(out.finish, FinishReason::Cancelled);
        assert_eq!(s.p3_hooks().unwrap().pool_batch_leases(), 0);
        s.p3_hooks().unwrap().cancel_at_stream_plane_for_test(0);
        assert_eq!(turn(&mut s, NEXT), reference, "turn after cancel");
        drop(s);

        // Storage loss mid-stream fails the turn. The seam kills every
        // replica, so the store stays FatalIo and the session is poisoned:
        // slot reuse after a drain is covered by the cancel leg above; here
        // the next turn must fail cleanly, and a fresh session serves.
        let mut s = Session::new(&streamed(true)).expect("session");
        s.fail_storage_in_stream_read_for_test();
        let mut sink = |_: &str| {};
        assert!(s.turn(&TurnRequest::new(LONG), &mut sink).is_err());
        assert_eq!(s.p3_hooks().unwrap().pool_batch_leases(), 0);
        assert!(matches!(
            s.turn(&TurnRequest::new(NEXT), &mut sink),
            Err(golem_engine::EngineError::StorageUnavailable)
        ));
        assert_eq!(s.p3_hooks().unwrap().pool_batch_leases(), 0);
        drop(s);
        assert_eq!(
            turn(&mut Session::new(&streamed(true)).unwrap(), NEXT),
            reference,
            "turn after storage loss"
        );
    }
    let after = expert_placement();
    assert_eq!(
        after.on_cpu, before.on_cpu,
        "expert MUL_MAT_ID ran on the CPU"
    );
}

/// HIP leg: the store-backed knobs on the GPU split with the
/// engine-registered hooks, K = 4 device slots < one expert tensor
/// (8 experts): greedy output unchanged and every expert op on the GPU.
#[cfg(feature = "hip-tests")]
#[test]
fn store_knobs_keep_output_and_experts_on_gpu() {
    use golem_engine::session::expert_placement;
    let gpu = || {
        let off = common::offload::offload_config();
        EngineConfig {
            n_gpu_layers: off.n_gpu_layers,
            cpu_moe_layers: off.cpu_moe_layers,
            ..pooled()
        }
    };
    let base = gpu();
    let before = expert_placement();
    let want = greedy(&base).unwrap();
    assert!(!want.is_empty());
    for cfg in [
        EngineConfig {
            io_depth: 2,
            ..gpu()
        },
        EngineConfig {
            numa_interleave: true,
            ..gpu()
        },
        EngineConfig {
            o_direct: true,
            ..gpu()
        },
    ] {
        assert_eq!(greedy(&cfg).unwrap(), want);
    }
    let after = expert_placement();
    let (on_gpu, on_cpu) = (after.on_gpu - before.on_gpu, after.on_cpu - before.on_cpu);
    assert!(on_cpu == 0 && on_gpu > 0, "on_gpu={on_gpu} on_cpu={on_cpu}");
}

/// The victim ring is off by default; on, it needs exclusive residency and
/// the RAM pool on any substrate, and a device substrate on the CPU one.
#[test]
fn victim_ring_preconditions_are_refused() {
    use golem_engine::config::Residency;
    assert_eq!(EngineConfig::default().victim_ring_slots, 0);
    let inclusive = EngineConfig {
        victim_ring_slots: 4,
        ..pooled()
    };
    let e = greedy(&inclusive).unwrap_err().to_string();
    assert!(
        e.contains("victim_ring_slots") && e.contains("exclusive"),
        "{e}"
    );
    let no_pool = EngineConfig {
        residency: Residency::Exclusive,
        ram_experts_gib: 0,
        ..inclusive.clone()
    };
    let e = greedy(&no_pool).unwrap_err().to_string();
    assert!(
        e.contains("victim_ring_slots") && e.contains("ram_experts_gib > 0"),
        "{e}"
    );
    if golem_llama_sys::substrate_backend() == golem_llama_sys::Backend::Cpu {
        let cpu = EngineConfig {
            residency: Residency::Exclusive,
            ..inclusive
        };
        let e = greedy(&cpu).unwrap_err().to_string();
        assert!(
            e.contains("victim_ring_slots") && e.contains("substrate"),
            "{e}"
        );
    }
}

/// HIP: exclusive residency with the victim ring, K = 4 < 8 experts so slab
/// evictions park and D2H through the ring, through the engine-registered
/// hooks: two turns decode the inclusive baseline's text, planes went
/// through the ring, and every expert op stayed on the GPU.
#[cfg(feature = "hip-tests")]
#[test]
fn victim_ring_exclusive_matches_inclusive_output() {
    use golem_engine::TurnRequest;
    use golem_engine::config::Residency;
    use golem_engine::session::expert_placement;

    const LONG: &[ChatTurn<'static>] = &[ChatTurn {
        role: "user",
        content: "one two three four five six seven eight nine ten eleven twelve thirteen \
                  fourteen fifteen sixteen seventeen eighteen nineteen twenty",
    }];
    const NEXT: &[ChatTurn<'static>] = &[ChatTurn {
        role: "user",
        content: "count to ten",
    }];
    let gpu = |residency: Residency, victim_ring_slots: u32| {
        let off = common::offload::offload_config();
        EngineConfig {
            n_gpu_layers: off.n_gpu_layers,
            cpu_moe_layers: off.cpu_moe_layers,
            residency,
            victim_ring_slots,
            ..pooled()
        }
    };
    let turn = |s: &mut Session, msgs: &[ChatTurn<'_>]| {
        let mut out = String::new();
        s.turn(&TurnRequest::new(msgs).max_tokens(Some(16)), &mut |t| {
            out.push_str(t)
        })
        .expect("turn");
        out
    };
    let two_turns = |cfg: &EngineConfig| {
        let mut s = Session::new(cfg).expect("session");
        let a = turn(&mut s, LONG);
        let b = turn(&mut s, NEXT);
        (a, b, s.device_cache_stats().expect("pooled GPU split"))
    };
    let before = expert_placement();
    let (a0, b0, _) = two_turns(&gpu(Residency::Inclusive, 0));
    assert!(!a0.is_empty() && !b0.is_empty());
    for slots in [1, 4] {
        let (a1, b1, dc) = two_turns(&gpu(Residency::Exclusive, slots));
        assert_eq!(
            (&a1, &b1),
            (&a0, &b0),
            "victim ring ({slots} slots) changed output"
        );
        assert!(dc.evictions > 0, "K = 4 never evicted: {dc:?}");
        // Logged, not asserted: the fixture does not guarantee >1 parked
        // eviction per (layer, compute), so a 1-slot ring may never be full.
        eprintln!(
            "victim ring {slots} slots: units={} full={}",
            dc.victim_ring_units, dc.victim_ring_full
        );
        assert!(
            dc.victim_ring_units > 0,
            "no plane went through the ring: {dc:?}"
        );
    }
    assert_eq!(
        expert_placement().on_cpu,
        before.on_cpu,
        "expert MUL_MAT_ID ran on the CPU"
    );
}

/// A ring whose device allocation fails (injected through the substrate's
/// test seam, engine-registered hooks) fails the turn instead of silently
/// falling back to the same-stream D2H the run did not ask for.
#[cfg(feature = "hip-tests")]
#[test]
#[allow(unsafe_code)] // FFI test seam (ring allocation failure, cache reset)
fn victim_ring_setup_failure_fails_the_turn() {
    use golem_engine::TurnRequest;
    use golem_engine::config::Residency;
    use golem_engine::error::EngineError;

    let off = common::offload::offload_config();
    let cfg = EngineConfig {
        n_gpu_layers: off.n_gpu_layers,
        cpu_moe_layers: off.cpu_moe_layers,
        residency: Residency::Exclusive,
        victim_ring_slots: 4,
        ..pooled()
    };
    let mut s = Session::new(&cfg).expect("session");
    let ctx = s.p3_ctx_ptr().expect("pooled ⇒ live ctx");
    // SAFETY: FFI on the session's live ctx between turns; the ring is
    // allocated at the first eviction park of the next turn.
    unsafe {
        golem_llama_sys::raw::golem_reset_device_cache(ctx);
        golem_llama_sys::raw::golem_inject_victim_ring_alloc_failure(ctx);
    }
    let msgs = [ChatTurn {
        role: "user",
        content: "one two three four five six seven eight nine ten eleven twelve thirteen \
                  fourteen fifteen sixteen seventeen eighteen nineteen twenty",
    }];
    let err = s
        .turn(&TurnRequest::new(&msgs).max_tokens(Some(16)), &mut |_| {})
        .expect_err("a failed ring setup must fail the turn");
    assert!(matches!(err, EngineError::VictimRingUnavailable), "{err}");
}
