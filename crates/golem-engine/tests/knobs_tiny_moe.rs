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
fn io_depth_default_and_two_keep_greedy_output() {
    let d = EngineConfig::default().io_depth;
    let (base, want) = assert_default_unchanged(|c| c.io_depth = d);
    let two = EngineConfig {
        io_depth: 2,
        ..base
    };
    assert_eq!(greedy(&two).unwrap(), want);
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

#[test]
fn async_h2d_default_keeps_output_and_on_is_refused() {
    let d = EngineConfig::default().async_h2d;
    assert!(!d, "the scheduler's in-stream H2D is today's behaviour");
    let (base, _) = assert_default_unchanged(|c| c.async_h2d = d);
    let on = EngineConfig {
        async_h2d: true,
        ..base
    };
    let e = greedy(&on).unwrap_err().to_string();
    assert!(e.contains("async_h2d"), "{e}");
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
