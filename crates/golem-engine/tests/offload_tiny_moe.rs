//! Acceptance: GPU-offload plumbing in `EngineConfig`. Defaults
//! must leave the tiny-moe CPU decode byte-identical, and the `offloaded()`
//! dump-header hook must report the config's offload intent. CPU substrate.
mod common;

use golem_engine::config::EngineConfig;
use golem_engine::session::{ChatTurn, Session};
use golem_llama_sys::{Backend, substrate_backend};

fn turns() -> [ChatTurn<'static>; 1] {
    [ChatTurn {
        role: "user",
        content: "hello",
    }]
}

fn decode(session: &mut Session) -> (u32, String) {
    let mut out = String::new();
    let n = session
        .chat_turn(&turns(), &mut |piece| out.push_str(piece))
        .unwrap();
    (n, out)
}

#[test]
fn default_config_is_not_offloaded() {
    let cfg = EngineConfig {
        model_path: common::tiny_moe_path(),
        n_predict: 8,
        ..Default::default()
    };
    let session = Session::new(&cfg).unwrap();
    assert!(!session.offloaded(), "n_gpu_layers defaults to 0");
}

#[test]
fn offloaded_reports_requested_layers() {
    let cfg = EngineConfig {
        model_path: common::tiny_moe_path(),
        n_predict: 8,
        n_gpu_layers: 99,
        ..Default::default()
    };
    let mut session = Session::new(&cfg).unwrap();
    assert!(session.offloaded());
    // A CPU-only substrate has no devices to offload to, so the decode must
    // still complete (llama.cpp clamps to 0 actually-offloaded layers).
    let (n, out) = decode(&mut session);
    assert!(
        n > 0 && !out.is_empty(),
        "offload config must not break decode"
    );
}

#[test]
fn cpu_moe_overrides_keep_cpu_decode_byte_identical() {
    let baseline = {
        let cfg = EngineConfig {
            model_path: common::tiny_moe_path(),
            n_predict: 8,
            ..Default::default()
        };
        decode(&mut Session::new(&cfg).unwrap())
    };
    let offload_cfg = EngineConfig {
        model_path: common::tiny_moe_path(),
        n_predict: 8,
        n_gpu_layers: 99,
        cpu_moe_layers: Some(2),
        ..Default::default()
    };
    let with_overrides = decode(&mut Session::new(&offload_cfg).unwrap());
    // On the CPU substrate there are no GPU devices: n_gpu_layers clamps to 0
    // actually-offloaded layers and the expert buft overrides re-pin tensors
    // that were already on CPU. The decode must be byte-identical (CPU
    // reduction order is bit-deterministic; gate runs the cpu backend).
    if substrate_backend() == Backend::Cpu {
        assert_eq!(baseline, with_overrides);
    }
}

#[test]
fn flash_attn_on_loads_and_decodes() {
    let cfg = EngineConfig {
        model_path: common::tiny_moe_path(),
        n_predict: 8,
        flash_attn_on: true,
        ..Default::default()
    };
    let mut session = Session::new(&cfg).unwrap();
    let (n, out) = decode(&mut session);
    assert!(
        n > 0 && !out.is_empty(),
        "flash_attn_on must not break the decode"
    );
}
