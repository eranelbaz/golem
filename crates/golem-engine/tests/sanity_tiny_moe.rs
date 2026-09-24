//! tiny-moe must pass the §4.3-subset sanity checks and load + self-check-probe
//! on whatever backend is built. Runs on CPU in CI.
mod common;

use golem_engine::config::EngineConfig;
use golem_engine::gguf;
use golem_engine::sanity;
use golem_engine::session;

#[test]
fn tiny_moe_passes_sanity_and_loads_with_probe() {
    let path = common::tiny_moe_path();
    let meta = gguf::read_metadata(&path).unwrap();
    assert_eq!(meta.kv_str("general.architecture"), Some("glm4moe"));
    sanity::check(&meta).unwrap();

    let cfg = EngineConfig {
        model_path: path,
        ..EngineConfig::default()
    };
    let loaded = session::load(&cfg).unwrap();
    assert_eq!(loaded.metadata_arch(), "glm4moe");
}
