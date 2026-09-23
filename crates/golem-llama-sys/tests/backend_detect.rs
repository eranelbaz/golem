#![allow(unsafe_code, missing_docs)]

fn auto_rule() -> golem_llama_sys::Backend {
    if golem_llama_sys::backend_probe::usable_rocm_toolchain() {
        golem_llama_sys::Backend::Hip
    } else {
        golem_llama_sys::Backend::Cpu
    }
}

#[test]
fn backend_matches_resolution_rules() {
    match golem_llama_sys::backend_env_input() {
        "cpu" => assert_eq!(
            golem_llama_sys::substrate_backend(),
            golem_llama_sys::Backend::Cpu
        ),
        "hip" => assert_eq!(
            golem_llama_sys::substrate_backend(),
            golem_llama_sys::Backend::Hip
        ),
        "auto" => assert_eq!(golem_llama_sys::substrate_backend(), auto_rule()),
        other => panic!("build.rs recorded unknown env input {other:?}"),
    }
}
