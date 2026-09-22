#![allow(unsafe_code, missing_docs)]

fn auto_rule() -> golem_llama_sys::Backend {
    let usable_rocm = std::path::Path::new("/opt/rocm/bin/hipcc").exists()
        && std::path::Path::new("/opt/rocm/lib/libamdhip64.so").exists();
    if usable_rocm {
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
