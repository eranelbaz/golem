//! Build script: resolves the compute backend from `GOLEM_LLAMA_SYS_BACKEND`
//! (auto|cpu|hip) and emits rustc-env vars plus `backend_info.rs`.
//! The vendored llama.cpp cmake build lands in Task 2 — not here.

// build.rs (Task 1 slice)
fn detect_backend() -> (&'static str, &'static str) {
    match std::env::var("GOLEM_LLAMA_SYS_BACKEND").as_deref() {
        Ok("cpu") => ("cpu", "cpu"),
        Ok("hip") => ("hip", "hip"),
        Ok(other) => panic!(
            "GOLEM_LLAMA_SYS_BACKEND: unknown value {other:?} (expected auto|cpu|hip); \
             hint: set GOLEM_LLAMA_SYS_BACKEND=cpu to force the CPU substrate"
        ),
        _ => {
            let usable_rocm = std::path::Path::new("/opt/rocm/bin/hipcc").exists()
                && std::path::Path::new("/opt/rocm/lib/libamdhip64.so").exists();
            ("auto", if usable_rocm { "hip" } else { "cpu" })
        }
    }
}

fn main() {
    let (raw, resolved) = detect_backend();
    println!("cargo:rustc-env=GOLEM_LLAMA_SYS_BACKEND={resolved}");
    println!("cargo:rustc-env=GOLEM_LLAMA_SYS_BACKEND_INPUT={raw}");
    println!("cargo:rerun-if-env-changed=GOLEM_LLAMA_SYS_BACKEND");
    // backend_info.rs
    let out_dir = std::env::var("OUT_DIR").unwrap();
    std::fs::write(
        format!("{out_dir}/backend_info.rs"),
        format!(
            "const BACKEND: Backend = Backend::{};\nconst BACKEND_INPUT: &str = {raw:?};\n",
            if resolved == "hip" { "Hip" } else { "Cpu" }
        ),
    )
    .unwrap();
}
