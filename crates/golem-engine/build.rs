//! Re-exports the substrate backend as a compile-time cfg so HIP-only FFI
//! (GPU-pinned host memory, §3.5) compiles only where the HIP runtime
//! is actually linked.

fn main() {
    println!("cargo:rustc-check-cfg=cfg(golem_backend_hip)");
    if std::env::var("DEP_LLAMA_CPP_BACKEND").as_deref() == Ok("hip") {
        println!("cargo:rustc-cfg=golem_backend_hip");
    }
}
