//! Re-exports the substrate backend as `cfg(golem_backend_hip)` (mirrors
//! golem-engine/build.rs): `/metrics` emits the I2 off-GPU series only
//! where expert math is required to run on the GPU.

fn main() {
    println!("cargo:rustc-check-cfg=cfg(golem_backend_hip)");
    if std::env::var("DEP_LLAMA_CPP_BACKEND").as_deref() == Ok("hip") {
        println!("cargo:rustc-cfg=golem_backend_hip");
    }
}
