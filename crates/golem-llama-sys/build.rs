#![allow(missing_docs)]
#[path = "backend_probe.rs"]
mod backend_probe;
#[path = "pin_gate.rs"]
mod pin_gate;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const BACKEND_ENV: &str = "GOLEM_LLAMA_SYS_BACKEND";
const SKIP_GATE_ENV: &str = "GOLEM_LLAMA_SYS_SKIP_PIN_GATE";
const AMDGPU_TARGET_ENV: &str = "GOLEM_LLAMA_SYS_AMDGPU_TARGET";
const DEFAULT_AMDGPU_TARGET: &str = "gfx1100";
const ROCM_HIP_COMPILER: &str = "/opt/rocm/llvm/bin/clang++";

fn detect_backend() -> (&'static str, &'static str) {
    match env::var(BACKEND_ENV).as_deref() {
        Ok("cpu") => ("cpu", "cpu"),
        Ok("hip") => ("hip", "hip"),
        Ok("auto") | Err(env::VarError::NotPresent) => {
            let hip = backend_probe::usable_rocm_toolchain();
            ("auto", if hip { "hip" } else { "cpu" })
        }
        Ok(other) => panic!(
            "{BACKEND_ENV}: unknown value {other:?} (expected auto|cpu|hip); \
             hint: set GOLEM_LLAMA_SYS_BACKEND=cpu to force the CPU substrate"
        ),
        Err(env::VarError::NotUnicode(value)) => panic!(
            "{BACKEND_ENV}: non-UTF-8 value {value:?} (expected auto|cpu|hip); \
             hint: set GOLEM_LLAMA_SYS_BACKEND=cpu to force the CPU substrate"
        ),
    }
}

fn main() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        panic!(
            "golem-llama-sys is Linux-only at this pin: the ordered link table \
             (dylib=stdc++/gomp/pthread/dl/m) has no macOS/Windows derivation"
        );
    }
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir.join("../..");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    println!(
        "cargo:rerun-if-changed={}",
        repo_root.join("vendor/llama.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        repo_root.join("vendor/VENDOR_SHA").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        repo_root.join("engine/llama-cpp.lock").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        repo_root
            .join("vendor/llama.cpp/ggml/include/golem-hooks.h")
            .display()
    );
    println!("cargo:rerun-if-env-changed={BACKEND_ENV}");
    println!("cargo:rerun-if-env-changed={SKIP_GATE_ENV}");
    println!("cargo:rerun-if-env-changed={AMDGPU_TARGET_ENV}");
    println!("cargo:rustc-check-cfg=cfg(golem_backend_hip)");

    let (raw, resolved) = detect_backend();
    // links = "llama-cpp" surfaces this downstream as DEP_LLAMA_CPP_BACKEND.
    println!("cargo:backend={resolved}");
    println!("cargo:rustc-env={BACKEND_ENV}={resolved}");
    println!("cargo:rustc-env=GOLEM_LLAMA_SYS_BACKEND_INPUT={raw}");

    fs::write(
        out_dir.join("backend_info.rs"),
        format!(
            "const BACKEND: Backend = Backend::{};\nconst BACKEND_INPUT: &str = {raw:?};\n",
            if resolved == "hip" { "Hip" } else { "Cpu" }
        ),
    )
    .expect("write backend_info.rs");

    let pin_skipped = run_pin_gate(&repo_root);
    let binary_dir = build_substrate(&repo_root, resolved, &out_dir, pin_skipped);

    if !pin_skipped {
        if let Err(e) = pin_gate::verify_pin(
            &repo_root.join("vendor/llama.cpp"),
            &repo_root.join("vendor/VENDOR_SHA"),
        ) {
            panic!("pin gate RE-CHECK after the cmake build failed: {e}");
        }
    }

    println!(
        "cargo:rustc-env=GOLEM_LLAMA_PIN_GATE_SKIPPED={}",
        if pin_skipped { "1" } else { "0" }
    );

    let openmp_enabled = read_openmp_enabled(&binary_dir);
    emit_link_directives(resolved, openmp_enabled, &binary_dir);
    generate_bindings(&repo_root, &out_dir);
}

fn run_pin_gate(repo_root: &Path) -> bool {
    let vendor = repo_root.join("vendor/llama.cpp");
    let vendor_sha = repo_root.join("vendor/VENDOR_SHA");

    if !vendor.is_dir() {
        panic!(
            "{} is absent — the vendor tree must be materialized before building; \
             run scripts/build-substrate.sh",
            vendor.display()
        );
    }

    if env::var(SKIP_GATE_ENV).as_deref() != Ok("1") {
        if let Err(e) = pin_gate::verify_pin(&vendor, &vendor_sha) {
            panic!(
                "pin gate failed: {e}\n\
                 remediation (re-vendor via scripts/build-substrate.sh):\n\
                 \x20 git clean -fdx vendor/llama.cpp && git checkout -- vendor/"
            );
        }
        let lock = fs::read_to_string(repo_root.join("engine/llama-cpp.lock"))
            .unwrap_or_else(|e| panic!("cannot read engine/llama-cpp.lock: {e}"));
        let commit = pin_gate::read_kv_value(&lock, "commit")
            .filter(|c| !c.is_empty())
            .unwrap_or_else(|| panic!("commit key missing in engine/llama-cpp.lock"));
        println!("pin-ok: {commit}");
        return false;
    }

    println!(
        "cargo:warning={SKIP_GATE_ENV}=1: SKIPPING the vendor pin gate — \
         never ship or commit a build made with the hatch set"
    );
    match pin_gate::verify_pin(&vendor, &vendor_sha) {
        Ok(()) => println!(
            "cargo:warning={SKIP_GATE_ENV}=1 is unnecessary: the vendor tree is \
             PRISTINE (pin-ok) — unset the hatch"
        ),
        Err(e) => println!(
            "cargo:warning={SKIP_GATE_ENV}=1: the vendor tree is genuinely drifted \
             ({e}) — remember to revert: git clean -fdx vendor/llama.cpp && git checkout -- vendor/"
        ),
    }
    true
}

fn build_substrate(repo_root: &Path, backend: &str, out_dir: &Path, pin_skipped: bool) -> PathBuf {
    let vendor = repo_root.join("vendor/llama.cpp");
    let build_root = out_dir.join(format!("build-{backend}"));

    if backend == "hip" {
        if !Path::new(ROCM_HIP_COMPILER).exists() {
            panic!(
                "backend hip requires the ROCm HIP compiler at {ROCM_HIP_COMPILER} (absent) — \
                 install ROCm or set GOLEM_LLAMA_SYS_BACKEND=cpu"
            );
        }
        rocm_advisory_check(repo_root);
    }

    let mut cfg = cmake::Config::new(&vendor);
    cfg.out_dir(&build_root)
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_COMMON", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_APP", "OFF")
        .define("GGML_BUILD_TESTS", "OFF")
        // Session-10: repack ON converts Q4_0 weights through a single
        // ~106 GiB host allocation — infeasible on 93 GiB RAM. Both
        // substrate builds disable it (matches the baseline-001 recipe).
        .define("GGML_CPU_REPACK", "OFF")
        .build_target("llama"); // not install: install builds common/tools/app transitively
    match backend {
        "cpu" => {
            cfg.define("GGML_HIP", "OFF");
        }
        "hip" => {
            let amdgpu_target = env::var(AMDGPU_TARGET_ENV)
                .ok()
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| DEFAULT_AMDGPU_TARGET.to_string());
            // CMake's HIP-language setup needs the ROCm root to emit `-x
            // hip`; without it clang compiles .cu as CUDA and rejects gfx1100.
            let rocm_root = std::env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".into());
            cfg.define("GGML_HIP", "ON")
                .env("ROCM_PATH", &rocm_root)
                .define("CMAKE_HIP_COMPILER_ROCM_ROOT", &rocm_root)
                .define("AMDGPU_TARGETS", &amdgpu_target)
                // CMake ≥ 4 HIP-language builds read this, not the legacy
                // AMDGPU_TARGETS; without it clang falls back to CUDA arch
                // probing and rejects gfx1100.
                .define("CMAKE_HIP_ARCHITECTURES", &amdgpu_target)
                .define("CMAKE_HIP_COMPILER", ROCM_HIP_COMPILER);
        }
        other => unreachable!("detect_backend only yields cpu|hip, got {other:?}"),
    }
    if pin_skipped {
        println!("cargo:warning={SKIP_GATE_ENV}=1 build: post-build pin re-check suppressed");
    }
    cfg.build();

    build_root.join("build")
}

fn rocm_advisory_check(repo_root: &Path) {
    let lock = repo_root.join("engine/llama-cpp.lock");
    let text =
        fs::read_to_string(&lock).unwrap_or_else(|e| panic!("cannot read {}: {e}", lock.display()));
    let pinned = pin_gate::read_kv_value(&text, "rocm")
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| panic!("rocm key missing in {}", lock.display()));
    let Some(installed) = fs::read_to_string("/opt/rocm/.info/version")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
    else {
        return;
    };
    if major_minor(&installed) != major_minor(&pinned) {
        println!(
            "cargo:warning=WARN: installed ROCm {installed}, lock pins {pinned} — \
             advisory minimum; continuing"
        );
    }
}

fn generate_bindings(repo_root: &Path, out_dir: &Path) {
    let vendor = repo_root.join("vendor/llama.cpp");
    let hooks_header = vendor.join("ggml/include/golem-hooks.h");
    let llama_header = vendor.join("include/llama.h");
    // size_t literal to the header; non-exhaustive enums keep future
    // llama.cpp enum additions non-breaking.
    let bindings = bindgen::Builder::default()
        .header(
            hooks_header
                .to_str()
                .expect("non-UTF-8 path to golem-hooks.h"),
        )
        .header(llama_header.to_str().expect("non-UTF-8 path to llama.h"))
        .clang_arg(format!("-I{}", vendor.join("include").display()))
        .clang_arg(format!("-I{}", vendor.join("ggml/include").display()))
        .allowlist_function("llama_.*")
        .allowlist_function("golem_.*")
        // GPU-offload plumbing (M1.4 Task 3): the `-ncmoe`-equivalent buft
        // overrides pin expert tensors to the CPU buffer type by handle; the
        // loader compares overrides against exactly this pointer.
        .allowlist_function("ggml_backend_cpu_buffer_type")
        // T1 op-parity leg (M1.4 Task 4): the minimal ggml FFI surface for
        // golem-engine/src/parity_op.rs (design §13: raw ggml FFI is confined
        // to that module, so the bindings stay as narrow as possible).
        // Context lifecycle:
        .allowlist_function("ggml_init")
        .allowlist_function("ggml_free")
        // Synthetic tensor creation (shapes [k, m, n_expert],
        // [k, 1, n_ids], [1, n_ids] i32 — mul_mat_id requires 2d/3d tensors):
        .allowlist_function("ggml_new_tensor_2d")
        .allowlist_function("ggml_new_tensor_3d")
        .allowlist_function("ggml_nbytes")
        // P1 hook-registration tests: tensor naming + expert-metadata math
        // (golem_tensor_expert_count/stride_bytes consume these over FFI).
        .allowlist_function("ggml_set_name")
        .allowlist_function("ggml_n_dims")
        .allowlist_function("ggml_type_size")
        .allowlist_function("ggml_blck_size")
        // The op itself plus graph build; compute goes through the backend
        // registry below (Task 4a), so no CPU-only compute entry is bound.
        .allowlist_function("ggml_mul_mat_id")
        .allowlist_function("ggml_new_graph")
        .allowlist_function("ggml_build_forward_expand")
        // Backend-registry compute (M1.4 Task 4a): the op must run on the
        // HIP device when the binary is HIP-linked, so compute goes through
        // the device registry instead of the CPU-only path above.
        .allowlist_function("ggml_backend_dev_by_type")
        .allowlist_function("ggml_backend_dev_type")
        .allowlist_function("ggml_backend_dev_name")
        .allowlist_function("ggml_backend_dev_init")
        .allowlist_function("ggml_backend_dev_buffer_type")
        .allowlist_function("ggml_backend_free")
        .allowlist_function("ggml_backend_alloc_ctx_tensors_from_buft")
        .allowlist_function("ggml_backend_buffer_free")
        .allowlist_function("ggml_backend_tensor_set")
        .allowlist_function("ggml_backend_tensor_get")
        .allowlist_function("ggml_backend_graph_compute")
        .allowlist_function("ggml_backend_cpu_set_n_threads")
        // Q8_0 variant quantizer:
        .allowlist_function("ggml_quantize_chunk")
        // Types used (or pinned for stability) by the parity_op.rs FFI:
        // the opaque handle trio, the init params struct, and the graph-
        // compute status enum (ggml_type arrives transitively via the
        // ggml_new_tensor_* signatures). ggml_threadpool is deliberately
        // absent — the leg pins threads via ggml_backend_cpu_set_n_threads.
        .allowlist_type("ggml_context")
        .allowlist_type("ggml_tensor")
        .allowlist_type("ggml_cgraph")
        .allowlist_type("ggml_init_params")
        .allowlist_type("ggml_status")
        // Opaque backend-registry handles + device-type enum (Task 4a):
        .allowlist_type("ggml_backend")
        .allowlist_type("ggml_backend_dev")
        .allowlist_type("ggml_backend_buffer")
        .allowlist_type("ggml_backend_buffer_type")
        .allowlist_type("ggml_backend_dev_type")
        .allowlist_type("llama_.*")
        .allowlist_type("golem_.*")
        .allowlist_var("LLAMA_.*")
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: true,
        })
        .size_t_is_usize(false)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .generate()
        .expect("bindgen failed on llama.h + golem-hooks.h");
    bindings.write_to_file(out_dir.join("bindings.rs")).unwrap();
}

fn major_minor(version: &str) -> String {
    version.split('.').take(2).collect::<Vec<_>>().join(".")
}

fn read_openmp_enabled(binary_dir: &Path) -> bool {
    let cache_path = binary_dir.join("CMakeCache.txt");
    let cache = fs::read_to_string(&cache_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", cache_path.display()));
    for line in cache.lines() {
        let Some(rest) = line
            .strip_prefix("GGML_OPENMP_ENABLED=")
            .or_else(|| line.strip_prefix("GGML_OPENMP_ENABLED:"))
        else {
            continue;
        };
        let value = match rest.split_once('=') {
            Some((_type, value)) => value,
            None => rest,
        };
        return match value.trim() {
            "ON" => true,
            "OFF" => {
                println!(
                    "cargo:warning=GGML_OPENMP_ENABLED=OFF in {} — the substrate \
                     built without OpenMP; dylib=gomp skipped",
                    cache_path.display()
                );
                false
            }
            other => panic!(
                "unexpected GGML_OPENMP_ENABLED value {other:?} in {} — \
                 the link table needs re-derivation",
                cache_path.display()
            ),
        };
    }
    panic!(
        "GGML_OPENMP_ENABLED not found in {} — the link table needs re-derivation",
        cache_path.display()
    )
}

fn emit_link_directives(backend: &str, openmp_enabled: bool, binary_dir: &Path) {
    let archives: &[(&str, &str)] = match backend {
        "cpu" => &[
            ("llama", "src"),
            ("ggml", "ggml/src"),
            ("ggml-cpu", "ggml/src"),
            ("ggml-base", "ggml/src"),
        ],
        "hip" => &[
            ("llama", "src"),
            ("ggml", "ggml/src"),
            ("ggml-cpu", "ggml/src"),
            ("ggml-hip", "ggml/src/ggml-hip"),
            ("ggml-base", "ggml/src"),
        ],
        other => unreachable!("detect_backend only yields cpu|hip, got {other:?}"),
    };

    let mut searched: Vec<PathBuf> = Vec::new();
    for (_, dir) in archives {
        let path = binary_dir.join(dir);
        if !searched.contains(&path) {
            searched.push(path);
        }
    }
    if backend == "hip" {
        searched.push(PathBuf::from("/opt/rocm/lib"));
    }
    for path in &searched {
        println!("cargo:rustc-link-search=native={}", path.display());
    }

    for (name, dir) in archives {
        let archive = binary_dir.join(dir).join(format!("lib{name}.a"));
        if !archive.exists() {
            panic!(
                "expected static archive {} is missing after the cmake build — \
                 inspect $OUT_DIR/build-{backend}/build and re-derive the link table",
                archive.display()
            );
        }
        println!("cargo:rustc-link-lib=static={name}");
    }

    println!("cargo:rustc-link-lib=dylib=stdc++");
    if openmp_enabled {
        println!("cargo:rustc-link-lib=dylib=gomp");
    }
    if backend == "hip" {
        println!("cargo:rustc-link-lib=dylib=hipblas");
        println!("cargo:rustc-link-lib=dylib=rocblas");
        println!("cargo:rustc-link-lib=dylib=amdhip64");
        println!("cargo:rustc-cfg=golem_backend_hip");
    }
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=m");
}
