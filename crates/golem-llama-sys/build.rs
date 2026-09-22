//! Build script for `golem-llama-sys` (Task 2 slice):
//!
//! 1. resolve the compute backend (`GOLEM_LLAMA_SYS_BACKEND`=auto|cpu|hip —
//!    env wins, auto picks hip when a usable ROCm toolchain exists);
//! 2. ADR 0001 pin gate over `vendor/llama.cpp` (Rust port of the canonical
//!    `cs()` recipe, proven byte-identical to the shell — see `pin_gate.rs`
//!    and `tests/pin_gate_conformance.rs`);
//! 3. cmake configure+build of the vendored substrate via the `cmake` crate
//!    (Release, static, `.build_target("llama")` — NOT the default `install`
//!    target, which transitively builds common/tools/app);
//! 4. ordered per-backend link directives (see `emit_link_directives`);
//! 5. bindgen bindings over `llama.h` + `headers/golem-hooks.h` →
//!    `$OUT_DIR/bindings.rs`, included as `golem_llama_sys::raw` (see
//!    `generate_bindings`).

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

/// Usable-ROCm probe: `auto` selects hip only when both the HIP compiler and
/// the shared HIP runtime library exist AND hipcc is executable — a partial
/// install (files present, exec bits lost) must fall through to cpu instead
/// of failing deep inside cmake with a confusing error.
fn usable_rocm_toolchain() -> bool {
    #[cfg(unix)]
    fn hipcc_executable() -> bool {
        use std::os::unix::fs::PermissionsExt;
        let path = Path::new("/opt/rocm/bin/hipcc");
        path.is_file()
            && path
                .metadata()
                .map(|m| m.permissions().mode() & 0o111 != 0)
                .unwrap_or(false)
    }
    #[cfg(not(unix))]
    fn hipcc_executable() -> bool {
        Path::new("/opt/rocm/bin/hipcc").is_file()
    }
    hipcc_executable() && Path::new("/opt/rocm/lib/libamdhip64.so").exists()
}

/// Resolve the compute backend. Returns the raw env input (`auto|cpu|hip`) and
/// the resolved backend (`cpu|hip`). Explicit `auto` behaves identically to
/// unset: probe for a usable ROCm toolchain, hip when present, cpu otherwise
/// (final-review finding: `auto` used to fall into the unknown-value panic).
/// A non-UTF-8 env value panics instead of silently falling through to auto
/// (controller addendum #1).
fn detect_backend() -> (&'static str, &'static str) {
    match env::var(BACKEND_ENV).as_deref() {
        Ok("cpu") => ("cpu", "cpu"),
        Ok("hip") => ("hip", "hip"),
        Ok("auto") | Err(env::VarError::NotPresent) => {
            let hip = usable_rocm_toolchain();
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
    // The link table below is glibc/Linux-shaped (dylib=pthread/dl/m,
    // libstdc++); fail fast naming the platform instead of a link error.
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        panic!(
            "golem-llama-sys is Linux-only at this pin: the ordered link table \
             (dylib=stdc++/gomp/pthread/dl/m) has no macOS/Windows derivation"
        );
    }
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    // build scripts run with cwd = crate root, but all paths are rooted at
    // CARGO_MANIFEST_DIR explicitly; vendor/ lives two levels up.
    let repo_root = manifest_dir.join("../..");
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    // (d) Rerun discipline — no latchable state. Paths are absolute: joined
    // onto the absolute CARGO_MANIFEST_DIR (repo_root keeps its `../..`
    // components un-normalized, which cargo resolves fine). The vendor dir
    // is watched recursively by cargo, so any in-tree drift (including the
    // files the pin gate hashes) re-runs this script. Parsed headers
    // additionally get automatic rerun-if-changed from bindgen's
    // CargoCallbacks in step (e); the golem-hooks.h entry also covers edits
    // bindgen skips over (comments/whitespace).
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
        manifest_dir.join("headers/golem-hooks.h").display()
    );
    println!("cargo:rerun-if-env-changed={BACKEND_ENV}");
    println!("cargo:rerun-if-env-changed={SKIP_GATE_ENV}");
    println!("cargo:rerun-if-env-changed={AMDGPU_TARGET_ENV}");

    let (raw, resolved) = detect_backend();
    println!("cargo:rustc-env={BACKEND_ENV}={resolved}");
    println!("cargo:rustc-env=GOLEM_LLAMA_SYS_BACKEND_INPUT={raw}");

    // backend_info.rs (Task 1 contract, unchanged).
    fs::write(
        out_dir.join("backend_info.rs"),
        format!(
            "const BACKEND: Backend = Backend::{};\nconst BACKEND_INPUT: &str = {raw:?};\n",
            if resolved == "hip" { "Hip" } else { "Cpu" }
        ),
    )
    .expect("write backend_info.rs");

    // (a) Pin gate — before any cmake work: never build a drifted tree.
    let pin_skipped = run_pin_gate(&repo_root);

    // (b) Configure + build the substrate for this backend.
    let binary_dir = build_substrate(&repo_root, resolved, &out_dir, pin_skipped);

    // (a') Post-build re-verify: the gate→cmake window is TOCTOU — re-hash so
    // a tree mutated mid-build never ships silently. Cheap relative to cmake.
    if !pin_skipped {
        if let Err(e) = pin_gate::verify_pin(
            &repo_root.join("vendor/llama.cpp"),
            &repo_root.join("vendor/VENDOR_SHA"),
        ) {
            panic!("pin gate RE-CHECK after the cmake build failed: {e}");
        }
    }

    // (f) Pin-gate provenance in the artifacts: a hatch-skipped build is
    // detectable downstream via `golem_llama_sys::pin_gate_skipped()`
    // (CI/test asserts this stays false for shipped builds).
    println!(
        "cargo:rustc-env=GOLEM_LLAMA_PIN_GATE_SKIPPED={}",
        if pin_skipped { "1" } else { "0" }
    );

    // (b) OpenMP presence decides the `dylib=gomp` entry of the link table.
    let openmp_enabled = read_openmp_enabled(&binary_dir);

    // (c) Ordered link directives.
    emit_link_directives(resolved, openmp_enabled, &binary_dir);

    // (e) Bindgen bindings over llama.h + the golem-hooks ABI stub. Kept
    // after the cmake section for a simple linear ordering (headers parse
    // fine without the build; the pin gate above already ran).
    generate_bindings(&manifest_dir, &repo_root, &out_dir);
}

/// (a) ADR 0001 pin gate. `GOLEM_LLAMA_SYS_SKIP_PIN_GATE=1` skips the gate with
/// a loud warning (escape hatch, brief (e)); when the tree is actually clean
/// the hatch is flagged as unnecessary. Thanks to (d) the hatch never sticks
/// silently: unset, the next build re-runs the gate.
fn run_pin_gate(repo_root: &Path) -> bool {
    let vendor = repo_root.join("vendor/llama.cpp");
    let vendor_sha = repo_root.join("vendor/VENDOR_SHA");

    if !vendor.is_dir() {
        panic!(
            "{} is absent — the vendor tree must be materialized before building; \
             run the vendoring flow per ADR 0001",
            vendor.display()
        );
    }

    if env::var(SKIP_GATE_ENV).as_deref() != Ok("1") {
        if let Err(e) = pin_gate::verify_pin(&vendor, &vendor_sha) {
            panic!(
                "pin gate failed: {e}\n\
                 remediation (re-vendor per ADR 0001):\n\
                 \x20 git clean -fdx vendor/llama.cpp && git checkout -- vendor/"
            );
        }
        let lock = fs::read_to_string(repo_root.join("engine/llama-cpp.lock"))
            .unwrap_or_else(|e| panic!("cannot read engine/llama-cpp.lock: {e}"));
        let commit = pin_gate::read_kv_value(&lock, "commit")
            .filter(|c| !c.is_empty())
            .unwrap_or_else(|| panic!("commit key missing in engine/llama-cpp.lock"));
        println!("pin-ok: {commit}"); // visible with `cargo build -vv`
        return false;
    }

    println!(
        "cargo:warning={SKIP_GATE_ENV}=1: SKIPPING the ADR 0001 vendor pin gate — \
         never ship or commit a build made with the hatch set"
    );
    // Leftover-hatch detection: compute the hash anyway and say which way it
    // would have gone.
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

/// (b) cmake configure+build of the vendored substrate.
///
/// `cmake-rs` puts the binary dir at `<out_dir>/build`, and we pass
/// `$OUT_DIR/build-{backend}` as out_dir → binary dir
/// `$OUT_DIR/build-{backend}/build` (per-backend isolation; both dirs grow
/// monotonically across backend flips — multi-GB each — so the flip ritual is
/// `cargo clean -p golem-llama-sys`).
///
/// BUILD_SHARED_LIBS=OFF is required: the default is ON on Linux at this pin.
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
        .define("BUILD_SHARED_LIBS", "OFF") // required — default is ON on Linux at this pin
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_COMMON", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_BUILD_APP", "OFF")
        .define("GGML_BUILD_TESTS", "OFF")
        // no LLAMA_CURL define: deprecated no-op at this pin
        .build_target("llama"); // NOT install — that transitively builds common/tools/app
    match backend {
        "cpu" => {
            cfg.define("GGML_HIP", "OFF");
        }
        "hip" => {
            // Default gfx1100 matches the M1 hardware; overridable per-machine
            // (rerun-if-env-changed above) so other GPUs need no source edit.
            let amdgpu_target = env::var(AMDGPU_TARGET_ENV)
                .ok()
                .filter(|v| !v.is_empty())
                .unwrap_or_else(|| DEFAULT_AMDGPU_TARGET.to_string());
            cfg.define("GGML_HIP", "ON")
                .define("AMDGPU_TARGETS", amdgpu_target)
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

/// Advisory ROCm version check, ported from `scripts/build-substrate.sh`:
/// equal major.minor → ok; otherwise WARN; never fail. No installed-version
/// file → skip. Only meaningful for the hip backend.
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
        println!("cargo:warning=rocm-check: skipped (no /opt/rocm/.info/version)");
        return;
    };
    if major_minor(&installed) != major_minor(&pinned) {
        println!(
            "cargo:warning=WARN: installed ROCm {installed}, lock pins {pinned} — \
             advisory minimum; continuing"
        );
    }
    // equal major.minor → ok, silent (the shell's "rocm-check: ok" is visible
    // only under -vv anyway; absence of the WARN is the ok signal here).
}

/// (e) Bindgen bindings over `llama.h` + `headers/golem-hooks.h`, written to
/// `$OUT_DIR/bindings.rs` and included by `src/lib.rs` as `pub mod raw`.
///
/// `size_t_is_usize(false)` keeps generated signatures literal to the header;
/// Rust `{ non_exhaustive: true }` enums keep future llama.cpp enum additions
/// non-breaking; `CargoCallbacks` gives automatic rerun-if-changed for every
/// parsed header (llama.h transitively pulls the ggml headers, hence the two
/// `-I` search paths).
fn generate_bindings(manifest_dir: &Path, repo_root: &Path, out_dir: &Path) {
    let vendor = repo_root.join("vendor/llama.cpp");
    let hooks_header = manifest_dir.join("headers/golem-hooks.h");
    let llama_header = vendor.join("include/llama.h");
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

/// Read `GGML_OPENMP_ENABLED` from the binary dir's CMakeCache.txt.
/// `dylib=gomp` is emitted only when it is ON; OFF means the link table needs
/// re-derivation, which we refuse to guess.
fn read_openmp_enabled(binary_dir: &Path) -> bool {
    let cache_path = binary_dir.join("CMakeCache.txt");
    let cache = fs::read_to_string(&cache_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", cache_path.display()));
    for line in cache.lines() {
        if let Some(rest) = line.strip_prefix("GGML_OPENMP_ENABLED") {
            let Some((_type, value)) = rest.split_once('=') else {
                continue;
            };
            return match value.trim() {
                "ON" => true,
                "OFF" => panic!(
                    "GGML_OPENMP_ENABLED=OFF at this pin — the ordered link table \
                     (dylib=gomp) needs re-derivation"
                ),
                other => panic!(
                    "unexpected GGML_OPENMP_ENABLED value {other:?} in {} — \
                     the link table needs re-derivation",
                    cache_path.display()
                ),
            };
        }
    }
    panic!(
        "GGML_OPENMP_ENABLED not found in {} — the link table needs re-derivation",
        cache_path.display()
    )
}

/// (c) Ordered per-backend link directives. Static archives resolve
/// left-to-right and dependents must precede dependencies; order verified
/// against the real CMake graph at pin 86ebfef2:
/// `llama → ggml → {ggml-cpu, ggml-hip} → ggml-base` (acyclic).
///
/// As-built table (lib set and ordering match the plan verbatim; archive
/// LOCATIONS confirmed empirically in Step 3 — the ggml archives are flat in
/// `ggml/src/` at this pin except `ggml-hip`, which lands in its per-target
/// output subdir `ggml/src/ggml-hip`). Cross-checked against
/// `ldd` of the M0-built `.m0-work/build-cpu/bin/llama-cli`, whose system-lib
/// set is libstdc++/libgomp/libm/libc — libpthread/libdl live in libc at
/// glibc ≥ 2.34, and libssl/brotli/zstd there come from curl in common/server,
/// which this minimal `llama`-target build does not use:
///
/// CPU: static=llama (src/), static=ggml (ggml/src/),
///      static=ggml-cpu (ggml/src/), static=ggml-base (ggml/src/)
///      then dylib=stdc++, dylib=gomp (only when GGML_OPENMP_ENABLED=ON),
///           dylib=pthread, dylib=dl, dylib=m
/// HIP: static=llama (src/), static=ggml (ggml/src/),
///      static=ggml-cpu (ggml/src/), static=ggml-hip (ggml/src/ggml-hip),
///      static=ggml-base (ggml/src/)
///      then dylib=stdc++, dylib=gomp (conditional as above), dylib=hipblas,
///           dylib=rocblas, dylib=amdhip64, dylib=pthread, dylib=dl, dylib=m
///      extra search path /opt/rocm/lib (ROCm 7.2.4 ships these as .so only).
fn emit_link_directives(backend: &str, openmp_enabled: bool, binary_dir: &Path) {
    // (archive name, archive dir relative to the binary dir). Order is the
    // link order. cmake-rs does not override ARCHIVE_OUTPUT_DIRECTORY; at this
    // pin llama lands in src/ and the ggml archives in ggml/src/, except
    // ggml-hip, which lands in its per-target subdir ggml/src/ggml-hip.
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
            ("ggml-hip", "ggml/src/ggml-hip"), // separate target output dir at this pin
            ("ggml-base", "ggml/src"),
        ],
        other => unreachable!("detect_backend only yields cpu|hip, got {other:?}"),
    };

    // Search paths first (deduped, table order), then libs in order.
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
    }
    println!("cargo:rustc-link-lib=dylib=pthread");
    println!("cargo:rustc-link-lib=dylib=dl");
    println!("cargo:rustc-link-lib=dylib=m");
}
