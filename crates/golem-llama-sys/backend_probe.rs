//! Usable-ROCm-toolchain probe, `#[path]`-shared by build.rs (backend
//! auto-detect) and tests/backend_detect.rs — one probe, so the two
//! consumers cannot drift.
use std::path::Path;

pub fn usable_rocm_toolchain() -> bool {
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
