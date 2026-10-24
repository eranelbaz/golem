use std::path::{Path, PathBuf};

pub const ROCM_PATH_ENV: &str = "ROCM_PATH";

pub fn rocm_root() -> PathBuf {
    PathBuf::from(
        std::env::var(ROCM_PATH_ENV)
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "/opt/rocm".into()),
    )
}

/// ROCm toolchain files under `root` that auto-detect requires but cannot find.
pub fn missing_rocm_paths(root: &Path) -> Vec<PathBuf> {
    #[cfg(unix)]
    fn executable(path: &Path) -> bool {
        use std::os::unix::fs::PermissionsExt;
        path.is_file()
            && path
                .metadata()
                .map(|m| m.permissions().mode() & 0o111 != 0)
                .unwrap_or(false)
    }
    #[cfg(not(unix))]
    fn executable(path: &Path) -> bool {
        path.is_file()
    }
    let mut missing: Vec<PathBuf> = ["bin/hipcc", "llvm/bin/clang++"]
        .iter()
        .map(|p| root.join(p))
        .filter(|p| !executable(p))
        .collect();
    let libhip = root.join("lib/libamdhip64.so");
    if !libhip.exists() {
        missing.push(libhip);
    }
    missing
}

pub fn usable_rocm_toolchain() -> bool {
    missing_rocm_paths(&rocm_root()).is_empty()
}
