//! Resolves the tiny-moe fixture: generated in-process via the xtask dev-dep,
//! cached under .test-cache/ with a vocab-versioned filename. Weights never in git.
use std::path::PathBuf;

pub fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

pub fn tiny_moe_path() -> PathBuf {
    let cache = repo_root().join(".test-cache");
    std::fs::create_dir_all(&cache).unwrap();
    let p = cache.join("tiny-moe-320.gguf");
    if !p.exists() {
        let tmp = cache.join(format!("tiny-moe-320-{}.gguf.tmp", std::process::id()));
        let f = std::fs::File::create(&tmp).unwrap();
        xtask::tiny_moe::generate(&mut std::io::BufWriter::new(f), 0x67_6f_6c_65_6d).unwrap();
        std::fs::rename(&tmp, &p).unwrap();
    }
    p
}
