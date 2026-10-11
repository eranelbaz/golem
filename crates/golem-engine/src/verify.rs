//! Start-time model check (§4.3 R-8): per shard size + mtime against the
//! manifest plus 16 sampled 1 MiB SHA-256 windows; `full` re-hashes every
//! byte. Full verification belongs to import; this catches a swapped,
//! truncated or rewritten shard before weights load.

use std::path::Path;

use golem_store::import::{sampled_sha256, verify_replica};
use golem_store::{ModelManifest, load_manifest};

use crate::error::EngineError;

/// Checks the shards of `manifest_path` that live next to `model` (the
/// replica the instance loads). Returns warnings (shards without samples).
pub fn verify_model(
    model: &Path,
    manifest_path: &Path,
    full: bool,
) -> Result<Vec<String>, EngineError> {
    let m = load_manifest(manifest_path).map_err(|e| EngineError::InvalidModel(e.to_string()))?;
    verify_shards(model.parent().unwrap_or(Path::new(".")), &m, full)
}

/// [`verify_model`] over an already loaded manifest, shards under `dir`.
pub fn verify_shards(
    dir: &Path,
    m: &ModelManifest,
    full: bool,
) -> Result<Vec<String>, EngineError> {
    let bad = |file: &str, what: String| EngineError::InvalidModel(format!("shard {file}: {what}"));
    let mut warnings = Vec::new();
    for s in &m.shards {
        let p = dir.join(&s.file);
        let md = std::fs::metadata(&p).map_err(|e| bad(&s.file, e.to_string()))?;
        if md.len() != s.bytes {
            return Err(bad(
                &s.file,
                format!("size {} != manifest {}", md.len(), s.bytes),
            ));
        }
        if let Some(want) = s.mtime {
            let got = md
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map_or(0, |d| d.as_secs());
            if got != want {
                return Err(bad(&s.file, format!("mtime {got} != manifest {want}")));
            }
        }
        if full {
            verify_replica(&p, &s.sha256, golem_store::import::CHUNK_BYTES)
                .map_err(|e| bad(&s.file, e.to_string()))?;
            continue;
        }
        match &s.sampled_sha256 {
            Some(want) => {
                let got = sampled_sha256(&p).map_err(|e| bad(&s.file, e.to_string()))?;
                if let Some(i) = (0..want.len().max(got.len())).find(|&i| want.get(i) != got.get(i))
                {
                    return Err(bad(&s.file, format!("sampled sha256 window {i} differs")));
                }
            }
            None => warnings.push(format!(
                "shard {}: manifest has no sampled_sha256 (pre-R-8 import); size + mtime only",
                s.file
            )),
        }
    }
    Ok(warnings)
}

#[cfg(test)]
mod tests {
    use super::*;
    use golem_store::ShardEntry;
    use std::os::unix::fs::FileExt;

    /// `len`-byte shard + manifest stamped from it; returns (dir, manifest).
    fn fixture(tag: &str, len: u32) -> (std::path::PathBuf, ModelManifest) {
        let dir = std::env::temp_dir().join(format!("golem-verify-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("m-00001-of-00001.gguf");
        let bytes: Vec<u8> = (0..len).map(|i| (i * 31 % 251) as u8).collect();
        std::fs::write(&p, &bytes).unwrap();
        let md = std::fs::metadata(&p).unwrap();
        let mtime = md
            .modified()
            .unwrap()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        let m = ModelManifest {
            name: "t".into(),
            shards: vec![ShardEntry {
                file: "m-00001-of-00001.gguf".into(),
                bytes: md.len(),
                sha256: golem_store::import::sha256_hex(&bytes),
                replicas: vec![dir.clone()],
                sampled_sha256: Some(sampled_sha256(&p).unwrap()),
                mtime: Some(mtime.as_secs()),
            }],
            ..Default::default()
        };
        (dir, m)
    }

    /// Flips the byte at `off`, keeping the file's mtime.
    fn flip(dir: &Path, off: u64) {
        let p = dir.join("m-00001-of-00001.gguf");
        let mtime = std::fs::metadata(&p).unwrap().modified().unwrap();
        let f = std::fs::File::options()
            .read(true)
            .write(true)
            .open(&p)
            .unwrap();
        let mut b = [0u8];
        f.read_exact_at(&mut b, off).unwrap();
        f.write_all_at(&[b[0] ^ 0xff], off).unwrap();
        f.set_modified(mtime).unwrap();
    }

    // 32 MiB shard: windows start every 31/15 MiB and are 1 MiB long, so
    // [1 MiB, 2 MiB) is unsampled and the last window ends at EOF.
    const LEN: u32 = 32 << 20;

    #[test]
    fn flip_inside_sampled_window_fails() {
        let (dir, m) = fixture("in", LEN);
        flip(&dir, u64::from(LEN) - 1); // last byte: window 15 ends at EOF
        assert!(verify_shards(&dir, &m, false).is_err());
    }

    #[test]
    fn flip_outside_windows_passes_sampled_but_full_catches_it() {
        let (dir, m) = fixture("out", LEN);
        flip(&dir, (1 << 20) + 17);
        assert!(
            verify_shards(&dir, &m, false).is_ok(),
            "outside every window"
        );
        assert!(
            verify_shards(&dir, &m, true).is_err(),
            "--full-verify catches it"
        );
    }

    #[test]
    fn size_or_mtime_mismatch_fails_and_missing_samples_warn() {
        let (dir, mut m) = fixture("meta", 1 << 20);
        m.shards[0].bytes += 1;
        assert!(verify_shards(&dir, &m, false).is_err(), "size");
        m.shards[0].bytes -= 1;
        m.shards[0].mtime = m.shards[0].mtime.map(|t| t + 1);
        assert!(verify_shards(&dir, &m, false).is_err(), "mtime");
        m.shards[0].mtime = m.shards[0].mtime.map(|t| t - 1);
        m.shards[0].sampled_sha256 = None;
        assert_eq!(verify_shards(&dir, &m, false).unwrap().len(), 1, "warns");
    }
}
