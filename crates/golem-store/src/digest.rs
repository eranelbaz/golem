//! Shard digests for the start-time check (§4.3 R-8): full SHA-256 and the
//! sampled-window SHA-256 recorded in the manifest.

use sha2::Digest;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Once;

/// Digest failure.
#[derive(Debug, thiserror::Error)]
pub enum DigestError {
    /// I/O failure.
    #[error("{path}: {source}")]
    Io {
        /// File path.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: std::io::Error,
    },
    /// The file's hash disagrees with the manifest.
    #[error("{path}: sha256 {got} != manifest {want}")]
    Mismatch {
        /// Hashed file.
        path: PathBuf,
        /// Computed digest.
        got: String,
        /// Manifest digest.
        want: String,
    },
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Lowercase sha256 hex.
#[must_use]
pub fn sha256_hex(bytes: &[u8]) -> String {
    hex(&sha2::Sha256::digest(bytes))
}

/// Default read chunk: big enough for NVMe sequential throughput.
pub const CHUNK_BYTES: usize = 8 << 20;

/// Warn once per process when a `POSIX_FADV_DONTNEED` advise fails (advisory:
/// a failure costs cache pressure, never correctness).
static DONTNEED_WARN: Once = Once::new();

fn warn_dontneed(path: &Path) {
    DONTNEED_WARN.call_once(|| {
        tracing::warn!(path = %path.display(), "posix_fadvise DONTNEED failed; \
            expert bytes may linger in the page cache");
    });
}

fn read_span(f: &mut std::fs::File, path: &Path, buf: &mut [u8]) -> Result<usize, DigestError> {
    let mut done = 0;
    while done < buf.len() {
        match f.read(&mut buf[done..]).map_err(|e| DigestError::Io {
            path: path.to_path_buf(),
            source: e,
        })? {
            0 => break,
            n => done += n,
        }
    }
    Ok(done)
}

/// Re-hashes all of `path` in `chunk`-byte reads (pages dropped behind the
/// cursor) and compares against `want_sha`.
///
/// # Errors
/// [`DigestError::Io`] on filesystem failures, [`DigestError::Mismatch`]
/// when the digest disagrees.
pub fn verify_sha256(path: &Path, want_sha: &str, chunk: usize) -> Result<(), DigestError> {
    let mut f = std::fs::File::open(path).map_err(|e| DigestError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut hasher = sha2::Sha256::new();
    let mut buf = vec![0u8; chunk.max(1)];
    let mut off = 0u64;
    loop {
        let n = read_span(&mut f, path, &mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
        if !crate::sys::drop_page_cache(&f, off, n) {
            warn_dontneed(path);
        }
        off += n as u64;
    }
    let got = hex(&hasher.finalize());
    if got == want_sha {
        Ok(())
    } else {
        Err(DigestError::Mismatch {
            path: path.to_path_buf(),
            got,
            want: want_sha.to_owned(),
        })
    }
}

/// R-8 sample count and window: 16 × 1 MiB per shard.
pub const SAMPLES: u64 = 16;
/// R-8 sample window bytes.
pub const SAMPLE_BYTES: u64 = 1 << 20;

/// SHA-256 hex of each window `[i·(len−1 MiB)/15, +1 MiB)` of `path`: the
/// first starts at 0 and the last ends at EOF (a file under 1 MiB is one
/// window repeated). Pages dropped after each read so expert bytes stay out
/// of the page cache. Target: 6 shards × 16 MiB = 96 MiB read at start vs
/// 186 GiB for a full hash.
///
/// # Errors
/// [`DigestError::Io`] on filesystem failures.
pub fn sampled_sha256(path: &Path) -> Result<Vec<String>, DigestError> {
    use std::os::unix::fs::FileExt;
    let io = |e| DigestError::Io {
        path: path.to_path_buf(),
        source: e,
    };
    let f = std::fs::File::open(path).map_err(io)?;
    let len = f.metadata().map_err(io)?.len();
    let mut buf = vec![0u8; SAMPLE_BYTES as usize];
    (0..SAMPLES)
        .map(|i| {
            let off = i * len.saturating_sub(SAMPLE_BYTES) / (SAMPLES - 1);
            let n = SAMPLE_BYTES.min(len - off) as usize;
            f.read_exact_at(&mut buf[..n], off).map_err(io)?;
            if !crate::sys::drop_page_cache(&f, off, n) {
                warn_dontneed(path);
            }
            Ok(sha256_hex(&buf[..n]))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file(tag: &str, bytes: &[u8]) -> PathBuf {
        let p = std::env::temp_dir().join(format!("golem-digest-{tag}-{}", std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn sampled_windows_span_first_and_last_mib() {
        let len = 3 << 20;
        let bytes: Vec<u8> = (0..len).map(|i| (i * 31 % 251) as u8).collect();
        let p = file("span", &bytes);
        let w = sampled_sha256(&p).unwrap();
        assert_eq!(w.len(), SAMPLES as usize);
        assert_eq!(w[0], sha256_hex(&bytes[..1 << 20]));
        assert_eq!(w[15], sha256_hex(&bytes[len - (1 << 20)..]));
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn short_and_empty_files_repeat_one_window() {
        let p = file("short", &[7u8; 4096]);
        let w = sampled_sha256(&p).unwrap();
        assert!(w.iter().all(|h| *h == sha256_hex(&[7u8; 4096])));
        std::fs::write(&p, b"").unwrap();
        let w = sampled_sha256(&p).unwrap();
        assert_eq!(w, vec![sha256_hex(b""); SAMPLES as usize]);
        let _ = std::fs::remove_file(p);
    }

    #[test]
    fn full_hash_is_chunk_invariant_and_reports_mismatch() {
        let bytes: Vec<u8> = (0..100_003u32).map(|i| (i % 251) as u8).collect();
        let p = file("full", &bytes);
        let want = sha256_hex(&bytes);
        for chunk in [1, 7, 4096, CHUNK_BYTES] {
            verify_sha256(&p, &want, chunk).unwrap();
        }
        assert!(matches!(
            verify_sha256(&p, &sha256_hex(b"x"), 4096),
            Err(DigestError::Mismatch { .. })
        ));
        let _ = std::fs::remove_file(p);
    }
}
