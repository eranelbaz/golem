//! Pinned tiny-GGUF fixture resolver shared by this crate's integration tests
//! (M1.3's hardware tests reuse `common::fixture()`).
//!
//! Safe std/sha2 only — no unsafe, no lint attributes (if one is ever added,
//! it must be line 1 of the file, `mod common;` inner attr).

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use sha2::{Digest, Sha256};

/// Pinned fixture: stories15M q4_0 — the exact file (same sha256) the vendored
/// substrate's own suite pins in `tests/CMakeLists.txt` at this commit, from
/// the same repo path its `cmake/download-models.cmake` fetches.
const FIXTURE_URL: &str =
    "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf";
const FIXTURE_SHA256: &str = "66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739";
/// Hash-embedded filename: `stories15M-q4_0-<first12ofsha256>.gguf`.
const FIXTURE_NAME: &str = "stories15M-q4_0-66967fbece6d.gguf";

/// Memoized resolution — one download (and one `fixture-ok`) per process, no
/// concurrent-test re-fetch.
static FIXTURE: OnceLock<PathBuf> = OnceLock::new();

/// Resolve the pinned test model: `GOLEM_TEST_MODEL` override, then cache
/// slots (`~/.cache/golem/test-models/` download destination, repo-local
/// `.test-cache/` read-only pre-seed), then pinned download. Prints
/// `fixture-ok: <sha256> <resolved-path>` on every resolution.
pub fn fixture() -> PathBuf {
    FIXTURE.get_or_init(resolve).clone()
}

fn resolve() -> PathBuf {
    // 1. Explicit override wins outright — but is still hash-verified.
    if let Ok(p) = std::env::var("GOLEM_TEST_MODEL") {
        let p = PathBuf::from(p);
        assert!(p.is_file(), "GOLEM_TEST_MODEL={p:?} is not a regular file");
        let got = sha256_file(&p).unwrap_or_else(|e| panic!("GOLEM_TEST_MODEL={p:?}: {e}"));
        assert!(
            got == FIXTURE_SHA256,
            "GOLEM_TEST_MODEL={p:?} hash mismatch: got sha256 {got}, want {FIXTURE_SHA256}"
        );
        println!("fixture-ok: {got} {}", p.display());
        return p;
    }

    // 2. Download slot (~/.cache/golem/test-models/) — CI-cache-compatible,
    //    never dirties the worktree; the only slot this helper writes.
    let cached = download_slot().join(FIXTURE_NAME);
    if let Some(got) = check_download_slot_hit(&cached) {
        println!("fixture-ok: {got} {}", cached.display());
        return cached;
    }

    // 3. Repo-local pre-seed slot — checked but NEVER written (no deletes
    //    here either; a corrupt pre-seed just warns and falls through).
    let preseed = preseed_slot().join(FIXTURE_NAME);
    if preseed.is_file() {
        match sha256_file(&preseed) {
            Ok(got) if got == FIXTURE_SHA256 => {
                println!("fixture-ok: {got} {}", preseed.display());
                return preseed;
            }
            Ok(got) => eprintln!(
                "warning: corrupt fixture {preseed:?}: got sha256 {got}, want {FIXTURE_SHA256} — ignoring pre-seed"
            ),
            Err(e) => eprintln!("warning: unreadable fixture {preseed:?}: {e} — ignoring pre-seed"),
        }
    }

    // 4. Pinned download into the download slot.
    download();
    let final_path = download_slot().join(FIXTURE_NAME);
    let got = sha256_file(&final_path)
        .unwrap_or_else(|e| panic!("downloaded fixture {final_path:?} unreadable: {e}"));
    assert!(
        got == FIXTURE_SHA256,
        "downloaded fixture {final_path:?} hash mismatch: got sha256 {got}, want {FIXTURE_SHA256}"
    );
    println!("fixture-ok: {got} {}", final_path.display());
    final_path
}

/// Download slot directory (the helper creates it before any write).
fn download_slot() -> PathBuf {
    let home = std::env::var("HOME").expect("HOME not set — cannot resolve fixture cache");
    Path::new(&home).join(".cache/golem/test-models")
}

/// Repo-local pre-seed slot (relative to this crate's manifest dir).
fn preseed_slot() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.test-cache")
}

/// Hash-check a candidate in the writable download slot. On hash mismatch:
/// delete the corrupt file, warn, fall through. `Some(sha256)` on a hit.
fn check_download_slot_hit(path: &Path) -> Option<String> {
    if !path.is_file() {
        return None;
    }
    let got = sha256_file(path).unwrap_or_else(|e| panic!("cache slot fixture {path:?}: {e}"));
    if got == FIXTURE_SHA256 {
        return Some(got);
    }
    fs::remove_file(path)
        .unwrap_or_else(|e| eprintln!("warning: could not remove corrupt {path:?}: {e}"));
    eprintln!(
        "warning: deleted corrupt fixture {path:?}: got sha256 {got}, want {FIXTURE_SHA256} — falling through to download"
    );
    None
}

/// Streamed sha256 (hex) — no whole-file buffering.
fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let digest = hasher.finalize();
    Ok(digest.iter().map(|b| format!("{b:02x}")).collect())
}

/// `curl -fL` the pinned fixture into a pid+thread-unique temp file inside the
/// download slot (same filesystem → atomic rename, no concurrent-test
/// collision), sha256-verify BEFORE renaming into place, panic loudly with
/// remediation on any failure.
fn download() {
    let slot = download_slot();
    fs::create_dir_all(&slot).unwrap_or_else(|e| panic!("cannot create cache dir {slot:?}: {e}"));
    let tmp = slot.join(format!(
        ".{FIXTURE_NAME}.{}.{}.tmp",
        std::process::id(),
        // e.g. ThreadId(3) — Debug format, unique per process
        format!("{:?}", std::thread::current().id())
    ));
    eprintln!("downloading pinned fixture to {tmp:?}");
    let status = std::process::Command::new("curl")
        .args([
            "-fL",
            "--retry",
            "3",
            "--retry-delay",
            "2",
            "--connect-timeout",
            "15",
            "--max-time",
            "300",
            "-o",
        ])
        .arg(&tmp)
        .arg(FIXTURE_URL)
        .status()
        .unwrap_or_else(|e| panic!("failed to spawn curl for {FIXTURE_URL}: {e}"));
    if !status.success() {
        clean_fail(&tmp, format!("curl exited {status} for {FIXTURE_URL}"));
    }
    let got = match sha256_file(&tmp) {
        Ok(got) => got,
        Err(e) => clean_fail(&tmp, format!("downloaded temp file unreadable: {e}")),
    };
    if got != FIXTURE_SHA256 {
        clean_fail(
            &tmp,
            format!(
                "downloaded {FIXTURE_URL} hash mismatch: got sha256 {got}, want {FIXTURE_SHA256}"
            ),
        );
    }
    if let Err(e) = fs::rename(&tmp, slot.join(FIXTURE_NAME)) {
        clean_fail(&tmp, format!("atomic rename into {slot:?} failed: {e}"));
    }
}

/// Remove the temp file and panic with the got/want evidence plus remediation.
fn clean_fail(tmp: &Path, msg: String) -> ! {
    let _ = fs::remove_file(tmp);
    panic!(
        "{msg}\
         \nremediation: set GOLEM_TEST_MODEL=/path/to/{FIXTURE_NAME} \
         (sha256 {FIXTURE_SHA256}), or pre-seed {} or {} with that file",
        download_slot().join(FIXTURE_NAME).display(),
        preseed_slot().join(FIXTURE_NAME).display(),
    );
}
