//! Pinned tiny-GGUF fixture resolver shared by this crate's integration tests.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use sha2::{Digest, Sha256};

/// The exact file (same sha256) the vendored substrate's own suite pins in
/// `tests/CMakeLists.txt` at this commit.
const FIXTURE_URL: &str =
    "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf";
const FIXTURE_SHA256: &str = "66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739";

/// Hash-derived filename — never transcribed, so it can't drift from the pin.
fn fixture_name() -> String {
    format!("stories15M-q4_0-{}.gguf", &FIXTURE_SHA256[..12])
}

/// One download per process; no concurrent-test re-fetch.
static FIXTURE: OnceLock<PathBuf> = OnceLock::new();

/// Resolve the pinned test model: `GOLEM_TEST_MODEL` override, cache slot
/// (`~/.cache/golem/test-models/`), repo-local `.test-cache/` pre-seed, then
/// pinned download. Prints `fixture-ok: <sha256> <resolved-path>` on success.
pub fn fixture() -> PathBuf {
    FIXTURE.get_or_init(resolve).clone()
}

fn resolve() -> PathBuf {
    // Explicit override wins outright — but is still hash-verified.
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

    let cached = download_slot().join(fixture_name());
    if let Some(got) = check_download_slot_hit(&cached) {
        println!("fixture-ok: {got} {}", cached.display());
        return cached;
    }

    // Pre-seed slot is checked but never written; a corrupt pre-seed warns
    // and falls through.
    let preseed = preseed_slot().join(fixture_name());
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

    download();
    let final_path = download_slot().join(fixture_name());
    let got = sha256_file(&final_path)
        .unwrap_or_else(|e| panic!("downloaded fixture {final_path:?} unreadable: {e}"));
    assert!(
        got == FIXTURE_SHA256,
        "downloaded fixture {final_path:?} hash mismatch: got sha256 {got}, want {FIXTURE_SHA256}"
    );
    println!("fixture-ok: {got} {}", final_path.display());
    final_path
}

fn download_slot() -> PathBuf {
    let home = std::env::home_dir()
        .filter(|p| !p.as_os_str().is_empty())
        .expect("cannot resolve home directory — cannot resolve fixture cache");
    home.join(".cache/golem/test-models")
}

fn preseed_slot() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.test-cache")
}

/// Hash-check a candidate in the download slot. A vanished file (a sibling
/// nextest process may be deleting a corrupt copy right now) is a miss,
/// never a panic. On mismatch: delete, warn, fall through.
fn check_download_slot_hit(path: &Path) -> Option<String> {
    if !path.is_file() {
        return None;
    }
    let got = match sha256_file(path) {
        Ok(got) => got,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return None,
        Err(e) => panic!("cache slot fixture {path:?}: {e}"),
    };
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

/// Download to a pid+thread-unique temp file in the same dir (atomic rename),
/// sha256-verify before renaming into place.
fn download() {
    let slot = download_slot();
    fs::create_dir_all(&slot).unwrap_or_else(|e| panic!("cannot create cache dir {slot:?}: {e}"));
    let tmp = slot.join(format!(
        ".{}.{}.{}.tmp",
        fixture_name(),
        std::process::id(),
        format_args!("{:?}", std::thread::current().id())
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
    if let Err(e) = fs::rename(&tmp, slot.join(fixture_name())) {
        clean_fail(&tmp, format!("atomic rename into {slot:?} failed: {e}"));
    }
}

fn clean_fail(tmp: &Path, msg: String) -> ! {
    let _ = fs::remove_file(tmp);
    let name = fixture_name();
    panic!(
        "{msg}\
         \nremediation: set GOLEM_TEST_MODEL=/path/to/{name} \
         (sha256 {FIXTURE_SHA256}), or pre-seed {} or {} with that file",
        download_slot().join(&name).display(),
        preseed_slot().join(&name).display(),
    );
}
