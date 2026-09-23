use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use sha2::{Digest, Sha256};

const FIXTURE_URLS_BY_PRIORITY: [&str; 2] = [
    "https://huggingface.co/ggml-org/models/resolve/main/tinyllamas/stories15M-q4_0.gguf",
    "https://huggingface.co/ggml-org/models-moved/resolve/main/tinyllamas/stories15M-q4_0.gguf",
];
pub const FIXTURE_SHA256: &str = "66967fbece6dbe97886593fdbb73589584927e29119ec31f08090732d1861739";

fn fixture_name() -> String {
    format!("stories15M-q4_0-{}.gguf", &FIXTURE_SHA256[..12])
}

static FIXTURE: OnceLock<PathBuf> = OnceLock::new();

pub fn fixture() -> PathBuf {
    if let Some(path) = FIXTURE.get() {
        if let Some(sha) = verify_memoized_hash(path) {
            println!("fixture-ok: {sha} {}", path.display());
            return path.clone();
        }
    }
    let path = resolve();
    let _ = FIXTURE.set(path.clone());
    path
}

fn verify_memoized_hash(path: &Path) -> Option<String> {
    match sha256_file(path) {
        Ok(got) if got == FIXTURE_SHA256 => Some(got),
        Ok(got) => {
            eprintln!(
                "warning: memoized fixture {path:?} changed under us: got sha256 {got}, want {FIXTURE_SHA256} — re-resolving"
            );
            None
        }
        Err(e) => {
            eprintln!("warning: memoized fixture {path:?} unreadable: {e} — re-resolving");
            None
        }
    }
}

fn resolve() -> PathBuf {
    if let Some(path) = golem_test_model_override() {
        return path;
    }
    if let Some(path) = cached_download() {
        return path;
    }
    if let Some(path) = preseeded_copy() {
        return path;
    }
    download_and_verify()
}

fn golem_test_model_override() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("GOLEM_TEST_MODEL") {
        let p = PathBuf::from(p);
        assert!(p.is_file(), "GOLEM_TEST_MODEL={p:?} is not a regular file");
        let got = sha256_file(&p).unwrap_or_else(|e| panic!("GOLEM_TEST_MODEL={p:?}: {e}"));
        assert!(
            got == FIXTURE_SHA256,
            "GOLEM_TEST_MODEL={p:?} hash mismatch: got sha256 {got}, want {FIXTURE_SHA256}"
        );
        println!("fixture-ok: {got} {}", p.display());
        Some(p)
    } else {
        None
    }
}

fn cached_download() -> Option<PathBuf> {
    let cached = download_slot().join(fixture_name());
    if let Some(got) = verified_existing(&cached) {
        println!("fixture-ok: {got} {}", cached.display());
        Some(cached)
    } else {
        None
    }
}

fn preseeded_copy() -> Option<PathBuf> {
    let preseed = preseed_slot().join(fixture_name());
    if !preseed.is_file() {
        return None;
    }
    match sha256_file(&preseed) {
        Ok(got) if got == FIXTURE_SHA256 => {
            println!("fixture-ok: {got} {}", preseed.display());
            Some(preseed)
        }
        Ok(got) => {
            eprintln!(
                "warning: corrupt fixture {preseed:?}: got sha256 {got}, want {FIXTURE_SHA256} — ignoring pre-seed"
            );
            None
        }
        Err(e) => {
            eprintln!("warning: unreadable fixture {preseed:?}: {e} — ignoring pre-seed");
            None
        }
    }
}

fn download_and_verify() -> PathBuf {
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

fn verified_existing(path: &Path) -> Option<String> {
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
    replace_corrupt_if_still_corrupt(path, &got);
    None
}

fn replace_corrupt_if_still_corrupt(path: &Path, got: &str) {
    match sha256_file(path) {
        Ok(now) if now != got => {
            eprintln!(
                "warning: corrupt fixture {path:?} changed under us ({got} → {now}) — another writer got there first, keeping it"
            );
            return;
        }
        Err(e) => {
            eprintln!(
                "warning: corrupt fixture {path:?} unreadable on re-check: {e} — falling through to download"
            );
            return;
        }
        Ok(_) => {}
    }
    let tmp = path.with_file_name(format!(
        ".{}.{}.{}.corrupt.tmp",
        fixture_name(),
        std::process::id(),
        format_args!("{:?}", std::thread::current().id())
    ));
    let replaced = fs::File::create(&tmp).and_then(|f| {
        drop(f);
        fs::rename(&tmp, path)
    });
    match replaced {
        Ok(()) => eprintln!(
            "warning: replaced corrupt fixture {path:?}: got sha256 {got}, want {FIXTURE_SHA256} — falling through to download"
        ),
        Err(e) => eprintln!(
            "warning: could not replace corrupt {path:?}: {e} — falling through to download"
        ),
    }
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

fn download() {
    let slot = download_slot();
    fs::create_dir_all(&slot).unwrap_or_else(|e| panic!("cannot create cache dir {slot:?}: {e}"));
    let tmp = slot.join(format!(
        ".{}.{}.{}.tmp",
        fixture_name(),
        std::process::id(),
        format_args!("{:?}", std::thread::current().id())
    ));
    for url in FIXTURE_URLS_BY_PRIORITY {
        eprintln!("downloading pinned fixture from {url} to {tmp:?}");
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
            .arg(url)
            .status()
            .unwrap_or_else(|e| panic!("failed to spawn curl for {url}: {e}"));
        if !status.success() {
            let _ = fs::remove_file(&tmp);
            eprintln!("warning: curl exited {status} for {url} — trying next mirror");
            continue;
        }
        let got = match sha256_file(&tmp) {
            Ok(got) => got,
            Err(e) => {
                let _ = fs::remove_file(&tmp);
                eprintln!(
                    "warning: downloaded temp file unreadable from {url}: {e} — trying next mirror"
                );
                continue;
            }
        };
        if got != FIXTURE_SHA256 {
            let _ = fs::remove_file(&tmp);
            eprintln!(
                "warning: downloaded {url} hash mismatch: got sha256 {got}, want {FIXTURE_SHA256} — trying next mirror"
            );
            continue;
        }
        if let Err(e) = fs::rename(&tmp, slot.join(fixture_name())) {
            clean_fail(&tmp, format!("atomic rename into {slot:?} failed: {e}"));
        }
        return;
    }
    clean_fail(&tmp, format!("all fixture URLs failed: {FIXTURE_URLS_BY_PRIORITY:?}"));
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

pub fn repo_text(rel: &str) -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join(rel)
        .canonicalize()
        .unwrap_or_else(|e| panic!("{rel} not found — conformance anchors to the live repo: {e}"));
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

pub const CANONICAL_CS_CORE: &str = "find . -path ./.git -prune -o -path ./build -prune -o -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum";

pub fn cs_recipe_line<'a>(text: &'a str, source: &str) -> &'a str {
    text.lines()
        .find(|l| l.contains("cs() {"))
        .unwrap_or_else(|| {
            panic!(
                "no `cs() {{` recipe line in {source} — the ADR 0001 content-hash recipe moved; restore it or re-anchor this test"
            )
        })
}

pub fn core_recipe(cs_line: &str) -> &str {
    const END: &str = "xargs -0 -r sha256sum";
    let start = cs_line
        .find("find .")
        .unwrap_or_else(|| panic!("cs() line has no `find .` pipeline start: {cs_line:?}"));
    let tail = &cs_line[start..];
    let end = tail
        .find(END)
        .unwrap_or_else(|| panic!("cs() line has no `{END}` pipeline end: {cs_line:?}"));
    &tail[..end + END.len()]
}

pub fn cs_stream_recipe(text: &str, source: &str) -> String {
    let line = cs_recipe_line(text, source);
    let body = line.trim_start().strip_prefix("cs() {").unwrap().trim_end();
    let body = body
        .strip_suffix('}')
        .expect("cs() must be a single-line `cs() {{ ...; }}` definition");
    let body = body.trim_end().strip_suffix(';').unwrap_or(body);
    let body = body
        .trim()
        .strip_prefix("cd \"$1\" && ")
        .expect("cs() must begin with `cd \"$1\" && `");
    let stream = body
        .strip_suffix("| sha256sum | awk '{print $1}'")
        .expect("cs() must end with the `| sha256sum | awk '{{print $1}}'` outer-hash fold");
    stream.trim_end().to_string()
}
