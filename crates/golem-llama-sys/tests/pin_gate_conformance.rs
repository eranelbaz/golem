//! Byte-compares `pin_gate` against the shell `cs()` recipe (ADR 0001).

#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

fn unique_fixture_root(tag: &str) -> PathBuf {
    let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
    let dir = std::env::temp_dir().join(format!(
        "golem-pin-gate-fixture-{}-{id}-{tag}",
        std::process::id()
    ));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).expect("create fixture root");
    dir
}

struct FixtureGuard(PathBuf);
impl Drop for FixtureGuard {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn make_fixture(root: &Path) {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;

    fs::create_dir_all(root.join("sub")).expect("sub");
    fs::create_dir_all(root.join("sub/build")).expect("nested build dir");
    fs::create_dir_all(root.join(".git/objects")).expect(".git dir (pruned)");
    fs::create_dir_all(root.join("build")).expect("build dir (pruned)");
    fs::create_dir_all(root.join("emptydir")).expect("empty dir (skipped)");
    fs::write(root.join("sub/.git"), b"nested git file: must be hashed").expect("sub/.git file");
    fs::write(
        root.join("sub/build/nested.o"),
        b"nested build file: must be hashed (prune is top-level-only)",
    )
    .expect("nested build file");
    fs::write(
        root.join(".git/HEAD"),
        b"top-level git dir content: must be pruned",
    )
    .expect(".git/HEAD");
    fs::write(
        root.join("build/artifact.o"),
        b"build artifact: must be pruned",
    )
    .expect("build artifact");
    fs::write(root.join("empty.txt"), b"").expect("empty file (hashed)");
    fs::write(root.join("data.bin"), b"fixture payload").expect("data.bin");
    fs::write(root.join("we\\ird.txt"), b"backslash name: escaping mode").expect("backslash file");
    fs::write(root.join("a\nb.txt"), b"newline name: escaping mode").expect("newline file");
    fs::write(
        root.join(OsStr::from_bytes(b"high\xFF.bin")),
        b"non-utf8 name",
    )
    .expect("non-utf8 file");
    std::os::unix::fs::symlink("data.bin", root.join("link.bin")).expect("symlink (skipped)");
}

/// Verbatim from `scripts/build-substrate.sh` (ADR 0001: paste, do not re-type).
const CS_BODY: &str = "find . -path ./.git -prune -o -path ./build -prune -o -type f -print0 \
                       | LC_ALL=C sort -z | xargs -0 sha256sum";

#[test]
fn recipe_body_is_verbatim() {
    assert!(CS_BODY.starts_with("find . -path ./.git -prune"));
    assert!(CS_BODY.ends_with("xargs -0 sha256sum"));
    assert!(!CS_BODY.contains('\n'));
}

fn shell_stream(dir: &Path) -> Vec<u8> {
    let out = Command::new("sh")
        .arg("-c")
        .arg(format!("cd \"$1\" && {CS_BODY}"))
        .arg("cs")
        .arg(dir)
        .output()
        .expect("spawn sh");
    assert!(
        out.status.success(),
        "shell cs() stream failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

fn shell_outer_digest(dir: &Path) -> String {
    let out = Command::new("sh")
        .arg("-c")
        .arg(format!(
            "cd \"$1\" && {CS_BODY} | sha256sum | awk '{{print $1}}'"
        ))
        .arg("cs")
        .arg(dir)
        .output()
        .expect("spawn sh");
    assert!(
        out.status.success(),
        "shell cs() outer hash failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout)
        .expect("utf8")
        .trim()
        .to_string()
}

#[test]
fn rust_gate_stream_is_byte_identical_to_shell() {
    let dir = unique_fixture_root("stream");
    let _guard = FixtureGuard(dir.clone());
    make_fixture(&dir);

    let rust_stream = golem_llama_sys::pin_gate::hash_stream(&dir).expect("rust stream");
    let shell = shell_stream(&dir);

    if rust_stream != shell {
        for (i, (r, s)) in rust_stream
            .split(|&b| b == b'\n')
            .zip(shell.split(|&b| b == b'\n'))
            .enumerate()
        {
            assert_eq!(
                r,
                s,
                "pin-gate streams diverge at line {}: rust {r:?} vs shell {s:?}",
                i + 1
            );
        }
        panic!(
            "pin-gate streams differ in length: rust {} bytes vs shell {} bytes",
            rust_stream.len(),
            shell.len()
        );
    }
}

#[test]
fn rust_gate_outer_digest_agrees_with_shell() {
    let dir = unique_fixture_root("digest");
    let _guard = FixtureGuard(dir.clone());
    make_fixture(&dir);

    let rust_digest = golem_llama_sys::pin_gate::content_hash(&dir).expect("rust digest");
    let shell_digest = shell_outer_digest(&dir);

    assert_eq!(
        rust_digest, shell_digest,
        "outer digests disagree (rust {rust_digest} vs shell {shell_digest})"
    );
}

#[test]
fn fixture_digest_matches_recorded_anchor() {
    let dir = unique_fixture_root("anchor");
    let _guard = FixtureGuard(dir.clone());
    make_fixture(&dir);

    let rust_digest = golem_llama_sys::pin_gate::content_hash(&dir).expect("digest");
    assert_eq!(
        rust_digest, ANCHOR_FIXTURE_DIGEST,
        "fixture digest moved — either the fixture layout changed (update the \
         anchor deliberately) or the recipe semantics drifted (investigate!)"
    );
}

const ANCHOR_FIXTURE_DIGEST: &str =
    "4da68c99a9896bb0cacc2de6ef6e5aa6e00f891e914aed5118c6745e8e8265d6";
