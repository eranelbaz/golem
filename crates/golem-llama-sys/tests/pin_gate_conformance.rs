//! Regression guard on the trust anchor: byte-compares the Rust pin-gate
//! implementation ([`golem_llama_sys::pin_gate`], the same source `build.rs`
//! gates every build with) against the exact shell `cs()` recipe from
//! `scripts/build-substrate.sh` / ADR 0001, run on a fixture directory.
//!
//! The fixture exercises the tricky edges of the recipe:
//! - a nested `sub/.git` FILE — must be hashed (top-level prune only);
//! - a top-level `.git` DIR — pruned;
//! - a top-level `build` DIR — pruned, while a nested `sub/build/` file is
//!   hashed (prune is top-level-only for BOTH names);
//! - a backslash-named file — must trigger GNU sha256sum's escaping mode
//!   (`\` line prefix, backslash doubled);
//! - a newline-named file — the other escaping edge (`\n` rendered as
//!   backslash + `n`), the single trickiest coreutils path;
//! - a non-UTF-8-named file — the byte-exact `os_bytes` path and `LC_ALL=C`
//!   sort ordering of high-byte names;
//! - a symlink — never followed nor hashed;
//! - an empty file — hashed (e3b0c442…);
//! - an empty dir — skipped.
//!
//! Both the full streams and the outer digests are compared, so any future
//! divergence localizes to the first differing line. CI runs this per-commit
//! via nextest. `#[cfg(unix)]`: the shell recipe is a Unix contract.

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
    let _ = fs::remove_dir_all(&dir); // stale leftovers from crashed runs
    fs::create_dir_all(&dir).expect("create fixture root");
    dir
}

/// Removes the fixture tree on drop — assertion failures no longer leak
/// temp dirs.
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
    // Invalid-UTF-8 filename via raw bytes (0xFF is never valid UTF-8).
    fs::write(
        root.join(OsStr::from_bytes(b"high\xFF.bin")),
        b"non-utf8 name",
    )
    .expect("non-utf8 file");
    std::os::unix::fs::symlink("data.bin", root.join("link.bin")).expect("symlink (skipped)");
}

/// The canonical recipe body, verbatim from `scripts/build-substrate.sh`
/// (ADR 0001: paste, do not re-type).
const CS_BODY: &str = "find . -path ./.git -prune -o -path ./build -prune -o -type f -print0 \
                       | LC_ALL=C sort -z | xargs -0 sha256sum";

#[test]
fn recipe_body_is_verbatim() {
    // The `\`-continuation above makes the body fragile to reformat; assert
    // its head so an accidental indent leak can't go unnoticed.
    assert!(CS_BODY.starts_with("find . -path ./.git -prune"));
    assert!(CS_BODY.ends_with("xargs -0 sha256sum"));
    assert!(!CS_BODY.contains('\n'));
}

fn shell_stream(dir: &Path) -> Vec<u8> {
    let out = Command::new("sh")
        .arg("-c")
        .arg(format!("cd \"$1\" && {CS_BODY}"))
        .arg("cs") // $0
        .arg(dir) // $1
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
        .arg("cs") // $0
        .arg(dir) // $1
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

    // First-divergence localization: report the line number where the streams
    // part, if they part at all.
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

/// Fixture layout anchor: the digest the shell recipe produced on this exact
/// fixture (recorded at fixture-construction time — see the task-2 report).
/// Guards against lockstep drift where BOTH implementations change together
/// and the pairwise comparisons above would keep passing while the recipe's
/// meaning silently moved.
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

/// sha256 over the fixture's emulated sha256sum stream, computed by the shell
/// recipe on the exact `make_fixture` layout above.
const ANCHOR_FIXTURE_DIGEST: &str =
    "4da68c99a9896bb0cacc2de6ef6e5aa6e00f891e914aed5118c6745e8e8265d6";
