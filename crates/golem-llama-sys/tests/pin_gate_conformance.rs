//! Byte-compares `pin_gate` against the live shell `cs()` recipe, extracted
//! from `scripts/build-substrate.sh` at test time (ADR 0001).

#![cfg(unix)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
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

/// Extract the streaming part of the LIVE `cs()` recipe from
/// `scripts/build-substrate.sh`: the pipeline between the leading
/// `cd "$1" && ` and the trailing `| sha256sum | awk '{print $1}'`
/// outer-hash fold (the fold is applied by the shell_* helpers here).
/// Panics loudly if the recipe is missing or reshaped — an embedded copy
/// would reintroduce the silent-drift this test exists to prevent.
fn cs_stream_recipe() -> &'static str {
    static RECIPE: OnceLock<String> = OnceLock::new();
    RECIPE
        .get_or_init(|| {
            let script = Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../../scripts/build-substrate.sh")
                .canonicalize()
                .expect(
                    "scripts/build-substrate.sh not found — conformance anchors to the live script",
                );
            let text = fs::read_to_string(&script).expect("read build-substrate.sh");
            let line = text
                .lines()
                .find(|l| l.trim_start().starts_with("cs() {"))
                .unwrap_or_else(|| {
                    panic!(
                        "cs() definition not found in {} — restore it or re-anchor this test",
                        script.display()
                    )
                });
            let body = line.trim_start().strip_prefix("cs() {").unwrap().trim_end();
            let body = body
                .strip_suffix('}')
                .expect("cs() must be a single-line `cs() { ...; }` definition");
            let body = body.trim_end().strip_suffix(';').unwrap_or(body);
            let body = body
                .trim()
                .strip_prefix("cd \"$1\" && ")
                .expect("cs() must begin with `cd \"$1\" && `");
            let stream = body
                .strip_suffix("| sha256sum | awk '{print $1}'")
                .expect("cs() must end with the `| sha256sum | awk '{print $1}'` outer-hash fold");
            stream.trim_end().to_string()
        })
        .as_str()
}

#[test]
fn recipe_body_is_verbatim() {
    let recipe = cs_stream_recipe();
    assert!(
        recipe.starts_with("find . -path ./.git -prune"),
        "live recipe lost its canonical shape: {recipe:?}"
    );
    assert!(
        recipe.ends_with("xargs -0 -r sha256sum"),
        "live recipe must not run sha256sum on an empty stream (-r): {recipe:?}"
    );
    assert!(!recipe.contains('\n'));
}

fn shell_stream(dir: &Path) -> Vec<u8> {
    let recipe = cs_stream_recipe();
    let out = Command::new("sh")
        .arg("-c")
        .arg(format!("cd \"$1\" && {recipe}"))
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
    let recipe = cs_stream_recipe();
    let out = Command::new("sh")
        .arg("-c")
        .arg(format!(
            "cd \"$1\" && {recipe} | sha256sum | awk '{{print $1}}'"
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
fn empty_tree_streams_and_digests_agree_with_shell() {
    let dir = unique_fixture_root("empty");
    let _guard = FixtureGuard(dir.clone());

    let rust_stream = golem_llama_sys::pin_gate::hash_stream(&dir).expect("rust stream");
    assert!(
        rust_stream.is_empty(),
        "rust gate emitted output for an empty tree"
    );
    assert_eq!(
        rust_stream,
        shell_stream(&dir),
        "empty tree: shell stream must be empty too (xargs -r — no `…  -` stdin-hash line)"
    );

    let rust_digest = golem_llama_sys::pin_gate::content_hash(&dir).expect("rust digest");
    assert_eq!(
        rust_digest,
        shell_outer_digest(&dir),
        "empty tree: outer digests diverge"
    );
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
