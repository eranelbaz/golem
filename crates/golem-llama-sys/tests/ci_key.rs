//! CI cache-key conformance: the `test-models-…` cache key literal in
//! `.github/workflows/ci.yml` is hand-transcribed — this test anchors it to
//! `FIXTURE_SHA256` so the workflow and the fixture hash cannot silently
//! drift apart.

#[allow(dead_code)] // only the consts are consumed here
mod common;

use std::fs;
use std::path::Path;

fn ci_workflow_text() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../.github/workflows/ci.yml")
        .canonicalize()
        .expect(".github/workflows/ci.yml not found — CI conformance anchors to the live workflow");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

fn vendor_pin_workflow_text() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../.github/workflows/vendor-pin.yml")
        .canonicalize()
        .expect(".github/workflows/vendor-pin.yml not found — pin-integrity conformance anchors to the live workflow");
    fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()))
}

/// Extract the fixture cache key literal from the workflow text.
/// Panics loudly if the `key: test-models-…` line is missing or reshaped.
fn fixture_cache_key(workflow: &str) -> &str {
    workflow
        .lines()
        .find_map(|l| l.trim_start().strip_prefix("key: test-models-"))
        .unwrap_or_else(|| {
            panic!(
                "no `key: test-models-…` literal in ci.yml — the fixture cache anchor moved; restore it or re-anchor this test"
            )
        })
}

#[test]
fn ci_cache_key_is_pinned_to_fixture_sha() {
    let workflow = ci_workflow_text();
    let key = fixture_cache_key(&workflow);
    let prefix = &common::FIXTURE_SHA256[..12];
    assert!(
        key.starts_with(prefix),
        "ci.yml cache key `test-models-{key}` must start with FIXTURE_SHA256[..12] = {prefix} — re-pin the literal to the fixture hash"
    );
    assert!(
        key.chars().all(|c| c.is_ascii_hexdigit()),
        "ci.yml cache key suffix {key:?} must be pure hex"
    );
}

/// The canonical cs() content-hash recipe (ADR 0001) is hand-copied into two
/// places that must never drift apart: `scripts/build-substrate.sh` (build-time
/// gate) and `.github/workflows/vendor-pin.yml` (CI gate). This test extracts
/// the recipe from the workflow and binds it to the live build-script copy.
fn cs_recipe_line<'a>(text: &'a str, source: &str) -> &'a str {
    text.lines()
        .find(|l| l.contains("cs() {"))
        .unwrap_or_else(|| {
            panic!(
                "no `cs() {{` recipe line in {source} — the ADR 0001 content-hash recipe moved; restore it or re-anchor this test"
            )
        })
}

/// The core recipe: the find → sort → xargs sha256sum pipeline inside a cs()
/// line (everything from `find .` through `xargs -0 -r sha256sum`).
fn core_recipe(cs_line: &str) -> &str {
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

#[test]
fn vendor_pin_workflow_recipe_matches_build_substrate() {
    let workflow = vendor_pin_workflow_text();
    let script = fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../scripts/build-substrate.sh")
            .canonicalize()
            .expect("scripts/build-substrate.sh not found — recipe conformance anchors to the live build script"),
    )
    .expect("cannot read scripts/build-substrate.sh");

    let wf_recipe = core_recipe(cs_recipe_line(
        &workflow,
        ".github/workflows/vendor-pin.yml",
    ));
    let script_recipe = core_recipe(cs_recipe_line(&script, "scripts/build-substrate.sh"));
    assert_eq!(
        wf_recipe, script_recipe,
        "vendor-pin.yml's cs() recipe drifted from scripts/build-substrate.sh's canonical copy (ADR 0001: paste verbatim, do not re-type) — live recipe: {script_recipe:?}"
    );
    // Guard against both copies drifting together off the canonical form: the
    // build-pruned, NUL-delimited, C-locale shape is load-bearing (ADR 0001 §3).
    assert_eq!(
        wf_recipe,
        "find . -path ./.git -prune -o -path ./build -prune -o -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum",
        "both cs() copies changed shape — re-check ADR 0001 §3 and re-pin this literal to the new canonical recipe"
    );
}

#[test]
fn vendor_pin_recipe_extraction_is_exact() {
    let line = "          cs() { cd \"$1\" && find . -path ./.git -prune -o -path ./build -prune -o -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}'; }";
    assert_eq!(
        core_recipe(cs_recipe_line(line, "inline fixture")),
        "find . -path ./.git -prune -o -path ./build -prune -o -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum",
        "must extract the find/xargs core, not the cd prefix or sha256sum|awk digest tail"
    );
}

#[test]
fn fixture_cache_key_extraction_is_exact() {
    let workflow = "\
jobs:
  test:
    steps:
      - uses: actions/cache@v4
        with:
          path: ~/.cache/golem/test-models
          key: test-models-66967fbece6dbe97
          restore-keys: |
            test-models-
";
    assert_eq!(
        fixture_cache_key(workflow),
        "66967fbece6dbe97",
        "must extract the literal after `key: test-models-`, not restore-keys or neighbors"
    );
}
