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
