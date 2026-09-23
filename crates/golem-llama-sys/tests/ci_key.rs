//! Anchors the hand-transcribed `test-models-…` cache key in ci.yml to `FIXTURE_SHA256`.

#[allow(dead_code)]
mod common;

use common::{CANONICAL_CS_CORE, FIXTURE_SHA256, core_recipe, cs_recipe_line, repo_text};

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
    let workflow = repo_text(".github/workflows/ci.yml");
    let key = fixture_cache_key(&workflow);
    let prefix = &FIXTURE_SHA256[..12];
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
fn vendor_pin_workflow_recipe_matches_build_substrate() {
    let workflow = repo_text(".github/workflows/vendor-pin.yml");
    let script = repo_text("scripts/build-substrate.sh");

    let wf_recipe = core_recipe(cs_recipe_line(
        &workflow,
        ".github/workflows/vendor-pin.yml",
    ));
    let script_recipe = core_recipe(cs_recipe_line(&script, "scripts/build-substrate.sh"));
    assert_eq!(
        wf_recipe, script_recipe,
        "vendor-pin.yml's cs() recipe drifted from scripts/build-substrate.sh's canonical copy (ADR 0001: paste verbatim, do not re-type) — live recipe: {script_recipe:?}"
    );
    // Guards against both copies drifting jointly off the canonical shape (ADR 0001 §3)
    assert_eq!(
        wf_recipe, CANONICAL_CS_CORE,
        "both cs() copies changed shape — re-check ADR 0001 §3 and re-pin this literal to the new canonical recipe"
    );
}

#[test]
fn vendor_pin_recipe_extraction_is_exact() {
    let line = "          cs() { cd \"$1\" && find . -path ./.git -prune -o -path ./build -prune -o -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}'; }";
    assert_eq!(
        core_recipe(cs_recipe_line(line, "inline fixture")),
        CANONICAL_CS_CORE,
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
