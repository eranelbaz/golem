# ADR 0001: Fork pin — the GLM-5.3-Flash compute substrate

## Status

Accepted, 2026-09-19.

## Context

GLM-5.3-Flash (`glm5next`) support is not merged into ggml-org/llama.cpp; the spec itself flags the feature as unmerged and moving (tech-design §3.2, §14). At pin-selection time, three competing PRs carried the support, and they disagree on the GGUF architecture string their loaders accept. Decision D4 (tech-design) pins the v1 weights to `unsloth/GLM-5.3-Flash-GGUF` (UD-Q4_K_XL) — those GGUFs carry `general.architecture = glm5next` and are built and tested against a specific PR. The arch string a PR implements is therefore the deciding constraint, not code quality or completeness:

| PR | Author | Head branch | Arch string | State | GGUF compatibility vs D4 weights (`unsloth/GLM-5.3-Flash-GGUF`) |
|----|--------|-------------|-------------|-------|-----------------------------------------------------------------|
| [#27754](https://github.com/ggml-org/llama.cpp/pull/27754) | `danielhanchen` (head branch hosted on the **unslothai** fork) | `glm5next/upstream` | `glm5next` | OPEN | **Match** — same arch literal as the GGUFs (confirmed in the vendored tree, `src/llama-arch.cpp:88`); the Unsloth GGUFs are built and tested against this PR |
| [#27773](https://github.com/ggml-org/llama.cpp/pull/27773) | `timkhronos` | `GLM5.3-Flash` | `glm5-next` | OPEN | **Mismatch** — `glm5-next` ≠ `glm5next`; the loader would reject D4's GGUFs |
| [#27752](https://github.com/ggml-org/llama.cpp/pull/27752) | `eauchs` | `glm5next/add-glm-5.3-flash` | `glm5next` (text-only) | OPEN | Arch matches, but this is not the PR D4's GGUFs are built/tested against, and the text-only scope is unproven against the pinned weights |

*Snapshot as of 2026-09-19, head `86ebfef2…` — recorded data, not live claims. PR numbers, authors, head branches, and states were live-checked at the stamp date; #27754's head/base SHAs and the `glm5next` arch literal are execution-verified against the vendored tree; sibling arch strings and the text-only scope note are plan-time research not re-verified against the sibling trees.*

Only #27754 satisfies D4: it both implements the GGUFs' `glm5next` arch literal and is the PR those GGUFs are built and tested against.

The pin also inherits a liveness risk: PR #27754 is hosted on ggml-org/llama.cpp, but its head branch `glm5next/upstream` lives on the third-party **unslothai** fork — an account Golem does not control. Upstream force-pushes or fork/branch deletion would silently invalidate any bare SHA claim, so the verification and acceptance mechanisms below are designed around a recomputable fingerprint rather than trust in the remote.

## Decision

1. **Pin.** `engine/llama-cpp.lock` and the vendored tree pin **unslothai/llama.cpp, branch `glm5next/upstream`, commit `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`** (merge-base on ggml-org/llama.cpp: `base_upstream = 930e2fa5995789efbf249a8bf61325bb626e417b`), head of PR #27754. Verified live at pin time with:

   ```bash
   gh pr view 27754 --repo ggml-org/llama.cpp --json headRefOid,baseRefOid,headRepositoryOwner,state
   # → headRefOid = 86ebfef2…, baseRefOid = 930e2fa5…, headRepositoryOwner.login = "unslothai", state = "OPEN"
   ```

   Note the PR number resolves against **ggml-org/llama.cpp**, not the fork — a fork has its own PR numbering, and `--repo unslothai/llama.cpp` fails for #27754.

2. **Vendoring form.** The tree is vendored as plain files under `vendor/llama.cpp/` in a single squashed commit: no nested `.git`, no gitlink, no submodule. Upstream's `.gitmodules` at the pin is a tracked **0-byte empty blob** and is kept for pristineness; submodule-freedom was proven with `git submodule status` (empty output). A **no-symlinks invariant** holds (`find vendor/llama.cpp -type l` must be empty), which makes `file_count` — regular files with `./.git` pruned — and the `git ls-files` entry count equal by definition. At the pin: **`file_count = 3599`**.

3. **Pin integrity = recomputable content hash, not a SHA claim.** Because the pin lives on a mutable third-party fork, the accepted anchor is a tree fingerprint recomputable forever from the vendored files. Canonical recipe (paste verbatim; do not re-type):

   ```bash
   (cs() { cd "$1" && find . -path ./.git -prune -o -path ./build -prune -o -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'; }; cs DIR)
   ```

   **Amendment (M0.2 follow-ups):** the recipe now prunes `./build` in addition to `./.git` (M0.1 original pruned only `./.git`). Rationale: `scripts/build-substrate.sh` (M0.2) hashes the vendored tree at build time, and build artifacts under `vendor/llama.cpp/build/` would otherwise make every post-build re-fingerprint disagree with `vendor/VENDOR_SHA`. The prune is a no-op on a pristine tree — the pin's hash value is unchanged (`812b9b4a…`), verified before amending. CI (`vendor-pin-check`) runs on a fresh checkout with no build dir and keeps its `./.git`-only form deliberately; any consumer fingerprinting a tree *after* a build must use this build-pruned form.

   The `cd`-inside + `find .` form and `LC_ALL=C` are load-bearing: `sha256sum` output embeds file paths, so a repo-root invocation hashes a different path prefix and can never match across checkouts, and `sort` order is locale-dependent and would diverge between machines. The recipe must reproduce itself — the stability canary, run twice before its output is trusted anywhere.

   At the pin: **`content_sha256 = 812b9b4a7848e34a76c8d444e244fa787f6adb0a586e85a7317477a53ea9f794`**. It is recorded in `vendor/VENDOR_SHA` (format contract: bare `key=value` lines; comments only on their own lines starting with `#`; **no inline comments** — the file is parsed) and in `engine/llama-cpp.lock`, and is re-derived by the **`vendor-pin-check`** CI job on every push (`.github/workflows/vendor-pin.yml`, unconditional push/pull_request trigger) using the identical command.

4. **`archive_sha256` is provenance only.** `archive_sha256 = dfe22d466b1ceb824651ada7c6f20347f4f18caa241dc3459fb5f10560d265e3` — the sha256 of the `git archive HEAD` output at the pin, **not** the sha256 of the parked recovery tarball (different artifact, different hash) — is kept as a tarball-era provenance note. It is unverifiable once the tarball is gone, and no gate depends on it — `content_sha256` is the only acceptance anchor.

5. **Recovery copies, until merge.** A pristine tarball is parked at `~/.cache/golem/vendor/golem-vendor-tarball.tar.gz` until this PR merges, created with `tar --exclude=.git -czf …` — **not** `--exclude-vcs`, which additionally drops tracked VCS *files* (here 17: 16 `.gitignore`s + `.gitmodules`) and breaks tree-identity fidelity checks. The pinning branch is pushed to origin immediately after vendoring; the pushed commit is itself a durable pristine copy, independent of the tarball.

6. **Fork-deletion end-state.** If the unslothai fork or the `glm5next/upstream` branch is ever deleted, the vendored tree + `vendor/VENDOR_SHA` are, by design, the only source of truth: re-vendoring from upstream becomes impossible. This is accepted.

## Re-vendor procedure (verbatim, future re-pins use this)

Canonical recipe (identical to Decision §3; paste verbatim):

```bash
(cs() { cd "$1" && find . -path ./.git -prune -o -path ./build -prune -o -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'; }; cs DIR)
```

All steps mandatory, in order:

1. **Re-verify the candidate PR head** (Task 1 Step 0 checks):

   ```bash
   gh pr view <PR> --repo ggml-org/llama.cpp --json headRefOid,baseRefOid,headRepositoryOwner,state
   ```

   Distinguish `gh` failure (unfetchable → investigate) from field mismatch (a re-pin decision). Clone the fork, check out the candidate commit, verify the merge-base against `base_upstream`, extract the GGUF arch literal from `src/llama-arch.*` case-sensitively (non-empty, single value), and confirm `git submodule status` is empty. If the old head was force-pushed away and cannot be fetched, do **not** silently re-pin — that requires a new ADR round.

2. **Re-fingerprint** the pristine tree with the canonical `cs()` recipe above — same commands, including the **stability canary** (run the recipe twice on the same tree; both outputs must be identical before anything is trusted). Record `content_sha256`, `file_count` (regular files, `./.git` pruned, explicit `-print`), and `commit`.

3. **Transfer:** `rsync -a --delete --exclude .git <pristine-tree>/ vendor/llama.cpp/`. The `--delete` is mandatory: re-vendoring without it retains deleted-upstream files → stale-header build chimeras.

4. **Re-assert the invariants — abort on violation:** symlinks present (`find vendor/llama.cpp -type l` non-empty → fingerprint definition mismatch, stop); `.gitmodules` non-empty or `git submodule status` non-empty (a 0-byte `.gitmodules` is expected upstream content); any file >50 MB; any LFS pointer.

5. **Stage and gate on the index:** `git add -f vendor/llama.cpp vendor/VENDOR_SHA` (`-f` neutralizes both ignore layers — the vendored upstream `.gitignore`s and golem's root `.gitignore` would otherwise silently strip vendored files). Index gates: `git ls-files vendor/llama.cpp | wc -l` must equal `file_count`, and zero `160000` (gitlink) entries may exist.

6. **Update `vendor/VENDOR_SHA` and `engine/llama-cpp.lock` in the same commit** as the new tree — a lock describing a tree the commit does not contain is a broken pin.

7. **CI must pass:** the `vendor-pin-check` job re-derives `content_sha256` with the identical canonical command and requires a match against `vendor/VENDOR_SHA` and the lock.

## Consequences

- **Lock schema supersession.** `engine/llama-cpp.lock`'s schema supersedes tech-design §3.2's template (repo, key set, `rocm` semantics — amended in this PR); this ADR is the declaration of that supersession.
- **Acceptance is content-hash-based.** Not `git -C vendor/llama.cpp rev-parse HEAD` — there is no nested `.git` by design. The only acceptance oracle is the canonical recipe's `content_sha256`, recomputed from the vendored files.
- **The D2 re-open trigger of design §3.2 still applies.** If rebasing Golem's patches onto the moving fork turns into a maintenance sink (>2 consecutive rebase failures) or the fork direction fails, design §3.2's documented trigger to re-open decision D2 governs. This pin is a substrate freeze, not a long-term endorsement of the fork.
- **Repo-size cost of squashed re-vendoring is accepted deliberately.** Each re-pin grows a full-tree set of blobs; git history grows linearly in re-pin count × tree size. If the fork survives more than a couple of re-pins, revisit the vendoring mechanism (subtree, submodule, or vendor branch) with numbers.
