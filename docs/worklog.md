# Golem — Work Log & Execution Protocol

This file is the **single source of truth for execution state**. The design is
[`tech-design.md`](tech-design.md) (stable); this file is churny. Every session
must keep it accurate — a future session with zero other context must be able
to resume from this file alone.

---

## 0. Continuation protocol (read this first in a new session)

1. **Read, in order:** `product-spec.md` → `docs/tech-design.md` (§1 decisions, §2, and the sections the task references) → this file top to bottom.
2. **Verify state:**
   - `git -C ~/Projects/golem status` (clean? on `main`?)
   - `git -C ~/Projects/golem pull --ff-only` (this file may have moved)
   - `gh pr list --repo eranelbaz/golem --state open` (are prior task PRs merged? If a PR is merged: check its task done here; if a PR has review comments, address them **first** on its branch)
3. **Pick the first task** on the board below with status `todo` whose dependencies are all `done` (or `n/a`). Do **not** skip ahead.
4. **Branch:** `git checkout -b task/<task-id>-<slug>` from up-to-date `main` (e.g. `task/b1-workspace-skeleton`). A plain branch in the main checkout is fine here — the writing-plans skill's isolated-worktree advice is optional, not required.
5. **Write the implementation plan first — no code before it exists:** load the writing-plans skill at `~/.pi/agent/git/github.com/eranelbaz/superpowers/skills/writing-plans/SKILL.md` and author `docs/superpowers/plans/YYYY-MM-DD-<task-id>-<slug>.md` in its exact format — skill header (agentic-workers banner, Goal / Architecture / Tech Stack / Spec lines), Global Constraints, bite-sized tasks with exact files, exact commands, failing-test-first steps, and per-task commit points; no placeholders. The design section(s) the board task cites are the plan's spec. Where the plan must refine the board's wording (e.g. a concrete acceptance command), fix the board line in the same PR. Commit the plan.
6. **Plan review gate:** dispatch ONE zai reviewer subagent (quick mode) using `~/.pi/agent/git/github.com/eranelbaz/superpowers/skills/writing-plans/plan-document-reviewer-prompt.md`, passing the plan path and the spec path (the board task's cited design sections). If Status = **Issues Found** → fix and re-dispatch the whole plan, same mode, max 3 rounds; still unresolved at round 3 → stop and list the open items for the human. Only an **Approved** plan unlocks step 7. Summarize the review verdict in the PR body.
7. **Implement exactly per the approved plan**, in plan order, ticking its checkboxes as steps complete (`- [ ]` → `- [x]`), following design §13 (clippy `-D warnings`, fmt, nextest; `unsafe` rules). If reality forces a deviation, update the plan doc in the same PR so it stays truthful — the plan, not chat memory, is the record.
8. **Verify acceptance** — run the exact commands the task/plan lists; paste the relevant output tail into the PR body.
9. **Commit** with Conventional Commits (`feat:`, `test:`, `fix:`, `docs:`, `build:`). The plan doc, its checkbox updates, and the implementation travel in the same PR. Multiple commits fine; keep them reviewable.
10. **Push and open the PR:** compose the body (task id, plan link, design sections implemented, acceptance evidence, plan-review verdict, deviation notes) and run `gh pr create --repo eranelbaz/golem --title "task(<id>): <slug>" --body-file <body-file>` — do **not** use `--fill`, it cannot produce the mandated fields.
11. **Update this file with the real PR number** (step 10 created it): set the task's status to `PR open (#N)`, append an entry under **Log** (newest first) with: date, task id, branch, PR number, **plan path**, what landed, any deviations from the design (deviations must also be flagged in the PR body). Mark `PR open (#N)` **only** — `done` is set exclusively after step 2's `gh pr list` confirms the merge in a later session. Commit and push the worklog update.
12. **Report to the user:** branch, PR link, plan path, what to review, what the next session should pick up.

**Rules**

- One task = one branch = one PR. No drive-by refactors; fix-through-PR if something is broken.
- **Plan-first is absolute:** no implementation commits for a board task before its plan doc exists and its review returned **Approved** (steps 5–6). Plans live in `docs/superpowers/plans/`, travel with the task's PR, and are historical record after merge — corrections go in follow-up PRs, never silent edits.
- Docs/process changes that are not board tasks use a `docs/<slug>` branch with the same PR discipline (plan not required).
- Never commit `vendor/llama.cpp` history — vendor as a **squashed single commit** (record the upstream SHA in the commit message and in `engine/llama-cpp.lock`).
- Model weights never enter git (`.gitignore` covers `models/`, `*.gguf`).
- If a task turns out larger than one session: split it on the board (edit this file in the PR), finish the first coherent half, note the split in Log.
- If reality contradicts the design: stop, write the finding in Log under `Deviations`, propose the design amendment in the PR — do not silently diverge.
- Machine facts live in **§2 Environment** below; re-verify before hardware tasks, update if changed.

**Status vocabulary:** `todo` → `in-progress` (branch exists) → `PR open (#N)` → `done (PR #N)` / `blocked: <reason>`.

---

## 1. Status snapshot

- **Phase:** M0 — feasibility spike (gates everything; design §14)
- **Next task:** **M0.3** (model-manifest) — after M0.2's PR merges
- **Open PRs:** M0.2 hip-build (session 5)
- **Last session:** session 5 — M0.2 executed, PR open

---

## 2. Environment (verified 2026-09-19; re-verify before hardware tasks)

| Fact | Value |
|------|-------|
| Machine | Arch Linux, author's own box — the v1 target machine (spec §Hardware) |
| GPU | AMD RX 7900 XTX (Navi 31, PCI `1002:744c`), `amdgpu` bound — **gfx1100** |
| ROCm | **7.2.4** at `/opt/rocm` (`hipcc`, `rocminfo` present; exact version recorded 2026-09-19 — pin goes into `llama-cpp.lock` at M0.2) |
| RAM | 93 GiB total, ~75 GiB available |
| Drives | `nvme0n1` Samsung 990 PRO 4TB (~1.9 TB free, hosts `/home`) · `nvme1n1` PNY CS2130 1TB — **two physical NVMe** for replica mode (design §5) |
| Toolchain | rustc/cargo 1.98.1 (edition 2024), `gh` authed as `eranelbaz` (repo+workflow scopes) |
| Remote | `github.com/eranelbaz/golem` (private) |
| Mount check pending | is `nvme1n1` mounted and writable for a model replica? Verify at M3.3 / `lsblk -f` |

---

## 3. Task board

Legend: ☐ `todo` · ◐ `in-progress` · ◑ `PR open` · ☑ `done`. IDs are stable — never renumber, only append.

Every task executes **plan-first** (protocol steps 5–6): a per-task implementation plan in writing-plans format, stored under `docs/superpowers/plans/`, must pass plan review before any implementation commit; the plan lands in the same PR as its implementation.

### Phase B — Bootstrap (no GPU needed)

- ☑ **B1 · workspace-skeleton** — Cargo workspace per design §13: crates `golem-cli`, `golem-server`, `golem-engine`, `golem-llama-sys`, `golem-placement`, `golem-store`, `golem-plan`, `golem-api`, `golem-registry`, plus `xtask`; `rust-toolchain.toml`; clippy/fmt/deny config; error envelope in `golem-api` (design §8.4) with table-driven wire-format tests; CI enforces fmt + clippy `-D warnings` + nextest + cargo-deny. Accept: `cargo fmt --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo test --workspace` all green locally; same gates (nextest) enforced in CI. *(done — PR #1)*

### Phase M0 — Feasibility spike on the 7900 XTX (gates everything; exit criteria design §14)

- ☑ **M0.1 · fork-pin** — Research the state of GLM-5.3-Flash (`glm5next`) text support upstream: pin the head of PR #27754 (PR hosted on `ggml-org/llama.cpp`; head branch `glm5next/upstream` from the unslothai fork); select the commit. Vendor it **squashed** into `vendor/llama.cpp/` (no patches yet, no nested `.git`). Write `engine/llama-cpp.lock` (repo, branch, commit, pr, base_upstream, content hashes, file_count, ROCm line, `patches = none-yet`) and `docs/adr/0001-fork-pin.md` (why this commit, what it contains). Accept: CI's `vendor-pin-check` recomputes `content_sha256` from `vendor/llama.cpp` (canonical recipe, ADR) and matches `vendor/VENDOR_SHA` and the lock; design §3.2. Deps: B1. *(done — PR #3, merge ac09f05)*
- ☐ **M0.2 · hip-build** — *(PR open — session 5)* Scripted substrate build: `scripts/build-substrate.sh` (cmake `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100`; warns on `rocminfo` vs lock `rocm` mismatch and skips the check when `rocminfo` is absent — M0.1 pre-fills `rocm = 7.2.4`). Smoke: `llama-cli` with a small GGUF on the 7900 XTX. Accept: build + smoke log committed under `docs/baselines/m0-notes.md`; ROCm version pinned. Deps: M0.1.
- ☐ **M0.3 · model-manifest** — Download `unsloth/GLM-5.3-Flash-GGUF` **UD-Q4_K_XL** (exact revision pinned) to a chosen drive location + replica location on the second NVMe; compute per-shard SHA-256; hand-write the `manifest.json` per design §4.2 (the `golem pull` tool itself comes in M3.2); record exact byte sizes, layer/expert counts, bytes-per-expert in `docs/baselines/m0-notes.md`. Also fetch **Q8_0** of the same revision for parity work. Accept: manifests committed, hashes verified, sizes recorded. Deps: M0.2.
- ☐ **M0.4 · m0-measurements** — The spike proper: run the pinned build with host-buffered experts (`-ncmoe`-style) on the real model; measure per design §14/M0: cold-expert copy cost per step, KV bytes/token, ROCm VRAM overhead, tok/s cold vs warm, FA/precision settings that make MLA correct on ROCm (design §3.5 open risk). Freeze validated settings + numbers into `engine/llama-cpp.lock`, `docs/baselines/baseline-001.json` (design §12.3 fields). Accept: all five measurement classes recorded; settings frozen. Deps: M0.3.
- ☐ **M0.5 · parity-floor** — 8-prompt greedy parity: HIP run vs CPU reference run of the *same pinned build*; compute match rate; set the T2 threshold at ≥ (reference-noise-floor + margin) per design §12.2. Accept: numbers + threshold recorded in `docs/baselines/baseline-001.json`; **M0 exit review** written (continue / re-scope verdict per design §14). Deps: M0.4.

### Phase M1 — Substrate integration (exit criteria design §14; tests T1, T2)

- ☐ **M1.1 · llama-sys** — `golem-llama-sys`: bindgen over `llama.h` (+ `golem-hooks.h` header stub), build the vendored fork via the `cmake` crate (CPU backend for CI, HIP on hardware), `unsafe` smoke test: load a tiny GGUF, decode one token. Accept: `cargo test -p golem-llama-sys` green in CI (CPU) and locally. Deps: B1, M0.2 (for hardware path).
- ☐ **M1.2 · tiny-moe** — `xtask gen-tiny-moe`: writes the synthetic 2-layer / 8-expert GGUF with upstream tensor naming (design §12.1). Accept: generated file loads in the pinned substrate; generator unit-tested. Deps: M1.1.
- ☐ **M1.3 · engine-decode** — `golem-engine`: load config, GGUF sanity checks (§4.3 subset), greedy single-slot decode loop over `llama_batch`, chat-template application via substrate. Minimal `golem run --greedy <model>` in `golem-cli`. Accept: chat completes on tiny-moe (CI) and on the real model (hardware); T2 harness scaffolding (`xtask parity`) exists. Deps: M1.1, M1.2.
- ☐ **M1.4 · parity-suite** — Formalize T1 (op parity) + T2 (end-to-end greedy parity vs reference build) as runnable harness with reports; run on hardware with real model. Accept: T1/T2 executed, results in `docs/baselines/`, thresholds frozen. Deps: M1.3, M0.5.

### Phase M2 — Placement engine (tests T3, T4, T6)

- ☐ **M2.1 · patch-p1** — Fork patch P1 (hook plumbing + expert metadata; upstream behavior unchanged when hooks unset) + rebase/T4 gate wired into CI. Accept: upstream tests green; unset-hooks runs byte-identical to pristine build on tiny-moe. Deps: M1.1.
- ☐ **M2.2 · placement-core** — `golem-placement` pure logic: decayed-frequency scores, tier bitsets, hysteresis+min-dwell transitions, floors, budgets (§6.1–6.4). Table-driven unit tests, no FFI. Accept: policy properties (floor always resident; no thrash under oscillation; budget never exceeded) hold under randomized access traces. Deps: B1.
- ☐ **M2.3 · patch-p3-pool** — P3 host-residency callback + mlock'd RAM pool + wiring to `golem-placement`; prefetch thread (§6.5) with demand-priority semaphore. Accept: T6 loom test over (access, page-in, evict) interleavings green; P3 slow-path counters visible. Deps: M2.1, M2.2.
- ☐ **M2.4 · patch-p2-device** — P2 persistent device expert-copy cache (validity bitset, policy-ordered eviction) + P4 counters. Accept: T3 cache-invariance test green on tiny-moe (4 cache states, design §12.2); skip-ratio metric shows hot experts not re-copied. Deps: M2.3.
- ☐ **M2.5 · journal-seeding** — Placement journal ring buffer (§11), score seeding on warm start, 24 h soak script for hardware. Accept: warm-start floor+top-N pre-promotion demonstrably reduces early P3 slow paths on a replayed trace. Deps: M2.4.

### Phase M3 — Storage tier (tests T5)

- ☐ **M3.1 · store-core** — `golem-store`: manifest load, replica registry, same-drive detection (§5.1), striped reads (§5.2), CRC32C page validation (§5.3), thread-pool `pread64` engine. Accept: T5 table green (1/2/4 replicas, injected EIO, injected rot) on synthetic fixtures. Deps: B1.
- ☐ **M3.2 · golem-pull** — `golem pull`: HF resolution, resumable verified download, `expert_index` builder, replica cloning (§5.5). Accept: end-to-end pull of the pinned model re-produces M0.3's manifest (diff-clean). Deps: M3.1.
- ☐ **M3.3 · store-wiring** — Wire store into P3 slow path; degraded/fatal-io state machine (§5.2); two-drive replica setup on this machine (`nvme1n1` mount verified, §2); yank-drive failover test procedure documented + executed. Accept: failover demonstrated live, `/health` shows `degraded:replica`, demand reads continue. Deps: M2.3, M3.1.

### Phase M4 — Product surface (tests T7, T8)

- ☐ **M4.1 · api-types** — `golem-api`: OpenAI types, error envelope + codes (§8.4), field-policy table (§8.2) as data + generated tests. Accept: T7 round-trip tests green. Deps: B1.
- ☐ **M4.2 · server-http** — `golem-server`: axum instance, `/v1/chat/completions` non-streaming, `/v1/models`, `/health`; slot scheduler v0 (single slot). Accept: OpenAI-python client completes a non-streaming chat against tiny-moe. Deps: M1.3, M4.1.
- ☐ **M4.3 · sse-reasoning** — SSE streaming (§8.3), `reasoning_content` splitting (§7.3), usage reporting, client-cancel → slot preemption. Accept: streaming golden tests; cancel observed in decode-loop logs. Deps: M4.2.
- ☐ **M4.4 · batching** — Dynamic micro-batching, chunked prefill, `--parallel` (§7.1–7.2). Accept: 4-slot concurrent chat correct; parity (T2 harness) still passes at batch>1 with documented deltas. Deps: M4.2.
- ☐ **M4.5 · registry-cli** — `golem-registry` + `golem list` / `golem stop` / `serve --detach` / `run` ensure-or-attach (§9). Accept: detach → list → stop round-trip green; stale-entry reaping works. Deps: M4.2.
- ☐ **M4.6 · golem-plan** — `golem-plan` arithmetic (§10) shared with runtime feasibility check (§6.4); `golem plan` CLI + `--json`. Accept: T8 — predicted vs actual VRAM within 256 MiB across a ctx sweep on hardware. Deps: M3.3, M4.2.
- ☐ **M4.7 · ops-hardening** — `/metrics` (§11), config precedence (§9.4), API key, graceful drain, logging defaults. Accept: metrics scraped by a plain `curl` parse test; precedence tested. Deps: M4.5.

### Phase M5 — Validation hardening (spec §Acceptance)

- ☐ **M5.1 · full-parity** — 50-prompt T2 suite + perplexity delta on wikitext-2 slice; thresholds final. Deps: M1.4, M2.4.
- ☐ **M5.2 · soak** — 24 h hardware soak (T6 runtime leg) + loom in CI. Deps: M2.5, M3.3.
- ☐ **M5.3 · acceptance-doc** — `golem bench` matrix runs (cold/warm × ctx × io-engine), baselines committed, README quickstart, spec §Acceptance checklist signed off in `docs/baselines/acceptance.md`. Deps: M4.7, M5.1, M5.2.

---

## 4. Log (append-only, newest first)

### 2026-09-20 — session 5: M0.2 hip-build (PR open)

> PR number to be filled in after `gh pr create` (this entry is committed before the PR exists — protocol ordering).

- Executed M0.2 plan-first on `task/m0.2-hip-build` (plan: `docs/superpowers/plans/2026-09-19-m0.2-hip-build.md`; plan review 3 rounds — rounds 1–2 caught real script bugs: pin-hash didn't prune the build dir (poisoned re-runs), dead smoke URLs, untestable skip-leg; round 3 **Approved** with empirical verification on this machine).
- Landed: `scripts/build-substrate.sh` (pin-integrity gate with `./build` pruned from the ADR canonical hash, advisory rocm check warn-or-skip-never-fail, HIP gfx1100 configure, llama-cli target); real HIP build on the 7900 XTX (83 s wall, first compile); GPU smoke run `stories15M-q4_0` at `-ngl 99` — exit 0, 7/7 layers on ROCm0, 1367 t/s generation. Evidence in `docs/baselines/m0-notes.md`.
- Deviations from plan: none substantive. Two execution notes (recorded in m0-notes): the fork's `llama-cli` chat UI needs `-st` for non-interactive runs (plan's `--no-warmup` invocation hung in interactive mode on first attempt; `-no-cnv` doesn't exist at this pin) and the UI suppresses backend logs (verbose evidence captured with `-v`).
- Next session: after merge, mark M0.2 `done` and start **M0.3** (model-manifest: download UD-Q4_K_XL + Q8_0, pin revision + SHA-256s, hand-write manifest).
### 2026-09-19 — session 4: M0.1 fork-pin (PR #3)

- PR #2 (planning protocol) merged. M0.1 executed plan-first on `task/m0.1-fork-pin` (plan: `docs/superpowers/plans/2026-09-19-m0.1-fork-pin.md`; per-task review gates clean). Status: **PR open (#3)**.
- **Research findings (as of 2026-09-19):** three competing PRs (drafts at research time; all OPEN as of execution) on GLM-5.3-Flash support disagree on the GGUF arch string — #27773 (`glm5-next`), #27754 (`glm5next`), #27752 (`glm5next`, text-only). Design D4 pins `unsloth/GLM-5.3-Flash-GGUF`, whose GGUFs carry arch `glm5next` → **#27754 selected** (head `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9`, branch `glm5next/upstream` from unslothai/llama.cpp; PR hosted on **ggml-org/llama.cpp**, base `930e2fa5…`, state OPEN live-verified). M0.3's gate will cross-check the actual GGUF `general.architecture`.
- **Vendored:** squashed single commit `5af7f1f` — plain files, no nested `.git`, no submodule; 3,599 files; 16 `.gitignore` files force-added past both ignore layers; pristine tarball parked at `~/.cache/golem/vendor/golem-vendor-tarball.tar.gz` until merge (delete after).
- **Pin recorded:** `engine/llama-cpp.lock` (`ca61722`; LOCK-MATCHES-VENDOR) + `vendor/VENDOR_SHA`; `content_sha256 = 812b9b4a7848e34a76c8d444e244fa787f6adb0a586e85a7317477a53ea9f794` (canonical `cs()` recipe, ADR 0001).
- **Design amendments in this PR (`a25661d`):** repo-wide arch string `glm5next`; lock schema §3.2 (superseded by lock+ADR); ROCm 7.2.4 advisory wording (M0.2 warns/skips — no hard assert); worklog M0.1/M0.2 task lines; §13 vendor-layout comment. Acceptance form changed git-rev-parse → content-hash (ADR `ecd37a9`).
- **Verification mechanism:** recomputable `content_sha256` (ONE canonical recipe used verbatim in fingerprint, verification, CI `vendor-pin-check` at `bdf1da5`, and ADR re-vendor procedure) + parked pristine tarball. All gates green: BASE-OK, ARCH=glm5next (src/llama-arch.cpp:88), NO-SUBMODULES, NO-NESTED-GIT-OK, CONTENT-HASH-OK, COUNT-OK, TREE-IDENTICAL, INDEX-OK (re-run fresh on the committed branch pre-PR).
- **Deviations:** brief's `--repo unslothai/llama.cpp` wrong (PR is on ggml-org — pin verified live, no re-pin); 0-byte tracked `.gitmodules` kept (upstream content; gate substance = `git submodule status` empty); `file_count=3599` (brief's `fc` form counts pruned `./.git` → 3600); tarball via `--exclude=.git` (`--exclude-vcs` drops 17 tracked files → broke TREE-IDENTICAL); Task-2 verify one-liner normalized (whitespace/order artifacts, values byte-identical); ADR records PR state OPEN (live-verified), not "draft".
- merge SHA: ac09f05

### 2026-09-19 — session 3: planning protocol (writing-plans wired in)

> Old protocol steps 5–10 were renumbered 5–12 in this session; session-2's "step 8" note below now refers to step 10.

- Merged PR #1 (CI green, 1m0s); B1 → `done (PR #1)`.
- Wired **superpowers:writing-plans** into the execution flow — protocol steps 3–12 now read: pick task → branch → **write plan** (`docs/superpowers/plans/YYYY-MM-DD-<id>-<slug>.md`, skill format, no placeholders) → **plan review gate** (one zai reviewer, `plan-document-reviewer-prompt.md`, ≤3 fix rounds, only Approved unlocks implementation) → implement per plan checkboxes → verify → PR carries plan + implementation together. Rules: plan-first is absolute; plans are historical record after merge; docs-only changes use `docs/<slug>` branches.
- Design §13/§14 updated: plans directory in repo layout; plan-first execution noted at milestones. ROCm version recorded (7.2.4).
- **Deliberately no task implementation this session.** Next chat: write + review the M0.1 plan, then implement on its own branch (PR #3).

### 2026-09-19 — session 2: B1 workspace skeleton (PR #1)

- Implemented B1 on `task/b1-workspace-skeleton`: 9 crates + `xtask` + `golem` bin, workspace lints (`unsafe_code` deny), toolchain pin `1.98.1`, `deny.toml`, CI (fmt / clippy `-D warnings` / nextest / cargo-deny).
- `golem-api` ships the full error envelope (design §8.4): 10 codes with exact `golem_*` wire strings, table-driven serde tests, `detail` field for structured payloads.
- Zai reviewer subagent quality pass (2 Critical / 5 Important / 5 Minor) — **all 12 applied**:
  - Design amendments: `golem_not_found` added to §8.4 code list (was referenced by §8.1 but missing — 9 → 10 codes); §15 citation §9.5 → §9.3.
  - Worklog protocol: step 8 now forbids marking `done` before merge confirmation (step 2 verifies).
  - CI now runs nextest + cargo-deny (was orphaned); smoke tests assert crate name, not version; README layout annotated with landing tasks.
- **Deviations:** nextest not installed locally — local acceptance runs `cargo test --workspace`; CI enforces `cargo nextest run`. Recorded here and in the PR body per protocol.
- Next session: merge PR #1 (if green), mark B1 `done (PR #1)`, then start **M0.1** (fork-pin).

### 2026-09-19 — session 1: bootstrap

- Wrote `docs/tech-design.md` v1.0 (all spec-deferred decisions resolved; grounded in context7 findings on llama.cpp HIP/MoE scheduler, candle/mistral.rs backend reality).
- Created repo `github.com/eranelbaz/golem` (private), `main` seeded with spec + design + this worklog + README.
- Verified environment: RX 7900 XTX present, ROCm installed, two-NVMe replica layout possible (see §2).
- Design self-review fixes applied: D12 patch count (3→4, P1–P4), lock-file patch list, §9 numbering, config-precedence subsection added (§9.4).
- No code yet. Next: **B1**.
