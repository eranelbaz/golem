# Golem — גולם

Dormant experts, awakened on demand.

Golem is a server for running Mixture-of-Experts LLMs whose expert weights
stay dormant on disk and are promoted to system RAM and then VRAM only as
they are used — with the hard guarantee that **every selected expert's matrix
multiplication executes on the GPU**, including cold experts fetched from disk
on demand. v1 target: GLM-5.3-Flash on an AMD RX 7900 XTX (ROCm/HIP).

See [`docs/product-spec.md`](docs/product-spec.md) for the what and why, and
[`docs/tech-design.md`](docs/tech-design.md) for the how — both are local-only
(with the worklog and per-task plans) and not committed to the repo.

## Status

Design approved; execution in progress (M0 — feasibility spike). Task board and
current phase: `docs/worklog.md` (local-only; not committed). The CLI below is
the designed interface — it lands across M3/M4 and is documented here as the
contract the implementation is built against.

## Using Golem

Golem never downloads models. You obtain the weights yourself (e.g.
`hf download unsloth/GLM-5.3-Flash-GGUF`), then hand Golem local paths.

```bash
# 1. Import: verify shards, build the manifest, mirror to more disks (optional,
#    but recommended — one path per physical drive = read striping + redundancy)
golem import ~/models/glm-5.3-flash-ud-q4k-xl \
             /mnt/disk2/models/glm-5.3-flash-ud-q4k-xl

# 2. Check what a config will do to your memory before starting
golem plan glm-5.3-flash-ud-q4k-xl --ctx 16384

# 3. Serve (foreground) or chat (auto-starts a server in the background)
golem serve glm-5.3-flash-ud-q4k-xl
golem run   glm-5.3-flash-ud-q4k-xl

# 4. Inspect / stop
golem list            # running instances (add --models for imported models)
golem stop <id|name>
```

Notes:

- `import`'s first path is the source of truth; every additional path becomes a
  **full mirror** that Golem creates itself (streamed SHA-256-verified clone).
  Full mirrors only — one complete shard set per path.
- Two replica paths on the same physical disk work but gain no bandwidth
  (Golem warns at import time).
- OpenAI-compatible API on `127.0.0.1`: `/v1/chat/completions` (streaming +
  non-streaming), `/v1/models`, `/health`. Reasoning tokens surface as
  `reasoning_content`.

## Repository layout

```
engine/        llama-cpp.lock — pinned fork commit + validated flags (M0.1 ✓)
vendor/        vendored llama.cpp fork — squashed import (M0.1 ✓)
crates/        Rust workspace (golem-cli, -server, -engine, -llama-sys,
               -placement, -store, -plan, -api, -registry) + xtask
scripts/       substrate build + measurement scripts (M0.2/M0.4 ✓)
docs/          baselines/ committed (manifests, baseline-001.json); design,
               worklog, plans, notes are local-only
```

## Working on Golem

Each unit of work is a task from the worklog board: branch `task/<id>-<slug>`,
write and plan-review an implementation plan under
`docs/superpowers/plans/` (local-only), implement per the
approved plan, verify the task's acceptance commands, PR. The full protocol is at
the top of `docs/worklog.md` (local-only).
