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

Design approved, execution starting. Task board and current phase:
`docs/worklog.md` (local-only; not committed).

## Repository layout

```
docs/          local-only docs: product spec, tech design, worklog, per-task plans (superpowers), baselines, ADRs
engine/        llama-cpp.lock — pinned fork commit + validated flags (lands M0.1)
vendor/        vendored llama.cpp fork — squashed import (lands M0.1)
crates/        Rust workspace (golem-cli, -server, -engine, -llama-sys,
               -placement, -store, -plan, -api, -registry) + xtask
scripts/       substrate build + hardware test scripts (lands M0.2)
```

## Working on Golem

Each unit of work is a task from the worklog board: branch `task/<id>-<slug>`,
write and plan-review an implementation plan under
`docs/superpowers/plans/` (local-only), implement per the
approved plan, verify the task's acceptance commands, PR. The full protocol is at
the top of `docs/worklog.md` (local-only).
