# Golem — גולם

Dormant experts, awakened on demand.

Golem is a server for running Mixture-of-Experts LLMs whose expert weights
stay dormant on disk and are promoted to system RAM and then VRAM only as
they are used — with the hard guarantee that **every selected expert's matrix
multiplication executes on the GPU**, including cold experts fetched from disk
on demand. v1 target: GLM-5.3-Flash on an AMD RX 7900 XTX (ROCm/HIP).

## Status

Storage and placement tiers are in place (milestone M3); work is tracked in
the GitHub issues. Measured numbers live in [`docs/baselines/`](docs/baselines/).

## Repository layout

```
engine/        llama-cpp.lock — pinned fork commit + validated flags
vendor/        vendored llama.cpp fork (squashed import + Golem hook patches)
crates/        Rust workspace (golem-cli, -server, -engine, -llama-sys, -gguf,
               -placement, -store, -plan, -api, -registry) + xtask
scripts/       substrate build + hardware test scripts
docs/          baselines/ (measured numbers), procedures/ (hardware test procedures)
```

Engineering rules for contributors and agents: [`AGENTS.md`](AGENTS.md).
