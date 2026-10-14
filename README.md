# Golem — גולם

Dormant experts, awakened on demand.

Golem is a server for running Mixture-of-Experts LLMs whose expert weights
stay dormant on disk and are promoted to system RAM and then VRAM only as
they are used — with the hard guarantee that **every selected expert's matrix
multiplication executes on the GPU**, including cold experts fetched from disk
on demand. v1 target: GLM-5.3-Flash on an AMD RX 7900 XTX (ROCm/HIP).

## Status

Storage and placement tiers (M3) and serving (M4: OpenAI-compatible HTTP API,
registry, `golem plan`, `golem optimize`) are in place; M5 (validation and
hardening) is tracked in the GitHub issues labelled `phase-m5`. Measured
numbers live in [`docs/baselines/`](docs/baselines/).

## Quick start

```sh
scripts/build.sh build                       # release build → target/release/golem
M=~/models/<model>/<first-shard>.gguf
scripts/build.sh run plan "$M" --ctx 32768   # memory plan; exit 2 = does not fit
scripts/build.sh run serve "$M" --port 8080 --ctx 32768 --n-predict 4096 --prefill-stream
```

Any OpenAI client then talks to `http://127.0.0.1:8080/v1` (`/v1/chat/completions`
with streaming, tools and `reasoning_effort` — `low`/`high`, anything else is
`max`; `/v1/models`; `/health`; `/metrics`). Ctrl-C cancels in-flight work and
exits; `--detach` runs in the background (`golem list`, `golem stop <id>`). `golem optimize --model "$M"
--ctx <ctx> --max-minutes <n>` sweeps scheduling knobs on this machine and
saves a winner that `serve`/`run` apply automatically (flags still win).

Anthropic clients (Claude Code, the Anthropic SDKs) use `POST /v1/messages` on
the same port and key (`x-api-key` or `Authorization: Bearer`). The model name
is strict: send the instance's `--model-name`.

```sh
M_NAME=<--model-name>
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_AUTH_TOKEN=<--api-key or any> \
ANTHROPIC_MODEL=$M_NAME ANTHROPIC_SMALL_FAST_MODEL=$M_NAME \
ANTHROPIC_DEFAULT_HAIKU_MODEL=$M_NAME ANTHROPIC_DEFAULT_SONNET_MODEL=$M_NAME \
ANTHROPIC_DEFAULT_OPUS_MODEL=$M_NAME claude
```

Claude Code's background calls (titles, summaries) use the small/fast and
per-tier models; unset, they name a `claude-*` model and get
`golem_model_mismatch`.

Claude Code's first request carries ≈ 53 KB of system prompt and tool schemas
(≈ 13–15k tokens, measured 2026-10-14), so `--ctx` must exceed that plus the
conversation, and a cold first turn is a long prefill (design §8.6).

## Repository layout

```
engine/        llama-cpp.lock — pinned fork commit + validated flags
vendor/        vendored llama.cpp fork (squashed import + Golem hook patches)
crates/        Rust workspace (golem-cli, -server, -engine, -llama-sys, -gguf,
               -placement, -store, -plan, -api, -registry) + xtask
scripts/       substrate build + hardware test scripts
docs/          tech-design.md, baselines/ (measured numbers), procedures/ (hardware test procedures)
```

Engineering rules for contributors and agents: [`AGENTS.md`](AGENTS.md).
