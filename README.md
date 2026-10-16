# GoLeM — גולם

**Dormant experts, awakened on demand.**

GoLeM is a server for running Mixture-of-Experts LLMs whose expert weights stay
dormant on disk and are promoted to system RAM and then VRAM only as they are
used — with the hard guarantee that **every selected expert's matrix
multiplication executes on the GPU**, including cold experts fetched from disk
on demand. v1 target model: [GLM-5.3-Flash](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) (hence the name).

## Target hardware

Road-tested on one machine; smaller configs work, larger ones just add headroom:

| Component | Spec |
|---|---|
| GPU | AMD RX 7900 XTX, 24 GiB VRAM (ROCm/HIP) |
| RAM | 93 GiB DDR5 |
| Storage | Samsung 990 PRO 4TB NVMe SSD (expert streaming is SSD-bound) |

Any OpenAI-compatible client talks to `http://127.0.0.1:8080/v1`.

Anthropic clients (Claude Code, the SDKs) use the same port via `POST /v1/messages`.

## Benchmarks

Measured on the target machine — GLM-5.3-Flash UD-Q4_K_XL, ctx 262144, greedy,
release HIP build:

| GPU | RAM | Storage | Quant | Context | Prefill (cold/warm) | token/s (cold/warm) |
|---|---|---|---|---|---|---|
| AMD RX 7900 XTX | 93 GiB DDR5 | Samsung 990 PRO 4TB NVMe | UD-Q4_K_XL | 262144 | 11.1 s / 13.4 s | 2.42 / 2.12 |

Run [`benchmarks/bench.sh`](benchmarks/bench.sh) on your machine and add a
row to [`BENCHMARKS.md`](BENCHMARKS.md).

Community results from other machines live in
[`BENCHMARKS.md`](BENCHMARKS.md).

## Quick start

```sh
scripts/build.sh build                       # release build → target/release/golem

# 1. Pull the model from Hugging Face (example: unsloth dynamic UD-IQ4_XS)
hf download unsloth/GLM-5.3-Flash-GGUF \
  --include "UD-IQ4_XS/*" --local-dir ~/models/glm-5.3-flash-ud-iq4-xs

# 2. Plan memory (exit 2 = does not fit)
scripts/build.sh run plan ~/models/glm-5.3-flash-ud-iq4-xs/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf --ctx 32768

# 3. Serve
scripts/build.sh run serve ~/models/glm-5.3-flash-ud-iq4-xs/UD-IQ4_XS/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf \
  --port 8080 --ctx 32768 --n-predict 4096 --prefill-stream
```

`golem plan` prints the memory plan; `golem optimize` sweeps scheduling knobs on your
machine and saves a winner that `serve`/`run` apply automatically; `--detach`
runs in the background (`golem list`, `golem stop <id>`). (`golem import` —
ingesting an HF cache layout into the store — is planned but not implemented.)

Anthropic clients:

```sh
M_NAME=<--model-name>
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_AUTH_TOKEN=<--api-key or any> \
ANTHROPIC_MODEL=$M_NAME ANTHROPIC_SMALL_FAST_MODEL=$M_NAME \
ANTHROPIC_DEFAULT_HAIKU_MODEL=$M_NAME ANTHROPIC_DEFAULT_SONNET_MODEL=$M_NAME \
ANTHROPIC_DEFAULT_OPUS_MODEL=$M_NAME claude
```

Claude Code's background calls (titles, summaries) use the small/fast and
per-tier models; unset, they name a `claude-*` model and get
`golem_model_mismatch`. Claude Code's first request carries ≈ 53 KB of system
prompt and tool schemas, so `--ctx` must exceed that plus the conversation.

## Repository layout

```
engine/        llama-cpp.lock — pinned fork commit + validated flags
vendor/        vendored llama.cpp fork (squashed import + Golem hook patches)
crates/        Rust workspace (golem-cli, -server, -engine, -llama-sys, -gguf,
               -placement, -store, -plan, -api, -registry) + xtask
scripts/       substrate build + hardware test scripts
benchmarks/    community benchmark runner (bench.sh)
docs/          tech-design.md (design doc, referenced across the codebase), procedures/
BENCHMARKS.md  community benchmark table + methodology
```

## Provenance

This repository is vibe-coded: built by Claude Opus 5/5.5 and GLM-5.3-Flash
(running both hosted and locally via GoLeM itself), with human review. See
[`AGENTS.md`](AGENTS.md) for the engineering rules the agents work under.

## License

See [`LICENSE`](LICENSE).
