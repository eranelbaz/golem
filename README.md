# GoLeM — גולם

**Dormant experts, awakened on demand.**

GoLeM is a server for running Mixture-of-Experts LLMs whose expert weights stay
dormant on disk and are promoted to system RAM and then VRAM only as they are
used — with the hard guarantee that **every selected expert's matrix
multiplication executes on the GPU**, including cold experts fetched from disk
on demand. v1 target model: [GLM-5.3-Flash](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) (hence the name).

Supported models:

| Model | `general.architecture` | Status |
|---|---|---|
| [GLM-5.3-Flash](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) | `glm5next` / `glm5-next` | target, measured |
| [DeepSeek-V4.1-Flash](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) | `deepseek41` | experimental (serves UD-IQ2_XXS; no baseline yet) |

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

## Requirements

| | |
|---|---|
| OS | Linux only (`golem-llama-sys` refuses to build elsewhere) |
| GPU | AMD with ROCm/HIP, ROCm ≥ 7.2.4 (advisory pin in `engine/llama-cpp.lock`) |
| Toolchain | Rust ≥ 1.99 (`rust-toolchain.toml` pins 1.99.1 under rustup), cmake, a C++ compiler, OpenMP (libgomp), libclang (bindgen) |
| Tools | [`hf`](https://huggingface.co/docs/huggingface_hub/guides/cli) CLI for pulling models; `cargo-nextest` for the tests |
| VRAM / RAM | Tested at 24 GiB VRAM / 93 GiB RAM; experts that fit neither stream from SSD |
| Disk | Fast NVMe; GLM-5.3-Flash UD-IQ4_XS is 100+ GiB, DeepSeek-V4.1-Flash UD-IQ2_XXS ≈ 280 GiB |

Sizing against the target (reference quant UD-Q4_K_XL, MB = 10⁶ bytes):
GLM-5.3-Flash experts are 42 layers × 288 experts × (4.7 + 4.7 + 5.8) MB
≈ 184 GB ≈ 171 GiB, more than 24 GiB VRAM + 93 GiB RAM = 117 GiB, so the cold
set is read from SSD.
After downloading, `golem plan` reports whether a given context fits.

**Not supported yet:** NVIDIA/CUDA, macOS, Windows. Linux only for now: the
substrate link table, `O_DIRECT`/`posix_fadvise` expert streaming and the ROCm
probe have no macOS/Windows equivalents yet.

### Environment variables

| Variable | When | Default | Effect |
|---|---|---|---|
| `GOLEM_LLAMA_SYS_BACKEND` | build | `auto` | `auto` picks HIP when `$ROCM_PATH` has `bin/hipcc`, `llvm/bin/clang++` and `lib/libamdhip64.so`, else builds a CPU-only substrate with a `cargo:warning`; `hip` requires ROCm; `cpu` forces CPU |
| `ROCM_PATH` | build | `/opt/rocm` | ROCm install root |
| `GOLEM_LLAMA_SYS_AMDGPU_TARGET` | build | `gfx1100` | GPU arch; set it for any card other than RDNA3 Navi 31 |
| `GOLEM_LLAMA_SYS_SKIP_PIN_GATE` | build | unset | `1` skips the vendor pin check (warns) |
| `GOLEM_ALLOW_PARALLEL` | run | unset | `1` allows `serve --parallel > 1` |

Example for an RX 6800 XT (gfx1030) with ROCm outside `/opt/rocm`:

```sh
ROCM_PATH=/opt/rocm-7.2.4 GOLEM_LLAMA_SYS_BACKEND=hip \
GOLEM_LLAMA_SYS_AMDGPU_TARGET=gfx1030 scripts/build.sh build
```

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

DeepSeek-V4.1-Flash: pull a quant the same way (UD-IQ2_XXS ≈ 280 GiB) and
serve its first shard. Until the hybrid snapshot restore is fixed, serve with
`--snapshot-ram-gib 0`; on CPUs with an integrated GPU, set
`HIP_VISIBLE_DEVICES=0` so only the discrete GPU is used.

### Several models, one endpoint

`[models.<name>]` tables in `golem.toml` add served models. Each lists in
`/v1/models`; a request's `model` picks one. At most one is loaded: a
request naming another waits while the loaded one is dropped and the other
loads (warm-start scores are kept per model). Sampling stays with the
client.

```toml
model = "~/models/glm/GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf"
model_name = "glm-5.3-flash"
idle_unload_min = 30   # unload after 30 min without a request; 0/unset keeps it

[models."deepseek-v4.1-flash"]
path = "~/models/ds/UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf"
snapshot_ram_gib = 0   # any engine key here overrides golem.toml for this model
```

With `idle_unload_min` set, `serve` starts with nothing loaded
(`/health` → `idle`), the first request loads its model, and the model is
unloaded again after that many idle minutes. Without a top-level `model`,
the `[models.*]` entry named by `model_name` (else the first by name) is the
default. Each model gets its own `golem optimize` profile. Several models or
idle unload need `parallel = 1` and no `timing_csv`/`routing_trace`, and all
models must agree on setting `ram_experts_gib`/`cpu_moe_layers`. Engine
flags and `GOLEM_*` env apply to every model and beat per-model keys.

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

## Reasoning control

Reasoning (thinking) output can be turned off per request:

- OpenAI protocol (`/v1/chat/completions`): `"reasoning_effort": "none"`.
- Anthropic protocol (`/v1/messages`): `"thinking": {"type": "disabled"}`.

The GLM-5.3-Flash template has no `enable_thinking` switch, so no-think is
rendered as the template's own empty-think prefill (`<think></think>`): the
model emits content directly, with no reasoning tokens.

Honoured `reasoning_effort` values are exactly `low`, `high` and `none`
(case-sensitive); `thinking.type` accepts `enabled`, `adaptive` and
`disabled`. Unknown values of any type are rejected with a 400, never
silently mapped to the maximum effort. Omitting the field keeps the
template default (`max`). Combining `thinking: disabled` with
`output_config.effort` is a 400 conflict; `output_config.effort: "max"`
maps to the template default.

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
