# Golem

## Name

Golem — גולם

The Golem is an artificial being brought to life from inert material. That's a
nice metaphor for experts sitting dormant on disk and becoming active when
needed. It's also immediately evocative for an AI project.

Golem — dormant experts, awakened on demand.

## Product purpose

Golem is a server for running Mixture-of-Experts (MoE) LLMs that do not need
to be fully loaded in memory at all times.
[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) is one
example of a model designed to keep some of its parameters offloadable —
though in that model's case it's the N-gram embedding table, not MoE
experts. Golem applies that same "keep it dormant until needed" idea
specifically to MoE experts: keeping expert weights dormant on disk and
promoting them to system RAM and then VRAM as they're used, computing on the
GPU rather than the CPU.

## Core requirement: GPU compute, not just GPU caching

This is the project's differentiator, so it must be unambiguous:

- **Disk** holds expert weights that are not currently cached.
- **System RAM** caches expert weights and stages transfers toward the GPU.
- **VRAM** holds expert weights needed for GPU execution.
- **CPU** handles orchestration, file I/O, tokenization, and routing
  decisions — not tensor math.
- Every selected expert's matrix multiplication must execute on the GPU,
  **including a cold expert fetched from disk on demand** — not only experts
  that happen to already be resident in VRAM. There is no silent fallback to
  CPU compute for expert math; if GPU execution isn't possible for a given
  configuration, Golem must fail with an actionable error, not silently
  compute on CPU.
- Caching must never change the model's expert selection. Routing always
  picks experts the same way the reference model would; caching only affects
  *where* a selected expert's weights are loaded from, never *which* experts
  are selected.

### Comparison to existing projects

There are similar projects already:

- [rabbit](https://github.com/ferrumox/rabbit) — CPU-only inference, with
  disk-streamed experts, RAM caching, and multi-disk support.
- [deltafin](https://github.com/gavamedia/deltafin) — has GPU paths for
  attention and experts (CUDA, Metal). It also has a ROCm/HIP build path, but
  its own docs state that path has never been run on real AMD hardware.
- [colibri](https://github.com/JustVugg/colibri) — has GPU backends
  (CUDA/Metal/Vulkan) and disk/RAM/VRAM tiering. Its documented Vulkan path
  runs *hot* experts on GPU (uploaded once at startup) while *remaining*
  experts still compute on CPU. GPU support also varies by model.

None of these guarantee GPU compute for every selected expert regardless of
cache state, validated on AMD. That's the gap Golem targets.

## Expert placement: disk → RAM → VRAM

Placement is a strict staging pipeline: an expert promotes from disk to
system RAM, and from RAM to VRAM. VRAM residency is not populated directly
from disk, skipping the RAM stage.

Placement is controlled **per-expert**, not per-layer. (Params like
"how many layers on GPU" from earlier drafts are replaced by
per-expert placement limits — see CLI parameters below.)

Placement combines two mechanisms:

- **Automatic, frequency-based caching** promotes/evicts experts between
  disk, RAM, and VRAM based on usage (most-used experts stay closer to the
  GPU). The exact algorithm (LRU, LFU, thresholds, hysteresis) is a
  technical-design decision, not a product requirement — but the
  frequency-based *intent* itself is a product requirement.
- **User-set expert-count limits** (e.g. "keep at least N experts resident in
  VRAM") act as a **floor on top of automatic caching** — they guarantee a
  minimum resident set, but automatic promotion/eviction still operates for
  the rest. They are not a manual override that disables automatic caching.

## Model scope

### v1 target model

[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) — 321B total
(rounded to 320B in the model's own README) / 18B active parameters,
`glm5next` architecture, text (the vision tower is out of scope for v1; see
below).

**Weight source, in priority order:**

1. **Preferred:** an existing quantized GGUF build, e.g.
   [unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF).
   A specific quantization variant and file revision must be pinned before
   technical design starts (the repo has several: UD-Q4_K_XL, UD-Q5_K_XL,
   Q8_0, etc. — these are not interchangeable and have very different
   RAM/VRAM/quality tradeoffs).
2. **Fallback, only if no suitable GGUF quantization is available or
   compatible with Golem's GPU-compute requirement:** the official
   [zai-org/GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) FP8
   checkpoint (321B params, ~314GB in FP8, 62 safetensors shards, no GGUF),
   converted locally with Golem's own quantizer. This path requires building
   a source→quantized conversion pipeline as part of v1 scope (not deferred),
   and needs upfront verification that the target AMD GPU has usable
   low-precision matmul support for whatever format is chosen.

Golem should detect and report which path it's using; this is a build/setup
decision, not something silently chosen at runtime.

**Multimodal:** out of scope for v1. GLM-5.3-Flash ships a vision tower
(`mmproj` files); Golem v1 only needs to support the text path.

### Model preparation / quantization (separate capability)

Independent of which weight source v1 ends up using, Golem may eventually
ship its own **model preparation pipeline**: converting a supported source
checkpoint into a Golem-supported quantized format. This is useful for
reducing storage/transfer/cache pressure, but it is *not* automatically a
quality or performance improvement over existing prequantized GGUF builds —
it should be benchmarked against them, not assumed superior.

[Colibri's `convert_glm53.py`](https://github.com/JustVugg/colibri/blob/main/c/tools/convert_glm53.py)
is a useful reference for what this involves, with important caveats:

- It's model-specific (targets GLM-5.3 exactly), not a generic quantizer —
  it fails loudly on unrecognized tensor names rather than skipping them.
- It outputs a Colibri-specific container layout, not GGUF. Golem would need
  its own equivalent, not a drop-in reuse.
- Reference conversion cost: ~194.7GB output, ~25 hours on the reference
  machine. v1 UX must budget for this.
- Default behavior deletes source shards as it goes (disk-safe streaming
  conversion). A "preserve original files" mode needs disk-budget-aware
  design if wanted.

If/when this ships, it must specify: supported source model(s), output
quantization format, peak RAM/temp-disk requirements, resume/recovery
behavior, an explicit source-file preservation policy (never silently delete
user-owned originals), revision/provenance and integrity metadata,
tokenizer/chat-template preservation, and a quality/performance comparison
against a reference implementation.

### Extensibility to other models

The architecture should support extending to other MoE models, e.g.
[Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) — but this is a design
goal for future work, not a v1 guarantee. Different MoE model families differ
in attention mechanism, routing, tensor layout, and quantization format;
Kimi-K3 specifically ships safetensors with custom `trust_remote_code`
modeling code and MXFP4 precision, unrelated to GGUF/llama.cpp's loading
path. Supporting a second model family like this likely means maintaining a
second, mostly-independent inference stack — not just adding an architecture
case to one shared loader. The spec does not promise this works out of the
box; technical design should treat each new model family as its own scoped
effort.

## Hardware scope

**v1 target:** the author's own machine, AMD RX 7900 XTX GPU.

**NVIDIA:** should not require a redesign to add later, but is explicitly
**not** invested in or validated for v1 — there's no NVIDIA hardware to test
against right now.

### Known open risk: AMD correctness/perf is unverified

The most concrete existing reference for GPU-side expert compute on
GLM-5.3-Flash's architecture is
[llama.cpp PR #27754](https://github.com/ggml-org/llama.cpp/pull/27754),
which is:

- **still open/unmerged** — an unstable, moving target, not a shipped
  feature. Technical design will need to pin a specific commit of whatever
  engine dependency it ends up using, and accept that upstream changes may
  break Golem.
- **validated on NVIDIA (B200) only.** Its correctness fixes
  (`NVIDIA_TF32_OVERRIDE=0`, `-fa off` for MLA) are NVIDIA/CUDA-specific
  workarounds; there is no evidence they're needed, sufficient, or even
  applicable on ROCm/HIP.

Since AMD GPU compute is the whole point of this project, an early technical
feasibility check on the actual 7900 XTX — correctness first, then
performance — should happen before committing to a full architecture.

## Multi-disk support

Golem needs to support reading the model from multiple disks at once (e.g.
two paths where the model lives), since this workload is disk-heavy. This
needs to be specified precisely — see
[Colibri's dual-SSD design](https://github.com/JustVugg/colibri#dual-ssd-two-copies-of-the-model-twice-the-read-bandwidth)
as a reference for the kind of behavior to define:

- **Mode:** full mirrored copies on separate physical drives (for combined
  read bandwidth), vs. sharded/partial placement across drives. Pick one for
  v1 — "two paths" is ambiguous between these.
- **Validation:** how mismatched or partial copies across drives are
  detected and handled.
- **Failure behavior:** what happens when one drive becomes unavailable
  mid-run — degrade to the remaining drive(s), or fail.
- Two directories that happen to be on the *same* physical drive must not be
  assumed to provide extra bandwidth.

## API and CLI

### OpenAI API compatibility

Golem should support chatting with the model via an OpenAI-compatible HTTP
API. The exact subset needs to be pinned before technical design, at least:

- `POST /v1/chat/completions`, streaming and non-streaming.
- `GET /v1/models`.
- Whether tool/function calling is supported in v1.
- Whether reasoning/thinking output is exposed, and how.
- Explicit, typed errors for anything requested but unsupported (rather than
  silently ignoring unsupported fields).

### Commands

- `golem run` — interactive chat.
- `golem serve` — HTTP OpenAI-compatible API server.
- `golem list` — see all running Golem instances.
- `golem stop` — stop a running instance from the list.
- `golem plan` — estimate VRAM/RAM/disk usage and other useful info for a
  given model + configuration, without actually starting it.

Behavioral details technical design needs to pin down:

- Does `golem run` start its own model process, or attach to an
  already-running `golem serve` instance?
- Does `golem serve` run in the foreground or detach to the background?
- Does `golem list` show running processes, downloaded models on disk, or
  both?
- Does `golem stop` terminate a process, unload a model but keep the process
  alive, or both (as separate operations)?
- Can multiple models genuinely run concurrently sharing one GPU, or does
  each get an independent managed process? What happens when their combined
  memory budgets don't fit?

### Parameters

Golem should support the standard LLM server parameters, including at least:

- Context length.
- Per-expert placement limits on GPU (VRAM) and system RAM — see "Expert
  placement" above. This replaces any per-layer placement knob; placement in
  Golem is per-expert.

`golem plan`'s memory estimate must account for more than just weight/expert
byte counts: non-expert (dense/attention) weights, attention or recurrent
state, KV cache, prefill/decode workspaces, transfer/staging buffers, and the
expert cache itself. Configurations that don't fit should produce an
actionable error, not a silent CPU fallback or a crash partway through
loading.

## Acceptance criteria

No performance targets exist yet, and none should be invented here — GPU
compute does not guarantee faster inference when disk or PCIe transfer is the
bottleneck, so targets should come from measurement, not assumption. What's
required:

- A documented baseline: RAM, storage devices, OS, exact model quantization,
  and context length used for testing.
- Correctness must be validated against the same quantized model run in a
  reference runtime (e.g. plain llama.cpp for the GGUF path) — output should
  match, not just "look reasonable."
- Cold-cache and warm-cache behavior should be measured and reported
  separately, since they're expected to differ substantially in this
  architecture.

## Decisions left to technical design

Explicitly out of scope for this product spec, to be decided during
technical design instead:

- Implementation language and framework choices.
- Vulkan vs. ROCm/HIP (or both) as the AMD compute backend.
- Exact caching algorithm (LRU/LFU/thresholds/hysteresis) implementing the
  frequency-based placement policy described above.
- Internal module boundaries and code architecture.
- Which inference engine (if any existing one) to build on vs. build from
  scratch.
