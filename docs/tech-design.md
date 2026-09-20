# Golem — Technical Design

**Status:** Approved-for-spike (v1.0) — execution state is tracked in [`docs/worklog.md`](worklog.md); milestone exit criteria live in §14.
**Spec:** [`product-spec.md`](../product-spec.md) — read alongside; this document resolves the questions the spec defers to technical design.
**Date:** 2026-09-19

---

## 1. Decision summary

Every decision the product spec leaves open, resolved. Rationale in the referenced sections.

| # | Decision | Choice |
|---|----------|--------|
| D1 | Implementation language | **Rust** (stable, edition 2024) for everything except the vendored compute substrate |
| D2 | Inference engine | **Thin fork of llama.cpp (pinned commit) as compute/kernels substrate**, embedded in a Rust process via C ABI; Golem owns placement, paging, I/O, serving, CLI (§3) |
| D3 | AMD compute backend | **ROCm/HIP** on gfx1100 (RX 7900 XTX). Vulkan/wgpu deferred; NVIDIA later = same fork built for CUDA (§3.5) |
| D4 | v1 weights | **`unsloth/GLM-5.3-Flash-GGUF`, UD-Q4_K_XL**, exact repo revision + SHA-256 pinned in the model manifest at M0; the in-house quantizer is **not** v1 scope (§4) |
| D5 | Caching algorithm | Decayed-frequency score (LRU×LFU hybrid) + promotion/eviction hysteresis + min-dwell, per (layer, expert) key, independently per tier; user limits are **floors** (§6) |
| D6 | Multi-disk mode | **Full mirrored replicas**, striped per-read across drives; manifest-validated; degrade-on-drive-failure (§5) |
| D7 | OpenAI API subset | Chat completions (stream + non-stream), `/v1/models`, `/health`. Tools → typed error. Reasoning → `reasoning_content`. Strict-allowlist field policy (§8) |
| D8 | CLI semantics | `run` = ensure-instance + attach TUI; `serve` foreground by default (`--detach` supported); `list` shows instances by default, models with `--models`; `stop` = graceful terminate, `--force` for SIGKILL (§9) |
| D9 | Concurrency model | One process per instance; Rust-driven decode loop over libllama; dynamic micro-batching; tokenization + sampling via llama.cpp for bit-parity with the reference runtime (§7) |
| D10 | Disk I/O engine | Dedicated pool of blocking reader threads doing `pread64` into an **mlock'd** RAM expert pool; io_uring and O_DIRECT are opt-in flags, not defaults (§5.4) |
| D11 | Instance registry | JSON manifests + PID heartbeat + Unix-domain control socket under `$XDG_RUNTIME_DIR/golem/` (§9.4) |
| D12 | Substrate patch seam | Four surgical patches to the fork (P1–P4) exposing C-ABI policy hooks around the expert-copy path; policy state computed in Rust, applied in C (§3.4) |

Non-decisions, deliberately: no performance targets (spec §Acceptance — they come from M0 measurement); no second model family commitment (spec treats each as a scoped future effort).

---

## 2. Architecture overview

### 2.1 Process model

One OS process per running instance (`golem serve`, or `golem run` auto-starting one).
Inside the instance process:

```
┌────────────────────────── golem instance process ──────────────────────────┐
│                                                                            │
│  tokio runtime (async)                    dedicated decode thread(s)       │
│  ┌────────────────────────┐               ┌─────────────────────────────┐  │
│  │ axum HTTP server       │               │ decode loop                 │  │
│  │  /v1/chat/completions  │──queue──┐     │  build batch from slots     │  │
│  │  /v1/models  /health   │         │     │  llama_decode()             │  │
│  └────────────────────────┘         ▼     │   └─► expert-copy hooks ─┐  │  │
│  ┌────────────────────────┐   ┌──────────┐│  sampler (llama.cpp)     │  │  │
│  │ slot scheduler         │──►│ req queue││  detokenize + SSE emit   │  │  │
│  │ (dynamic micro-batch)  │   └──────────┘│                     │    │  │  │
│  └────────────────────────┘               └─────────────────────┼────┘  │  │
│  ┌────────────────────────┐   ┌─────────────────────────────┐   │       │  │
│  │ placement manager      │◄──┤ policy state (scores,       │◄──┘       │  │
│  │ (Rust)                 │   │ residency bitsets, floors)  │ C ABI     │  │
│  └─────────┬──────────────┘   └─────────────────────────────┘           │  │
│            │ page requests                                              │  │
│  ┌─────────▼──────────────┐                                             │  │
│  │ store (multi-disk I/O) │── pread64, striped across replicas ──► disk │  │
│  └────────────────────────┘                                             │  │
└────────────────────────────────────────────────────────────────────────────┘
        ▲ libllama (vendored fork, HIP backend) linked into the process
```

`golem list` / `golem stop` / `golem plan` are thin clients: the first two talk to the registry (§9.2), the last is a pure offline computation (§10).

### 2.2 The three tiers

| Tier | Backed by | Managed by | Size |
|------|-----------|------------|------|
| Disk | GGUF shard files on 1..N drives, mirrored | `golem-store` | full model (~165–200 GB est. for Q4_K_XL-class quant; authoritative numbers come from the GGUF tensor table, never hardcoded) |
| RAM | one mlock'd expert pool per instance (host buffer of the vendored substrate) | `golem-placement` + store | default: min(75% of free RAM at start, total expert bytes); `--ram-experts-gib` override |
| VRAM | device expert-copy cache (persistent across steps) + non-expert weights + KV + workspaces | scheduler patches (P2) under Rust policy | `--vram-experts-gib` budget / `--vram-experts-min` floor |

Everything that is **not** an expert weight (embeddings, attention projections, norms, router, output head) is loaded into VRAM at startup and stays there for the life of the instance. Only expert FFN weights participate in tiering — placement is per-expert, never per-layer (spec requirement).

### 2.3 Life of a decode step

1. Slot scheduler assembles the next-token batch for all active slots (§7.2).
2. `llama_decode` builds the graph and runs it on the HIP backend. Attention, router, and all shared weights live entirely in VRAM.
3. When the graph reaches a layer's `GGML_OP_MUL_MAT_ID` (expert FFN), the scheduler's expert-copy path needs that layer's selected experts resident in VRAM:
   - **P3 hook (host residency):** before reading host bytes, call into Rust. Fast path: residency bitset hit (no-op). Slow path: synchronous striped `pread64` from replicas into the RAM pool, then return. A prefetch thread (§6.5) makes the slow path rare.
   - **P2 patch (device resident-set):** copy only the byte ranges of experts whose device copies are invalid; skip ranges already resident in the device expert-copy cache from previous steps. Evictions happen here when the device cache exceeds its budget, in the order Rust's policy thread supplied.
4. Routing ids are produced by the router matmul inside the graph from activations, exactly as the reference model computes them — Golem never inspects, filters, or overrides ids. Access notifications (`notify-access(expert_id)`) are emitted to Rust per step, feeding the frequency scores.
5. Sample → detokenize → SSE frames.

**Invariant I1 (routing invariance):** caching affects only *where selected experts' bytes are staged from*. Weight bytes are bit-identical across tiers (verified by manifest hashes), the kernels and op order are identical (same vendored substrate), therefore activations, router logits, and expert selection are identical regardless of cache state. Tested explicitly (§12.2, T3).

**Invariant I2 (no CPU expert math):** expert tensors are attached to the graph as host-buffer inputs consumed by device-side `MUL_MAT_ID` — the CPU never executes an expert matmul. If the HIP backend is unavailable at startup, the instance **fails to start** with an actionable error (§8.4, `golem_backend_unavailable`); there is no CPU fallback path in the binary for expert math. `golem plan` rejects configurations that would require one.

---

## 3. Compute substrate

### 3.1 Options considered

| Option | Verdict | Why |
|--------|---------|-----|
| **A. Thin fork of llama.cpp + Rust orchestration** | **chosen** | HIP backend has full K-quant/MoE/FlashAttention support (verified feature matrix, §3.6); the split scheduler already implements just-in-time per-expert RAM→VRAM copies of only the routed experts (verified in `ggml-backend.cpp`); acceptance-criterion parity is nearly free because the compute substrate *is* the reference runtime; GLM arch fixes arrive as upstream merges we rebase onto |
| B. Pure-Rust engine on vendored ggml (own graph builder) | rejected for v1 | Same kernels as A but we re-implement the whole model graph (MLA attention, RoPE, KV management, samplers) in Rust over raw ggml-sys; loses upstream fix-tracking; parity becomes our regression to own. Revisit only if fork patches rot |
| C. candle / mistral.rs | rejected | candle's quantized-GPU path is CUDA-only (`--features cuda`); no HIP backend; mistral.rs inherits that and its offload is layer/topology-granular, not per-expert from disk. Wrong hardware target (gfx1100) |
| D. From-scratch Rust engine on wgpu/Vulkan | rejected for v1 | Vulkan K-quant kernels are documented as slow vs HIP/CUDA in llama.cpp's own feature matrix; writing competitive K-quant + MLA kernels for RDNA3 is a kernel-engineering project, not an orchestration project; portable-backend goals are secondary to the v1 hardware target |
| E. Full llama.cpp server fork (placement in C++) | rejected | Drags the disk tier, cache policy, and multi-disk I/O into C++, exactly where the project's differentiation lives. Rust stays the product; the fork stays a substrate |

### 3.2 The fork

- Upstream: the **unslothai/llama.cpp** fork, pinned to the head of PR #27754 (the PR itself is hosted on `ggml-org/llama.cpp`; its head branch `glm5next/upstream` lives on the unslothai fork) — a commit that contains GLM-5.3-Flash (`glm5next`) text support. The spec itself flags that this support is unmerged/moving; therefore the commit is selected and frozen during **M0** (§14) and recorded in `engine/llama-cpp.lock`:

  ```
  # engine/llama-cpp.lock
  repo = https://github.com/unslothai/llama.cpp
  branch = glm5next/upstream
  commit = 86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9   # head of PR #27754
  pr = 27754
  base_upstream = 930e2fa5995789efbf249a8bf61325bb626e417b  # ggml-org/llama.cpp merge-base at pin
  content_sha256 = 812b9b4a7848e34a76c8d444e244fa787f6adb0a586e85a7317477a53ea9f794  # recomputable tree hash; CI re-derives it
  file_count = 3599                       # regular-file count at pin
  archive_sha256 = dfe22d466b1ceb824651ada7c6f20347f4f18caa241dc3459fb5f10560d265e3  # provenance only
  rocm = 7.2.4            # advisory minimum; M0.2 warns on rocminfo mismatch, skips the check when rocminfo is absent
  patches = none-yet      # pristine upstream-at-pin; P1–P4 land in M2.x (§3.4)
  # fa = <set at M0.4>    # FlashAttention/precision setting for MLA on ROCm (§3.5)
  ```

  *(values as of the M0.1 pin; `engine/llama-cpp.lock` is authoritative on divergence)*

- The vendored tree carries no nested `.git` by design, so the acceptance is content-based, not `git rev-parse`: CI's `vendor-pin-check` recomputes `content_sha256` from `vendor/llama.cpp` (canonical recipe, ADR 0001) and it must match `vendor/VENDOR_SHA` and the lock.
- The fork is vendored in-tree (`vendor/llama.cpp/`) and built by the `golem-llama-sys` build script via the `cmake` crate with `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100`.
- Patch discipline: total diff vs upstream ≤ ~1,000 lines, confined to the files listed in §3.4. Every patch has an in-tree regression test (§12.2, T4). Upstream rebase is a scheduled maintenance task (monthly + on GLM-arch fixes), never an ad-hoc drift.
- The fork carries **no** model-architecture code of its own and **no** product logic in C++ beyond the hooks. If a future rebase turns patches into a maintenance sink (>2 consecutive rebase failures), that is the documented trigger to re-open decision D2, option B.

### 3.3 What we use from the substrate

| Concern | Source |
|---------|--------|
| GGUF parsing, tensor loading, mmap | llama.cpp loader, driven through `llama.h` |
| Model graph (glm5next text path), attention, RoPE, norms, router | llama.cpp |
| KV cache (allocation, quantized-KV options, defrag) | llama.cpp (`-ctk/-ctv` exposed as config) |
| K-quant GEMV/GEMM kernels on HIP | ggml HIP backend |
| Tokenization (BPE + GGUF chat template) | llama.cpp — *not* a Rust tokenizer; byte-parity with the reference runtime matters more than library purity |
| Sampling | llama.cpp sampler chain, driven by `llama.h` with an explicit seed — reproducibility is an acceptance requirement (§12.2) |
| Continuous batching / HTTP server / placement logic | **not** the substrate's — Golem replaces `llama-server` entirely; we drive `llama_decode` ourselves (§7) |

### 3.4 Substrate patches (the entire C++ delta)

**P1 — hook plumbing.** `ggml/src/ggml-backend.cpp` + a new `ggml/include/golem-hooks.h`: registers three C function pointers (`ensure_host_resident`, `policy_query`, `notify_access`) and per-tensor expert metadata (expert count, per-expert byte size, stride) at buffer-usage-weights time. No behavior change when hooks are unset (upstream-semantics default), which keeps upstream tests green.

**P2 — persistent device expert-copy cache.** In the split-scheduler expert-copy path (the code that already groups consecutive routed experts and calls `ggml_backend_tensor_set_async`): maintain, per weight tensor, a validity bitset over expert ranges in the device copy buffer. Copy only invalid ranges of the routed set; when the total valid bytes exceed the device budget, evict in the order most recently supplied by `policy_query` (Rust), i.e. true LRU-of-policy ordering applied in C with no callback latency in the hot loop. Previously-copied hot experts are skipped entirely — this is the mechanism that makes "hot experts live in VRAM" real without touching graph construction.

**P3 — host residency callback.** Immediately before the host→device range copy: `ensure_host_resident(tensor, first_id, last_id)` → Rust blocks until ranges are paged into the mlock'd RAM pool (bitset fast path; synchronous striped pread slow path). Callback runs on the decode thread; §6.5 explains why the blocking slow path is rare.

**P4 — counters.** Per-step counters (bytes copied host→device, bytes skipped because resident, page-ins served from RAM vs disk, evictions) exposed through a lock-free stats block read by Rust; exported as metrics (§11).

### 3.5 Backend: ROCm/HIP, and what "not NVIDIA" means

- v1 target: **gfx1100 (RX 7900 XTX)**, ROCm **7.2.4** as the advisory minimum recorded in `engine/llama-cpp.lock` (M0.2 warns on `rocminfo` vs lock mismatch and skips the check when `rocminfo` is absent — CI/no-GPU), HIP backend of the fork with `AMDGPU_TARGETS=gfx1100`.
- The known NVIDIA-specific workarounds from PR #27754 (`NVIDIA_TF32_OVERRIDE=0`, `-fa off` for MLA) are CUDA-specific. Their ROCm analogues (hipBLASLt reduced-precision options, FlashAttention on MLA) are **unknown on gfx1100** — resolving them with correctness evidence is an explicit M0 exit criterion (§14), not an assumption. Whatever setting M0 validates is written into `engine/llama-cpp.lock` and made the default; a config escape hatch (`--fa {on,off}`) stays exposed.
- Vulkan: out of v1. The seam is clean — the substrate's Vulkan backend is a build-flag swap, and Golem's policy/patch seam (P1–P3) is backend-agnostic. When Vulkan K-quant kernels mature, support is a build+validation effort, not a redesign.
- NVIDIA: same story — the fork's CUDA backend is a build-flag swap of the same kernels. No NVIDIA-specific code anywhere in Golem's Rust layers.

### 3.6 Evidence base (from context7, 2026-09)

- llama.cpp feature matrix: ROCm column — K-quants ✅, I-quants ✅, MoE ✅, Flash Attention ✅, K-cache quants ✅; Vulkan K-quants marked slow ("the CUDA or ROCM backends are faster"). This is the empirical basis for D3.
- `ggml-backend.cpp` split scheduler: when MoE weights are offloaded to a host buffer, it reads the ids tensor, builds a used-expert bitset, groups consecutive ids, and copies **only those ranges** host→device per eval (`ggml_backend_tensor_set_async`), including small padding guards for MMQ. This is upstream's existing just-in-time cold-expert-to-GPU path; Golem's P2/P3 wrap policy around it.
- `--n-cpu-moe` / `-ot exps=CPU` are shipped, exercised configurations that put expert weights in host buffers while the graph runs on GPU — i.e., the "cold expert computes on GPU" configuration is upstream-supported and debugged, not speculative.
- `--lazy-mode` (on-demand reads of selected tensors from disk) exists upstream but only for specific tensor kinds (e.g. oversized embeddings), never experts, and rides the kernel page cache. Golem's explicit RAM pool (§5.4) supersedes it for experts: bounded memory, multi-disk striping, and accounting — three things page cache cannot give us.

---

## 4. Model loading and GGUF handling

### 4.1 v1 weights (D4)

- Primary: `unsloth/GLM-5.3-Flash-GGUF`, quantization **UD-Q4_K_XL** — the best capacity/quality tradeoff for a 24 GB VRAM + disk-tiered architecture (≈4.5–5 bpw keeps the resident-set calculus meaningful). Exact repo revision and per-shard SHA-256s are pinned into the model manifest during M0 (the repo carries several non-interchangeable variants; the manifest makes the choice auditable and `golem plan` exact).
- Auxiliary for validation: **Q8_0** of the same revision — used by the parity harness (§12.2) because high-precision quant minimizes quantization noise when bisecting correctness issues across GPU/CPU execution.
- The official FP8 safetensors checkpoint + in-house quantizer is **not** v1 scope (the spec makes it a fallback "only if no suitable GGUF exists"; M0 confirms the GGUF path works on gfx1100 before anything else). The quantization pipeline stays a separately-scoped future capability per spec §Model preparation.

### 4.2 Manifest

Every downloaded model gets `$GOLEM_DATA_DIR/models/<name>/manifest.json`:

```jsonc
{
  "name": "glm-5.3-flash-ud-q4k-xl",
  "repo": "unsloth/GLM-5.3-Flash-GGUF",
  "revision": "<hf commit sha>",
  "quant": "UD-Q4_K_XL",
  "arch": "glm5next",
  "shards": [ { "file": "glm-5.3-flash-UD-Q4_K_XL-00001-of-000xx.gguf",
                "bytes": 0, "sha256": "<...>",
                "replicas": ["/mnt/ssd0/golem/models/...", "/mnt/ssd1/golem/models/..."] } ],
  "expert_index": [ { "tensor": "blk.3.ffn_gate_exps", "layer": 3, "experts": 256,
                      "bytes_per_expert": 0, "offsets_bytes": [/* per-expert */] } ],
  "chat_template_sha": "<...>"
}
```

`expert_index` is produced once at download/import time by walking the GGUF tensor table (expert tensors are identified by the upstream MoE regex `ffn_(up|down|gate|gate_up)_(ch|)exps`); it is the index the store and placement manager share. Vision-tower (`mmproj*`) tensors and files are refused at import with a clear message ("text path only in v1") rather than silently skipped — the spec bans silent anything.

### 4.3 Load-time checks (fail fast, fail loud)

1. Manifest SHA-256 over every shard (streamed, ~1–2 min for ~170 GB on NVMe; `--skip-verify` available and reported loudly in logs).
2. Expert count and bytes-per-expert uniformity per layer (GGUF guarantee; a violation means a corrupted/mismatched mirror — refuse to start).
3. VRAM budget feasibility: non-expert bytes + KV + workspace + floor-set bytes ≤ free VRAM minus measured ROCm overhead (M0 number). Failure → `golem_plan_infeasible` error with the exact over-budget line items (§10), never a degraded start.
4. Backend probe: `hipGetDeviceProperties` on gfx1100, kernel self-test (one expert matmul GPU vs known-good CPU result within tolerance). Failure → `golem_backend_unavailable` with install instructions (ROCm version, env vars) in the error body.

---

## 5. Storage tier (`golem-store`)

### 5.1 Mode (D6): full mirrored replicas

v1 supports **N ≥ 1 full copies** of the model on separate paths (the model bytes are identical; the manifest's `replicas` array names them). Mirrored full copies — not sharded placement — because: (a) per-read striping across replicas already extracts aggregate bandwidth, so sharding adds a failure mode without extra bandwidth; (b) replicas double as validation targets for each other; (c) sharding can be layered later without interface changes (`replicas` → `shards` is a manifest evolution, not an API break).

**Same-physical-drive detection:** two replica paths whose resolved `st_dev` maps to partitions of the same parent block device (resolved by walking `/sys/block/<parent>/<partition>`) are flagged at startup: `warn: replicas share a physical device; no bandwidth aggregation`. They still function (fault-tolerance is zero there, and we say so), we just don't advertise aggregate bandwidth or prefer them for striping.

### 5.2 Read path

- Unit of I/O: one expert's byte range within one tensor (from `expert_index`), typically 1–8 MB per expert per tensor per layer.
- A read fans out across healthy replicas in contiguous chunks (chunk count = replica count, chunk size = bytes/replicas) issued concurrently; completion is when all chunks land. Chunk boundaries are deterministic (offset arithmetic), so a retry on another replica is trivially resumable.
- Reads land directly in pre-pinned slots of the mlock'd RAM pool (no bounce buffers); completion updates the residency bitset and releases the placement manager's waiter.
- Errors on one replica (EIO, device gone): mark replica degraded (health state, metric, log), transparently retry the failed chunk on the next healthy replica. If **all** replicas of a shard are unreadable: instance transitions to `degraded:fatal-io` — in-flight requests complete with an error, new requests get `golem_storage_unavailable`; the process stays alive so `golem stop`/`golem list` keep working and the user can pull the drive and recover. (Fail-soft per spec "degrade or fail": we degrade until reads are impossible, then fail loudly but stay manageable.)

### 5.3 Validation

- Full SHA-256 verify at import and at instance start (§4.3). Sizes re-checked on open.
- Continuous in-flight hashing would tax the read path; instead each pool page-in is covered by a cheap **CRC32C** appended into the page table entry, verified on use. Mismatch → treat as corrupted replica (same path as EIO) and log the offending shard+range. This catches silent rot between full verifies at ~3 GB/s/core cost, negligible against disk reads.
- On any detected mismatch, `golem list` shows the instance as `degraded`, naming the shard and replica — actionable, not silent.

### 5.4 I/O engine (D10)

- **Default:** a fixed pool (default 8, `--io-threads`) of dedicated reader threads doing `pread64` on `O_RDONLY` fds, one per replica file. Rationale: expert reads are large (MB-scale), few, and schedulable; thread-pool pread is within ~10% of io_uring at this op size, and it keeps the decode-thread slow path (§6.5) simple and bounded.
- The RAM expert pool is one `mmap(MAP_ANONYMOUS | MAP_NORESERVE)` + `mlock` region carved into fixed-size expert slots (per-layer uniform bytes-per-expert from the manifest). mlock prevents the kernel from double-evicting what our own cache thinks is resident — the class of bug that produces nondeterministic "disk reads" inside a supposedly-warm cache.
- Host page cache: opened with `posix_fadvise(DONTNEED)` management on the replica fds is *not* used in v1 (double-caching of expert bytes in page cache is bounded by the pool being ≤75% of RAM; revisit if M0 measurement shows memory pressure).
- io_uring (`tokio-uring`) and `O_DIRECT` remain compile-time-available, runtime-opt-in (`--io-engine {threads,io_uring}`, `--direct-io`) for M0's bandwidth experiments; they are not defaults until they win a benchmark.

### 5.5 Download/import

`golem pull <repo> [--quant ...] [--replica-on /path ...]` — resolves the HF repo, downloads shards (resumable, verified), builds `expert_index` by walking the GGUF header, writes the manifest. Two replicas requested → the second copy is a byte-for-byte clone verified against the first's hashes. (This command is additive to the spec's CLI list — the spec's `list --models` presumes downloaded models exist; import is the missing verb.)

---

## 6. Expert placement (`golem-placement`)

### 6.1 Identity and state

Key: `(layer, expert_id)` — model-agnostic, works for any MoE arch the substrate loads.

Per key, the placement manager holds: `tier ∈ {disk, ram, vram}` residency flags (independent per tier: an expert can be in RAM but not VRAM, etc.), `score: f64`, `last_access_step`, `dwell_step` (steps since last tier transition), and a CRC over its RAM slot. State is a flat `Vec` indexed by `layer * E + e` — no heap per step, lock split per layer shard.

### 6.2 Score (D5)

```
score(k) ← decay · score(k)          every step (decay = e^(−1/H), H default 512 steps)
score(k) ← score(k) + 1              on access (notified via P1 hook)
```

This is exponential-decay frequency: a blend of LRU (recency dominates for small H) and LFU (accumulated use dominates for large H), tunable by one parameter. Defaults (H = 512, thresholds below) chosen so that a "hot set" is an expert accessed more than ~once per H steps; M0/M4 tune on real traces.

### 6.3 Transitions

- **disk → RAM** (page-in): on demand via the P3 slow path, or speculatively by the prefetcher (§6.5). Admission if pool has a free slot, else evict lowest-score RAM-resident expert whose `dwell_step > D_min` (min-dwell, default 256 steps) and which is not floor-pinned.
- **RAM → VRAM** (promotion): a background policy pass (runs on the placement thread at ≥1 kHz or every step, whichever is coarser) orders RAM-resident experts by score and marks the top ones as *prefer-resident* up to the VRAM expert budget; P2 consumes this ordering at its next eviction decision. Thresholds are hysteresis-widened: promote above `θ_hi = 1.0`, demote-below `θ_lo = 0.4` (an expert that oscillates across the boundary is damped by min-dwell, not by threshold gap alone).
- **VRAM → RAM** (eviction): P2 evicts by policy order when the device budget overflows; the RAM copy (if present) is untouched, so re-promotion costs a device-side copy only if the RAM copy survived, else a page-in.
- **Floors:** `--vram-experts-min N` and `--ram-experts-min N` pin the current top-N by score in the respective tier. Floors are computed once per policy pass from current scores — they guarantee a minimum resident set (spec: "floor, not override") while automatic promotion/eviction continues to operate above the floor. `--vram-experts-gib`/`--ram-experts-gib` are budgets (caps on the cache), orthogonal to floors; both may be set.

Startup warm-up: if a previous run's placement journal (§11) exists, scores are seeded from it, and the floor set + top-scored experts are page-in'd and promoted before the first request — this is what makes "warm start" measurable separately from cold (§12.3).

### 6.4 Budgets and feasibility

The placement manager refuses at load time (not at step 50,000) any configuration whose floor set cannot fit: `vram_experts_min · max_bytes_per_expert > vram_budget_for_experts` → `golem_plan_infeasible` naming the numbers. All budgets derive from §10's arithmetic, so `golem plan` and the runtime agree by construction (same crate, `golem-plan`, computes both).

### 6.5 Prefetch

Autoregressive decoding means step N+1's routing is highly correlated with step N's. The placement thread receives step N's access set per layer and issues page-ins for the union of the last k=3 steps' sets that aren't RAM-resident (k and on/off are config; measured in M0). Prefetch reads use the same striped read path at lower priority (a semaphore caps outstanding prefetch I/O so demand reads always win). Expected effect, to be *measured* not assumed: turning a large fraction of P3 slow paths into bitset hits. The counters from P4 make the hit rate directly observable (§11), so the hypothesis is falsifiable in the first week of real use.

### 6.6 Correctness under eviction

An expert may be evicted from RAM only if no in-flight read targets it and it is not floor-pinned; in-flight P3 waits hold a reference. All transitions happen on the placement thread against the bitsets; the decode thread's P3 callback consults the same bitsets under a per-layer shard lock held only for bitset flip + slot handoff (never across I/O). The lock discipline is: **bitset ops are the critical section; I/O happens outside locks** — reviewed against data races in T6 (§12.2) with a loom test over (page-in, evict, access) interleavings.

---

## 7. Inference runtime

### 7.1 Slot scheduler

- A slot = one chat sequence: KV-cache sequence id, request state, sampling params, output channel.
- v1 concurrency: up to `--parallel N` slots (default 1 for `golem run`, 4 for `golem serve`). MoE + disk-paged experts makes heavy batching a bandwidth amplifier (each slot's routed experts union); v1 keeps batching modest and *measures* before raising defaults — batching interacts with the placement hot set, so the default is conservative by design.
- Batching is dynamic: each decode iteration includes every slot whose tokens are ready (continuous batching, slot join/leave at iteration boundaries — the same discipline `llama-server` uses, re-implemented over `llama.h`, which supports multi-sequence batched decode).

### 7.2 Decode loop

Dedicated thread(s) (`--decode-threads`, default 1 — HIP kernels own the GPU):

1. Drain ready continuations from the queue into a `llama_batch` (prompt chunks on join; single tokens while generating).
2. `llama_decode`.
3. Per slot: sampler chain (llama.cpp, seeded; §12.2 reproducibility) → append token → emit delta.
4. Notify placement manager of step completion (score decay tick, access notification flush).
5. Loop; preempt on stop/preemption signals from the async side (slot cancel = KV sequence rewind/removal per llama.cpp state APIs).

Prefill and decode are the same loop; a long prompt is processed as chunked prefill batches (chunk = 512 tokens default) so a cold joiner doesn't starve generating slots beyond one chunk.

### 7.3 Chat template and reasoning content

- Prompt construction applies the GGUF-embedded chat template via llama.cpp — parity with the reference runtime again. No Golem-side template code.
- GLM-5.3-Flash's thinking output: the template's reasoning delimiters are exposed as `reasoning_content` on the message/delta (DeepSeek-style convention), split by the template's known marker tokens (the exact markers are read from the chat-template metadata and pinned in M0; they are a property of the model revision, which the manifest already pins). `--reasoning {inline,separate}` selects behavior; default `separate`. When `inline`, raw tokens stream as `content` with markers intact (clients that don't understand `reasoning_content` see the model's native output — never silently stripped; spec bans silent behavior).

### 7.4 Determinism

Same weights (bit-identical), same kernels, same op order ⇒ same activations *for a fixed batch composition*. Known substrate nondeterminism (batch-size-dependent reduction orders; FA kernels) is controlled in v1: parity harness fixes batch=1, context length, FA setting (M0-validated), and seed. Production serving does not promise cross-request bitwise determinism (no LLM server does); it promises per-(seed, batch-composition) determinism, which the harness exercises.

---

## 8. HTTP API (D7)

### 8.1 Endpoints

| Endpoint | Status | Notes |
|----------|--------|-------|
| `POST /v1/chat/completions` | v1 | streaming (`stream: true`, SSE) and non-streaming |
| `GET /v1/models` | v1 | the instance's single model |
| `GET /health` | v1 | Golem extension: liveness + tier states + degraded flags (§11) |
| `GET /metrics` | v1 | Prometheus-format (§11) |
| everything else | — | 404 `golem_not_found` (typed, §8.4) |

### 8.2 Request field policy

Strict allowlist — anything requested-but-unsupported gets an explicit typed error, per spec. Tolerated-only fields (no functional effect) are `metadata`, `user`, `request_id`.

| Field | v1 behavior |
|-------|-------------|
| `model` | must match instance model name (else 400 `golem_model_mismatch`) |
| `messages` | supported; text content only; image/audio parts → 400 `golem_multimodal_unsupported` |
| `temperature`, `top_p`, `max_tokens` (`max_completion_tokens` aliased), `stop`, `seed`, `frequency_penalty`, `presence_penalty`, `n` (must be 1) | supported |
| `stream`, `stream_options.include_usage` | supported (SSE; final frame carries usage) |
| `tools`, `tool_choice`, `response_format` (json schema), `logit_bias`, `logprobs`, `reasoning_effort` | **400 `golem_unsupported_field`** naming the field and the reason. Clients probing capabilities get answers, not silence |
| `top_k` | Golem extension, supported |

Usage reporting: prompt/completion token counts from llama.cpp counters — cost-accurate, since we own the loop.

### 8.3 SSE shape

Standard OpenAI chunk framing (`data: {chunk}\n\n`, `data: [DONE]`), deltas carrying `content` and/or `reasoning_content` (§7.3), terminal frame `usage` when `include_usage`. Backpressure: SSE streams are tokio-guarded; a dead client cancels its slot (decode loop preempts at the next iteration boundary).

### 8.4 Error model

One envelope everywhere (HTTP + CLI + logs):

```jsonc
{ "error": { "code": "golem_plan_infeasible", "message": "...",
             "detail": { /* structured, e.g. per-tier byte overruns */ } } }
```

Codes (stable, documented): `golem_backend_unavailable`, `golem_plan_infeasible`, `golem_model_mismatch`, `golem_multimodal_unsupported`, `golem_unsupported_field`, `golem_storage_unavailable`, `golem_instance_not_found`, `golem_not_found`, `golem_busy`, `golem_invalid_request`. All 4xx/5xx carry `code`; the doc table maps codes → remediation.

### 8.5 Server posture

Binds `127.0.0.1` by default (port: auto-select, recorded in the registry; `--host/--port` to override). Optional bearer token (`--api-key`). No TLS in v1 (loopback tool; document the reverse-proxy path). `golem serve` runs in the foreground by default — logs to stderr, Ctrl-C = graceful drain — with `--detach` (double-fork, logs to `$GOLEM_DATA_DIR/logs/<instance>.log`, PID in registry) for the background habit.

---

## 9. CLI and instance lifecycle (D8)

| Command | Behavior (pinned) |
|---------|-------------------|
| `golem run [model]` | Ensures an instance for (model, compatible config) is running — starts one in the background if not (equivalent to `golem serve --detach` + wait-for-health) — then attaches an interactive chat client (stdin REPL over the HTTP API, streaming). Flags map onto instance config. Attach-vs-spawn is *not* a fork in the code path: `run` is a client + an ensure step |
| `golem serve [model]` | Foreground instance (§8.5). `--detach` supported. One model per process |
| `golem list` | Default: running instances (id, model, health, tier fill %, req/s, tok/s, degraded flags) from the registry. `--models`: downloaded models from the data dir with manifest summary (quant, size, replicas, last-used). Both, never conflated |
| `golem stop <id\|name>` | Graceful: stop accepting, drain in-flight requests (grace period 30 s), unload, exit. `--force`: SIGKILL. `--now`: skip drain. Stopping a nonexistent/dead instance is a typed error, not a shrug |
| `golem plan [model] --ctx N --vram-experts-min N ...` | §10. No GPU required; reads manifest + flags, prints per-tier byte table and a fit verdict with exact overages |
| `golem pull` | §5.5 |

### 9.2 Instance registry (D11)

`$XDG_RUNTIME_DIR/golem/instances/<id>.json`: id (slug: model + port + short uuid), pid, http endpoint, model name, config hash, started-at, health cache. Liveness = PID exists **and** UDS heartbeat (< 5 s old). Stale entries are reaped by any golem client that observes them (registry is advisory state, not a database). Control (stop, status detail) goes over a per-instance UDS carrying the same JSON-RPC-ish envelope as the error model — `golem stop` works even when HTTP is wedged.

### 9.3 Multiple instances / models

Each instance is an independent process with its own VRAM budget; concurrent instances are allowed and *individually* planned, but combined-fit is the operator's responsibility — `golem plan` checks a single instance; before starting a second instance, the CLI sums planned VRAM of running instances and **warns** (does not block) when the sum exceeds VRAM. Documented, not hidden. (Hard multi-instance VRAM arbitration is future work; the spec only asks the question be answered honestly.)

### 9.4 Config precedence

Flags > env (`GOLEM_*`, uppercase, one per flag) > config file (`$GOLEM_CONFIG_DIR/golem.toml`) > built-in defaults. The instance's effective config hash is recorded in the registry (§9.2); `golem run` only attaches to an instance whose config hash is *compatible* (same model; placement budgets that fit the running instance), else it reports the conflict and offers to start a new instance.

---

## 10. `golem plan` — the memory model

All figures derive from the GGUF header via the manifest — never hardcoded model tables. Let `E` = experts/layer, `L` = expert layers, `b_e` = bytes per expert per layer, and `T_ne` = total bytes of non-expert tensors (computed by filtering the tensor table through the upstream MoE regex — the same classifier llama.cpp uses, so "expert" means the same thing in both).

```
VRAM =
  T_ne                                    non-expert weights (resident forever)
+ kv_per_token · ctx · parallel            KV cache
+ ws                                       decode/prefill workspace (subsidy measured in M0; starts at 512 MiB)
+ staging                                  max b_e · 1.5 (P2 copy padding guard, §3.6)
+ device_expert_cache                      ≤ --vram-experts-gib, ≥ ceil of floor set
+ rocm_overhead                            measured in M0 (starts 768 MiB)

RAM =
  ram_pool                                ≤ --ram-experts-gib (default min(.75·free, Σ expert bytes))
+ host_overhead                            loader + activations (measured, starts 2 GiB)

DISK =
  model_bytes · replicas + logs
```

Output: table per tier, the four governing numbers (VRAM headroom, RAM pool, per-token KV cost, max cold-expert latency estimate from measured bandwidth), and a verdict. Infeasible → exit code 2, `golem_plan_infeasible` with the failing inequality printed with actual numbers. `--json` for tooling. The same functions serve the runtime's load-time feasibility check (§6.4) — one implementation, two callers, no drift.

KV per token without loading the model: read from the arch table (glm5next entry created in M0 from the loaded context's actual allocation, then frozen as a formula in metadata keys); when the GGUF carries the needed dims, the formula computes exactly.

---

## 11. Observability

- **Tracing:** `tracing` + JSON logs to stderr/file; spans for request lifecycle, decode steps (sampled), page-ins, evictions.
- **Metrics** (`/metrics`, Prometheus): token rates per slot; bytes disk→RAM (demand vs prefetch), RAM→VRAM (demand vs promote), VRAM evictions; P2 skip ratio (device-cache hit rate); P3 slow-path rate (the key health number — should be ≈0 steady-state); tier fill levels; replica health; per-code error counters.
- **Placement journal:** ring buffer (last N steps of access sets + transitions) dumped to `$GOLEM_DATA_DIR/journals/<instance>/` on shutdown, reused for score seeding (§6.3) and offline policy tuning.
- **Health states:** `healthy` / `degraded:{replica,io,checksum}` / `draining` / `fatal-io` — surfaced in `/health`, `golem list`, and log lines. Degraded is a first-class, visible state (spec: detect and report, never silently).

---

## 12. Testing and acceptance

### 12.1 Layout

- Unit tests per crate (policy math, manifest, striping, plan arithmetic — pure and table-driven).
- Integration tests run against a **synthetic tiny MoE GGUF** (a 2-layer, 8-expert model generated by a test helper that writes valid GGUF with the upstream tensor-naming scheme): exercises the full load → route → page → decode path in CI without 170 GB or a GPU (CPU-backend substrate build for CI; HIP build tested on hardware in the hardware lab job).
- Hardware tests (M0+ machine): nightly script, not CI-gated.

### 12.2 Test matrix

| ID | Test | Asserts |
|----|------|---------|
| T1 | Op parity: expert matmul (`MUL_MAT_ID`) on fixed weights/ids, HIP vs CPU substrate, Q8_0 and UD-Q4_K_XL | max abs diff within M0-calibrated bound (start: ≤ 1e-2 on fp16 accumulate path) |
| T2 | End-to-end greedy parity vs reference `llama.cpp` (same pinned commit, same GGUF): 50-prompt suite, batch=1, seeded | token-exact match ≥ 99.5%; every divergence reported (position, tokens); perplexity delta on wikitext-2 slice ≤ 0.5%. Thresholds recalibrated once at M0 against reference-vs-reference noise floor, then frozen |
| T3 | **Cache invariance** (spec I1): identical prompt under (a) fully pre-promoted cache, (b) cold start, (c) randomized eviction stress, (d) floors-only | all four produce **identical greedy token IDs** — the direct test that caching never changes selection |
| T4 | Fork patches: upstream test suite green; P1 unset-hooks behavior byte-identical to upstream; P2 validity-bitset invariants fuzzed | no behavioral delta without hooks |
| T5 | Store: striping correctness across 1/2/4 replicas, per-chunk failover mid-read, checksum-rot detection (injected), same-drive warning | reads correct under all replica-count/failure combos |
| T6 | Placement concurrency: loom test over (access, page-in, evict, promote) interleavings; 24 h soak with `--io-engine io_uring` on hardware | no races, no residency-bitset drift vs pool contents |
| T7 | API contract: field-policy table (§8.2) becomes generated round-trip tests; SSE framing golden tests | unknown field ⇒ typed 400; stream shape stable |
| T8 | Plan vs reality: `golem plan` predicted VRAM vs actual after load, on hardware, over ctx sweep | |predicted − actual| ≤ 256 MiB across sweep |

### 12.3 Benchmarks and the documented baseline

`golem bench` (dev binary, not a product subcommand): cold-start TTFP, warm TTFP, tok/s at ctx ∈ {4k, 16k, 32k} × cache ∈ {cold, warm, pre-promoted} × io-engine ∈ {threads, io_uring} — the cold/warm split the spec demands. Every run stamps the baseline record: RAM, drives (model + link), kernel, ROCm version, fork commit, quant + revision, ctx. Baselines live in `docs/baselines/` as committed JSON. **No target numbers exist until the first M0 baseline lands**; the spec forbids inventing them and so does this design.

---

## 13. Repository layout and toolchain

```
golem/
├── product-spec.md, docs/  (tech-design.md, worklog.md, superpowers/plans/, baselines/, adr/ ...)
├── engine/llama-cpp.lock          # pinned fork commit, ROCm line, patch list, M0-validated flags
├── vendor/llama.cpp/              # pristine fork at pin; P1–P4 patches + golem-hooks.h land in M2.x
├── crates/
│   ├── golem-cli/                 # clap commands; thin — all verbs delegate to libraries
│   ├── golem-server/              # instance process: axum API, slot scheduler, decode loop
│   ├── golem-engine/              # llama.h glue: load config, sampler chain, batch building, KV mgmt
│   ├── golem-llama-sys/           # bindgen over llama.h + golem-hooks.h; cmake build of the fork
│   ├── golem-placement/           # scores, tiers, floors, prefetch; policy consumed by P2/P3
│   ├── golem-store/               # replicas, striping, checksums, pool, pull/import
│   ├── golem-plan/                # §10 arithmetic (shared by CLI and runtime feasibility check)
│   ├── golem-api/                 # OpenAI types, SSE, error envelope, field-policy table
│   └── golem-registry/            # instance registry, UDS control, heartbeat
└── xtask/                         # bench harness, tiny-moe generator, parity suite runner
```

Toolchain and standards: Rust stable (currently 1.98, `rust-toolchain.toml` pins; edition 2024). `cargo clippy -D warnings`, `cargo fmt` check, `cargo nextest` in CI; deny.toml for license/advisory audit; `unsafe` confined to `golem-llama-sys` and the two FFI glue modules in `golem-engine`/`golem-placement` (hook trampolines), each with a `// SAFETY:` contract comment and reviewed in T6. Key crates: tokio, axum 0.8 + tower-http, serde/serde_json, clap derive, thiserror/anyhow (thiserror in libraries, anyhow at binaries), tracing, bindgen, cmake, sha2, crc32c, libc, sysinfo, uuid.

---

## 14. Milestones

Execution is tracked task-by-task in [`docs/worklog.md`](worklog.md) (task board, statuses, PRs). Each milestone below defines *exit criteria*; the worklog decomposes them into branch/PR-sized tasks.

Execution is **plan-first**: every board task is implemented from a written implementation plan in the superpowers:writing-plans format, stored under `docs/superpowers/plans/`, and that plan must pass a dedicated plan review (one reviewer subagent, quick mode; fix-and-redo loop capped at 3 rounds) **before** any implementation commit. The plan travels in the task's PR and is the authoritative execution record afterwards (worklog §0, steps 5–9).

**M0 — Feasibility spike on the 7900 XTX (gates everything).**
Select and lock the fork commit (GLM-5.3-Flash text support present); build HIP for gfx1100; validate FA/precision settings for MLA on ROCm (the open risk the spec flags); run `--n-cpu-moe`-style host-buffered experts; measure: cold-expert copy cost per step, achievable disk bandwidth (threads vs io_uring, 1 vs 2 drives), ROCm overhead, KV bytes/token, tok/s cold vs warm. Exit: T2 parity ≥ threshold on 8 prompts *and* measured cold tok/s above the "viable" line the author sets from the baseline (if disk→PCIe math says cold decode is unusable even with prefetch, the project re-scopes before more code exists). Deliverables: `engine/llama-cpp.lock`, baseline JSON #1, the M0 numbers feeding §10 constants.

**M1 — Substrate integration.** `golem-llama-sys` + `golem-engine`: load pinned GGUF, greedy single-slot decode from CLI, all non-expert weights resident, experts host-buffered with upstream default copy behavior. Exit: T1, T2 green on hardware; a chat completes.

**M2 — Placement engine.** P1–P4 patches + `golem-placement` (scores, hysteresis, floors, VRAM resident-set). Exit: T3, T4, T6 green; P2 skip-ratio and P3 slow-path metrics demonstrably better than naive per-step copies on the same trace.

**M3 — Storage tier.** `golem-store` (replicas, striping, checksums, pool, pull) wired into P3/prefetch; two-drive mode. Exit: T5 green; cold-start TTFP and cold tok/s measured on 1 vs 2 replicas; degraded-replica failover demonstrated by yanking a drive (literal test procedure in docs).

**M4 — Product surface.** `golem-api` + `golem-server` HTTP, slots/batching, reasoning content; `golem` CLI (run/serve/list/stop/plan/pull), registry; `golem-plan` with T8 calibration. Exit: T7, T8 green; the spec's five commands behave as §9 pins them.

**M5 — Validation hardening.** Full T2 suite vs reference runtime, 24 h soak (T6), baselines for cold/warm across the §12.3 matrix, README quickstart, baseline doc per spec §Acceptance. Exit: acceptance criteria of the product spec demonstrably met on the author's machine, with numbers committed under `docs/baselines/`.

---

## 15. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GLM support unmerged upstream; fork chases a moving PR | certain (spec states it) | high | pin commit in M0; patches confined and rebased monthly; documented trigger to re-open D2 |
| ROCm MLA/precision behavior differs from NVIDIA workarounds | likely | high | explicit M0 exit criterion; `--fa` escape hatch; settings frozen into lock file once validated |
| PCIe becomes the bottleneck; cold decode unusably slow | unknown | existential for the product | M0 measures before building; prefetch + resident-set design attacks exactly this; if the floor isn't met, re-scope happens at M0, not M4 |
| Fork patch conflicts on rebase | medium | medium | ≤1k-line diff, confined files, T4 as rebase gate; P2 is the deepest patch and is isolated to one scheduler function |
| Cache-state-dependent nondeterminism (I1 violation) | low | fatal to the core claim | T3 is a first-class test from M2 onward, run in nightly hardware CI |
| Multi-instance VRAM oversubscription crashes the driver | medium | medium | plan-time warnings (§9.3), per-instance budgets, documented limits; no silent overcommit |
| RAM pool + page cache pressure on smaller-RAM machines | medium | low | pool default ≤75% free RAM, mlock'd; `--ram-experts-gib`; plan reports RAM fit explicitly |
| Checksum verify slows startup (~170 GB) | certain | low | ~1–2 min amortized, `--skip-verify` loud-flagged; CRC32C keeps steady-state cost near zero |

---

## 16. References

- Product spec: `product-spec.md` (same repo).
- llama.cpp: GLM MoE support PR #27754 (open at spec-writing time); feature matrix (ROCm/Vulkan capability evidence); `ggml-backend.cpp` split-scheduler expert-copy path; `-ncmoe`/`-ot`/`--lazy-mode` docs — all retrieved via context7, 2026-09.
- Comparison projects per spec: rabbit (CPU-only disk-streamed experts), deltafin (GPU paths; ROCm path unvalidated), colibri (tiering + dual-SSD reference; GPU-hot/CPU-rest split — the exact gap Golem closes).
- context7 findings on alternatives: candle quantized path CUDA-only; mistral.rs layer-granular topology, no per-expert disk tier (§3.1, option C).
