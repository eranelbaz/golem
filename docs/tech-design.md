# Golem — Technical Design

**Status:** current design. What exists in the repo is in §14; known gaps in §17.
**Spec:** `product-spec.md` (local) — read alongside; this document resolves the questions the spec defers to technical design.

---

## 1. Decision summary

Design choices for what the product spec leaves open. Rationale in the referenced sections.

| # | Area | Choice |
|---|------|--------|
| D1 | Implementation language | **Rust** (stable, edition 2024) for everything except the vendored compute substrate |
| D2 | Inference engine | **Thin fork of llama.cpp (pinned commit) as compute/kernels substrate**, embedded in a Rust process via C ABI; Golem owns placement, paging, I/O, serving, CLI (§3) |
| D3 | AMD compute backend | **ROCm/HIP** on gfx1100 (RX 7900 XTX). HIP only; a CUDA build would reuse the same fork and kernels (§3.5) |
| D4 | Weights | **`unsloth/GLM-5.3-Flash-GGUF`; the served quant is whatever `golem.toml` (`[models.*].path`, §9) points at** — any quant of that repo may run (O7); exact repo revision + SHA-256 pinned in the model manifest (§4) |
| D5 | Caching algorithm | Decayed-frequency score (LRU×LFU hybrid) + relative swap margin + min-dwell, per (layer, expert) key, independently per tier; user limits are **caps** (`--device-cache-experts` / `--vram-experts-frac`, `--ram-experts-gib`; `golem plan --vram-experts-gib`), with no floors (O3, §6) |
| D6 | Multi-disk mode | **Full mirrored replicas**, striped across drives through one shared chunk queue (§5.2); manifest-validated; degrade-on-drive-failure (§5) |
| D7 | OpenAI API subset | Chat completions (stream + non-stream), `/v1/models`, `/health`. Tools **supported** via the pinned chat template (O2). Reasoning → `reasoning_content`. Strict-allowlist field policy (§8) |
| D8 | CLI semantics | `run` = attach to a compatible instance, else run in-process in the foreground (`--keep` leaves a server; O4); `serve` foreground by default (`--detach` supported); `list` shows instances by default, models with `--models`; `stop` = graceful terminate, `--force` for SIGKILL (§9) |
| D9 | Concurrency model | One process per instance; Rust-driven decode loop over libllama; dynamic micro-batching; tokenization + sampling via llama.cpp for bit-parity with the reference runtime (§7) |
| D10 | Disk I/O engine | Blocking reader threads doing `pread64` into an **mlock'd, GPU-pinned** RAM expert pool, 1–2 in-flight reads per drive; `O_DIRECT` where the filesystem allows it (else `DONTNEED`); blocking `pread64`, no io_uring (§5.4) |
| D11 | Instance registry | JSON manifests + PID heartbeat + Unix-domain control socket under `$XDG_RUNTIME_DIR/golem/` (§9.2) |
| D12 | Substrate patch seam | Patches to the fork's split-scheduler expert-input path (P1–P5), behind a per-model C-ABI context; policy state computed in Rust, applied in C. Not a small fixed patch set (§3.2, §3.4) |

Out of scope: performance targets (spec §Acceptance; they come from measurement).

**I2 (GPU-only expert math):** arithmetic in §2.4 puts GPU-only decode at a PCIe ceiling of ≈5 tok/s at zero VRAM hit rate, against ≈12–16 tok/s for upstream's CPU-hybrid (`-ncmoe`) at the same RAM footprint. I2 is the default, with an opt-in, decode-only CPU path (O1).

**Policy choices:**

| # | Area | Choice | Where |
|---|----------|--------|-------|
| O1 | CPU expert math for non-resident experts | **Opt-in CPU path.** A decode-only CPU expert-math path for experts not resident in VRAM (`--cpu-expert-math`, **off by default**, reported in metrics). The default keeps I2 and I1/T3. With the flag on, I1 weakens to per-cache-state determinism (CPU and GPU kernels round differently). Prefill is GPU-only. Evidence for keeping GPU-only as the default: pooled GPU-only decode 0.33–0.40 tok/s cold ≈ 2 GB/s of expert bytes against ≈25 GB/s PCIe, so the bottleneck is disk misses, not compute location. The unpooled leg thrashed mmap (172.8 GiB of experts > 93 GiB RAM), so it is not a CPU-math measurement | §1, §2.3 I2, §2.4 |
| O2 | `tools`/`tool_choice` | **Supported** via the pinned chat template | §8.2 |
| O3 | Tier limits | **Caps only**; there are no tier floors | §6.3 (also §2.2, §4.3, §6.4, §6.6, §9, §10, T3) |
| O4 | `golem run` with no running instance | **In-process, foreground**; attach if a compatible instance exists; `--keep` leaves a server | §2.1, §9 |
| O5 | Second instance whose planned VRAM/locked RAM exceeds physical | **Blocked by default**; `--allow-overcommit` | §9.3 |
| O7 | Expert quant floor | **No floor.** Any GLM-5.3-Flash quant may run; the served one comes from `golem.toml` (`[models.*].path`, §9). Measurements name the quant they ran on | §4.1, §2.4, §15 |
| O8 | Choosing between speed levers | **Decode performance first**: the lever that gives the most decode tok/s at the configured quant (golem.toml, O7) in the §12.3 A/B is the default. GPU-only (I2) is the default; O1's CPU+DMA split is opt-in. §2.4 estimates its gain at ≈+7–10% at this RAM size (20–30% of expert bytes from disk), +42–47% at 10%, and 1.7–2.5× only as the disk share approaches 0 | §2.4, §14 |
| O10 | RAM/VRAM expert residency | **Inclusive is the default**; exclusive is opt-in (`--residency exclusive`). With inclusive residency an expert in VRAM keeps its RAM pool slot, so ≈750 VRAM slots × 15.2 MB ≈ 10.6 GiB of the pool (≈17% of a 63 GiB pool) is duplicated. Exclusive residency measures −19% cold tok/s for −2.6% disk bytes, because kept evictions D2H on the refill stream | §6.3 |

---

## 2. Architecture overview

### 2.1 Process model

One OS process per running instance (`golem serve`, or `golem run` running in-process when no compatible instance exists: O4).
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
│  │ residency map + order  │◄──┤ policy state (scores,       │◄──┘       │  │
│  │ publisher (Rust)       │   │ residency bitsets, caps)    │ C ABI     │  │
│  └─────────┬──────────────┘   └─────────────────────────────┘           │  │
│            │ page requests                                              │  │
│  ┌─────────▼──────────────┐                                             │  │
│  │ store (multi-disk I/O) │── pread64, striped across replicas ──► disk │  │
│  └────────────────────────┘                                             │  │
└────────────────────────────────────────────────────────────────────────────┘
        ▲ libllama (vendored fork, HIP backend) linked into the process
```

`golem list` / `golem stop` / `golem plan` are thin clients: the first two talk to the registry (§9.2), the last is a pure offline computation (§10).

There is **no placement thread**. RAM residency transitions happen inside `ResidencyMap::claim` on the decode thread. Scores and recency are booked by the `routing` hook on the decode thread (atomics only). One `golem-order-publisher` thread rebuilds each dirty layer's eviction order off the decode thread and pushes it with `golem_policy_order_push`. `golem-placement::Policy` has no production consumer: the engine uses only its score math (`HALF_LIFE_STEPS`, `decay_factor`).

### 2.2 The three tiers

| Tier | Backed by | Managed by | Size |
|------|-----------|------------|------|
| Disk | GGUF shard files on 1..N drives, mirrored | `golem-store` | full model (~165–200 GB at the reference Q4_K_XL-class quant, §2.4; the served quant comes from `golem.toml` §9, so authoritative numbers come from the GGUF tensor table, never hardcoded) |
| RAM | one mlock'd expert pool per instance (host buffer of the vendored substrate) | `golem-placement` + store | default: min(RLIMIT_MEMLOCK, 75% of available RAM at start, total expert bytes), floored to GiB; `--ram-experts-gib` override |
| VRAM | device expert-copy cache (persistent across steps) + non-expert weights + KV + workspaces | scheduler patches (P2) under Rust policy | `--device-cache-experts` slots per layer, or derived from free VRAM × `--vram-experts-frac` after `--vram-reserve-gib` (cap only, O3) |

Everything that is **not** a routed-expert weight (embeddings, attention projections, the linear-attention/indexer tensors, norms, router, output head, and the per-layer **shared expert** `ffn_*_shexp`, which runs on every token) is loaded into VRAM at startup and stays there for the life of the instance. Only expert FFN weights participate in tiering — placement is per-expert, never per-layer (spec requirement). The MTP/NextN block (blk.45) carries its own routed experts; it is skipped at load while `load_mtp = false` and is **excluded from pool geometry**.

### 2.3 Life of a decode step

1. Slot scheduler assembles the next-token batch for all active slots (§7.2).
2. `llama_decode` builds the graph and runs it on the HIP backend. Attention, router, and all shared weights live entirely in VRAM.
3. When the graph reaches a layer's `GGML_OP_MUL_MAT_ID` (expert FFN), the scheduler's expert-copy path needs that layer's selected experts resident in VRAM:
   - **P3 hook (host residency):** before reading host bytes, call into Rust. Fast path: residency bitset hit (no-op). Slow path: the layer's misses are read through the shared striped queue (§5.2) into the RAM pool, then the call returns. With `--prefetch read` (§6.5, default `off`) predicted experts may already be resident; nothing guarantees it.
   - **P2 patch (device resident-set):** copy only the byte ranges of experts whose device copies are invalid; skip ranges already resident in the device expert-copy cache from previous steps. When a routed expert needs a slot and none is free, the first non-routed slot owner in the order Rust published is evicted.
4. Routing ids are produced by the router matmul inside the graph from activations, exactly as the reference model computes them — Golem never filters or overrides ids. After each layer's ids readback, the scheduler gathers the ids densely and fires the **`routing` hook once per changed ids tensor per step** (not per expert, which would be ≈1,000 FFI calls per token). That one call books recency, the decayed score and the layer's dirty mark. The eviction order is pushed to C as model-context-owned triple-buffered data by the order publisher thread (§2.1), never queried per tensor. Layers whose routed experts are all device-resident take no readback and fire no `routing` call.
5. Sample → detokenize → SSE frames.

**Invariant I1 (routing invariance):** caching affects only *where selected experts' bytes are staged from*. Weight bytes are bit-identical across tiers (verified by manifest hashes), the kernels and op order are identical (same vendored substrate), therefore activations, router logits, and expert selection are identical regardless of cache state. Tested explicitly (§12.2, T3).

**Invariant I2 (no CPU expert math):** expert tensors are attached to the graph as host-buffer inputs consumed by device-side `MUL_MAT_ID` — the CPU never executes an expert matmul. If the HIP backend is unavailable at startup, the instance **fails to start** with an actionable error (§8.4, `golem_backend_unavailable`); there is no CPU fallback path in the binary for expert math. `golem plan` rejects configurations that would require one. *(O1: this is the default; an opt-in, decode-only `--cpu-expert-math` exception exists. See §1.)*

### 2.4 Throughput ceilings

These are first-order bounds, computed on UD-Q4_K_XL — reference arithmetic for that quant, not the served default (§9, `golem.toml`) — on this machine. They are arithmetic, not targets: the governing numbers each design choice below must move.

- **Bytes per token.** One routed expert costs gate 4.72 + up 4.72 + down 5.77 ≈ 15.2 MB per layer. × 8 routed × 42 trunk MoE layers ≈ **5.1 GB of expert weights per token**. The GGUF carries 43 expert layers, but blk.45 is the MTP/NextN block, which the loader skips while `load_mtp = false`.
- **PCIe ceiling (GPU-only, I2).** Every expert not already in VRAM crosses PCIe 4.0 x16 (≈25 GB/s pinned, markedly less when pageable). Decode ≤ 25 / (5.1 · (1 − h)) ≈ **4.9 / (1 − h) tok/s**, where h is the VRAM hit rate. After non-expert weights (8.35 GiB), ROCm overhead (2.64 GiB), KV and workspace, about 8–11 GiB remains for experts, i.e. ≈550–750 of 12,096 trunk experts (≈5%). So h is bounded by routing skew and must be *measured* (P4 counters), not assumed. *(Measured 27 GB/s pinned, so the ceiling is ≈5.3 / (1 − h); the §2.4 split table uses 27.)*
- **Disk-miss cost.** One expert-layer from disk ≈ 15.2 MB / 5.8 GiB/s ≈ 2.4 ms (nvme0n1), ≈1.8 ms with shared-queue striping across both drives (§5.2). At 336 expert-layer loads per token, a 5% disk-miss rate adds ≈35–40 ms/token (cap ≈25 tok/s); 20% caps decode at ≈6 tok/s. Measured at QD1: the shared queue reads at **4.73 GB/s** across both drives at QD1 per drive with `O_DIRECT` (nvme0n1 alone 3.6–5.0, nvme1n1 alone 1.74). That is ≈3.2 ms per 15.2 MB unit, not 1.8, and two reader threads per drive regress (0.70 GB/s). Disk terms in this document that use 8.05 GiB/s or 8.64 GB/s are about **1.8× optimistic**. Disk time per token at RAM-hit rate r is 5.1 · (1 − r) / 4.73 GB/s: r = 0.6 → 431 ms (≤ 2.3 tok/s), r = 0.8 → 216 ms (≤ 4.6 tok/s). Per-drive queue depth is the I/O lever. Current drives (`docs/baselines/m6.3-layer-batch-io-depth.md`): fio reads 8.4 + 6.3 GB/s at QD1 and 11.9 + 7.5 GB/s at QD2 (4.8 MB reads). A layer's misses drain as one batch, and the default is two reader threads per drive: the shared queue reaches 17.6 GB/s on a 24-plane batch (14.6 at one thread per drive, 17.0 at three). The 0.70 GB/s two-thread regression does not occur with layer batching. GLM-5.3-Flash UD-IQ3_XXS decode at a 63 GiB pool runs at 6.74–6.77 tok/s with batched misses (6.55–6.65 with per-expert reads), disk wait 47–48 ms/token (49–51).
- **CPU-hybrid reference (what I2 forbids).** CPU matvec over RAM-resident experts is DDR5-bound (≈60–80 GB/s on the 9950X3D) → ≈12–16 tok/s. That is ≈2.5–3× the GPU-only PCIe ceiling at low h.
- **Prefill.** At ubatch ≥ 32 almost all 288 experts are routed per layer, so a naive per-ubatch copy moves the whole expert set (≈184 GB) over PCIe (≈7–8 s per 512-token ubatch). Prefill therefore uses **layer streaming** (§7.6): layer-by-layer expert-sorted streaming through a double-buffered device slab. Small ubatches (chunk cap 1) measure **0.24 tok/s cold**. At ubatch 2048, 16,384 routings over 288 experts leave P(expert unused) ≈ e⁻⁵⁷ ≈ 0, so each ubatch moves the full set: 644,022,272 B × 288 = **185.5 GB**. PCIe: 185.5 / 27 GB/s (measured pinned) ≈ 6.9 s. Disk: the pool holds 4,550 / 12,096 units (37.6%), so misses are ≈115.7 GB (107.8 GiB) / 8.05 GiB/s (both drives, §5.2) ≈ **13.4 s per ubatch**. Disk is the bound, and PCIe plus compute overlap it. That gives ≈150 tok/s at ubatch 2048 and ≈38 tok/s at 512, before GPU compute is measured. At the measured 4.73 GB/s the disk term is 115.7 GB / 4.73 ≈ 24.5 s per ubatch, which gives **≈85 tok/s at 2048 and ≈21 at 512**. External reference, not this machine: a 5090 on PCIe 5.0 with two NVMe drives and the same scheme reports ≈230–250 tok/s at ubatch 2048 (§16).

Consequence: I2 trades roughly 2.5–3× decode throughput for "all expert math on GPU". O1 (§1) provides an opt-in, decode-only path with this rule: "GPU when the expert is VRAM-resident or when the transfer is cheaper than CPU compute; CPU otherwise, as an explicit, configured, reported policy, never a silent fallback."

Under the O1 opt-in, RAM-resident misses are split between CPU and PCIe, not sent all to the CPU. Each layer's misses are split: VRAM hits run on the GPU, a measured fraction f of misses is DMA'd to the GPU on the copy engine (issued as soon as the ids are known), and the rest runs on the CPU, all concurrently. The best f depends on what the CPU kernel is bound by: lower when it is RAM-bandwidth-bound (the DMA competes for the same DRAM), higher when it is arithmetic-bound. With the GPU also pulling from RAM, the CPU's stream rate drops below its solo rate, so the 60–80 GB/s above is an upper bound; the arithmetic below assumes ≈41 GB/s concurrent, and it is unmeasured. Arithmetic for Golem at h ≈ 0 with all misses RAM hits: per token 5.1 GB; all-PCIe 5.1 / 27 ≈ 189 ms; all-CPU at 41 GB/s ≈ 124 ms; split f to PCIe balances at f = 27 / (27 + 41) ≈ 0.40, giving 5.1 × 0.60 / 41 ≈ 75 ms ≈ 13 tok/s, before compute and launch overhead. Splitting across tokens instead (two token groups of one verify window, one per device) reads the dense weights twice and shares fewer experts per group, so the split is across experts, not tokens. Disk misses are unchanged by any of this: they cost the same wherever the math runs.

What the split buys with disk in the loop (O8): the h ≈ 0, all-RAM-hit figure above is the best case. With a disk share d of the expert bytes, per layer and without lookahead, the RAM-hit transfer overlaps the disk read of the misses, and the misses transfer or compute after they land. That gives t = max(R / X, D / 4.73) + D / X per token, where:
- R = (1 − v − d) · 5.1 GB (RAM-hit bytes) and D = d · 5.1 GB (disk bytes), with VRAM hit v ≈ 0.10;
- X = 27 GB/s for GPU-only, or 47–68 GB/s for the split (CPU 20–41 + PCIe 27; the CPU's K-quant rate on the 9950X3D is unmeasured).

A disk-missed expert lands in the pool, so the CPU can compute it too. Results at the measured 4.73 GB/s:

| d | GPU-only (X = 27) | split (X = 47–68) | gain |
|---|---|---|---|
| 0.3 | max(113, 323) + 57 = 380 ms → 2.6 tok/s | 346–356 ms → 2.8–2.9 | +7–10% |
| 0.2 | max(132, 216) + 38 = 254 ms → 3.9 | 231–238 ms → 4.2–4.3 | +7–10% |
| 0.1 | max(151, 108) + 19 = 170 ms → 5.9 | 116–119 ms → 8.4–8.7 | +42–47% |
| 0 | 170 ms → 5.9 | 68–98 ms → 10.2–14.8 | 1.7–2.5× |

Reading the table:
- The PCIe cap on GPU-only decode is 4.59 GB / 27 GB/s = 170 ms ≈ 5.9 tok/s, whatever the cache does. The split lifts it to ≈10–15 tok/s.
- The split pays only once disk stops being the bound. At 4.73 GB/s that needs d ≤ ≈0.1.
- At a 63 GiB pool (36.5% of the expert bytes, ≈42% with exclusive residency) and moderate routing skew (assumed ≈50% of hits from ≈18% of capacity), d ≈ 0.2–0.35. There the split gains ≈+5–10%; the disk dominates.
- Lowering d (bigger pool, exclusive residency, lookahead) is the first lever. The split is the second.
- Concurrent DDR5 traffic (CPU 41 + PCIe 27 + disk DMA 5) ≈ 73 GB/s, against ≈96 GB/s theoretical on dual-channel DDR5-6000, so the CPU rate *under* concurrent DMA is unmeasured.

### 2.5 The model's actual architecture

GLM-5.3-Flash (`glm5next`) is a **hybrid**:

- 45 trunk layers (`config.json` `layer_types`): **34 `linear_attention`** layers with a fixed-size recurrent state, and **11 `deepseek_sparse_attention`** layers with MLA-style K-only KV plus a sparse-attention indexer (`indexer.*` tensors). Layers 0–2 are dense, layers 3–44 are MoE (288 routed + 1 shared, top-8), and blk.45 is the MTP/NextN block.
- Memory per sequence is **constant-size recurrent state** (145.56 MiB R/S state) **plus** KV for the 11 attention layers only (19,712 B/token measured at f16; 14,432 B/token at q8_0, §10).
- Recurrent state **cannot be partially rewound** (`llama-memory-recurrent.cpp:182`). Anything that assumes "truncate the KV to position p" (slot cancel with rollback, stop-sequence rollback, prefix reuse) must instead **snapshot and restore** state (§7.1).
- Long-context cost is governed by the sparse indexer, not dense attention, so the §12.3 ctx sweep must be chosen from measurement.

Every "MLA" reference below means the 11 sparse-attention layers only.

**DeepSeek-V4.1-Flash (`deepseek41`).** A second supported architecture: 40 blocks, n_embd 5120, 64 heads MLA (kv 512, q_lora 1280), 384 routed experts top-6 + 1 shared (expert FFN 2304, sqrt-softplus gating), SWA 128 with per-layer compressed attention plus a 32-head indexer (top_k 512), 4-way hyper-connections, and an engram table (`per_layer_token_embd`, ≈ 126 GiB) read lazily from disk and never placed in VRAM. Its KV is priced per cache by `golem-plan` (`dsv41_state_terms`): raw SWA rows `pad256(n_swa + n_ubatch)` plus compressed rows `n_ctx / ratio` per layer; target at q8_0 ≈ 1,700 B/token. Expert planes are mixed-quant (IQ2/MXFP4); the stream slabs carry one stride of slack for HIP's 512-element row padding (expert ne0 2304 → 2560). P5 prediction nodes mirror its router (sqrt-softplus + `exp_probs_b`). Unmeasured: the `ws` compute-buffer fit (fitted on GLM), cold/warm throughput, snapshot restore (diverges on hybrids at this pin; serve with snapshots off).

---

## 3. Compute substrate

### 3.1 Substrate choice

Golem is a **thin fork of llama.cpp with Rust orchestration**. The HIP backend has full K-quant/MoE/FlashAttention support (verified feature matrix, §3.6); the split scheduler already implements just-in-time per-expert RAM→VRAM copies of only the routed experts (verified in `ggml-backend.cpp`); acceptance-criterion parity is nearly free because the compute substrate *is* the reference runtime; and GLM arch fixes arrive as upstream merges the fork rebases onto. The disk tier, cache policy and multi-disk I/O stay in Rust: Rust is the product, the fork is a substrate.

### 3.2 The fork

- Upstream: ggml-org/llama.cpp PR #30817 head `b72050d` (base `f174553`), which adds `deepseek41` and upstream `glm5-next` (the unsloth `glm5next` name is folded onto it). `engine/llama-cpp.lock` is authoritative for the pin. Its fields are: `repo`, `branch`, `commit`, `pr`, `base_upstream` (merge-base at pin), `content_sha256` (recomputable tree hash, enforced by the pin gate in `crates/golem-llama-sys/pin_gate.rs` and re-derived by CI; the regular-file count lives in `vendor/VENDOR_SHA`), `rocm` (advisory minimum; the check warns on `rocminfo` mismatch and is skipped when `rocminfo` is absent), `patches`, and `fa` (the FlashAttention/precision setting for MLA on ROCm, §3.5).
- The vendored tree carries no nested `.git` by design, so acceptance is content-based, not `git rev-parse`: CI's `vendor-pin-check` recomputes `content_sha256` from `vendor/llama.cpp` (recipe in `scripts/vendor-rehash.sh` and `pin_gate.rs`) and it must match `vendor/VENDOR_SHA` and the lock.
- The fork is vendored in-tree (`vendor/llama.cpp/`) and built by the `golem-llama-sys` build script via the `cmake` crate with `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100`.
- Patch discipline: the delta vs upstream is confined to the files listed in §3.4, with no fixed line budget: the P2/P3 design (pool as H2D source, slot-indirected device cache, per-model hook context, a post-router P5 hook, prefill streaming) reworks how the split scheduler handles expert inputs, and `ggml-backend.cpp`'s scheduler is one of upstream's most-changed files. The delta is isolated behind one upstream-shaped extension point (a custom buffer type plus a `MUL_MAT_ID` input resolver). Every patch has an in-tree regression test (§12.2, T4). Upstream rebase is scheduled maintenance (monthly + on GLM-arch fixes), never ad-hoc drift.
- The fork carries **no** model-architecture code of its own and **no** product logic in C++ beyond the hooks.

### 3.3 What we use from the substrate

| Concern | Source |
|---------|--------|
| GGUF parsing, tensor loading, mmap | llama.cpp loader, driven through `llama.h` |
| Model graph (glm5next text path), attention, RoPE, norms, router | llama.cpp |
| KV cache (allocation, quantized-KV options, defrag) | llama.cpp (`--cache-type-k/--cache-type-v`, config `cache_type_k`/`cache_type_v`, default f16) |
| K-quant GEMV/GEMM kernels on HIP | ggml HIP backend |
| Tokenization (BPE + GGUF chat template) | llama.cpp — *not* a Rust tokenizer; byte-parity with the reference runtime matters more than library purity |
| Sampling | llama.cpp sampler chain, driven by `llama.h` with an explicit seed — reproducibility is an acceptance requirement (§12.2) |
| Continuous batching / HTTP server / placement logic | **not** the substrate's — Golem replaces `llama-server` entirely; we drive `llama_decode` ourselves (§7) |

### 3.4 Substrate patches (the entire C++ delta)

**P1 — hook plumbing.** `ggml/src/ggml-backend.cpp` + a new `ggml/include/golem-hooks.h`: registers C function pointers per `golem_ctx` (`ensure_host_resident`, `routing`, `copy_done`, `evict_batch`, `evict_abandon`, `stream_source`, `stream_release`, plus `predict` and `cpu_lease` through their own setters; hook ABI version 22, `golem_hooks_abi_version()`) and per-tensor expert metadata (expert count, per-expert byte size, stride). Expert metadata is queried lazily at the first callback. The name match accepts the runtime tensor names `blk.N.ffn_{gate,up,down}_exps.weight`; the fused `ffn_gate_up_exps` is deliberately excluded. Every hook takes a `golem_ctx*` owned by one `llama_model`; registration returns a handle, and unregister waits for in-flight callbacks. There is no process-global hook state, device-cache map, budget or stats. No behavior change when hooks are unset (upstream-semantics default), which keeps upstream tests green.

**P2 — slot-indirected device expert cache.** A full-size device twin per expert tensor would total ≈184 GB on this model and OOM a 24 GB card, so the device holds a slab of `K` expert slots per (layer, kind), sized from the VRAM budget. The slot table is one per layer, shared by gate, up and down (the three tensors consume the same `selected_experts` ids per step); each tensor keeps its own per-slot validity bits and slab, and an eviction clears the slot's bits in all three. A host-side `expert → slot` map feeds `MUL_MAT_ID` the slab as `as` (`ne[2] = K`) plus a remapped 2-D ids tensor (values in `[0, K)`). This also replaces upstream's full-size transient `input_cpy` for expert inputs. The budget is a per-(layer, kind) slot count, not a byte cap derived from one tensor's stride. Only invalid slots of the routed set are copied (the split-scheduler path that groups consecutive routed experts and calls `ggml_backend_tensor_set_async`); when slots are exhausted, eviction follows the per-layer order most recently pushed by Rust (`golem_policy_order_push`), applied in C with no callback latency in the hot loop. Already-copied hot experts are skipped entirely — this is the mechanism that makes "hot experts live in VRAM" real without touching graph construction.

**P2 prefill-stream mode.** When a ubatch is above the streaming threshold (§7.6), the expert input for each plane comes from a **stream slab** instead: the same slot-slab machinery with `K = n_expert` and identity ids, so no ids remap is needed. Two stream slabs, each sized for the largest plane of any class, are ping-ponged: plane *p* of layer *L* computes while plane *p+1* (or layer *L+1*'s gate) fills over H2D on a separate stream, event-synchronized. Slot-cache slabs are untouched during prefill, so the decode hot set survives a long prompt.

**P3 — host residency callback.** The host expert bytes live in a read-only file mmap and a writable host buffer of model size cannot fit in RAM, so **the RAM pool is the H2D source**: `ensure_host_resident` returns a pool pointer per expert, `copy_experts_to` copies from it, and the lease is held until the async copy completes (the `copy_done` hook, after the scheduler's event sync). Expert tensors are never populated from the file by the loader: no `MAP_POPULATE` of expert ranges (the fork uses the lazy-tensor path, `set_tensor_data` loading, or a zero-backing buffer type). The callback runs on the decode thread and blocks until ranges are paged into the mlock'd RAM pool (bitset fast path; synchronous striped pread slow path); §6.5 explains why the blocking slow path is rare.

*Layer-batched cold reads.* P3 takes **one call per (layer, step)** with the layer's full miss list; all of a layer's misses are known before the first read. Rust submits every plane of every missing expert to the §5.2 shared queue at once and returns when all have landed (per-expert leases unchanged). Per-expert calls would make a cold layer up to 8 × 3 = 24 blocking reads of 4.7–6.9 MB at queue depth 1 while the GPU idles and the prefetcher stands down (demand gate). Arithmetic: a fully cold layer is 8 × 15.2 MB ≈ 122 MB; at the ≈2 GB/s measured for the serial path that is ≈61 ms, at both drives' 8.05 GiB/s ≈14 ms. At zero RAM hits a token is 5.1 GB / 8.64 GB/s ≈ 0.59 s ≈ 1.7 tok/s ceiling, against the 0.33–0.40 tok/s measured with the serial path. BMoE (§16) got ≈3× from parallel read lanes on phone storage.

*Role-major page-in (opt-in, `--role-major-pagein`, default off).* Split rule: within one `ensure_host_resident` call, a disk-missed unit whose miss list does not name its down plane (the current split segment needs only gate/up) is split, up to `min(K, n_experts)` units per layer: gate/up are read into the slot and handed to the H2D at once, and the slot's down region `[region[0] + region[1], slot end)` goes to one `golem-down-reader` thread as a window, read in the background (counted in `split_units`). Every other miss pages in single-phase. Single-owner settle: the split consumes the unit's fill claim; the claim is reachable again only by rejoining the window, so no settle can recycle a slot the reader is still writing, and a failed down read poisons the unit (cancelled at the step end, never published). Join points: a later miss naming a split unit, a new split on the same layer, the step end (`copy_done`, before any lease drain or D2H commit), and the idle settle (failed decode, device-cache reset, teardown). Bounded wait: one deadline (120 s, the device-wait bound) is shared by every layer a join waits on; outliving it fails with `DownStalled`, which the engine treats as fatal (`p3_fatal`) rather than hang. Memory: per (layer, unit) 48 B job window + 40 B split claim + 24 B poisoned-claim slot = 112 B, sized once at enable; GLM-5.3-Flash 42 layers × min(K, 288) × 112 B ≤ 1.35 MB. Off: no thread, no state.

**P4 — counters.** Per-step counters (bytes copied host→device, bytes skipped because resident, page-ins served from RAM vs disk, evictions) exposed through a lock-free stats block read by Rust; exported as metrics (§11).

**P5 — post-router hook.** A hook per MoE layer after routing, for next-layer routing prediction (§6.5).

**Patch set at the pin.** `engine/llama-cpp.lock` `patches =` names every patch in the vendored tree; each is tied to the hook ABI version it introduced:
- `P1`, `P2`, `P3`: hook plumbing, slot-indirected device cache, host residency callback (above).
- `P3copy`: `golem_copy_pooled_expert_run`, the pooled-run H2D copy plan (a padding tail is the last expert's own bytes, never past the slot).
- `P1place`: `golem_get_expert_placement_stats`, per-backend expert `MUL_MAT_ID` counters at split, the I2 proof.
- `P1ctx`: the per-model `golem_ctx` hook context, batched ensure and `routing` (v5).
- `LLctx`: `llama_golem_model_attach/detach`, the libllama seam that sets the ctx on the model and re-applies it per `sched_reserve`.
- `P5pred`: `golem_pred-<target>-<src>` prediction nodes (glm5-next, deepseek41) and the `predict` hook (v6).
- `LLarch`: unsloth `glm5next` files fold onto upstream `glm5-next`; lazy expert planes (`TENSOR_READ_LAZY`) in both arches.
- `O10excl`: `evict_batch` hook; exclusive-residency slab evictions D2H into the pool (v7, batched per (layer, compute) in v8).
- `P2slabfail`: a device-slab allocation failure fails the compute (`llama_decode` −2) instead of aborting; the engine resets and lowers K (v12).
- `P6abort`: a prefill-stream source returning cancel aborts the compute (v11).
- `P3ensurefail`: an ensure hook returning nonzero fails the compute instead of aborting (v13).
- `P4resident`: `golem_device_cache_resident_bytes`, live slab plus stream-slab device bytes (v14).
- `P6srclost`: a stream source returning `GOLEM_STREAM_SRC_STORAGE_LOST` fails the compute (v15).
- `TSgate`: test seams compiled only under `GOLEM_TEST_SUPPORT` (v15).
- `P2h2dstats`: device-cache stats count H2D expert copies (v15).
- `P3copystream`: `golem_set_async_h2d`/`golem_take_h2d_timing`, pooled and slab H2D on a dedicated HIP copy stream, event-joined with compute (v15).
- `O10ring`: `golem_set_victim_ring`, a VRAM victim ring that moves exclusive-residency eviction D2H off the refill path onto a dedicated D2H stream (v18).
- `P11syncgap`: `golem_take_sync_gap_stats`, per-step host time blocked in the ids-readback sync and the gap to the next split launch (v19).
- `P12devwait`: `golem_set_device_wait` and friends; the decode path publishes routed ids to a mapped ring and waits on the device (`hipStreamWaitValue32` or a wait kernel) instead of the per-layer ids readback (v20).
- `O10ringfail`: `golem_take_victim_ring_stats` returns −1 when a setup failure disabled the ring; `golem_set_victim_ring` refuses at load without a copy-stream backend (v21).
- `P13tiers`: `golem_device_cache_layer_resident`, a lock-free per-layer count of slab slots with all three planes valid (v22).

The ABI also carries the abandoned-D2H hook `evict_abandon` (v9), the prefill-stream hooks (v10), the opt-in CPU expert lease (v16) and the ensure `hits_ready` early-H2D callback (v17).

**Engine-side invariant.** The engine forces every expert `MUL_MAT_ID` onto the GPU (upstream keeps it on the CPU below batch 32) and verifies after the first decode that it ran there. Failure → `golem_backend_unavailable`.

### 3.5 Backend: ROCm/HIP, and what "not NVIDIA" means

- Target: **gfx1100 (RX 7900 XTX)**, ROCm **7.2.4** as the advisory minimum recorded in `engine/llama-cpp.lock` (a `rocminfo` vs lock mismatch warns; the check is skipped when `rocminfo` is absent, e.g. CI/no-GPU), HIP backend of the fork with `AMDGPU_TARGETS=gfx1100`. ROCm/HIP is the right choice for gfx1100: its K-quant MMQ/MMVQ kernels are mature.
- The known NVIDIA-specific workarounds from upstream's GLM MoE support PR (`ggml-org/llama.cpp#27754`: `NVIDIA_TF32_OVERRIDE=0`, `-fa off` for MLA) are CUDA-specific. Their ROCm analogues (hipBLASLt reduced-precision options, FlashAttention on MLA) are validated with correctness evidence, and the validated setting is written into `engine/llama-cpp.lock` and is the default; `--flash-attn` forces FlashAttention on; without it llama.cpp's AUTO default applies.
- **AMD-specific requirements:**
  - **Pinned host memory.** The RAM expert pool is mlock'd *and* registered with `hipHostRegister` (portable). Pageable H2D goes through driver bounce buffers, runs at roughly half of PCIe bandwidth and is not truly async, and PCIe is the §2.4 bottleneck. Pinned vs pageable H2D GB/s on the 7900 XTX are recorded in the baselines. `MADV_HUGEPAGE` is not applied to the arena.
  - **HIP graphs.** `GGML_HIP_GRAPHS` defaults ON. A blocking host callback inside the scheduler (P3) must never run inside a captured region: either verified, or graph capture is disabled for split graphs.
  - **Op-offload threshold.** Upstream keeps `MUL_MAT_ID` with host-resident weights on the CPU backend below batch 32 (`GGML_OP_OFFLOAD_MIN_BATCH`). The engine (not tests) forces device placement and **verifies at runtime** that every expert `MUL_MAT_ID` ran on the GPU. Otherwise I2 is silently violated.
  - Backend-specific pieces (pinned-memory registration, the P2 slot remap) stay behind the P-hook seam.

### 3.6 Evidence base

Substrate claims cite the **vendored tree** (`rg` in `vendor/llama.cpp`, file:line), not doc summaries.

- llama.cpp feature matrix: ROCm column — K-quants ✅, I-quants ✅, MoE ✅, Flash Attention ✅, K-cache quants ✅; Vulkan K-quants marked slow ("the CUDA or ROCM backends are faster"). This is the empirical basis for D3.
- `ggml-backend.cpp` split scheduler: when MoE weights are offloaded to a host buffer, it reads the ids tensor, builds a used-expert bitset, groups consecutive ids, and copies **only those ranges** host→device per eval (`ggml_backend_tensor_set_async`), including small padding guards for MMQ. This is upstream's existing just-in-time cold-expert-to-GPU path; Golem's P2/P3 wrap policy around it.
- `--n-cpu-moe` / `-ot exps=CPU` are shipped, exercised configurations that put expert weights in host buffers while the graph runs on GPU — i.e., the "cold expert computes on GPU" configuration is upstream-supported and debugged, not speculative.
- `--lazy-mode` (on-demand reads of selected tensors from disk) exists upstream for any tensor an arch marks `TENSOR_READ_LAZY` (`llama-model-loader.cpp:1335`), including experts, and rides the kernel page cache. No vendored arch sets it on experts. It is relevant to P3's load-mode (lazy-tensor path). Golem's explicit RAM pool (§5.4) is used for experts instead: bounded memory, multi-disk striping, and accounting — three things page cache cannot give us.

---

## 4. Model loading and GGUF handling

### 4.1 Weights (D4)

- Served quant: whatever `golem.toml` (`[models.*].path`, §9) points at. The quants served today (§9): GLM-5.3-Flash at UD-IQ3_XXS, DeepSeek-V4.1-Flash at UD-IQ2_XXS. Any quant of the repo may run; there is no default quant (O7). The exact repo revision and per-shard SHA-256s are pinned into the model manifest (the repo carries several non-interchangeable variants; the manifest makes the choice auditable and `golem plan` exact). Sizing numbers elsewhere in this doc name the quant they were measured on.
- Reference arithmetic only: the capacity/quality tradeoff notes below were computed on **UD-Q4_K_XL** (≈4.5–5 bpw keeps the resident-set calculus meaningful). It is not a default.
- Auxiliary for validation: **Q8_0** of the same revision — used by the parity harness (§12.2) because high-precision quant minimizes quantization noise when bisecting correctness issues across GPU/CPU execution.
- **Quantization is the largest single performance lever.** Bytes per token, and with them the §2.4 PCIe ceiling, the disk-miss cost and the number of experts that fit in RAM/VRAM, scale linearly with expert bits. A ~3-bit expert quant (UD-Q3_K_XL / IQ3_XXS class) cuts ≈30–35% of expert bytes, raises the PCIe ceiling proportionally, and fits ≈50% more experts per tier. Parity is judged against "the same quantized model", so a lower quant does not break acceptance. Its quality cost is gated separately: perplexity delta on the wikitext slice. `golem plan` therefore treats the quant as a first-class input and reports the §2.4 ceilings per candidate quant. Any quant may run; the served one comes from `golem.toml` (`[models.*].path`, §9) (O7).

### 4.2 Manifest

A model's manifest lives at `$GOLEM_DATA_DIR/models/<name>/manifest.json`:

```jsonc
{
  "name": "glm-5.3-flash-<quant>",
  "repo": "unsloth/GLM-5.3-Flash-GGUF",
  "revision": "<hf commit sha>",
  "quant": "<quant per golem.toml [models.*].path, e.g. UD-IQ3_XXS>",
  "arch": "glm5next",
  "shards": [ { "file": "glm-5.3-flash-<quant>-00001-of-000xx.gguf",
                "bytes": 0, "sha256": "<...>",
                "replicas": ["/mnt/ssd0/golem/models/...", "/mnt/ssd1/golem/models/..."],
                "sampled_sha256": ["<16 window hashes, optional>"], "mtime": 0 } ],
  "expert_index": [ { "tensor": "blk.3.ffn_gate_exps.weight", "layer": 3, "experts": 288,
                      "bytes_per_expert": 0, "offsets_bytes": [/* per-expert */],
                      "crc32c": [/* per-expert, optional */] } ],
  "chat_template_sha": "<...>"
}
```

The struct is `golem_store::ModelManifest`. `name` and `shards` are required; `repo`, `revision`, `revision_source`, `quant`, `arch`, `expert_index` and `chat_template_sha` are optional pass-throughs. Per shard, `sampled_sha256` and `mtime` are optional; per `expert_index` entry, `crc32c` is optional and must have `experts` entries. `layer` is the GGUF `blk.N` index, so GLM-5.3-Flash entries run from blk 3. `expert_index` is produced by walking the GGUF tensor table (`scripts/manifest-builder.py`; expert tensors match the upstream MoE regex `ffn_(up|down|gate|gate_up)_(ch|)exps`, and `tensor` is the full runtime name). It is required by the store-backed source (`StoreExpertSource::open` fails without it), which re-checks every entry against each replica's GGUF tensor table. Vision-tower (`mmproj*`) tensors and files must not appear in a manifest (text path only); this is not enforced by any tool — the spec bans silent anything.

### 4.3 Load-time checks (fail fast, fail loud)

1. Manifest integrity. Full SHA-256 verification happens only with `--full-verify` (≈32 s of pure read at 5.8 GiB/s plus ≈90 s single-core SHA, reading 186 GiB off the SSD and flushing the page cache). At instance start, check size, plus mtime and the sampled per-shard hash where the manifest records them: 16 × 1 MiB windows at `i·(len − 1 MiB)/15`, the last ending at EOF, with pages dropped after each read (`golem_store::digest::sampled_sha256`). A shard without `sampled_sha256` is checked by size and mtime only, with a warning. Also check the expert layer set against `block_count − nextn_predict_layers`, so the MTP block is never counted as a trunk layer.
2. Expert layer coverage and count: every layer from the dense lead through `block_count − nextn_predict_layers − 1` bears gate/up/down expert tensors (the MTP/NextN block is excluded), and per-layer expert count and bytes-per-expert are uniform and non-zero (GGUF guarantee; a violation means a corrupted/mismatched mirror — refuse to start).
3. VRAM budget feasibility: non-expert bytes (incl. shared experts) + KV (11 attention layers) + recurrent state × parallel + workspace + VRAM expert-cache budget ≤ free VRAM minus measured ROCm overhead. Failure → `golem_plan_infeasible` error with the exact over-budget line items (§10), never a degraded start.
4. Backend probe: `hipGetDeviceProperties` on gfx1100, kernel self-test (one expert matmul GPU vs known-good CPU result within tolerance). Failure → `golem_backend_unavailable` with install instructions (ROCm version, env vars) in the error body.

---

## 5. Storage tier (`golem-store`)

### 5.1 Mode (D6): full mirrored replicas

The store supports **N ≥ 1 full copies** of the model on separate paths (the model bytes are identical; the manifest's `replicas` array names them). Mirrored full copies — not sharded placement — because: (a) per-read striping across replicas already extracts aggregate bandwidth, so sharding adds a failure mode without extra bandwidth; (b) replicas double as validation targets for each other.

**Same-physical-drive detection:** two replica paths whose resolved `st_dev` maps to partitions of the same parent block device (resolved by walking `/sys/block/<parent>/<partition>`) are flagged at startup: `warn: replicas share a physical device; no bandwidth aggregation`. They still function (fault-tolerance is zero there, and we say so), we just don't advertise aggregate bandwidth or prefer them for striping.

### 5.2 Read path

- Unit of I/O: one expert's byte range within one tensor (from `expert_index`), typically 1–8 MB per expert per tensor per layer.
- **Shared-queue striping.** Every outstanding read (demand, prefetch, prefill stream) is cut into chunks of at most **8 MiB** (`golem_store::stripe::CHUNK`) on a grid anchored at the 4 KiB-aligned read start, so every chunk meets the `O_DIRECT` offset and length rule (§5.4) and a 4.7–6.9 MB expert plane is one `pread`. The chunks go into **one queue shared by all replicas**; demand chunks drain strictly before prefetch chunks. Each drive runs its reader threads (§5.4), and each thread pulls the next chunk into its destination region. The faster drive takes more chunks without any bandwidth probe, EWMA or per-read plan, because chunks are pulled rather than pre-split; single-chunk drains rotate their first drive so they alternate drives. Under layer batching a cold layer is up to 24 plane chunks, and the tail costs at most one slow-drive chunk (6.9 MB / 2.24 GiB/s ≈ 2.9 ms) per queue drain, not per unit. Under prefill streaming the misses of one ubatch are ≈(12,096 − 4,550) units × 3 planes ≈ 22,600 chunks. Completion is when all of a read's chunks land. Chunk boundaries are deterministic (offset arithmetic), so a retry on another replica is trivially resumable. A chunk that fails on replica X is requeued onto the same drain with X set in its exclude mask, so a healthy drive's thread retries it within the same wait. A degraded drive self-throttles because it simply pulls fewer chunks, so health needs only a per-replica error count. CRCs are per expert plane, verified after all of a plane's chunks land (§5.3).
- **Persistent drain readers.** Each queue owns up to 3 crews of parked reader threads instead of spawning per drain. A drain pops an idle crew, grows it to `drives × io_depth` (or `--io-threads N`) threads if needed, hands it the drain task and blocks until every worker has released it; once warm, a drain spawns nothing and allocates nothing. Each crew thread pins to the engine cpuset once at spawn. Bound on the target (2 drives, depth 2): 3 × 2 × 2 = 12 threads, 12 × 2 MiB = 24 MiB reserved virtual stack (resident is touched pages only); with `--io-threads N`, 3 × N. A 4th overlapping drain runs serially on its caller, unpinned, and is counted (`serial_fallbacks`). A worker panic aborts its drain and resumes on the caller after all workers check out.
- Reads land directly in pre-pinned slots of the mlock'd RAM pool (no bounce buffers), except prefill-stream misses, which land in the pinned staging ring (§7.6); completion publishes the unit in the residency map and releases the waiting `claim`.
- Errors on one replica (EIO, device gone): mark replica degraded (health state, metric, log), transparently retry the failed chunk on the next healthy replica. If **all** replicas of a shard are unreadable: instance transitions to `degraded:fatal-io` — in-flight requests complete with an error, new requests get `golem_storage_unavailable`; the process stays alive so `golem stop`/`golem list` keep working and the user can pull the drive and recover. (Fail-soft per spec "degrade or fail": we degrade until reads are impossible, then fail loudly but stay manageable.)

### 5.3 Validation

- Full SHA-256 verify at instance start when `--full-verify` is set (§4.3). Sizes re-checked on open.
- Each pool page-in is covered by a cheap **CRC32C** stored in the page table entry, verified **at page-in** (verifying on use would mean hashing ≈5 GB per token on the decode thread). CRCs are per fixed unit, i.e. per expert plane, and are stored in the manifest `expert_index`; no tool generates them, and a manifest without them skips page-in CRC. They are never per striped chunk. Mismatch → treat as corrupted replica (same path as EIO) and log the offending shard+range. This catches silent rot between full verifies at ~3 GB/s/core cost, negligible against disk reads.
- On any detected mismatch, `golem list` shows the instance as `degraded`, naming the shard and replica — actionable, not silent.

### 5.4 I/O engine (D10)

- **Default:** dedicated reader threads doing `pread64` on `O_RDONLY` fds, one per replica file. One thread measured 5.81 GiB/s (nvme0n1) while 8 threads regressed to ≈2.0 GiB/s on both drives, so the default is **2** reader threads per drive with `O_DIRECT` (`golem_store::DEFAULT_IO_DEPTH`, `--io-depth`), with `--io-threads` as an override. Rationale: expert reads are large (MB-scale), few, and schedulable; thread-pool pread is within ~10% of io_uring at this op size, and it keeps the decode-thread slow path (§6.5) simple and bounded.
- The RAM expert pool is one `mlock`'d mapping (`mmap(MAP_ANONYMOUS | MAP_NORESERVE)` + `mlock`) carved into expert slots. mlock prevents the kernel from double-evicting what our own cache thinks is resident — the class of bug that produces nondeterministic "disk reads" inside a supposedly-warm cache.
- Host page cache: expert reads must not populate it (memory pressure was observed on the target, with desktop freezes). Use `O_DIRECT` straight into the (4 KiB-aligned) pool slot where the filesystem supports it, probed per replica at open, since btrfs-zstd `/home` returns EIO under concurrent `O_DIRECT`. Otherwise, and after the first direct-read failure on a replica, read buffered and `posix_fadvise(DONTNEED)` the range after each read. `--o-direct` (needs a manifest) makes direct reads mandatory: a replica without a direct handle, or a failed direct read, fails the read instead of falling back. The loader does not `MAP_POPULATE` expert tensors either.
- Reads use blocking `pread64`, not io_uring: thread-pool pread is within ~10% at MB sizes, and `tokio-uring` is effectively unmaintained.
- Slots are not one uniform size. There is one slab per plane-stride class, because max-stride slots would waste ≈12%. There is no static per-layer quota: one global pool with a global CLOCK, and per-layer locks for residency bits only. The **unit of residency is (layer, expert)**: its gate, up and down planes are claimed, filled, leased and evicted together, in RAM and in VRAM.
- The arena is **one mlock'd mapping carved into per-class regions** (class = distinct unit byte size), with one free list and one CLOCK hand per class under the existing two locks. Per-class budget is proportional to the class's byte share, computed as `⌊pool × layers_c / Σ unit bytes⌋`. Target arithmetic (GLM-5.3-Flash UD-Q4_K_XL, blk 3–44 live, blk 45 MTP skipped): std 39 layers @ 15,204,352 B/unit, q6d 2 @ 16,318,464, q5q6 1 @ 18,415,616; Σ = 644,022,272 B. 65 GiB → std 4,226, q6d 216, q5q6 108 = **4,550 units (+34.6% vs max-stride slots' 3,380)**. Per-class floor: one cold step leases `layers_c × top-k` units (std 312, q6d 16, q5q6 8 — all fit). The derived device slot count K is priced by Σ unit bytes → K = 9.

### 5.5 Import

There is no import path and no `golem import`. Golem reads the customer's GGUF shards in place (obtained however they like — `hf download`, another machine, purchased copy) and never talks to Hugging Face. A manifest (§4.2) is optional: without `manifest.json` next to the model, the engine reads the GGUF directly; with one, it is written by hand (`sampled_sha256`, `mtime` and per-plane CRCs are optional; absent ones are not checked) and §5.1–§5.4 apply to its replica set. Multi-replica striping is not the default path: the real-target A/B (GLM-5.3-Flash UD-IQ3_XXS, 120.4 GB) measured striping nvme1n1 + 990 PRO at 5.4–6.6 GB/s vs 7.2–8.0 GB/s for nvme1n1 alone; static round-robin homing caps the pair at the harmonic mean 2 / (1/7.85 + 1/5.97) = 6.78 GB/s, so a second mirror buys failover only, not bandwidth.

---
## 6. Expert placement (`golem-placement`)

### 6.1 Identity and state

Key: `(layer, expert_id)` — model-agnostic, works for any MoE arch the substrate loads. It is also the unit of paging in every tier: all three planes of an expert move together.

Per key, state is split by tier:
- **RAM** (`golem-placement::pool::ResidencyMap`): the key → slot map, lease counts and a CLOCK reference bit, under one lock per layer shard. Prefetch-filled and prediction-protected bits sit beside them.
- **VRAM**: the substrate's per-layer slot table (`golem-device-cache.hpp`: `slot_of`, `owner_of`, per-kind validity bits). Rust never holds VRAM residency.
- **Policy** (`p3_glue`): per key, the last-access step and the §6.2 score packed with the step it was booked at, in one `AtomicU64`.

There is no per-key CRC: CRCs live per plane in the manifest (§5.3). Every table is a flat `Vec` indexed by `layer * E + e`, with no heap allocation per step.

### 6.2 Score (D5)

```
score(k, now) = score(k, t) · e^(−(now − t)/H)   decayed lazily to the reader's step (H = HALF_LIFE_STEPS = 512)
score(k) ← score(k, now) + 1                    on access, booked by the `routing` hook (§2.3)
```

This is exponential-decay frequency: a blend of LRU (recency dominates for small H) and LFU (accumulated use dominates for large H), tunable by one parameter. No per-step sweep runs: each key stores its score with the step it was booked at, and readers decay it on read (`golem_placement::decay_factor`). With H = 512 a "hot set" is an expert accessed more than about once per H steps.

- **Step.** One step = one `llama_decode` call on the trunk, counted in Rust (Rust drives decode). Prefill chunks are **excluded** from scoring, because a prefill ubatch routes nearly every expert and would flatten all scores.
- **Scale.** An average expert's steady-state score is p / (1 − e^(−1/H)) with p = 8/288, i.e. ≈14.2 at H = 512. Absolute thresholds therefore sit an order of magnitude below the typical score, which is why §6.3 promotes by relative margin.

### 6.3 Transitions

- **disk → RAM** (page-in): on demand via the P3 slow path, or speculatively by the prefetcher (§6.5). `ResidencyMap::claim` takes a free slot of the unit's size class, else runs that class's CLOCK hand: a key with its reference bit set (set on every lease grant) gets its bit cleared and a second chance; the first unleased, unprotected key without the bit is the victim. A prefetch claim never evicts a key whose reference bit is set, and returns busy instead. Under exclusive residency the published scores add a floor: keys scoring above the incoming unit are shielded. There is no min-dwell in the RAM tier.
- **RAM → VRAM** (admission): on demand only. When a routed expert has no slab slot, P2 takes a free slot, else evicts the first non-routed slot owner in the published order and copies the expert's planes from its pool slot. The order is ascending §6.2 score (`--vram-order score`, default) or least recently routed first (`--vram-order recency`). No relative swap margin or min-dwell is applied at admission: a routed miss always takes a slot. `golem_placement::Policy` implements the margin and dwell rule (`SWAP_MARGIN = 1.2`, `min_dwell_steps = 256`), but no production path calls it.
- **VRAM → RAM** (eviction): P2 evicts by the published order when a routed miss needs a slot; under inclusive residency the pool copy is untouched, so re-admission is an H2D from the pool if the unit is still resident there, else a page-in first.
- **Limits = caps.** `--device-cache-experts` (or `--vram-experts-frac`) / `--ram-experts-gib` cap each tier's expert cache; within the cap, the scorer decides residency. There are no per-tier floors: a floor that pins "the current top-N by score" pins what the cache already keeps, and it only changes behavior when it overrides the scorer, which lowers the hit rate. The per-expert placement limits are met by the caps.
- **No background promotion.** Every VRAM admission is a demand copy inside the step that routes the expert (above); there is no score-driven promotion stream. Replay showed background promotion lowers the VRAM hit rate (44.2 → 43.9% at 8/step).
- **Residency is inclusive by default.** P2 fills a VRAM slot from the expert's pool slot, and an inclusive VRAM eviction does no D2H (`golem-device-cache.hpp`). So every VRAM-resident expert also holds a RAM slot: ≈750 × 15.2 MB ≈ 10.6 GiB, ≈17% of a 63 GiB pool. The disk → RAM → VRAM staging order holds.
- **Exclusive residency (`--residency exclusive`, `evict_batch` hook, Rust `evict_dest_batch`).** Opt-in:
  - Admission to VRAM releases the expert's pool slot once its H2D copy lands.
  - Only fully plane-valid slots are reported on eviction, one `evict_batch` call per (layer, compute). Rust either returns pool destinations for the three planes (`keep`), and C issues one D2H per plane, or drops the bytes when the pool's victim scores higher. The returned pool slots stay gated until the matching `copy_done`, after the substrate has synchronized its backends. An evicted expert is never re-read from disk while a VRAM copy exists.
  - Stream: without the victim ring, the D2H runs on the split backend's stream ahead of the H2D refills of the stolen slots, so stream order lands the D2H first. With `--victim-ring N` (default 0, off), evicted planes go to an N-slot VRAM victim ring and drain to the pool on a dedicated D2H stream, off the refill path. `evict_abandon` releases pool destinations for D2H that were never issued.
  - Cost: one D2H per VRAM eviction (15.2 MB at ≈25 GB/s ≈ 0.6 ms) plus a pool-slot claim on the eviction path. Exclusive frees ≈10.6 GiB of unique capacity but measures −19% cold tok/s (O10). Exclusive residency and the device-side landed wait (`--device-wait`) are mutually exclusive; `golem_set_device_wait` refuses with `evict_batch` set.
- **Policy replay** (`docs/baselines/m3.7-serve-trace-replay.md`). Four real 512-token traces replayed at production geometry (K = 18):
  - **VRAM.** The §6.2 decayed score beats LRU by 2.1 points of VRAM hit.
  - **RAM.** CLOCK stays under inclusive residency: decay at both tiers duplicates the hot set (d 17.8% vs 16.7% at 63 GiB).
  - **Exclusive residency + decay at both tiers:** d 13.8% vs 18.0% with inclusive, so exclusive residency uses a decayed RAM tier.
  - **`Policy::pass`** has no production consumer: per-layer forced-admission slabs consume only its score math.
- **Sizes and context.** The RAM pool is 4,550 / 12,096 units = 37.6%; VRAM K = 9–10 of 288 per layer ≈ 3.5%. Reported results from other systems (BMoE, §16) do not transfer to these sizes, so policy choices rest on the replay above.

Startup warm-up: if a previous run's score file (§11; `journals/<instance>/scores-<fp>.txt`, policy words and step) exists, the per-expert score words and step are restored verbatim, the VRAM order is republished from them, and the top-scored experts (up to each pool class's free headroom) are page-in'd in the background before the first request — the first request never waits on it — this is what makes "warm start" measurable separately from cold (§12.3). A first start has no scores file.

### 6.4 Budgets and feasibility

The engine's load-time plan check (`check_plan`) refuses (not at step 50,000) any configuration whose caps cannot fit: a VRAM expert budget smaller than one layer's routed set (top-k × max stride per plane), or larger than VRAM left after §10's fixed terms → `golem_plan_infeasible` naming the numbers. All budgets derive from §10's arithmetic, so `golem plan` and the runtime agree by construction (same crate, `golem-plan`, computes both).

### 6.5 Prefetch

Autoregressive decoding means step N+1's routing is correlated with step N's, but a last-k union can only re-fetch experts that were just demand-paged and later evicted; it never loads an expert before its first use. Prefetch therefore uses **next-layer routing prediction** (P5), and ships behind a switch: `--prefetch {off,protect,read}` (default `off`).

BMoE's next-layer predictor (the same scheme, §16) was right 80–89% of the time, yet acting on it cost 21–38% throughput when storage was already the bottleneck: wrong guesses spend the disk bandwidth that demand misses need. Using the prediction only to protect predicted experts from eviction gained ≈0.4 hit-rate points at no I/O cost. Golem is disk-bound on cold decode too (§2.4), and once layer-batched reads (§3.4 P3) saturate the queue during misses there is little idle bandwidth to spend. The default is `off`; modes are compared on decode tok/s off vs protect vs read, cold and warm, A/B in both orders (§12.3).

P5 is a graph node per MoE layer L, built only when `--prefetch` is not `off`:
- **Prediction.** `golem_pred-<L+d>-<L>` = the top (k + 4) of layer L+d's router over layer L's FFN input (the `hc_ffn_pre` output), normed with L+d's `ffn_norm`, with the same gating and selection bias. d = 1..`--prefetch-lookahead`. This gives about one layer of compute as lead time. Real routing is untouched, because only staging is speculative, so I1 holds.
- **Delivery.** The scheduler reads the predictions in layer L's pre-pass, right after its ids readback has synchronized the device, and passes them to the `predict` hook (ABI v6). There are no eval-callback views and no extra syncs.
- **Modes.** `protect` shields the predicted union from RAM eviction. `read` queues it for the single prefetch thread, so at most one prefetch page-in is in flight. Its reads drain behind every demand chunk (§5.2), a round stops while demand page-ins are in flight, and its claims never evict a key whose CLOCK reference bit is set (§6.3).
- **Metrics.** `prefetch_hit_rate` = prefetched keys accessed before eviction; `prefetch_wasted_bytes` = prefetched and evicted unaccessed. The P4 counters make the hit rate observable (§11).
- **Real model.** Recall is 84% at lookahead 1 and 87% at 2; output is byte-identical across modes.

### 6.6 Correctness under eviction

An expert may be evicted from RAM only if no in-flight read targets it; in-flight P3 waits hold a reference. Transitions run against the bitsets inside `claim` on the decode thread; there is no placement thread. The decode thread's P3 callback consults the same bitsets under a per-layer shard lock held only for bitset flip + slot handoff (never across I/O). The lock discipline is: **bitset ops are the critical section; I/O happens outside locks** — reviewed against data races in T6 (§12.2) with a loom test over (page-in, evict, access) interleavings. The decode thread claims each (layer, expert) unit once per scheduler compute — the unit's other planes reuse that lease — and reads the policy snapshot once per compute: 42 × 8 = 336 shard-lock claims per token instead of 1,008.

---

## 7. Inference runtime

### 7.1 Slot scheduler

- A slot = one chat sequence: KV-cache sequence id, request state, sampling params, output channel.
- **Per-slot state snapshot/restore.** Prefill is the most expensive operation on this architecture (§2.4), so re-prefilling the whole transcript each turn would cost O(transcript). At turn end, the engine snapshots the sequence state (`llama_state_seq_get_data`: recurrent state + KV), keyed by the token prefix. When a request extends a snapshotted prefix, it restores the snapshot and prefills only the delta. Snapshots live in one preallocated pinned arena (`--snapshot-ram-gib`; omitted = `max(2 GiB, 2 × floor)`, shrunk toward the floor when RAM is short; 0 disables). The floor is one full-context snapshot per slot (≈145 MiB recurrent state + KV each), and an arena below it refuses the load (`golem_plan_infeasible`). Entries are placed first-fit and LRU-evicted until a new entry fits; at most 64 entries live (`snapshot::MAX_ENTRIES`), and a snapshot larger than the whole arena is dropped. A failed restore drops the entry and prefills in full. Snapshots are the only correct form of prefix reuse on a recurrent model.
- Concurrency: up to `--parallel N` slots (default 1 for both `golem run` and `golem serve`). 4 independent slots route ≈30.7 unique experts per layer per step versus 32 with no sharing, so under the §2.4 PCIe/disk ceilings aggregate tok/s is ≈flat and each slot runs at ≈¼ speed. `serve --parallel` > 1 is refused unless `GOLEM_ALLOW_PARALLEL=1` (batching measurements: `docs/baselines/m7-measurements.md`). MoE + disk-paged experts makes heavy batching a bandwidth amplifier (each slot's routed experts union), and batching interacts with the placement hot set, so the default is conservative by design.
- Batching is dynamic: each decode iteration includes every slot whose tokens are ready (continuous batching, slot join/leave at iteration boundaries — the same discipline `llama-server` uses, re-implemented over `llama.h`, which supports multi-sequence batched decode).

### 7.2 Decode loop

One dedicated decode thread (HIP kernels own the GPU). The **CPU backend threadpool** is separate (`--threads`): it runs sampling, the ids readback and any CPU-side op, so it defaults to physical cores / 2. The decode thread is pinned to the V-cache CCD and the I/O threads to the other CCD on the 9950X3D:

1. Drain ready continuations from the queue into a `llama_batch` (prompt chunks on join; single tokens while generating).
2. `llama_decode`.
3. Per slot: sampler chain (llama.cpp, seeded; §12.2 reproducibility) → append token → emit delta.
4. Advance the step counter. Scores need no per-step tick (§6.2 decays on read), and accesses were already booked by the `routing` hook during `llama_decode`.
5. Loop; preempt on stop/preemption signals from the async side (slot cancel = sequence removal, or restore from the last snapshot; recurrent state cannot be partially rewound). Preemption mid-`llama_decode` uses `llama_set_abort_callback`.

Prefill and decode are the same loop; a long prompt is processed as chunked prefill batches (chunk = 512 tokens default) so a cold joiner doesn't starve generating slots beyond one chunk. Per §2.4, a naive 512-token ubatch copies nearly the whole expert set over PCIe; per-token prefill measured 0.24 tok/s at chunk cap 1. Chunks above the §7.6 streaming threshold go through layer streaming at `n_ubatch` up to 2048. Below the threshold, the per-token cap `max(1, K/top_k)` stays.


### 7.3 Chat template and reasoning content

- Prompt construction applies the GGUF-embedded chat template via llama.cpp — parity with the reference runtime again. No Golem-side template code.
- GLM-5.3-Flash's thinking output: the template's reasoning delimiters are exposed as `reasoning_content` on the message/delta (DeepSeek-style convention), split by the template's known marker tokens (the exact markers are read from the chat-template metadata and pinned; they are a property of the model revision, which the manifest already pins). `--reasoning {inline,separate}` selects behavior; default `separate`. When `inline`, raw tokens stream as `content` with markers intact (clients that don't understand `reasoning_content` see the model's native output — never silently stripped; spec bans silent behavior).

### 7.4 Determinism

Same weights (bit-identical), same kernels, same op order ⇒ same activations *for a fixed batch composition*. Known substrate nondeterminism (batch-size-dependent reduction orders; FA kernels) is controlled: the parity harness fixes batch=1, context length, FA setting, and seed. Production serving does not promise cross-request bitwise determinism (no LLM server does); it promises per-(seed, batch-composition) determinism, which the harness exercises.

### 7.6 Prefill layer streaming



**Problem.** Per-token prefill pays a full cold decode step per prompt token (measured ≈4.2 s/token cold, 0.24 tok/s). A ubatch of 2048 routes all 288 experts in every layer (§2.4), so batching only helps if each expert's bytes cross PCIe **once per ubatch**, not once per token.

**Mechanism.** The P2 prefill-stream mode (§3.4) runs per ubatch whose size is ≥ `--prefill-stream-min` (default 256; streaming is opt-in via `--prefill-stream`, needs the RAM pool, and is disabled with a warning when the slabs plus K ≥ top-k do not fit; the arithmetic crossover is ≈4 tokens, because per-token costs ≈4.2 s cold and one streamed ubatch costs ≈13.4 s).
1. **Plane granularity.** The pipeline unit is one plane (gate, up or down) of one layer for all 288 experts. Two stream slabs, each `n_expert × max plane stride` over all classes. Strides from the tensor table: q4 planes 4,718,592 B, q5 5,767,168 B, q6 6,881,280 B (std = q4+q4+q5, q6d = q4+q4+q6, q5q6 = q5+q5+q6). So the max plane is 6,881,280 B, and 2 × 288 × 6,881,280 B = 3,963,617,280 B = **3.69 GiB VRAM**, plus one stride of row-padding slack per slab (`stream_slabs` = 2 × 289 × 6,881,280 = 3,977,379,840 B). VRAM fit on the target: ≈13.0 GiB free after non-expert weights and ROCm (§2.4), minus the P2 slot cache at derived K = 9–10 (§5.4; at most 10 × 644,022,272 B = 6.0 GiB), minus 3.70 GiB of slabs, leaves **≥ 3.3 GiB for `ws` at the prefill ubatch plus KV**. If `ws` does not fit, `golem plan` lowers `--prefill-ubatch` (it does not drop the slot cache). Plane granularity is used because whole-layer double-buffering (2 × 4.08 GiB = 8.2 GiB) fits only without the P2 slot cache.
2. **Sources.** Each plane's expert ranges come from the RAM pool when resident: H2D from the pinned slot directly, holding a lease until the copy event completes, as P3 does. Misses are read through the shared queue (§5.2) into a **pinned staging ring** of 6 slots (`stream::STAGING_SLOTS`), each sized for the largest plane's page-aligned per-expert regions: 6 × 288 × 6,881,280 B ≈ 11.1 GiB of mlock'd, `hipHostRegister`ed RAM at UD-Q4_K_XL. Peak use is 5 slots (two sourced planes held until their slab is reused, plus up to three warm planes ahead of the copy loop). Misses never land in pool slots. So a long prompt does not evict the decode hot set, and the bytes are copied at most once disk→staging and once staging→device, satisfying the AGENTS.md copy rule.
3. **Overlap.** Three `golem-stream-reader` threads keep up to three planes reading ahead of the copy loop, while H2D for plane *p+1* and compute for plane *p* proceed on the device. Layer L's routing readback warms its up and down planes (routed-only) and layer L+1's gate plane. The pipeline runs at its slowest stage, which on this machine is the disk (§2.4).
4. **Scoring.** Streamed accesses are not scored (§6.2: prefill chunks are excluded), and they do not change pool residency.
5. **Invariants.** Every streamed `MUL_MAT_ID` runs on the GPU (I2; the runtime check covers the stream path). The bytes are bit-identical (I1). Output is deterministic per batch composition (§7.4); T2 documents the deltas at prefill ubatch > 1 versus the per-token path.
6. **Kernel risk.** The MMQ `MUL_MAT_ID` path at `n_ubatch` ≥ 2048 is unvalidated on gfx1100. An external report shows a padding bug crashing it on Blackwell/CUDA (§16).

**Ubatch size.** Streamed bytes per ubatch are constant (185.5 GB, ≈13.4 s disk-bound cold), so throughput scales linearly with ubatch until `ws` stops fitting. Estimates: 4096 → ≈305 tok/s, 8192 → ≈610 tok/s. `--prefill-ubatch` defaults to 2048; `golem plan` suggests the largest of 512/1024/2048/4096/8192 that fits §10. Evidence on the same GPU: exllamav3-rocm-moe measured 8192-token chunks at ≈945–990 tok/s vs ≈600–680 at 2048 (RX 7900 XTX, different model and quant, §16). The VRAM left after slabs and the slot cache (≈3.3 GiB, point 1) is the binding term for `ws` per ubatch.

**Routed-only fill for short ubatches (BMoE, §16).** Streaming a full plane is waste when a ubatch routes few experts. Under uniform top-8-of-288 routing a ubatch of T tokens touches 1 − (1 − 8/288)^T of each layer's experts: T = 16 → 36%, 32 → 59%, 64 → 83%, ≥ 128 → ≈97%; real routing is skewed, so the true share is lower. Later chat turns (after §7.1 snapshots) prefill only the new message, often T ≤ 64. The stream slab already uses identity ids with `K = n_expert`, so filling only the routed experts' ranges needs no new mechanism; unfilled ranges are never read by `MUL_MAT_ID`. Up and down are filled routed-only for free: the layer's ids are known once its router has run, before up/down compute. The gate plane of layer L+1 is filled before L+1's router runs, so it stays whole (default) or is predicted from the previous ubatch's layer-L+1 routing with misses filled after the router (BMoE's approach). Switch to whole planes when the routed share exceeds ≈85% (BMoE's threshold; Golem measures its own). Streamed bytes for T = 32 under the uniform estimate: gate 4.72 MB × 288 + (up + down) 10.49 MB × 0.59 × 288 ≈ 1.36 + 1.78 ≈ 3.1 GB per layer vs 4.4 GB whole (≈30% less); the ids-dependent half is what shrinks.

**Budget.** §10 carries the `stream_slabs` (VRAM) and `staging_ring` (RAM) terms, and `ws` grows with `n_ubatch` (compute buffer at 2048 measured, not assumed). `golem plan` refuses a `--prefill-ubatch` whose terms do not fit, and suggests the largest that does.

**Acceptance.** `ws` measured per ubatch (512–8192); cold and warm prefill tok/s on the real model at 512, 2048 and the largest ubatch that fits (separately), compared with the §2.4 estimate. The HIP gate must report zero SKIPPED. The decode hot set must be unchanged across a long prefill: P4 RAM-hit rate before and after.

---

## 8. HTTP API (D7)

### 8.1 Endpoints

| Endpoint | Status | Notes |
|----------|--------|-------|
| `POST /v1/chat/completions` | v1 | streaming (`stream: true`, SSE) and non-streaming |
| `POST /v1/messages` | v1 | Anthropic Messages API, streaming and non-streaming (§8.6) |
| `GET /v1/models` | v1 | every served model (`id` = served name, `owned_by: "golem"`), none special |
| `GET /health` | v1 | Golem extension, never behind the API key: `state`, `tiers{vram_fill,ram_fill}`, `degraded` (`device_cache`, `stalled`), `model`, `turns`, `device_cache_skip_ratio` (§11). 200 for `healthy`/`degraded`/`idle`, 503 for `loading`/`draining`/`releasing`/`fatal-io`, 500 for `fatal` |
| `GET /metrics` | v1 | Prometheus-format (§11) |
| `GET /v1/golem/experts` | v1 | Golem extension, behind the API key: the last decode step's routed experts per layer with their tier, or `{"state":<serve state>}` (§11 expert tiers) |
| everything else | — | 404 `golem_not_found` (typed, §8.4) |

`golem.toml` `[models.<name>]` tables add served models to one instance. Each has `path` (GGUF) and optional engine keys that override the file layer for that model only (flags and env still win); its own `golem optimize` profile applies (fingerprint per path). A table may not set `model` or `model_name` (the table name is the served name); a served name or canonical path used twice is a start error.

Start model: a `model` path (flag > `GOLEM_MODEL` > `golem.toml` `model`) is preloaded and served as `model_name` (flag > env > file), else GGUF `general.name`, else the file stem. With only `[models.*]` tables the instance starts `idle` and the first request loads the model it names; `model_name` without a `model` path is a start error. `idle_unload_min` > 0 also starts `idle`.

There is no default model. Every request must name a served model in `model`; a missing or unknown name is 400 `golem_model_mismatch` listing the served names, and nothing is loaded or swapped. `/v1/models` lists every served model. `/health` `model` is the loaded model, `null` when idle.

- **One resident model (default).** Unless `coresident`, at most one model is resident. On the target, GLM-5.3-Flash UD-IQ3_XXS (~24 GiB VRAM, `ram_experts_gib` 50 + KV/snapshots ≈ 70 GiB RAM) and DS41 UD-IQ2_XXS (24 GiB VRAM planned, 69.8 GiB RAM planned) sum to ≈ 48 GiB VRAM and ≈ 140 GiB RAM, against 24 and 93, so the worker swaps on demand.
- **Swap.** A job naming the unloaded model is held by the worker and still counts as queued. Admission stops and nothing behind it is admitted, so arrival order is strict: no request starves, and each waits at most the turns ahead of it plus swaps. Once every slot is idle, the loaded session saves warm scores and is dropped, the other model loads, and the job runs. A cancelled or draining job is refused rather than swapped for. A failed load answers that job and every job queued behind it `golem_backend_unavailable` and leaves the instance `idle`.
- **Idle unload.** `idle_unload_min` > 0 starts the instance `idle` (no model; `/health` 200 `idle`) and drops the loaded model after that many minutes with every slot free and no job queued. The next request loads its model. Several models, idle unload or no preload refuse `timing_csv`/`routing_trace` (one recorder per session). Idle unload and the lazy first load (the first request's model, when none is preloaded) work at any `parallel`: neither swaps under a running turn, and jobs arriving during the load queue as usual.
- **Coresident.** `coresident = true` keeps every loaded model resident. It is non-pooled only: the engine allows one pooled session per process, so it is refused with `ram_experts_gib`/`cpu_moe_layers`. A switch to a resident model drains the slots and changes the active session without a load. A load that fails beside the others drops them and retries once.
- **Queue and slots.** The CLI applies one `parallel` to every model. The job queue is unbounded and served in arrival order: a request never gets `golem_busy` for load, and a cancelled client leaves the queue. Each extra slot costs its KV and recurrent state while loaded (target GLM-5.3-Flash UD-IQ3_XXS at ctx 262144: 19,712 B × 262,144 ≈ 4.8 GiB + 145.56 MiB per slot, §10), so `parallel` stays within the machine as without idle unload.
- **State and limits.** Load/unload state changes never leave `draining`/`releasing`/`fatal`. All models must agree on forcing expert offload: the first load fixes the process-wide op-offload threshold (`init_backend`). Sampling is client-side. The registry and the admission check see the start model only; the operator keeps every served model within the machine.

### 8.2 Request field policy

Strict allowlist — anything requested-but-unsupported gets an explicit typed error, per spec. Tolerated-only fields (no functional effect) are `metadata`, `user`, `request_id`, `store` (`false`/`null` only; `store: true` is rejected with `golem_unsupported_field` since completions are not stored).

| Field | v1 behavior |
|-------|-------------|
| `model` | required; must name a served model (else 400 `golem_model_mismatch`, no load) |
| `messages` | supported; text content only; image/audio parts → 400 `golem_multimodal_unsupported` |
| `temperature`, `top_p`, `max_tokens` (`max_completion_tokens` aliased), `stop`, `seed`, `frequency_penalty`, `presence_penalty`, `n` (must be 1) | supported |
| `stream`, `stream_options.include_usage` | supported (SSE; final frame carries usage) |
| `tools`, `tool_choice` | supported. Tools are rendered through the pinned chat template, and tool calls are parsed from the model output by llama.cpp's chat layer for that template, then returned as OpenAI `tool_calls` (streaming: `delta.tool_calls`). `tool_choice`: `auto`, `none`, `required`, and a named function (enforced by the template/grammar where the substrate supports it, else a typed 400). Parallel tool calls follow the template. Acceptance: a T7 golden test per template path, plus a T2-style parity check that the rendered prompt is byte-identical to the reference runtime's. |
| `reasoning_effort` | supported, per the GLM-5.3-Flash model card. Exactly `low`, `high` and `none` (case-sensitive strings) are honoured: `low`/`high` are passed to the chat template as `reasoning_effort`; `none` renders the template's no-think form (§8.6). Any other value of any type (`max`, `medium`, `minimal`, numbers, objects, …) is a 400 `golem_invalid_request` — never silently mapped to `max`. `null`/absent sends nothing, so the template default `max` applies. |
| `response_format` (json schema), `logit_bias`, `logprobs` | **400 `golem_unsupported_field`** naming the field and the reason. Clients probing capabilities get answers, not silence |
| `top_k` | Golem extension, supported |

Usage reporting: prompt/completion token counts from llama.cpp counters — cost-accurate, since we own the loop.

### 8.3 SSE shape

Standard OpenAI chunk framing (`data: {chunk}\n\n`, `data: [DONE]`): a first chunk with `delta.role: "assistant"`, then deltas carrying `content` and/or `reasoning_content` (§7.3), terminal frame `usage` when `include_usage`. Backpressure: SSE streams are tokio-guarded; a dead client cancels its slot (decode loop preempts at the next iteration boundary). An error mid-stream (drain, worker death) sends the error frame, then `data: [DONE]`; a live client that cannot keep up with the event buffer gets a `golem_backend_unavailable` error frame, never a truncated `finish_reason: stop`.

### 8.4 Error model

One envelope everywhere (HTTP + CLI + logs):

```jsonc
{ "error": { "code": "golem_plan_infeasible", "message": "...",
             "detail": { /* structured, e.g. per-tier byte overruns */ } } }
```

Codes (stable, documented): `golem_backend_unavailable`, `golem_plan_infeasible`, `golem_model_mismatch`, `golem_multimodal_unsupported`, `golem_unsupported_field`, `golem_storage_unavailable`, `golem_instance_not_found`, `golem_not_found`, `golem_busy`, `golem_stop_timeout` (CLI only: `golem stop` gave up waiting for the exit; retry with `--force`), `golem_invalid_request`. All 4xx/5xx carry `code`. HTTP status per code: 400 for `golem_unsupported_field`, `golem_invalid_request`, `golem_model_mismatch`, `golem_multimodal_unsupported`; 404 for `golem_not_found`; 503 for `golem_busy` (with `Retry-After: 1`) and `golem_storage_unavailable`; 500 for the rest. A missing or wrong API key is 401 `golem_invalid_request`.

### 8.5 Server posture

Binds `127.0.0.1` by default (port: auto-select, recorded in the registry; `--host/--port` to override). Optional API key (`--api-key`, `GOLEM_API_KEY` or `api_key`): `Authorization: Bearer` or `x-api-key` must match on `/v1/*` and `/metrics`; `/health` stays open. No TLS in v1 (loopback tool; document the reverse-proxy path). `golem serve` runs in the foreground by default — logs to stderr. Ctrl-C cancels in-flight requests at once (a second Ctrl-C exits without teardown), SIGTERM and `golem stop` keep the 30 s drain grace (`golem stop --now` cancels at once). `--detach` (double-fork, logs to `$GOLEM_DATA_DIR/logs/<instance>.log`, PID in registry) serves the background habit.

### 8.6 Anthropic Messages API

`POST /v1/messages` serves Anthropic-protocol clients (Claude Code, the Anthropic SDKs) on the same port and key. There is one decode path: `golem_api::anthropic::to_chat` translates the body into the §8.2 chat request. That request then runs through the same `policy::validate`, worker, chat template and tool parser, and the result is translated back.

- **Fields.** A strict allowlist, separate from §8.2's:
  - Supported: `model`, `messages`, `system`, `max_tokens` (required), `stop_sequences`, `temperature`, `top_p`, `top_k`, `stream`, `tools`, `tool_choice`, `thinking`, and `output_config.effort` (→ `reasoning_effort`).
  - Tolerated with no effect: `metadata`, `context_management`, `safeguards`. Claude Code sends all three.
  - Anything else is 400 `golem_unsupported_field`. That includes server tools, `redacted_thinking` and other `output_config` keys.
  - Every block accepts `cache_control` and `citations`, with no effect.
- **Messages.**
  - `system` (a string or text blocks) becomes the system message.
  - User `tool_result` blocks become `tool` messages (`tool_call_id` = `tool_use_id`), followed by the remaining text as one user message.
  - Assistant blocks: `text` becomes `content`, `thinking` becomes `reasoning_content`, and `tool_use` becomes `tool_calls`.
  - `image`/`document` are 400 `golem_multimodal_unsupported`.
- **Tool choice.** `auto`/`any`/`tool`/`none` map to `auto`/`required`/named/`none`.
- **Thinking.** Reasoning is always split into `thinking` blocks (`signature: ""`), whatever `--reasoning` is set to.
  - `enabled` and `adaptive` keep the template default. `budget_tokens` and `display` have no effect.
  - `disabled` renders the template's no-think form: the request carries `reasoning_effort: "none"` on the §8.2 side. The GLM-5.3-Flash template has no `enable_thinking` kwarg, so no-think is the template's own empty-think form — a `<|assistant|><think></think>` prefill — with the `Reasoning Effort: Low` header retained (the header cannot be suppressed; no-think pins it to `low`). Template provenance: the UD-Q4_K_XL GGUF kv `tokenizer.chat_template`, pinned by the checked-in fixture `crates/golem-engine/tests/fixtures/glm-5.3-flash.jinja` (verified byte-identical).
  - `disabled` together with `output_config.effort` is a 400 conflict — never a silent re-enable. `output_config.effort: "max"` maps to absent (the template default); other unrecognised strings are 400.
- **Response.** Blocks come in the order `thinking`, `text`, `tool_use` (`input` is the parsed arguments; unparseable arguments are a 500, never `{}`).
  - `stop_reason` is `end_turn`, `max_tokens`, `tool_use`, or `stop_sequence`, which echoes the matched string via `TurnOutcome::stop_sequence`.
  - `usage` carries `input_tokens`/`output_tokens`.
- **Streaming.** Named events: `message_start` (sent before prefill), then `content_block_start`/`_delta` (`text_delta`, `thinking_delta`, `input_json_delta`)/`_stop`, then `message_delta` (stop reason, usage) and `message_stop`.
  - A `ping` event (not an SSE comment) goes out every 10 s while idle, because a cold agent prefill runs for minutes.
  - Errors mid-stream are an `error` event.
  - A disconnect cancels the slot, as on `/v1/chat/completions`.
- **Errors and auth.**
  - Error envelope: `{"type":"error","error":{"type","message":"<golem code>: …"}}` with the §8.4 status. `type` is one of `invalid_request_error`, `authentication_error`, `not_found_error`, `overloaded_error`, `api_error`.
  - Auth accepts `x-api-key` or `Authorization: Bearer`. A 401 or 404 under `/v1/messages` uses this envelope.
  - `anthropic-version` and `anthropic-beta` are ignored.
- **Model.** Strict, as in §8.2. Claude Code sets it with `ANTHROPIC_MODEL`.
- **Cost.** A captured Claude Code request is 66 KB of JSON: ≈ 47.5k chars of tool schemas plus ≈ 5.8k of system prompt. At ≈ 3.5–4 chars/token that is ≈ 13–15k tokens. At the measured 5.6 tok/s cold streamed prefill that is ≈ 14 000 / 5.6 ≈ 42 min for a cold first turn. Later turns depend on §7.1 prefix snapshots. Turn-1/turn-2 TTFT on the target is unmeasured.

---

## 9. CLI and instance lifecycle (D8)

| Command | Behavior (pinned) |
|---------|-------------------|
| `golem run [model]` | If a compatible instance (§9.4 config hash) is running, attach an interactive chat client (stdin REPL over the HTTP API, streaming). Otherwise load the model **in-process, in the foreground**, and run the REPL directly over the engine. Exit (EOF / Ctrl-C) unloads and exits: no daemon is left holding the GPU and locked RAM. `--keep` instead starts a detached `serve` and attaches to it. Flags map onto instance config. Ctrl-C mid-reply cancels that turn and keeps the REPL; Ctrl-C at the prompt, SIGTERM, or a signal during `--prompt` exits `128 + signal` (130 / 143) with `golem: interrupted` and no error envelope; a second signal before the first is handled kills the process |
| `golem serve [model]` | Foreground instance (§8.5). `--detach` supported (`--load-timeout`, default 600 s: the parent kills the instance and reaps its entry if it is not serving by then). Serves the start model plus any `[models.<name>]` tables (§8.1). `--no-warm-start` neither restores nor saves warm-start scores (§11) |
| `golem list` | Default: running instances from the registry (id, model, kind, health, device-cache hit % as `TIER%`, `degraded` flag; `serve` health comes live over the control socket), plus `golem` processes with no entry as `unregistered`. `--models`: manifests under `$GOLEM_DATA_DIR` (name, quant, size, replicas). Both, never conflated |
| `golem stop <id\|name>` | `name` is the served model name; one served by several instances needs an id. Graceful: stop accepting, drain in-flight requests (grace period 30 s), unload, exit. `--force`: SIGKILL. `--now`: skip drain. Stopping a nonexistent/dead instance is a typed error, not a shrug. Socketless kinds get SIGTERM then SIGKILL after a per-kind grace — `run` 60 s (teardown; the SIGKILL may land mid-load), others 5 s |
| `golem plan [model] --ctx N --vram-experts-gib N --ram-experts-gib N ...` | §10. No GPU required; reads manifest + flags, prints per-tier byte table and a fit verdict with exact overages. Prices the §8.1 start model; `--model-name` picks a `[models.*]` entry |
| `golem optimize [--model M]` | Sweeps the profile knobs (`threads`, `numa_interleave`, `o_direct`, `io_threads`, `io_depth`, `async_h2d`, `vram_experts_frac`, `cpu_expert_math`, `cpu_expert_frac`, `prefill_ubatch`; `--axes` overrides) by bounded coordinate descent (≤ 2 passes, `--max-minutes` default 120), one child process per measurement, each candidate paired with the incumbent in random order. A decode win needs +3% tok/s, hit rate within 0.5 pp, TTFT p50/p99 within 1.2× of baseline and matching greedy output; a prefill axis needs ≥ 10% faster long-prompt prefill with decode within 3%. The winner is confirmed by a reverse-order rerun and saved as the profile layer (§9.4) |

Engine flags that change I/O, residency or state (each also `GOLEM_<FLAG>` / `golem.toml`, §9.4):

- `--o-direct` (off): require `O_DIRECT` store reads; fail instead of falling back to buffered + `DONTNEED` (§5.4).
- `--numa-interleave` (off): interleave the RAM pool across NUMA nodes; no-op on one node.
- `--async-h2d` (off, hidden): pooled H2D on a dedicated HIP copy stream; needs the RAM pool.
- `--device-wait off|on|kernel` (off): the GPU waits for each decode MoE layer's experts to land instead of a host sync per layer; needs `--async-h2d`, the pool on HIP, inclusive residency, no `--cpu-expert-math`, top-k ≤ 32.
- `--victim-ring N` (0 = off; file key `victim_ring_slots`): VRAM victim ring slots for `--residency exclusive` (§6.3 O10), 3 × max plane stride each (≈ 20.6 MB at UD-Q4_K_XL).
- `--snapshot-ram-gib G` (default `max(2 GiB, 2 × floor)`, shrunk toward the floor when RAM is short; 0 disables): state-snapshot arena (§7.1); below the floor the load is `golem_plan_infeasible`.

### 9.2 Instance registry (D11)

`$XDG_RUNTIME_DIR/golem/instances/<id>.json` (a detached `serve` without user linger registers under `$XDG_STATE_HOME/golem/instances`, which is also read): id (8 hex chars of a uuid), pid and its start time, kind (`serve`, `run`, `optimize`), http endpoint (`serve` only), served model name, config hash, effective config, planned VRAM/RAM, started-at, health cache, child pids. A heartbeat thread rewrites the entry every 1 s. An entry is stale (reaped) only when its PID is gone or reused (start time differs); heartbeat staleness marks a live instance **hung**, never reaped — `golem stop` escalates it to SIGKILL when the control socket is also unreachable and the heartbeat stays stale over two samples ≥ 10 s apart. Stale entries are reaped by any golem client that observes them (registry is advisory state, not a database). Control (stop, status detail) goes over a per-instance UDS carrying the same JSON-RPC-ish envelope as the error model — `golem stop` works even when HTTP is wedged.

### 9.3 Multiple instances / models

Each instance is an independent process with its own VRAM budget; concurrent instances are allowed and *individually* planned, but combined-fit is the operator's responsibility — `golem plan` checks a single instance; before starting a second instance, the CLI sums planned VRAM **and planned locked RAM** of running instances. It **blocks** by default when either sum exceeds physical capacity (`golem_plan_infeasible`, naming each instance's share), and `--allow-overcommit` proceeds with a loud warning. For this model one instance *is* the GPU and most of the RAM, so an overcommitted second instance would thrash or OOM mid-request, which the spec bans. Documented, not hidden. (Golem does not arbitrate VRAM across instances; the spec only asks the question be answered honestly.)

### 9.4 Config precedence

Flags > env (`GOLEM_*`, uppercase, one per flag) > config file (`$GOLEM_CONFIG_DIR/golem.toml`) > `golem optimize` profile > built-in defaults. The config file is `golem.toml` in `$GOLEM_CONFIG_DIR`, else `$XDG_CONFIG_HOME/golem`, else `~/.config/golem` (one resolver shared with the profile's tuning dir); a missing file is no file layer, an unparsable one is an error naming the path, and which file was loaded (or that none was found) is logged at info. The model path resolves the same way: the positional argument of `run`/`serve`/`plan` or `--model` of `optimize` > `GOLEM_MODEL` > `model` in `golem.toml` (`~` expanded); with none set, `serve`, `plan` and `optimize` fail naming all three, `run` prints its help. A `run --keep` detached `serve` resolves it through the same layers. The profile (`$XDG_CONFIG_HOME/golem/tuning/<fingerprint>.json`, per KV pair, `~/.config` when unset) applies only when it holds a winner, the hardware/model/engine fingerprint matches (GPU, CPU topology, model shards' name/size/mtime, engine version and vendor pin), and free RAM/VRAM now are at least what the sweep measured minus 256 MiB; `--no-optimize-profile` skips it. A profile also records the cache element types it was measured under (`cache_type_k`/`cache_type_v`, written `k/v`; entries written without the field ran llama's f16 default and read as `f16/f16`) and does not apply to a run resolving other types — the types price `kv_per_token` (GLM-5.3-Flash at ctx 131,072: 19,712 B/token at f16 vs 14,432 B at q8_0, 660 MiB of VRAM over the sweep's candidate slot counts), so the knobs would be tuned against another free-VRAM figure; the skip, like the free-memory one, is logged at info with its reason. Each pair has its own file, `tuning/<fingerprint>-<k>-<v>.json`, so sweeps under different types do not replace each other; the untyped legacy `<fingerprint>.json` is read only when no typed file exists. The instance's effective config hash is recorded in the registry (§9.2); `golem run` only attaches to a live `serve` instance (healthy, degraded or idle) whose config hash is *compatible* (same model; placement budgets that fit the running instance), else it reports the conflict and runs in-process (`--new` skips the check; `--keep` starts a detached `serve` and attaches).

---

## 10. `golem plan` — the memory model

All figures derive from the GGUF header via the manifest — never hardcoded model tables. Let `E` = experts/layer, `L` = expert layers, `b_e` = bytes per expert per layer, and `T_ne` = total bytes of non-expert tensors (computed by filtering the tensor table through the upstream MoE regex — the same classifier llama.cpp uses, so "expert" means the same thing in both).

```
VRAM =
  T_ne                                    non-expert weights (resident forever)
+ kv_per_token · ctx · parallel            KV cache, 11 sparse-attention layers only
+ rs_bytes · parallel                     recurrent state, 34 linear-attention layers (constant per sequence; 145.56 MiB measured)
+ shexp_bytes                             shared experts, VRAM-resident (part of T_ne, listed so it is never tiered)
+ ws                                       decode/prefill workspace (fitted on gfx1100, `WS_MEASURED`: max(320 MiB, ubatch · (699 500 + 3·ctx)) plus a 9·ctx score-buffer term; `docs/baselines/m4.6-plan-t8.md`)
+ device_expert_cache                      ≤ --vram-experts-gib, as slot slab K × stride per (layer, kind), §3.4 P2; `resident_experts` (all experts in VRAM) for a non-pooled session
+ stream_slabs                             2 × (n_expert + 1) × max plane stride, prefill layer streaming (§7.6; 3,977,379,840 B ≈ 3.70 GiB on the target)
+ victim_ring                              VRAM victim ring slots (docs/baselines/m6.6-vram-victim-ring.md)
+ rocm_overhead                            measured: 2.64 GiB combined overhead on gfx1100
+ vram_reserve                             max(1.5 GiB, 6% of device total)

RAM =
  ram_pool                                ≤ --ram-experts-gib (default min(.75·free, Σ expert bytes))
+ host_overhead                            loader + activations (measured, starts 2 GiB)
+ state_snapshots                          per-slot snapshot cache (§7.1)
+ staging_ring                             pinned disk-miss staging for prefill streaming, priced at 2 × n_expert × max plane stride (3.69 GiB on the target; the engine allocates 6 slots ≈ 11.1 GiB, §7.6, §17)

DISK =
  model_bytes · replicas + logs
```

Output: table per tier, the four governing numbers (VRAM headroom, RAM pool, per-token KV cost, max cold-expert latency estimate from measured bandwidth), and a verdict. Infeasible → exit code 2, `golem_plan_infeasible` with the failing inequality printed with actual numbers. `--json` for tooling. With the P2 slot-indirected device cache the cache is `K` slots × stride per (layer, kind); there is no full-size transient expert copy. The same functions serve the runtime's load-time feasibility check (§6.4) — one implementation, two callers, no drift.

KV per token without loading the model: read from the arch table (glm5next entry created from the loaded context's actual allocation, then frozen as a formula in metadata keys); when the GGUF carries the needed dims, the formula computes exactly.

`kv_per_token` is priced per cache type: each attention layer costs `heads_kv · (row(type_k, key_length) + row(type_v, value_length)) + idx_heads · row(type_idx, indexer_key_length)`, where `row` is ggml's block size (q8_0 = 34 B per 32 elements). MLA stores no V, so it requires `type_k == type_v` (as llama does). On glm5next the indexer cache stays f16 under a quantized K, because it also holds the compressor gates (`llama-model.cpp`). Target at ctx 131,072: f16 19,712 B/token (2.41 GiB), q8_0 11 × (16 × 34 + 384 × 2) = 14,432 B/token (748 + 1,056 MiB, matching llama's log). The quantized V turns on flash attention. A/B on UD-IQ3_XXS (`scripts/kv-ab.sh`, both orders): needle recall at 10/50/90 % of 129k tokens passed for both types, prefill ≈ 78–84 tok/s and decode ≈ 3.3 tok/s for both, and the 660 MiB saved fit 2 more device-cache slots (`k_decode` 21 vs 19). The registry records the types, so `compatible` refuses to reuse an instance with different ones.

---

## 11. Observability

- **Tracing:** `tracing` + JSON logs to stderr/file; spans for request lifecycle, decode steps (sampled), page-ins, evictions. `serve` logs at INFO for `golem_server`, `tower_http` and `golem_engine::slot` and at WARN otherwise (`RUST_LOG` overrides); JSON when detached or off a TTY, text otherwise. It logs every HTTP request/response with status and latency, and one `chat completion` line per request with status (499 = client disconnected), finish reason, prompt/completion tokens, `prefill_tok_s` (admission → first piece; a snapshot-restored prompt or `--parallel` > 1 reads as latency) and `gen_tok_s`.
- **Metrics** (`/metrics`, Prometheus): token rates per slot; bytes disk→RAM (demand vs prefetch), RAM→VRAM (demand vs promote), VRAM evictions; P2 skip ratio (device-cache hit rate); P3 slow-path rate (the key health number — should be ≈0 steady-state); `prefetch_hit_rate` (prefetched keys accessed before eviction, §6.5), `prefetch_wasted_bytes` (prefetched, evicted unaccessed, §6.5), `state_snapshot_hits` (§7.1), and an I2 guard counter of expert `MUL_MAT_ID`s executed off-GPU (must stay 0); tier fill levels; per-code error counters. As exported: `golem_device_cache_{bytes_copied,bytes_skipped,evictions}_total`, `golem_pool_{demand_pageins,fast_hits,evictions,bytes_paged,source_errors}_total`, `golem_prefetch_hit_rate`, `golem_prefetch_wasted_bytes`, `golem_state_snapshot_{hits,misses,restore_failures,evictions,dropped_oversize}_total`, `golem_state_snapshot_bytes`, `golem_expert_mul_mat_id_off_gpu_total` (HIP builds), `golem_locked_ram_bytes`, `golem_vram_free_bytes`, `golem_queue_depth`, `golem_in_flight`, `golem_errors_total{code}`, `golem_slot_tokens_total{slot}`. Also exported: `golem_prefetch_standdown_total`, the `golem_stream_*`, `golem_cpu_expert_math_*` and `golem_victim_ring_*` families, `golem_store_serial_fallbacks_total`, `golem_expert_early_h2d_units_total`, `golem_pool_split_units_total`. `golem_expert_h2d_copies_total{path="cache"|"stream"}` (expert planes copied host→device by device-cache fill or streamed prefill; the stream path counts whole-tensor fills, which can exceed routed demand; pooled offloaded sessions only) and `golem_expert_mul_mat_id_total{device="gpu"|"cpu"}` (expert `MUL_MAT_ID` graph-node placement; offloaded sessions only; `golem_expert_mul_mat_id_off_gpu_total` is its deprecated `{device="cpu"}` alias). Tier fill is reported in `/health`.
- **Per-step timing (BMoE, §16).** Counters say how many misses; they cannot say why cold decode moves ≈2 GB/s against 5.8–8 GiB/s of disk (§2.4). `--timing-csv <path>` writes one row per decode step: step, tokens, wall ms, P3 wait ms (time inside `ensure_host_resident`, measured in the Rust trampoline), disk read ms and bytes, H2D ms (HIP events on the copy stream) and bytes, policy ms, and compute ms = wall − the measured parts. A column that is not measured in a given build is written `unmeasured`, never `0`. The file starts with a header carrying engine version, fork commit and the full resolved config. Cost: a few `Instant::now()` per layer and one event pair per H2D batch, no allocation per row (preallocated ring flushed at turn end), so it obeys the hot-path rules and can stay on during benchmarks. With `async_h2d`, `h2d_ms` is the sum over the step's copying splits of first copy start to last copy end on the copy stream (gaps between splits excluded), so it includes copy-stream waits on compute within a split; a step with more than 256 copying splits (the event ring size) writes `unmeasured`. The copy stream waits on the compute-stream tail before each plane (conservative WAR rule), so GPU-side H2D/compute overlap is ≈0 and only host-side queueing runs ahead; an A/B is expected to show that.
- **Run trailers and fixed-length runs.** The timing CSV ends each turn with `# prefill_ms=<ms> prompt_tokens=<n>` when the turn prefilled, and `golem run` appends end-of-run trailers after the rows: `# <prefetch stats>`, `# store scratch_planes=…`, `# device_cache …`, and `# pool split_units=<n> ensure_calls=<n>` (role-major units split, `ensure_host_resident` calls; the same line goes to stderr, and `split_units` is exported as `golem_pool_split_units_total`). `golem run --ignore-eos` never samples an end-of-generation token, so every completion runs to `--n-predict` (fixed-length benchmarks); like `--role-major-pagein` it always loads in-process, since an attached instance cannot honour it. The resolved-config header records `ignore_eos`, `role_major_pagein`, `async_h2d` and `cpu_expert_math`.
- **Warm-start scores:** `golem-scores v1` text file of the raw §6.2 score words plus the decode step, written atomically to `$GOLEM_DATA_DIR/journals/<instance>/scores-<fp>.txt` (fp = hash of shard names, sizes and mtimes) on graceful shutdown and every ≥ 300 s at turn end; restored for score seeding (§6.3). Offline policy tuning uses routing traces (`docs/baselines/traces/`).
- **Expert tiers (`GET /v1/golem/experts`).** A read-only, best-effort view of where the last decode step's routed experts came from, per expert layer. Response: `{"model","step","step_kind","n_layers","n_experts","top_k","layers":[{"layer","step","vram_resident","ram_resident","truncated","routed":[{"id","tier"}]}]}`; `{"state":<serve state>}` without a tier view (no RAM pool) or before the first decode step: `"idle"` with no model loaded, else the `/health` state (e.g. `"healthy"`). Fields:
  - `tier`: `vram` = already assigned in the device slab, no host copy (the unmarked default only when the session's device slab is armed: HIP build, layers offloaded, K > 0; otherwise the substrate reads every unit from the host pool and the unmarked default is `ram`); `ram` = copied from the RAM pool, resident before the step; `disk` = read from disk into the pool, then copied; `cpu` = computed on the CPU from the pool (CPU expert math). A unit missed in several split segments keeps the slowest tier seen (`disk` over `ram`).
  - `routed`: the step's distinct ids in first-routed order, up to `parallel × top_k` (`truncated` when more were routed).
  - `vram_resident`: device slab slots with gate, up and down all valid at the step end (≤ K); `ram_resident`: pool units claimed at the step end, mid-fill included.
  - `step` / `step_kind`: the last closed decode step and the kind of the last hooked compute (`prefill` / `decode`). `step` is the newest published row's step. The session arms the rows only for a batch made entirely of generation tokens; prefill and mixed prefill+decode batches never touch the rows and set `step_kind: "prefill"` only.
  - Semantics are step-granular: the hooks write a working row readers never see, and after the decode's synchronize each opened working row is copied into its published row under that row's seqlock (≤ `slots` words + 4 counters per layer: GLM 42 × 8 = 336 words per step). Readers see only published rows, so a read mid-generation returns the previous step; only a read torn by a concurrent publish (after retries) comes back `"open": true` with `routed: []`. A failed or aborted step publishes nothing and does not advance `step`. The server reads atomics only and never blocks the decode thread. Cost per routed id: one relaxed store at routing, a ≤ `slots` scan per miss or CPU lease; per layer per step: one lock-free C getter (`golem_device_cache_layer_resident`, ABI v22) and two loads. Size: GLM 42 × 8 × 4 B = 1.3 KiB at parallel 1.
- **Health states:** `loading` / `healthy` / `degraded` / `idle` / `draining` / `releasing` / `fatal` / `fatal-io`, with `degraded` listing `device_cache` (device cache degraded) and `stalled` (no decode progress past the stall bound while a job is active) — surfaced in `/health`, `golem list`, and log lines. Degraded is a first-class, visible state (spec: detect and report, never silently).

---

## 12. Testing and acceptance

### 12.1 Layout

- Unit tests per crate (policy math, manifest, striping, plan arithmetic — pure and table-driven).
- Integration tests run against a **synthetic tiny MoE GGUF** (a 2-layer, 8-expert model generated by a test helper that writes valid GGUF with the upstream tensor-naming scheme): exercises the full load → route → page → decode path in CI without 186 GiB or a GPU (CPU-backend substrate build for CI; HIP build tested on hardware in the hardware lab job).
- Hardware tests (7900 XTX machine): nightly script, not CI-gated.

### 12.2 Test matrix

| ID | Test | Asserts |
|----|------|---------|
| T1 | Op parity: expert matmul (`MUL_MAT_ID`) on fixed weights/ids, HIP vs CPU substrate, Q8_0 and UD-Q4_K_XL | max abs diff within the bound (`--bound`, default 1e-2, on the fp16 accumulate path) |
| T2 | End-to-end greedy parity vs reference `llama.cpp` (same pinned commit, same GGUF): 50-prompt suite, batch=1, seeded | token-ID comparison where a divergence **passes** only if the reference leg's top-2 logit gap at the first divergent position is below epsilon (`--epsilon`, default 1e-2; strict `<`; a benign tie); every divergence is reported (position, tokens, gap). Perplexity delta on wikitext-2 slice ≤ 0.5% |
| T3 | **Cache invariance** (spec I1): identical prompt under (a) fully pre-promoted cache, (b) cold start, (c) randomized eviction stress, (d) minimum VRAM cap (K = one routed set) | all four produce **identical greedy token IDs** — the direct test that caching never changes selection |
| T4 | Fork patches: upstream test suite green; P1 unset-hooks behavior byte-identical to upstream; P2 validity-bitset invariants fuzzed | no behavioral delta without hooks |
| T5 | Store: striping correctness across 1/2/4 replicas, per-chunk failover mid-read, checksum-rot detection (injected), same-drive warning | reads correct under all replica-count/failure combos |
| T6 | Placement concurrency: loom test over (access, page-in, evict, promote) interleavings; 24 h soak on hardware (default I/O engine) | no races, no residency-bitset drift vs pool contents |
| T7 | API contract: field-policy table (§8.2) becomes generated round-trip tests; SSE framing golden tests | unknown field ⇒ typed 400; stream shape stable |
| T8 | Plan vs reality: `golem plan` predicted VRAM vs actual after load, on hardware, over ctx sweep | actual − predicted ≤ 256 MiB, predicted − actual ≤ 1 GiB across sweep (ws is a max of fitted slopes; over-prediction only shrinks the derived cache) |

### 12.3 Benchmarks and the documented baseline

A test that cannot meet its precondition **fails**; it never returns early as a pass. Every P1–P5 change is evidenced by `--features hip-tests` output with zero `SKIPPED` lines through **engine-registered** hooks. Fixtures match real shape where it matters: `.weight` names, a split GGUF with a metadata-only shard 1, mixed-stride K-quant planes, and a device budget smaller than one expert tensor. Every design deviation carries one line of arithmetic against the real model's numbers.

The benchmark harness (`scripts/benchmark.sh`, not a product subcommand): cold-start TTFP, warm TTFP, tok/s at ctx ∈ {4k, 16k, 32k} × cache ∈ {cold, warm, pre-promoted} × replicas ∈ {1, 2} — the cold/warm split the spec demands. Every run stamps the baseline record: RAM, drives (model + link), kernel, ROCm version, fork commit, quant + revision, ctx. Baselines live in `docs/baselines/`. No target numbers are invented; the spec forbids it.

**Teacher-forced top-k parity.** Next to T2's greedy match rate, report a teacher-forced comparison against the reference runtime on the same token sequences. Per position, both sides emit their top-20 log-probs. Report:
- argmax agreement and top-5/top-10 overlap;
- KL over the reference's top 20 plus one bucket for the rest;
- perplexity of each side;
- disagreements bucketed by the reference's top-1/top-2 gap (< 0.2, 0.2–0.5, ≥ 0.5).

A raw match rate counts divergences; the gap buckets show whether they are near-ties. This is also the form in which O1 (CPU math) states its determinism exception: "every disagreement has gap < x".

**A/B method.** A comparison between two settings runs as A, B, then B, A (cold state re-established before each run); a difference that follows the setting in both orders counts, one that follows run position does not. Every results file in `docs/baselines/` embeds the resolved config, engine version and fork commit (the §11 timing header), and decode numbers cite the per-step timing CSV that produced them.

---


## 13. Repository layout and toolchain

```
golem/
├── docs/                   # product-spec.md, tech-design.md, baselines/ (measured numbers), procedures/
├── engine/llama-cpp.lock          # pinned fork commit, ROCm line, patch list, validated flags
├── vendor/llama.cpp/              # fork at pin; P1–P5 patches + golem-hooks.h
├── crates/
│   ├── golem-cli/                 # clap commands; thin — all verbs delegate to libraries
│   ├── golem-server/              # instance process: axum API, slot scheduler, decode loop
│   ├── golem-engine/              # llama.h glue: load config, sampler chain, batch building, KV mgmt
│   ├── golem-llama-sys/           # bindgen over llama.h + golem-hooks.h; cmake build of the fork
│   ├── golem-placement/           # scores, tiers, caps, prefetch; policy consumed by P2/P3
│   ├── golem-store/               # replicas, striping, checksums, pool
│   ├── golem-plan/                # §10 arithmetic (shared by CLI and runtime feasibility check)
│   ├── golem-gguf/                # GGUF reader
│   ├── golem-api/                 # OpenAI types, SSE, error envelope, field-policy table
│   └── golem-registry/            # instance registry, UDS control, heartbeat
└── xtask/                         # bench harness, tiny-moe generator, parity suite runner
```

Toolchain and standards: Rust stable (currently 1.99, `rust-toolchain.toml` pins; edition 2024). `cargo clippy -D warnings`, `cargo fmt` check, `cargo nextest` in CI; deny.toml for license/advisory audit; `unsafe` is confined to `golem-llama-sys` and the FFI glue modules (hook trampolines, `golem-engine/src/{session,p3_glue,stream,chat,parity}.rs`, `golem-cli/src/{run,sys}.rs`, `golem-store`, `golem-registry/src/sys.rs`) and `golem-placement/src/pool/`; each site carries `allow(unsafe_code)`. Safe public APIs cannot permit UB: slot access goes through lease/claim tokens, each with a `// SAFETY:` contract comment and reviewed in T6. Key crates: tokio, axum 0.8 + tower-http, serde/serde_json, clap derive, thiserror/anyhow (thiserror in libraries, anyhow at binaries), tracing, bindgen, cmake, sha2, crc32c, libc, sysinfo, uuid.

---

## 14. Components

What exists in the repo; the design sections define behavior.

- Substrate: pinned fork under `vendor/llama.cpp/` with P1–P5 patches and `golem-hooks.h`, locked in `engine/llama-cpp.lock`; `golem-llama-sys` and `golem-engine` (load, greedy and sampled decode, per-slot state snapshot/restore, start-time verify, CCD pinning).
- Placement and storage: `golem-placement` (scores, CLOCK eviction, caps, slabs, pool, pinned arena) and `golem-store` (replicas, shared-queue striping, checksums, `O_DIRECT` reads), wired into P2/P3 with predictive prefetch and prefill layer streaming (§7.6).
- Product surface: `golem-api`, `golem-server` (axum API, slots), `golem-registry`, `golem-plan` with T8 calibration, and the `golem` CLI (run/serve/list/stop/plan; no import, §5.5).
- Evidence: measured baselines in `docs/baselines/` (pool-size, trace replay, hot-set replay, plan vs actual, drain read depth, prefill staging, persistent drain readers, layer-batch I/O depth, VRAM victim ring, tier endpoint overhead, role-major page-in, CPU expert math).


---

## 15. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GLM support unmerged upstream; fork chases a moving PR | certain (spec states it) | high | pin commit; patches confined and rebased monthly |
| ROCm MLA/precision behavior differs from NVIDIA workarounds | likely | high | `--flash-attn` switch; validated settings frozen into the lock file |
| PCIe becomes the bottleneck; cold decode unusably slow | **likely** (§2.4: ≈4.9/(1−h) tok/s ceiling; CPU-hybrid ≈2.5–3× faster at low h) | existential for the product | levers in order: lower-bit expert quant (§4.1, O7), pinned H2D (§3.5), VRAM hit rate (P2 slot cache + predictive prefetch), prefill streaming (§7.6), shared-queue striping (§5.2); opt-in CPU expert math (O1, §1) |
| Fork patch conflicts on rebase | **high** | **high** | the P2/P3/P5 work sits in the scheduler's expert-input path, one of upstream's most-changed areas; isolate behind one extension point; T4 as rebase gate |
| Cold prefill unusably slow (0.24 tok/s without streaming) | certain without streaming | high (agentic prompts are 10k–35k tokens) | §7.6 layer streaming at ubatch ≤ 2048 (≈150 tok/s estimated, disk-bound); MMQ `MUL_MAT_ID` kernel validated at large ubatch on gfx1100 |
| Cache-state-dependent nondeterminism (I1 violation) | low | fatal to the core claim | T3 is a first-class test, run in nightly hardware CI |
| Multi-instance VRAM oversubscription crashes the driver | medium | medium | blocked by default when planned VRAM or locked RAM exceeds physical (§9.3, O5), with `--allow-overcommit` as a loud override; per-instance budgets |
| RAM pool + page cache pressure on smaller-RAM machines | medium | low | pool default ≤75% free RAM, mlock'd; `--ram-experts-gib`; plan reports RAM fit explicitly |
| Hybrid recurrent architecture breaks KV-rewind assumptions (slot cancel, prefix reuse) | certain | high | state snapshot/restore (§7.1); no partial rewind anywhere |
| Checksum verify slows startup (~186 GiB) | certain | low | start-time size + mtime + sampled hash (16 × 1 MiB SHA-256 windows; 0.86 s cold on UD-IQ3_XXS vs 1452 s for `--full-verify`); `--full-verify` opt-in; CRC32C at page-in |

---

## 16. References

- Product spec: `product-spec.md` (same repo).
- llama.cpp: GLM MoE support PR #27754; feature matrix (ROCm/Vulkan capability evidence); `ggml-backend.cpp` split-scheduler expert-copy path; `-ncmoe`/`-ot`/`--lazy-mode` docs.
- Comparison projects per spec: rabbit (CPU-only disk-streamed experts), deltafin (GPU paths; ROCm path unvalidated), colibri (tiering + dual-SSD reference; GPU-hot/CPU-rest split — the exact gap Golem closes).
- r/LocalAIStack, "GLM-5.3-Flash UD-IQ4_XS running on 5090+64GB Ram with dual SSD-Offloading" (u/zMytze; project `glm53-ssd-streaming`, built on unmerged llama.cpp expert-streaming PR #25294). External reference for §7.6 and §5.2: layer-streamed prefill through pinned double buffers at ubatch 2048, both NVMe drives reading one mirrored GGUF from a shared work queue, an `O_DIRECT` RAM expert cache, and an MMQ padding fix for `-ub 2048` on Blackwell. Reported: ≈230–250 tok/s prefill, 5.6 tok/s decode, 78% RAM-cache hit, ≈1.3 GB/token from SSD. Not measured on this machine and not targets (spec §Acceptance).
- AMD/ROCm prior art (none drop-in for Golem): **bsvinay/exllamav3-rocm-moe** (ExLlamaV3 ROCm port, RX 7900 XTX + 7950X3D + 60 GB). Hot experts sit in VRAM and the rest run on CPU AVX-512 kernels (DRAM-bound ≈50 GB/s). Its learned placement is persisted across restarts (CPU share 0.47 → 0.15 on the first request), which corroborates persisting the §6.2 scores file for warm start. Prefill at 8192-token chunks ≈2× faster than at 2048 (§7.6 ubatch size). MTP 2-token drafts take decode from 53 to 83 tok/s on code. Its llama.cpp `--n-cpu-moe` reference on the same GPU is 17.5 tok/s. **GenerelSchwerz/llama.cpp** `moe-cache`: a VRAM MoE expert cache, CUDA-only by its own docs, so it gives no ROCm evidence. All numbers are the authors'.
- Upstream MoE expert caches (authors' numbers, all-experts-in-RAM hosts, so their 1.57–2.2× decode does not transfer: Golem's 172.8 GiB of experts vs 93 GiB RAM leaves disk at 151–237 ms/token): **llama.cpp #29887** (merged): `--moe-cache-mib` LRU over host experts for `MUL_MAT_ID` at batch ≤ 32, the same job as P2; 72% hit at 6.4 GB, 89% at 18.6 GB, on 65 GiB of experts (4090/5090). **#30112** (merged): the same per GPU (single-GPU here, n/a); prompt t/s drops to 0.91×/0.73× with the cache on. **#27861** (open): 54k-record Qwen3.8-Flash-Next routing trace, top-32 static list ≈10% coverage vs per-layer LRU-64 ≈67% and LRU-128 ≈81%, consistent with keeping the decayed score over LRU (+2.1 VRAM-hit points, §6.2); its CPU/GPU split (cached ids → second `mul_mat_id` chain, uncached → zero slot, CPU skips cached ids, outputs summed, async uploads) is the template for O1. **Discussion #24528**: hybrid hit/miss RFC; fill on decode only, since prompt routing is flat and thrashes. **neurall/llama.cpp**: GLM-5.3-Flash at 3.0 bit (O7 forbids), MTP no gain (Golem skips the MTP block, §2.2).
- Reference designs (authors' numbers, other hardware, not targets): **Helldez/BigMoeOnEdge** (BMoE): CPU-only, lossless expert streaming on phones through llama.cpp's public API (`cb_eval` on the `ffn_moe_topk` node, `->data` repointing). Source of: parallel read lanes (≈3× on phone flash; §3.4 P3 layer-batched reads), per-token timing CSV with `unmeasured` columns and config header (§11), routed-only prefill fill with an 85% whole-layer switch (§7.6), next-layer prediction at 80–89% accuracy that still lost 21–38% when I/O-bound (§6.5 kill switch), hit rate ≈ linear in cache size and a smarter policy +2 points / −30% speed (§6.3 replay), MTP acceptance 71/60/52% at k = 2/3/4 and batch-dependent output (Golem skips the MTP block, §2.2; batch dependence, §7.4), A/B in both orders (§12.3).

---

## 17. Known gaps

- `golem plan` / T8: the total-delta gate depends on device-cache residency accounting; ws is fitted (`golem_plan::WS_MEASURED`) but residual scatter is non-ws (`docs/baselines/m4.6-plan-t8.md`).
- A second concurrent session or a re-enable is refused (`ENABLE_LOCK` in `p3_glue.rs`) rather than isolated; sequential models carry their own `golem_ctx`.
- CCD pinning (`--pin auto`) throughput effect is unmeasured.
- Two GGUF readers beyond `golem-gguf` remain: `golem-engine/src/gguf.rs` (KV/tensor layer) and `xtask/src/gguf.rs` (writer + `read_arch`); the manifest `expert_index` and the pool geometry are two expert indexes.
- `Policy::pass` allocates per pass.
- A hand-written `Display` impl remains for `SourceError` while `thiserror` is a workspace dependency.
- `golem plan` prices `staging_ring` at 2 max planes (3.69 GiB on the target) while the engine's staging ring allocates `STAGING_SLOTS` = 6 slots (≈ 11.1 GiB at UD-Q4_K_XL), so the RAM tier under-prices prefill streaming by ≈ 7.4 GiB.
- Hardware evidence for the shared-queue store's live failover demo and cold decode before/after is not committed.
