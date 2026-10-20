# Golem — Technical Design

**Status:** Approved-for-spike (v1.0), amended 2026-09-29 from the independent design/code review (findings C/H/D/R; the still-open ones are in §17; every amended passage is marked *Amended 2026-09-29* with its finding ID where one applies; **user decisions O1–O7** ruled 2026-09-29 (O7 2026-10-03, re-ruled 2026-10-17: no quant floor) and collected in §1; O1 and O6 ruled 2026-10-03 on the M3.5 numbers; amended 2026-10-03 with prefill layer streaming §7.6, shared-queue striping §5.2, quant floor O7 and rejected option F (ExLlamaV3); amended again 2026-10-03 from external reference designs (§16): layer-batched cold reads §3.4 P3, per-step timing §11/§12.3, routed-only short prefill §7.6, policy replay + async VRAM promotion §6.3, prefetch kill switch §6.5, MTP determinism and draft length §7.5/O6, CPU/DMA miss split §2.4, A/B method §12.3; amended 2026-10-06 from the M3.3 store measurements: rulings O8–O10, measured two-drive QD1 bandwidth §2.4, CPU+DMA split re-estimate §2.4, exclusive RAM/VRAM residency and routing profile §6.3, substrate MTP + prompt lookup §7.5, top-k KL parity §12.3, open review findings §17); milestone exit criteria live in §14.
**Spec:** `product-spec.md` (local) — read alongside; this document resolves the questions the spec defers to technical design.
**Date:** 2026-09-19

---

## 1. Decision summary

Every decision the product spec leaves open, resolved. Rationale in the referenced sections.

| # | Decision | Choice |
|---|----------|--------|
| D1 | Implementation language | **Rust** (stable, edition 2024) for everything except the vendored compute substrate |
| D2 | Inference engine | **Thin fork of llama.cpp (pinned commit) as compute/kernels substrate**, embedded in a Rust process via C ABI; Golem owns placement, paging, I/O, serving, CLI (§3) |
| D3 | AMD compute backend | **ROCm/HIP** on gfx1100 (RX 7900 XTX). Vulkan/wgpu deferred; NVIDIA later = same fork built for CUDA (§3.5) |
| D4 | v1 weights | **`unsloth/GLM-5.3-Flash-GGUF`, UD-Q4_K_XL** as the default; any quant of that repo may run (*amended 2026-10-17*, O7), exact repo revision + SHA-256 pinned in the model manifest at M0; the in-house quantizer is **not** v1 scope (§4) |
| D5 | Caching algorithm | Decayed-frequency score (LRU×LFU hybrid) + relative swap margin (amended 2026-09-29, D-9; was θ-hysteresis) + min-dwell, per (layer, expert) key, independently per tier; user limits are **caps** (`--vram-experts-gib`, `--ram-experts-gib`); floors dropped (O3, §6) |
| D6 | Multi-disk mode | **Full mirrored replicas**, striped across drives through one shared chunk queue (amended 2026-10-03, §5.2); manifest-validated; degrade-on-drive-failure (§5) |
| D7 | OpenAI API subset | Chat completions (stream + non-stream), `/v1/models`, `/health`. Tools **supported** via the pinned chat template (O2). Reasoning → `reasoning_content`. Strict-allowlist field policy (§8) |
| D8 | CLI semantics | `run` = attach to a compatible instance, else run in-process in the foreground (`--keep` leaves a server; O4); `serve` foreground by default (`--detach` supported); `list` shows instances by default, models with `--models`; `stop` = graceful terminate, `--force` for SIGKILL (§9) |
| D9 | Concurrency model | One process per instance; Rust-driven decode loop over libllama; dynamic micro-batching; tokenization + sampling via llama.cpp for bit-parity with the reference runtime (§7) |
| D10 | Disk I/O engine | Blocking reader threads doing `pread64` into an **mlock'd, GPU-pinned** RAM expert pool, 1–2 in-flight reads per drive; `O_DIRECT` where the filesystem allows it (else `DONTNEED`); io_uring dropped from v1 (§5.4; amended 2026-09-29) |
| D11 | Instance registry | JSON manifests + PID heartbeat + Unix-domain control socket under `$XDG_RUNTIME_DIR/golem/` (§9.4) |
| D12 | Substrate patch seam | Patches to the fork's split-scheduler expert-input path (P1–P5), behind a per-model C-ABI context; policy state computed in Rust, applied in C. *Amended 2026-09-29:* no longer "four surgical patches, ≤1k lines" (§3.2, §3.4) |

Non-decisions, deliberately: no performance targets (spec §Acceptance — they come from M0 measurement); no second model family commitment (spec treats each as a scoped future effort).

**Open decision (amended 2026-09-29, review D-1):** whether I2 (GPU-only expert math) stays absolute. Arithmetic in §2.4 puts GPU-only decode at a PCIe ceiling of ≈5 tok/s at zero VRAM hit rate, against ≈12–16 tok/s for upstream's CPU-hybrid (`-ncmoe`) at the same RAM footprint. Resolved by the M3.5 benchmark gate (§14) **before M4 starts**; the user decides on the measured numbers. **Resolved 2026-10-03 (O1 below):** I2 is amended with an opt-in, decode-only CPU path; the default is unchanged.

**User decisions** (ruled 2026-09-29; each is marked *Decided (On)* where it applies):

| # | Question | Ruling | Where | Review |
|---|----------|--------|-------|--------|
| O1 | Keep I2 (GPU-only expert math) absolute, or allow an explicit, reported CPU path for non-resident experts | **Ruled 2026-10-03: amend I2, opt-in.** Add a decode-only CPU expert-math path for experts not resident in VRAM (`--cpu-expert-math`, **off by default**, reported in metrics). The default keeps I2 and I1/T3. With the flag on, I1 weakens to per-cache-state determinism (CPU and GPU kernels round differently). Prefill stays GPU-only. Build it after H-6/H-8/H-9. Make it the default only if it measurably beats GPU-only with a pooled-RAM leg, which M3.5's D-1 leg did not test. M3.5 evidence: pooled GPU-only decode 0.33–0.40 tok/s cold ≈ 2 GB/s of expert bytes against ≈25 GB/s PCIe, so the bottleneck is disk misses, not compute location. The unpooled leg thrashed mmap (172.8 GiB of experts > 93 GiB RAM), so it is not a CPU-math measurement (`docs/baselines/m3.5-bench.md`) | §1, §2.3 I2, §2.4 | D-1 |
| O2 | Support `tools`/`tool_choice` in v1 via the pinned chat template | **Support in v1** | §8.2 | R-14 |
| O3 | Keep tier **floors** (`--*-experts-min`) or ship caps only | **Caps only; floors dropped** | §6.3 (also §2.2, §4.3, §6.4, §6.6, §9, §10, T3) | R-7 |
| O4 | `golem run` with no running instance: spawn a detached server, or run in-process in the foreground | **In-process, foreground**; attach if a compatible instance exists; `--keep` leaves a server | §2.1, §9 | R-12 |
| O5 | Second instance whose planned VRAM/locked RAM exceeds physical: warn or block by default | **Block by default**; `--allow-overcommit` | §9.3 | R-13 |
| O6 | MTP speculative decoding as a v1 track | **Ruled 2026-10-03: deferred, spike first.** M3.5 could not measure it: no MTP draft or verify code exists. Spike: from P3 routing traces, measure the routed-expert union of consecutive decode tokens per layer. Adopt MTP as a track only if the union is ≤ ≈1.3× one token's expert bytes. Arithmetic: one MTP layer gives 1 draft token; at ≈85–90% acceptance (DeepSeek-V3, not measured here) that is ≈1.85 tokens per verify pass, so speedup ≈ 1.85 / (union ÷ single-token bytes), i.e. ≈1.15× at a 1.6× union and ≈1.4× at 1.3×. *Amended 2026-10-03 (BMoE, §7.5):* external acceptance is below the borrowed 85–90% (BMoE 71/60/52% at draft lengths 2/3/4), one NextN layer can draft k > 1 tokens recursively, and on llama.cpp kernels MTP-verified output need not be bit-identical to plain greedy (a k-token verify batch changes reduction order, §7.4; BMoE saw it even between ubatch sizes). So the spike measures the union ratio U_k for windows k ∈ {2, 3, 4}; the bar generalizes to expected tokens per pass ÷ U_k ≥ ≈1.4; and adoption means an opt-in, reported `--mtp` with the same per-configuration determinism exception as O1. **Ruled 2026-10-07 (M3.10 spike): rejected for v1.** Measured U_2/U_3/U_4 = 1.71/2.33/2.91, so even at 100% acceptance tokens per pass ÷ U_k ≤ 1.17/1.29/1.37 < 1.4 (`docs/baselines/m3.10-mtp-spike.md`) | §7.5 | R-2 |
| O7 | Expert quant floor: may Golem trade quality for bytes below UD-Q4_K_XL? | ~~Ruled 2026-10-03: no. UD-Q4_K_XL is the floor; no ~3-bit (Q3/IQ3) expert quant.~~ **Re-ruled 2026-10-17: yes.** No quant floor: any GLM-5.3-Flash quant may run, UD-Q4_K_XL stays the default. Measurements name the quant they ran on | §4.1, §2.4, §14 M3.5 (d), §15 | D-5 |
| O8 | What decides between competing speed levers? | **Ruled 2026-10-06: decode performance first.** Adopt whatever measurably gives the most decode tok/s at the default UD-Q4_K_XL (O7), judged by the §12.3 A/B. The GPU-only default (I2) stays in principle; O1's opt-in CPU+DMA split becomes the default only if it wins that A/B. §2.4 re-estimates its gain at ≈+7–10% at this RAM size (20–30% of expert bytes from disk), +42–47% at 10%, and 1.7–2.5× only as the disk share approaches 0 | §2.4, §14 | — |
| O9 | MTP (O6) before or after the O1 CPU+DMA split? | **Ruled 2026-10-06: before**, as long as it stays cheaper and the U_k spike shows a gain. The vendored substrate already carries a GLM5NEXT MTP draft context (§7.5), so the cost is wiring, not a draft model | §7.5, §14 | — |
| O10 | Inclusive or exclusive RAM/VRAM expert residency? | **Ruled 2026-10-06: exclusive**, as a separate task after the M3.7 policy replay sizes the gain. Today an expert in VRAM keeps its RAM pool slot, so ≈750 VRAM slots × 15.2 MB ≈ 10.6 GiB of the pool (≈17% of a 63 GiB pool) is duplicated. **Re-ruled 2026-10-08 on the M3.15 A/B: inclusive stays the default**, exclusive is opt-in (`--residency exclusive`): −19% cold tok/s for −2.6% disk bytes, because kept evictions D2H on the refill stream (`docs/baselines/m3.15-exclusive-residency.md`) | §6.3 | — |

The decision table rows D5, D7, D8, D10 and D12 are also affected by the amendments below (§6, §8.2, §9, §5.4, §3.4).

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
│  │ placement manager      │◄──┤ policy state (scores,       │◄──┘       │  │
│  │ (Rust)                 │   │ residency bitsets, caps)    │ C ABI     │  │
│  └─────────┬──────────────┘   └─────────────────────────────┘           │  │
│            │ page requests                                              │  │
│  ┌─────────▼──────────────┐                                             │  │
│  │ store (multi-disk I/O) │── pread64, striped across replicas ──► disk │  │
│  └────────────────────────┘                                             │  │
└────────────────────────────────────────────────────────────────────────────┘
        ▲ libllama (vendored fork, HIP backend) linked into the process
```

`golem list` / `golem stop` / `golem plan` are thin clients: the first two talk to the registry (§9.2), the last is a pure offline computation (§10).

*Amended 2026-09-29 (review R-10):* at HEAD there is **no placement thread**. RAM residency transitions happen inside `ResidencyMap::claim` on the decode thread, and `golem-placement::Policy` has no production consumer. The "placement manager" in the diagram is the target, not the current state.

### 2.2 The three tiers

| Tier | Backed by | Managed by | Size |
|------|-----------|------------|------|
| Disk | GGUF shard files on 1..N drives, mirrored | `golem-store` | full model (~165–200 GB est. for Q4_K_XL-class quant; authoritative numbers come from the GGUF tensor table, never hardcoded) |
| RAM | one mlock'd expert pool per instance (host buffer of the vendored substrate) | `golem-placement` + store | default: min(75% of free RAM at start, total expert bytes); `--ram-experts-gib` override |
| VRAM | device expert-copy cache (persistent across steps) + non-expert weights + KV + workspaces | scheduler patches (P2) under Rust policy | `--vram-experts-gib` budget (floors dropped, O3) |

Everything that is **not** a routed-expert weight (embeddings, attention projections, the linear-attention/indexer tensors, norms, router, output head, and the per-layer **shared expert** `ffn_*_shexp`, which runs on every token: review R-3) is loaded into VRAM at startup and stays there for the life of the instance. Only expert FFN weights participate in tiering — placement is per-expert, never per-layer (spec requirement). *Amended 2026-09-29 (review R-2):* the MTP/NextN block (blk.45) carries its own routed experts; it is skipped at load while `load_mtp = false` and must be **excluded from pool geometry** (it is today counted as a 43rd expert layer). If MTP speculative decoding is adopted (§7.5, O6), its experts join the tiers as layer 42.

### 2.3 Life of a decode step

1. Slot scheduler assembles the next-token batch for all active slots (§7.2).
2. `llama_decode` builds the graph and runs it on the HIP backend. Attention, router, and all shared weights live entirely in VRAM.
3. When the graph reaches a layer's `GGML_OP_MUL_MAT_ID` (expert FFN), the scheduler's expert-copy path needs that layer's selected experts resident in VRAM:
   - **P3 hook (host residency):** before reading host bytes, call into Rust. Fast path: residency bitset hit (no-op). Slow path: synchronous striped `pread64` from replicas into the RAM pool, then return. A prefetch thread (§6.5) aims to make the slow path rare. That is a hypothesis to measure, not a guarantee.
   - **P2 patch (device resident-set):** copy only the byte ranges of experts whose device copies are invalid; skip ranges already resident in the device expert-copy cache from previous steps. Evictions happen here when the device cache exceeds its budget, in the order Rust's policy thread supplied.
4. Routing ids are produced by the router matmul inside the graph from activations, exactly as the reference model computes them — Golem never inspects, filters, or overrides ids. Access notifications (`notify-access(expert_id)`) are emitted to Rust per step, feeding the frequency scores. *Amended 2026-09-29 (review R-6):* replace per-expert `notify_access` callbacks (≈1,000 FFI calls per token) with one read of each MoE layer's ids tensor per step, and push the eviction order to C as model-context-owned data updated off the hot path, instead of the per-tensor `policy_query` callback.
5. Sample → detokenize → SSE frames.

**Invariant I1 (routing invariance):** caching affects only *where selected experts' bytes are staged from*. Weight bytes are bit-identical across tiers (verified by manifest hashes), the kernels and op order are identical (same vendored substrate), therefore activations, router logits, and expert selection are identical regardless of cache state. Tested explicitly (§12.2, T3).

**Invariant I2 (no CPU expert math):** expert tensors are attached to the graph as host-buffer inputs consumed by device-side `MUL_MAT_ID` — the CPU never executes an expert matmul. If the HIP backend is unavailable at startup, the instance **fails to start** with an actionable error (§8.4, `golem_backend_unavailable`); there is no CPU fallback path in the binary for expert math. `golem plan` rejects configurations that would require one. *(O1, ruled 2026-10-03: the default stays as written; an opt-in, decode-only `--cpu-expert-math` exception is planned. See §1.)*

### 2.4 Throughput ceilings (amended 2026-09-29)

These are first-order bounds for UD-Q4_K_XL on this machine. They are arithmetic, not targets: the governing numbers each design choice below must move.

- **Bytes per token.** One routed expert costs gate 4.72 + up 4.72 + down 5.77 ≈ 15.2 MB per layer. × 8 routed × 42 trunk MoE layers ≈ **5.1 GB of expert weights per token**. The GGUF carries 43 expert layers, but blk.45 is the MTP/NextN block, which the loader skips while `load_mtp = false`.
- **PCIe ceiling (GPU-only, I2).** Every expert not already in VRAM crosses PCIe 4.0 x16 (≈25 GB/s pinned, markedly less when pageable). Decode ≤ 25 / (5.1 · (1 − h)) ≈ **4.9 / (1 − h) tok/s**, where h is the VRAM hit rate. After non-expert weights (8.35 GiB), ROCm overhead (2.64 GiB), KV and workspace, about 8–11 GiB remains for experts, i.e. ≈550–750 of 12,096 trunk experts (≈5%). So h is bounded by routing skew and must be *measured* (P4 counters), not assumed. *(M3.5 measured 27 GB/s pinned, so the ceiling is ≈5.3 / (1 − h); the §2.4 split table uses 27.)*
- **Disk-miss cost (review D-6).** One expert-layer from disk ≈ 15.2 MB / 5.8 GiB/s ≈ 2.4 ms (nvme0n1), ≈1.8 ms with shared-queue striping across both drives (§5.2). At 336 expert-layer loads per token, a 5% disk-miss rate adds ≈35–40 ms/token (cap ≈25 tok/s); 20% caps decode at ≈6 tok/s. *Amended 2026-10-06 (M3.3 measured, `docs/baselines/m3.3-store-bench.md`):* the shared queue reads at **4.73 GB/s** across both drives at QD1 per drive with `O_DIRECT` (nvme0n1 alone 3.6–5.0, nvme1n1 alone 1.74). That is ≈3.2 ms per 15.2 MB unit, not 1.8, and two reader threads per drive regress (0.70 GB/s). Every disk term in this document that used 8.05 GiB/s or 8.64 GB/s is about **1.8× optimistic**. Disk time per token at RAM-hit rate r is 5.1 · (1 − r) / 4.73 GB/s: r = 0.6 → 431 ms (≤ 2.3 tok/s), r = 0.8 → 216 ms (≤ 4.6 tok/s). Per-drive queue depth is the remaining I/O lever.
- **CPU-hybrid reference (what I2 forbids).** CPU matvec over RAM-resident experts is DDR5-bound (≈60–80 GB/s on the 9950X3D) → ≈12–16 tok/s. That is ≈2.5–3× the GPU-only PCIe ceiling at low h.
- **Prefill (review D-7).** At ubatch ≥ 32 almost all 288 experts are routed per layer, so a naive per-ubatch copy moves the whole expert set (≈184 GB) over PCIe (≈7–8 s per 512-token ubatch). Prefill therefore needs its own strategy: layer-by-layer expert-sorted streaming through a double-buffered device slab, or deliberately small ubatches. Measure before choosing. *Amended 2026-10-03:* M3.5 measured the small-ubatch option (chunk cap 1) at **0.24 tok/s cold**, so prefill uses **layer streaming** (§7.6). At ubatch 2048, 16,384 routings over 288 experts leave P(expert unused) ≈ e⁻⁵⁷ ≈ 0, so each ubatch moves the full set: 644,022,272 B × 288 = **185.5 GB**. PCIe: 185.5 / 27 GB/s (measured pinned) ≈ 6.9 s. Disk: the pool holds 4,550 / 12,096 units (37.6%), so misses are ≈115.7 GB (107.8 GiB) / 8.05 GiB/s (both drives, §5.2) ≈ **13.4 s per ubatch**. Disk is the bound, and PCIe plus compute overlap it. That gives ≈150 tok/s at ubatch 2048 and ≈38 tok/s at 512, before GPU compute is measured. *Amended 2026-10-06 (M3.3 measured):* at 4.73 GB/s the disk term is 115.7 GB / 4.73 ≈ 24.5 s per ubatch, which gives **≈85 tok/s at 2048 and ≈21 at 512**. External reference, not this machine: a 5090 on PCIe 5.0 with two NVMe drives and the same scheme reports ≈230–250 tok/s at ubatch 2048 (§16).

Consequence: I2 as written trades roughly 2.5–3× decode throughput for "all expert math on GPU". *O1 ruled 2026-10-03 (§1): this amendment is adopted as an opt-in, decode-only path.* The amendment text: "GPU when the expert is VRAM-resident or when the transfer is cheaper than CPU compute; CPU otherwise, as an explicit, configured, reported policy, never a silent fallback."

*Amended 2026-10-03: under the O1 opt-in, split RAM-resident misses between CPU and PCIe, not all-CPU.* Each layer's misses are split: VRAM hits run on the GPU, a measured fraction f of misses is DMA'd to the GPU on the copy engine (issued as soon as the ids are known), and the rest runs on the CPU, all concurrently. The best f depends on what the CPU kernel is bound by: lower when it is RAM-bandwidth-bound (the DMA competes for the same DRAM), higher when it is arithmetic-bound. With the GPU also pulling from RAM, the CPU's stream rate drops below its solo rate, so the 60–80 GB/s above is an upper bound; the arithmetic below assumes ≈41 GB/s concurrent, and M3.9 measures it. Arithmetic for Golem at h ≈ 0 with all misses RAM hits: per token 5.1 GB; all-PCIe 5.1 / 27 ≈ 189 ms; all-CPU at 41 GB/s ≈ 124 ms; split f to PCIe balances at f = 27 / (27 + 41) ≈ 0.40, giving 5.1 × 0.60 / 41 ≈ 75 ms ≈ 13 tok/s, before compute and launch overhead. Splitting across tokens instead (two token groups of one verify window, one per device) reads the dense weights twice and shares fewer experts per group, so the split is across experts, not tokens. Disk misses are unchanged by any of this: they cost the same wherever the math runs.

*Amended 2026-10-06 (O8): what the split buys with disk in the loop.* The h ≈ 0, all-RAM-hit figure above is the best case. With a disk share d of the expert bytes, per layer and without lookahead, the RAM-hit transfer overlaps the disk read of the misses, and the misses transfer or compute after they land. That gives t = max(R / X, D / 4.73) + D / X per token, where:
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
- Lowering d (bigger pool, exclusive residency, routing profile, lookahead) is the first lever. The split is the second.
- Concurrent DDR5 traffic (CPU 41 + PCIe 27 + disk DMA 5) ≈ 73 GB/s, against ≈96 GB/s theoretical on dual-channel DDR5-6000, so M3.9 must measure the CPU rate *under* concurrent DMA.

### 2.5 The model's actual architecture (amended 2026-09-29, review R-1)

The original text assumed an MLA/KV-cache model. GLM-5.3-Flash (`glm5next`) is a **hybrid**:

- 45 trunk layers (`config.json` `layer_types`): **34 `linear_attention`** layers with a fixed-size recurrent state, and **11 `deepseek_sparse_attention`** layers with MLA-style K-only KV plus a sparse-attention indexer (`indexer.*` tensors). Layers 0–2 are dense, layers 3–44 are MoE (288 routed + 1 shared, top-8), and blk.45 is the MTP/NextN block.
- Memory per sequence is **constant-size recurrent state** (m0-notes: 145.56 MiB R/S state) **plus** KV for the 11 attention layers only (19,712 B/token measured at f16; 14,432 B/token at q8_0, §10).
- Recurrent state **cannot be partially rewound** (`llama-memory-recurrent.cpp:182`). Anything that assumes "truncate the KV to position p" (slot cancel with rollback, stop-sequence rollback, prefix reuse) must instead **snapshot and restore** state (§7.1).
- Long-context cost is governed by the sparse indexer, not dense attention, so the §12.3 ctx sweep must be chosen from measurement.

Every "MLA" reference below means the 11 sparse-attention layers only.

*Amended 2026-10-20 (#165): DeepSeek-V4.1-Flash (`deepseek41`).* A second supported architecture: 40 blocks, n_embd 5120, 64 heads MLA (kv 512, q_lora 1280), 384 routed experts top-6 + 1 shared (expert FFN 2304, sqrt-softplus gating), SWA 128 with per-layer compressed attention plus a 32-head indexer (top_k 512), 4-way hyper-connections, and an engram table (`per_layer_token_embd`, ≈ 126 GiB) read lazily from disk and never placed in VRAM. Its KV is priced per cache by `golem-plan` (`dsv41_state_terms`): raw SWA rows `pad256(n_swa + n_ubatch)` plus compressed rows `n_ctx / ratio` per layer; target at q8_0 ≈ 1,700 B/token. Expert planes are mixed-quant (IQ2/MXFP4); the stream slabs carry one stride of slack for HIP's 512-element row padding (expert ne0 2304 → 2560). P5 prediction nodes mirror its router (sqrt-softplus + `exp_probs_b`). Unmeasured: the `ws` compute-buffer fit (fitted on GLM), cold/warm throughput, snapshot restore (diverges on hybrids at this pin; serve with snapshots off).

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
| F. ExLlamaV3 (ROCm port: bsvinay/exllamav3-rocm-moe) instead of llama.cpp (added 2026-10-03) | rejected | Its RDNA3 kernels and `glm5_next` + MTP support are real, but: (1) its MoE speed comes from running cold experts on the CPU, which I2 forbids by default; (2) it has no disk expert tier, since every expert is pinned in RAM (85 GB model), while ours needs ≈185 GB of experts streamed against 93 GiB of RAM; (3) it is a Python/PyTorch extension with no C ABI, so embedding it costs about as much as option B; (4) it is EXL3-only, with no GGUF, so the pinned manifest, baselines and T2 parity against the reference runtime are all lost; (5) its main advantage, sub-4-bit quality, is moot under O7; (6) it is a single-maintainer fork. Its ideas are adopted instead: large prefill chunks (§7.6), persisted learned placement (M2.5), MTP (O6). Revisit only if llama.cpp's HIP MoE kernels prove the bottleneck at UD-Q4_K_XL |

### 3.2 The fork

- *Amended 2026-10-20 (#165):* re-pinned to ggml-org/llama.cpp PR #30817 head `b72050d` (base `f174553`), which adds `deepseek41` and upstream `glm5-next` (the unsloth `glm5next` name is folded onto it). Golem's patches were replayed; `engine/llama-cpp.lock` is authoritative. The original pin follows.
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
  patches = none-yet      # pristine upstream-at-pin; P1–P4 land in M2.x, P5 + rework per the 2026-09-29 amendment (§3.4)
  # fa = <set at M0.4>    # FlashAttention/precision setting for MLA on ROCm (§3.5)
  ```

  *(values as of the M0.1 pin; `engine/llama-cpp.lock` is authoritative on divergence)*

- The vendored tree carries no nested `.git` by design, so the acceptance is content-based, not `git rev-parse`: CI's `vendor-pin-check` recomputes `content_sha256` from `vendor/llama.cpp` (canonical recipe, ADR 0001) and it must match `vendor/VENDOR_SHA` and the lock.
- The fork is vendored in-tree (`vendor/llama.cpp/`) and built by the `golem-llama-sys` build script via the `cmake` crate with `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100`.
- Patch discipline: total diff vs upstream ≤ ~1,000 lines, confined to the files listed in §3.4. *Amended 2026-09-29 (review D-14):* this budget no longer holds. The required P2/P3 rework (pool as H2D source, slot-indirected device cache, per-model hook context, a post-router P5 hook, possibly prefill streaming) reworks how the split scheduler handles expert inputs, and `ggml-backend.cpp`'s scheduler is one of upstream's most-changed files. Re-estimate the maintenance cost in the §3.4 amendment PR, and isolate the delta behind one upstream-shaped extension point (a custom buffer type plus an `MUL_MAT_ID` input resolver). **Trigger to re-open D2 option B:** the scheduler diff passes ~1.5k lines, or two consecutive rebases conflict in it. Every patch has an in-tree regression test (§12.2, T4). Upstream rebase is a scheduled maintenance task (monthly + on GLM-arch fixes), never an ad-hoc drift.
- The fork carries **no** model-architecture code of its own and **no** product logic in C++ beyond the hooks. If a future rebase turns patches into a maintenance sink (>2 consecutive rebase failures), that is the documented trigger to re-open decision D2, option B.

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

**P1 — hook plumbing.** `ggml/src/ggml-backend.cpp` + a new `ggml/include/golem-hooks.h`: registers three C function pointers (`ensure_host_resident`, `policy_query`, `notify_access`) and per-tensor expert metadata (expert count, per-expert byte size, stride) at buffer-usage-weights time. No behavior change when hooks are unset (upstream-semantics default), which keeps upstream tests green.

**P2 — persistent device expert-copy cache.** In the split-scheduler expert-copy path (the code that already groups consecutive routed experts and calls `ggml_backend_tensor_set_async`): maintain, per weight tensor, a validity bitset over expert ranges in the device copy buffer. Copy only invalid ranges of the routed set; when the total valid bytes exceed the device budget, evict in the order most recently supplied by `policy_query` (Rust), i.e. true LRU-of-policy ordering applied in C with no callback latency in the hot loop. Previously-copied hot experts are skipped entirely — this is the mechanism that makes "hot experts live in VRAM" real without touching graph construction.

**P3 — host residency callback.** Immediately before the host→device range copy: `ensure_host_resident(tensor, first_id, last_id)` → Rust blocks until ranges are paged into the mlock'd RAM pool (bitset fast path; synchronous striped pread slow path). Callback runs on the decode thread; §6.5 explains why the blocking slow path is rare.

**P4 — counters.** Per-step counters (bytes copied host→device, bytes skipped because resident, page-ins served from RAM vs disk, evictions) exposed through a lock-free stats block read by Rust; exported as metrics (§11).

*Amended 2026-09-29: required changes to P1–P4, and a new P5.*
- **P1 (review C-1, R-5, H-10, R-10).** Expert metadata is queried lazily at the first callback, not "at buffer-usage-weights time". The name match must accept the real runtime names (`blk.N.ffn_*_exps.weight`); today it matches none of them. Every hook takes a `golem_ctx*` owned by one `llama_model`. Registration returns a handle, and unregister waits for in-flight callbacks. No process-global hook state, device-cache map, budget or stats.
- **P2 (review C-3, H-5, M-13).** A full-size device twin per expert tensor totals ≈184 GB on this model and OOMs a 24 GB card. Replace it with a **slot-indirected device cache**: per (layer, kind) a device slab of `K` expert slots sized from the VRAM budget, a host-side `expert → slot` map, and `MUL_MAT_ID` fed the slab as `as` (`ne[2] = K`) plus a remapped 2-D ids tensor (values in `[0, K)`). This also replaces upstream's full-size transient `input_cpy` for expert inputs. The budget is a per-(layer, kind) slot count, not a byte cap derived from one tensor's stride.
- **P2, further amended 2026-10-01 (review D-11, implemented):** the slot table is **one per layer, shared by gate, up and down** (safe: the three tensors consume the same `selected_experts` ids per step), while each tensor keeps its **own per-slot validity bits and slab**; an eviction clears the slot's bits in all three. The P1/P2 ABI is unchanged (no signature change); the policy order supplied by Rust becomes per layer.
- **P3 (review C-2, H-3).** The host expert bytes live in a read-only file mmap (writing them SIGSEGVs), and a writable host buffer of model size cannot fit in RAM. So **the RAM pool is the H2D source**: `ensure_host_resident` returns a pool pointer per expert, `copy_experts_to` copies from it, and the lease is held until the async copy completes (a `copy_done` hook after the scheduler's event sync). Expert tensors must never be populated from the file by the loader: no `MAP_POPULATE` of expert ranges (use the fork's lazy-tensor path, `set_tensor_data` loading, or a zero-backing buffer type).
- **P3, further amended 2026-10-03 (BMoE, §16): layer-batched cold reads.** The slab path calls `ensure_host_resident` once per missing expert (`ggml-backend.cpp`, slot-slab fill loop), and `page_in` reads gate, up and down one after another, so a cold layer is up to 8 × 3 = 24 blocking reads of 4.7–6.9 MB at queue depth 1 while the GPU idles and the prefetcher stands down (demand gate). All of a layer's misses are known before the first read. So P3 takes **one call per (layer, step)** with the layer's full miss list; Rust submits every plane of every missing expert to the §5.2 shared queue at once and returns when all have landed (per-expert leases unchanged). This is a P1 ABI change, bundled with the M3.8 ABI bump. Arithmetic: a fully cold layer is 8 × 15.2 MB ≈ 122 MB; at the M3.5-measured ≈2 GB/s that is ≈61 ms, at both drives' 8.05 GiB/s ≈14 ms. At zero RAM hits a token is 5.1 GB / 8.64 GB/s ≈ 0.59 s ≈ 1.7 tok/s ceiling, against the 0.33–0.40 tok/s measured. BMoE got ≈3× from parallel read lanes on phone storage.
- **P5, new (review H-6).** A post-router hook per MoE layer for next-layer routing prediction (§6.5).
- **P2 prefill-stream mode, new (amended 2026-10-03, D-7).** When a ubatch is above the streaming threshold (§7.6), the expert input for each plane comes from a **stream slab** instead: the same slot-slab machinery with `K = n_expert` and identity ids, so no ids remap is needed and the extension point stays the same. Two stream slabs, each sized for the largest plane of any class, are ping-ponged: plane *p* of layer *L* computes while plane *p+1* (or layer *L+1*'s gate) fills over H2D on a separate stream, event-synchronized. P2 slot-cache slabs are left untouched during prefill, so the decode hot set survives a long prompt.
- **Engine-side invariant (review C-4).** The engine forces every expert `MUL_MAT_ID` onto the GPU (upstream keeps it on the CPU below batch 32) and verifies after the first decode that it ran there. Failure → `golem_backend_unavailable`.

### 3.5 Backend: ROCm/HIP, and what "not NVIDIA" means

- v1 target: **gfx1100 (RX 7900 XTX)**, ROCm **7.2.4** as the advisory minimum recorded in `engine/llama-cpp.lock` (M0.2 warns on `rocminfo` vs lock mismatch and skips the check when `rocminfo` is absent — CI/no-GPU), HIP backend of the fork with `AMDGPU_TARGETS=gfx1100`.
- The known NVIDIA-specific workarounds from PR #27754 (`NVIDIA_TF32_OVERRIDE=0`, `-fa off` for MLA) are CUDA-specific. Their ROCm analogues (hipBLASLt reduced-precision options, FlashAttention on MLA) are **unknown on gfx1100** — resolving them with correctness evidence is an explicit M0 exit criterion (§14), not an assumption. Whatever setting M0 validates is written into `engine/llama-cpp.lock` and made the default; a config escape hatch (`--fa {on,off}`) stays exposed.
- Vulkan: out of v1. The seam is clean — the substrate's Vulkan backend is a build-flag swap, and Golem's policy/patch seam (P1–P5) is backend-agnostic. When Vulkan K-quant kernels mature, support is a build+validation effort, not a redesign.
- NVIDIA: same story — the fork's CUDA backend is a build-flag swap of the same kernels. No NVIDIA-specific code anywhere in Golem's Rust layers.
- **ROCm/HIP stays the right v1 choice for gfx1100** (amended 2026-09-29): its K-quant MMQ/MMVQ kernels are mature. Vulkan remains a build-flag swap.
- **AMD-specific requirements** (review H-4, C-4, D-8):
  - **Pinned host memory.** The RAM expert pool is mlock'd *and* registered with `hipHostRegister` (portable). Pageable H2D goes through driver bounce buffers, runs at roughly half of PCIe bandwidth and is not truly async, and PCIe is the §2.4 bottleneck. Record pinned vs pageable H2D GB/s on the 7900 XTX in the baselines. Consider `MADV_HUGEPAGE` on the arena before mlock.
  - **HIP graphs.** `GGML_HIP_GRAPHS` defaults ON. A blocking host callback inside the scheduler (P3) must never run inside a captured region: verify this, or disable graph capture for split graphs.
  - **Op-offload threshold.** Upstream keeps `MUL_MAT_ID` with host-resident weights on the CPU backend below batch 32 (`GGML_OP_OFFLOAD_MIN_BATCH`). The engine (not tests) must force device placement and **verify at runtime** that every expert `MUL_MAT_ID` ran on the GPU. Otherwise I2 is silently violated.
  - Backend-specific pieces (pinned-memory registration, the P2 slot remap) stay behind the P-hook seam, so Vulkan and CUDA can implement them later.

### 3.6 Evidence base (from context7, 2026-09)

*Amended 2026-09-29 (review R-16):* substrate claims must cite the **vendored tree** (`rg` in `vendor/llama.cpp`, file:line), not doc summaries. Correction to the `--lazy-mode` bullet below: the loader supports per-tensor on-demand reads via `TENSOR_READ_LAZY` (`llama-model-loader.cpp:1335`), and an arch file can set it on any tensor, including experts. It is relevant to P3's load-mode fix. No vendored arch sets it on experts today.

- llama.cpp feature matrix: ROCm column — K-quants ✅, I-quants ✅, MoE ✅, Flash Attention ✅, K-cache quants ✅; Vulkan K-quants marked slow ("the CUDA or ROCM backends are faster"). This is the empirical basis for D3.
- `ggml-backend.cpp` split scheduler: when MoE weights are offloaded to a host buffer, it reads the ids tensor, builds a used-expert bitset, groups consecutive ids, and copies **only those ranges** host→device per eval (`ggml_backend_tensor_set_async`), including small padding guards for MMQ. This is upstream's existing just-in-time cold-expert-to-GPU path; Golem's P2/P3 wrap policy around it.
- `--n-cpu-moe` / `-ot exps=CPU` are shipped, exercised configurations that put expert weights in host buffers while the graph runs on GPU — i.e., the "cold expert computes on GPU" configuration is upstream-supported and debugged, not speculative.
- `--lazy-mode` (on-demand reads of selected tensors from disk) exists upstream but ~~only for specific tensor kinds (e.g. oversized embeddings), never experts~~ *(corrected above: any tensor an arch marks `TENSOR_READ_LAZY`)*, and rides the kernel page cache. Golem's explicit RAM pool (§5.4) supersedes it for experts: bounded memory, multi-disk striping, and accounting — three things page cache cannot give us.

---

## 4. Model loading and GGUF handling

### 4.1 v1 weights (D4)

- Primary: `unsloth/GLM-5.3-Flash-GGUF`, quantization **UD-Q4_K_XL** — the best capacity/quality tradeoff for a 24 GB VRAM + disk-tiered architecture (≈4.5–5 bpw keeps the resident-set calculus meaningful). Exact repo revision and per-shard SHA-256s are pinned into the model manifest during M0 (the repo carries several non-interchangeable variants; the manifest makes the choice auditable and `golem plan` exact).
- Auxiliary for validation: **Q8_0** of the same revision — used by the parity harness (§12.2) because high-precision quant minimizes quantization noise when bisecting correctness issues across GPU/CPU execution.
- **Quantization is the largest single performance lever** (amended 2026-09-29, review D-5). Bytes per token, and with them the §2.4 PCIe ceiling, the disk-miss cost and the number of experts that fit in RAM/VRAM, scale linearly with expert bits. A ~3-bit expert quant (UD-Q3_K_XL / IQ3_XXS class) cuts ≈30–35% of expert bytes, raises the PCIe ceiling proportionally, and fits ≈50% more experts per tier. Parity is judged against "the same quantized model", so a lower quant does not break acceptance. Its quality cost is gated separately: perplexity delta on the wikitext slice (M5.1). `golem plan` therefore treats the quant as a first-class input and reports the §2.4 ceilings per candidate quant. *~~Decided 2026-10-03 (O7): the lever is closed downward.~~ Re-ruled 2026-10-17 (O7): the lever is open; any quant may run, UD-Q4_K_XL stays the default.*
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

1. ~~Manifest SHA-256 over every shard (streamed, ~1–2 min for ~170 GB on NVMe; `--skip-verify` available and reported loudly in logs).~~ *Amended 2026-09-29 (review R-8):* the full hash is ≈32 s of pure read at 5.8 GiB/s plus ≈90 s single-core SHA, reads 186 GiB off the SSD and flushes the page cache at every start. Full verify happens **at import**. At instance start, check size + mtime + a sampled per-shard hash (e.g. 64 × 1 MiB at deterministic offsets), with `--full-verify` opt-in. Also check the expert layer set against `block_count − nextn_predict_layers`, so the MTP block is never counted as a trunk layer.
2. Expert count and bytes-per-expert uniformity per layer (GGUF guarantee; a violation means a corrupted/mismatched mirror — refuse to start).
3. VRAM budget feasibility: non-expert bytes (incl. shared experts) + KV (11 attention layers) + recurrent state × parallel + workspace + VRAM expert-cache budget ≤ free VRAM minus measured ROCm overhead (M0 number). Failure → `golem_plan_infeasible` error with the exact over-budget line items (§10), never a degraded start.
4. Backend probe: `hipGetDeviceProperties` on gfx1100, kernel self-test (one expert matmul GPU vs known-good CPU result within tolerance). Failure → `golem_backend_unavailable` with install instructions (ROCm version, env vars) in the error body.

---

## 5. Storage tier (`golem-store`)

### 5.1 Mode (D6): full mirrored replicas

v1 supports **N ≥ 1 full copies** of the model on separate paths (the model bytes are identical; the manifest's `replicas` array names them). Mirrored full copies — not sharded placement — because: (a) per-read striping across replicas already extracts aggregate bandwidth, so sharding adds a failure mode without extra bandwidth; (b) replicas double as validation targets for each other; (c) sharding can be layered later without interface changes (`replicas` → `shards` is a manifest evolution, not an API break).

**Same-physical-drive detection:** two replica paths whose resolved `st_dev` maps to partitions of the same parent block device (resolved by walking `/sys/block/<parent>/<partition>`) are flagged at startup: `warn: replicas share a physical device; no bandwidth aggregation`. They still function (fault-tolerance is zero there, and we say so), we just don't advertise aggregate bandwidth or prefer them for striping.

### 5.2 Read path

- Unit of I/O: one expert's byte range within one tensor (from `expert_index`), typically 1–8 MB per expert per tensor per layer.
- ~~A read fans out across healthy replicas in contiguous chunks (chunk count = replica count) issued concurrently.~~ *Amended 2026-09-29 (review H-8):* ~~chunk sizes are **proportional to each replica's measured bandwidth** (probed at open, EWMA-updated from completed reads).~~ The motivating fact still holds: M0 measured 5.81 vs 2.24 GiB/s, so equal halves give ≈4.5 GiB/s, *below* nvme0n1 alone. ~~Below a minimum chunk (e.g. 1 MiB) the read goes to one replica unsplit. The plan is fixed per read and recorded with the request;~~ completion is when all chunks land. Chunk boundaries are deterministic (offset arithmetic), so a retry on another replica is trivially resumable.
- *Amended 2026-10-03 (H-8, supersedes the proportional split):* **shared-queue striping.** Every outstanding read (demand, prefetch, prefill stream) is cut into fixed **2 MiB chunks** on 4 KiB file-offset boundaries, as `O_DIRECT` requires (§5.4). The chunks go into **one queue shared by all replicas**, with demand chunks ahead of prefetch chunks. Each drive runs its 1–2 reader threads (§5.4), and each thread pulls the next chunk into the destination slot when it is free. The faster drive takes more chunks without any bandwidth probe, EWMA or per-read plan: with 5.81 and 2.24 GiB/s it takes ≈72%, so a 15.2 MB unit reads in ≈1.76 ms vs 2.44 ms on nvme0n1 alone. The tail costs at most one slow-drive chunk (2 MiB / 2.24 GiB/s ≈ 0.9 ms) per queue drain, not per unit, and is negligible under prefill streaming, where the queue holds ≈55,000 chunks per ubatch. A chunk that fails on replica X is requeued tagged "not X". A degraded drive self-throttles because it simply pulls fewer chunks, so health needs only a per-replica error count. CRC granularity is unchanged: CRCs stay per expert plane, verified after all of a plane's chunks land (§5.3, review D-4).
- Reads land directly in pre-pinned slots of the mlock'd RAM pool (no bounce buffers), except prefill-stream misses, which land in the pinned staging ring (§7.6); completion updates the residency bitset and releases the placement manager's waiter.
- Errors on one replica (EIO, device gone): mark replica degraded (health state, metric, log), transparently retry the failed chunk on the next healthy replica. If **all** replicas of a shard are unreadable: instance transitions to `degraded:fatal-io` — in-flight requests complete with an error, new requests get `golem_storage_unavailable`; the process stays alive so `golem stop`/`golem list` keep working and the user can pull the drive and recover. (Fail-soft per spec "degrade or fail": we degrade until reads are impossible, then fail loudly but stay manageable.)

### 5.3 Validation

- Full SHA-256 verify at import and at instance start (§4.3). Sizes re-checked on open.
- Continuous in-flight hashing would tax the read path; instead each pool page-in is covered by a cheap **CRC32C** appended into the page table entry, verified **at page-in** (*amended 2026-09-29, review R-8, D-4:* "on use" would mean hashing ≈5 GB per token on the decode thread. CRCs are per fixed unit, i.e. per expert plane, computed at import and stored in the manifest `expert_index`, never per striped chunk). Mismatch → treat as corrupted replica (same path as EIO) and log the offending shard+range. This catches silent rot between full verifies at ~3 GB/s/core cost, negligible against disk reads.
- On any detected mismatch, `golem list` shows the instance as `degraded`, naming the shard and replica — actionable, not silent.

### 5.4 I/O engine (D10)

- **Default:** ~~a fixed pool (default 8, `--io-threads`) of~~ dedicated reader threads doing `pread64` on `O_RDONLY` fds, one per replica file. *Amended 2026-09-29 (review D-12):* M0 measured one thread at 5.81 GiB/s (nvme0n1) and 8 threads regressing to ≈2.0 GiB/s on both drives. The default is **1–2 in-flight reads per drive**, derived from the replica set, re-measured after O_DIRECT lands, with `--io-threads` as an override. Rationale: expert reads are large (MB-scale), few, and schedulable; thread-pool pread is within ~10% of io_uring at this op size, and it keeps the decode-thread slow path (§6.5) simple and bounded.
- The RAM expert pool is one `mmap(MAP_ANONYMOUS | MAP_NORESERVE)` + `mlock` region carved into fixed-size expert slots (per-layer uniform bytes-per-expert from the manifest). mlock prevents the kernel from double-evicting what our own cache thinks is resident — the class of bug that produces nondeterministic "disk reads" inside a supposedly-warm cache.
- Host page cache: *amended 2026-09-29 (review H-9).* M0 **did** show memory pressure (three desktop freezes on 2026-09-21), so expert reads must not populate the page cache. Use `O_DIRECT` straight into the (4 KiB-aligned) pool slot where the filesystem supports it, probed per replica, since btrfs-zstd `/home` returns EIO under concurrent `O_DIRECT`. Otherwise use `posix_fadvise(DONTNEED)` on the range after each read. The loader must not `MAP_POPULATE` expert tensors either (review H-3).
- ~~io_uring (`tokio-uring`) and `O_DIRECT` remain compile-time-available, runtime-opt-in (`--io-engine {threads,io_uring}`, `--direct-io`)~~ *Amended 2026-09-29 (review R-17):* io_uring is **dropped from v1** (M0 never ran the leg, `tokio-uring` is effectively unmaintained, and the design itself puts thread-pool pread within ~10% at MB sizes). `O_DIRECT` is the default where supported (above). Revisit io_uring only if O_DIRECT plus proportional striping leaves measured bandwidth unused.
- *Amended 2026-09-29 (review D-2, D-3, D-11):* slots are **not** one uniform size. Use one slab per plane-stride class, because max-stride slots waste ≈12%. There is no static per-layer quota: one global pool with a global CLOCK, and per-layer locks for residency bits only. The **unit of residency is (layer, expert)**: its gate, up and down planes are claimed, filled, leased and evicted together, in RAM and in VRAM.
- *Amended 2026-10-01 (review D-2, D-11, implemented):* the arena is **one mlock'd mapping carved into per-class regions** (class = distinct unit byte size), one free list and one CLOCK hand per class under the existing two locks. Per-class budget ∝ the class's byte share, computed as `⌊pool × layers_c / Σ unit bytes⌋`. Target arithmetic (GLM-5.3-Flash UD-Q4_K_XL, blk 3–44 live, blk 45 MTP skipped): std 39 layers @ 15,204,352 B/unit, q6d 2 @ 16,318,464, q5q6 1 @ 18,415,616; Σ = 644,022,272 B. 65 GiB → std 4,226, q6d 216, q5q6 108 = **4,550 units (+34.6% vs max-stride HEAD's 3,380)**. Per-class floor: one cold step leases `layers_c × top-k` units (std 312, q6d 16, q5q6 8 — all fit). Derived K (C-3) is priced by Σ unit bytes → K = 9.

### 5.5 Import (amended 2026-09-20, supersedes the original HF-download design)

`golem import <source-path> [<replica-path> ...] [--quant ...] [--name ...] [--revision-label ...]` — Golem never talks to Hugging Face. The customer supplies local paths to weights already on their machine (obtained however they like — `hf download`, another machine, purchased copy).

- **`source-path` (first path)** = source of truth: a directory containing the model's shard files. Golem walks the GGUF headers (arch gate, tensor table, expert regex — §4.2), streams per-shard SHA-256, and writes `manifest.json` into `$GOLEM_DATA_DIR/models/<name>/`.
- **`replica-path`s (additional paths)** = full-mirror targets, one per disk. Golem *creates* the mirrors itself: streamed verified clone of the source shard set into each target, re-hashed against the source's hashes (no trust in pre-existing copies). Full mirrors only (D6) — every replica path must end with the complete shard set; partial/sharded placement stays out of v1.
- **Same-disk detection** (§5.1) runs on the supplied paths: two paths resolving to one physical device → warn, no aggregate bandwidth claimed.
- **Revision metadata is best-effort**: no HF authority exists in this flow. The manifest takes an optional user-supplied `--revision-label` (or reads the revision when the directory carries `hf download` cache metadata); `revision_source` records which of {user, hf-cache, unknown} applied.
- HF download remains out of product scope; the README quickstart documents `hf download <repo>` + `golem import <dir>` as the recipe. (The original `golem pull <repo>` download design is retired; spec §CLI never listed `pull` — the verb is now `import`.)

---

## 6. Expert placement (`golem-placement`)

### 6.1 Identity and state

Key: `(layer, expert_id)` — model-agnostic, works for any MoE arch the substrate loads.

Per key, the placement manager holds: `tier ∈ {disk, ram, vram}` residency flags (independent per tier: an expert can be in RAM but not VRAM, etc.), `score: f64`, `last_access_step`, `dwell_step` (steps since last tier transition), and a CRC over its RAM slot. *(Amended 2026-09-29, review D-11, R-10: the key (layer, expert) is also the unit of paging in every tier. All three planes move together, so the pool's (layer, kind, expert) key is wrong. No per-key CRC exists in code. CRCs live per plane in the manifest, §5.3.)* State is a flat `Vec` indexed by `layer * E + e` — no heap per step, lock split per layer shard.

### 6.2 Score (D5)

```
score(k) ← decay · score(k)          every step (decay = e^(−1/H), H default 512 steps)
score(k) ← score(k) + 1              on access (notified via P1 hook)
```

This is exponential-decay frequency: a blend of LRU (recency dominates for small H) and LFU (accumulated use dominates for large H), tunable by one parameter. Defaults (H = 512, thresholds below) chosen so that a "hot set" is an expert accessed more than ~once per H steps; M0/M4 tune on real traces.

*Amended 2026-09-29.*
- **Step (review D-10).** One step = one `llama_decode` call on the trunk, counted in Rust (Rust drives decode). Prefill chunks are **excluded** from scoring, because a prefill ubatch routes nearly every expert and would flatten all scores. The current wall-clock "turn" heuristic (`TURN_GAP_NS = 500 ms`) makes a whole reply one step and must go.
- **Scale (review D-9).** An average expert's steady-state score is p / (1 − e^(−1/H)) with p = 8/288, i.e. ≈14.2 at H = 512. So the absolute θ values in §6.3 sit an order of magnitude below the typical score.

### 6.3 Transitions

- **disk → RAM** (page-in): on demand via the P3 slow path, or speculatively by the prefetcher (§6.5). Admission if pool has a free slot, else evict lowest-score RAM-resident expert whose `dwell_step > D_min` (min-dwell, default 256 steps) and which is not leased.
- **RAM → VRAM** (promotion): a background policy pass (runs on the placement thread at ≥1 kHz or every step, whichever is coarser) orders RAM-resident experts by score and marks the top ones as *prefer-resident* up to the VRAM expert budget; P2 consumes this ordering at its next eviction decision. Thresholds are hysteresis-widened: promote above `θ_hi = 1.0`, demote-below `θ_lo = 0.4` (an expert that oscillates across the boundary is damped by min-dwell, not by threshold gap alone). *Amended 2026-09-29 (review D-9):* absolute θ values don't filter anything at this scale (any single access clears θ_hi, and one access in ≈470 steps stays above θ_lo). Replace them with a **relative swap margin**: a candidate displaces the weakest resident only if its score exceeds that resident's by ≥ 20%, subject to min-dwell.
- **VRAM → RAM** (eviction): P2 evicts by policy order when the device budget overflows; the RAM copy (if present) is untouched, so re-promotion costs a device-side copy only if the RAM copy survived, else a page-in.
- **Limits = caps.** `--vram-experts-gib` / `--ram-experts-gib` cap each tier's expert cache; within the cap, the scorer decides residency. **Decided 2026-09-29 (O3, review R-7):** floors (`--vram-experts-min`, `--ram-experts-min`) are **dropped**. A floor that pins "the current top-N by score" pins what the cache already keeps, and it only changes behavior when it overrides the scorer, which lowers the hit rate. The spec's per-expert placement limits are met by the caps. Revisit only if a trace shows a floor beating the scorer. Code to remove: the `vram_experts_min`/`ram_experts_min` fields and floor checks in `golem-placement::Policy`, `PolicySnapshot::pin`, the warm-plan floor split, and T3 state (d).
- *Amended 2026-10-03 (BMoE, §16): replay before building the policy thread.* BMoE measured hit rate growing roughly linearly with cache size (reuse is broad, not concentrated); a per-layer smarter policy gained ≈2 points but ran ≈30% slower from allocator churn Golem's fixed slabs don't have. Golem's sizes differ (RAM pool 4,550 / 12,096 units = 37.6%; VRAM K = 9–10 of 288 per layer ≈ 3.5%), so that result does not transfer. Before M3.7 writes the placement thread, an offline replay of M3.8 routing traces (`xtask`, no engine) reports RAM and VRAM hit rate for CLOCK (current), LRU and the §6.2 decayed score with swap margin, at those sizes, on ≥ 3 real prompts. If the decayed score is within ≈1 point of CLOCK at both tiers, M3.7 keeps CLOCK and drops the scorer; the replay predicts hit rate only, not policy cost.
- *Amended 2026-10-03: promotion never stalls a step.* Copying promotions between steps and making the next step wait on them stalls decode. Golem's RAM → VRAM promotion therefore runs on a separate HIP stream and admits a slot (sets its validity bits) only when the copy event completes; until then the slot is free and a routed hit on the newcomer is a normal miss. Bounded per step so promotion H2D never competes with a demand copy: ≤ one layer's routed set (8 × 15.2 MB ≈ 122 MB ≈ 4.5 ms at 27 GB/s).

- *Amended 2026-10-06 (O10): exclusive RAM/VRAM residency.* Today residency is **inclusive**. Tier bits are independent (`golem-placement/src/lib.rs` `TIER_RAM`/`TIER_VRAM`), P2 fills a VRAM slot from the expert's pool slot, and VRAM eviction does no D2H (`golem-device-cache.hpp`). So every VRAM-resident expert also holds a RAM slot: ≈750 × 15.2 MB ≈ 10.6 GiB, ≈17% of a 63 GiB pool. Target behaviour:
  - Promotion to VRAM releases the expert's pool slot once its H2D copy lands.
  - VRAM eviction copies the slot D2H into a free pool slot, or drops it when the pool's victim scores higher. It is never re-read from disk while a VRAM copy exists.
  - The disk → RAM → VRAM staging order is kept (spec §Expert placement).
  - Cost: one D2H per VRAM eviction (15.2 MB at ≈25 GB/s ≈ 0.6 ms) on the split backend's stream — the SAME stream the H2D refills of the stolen slots use, issued first so stream order lands the D2H before the refill (not a separate promotion stream); measured decode impact pending A/B. Plus a pool-slot claim on the eviction path.
  - Sized by the M3.7 replay: build it if the extra ≈10.6 GiB of unique capacity moves the replayed RAM hit rate. Built behind `--residency inclusive|exclusive` (ABI v8 `evict_dest_batch` hook); **inclusive stays the default** until the decode tok/s A/B (cold + warm, both orders) proves exclusive on this machine.
- *Amended 2026-10-06: routing profile for first start.* The persisted §6.2 scores file (`journals/<instance>/scores-<fp>.txt`, policy words and step) seeds a warm start. A first start has no scores file, so ship or record an offline **routing profile**: per-(layer, expert) access counts from M3.8 traces over real prompts. Seed scores from it and pre-fill the pool and the VRAM slots before the first request. The hit-rate curve comes from the M3.7 replay.
- *Amended 2026-10-07 (M3.7 replay, `docs/baselines/m3.7-policy-replay.md`).* Four real 512-token traces replayed at production geometry (K = 18). Findings:
  - **VRAM.** The §6.2 decayed score beats LRU by 2.1 points of VRAM hit and becomes the P2 evict-first order (`--vram-order score`, default).
  - **RAM.** CLOCK stays under inclusive residency: decay at both tiers duplicates the hot set (d 17.8% vs 16.7% at 63 GiB).
  - **Exclusive residency + decay at both tiers:** d 13.8% vs 18.0% today, so O10 builds it with a decayed RAM tier.
  - **Score-driven background promotion** lowers the VRAM hit rate (44.2 → 43.9% at 8/step). Not built (user ruling).
  - **The routing profile** leaves decode d unchanged (18.0 → 17.9%); it is deferred until after §7.6 streaming (user ruling).
  - **`Policy::pass`** keeps no production consumer: per-layer forced-admission slabs consume only its score math.

Startup warm-up: if a previous run's score file (§11) exists, the per-expert score words and step are restored verbatim, the VRAM order is republished from them, and the top-scored experts (up to each pool class's free headroom) are page-in'd in the background before the first request — the first request never waits on it — this is what makes "warm start" measurable separately from cold (§12.3).

### 6.4 Budgets and feasibility

The placement manager refuses at load time (not at step 50,000) any configuration whose caps cannot fit: a VRAM expert budget smaller than one layer's routed set (top-k × max stride per plane), or larger than VRAM left after §10's fixed terms → `golem_plan_infeasible` naming the numbers. *(Decided 2026-09-29, O3: the floor-set check is removed along with floors.)* All budgets derive from §10's arithmetic, so `golem plan` and the runtime agree by construction (same crate, `golem-plan`, computes both).

### 6.5 Prefetch

Autoregressive decoding means step N+1's routing is highly correlated with step N's. The placement thread receives step N's access set per layer and issues page-ins for the union of the last k=3 steps' sets that aren't RAM-resident (k and on/off are config; measured in M0). Prefetch reads use the same striped read path at lower priority (a semaphore caps outstanding prefetch I/O so demand reads always win). Expected effect, to be *measured* not assumed: turning a large fraction of P3 slow paths into bitset hits. The counters from P4 make the hit rate directly observable (§11), so the hypothesis is falsifiable in the first week of real use.

*Amended 2026-09-29 (review H-6):* the last-k union above can only re-fetch experts that were *just* demand-paged and later evicted. It never loads an expert before its first use, so it cannot turn cold misses into hits. The replacement is **next-layer routing prediction**: after layer L's router runs, apply layer L+1's gate weights (VRAM-resident, tiny) to layer L's hidden state, and prefetch the predicted top-k (plus a margin) for layer L+1. That gives about one layer of compute as lead time. Real routing is untouched, because only staging is speculative, so I1 holds. It needs a post-router hook (a new patch, P5). Success metric: `prefetch_hit_rate` = prefetched keys accessed before eviction.

*Amended 2026-10-03 (BMoE, §16): prefetch ships behind a kill switch, off by default until it measurably wins.* BMoE's next-layer predictor (the same scheme) was right 80–89% of the time, yet acting on it cost 21–38% throughput when storage was already the bottleneck: wrong guesses spend the disk bandwidth that demand misses need. Using the prediction only to protect predicted experts from eviction gained ≈0.4 hit-rate points at no I/O cost. Golem is disk-bound on cold decode too (§2.4), and once layer-batched reads (§3.4 P3) saturate the queue during misses there is little idle bandwidth to spend. So: `--prefetch {off,protect,read}` (default `off`); `read` issues reads only at prefetch priority behind demand (§5.2); `prefetch_wasted_bytes` (prefetched, evicted unaccessed) is a metric; M3.7 acceptance is decode tok/s off vs protect vs read, cold and warm, A/B in both orders (§12.3), and the default changes only for a mode that wins. Build it after layer-batched reads.

*Amended 2026-10-07 (M3.7, implemented).* P5 is a graph node per MoE layer L, built only when `--prefetch` is not `off`:
- **Prediction.** `golem_pred-<L+d>-<L>` = the top (k + 4) of layer L+d's router over layer L's FFN input (the `hc_ffn_pre` output), normed with L+d's `ffn_norm`, with the same gating and selection bias. d = 1..`--prefetch-lookahead`.
- **Delivery.** The scheduler reads the predictions in layer L's pre-pass, right after its ids readback has synchronized the device, and passes them to the `predict` hook (ABI v6). There are no eval-callback views and no extra syncs.
- **Modes.** `protect` shields the predicted union from RAM eviction. `read` queues it for the prefetch thread, which runs behind demand and never evicts a reference-bit key (P17). The last-k union prefetcher is removed.
- **Real model.** Recall is 84% at lookahead 1 and 87% at 2; output is byte-identical across modes.

### 6.6 Correctness under eviction

An expert may be evicted from RAM only if no in-flight read targets it (floors dropped, O3); in-flight P3 waits hold a reference. All transitions happen on the placement thread against the bitsets *(amended 2026-09-29, review R-10: at HEAD they happen inside `claim` on the decode thread; there is no placement thread yet)*; the decode thread's P3 callback consults the same bitsets under a per-layer shard lock held only for bitset flip + slot handoff (never across I/O). The lock discipline is: **bitset ops are the critical section; I/O happens outside locks** — reviewed against data races in T6 (§12.2) with a loom test over (page-in, evict, access) interleavings. *(Amended 2026-10-04, M3.11: the decode thread claims each (layer, expert) unit once per scheduler compute — the unit's other planes reuse that lease — and reads the policy snapshot once per compute; 42 × 8 = 336 shard-lock claims per token instead of 1,008.)*

---

## 7. Inference runtime

### 7.1 Slot scheduler

- A slot = one chat sequence: KV-cache sequence id, request state, sampling params, output channel.
- *Amended 2026-09-29 (review R-4, R-1):* **per-slot state snapshot/restore is a v1 requirement.** The engine today clears memory and re-prefills the whole transcript every turn (`session.rs` "stateless per turn"), and prefill is the most expensive operation on this architecture (§2.4), so turn N costs O(transcript). At turn end, snapshot the sequence state (`llama_state_seq_get_data`: recurrent state + KV), keyed by the token prefix. When a request extends a snapshotted prefix, restore it and prefill only the delta. Snapshots are host-RAM-budgeted (≈145 MiB recurrent + KV per snapshot) and LRU-evicted. They are the only correct form of prefix reuse on a recurrent model.
- v1 concurrency: up to `--parallel N` slots (default 1 for `golem run`, ~~4~~ **1** for `golem serve`; *amended 2026-09-29, review D-13:* 4 independent slots route ≈30.7 unique experts per layer per step versus 32 with no sharing, so under the §2.4 PCIe/disk ceilings aggregate tok/s is ≈flat and each slot runs at ≈¼ speed. Raise the default only when M3.5 measures a batching win). *Amended 2026-10-10:* `serve --parallel` > 1 is refused unless `GOLEM_ALLOW_PARALLEL=1` until the target HW batching leg (`docs/baselines/m4.4-batching.md`, #83) passes. MoE + disk-paged experts makes heavy batching a bandwidth amplifier (each slot's routed experts union); v1 keeps batching modest and *measures* before raising defaults — batching interacts with the placement hot set, so the default is conservative by design.
- Batching is dynamic: each decode iteration includes every slot whose tokens are ready (continuous batching, slot join/leave at iteration boundaries — the same discipline `llama-server` uses, re-implemented over `llama.h`, which supports multi-sequence batched decode).

### 7.2 Decode loop

Dedicated thread(s) (`--decode-threads`, default 1 — HIP kernels own the GPU). *Amended 2026-09-29 (review R-15):* the **CPU backend threadpool** is separate and must not default to 1. It still runs sampling, the ids readback and any CPU-side op, so default to physical cores / 2. Pin the decode thread to the V-cache CCD and the I/O threads to the other CCD on the 9950X3D:

1. Drain ready continuations from the queue into a `llama_batch` (prompt chunks on join; single tokens while generating).
2. `llama_decode`.
3. Per slot: sampler chain (llama.cpp, seeded; §12.2 reproducibility) → append token → emit delta.
4. Notify placement manager of step completion (score decay tick, access notification flush).
5. Loop; preempt on stop/preemption signals from the async side (slot cancel = ~~KV sequence rewind~~ sequence **removal**, or restore from the last snapshot; *amended 2026-09-29, review R-1:* recurrent state cannot be partially rewound). Preemption mid-`llama_decode` uses `llama_set_abort_callback`.

Prefill and decode are the same loop; a long prompt is processed as chunked prefill batches (chunk = 512 tokens default) so a cold joiner doesn't starve generating slots beyond one chunk. *(Amended 2026-09-29: the 512 default is provisional. Per §2.4, a naive 512-token ubatch copies nearly the whole expert set over PCIe. The prefill chunk size and strategy are set by the M3.5 measurements.)* *(Amended 2026-10-03: M3.5 measured 0.24 tok/s at chunk cap 1. Chunks above the §7.6 streaming threshold go through layer streaming at `n_ubatch` up to 2048. Below the threshold, the per-token cap `max(1, K/top_k)` stays.)*

### 7.3 Chat template and reasoning content

- Prompt construction applies the GGUF-embedded chat template via llama.cpp — parity with the reference runtime again. No Golem-side template code.
- GLM-5.3-Flash's thinking output: the template's reasoning delimiters are exposed as `reasoning_content` on the message/delta (DeepSeek-style convention), split by the template's known marker tokens (the exact markers are read from the chat-template metadata and pinned in M0; they are a property of the model revision, which the manifest already pins). `--reasoning {inline,separate}` selects behavior; default `separate`. When `inline`, raw tokens stream as `content` with markers intact (clients that don't understand `reasoning_content` see the model's native output — never silently stripped; spec bans silent behavior).

### 7.4 Determinism

Same weights (bit-identical), same kernels, same op order ⇒ same activations *for a fixed batch composition*. Known substrate nondeterminism (batch-size-dependent reduction orders; FA kernels) is controlled in v1: parity harness fixes batch=1, context length, FA setting (M0-validated), and seed. Production serving does not promise cross-request bitwise determinism (no LLM server does); it promises per-(seed, batch-composition) determinism, which the harness exercises.

### 7.5 MTP speculative decoding (amended 2026-09-29, review R-2; *Decided (O6), 2026-10-03: deferred; adopt only if the expert-union spike passes, see §1*)

The GGUF ships one NextN/MTP block (`nextn_predict_layers = 1`, blk.45) with its own routed experts, and the fork can load it (`load_mtp`). MTP drafts the next token(s) cheaply, and the trunk verifies k drafted tokens in **one** forward pass. Those k tokens largely share routed experts, so expert bytes per *accepted* token fall roughly with the acceptance rate. Under the §2.4 PCIe ceiling this is the largest lever after quantization. It interacts with I1: verification uses the trunk's own routing, so selection per verified token is unchanged, and it interacts with T2 (greedy MTP-verified output must equal plain greedy). Measure the acceptance rate on real prompts at M3.5 before committing. *(M3.5 could not measure it because no MTP code exists. Under O6, the consecutive-token expert-union spike from §1 gates adoption.)*

*Amended 2026-10-03 (BMoE, §16): determinism and draft length.* "MTP-verified output must equal plain greedy" holds only when the verify pass computes each position exactly as a single-token decode would. On llama.cpp kernels it need not: a k-token verify batch changes reduction order and kernel choice (§7.4), and BMoE saw output differ even between plain decode at ubatch 1 and its default ubatch. T2 tests it with forced wrong drafts and rollbacks. So T2 compares MTP-verified output to plain greedy and reports the first divergence; equality is the goal, and a divergence that traces to batch reduction order is accepted as a documented per-configuration exception under an opt-in `--mtp` (like O1), never a silent default. Acceptance is lower than the borrowed 85–90%: BMoE measured 71/60/52% per draft at k = 2/3/4. A confidence gate (draft only while the drafter's p ≥ 0.5) trades draft length for acceptance. The spike therefore measures acceptance per draft position from the real NextN layer alongside the union U_k, at k ∈ {2, 3, 4}, with and without a confidence gate.

*Amended 2026-10-06 (O9): the substrate already drafts with NextN.* M3.5's "no MTP code exists" holds for `crates/`, not for the vendored fork. Three pieces exist there:
- `LLAMA_CONTEXT_TYPE_MTP` (`llama.h:225`);
- GLM5NEXT cache filters that give the draft context a cache for the NextN block only (`llama-model.cpp:2554`, `mtp_ctx`), so the trunk's recurrent layers are never rolled back;
- `COMMON_SPECULATIVE_TYPE_DRAFT_MTP` in `common/speculative.cpp:2540`.

The spike's cost is therefore wiring:
- load with `load_mtp`;
- create the MTP context;
- extend P1–P3 coverage to blk.45's experts (layer 42 of the pool geometry, §2.2);
- drive the verify window from `golem-engine`.

Writing a draft model is not part of it.

**Prompt lookup** rides the same verify window: draft up to 5 tokens by matching the last n-gram against the context, with no draft weights. It is enabled only where its measured acceptance pays. The spike reports it alongside NextN.

### 7.6 Prefill layer streaming (amended 2026-10-03, D-7)

**Problem.** Per-token prefill pays a full cold decode step per prompt token (measured ≈4.2 s/token cold, 0.24 tok/s). A ubatch of 2048 routes all 288 experts in every layer (§2.4), so batching only helps if each expert's bytes cross PCIe **once per ubatch**, not once per token.

**Mechanism.** The P2 prefill-stream mode (§3.4) runs per ubatch whose size is ≥ `--prefill-stream-min` (default chosen by measurement; the arithmetic crossover is ≈4 tokens, because per-token costs ≈4.2 s cold and one streamed ubatch costs ≈13.4 s).
1. **Plane granularity.** The pipeline unit is one plane (gate, up or down) of one layer for all 288 experts. Two stream slabs, each `n_expert × max plane stride` over all classes. Strides from the tensor table: q4 planes 4,718,592 B, q5 5,767,168 B, q6 6,881,280 B (std = q4+q4+q5, q6d = q4+q4+q6, q5q6 = q5+q5+q6). So the max plane is 6,881,280 B, and 2 × 288 × 6,881,280 B = 3,963,617,280 B = **3.69 GiB VRAM**. VRAM fit on the target: ≈13.0 GiB free after non-expert weights and ROCm (§2.4), minus the P2 slot cache at derived K = 10 (10 × 644,022,272 B = 6.0 GiB), minus 3.69 GiB of slabs, leaves **≈3.3 GiB for `ws` at the prefill ubatch plus KV**. `ws` at 2048 is not measured yet. If it does not fit, `golem plan` lowers `--prefill-ubatch` (it does not drop the slot cache). Whole-layer double-buffering (2 × 4.08 GiB = 8.2 GiB) was rejected: it fits only if the P2 slot cache is dropped.
2. **Sources.** Each plane's expert ranges come from the RAM pool when resident: H2D from the pinned slot directly, holding a lease until the copy event completes, as P3 already does. Misses are read through the shared queue (§5.2) into a **pinned staging ring** (mlock'd and `hipHostRegister`ed, ≥ 2 max planes = 2 × 288 × 6,881,280 B ≈ 3.7 GiB of RAM), never into pool slots. So a long prompt does not evict the decode hot set, and the bytes are copied at most once disk→staging and once staging→device, satisfying the AGENTS.md copy rule.
3. **Overlap.** Disk reads for plane *p+2*, H2D for plane *p+1* and compute for plane *p* proceed concurrently. With a ≥ 2-plane read-ahead, the pipeline runs at the slowest stage. That stage is the disk on this machine (§2.4).
4. **Scoring.** Streamed accesses are not scored (§6.2: prefill chunks are excluded), and they do not change pool residency.
5. **Invariants.** Every streamed `MUL_MAT_ID` runs on the GPU (I2; the C-4 runtime check covers the stream path). The bytes are bit-identical (I1). Output is deterministic per batch composition (§7.4); T2 documents the deltas at prefill ubatch > 1 versus the per-token path.
6. **Kernel risk.** The MMQ `MUL_MAT_ID` path at `n_ubatch` ≥ 2048 is unvalidated on gfx1100. An external report shows a padding bug crashing it on Blackwell/CUDA (§16). The HIP acceptance leg must run every ubatch the default can select (512–8192) on the real model.

**Ubatch size.** Streamed bytes per ubatch are constant (185.5 GB, ≈13.4 s disk-bound cold), so throughput scales linearly with ubatch until `ws` stops fitting. Estimates: 4096 → ≈305 tok/s, 8192 → ≈610 tok/s. `--prefill-ubatch` therefore defaults to the **largest of 512/1024/2048/4096/8192 that fits §10**, not a fixed 2048. Evidence on the same GPU: exllamav3-rocm-moe measured 8192-token chunks at ≈945–990 tok/s vs ≈600–680 at 2048 (RX 7900 XTX, different model and quant, §16). The VRAM left after slabs and the slot cache (≈3.3 GiB, point 1) is the binding term, so `ws` per ubatch must be measured first.

**Routed-only fill for short ubatches (amended 2026-10-03, BMoE, §16).** Streaming a full plane is waste when a ubatch routes few experts. Under uniform top-8-of-288 routing a ubatch of T tokens touches 1 − (1 − 8/288)^T of each layer's experts: T = 16 → 36%, 32 → 59%, 64 → 83%, ≥ 128 → ≈97%; real routing is skewed, so the true share is lower. Follow-up chat turns (after M4.8 snapshots) prefill only the new message, often T ≤ 64. The stream slab already uses identity ids with `K = n_expert`, so filling only the routed experts' ranges needs no new mechanism; unfilled ranges are never read by `MUL_MAT_ID`. Up and down are filled routed-only for free: the layer's ids are known once its router has run, before up/down compute. The gate plane of layer L+1 is filled before L+1's router runs, so it stays whole (default) or is predicted from the previous ubatch's layer-L+1 routing with misses filled after the router (BMoE's approach). Switch to whole planes when the routed share exceeds ≈85% (BMoE's threshold; Golem measures its own). Streamed bytes for T = 32 under the uniform estimate: gate 4.72 MB × 288 + (up + down) 10.49 MB × 0.59 × 288 ≈ 1.36 + 1.78 ≈ 3.1 GB per layer vs 4.4 GB whole (≈30% less); the ids-dependent half is what shrinks.

**Budget.** §10 gains the `stream_slabs` (VRAM) and `staging_ring` (RAM) terms, and `ws` grows with `n_ubatch` (compute buffer at 2048 measured, not assumed). `golem plan` refuses a `--prefill-ubatch` whose terms do not fit, and suggests the largest that does.

**Acceptance.** `ws` measured per ubatch (512–8192); cold and warm prefill tok/s on the real model at 512, 2048 and the largest ubatch that fits (separately), compared with the §2.4 estimate. The HIP gate must report zero SKIPPED. The decode hot set must be unchanged across a long prefill: P4 RAM-hit rate before and after.

---

## 8. HTTP API (D7)

### 8.1 Endpoints

| Endpoint | Status | Notes |
|----------|--------|-------|
| `POST /v1/chat/completions` | v1 | streaming (`stream: true`, SSE) and non-streaming |
| `POST /v1/messages` | v1 | Anthropic Messages API, streaming and non-streaming (§8.6, #106) |
| `GET /v1/models` | v1 | the instance model, then each `[models.<name>]` entry (*amended 2026-10-20*) |
| `GET /health` | v1 | Golem extension: `state`, `tiers{vram_fill,ram_fill}`, `degraded`, `model`, `turns`, `device_cache_skip_ratio` (§11) |
| `GET /metrics` | v1 | Prometheus-format (§11) |
| everything else | — | 404 `golem_not_found` (typed, §8.4) |

*Amended 2026-10-20 (several models, idle unload):* `golem.toml` `[models.<name>]` tables add served models to one instance. Each has `path` (GGUF) and optional engine keys that override the file layer for that model only (flags and env still win); its own `golem optimize` profile applies (fingerprint per path). `/v1/models` lists the instance model then each entry; `model` must name one, else `golem_model_mismatch`. At most one model is resident — on the target GLM-5.3-Flash UD-IQ3_XXS (~24 GiB VRAM, `ram_experts_gib` 50 + KV/snapshots ≈ 70 GiB RAM) and DS41 UD-IQ2_XXS (24 GiB VRAM planned, 69.8 GiB RAM planned) sum to ≈ 48 GiB VRAM and ≈ 140 GiB RAM, against 24 and 93 — so the worker swaps on demand: a job naming the unloaded model is dequeued only with every slot idle (parallel 1), the loaded session saves warm scores and is dropped, the other loads, then the job runs. A cancelled or draining job is refused rather than swapped for. A failed load answers that job `golem_backend_unavailable` and leaves the instance `idle`. `idle_unload_min` > 0 starts the instance `idle` (no model; `/health` 200 `idle`) and drops the loaded model after that many minutes without a job; the next request loads its model. Load/unload state changes never leave `draining`/`releasing`/`fatal`. All models must agree on forcing expert offload: the first load fixes the process-wide op-offload threshold (`init_backend`). Sampling is client-side. The registry and the admission check see the start model only; the operator keeps every served model within the machine.

### 8.2 Request field policy

Strict allowlist — anything requested-but-unsupported gets an explicit typed error, per spec. Tolerated-only fields (no functional effect) are `metadata`, `user`, `request_id`, `store` (`false`/`null` only; `store: true` is rejected with `golem_unsupported_field` since completions are not stored).

| Field | v1 behavior |
|-------|-------------|
| `model` | must match instance model name (else 400 `golem_model_mismatch`) |
| `messages` | supported; text content only; image/audio parts → 400 `golem_multimodal_unsupported` |
| `temperature`, `top_p`, `max_tokens` (`max_completion_tokens` aliased), `stop`, `seed`, `frequency_penalty`, `presence_penalty`, `n` (must be 1) | supported |
| `stream`, `stream_options.include_usage` | supported (SSE; final frame carries usage) |
| `tools`, `tool_choice` | **Decided 2026-09-29 (O2, review R-14):** supported. Tools are rendered through the pinned chat template, and tool calls are parsed from the model output by llama.cpp's chat layer for that template, then returned as OpenAI `tool_calls` (streaming: `delta.tool_calls`). `tool_choice`: `auto`, `none`, `required`, and a named function (enforced by the template/grammar where the substrate supports it, else a typed 400). Parallel tool calls follow the template. Acceptance: a T7 golden test per template path, plus a T2-style parity check that the rendered prompt is byte-identical to the reference runtime's. |
| `reasoning_effort` | *Amended 2026-10-14 (#117), re-amended 2026-10-18 (#134):* supported, per the GLM-5.3-Flash model card. Exactly `low`, `high` and `none` (case-sensitive strings) are honoured: `low`/`high` are passed to the chat template as `reasoning_effort`; `none` renders the template's no-think form (§8.6). Any other value of any type (`max`, `medium`, `minimal`, numbers, objects, …) is a 400 `golem_invalid_request` — never silently mapped to `max`. `null`/absent sends nothing, so the template default `max` applies. |
| `response_format` (json schema), `logit_bias`, `logprobs` | **400 `golem_unsupported_field`** naming the field and the reason. Clients probing capabilities get answers, not silence |
| `top_k` | Golem extension, supported |

Usage reporting: prompt/completion token counts from llama.cpp counters — cost-accurate, since we own the loop.

### 8.3 SSE shape

Standard OpenAI chunk framing (`data: {chunk}\n\n`, `data: [DONE]`), deltas carrying `content` and/or `reasoning_content` (§7.3), terminal frame `usage` when `include_usage`. Backpressure: SSE streams are tokio-guarded; a dead client cancels its slot (decode loop preempts at the next iteration boundary). An error mid-stream (drain, worker death) sends the error frame, then `data: [DONE]` (*Amended 2026-10-13:* previously the stream ended after the error frame with no `[DONE]`); a live client that cannot keep up with the event buffer gets a `golem_backend_unavailable` error frame, never a truncated `finish_reason: stop`.

### 8.4 Error model

One envelope everywhere (HTTP + CLI + logs):

```jsonc
{ "error": { "code": "golem_plan_infeasible", "message": "...",
             "detail": { /* structured, e.g. per-tier byte overruns */ } } }
```

Codes (stable, documented): `golem_backend_unavailable`, `golem_plan_infeasible`, `golem_model_mismatch`, `golem_multimodal_unsupported`, `golem_unsupported_field`, `golem_storage_unavailable`, `golem_instance_not_found`, `golem_not_found`, `golem_busy`, `golem_stop_timeout` (CLI only: `golem stop` gave up waiting for the exit; retry with `--force`), `golem_invalid_request`. All 4xx/5xx carry `code`; the doc table maps codes → remediation.

### 8.5 Server posture

Binds `127.0.0.1` by default (port: auto-select, recorded in the registry; `--host/--port` to override). Optional bearer token (`--api-key`). No TLS in v1 (loopback tool; document the reverse-proxy path). `golem serve` runs in the foreground by default — logs to stderr; *amended 2026-10-10:* Ctrl-C cancels in-flight requests at once (a second Ctrl-C exits without teardown), SIGTERM and `golem stop` keep the 30 s drain grace (`golem stop --now` cancels at once) — with `--detach` (double-fork, logs to `$GOLEM_DATA_DIR/logs/<instance>.log`, PID in registry) for the background habit.

### 8.6 Anthropic Messages API

*Added 2026-10-14 (#106).* `POST /v1/messages` serves Anthropic-protocol clients (Claude Code, the Anthropic SDKs) on the same port and key. There is one decode path: `golem_api::anthropic::to_chat` translates the body into the §8.2 chat request. That request then runs through the same `policy::validate`, worker, chat template and tool parser, and the result is translated back.

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
- **Cost.** A captured Claude Code request (2026-10-14) is 66 KB of JSON: ≈ 47.5k chars of tool schemas plus ≈ 5.8k of system prompt. At ≈ 3.5–4 chars/token that is ≈ 13–15k tokens. At the measured 5.6 tok/s cold streamed prefill (m3.6) that is ≈ 14 000 / 5.6 ≈ 42 min for a cold first turn. Later turns depend on §7.1 prefix snapshots. Turn-1/turn-2 TTFT on the target is still to be measured.

---

## 9. CLI and instance lifecycle (D8)

| Command | Behavior (pinned) |
|---------|-------------------|
| `golem run [model]` | **Decided 2026-09-29 (O4, review R-12):** if a compatible instance (§9.4 config hash) is running, attach an interactive chat client (stdin REPL over the HTTP API, streaming). Otherwise load the model **in-process, in the foreground**, and run the REPL directly over the engine. Exit (EOF / Ctrl-C) unloads and exits: no daemon is left holding the GPU and locked RAM. `--keep` instead starts a detached `serve` and attaches to it. Flags map onto instance config. *Amended 2026-10-13:* Ctrl-C mid-reply cancels that turn and keeps the REPL; Ctrl-C at the prompt, SIGTERM, or a signal during `--prompt` exits `128 + signal` (130 / 143) with `golem: interrupted` and no error envelope; a second signal before the first is handled kills the process |
| `golem serve [model]` | Foreground instance (§8.5). `--detach` supported. One model per process |
| `golem list` | Default: running instances (id, model, health, tier fill %, req/s, tok/s, degraded flags) from the registry. `--models`: downloaded models from the data dir with manifest summary (quant, size, replicas, last-used). Both, never conflated |
| `golem stop <id\|name>` | Graceful: stop accepting, drain in-flight requests (grace period 30 s), unload, exit. `--force`: SIGKILL. `--now`: skip drain. Stopping a nonexistent/dead instance is a typed error, not a shrug. *Amended 2026-10-13:* socketless kinds get SIGTERM then SIGKILL after a per-kind grace — `run` 60 s (teardown; the SIGKILL may land mid-load), others 5 s |
| `golem plan [model] --ctx N --vram-experts-gib N --ram-experts-gib N ...` | §10. No GPU required; reads manifest + flags, prints per-tier byte table and a fit verdict with exact overages |
| `golem import` | §5.5 |

### 9.2 Instance registry (D11)

`$XDG_RUNTIME_DIR/golem/instances/<id>.json`: id (slug: model + port + short uuid), pid, http endpoint, model name, config hash, started-at, health cache. Liveness = PID exists **and** UDS heartbeat (< 5 s old). *Amended 2026-10-13:* an entry is stale (reaped) only when its PID is gone or reused; heartbeat staleness marks a live instance **hung**, never reaped — `golem stop` escalates it to SIGKILL when the control socket is also unreachable and the heartbeat stays stale over two samples ≥ 10 s apart. Stale entries are reaped by any golem client that observes them (registry is advisory state, not a database). Control (stop, status detail) goes over a per-instance UDS carrying the same JSON-RPC-ish envelope as the error model — `golem stop` works even when HTTP is wedged.

### 9.3 Multiple instances / models

Each instance is an independent process with its own VRAM budget; concurrent instances are allowed and *individually* planned, but combined-fit is the operator's responsibility — `golem plan` checks a single instance; before starting a second instance, the CLI sums planned VRAM **and planned locked RAM** of running instances. **Decided 2026-09-29 (O5, review R-13):** it **blocks** by default when either sum exceeds physical capacity (`golem_plan_infeasible`, naming each instance's share), and `--allow-overcommit` proceeds with a loud warning. For this model one instance *is* the GPU and most of the RAM, so an overcommitted second instance would thrash or OOM mid-request, which the spec bans. Documented, not hidden. (Hard multi-instance VRAM arbitration is future work; the spec only asks the question be answered honestly.)

### 9.4 Config precedence

Flags > env (`GOLEM_*`, uppercase, one per flag) > config file (`$GOLEM_CONFIG_DIR/golem.toml`) > `golem optimize` profile > built-in defaults. *Amended 2026-10-18:* the config file is `golem.toml` in `$GOLEM_CONFIG_DIR`, else `$XDG_CONFIG_HOME/golem`, else `~/.config/golem` (one resolver shared with the profile's tuning dir); a missing file is no file layer, an unparsable one is an error naming the path, and which file was loaded (or that none was found) is logged at info. The model path resolves the same way: the positional argument of `run`/`serve`/`plan` > `GOLEM_MODEL` > `model` in `golem.toml` (`~` expanded); with none set, `serve` and `plan` fail naming all three, `run` prints its help. A `run --keep` detached `serve` resolves it through the same layers. *Amended 2026-10-10:* the profile (`$XDG_CONFIG_HOME/golem/tuning/<fingerprint>.json`, `~/.config` when unset) applies only when it holds a winner, the hardware/model/engine fingerprint matches (GPU, CPU topology, model shards' name/size/mtime, engine version and vendor pin), and free RAM/VRAM now are at least what the sweep measured minus 256 MiB; `--no-optimize-profile` skips it. The instance's effective config hash is recorded in the registry (§9.2); `golem run` only attaches to an instance whose config hash is *compatible* (same model; placement budgets that fit the running instance), else it reports the conflict and runs in-process (*amended 2026-10-10:* `--new` skips the check; `--keep` starts a detached `serve` and attaches).

---

## 10. `golem plan` — the memory model

All figures derive from the GGUF header via the manifest — never hardcoded model tables. Let `E` = experts/layer, `L` = expert layers, `b_e` = bytes per expert per layer, and `T_ne` = total bytes of non-expert tensors (computed by filtering the tensor table through the upstream MoE regex — the same classifier llama.cpp uses, so "expert" means the same thing in both).

```
VRAM =
  T_ne                                    non-expert weights (resident forever)
+ kv_per_token · ctx · parallel            KV cache, 11 sparse-attention layers only (amended 2026-09-29, R-1)
+ rs_bytes · parallel                     recurrent state, 34 linear-attention layers (constant per sequence; 145.56 MiB measured)
+ shexp_bytes                             shared experts, VRAM-resident (R-3; part of T_ne, listed so it is never tiered)
+ ws                                       decode/prefill workspace (fitted on gfx1100: max(320 MiB + 5 138 B·ctx, 1 105 975 B·ubatch); m4.6-plan-t8)
+ staging                                  max b_e · 1.5 (P2 copy padding guard, §3.6)
+ device_expert_cache                      ≤ --vram-experts-gib, as slot slab K × stride per (layer, kind), §3.4 P2 (floors dropped, O3)
+ stream_slabs                             2 × n_expert × max plane stride, prefill layer streaming (§7.6; 3.69 GiB on the target)
+ rocm_overhead                            measured in M0: 2.64 GiB combined overhead on gfx1100 (m0-notes)

RAM =
  ram_pool                                ≤ --ram-experts-gib (default min(.75·free, Σ expert bytes))
+ host_overhead                            loader + activations (measured, starts 2 GiB)
+ state_snapshots                          per-slot snapshot cache (§7.1; amended 2026-09-29, R-4)
+ staging_ring                             pinned disk-miss staging for prefill streaming, ≥ 2 max planes (§7.6; ≈3.7 GiB on the target)

DISK =
  model_bytes · replicas + logs
```

Output: table per tier, the four governing numbers (VRAM headroom, RAM pool, per-token KV cost, max cold-expert latency estimate from measured bandwidth), and a verdict. Infeasible → exit code 2, `golem_plan_infeasible` with the failing inequality printed with actual numbers. `--json` for tooling. *(Amended 2026-09-29: the `staging` and `device_expert_cache` terms assume upstream's full-size transient expert copy. They change when the P2 slot-indirected device cache lands (review C-3/H-5): staging goes to ≈0 and the cache becomes `K` slots × stride per (layer, kind). `golem plan` also reports the §2.4 ceilings per candidate quant.)* The same functions serve the runtime's load-time feasibility check (§6.4) — one implementation, two callers, no drift.

KV per token without loading the model: read from the arch table (glm5next entry created in M0 from the loaded context's actual allocation, then frozen as a formula in metadata keys); when the GGUF carries the needed dims, the formula computes exactly.

*Amended 2026-10-19 (#107):* `kv_per_token` is priced per cache type: each attention layer costs `heads_kv · (row(type_k, key_length) + row(type_v, value_length)) + idx_heads · row(type_idx, indexer_key_length)`, where `row` is ggml's block size (q8_0 = 34 B per 32 elements). MLA stores no V, so it requires `type_k == type_v` (as llama does). On glm5next the indexer cache stays f16 under a quantized K, because it also holds the compressor gates (`llama-model.cpp`). Target at ctx 131,072: f16 19,712 B/token (2.41 GiB), q8_0 11 × (16 × 34 + 384 × 2) = 14,432 B/token (748 + 1,056 MiB, matching llama's log). The quantized V turns on flash attention. A/B on UD-IQ3_XXS (`scripts/kv-ab.sh`, both orders): needle recall at 10/50/90 % of 129k tokens passed for both types, prefill ≈ 78–84 tok/s and decode ≈ 3.3 tok/s for both, and the 660 MiB saved fit 2 more device-cache slots (`k_decode` 21 vs 19). The registry records the types, so `compatible` refuses to reuse an instance with different ones.

---

## 11. Observability

- **Tracing:** `tracing` + JSON logs to stderr/file; spans for request lifecycle, decode steps (sampled), page-ins, evictions. *Amended 2026-10-10:* `serve` logs at INFO for `golem_server`, `tower_http` and `golem_engine::slot` and at WARN otherwise (`RUST_LOG` overrides); JSON when detached or off a TTY, text otherwise. It logs every HTTP request/response with status and latency, and one `chat completion` line per request with status (499 = client disconnected), finish reason, prompt/completion tokens, `prefill_tok_s` (admission → first piece; a snapshot-restored prompt or `--parallel` > 1 reads as latency) and `gen_tok_s`.
- **Metrics** (`/metrics`, Prometheus): token rates per slot; bytes disk→RAM (demand vs prefetch), RAM→VRAM (demand vs promote), VRAM evictions; P2 skip ratio (device-cache hit rate); P3 slow-path rate (the key health number — should be ≈0 steady-state); *amended 2026-09-29:* `prefetch_hit_rate` (prefetched keys accessed before eviction, §6.5 H-6), `prefetch_wasted_bytes` (prefetched, evicted unaccessed, §6.5, amended 2026-10-03), `mtp_acceptance_rate` (§7.5), `state_snapshot_hits` (§7.1), and an I2 guard counter of expert `MUL_MAT_ID`s executed off-GPU (must stay 0, C-4); tier fill levels; replica health; per-code error counters. *Amended 2026-10-10 (as exported in M4):* `golem_device_cache_{bytes_copied,bytes_skipped,evictions}_total`, `golem_pool_{demand_pageins,fast_hits,evictions,bytes_paged,source_errors}_total`, `golem_prefetch_hit_rate`, `golem_prefetch_wasted_bytes`, `golem_state_snapshot_{hits,misses,restore_failures,evictions,dropped_oversize}_total`, `golem_state_snapshot_bytes`, `golem_expert_mul_mat_id_off_gpu_total` (HIP builds), `golem_locked_ram_bytes`, `golem_vram_free_bytes`, `golem_queue_depth`, `golem_in_flight`, `golem_errors_total{code}`, `golem_slot_tokens_total{slot}`. *Amended 2026-10-13:* `golem_expert_h2d_copies_total{path="cache"|"stream"}` (expert planes copied host→device by device-cache fill or streamed prefill; the stream path counts whole-tensor fills, which can exceed routed demand; pooled offloaded sessions only) and `golem_expert_mul_mat_id_total{device="gpu"|"cpu"}` (expert `MUL_MAT_ID` graph-node placement; offloaded sessions only; `golem_expert_mul_mat_id_off_gpu_total` is its deprecated `{device="cpu"}` alias). The per-expert CPU count of the §2.4 split is 0 by construction until the CPU split (#52) exists, so the DMA count is the whole miss traffic. Not yet exported: `mtp_acceptance_rate` (MTP deferred), RAM→VRAM demand vs promote split, replica health; tier fill is in `/health`.
- **Per-step timing (amended 2026-10-03, BMoE, §16).** Counters say how many misses; they cannot say why cold decode moves ≈2 GB/s against 5.8–8 GiB/s of disk (§2.4). `--timing-csv <path>` writes one row per decode step: step, tokens, wall ms, P3 wait ms (time inside `ensure_host_resident`, measured in the Rust trampoline), disk read ms and bytes, H2D ms (HIP events on the copy stream) and bytes, policy ms, and compute ms = wall − the measured parts. A column that is not measured in a given build is written `unmeasured`, never `0`. The file starts with a header carrying engine version, fork commit and the full resolved config. Cost: a few `Instant::now()` per layer and one event pair per H2D batch, no allocation per row (preallocated ring flushed at turn end), so it obeys the hot-path rules and can stay on during benchmarks. *Amended 2026-10-13:* with `async_h2d`, `h2d_ms` is the sum over the step's copying splits of first copy start to last copy end on the copy stream (gaps between splits excluded), so it includes copy-stream waits on compute within a split; a step with more than 256 copying splits (the event ring size) writes `unmeasured`. The copy stream waits on the compute-stream tail before each plane (conservative WAR rule), so GPU-side H2D/compute overlap is ≈0 and only host-side queueing runs ahead; the #83 A/B is expected to show that.
- **Warm-start scores:** `golem-scores v1` text file of the raw §6.2 score words plus the decode step, written atomically to `$GOLEM_DATA_DIR/journals/<instance>/scores-<fp>.txt` (fp = hash of shard names, sizes and mtimes) on graceful shutdown and every ≥ 300 s at turn end; restored for score seeding (§6.3). Offline policy tuning uses routing traces (M3.8).
- **Health states:** `healthy` / `degraded:{replica,io,checksum}` / `draining` / `fatal-io` — surfaced in `/health`, `golem list`, and log lines. Degraded is a first-class, visible state (spec: detect and report, never silently).

---

## 12. Testing and acceptance

### 12.1 Layout

- Unit tests per crate (policy math, manifest, striping, plan arithmetic — pure and table-driven).
- Integration tests run against a **synthetic tiny MoE GGUF** (a 2-layer, 8-expert model generated by a test helper that writes valid GGUF with the upstream tensor-naming scheme): exercises the full load → route → page → decode path in CI without 186 GiB or a GPU (CPU-backend substrate build for CI; HIP build tested on hardware in the hardware lab job).
- Hardware tests (M0+ machine): nightly script, not CI-gated.

### 12.2 Test matrix

| ID | Test | Asserts |
|----|------|---------|
| T1 | Op parity: expert matmul (`MUL_MAT_ID`) on fixed weights/ids, HIP vs CPU substrate, Q8_0 and UD-Q4_K_XL | max abs diff within M0-calibrated bound (start: ≤ 1e-2 on fp16 accumulate path) |
| T2 | End-to-end greedy parity vs reference `llama.cpp` (same pinned commit, same GGUF): 50-prompt suite, batch=1, seeded | ~~token-exact match ≥ 99.5%~~ *Amended 2026-09-29 (review R-9):* measured 25% (2/8) HIP vs CPU at M0.4/M1.4, so the frozen number is dead. The M1.4 rule: token-ID comparison where a divergence **passes** only if the reference leg's top-2 logit gap at the first divergent position is ≤ the calibrated epsilon (a benign tie), and every divergence is reported (position, tokens, gap). Write the calibrated epsilon here. Perplexity delta on wikitext-2 slice ≤ 0.5% |
| T3 | **Cache invariance** (spec I1): identical prompt under (a) fully pre-promoted cache, (b) cold start, (c) randomized eviction stress, ~~(d) floors-only~~ (dropped with floors, O3), (d) minimum VRAM cap (K = one routed set) | all four produce **identical greedy token IDs** — the direct test that caching never changes selection |
| T4 | Fork patches: upstream test suite green; P1 unset-hooks behavior byte-identical to upstream; P2 validity-bitset invariants fuzzed | no behavioral delta without hooks |
| T5 | Store: striping correctness across 1/2/4 replicas, per-chunk failover mid-read, checksum-rot detection (injected), same-drive warning | reads correct under all replica-count/failure combos |
| T6 | Placement concurrency: loom test over (access, page-in, evict, promote) interleavings; 24 h soak on hardware (amended 2026-09-29: default I/O engine; io_uring dropped, R-17) | no races, no residency-bitset drift vs pool contents |
| T7 | API contract: field-policy table (§8.2) becomes generated round-trip tests; SSE framing golden tests | unknown field ⇒ typed 400; stream shape stable |
| T8 | Plan vs reality: `golem plan` predicted VRAM vs actual after load, on hardware, over ctx sweep | ~~\|predicted − actual\| ≤ 256 MiB across sweep~~ (2026-10-18, #116) actual − predicted ≤ 256 MiB, predicted − actual ≤ 1 GiB across sweep (ws is a max of fitted slopes; over-prediction only shrinks the derived cache) |

### 12.3 Benchmarks and the documented baseline

*Amended 2026-09-29 (review H-7):* a test that cannot meet its precondition **fails**; it never returns early as a pass. Every P1–P5 change is evidenced by `--features hip-tests` output with zero `SKIPPED` lines through **engine-registered** hooks. Fixtures must match real shape where it matters: `.weight` names, a split GGUF with a metadata-only shard 1, mixed-stride K-quant planes, and a device budget smaller than one expert tensor. Every design deviation carries one line of arithmetic against the real model's numbers.

`golem bench` (dev binary, not a product subcommand): cold-start TTFP, warm TTFP, tok/s at ctx ∈ {4k, 16k, 32k} × cache ∈ {cold, warm, pre-promoted} × replicas ∈ {1, 2} (amended 2026-09-29: io_uring leg dropped, R-17) — the cold/warm split the spec demands. Every run stamps the baseline record: RAM, drives (model + link), kernel, ROCm version, fork commit, quant + revision, ctx. Baselines live in `docs/baselines/` as committed JSON. **No target numbers exist until the first M0 baseline lands**; the spec forbids inventing them and so does this design.

*Amended 2026-10-06: teacher-forced top-k parity.* Next to T2's greedy match rate, report a teacher-forced comparison against the reference runtime on the same token sequences. Per position, both sides emit their top-20 log-probs. Report:
- argmax agreement and top-5/top-10 overlap;
- KL over the reference's top 20 plus one bucket for the rest;
- perplexity of each side;
- disagreements bucketed by the reference's top-1/top-2 gap (< 0.2, 0.2–0.5, ≥ 0.5).

A raw match rate counts divergences; the gap buckets show whether they are near-ties. This is also the form in which O1 (CPU math) and O9 (MTP verify) state their determinism exceptions: "every disagreement has gap < x".

*Amended 2026-10-03 (BMoE, §16): A/B method.* A comparison between two settings runs as A, B, then B, A (cold state re-established before each run); a difference that follows the setting in both orders counts, one that follows run position does not. Every results file in `docs/baselines/` embeds the resolved config, engine version and fork commit (the §11 timing header), and decode numbers cite the per-step timing CSV that produced them.

---

## 13. Repository layout and toolchain

```
golem/
├── docs/                   # product-spec.md, tech-design.md, baselines/ (measured numbers), procedures/
├── engine/llama-cpp.lock          # pinned fork commit, ROCm line, patch list, M0-validated flags
├── vendor/llama.cpp/              # pristine fork at pin; P1–P5 patches + golem-hooks.h (M2.x + 2026-09-29 rework)
├── crates/
│   ├── golem-cli/                 # clap commands; thin — all verbs delegate to libraries
│   ├── golem-server/              # instance process: axum API, slot scheduler, decode loop
│   ├── golem-engine/              # llama.h glue: load config, sampler chain, batch building, KV mgmt
│   ├── golem-llama-sys/           # bindgen over llama.h + golem-hooks.h; cmake build of the fork
│   ├── golem-placement/           # scores, tiers, caps, prefetch; policy consumed by P2/P3
│   ├── golem-store/               # replicas, striping, checksums, pool, import
│   ├── golem-plan/                # §10 arithmetic (shared by CLI and runtime feasibility check)
│   ├── golem-api/                 # OpenAI types, SSE, error envelope, field-policy table
│   └── golem-registry/            # instance registry, UDS control, heartbeat
└── xtask/                         # bench harness, tiny-moe generator, parity suite runner
```

Toolchain and standards: Rust stable (currently 1.98, `rust-toolchain.toml` pins; edition 2024). `cargo clippy -D warnings`, `cargo fmt` check, `cargo nextest` in CI; deny.toml for license/advisory audit; `unsafe` confined to `golem-llama-sys` and the two FFI glue modules in `golem-engine`/`golem-placement` (hook trampolines) *(amended 2026-09-29, R-10: also present in `golem-cli/src/run.rs`, `golem-engine/src/session.rs`, `parity_op.rs`, and `golem-placement/src/pool/arena.rs`. Either list them or move them. Safe public APIs must not permit UB: slot access goes through lease/claim tokens, review H-1)*, each with a `// SAFETY:` contract comment and reviewed in T6. Key crates: tokio, axum 0.8 + tower-http, serde/serde_json, clap derive, thiserror/anyhow (thiserror in libraries, anyhow at binaries), tracing, bindgen, cmake, sha2, crc32c, libc, sysinfo, uuid.

---

## 14. Milestones

Each milestone below defines its *exit criteria*.

**M0 — Feasibility spike on the 7900 XTX (gates everything).**
Select and lock the fork commit (GLM-5.3-Flash text support present); build HIP for gfx1100; validate FA/precision settings for MLA on ROCm (the open risk the spec flags); run `--n-cpu-moe`-style host-buffered experts; measure: cold-expert copy cost per step, achievable disk bandwidth (1 vs 2 drives; io_uring leg dropped 2026-09-29, R-17), ROCm overhead, KV bytes/token, tok/s cold vs warm. Exit: T2 parity ≥ threshold on 8 prompts *and* measured cold tok/s above the "viable" line the author sets from the baseline (if disk→PCIe math says cold decode is unusable even with prefetch, the project re-scopes before more code exists). Deliverables: `engine/llama-cpp.lock`, baseline JSON #1, the M0 numbers feeding §10 constants.

**M1 — Substrate integration.** `golem-llama-sys` + `golem-engine`: load pinned GGUF, greedy single-slot decode from CLI, all non-expert weights resident, experts host-buffered with upstream default copy behavior. Exit: T1, T2 green on hardware; a chat completes.

**M2 — Placement engine.** P1–P4 patches (P5 + rework per §3.4 amendment) + `golem-placement` (scores, swap margin, caps, VRAM resident-set). Exit: T3, T4, T6 green; P2 skip-ratio and P3 slow-path metrics demonstrably better than naive per-step copies on the same trace.

**M3 — Storage tier.** `golem-store` (replicas, striping, checksums, pool, import) wired into P3/prefetch; two-drive mode. Exit: T5 green; cold-start TTFP and cold tok/s measured on 1 vs 2 replicas; degraded-replica failover demonstrated by yanking a drive (literal test procedure in the local M3 doc).

**M3.5 — Throughput decision gate (amended 2026-09-29, blocks M4).** Prerequisite: the Critical/High review fixes (all landed by PR #48; the remaining open findings are in §17), which stopped the pooled GPU path from running on the real model. In suggested order:
1. Test skips must fail (H-7); unsafe pool-slot API replaced by lease tokens (H-1).
2. Expert-name matching (C-1/M-9) and a multi-shard source (C-5).
3. Engine-enforced GPU `MUL_MAT_ID` (C-4).
4. The §3.4/§5.4 amendment: pool as the H2D host source (C-2), a slot-indirected device cache (C-3/H-5), load-mode policy (H-3), a pinned arena (H-4), and a per-model hook context (H-10).
5. CLOCK eviction plus slabs and a global pool (H-2, D-2/D-3); (layer, expert) as the residency unit (D-11); step = one decode (D-10); relative swap margin (D-9); I/O-thread and `--parallel` defaults (D-12, D-13).
6. Architecture fixes: MTP layer excluded from pool geometry (R-2), recurrent state in `golem plan` (R-1), per-slot state snapshot/restore (R-4), per-model hook context and batched routing readback (R-5/R-6).

Then benchmark on the real model, cold and warm reported separately, at the same RAM footprint:
- (a) Golem GPU-only.
- (b) Upstream `-ncmoe` CPU-hybrid.
- (c) Golem with a VRAM hit-rate sweep.
- (d) ~~(a)–(c) repeated with one lower-bit expert quant (§4.1).~~ *Dropped 2026-10-03 (O7); O7 re-ruled 2026-10-17 (no floor), so this may return as an optional leg.*
- (e) MTP speculative decoding acceptance rate and tok/s (§7.5).
- (f) Multi-turn chat latency with vs without state snapshots (§7.1).

Exit: the numbers are in `docs/baselines/`, compared against the §2.4 ceilings, and the user rules on O1 (I2) and O6 (MTP) with the measured numbers; O2–O5 were decided 2026-09-29. Prefetch redesign (H-6, next-layer router prediction) and store I/O (H-8 shared-queue striping, §5.2 as amended 2026-10-03; H-9 `O_DIRECT`) proceed in M3.3 alongside this gate.

**M3.6 — Prefill layer streaming (added 2026-10-03, D-7).** Prerequisite: M3.3, because streaming needs the shared-queue reader and `O_DIRECT` staging. Build §7.6: the P2 prefill-stream mode, the pinned staging ring, the plane pipeline, the §10 `stream_slabs`/`staging_ring` terms and `--prefill-stream-min`/`--prefill-ubatch`. Exit: HIP gate zero SKIPPED; MMQ `MUL_MAT_ID` verified on gfx1100 at ubatch 512–8192; `ws` measured per ubatch; cold and warm prefill tok/s on the real model at 512, 2048 and the largest ubatch that fits recorded in `docs/baselines/` against the §2.4 estimate (≈21 / ≈85 tok/s at the measured 4.73 GB/s; ≈38 / ≈150 before M3.3); P4 RAM-hit rate for decode unchanged across a long prefill.

*Amended 2026-10-03 (BMoE, §16): which milestone carries each 2026-10-03 amendment.* M3.3 store-wiring: layer-batched cold reads (§3.4 P3) feeding the shared queue, and the per-step timing CSV (§11) so its throughput claim cites I/O vs H2D vs compute. M3.8 hook-context: the P3 per-layer ABI change, in the same ABI bump. M3.7 predictive-prefetch: offline policy replay first (§6.3), then the prefetch kill switch (§6.5) and async VRAM promotion (§6.3). M3.6 prefill-stream: routed-only fill for short ubatches (§7.6). M3.9 cpu-expert-math: CPU/DMA miss split (§2.4). M3.10 mtp-spike: acceptance per draft position, U_k at k ∈ {2, 3, 4}, divergence vs plain greedy (§7.5). Every bench in these uses the §12.3 A/B method.

*Amended 2026-10-06 (O8–O10): attack order for the remaining M3 work*, ranked by expected decode gain under O8:
1. **Pool-size baseline.** Measure only. Decode tok/s at `--ram-experts-gib` 10 / 40 / 63 (the 63 is the 75%-of-available preflight cap on this box), cold and warm, A/B both orders, with the M3.3 timing CSV. Every Golem decode number so far used a 10 GiB pool (5.8% of the expert bytes).
2. **M3.8 hook-context.** Routing traces, which feed the profile, the replay and U_k.
3. **M3.7 predictive-prefetch.** Replay → hit-rate-vs-capacity curve and d per pool size; routing profile + async VRAM promotion (§6.3); lookahead `protect`/`read` A/B (§6.5).
4. **Exclusive residency (O10, §6.3).** A new task, sized by the step-3 curve.
5. **M3.10 mtp-spike (O9, §7.5).** On the substrate's MTP context, with prompt lookup.
6. **M3.9 cpu-expert-math (O8, §2.4).** The CPU+DMA split, gated on the measured d.
7. **M3.6 prefill-stream and M4.8 state-snapshots.** Unchanged.

**M4 — Product surface.** `golem-api` + `golem-server` HTTP, slots/batching, reasoning content; `golem` CLI (run/serve/list/stop/plan/import), registry; `golem-plan` with T8 calibration. Exit: T7, T8 green; the spec's five commands behave as §9 pins them.

**M5 — Validation hardening.** Full T2 suite vs reference runtime, 24 h soak (T6), baselines for cold/warm across the §12.3 matrix, README quickstart, baseline doc per spec §Acceptance. Exit: acceptance criteria of the product spec demonstrably met on the author's machine, with numbers committed under `docs/baselines/`.

---

## 15. Risks and mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| GLM support unmerged upstream; fork chases a moving PR | certain (spec states it) | high | pin commit in M0; patches confined and rebased monthly; documented trigger to re-open D2 |
| ROCm MLA/precision behavior differs from NVIDIA workarounds | likely | high | explicit M0 exit criterion; `--fa` escape hatch; settings frozen into lock file once validated |
| PCIe becomes the bottleneck; cold decode unusably slow | **likely** (§2.4: ≈4.9/(1−h) tok/s ceiling; CPU-hybrid ≈2.5–3× faster at low h) | existential for the product | M3.5 gate benchmarks GPU-only vs CPU-hybrid before M4; levers in order: lower-bit expert quant (§4.1, reopened by O7 2026-10-17), pinned H2D (§3.5), VRAM hit rate (P2 slot cache + predictive prefetch), prefill streaming (§7.6), shared-queue striping (§5.2); if still below the hybrid, user rules on relaxing I2 (§1) |
| Fork patch conflicts on rebase | **high** (amended 2026-09-29, D-14) | **high** | the required P2/P3/P5 rework is in the scheduler's expert-input path, one of upstream's most-changed areas; isolate behind one extension point; T4 as rebase gate; re-open D2 option B past ~1.5k lines or two conflicting rebases |
| Cold prefill unusably slow (0.24 tok/s measured at M3.5) | certain without streaming | high (agentic prompts are 10k–35k tokens) | §7.6 layer streaming at ubatch ≤ 2048 (≈150 tok/s estimated, disk-bound); validate the MMQ `MUL_MAT_ID` kernel at large ubatch on gfx1100 first |
| Cache-state-dependent nondeterminism (I1 violation) | low | fatal to the core claim | T3 is a first-class test from M2 onward, run in nightly hardware CI |
| Multi-instance VRAM oversubscription crashes the driver | medium | medium | blocked by default when planned VRAM or locked RAM exceeds physical (§9.3, O5), with `--allow-overcommit` as a loud override; per-instance budgets |
| RAM pool + page cache pressure on smaller-RAM machines | medium | low | pool default ≤75% free RAM, mlock'd; `--ram-experts-gib`; plan reports RAM fit explicitly |
| Hybrid recurrent architecture breaks KV-rewind assumptions (slot cancel, prefix reuse) | certain (amended 2026-09-29, R-1) | high | state snapshot/restore (§7.1); no partial rewind anywhere |
| Checksum verify slows startup (~186 GiB) | certain | low | amended 2026-09-29 (R-8): full verify at import only; start-time size + mtime + sampled hash; `--full-verify` opt-in; CRC32C at page-in |

---

## 16. References

- Product spec: `product-spec.md` (same repo).
- llama.cpp: GLM MoE support PR #27754 (open at spec-writing time); feature matrix (ROCm/Vulkan capability evidence); `ggml-backend.cpp` split-scheduler expert-copy path; `-ncmoe`/`-ot`/`--lazy-mode` docs — all retrieved via context7, 2026-09.
- Comparison projects per spec: rabbit (CPU-only disk-streamed experts), deltafin (GPU paths; ROCm path unvalidated), colibri (tiering + dual-SSD reference; GPU-hot/CPU-rest split — the exact gap Golem closes).
- context7 findings on alternatives: candle quantized path CUDA-only; mistral.rs layer-granular topology, no per-expert disk tier (§3.1, option C).
- r/LocalAIStack, "GLM-5.3-Flash UD-IQ4_XS running on 5090+64GB Ram with dual SSD-Offloading" (u/zMytze, 2026-09; project `glm53-ssd-streaming`, built on unmerged llama.cpp expert-streaming PR #25294). It is the external reference for §7.6 and the 2026-10-03 §5.2 amendment: layer-streamed prefill through pinned double buffers at ubatch 2048, both NVMe drives reading one mirrored GGUF from a shared work queue, an `O_DIRECT` RAM expert cache, and an MMQ padding fix for `-ub 2048` on Blackwell. Reported: ≈230–250 tok/s prefill, 5.6 tok/s decode, 78% RAM-cache hit, ≈1.3 GB/token from SSD. These are not measured on this machine and are not targets (spec §Acceptance).
- AMD/ROCm prior art surveyed 2026-10-03 (none are drop-in for Golem): **bsvinay/exllamav3-rocm-moe** (ExLlamaV3 ROCm port, RX 7900 XTX + 7950X3D + 60 GB). Hot experts sit in VRAM and the rest run on CPU AVX-512 kernels (DRAM-bound ≈50 GB/s). Its learned placement is persisted across restarts (CPU share 0.47 → 0.15 on the first request), which corroborates persisting the §6.2 scores file for warm start. Prefill at 8192-token chunks ≈2× faster than at 2048 (§7.6 ubatch size). MTP 2-token drafts take decode from 53 to 83 tok/s on code (O6). Its llama.cpp `--n-cpu-moe` reference on the same GPU is 17.5 tok/s. **GenerelSchwerz/llama.cpp** `moe-cache`: a VRAM MoE expert cache, CUDA-only by its own docs, so it gives no ROCm evidence. All numbers are the authors', not measured here.
- Upstream MoE expert caches surveyed 2026-10-08 (authors' numbers, all-experts-in-RAM hosts, so their 1.57–2.2× decode does not transfer: Golem's 172.8 GiB of experts vs 93 GiB RAM leaves disk at 151–237 ms/token, m3.9). **llama.cpp #29887** (merged): `--moe-cache-mib` LRU over host experts for `MUL_MAT_ID` at batch ≤ 32, the same job as P2; 72% hit at 6.4 GB, 89% at 18.6 GB, on 65 GiB of experts (4090/5090). **#30112** (merged): the same per GPU (single-GPU here, n/a); prompt t/s drops to 0.91×/0.73× with the cache on. **#27861** (open): 54k-record Qwen3.8-Flash-Next routing trace, top-32 static list ≈10% coverage vs per-layer LRU-64 ≈67% and LRU-128 ≈81%, consistent with m3.7 keeping the decayed score over LRU (+2.1 VRAM-hit points, §6.2); its exact CPU/GPU split (cached ids → second `mul_mat_id` chain, uncached → zero slot, CPU skips cached ids, outputs summed, async uploads) is the template for O1. **Discussion #24528**: hybrid hit/miss RFC; fill on decode only, since prompt routing is flat and thrashes. **neurall/llama.cpp**: GLM-5.3-Flash at 3.0 bit (O7 forbids), MTP no gain (matches O6).
- Reference designs surveyed 2026-10-03 for the second 2026-10-03 amendment (authors' numbers, other hardware, not targets): **Helldez/BigMoeOnEdge** (BMoE): CPU-only, lossless expert streaming on phones through llama.cpp's public API (`cb_eval` on the `ffn_moe_topk` node, `->data` repointing). Source of: parallel read lanes (≈3× on phone flash; §3.4 P3 layer-batched reads), per-token timing CSV with `unmeasured` columns and config header (§11), routed-only prefill fill with an 85% whole-layer switch (§7.6), next-layer prediction at 80–89% accuracy that still lost 21–38% when I/O-bound (§6.5 kill switch), hit rate ≈ linear in cache size and a smarter policy +2 points / −30% speed (§6.3 replay), MTP acceptance 71/60/52% at k = 2/3/4 and batch-dependent output (§7.5), A/B in both orders (§12.3).

---

## 17. Open review findings (moved here 2026-10-06)

Every finding from the design/code reviews (2026-09-29 onward) is either fixed in code or listed below, re-checked against `7a43607`. IDs are kept because amended passages above cite them.

**Closed since the last review pass:**
- H-8: M3.3 shared-queue striping (§5.2).
- D-4: per-plane CRC32C computed at import and carried in the manifest `expert_index` (§5.3).
- D-5: O7.
- D-6: §2.4.
- H-6: design side done (§6.5); the code landed in M3.7.
- P11: `PolicySnapshot` excludes are a per-layer bitset (M3.7).
- P17: a prefetch claim never evicts a reference-bit key and leaves CLOCK state untouched (M3.7).
- P23: `Policy::pass` stays unconsumed: production VRAM is per-layer forced-admission slabs, so only its score math is used (`--vram-order score`, §6.3, M3.7). Not replayed.

**Design / architecture:**
- **R-1 (§2.5, §10):** `golem plan` must derive memory from the hybrid architecture, per layer type: recurrent state × parallel, plus KV for the 11 attention layers only. The §12.3 ctx sweep is still picked blind. Owner: M4.6. *Status (M4, #32):* `golem_plan::plan` derives KV over the attention layers and recurrent state × parallel from the header (target: 19,712 B/token, 145.56 MiB); `golem plan` and the engine's load-time check share it. The T8 hardware sweep (`t8_plan_vram.rs`) has run (PR #118 and the #122
acceptance run); `golem_plan::WS_MEASURED` is fitted from the measured
compute buffer (see `docs/baselines/m4.6-plan-t8.md`). The ws fit holds
within 256 MiB at all fitted points, but the total-delta gate on the
widened grid is still open — the residual scatter is non-ws (device-cache
residency accounting), tracked in that baseline.
- **R-4 (§7.1):** every `chat_turn` re-prefills the whole transcript (`session.rs`, `llama_memory_clear` per turn). The fix is state snapshot/restore. *Status (M4, #54):* implemented — prompt-prefill snapshots in a pinned arena, restore + delta prefill (`snapshot.rs`); byte-identical on tiny-moe and tiny-hybrid. Real model (`docs/baselines/m4.8-snapshots.md`): turn-5 TTFT 49.6–52.6 s → 9.9–10.1 s cold, 49.4–54.7 s → 9.8 s warm; greedy output not byte-identical on vs off (open).
- **R-5 / H-10 (§3.4 P1):** hooks, device-cache map, budget and stats are process-global. The fix is a per-model `golem_ctx*` and an unregister that waits for in-flight callbacks. Mitigated today by refusing a second session or a re-enable.
- **R-6 (§2.3 step 4):** the per-tensor `policy_query` and per-expert `notify_access` callbacks should become one ids-tensor read per MoE layer per step, with the eviction order pushed to C as context-owned data. The per-call cost is already allocation- and lock-free (M-1).
- **R-8 (§4.3):** start-time verify should be size + mtime + sampled hash, with `--full-verify` opt-in. *Status (M4, #33):* implemented — `golem-engine/src/verify.rs` checks size + mtime + 16 × 1 MiB sampled SHA-256 (`i·len/16`) against manifest fields written by `golem import` (manifests without them: size + mtime + a warning); `--full-verify` re-hashes every byte. Real-model start times (cold, sampled vs full) pending the HW leg in `docs/baselines/m4.7-start-verify.md`. The expert-layer-set check against `block_count − nextn_predict_layers` is not part of it.
- **D-14 (§3.2):** the scheduler patch-size trigger (≈1.5k lines, or two conflicting rebases) re-opens D2 option B. Re-estimate at every vendor rebase.
- **R-15:** CCD pinning, with decode on the V-cache CCD and I/O threads elsewhere. *Status (M4, #33):* implemented — `--pin auto` (default) detects the V-cache CCD from `cache/index3/size`, the decode worker pins itself before `Session::new` (ggml's threadpool inherits), store readers and the prefetch thread pin to the other CCD at spawn; no-op with one CCD. CPU threadpool default is physical cores / 2. Throughput effect unmeasured.

**Code:**
- **M-2:** `golem-store` `read_page` takes a caller covering buffer, so it no longer bounces. Confirm no `Vec` bounce is left on the P3 path once M3.3's `store_source` is the only source. Otherwise closed.
- **M-3:** three GGUF readers remain: `golem-gguf`, `golem-engine/src/gguf.rs` (a KV/tensor layer on top of it) and `xtask/src/gguf.rs` (writer + `read_arch`). There are also two expert indexes: the manifest `expert_index` and the pool geometry. Collapse to one reader and one index (AGENTS.md "one implementation per concern").
- **M-4:** `golem-llama-sys/pin_gate.rs` hand-rolls SHA-256 (`struct Sha256`). Use workspace `sha2` as a build-dependency and keep the `sha256sum` conformance test.
- **M-7:** `ResidencyMap::claim` spins on `yield_now` when every slot is leased (`residency.rs`). Park on the shard condvar and notify from `release_lease`.
- **M-11:** milestone jargon in `scripts/` docstrings (`parity-suite.py`, `manifest-builder.py`, `disk-bandwidth.py`, `gguf-header-gate.py`).
- **L-3:** `session.rs` panics when `llama_get_logits_ith` returns null, in two places. A library should return `EngineError`.
- **L-4:** `#![allow(unexpected_cfgs)]` in six `golem-placement/src/pool/*.rs` files. Declare `check-cfg = ['cfg(loom)']` under `[lints.rust]` instead.
- **L-5:** hand-written `Display` impls for `ArenaError`, `SourceError`, `PoolError` and `PinError`, while `thiserror` is a workspace dependency.
- **L-6:** `Policy::decay_factor` uses `max − min`, which hides a non-monotonic step. Use `saturating_sub` plus a `debug_assert`.
- **L-7:** `Policy::pass` clones the config and allocates per pass. Reuse scratch once the placement thread runs at ≥ 1 kHz.
- **L-8:** `[workspace.dependencies]` predeclares tokio, axum, anyhow and uuid before any crate uses them. Add them when used (M4).
- **L-9:** `bindgen.size_t_is_usize(false)` in `golem-llama-sys/build.rs` forces casts. The default (`true`) is idiomatic on Linux.

**Hardware evidence still owed:** the M3.3 live failover demo and the cold decode before/after (`docs/baselines/m3.3-store-bench.md` §3–4, deferred).
