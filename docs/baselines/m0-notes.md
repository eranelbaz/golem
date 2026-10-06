# M0 notes — running record of the feasibility spike

## M0.2 — HIP build + smoke (executed 2026-09-20, PR #4)

- Pin built: `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9` (content hash verified pre-build by the script — `pin-ok` gate uses the ADR 0001 canonical `cs()` recipe with `./build` pruned; pristine-tree hash identical to `vendor/VENDOR_SHA`; ADR amended to canonize the build-pruned form).
- ROCm: **7.2.4** at `/opt/rocm` (`/opt/rocm/.info/version`); matches lock `rocm = 7.2.4` — script prints `rocm-check: ok (7.2.4 satisfies lock 7.2.4)` (wording updated in follow-ups; PR #4's log said `matches`).
- Configure: `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release`; build wall-clock **≈83 s** on 32 threads (uncaptured `time` — from session wall clock; build-log.txt is untimed — future tasks: capture `time` output, M0.4 is measurement-heavy).
- CMakeCache: `GGML_HIP:BOOL=ON`, `AMDGPU_TARGETS:UNINITIALIZED=gfx1100`.
- Devices: `llama-cli --list-devices` →
  ```
  Available devices:
    ROCm0: AMD Radeon RX 7900 XTX (24560 MiB, 24524 MiB free)
  ```
- Smoke: `stories15M-q4_0.gguf` (`llama` arch, 19,077,344 bytes — build validation only; GLM gate is M0.3), `-ngl 99 --no-warmup --seed 42 -st`, exit 0. Note: this fork's `llama-cli` is a chat UI; `-st` (single-turn) is the non-interactive mode — `-no-cnv` no longer exists. Key log lines (source `.m0-work/smoke-verbose.log` unless noted):
  ```
  llama_prepare_model_devices: using device ROCm0 (AMD Radeon RX 7900 XTX) (0000:03:00.0) - 24524 MiB free
  load_tensors: offloaded 7/7 layers to GPU
  system_info: ... | ROCm : NO_VMM = 1 | FA_QUANTS = q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16 | ...
  [ Prompt: 305.5 t/s | Generation: 1366.0 t/s ]
  ```
  (PR #4's notes quoted `289.5/1367.7 t/s` from the non-verbose run, `.m0-work/smoke-log.txt`; the verbose-run numbers above are from the same seed/config. Output tokens coherent for tinyllamas (tiny-model quality). GPU-offload confirmed by layer assignment lines (layers 0–6 all `assigned to device ROCm0`).)
- Build log excerpt (tail):
  ```
  [100%] Building CXX object tools/cli/CMakeFiles/llama-cli.dir/main.cpp.o
  [100%] Linking CXX executable ../../bin/llama-cli
  [100%] Built target llama-cli
  BUILD-OK: vendor/llama.cpp/build/bin/llama-cli
  ```
- Script failure modes exercised before the real build: drift injection → `FATAL` + exit 1; restricted PATH + cmake stub → `rocm-check: skipped (rocminfo absent)`, flow proven to reach configure with exactly `-S vendor/llama.cpp -B vendor/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100`.

## M0.3 — model-manifest (executed 2026-09-20, this PR)

- Repo `unsloth/GLM-5.3-Flash-GGUF`, revision pinned `621d456e93e926e4b52f85cff5f634358c1828f9`, arch **`glm5next`** (gate on real shard-1 headers of both quants — `GATE-OK`; M0.1 pin cross-check passed).
- Header facts (both quants identical): `block_count` 46, `expert_count` 288, `expert_used_count` 8, `leading_dense_block_count` 3, `embedding_length` 4096, `attention.head_count` 64, `head_count_kv` 1, `context_length` 1,048,576.
- **UD-Q4_K_XL** (primary): 6 shards, total **199,707,321,347 bytes** at `/home/eranelbaz/models/glm-5.3-flash-ud-q4k-xl/UD-Q4_K_XL/`; replica (full mirror, byte-verified) at `/mnt/data/golem/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/` (nvme1n1).
- **Q8_0** (parity): 8 shards, total **340,981,966,112 bytes** (single copy) at `/home/eranelbaz/models/glm-5.3-flash-q8-0/Q8_0/`.
- Expert tensors: **129** = (46 − 3 dense) × 3 (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`), 288 experts/layer, bytes-per-expert uniform **per (layer, tensor)** (division asserted). Q4 typical: gate/up 4,718,592, down 5,767,168 — outliers (see manifest): layer 11 gate/up = 5,767,168; layers 11/12/44 down = 6,881,280. The §4.3 uniformity check must therefore compare per `(layer, tensor)`, never per tensor kind. Q8 bytes/expert: 8,912,896 (all three).
- Provenance: shards downloaded 2026-09-20 (earlier same-day session); this session verified all 14 shards **content-identical to the Hub head via LFS-sha256 comparison** (tree API `lfs.oid` vs local `sha256sum`); manifests then built by `scripts/manifest-builder.py` (independent streamed hashing) and re-verified with `sha256sum -c` on primary **and** replica — all OK.
- Manifests: `docs/baselines/manifests/glm-5.3-flash-ud-q4k-xl.json`, `docs/baselines/manifests/glm-5.3-flash-q8_0.json` (design §4.2 schema; `chat_template_sha` over `tokenizer.chat_template`). M3.2 `golem pull` must reproduce the Q4 manifest diff-clean.
- Replica-path convention: the `shards[].replicas` arrays hold host-local absolute paths as-of pinning (primary `/home/eranelbaz/...`, replica `/mnt/data/...`). These are provenance records, not a portable contract — M3.2 `golem pull` must rebuild `replicas` from its own resolved locations and compare only file/bytes/sha256 for the diff-clean check.
- Timing caveat: wall-clock times for hashing were not captured per-step; NVMe hashing ran at ≳1 GB/s aggregate. M0.4 is the measurement task.

## M0.4 — m0-measurements (executed 2026-09-20, this branch)

Plan: `docs/superpowers/plans/2026-09-20-m0.4-m0-measurements.md` (plan review 3 rounds; round-1 Critical caught the `-ncmoe` layer-arithmetic error, rounds 2–3 Minor numeric/unit fixes). Kernel: `$(uname -r)` printed above this section's execution.

### As-executed deviations from plan (all recorded in PR body)

1. **`-ncmoe` semantics at this pin are whole-tensor**: blocks ≥ N get full VRAM expert buffers (`-ncmoe 16` probe OOM'd allocating 136 GiB on ROCm0). Verbose logs show the feasible mode: blocks < N experts = **CPU_Mapped** (mmap-backed; 46,136+47,621+46,329+45,039 MiB), ROCm0 model buffer 8,550.87 MiB. Feasible band N ∈ {43,44,45} + `-cmoe`; original sweep N ∈ {16,12,8} infeasible.
2. **Model 186 GiB > RAM+VRAM 117 GiB** ⇒ every viable run is nearly-all-mmap-paged = cold disk tier via page cache. Cold gen tok/s **0.6–0.8**.
3. **FA decision via 256-token greedy agreement** (ppl infeasible: prompt 0.6 t/s ⇒ 2048-token window ≈ 1 h): fa-off vs fa-on both coherent, 48-word common prefix then gradual divergence → locked `fa = on`; formal ppl/parity gate = M0.5.
4. **Copy cost inconclusive**: sweep N ∈ {45,44,43} → 0.8/0.7/0.6 t/s, INVERTED vs copy model; page-cache noise ±20% >> ~5% expected copy signal. P4 counters (M2.4) own this measurement.
5. **KV = 19,712 B/token** (log-derived; VRAM-delta cross-check 19,802 = 0.4% agreement). MLA: K-only f16 (V=0), two buffers (512+128), plus fixed 145.56 MiB R/S decode state. **VRAM overhead combined = 2.64 GiB** (peak − idle − 8.35 GiB ROCm0 model buffer − KV).
6. **Disk bench on shard 2** (shard 1 is 9.4 MB metadata-only). O_DIRECT pread64, 4 GiB/leg: nvme0n1 seq/rand t1 **5.81/5.76** GiB/s; nvme1n1 seq/rand t1 **2.24/2.16**; striped 2-drive seq **4.7–4.9**, rand **4.6** (t2). t8 legs REGRESS to ~1.9–2.0 on both drives. **Hardware flag: reproducible EIO on nvme0n1 under concurrent O_DIRECT** (intermittent t2, consistent t8, single-thread clean; M0.3 single-stream hashes all passed) — recommend SMART long test before M3.
7. GNU `/usr/bin/time` absent (no sudo) → python wrapper (`MEAS wall_s=… max_rss_kib=…`); rocm-smi on ROCm 7.2.4 rejects `--gpu 0` → `--json` sampling.

### GLM-RUN summary lines (verbatim)

```
GLM-RUN: name=smoke-tiny rc=0 idle_vram_bytes=1891880960 peak_vram_bytes=1891930112 ts="[ Prompt: 302.3 t/s | Generation: 1052.5 t/s ]" gen_ts="Generation: 1052.5 t/s"
GLM-RUN: name=probe-ncmoe45 rc=0 idle_vram_bytes=1947078656 peak_vram_bytes=13808099328 ts="[ Prompt: 0.6 t/s | Generation: 0.6 t/s ]" gen_ts="Generation: 0.6 t/s"
GLM-RUN: name=fa-off rc=0 idle_vram_bytes=1985232896 peak_vram_bytes=13849489408 ts="[ Prompt: 0.5 t/s | Generation: 0.8 t/s ]" gen_ts="Generation: 0.8 t/s"
GLM-RUN: name=fa-on rc=0 idle_vram_bytes=1916751872 peak_vram_bytes=13785247744 ts="[ Prompt: 0.6 t/s | Generation: 0.8 t/s ]" gen_ts="Generation: 0.8 t/s"
GLM-RUN: name=kv-4k rc=0 idle_vram_bytes=1916751872 peak_vram_bytes=13796962304 ts="[ Prompt: 0.5 t/s | Generation: 0.6 t/s ]" gen_ts="Generation: 0.6 t/s"
GLM-RUN: name=kv-16k rc=0 idle_vram_bytes=1940242432 peak_vram_bytes=14040285184 ts="[ Prompt: 0.6 t/s | Generation: 0.6 t/s ]" gen_ts="Generation: 0.6 t/s"
GLM-RUN: name=sweep-44 rc=0 idle_vram_bytes=1940242432 peak_vram_bytes=18515132416 ts="[ Prompt: 0.6 t/s | Generation: 0.7 t/s ]" gen_ts="Generation: 0.7 t/s"
GLM-RUN: name=sweep-43 rc=0 idle_vram_bytes=1924358144 peak_vram_bytes=22872653824 ts="[ Prompt: 0.6 t/s | Generation: 0.6 t/s ]" gen_ts="Generation: 0.6 t/s"
```
(fa-off leg at `-ncmoe 42` OOM'd pre-amendment: `allocating 129960.87 MiB on device 0: cudaMalloc failed: out of memory` — the probe that established whole-tensor placement.)

### Generation texts (fa-off / fa-on, first ~600 chars)

Stored verbatim in `.m0-work/m04/fa-{off,on}.out`. Both begin `[Start thinking]\n\nThe user wants a short story about a robot who learns to paint…` with structured brainstorming; fa-on diverges from fa-off at word 48 and stays coherent.

### M0.4 follow-up — EIO diagnosis (same day)

Hardware flag RESOLVED: not the drive. btrfs (`/home`, `compress=zstd:3`) fails concurrent O_DIRECT reads
with EIO (t2 2/5, t4+ 5/5 legs), ext4 (`/mnt/data`) never fails; single-thread always clean (full 45.8 GiB
sweep OK); journal shows no nvme errors; AER clean. Design consequence for M3.1: O_DIRECT viability probe
per replica at open + buffered fallback. The separate "8 threads slower than 1" regression reproduces on
ext4 too and stays unexplained (recorded in baseline JSON).

## M0.5 (2026-09-21, session 10) — parity legs + baseline-001
As-executed: legs from ext4 replica; CPU substrate GGML_CPU_REPACK=OFF; desktop-safe cgroup recipe (see plan session-10 amendment). Driver: .m0-work/m05/run-all.sh under golem-m05-legs.service; ~5.5 h total.
Leg wall times (approx): smoke-cpu 11.4 min (load 205 s + gen 480 s); cpu-t16 ~111 min; cpu-t4 ~93 min; hip ~180 min (slower than expected — per-prompt reload dominates).
No n reduction needed (96 tokens throughout, zero prompt failures across all legs).
Verbatim PARITY lines:
  PARITY: a=cpu-t16 b=cpu-t4 match=8/8 prefixes=[None×8]
  PARITY: a=hip b=cpu-t16 match=2/8 prefixes=[49,25,33,31,34,27,None,None]
  PARITY: a=hip b=cpu-t4 match=2/8 prefixes=[49,25,33,31,34,27,None,None]
  PARITY-LEG: name=smoke-cpu ok=1/1 / cpu-t16 ok=8/8 / cpu-t4 ok=8/8 / hip ok=8/8
Floor 100% (CPU-only pair) ⇒ threshold 100 ⇒ parity 25% < 100 → RE-SCOPE (method-level; token-ID T2 brought forward to M1.4). See m0-exit-review.md.

## Lazy-regime recipe for the warm/cold re-baseline

The 2026-09-21 kernel update (linux-omarchy 7.2.5-4) livelocks llama.cpp's `MAP_POPULATE` expert pre-fault, so `baseline-001.json`'s warm/cold tok/s pair is deferred. Re-run cold and warm under this exact regime, then gate:

- Flags: `-ngl 99 -ncmoe 45 -fa on -t 16 -c 4096 --seed 42 --lazy-mode on --ctx-checkpoints 0 --no-op-offload`.
  - `--lazy-mode on` switches CPU-resident experts to per-row `read()`, bypassing the broken `MAP_POPULATE` fault path.
  - `--ctx-checkpoints 0` avoids an embedded-server task-queue deadlock.
  - `-st` deadlocks the server flow, so it is not used.
  - Use `-p` (not `-f`) and pipe `/exit` to stdin for a clean rc=0.
- An `LD_PRELOAD` shim (machine-local, `.m0-work/m04/mmapshim4.c`) strips `MAP_POPULATE` from `.gguf` mappings and sets `MADV_NOHUGEPAGE`. It affects the load path only, not generation math.
- 5-leg protocol, warm-up leg discarded; tok/s = `gen_ts` from the GLM-RUN summary.
- Cold legs need page-cache eviction first (`scripts/evict-model-cache.py`; no root on the rig).
- Model: unsloth/GLM-5.3-Flash-GGUF@621d456 UD-Q4_K_XL.
