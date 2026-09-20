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
