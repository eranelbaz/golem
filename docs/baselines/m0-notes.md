# M0 notes — running record of the feasibility spike

## M0.2 — HIP build + smoke (2026-09-19)

- Pin built: `86ebfef2c6a0f3359a2a07d2c215d61b0fa885c9` (content hash verified pre-build by the script — `pin-ok` gate uses the ADR 0001 canonical `cs()` recipe with `./build` pruned; pristine-tree hash identical to `vendor/VENDOR_SHA`).
- ROCm: **7.2.4** at `/opt/rocm` (`/opt/rocm/.info/version`); matches lock `rocm = 7.2.4` — script prints `rocm-check: ok (7.2.4 matches lock 7.2.4)`.
- Configure: `-DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100 -DCMAKE_BUILD_TYPE=Release`; build wall-clock **83 s** on 32 threads (first full HIP build; precompiled HIP kernels ship in ROCm, so compile cost is mostly C++).
- CMakeCache: `GGML_HIP:BOOL=ON`, `AMDGPU_TARGETS:UNINITIALIZED=gfx1100`.
- Devices: `llama-cli --list-devices` →
  ```
  Available devices:
    ROCm0: AMD Radeon RX 7900 XTX (24560 MiB, 24524 MiB free)
  ```
- Smoke: `stories15M-q4_0.gguf` (`llama` arch, 19,077,344 bytes — build validation only; GLM gate is M0.3), `-ngl 99 --no-warmup --seed 42 -st`, exit 0. Note: this fork's `llama-cli` is a chat UI; `-st` (single-turn) is the non-interactive mode — `-no-cnv` no longer exists. Key log lines:
  ```
  llama_prepare_model_devices: using device ROCm0 (AMD Radeon RX 7900 XTX) (0000:03:00.0) - 24524 MiB free
  load_tensors: offloaded 7/7 layers to GPU
  system_info: ... | ROCm : NO_VMM = 1 | FA_QUANTS = q4_0-q4_0,q8_0-q8_0,f16-f16,bf16-bf16 | ...
  [ Prompt: 289.5 t/s | Generation: 1367.7 t/s ]
  ```
  Output tokens coherent for tinyllamas (tiny-model quality). GPU-offload confirmed by layer assignment lines (layers 0–6 all `assigned to device ROCm0`).
- Build log excerpt (tail):
  ```
  [100%] Building CXX object tools/cli/CMakeFiles/llama-cli.dir/main.cpp.o
  [100%] Linking CXX executable ../../bin/llama-cli
  [100%] Built target llama-cli
  BUILD-OK: vendor/llama.cpp/build/bin/llama-cli
  ```
- Script failure modes exercised before the real build: drift injection → `FATAL` + exit 1; restricted PATH + cmake stub → `rocm-check: skipped (rocminfo absent)`, flow proven to reach configure with exactly `-S vendor/llama.cpp -B vendor/llama.cpp/build -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100`.
