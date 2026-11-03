# #208: prefill staging reads planes in place

Machine and model as in `205-drain-read-depth.md`: 32 threads, 93 GiB RAM, 24 GiB VRAM (gfx1100), two replicas (9100 PRO + 990 PRO), GLM-5.3-Flash UD-IQ3_XXS, 4 split shards, knobs from `~/.config/golem/golem.toml`. Baseline = `origin/main` 2217755, new = this branch.

`PERF_LEGS='^(base|new)-(d2|pf)$' scripts/perf-205.sh`, 4 reps ABBA (even reps reverse leg order and cold/warm order). Median [min–max].

## Cause

Every expert tensor base sits 32–4000 B past a page boundary; strides are page multiples. The prefill stream reader staged each plane at `head = 0` in an exact plane-sized buffer, so every staged plane failed the head check and took the scratch copy. Staged experts now sit at `id × plane_len + head` with a page-rounded read region; pointers stay `plane_len` apart, so C still coalesces H2D runs.

## Result

| | base | new |
|---|---|---|
| prefill s cold | 45.9 | 39.5 (−14%) |
| prefill s warm | 47.4 | 38.2 (−19%) |
| non-decode s cold (pf) | 68.0 [67.5–69.8] | 61.7 [61.4–72.9] |
| non-decode s warm (pf) | 67.5 [65.5–70.1] | 58.2 [57.6–59.8] |
| decode tok/s cold | 6.416 [6.345–6.425] | 6.405 [6.376–6.428] |
| decode tok/s warm | 6.436 [6.418–6.451] | 6.396 [6.346–6.440] |
| disk GB/token | 0.58 | 0.58 |
| scratch_planes, prefill run | 58 273 | 126 |
| scratch_planes, decode run | 126 | 126 |

The remaining 126 are the load-time verification read (one per expert tensor, 42 layers × 3 planes, head 0): `head=126 len=0 align=0`. Decode ranges overlap; the warm median is −0.6%, within spread.

## Memory

Staging grows by one page per slot: 6 × 4 KiB = 24 KiB RAM on 6 × 288 × 4.46 MB ≈ 7.2 GiB of staging. VRAM unchanged (device slabs stay 2 × 288 × max stride).
