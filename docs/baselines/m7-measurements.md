# M7: tier endpoint overhead, role-major page-in, CPU expert math (GLM)

Machine: 32 threads, 93 GiB RAM, 24 GiB VRAM (gfx1100). Model: GLM-5.3-Flash UD-IQ3_XXS, 4 split shards, manifest. Release builds: new = `d463fbe`, baseline = origin/main `6c75950`. Driver: `scripts/m7-measure.sh` (run 2026-11-08, `.ab-work/m7-20261108-160725`).

Every leg: `ab-bench.sh` with `--new --async-h2d`, 63 GiB pool, 128 tokens, ctx 1024, greedy, plus golem.toml's GLM knobs (`--device-cache-experts 8 --vram-reserve-gib 3.0 --vram-experts-frac 0.8 --prefill-ubatch 512 --flash-attn`) and top-level keys (`prefetch = read`, `prefill_gate = predicted`, q8_0 KV, 8 threads). Every arm ran 4 reps cold and 4 warm, ABBA then BAAB. Cold = `drop_caches` right before the leg. Means are Σtokens / Σwall over the 127 decode rows.

Memory: 63 GiB pool + ≈ 20 GiB process RSS ≈ 83 of 93 GiB.

## #200: `GET /v1/golem/experts` overhead

| arm | tok/s cold (4 reps) | tok/s warm (4 reps) | mean cold / warm |
|---|---|---|---|
| main `6c75950` | 6.988, 7.531, 7.553, 7.538 | 7.549, 7.545, 7.550, 7.550 | 7.402 / 7.548 |
| this tree | 7.111, 7.534, 7.554, 7.546 | 7.550, 7.556, 7.559, 7.563 | 7.436 / 7.557 |

No regression: +0.5% cold, +0.1% warm. Both arms' first cold run is the slow one (6.99 / 7.11), the first load after the build. Per-step timing is unchanged (h2d 81 vs 83 ms, compute 38 vs 37 ms).

## #197: `--role-major-pagein`

| arm | tok/s cold | tok/s warm | mean cold / warm | p3_wait ms | disk_wait ms | compute ms |
|---|---|---|---|---|---|---|
| off | 7.542, 7.542, 7.548, 7.550 | 7.553, 7.556, 7.555, 7.560 | 7.545 / 7.556 | 15.4 | 12.2 | 37.1 |
| on | 7.394, 7.395, 7.387, 7.407 | 7.387, 7.423, 7.405, 7.397 | 7.396 / 7.403 | 20.1 | 8.6 | 35.1 |

−2.0% cold and warm. Greedy `--parity-dump` is byte-identical. Splits: 6 071–6 125 units per run over 21 033 ensure calls, ≈ 48 units per decode step of 42 × 8 = 336 routed. The split does take 3.6 ms/step off the disk wait, but the decode thread then waits 4.7 ms/step longer in P3 for the down planes, so the step ends ≈ 2.7 ms slower (132.5 → 135.2 ms). Stays opt-in, default off.

## #196: `--cpu-expert-math`

| arm | mean tok/s cold / warm | vs off | h2d ms | CPU experts/step | CPU ms | join wait ms | compute ms |
|---|---|---|---|---|---|---|---|
| off | 7.540 / 7.544 | — | 80.0 | — | — | — | 37.1 |
| frac 0.5 | **9.566 / 9.534** | **+26.9% / +26.4%** | 41.3 | 126 | 36.6 | 5.3 | 44.2 |
| frac 0.7 | 9.168 / 9.237 | +21.6% / +22.4% | 24.6 | 194 | 48.1 | 20.2 | 65.7 |
| frac 0.9 | 7.316 / 7.332 | −3.0% / −2.8% | 4.0 | 307 | 65.5 | 44.1 | 117.3 |
| frac 0.5, default (8) threads | 9.543 / 9.506 | +26.6% / +26.0% | 41.4 | 126 | 36.9 | 5.4 | 44.3 |
| frac 0.5, 4 threads | 9.272 / 9.185 | +23.0% / +21.8% | 41.5 | 127 | 46.8 | 13.1 | 48.4 |
| frac 0.5, 8 threads | 9.544 / 9.466 | +26.6% / +25.5% | 41.8 | 127 | 36.8 | 5.2 | 44.2 |

Best: frac 0.5 at the default 8 threads. Routing 126 of 336 experts per step to the CPU halves the H2D time (80 → 41 ms/step). The CPU work (≈ 37 ms) overlaps the GPU, so only ≈ 5 ms of join wait shows up in the step. Past frac 0.7 the CPU becomes the critical path: at 0.9 the join wait is 44 ms and the step is slower than off.

**Not bit-exact.** Greedy `--parity-dump` against off first picks a different token at position 44 (largest top-1 logit difference 58% relative). CPU expert math is a different numeric path. It needs a quality check before it is called a free speedup.

## #198: DeepSeek baseline (pending)

Not measured yet. In the first run, the stock `llama-cli` leg drew its UI on `/dev/tty`, so its log, including the t/s line, stayed empty. The run was stopped there. `m7-measure.sh` now passes `--simple-io` and closes stdin. Rerun with `M7_STAGES=198 scripts/m7-measure.sh`.
