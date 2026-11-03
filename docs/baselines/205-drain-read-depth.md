# #205: store drain bookkeeping, bounded batch scratch, read depth per drive

Machine: 32 threads, 93 GiB RAM, 24 GiB VRAM (gfx1100), two replicas (9100 PRO + 990 PRO). Model: GLM-5.3-Flash UD-IQ3_XXS, 4 split shards, model and knobs from `~/.config/golem/golem.toml` (`scripts/golem-toml-model.py`: 52 GiB pool, `--device-cache-experts 8 --vram-reserve-gib 3.0 --vram-experts-frac 0.8 --prefill-ubatch 512 --flash-attn`). Baseline = `origin/main` f69a4ad, new = this branch.

`scripts/perf-205.sh`, 4 reps, ABBA: even reps reverse both leg order and cold/warm order. Cold = `drop_caches` before the leg. Decode legs: 128 tokens, ctx 1024. Prefill legs (`pf`): ~700-token prompt, ctx 4096, 16 decode tokens; prefill s = their non-decode wall minus the same binary/temp short leg's. Median [min–max]; full table via `scripts/perf-205-report.py .ab-work/perf-205`.

## Baseline vs new (default io_depth 2)

| | base | new |
|---|---|---|
| decode tok/s cold | 6.397 [6.051–6.423] | 6.409 [6.381–6.422] |
| decode tok/s warm | 6.422 [6.393–6.448] | 6.425 [6.392–6.439] |
| prefill s cold | 45.2 | 45.8 |
| prefill s warm | 45.7 | 44.8 |
| disk GB/token | 0.58 | 0.58 |

No change outside the spread. The one-unit scratch cap costs nothing measurable although real planes use it: `scratch_planes` = 126 per decode run, 58 273 per prefill run. Scratch per thread is now ≤ 3 × the largest page-rounded plane covering instead of the whole layer's scratch planes.

## Read depth (new, decode)

| config | reader threads | tok/s cold | tok/s warm | disk wait ms/token |
|---|---|---|---|---|
| `--io-threads 1` | 1 | 5.408 | 5.416 | 49.7 |
| `--io-depth 1` | 2 | 6.260 | 6.334 | 33.4 |
| `--io-threads 2` | 2 | 6.245 | 6.281 | 33.6 |
| default (`--io-depth 2`) | 4 | 6.409 | 6.425 | 30.6 |
| `--io-threads 4` | 4 | 6.426 | 6.442 | 30.4 |
| `--io-depth 4` | 8 | 6.400 | 6.424 | 30.6 |

Prefill is flat across depth (44.5–45.8 s). Depth 2 per drive is the knee (+2.4% decode over depth 1); depth 4 gains nothing and costs nothing, so the per-drain thread spawns are not measurable at this scale. Multiple reads in flight per thread (two `pread`s / io_uring) is not pursued: its ceiling is the depth-2 → depth-4 delta, which is zero.
