# #240: one shard lock per MoE layer for resident claims

Change: `ExpertPool::claim_misses` (the P3 layer-batched `ensure_batch` path) first claims every resident unit of the layer's miss list under one shard lock (`ResidencyMap::claim_resident_batch`: lease++, prefetch mark cleared, CLOCK reference bit set, same as `claim_for`'s `Resident` arm). Only non-resident, mid-fill or lease-saturated units fall through to the per-unit `claim_for` (fill, evict, park, `Busy`), unchanged.

Arithmetic (GLM-5.3-Flash, 42 MoE layers × K=8 routed): before, 42 × 8 = 336 shard locks per token on an all-hit step; after, 42 × 1 = 42, plus one `claim_for` per actual miss. The run below hit 83.9% recall, i.e. ~1.3 misses per layer.

Machine: 32 threads, 93 GiB RAM, 24 GiB VRAM (gfx1100). Model: GLM-5.3-Flash UD-IQ3_XXS, 4 split shards, knobs from `golem.toml` `[models."glm-5.3-flash"]` (52 GiB pool, `--device-cache-experts 8 --vram-reserve-gib 3.0 --prefill-ubatch 512 --flash-attn`). Base = HEAD db1669c plus the worktree's other uncommitted edits, minus `pool/pager.rs` and `pool/residency.rs`; new = full worktree. `scripts/ab-bench.sh`, 128 decode tokens, ctx 1024, 4 reps ABBA (even reps reverse both leg order and cold/warm order). Cold = all four shards evicted with `POSIX_FADV_DONTNEED` before the leg (`sudo` for `drop_caches` was not available). Median [min–max] of `scripts/ab-report.py` tok/s.

| | base | new |
|---|---|---|
| decode tok/s cold | 6.351 [6.340–6.364] | 6.348 [6.321–6.359] |
| decode tok/s warm | 6.333 [6.315–6.345] | 6.333 [6.316–6.353] |
| disk GB/token | 0.33 | 0.33 |

No measurable tok/s change: the ranges overlap and the medians are within 0.05%. The removed ~294 uncontended lock/unlock pairs per token are far below the ~157 ms token time.
