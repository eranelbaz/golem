# #241: lock-free `entry_for`

Machine: 24 GiB VRAM (gfx1100), 93 GiB RAM. Model: GLM-5.3-Flash UD-IQ3_XXS, 4 split shards, `--manifest`, 52 GiB pool, `--device-cache-experts 8 --vram-reserve-gib 3.0 --vram-experts-frac 0.8 --prefill-ubatch 512 --flash-attn`, golem.toml top-level keys (q8_0 KV, `prefetch = protect`). Base = HEAD db1669c, new = db1669c + only the `p3_glue.rs` change (both built in one clean worktree). `scripts/ab-bench.sh`, 128 tokens, ctx 1024, greedy; 2 reps, ABBA then BAAB, cold and warm per arm slot. Cold = `POSIX_FADV_DONTNEED` on every shard right before the leg (`drop_caches` needs a sudo password on this host); expert reads are O_DIRECT either way. Decode tok/s = Σtokens / Σwall over the 127 decode rows.

Arithmetic: `entry_for` runs at ≤ 2 call sites (ensure, evict-dest) × 42 layers × 3 kinds = 252 calls/token; at ~100 ns per old miss path (2 lock/unlock pairs + SipHash lookup) that is ~25 µs against a 158 ms token (6.34 tok/s), 0.016%, below what decode tok/s can resolve. The table is 42 × 3 × 8 B = 1 008 B.

| arm | tok/s cold (4 legs) | tok/s warm (4 legs) | mean cold / warm | p3_wait ms/step |
|---|---|---|---|---|
| base | 6.323 6.327 6.335 6.356 | 6.353 6.335 6.352 6.326 | 6.335 / 6.341 | 42.4 / 42.4 |
| new | 6.336 6.335 6.356 6.356 | 6.331 6.337 6.337 6.322 | 6.346 / 6.332 | 42.3 / 42.5 |

No measurable change: +0.2% cold, −0.1% warm, ranges overlap in both temperatures. All 16 legs produced byte-identical greedy output.
