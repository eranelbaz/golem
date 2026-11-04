# #211: persistent drain readers, per-session ring stats, atomic enable

Machine: 32 threads, 93 GiB RAM, 24 GiB VRAM (gfx1100), two replicas (9100 PRO + 990 PRO). Model: GLM-5.3-Flash UD-IQ3_XXS, 4 split shards, model and knobs from `~/.config/golem/golem.toml` (`scripts/golem-toml-model.py`: 52 GiB pool, `--device-cache-experts 8 --vram-reserve-gib 3.0 --vram-experts-frac 0.8 --prefill-ubatch 512 --flash-attn`). Baseline = `origin/main` 35b9363, new = 35b9363 + this branch's worktree. `scripts/perf-211.sh` runs the HIP gate, the pop bench and `PERF_LEGS='^(base|new)-(d2|pf)$' scripts/perf-205.sh`: 4 reps ABBA, even reps reverse both leg order and cold/warm order. Cold = `drop_caches` before the leg. Decode legs: 128 tokens, ctx 1024. Prefill legs: ~700-token prompt, ctx 4096, 16 decode tokens; prefill s = their non-decode wall minus the same binary/temp short leg's. Median [min–max]; full table via `scripts/perf-205-report.py .ab-work/perf-211/ab`.

## Baseline vs new (default io_depth 2)

| | base | new |
|---|---|---|
| decode tok/s cold | 6.443 [6.409–6.454] | 6.482 [6.474–6.500] |
| decode tok/s warm | 6.455 [6.443–6.470] | 6.502 [6.496–6.508] |
| prefill-leg decode tok/s cold | 3.204 [3.183–3.217] | 3.225 [3.208–3.256] |
| prefill-leg decode tok/s warm | 3.188 [3.173–3.217] | 3.214 [3.206–3.246] |
| prefill s cold | 36.8 | 35.5 |
| prefill s warm | 36.5 | 35.1 |
| disk wait ms/token (decode) cold / warm | 30.4 / 30.1 | 29.6 / 29.5 |
| disk GB/token (decode) | 0.58–0.59 | 0.59 |

Decode +0.6% cold, +0.7% warm, min–max ranges disjoint in both temperatures; disk bytes per token unchanged, disk wait −0.8 / −0.6 ms/token. One decode token runs 42 drains, each previously spawning `drives × io_depth` = 4 threads (168 spawns/token); spawn cost itself was not measured, and #205 found depth 2 → 4 (double the spawns) unmeasurable, so the attribution is unproven. Prefill −1.3 s cold, −1.4 s warm (−3.5%/−3.8%) is too large for spawn removal (~84 drains for a 2-ubatch prompt would need ~15 ms each); not explained.

## Pop cost with one replica degraded (warm)

`GOLEM_POP_BENCH=1`, 864 chunks per batch, two sessions of 3 runs: 2080, 2149, 2345 and 2413, 2250, 2373 ns/pop (whole-batch time / pops, an upper bound). Threshold for per-drive queues is 75 µs; measured is ≥31× under, so the single shared queue stays. Cold not measured: the bench reads from a null source, so page-cache state does not enter it.

## Zero allocation

`warm_batch_drain_allocates_nothing` (store, all threads) and `store_backed_decode_allocates_nothing_on_any_thread` (engine, 200 store-backed steps, every step a demand page-in through the drain, all threads) both assert 0 allocations. The latter caught 2 allocations per step in `publish_scored_snapshot` (exclusive residency only; fixed by snapshot recycling: 2 heap ops per decode token before).

## HIP gate

`cargo nextest run -p golem-engine --features hip-tests`: 202/202 passed. Real-model legs (`--run-ignored only -E 'test(/^hw_/)' --test-threads 1`; serial because each pins a 52 GiB pool, two exceed 93 GiB): 5/5 passed — `hw_cancel_inside_streamed_prefill`, `hw_glm_jinja_prompt_reasoning`, `hw_soak_no_gpu_reset` (300 s), `hw_snapshot_turn_ttft`, `hw_start_verify_sampled_vs_full_cold`. Zero `SKIPPED` lines in either log.

Figures below are from a `--no-capture` rerun of the two tests (`hip-hw-numbers.log`; both passed).

`hw_start_verify_sampled_vs_full_cold` (all shards `fincore RES=0`, no mapping pid): sampled 0.86 s, full 1449.66 s, 4 shards, 0 warnings.

`hw_snapshot_turn_ttft` (order `off,on`; TTFT ms, turns 1–5, prompt 17/49/82/117/153 tokens):

| mode / pass | t1 | t2 | t3 | t4 | t5 |
|---|---|---|---|---|---|
| off cold | 10 827 | 5 797 | 6 913 | 11 499 | 16 875 |
| on cold | 10 866 | 4 684 | 3 699 | 4 603 | 4 015 |
| off warm | 2 783 | 4 054 | 6 655 | 10 593 | 16 742 |
| on warm | 3 467 | 2 935 | 2 858 | 3 888 | 3 507 |

`outputs_equal_cold=true`. Only the `off,on` order was run, so this table is functional evidence, not an A/B performance claim.
