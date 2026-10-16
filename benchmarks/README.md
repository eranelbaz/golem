# GoLeM community benchmarks

Run [`bench.sh`](bench.sh) against your machine and append a row to
[`RESULTS.md`](RESULTS.md). One number per column, cite your cold and warm
legs — see the README's Benchmarks table for the target-machine reference.

Columns: `GPU | VRAM | RAM | SSD | Quant | Prefill TTFT (cold/warm) | Decode tok/s (cold/warm)`

Cold = the model's page cache dropped before the run; warm = the immediate
re-run. `bench.sh` produces both legs via `scripts/ab-bench.sh` (`--ctx 1024`,
128 greedy tokens, `--ram-experts-gib 59`), so set the same env defaults when
comparing rows. Prerequisites (from `ab-bench.sh`):

- release build (`scripts/build.sh build`)
- `ulimit -l unlimited` (the pool is mlocked)
- the cold leg needs a passwordless sudoers entry for `drop_caches`

Report anything unusual about machine state (desktop idleness, GPU sharing).

Note: the reference row in `RESULTS.md` was measured with `golem optimize` at
ctx 262144 and a 62 GiB pool, so treat it as indicative rather than directly
comparable to `bench.sh` rows.
