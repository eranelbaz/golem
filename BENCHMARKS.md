# Benchmarks

Community benchmark table for GoLeM. Add your machine, send a PR.

## How to add a benchmark

1. Build the release binary: `scripts/build.sh build` (ROCm/HIP required).
2. Check `ulimit -l` is `unlimited` (the expert pool is mlocked).
3. Run the two legs with [`benchmarks/bench.sh`](benchmarks/bench.sh):

   ```sh
   GOLEM_MODEL=<shard-1.gguf> GOLEM_QUANT=<quant> GOLEM_SSD="<ssd model>" benchmarks/bench.sh
   ```

   It runs a **cold** leg (page cache dropped via `drop_caches`; needs a
   passwordless sudoers entry) and a **warm** leg (immediate re-run), and
   prints a ready-to-paste row.
4. Append the row to the table below and note anything unusual about machine
   state (desktop idleness, GPU sharing).

**Method** (what makes numbers comparable):

- Greedy decoding, release HIP build, one prompt, `--n-predict 128`, fixed
  ctx — report the ctx you used.
- Cold = page cache evicted before the run; warm = immediate re-run.
- Decode tok/s = tokens ÷ Σ wall_ms from the `--timing-csv` rows of the leg
  (`benchmarks/bench.sh` does this for you).
- Prefill TTFT from `golem serve` request logs or `golem optimize` output.
- Any A/B claim runs in both orders (A→B and B→A) so run position doesn't
  masquerade as a win.
- Cite the commit you measured (`bench.sh` records it in the leg's
  `meta.txt`).

## Results

`Prefill` is TTFT cold/warm; `token/s` is decode cold/warm.

| GPU | RAM | Storage | Quant | Context | Prefill (cold/warm) | token/s (cold/warm) |
|---|---|---|---|---|---|---|
| AMD RX 7900 XTX (reference) | 93 GiB DDR5 | Samsung 990 PRO 4TB NVMe | UD-Q4_K_XL | 262144 | 11.1 s / 13.4 s | 2.42 / 2.12 |
