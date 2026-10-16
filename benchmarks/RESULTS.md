# Community benchmark results

Add one row per machine. Run [`bench.sh`](bench.sh) (cold + warm legs, decode
tok/s from `--timing-csv`), fill in your hardware columns, and append below.
Prefix (cold/warm) means page cache evicted / immediate re-run.

Format: `GPU | VRAM | RAM | SSD | Quant | Prefill TTFT cold/warm | Decode tok/s cold/warm`

| GPU | VRAM | RAM | SSD | Quant | Prefill TTFT (cold/warm) | Decode tok/s (cold/warm) |
|---|---|---|---|---|---|---|
| AMD RX 7900 XTX (reference) | 24 GiB | 93 GiB DDR5 | Samsung 990 PRO 4TB NVMe | UD-Q4_K_XL | 11.1 s / 13.4 s | 2.42 / 2.12 |

Notes:
- Reference row measured 2026-10-10 (`docs/baselines/m4.9-optimize.md`),
  ctx 262144, greedy, release HIP build, desktop GPU sharing.
- `bench.sh` reports decode tok/s only; TTFT comes from `golem serve`
  request logs or `golem optimize` output.
