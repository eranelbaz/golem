# Import + replica set: real-target read_bench A/B (#171)

Acceptance for `golem import` + `replicas` in `golem.toml` on the real target:
mirror GLM-5.3-Flash UD-IQ3_XXS onto the second drive, then A/B the store read
path (single drive vs both) cold and warm, in both orders.

## Machine and target

- Drives: replica 0 `/home` = nvme1n1p2 (btrfs on LUKS), replica 1 `/mnt/ssd1`
  = nvme0n1p1 (xfs, the 990 PRO). Distinct physical devices, so no §5.1
  same-device warning.
- Model: `glm-5.3-flash-ud-iq3-xxs`, quant `UD-IQ3_XXS`, arch `glm5-next`,
  4 shards, **120 367 571 840 B = 120.4 GB (112.1 GiB)** total; shard 1 is
  metadata-only (9.4 MB), shards 2–4 are 49.3 / 49.1 / 22.0 GB.
- Expert geometry: 42 layers × 3 planes = 126 planes; per-expert strides
  2 686 976 B (82), 3 604 480 B (41), 4 456 448 B (3). Bench reads use
  3 604 480 B — the middle-layer `ffn_down_exps` plane.
- RAM/VRAM context: 93 GiB RAM, 24 GiB VRAM; the pool is not loaded for this
  bench (store read path only).
- Space arithmetic: two full mirrors = 2 × 120.4 GB = 240.8 GB against
  2.9 TiB free (`/mnt/ssd1`) + 1.4 TiB free (`/home`).

## Import (verified mirror + manifests)

The source dir already carried hand-edited manifests pointing at a replica path
that no longer existed (`manifest.two.json` → `/mnt/data/...`), the exact bug in
the issue.

```sh
golem import /home/eranelbaz/models/glm-5.3-flash-ud-iq3-xxs/UD-IQ3_XXS \
  /mnt/ssd1/golem/glm-5.3-flash-ud-iq3-xxs/UD-IQ3_XXS \
  --name glm-5.3-flash-ud-iq3-xxs --quant UD-IQ3_XXS
```

| Run | Result | Wall | Notes |
|---|---|---|---|
| re-import, replica complete | `4 replica shard(s) kept` | **64.9 s** | source 120.4 GB re-hashed (sha256+CRC32C in one pass, 1.86 GB/s); each replica reads 4 × 16 MiB = 64 MiB sampled windows |
| re-import, shard 3 deleted | `3 replica shard(s) kept` | **98.5 s** | +33.6 s to clone 49.26 GB: write 49.26 GB + full re-hash verify 49.26 GB = 98.5 GB device traffic ≈ 2.93 GB/s |

Both `manifest.json`s (source dir and replica) load and list the same two
replica roots; a re-run is byte-identical and skips every kept shard.

## read_bench A/B

3 large shards only (shard 1 is metadata and would read from page cache):
`--n 2000 --plane-bytes 3604480 --seed 7` = 7.21 GB logical per leg.
Cold = `sync` + `drop_caches` before each run; warm = same command run twice,
second measured (the store DONTNEEDs behind the cursor, so warm ≈ cold by
design; "warm" here is steady state, not a page-cache number).

| Order | Leg | Wall GB/s | Device GB/s |
|---|---|---|---|
| 1 | cold, both drives | 6.29 | 6.51 |
| 1 | cold, nvme1n1 alone | 7.53 | 7.85 |
| 1 | warm, both drives | 6.37 | 6.61 |
| 1 | warm, nvme1n1 alone | 7.63 | 7.98 |
| 2 | cold, both drives | 6.34 | 6.56 |
| 2 | cold, nvme1n1 alone | 7.40 | 7.75 |
| 2 | warm, nvme1n1 alone | 7.24 | 7.55 |
| 2 | warm, both drives | 5.44 | 5.61 |

Standalone per drive (`--whole`, same offsets): nvme1n1 7.85 cold / 7.94 warm,
nvme0n1 5.97 cold / 6.03 warm.

## Answer

- **Striping does not aggregate on this unequal pair; it is a net loss.** Both
  drives land at 5.4–6.6 GB/s versus 7.2–8.0 GB/s for the faster drive alone
  (17–25% slower), because chunk homing is static round-robin (chunk index %
  drives) and each drive's worker only takes its own home chunks. Aggregate is
  the harmonic mean: 2 / (1/7.85 + 1/5.97) = 6.78 GB/s expected, 6.31–6.56 GB/s
  measured with queue overhead. §5.2's "the faster drive takes more chunks"
  holds only for equal-speed drives under this homing.
- **The replica set pays for redundancy and failover, not bandwidth** on this
  pair: a lost 990 PRO still leaves the model readable at 7.9 GB/s off
  `/home`. Bandwidth striping would need speed-weighted homing (or demand-first
  pulling instead of home-chunk preference) before it can beat one drive.
- Import itself is dominated by hashing the source once (1.86 GB/s
  sha256+CRC32C); a fresh 49 GB replica shard costs 33.6 s on top, and a
  re-import with no change writes nothing.
