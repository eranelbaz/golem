# GoLeM community benchmarks

Run [`bench.sh`](bench.sh) against your machine and append a row to
[`RESULTS.md`](RESULTS.md). One number per column, cite your cold and warm
legs — see the README's Benchmarks table for the target-machine reference.

Columns: `GPU | VRAM | RAM | SSD | Quant | Prefill TTFT (cold/warm) | Decode tok/s (cold/warm)`

Cold = the model shards were evicted from the OS page cache before the run
(`scripts/evict-model-cache.py` or a reboot); warm = the immediate re-run.
Both numbers use a fixed prompt set, greedy decoding, and grayscale honesty:
report the machine state (desktop idleness, GPU sharing) if it was unusual.
