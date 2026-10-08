#!/usr/bin/env python3
"""Summarise ab-bench.sh legs: decode tok/s, disk GB/token and the run's
prefetch stats line, one row per leg, in the given order.
usage: ab-report.py <work-dir> <leg>..."""
import csv
import re
import sys
from pathlib import Path


def leg_row(d: Path) -> str:
    if not (d / "meta.txt").is_file() or "rc=0" not in (d / "meta.txt").read_text().split():
        sys.exit(f"{d.name}: run did not finish with rc=0")
    lines = (d / "timing.csv").read_text().splitlines()
    rows = list(csv.DictReader(l for l in lines if not l.startswith("#")))
    if not rows:
        sys.exit(f"{d.name}: no decode rows")
    try:
        wall = sum(float(r["wall_ms"]) for r in rows)
        # The plain-GGUF leg (no manifest) has no store counters.
        disk = None if rows[0]["disk_bytes"] == "unmeasured" else sum(float(r["disk_bytes"]) for r in rows)
    except (KeyError, ValueError) as e:
        sys.exit(f"{d.name}: unusable timing.csv ({e})")
    trailer = next((l[2:] for l in lines if l.startswith("# prefetch")), "")
    stats = dict(re.findall(r"(\w+)=(\S+)", trailer))
    return "| {} | {} | {:.3f} | {} | {} | {} | {} |".format(
        d.name,
        len(rows),
        len(rows) / wall * 1000,
        "-" if disk is None else f"{disk / len(rows) / 1e9:.2f}",
        stats.get("prediction_recall", "-"),
        stats.get("prefetch_hit_rate", "-"),
        f"{int(stats['prefetch_wasted_bytes']) / len(rows) / 1e9:.2f}" if "prefetch_wasted_bytes" in stats else "-",
    )


def main() -> None:
    work = Path(sys.argv[1])
    print("| leg | steps | tok/s | disk GB/token | recall | prefetch hit | wasted GB/token |")
    print("|---|---|---|---|---|---|---|")
    for leg in sys.argv[2:]:
        print(leg_row(work / leg))


if __name__ == "__main__":
    main()
