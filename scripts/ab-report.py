#!/usr/bin/env python3
"""Summarise ab-bench.sh legs, one row per leg, in the given order: decode
tok/s (sum of tokens over sum of wall time), disk GB/token, the run's
prefetch stats line, per-step means of the timing columns, and the prefill
trailer.

Token 1 is sampled from the prefill logits and has no row; the decode step
with `step = s` produces token s + 2. `--window FIRST-LAST` (tokens, 1-based,
FIRST >= 2) therefore keeps rows with FIRST-2 <= step <= LAST-2.

usage: ab-report.py [--window FIRST-LAST] <work-dir> <leg>..."""
import argparse
import csv
import re
import sys
from pathlib import Path

# Per-step means over measured rows; `compute_ms` is the recorder's clamped
# remainder (wall - p3_wait - h2d - policy, floored at 0).
MEAN_COLUMNS = [
    "p3_wait_ms",
    "disk_wait_ms",
    "h2d_ms",
    "compute_ms",
    "cpu_expert_math_experts",
    "cpu_expert_math_bytes",
    "cpu_expert_math_ms",
    "cpu_expert_join_wait_ms",
]


def parse_window(s: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)-(\d+)", s)
    if not m:
        raise argparse.ArgumentTypeError(f"window {s!r} is not FIRST-LAST")
    first, last = int(m[1]), int(m[2])
    if first < 2:
        raise argparse.ArgumentTypeError("window FIRST must be >= 2 (token 1 has no row)")
    if last < first:
        raise argparse.ArgumentTypeError(f"window {s!r} has LAST < FIRST")
    return first, last


def mean_cell(rows: list[dict], col: str) -> str:
    """Mean over measured rows; `value (n=k)` when some rows are unmeasured,
    `-` when none is measured."""
    vals = [float(r[col]) for r in rows if r.get(col, "unmeasured") != "unmeasured"]
    if not vals:
        return "-"
    m = f"{sum(vals) / len(vals):.3f}"
    return m if len(vals) == len(rows) else f"{m} (n={len(vals)})"


def leg_row(d: Path, window: tuple[int, int] | None) -> str:
    if not (d / "meta.txt").is_file() or "rc=0" not in (d / "meta.txt").read_text().split():
        sys.exit(f"{d.name}: run did not finish with rc=0")
    lines = (d / "timing.csv").read_text().splitlines()
    rows = list(csv.DictReader(l for l in lines if not l.startswith("#")))
    if not rows:
        sys.exit(f"{d.name}: no decode rows")
    try:
        if window is not None:
            first, last = window
            max_step = max(int(r["step"]) for r in rows)
            if last - 2 > max_step:
                sys.exit(
                    f"{d.name}: window {first}-{last} needs step {last - 2}, "
                    f"last row is step {max_step} (token {max_step + 2})"
                )
            rows = [r for r in rows if first - 2 <= int(r["step"]) <= last - 2]
            if not rows:
                sys.exit(f"{d.name}: no rows in window {first}-{last}")
        wall = sum(float(r["wall_ms"]) for r in rows)
        tokens = sum(int(r["tokens"]) for r in rows)
        # The plain-GGUF leg (no manifest) has no store counters.
        disk = None if any(r["disk_bytes"] == "unmeasured" for r in rows) else sum(float(r["disk_bytes"]) for r in rows)
        means = [mean_cell(rows, c) for c in MEAN_COLUMNS]
    except (KeyError, ValueError) as e:
        sys.exit(f"{d.name}: unusable timing.csv ({e})")
    trailer = next((l[2:] for l in lines if l.startswith("# prefetch")), "")
    stats = dict(re.findall(r"(\w+)=(\S+)", trailer))
    pf = next((l for l in lines if l.startswith("# prefill_ms=")), None)
    pf_ms = pf_tok = None
    if pf:
        m = re.fullmatch(r"# prefill_ms=(\S+) prompt_tokens=(\d+)", pf)
        if not m:
            sys.exit(f"{d.name}: unusable prefill trailer {pf!r}")
        pf_ms, pf_tok = float(m[1]), int(m[2])
    cells = [
        d.name,
        str(len(rows)),
        f"{tokens / wall * 1000:.3f}",
        "-" if disk is None else f"{disk / tokens / 1e9:.2f}",
        stats.get("prediction_recall", "-"),
        stats.get("prefetch_hit_rate", "-"),
        f"{int(stats['prefetch_wasted_bytes']) / tokens / 1e9:.2f}" if "prefetch_wasted_bytes" in stats else "-",
        *means,
        "-" if pf_ms is None else f"{pf_ms:.1f}",
        "-" if pf_tok is None else str(pf_tok),
        f"{pf_tok / pf_ms * 1000:.1f}" if pf_ms and pf_tok is not None else "-",
    ]
    return "| " + " | ".join(cells) + " |"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--window", type=parse_window, metavar="FIRST-LAST",
                    help="report only tokens FIRST..LAST (1-based, FIRST >= 2)")
    ap.add_argument("work", type=Path)
    ap.add_argument("legs", nargs="+")
    a = ap.parse_args(argv)
    head = ["leg", "steps", "tok/s", "disk GB/token", "recall", "prefetch hit", "wasted GB/token",
            *MEAN_COLUMNS, "prefill ms", "prompt tokens", "prefill tok/s"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for leg in a.legs:
        print(leg_row(a.work / leg, a.window))


if __name__ == "__main__":
    main()
