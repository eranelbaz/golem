#!/usr/bin/env python3
"""Summarise perf-205.sh legs: per leg × temp, median [min–max] over reps of
decode tok/s, disk GB/token, disk wait ms/token, non-decode wall, and the
store scratch_planes trailer. Prefill legs (`-pf`) also report prefill s =
their non-decode wall minus the same binary/temp short leg's.
usage: perf-205-report.py <work-dir>"""
import csv
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def measured(rows, col):
    vals = [float(r[col]) for r in rows if r.get(col) not in (None, "", "unmeasured")]
    return sum(vals) if vals else None


def leg(d: Path):
    meta = (d / "meta.txt").read_text() if (d / "meta.txt").is_file() else ""
    if "rc=0" not in meta.split() or not (d / "timing.csv").is_file():
        return None
    lines = (d / "timing.csv").read_text().splitlines()
    rows = list(csv.DictReader(l for l in lines if not l.startswith("#")))
    if not rows:
        return None
    tokens = sum(int(r["tokens"]) for r in rows)
    wall = sum(float(r["wall_ms"]) for r in rows)
    disk, dwait = measured(rows, "disk_bytes"), measured(rows, "disk_wait_ms")
    total = re.search(r"wall_ms=(\d+)", meta)
    scratch = next((re.search(r"scratch_planes=(\d+)", l) for l in lines if "scratch_planes=" in l), None)
    return {
        "tok_s": tokens / wall * 1000,
        "gb_tok": None if disk is None else disk / tokens / 1e9,
        "wait_ms_tok": None if dwait is None else dwait / tokens,
        "nondecode_s": None if total is None else (float(total.group(1)) - wall) / 1000,
        "scratch": int(scratch.group(1)) if scratch else None,
    }


def stat(rs, k, fmt):
    vals = [r[k] for r in rs if r[k] is not None]
    if not vals:
        return "unmeasured"
    return f"{statistics.median(vals):{fmt}} [{min(vals):{fmt}}–{max(vals):{fmt}}]"


def main() -> None:
    work = Path(sys.argv[1])
    groups = defaultdict(list)
    failed = []
    for d in sorted(p for p in work.iterdir() if p.is_dir() and re.search(r"-r\d+$", p.name)):
        r = leg(d)
        (groups[re.sub(r"-r\d+$", "", d.name)].append(r) if r else failed.append(d.name))

    def nondecode_med(name):
        vals = [r["nondecode_s"] for r in groups.get(name, []) if r["nondecode_s"] is not None]
        return statistics.median(vals) if vals else None

    print("| leg | n | decode tok/s | disk GB/tok | disk wait ms/tok | non-decode s | prefill s | scratch_planes |")
    print("|---|---|---|---|---|---|---|---|")
    for name, rs in sorted(groups.items()):
        prefill = "-"
        m = re.match(r"(base|new)-pf(?:-([^-]+))?-(cold|warm)$", name)
        if m:
            short = f"{m.group(1)}-{m.group(2) or 'd2'}-{m.group(3)}"
            a, b = nondecode_med(name), nondecode_med(short)
            prefill = f"{a - b:.1f} (vs {short})" if a is not None and b is not None else "n/a"
        scr = [r["scratch"] for r in rs if r["scratch"] is not None]
        print(f"| {name} | {len(rs)} | {stat(rs, 'tok_s', '.3f')} | {stat(rs, 'gb_tok', '.2f')} | "
              f"{stat(rs, 'wait_ms_tok', '.1f')} | {stat(rs, 'nondecode_s', '.1f')} | {prefill} | "
              f"{max(scr) if scr else 'n/a'} |")
    if failed:
        print(f"\nfailed legs: {' '.join(failed)}")


if __name__ == "__main__":
    main()
