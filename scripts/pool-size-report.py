#!/usr/bin/env python3
"""Pool-size report: parse timing.csv legs and print a markdown rollup .

Usage: python3 pool-size-report.py <workdir>
Each leg is a dir containing timing.csv; legs named asc|desc|smoke|probe-<gib>-<temp>.
Constants: 336 expert-layer loads/token, 15.2 MB per expert unit, 27 GB/s pinned
H2D bus, 4.73 GB/s disk (measured, scripts/disk-bandwidth.py).
Total expert bytes: 12,096 units x 15.34 MB = 172.8 GiB (185.5 GB); tables
below use the plan-verbatim rounded constant EXPERT_BYTES = 15.2 MB (<1% low).
"""
import argparse, csv, glob, os, re, statistics, sys

EXPERT_LAYER_LOADS = 336          # 336 expert-layer loads/token (model constant)
EXPERT_BYTES = 15.2e6             # bytes per expert unit
H2D_GBPS = 27.0                   # GB/s pinned H2D
DISK_GBPS = 4.73                  # GB/s disk

COLS = ["step", "tokens", "wall_ms", "p3_wait_ms", "disk_worker_ms", "disk_bytes",
        "h2d_ms", "h2d_bytes", "policy_ms", "compute_ms"]
UNMEASURED = {"h2d_ms", "h2d_bytes", "policy_ms"}


def parse_num(col, raw):
    if col in UNMEASURED and raw == "unmeasured":
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"column {col!r}: bad value {raw!r}")


def load_leg(path):
    rows = []
    with open(path) as f:
        first = f.readline()
        if not first.startswith("# "):
            raise ValueError("missing '# engine_version=...' header comment")
        rdr = csv.reader(f)
        header = next(rdr, None)
        if header != COLS:
            raise ValueError(f"bad header: {header}")
        for i, row in enumerate(rdr):
            if len(row) != len(COLS):
                raise ValueError(f"data row {i+1}: expected {len(COLS)} columns, got {len(row)}")
            rows.append({c: parse_num(c, v) for c, v in zip(COLS, row)})
    if not rows:
        raise ValueError("no data rows")
    return rows


def leg_stats(rows):
    gen = sum(r["tokens"] for r in rows)
    wall = sum(r["wall_ms"] for r in rows)
    if gen <= 0 or wall <= 0:
        raise ValueError("nonpositive tokens or wall_ms")
    disk = sum(r["disk_bytes"] for r in rows)
    d = disk / (EXPERT_LAYER_LOADS * EXPERT_BYTES * gen)
    return {
        "gen": gen,
        "tok_s": gen / (wall / 1000.0),
        "ms_tok": wall / gen,
        "disk_gb_tok": disk / gen / 1e9,
        "disk_share": d,
        "disk_ms_share": sum(r["disk_worker_ms"] for r in rows) / wall,
        "compute_share": sum(r["compute_ms"] for r in rows) / wall,
    }


def predict(s):
    """§2.4 arithmetic against the real model: 12,096 expert units x ~15.34 MB = 172.8 GiB
    (tables compute with the plan's rounded 15.2 MB constant, <1% low)
    total; 336 expert-layer loads/token. Pool R=GB/token resident, disk D=GB/token."""
    d = s["disk_share"]
    total = EXPERT_LAYER_LOADS * EXPERT_BYTES / 1e9  # GB/token demanded
    D = d * total
    R = (1 - d) * total
    # t = max(R/27 pinned H2D, D/4.73 disk) + D/27 bus transfer of disk-sourced bytes
    t_pred_ms = (max(R / H2D_GBPS, D / DISK_GBPS) + D / H2D_GBPS) * 1000.0
    return t_pred_ms, D, R


def leg_gib_temp(name):
    m = re.match(r"(asc|desc)-(\d+)-(cold|warm)$", name)   # smoke/probe legs excluded from pool means
    if not m:
        return None
    return int(m.group(2)), m.group(3), m.group(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("workdir", help="dir containing <leg>/timing.csv")
    args = ap.parse_args()
    legs = {}
    for path in sorted(glob.glob(os.path.join(args.workdir, "*", "timing.csv"))):
        name = os.path.basename(os.path.dirname(path))
        try:
            legs[name] = leg_stats(load_leg(path))
        except (OSError, ValueError) as e:
            print(f"ERROR: leg {name}: {e}", file=sys.stderr)
            sys.exit(1)
    if not legs:
        print(f"ERROR: no legs with timing.csv under {args.workdir}", file=sys.stderr)
        sys.exit(1)
    print(f"legs found: {', '.join(legs)}\n")

    print("## Per-leg\n")
    print("| leg | gen_tokens | tok/s | ms/token | disk GB/token | disk share % | disk ms share % | compute+H2D ms share % |")
    print("|---|---|---|---|---|---|---|---|")
    for name, s in legs.items():
        print(f"| {name} | {int(s['gen'])} | {s['tok_s']:.3f} | {s['ms_tok']:.1f} | "
              f"{s['disk_gb_tok']:.3f} | {100*s['disk_share']:.1f} | "
              f"{100*s['disk_ms_share']:.1f} | {100*s['compute_share']:.1f} |")
    print("\ncompute column = compute + H2D (H2D unmeasured).\n"
          "Footnote: disk ms share can exceed 100% — disk_worker_ms summed across "
          "parallel per-drive workers overlaps wall time.")

    # per-pool-size means across both orders
    print("\n## Per-pool means\n")
    groups = {}
    unmatched = []
    for name, s in legs.items():
        gt = leg_gib_temp(name)
        if gt:
            groups.setdefault(gt[0], []).append((gt, s))
        else:
            unmatched.append(name)
    print("| GiB | cold tok/s | warm tok/s | mean disk share % |")
    print("|---|---|---|---|")
    if unmatched:
        print(f"(# unmatched: {', '.join(sorted(unmatched))})\n")
    for gib in sorted(groups):
        items = groups[gib]
        colds = [s["tok_s"] for (_, t, _), s in items if t == "cold"]
        warms = [s["tok_s"] for (_, t, _), s in items if t == "warm"]
        shares = [s["disk_share"] for _, s in items]
        cs = f"{statistics.mean(colds):.3f}" if colds else "—"
        ws = f"{statistics.mean(warms):.3f}" if warms else "—"
        print(f"| {gib} | {cs} | {ws} | {100*statistics.mean(shares):.1f} |")
    # flag cold/warm inversions: warm slower than the cold leg of same order+size
    by_key = {}
    for name, s in legs.items():
        gt = leg_gib_temp(name)
        if gt:
            by_key[(gt[2], gt[0], gt[1])] = (name, s)
    flags = []
    for (order, gib, temp), (name, s) in sorted(by_key.items()):
        if temp == "warm":
            ck = by_key.get((order, gib, "cold"))
            if ck and s["tok_s"] < ck[1]["tok_s"]:
                flags.append(f"FLAG: {name} warm ({s['tok_s']:.3f} tok/s) slower than cold {ck[0]} "
                             f"({ck[1]['tok_s']:.3f} tok/s)")
    if flags:
        print("\n" + "\n".join(flags))

    print("\n## §2.4 predicted vs measured\n")
    print("d = measured disk share; total expert bytes = 12,096 units x ~15.34 MB = 172.8 GiB; "
          "GB/token = d x 336 x 15.2 MB (rounded constant); t_pred = max(R/27, D/4.73) + D/27 GB/s, x1000 ms.\n")
    print("| leg | disk share d | D GB/tok | R GB/tok | pred ms/tok | meas ms/tok | delta % | arithmetic |")
    print("|---|---|---|---|---|---|---|---|")
    for name, s in legs.items():
        t_pred, D, R = predict(s)
        delta = 100 * (s["ms_tok"] - t_pred) / t_pred
        flag = " **FLAG**" if abs(delta) > 20 else ""
        arith = f"max({R:.2f}/{H2D_GBPS:.0f}, {D:.2f}/{DISK_GBPS})+{D:.2f}/{H2D_GBPS:.0f} = {t_pred:.0f} ms"
        print(f"| {name} | {s['disk_share']:.3f} | {D:.2f} | {R:.2f} | {t_pred:.0f} | "
              f"{s['ms_tok']:.0f} | {delta:+.1f}{flag} | {arith} |")


if __name__ == "__main__":
    main()
