#!/usr/bin/env python3
"""Compare parity dumps: exact tokens, logits within tolerance.

Usage: t4_gate_cmp.py <baseline.jsonl> <candidate.jsonl>
Tolerance: 1e-5 relative (covers last-ulp codegen drift across machines;
a real numeric regression shifts logits far more, and flips tokens first
under greedy decode).
"""
import json
import math
import sys

REL_TOL = 1e-5


def main() -> int:
    base, cand = sys.argv[1], sys.argv[2]
    with open(base) as fb, open(cand) as fc:
        blines, clines = fb.readlines(), fc.readlines()
    if len(blines) != len(clines):
        print(f"line count: baseline {len(blines)} vs candidate {len(clines)}", file=sys.stderr)
        return 1
    worst = 0.0
    for i, (b, c) in enumerate(zip(blines, clines)):
        jb, jc = json.loads(b), json.loads(c)
        if set(jb) != set(jc):
            print(f"line {i}: key mismatch {sorted(jb)} vs {sorted(jc)}", file=sys.stderr)
            return 1
        for k in jb:
            if k in ("top1", "top2"):
                bv, cv = float(jb[k]), float(jc[k])
                rel = abs(bv - cv) / max(abs(bv), abs(cv), 1e-30)
                worst = max(worst, rel)
                if not (math.isfinite(bv) and math.isfinite(cv)) or rel > REL_TOL:
                    print(f"line {i}: {k} {bv} vs {cv} (rel {rel:.2e} > {REL_TOL})", file=sys.stderr)
                    return 1
            elif jb[k] != jc[k]:
                print(f"line {i}: {k} {jb[k]!r} vs {jc[k]!r}", file=sys.stderr)
                return 1
    print(f"max relative logit delta: {worst:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
