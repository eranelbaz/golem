"""Command-line entry point for the parity-floor harness."""
from __future__ import annotations

import argparse
import sys

from .compare import compare
from .config import DEFAULT_OUTDIR, DEFAULT_MODEL
from .runner import run_leg
from .selftest import selftest


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Parity-floor harness: greedy text comparison between two llama-cli builds (HIP vs CPU).")
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    rl = sub.add_parser("run-leg")
    rl.add_argument("--bin", required=True)
    rl.add_argument("--name", required=True)
    rl.add_argument("--model", default=DEFAULT_MODEL)
    rl.add_argument("--hip", action="store_true")
    rl.add_argument("--threads", type=int, default=None)
    rl.add_argument("--prompts", type=int, default=None, help="run only prompt index N")
    rl.add_argument("--n", type=int, default=96, help="tokens to generate per prompt")
    rl.add_argument("--timeout", type=int, default=3600, help="per-prompt timeout seconds")
    rl.add_argument("--out", default=DEFAULT_OUTDIR)
    cp = sub.add_parser("compare")
    cp.add_argument("a")
    cp.add_argument("b")
    cp.add_argument("--out", default=DEFAULT_OUTDIR)
    ns = ap.parse_args()
    if ns.selftest:
        return selftest()
    if ns.cmd == "run-leg":
        return run_leg(ns.bin, ns.name, ns.model, ns.hip, ns.threads,
                       ns.prompts, ns.n, ns.timeout, ns.out)
    if ns.cmd == "compare":
        return compare(ns.a, ns.b, ns.out)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
