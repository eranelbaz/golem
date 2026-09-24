"""Command-line entry point for the parity-suite driver."""
from __future__ import annotations

import argparse
import sys

from .driver import (DEFAULT_BOUND, DEFAULT_EPSILON, DEFAULT_OUTDIR, DEFAULT_QUANT,
                     DEFAULT_THRESHOLD_PCT, DEFAULT_TIMEOUT_S, run_suite)
from .selftest import selftest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="parity-suite.py",
        description="M1.4 parity suite: T1 op parity + T2 end-to-end greedy token parity "
                    "over two substrate builds (run from the repo root: vendor/ and "
                    "engine/ are resolved relative to cwd).")
    ap.add_argument("--selftest", action="store_true",
                    help="fabricated dumps end-to-end through the real comparator; "
                         "no builds, no hardware")
    sub = ap.add_subparsers(dest="cmd")
    run = sub.add_parser("run", help="provenance gate → T1 legs → T2 legs → compare → report")
    run.add_argument("--cpu-bin", required=True, help="cpu-linked golem-cli binary")
    run.add_argument("--hip-bin", help="hip-linked golem-cli binary (required unless --skip-hip)")
    run.add_argument("--model", required=True, help="GGUF model file (identity recorded as "
                                                    "sha256 + bytes only)")
    run.add_argument("--out", default=DEFAULT_OUTDIR, help="output dir for dumps, logs, "
                                                           "report.json")
    run.add_argument("--quant", choices=["fp32", "q8_0"], default=DEFAULT_QUANT,
                     help="T1 op-leg weight encoding")
    run.add_argument("--skip-hip", action="store_true",
                     help="run the cpu leg only and self-compare (reference-vs-reference)")
    run.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S,
                     help="per-leg timeout seconds (whole process group is killed)")
    run.add_argument("--xtask", default=None,
                     help="prebuilt xtask binary; default: target/debug|release/xtask, "
                          "else cargo run (may compile)")
    run.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                     help="T2 logit-gap epsilon")
    run.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_PCT,
                     help="T2 required match-rate percentage")
    run.add_argument("--bound", type=float, default=DEFAULT_BOUND,
                     help="T1 max-abs checksum bound")
    run.add_argument("--threads-a", type=int, default=None,
                     help="decode threads for the cpu (leg A) T2 golem invocations "
                          "(golem run --threads); default: no flag, engine default")
    run.add_argument("--threads-b", type=int, default=None,
                     help="decode threads for the hip (leg B) T2 golem invocations "
                          "(golem run --threads); default: no flag, engine default")
    ns = ap.parse_args(argv)
    if ns.selftest:
        return selftest()
    if ns.cmd == "run":
        if not ns.skip_hip and not ns.hip_bin:
            ap.error("--hip-bin is required unless --skip-hip")
        return run_suite(cpu_bin=ns.cpu_bin, hip_bin=ns.hip_bin, model=ns.model,
                         outdir=ns.out, quant=ns.quant, skip_hip=ns.skip_hip,
                         timeout_s=ns.timeout, xtask=ns.xtask, epsilon=ns.epsilon,
                         threshold_pct=ns.threshold, bound=ns.bound,
                         threads_a=ns.threads_a, threads_b=ns.threads_b)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
