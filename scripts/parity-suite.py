#!/usr/bin/env python3
"""Parity-suite driver: T1 (op parity) + T2 (end-to-end greedy token
parity over the 8 frozen benchmark prompts) between two golem-cli substrate builds.

Thin shim — implementation lives in the scripts/parity_suite/ package.
Commands:
  run --cpu-bin <path> --hip-bin <path> --model <path> --out DIR
      [--quant fp32|q8_0] [--skip-hip] [--timeout S] [--xtask BIN]
      [--epsilon E] [--threshold PCT] [--bound B]
      [--threads-a N] [--threads-b N]
  --selftest
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parity_suite.cli import main

if __name__ == "__main__":
    sys.exit(main())
