#!/usr/bin/env python3
"""Parity-floor harness: greedy text comparison between two llama-cli builds (HIP vs CPU).

Thin shim — implementation lives in the scripts/parity_floor/ package.
Commands:
  run-leg  --bin <dir>/llama-cli --name <leg> [--hip] [--threads N] [--prompts N]
           [--n N] [--timeout S] [--out DIR]
  compare  <legA> <legB> [--out DIR]
  --selftest
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from parity_floor.cli import main

if __name__ == "__main__":
    sys.exit(main())
