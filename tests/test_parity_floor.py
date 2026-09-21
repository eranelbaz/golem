"""Pytest entry point for parity-floor harness tests.

Single source of truth is scripts/parity_floor/tests.py (also runnable
offline via `scripts/parity-floor.py --selftest`).
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from parity_floor import tests as parity_tests  # noqa: E402,F401


def _reexport():
    for name in dir(parity_tests):
        if name.startswith("test_"):
            globals()[name] = getattr(parity_tests, name)


_reexport()
del _reexport
