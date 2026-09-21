"""--selftest entry point: runs tests/test_parity_floor.py without pytest."""
from __future__ import annotations

import importlib.util
import os
import sys
import traceback

TESTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          "tests", "test_parity_floor.py")


def _load_tests():
    spec = importlib.util.spec_from_file_location("parity_floor_tests", TESTS_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["parity_floor_tests"] = module
    spec.loader.exec_module(module)
    return module


def selftest() -> int:
    if not os.path.exists(TESTS_PATH):
        print(f"SELFTEST-FAILED: tests not found at {TESTS_PATH}")
        return 1
    tests_module = _load_tests()
    failed = 0
    for name in sorted(n for n in dir(tests_module) if n.startswith("test_")):
        try:
            getattr(tests_module, name)()
            print(f"ok   {name}")
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    if failed:
        print(f"SELFTEST-FAILED ({failed} test(s))")
        return 1
    print("SELFTEST-OK")
    return 0
