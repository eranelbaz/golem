"""--selftest entry point: runs every test_* in parity_floor.tests without pytest."""
from __future__ import annotations

import traceback

from . import tests as _tests


def selftest() -> int:
    failed = 0
    for name in sorted(n for n in dir(_tests) if n.startswith("test_")):
        fn = getattr(_tests, name)
        try:
            fn()
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
