"""Parity-suite driver package.

Sequences the T1 (MUL_MAT_ID op parity) and T2 (end-to-end greedy token
parity, the 8 frozen benchmark prompts) harness legs over two substrate builds and merges the
per-pair comparator reports into one scrubbed report.json:
  - model identity is sha256 + byte size only (never a path)
  - binaries appear as basenames; the provenance anchor is the vendored-tree
    pin_sha, gated against vendor/VENDOR_SHA and engine/llama-cpp.lock every run
  - a leg labeled `hip` must carry dump headers proving a hip-linked binary
Shared leg plumbing (flock, atomic writes, vendor pin, subprocess lifecycle
with process-group kill) is imported from scripts/parity_floor/, never copied.
"""
from .driver import (BACKEND_ENV_VAR, DEFAULT_BOUND, DEFAULT_EPSILON, DEFAULT_OUTDIR,
                     DEFAULT_QUANT, DEFAULT_THRESHOLD_PCT, DEFAULT_TIMEOUT_S, LOCK_FILE,
                     SUITE_VERSION, build_report, check_lock_pin, dump_header_identity,
                     merged_t1, merged_t2, per_prompt_verdicts, resolve_comparator,
                     run_comparator, run_suite, verify_dump_backend, write_report)
from .selftest import SELFTEST_MODEL_SHA, selftest

__all__ = [
    "BACKEND_ENV_VAR", "DEFAULT_BOUND", "DEFAULT_EPSILON", "DEFAULT_OUTDIR",
    "DEFAULT_QUANT", "DEFAULT_THRESHOLD_PCT", "DEFAULT_TIMEOUT_S", "LOCK_FILE",
    "SUITE_VERSION", "SELFTEST_MODEL_SHA",
    "build_report", "check_lock_pin", "dump_header_identity", "merged_t1",
    "merged_t2", "per_prompt_verdicts", "resolve_comparator", "run_comparator",
    "run_suite", "selftest", "verify_dump_backend", "write_report",
]
