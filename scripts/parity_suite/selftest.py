"""--selftest: fabricated dumps through the real comparator + report path.

Two tiny synthetic dump pairs (in-script constants — one exact pair that must
pass) are written to a temp dir and pushed through the same comparator
resolution, parity-compare invocation, merge, and scrubbed-report write the
`run` command uses. No builds beyond the cargo-run fallback, no hardware, no
model. Asserts comparator exit 0 and a report that parses with pass=true and
carries no absolute paths.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import traceback

from .driver import (DEFAULT_BOUND, DEFAULT_EPSILON, DEFAULT_THRESHOLD_PCT, PASS_BASIS,
                     SUITE_VERSION, dump_header_identity, merged_t1, merged_t2,
                     per_prompt_verdicts, resolve_comparator, run_comparator,
                     verify_dump_backend, write_report)

SELFTEST_MODEL_SHA = hashlib.sha256(b"parity-suite-selftest").hexdigest()
SELFTEST_TIMEOUT_S = 1800  # cold-start headroom for the cargo-run fallback

# Exact pair: identical steps on both legs → comparator must report pass.
_SELFTEST_STEPS = [
    {"position": 0, "token": 11, "top1": 5.5, "top2": 1.25},
    {"position": 1, "token": -3, "top1": 0.0, "top2": 0.0},
    {"position": 2, "token": 42, "top1": -7.75, "top2": -9.5},
]


# Diverged pair: position 1 flips token with a huge gap on both legs (large
# gap on leg A too, else the divergence lands inside ε and gets tolerated).
# The token rule must FAIL it, proving the honest-FAIL branch of the merged
# rule is reachable.
def _diverged_steps(hip: bool) -> list[dict]:
    steps = [dict(step) for step in _SELFTEST_STEPS]
    steps[1] = {**steps[1], "top1": 9.0, "top2": -9.0}
    if hip:
        steps[1]["token"] = 777
    return steps


def _t2_dump(backend: str, steps: list[dict] | None = None) -> str:
    steps = _SELFTEST_STEPS if steps is None else steps
    header = json.dumps({"format": "golem-parity-dump", "version": 1,
                         "model_sha256": SELFTEST_MODEL_SHA, "model_bytes": 1234,
                         "backend": backend, "offload": backend == "hip",
                         "prompt_tokens": 9, "n_predict": len(steps)})
    return header + "\n" + "".join(json.dumps(step) + "\n" for step in steps)


def _t1_dump(backend: str) -> str:
    header = json.dumps({"format": "golem-parity-op", "version": 1, "backend": backend,
                         "m": 256, "k": 128, "n_expert": 8, "n_ids": 4, "seed": 1234,
                         "quant": "fp32"})
    rows = "".join(json.dumps({"checksum": 1.5 * (i + 1), "id": i}) + "\n" for i in range(4))
    return header + "\n" + rows


def selftest() -> int:
    try:
        with tempfile.TemporaryDirectory(prefix="parity-suite-selftest-") as tmp:
            dumps = {}
            for backend in ("cpu", "hip"):
                for kind, render in (("t1", _t1_dump), ("t2", _t2_dump)):
                    path = os.path.join(tmp, f"{kind}-{backend}.jsonl")
                    with open(path, "w") as dump_file:
                        dump_file.write(render(backend))
                    dumps[f"{kind}-{backend}"] = path
                    verify_dump_backend(path, backend)

            prefix = resolve_comparator(None)
            t1 = run_comparator(prefix, ["parity-compare", "t1", "--a", dumps["t1-cpu"],
                                         "--b", dumps["t1-hip"], "--bound",
                                         repr(DEFAULT_BOUND)], SELFTEST_TIMEOUT_S)
            t2_pair = run_comparator(prefix, ["parity-compare", "t2", "--a", dumps["t2-cpu"],
                                              "--b", dumps["t2-hip"], "--epsilon",
                                              repr(DEFAULT_EPSILON), "--threshold",
                                              repr(DEFAULT_THRESHOLD_PCT)], SELFTEST_TIMEOUT_S)
            if t1["pass"] is not True:
                sys.exit(f"SELFTEST-FAILED: t1 pass={t1['pass']!r} (expected true)")
            if t2_pair["pass"] is not True:
                sys.exit(f"SELFTEST-FAILED: t2 pass={t2_pair['pass']!r} (expected true)")

            state = {
                "status": "complete", "pass": None,
                "pin_sha": "0" * 64, "backend_env_input": "selftest",
                "comparator": "xtask", "quant": "fp32", "skip_hip": False,
                "epsilon": DEFAULT_EPSILON, "threshold_pct": DEFAULT_THRESHOLD_PCT,
                "bound": DEFAULT_BOUND,
                "model": dump_header_identity([dumps["t2-cpu"], dumps["t2-hip"]]),
                "t1": merged_t1(t1),
                "t2": merged_t2([t2_pair], DEFAULT_EPSILON, DEFAULT_THRESHOLD_PCT),
                "per_prompt": per_prompt_verdicts([t2_pair]),
                "legs": {"cpu": {"bin": "selftest-cpu", "t1": ["t1-cpu.jsonl"],
                                 "t2": ["t2-cpu.jsonl"]},
                         "hip": {"bin": "selftest-hip", "t1": ["t1-hip.jsonl"],
                                 "t2": ["t2-hip.jsonl"]}},
            }
            state["pass"] = bool(state["t1"]["pass"] and state["t2"]["pass"])
            write_report(state, tmp)

            with open(os.path.join(tmp, "report.json")) as report_file:
                raw = report_file.read()
            report = json.loads(raw)  # parses
            if report.get("pass") is not True:
                sys.exit(f"SELFTEST-FAILED: report pass={report.get('pass')!r} (expected true)")
            if report["model"]["sha256"] != SELFTEST_MODEL_SHA:
                sys.exit("SELFTEST-FAILED: report model identity mismatch")
            if report.get("pass_basis") != PASS_BASIS:
                sys.exit("SELFTEST-FAILED: report pass_basis missing/wrong")
            if tmp in raw:
                sys.exit("SELFTEST-FAILED: report.json leaks the temp dir path")

            # Merged-rule coverage: run the diverged pair through the real
            # comparator, then assert both branches of the landed rule
            # (t1 AND (token OR bound)) that run_suite computes.
            for backend in ("cpu", "hip"):
                path = os.path.join(tmp, f"t2-div-{backend}.jsonl")
                with open(path, "w") as dump_file:
                    dump_file.write(_t2_dump(backend, _diverged_steps(backend == "hip")))
                dumps[f"t2-div-{backend}"] = path
            t2_div = run_comparator(prefix, ["parity-compare", "t2",
                                             "--a", dumps["t2-div-cpu"],
                                             "--b", dumps["t2-div-hip"], "--epsilon",
                                             repr(DEFAULT_EPSILON), "--threshold",
                                             repr(DEFAULT_THRESHOLD_PCT)],
                                    SELFTEST_TIMEOUT_S)
            if t2_div["pass"] is not False or t2_div["diverged"] != 1:
                sys.exit("SELFTEST-FAILED: diverged pair must fail the token rule")
            token_only = merged_t2([t2_div], DEFAULT_EPSILON, DEFAULT_THRESHOLD_PCT)
            bounded = merged_t2([t2_div], DEFAULT_EPSILON, DEFAULT_THRESHOLD_PCT,
                                gpu_dev_bound=9.5)  # fixed: > diverged pair's max_logit_dev 9.0
            suite_token = bool(t1["pass"] and (token_only["pass"] or token_only["gpu_leg_pass"] is True))
            suite_bound = bool(t1["pass"] and (bounded["pass"] or bounded["gpu_leg_pass"] is True))
            if suite_token is not False or suite_bound is not True:
                sys.exit(f"SELFTEST-FAILED: merged rule wrong (token={suite_token} "
                         f"bound={suite_bound})")
            print(f"SELFTEST: t1 max_abs={t1['max_abs']} t2 verdict="
                  f"{report['per_prompt'][0]['verdict']} pass={report['pass']} "
                  f"merged-rule token/bound={suite_token}/{suite_bound}")
    except (SystemExit, Exception) as failure:
        if isinstance(failure, SystemExit) and str(failure).startswith("SELFTEST-FAILED"):
            print(str(failure), file=sys.stderr)
            return 1
        print("SELFTEST-FAILED:", file=sys.stderr)
        traceback.print_exc()
        return 1
    print("SELFTEST-OK")
    return 0
