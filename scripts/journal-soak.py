#!/usr/bin/env python3
"""24 h placement-journal soak (§6.3/§11 — hardware, not CI-gated).

Drives the placement journal cycle (start → seed-from-journal → run trace →
graceful stop → next iteration seeds from the previous journal) against the
real model for --duration seconds (default 24 h), appending one JSONL line per
iteration to --out:

    {"iter": i, "warm": bool, "early_pageins": n, "elapsed_s": t}

Iteration 0 is the cold baseline (no journal to seed from). The script's own
pass criterion — printed, never CI-enforced — is a drop in early_pageins after
iteration 1.

RUNNER CONTRACT (--runner <cmd>, repeatable, required outside --selftest):
  - the runner command receives the replay trace on stdin: one flat key
    (decimal usize) per line, blank line = step boundary;
  - the runner owns instance start, seed-from-journal (with GOLEM_GAP_STEPS
    env for --gap-steps) and graceful stop;
  - on exit it must print exactly one JSON line: {"early_pageins": n}
    (early-window slow-path count for this iteration);
  - the binary implementing this contract is an M3 deliverable (golem-server
    is a stub today), so until it exists this script is scaffolding behind
    --selftest.

--gap-steps (default 0) is passed to the runner as the GOLEM_GAP_STEPS env var
(not a CLI arg): restarts are seconds apart so 0 is realistic; the flag exists
so a realistic journal-age gap is measurable. The runner owns the actual
seeding math.

Degradation: a runner that exits nonzero or emits unparseable JSON is warned
about and recorded as an iteration with early_pageins=null; the soak never
aborts on it. (The runner owns journal read/parse too — a journal failure is
its problem to cold-start from; ours is to keep looping regardless.)
"""
import argparse, json, os, shlex, subprocess, sys, time

DEFAULT_DURATION_S = 24 * 3600
SUMMARY_KEYS = {"iter", "warm", "early_pageins", "elapsed_s"}


def parse_trace(text):
    """Trace text → list of steps (lists of flat keys); blank line = step boundary."""
    steps, step = [], []
    for line in text.splitlines():
        if line.strip():
            step.append(int(line))
        elif step:
            steps.append(step)
            step = []
    if step:
        steps.append(step)
    return steps


def run_iteration(runner_cmd, trace_text, gap_steps, env=None):
    """Feed trace to one runner invocation; return its {"early_pageins": n} (or None).

    Runner failure (nonzero exit, no/invalid JSON) → warn + None; never raises.
    """
    cmd = shlex.split(runner_cmd)
    renv = dict(env or os.environ, GOLEM_GAP_STEPS=str(gap_steps))
    try:
        proc = subprocess.run(cmd, input=trace_text, capture_output=True,
                              text=True, env=renv, timeout=None)
    except OSError as e:
        print(f"WARN: runner {cmd[0]!r} failed to start: {e}", file=sys.stderr)
        return None
    if proc.returncode != 0:
        print(f"WARN: runner exited {proc.returncode}: {proc.stderr.strip()[:500]}",
              file=sys.stderr)
    try:
        res = json.loads(proc.stdout.strip().splitlines()[-1])
        if not isinstance(res.get("early_pageins"), int):
            raise ValueError("early_pageins missing/not int")
        return {"early_pageins": res["early_pageins"]}
    except (ValueError, json.JSONDecodeError, IndexError):
        print('WARN: runner emitted no parseable {"early_pageins": n} line '
              f"(stdout tail: {proc.stdout.strip()[-200:]!r})", file=sys.stderr)
        return None


def summary_line(iter_, warm, early_pageins, elapsed_s):
    """One JSONL summary line; raises if the shape ever drifts."""
    line = {"iter": iter_, "warm": bool(warm),
            "early_pageins": early_pageins, "elapsed_s": round(elapsed_s, 3)}
    assert set(line) == SUMMARY_KEYS
    return json.dumps(line)


def pass_criterion(pageins):
    """True if early_pageins dropped after iteration 1: [cold, w0, ...] with w0 < cold."""
    return len(pageins) >= 2 and pageins[0] is not None and pageins[1] is not None \
        and pageins[1] < pageins[0]


def selftest():
    trace = "12\n34\n\n56\n\n"
    steps = parse_trace(trace)
    assert steps == [[12, 34], [56]], steps  # blank line splits; trailing blank dropped

    echo_runner = "python3 -c 'import json,sys; keys=[l for l in sys.stdin if l.strip()];" \
        " print(json.dumps({\"early_pageins\": len(keys)}))'"
    res = run_iteration(echo_runner, trace, gap_steps=7)
    assert res == {"early_pageins": 3}, res  # 3 flat keys fed through stdin

    import tempfile
    with tempfile.TemporaryDirectory(prefix="journal-soak-selftest-") as tmp:
        out = os.path.join(tmp, "soak.jsonl")
        with open(out, "a") as f:
            f.write(summary_line(0, False, 100, 1.25) + "\n")
            f.write(summary_line(1, True, 10, 0.5) + "\n")
        lines = [json.loads(l) for l in open(out)]
        assert len(lines) == 2
        assert all(set(l) == SUMMARY_KEYS for l in lines), lines
        assert lines[0] == {"iter": 0, "warm": False, "early_pageins": 100, "elapsed_s": 1.25}
        assert lines[1]["warm"] is True and lines[1]["early_pageins"] == 10

    assert pass_criterion([100, 10, 9]) is True
    assert pass_criterion([100, 100]) is False
    assert pass_criterion([100]) is False  # single cold iteration: no verdict

    # Full loop with a fake runner (drop in early_pageins on iteration 1).
    import tempfile
    with tempfile.TemporaryDirectory(prefix="journal-soak-selftest-") as tmp:
        out = os.path.join(tmp, "soak.jsonl")
        fake = "python3 -c 'import json,os,sys; sys.stdin.read();" \
            " print(json.dumps({\"early_pageins\": int(os.environ[\"FAKE_PAGEINS\"])}))'"
        env = dict(os.environ, FAKE_PAGEINS="50", GOLEM_GAP_STEPS="0")
        soak({"runner": [fake], "duration": 0.01, "out": out, "gap_steps": 0,
              "trace": trace}, env=env,
             fake_env_overrides={1: {"FAKE_PAGEINS": "5"}})
        lines = [json.loads(l) for l in open(out)]
        assert len(lines) == 2 and lines[0]["early_pageins"] == 50 \
            and lines[1]["early_pageins"] == 5, lines
        assert [l["warm"] for l in lines] == [False, True]
    print("SELFTEST-OK")


def soak(argv, env=None, fake_env_overrides=None):
    """Main loop: invoke runner(s) until --duration elapses, one JSONL line per run.

    fake_env_overrides (selftest only): {iter: env-dict} merged into the runner
    env for that iteration, so a fake runner can emit scripted pageins counts.
    """
    runners = argv["runner"]
    deadline = time.monotonic() + argv["duration"]
    pageins, iter_ = [], 0
    while iter_ < 2 or time.monotonic() < deadline:  # always ≥2: cold + one warm
        warm = iter_ > 0
        renv = dict(env or os.environ)
        renv.update((fake_env_overrides or {}).get(iter_, {}))
        t0 = time.monotonic()
        res = run_iteration(runners[iter_ % len(runners)], argv["trace"],
                            argv["gap_steps"], env=renv)
        elapsed = time.monotonic() - t0
        n = res["early_pageins"] if res else None
        pageins.append(n)
        line = summary_line(iter_, warm, n, elapsed)
        print(line)
        if argv["out"]:
            with open(argv["out"], "a") as f:
                f.write(line + "\n")
        iter_ += 1
    verdict = pass_criterion(pageins)
    print(f"SOAK: {iter_} iterations; pass criterion (early_pageins drop after "
          f"iteration 1): {'PASS' if verdict else 'NOT MET' if len(pageins) >= 2 else 'N/A (single iteration)'}")
    return verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runner", action="append",
                    help="runner command (shlex-split); owns start/seed/stop, "
                         "trace on stdin, emits {\"early_pageins\": n}. Repeatable: "
                         "iterations cycle through runners")
    ap.add_argument("--duration", type=float, default=DEFAULT_DURATION_S,
                    help=f"soak wall-clock budget in seconds (default {DEFAULT_DURATION_S})")
    ap.add_argument("--gap-steps", type=int, default=0, dest="gap_steps",
                    help="journal age in steps, passed to runner as GOLEM_GAP_STEPS")
    ap.add_argument("--out", help="JSONL summary path (gitignored scratch)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        sys.exit(selftest())
    if not args.runner:
        ap.error("--runner is required (unless --selftest): no runner binary exists yet "
                 "until the M3 deliverable implements the contract (see docstring)")
    # Synthetic default trace (3 steps × 4 keys) until the M3 replay tooling
    # emits the real one; override via GOLEM_SOAK_TRACE.
    argv = vars(args)
    argv["trace"] = os.environ.get("GOLEM_SOAK_TRACE", "101\n102\n103\n104\n\n"
                                   "201\n202\n203\n204\n\n301\n302\n303\n304\n")
    soak(argv)


if __name__ == "__main__":
    main()
