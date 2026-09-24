"""Suite leg orchestration: provenance gate, sequential legs, compare, report.

Sequencing (M1.4): provenance gate → T1 op legs → T2 prompt legs (8 M0.5
prompts, one dump per prompt per backend) → `xtask parity-compare` per pair →
merged scrubbed report.json, rewritten incrementally after every stage.

Path-scrub policy (session-8): report.json and stdout summaries never carry
machine-local absolute paths — binaries appear as basenames, the model as
sha256 + byte size (taken from the dump headers the binaries themselves
wrote), everything else as hashes, verdicts, or dump-file basenames.

Shared plumbing (flock, atomic write, vendor pin, subprocess lifecycle with
process-group kill) is imported from scripts/parity_floor/, never copied.
Run from the repo root: vendor/ and engine/ are resolved relative to cwd.
"""
from __future__ import annotations

import json
import os
import sys

from parity_floor.config import PROMPTS, check_vendor_pin, pin_sha
from parity_floor.files import atomic_write
from parity_floor.locking import acquire_lock, release_lock
from parity_floor.runner import run_prompt

SUITE_VERSION = 1
DEFAULT_OUTDIR = os.environ.get("GOLEM_M14_WORK", ".m1-work/parity-suite")
DEFAULT_QUANT = "fp32"
DEFAULT_TIMEOUT_S = 3600
DEFAULT_EPSILON = 1e-2
DEFAULT_THRESHOLD_PCT = 100.0
DEFAULT_BOUND = 1e-2
LOCK_FILE = "engine/llama-cpp.lock"
BACKEND_ENV_VAR = "GOLEM_LLAMA_SYS_BACKEND"


def check_lock_pin(tree_hash: str) -> None:
    """Die unless the vendored-tree hash also matches the content_sha256 the
    build contract records (engine/llama-cpp.lock) — vendor/ and the lock file
    must describe the same pin, not just VENDOR_SHA and the tree."""
    try:
        with open(LOCK_FILE) as lock_file:
            for line in lock_file:
                if line.lstrip().startswith("#"):
                    continue
                name, _, value = line.partition("=")
                if name.strip() == "content_sha256":
                    locked = value.split("#", 1)[0].strip()
                    break
            else:
                sys.exit(f"FATAL: {LOCK_FILE} has no content_sha256 line")
    except OSError:
        sys.exit(f"FATAL: cannot read {LOCK_FILE}")
    if locked != tree_hash:
        sys.exit(f"FATAL: build lock drifted ({tree_hash[:12]} != {locked[:12]}) — "
                 f"{LOCK_FILE} and the vendored tree disagree")


def _check_bin(role: str, path: str) -> None:
    if not os.path.exists(path):
        sys.exit(f"FATAL: {role} binary not found: {path}")
    if not os.access(path, os.X_OK):
        sys.exit(f"FATAL: {role} binary not executable: {path}")


def t1_args(bin_path: str, dump_path: str, quant: str) -> list[str]:
    return [bin_path, "parity-op", "--dump", dump_path, "--quant", quant]


def t2_args(bin_path: str, model: str, prompt: str, dump_path: str,
            threads: int | None = None) -> list[str]:
    """One T2 leg invocation. `threads` (the Task 7 calibration knob) maps to
    `golem run --threads N` → EngineConfig.threads; None passes no flag, so
    the leg runs at the engine default."""
    args = [bin_path, "run", "--greedy", model, "--prompt", prompt,
            "--parity-dump", dump_path]
    if threads is not None:
        args += ["--threads", str(threads)]
    return args


def resolve_comparator(explicit: str | None) -> list[str]:
    """Comparator invocation prefix: an explicit --xtask path, else a prebuilt
    target/{debug,release}/xtask, else `cargo run` (which may compile — the
    selftest avoids it whenever a prebuilt binary exists)."""
    if explicit is not None:
        _check_bin("--xtask", explicit)
        return [explicit]
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for candidate in (os.path.join(root, "target", "debug", "xtask"),
                      os.path.join(root, "target", "release", "xtask")):
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return [candidate]
    print("SUITE: no prebuilt xtask binary found — falling back to cargo run "
          "(may compile)", flush=True)
    return [os.environ.get("CARGO", "cargo"), "run", "-q", "-p", "xtask", "--"]


def run_comparator(prefix: list[str], argv_tail: list[str], timeout_s: int) -> dict:
    """One parity-compare invocation → its parsed report JSON. Comparator
    exit 0 is mandatory: any failure (guards exit 2) aborts the suite."""
    rc, stdout, stderr, _ = run_prompt(prefix + argv_tail, timeout_s)
    if rc != 0:
        sys.exit(f"FATAL: parity-compare {' '.join(argv_tail[:2])}... failed rc={rc}: "
                 f"{(stderr or '').strip()[:400]}")
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        sys.exit("FATAL: parity-compare produced no report line on stdout")
    try:
        return json.loads(lines[-1])
    except json.JSONDecodeError as err:
        sys.exit(f"FATAL: parity-compare report line is not JSON ({err})")


def _dump_header(dump_path: str) -> dict:
    try:
        with open(dump_path) as dump_file:
            for line in dump_file:
                if line.strip():
                    return json.loads(line)
    except (OSError, json.JSONDecodeError) as err:
        sys.exit(f"FATAL: cannot read dump header {os.path.basename(dump_path)} ({err})")
    sys.exit(f"FATAL: dump has no header line: {os.path.basename(dump_path)}")


def verify_dump_backend(dump_path: str, expected: str) -> None:
    """A leg labeled `hip` must have been produced by a hip-linked binary (and
    vice versa): the dump header records the substrate the binary was built
    for, so a mislabeled leg dies here instead of poisoning the comparison."""
    backend = _dump_header(dump_path).get("backend")
    if backend != expected:
        sys.exit(f"FATAL: {os.path.basename(dump_path)} records backend={backend!r}, "
                 f"expected {expected!r} — mislabeled leg binary")


def dump_header_identity(dump_paths: list[str]) -> dict:
    """Model identity (sha256 + byte size only) from the T2 dump headers; all
    dumps across all legs must describe the same model bytes."""
    identity: tuple[str, int] | None = None
    for path in dump_paths:
        header = _dump_header(path)
        key = (header.get("model_sha256"), header.get("model_bytes"))
        if not isinstance(key[0], str) or not isinstance(key[1], int):
            sys.exit(f"FATAL: {os.path.basename(path)} header has no model sha256/bytes")
        if identity is None:
            identity = key
        elif key != identity:
            sys.exit(f"FATAL: model identity mismatch across T2 dumps "
                     f"({identity[0][:12]} vs {key[0][:12]}) — legs ran different models")
    if identity is None:
        sys.exit("FATAL: no T2 dumps to read model identity from")
    return {"sha256": identity[0], "bytes": identity[1]}


def merged_t1(t1_report: dict) -> dict:
    """The single T1 comparison is the merged T1 section as-is (frozen
    comparator contract: harness/max_abs/bound/pass/backends)."""
    return t1_report


def merged_t2(pair_reports: list[dict], epsilon: float, threshold_pct: float) -> dict:
    """Merge per-pair comparator reports into one T2 section under the frozen
    pass rule: zero diverged prompts AND match_rate_pct >= threshold_pct."""
    exact = sum(r["exact"] for r in pair_reports)
    tolerated = sum(r["tolerated"] for r in pair_reports)
    diverged = sum(r["diverged"] for r in pair_reports)
    prompts = sum(r["prompts"] for r in pair_reports)
    match_rate_pct = 100.0 * exact / prompts if prompts else 0.0
    divergences = [{"prompt": idx, **d}
                   for idx, r in enumerate(pair_reports) for d in r["divergences"]]
    return {
        "harness": "t2-parity",
        "version": 1,
        "epsilon": epsilon,
        "threshold_pct": threshold_pct,
        "prompts": prompts,
        "exact": exact,
        "tolerated": tolerated,
        "diverged": diverged,
        "match_rate_pct": match_rate_pct,
        "pass": diverged == 0 and match_rate_pct >= threshold_pct,
        "divergences": divergences,
        "post_split_positions": sum(r["post_split_positions"] for r in pair_reports),
        "backends": pair_reports[0]["backends"] if pair_reports else [],
    }


def per_prompt_verdicts(pair_reports: list[dict]) -> list[dict]:
    """First-divergence verdict per prompt pair (the comparator decides per
    pair at the first divergence; later positions are post-split)."""
    verdicts = []
    for idx, report in enumerate(pair_reports):
        if report["diverged"]:
            verdict = "diverged"
        elif report["tolerated"]:
            verdict = "tolerated"
        else:
            verdict = "exact"
        first = report["divergences"][0] if report["divergences"] else None
        verdicts.append({"idx": idx, "verdict": verdict,
                         "first_divergence": ({"prompt": idx, **first} if first else None)})
    return verdicts


def build_report(state: dict) -> dict:
    """Scrubbed report: the only place internal state reaches report.json.
    Binaries are reduced to basenames here — internal state may carry
    absolute paths, the committed artifact never does."""
    legs = {}
    for backend in ("cpu", "hip"):
        leg = state.get("legs", {}).get(backend)
        if leg:
            legs[backend] = {"bin": os.path.basename(leg["bin"]),
                             "t1": list(leg.get("t1", [])),
                             "t2": list(leg.get("t2", []))}
    report = {
        "harness": "parity-suite",
        "version": SUITE_VERSION,
        "status": state.get("status", "running"),
        "pass": state.get("pass"),
        "provenance": {
            "pin_sha": state.get("pin_sha"),
            "backend_env_input": {BACKEND_ENV_VAR: state.get("backend_env_input", "unset")},
        },
        "comparator": state.get("comparator", "xtask"),
        "quant": state.get("quant"),
        "skip_hip": state.get("skip_hip", False),
        "params": {"epsilon": state.get("epsilon"), "threshold_pct": state.get("threshold_pct"),
                   "bound": state.get("bound"),
                   "threads_a": state.get("threads_a"), "threads_b": state.get("threads_b")},
        "model": state.get("model"),
        "legs": legs,
        "t1": state.get("t1"),
        "t2": state.get("t2"),
        "per_prompt": state.get("per_prompt", []),
    }
    if state.get("error"):
        report["error"] = state["error"]
    return report


def write_report(state: dict, outdir: str) -> None:
    """Incremental checkpoint: rewritten after every stage (intentional
    crash-resume durability, cheap at this artifact size)."""
    body = json.dumps(build_report(state), indent=2, sort_keys=True) + "\n"
    atomic_write(os.path.join(outdir, "report.json"), body)


def _run_leg(state: dict, outdir: str, logdir: str, name: str,
             args: list[str], timeout_s: int, expected_backend: str) -> None:
    """One leg subprocess: run under the per-leg timeout (process-group kill
    via parity_floor.run_prompt), persist its output log, verify the dump
    header's backend label, checkpoint the report. Any failure is fatal."""
    rc, stdout, stderr, wall_s = run_prompt(args, timeout_s)
    atomic_write(os.path.join(logdir, f"{name}.log"), (stdout or "") + (stderr or ""))
    if rc != 0:
        state["status"] = "failed"
        state["error"] = f"leg {name} rc={rc} (log: {name}.log)"
        write_report(state, outdir)
        reason = " (timed out)" if rc == -9 else ""
        sys.exit(f"FATAL: leg={name} rc={rc}{reason} — log saved to logs/{name}.log")
    dump_path = args[args.index("--dump") + 1] if "--dump" in args else \
        args[args.index("--parity-dump") + 1]
    verify_dump_backend(dump_path, expected_backend)
    print(f"SUITE: leg={name} rc=0 wall_s={wall_s:.1f}", flush=True)
    write_report(state, outdir)


def _check_threads(threads_a: int | None, threads_b: int | None) -> None:
    """The knobs must be engine-plausible before any leg starts (golem run
    itself rejects `--threads 0`; failing here keeps the failure message
    beside the knob that caused it)."""
    for knob, threads in (("threads-a", threads_a), ("threads-b", threads_b)):
        if threads is not None and threads < 1:
            sys.exit(f"FATAL: --{knob} must be >= 1, got {threads}")


def run_suite(cpu_bin: str, hip_bin: str | None, model: str, outdir: str, quant: str,
              skip_hip: bool, timeout_s: int, xtask: str | None,
              epsilon: float, threshold_pct: float, bound: float,
              threads_a: int | None = None, threads_b: int | None = None) -> int:
    """Full suite: provenance gate → T1 legs → T2 legs (8 prompts × legs) →
    per-pair compare → merged report. Returns 0 iff every leg succeeded and
    both merged T1/T2 sections pass.

    `threads_a` / `threads_b` are the Task 7 calibration knobs: per-leg T2
    decode-thread counts (leg A = cpu, leg B = hip). None (the default) runs
    the leg with no `--threads` flag at the engine default; together they
    express the reference-vs-reference matrix (e.g. cpu-t16 vs cpu-t4)."""
    _check_threads(threads_a, threads_b)
    tree_hash = pin_sha()
    check_vendor_pin(tree_hash)
    check_lock_pin(tree_hash)
    _check_bin("cpu", cpu_bin)
    if skip_hip:
        hip_bin = cpu_bin  # reference-vs-reference: the cpu leg self-compares
    else:
        _check_bin("hip", hip_bin)
    if not os.path.isfile(model):
        sys.exit(f"FATAL: model not found: {model}")

    os.makedirs(outdir, exist_ok=True)
    logdir = os.path.join(outdir, "logs")
    os.makedirs(logdir, exist_ok=True)
    lock_fd = acquire_lock(outdir)
    try:
        comparator_prefix = resolve_comparator(xtask)
        comparator_mode = "cargo-run" if comparator_prefix[0].endswith("cargo") else "xtask"
        b_backend = "cpu" if skip_hip else "hip"  # --skip-hip: the cpu leg self-compares
        backends = ["cpu"] if skip_hip else ["cpu", "hip"]
        bins = {"cpu": cpu_bin, "hip": hip_bin}
        state = {
            "status": "running", "pass": None,
            "pin_sha": tree_hash,
            "backend_env_input": os.environ.get(BACKEND_ENV_VAR, "unset"),
            "comparator": comparator_mode,
            "quant": quant, "skip_hip": skip_hip,
            "epsilon": epsilon, "threshold_pct": threshold_pct, "bound": bound,
            "threads_a": threads_a, "threads_b": threads_b,
            "model": None, "t1": None, "t2": None, "per_prompt": [],
            "legs": {backend: {"bin": bins[backend], "t1": [], "t2": []}
                     for backend in backends},
        }
        write_report(state, outdir)
        print(f"SUITE: pin={tree_hash[:12]} lock=ok backend_env="
              f"{state['backend_env_input']} quant={quant} "
              f"legs={','.join(backends)} comparator={comparator_mode}", flush=True)

        # T1 op legs: synthetic tensors, one dump per backend.
        t1_dumps = {}
        for backend in backends:
            dump_path = os.path.join(outdir, f"t1-{backend}-{quant}.jsonl")
            state["legs"][backend]["t1"].append(os.path.basename(dump_path))
            _run_leg(state, outdir, logdir, f"t1-{backend}",
                     t1_args(bins[backend], dump_path, quant), timeout_s, backend)
            t1_dumps[backend] = dump_path

        # T2 legs: one dump per (backend, prompt); same prompt text on both
        # legs. The calibration knobs set each leg's decode threads (leg A =
        # cpu → threads_a, leg B = hip → threads_b); T1 legs never get one
        # (parity-op has no threads surface). With --skip-hip only leg A runs.
        leg_threads = {"cpu": threads_a, "hip": threads_b}
        t2_dumps: dict[str, list[str]] = {}
        for backend in backends:
            t2_dumps[backend] = []
            for idx, prompt in enumerate(PROMPTS):
                dump_path = os.path.join(outdir, f"t2-{backend}-{idx}.jsonl")
                state["legs"][backend]["t2"].append(os.path.basename(dump_path))
                _run_leg(state, outdir, logdir, f"t2-{backend}-{idx}",
                         t2_args(bins[backend], model, prompt, dump_path,
                                 leg_threads[backend]), timeout_s, backend)
                t2_dumps[backend].append(dump_path)

        state["model"] = dump_header_identity(t2_dumps["cpu"] + t2_dumps.get("hip", []))
        write_report(state, outdir)
        print(f"SUITE: model sha256={state['model']['sha256'][:12]} "
              f"bytes={state['model']['bytes']}", flush=True)

        # Compare: T1 once, then T2 per prompt pair (incremental report).
        t1_report = run_comparator(
            comparator_prefix,
            ["parity-compare", "t1", "--a", t1_dumps["cpu"],
             "--b", t1_dumps[b_backend],
             "--bound", repr(bound)],
            timeout_s)
        state["t1"] = merged_t1(t1_report)
        write_report(state, outdir)
        print(f"SUITE: compare=t1 pass={str(state['t1']['pass']).lower()} "
              f"max_abs={state['t1']['max_abs']}", flush=True)

        pair_reports = []
        for idx in range(len(PROMPTS)):
            pair_reports.append(run_comparator(
                comparator_prefix,
                ["parity-compare", "t2", "--a", t2_dumps["cpu"][idx],
                 "--b", t2_dumps[b_backend][idx],
                 "--epsilon", repr(epsilon), "--threshold", repr(threshold_pct)],
                timeout_s))
            state["t2"] = merged_t2(pair_reports, epsilon, threshold_pct)
            state["per_prompt"] = per_prompt_verdicts(pair_reports)
            write_report(state, outdir)
            print(f"SUITE: compare=t2-{idx} "
                  f"verdict={state['per_prompt'][idx]['verdict']}", flush=True)

        state["status"] = "complete"
        state["pass"] = bool(state["t1"]["pass"] and state["t2"]["pass"])
        write_report(state, outdir)
        t2 = state["t2"]
        print(f"SUITE: t2 prompts={t2['prompts']} exact={t2['exact']} "
              f"tolerated={t2['tolerated']} diverged={t2['diverged']} "
              f"match_rate_pct={t2['match_rate_pct']}", flush=True)
        print(f"SUITE: {'PASS' if state['pass'] else 'FAIL'} "
              f"(report written: report.json)", flush=True)
        return 0 if state["pass"] else 1
    finally:
        release_lock(lock_fd)
