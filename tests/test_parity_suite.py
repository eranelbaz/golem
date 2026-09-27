"""Parity-suite driver tests.

`test_selftest_end_to_end_passes` shells out to `scripts/parity-suite.py
--selftest`, which runs fabricated dumps end-to-end through the real
comparator (xtask parity-compare); the remaining tests drive the
scripts/parity_suite/ package API directly.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest

import parity_suite
from parity_suite.driver import (dump_header_identity, merged_t2, per_prompt_verdicts,
                                 verify_dump_backend, write_report)


def test_package_exports_suite_version():
    assert parity_suite.SUITE_VERSION == 1

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "parity-suite.py")

SELFTEST_TIMEOUT_S = 3600  # covers the cold cargo-run fallback on a fresh clone


def test_selftest_end_to_end_passes():
    """The selftest must exit 0 through the real comparator path (exit 0 of
    every parity-compare invocation and a report that parses with pass=true
    are asserted inside the selftest itself)."""
    proc = subprocess.run([sys.executable, SCRIPT, "--selftest"], cwd=REPO_ROOT,
                          capture_output=True, text=True, timeout=SELFTEST_TIMEOUT_S)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "SELFTEST-OK" in proc.stdout


def test_report_scrubs_absolute_paths(tmp_path):
    """Feed absolute bin/model paths in; the written report.json must carry
    basenames and hashes only — the temp dir string must not appear anywhere."""
    bin_path = str(tmp_path / "golem-all-your-paths-belong-to-us")
    model_path = str(tmp_path / "secret-model.gguf")
    # cpu_bin/hip_bin/model_path are internal-state decoys: absolute paths that
    # build_report's whitelist must never let into the artifact.
    state = {
        "cpu_bin": bin_path,
        "hip_bin": bin_path,
        "model_path": model_path,
        "model": {"sha256": "a" * 64, "bytes": 7},
        "legs": {"cpu": {"bin": bin_path, "t1": ["t1-cpu-fp32.jsonl"], "t2": ["t2-cpu-0.jsonl"]},
                 "hip": {"bin": bin_path, "t1": ["t1-hip-fp32.jsonl"], "t2": ["t2-hip-0.jsonl"]}},
    }
    write_report(state, str(tmp_path))
    raw = (tmp_path / "report.json").read_text()
    assert str(tmp_path) not in raw, "absolute out dir leaked into report.json"
    assert bin_path not in raw, "absolute bin path leaked into report.json"
    assert model_path not in raw, "absolute model path leaked into report.json"
    report = json.loads(raw)
    assert report["legs"]["cpu"]["bin"] == os.path.basename(bin_path)
    assert report["model"] == {"sha256": "a" * 64, "bytes": 7}


def _t2_header(sha: str = "b" * 64, backend: str = "cpu") -> str:
    return json.dumps({"format": "golem-parity-dump", "version": 1, "model_sha256": sha,
                       "model_bytes": 1234, "backend": backend, "offload": False,
                       "prompt_tokens": 9, "n_predict": 3})


def test_dump_header_identity_accepts_consistent_headers(tmp_path):
    dumps = []
    for backend in ("cpu", "hip"):
        path = tmp_path / f"t2-{backend}-0.jsonl"
        path.write_text(_t2_header(backend=backend) + "\n" +
                        '{"position":0,"token":5,"top1":1.0,"top2":0.5}\n')
        dumps.append(str(path))
    assert dump_header_identity(dumps) == {"sha256": "b" * 64, "bytes": 1234}


def test_dump_header_identity_rejects_model_mismatch(tmp_path):
    a = tmp_path / "t2-cpu-0.jsonl"
    b = tmp_path / "t2-hip-0.jsonl"
    a.write_text(_t2_header(sha="b" * 64) + "\n")
    b.write_text(_t2_header(sha="c" * 64) + "\n")
    with pytest.raises(SystemExit, match="model identity"):
        dump_header_identity([str(a), str(b)])


def _pair_report(exact=0, tolerated=0, diverged=0, post_split=0, divergences=None,
                 max_logit_dev=0.0):
    return {"harness": "t2-parity", "version": 2, "epsilon": 0.01, "threshold_pct": 100.0,
            "prompts": 1, "exact": exact, "tolerated": tolerated, "diverged": diverged,
            "match_rate_pct": 100.0 * exact, "pass": diverged == 0 and exact == 1,
            "divergences": divergences or [], "post_split_positions": post_split,
            "backends": ["cpu", "hip"], "max_logit_dev": max_logit_dev,
            "gpu_dev_bound": None, "gpu_leg_pass": None}


def test_merged_t2_aggregates_and_applies_pass_rule():
    exact = _pair_report(exact=1)
    tolerated = _pair_report(tolerated=1, post_split=2,
                             divergences=[{"position": 1, "a_token": 3, "b_token": 4,
                                           "gap_a": 0.125, "gap_b": 0.5, "tolerated": True}])
    merged = merged_t2([exact, tolerated], epsilon=0.01, threshold_pct=100.0)
    assert merged["prompts"] == 2
    assert (merged["exact"], merged["tolerated"], merged["diverged"]) == (1, 1, 0)
    assert merged["match_rate_pct"] == 50.0
    assert merged["pass"] is False, "match_rate below threshold must fail"
    assert merged["divergences"][0]["prompt"] == 1, "divergences carry their prompt index"
    assert merged["post_split_positions"] == 2
    assert merged["backends"] == ["cpu", "hip"]
    assert merged["gpu_dev_bound"] is None
    assert merged["gpu_leg_pass"] is None
    assert merged["max_logit_dev"] == 0.0


def test_merged_t2_gpu_dev_bound():
    diverged = _pair_report(diverged=1, max_logit_dev=0.35,
                            divergences=[{"position": 0, "a_token": 1, "b_token": 2,
                                          "gap_a": 0.5, "gap_b": 0.5, "tolerated": False}])
    merged = merged_t2([diverged], epsilon=0.01, threshold_pct=99.5,
                       gpu_dev_bound=0.5)
    assert merged["pass"] is False, "token pass stays the token rule"
    assert merged["gpu_dev_bound"] == 0.5
    assert merged["max_logit_dev"] == 0.35
    assert merged["gpu_leg_pass"] is True, "bounded deviation passes the GPU leg"

    over = merged_t2([_pair_report(diverged=1, max_logit_dev=0.6)],
                     epsilon=0.01, threshold_pct=99.5, gpu_dev_bound=0.5)
    assert over["gpu_leg_pass"] is False


def test_per_prompt_verdicts():
    exact = _pair_report(exact=1)
    diverged = _pair_report(
        diverged=1,
        divergences=[{"position": 0, "a_token": 1, "b_token": 2,
                      "gap_a": 0.5, "gap_b": 0.5, "tolerated": False}])
    verdicts = per_prompt_verdicts([exact, diverged])
    assert [v["verdict"] for v in verdicts] == ["exact", "diverged"]
    assert verdicts[0]["first_divergence"] is None
    assert verdicts[1]["first_divergence"]["position"] == 0


def test_selftest_dumps_use_selftest_model_hash():
    """The in-script fabricated dumps must not reference any real model
    identity; their sha is derived from the selftest marker bytes."""
    from parity_suite.selftest import SELFTEST_MODEL_SHA
    assert SELFTEST_MODEL_SHA == hashlib.sha256(b"parity-suite-selftest").hexdigest()


# ---- run mode against stub binaries (no builds, no hardware) -------------

# Stub golem-cli: derives its backend label from its own file name
# (golem-cpu → cpu, golem-hip → hip), hashes the model file it was passed
# for the dump header (like the real CLI), and writes deterministic dumps.
STUB_GOLEM = '''#!/usr/bin/env python3
import hashlib, json, os, sys
args = sys.argv[1:]
# Flag whitelist: a driver typo (e.g. --n-gpu-layer) must fail the leg
# loudly, never silently vanish from the dump headers.
VALUE_FLAGS = {"--dump", "--quant", "--greedy", "--prompt", "--parity-dump",
               "--n-predict", "--threads", "--n-gpu-layers", "--cpu-moe-layers"}
ALLOWED = {"parity-op", "run", "--flash-attn"} | VALUE_FLAGS
i = 0
while i < len(args):
    flag = args[i]
    assert flag in ALLOWED, f"stub got unexpected arg {flag!r} (driver typo?)"
    i += 2 if flag in VALUE_FLAGS else 1
def value(flag):
    return args[args.index(flag) + 1]
backend = os.path.basename(sys.argv[0]).rsplit("-", 1)[-1]
mode = args[0]
if mode == "parity-op":
    out, quant = value("--dump"), value("--quant")
    header = {"format": "golem-parity-op", "version": 1, "backend": backend, "m": 256,
              "k": 128, "n_expert": 8, "n_ids": 4, "seed": 1234, "quant": quant}
    body = "".join(json.dumps({"checksum": 1.5 * (i + 1), "id": i}) + "\\n" for i in range(4))
else:
    model = open(args[args.index("--greedy") + 1], "rb").read()
    sha = hashlib.sha256(model).hexdigest()
    out, prompt = value("--parity-dump"), value("--prompt")
    # Record the pass-through `--threads` value (Task 7 calibration knob) so
    # tests can assert exactly which flag each leg received; None = flag absent.
    threads = int(args[args.index("--threads") + 1]) if "--threads" in args else None
    npredict = value("--n-predict") if "--n-predict" in args else None
    header = {"format": "golem-parity-dump", "version": 1, "model_sha256": sha,
              "model_bytes": len(model), "backend": backend, "offload": backend == "hip",
              "prompt_tokens": len(prompt.split()), "n_predict": int(npredict),
              "threads": threads,
              "gpu_layers": int(value("--n-gpu-layers")) if "--n-gpu-layers" in args else None,
              "cpu_moe": int(value("--cpu-moe-layers")) if "--cpu-moe-layers" in args else None,
              "flash_attn": "--flash-attn" in args}
    body = "".join(json.dumps({"position": p, "token": p + 1, "top1": 2.0, "top2": 0.5}) + "\\n"
                   for p in range(3))
with open(out, "w") as f:
    f.write(json.dumps(header) + "\\n" + body)
'''

FAILING_STUB = "#!/usr/bin/env sh\nexit 1\n"

# Hangs forever on the T2 leg so the suite's per-leg timeout drives the
# process-group kill; writes valid T1 dumps so earlier legs pass.
SLEEP_GOLEM = '''#!/usr/bin/env python3
import json, os, sys, time
args = sys.argv[1:]
def value(flag):
    return args[args.index(flag) + 1]
backend = os.path.basename(sys.argv[0]).rsplit("-", 1)[-1]
if args[0] == "parity-op":
    with open(value("--dump"), "w") as f:
        f.write(json.dumps({"format": "golem-parity-op", "version": 1, "backend": backend,
                            "m": 256, "k": 128, "n_expert": 8, "n_ids": 4, "seed": 1234,
                            "quant": value("--quant")}) + "\\n")
    sys.exit(0)
time.sleep(30)
'''

# A hip-linked binary that never asked for device execution: the offload
# gate must catch what the backend label alone cannot.
HIP_NO_OFFLOAD_STUB = STUB_GOLEM.replace('"offload": backend == "hip"', '"offload": False')


def _write_stub(path, content):
    path.write_text(content)
    path.chmod(0o755)
    return str(path)


def _run_suite(tmp_path, extra_args, cpu=None, hip=None):
    """Runs scripts/parity-suite.py run against stub binaries from the repo
    root (the provenance gate hashes the real vendored tree). --hip-bin is
    omitted under --skip-hip (passing both is a CLI error)."""
    bins = tmp_path / "bins"
    bins.mkdir(exist_ok=True)
    cpu = cpu or _write_stub(bins / "golem-cpu", STUB_GOLEM)
    hip = hip or _write_stub(bins / "golem-hip", STUB_GOLEM)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"stub model weights\n")
    out = str(tmp_path / "out")
    args = ["--cpu-bin", cpu]
    if "--skip-hip" not in extra_args:
        args += ["--hip-bin", hip]
    args += ["--model", str(model), "--out", out, *extra_args]
    proc = subprocess.run(
        [sys.executable, SCRIPT, "run", *args],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=SELFTEST_TIMEOUT_S)
    report_path = tmp_path / "out" / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    return proc, report


def test_run_mode_end_to_end_passes_with_stub_binaries(tmp_path):
    proc, report = _run_suite(tmp_path, [])
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert report["status"] == "complete" and report["pass"] is True
    assert report["model"] == {"sha256": hashlib.sha256(b"stub model weights\n").hexdigest(),
                               "bytes": len(b"stub model weights\n")}
    assert report["t1"]["pass"] is True and report["t1"]["max_abs"] == 0.0
    t2 = report["t2"]
    assert (t2["prompts"], t2["exact"], t2["diverged"]) == (8, 8, 0)
    assert [v["verdict"] for v in report["per_prompt"]] == ["exact"] * 8
    assert report["legs"]["cpu"]["bin"] == "golem-cpu"
    assert report["legs"]["hip"]["bin"] == "golem-hip"
    # Threads knobs unset → no injection at all (engine default on both legs).
    assert report["params"]["threads_a"] is None and report["params"]["threads_b"] is None
    header = _dump_header(tmp_path / "out" / "t2-cpu-0.jsonl")
    assert header["threads"] is None, "no --threads flag may reach the leg"
    # Scrub: neither the stub paths nor the out dir may reach report or stdout.
    raw_report = (tmp_path / "out" / "report.json").read_text()
    for artifact in (raw_report, proc.stdout):
        assert str(tmp_path) not in artifact


def _dump_header(path) -> dict:
    return json.loads(open(path).readline())


def test_run_mode_passes_threads_through_to_t2_legs(tmp_path):
    """--threads-a/--threads-b map onto `golem run --threads N` per leg
    (leg A = cpu, leg B = hip) and land in report params; T1 legs never
    receive a threads flag (parity-op has no threads surface)."""
    proc, report = _run_suite(tmp_path, ["--threads-a", "16", "--threads-b", "4"])
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert report["status"] == "complete" and report["pass"] is True
    assert report["params"]["threads_a"] == 16 and report["params"]["threads_b"] == 4
    out = tmp_path / "out"
    for idx in range(8):
        assert _dump_header(out / f"t2-cpu-{idx}.jsonl")["threads"] == 16
        assert _dump_header(out / f"t2-hip-{idx}.jsonl")["threads"] == 4
    for backend in ("cpu", "hip"):
        t1_header = _dump_header(out / f"t1-{backend}-fp32.jsonl")
        assert "threads" not in t1_header, "T1 legs must not carry a threads flag"


def test_run_mode_rejects_non_positive_threads_before_any_leg(tmp_path):
    proc, report = _run_suite(tmp_path, ["--threads-a", "0"])
    assert proc.returncode == 1
    assert report is None, "validation fails before the out dir is created"
    assert "--threads-a" in proc.stderr and "FATAL" in proc.stderr


def test_run_mode_hardware_leg_flags_reach_t2_legs(tmp_path):
    """Task-7 hardware-leg contract: both T2 legs generate n_predict=96 with
    flash-attn forced on; only the GPU (hip) leg receives the offload knobs
    (--n-gpu-layers 99 --cpu-moe-layers 45, the -ncmoe equivalent). T1 legs
    are synthetic and must never receive T2 flags."""
    proc, report = _run_suite(tmp_path, [])
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    out = tmp_path / "out"
    for backend in ("cpu", "hip"):
        for idx in range(8):
            h = _dump_header(out / f"t2-{backend}-{idx}.jsonl")
            assert h["n_predict"] == 96, "hardware legs generate 96 tokens"
            assert h["flash_attn"] is True, "fa=on on both hardware legs"
    hip_h = _dump_header(out / "t2-hip-0.jsonl")
    cpu_h = _dump_header(out / "t2-cpu-0.jsonl")
    assert hip_h["gpu_layers"] == 99 and hip_h["cpu_moe"] == 45
    assert cpu_h["gpu_layers"] is None and cpu_h["cpu_moe"] is None, \
        "the CPU reference leg must stay fully on CPU"
    for backend in ("cpu", "hip"):
        t1_header = _dump_header(out / f"t1-{backend}-fp32.jsonl")
        assert "flash_attn" not in t1_header and "n_predict" not in t1_header
    assert report["params"]["n_predict"] == 96
    assert report["params"]["gpu_layers"] == 99
    assert report["params"]["cpu_moe"] == 45
    assert report["params"]["flash_attn_a"] is True
    assert report["params"]["flash_attn_b"] is True


def test_run_mode_per_leg_flash_attn_knobs(tmp_path):
    """fa on-vs-off falsification: --no-flash-attn-a / --no-flash-attn-b must
    land on their own leg only."""
    proc, report = _run_suite(tmp_path, ["--no-flash-attn-a"])
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    out = tmp_path / "out"
    assert _dump_header(out / "t2-cpu-0.jsonl")["flash_attn"] is False
    assert _dump_header(out / "t2-hip-0.jsonl")["flash_attn"] is True
    assert report["params"]["flash_attn_a"] is False
    assert report["params"]["flash_attn_b"] is True


def test_run_mode_skip_hip_applies_no_gpu_knobs(tmp_path):
    """--skip-hip self-comparison is CPU-vs-CPU: no leg may receive the
    offload knobs."""
    proc, report = _run_suite(tmp_path, ["--skip-hip"])
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    h = _dump_header(tmp_path / "out" / "t2-cpu-0.jsonl")
    assert h["gpu_layers"] is None and h["cpu_moe"] is None


def test_run_mode_skip_hip_self_compares_cpu(tmp_path):
    proc, report = _run_suite(tmp_path, ["--skip-hip"])
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert report["skip_hip"] is True and report["pass"] is True
    assert "hip" not in report["legs"]
    assert report["t2"]["backends"] == ["cpu", "cpu"]


def test_run_mode_failing_leg_writes_failed_report(tmp_path):
    proc, report = _run_suite(tmp_path, [], cpu=_write_stub(tmp_path / "broken", FAILING_STUB))
    assert proc.returncode == 1
    assert report["status"] == "failed"
    assert "t1-cpu" in report["error"]
    assert report["pass"] is not True
    assert (tmp_path / "out" / "logs" / "t1-cpu.log").exists()


def test_verify_dump_backend_rejects_mislabeled_leg(tmp_path):
    dump = tmp_path / "t2-hip-0.jsonl"
    dump.write_text(_t2_header(backend="cpu") + "\n")  # hip leg, cpu-linked dump
    with pytest.raises(SystemExit, match="mislabeled leg binary"):
        verify_dump_backend(str(dump), "hip")


def test_verify_dump_backend_requires_offload_for_gpu_leg(tmp_path):
    """backend=hip with offload=false is a hip-linked binary that never asked
    for device execution — the T2 gate must fail it, not trust the label."""
    dump = tmp_path / "t2-hip-0.jsonl"
    dump.write_text(_t2_header(backend="hip") + "\n")  # _t2_header: offload False
    with pytest.raises(SystemExit, match="never asked for device execution"):
        verify_dump_backend(str(dump), "hip", require_offload=True)
    # Without the flag (T1 legs) the backend label alone decides.
    verify_dump_backend(str(dump), "hip")


def test_comparator_failure_error_is_scrubbed(tmp_path, monkeypatch):
    """Comparator stderr embeds absolute dump paths; the report error field
    must carry basenames only (path-scrub policy covers failures too)."""
    from parity_suite import driver
    monkeypatch.setattr(
        driver, "run_prompt",
        lambda args, timeout_s: (2, "",
                                 "guard: read /tmp/secret-dir/t2-cpu-0.jsonl: bad dump", 0.1))
    state = {"status": "running", "pass": None, "legs": {}}
    with pytest.raises(SystemExit):
        driver._compare_or_die(state, str(tmp_path), ["xtask"],
                               ["parity-compare", "t2", "--a", "a.jsonl"], 10)
    raw = (tmp_path / "report.json").read_text()
    assert "/tmp/secret-dir" not in raw, "comparator stderr leaked absolute paths"
    assert "t2-cpu-0.jsonl" in json.loads(raw)["error"]


def test_run_mode_leg_timeout_kills_group_and_writes_failed_report(tmp_path):
    """The riskiest subprocess path — per-leg timeout → process-group SIGKILL
    → rc=-9 — must surface as a `failed` report, not a stranded `running`."""
    # Stub names must be golem-<backend>: the stub derives its dump header's
    # backend label from its own file name.
    (tmp_path / "bins").mkdir(exist_ok=True)
    proc, report = _run_suite(
        tmp_path, ["--timeout", "1"],
        cpu=_write_stub(tmp_path / "bins" / "golem-cpu", SLEEP_GOLEM),
        hip=_write_stub(tmp_path / "bins" / "golem-hip", SLEEP_GOLEM))
    assert proc.returncode == 1, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert report["status"] == "failed"
    assert "t2-cpu-0" in report["error"] and "rc=-9" in report["error"]
    assert "(timed out)" in report["error"]
    assert report["pass"] is not True


def test_run_mode_rejects_hip_leg_that_never_offloads(tmp_path):
    (tmp_path / "bins").mkdir(exist_ok=True)
    proc, report = _run_suite(
        tmp_path, [], hip=_write_stub(tmp_path / "bins" / "golem-hip", HIP_NO_OFFLOAD_STUB))
    assert proc.returncode == 1
    assert report["status"] == "failed"
    assert "never asked for device execution" in report["error"]


def test_run_mode_rejects_non_positive_knobs_before_any_leg(tmp_path):
    for flag, bad in (("--n-predict", "0"), ("--gpu-layers", "0"),
                      ("--timeout", "0"), ("--cpu-moe", "-1")):
        proc, report = _run_suite(tmp_path, [flag, bad])
        assert proc.returncode == 1, flag
        assert report is None, f"{flag}: validation must precede the out dir"
        assert flag.lstrip("-") in proc.stderr and "FATAL" in proc.stderr, flag


def test_run_mode_rejects_skip_hip_with_hip_bin(tmp_path):
    bins = tmp_path / "bins"
    bins.mkdir()
    cpu = _write_stub(bins / "golem-cpu", STUB_GOLEM)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"stub model weights\n")
    proc = subprocess.run(
        [sys.executable, SCRIPT, "run", "--cpu-bin", cpu, "--hip-bin", cpu,
         "--model", str(model), "--out", str(tmp_path / "out"), "--skip-hip"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=SELFTEST_TIMEOUT_S)
    assert proc.returncode == 2
    assert "--skip-hip" in proc.stderr and "--hip-bin" in proc.stderr
    assert not (tmp_path / "out" / "report.json").exists()


def test_stub_rejects_driver_flag_typos(tmp_path):
    """The stub whitelists flags: a driver typo like --n-gpu-layer (missing
    's') must fail the leg instead of silently dropping the knob."""
    bins = tmp_path / "bins"
    bins.mkdir()
    stub = _write_stub(bins / "golem-cpu", STUB_GOLEM)
    model = tmp_path / "model.gguf"
    model.write_bytes(b"stub model weights\n")
    proc = subprocess.run(
        [sys.executable, stub, "run", "--greedy", str(model), "--prompt", "hi",
         "--parity-dump", str(tmp_path / "d.jsonl"), "--n-predict", "3",
         "--n-gpu-layer", "99"],
        capture_output=True, text=True, timeout=60)
    assert proc.returncode != 0
    assert "unexpected arg" in proc.stderr and "n-gpu-layer" in proc.stderr
