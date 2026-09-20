#!/usr/bin/env python3
"""Parity-floor harness: greedy text comparison between two llama-cli builds (HIP vs CPU).

Commands:
  run-leg  --bin <dir>/llama-cli --name <leg> [--hip] [--threads N] [--prompts N] [--out DIR]
  compare  <legA> <legB> [--out DIR]
  --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

DEFAULT_MODEL = (
    "/home/eranelbaz/models/glm-5.3-flash-ud-q4k-xl/UD-Q4_K_XL/"
    "GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf"
)

PROMPTS = [
    "Write a short story about a robot who learns to paint.",
    "Summarize the plot of Romeo and Juliet in three sentences.",
    "Explain how photosynthesis works to a ten-year-old.",
    "List five healthy breakfast ideas.",
    "Translate 'the weather is nice today' into French.",
    "Write a haiku about autumn rain.",
    "What is the capital of Australia? Answer briefly.",
    "Describe the water cycle step by step.",
]

GEN_RE = re.compile(r"Generation: ([0-9.]+) t/s")


def extract_text(raw: str) -> str:
    """Generation text = stdout with trailing whitespace stripped (never None)."""
    return raw.rstrip()


def prefix_words(a: str, b: str) -> int | None:
    """Common whitespace-word prefix length; None when texts are byte-equal."""
    if a == b:
        return None
    wa, wb = a.split(), b.split()
    n = 0
    for x, y in zip(wa, wb):
        if x != y:
            break
        n += 1
    return n


def match_count(texts_a: list[str], texts_b: list[str]) -> int:
    """Count of byte-equal prompt pairs."""
    return sum(1 for a, b in zip(texts_a, texts_b) if a == b)


def leg_args(bin_path: str, model: str, prompt: str, hip: bool, threads: int | None) -> list[str]:
    args = [
        bin_path, "-m", model, "-st", "--temp", "0", "--seed", "42",
        "-n", "64", "-b", "512", "-ub", "512", "--no-display-prompt",
        "-fa", "on", "-p", prompt,
    ]
    if hip:
        args += ["-ngl", "99", "-ncmoe", "45", "-t", "16"]
    elif threads is not None:
        args += ["-t", str(threads)]
    return args


def run_leg(bin_path: str, name: str, model: str, hip: bool, threads: int | None,
            only_idx: int | None, outdir: str) -> int:
    legdir = os.path.join(outdir, name)
    os.makedirs(legdir, exist_ok=True)
    idxs = [only_idx] if only_idx is not None else range(len(PROMPTS))
    results = []
    for idx in idxs:
        txt_path = os.path.join(legdir, f"prompt-{idx:02d}.txt")
        log_path = os.path.join(legdir, f"prompt-{idx:02d}.log")
        if os.path.exists(txt_path):
            results.append({"idx": idx, "rc": 0, "gen_s": None, "text": extract_text(open(txt_path).read())})
            print(f"PARITY: leg={name} prompt={idx} cached")
            continue
        args = leg_args(bin_path, model, PROMPTS[idx], hip, threads)
        t0 = time.perf_counter()
        proc = subprocess.run(args, capture_output=True, text=True)
        wall = time.perf_counter() - t0
        with open(txt_path, "w") as f:
            f.write(proc.stdout)
        with open(log_path, "w") as f:
            f.write(proc.stderr)
        m = GEN_RE.search(proc.stderr)
        gen_s = round(64 / float(m.group(1)), 1) if m and float(m.group(1)) > 0 else None
        results.append({"idx": idx, "rc": proc.returncode, "gen_s": gen_s,
                        "wall_s": round(wall, 1), "text": extract_text(proc.stdout)})
        print(f"PARITY: leg={name} prompt={idx} rc={proc.returncode} "
              f"gen_s={gen_s} wall_s={wall:.1f}", flush=True)
    ok = sum(1 for r in results if r["rc"] == 0)
    with open(os.path.join(legdir, "leg.json"), "w") as f:
        json.dump({"leg": name, "bin": bin_path,
                   "args_sha": hashlib.sha256(json.dumps(leg_args(bin_path, model, "", hip, threads)).encode()).hexdigest()[:12],
                   "ok": ok, "prompts": results}, f, indent=2)
    print(f"PARITY-LEG: name={name} ok={ok}/{len(results)}")
    return 0 if ok == len(results) else 1


def compare(a: str, b: str, outdir: str) -> int:
    la = json.load(open(os.path.join(outdir, a, "leg.json")))
    lb = json.load(open(os.path.join(outdir, b, "leg.json")))
    ia = {r["idx"]: r for r in la["prompts"]}
    ib = {r["idx"]: r for r in lb["prompts"]}
    idxs = sorted(set(ia) & set(ib))
    ta = [ia[i]["text"] for i in idxs]
    tb = [ib[i]["text"] for i in idxs]
    mc = match_count(ta, tb)
    prefixes = [prefix_words(x, y) for x, y in zip(ta, tb)]
    report = {"a": a, "b": b, "prompts": len(idxs), "match_count": mc,
              "match_rate": round(100.0 * mc / len(idxs), 2) if idxs else 0.0,
              "prefix_words": prefixes}
    out = os.path.join(outdir, f"{a}_vs_{b}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(f"PARITY: a={a} b={b} match={mc}/{len(idxs)} prefixes={prefixes}")
    print(f"PARITY: wrote {out}")
    return 0


def selftest() -> int:
    assert extract_text("hello\n\n") == "hello"
    assert extract_text("") == ""
    assert prefix_words("a b c", "a b d") == 2
    assert prefix_words("x", "x") is None
    assert prefix_words("", "a") == 0
    assert prefix_words("a b", "a b c") == 2
    assert match_count(["x", "y"], ["x", "z"]) == 1
    assert len(PROMPTS) == 8
    assert PROMPTS[0].startswith("Write a short story about a robot")
    # args builder: HIP legs carry placement + fa; CPU legs carry -t only
    hip = leg_args("BIN", "M", "P", True, None)
    assert "-ncmoe" in hip and "45" in hip and hip[hip.index("-t") + 1] == "16"
    cpu = leg_args("BIN", "M", "P", False, 32)
    assert "-ncmoe" not in cpu and cpu[cpu.index("-t") + 1] == "32"
    for flag_set in (hip, cpu):
        assert "--temp" in flag_set and "0" in flag_set and "--no-display-prompt" in flag_set
    print("SELFTEST-OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selftest", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    rl = sub.add_parser("run-leg")
    rl.add_argument("--bin", required=True)
    rl.add_argument("--name", required=True)
    rl.add_argument("--model", default=DEFAULT_MODEL)
    rl.add_argument("--hip", action="store_true")
    rl.add_argument("--threads", type=int, default=None)
    rl.add_argument("--prompts", type=int, default=None, help="run only prompt index N")
    rl.add_argument("--out", default=os.environ.get("GOLEM_M05_WORK", ".m0-work/m05"))
    cp = sub.add_parser("compare")
    cp.add_argument("a")
    cp.add_argument("b")
    cp.add_argument("--out", default=os.environ.get("GOLEM_M05_WORK", ".m0-work/m05"))
    ns = ap.parse_args()
    if ns.selftest:
        return selftest()
    if ns.cmd == "run-leg":
        return run_leg(ns.bin, ns.name, ns.model, ns.hip, ns.threads, ns.prompts, ns.out)
    if ns.cmd == "compare":
        return compare(ns.a, ns.b, ns.out)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
