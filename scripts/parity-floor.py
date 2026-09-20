#!/usr/bin/env python3
"""Parity-floor harness: greedy text comparison between two llama-cli builds (HIP vs CPU).

Commands:
  run-leg  --bin <dir>/llama-cli --name <leg> [--hip] [--threads N] [--prompts N] [--n N]
           [--timeout S] [--out DIR]
  compare  <legA> <legB> [--out DIR]
  --selftest

Integrity rules (the harness must never manufacture a clean result):
  - a prompt's text is persisted only when the run exited 0 AND extraction is non-empty
  - empty extraction on rc=0 is a harness error: the leg aborts immediately
  - cached prompts skip only when the config sidecar matches exactly
  - compare refuses legs that are not ok=8/8 with identical -n and non-empty texts
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

GEN_RE = re.compile(r"Generation: ([0-9.]+) t/s\s*\]")
FA_WARN_RE = re.compile(r"^.*(flash.?att|fa ).*(not support|fallback|disabl|ignor).*$", re.IGNORECASE | re.MULTILINE)


def extract_text(raw: str) -> str:
    """Generation text = stdout with trailing whitespace stripped (never None)."""
    return raw.rstrip()


def extract_generation(raw: str, prompt: str) -> str:
    """Parse llama-cli stdout down to the generated text.

    stdout layout (with --simple-io --no-display-prompt): banner, 'Loading model...',
    a line starting '> ' (echoing the prompt), the generation, then the
    '[ Prompt: ... | Generation: ... t/s ]' stats line and 'Exiting...'.
    Returns '' when the layout is not recognized — callers treat that as a
    harness error, never as an (empty) generation.
    """
    lines = raw.splitlines()
    end = len(lines)
    for i, ln in enumerate(lines):
        if ln.startswith("[ Prompt:") or ln.startswith("Exiting..."):
            end = i
            break
    start = 0
    for i in range(end):
        if lines[i].startswith("> "):
            start = i + 1
            break
    else:
        return ""
    body = lines[start:end]
    if body and body[0].startswith(prompt):
        body[0] = body[0][len(prompt):]
    return "\n".join(body).strip("\n")


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


def leg_args(bin_path: str, model: str, prompt: str, hip: bool,
             threads: int | None, n_tokens: int) -> list[str]:
    args = [
        bin_path, "-m", model, "-st", "--temp", "0", "--seed", "42",
        "-n", str(n_tokens), "-b", "512", "-ub", "512", "--no-display-prompt",
        "--simple-io", "-fa", "on", "-p", prompt,
    ]
    if hip:
        args += ["-ngl", "99", "-ncmoe", "45", "-t", "16"]
    elif threads is not None:
        args += ["-t", str(threads)]
    return args


def config_sha(bin_path: str, model: str, hip: bool, threads: int | None, n_tokens: int) -> str:
    payload = json.dumps([bin_path, model, hip, threads, n_tokens])
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def atomic_write(path: str, content: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(content)
    os.replace(tmp, path)


def write_leg_json(legdir: str, meta: dict, results: list, ok: int) -> None:
    atomic_write(os.path.join(legdir, "leg.json"),
                 json.dumps({**meta, "ok": ok, "prompts": results}, indent=2))


def acquire_lock(outdir: str) -> None:
    """One model leg at a time on this machine (shared RAM/page cache/GPU)."""
    os.makedirs(outdir, exist_ok=True)
    lock = os.path.join(outdir, ".leg-lock")
    if os.path.exists(lock):
        other = open(lock).read().strip()
        sys.exit(f"FATAL: another leg holds {lock} (pid {other}). "
                 f"Verify no llama-cli is running, then remove the lock.")
    with open(lock, "w") as f:
        f.write(str(os.getpid()))


def release_lock(outdir: str) -> None:
    try:
        os.remove(os.path.join(outdir, ".leg-lock"))
    except FileNotFoundError:
        pass


def preflight(bin_path: str) -> None:
    r = subprocess.run(["pgrep", "-f", "llama-cli"], capture_output=True, text=True)
    if r.returncode == 0:
        sys.exit(f"FATAL: llama-cli already running (pid(s): {r.stdout.strip()}) — "
                 f"legs must run strictly sequentially.")
    if not os.path.exists(bin_path):
        sys.exit(f"FATAL: binary not found: {bin_path}")


def run_leg(bin_path: str, name: str, model: str, hip: bool, threads: int | None,
            only_idx: int | None, n_tokens: int, timeout_s: int, outdir: str) -> int:
    preflight(bin_path)
    acquire_lock(outdir)
    legdir = os.path.join(outdir, name)
    os.makedirs(legdir, exist_ok=True)
    sha = config_sha(bin_path, model, hip, threads, n_tokens)
    meta = {"leg": name, "bin": bin_path, "model": model, "hip": hip,
            "threads": threads, "n_tokens": n_tokens, "args_sha": sha,
            "wall_s_note": "informational only — page-cache warmth differs between legs"}
    try:
        idxs = [only_idx] if only_idx is not None else range(len(PROMPTS))
        results = []
        for idx in idxs:
            txt_path = os.path.join(legdir, f"prompt-{idx:02d}.txt")
            log_path = os.path.join(legdir, f"prompt-{idx:02d}.log")
            meta_path = os.path.join(legdir, f"prompt-{idx:02d}.meta.json")
            if os.path.exists(txt_path) and os.path.exists(meta_path):
                old = json.load(open(meta_path))
                if old.get("args_sha") == sha and old.get("rc") == 0:
                    results.append({"idx": idx, "rc": 0, "cached": True,
                                    "gen_s": old.get("gen_s"),
                                    "text": extract_text(open(txt_path).read())})
                    print(f"PARITY: leg={name} prompt={idx} cached (config match)", flush=True)
                    write_leg_json(legdir, meta, results, sum(1 for r in results if r["rc"] == 0))
                    continue
                print(f"PARITY: leg={name} prompt={idx} stale cache (config drift) — re-running", flush=True)
            args = leg_args(bin_path, model, PROMPTS[idx], hip, threads, n_tokens)
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout_s)
                rc, stdout, stderr = proc.returncode, proc.stdout, proc.stderr
            except subprocess.TimeoutExpired:
                rc, stdout, stderr = -9, "", f"TIMEOUT after {timeout_s}s"
            wall = time.perf_counter() - t0
            atomic_write(log_path, stderr)
            fa_warnings = FA_WARN_RE.findall(stderr)
            if fa_warnings:
                print(f"PARITY: leg={name} prompt={idx} FA-WARN {fa_warnings[:2]}", flush=True)
            if rc != 0:
                results.append({"idx": idx, "rc": rc, "gen_s": None,
                                "wall_s": round(wall, 1), "text": "", "fa_warnings": fa_warnings})
                print(f"PARITY: leg={name} prompt={idx} FAILED rc={rc}", flush=True)
                write_leg_json(legdir, meta, results, sum(1 for r in results if r["rc"] == 0))
                continue
            text = extract_generation(stdout, PROMPTS[idx])
            if not text:
                # Harness error, not an empty generation: refuse to record it.
                atomic_write(os.path.join(legdir, f"prompt-{idx:02d}.stdout.bad"), stdout)
                release_lock(outdir)
                sys.exit(f"FATAL: leg={name} prompt={idx} extraction produced no text on rc=0 "
                         f"(stdout saved to prompt-{idx:02d}.stdout.bad) — fix the harness, "
                         f"do not trust this leg.")
            m = GEN_RE.search(stderr)
            gen_s = round(n_tokens / float(m.group(1)), 1) if m and float(m.group(1)) > 0 else None
            if gen_s is None:
                print(f"PARITY: leg={name} prompt={idx} WARN no stats line matched (gen_s=None)", flush=True)
            atomic_write(txt_path, text)
            atomic_write(meta_path, json.dumps({"rc": 0, "args_sha": sha, "gen_s": gen_s}))
            results.append({"idx": idx, "rc": 0, "cached": False, "gen_s": gen_s,
                            "wall_s": round(wall, 1), "text": text, "fa_warnings": fa_warnings})
            print(f"PARITY: leg={name} prompt={idx} rc=0 gen_s={gen_s} "
                  f"wall_s={wall:.1f} words={len(text.split())}", flush=True)
            write_leg_json(legdir, meta, results, sum(1 for r in results if r["rc"] == 0))
        ok = sum(1 for r in results if r["rc"] == 0)
        print(f"PARITY-LEG: name={name} ok={ok}/{len(results)}")
        return 0 if ok == len(results) else 1
    finally:
        release_lock(outdir)


def compare(a: str, b: str, outdir: str) -> int:
    la = json.load(open(os.path.join(outdir, a, "leg.json")))
    lb = json.load(open(os.path.join(outdir, b, "leg.json")))
    if la.get("n_tokens") != lb.get("n_tokens"):
        sys.exit(f"FATAL: legs disagree on -n ({la.get('n_tokens')} vs {lb.get('n_tokens')})")
    ia = {r["idx"]: r for r in la["prompts"]}
    ib = {r["idx"]: r for r in lb["prompts"]}
    if sorted(ia) != list(range(len(PROMPTS))) or sorted(ib) != list(range(len(PROMPTS))):
        sys.exit(f"FATAL: incomplete legs (a has prompts {sorted(ia)}, b has {sorted(ib)}) — "
                 f"both legs must cover all {len(PROMPTS)} prompts")
    oka = sum(1 for r in la["prompts"] if r["rc"] == 0)
    okb = sum(1 for r in lb["prompts"] if r["rc"] == 0)
    if oka != len(PROMPTS) or okb != len(PROMPTS):
        sys.exit(f"FATAL: refuse to compare: a ok={oka}/{len(PROMPTS)}, b ok={okb}/{len(PROMPTS)}")
    ta = [ia[i]["text"] for i in range(len(PROMPTS))]
    tb = [ib[i]["text"] for i in range(len(PROMPTS))]
    empties = [i for i, (x, y) in enumerate(zip(ta, tb)) if not x or not y]
    if empties:
        sys.exit(f"FATAL: empty texts at prompt indices {empties} — legs are not trustworthy")
    mc = match_count(ta, tb)
    prefixes = [prefix_words(x, y) for x, y in zip(ta, tb)]
    report = {"a": a, "b": b, "n_tokens": la.get("n_tokens"), "prompts": len(PROMPTS),
              "match_count": mc,
              "match_rate": round(100.0 * mc / len(PROMPTS), 2),
              "prefix_words": prefixes}
    out = os.path.join(outdir, f"{a}_vs_{b}.json")
    atomic_write(out, json.dumps(report, indent=2) + "\n")
    print(f"PARITY: a={a} b={b} match={mc}/{len(PROMPTS)} prefixes={prefixes}")
    print(f"PARITY: wrote {out}")
    return 0


def selftest() -> int:
    assert extract_text("hello\n\n") == "hello"
    assert extract_text("") == ""
    # Fixture: real stdout layout captured from llama-cli (stories15M smoke, this machine);
    # banner bytes differ per model but the marker lines are the contract.
    real = (
        "\n\nLoading model... \n"
        "\nbuild      : b53-d773b57\nmodel      : m.gguf\n\n"
        "> Once upon a time\n"
        "Aweak in the air. Aweak of ice cream is a\n"
        "\n[ Prompt: 18912.5 t/s | Generation: 5114.2 t/s ]\n"
        "\n\nExiting...\n"
    )
    assert extract_generation(real, "Once upon a time") == \
        "Aweak in the air. Aweak of ice cream is a"
    # GLM-flavored layout: generation begins with the reasoning delimiter.
    glm = "> p\n[Start thinking]\nThe user wants\n\n[ Prompt: 1.0 t/s | Generation: 1.0 t/s ]\nExiting...\n"
    assert extract_generation(glm, "p") == "[Start thinking]\nThe user wants"
    # Extraction must never swallow a stats line into the text…
    assert "t/s" not in extract_generation(real, "Once upon a time")
    assert extract_generation("no marker here", "p") == ""
    assert extract_generation("> p\nalpha\n", "p") == "alpha"
    assert prefix_words("a b c", "a b d") == 2
    assert prefix_words("x", "x") is None
    assert prefix_words("", "a") == 0
    assert prefix_words("a b", "a b c") == 2
    assert match_count(["x", "y"], ["x", "z"]) == 1
    assert len(PROMPTS) == 8
    assert PROMPTS[0].startswith("Write a short story about a robot")
    # args builder: HIP legs carry placement + fa + fixed -t 16; CPU legs carry -t only
    hip = leg_args("BIN", "M", "P", True, None, 96)
    assert "-ncmoe" in hip and "45" in hip and hip[hip.index("-t") + 1] == "16"
    assert hip[hip.index("-n") + 1] == "96"
    cpu = leg_args("BIN", "M", "P", False, 32, 96)
    assert "-ncmoe" not in cpu and cpu[cpu.index("-t") + 1] == "32"
    for flag_set in (hip, cpu):
        assert "--temp" in flag_set and "0" in flag_set and "--simple-io" in flag_set
    # stats-line regex anchors to the composite line, not a bare substring
    assert GEN_RE.search("[ Prompt: 1.0 t/s | Generation: 12.5 t/s ]").group(1) == "12.5"
    assert GEN_RE.search("Generation: 12.5 t/s trailing junk") is None
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
    rl.add_argument("--n", type=int, default=96, help="tokens to generate per prompt")
    rl.add_argument("--timeout", type=int, default=3600, help="per-prompt timeout seconds")
    rl.add_argument("--out", default=os.environ.get("GOLEM_M05_WORK", ".m0-work/m05"))
    cp = sub.add_parser("compare")
    cp.add_argument("a")
    cp.add_argument("b")
    cp.add_argument("--out", default=os.environ.get("GOLEM_M05_WORK", ".m0-work/m05"))
    ns = ap.parse_args()
    if ns.selftest:
        return selftest()
    if ns.cmd == "run-leg":
        return run_leg(ns.bin, ns.name, ns.model, ns.hip, ns.threads,
                       ns.prompts, ns.n, ns.timeout, ns.out)
    if ns.cmd == "compare":
        return compare(ns.a, ns.b, ns.out)
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
