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
  - cached prompts skip only when the config sidecar matches exactly AND the
    recorded pin_sha equals the live vendored-tree hash AND the text hash verifies
  - compare refuses legs that are not ok=8/8 with identical -n and non-empty texts
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import fcntl
import signal
import subprocess
import sys
import time

DEFAULT_MODEL = os.environ.get(
    "GOLEM_M05_MODEL",
    "/home/eranelbaz/models/glm-5.3-flash-ud-q4k-xl/UD-Q4_K_XL/"
    "GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf",
)
VENDOR_DIR = "vendor/llama.cpp"
VENDOR_SHA_FILE = "vendor/VENDOR_SHA"
DEFAULT_OUTDIR = os.environ.get("GOLEM_M05_WORK", ".m0-work/m05")

PROMPTS = [  # all single-line: extract_generation strips the '> '-echo only when the
    # whole prompt sits on the first body line (multi-line prompts fail closed via
    # the empty-extraction FATAL in run_leg)
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
BUILD_RE = re.compile(r"^build\s*:\s*(\S+)", re.MULTILINE)
FA_WARN_RE = re.compile(r"^.*(flash[- ]?att|\bfa\b).*(not support|fallback|disabl|ignor).*$", re.IGNORECASE | re.MULTILINE)
HARNESS_VERSION = 4  # bump on any change to prompts, flag contract, or extraction logic


def pin_sha() -> str:
    """Content hash of the vendored substrate (ADR 0001 canonical recipe, python form).

    The llama.cpp banner build id is useless as provenance: with no .git inside
    vendor/, cmake walks up and reads the enclosing golem repo's HEAD — banners
    differ between build dirs of the identical pin. The tree hash is the only
    honest binary provenance.
    """
    paths = []
    for root, dirs, files in os.walk(VENDOR_DIR):
        dirs[:] = [d for d in dirs if d not in (".git", "build")]  # any depth
        paths += [os.path.relpath(os.path.join(root, f), VENDOR_DIR) for f in files]
    inner = hashlib.sha256()
    for p in sorted(paths, key=lambda s: s.encode()):
        with open(os.path.join(VENDOR_DIR, p), "rb") as f:
            inner.update(f"{hashlib.sha256(f.read()).hexdigest()}  ./{p}\n".encode())
    return inner.hexdigest()


def extract_text(raw: str) -> str:
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
    end = None
    for marker in ("[ Prompt:", "Exiting..."):
        last = max((k for k, ln in enumerate(lines) if ln.startswith(marker)), default=None)
        if last is not None:
            end = last if end is None else min(end, last)
    if end is None:
        return ""
    start = next((i + 1 for i in range(end) if lines[i].startswith("> ")), None)
    if start is None:
        return ""
    body = lines[start:end]
    if body and body[0].startswith(prompt):
        body[0] = body[0][len(prompt):]
    return "\n".join(body).strip("\n")


def prefix_words(a: str, b: str) -> int | None:
    """Common whitespace-word prefix length; None when texts are byte-equal."""
    if a == b:
        return None
    n = 0
    for x, y in zip(a.split(), b.split()):
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
    payload = json.dumps([HARNESS_VERSION, bin_path, model, hip, threads, n_tokens, PROMPTS])
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def atomic_write(path: str, content: str) -> None:
    tmp = f"{path}.tmp.{os.getpid()}"  # pid-suffixed: compare runs outside the leg lock
    with open(tmp, "w") as f:
        f.write(content)
    os.replace(tmp, path)


def prompt_paths(legdir: str, idx: int) -> dict[str, str]:
    base = os.path.join(legdir, f"prompt-{idx:02d}")
    return {"txt": base + ".txt", "log": base + ".log",
            "meta": base + ".meta.json", "stdout_bad": base + ".stdout.bad"}


def write_leg_json(legdir: str, meta: dict, results: list) -> None:
    # rewritten after every prompt: intentional crash-resume durability, cheap at n=8
    ok = sum(1 for r in results if r["rc"] == 0)
    atomic_write(os.path.join(legdir, "leg.json"),
                 json.dumps({**meta, "ok": ok, "prompts": results}, indent=2))


def acquire_lock(outdir: str) -> int:
    """One model leg at a time (shared RAM/page cache/GPU).

    flock advisory lock: the kernel releases it when the holder dies
    (SIGKILL/OOM), so stale reclaim is automatic and race-free — unlike
    O_EXCL create + liveness check, which has a reclaim TOCTOU window.
    Returns the fd (keep it open for the leg's lifetime).
    """
    os.makedirs(outdir, exist_ok=True)
    lock = os.path.join(outdir, ".leg-lock")
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        holder = ""
        try:
            holder = " (pid " + os.read(fd, 64).decode(errors="replace").strip() + ")"
        except OSError:
            pass
        os.close(fd)
        sys.exit(f"FATAL: another leg holds {lock}{holder}.")
    os.truncate(fd, 0)
    os.pwrite(fd, str(os.getpid()).encode(), 0)  # diagnostics only
    return fd


def release_lock(fd: int) -> None:
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def _live_model_pids(bin_path: str) -> list[str]:
    live = []
    targets = {os.path.realpath(bin_path)}
    r = subprocess.run(["pgrep", "-x", "llama-cli"], capture_output=True, text=True)
    for pid in r.stdout.split():
        try:
            with open(f"/proc/{pid}/stat", "rb") as f:
                # stat: pid (comm) state ... — comm can contain spaces/parens,
                # so anchor on the last ')'.
                state = f.read().rsplit(b")", 1)[1].split()[0]
            if state == b"Z":
                # pgrep -x also matches zombies (a defunct child systemd --user
                # never reaped), which would block legs forever. Treat as dead.
                continue
            try:
                exe = os.path.realpath(f"/proc/{pid}/exe")
                if exe not in targets:
                    continue
            except OSError:
                pass  # exe unreadable — fall back to the name match
        except OSError:
            continue
        live.append(pid)
    return live


def preflight(bin_path: str) -> None:
    if not os.path.exists(bin_path):
        sys.exit(f"FATAL: binary not found: {bin_path}")
    if not os.access(bin_path, os.X_OK):
        sys.exit(f"FATAL: binary not executable: {bin_path}")
    live = _live_model_pids(bin_path)
    if live:
        sys.exit(f"FATAL: llama-cli already running (pid(s): {', '.join(live)}) — "
                 f"legs must run strictly sequentially.")


def cached_result(legdir: str, idx: int, sha: str, psha: str) -> dict | None:
    """Previous result for this prompt when config, pin, and text hash all verify."""
    paths = prompt_paths(legdir, idx)
    if not (os.path.exists(paths["txt"]) and os.path.exists(paths["meta"])):
        return None
    with open(paths["meta"]) as f:
        old = json.load(f)
    if old.get("args_sha") != sha or old.get("rc") != 0:
        return None
    if old.get("pin_sha") != psha:
        print(f"PARITY: leg cache prompt={idx} rejected — pin drift "
              f"({str(old.get('pin_sha'))[:12]} != {psha[:12]})", flush=True)
        return None
    with open(paths["txt"]) as f:
        txt = extract_text(f.read())
    if not txt:
        return None
    recorded = old.get("text_sha")
    if recorded and hashlib.sha256(txt.encode()).hexdigest() != recorded:
        print(f"PARITY: leg cache prompt={idx} rejected — text hash mismatch (tamper/partial write)", flush=True)
        return None
    return {"idx": idx, "rc": 0, "cached": True, "gen_s": old.get("gen_s"),
            "pin_sha": old.get("pin_sha"), "build_id": old.get("build_id"),
            "wall_s": None, "text": txt, "fa_warnings": old.get("fa_warnings", [])}


def run_prompt(args: list[str], timeout_s: int) -> tuple[int, str, str, float]:
    """Run one llama-cli invocation; returns (rc, stdout, stderr, wall_s).

    Own process group: on timeout, kill the whole tree so an orphaned child
    cannot hold the stdout pipe open and stall communicate().
    """
    t0 = time.perf_counter()
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        try:
            stdout, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            # a grandchild that escaped the process group still holds the pipe;
            # abandon the pipes rather than hang forever
            proc.kill()
            stdout, stderr = "", ""
        stderr = (stderr or "") + f"\nTIMEOUT after {timeout_s}s (partial output preserved)"
        rc = -9
    return rc, stdout, stderr, time.perf_counter() - t0


def parse_run_output(combined: str, stderr: str) -> tuple[str | None, list[str]]:
    m = BUILD_RE.search(combined)
    build_id = m.group(1) if m else None
    return build_id, FA_WARN_RE.findall(stderr)


def run_leg(bin_path: str, name: str, model: str, hip: bool, threads: int | None,
            only_idx: int | None, n_tokens: int, timeout_s: int, outdir: str) -> int:
    if only_idx is not None and not 0 <= only_idx < len(PROMPTS):
        sys.exit(f"FATAL: --prompts must be 0..{len(PROMPTS) - 1}, got {only_idx}")
    preflight(bin_path)
    lock_fd = acquire_lock(outdir)
    try:
        legdir = os.path.join(outdir, name)
        os.makedirs(legdir, exist_ok=True)
        sha = config_sha(bin_path, model, hip, threads, n_tokens)
        psha = pin_sha()
        try:
            with open(VENDOR_SHA_FILE) as f:
                first = f.readline()
            want = first.split("=", 1)[1].strip()
        except (OSError, IndexError):
            sys.exit(f"FATAL: cannot read/parse {VENDOR_SHA_FILE} (expected '<name>=<sha256>')")
        if psha != want:
            sys.exit(f"FATAL: vendor tree drifted ({psha[:12]} != {want[:12]}) — re-vendor per ADR 0001")
        meta = {"leg": name, "bin": bin_path, "model": model, "hip": hip,
                "threads": threads, "n_tokens": n_tokens, "args_sha": sha, "pin_sha": psha,
                "harness_version": HARNESS_VERSION,
                "wall_s_note": "informational only — page-cache warmth differs between legs"}
        idxs = [only_idx] if only_idx is not None else range(len(PROMPTS))
        results: list = []
        if only_idx is not None:
            prev = os.path.join(legdir, "leg.json")
            if os.path.exists(prev):
                with open(prev) as f:
                    prev_rec = json.load(f)
                if prev_rec.get("args_sha") != sha:
                    sys.exit(f"FATAL: --prompts N resume of {prev} with a different config "
                             f"({sha} vs recorded {prev_rec.get('args_sha')}) — use a fresh --name.")
                if len(prev_rec.get("prompts", [])) == len(PROMPTS):
                    sys.exit(f"FATAL: --prompts N would clobber the complete leg record {prev} — "
                             f"use a fresh --name or delete the leg dir deliberately.")
                # resume: keep existing prompt records, replace only this idx
                results = [r for r in prev_rec.get("prompts", []) if r.get("idx") != only_idx]
        for idx in idxs:
            paths = prompt_paths(legdir, idx)
            hit = cached_result(legdir, idx, sha, psha)
            if hit:
                results.append(hit)
                print(f"PARITY: leg={name} prompt={idx} cached (config match)", flush=True)
                write_leg_json(legdir, meta, results)
                continue
            rc, stdout, stderr, wall = run_prompt(
                leg_args(bin_path, model, PROMPTS[idx], hip, threads, n_tokens), timeout_s)
            atomic_write(paths["log"], stderr)
            combined = stdout + "\n" + stderr
            build_id, fa_warnings = parse_run_output(combined, stderr)
            if fa_warnings:
                print(f"PARITY: leg={name} prompt={idx} FA-WARN {fa_warnings[:2]}", flush=True)
            if rc != 0:
                results.append({"idx": idx, "rc": rc, "gen_s": None,
                                "wall_s": round(wall, 1), "text": "", "fa_warnings": fa_warnings})
                print(f"PARITY: leg={name} prompt={idx} FAILED rc={rc}", flush=True)
                write_leg_json(legdir, meta, results)
                continue
            text = extract_generation(stdout, PROMPTS[idx])
            if not text:
                atomic_write(paths["stdout_bad"], stdout)
                sys.exit(f"FATAL: leg={name} prompt={idx} extraction produced no text on rc=0 "
                         f"(stdout saved to {os.path.basename(paths['stdout_bad'])}) — "
                         f"fix the harness, do not trust this leg.")
            m = GEN_RE.search(combined)
            gen_s = round(n_tokens / float(m.group(1)), 1) if m and float(m.group(1)) > 0 else None
            if gen_s is None:
                print(f"PARITY: leg={name} prompt={idx} WARN no stats line matched (gen_s=None)", flush=True)
            if "\ufffd" in text:
                print(f"PARITY: leg={name} prompt={idx} WARN U+FFFD in text (token-boundary truncation)", flush=True)
            atomic_write(paths["txt"], text)
            atomic_write(paths["meta"], json.dumps({"rc": 0, "args_sha": sha, "gen_s": gen_s,
                                                    "pin_sha": psha, "build_id": build_id,
                                                    "text_sha": hashlib.sha256(text.encode()).hexdigest(),
                                                    "fa_warnings": fa_warnings}))
            results.append({"idx": idx, "rc": 0, "cached": False, "gen_s": gen_s, "pin_sha": psha,
                            "build_id": build_id, "wall_s": round(wall, 1), "text": text,
                            "fa_warnings": fa_warnings})
            print(f"PARITY: leg={name} prompt={idx} rc=0 gen_s={gen_s} "
                  f"wall_s={wall:.1f} words={len(text.split())}", flush=True)
            write_leg_json(legdir, meta, results)
        ok = sum(1 for r in results if r["rc"] == 0)
        print(f"PARITY-LEG: name={name} ok={ok}/{len(results)}")
        return 0 if ok == len(results) else 1
    finally:
        release_lock(lock_fd)


def load_leg(outdir: str, leg: str) -> dict:
    path = os.path.join(outdir, leg, "leg.json")
    if not os.path.exists(path):
        sys.exit(f"FATAL: no leg record for '{leg}' under {outdir} — run-leg first")
    with open(path) as f:
        return json.load(f)


def compare(a: str, b: str, outdir: str) -> int:
    la, lb = load_leg(outdir, a), load_leg(outdir, b)
    if la.get("n_tokens") != lb.get("n_tokens"):
        sys.exit(f"FATAL: legs disagree on -n ({la.get('n_tokens')} vs {lb.get('n_tokens')})")
    if la.get("model") != lb.get("model"):
        sys.exit("FATAL: legs ran different models")
    if la.get("harness_version") != lb.get("harness_version"):
        sys.exit(f"FATAL: legs produced by different harness versions "
                 f"({la.get('harness_version')} vs {lb.get('harness_version')})")
    ia = {r["idx"]: r for r in la["prompts"]}
    ib = {r["idx"]: r for r in lb["prompts"]}
    if sorted(ia) != list(range(len(PROMPTS))) or sorted(ib) != list(range(len(PROMPTS))):
        sys.exit(f"FATAL: incomplete legs (a has prompts {sorted(ia)}, b has {sorted(ib)}) — "
                 f"both legs must cover all {len(PROMPTS)} prompts")
    oka = sum(1 for r in la["prompts"] if r["rc"] == 0)
    okb = sum(1 for r in lb["prompts"] if r["rc"] == 0)
    if oka != len(PROMPTS) or okb != len(PROMPTS):
        sys.exit(f"FATAL: refuse to compare: a ok={oka}/{len(PROMPTS)}, b ok={okb}/{len(PROMPTS)}")
    if la.get("hip") == lb.get("hip") and la.get("threads") == lb.get("threads"):
        sys.exit(f"FATAL: identical backend config on both legs (hip={la.get('hip')}, "
                 f"threads={la.get('threads')}) — that is not a parity comparison")
    pins_a = {r.get("pin_sha") for r in la["prompts"]}
    pins_b = {r.get("pin_sha") for r in lb["prompts"]}
    # Banner build ids are recorded but NOT compared — they track the enclosing
    # repo HEAD, not the pin (see pin_sha).
    if len(pins_a) != 1 or len(pins_b) != 1 or pins_a != pins_b or None in (pins_a | pins_b):
        sys.exit(f"FATAL: pin identity missing or mismatched (a={pins_a}, b={pins_b}) — "
                 f"both legs must run binaries built from the same pinned tree")
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
    glm = "> p\n[Start thinking]\nThe user wants\n\n[ Prompt: 1.0 t/s | Generation: 1.0 t/s ]\nExiting...\n"
    assert extract_generation(glm, "p") == "[Start thinking]\nThe user wants"
    # end anchor = LAST marker occurrence, so a generation containing literal
    # marker text must not truncate the extraction
    tricky = "> p\nabout Exiting... and [ Prompt: fake lines\nreal content\n\n[ Prompt: 1.0 t/s | Generation: 2.0 t/s ]\nExiting...\n"
    assert "real content" in extract_generation(tricky, "p")
    assert "t/s" not in extract_generation(real, "Once upon a time")
    assert extract_generation("no marker here", "p") == ""
    assert extract_generation("> p\nalpha\n", "p") == ""
    assert GEN_RE.search(extract_generation(real, "Once upon a time") + "\n" + real).group(1) == "5114.2"
    assert BUILD_RE.search(real).group(1) == "b53-d773b57"
    assert GEN_RE.search("Generation: 12.5 t/s trailing junk") is None
    assert prefix_words("a b c", "a b d") == 2
    assert prefix_words("x", "x") is None
    assert prefix_words("", "a") == 0
    assert prefix_words("a b", "a b c") == 2
    assert match_count(["x", "y"], ["x", "z"]) == 1
    assert len(PROMPTS) == 8
    assert PROMPTS[0].startswith("Write a short story about a robot")
    hip = leg_args("BIN", "M", "P", True, None, 96)
    assert "-ncmoe" in hip and "45" in hip and hip[hip.index("-t") + 1] == "16"
    assert hip[hip.index("-n") + 1] == "96"
    cpu = leg_args("BIN", "M", "P", False, 32, 96)
    assert "-ncmoe" not in cpu and cpu[cpu.index("-t") + 1] == "32"
    for flag_set in (hip, cpu):
        assert "--temp" in flag_set and "0" in flag_set and "--simple-io" in flag_set
    assert GEN_RE.search("[ Prompt: 1.0 t/s | Generation: 12.5 t/s ]").group(1) == "12.5"
    assert GEN_RE.search("Generation: 12.5 t/s trailing junk") is None
    # cached_result: config match, pin drift, and text-hash tamper detection
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        txt = "generated words"
        good = {"rc": 0, "args_sha": "AAA", "pin_sha": "PIN", "gen_s": 1.0,
                "build_id": "b", "text_sha": hashlib.sha256(txt.encode()).hexdigest(),
                "fa_warnings": []}
        base = os.path.join(td, "leg")
        os.makedirs(base)
        p = prompt_paths(base, 0)
        atomic_write(p["txt"], txt)
        atomic_write(p["meta"], json.dumps(good))
        hit = cached_result(base, 0, "AAA", "PIN")
        assert hit is not None and hit["text"] == txt and hit["wall_s"] is None
        assert cached_result(base, 0, "BBB", "PIN") is None          # config mismatch
        assert cached_result(base, 0, "AAA", "OTHER") is None        # pin drift
        tampered = dict(good, text_sha="0" * 64)
        atomic_write(p["meta"], json.dumps(tampered))
        assert cached_result(base, 0, "AAA", "PIN") is None          # hash mismatch
    # write_leg_json computes ok from rc
    with tempfile.TemporaryDirectory() as td:
        write_leg_json(td, {"leg": "x"}, [{"rc": 0}, {"rc": -9}])
        with open(os.path.join(td, "leg.json")) as f:
            assert json.load(f)["ok"] == 1
    # FA_WARN_RE matches fallback lines
    assert FA_WARN_RE.search("ggml_flash_attn_ext: flash attention not supported, using fallback")
    assert not FA_WARN_RE.search("all good, nothing suspicious here")
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
    rl.add_argument("--out", default=DEFAULT_OUTDIR)
    cp = sub.add_parser("compare")
    cp.add_argument("a")
    cp.add_argument("b")
    cp.add_argument("--out", default=DEFAULT_OUTDIR)
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
