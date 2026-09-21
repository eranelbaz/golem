"""Prompt execution: subprocess lifecycle, cache validation, leg orchestration."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time

from .config import GEN_RE, HARNESS_VERSION, PROMPTS, VENDOR_SHA_FILE, config_sha, leg_args, pin_sha
from .extract import extract_generation, extract_text, parse_run_output
from .files import atomic_write, prompt_paths, write_leg_json
from .locking import preflight, acquire_lock, release_lock


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
            proc.wait(timeout=5)
            stdout, stderr = "", ""
        stderr = (stderr or "") + f"\nTIMEOUT after {timeout_s}s (partial output preserved)"
        rc = -9
    return rc, stdout, stderr, time.perf_counter() - t0


def cached_result(legdir: str, idx: int, sha: str, psha: str) -> dict | None:
    """Previous result for this prompt when config, pin, and text hash all verify."""
    paths = prompt_paths(legdir, idx)
    if not (os.path.exists(paths["txt"]) and os.path.exists(paths["meta"])):
        return None
    with open(paths["meta"]) as f:
        try:
            old = json.load(f)
        except json.JSONDecodeError:
            print(f"PARITY: leg cache prompt={idx} rejected — corrupt meta", flush=True)
            return None
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
    if not recorded or hashlib.sha256(txt.encode()).hexdigest() != recorded:
        print(f"PARITY: leg cache prompt={idx} rejected — text hash missing or mismatched (tamper/partial write)", flush=True)
        return None
    return {"idx": idx, "rc": 0, "cached": True, "gen_s": old.get("gen_s"),
            "pin_sha": old.get("pin_sha"), "build_id": old.get("build_id"),
            "wall_s": None, "text": txt, "fa_warnings": old.get("fa_warnings", [])}


def run_leg(bin_path: str, name: str, model: str, hip: bool, threads: int | None,
            only_idx: int | None, n_tokens: int, timeout_s: int, outdir: str) -> int:
    if only_idx is not None and not 0 <= only_idx < len(PROMPTS):
        sys.exit(f"FATAL: --prompts must be 0..{len(PROMPTS) - 1}, got {only_idx}")
    lock_fd = acquire_lock(outdir)
    try:
        preflight(bin_path)
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
