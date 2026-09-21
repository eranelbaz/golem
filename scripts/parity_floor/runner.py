"""Prompt execution: subprocess lifecycle, cache validation, leg orchestration."""
from __future__ import annotations

import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass

from .config import GEN_RE, HARNESS_VERSION, PROMPTS, VENDOR_SHA_FILE, config_sha, leg_args, pin_sha
from .extract import extract_generation, extract_text, parse_run_output
from .files import atomic_write, prompt_paths, write_leg_json
from .locking import preflight, acquire_lock, release_lock


@dataclass
class LegContext:
    """Everything one leg run needs, fixed for its whole lifetime."""
    bin_path: str
    model: str
    hip: bool
    threads: int | None
    n_tokens: int
    timeout_s: int
    outdir: str
    name: str

    @property
    def leg_dir(self) -> str:
        return os.path.join(self.outdir, self.name)

    def config_hash(self) -> str:
        return config_sha(self.bin_path, self.model, self.hip, self.threads, self.n_tokens)


def run_prompt(args: list[str], timeout_s: int) -> tuple[int, str, str, float]:
    """Run one llama-cli invocation; returns (rc, stdout, stderr, wall_s)."""
    started = time.perf_counter()
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
        return_code = proc.returncode
    except subprocess.TimeoutExpired:
        return_code, stdout, stderr = _kill_process_group(proc, timeout_s)
    return return_code, stdout, stderr, time.perf_counter() - started


def _kill_process_group(proc: subprocess.Popen, timeout_s: int) -> tuple[int, str, str]:
    os.killpg(proc.pid, signal.SIGKILL)
    try:
        stdout, stderr = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        # a grandchild outside the process group still holds the pipes: abandon them
        proc.kill()
        proc.wait(timeout=5)
        stdout, stderr = "", ""
    stderr = (stderr or "") + f"\nTIMEOUT after {timeout_s}s (partial output preserved)"
    return -9, stdout, stderr


def cached_result(leg_dir: str, prompt_idx: int, config_hash: str, tree_hash: str) -> dict | None:
    """Previous result for this prompt, or None unless every check passes."""
    paths = prompt_paths(leg_dir, prompt_idx)
    if not (os.path.exists(paths["txt"]) and os.path.exists(paths["meta"])):
        return None

    def reject(reason: str) -> None:
        print(f"PARITY: leg cache prompt={prompt_idx} rejected — {reason}", flush=True)

    try:
        with open(paths["meta"]) as meta_file:
            sidecar = json.load(meta_file)
    except json.JSONDecodeError:
        reject("corrupt meta")
        return None
    if sidecar.get("args_sha") != config_hash or sidecar.get("rc") != 0:
        return None
    if sidecar.get("pin_sha") != tree_hash:
        reject(f"pin drift ({sidecar.get('pin_sha')} != {tree_hash})")
        return None
    with open(paths["txt"]) as text_file:
        text = extract_text(text_file.read())
    if not text:
        return None
    if sidecar.get("text_sha") != hashlib.sha256(text.encode()).hexdigest():
        reject("text hash missing or mismatched (tamper/partial write)")
        return None
    return {"idx": prompt_idx, "rc": 0, "cached": True, "gen_s": sidecar.get("gen_s"),
            "pin_sha": sidecar.get("pin_sha"), "build_id": sidecar.get("build_id"),
            "wall_s": None, "text": text, "fa_warnings": sidecar.get("fa_warnings", [])}


def _check_vendor_pin(tree_hash: str) -> None:
    try:
        with open(VENDOR_SHA_FILE) as pin_file:
            pinned_hash = pin_file.readline().split("=", 1)[1].strip()
    except (OSError, IndexError):
        sys.exit(f"FATAL: cannot read/parse {VENDOR_SHA_FILE} (expected '<name>=<sha256>')")
    if tree_hash != pinned_hash:
        sys.exit(f"FATAL: vendor tree drifted ({tree_hash[:12]} != {pinned_hash[:12]}) — "
                 f"re-vendor per ADR 0001")


def _load_resumable_results(leg_dir: str, config_hash: str, prompt_idx: int | None) -> list:
    """Existing prompt records from an interrupted leg, minus the one being re-run."""
    if prompt_idx is None:
        return []
    leg_record_path = os.path.join(leg_dir, "leg.json")
    if not os.path.exists(leg_record_path):
        return []
    with open(leg_record_path) as record_file:
        leg_record = json.load(record_file)
    if leg_record.get("args_sha") != config_hash:
        sys.exit(f"FATAL: --prompts N resume of {leg_record_path} with a different config "
                 f"({config_hash} vs recorded {leg_record.get('args_sha')}) — use a fresh --name.")
    if len(leg_record.get("prompts", [])) == len(PROMPTS):
        sys.exit(f"FATAL: --prompts N would clobber the complete leg record {leg_record_path} — "
                 f"use a fresh --name or delete the leg dir deliberately.")
    return [record for record in leg_record.get("prompts", []) if record.get("idx") != prompt_idx]


def _record_prompt(ctx: LegContext, leg_meta: dict, results: list,
                   prompt_idx: int, config_hash: str, tree_hash: str) -> None:
    """Run (or cache-load) one prompt and append its record to `results`."""
    prompt = PROMPTS[prompt_idx]
    paths = prompt_paths(ctx.leg_dir, prompt_idx)

    cached = cached_result(ctx.leg_dir, prompt_idx, config_hash, tree_hash)
    if cached:
        results.append(cached)
        print(f"PARITY: leg={ctx.name} prompt={prompt_idx} cached (config match)", flush=True)
        write_leg_json(ctx.leg_dir, leg_meta, results)
        return

    return_code, stdout, stderr, wall_s = run_prompt(
        leg_args(ctx.bin_path, ctx.model, prompt, ctx.hip, ctx.threads, ctx.n_tokens),
        ctx.timeout_s)
    atomic_write(paths["log"], stderr)
    build_id, fa_warnings = parse_run_output(stdout + "\n" + stderr, stderr)
    if fa_warnings:
        print(f"PARITY: leg={ctx.name} prompt={prompt_idx} FA-WARN {fa_warnings[:2]}", flush=True)
    if return_code != 0:
        results.append({"idx": prompt_idx, "rc": return_code, "gen_s": None,
                        "wall_s": round(wall_s, 1), "text": "", "fa_warnings": fa_warnings})
        print(f"PARITY: leg={ctx.name} prompt={prompt_idx} FAILED rc={return_code}", flush=True)
        write_leg_json(ctx.leg_dir, leg_meta, results)
        return

    text = _extract_or_die(stdout, prompt, prompt_idx, paths, ctx.name)
    gen_s = _generation_speed(stdout + "\n" + stderr, ctx.n_tokens, prompt_idx, ctx.name)
    if "\ufffd" in text:
        print(f"PARITY: leg={ctx.name} prompt={prompt_idx} WARN U+FFFD in text (token-boundary truncation)", flush=True)

    atomic_write(paths["txt"], text)
    atomic_write(paths["meta"], json.dumps({"rc": 0, "args_sha": config_hash, "gen_s": gen_s,
                                            "pin_sha": tree_hash, "build_id": build_id,
                                            "text_sha": hashlib.sha256(text.encode()).hexdigest(),
                                            "fa_warnings": fa_warnings}))
    results.append({"idx": prompt_idx, "rc": 0, "cached": False, "gen_s": gen_s,
                    "pin_sha": tree_hash, "build_id": build_id, "wall_s": round(wall_s, 1),
                    "text": text, "fa_warnings": fa_warnings})
    print(f"PARITY: leg={ctx.name} prompt={prompt_idx} rc=0 gen_s={gen_s} "
          f"wall_s={wall_s:.1f} words={len(text.split())}", flush=True)
    write_leg_json(ctx.leg_dir, leg_meta, results)


def _extract_or_die(stdout: str, prompt: str, prompt_idx: int, paths: dict, leg_name: str) -> str:
    text = extract_generation(stdout, prompt)
    if not text:
        atomic_write(paths["stdout_bad"], stdout)
        sys.exit(f"FATAL: leg={leg_name} prompt={prompt_idx} extraction produced no text on rc=0 "
                 f"(stdout saved to {os.path.basename(paths['stdout_bad'])}) — "
                 f"fix the harness, do not trust this leg.")
    return text


def _generation_speed(combined_output: str, n_tokens: int, prompt_idx: int, leg_name: str) -> float | None:
    stats_match = GEN_RE.search(combined_output)
    if not stats_match or float(stats_match.group(1)) <= 0:
        print(f"PARITY: leg={leg_name} prompt={prompt_idx} WARN no stats line matched (gen_s=None)", flush=True)
        return None
    return round(n_tokens / float(stats_match.group(1)), 1)


def run_leg(bin_path: str, name: str, model: str, hip: bool, threads: int | None,
            prompt_idx: int | None, n_tokens: int, timeout_s: int, outdir: str) -> int:
    if prompt_idx is not None and not 0 <= prompt_idx < len(PROMPTS):
        sys.exit(f"FATAL: --prompts must be 0..{len(PROMPTS) - 1}, got {prompt_idx}")
    ctx = LegContext(bin_path=bin_path, model=model, hip=hip, threads=threads,
                     n_tokens=n_tokens, timeout_s=timeout_s, outdir=outdir, name=name)
    lock_fd = acquire_lock(outdir)
    try:
        preflight(bin_path)
        os.makedirs(ctx.leg_dir, exist_ok=True)
        config_hash = ctx.config_hash()
        tree_hash = pin_sha()
        _check_vendor_pin(tree_hash)

        leg_meta = {"leg": name, "bin": bin_path, "model": model, "hip": hip,
                    "threads": threads, "n_tokens": n_tokens, "args_sha": config_hash,
                    "pin_sha": tree_hash, "harness_version": HARNESS_VERSION,
                    "wall_s_note": "informational only — page-cache warmth differs between legs"}
        prompt_indices = [prompt_idx] if prompt_idx is not None else list(range(len(PROMPTS)))
        results = _load_resumable_results(ctx.leg_dir, config_hash, prompt_idx)

        for idx in prompt_indices:
            _record_prompt(ctx, leg_meta, results, idx, config_hash, tree_hash)

        ok_count = sum(1 for record in results if record["rc"] == 0)
        print(f"PARITY-LEG: name={name} ok={ok_count}/{len(results)}")
        return 0 if ok_count == len(results) else 1
    finally:
        release_lock(lock_fd)
