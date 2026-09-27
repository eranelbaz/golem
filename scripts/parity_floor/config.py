"""Shared constants, prompt set, provenance hashing, and llama-cli flag contract."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys

DEFAULT_MODEL = os.environ.get("GOLEM_M05_MODEL")  # no committed machine-local default
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
    """Content hash of the vendored substrate (canonical cs() recipe from
    scripts/build-substrate.sh, python form).

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


def check_vendor_pin(tree_hash: str) -> None:
    """Die unless tree_hash equals the pin recorded in vendor/VENDOR_SHA
    ('content_sha256=<hex>' first line)."""
    try:
        with open(VENDOR_SHA_FILE) as pin_file:
            pinned_hash = pin_file.readline().split("=", 1)[1].strip()
    except (OSError, IndexError):
        sys.exit(f"FATAL: cannot read/parse {VENDOR_SHA_FILE} (expected '<name>=<sha256>')")
    if tree_hash != pinned_hash:
        sys.exit(f"FATAL: vendor tree drifted ({tree_hash[:12]} != {pinned_hash[:12]}) — "
                 f"re-vendor per scripts/build-substrate.sh")


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
