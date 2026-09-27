"""Parity-floor harness package.

Integrity rules (the harness must never manufacture a clean result):
  - a prompt's text is persisted only when the run exited 0 AND extraction is non-empty
  - empty extraction on rc=0 is a harness error: the leg aborts immediately
  - cached prompts skip only when the config sidecar matches exactly AND the
    recorded pin_sha equals the live vendored-tree hash AND the text hash verifies
  - compare refuses legs that are not ok=8/8 with identical -n and non-empty texts
"""
from .config import (DEFAULT_MODEL, DEFAULT_OUTDIR, HARNESS_VERSION, PROMPTS,
                     VENDOR_DIR, VENDOR_SHA_FILE, check_vendor_pin, config_sha,
                     leg_args, pin_sha)
from .extract import (extract_generation, extract_text, match_count,
                      parse_run_output, prefix_words)
from .files import atomic_write, prompt_paths, write_leg_json
from .locking import acquire_lock, preflight, release_lock
from .runner import cached_result, run_leg, run_prompt
from .compare import compare, load_leg, verify_texts
from .selftest import selftest

__all__ = [
    "DEFAULT_MODEL", "DEFAULT_OUTDIR", "HARNESS_VERSION", "PROMPTS",
    "VENDOR_DIR", "VENDOR_SHA_FILE", "check_vendor_pin", "config_sha",
    "leg_args", "pin_sha",
    "extract_generation", "extract_text", "match_count", "parse_run_output",
    "prefix_words", "atomic_write", "prompt_paths", "write_leg_json",
    "acquire_lock", "preflight", "release_lock",
    "cached_result", "run_leg", "run_prompt", "compare", "load_leg",
    "verify_texts", "selftest",
]
