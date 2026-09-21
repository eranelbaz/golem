"""Leg comparison with tamper re-verification."""
from __future__ import annotations

import hashlib
import json
import os
import sys

from .config import PROMPTS
from .extract import match_count, prefix_words
from .files import atomic_write, prompt_paths


def load_leg(outdir: str, leg_name: str) -> dict:
    leg_record_path = os.path.join(outdir, leg_name, "leg.json")
    if not os.path.exists(leg_record_path):
        sys.exit(f"FATAL: no leg record for '{leg_name}' under {outdir} — run-leg first")
    with open(leg_record_path) as leg_record_file:
        return json.load(leg_record_file)


def _require(condition: bool, message: str) -> None:
    if not condition:
        sys.exit(f"FATAL: {message}")


def _validated_prompts(leg_name: str, leg_record: dict) -> dict[int, dict]:
    """Prompt records keyed by index; dies unless the leg is complete and healthy."""
    by_index = {record["idx"]: record for record in leg_record["prompts"]}
    _require(sorted(by_index) == list(range(len(PROMPTS))),
             f"incomplete leg {leg_name} (has prompts {sorted(by_index)}) — "
             f"both legs must cover all {len(PROMPTS)} prompts")
    ok_count = sum(1 for record in by_index.values() if record["rc"] == 0)
    _require(ok_count == len(PROMPTS),
             f"refuse to compare: {leg_name} ok={ok_count}/{len(PROMPTS)}")
    return by_index


def _validated_pin(leg_a: str, leg_b: str,
                   prompts_a: dict[int, dict], prompts_b: dict[int, dict]) -> str:
    # Banner build ids are recorded but NOT compared — they track the enclosing
    # repo HEAD, not the pin (see config.pin_sha).
    pins_a = {record.get("pin_sha") for record in prompts_a.values()}
    pins_b = {record.get("pin_sha") for record in prompts_b.values()}
    pins_equal = len(pins_a) == 1 and len(pins_b) == 1 and pins_a == pins_b
    _require(pins_equal and None not in pins_a,
             f"pin identity missing or mismatched (a={pins_a}, b={pins_b}) — "
             f"both legs must run binaries built from the same pinned tree")
    _require(leg_a.get("hip") != leg_b.get("hip") or leg_a.get("threads") != leg_b.get("threads"),
             f"identical backend config on both legs (hip={leg_a.get('hip')}, "
             f"threads={leg_a.get('threads')}) — that is not a parity comparison")
    return pins_a.pop()


def _require_same_setup(leg_a: dict, leg_b: dict) -> None:
    _require(leg_a.get("n_tokens") == leg_b.get("n_tokens"),
             f"legs disagree on -n ({leg_a.get('n_tokens')} vs {leg_b.get('n_tokens')})")
    _require(leg_a.get("model") == leg_b.get("model"), "legs ran different models")
    _require(leg_a.get("harness_version") == leg_b.get("harness_version"),
             f"legs produced by different harness versions "
             f"({leg_a.get('harness_version')} vs {leg_b.get('harness_version')})")


def verify_texts(outdir: str, leg_name: str, leg_record: dict) -> list[str]:
    """Re-hash leg.json texts against the per-prompt meta sidecars (tamper check)."""
    leg_dir = os.path.join(outdir, leg_name)
    texts = []
    for record in leg_record["prompts"]:
        if record.get("rc") != 0:
            continue
        meta_path = prompt_paths(leg_dir, record["idx"])["meta"]
        try:
            with open(meta_path) as meta_file:
                recorded_hash = json.load(meta_file).get("text_sha")
        except OSError:
            sys.exit(f"FATAL: leg={leg_name} prompt={record['idx']} meta sidecar missing ({meta_path})")
        actual_hash = hashlib.sha256(record["text"].encode()).hexdigest()
        _require(recorded_hash == actual_hash,
                 f"leg={leg_name} prompt={record['idx']} text_sha mismatch vs sidecar — "
                 f"leg.json was altered after the run; re-run the leg")
        texts.append(record["text"])
    return texts


def compare(leg_name_a: str, leg_name_b: str, outdir: str) -> int:
    leg_a = load_leg(outdir, leg_name_a)
    leg_b = load_leg(outdir, leg_name_b)
    _require_same_setup(leg_a, leg_b)
    prompts_a = _validated_prompts(leg_name_a, leg_a)
    prompts_b = _validated_prompts(leg_name_b, leg_b)
    pin = _validated_pin(leg_a, leg_b, prompts_a, prompts_b)

    texts_a = [prompts_a[i]["text"] for i in range(len(PROMPTS))]
    texts_b = [prompts_b[i]["text"] for i in range(len(PROMPTS))]
    verify_texts(outdir, leg_name_a, leg_a)
    verify_texts(outdir, leg_name_b, leg_b)
    _require(all(text_a and text_b for text_a, text_b in zip(texts_a, texts_b)),
             "empty texts in one or both legs — legs are not trustworthy")

    match_total = match_count(texts_a, texts_b)
    prefixes = [prefix_words(text_a, text_b) for text_a, text_b in zip(texts_a, texts_b)]
    report = {"a": leg_name_a, "b": leg_name_b, "n_tokens": leg_a.get("n_tokens"),
              "prompts": len(PROMPTS), "pin_sha": pin, "match_count": match_total,
              "match_rate": round(100.0 * match_total / len(PROMPTS), 2),
              "prefix_words": prefixes}
    report_path = os.path.join(outdir, f"{leg_name_a}_vs_{leg_name_b}.json")
    atomic_write(report_path, json.dumps(report, indent=2) + "\n")
    print(f"PARITY: a={leg_name_a} b={leg_name_b} pin={pin[:12]} "
          f"match={match_total}/{len(PROMPTS)} prefixes={prefixes}")
    print(f"PARITY: wrote {report_path}")
    return 0
