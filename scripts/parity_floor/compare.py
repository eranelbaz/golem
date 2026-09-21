"""Leg comparison with tamper re-verification."""
from __future__ import annotations

import hashlib
import json
import os
import sys

from .config import PROMPTS
from .extract import match_count, prefix_words
from .files import atomic_write, prompt_paths


def load_leg(outdir: str, leg: str) -> dict:
    path = os.path.join(outdir, leg, "leg.json")
    if not os.path.exists(path):
        sys.exit(f"FATAL: no leg record for '{leg}' under {outdir} — run-leg first")
    with open(path) as f:
        return json.load(f)


def verify_texts(outdir: str, leg: str, rec: dict) -> list[str]:
    """Re-hash leg.json texts against the per-prompt meta sidecars (tamper check)."""
    legdir = os.path.join(outdir, leg)
    texts = []
    for r in rec["prompts"]:
        if r.get("rc") != 0:
            continue
        meta_path = prompt_paths(legdir, r["idx"])["meta"]
        try:
            with open(meta_path) as f:
                recorded = json.load(f).get("text_sha")
        except OSError:
            sys.exit(f"FATAL: leg={leg} prompt={r['idx']} meta sidecar missing ({meta_path})")
        if not recorded or hashlib.sha256(r["text"].encode()).hexdigest() != recorded:
            sys.exit(f"FATAL: leg={leg} prompt={r['idx']} text_sha mismatch vs sidecar — "
                     f"leg.json was altered after the run; re-run the leg")
        texts.append(r["text"])
    return texts


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
    verify_texts(outdir, a, la)
    verify_texts(outdir, b, lb)
    empties = [i for i, (x, y) in enumerate(zip(ta, tb)) if not x or not y]
    if empties:
        sys.exit(f"FATAL: empty texts at prompt indices {empties} — legs are not trustworthy")
    mc = match_count(ta, tb)
    prefixes = [prefix_words(x, y) for x, y in zip(ta, tb)]
    pin = next(iter(pins_a))
    report = {"a": a, "b": b, "n_tokens": la.get("n_tokens"), "prompts": len(PROMPTS),
              "pin_sha": pin,
              "match_count": mc,
              "match_rate": round(100.0 * mc / len(PROMPTS), 2),
              "prefix_words": prefixes}
    out = os.path.join(outdir, f"{a}_vs_{b}.json")
    atomic_write(out, json.dumps(report, indent=2) + "\n")
    print(f"PARITY: a={a} b={b} pin={pin[:12]} match={mc}/{len(PROMPTS)} prefixes={prefixes}")
    print(f"PARITY: wrote {out}")
    return 0
