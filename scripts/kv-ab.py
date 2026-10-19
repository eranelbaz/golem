#!/usr/bin/env python3
"""KV cache A/B probe against a running `golem serve` (issue #107).

Short-prompt parity answers plus needle recall at several depths of a long
filler context. Writes one JSON result file; `--compare A.json B.json`
diffs two of them.

usage: kv-ab.py probe --url URL --out FILE [--tokens N]
       kv-ab.py compare A.json B.json
"""
import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request

SHORT = [
    "What is 17 * 23? Answer with the number only.",
    "Name the capital of Australia in one word.",
    "Write a Python one-liner that reverses a string s.",
    "List the first five prime numbers, comma separated.",
]

NEEDLES = [
    (0.10, "The vault code for the Orion project is 4817-KESTREL.",
     "What is the vault code for the Orion project?", "4817-KESTREL"),
    (0.50, "Dr. Maren Holt's favourite tea is smoked lapsang from Fujian.",
     "What is Dr. Maren Holt's favourite tea?", "lapsang"),
    (0.90, "The backup server in rack 12 is named BLUE-HERON-7.",
     "What is the backup server in rack 12 named?", "BLUE-HERON-7"),
]

MODEL = ""

WORDS = ("river stone window market lantern harbor meadow copper signal garden "
         "engine ledger orchard compass thread canyon beacon quarry violet anchor").split()


def chat(url, messages, max_tokens):
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "seed": 1,
        "max_tokens": max_tokens,
        "reasoning_effort": "none",
    }).encode()
    req = urllib.request.Request(f"{url}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            d = json.load(r)
    except urllib.error.HTTPError as e:
        sys.exit(f"HTTP {e.code}: {e.read().decode(errors='replace')}")
    return d["choices"][0]["message"]["content"] or "", d.get("usage", {}), time.time() - t


def filler(n_lines, seed=7):
    rng = random.Random(seed)
    return [f"Note {i}: " + " ".join(rng.choice(WORDS) for _ in range(12)) + "."
            for i in range(n_lines)]


def haystack(lines, needle_at, needle):
    out = list(lines)
    out.insert(int(len(out) * needle_at), needle)
    return "\n".join(out)


def probe(a):
    global MODEL
    with urllib.request.urlopen(f"{a.url}/v1/models", timeout=30) as r:
        MODEL = json.load(r)["data"][0]["id"]
    res = {"url": a.url, "short": [], "needles": []}
    for q in SHORT:
        text, _, dt = chat(a.url, [{"role": "user", "content": q}], 128)
        res["short"].append({"q": q, "a": text.strip(), "s": round(dt, 2)})
        print(f"short  {dt:6.1f}s  {q[:40]!r} -> {text.strip()[:60]!r}", flush=True)

    # Calibrate tokens per filler line on the real tokenizer.
    _, u, _ = chat(a.url, [{"role": "user", "content": "\n".join(filler(200))}], 1)
    per_line = max(u.get("prompt_tokens", 0), 1) / 200
    n_lines = int(a.tokens / per_line)
    lines = filler(n_lines)
    print(f"filler {n_lines} lines ≈ {a.tokens} tokens ({per_line:.1f} tok/line)", flush=True)

    for depth, fact, q, want in NEEDLES:
        doc = haystack(lines, depth, fact)
        prompt = f"{doc}\n\nUsing only the notes above, answer briefly: {q}"
        text, u, dt = chat(a.url, [{"role": "user", "content": prompt}], 64)
        ok = want.lower() in text.lower()
        res["needles"].append({"depth": depth, "want": want, "a": text.strip(), "ok": ok,
                               "prompt_tokens": u.get("prompt_tokens"), "s": round(dt, 1)})
        print(f"needle {depth:.0%} {u.get('prompt_tokens')} tok {dt:7.1f}s "
              f"{'PASS' if ok else 'FAIL'} -> {text.strip()[:60]!r}", flush=True)

    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    return 0 if all(n["ok"] for n in res["needles"]) else 1


def compare(a):
    x, y = (json.load(open(p)) for p in (a.a, a.b))
    print(f"{'':8} {a.a:>30} {a.b:>30}")
    same = 0
    for s, t in zip(x["short"], y["short"]):
        eq = s["a"] == t["a"]
        same += eq
        print(f"short  {'same' if eq else 'DIFF'}  {s['q'][:44]!r}")
        if not eq:
            print(f"   A: {s['a'][:100]!r}\n   B: {t['a'][:100]!r}")
    print(f"short parity: {same}/{len(x['short'])} identical")
    for s, t in zip(x["needles"], y["needles"]):
        print(f"needle {s['depth']:.0%}  A {'PASS' if s['ok'] else 'FAIL'} ({s['s']}s)"
              f"  B {'PASS' if t['ok'] else 'FAIL'} ({t['s']}s)")
    return 0


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("probe")
    pr.add_argument("--url", required=True)
    pr.add_argument("--out", required=True)
    pr.add_argument("--tokens", type=int, default=120_000)
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    a = p.parse_args()
    return probe(a) if a.cmd == "probe" else compare(a)


if __name__ == "__main__":
    sys.exit(main())
