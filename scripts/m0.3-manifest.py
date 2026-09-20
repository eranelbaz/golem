#!/usr/bin/env python3
"""M0.3: build the design §4.2 manifest.json for one quant dir of GGUF shards.

Usage:
  m0.3-manifest.py <dir> <name> <repo> <revision> <quant> [--replica PATH ...] [--header-only]

<dir> holds the .gguf shards (nested <outer>/<QUANT>/ layout from `hf download
--local-dir`; pass the inner dir). --header-only skips hashing/streaming and
prints header facts + tensor-walk summary instead of writing the manifest
(usable while later shards are still downloading).
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys

GGUF_PY = os.path.join(os.path.dirname(__file__), "..", "vendor", "llama.cpp", "gguf-py")
sys.path.insert(0, GGUF_PY)

import gguf  # noqa: E402

MOE_RE = re.compile(r"ffn_(up|down|gate|gate_up)_(ch|)exps")
CHUNK = 1024 * 1024
EXPECT_ARCH = "glm5next"


def field_str(reader, key):
    f = reader.get_field(key)
    return bytes(f.parts[-1]).decode() if f is not None else None


def field_int(reader, key):
    f = reader.get_field(key)
    return int(f.parts[-1][0]) if f is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("name")
    ap.add_argument("repo")
    ap.add_argument("revision")
    ap.add_argument("quant")
    ap.add_argument("--replica", action="append", default=[])
    ap.add_argument("--header-only", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.gguf")))
    if not files:
        sys.exit(f"FATAL: no .gguf files in {args.dir}")
    for f in files:
        if os.path.basename(f).startswith("mmproj"):
            sys.exit(f"FATAL: mmproj (vision) file present, text path only in v1: {f}")

    readers = {f: gguf.GGUFReader(f) for f in files}
    first = files[0]
    arch = field_str(readers[first], "general.architecture")
    if arch != EXPECT_ARCH:
        sys.exit(f"GATE FAILED: arch is {arch!r}, expected {EXPECT_ARCH!r}")

    block_count = field_int(readers[first], f"{arch}.block_count")
    expert_count = field_int(readers[first], f"{arch}.expert_count")
    if expert_count is None or expert_count <= 0:
        sys.exit(f"FATAL: {arch}.expert_count missing or invalid in GGUF header")
    print(f"arch: {arch}  block_count: {block_count}  expert_count: {expert_count}")

    # expert_index: walk tensor tables of EVERY shard (shard 1 has zero tensors).
    expert_index = []
    tensor_count = 0
    for f in files:
        r = readers[f]
        for t in r.tensors:
            tensor_count += 1
            m = MOE_RE.search(t.name)
            if not m:
                continue
            layer = int(re.search(r"blk\.(\d+)\.", t.name).group(1)) if re.search(r"blk\.(\d+)\.", t.name) else sys.exit(
                f"FATAL: expert-like tensor without blk.N. prefix: {t.name}")
            if t.n_bytes % expert_count != 0:
                sys.exit(f"FATAL: {t.name} n_bytes {t.n_bytes} not divisible by {expert_count} experts")
            bpe = t.n_bytes // expert_count
            expert_index.append({
                "tensor": t.name,
                "layer": layer,
                "experts": expert_count,
                "bytes_per_expert": bpe,
                "offsets_bytes": [i * bpe for i in range(expert_count)],
            })
    expert_index.sort(key=lambda e: (e["layer"], e["tensor"]))
    by_name = {e["tensor"]: e for e in expert_index}
    if len(by_name) != len(expert_index):
        sys.exit("FATAL: same expert tensor name in multiple shards — split boundary corruption")
    print(f"tensors walked: {tensor_count}  expert tensors: {len(expert_index)}")
    for e in expert_index[:3]:
        print(f"  e.g. {e['tensor']}: layer {e['layer']}, {e['experts']} experts, "
              f"{e['bytes_per_expert']} bytes/expert")
    if args.header_only:
        print("HEADER-ONLY: manifest not written")
        return

    # Full pass: stream every shard through sha256, record exact sizes.
    primary = os.path.abspath(args.dir)
    shards = []
    for f in files:
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            while chunk := fh.read(CHUNK):
                h.update(chunk)
        size = os.path.getsize(f)
        shards.append({
            "file": os.path.basename(f),
            "bytes": size,
            "sha256": h.hexdigest(),
            "replicas": [primary] + [os.path.abspath(p) for p in args.replica],
        })
        print(f"hashed {os.path.basename(f)}  {size}  {shards[-1]['sha256']}")

    tmpl = field_str(readers[first], "tokenizer.chat_template")
    if tmpl is None:
        sys.exit("FATAL: tokenizer.chat_template missing from GGUF header")

    manifest = {
        "name": args.name,
        "repo": args.repo,
        "revision": args.revision,
        "quant": args.quant,
        "arch": arch,
        "shards": shards,
        "expert_index": expert_index,
        "chat_template_sha": hashlib.sha256(tmpl.encode()).hexdigest(),
    }
    out = os.path.join(args.dir, "manifest.json")
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2)
    total = sum(s["bytes"] for s in shards)
    print(f"MANIFEST-OK: {out}  shards={len(shards)}  total={total}")


if __name__ == "__main__":
    main()
