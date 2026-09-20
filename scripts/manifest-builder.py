#!/usr/bin/env python3
"""M0.3: build a manifest.json for one quant dir of GGUF shards.

Usage:
  manifest-builder.py <dir> <name> <repo> <revision> <quant> [--replica PATH ...] [--header-only]

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
BLK_RE = re.compile(r"blk\.(\d+)\.")
SHARD_RE = re.compile(r"-\d{5}-of-\d{5}\.gguf$")
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
    strays = [f for f in files if not SHARD_RE.search(f)]
    if strays:
        sys.exit(f"FATAL: non-shard-named .gguf files present (refusing to guess header source): {strays}")
    for f in files:
        if os.path.basename(f).startswith("mmproj"):
            sys.exit(f"FATAL: mmproj (vision) file present, text path only in v1: {f}")

    readers = {f: gguf.GGUFReader(f) for f in files}
    first = files[0]
    arch = field_str(readers[first], "general.architecture")
    if arch != EXPECT_ARCH:
        sys.exit(f"GATE FAILED: arch is {arch!r}, expected {EXPECT_ARCH!r}")
    # Every shard that carries general.architecture must agree (GGUF continuation
    # shards legally omit it) — a mixed-in wrong-quant shard must not pass.
    for f in files[1:]:
        other = field_str(readers[f], "general.architecture")
        if other is not None and other != arch:
            sys.exit(f"FATAL: arch mismatch across shards: {os.path.basename(first)} is {arch!r} "
                     f"but {os.path.basename(f)} is {other!r}")

    block_count = field_int(readers[first], f"{arch}.block_count")
    dense_count = field_int(readers[first], f"{arch}.leading_dense_block_count") or 0
    expert_count = field_int(readers[first], f"{arch}.expert_count")
    if expert_count is None or expert_count <= 0:
        sys.exit(f"FATAL: {arch}.expert_count missing or invalid in GGUF header")
    print(f"arch: {arch}  block_count: {block_count}  expert_count: {expert_count}  dense: {dense_count}")

    # expert_index: walk tensor tables of EVERY shard (shard 1 has zero tensors).
    expert_index = []
    tensor_count = 0
    for f in files:
        r = readers[f]
        for t in r.tensors:
            tensor_count += 1
            if not MOE_RE.search(t.name):
                if "mmproj" in t.name:
                    sys.exit(f"FATAL: mmproj (vision) tensor present, text path only in v1: {t.name}")
                continue
            m = BLK_RE.search(t.name)
            if not m:
                sys.exit(f"FATAL: expert-like tensor without blk.N. prefix: {t.name}")
            layer = int(m.group(1))
            if "mmproj" in t.name:
                sys.exit(f"FATAL: mmproj (vision) tensor present, text path only in v1: {t.name}")
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
    # Completeness: exactly 3 expert tensors per MoE layer (gate/up/down —
    # glm5next has no ffn_gate_up_exps). Catches truncated tensor tables at
    # split boundaries. If this script is ever reused for a gate_up-fused arch,
    # the * 3 here and the MOE_RE alternation must be revisited together.
    moe_layers = sorted({e["layer"] for e in expert_index})
    expected_layers = list(range(dense_count, block_count))
    if moe_layers != expected_layers or len(expert_index) != len(expected_layers) * 3:
        sys.exit(f"FATAL: expert_index incomplete: {len(moe_layers)} MoE layers / {len(expert_index)} tensors, "
                 f"expected layers {dense_count}..{block_count - 1} with 3 tensors each")
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
        st = os.stat(f)
        size = st.st_size
        h = hashlib.sha256()
        with open(f, "rb") as fh:
            fst = os.fstat(fh.fileno())
            if (fst.st_size != size or fst.st_ino != st.st_ino or fst.st_dev != st.st_dev):
                sys.exit(f"FATAL: {f} replaced/changed between stat and open — file in flight, aborting")
            while chunk := fh.read(CHUNK):
                h.update(chunk)
        if os.stat(f).st_size != size:
            sys.exit(f"FATAL: {f} changed size during hashing — file in flight, aborting")
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
