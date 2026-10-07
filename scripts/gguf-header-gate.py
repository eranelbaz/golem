#!/usr/bin/env python3
"""Gate: read GGUF headers via the vendored fork's gguf-py.

Asserts arch == glm5next (arch pin) across every shard that carries the
field, and prints the layer/expert counts needed for the manifest's
expert_index.

Usage:
  gguf-header-gate.py [DIR]           # default: $HOME/models/glm-5.3-flash-ud-q4k-xl/UD-Q4_K_XL
"""

import glob
import os
import sys

GGUF_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "vendor", "llama.cpp", "gguf-py")
sys.path.insert(0, GGUF_PY)

import gguf  # noqa: E402

DEFAULT_DIR = os.path.join(os.path.expanduser("~"), "models", "glm-5.3-flash-ud-q4k-xl", "UD-Q4_K_XL")


def fatal(msg):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def main():
    directory = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DIR
    if not os.path.isdir(directory):
        fatal(f"not a directory: {directory}")
    if not os.path.isdir(GGUF_PY):
        fatal(f"vendored gguf-py not found at {GGUF_PY}")

    os.chdir(directory)
    files = sorted(glob.glob("*.gguf"))
    if not files:
        fatal("no .gguf files in current directory")

    # Gate every shard that carries general.architecture (GGUF continuation
    # shards legally omit it) — don't let a mixed-in wrong-quant file pass.
    arch = None
    first = reader = None
    for f in files:
        r = gguf.GGUFReader(f)
        fld = r.get_field("general.architecture")
        if fld is None:
            print(f"arch field absent (continuation shard), skipped: {f}")
            continue
        a = bytes(fld.parts[-1]).decode()
        if arch is None:
            arch, first, reader = a, f, r
        elif a != arch:
            fatal(f"arch mismatch across shards: {first} is {arch!r} but {f} is {a!r}")

    if arch is None:
        fatal("no shard carries general.architecture")
    print("arch:", arch)
    if arch != "glm5next":
        fatal(f"GATE FAILED: arch is {arch!r}, expected 'glm5next' — arch pin mismatch, stop")

    for k in sorted(reader.fields):
        if any(s in k for s in ("expert_count", "expert_used", "block_count",
                                "embedding_length", "attention.head_count", "context_length")):
            fld = reader.get_field(k)
            if fld.types[0] == gguf.GGUFValueType.STRING:
                val = bytes(fld.parts[-1]).decode()
            else:
                val = fld.parts[-1].tolist() if getattr(fld.parts[-1], "ndim", 0) else fld.parts[-1][0]
            print(f"{k} = {val}")
    print("GATE-OK:", first)


if __name__ == "__main__":
    main()
