#!/usr/bin/env bash
# M0.3 gate: read GGUF header via the vendored fork's gguf-py.
# Asserts arch == glm5next (design D4 / M0.1 pin) and prints the
# layer/expert counts needed for the manifest's expert_index.
set -euo pipefail

DIR="${1:-$HOME/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/UD-Q4_K_XL}"
GGUF_PY="$HOME/Projects/golem/vendor/llama.cpp/gguf-py"

cd "$DIR"
[ -d "$GGUF_PY" ] || { echo "FATAL: vendored gguf-py not found at $GGUF_PY"; exit 1; }

PYTHONPATH="$GGUF_PY" python3 - <<'EOF'
import glob, sys
import gguf

files = sorted(glob.glob("*.gguf"))
if not files:
    sys.exit("FATAL: no .gguf files in current directory")
f = files[0]
r = gguf.GGUFReader(f)

arch = r.get_field("general.architecture")
if arch is None:
    sys.exit("FATAL: general.architecture not in GGUF header")
arch = bytes(arch.parts[-1]).decode()
print("arch:", arch)
if arch != "glm5next":
    sys.exit(f"GATE FAILED: arch is {arch!r}, expected 'glm5next' — M0.1 pin mismatch, stop")

for k in sorted(r.fields):
    if any(s in k for s in ("expert_count", "expert_used", "block_count",
                            "embedding_length", "attention.head_count", "context_length")):
        fld = r.get_field(k)
        if fld.types[0] == gguf.GGUFValueType.STRING:
            val = bytes(fld.parts[-1]).decode()
        else:
            val = fld.parts[-1].tolist() if getattr(fld.parts[-1], "ndim", 0) else fld.parts[-1][0]
        print(f"{k} = {val}")
print("GATE-OK:", f)
EOF
