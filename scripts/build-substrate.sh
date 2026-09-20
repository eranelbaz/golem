#!/usr/bin/env bash
# Build the pinned vendored llama.cpp fork for gfx1100 (HIP). Design §14/M0.2.
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

LOCK=engine/llama-cpp.lock
VENDOR=vendor/llama.cpp
BUILD_DIR="$VENDOR/build"
TARGET=gfx1100

# --- pin integrity: never build a drifted tree (ADR 0001 canonical recipe) ---
# The build-output dir is pruned: the canonical hash covers source, not artifacts, so
# re-runs after a (re)build still match VENDOR_SHA. Pruning changes nothing on a pristine tree.
cs() { cd "$1" && find . -path ./.git -prune -o -path ./build -prune -o -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}'; }
got=$(cs "$VENDOR")
want=$(grep '^content_sha256' vendor/VENDOR_SHA | cut -d= -f2)
[ "$got" = "$want" ] || { echo "FATAL: vendor tree drifted ($got != $want) — re-vendor per ADR 0001"; exit 1; }
pin=$(grep '^commit' "$LOCK" | sed 's/[[:space:]]*#.*//' | awk -F'= *' '{print $2}')
echo "pin-ok: $pin"

# --- advisory ROCm version check: WARN or skip, never fail (design §3.5) ---
lock_rocm=$(grep '^rocm' "$LOCK" | sed 's/[[:space:]]*#.*//' | awk -F'= *' '{print $2}')
if ! command -v rocminfo >/dev/null 2>&1; then
  echo "rocm-check: skipped (rocminfo absent)"
else
  # installed version: /opt/rocm/.info/version, falling back to hipcc's dir prefix
  inst=$(cat /opt/rocm/.info/version 2>/dev/null || true)
  if [ -z "$inst" ]; then echo "rocm-check: skipped (no /opt/rocm/.info/version)"; fi
  if [ -n "$inst" ] && [ "${inst%%.*}" != "${lock_rocm%%.*}" -o "$(printf '%s\n' "$inst" | cut -d. -f1-2)" != "$(printf '%s\n' "$lock_rocm" | cut -d. -f1-2)" ]; then
    echo "WARN: rocminfo reports ROCm $inst, lock pins $lock_rocm — advisory minimum (design §3.5); continuing"
  elif [ -n "$inst" ]; then
    echo "rocm-check: ok ($inst matches lock $lock_rocm)"
  fi
fi

# --- configure + build ---
[ -d "$BUILD_DIR" ] || mkdir -p "$BUILD_DIR"
cmake -S "$VENDOR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS="$TARGET"
cmake --build "$BUILD_DIR" --target llama-cli -j"$(nproc)"

echo "BUILD-OK: $BUILD_DIR/bin/llama-cli"
