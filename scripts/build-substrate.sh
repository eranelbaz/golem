#!/usr/bin/env bash
# Build the pinned vendored llama.cpp fork for gfx1100 (HIP).
set -euo pipefail
cd "$(dirname "$0")/.."   # repo root

LOCK=engine/llama-cpp.lock
VENDOR=vendor/llama.cpp
BUILD_DIR="$VENDOR/build"
TARGET=gfx1100

# --- pin integrity: never build a drifted tree (canonical recipe, mirrored in crates/golem-llama-sys/tests/common/mod.rs::CANONICAL_CS_CORE) ---
# The build-output dir is pruned: the canonical hash covers source, not artifacts, so
# re-runs after a (re)build still match VENDOR_SHA. Pruning changes nothing on a pristine tree.
# xargs -r: empty input → empty stream (never sha256sum reading stdin as "-").
cs() { cd "$1" && find . -path ./.git -prune -o -path ./build -prune -o -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | awk '{print $1}'; }
want=$(sed -n 's/^content_sha256 *= *//p' vendor/VENDOR_SHA)
[ -n "$want" ] || { echo "FATAL: content_sha256 key missing/empty in vendor/VENDOR_SHA"; exit 1; }
got=$(cs "$VENDOR")
[ "$got" = "$want" ] || { echo "FATAL: vendor tree drifted ($got != $want) — delete vendor/llama.cpp and re-run this script"; exit 1; }
pin=$(sed -n 's/^commit *= *//p' "$LOCK" | head -n1)
[ -n "$pin" ] || { echo "FATAL: commit key missing in $LOCK"; exit 1; }
echo "pin-ok: $pin"

# --- advisory ROCm version check: WARN or skip, never fail ---
lock_rocm=$(sed -n 's/^rocm *= *//p' "$LOCK" | head -n1)
[ -n "$lock_rocm" ] || { echo "FATAL: rocm key missing in $LOCK"; exit 1; }
if ! command -v rocminfo >/dev/null 2>&1; then
  echo "rocm-check: skipped (rocminfo absent)"
else
  # installed version from /opt/rocm/.info/version (no fallback needed on ROCm installs; absence → skip)
  inst=$(cat /opt/rocm/.info/version 2>/dev/null || true)
  if [ -z "$inst" ]; then echo "rocm-check: skipped (no /opt/rocm/.info/version)"; fi
  # advisory-minimum semantics: equal major.minor → ok; older/newer → WARN; never fail
  if [ -n "$inst" ] && [ "$(printf '%s\n' "$inst" | cut -d. -f1-2)" != "$(printf '%s\n' "$lock_rocm" | cut -d. -f1-2)" ]; then
    echo "WARN: rocminfo reports ROCm $inst, lock pins $lock_rocm — advisory minimum; continuing"
  elif [ -n "$inst" ]; then
    echo "rocm-check: ok ($inst satisfies lock $lock_rocm)"
  fi
fi

# --- configure + build ---
mkdir -p "$BUILD_DIR"
cmake -S "$VENDOR" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DAMDGPU_TARGETS="$TARGET"
cmake --build "$BUILD_DIR" --target llama-cli -j"$(nproc)"

[ -x "$BUILD_DIR/bin/llama-cli" ] || { echo "FATAL: build reported success but $BUILD_DIR/bin/llama-cli is missing"; exit 1; }
echo "BUILD-OK: $BUILD_DIR/bin/llama-cli"
printf '%s\n' "$pin" > "$BUILD_DIR/substrate-pin.txt"
