#!/usr/bin/env bash
# KV cache A/B (issue #107): serves the model once per KV type at 131k ctx,
# checks llama's logged KV size against `golem plan`, runs kv-ab.py, then
# compares. Needs the GPU to itself: stop any other golem serve first.
# usage: scripts/kv-ab.sh <model.gguf> [types...]   (default: q8_0 f16)
set -euo pipefail
cd "$(dirname "$0")/.."

M="${1:?usage: kv-ab.sh <model.gguf> [types...]}"; shift
TYPES=("$@")
[ $# -eq 0 ] && TYPES=(q8_0 f16)
CTX=131072
PORT="${KV_AB_PORT:-8090}"
OUT="${KV_AB_OUT:-/tmp/kv-ab}"
mkdir -p "$OUT"

./scripts/build.sh build

for T in "${TYPES[@]}"; do
  echo "=== $T ==="
  ./target/release/golem plan "$M" --ctx "$CTX" --cache-type-k "$T" --cache-type-v "$T" \
    | rg "KV .* B/token" | tee "$OUT/$T.plan"

  RUST_LOG=info ./target/release/golem serve "$M" \
    --host 127.0.0.1 --port "$PORT" --no-optimize-profile \
    --ctx "$CTX" --n-predict 512 --parallel 1 \
    --cache-type-k "$T" --cache-type-v "$T" \
    --flash-attn --prefill-stream --prefill-stream-min 128 \
    --ram-experts-gib 50 --io-threads 4 --vram-experts-frac 0.85 \
    --cpu-expert-math --cpu-expert-frac 1.0 --cpu-expert-threads 12 \
    --timing-csv "$OUT/$T.csv" > "$OUT/$T.log" 2>&1 &
  PID=$!
  trap 'kill $PID 2>/dev/null || true' EXIT

  for _ in $(seq 600); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
    kill -0 $PID 2>/dev/null || { echo "serve died; see $OUT/$T.log"; tail -20 "$OUT/$T.log"; exit 1; }
    sleep 1
  done
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "serve not healthy after 600 s"; exit 1; }

  echo "--- llama KV cache lines (expect K ($T)) ---"
  rg "llama_kv_cache: size|flash_attn|device cache" "$OUT/$T.log" | tee "$OUT/$T.kv" || true
  { cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null || echo n/a; } \
    | awk '$1 == "n/a" {print "VRAM used n/a"; exit} {s += $1} END {if (NR) print "VRAM used", int(s / 1048576), "MiB"}' \
    | tee "$OUT/$T.vram"

  python3 scripts/kv-ab.py probe --url "http://127.0.0.1:$PORT" --out "$OUT/$T.json" || true

  kill $PID 2>/dev/null || true; wait $PID 2>/dev/null || true
  trap - EXIT
done

if [ "${#TYPES[@]}" -ge 2 ] && [ -f "$OUT/${TYPES[0]}.json" ] && [ -f "$OUT/${TYPES[1]}.json" ]; then
  echo "=== compare ${TYPES[0]} vs ${TYPES[1]} ==="
  python3 scripts/kv-ab.py compare "$OUT/${TYPES[0]}.json" "$OUT/${TYPES[1]}.json"
fi
echo "results in $OUT"
