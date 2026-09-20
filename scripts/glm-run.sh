#!/usr/bin/env bash
# Wraps llama-cli with wall timing, VRAM sampling, and log capture.
# usage: glm-run.sh <name> <model.gguf> [llama-cli args...]
set -euo pipefail

NAME="$1"; shift
MODEL="$1"; shift

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$REPO/vendor/llama.cpp/build/bin"
LOGDIR="${GOLEM_M0_WORK:-$REPO/.m0-work/m04}"
LOG="$LOGDIR/$NAME.log"
VLOG="$LOGDIR/$NAME.vram"
TIMEF="$LOGDIR/$NAME.time"
mkdir -p "$LOGDIR"

vram_sample() {
  rocm-smi --showmeminfo vram --json | python3 -c \
    'import json,sys,time; d=json.load(sys.stdin); k=next(iter(d)); print(time.time(), d[k]["VRAM Total Used Memory (B)"])'
}
vram_sample > "$VLOG"
IDLE=$(tail -1 "$VLOG" | awk '{print $2}')
if ! [[ "$IDLE" =~ ^[0-9]+$ ]]; then
  echo "FATAL: could not read idle VRAM from rocm-smi (got '$IDLE')" >&2
  exit 2
fi
: > "$LOG"
: > "$TIMEF"

( while true; do vram_sample || echo "WARN: vram sample failed" >&2; sleep 0.5; done ) >> "$VLOG" &
SAMPLER=$!
trap 'kill "$SAMPLER" 2>/dev/null || true; wait "$SAMPLER" 2>/dev/null || true' EXIT

RC=0
python3 -c '
import resource, subprocess, sys, time
t0 = time.perf_counter()
rc = subprocess.call(sys.argv[1:])
dt = time.perf_counter() - t0
ru = resource.getrusage(resource.RUSAGE_CHILDREN)
print(f"MEAS wall_s={dt:.1f} max_rss_kib={ru.ru_maxrss}", file=sys.stderr)
sys.exit(rc)
' "$BIN/llama-cli" -m "$MODEL" "$@" > "$LOG" 2>&1 || RC=$?
grep -E 'MEAS wall_s' "$LOG" >> "$TIMEF" || true
PEAK=$(awk '{print $2}' "$VLOG" | sort -n | tail -1)
TS=$(grep -o '\[ Prompt: [^]]*\]' "$LOG" | tail -1 || true)
GEN=$(grep -o 'Generation: [0-9.]* t/s' "$LOG" | tail -1 || true)
echo "GLM-RUN: name=$NAME rc=$RC idle_vram_bytes=$IDLE peak_vram_bytes=$PEAK ts=\"$TS\" gen_ts=\"$GEN\""
exit "$RC"
