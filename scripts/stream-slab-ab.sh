#!/usr/bin/env bash
# #119 A/B on the real model: A = base rev (fit leaves K at k_stream), B =
# this tree (fit starts at k_decode). Legs: short / long (≥ 2048-token)
# prompt × cold (fadvise-evicted shards) / warm, run ABBA then BAAB. Each
# leg records peak VRAM (sysfs), the fitted K, device-cache counters and
# the timing CSV. Summary + per-leg stderr go to $LOG.
# usage: stream-slab-ab.sh [base-rev]   (default HEAD~1)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
BASE="${1:-HEAD~1}"
LOG="${GOLEM_AB_LOG:-/tmp/golem-119-ab.log}"
WORK="${GOLEM_AB_WORK:-/tmp/golem-119-ab}"
MODEL="${GOLEM_MODEL:-$HOME/models/glm-5.3-flash-ud-q4k-xl/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf}"
GIB="${GOLEM_AB_GIB:-59}"
TOKENS="${GOLEM_AB_TOKENS:-128}"
# Default 2048 / 1 GiB is plan-infeasible beside the stream_slabs term on
# the 24 GiB card (device_expert_cache 4.53 < 5.40 GiB floor).
UBATCH="${GOLEM_AB_UBATCH:-512}"
RESERVE="${GOLEM_AB_RESERVE_GIB:-0.5}"
mkdir -p "$WORK"
: > "$LOG"
log() { echo "$*" | tee -a "$LOG"; }

[[ -f $MODEL ]] || { log "FATAL: missing $MODEL"; exit 2; }
# Plain-GGUF path: run from a symlinked shard dir so the model dir's
# manifest.json (auto-discovered; its replicas may not exist here) is skipped.
shards=("${MODEL%-00001-of-*}"-*.gguf)
LINKS="$WORK/model"
mkdir -p "$LINKS"
ln -sf "${shards[@]}" "$LINKS/"
MODEL="$LINKS/${MODEL##*/}"
[[ "$(ulimit -l)" == unlimited ]] || { log "FATAL: RLIMIT_MEMLOCK=$(ulimit -l), need unlimited"; exit 2; }

build() {
  local tree="$1"
  log "build $(git -C "$tree" rev-parse --short HEAD) in $tree"
  (cd "$tree" && GOLEM_LLAMA_SYS_BACKEND=hip cargo build --release -q -p golem-cli) >> "$LOG" 2>&1
}
A_TREE="$WORK/base"
if [[ ! -d $A_TREE ]]; then
  git -C "$REPO" worktree add -q --detach "$A_TREE" "$BASE"
fi
build "$A_TREE"
build "$REPO"
A_BIN="$A_TREE/target/release/golem"
B_BIN="$REPO/target/release/golem"

SHORT="Write a detailed technical explanation of how a modern database engine executes a SQL query."
LONG="$(for i in $(seq 1 60); do printf 'Section %d: a database engine parses SQL into an AST, binds names against the catalog, rewrites views, enumerates join orders with a cost model over cardinality estimates, picks access paths, and executes a pipelined operator tree. ' "$i"; done) Summarize the sections above."

vram() { cat /sys/class/drm/card*/device/mem_info_vram_used | awk '{s += $1} END {print s}'; }

leg() {
  local ab="$1" prompt_kind="$2" temp="$3" n="$4"
  local bin prompt out
  [[ $ab == A ]] && bin="$A_BIN" || bin="$B_BIN"
  [[ $prompt_kind == short ]] && prompt="$SHORT" || prompt="$LONG"
  out="$WORK/$n-$ab-$prompt_kind-$temp"
  mkdir -p "$out"
  [[ $temp == cold ]] && python3 "$REPO/scripts/evict-model-cache.py" "${shards[@]}" > /dev/null
  ( while :; do vram; sleep 0.25; done ) > "$out/vram.txt" &
  local sampler=$!
  local t0 rc=0
  t0=$(date +%s.%N)
  RUST_LOG=golem_engine=info "$bin" run "$MODEL" \
    --n-gpu-layers 99 --cpu-moe-layers 99 --greedy --ram-experts-gib "$GIB" \
    --ctx 4096 --n-predict "$TOKENS" --prefill-stream \
    --prefill-ubatch "$UBATCH" --vram-reserve-gib "$RESERVE" \
    --timing-csv "$out/timing.csv" --prompt "$prompt" \
    > "$out/stdout.txt" 2> "$out/stderr.txt" || rc=$?
  kill "$sampler"; wait "$sampler" 2>/dev/null || true
  local peak
  peak=$(sort -n "$out/vram.txt" | tail -1)
  log "LEG n=$n ab=$ab prompt=$prompt_kind temp=$temp rc=$rc wall_s=$(awk -v a="$(date +%s.%N)" -v b="$t0" 'BEGIN{printf "%.1f", a-b}') peak_vram_gib=$(awk -v p="$peak" 'BEGIN{printf "%.2f", p/1073741824}')"
  rg -N "device cache|device_cache bytes|prefill-stream disabled|slab alloc|tok/s|prefill" "$out/stderr.txt" | sed 's/^/  /' | tee -a "$LOG" || true
  tail -3 "$out/timing.csv" 2>/dev/null | sed 's/^/  csv: /' | tee -a "$LOG" || true
}

n=0
for order in "A B B A" "B A A B"; do
  for prompt_kind in short long; do
    for temp in cold warm; do
      for ab in $order; do
        n=$((n + 1))
        leg "$ab" "$prompt_kind" "$temp" "$n"
      done
    done
  done
done
log "DONE"
