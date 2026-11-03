#!/usr/bin/env bash
# Store-drain A/B on the real model: baseline (origin/main) vs this tree, and
# read depth via io_depth 1/2/4 vs explicit io_threads. Every leg runs cold
# and warm, in both orders (ABBA). Decode and prefill (long prompt) are
# separate runs; prefill legs decode 16 tokens and report prefill as
# (long-leg non-decode wall) - (same binary/temp short-leg non-decode wall). Summary: $OUT/summary.md; per-leg logs under $OUT/<leg>/.
# usage: scripts/perf-205.sh   (env: GOLEM_MODEL, GOLEM_AB_TOKENS, PERF_REPS, PERF_LEGS, PERF_OUT)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
OUT="${PERF_OUT:-$REPO/.ab-work/perf-205}"
REPS="${PERF_REPS:-4}"
# Model, pool size and VRAM knobs come from golem.toml's default model
# (the plan OOMs without its VRAM knobs); env overrides win.
eval "$(scripts/golem-toml-model.py)"
export GOLEM_MODEL GOLEM_AB_GIB
COMMON="${PERF_FLAGS:-$GOLEM_MODEL_FLAGS}"
[[ -f $GOLEM_MODEL && -f ${GOLEM_MODEL%/*}/manifest.json ]] || { echo "FATAL: missing $GOLEM_MODEL or its manifest.json"; exit 2; }
mkdir -p "$OUT"
exec > >(tee "$OUT/driver.log") 2>&1
trap 'exec >&- 2>&-; wait' EXIT

[[ $EUID -ne 0 ]] || { echo "FATAL: run as your user, not via sudo (cargo/cmake PATH, file ownership); it asks for sudo itself"; exit 2; }
# Cold legs drop caches via `sudo -n`: prompt once, then keep the timestamp
# fresh for the multi-hour run.
sudo -v || { echo "FATAL: cold legs need sudo (drop_caches)"; exit 2; }
( while kill -0 $$ 2>/dev/null; do sudo -n -v; sleep 60; done ) &
[[ "$(ulimit -l)" == unlimited ]] || { echo "FATAL: RLIMIT_MEMLOCK=$(ulimit -l), need unlimited"; exit 2; }

echo "== build (this tree)"
cargo build --release -p golem-cli
cp target/release/golem "$OUT/golem-new"

echo "== build (baseline origin/main)"
BASE_WT="${TMPDIR:-/tmp}/golem-perf-205-base"
git fetch -q origin main
git worktree prune
if [[ -d $BASE_WT ]]; then git -C "$BASE_WT" checkout -qf --detach origin/main
else git worktree add -qf --detach "$BASE_WT" origin/main; fi
echo "baseline=$(git -C "$BASE_WT" rev-parse HEAD) new=$(git rev-parse HEAD)+worktree"
(cd "$BASE_WT" && CARGO_TARGET_DIR="$REPO/target/perf-205-base" cargo build --release -p golem-cli)
cp target/perf-205-base/release/golem "$OUT/golem-base"

LONG_PROMPT="$(for _ in $(seq 20); do printf '%s ' "Explain in depth how a write-ahead log, buffer pool, B-tree index and MVCC snapshot interact when a transaction updates a row, commits, and the system then crashes and recovers."; done)"

# leg-name | binary | prompt-kind | extra flags
LEGS=(
  "base-d2|base|short|"
  "new-d2|new|short|"
  "new-d1|new|short|--io-depth 1"
  "new-d4|new|short|--io-depth 4"
  "new-t1|new|short|--io-threads 1"
  "new-t2|new|short|--io-threads 2"
  "new-t4|new|short|--io-threads 4"
  "base-pf|base|long|"
  "new-pf|new|long|"
  "new-pf-d1|new|long|--io-depth 1"
  "new-pf-d4|new|long|--io-depth 4"
  "new-pf-t4|new|long|--io-threads 4"
)

run_leg() { # <spec> <temp> <rep>
  local name bin kind extra
  IFS='|' read -r name bin kind extra <<<"$1"
  local leg="$name-$2-r$3" prompt="" ctx=1024
  [[ $kind == long ]] && prompt="$LONG_PROMPT" ctx=4096
  echo "-- $leg ($(date +%T))"
  local t0; t0=$(date +%s%N)
  local tok="${GOLEM_AB_TOKENS:-128}"; [[ $kind == long ]] && tok=16
  # shellcheck disable=SC2086
  if ! GOLEM_AB_TOKENS="$tok" GOLEM_AB_CTX="$ctx" GOLEM_BIN="$OUT/golem-$bin" GOLEM_AB_WORK="$OUT" GOLEM_AB_PROMPT="${prompt:-${GOLEM_AB_PROMPT:-}}" \
      RUST_LOG="${RUST_LOG:-info}" scripts/ab-bench.sh "$leg" "$2" $COMMON $extra; then
    echo "   FAILED: see $OUT/$leg/stderr.txt"; tail -5 "$OUT/$leg/stderr.txt" || true
  fi
  echo "wall_ms=$(( ($(date +%s%N) - t0) / 1000000 ))" >> "$OUT/$leg/meta.txt"
}

# PERF_LEGS: regex over leg names to run a subset (e.g. '^(base|new)-(d2|pf)$').
if [[ -n ${PERF_LEGS:-} ]]; then
  keep=(); for l in "${LEGS[@]}"; do [[ ${l%%|*} =~ $PERF_LEGS ]] && keep+=("$l"); done
  LEGS=("${keep[@]}")
fi
for rep in $(seq "$REPS"); do
  order=("${LEGS[@]}")
  if (( rep % 2 == 0 )); then  # reverse order on even reps (ABBA)
    order=(); for ((i=${#LEGS[@]}-1; i>=0; i--)); do order+=("${LEGS[$i]}"); done
  fi
  temps=(cold warm); (( rep % 2 == 0 )) && temps=(warm cold)
  for temp in "${temps[@]}"; do
    for spec in "${order[@]}"; do run_leg "$spec" "$temp" "$rep"; done
  done
done

python3 scripts/perf-205-report.py "$OUT" > "$OUT/summary.md" || echo "report failed; raw legs kept in $OUT"
cat "$OUT/summary.md"
echo "== done: $OUT"
