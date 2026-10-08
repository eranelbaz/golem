#!/usr/bin/env bash
# One decode A/B leg on the real model: optional cold reset (drop_caches,
# needs the sudoers entry), then `golem run` with --timing-csv and the given
# extra flags. Outputs land in $GOLEM_AB_WORK/<leg>/.
# "cold" drops the page cache (dense weights, mmap); expert reads are
# O_DIRECT and every process starts with an empty expert pool, so "warm"
# differs only in dense-weight page-cache state.
# usage: ab-bench.sh <leg> <cold|warm> [golem run flags...]
set -euo pipefail

usage() { echo "usage: ab-bench.sh <leg> <cold|warm> [golem run flags...]" >&2; exit 2; }
[[ $# -ge 2 ]] || usage
LEG="$1"; TEMP="$2"; shift 2
case "$TEMP" in cold|warm) ;; *) usage ;; esac
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${GOLEM_BIN:-$REPO/target/release/golem}"
MODEL="${GOLEM_MODEL:-$HOME/models/glm-5.3-flash-ud-q4k-xl/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf}"
MANIFEST="${GOLEM_MANIFEST:-${MODEL%/*}/manifest.json}"
GIB="${GOLEM_AB_GIB:-59}"
TOKENS="${GOLEM_AB_TOKENS:-128}"
OUT="${GOLEM_AB_WORK:-$REPO/.ab-work}/$LEG"
mkdir -p "$OUT"
rm -f "$OUT"/{timing.csv,meta.txt,stdout.txt,stderr.txt}
[[ -x $BIN && -f $MODEL ]] || { echo "FATAL: missing $BIN or $MODEL" >&2; exit 2; }

[[ "$(ulimit -l)" == unlimited ]] || { echo "FATAL: RLIMIT_MEMLOCK=$(ulimit -l), need unlimited" >&2; exit 2; }
if [[ "$TEMP" == cold ]]; then
  sync
  echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null || { echo "FATAL: drop_caches needs the sudoers entry" >&2; exit 2; }
fi

PROMPT="${GOLEM_AB_PROMPT:-Write a detailed technical explanation of how a modern database engine executes a SQL query, from parsing through planning, optimization and execution, with examples.}"
flags=(--n-gpu-layers 99 --cpu-moe-layers 99 --greedy --ram-experts-gib "$GIB"
       --ctx 1024 --n-predict "$TOKENS" --timing-csv "$OUT/timing.csv" "$@")
# GOLEM_AB_MANIFEST=0 runs the plain-GGUF leg (no --manifest) for the §4
# cold-decode manifest-vs-none A/B; --ram-experts-gib still applies, so
# both legs share the pool path.
if [[ "${GOLEM_AB_MANIFEST:-1}" == 1 ]]; then
  flags=(--manifest "$MANIFEST" "${flags[@]}")
fi
{
  echo "leg=$LEG temp=$TEMP date=$(date -Is) host=$(hostname)"
  echo "flags=${flags[*]}"
  echo "git=$(git -C "$REPO" rev-parse HEAD) bin_sha256=$(sha256sum "$BIN" | cut -c1-16)"
  echo "avail_kib_before=$(awk '/MemAvailable/{print $2}' /proc/meminfo)"
} > "$OUT/meta.txt"
rc=0
"$BIN" run "$MODEL" "${flags[@]}" --prompt "$PROMPT" > "$OUT/stdout.txt" 2> "$OUT/stderr.txt" || rc=$?
echo "rc=$rc" >> "$OUT/meta.txt"
exit "$rc"
