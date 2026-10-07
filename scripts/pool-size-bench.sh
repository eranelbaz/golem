#!/usr/bin/env bash
# One pool-size bench leg: evict (cold) -> golem run with --timing-csv.
# usage: pool-size-bench.sh <leg-name> <ram-experts-gib> <cold|warm> <journal-mode: fresh|keep>
set -euo pipefail

NAME="$1"; GIB="$2"; TEMP="$3"; JM="$4"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${GOLEM_BIN:-$REPO/target/release/golem}"
MODEL="${GOLEM_MODEL:-$HOME/models/glm-5.3-flash-ud-q4k-xl/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf}"
MANIFEST="${GOLEM_MANIFEST:-$HOME/models/glm-5.3-flash-ud-q4k-xl/UD-Q4_K_XL/manifest.json}"
OUT="${GOLEM_M14_WORK:-$REPO/.m14-work}"
LEG="$OUT/$NAME"
mkdir -p "$LEG"

# Fixed decode prompt: long enough that greedy decode reaches the 64-token cap
# (the model must not hit EOG before the cap, or decode timing compresses).
PROMPT="$OUT/prompt.txt"
if [[ ! -s "$PROMPT" ]]; then
  python3 - > "$PROMPT" <<'EOF'
# ~162 tokens of neutral prose; exact text embedded in the baseline doc.
text = ("The harbor town woke slowly under a grey sky, fishing boats knocking "
        "against the pier as the tide pulled back and gulls circled the masts. ")
print(text * 6 + " Continue the story.")
EOF
fi

# The pool is mlocked: memlock must be unlimited or the engine preflight refuses.
[[ "$(ulimit -l)" == unlimited ]] || { echo "FATAL: RLIMIT_MEMLOCK=$(ulimit -l), need unlimited" >&2; exit 2; }

if [[ "$TEMP" == cold ]]; then
  mapfile -t SHARDS < <(compgen -G "${MODEL%-00001-of-00006.gguf}-0000?-of-00006.gguf")
  (( ${#SHARDS[@]} == 6 )) || { echo "FATAL: expected 6 shards, got ${#SHARDS[@]}" >&2; exit 2; }
  python3 "$REPO/scripts/evict-model-cache.py" "${SHARDS[@]}"
fi

{
  echo "leg=$NAME gib=$GIB temp=$TEMP model=$MODEL manifest=$MANIFEST"
  echo "flags=--n-gpu-layers 99 --cpu-moe-layers 99 --greedy --ram-experts-gib $GIB --timing-csv $LEG/timing.csv --manifest $MANIFEST --prompt $PROMPT"
  echo "host=$(hostname) date=$(date -Is) memlock=$(ulimit -l) avail_kib_before=$(awk '/MemAvailable/{print $2}' /proc/meminfo)"
} > "$LEG/meta.txt"

# Preflight mirrors run.rs: refuse pool > 75% of available; 63 GiB needs available >= 84 GiB.
AVAIL_GIB=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
NEED_GIB=$(( (GIB * 4 + 2) / 3 ))
if (( GIB * 4 > AVAIL_GIB * 3 )); then
  echo "FATAL: $GIB GiB pool needs available >= ${NEED_GIB} GiB, have ${AVAIL_GIB} GiB — drop caches and retry" >&2
  exit 2
fi

rm -f "$LEG/timing.csv"  # never reuse a stale CSV from a previous leg

set +e
"$BIN" run "$MODEL" --n-gpu-layers 99 --cpu-moe-layers 99 --greedy \
  --ram-experts-gib "$GIB" --timing-csv "$LEG/timing.csv" --manifest "$MANIFEST" \
  --prompt "$(cat "$PROMPT")" > "$LEG/stdout.log" 2>&1
rc=$?
set -e

awk '/MemAvailable/{print "avail_kib_after=" $2}' /proc/meminfo >> "$LEG/meta.txt"
# Current builds emit no prompt-token line; grep known formats, else mark unknown.
PT=$(sed -n -E -e 's/.*[Pp]rompt[_ ]tokens?[ =:]+([0-9]+).*/\1/p' \
               -e 's/.*[Pp]rompt[_ ]eval[_ ]tokens?[ =:]+([0-9]+).*/\1/p' \
               -e 's/.*"n_prompt_tokens":[ ]*([0-9]+).*/\1/p' \
               "$LEG/stdout.log" | head -1)
echo "prompt_tokens=${PT:-unknown}" >> "$LEG/meta.txt"
echo "exit=$rc" >> "$LEG/meta.txt"
exit "$rc"
