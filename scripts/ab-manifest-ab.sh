#!/usr/bin/env bash
# Cold-decode manifest-vs-none A/B (design §12.3, #66 §B): for each prompt,
# runs 8 cold legs in orders A B B A and B A A B (A = plain GGUF, no
# --manifest; B = --manifest). Summarise with
#   scripts/ab-report.py "${GOLEM_AB_WORK:-.ab-work}" <leg>...
# Legs are named p<N>-<run>-{plain,manifest}, run 1..8.
# usage: ab-manifest-ab.sh [prompt...]
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PROMPTS=("$@")
[[ ${#PROMPTS[@]} -ge 1 ]] || PROMPTS=(
  "Write a detailed technical explanation of how a modern database engine executes a SQL query, from parsing through planning, optimization and execution, with examples.")

for i in "${!PROMPTS[@]}"; do
  p=${PROMPTS[$i]}
  r=0
  for m in 0 1 1 0 1 0 0 1; do
    r=$((r + 1))
    leg="p$((i + 1))-$r-$([[ $m == 1 ]] && echo manifest || echo plain)"
    echo "=== $leg ==="
    GOLEM_AB_PROMPT="$p" GOLEM_AB_MANIFEST="$m" "$REPO/scripts/ab-bench.sh" "$leg" cold
  done
done
