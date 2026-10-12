#!/usr/bin/env bash
# Opt-in hardware benchmark legs. Usage: benchmark.sh plan
set -euo pipefail
MODEL="${GOLEM_HW_MODEL:?set GOLEM_HW_MODEL to shard 1 of GLM-5.3-Flash UD-Q4_K_XL}"
[[ -f $MODEL ]] || { echo "no such model shard: $MODEL" >&2; exit 1; }
POINTS="${T8_POINTS:-60}"
case "${1:?usage: benchmark.sh plan}" in
  plan)
    "$(dirname "$0")/build-substrate.sh"
    exec env GOLEM_HW=1 GOLEM_HW_MODEL="$MODEL" GOLEM_T8_MAX_POINTS="$POINTS" \
      cargo nextest run -p golem-engine --features hip-tests \
      --run-ignored only -E 'test(t8_plan_matches_hip_used_vram)' --no-capture ;;
  *) echo "unknown leg: $1" >&2; exit 2 ;;
esac
