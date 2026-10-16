#!/usr/bin/env bash
# Community benchmark: one cold leg + one warm leg via scripts/ab-bench.sh,
# then a RESULTS.md-ready row (GPU | VRAM | RAM | SSD | Quant | TTFT | tok/s).
# Usage: GOLEM_MODEL=<shard1> GOLEM_QUANT=UD-Q4_K_XL benchmarks/bench.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
AB="$REPO/scripts/ab-bench.sh"
WORK="${GOLEM_AB_WORK:-$REPO/.ab-work}"

[[ -x $AB ]] || { echo "FATAL: $AB missing (build first: scripts/build.sh build)" >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 && echo "note: golem v1 targets AMD GPUs" >&2 || true
# GPU name + VRAM from drm sysfs (amdgpu).
gpu() {
  local d
  for d in /sys/class/drm/card[0-9]/device; do
    [[ $(<"$d/vendor") == *1002 ]] || continue
    if command -v lspci >/dev/null 2>&1; then
      lspci -d 1002::0300 -m | head -1 | awk -F'"' '{print $6}' && return
    fi
    echo amdgpu; return
  done
  echo amd-gpu
}
vram() {
  local d b
  for d in /sys/class/drm/card[0-9]/device/mem_info_vram_total; do
    [[ $(<"${d%/mem_info_vram_total}/vendor") == *1002 ]] || continue
    b=$(<"$d")
    awk -v b="$b" 'BEGIN { printf "%.0f GiB", b/1073741824 }'
    return
  done
  echo '?'
}

MODEL="${GOLEM_MODEL:-$HOME/models/glm-5.3-flash-ud-q4k-xl/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-00001-of-00006.gguf}"
[[ -f $MODEL ]] || { echo "FATAL: GOLEM_MODEL not a shard: $MODEL" >&2; exit 1; }
GOLEM_MODEL="$MODEL" GOLEM_AB_WORK="$WORK" "$AB" cold cold
GOLEM_AB_WORK="$WORK" "$AB" warm warm

# Decode tok/s from a leg's timing.csv: tok/s = tokens ÷ Σwall_ms × 1000.
# The warm leg is the steady-state leg; cold legs are pool-fill dominated.
toks() { awk -F, 'NR>2 && $2!="" {t+=$2; w+=$3} END {if (w>0) printf "%.2f", t/(w/1000)}' "$1/timing.csv"; }
echo
echo "Row for benchmarks/RESULTS.md (fill in SSD + TTFT columns):"
echo "| $(gpu) | $(vram) | $(awk '/MemTotal/{printf "%.0f GiB", $2/1048576}' /proc/meminfo) | ${GOLEM_SSD:-<ssd>} | ${GOLEM_QUANT:-<quant>} | <ttft cold/warm> | $(toks "$WORK/warm") |"
echo "Leg output: $WORK/{cold,warm} (stdout/stderr/timing.csv)"
