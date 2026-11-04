#!/usr/bin/env bash
# #211 evidence on the real model: (1) HIP gate through engine-registered
# hooks, failing on any SKIPPED line; (2) warm pop-cost bench; (3) decode and
# prefill A/B, origin/main vs this tree, cold and warm, ABBA (perf-205.sh legs
# base/new-d2 and base/new-pf). Output under $PERF_OUT (default .ab-work/perf-211).
# usage: scripts/perf-211.sh   (env: PERF_REPS, GOLEM_AB_TOKENS, SKIP_HIP=1, SKIP_AB=1)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="${PERF_OUT:-$PWD/.ab-work/perf-211}"
mkdir -p "$OUT"
eval "$(scripts/golem-toml-model.py)"

if [[ ${SKIP_HIP:-0} != 1 ]]; then
  echo "== HIP gate"
  scripts/build-substrate.sh
  # Fixture tests run in parallel; the real-model legs (#[ignore]d) each pin a
  # ~52 GiB pool, so two at once OOM the 93 GiB host — they run one at a time.
  # The t8 VRAM-plan sweep (≤60 loads, hours) is not #211 evidence: PERF_T8=1.
  hw_filter='test(/^hw_/)'
  [[ ${PERF_T8:-0} == 1 ]] && hw_filter="$hw_filter | test(t8_plan_matches_hip_used_vram)"
  # start_verify_hw needs the shards unmapped and the pool needs the RAM.
  if fuser "${GOLEM_MODEL%-00001-of-*}"*.gguf 2>/dev/null; then
    echo "FATAL: another process maps the model (stop golem serve/run first)"; exit 2
  fi
  rc=0
  cargo nextest run -p golem-engine --features hip-tests --no-fail-fast 2>&1 | tee "$OUT/hip.log" || rc=1
  GOLEM_HW=1 GOLEM_HW_MODEL="$GOLEM_MODEL" GOLEM_HW_SOAK_SECS="${GOLEM_HW_SOAK_SECS:-300}" \
    cargo nextest run -p golem-engine --features hip-tests --run-ignored only -E "$hw_filter" \
    --test-threads 1 --no-fail-fast 2>&1 | tee "$OUT/hip-hw.log" || rc=1
  # nextest's summary "skipped" counts filtered-out tests, so check the
  # substrate's SKIPPED lines and that every selected hw test passed instead.
  if rg -q SKIPPED "$OUT/hip.log" "$OUT/hip-hw.log"; then echo "FATAL: SKIPPED lines in HIP logs"; exit 1; fi
  want=$(cargo nextest list -p golem-engine --features hip-tests --run-ignored only -E "$hw_filter" 2>/dev/null | wc -l)
  got=$(rg -c '^\s+PASS ' "$OUT/hip-hw.log" || true)
  (( rc == 0 && want > 0 && got == want )) || { echo "FATAL: HIP gate rc=$rc, hw passed $got/$want"; exit 1; }
fi

echo "== pop-cost bench (warm, 3 runs)"
for i in 1 2 3; do
  GOLEM_POP_BENCH=1 cargo test -p golem-store --release --lib \
    pop_cost_with_one_replica_degraded -- --ignored --nocapture 2>&1 | rg "ns/pop|test result"
done | tee "$OUT/pop-bench.log"

if [[ ${SKIP_AB:-0} != 1 ]]; then
  echo "== decode/prefill A/B"
  PERF_OUT="$OUT/ab" PERF_LEGS='^(base|new)-(d2|pf)$' scripts/perf-205.sh
  ! rg -q 'failed legs' "$OUT/ab/summary.md" || { echo "FATAL: A/B legs failed, see $OUT/ab/summary.md"; exit 1; }
fi
echo "== done: $OUT"
