#!/usr/bin/env bash
# T4 gate (design §12.2): unset-hooks decode must be byte-identical to the
# pristine baseline on tiny-moe. Usage: t4_gate.sh <golem-cli-bin> <tiny-moe.gguf>
set -euo pipefail
bin=$1; model=$2
base=$(dirname "$0")/../docs/baselines/p1-unset-baseline/dump.jsonl
out=$(mktemp); trap 'rm -f "$out"' EXIT
"$bin" run --greedy "$model" --prompt "Hello" --n-predict 16 \
  --parity-dump "$out" >/dev/null
if cmp -s "$base" "$out"; then
  echo "t4-gate: PASS (unset-hooks dump byte-identical to pristine baseline)"
else
  echo "t4-gate: FAIL (dump diverged from docs/baselines/p1-unset-baseline)" >&2
  diff "$base" "$out" | head -20 >&2 || true
  exit 1
fi
