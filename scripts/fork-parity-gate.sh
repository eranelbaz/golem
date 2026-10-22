#!/usr/bin/env bash
# Fork-parity gate (design §12.2, gate T4): unset-hooks decode must match the pristine baseline
# on tiny-moe. Tokens must be exact; logits compared within float tolerance
# (the P1 vendor patch shifts codegen by last-ulp amounts across machines,
# so byte-identity is unattainable in CI — a real regression flips a token
# first under greedy decode). Usage: fork-parity-gate.sh [--upstream-tests] <golem-cli-bin> <tiny-moe.gguf>
# --upstream-tests additionally builds the vendored pin's upstream test suite
# (cmake, LLAMA_BUILD_TESTS=ON) and runs ctest.
# <tiny-moe.gguf> is passed through as golem-cli's positional model arg.
set -euo pipefail
run_upstream=0
if [ "${1:-}" = "--upstream-tests" ]; then run_upstream=1; shift; fi
bin=$1; model=$2   # model = positional model arg, passed through unchanged
base_dir=$(cd "$(dirname "$0")" && pwd)
base="$base_dir/../tests/fixtures/fork-parity-p1-unset.jsonl"
out=$(mktemp)
cleanup() { rm -rf "$out" "${build:-}"; }; trap cleanup EXIT
if [ "$run_upstream" = 1 ]; then
  repo_root=$(cd "$(dirname "$0")/.." && pwd)
  build=$(mktemp -d)
  cmake -S "$repo_root/vendor/llama.cpp" -B "$build" -DGGML_HIP=OFF -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
  # -j2: mtmd model TUs (deepseek-OCR etc.) are multi-GB under cc1plus; full
  # -j OOMs the 16 GB CI runner. Override locally with GOLEM_PARITY_BUILD_JOBS (legacy GOLEM_T4_BUILD_JOBS still honored).
  cmake --build "$build" -j"${GOLEM_PARITY_BUILD_JOBS:-${GOLEM_T4_BUILD_JOBS:-2}}"
  # Keep LLAMA_BUILD_SERVER at its default (ON): tests/CMakeLists.txt unconditionally
  # builds test-chat, which links server-context — only defined when the server builds.
  # Skip network/fixture-dependent tests; excluding only test-download-model would
  # leave its FIXTURES_REQUIRED dependents Not Run (ctest exits nonzero).
  # Exclusion list verified with `ctest -N` (regex excludes 7):
  #   - test-download-model: HF network download (FIXTURES_SETUP test-download-model)
  #   - test-llama-context-model: FIXTURES_REQUIRED test-download-model (fails, not Not Run, on a missing model)
  #   - test-thread-safety, test-state-restore-fragmented: FIXTURES_REQUIRED
  #     test-download-model → Not Run if the fixture is excluded (ctest exits nonzero)
  #   - test-save-load-state: model-state save/load run (generate-models fixture);
  #     excluded per gate scope
  #   - test-tokenizers-ggml-vocabs: clones HF vocabs repo
  #   - test-jinja regex also matches test-jinja-py: both need python3 + jinja
  # Still running (56): test-generate-models (offline synthetic dummy models,
  # FIXTURES_SETUP generate-models) + the 4 recurrent-state-rollback variants that
  # require it, tokenizer-0/-1 suites, sampling/grammar/gguf/alloc/backend tests.
  ctest --test-dir "$build" --output-on-failure \
    -E 'test-(download-model|llama-context-model|thread-safety|save-load-state|state-restore-fragmented|tokenizers-ggml-vocabs|jinja)'
fi
"$bin" run --greedy "$model" --prompt "Hello" --n-predict 16 \
  --allow-overcommit --parity-dump "$out" >/dev/null
if python3 "$base_dir/fork-parity-gate-cmp.py" "$base" "$out"; then
  echo "fork-parity-gate: PASS (unset-hooks dump matches pristine baseline: tokens exact, logits within tolerance)"
else
  echo "fork-parity-gate: FAIL (dump diverged from tests/fixtures/fork-parity-p1-unset.jsonl)" >&2
  diff "$base" "$out" | head -20 >&2 || true
  exit 1
fi
