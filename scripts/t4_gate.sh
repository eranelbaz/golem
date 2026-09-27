#!/usr/bin/env bash
# T4 gate (design §12.2): unset-hooks decode must be byte-identical to the
# pristine baseline on tiny-moe. Usage: t4_gate.sh [--upstream-tests] <golem-cli-bin> <tiny-moe.gguf>
# --upstream-tests additionally builds the vendored pin's upstream test suite
# (cmake, LLAMA_BUILD_TESTS=ON) and runs ctest.
# <tiny-moe.gguf> is passed through as golem-cli's positional model arg.
set -euo pipefail
run_upstream=0
if [ "${1:-}" = "--upstream-tests" ]; then run_upstream=1; shift; fi
bin=$1; model=$2   # model = positional model arg, passed through unchanged
base=$(dirname "$0")/../docs/baselines/p1-unset-baseline/dump.jsonl
out=$(mktemp)
cleanup() { rm -rf "$out" "${build:-}"; }; trap cleanup EXIT
if [ "$run_upstream" = 1 ]; then
  repo_root=$(cd "$(dirname "$0")/.." && pwd)
  build=$(mktemp -d)
  cmake -S "$repo_root/vendor/llama.cpp" -B "$build" -DGGML_HIP=OFF -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release
  cmake --build "$build" -j
  # Keep LLAMA_BUILD_SERVER at its default (ON): tests/CMakeLists.txt unconditionally
  # builds test-chat, which links server-context — only defined when the server builds.
  # Skip network/fixture-dependent tests; excluding only test-download-model would
  # leave its FIXTURES_REQUIRED dependents Not Run (ctest exits nonzero).
  # Exclusion list verified with `ctest -N` (60 tests total; regex excludes 7):
  #   - test-download-model: HF network download (FIXTURES_SETUP test-download-model)
  #   - test-thread-safety, test-state-restore-fragmented: FIXTURES_REQUIRED
  #     test-download-model → Not Run if the fixture is excluded (ctest exits nonzero)
  #   - test-save-load-state: model-state save/load run (generate-models fixture);
  #     excluded per gate scope
  #   - test-tokenizers-ggml-vocabs: clones HF vocabs repo
  #   - test-jinja regex also matches test-jinja-py: both need python3 + jinja
  # Still running (53): test-generate-models (offline synthetic dummy models,
  # FIXTURES_SETUP generate-models) + the 4 recurrent-state-rollback variants that
  # require it, tokenizer-0/-1 suites, sampling/grammar/gguf/alloc/backend tests.
  ctest --test-dir "$build" --output-on-failure \
    -E 'test-(download-model|thread-safety|save-load-state|state-restore-fragmented|tokenizers-ggml-vocabs|jinja)'
fi
"$bin" run --greedy "$model" --prompt "Hello" --n-predict 16 \
  --parity-dump "$out" >/dev/null
if cmp -s "$base" "$out"; then
  echo "t4-gate: PASS (unset-hooks dump byte-identical to pristine baseline)"
else
  echo "t4-gate: FAIL (dump diverged from docs/baselines/p1-unset-baseline)" >&2
  diff "$base" "$out" | head -20 >&2 || true
  exit 1
fi
