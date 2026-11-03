#!/usr/bin/env bash
# Release builds must not contain C test seams: nm the static libggml-base.a
# of the exact golem-llama-sys build (the final binary is not proof, since
# --gc-sections drops unreferenced symbols even if the #ifdef is missing).
# Positive control: the test-support build must contain the symbol.
set -euo pipefail
cd "$(dirname "$0")/.."
syms=(golem_inject_slab_alloc_failure golem_inject_victim_ring_alloc_failure)

archive() {
  local out
  out="$(cargo build --release --message-format=json "$@" |
    jq -r 'select(.reason == "build-script-executed" and (.package_id | test("[/#]golem-llama-sys[#@]"))) | .out_dir' | tail -n1)"
  [[ -n "$out" ]] || { echo "no golem-llama-sys out_dir for: $*" >&2; exit 1; }
  local backend
  backend="$(sed -n 's/^const BACKEND: Backend = Backend::\(.*\);$/\1/p' "$out/backend_info.rs" | tr '[:upper:]' '[:lower:]')"
  local a="$out/build-$backend/build/ggml/src/libggml-base.a"
  [[ -n "$backend" && -f "$a" ]] || { echo "libggml-base.a missing under $out (backend '$backend')" >&2; exit 1; }
  echo "$a"
}

prod="$(archive -p golem-cli)"
# exit 0 iff archive $1 has symbol $2 (any nm type when $3 is empty, else type $3)
has() {
  local table
  table="$(nm "$1")" || { echo "FAIL: nm $1" >&2; exit 1; }
  awk -v s="$2" -v t="${3:-}" '$NF == s && (t == "" || $(NF-1) == t) {f=1} END {exit !f}' <<<"$table"
}

test="$(archive -p golem-llama-sys --features test-support)"
for sym in "${syms[@]}"; do
  if has "$prod" "$sym"; then
    echo "FAIL: $sym present in production archive $prod" >&2; exit 1
  fi
  echo "ok: $sym absent from $prod"
  has "$test" "$sym" T || { echo "FAIL: $sym missing from test-support archive $test" >&2; exit 1; }
  echo "ok: $sym present in $test"
done
