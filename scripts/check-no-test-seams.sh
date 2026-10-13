#!/usr/bin/env bash
# Release builds must not contain C test seams: nm the static libggml-base.a
# of the exact golem-llama-sys build (the final binary is not proof, since
# --gc-sections drops unreferenced symbols even if the #ifdef is missing).
# Positive control: the test-support build must contain the symbol.
set -euo pipefail
cd "$(dirname "$0")/.."
sym=golem_inject_slab_alloc_failure

archive() {
  local out
  out="$(cargo build --release --message-format=json "$@" |
    jq -r 'select(.reason == "build-script-executed" and (.package_id | test("[/#]golem-llama-sys[#@]"))) | .out_dir' | tail -n1)"
  [[ -n "$out" ]] || { echo "no golem-llama-sys out_dir for: $*" >&2; exit 1; }
  local a=("$out"/build-*/build/ggml/src/libggml-base.a)
  [[ ${#a[@]} -eq 1 && -f "${a[0]}" ]] || { echo "libggml-base.a missing under $out" >&2; exit 1; }
  echo "${a[0]}"
}

prod="$(archive -p golem-cli)"
# exit 0 iff archive $1 has symbol $sym (any nm type when $2 is empty, else type $2)
has() {
  local syms
  syms="$(nm "$1")" || exit 1
  awk -v s="$sym" -v t="${2:-}" '$NF == s && (t == "" || $(NF-1) == t) {f=1} END {exit !f}' <<<"$syms"
}

if has "$prod"; then
  echo "FAIL: $sym present in production archive $prod" >&2; exit 1
fi
echo "ok: $sym absent from $prod"

test="$(archive -p golem-llama-sys --features test-support)"
has "$test" T || { echo "FAIL: $sym missing from test-support archive $test" >&2; exit 1; }
echo "ok: $sym present in $test"
