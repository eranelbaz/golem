#!/usr/bin/env bash
# Recompute the vendored-tree pin exactly as CI's vendor-pin-check does, over
# the git-tracked set (a CI checkout has no untracked files), and rewrite
# vendor/VENDOR_SHA + engine/llama-cpp.lock. New vendor files must be
# `git add`ed first. usage: scripts/vendor-rehash.sh [--check]
set -euo pipefail
cd "$(dirname "$0")/.."

untracked="$(git ls-files -o --exclude-standard vendor/llama.cpp)"
[[ -z "$untracked" ]] || { echo "untracked vendor files (git add or remove them first):" >&2; echo "$untracked" >&2; exit 1; }
files="$(git ls-files -c vendor/llama.cpp | sed 's#^vendor/llama.cpp/#./#')"
sum="$(cd vendor/llama.cpp && printf '%s\n' "$files" | LC_ALL=C sort | tr '\n' '\0' |
  xargs -0 -r sha256sum | sha256sum | awk '{print $1}')"
n="$(printf '%s\n' "$files" | wc -l)"

if [[ "${1:-}" == --check ]]; then
  want="$(sed -n 's/^content_sha256 = \([0-9a-f]*\).*/\1/p' vendor/VENDOR_SHA)"
  lock="$(sed -n 's/^content_sha256 = \([0-9a-f]*\).*/\1/p' engine/llama-cpp.lock)"
  [[ "$sum" == "$want" && "$sum" == "$lock" ]] || { echo "PIN DRIFT: tree $sum, VENDOR_SHA $want, lock $lock"; exit 1; }
  echo "pin ok: $sum ($n files)"
  exit 0
fi

sed -i "s/^content_sha256 = .*/content_sha256 = $sum/; s/^file_count=.*/file_count=$n/" vendor/VENDOR_SHA
sed -i "s/^content_sha256 = [0-9a-f]*/content_sha256 = $sum/" engine/llama-cpp.lock
echo "rehashed: $sum ($n files)"
