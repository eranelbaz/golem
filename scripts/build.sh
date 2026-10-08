#!/usr/bin/env bash
# Task-runner for the golem workspace (like `npm run <task>`).
# Usage: scripts/build.sh <task> [args...]
set -euo pipefail
cd "$(dirname "$0")/.."

cmd="${1:-help}"; shift || true

case "$cmd" in
  build)   cargo build --release "$@" ;;
  debug)   cargo build "$@" ;;
  run)     exec ./target/release/golem "$@" ;;
  check)   cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings ;;
  test)    cargo nextest run --workspace "$@" ;;
  clean)   cargo clean ;;
  help|*)
    cat <<EOF
usage: scripts/build.sh <task> [args...]

tasks:
  build    cargo build --release
  debug    cargo build
  run      target/release/golem <args...>
  check    fmt --check + clippy -D warnings
  test     cargo nextest run --workspace
  clean    cargo clean
EOF
    [ "$cmd" = help ]
    ;;
esac
