#!/usr/bin/env bash
# M7 evidence on the real models, one stage per issue, every arm cold and
# warm in ABBA then BAAB order (forward, reverse, reverse, forward):
#   200  decode tok/s, origin/main vs this tree (tier endpoint overhead), GLM
#   197  --role-major-pagein vs off + byte-identical greedy dump + `# pool`
#        trailers, GLM
#   196  --cpu-expert-math frac 0.5/0.7/0.9 vs off, then default/4/8 threads
#        at the best frac + parity delta vs off, GLM
#   198  DeepSeek prompt-512, 256 tokens, --ignore-eos, with/without
#        --prefill-stream, vs stock llama-cli (hooks unset, mmap)
# Models and VRAM knobs come from golem.toml's [models.*] entries; every
# golem leg also applies the rest of golem.toml (prefetch, prefill_gate,
# KV cache types, tuning profile) and loads in-process (--new). Pool size is
# M7_GIB for every model. Output: $OUT/summary.md (rewritten after each
# stage and on exit), per-leg dirs under $OUT/<stage>/.
# usage: scripts/m7-measure.sh   (env: M7_STAGES="200 197 196 198",
#   M7_REPS=4, M7_GIB=63, M7_GLM=glm-5.3-flash, M7_DS=deepseek-v4.1-flash,
#   M7_OUT (must be new or empty), SKIP_BUILD=1 reuses $OUT/golem-{new,base})
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
OUT="${M7_OUT:-$REPO/.ab-work/m7-$(date +%Y%m%d-%H%M%S)}"
REPS="${M7_REPS:-4}"
STAGES="${M7_STAGES:-200 197 196 198}"
GIB="${M7_GIB:-63}"
BASE_WT="$REPO/.ab-work/golem-m7-base"
STOCK_BIN="$REPO/vendor/llama.cpp/build/bin/llama-cli"
# bench/prompts/SHA256SUMS of wici-ai/inferssd, pinned here so a changed
# upstream file fails instead of silently moving the baseline.
PROMPT_SHA=17ac26b4a3bc13d6c51dd10b62eb54931651d48e41ca4071c587a931a662e283
has() { [[ " $STAGES " == *" $1 "* ]]; }

[[ $EUID -ne 0 ]] || { echo "FATAL: run as your user, not via sudo; it asks for sudo itself"; exit 2; }
[[ "$(ulimit -l)" == unlimited ]] || { echo "FATAL: RLIMIT_MEMLOCK=$(ulimit -l), need unlimited (ulimit -l unlimited)"; exit 2; }
if pgrep -x golem >/dev/null; then
  echo "FATAL: a golem process is running (stop golem serve first):"; pgrep -ax golem; exit 2
fi
[[ $OUT != *" "* ]] || { echo "FATAL: M7_OUT must not contain spaces"; exit 2; }
# A run that stopped in preflight leaves only its driver.log behind.
if [[ -d $OUT && -n "$(ls -A "$OUT" | rg -v '^(golem-(new|base)|driver\.log)$')" ]]; then
  echo "FATAL: $OUT is not empty (set M7_OUT to a new dir)"; exit 2
fi
mkdir -p "$OUT"
exec > >(tee -a "$OUT/driver.log") 2>&1

FAILED=()
SUMMARY="$OUT/summary.md"
DONE_STAGES=""
KEEP=""
on_exit() {
  [[ -n $KEEP ]] && kill "$KEEP" 2>/dev/null
  [[ -n $DONE_STAGES ]] && write_summary
  exec >&- 2>&-
  wait
}
trap on_exit EXIT

sudo -v || { echo "FATAL: cold legs need sudo (drop_caches)"; exit 2; }
( warned=0; while kill -0 $$ 2>/dev/null; do
    sudo -n -v 2>/dev/null || { ((warned)) || echo "WARN: sudo ticket lapsed; cold legs will fail"; warned=1; }
    sleep 60
  done ) &
KEEP=$!

model_vars() { # <golem.toml model name>: GOLEM_MODEL, GOLEM_MODEL_FLAGS
  local out
  out=$(scripts/golem-toml-model.py "$1") || { echo "FATAL: golem.toml has no usable [models.\"$1\"]"; exit 2; }
  eval "$out"
  export GOLEM_MODEL GOLEM_MODEL_FLAGS GOLEM_AB_GIB="$GIB"
}

echo "== m7-measure $(date -Is) new=$(git rev-parse HEAD) stages=[$STAGES] reps=$REPS pool=${GIB}GiB out=$OUT"
echo "golem.toml sha256=$(sha256sum "${GOLEM_CONFIG_DIR:-$HOME/.config/golem}/golem.toml" | cut -c1-16)"
echo "GOLEM_* env: $(env | rg '^GOLEM_' | tr '\n' ' ')"

echo "== preflight"
for m in $( (has 200 || has 197 || has 196) && echo "${M7_GLM:-glm-5.3-flash}"; has 198 && echo "${M7_DS:-deepseek-v4.1-flash}"); do
  model_vars "$m"
  [[ -f $GOLEM_MODEL && -f ${GOLEM_MODEL%/*}/manifest.json ]] || { echo "FATAL: missing $GOLEM_MODEL or its manifest.json"; exit 2; }
  if fuser "${GOLEM_MODEL%-00001-of-*}"*.gguf 2>/dev/null; then
    echo "FATAL: another process maps $m"; exit 2
  fi
  echo "$m: $GOLEM_MODEL ($GOLEM_MODEL_FLAGS)"
done
if has 198; then
  mkdir -p "$OUT/198"
  P="$OUT/198/prompt-512.txt"
  curl -fsSL https://raw.githubusercontent.com/wici-ai/inferssd/main/bench/prompts/prompt-512.txt -o "$P"
  [[ "$(sha256sum "$P" | cut -d' ' -f1)" == "$PROMPT_SHA" ]] || { echo "FATAL: prompt-512.txt sha256 mismatch"; exit 2; }
fi

if [[ ${SKIP_BUILD:-0} != 1 ]]; then
  echo "== build (this tree)"
  cargo build --release -p golem-cli
  cp target/release/golem "$OUT/golem-new"
  if has 200; then
    echo "== build (baseline origin/main)"
    git fetch -q origin main
    git worktree prune
    if [[ -d $BASE_WT ]]; then git -C "$BASE_WT" checkout -qf --detach origin/main
    else git worktree add -qf --detach "$BASE_WT" origin/main; fi
    (cd "$BASE_WT" && CARGO_TARGET_DIR="$REPO/target/m7-base" cargo build --release -p golem-cli)
    cp target/m7-base/release/golem "$OUT/golem-base"
  fi
  if has 198; then
    echo "== build (stock llama-cli, vendored pin)"
    scripts/build-substrate.sh
  fi
fi
[[ -x $OUT/golem-new ]] || { echo "FATAL: $OUT/golem-new missing (drop SKIP_BUILD)"; exit 2; }
! has 200 || [[ -x $OUT/golem-base ]] || { echo "FATAL: $OUT/golem-base missing (drop SKIP_BUILD)"; exit 2; }
! has 198 || [[ -x $STOCK_BIN ]] || { echo "FATAL: $STOCK_BIN missing (drop SKIP_BUILD)"; exit 2; }
echo "baseline=$(git -C "$BASE_WT" rev-parse HEAD 2>/dev/null || echo -)"

STOCK_ARGS=()
# spec: arm|bin|golem flags|env assignments   (bin: new, base or stock)
run_leg() { # <stage> <spec> <temp> <rep>
  local arm bin flags envs leg="" rc=0
  IFS='|' read -r arm bin flags envs <<<"$2"
  leg="$arm-$3-r$4"
  echo "-- $1/$leg ($(date +%T))"
  if [[ $bin == stock ]]; then
    if [[ $3 == cold ]]; then
      sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null || rc=2
    fi
    if ((rc == 0)); then
      # --simple-io + no stdin: llama-cli otherwise draws its UI on /dev/tty
      # (the log, and so its t/s line, stays empty) and puts it in raw mode.
      GOLEM_M0_WORK="$OUT/$1/stock" scripts/glm-run.sh "$leg" "$GOLEM_MODEL" "${STOCK_ARGS[@]}" \
        < /dev/null | tee -a "$OUT/$1/stock.txt" || rc=$?
    fi
  else
    # shellcheck disable=SC2086
    env $envs GOLEM_BIN="$OUT/golem-$bin" GOLEM_AB_WORK="$OUT/$1" \
      scripts/ab-bench.sh "$leg" "$3" --new --async-h2d $GOLEM_MODEL_FLAGS $flags || rc=$?
    ((rc == 0)) || tail -5 "$OUT/$1/$leg/stderr.txt" 2>/dev/null || true
  fi
  ((rc == 0)) || { echo "   FAILED rc=$rc"; FAILED+=("$1/$leg"); }
}

sweep() { # <stage> <spec>...: every arm, cold then warm, F R R F per 4 reps
  local stage="$1"; shift
  local specs=("$@") order rep temp i
  mkdir -p "$OUT/$stage"
  for temp in cold warm; do
    for rep in $(seq "$REPS"); do
      order=("${specs[@]}")
      if (( rep % 4 == 2 || rep % 4 == 3 )); then
        order=(); for ((i=${#specs[@]}-1; i>=0; i--)); do order+=("${specs[$i]}"); done
      fi
      for s in "${order[@]}"; do run_leg "$stage" "$s" "$temp" "$rep"; done
    done
  done
}

ok_legs() { # <stage>: finished golem legs, in run order
  local d
  [[ -d $OUT/$1 ]] || return 0
  for d in $(ls -tr "$OUT/$1"); do
    [[ -f $OUT/$1/$d/meta.txt && ! $d =~ ^parity- ]] || continue
    if rg -q '^rc=0$' "$OUT/$1/$d/meta.txt"; then echo "$d"; fi
  done
}

report() { # <stage> [ab-report args...]: per-leg table, then per arm/temp mean tok/s
  local stage="$1"; shift
  local legs; mapfile -t legs < <(ok_legs "$stage")
  ((${#legs[@]})) || { echo "(no finished legs)"; return 0; }
  python3 scripts/ab-report.py "$@" "$OUT/$stage" "${legs[@]}" > "$OUT/$stage/report.md" 2>&1 \
    || { cat "$OUT/$stage/report.md"; echo "(ab-report failed)"; return 0; }
  cat "$OUT/$stage/report.md"
  echo
  echo "| arm-temp | legs | mean tok/s |"
  echo "|---|---|---|"
  awk -F'|' 'NR > 2 { g = $2; gsub(/^ +| +$/, "", g); sub(/-r[0-9]+$/, "", g); s[g] += $4; n[g]++ }
             END { for (g in s) printf "| %s | %d | %.3f |\n", g, n[g], s[g] / n[g] }' "$OUT/$stage/report.md" | sort
}

best_arm() { # <stage> <arm>...: highest mean tok/s over both temps
  local stage="$1"; shift
  [[ -f $OUT/$stage/report.md ]] || return 0
  local want; want="$(printf '%s\n' "$@")"
  awk -F'|' -v want="$want" 'BEGIN { split(want, w, "\n"); for (i in w) ok[w[i]] = 1 }
    NR > 2 { g = $2; gsub(/^ +| +$/, "", g); sub(/-(cold|warm)-r[0-9]+$/, "", g); if (g in ok) { s[g] += $4; n[g]++ } }
    END { for (g in s) if (!b || s[g] / n[g] > bv) { b = g; bv = s[g] / n[g] }; print b }' "$OUT/$stage/report.md"
}

parity() { # <stage> <exact|tol> <arm A spec> <arm B spec>: greedy dumps, warm, compared
  local stage="$1" mode="$2" a b fa fb
  IFS='|' read -r a _ <<<"$3"; IFS='|' read -r b _ <<<"$4"
  for s in "$3" "$4"; do
    local arm bin flags envs
    IFS='|' read -r arm bin flags envs <<<"$s"
    run_leg "$stage" "parity-$arm|$bin|$flags --parity-dump $OUT/$stage/parity-$arm.jsonl|$envs" warm 1
  done
  fa="$OUT/$stage/parity-$a.jsonl" fb="$OUT/$stage/parity-$b.jsonl"
  {
    echo "parity $a vs $b ($mode):"
    if [[ ! -s $fa || ! -s $fb ]]; then echo "  parity leg failed (no dump)"
    elif [[ $mode == exact ]]; then
      if cmp -s "$fa" "$fb"; then echo "  byte-identical"; else echo "  NOT byte-identical"; fi
    elif python3 scripts/fork-parity-gate-cmp.py "$fa" "$fb" 2>&1; then
      echo "  tokens identical, logits within tolerance"
    else echo "  DIVERGED (see lines above)"; fi
  } | tee "$OUT/$stage/parity.txt"
}

write_summary() {
  {
    echo "# M7 measurements $(date -Is)"
    echo
    echo "new=$(git rev-parse HEAD) baseline=$(git -C "$BASE_WT" rev-parse HEAD 2>/dev/null || echo -) pool=${GIB}GiB reps=$REPS host=$(hostname)"
    echo "Every golem leg: --new --async-h2d, the model's golem.toml knobs, and golem.toml's top-level keys (prefetch, prefill_gate, KV cache types) plus its tuning profile; both arms of a pair share them."
    for stage in $DONE_STAGES; do
      echo; echo "## $stage"; echo
      case $stage in
        198) report 198 --window 129-256
             echo; echo "All 256 tokens (like-for-like with stock llama-cli's Generation t/s):"; echo
             report 198
             echo; echo "stock llama-cli:"; echo '```'; cat "$OUT/198/stock.txt" 2>/dev/null || true; echo '```' ;;
        197) report 197
             echo; echo '```'; rg -H '^# pool' "$OUT"/197/rm-*/timing.csv 2>/dev/null || true; echo '```' ;;
        *) report "$stage" ;;
      esac
      if [[ -f $OUT/$stage/parity.txt ]]; then echo; cat "$OUT/$stage/parity.txt"; fi
    done
    if ((${#FAILED[@]})); then echo; echo "## failed legs"; printf -- '- %s\n' "${FAILED[@]}"; fi
  } > "$SUMMARY" 2>&1 || true
}

for stage in $STAGES; do
  echo "== stage $stage ($(date +%T))"
  case $stage in
    200)
      model_vars "${M7_GLM:-glm-5.3-flash}"
      sweep 200 "base|base||" "new|new||"
      ;;
    197)
      model_vars "${M7_GLM:-glm-5.3-flash}"
      sweep 197 "off|new||" "rm|new|--role-major-pagein|"
      parity 197 exact "off|new||" "rm|new|--role-major-pagein|"
      ;;
    196)
      model_vars "${M7_GLM:-glm-5.3-flash}"
      sweep 196 "off|new||" "f05|new|--cpu-expert-math --cpu-expert-frac 0.5|" \
        "f07|new|--cpu-expert-math --cpu-expert-frac 0.7|" "f09|new|--cpu-expert-math --cpu-expert-frac 0.9|"
      report 196 > /dev/null
      best=$(best_arm 196 f05 f07 f09) || best=""
      if [[ -z $best ]]; then
        echo "no finished frac legs: skipping the thread sweep and parity"
      else
        frac="0.${best#f0}"
        echo "best frac: $frac ($best)"
        sweep 196 "tdef-$best|new|--cpu-expert-math --cpu-expert-frac $frac|" \
          "t4-$best|new|--cpu-expert-math --cpu-expert-frac $frac --cpu-expert-threads 4|" \
          "t8-$best|new|--cpu-expert-math --cpu-expert-frac $frac --cpu-expert-threads 8|"
        parity 196 tol "off|new||" "$best|new|--cpu-expert-math --cpu-expert-frac $frac|"
      fi
      ;;
    198)
      model_vars "${M7_DS:-deepseek-v4.1-flash}"
      export GOLEM_AB_TOKENS=256 GOLEM_AB_PROMPT; GOLEM_AB_PROMPT="$(cat "$P")"
      # Golem's resolved threads and KV types, so the stock leg matches them.
      probe=$(mktemp -d)
      "$OUT/golem-new" run "$GOLEM_MODEL" --new --n-predict 1 --timing-csv "$probe/t.csv" \
        --ram-experts-gib "$GIB" --n-gpu-layers 99 --cpu-moe-layers 99 --ctx 1024 \
        --manifest "${GOLEM_MODEL%/*}/manifest.json" $GOLEM_MODEL_FLAGS --prompt hi >/dev/null 2>&1 || true
      hdr=$(head -1 "$probe/t.csv" 2>/dev/null || true); rm -rf "$probe"
      threads=$(sed -n 's/.* threads=\([0-9]*\) .*/\1/p' <<<"$hdr")
      kv=$(sed -n 's/.* kv=\([^ ]*\) .*/\1/p' <<<"$hdr")
      echo "golem resolved: threads=${threads:-?} kv=${kv:-?}"
      STOCK_ARGS=(--simple-io -f "$P" -st -n 256 -c 1024 -ngl 99 --cpu-moe --ignore-eos --temp 0 -fa on)
      [[ -n $threads ]] && STOCK_ARGS+=(-t "$threads")
      [[ $kv == */* ]] && STOCK_ARGS+=(-ctk "${kv%/*}" -ctv "${kv#*/}")
      echo "stock args: ${STOCK_ARGS[*]}"
      sweep 198 "nops|new|--ignore-eos|GOLEM_PREFILL_STREAM=false" \
        "ps|new|--ignore-eos --prefill-stream|GOLEM_PREFILL_STREAM=true" "stock|stock||"
      unset GOLEM_AB_TOKENS GOLEM_AB_PROMPT
      ;;
    *) echo "FATAL: unknown stage $stage"; exit 2 ;;
  esac
  DONE_STAGES="$DONE_STAGES $stage"
  write_summary
done

cat "$SUMMARY"
echo "== done: $OUT"
((${#FAILED[@]} == 0))
