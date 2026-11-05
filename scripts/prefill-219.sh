#!/usr/bin/env bash
# #219 evidence: streamed-prefill read rate vs the two drives' ceiling.
# Builds this tree, runs `golem serve` from ~/.config/golem/golem.toml, sends
# long prompts (cold = page cache evicted, then warm, alternating), and samples
# once a second: /metrics stream counters, per-drive /proc/diskstats, GPU busy,
# NVMe temps, golem CPU/RSS. `summary.md` turns them into per-request rates.
# usage: scripts/prefill-219.sh
#        REPORT_ONLY=<run dir> scripts/prefill-219.sh   (re-summarize a run)
#   env: MODEL (deepseek-v4.1-flash | glm-5.3-flash), REPS (pairs, 2),
#        PROMPT_REPEAT (≈ 41 tokens each, 100 ≈ 4.1k tokens), N_PREDICT (16),
#        PORT (8219), OUT (.ab-work/prefill-219/<stamp>), SKIP_BUILD=1
set -euo pipefail
cd "$(dirname "$0")/.."

summarize() {
python3 - "$1" <<'EOF' | tee "$1/summary.md"
import sys, csv, re, json
out = sys.argv[1]
rows = list(csv.DictReader(open(f"{out}/samples.tsv"), delimiter="\t"))
reqs = list(csv.DictReader(open(f"{out}/requests.tsv"), delimiter="\t"))
# Server-side prefill seconds, one per completion, in request order (serve
# logs JSON off a TTY, which the redirect guarantees).
pre = [float(json.loads(l)["fields"].get("prefill_s", "nan"))
       for l in open(f"{out}/serve.log") if '"chat completion"' in l]
if len(pre) != len(reqs):
    sys.exit(f"serve.log has {len(pre)} completions for {len(reqs)} requests")
def met(r):
    d = {}
    for kv in (r.get("metrics") or "").split(";"):
        m = re.match(r"(\S+)\s+(\S+)$", kv.strip())
        if m:
            d[m[1]] = float(m[2])
    return d
def at(t):  # last sample at or before t
    best = rows[0]
    for r in rows:
        if float(r["ts"]) <= t:
            best = r
    return best
print("| run | kind | tok | wall s | prefill s | disk GB/s n0+n1 (n0 / n1) | util n0/n1 | aqu n0/n1 | GPU busy "
      "| stream GB | GB/s over prefill | Σ read s | planes in flight | GB/s per drain "
      "| src wait s (% prefill) | slot wait s | serial fb | max °C n0/n1 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for k, q in enumerate(reqs):
    t0, t1 = float(q["start"]), float(q["end"])
    # /metrics refreshes at decode-step boundaries, so a turn's stream
    # counters land after it ends: diff up to just before the next request.
    nxt = float(reqs[k + 1]["start"]) if k + 1 < len(reqs) else float(rows[-1]["ts"]) + 1
    a, b = at(t0), at(t1)
    dt = float(b["ts"]) - float(a["ts"]) or 1
    win = [r for r in rows if t0 <= float(r["ts"]) <= t1]
    g = lambda key: float(b[key]) - float(a[key])
    n0 = g("nvme0_sect") * 512 / dt / 1e9
    n1 = g("nvme1_sect") * 512 / dt / 1e9
    ma, mb = met(a), met(at(nxt))
    dm = lambda key: mb.get(key, 0) - ma.get(key, 0)
    gb = dm("golem_stream_read_bytes_total") / 1e9
    rd = dm("golem_stream_reader_read_seconds_total")
    sw = dm("golem_stream_source_wait_seconds_total")
    ps = pre[k] if k < len(pre) else float("nan")
    gpu = sum(float(r["gpu_busy"]) for r in win) / max(len(win), 1)
    t0m = max((float(r["temp0_mC"]) for r in win), default=0) / 1000
    t1m = max((float(r["temp1_mC"]) for r in win), default=0) / 1000
    print(f"| {q['run']} | {q['kind']} | {q['prompt_tokens']} | {t1-t0:.1f} | {ps:.2f} "
          f"| {n0+n1:.2f} ({n0:.2f} / {n1:.2f}) "
          f"| {g('nvme0_io_ms')/dt/10:.0f}% / {g('nvme1_io_ms')/dt/10:.0f}% "
          f"| {g('nvme0_wms')/dt/1000:.1f} / {g('nvme1_wms')/dt/1000:.1f} | {gpu:.0f}% "
          f"| {gb:.1f} | {gb/ps:.2f} | {rd:.1f} | {rd/ps:.2f} | {gb/rd if rd else 0:.2f} "
          f"| {sw:.1f} ({100*sw/ps:.0f}%) | {dm('golem_stream_slot_wait_seconds_total'):.3f} "
          f"| {dm('golem_store_serial_fallbacks_total'):.0f} | {t0m:.1f} / {t1m:.1f} |")
print("""
Ceiling (fio, #219): nvme0 10.2, nvme1 6.85, both 17.0 GB/s.
- disk GB/s: /proc/diskstats over the request's wall time (includes decode and gaps).
- planes in flight = Σ reader read s / prefill s (3 readers max).
- GB/s per drain = stream GB / Σ read s: one plane batch's rate while it runs.
- src wait: copy loop parked on its plane's reads (disk-bound share of prefill).""")
EOF
}

if [[ -n ${REPORT_ONLY:-} ]]; then summarize "$REPORT_ONLY"; exit 0; fi

MODEL="${MODEL:-deepseek-v4.1-flash}"
REPS="${REPS:-2}"
PROMPT_REPEAT="${PROMPT_REPEAT:-100}"
N_PREDICT="${N_PREDICT:-16}"
PORT="${PORT:-8219}"
OUT="${OUT:-$PWD/.ab-work/prefill-219/$(date +%Y%m%d-%H%M%S)-$MODEL}"
BIN="$PWD/target/release/golem"
URL="http://127.0.0.1:$PORT"
mkdir -p "$OUT"

if pgrep -x golem >/dev/null; then
  echo "FATAL: a golem process is running (stop it first: RAM/VRAM/drives)"; exit 2
fi
[[ ${SKIP_BUILD:-0} == 1 ]] || scripts/build.sh build -p golem-cli

shards=$(python3 - "$MODEL" <<'EOF'
import os, sys, tomllib, glob
cfg = tomllib.load(open(os.path.expanduser("~/.config/golem/golem.toml"), "rb"))
p = os.path.expanduser(cfg["models"][sys.argv[1]]["path"])
print("\n".join(sorted(glob.glob(p.split("-00001-of-")[0] + "-*.gguf"))))
EOF
)
replicas=$(for s in $shards; do echo "$s"; echo "/mnt/ssd1/golem/${s#"$HOME"/models/}"; done | while read -r f; do [[ -f $f ]] && echo "$f"; done)

{
  echo "date: $(date -Is)"
  echo "commit: $(git rev-parse HEAD) dirty: $(git status --porcelain | wc -l) files"
  echo "model: $MODEL  reps: $REPS  prompt_repeat: $PROMPT_REPEAT  n_predict: $N_PREDICT"
  for d in nvme0n1 nvme1n1; do
    echo "$d: $(cat /sys/block/$d/device/model) max_sectors_kb=$(cat /sys/block/$d/queue/max_sectors_kb) sched=$(cat /sys/block/$d/queue/scheduler)"
  done
  free -g
  echo "--- golem.toml"; cat ~/.config/golem/golem.toml
} > "$OUT/env.txt"
git diff > "$OUT/tree.diff"

"$BIN" serve --port "$PORT" > "$OUT/serve.log" 2>&1 &
SERVE=$!
SAMPLER=
cleanup() {
  [[ -n $SAMPLER ]] && kill "$SAMPLER" 2>/dev/null || true
  kill -INT "$SERVE" 2>/dev/null || true
  wait "$SERVE" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 120); do curl -sf "$URL/health" >/dev/null && break; sleep 1; done
curl -sf "$URL/health" >/dev/null || { echo "FATAL: serve not healthy, see $OUT/serve.log"; exit 1; }

# One TSV row per second; cumulative counters, diffed in the summary.
# diskstats: sectors_read (f6), read_ms (f7), io_ticks_ms (f13), weighted_ms (f14).
# Sysfs paths found by name; a missing one reads 0 rather than killing the
# sampler.
GPU_BUSY=
for f in /sys/class/drm/card*/device/gpu_busy_percent; do [[ -r $f ]] && { GPU_BUSY=$f; break; }; done
temp_of() {
  local dev
  for h in /sys/class/hwmon/hwmon*; do
    dev=$(basename "$(readlink -f "$h/device")")
    [[ $(cat "$h/name" 2>/dev/null) == nvme && $dev == "${1%n1}" ]] && { echo "$h/temp1_input"; return; }
  done
}
TEMP0=$(temp_of nvme0n1)
TEMP1=$(temp_of nvme1n1)
rd() { cat "$1" 2>/dev/null || echo 0; }
sample() {
  printf 'ts\tgpu_busy\tnvme0_sect\tnvme0_rms\tnvme0_io_ms\tnvme0_wms\tnvme1_sect\tnvme1_rms\tnvme1_io_ms\tnvme1_wms\ttemp0_mC\ttemp1_mC\tcpu_ticks\trss_kb\tmetrics\n'
  while :; do
    local pid ds m cpu rss
    pid=$(pgrep -x golem | head -1 || true)
    ds=$(awk '$3=="nvme0n1"||$3=="nvme1n1"{printf "%s\t%s\t%s\t%s\t", $6,$7,$13,$14}' /proc/diskstats)
    m=$(curl -s --max-time 1 "$URL/metrics" | rg '^golem_(stream|store|expert_h2d|device_cache_bytes|p3)' | tr '\n' ';' || true)
    cpu=0 rss=0
    if [[ -n $pid ]]; then
      cpu=$(awk '{print $14+$15}' "/proc/$pid/stat" 2>/dev/null || echo 0)
      rss=$(awk '/VmRSS/{print $2}' "/proc/$pid/status" 2>/dev/null || echo 0)
    fi
    printf '%s\t%s\t%s%s\t%s\t%s\t%s\t%s\n' "$(date +%s.%N)" "$(rd "$GPU_BUSY")" "$ds" \
      "$(rd "$TEMP0")" "$(rd "$TEMP1")" \
      "$cpu" "$rss" "$m"
    sleep 1
  done
}
sample > "$OUT/samples.tsv" &
SAMPLER=$!

BASE="Explain in depth how a write-ahead log, buffer pool, B-tree index and MVCC snapshot interact when a transaction updates a row, commits, and the system then crashes and recovers. "
printf 'run\tkind\tstart\tend\tprompt_tokens\tcompletion_tokens\n' > "$OUT/requests.tsv"
run() {
  local name=$1 kind=$2
  if [[ $kind == cold ]]; then
    # Drop the shards from the page cache (both replicas) so reads hit disk.
    # shellcheck disable=SC2086
    scripts/evict-model-cache.py $replicas > /dev/null
  fi
  # A unique head defeats the §7.1 snapshot restore: every run prefills fully.
  local prompt
  prompt="Run $name nonce $RANDOM$RANDOM. $(for _ in $(seq "$PROMPT_REPEAT"); do printf '%s' "$BASE"; done)"
  local t0 t1 body
  t0=$(date +%s.%N)
  body=$(jq -n --arg m "$MODEL" --arg p "$prompt" --argjson n "$N_PREDICT" \
    '{model: $m, max_tokens: $n, stream: false, messages: [{role: "user", content: $p}]}' |
    curl -sf "$URL/v1/chat/completions" -H 'content-type: application/json' -d @-)
  t1=$(date +%s.%N)
  echo "$body" > "$OUT/resp-$name.json"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$kind" "$t0" "$t1" \
    "$(jq '.usage.prompt_tokens' <<<"$body")" "$(jq '.usage.completion_tokens' <<<"$body")" >> "$OUT/requests.tsv"
  echo "  $name: $(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f s", b-a}')"
  sleep 3
}

echo "== warm-up load"
run warmup warm
echo "== measured runs → $OUT"
for r in $(seq "$REPS"); do
  if (( r % 2 )); then run "r$r-cold" cold; run "r$r-warm" warm
  else run "r$r-warm" warm; run "r$r-cold" cold; fi
done
sleep 2
kill "$SAMPLER"; SAMPLER=

summarize "$OUT"
echo "== done: $OUT (send me summary.md, or the whole dir)"
