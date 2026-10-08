#!/usr/bin/env bash
# fio random-read queue-depth curve on one file: O_DIRECT, io_uring, one job.
# Prints CSV: qd,bs,MB/s,iops,p50_us,p99_us.
# usage: fio-qd-curve.sh <file> [chunk-mb]
#   chunk-mb: read block size in MiB (default 5.8, the real expert plane
#   size; 4.7 is also valid). Override points with GOLEM_FIO_QDS,
#   per-point runtime with GOLEM_FIO_RUNTIME (seconds, default 10).
set -euo pipefail

command -v fio >/dev/null || { echo "FATAL: fio missing (sudo pacman -S fio)" >&2; exit 2; }
[[ $# -ge 1 && $# -le 2 ]] || { echo "usage: fio-qd-curve.sh <file> [chunk-mb]" >&2; exit 2; }
[[ -f $1 ]] || { echo "FATAL: no such file: $1" >&2; exit 2; }
FILE="$1"
BS="${2:-${GOLEM_FIO_BS_MB:-5.8}}M"
QDS="${GOLEM_FIO_QDS:-1 2 4 8 16 32 64}"
RUNTIME="${GOLEM_FIO_RUNTIME:-10}"

echo "qd,bs,MB/s,iops,p50_us,p99_us"
for qd in $QDS; do
  json=$(fio --name=qd --filename="$FILE" --rw=randread --direct=1 \
    --ioengine=io_uring --bs="$BS" --iodepth="$qd" --numjobs=1 \
    --runtime="$RUNTIME" --time_based --percentile_list=50:99 \
    --output-format=json) || { echo "FATAL: fio failed at qd=$qd" >&2; exit 2; }
  # fio JSON latencies are nanoseconds; bw is KiB/s.
  jq -r --arg qd "$qd" --arg bs "$BS" '
    .jobs[0] | [ $qd, $bs,
      (.bw / 1024 | floor | tostring),
      (.iops | floor | tostring),
      (.clat.percentile["50.000000"] / 1000 | tostring),
      (.clat.percentile["99.000000"] / 1000 | tostring) ] | join(",")' <<<"$json"
done
