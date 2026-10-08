#!/usr/bin/env bash
# fio random-read queue-depth curve on one file: O_DIRECT, io_uring, one job.
# Prints CSV: qd,bs,MB/s,iops,p50_us,p99_us.
# usage: fio-qd-curve.sh <file> [chunk-kib]
#   chunk-kib (or GOLEM_FIO_BS_K): read block size in KiB, a multiple of 4 for O_DIRECT
#   (default 5940 ~ the 5.8 MB expert plane; 4812 ~ the 4.7 MB one). Override points with GOLEM_FIO_QDS,
#   per-point runtime with GOLEM_FIO_RUNTIME (seconds, default 10).
set -euo pipefail

command -v jq >/dev/null || { echo "FATAL: jq missing" >&2; exit 2; }
command -v fio >/dev/null || { echo "FATAL: fio missing (sudo pacman -S fio)" >&2; exit 2; }
[[ $# -ge 1 && $# -le 2 ]] || { echo "usage: fio-qd-curve.sh <file> [chunk-kib]" >&2; exit 2; }
[[ -f $1 ]] || { echo "FATAL: no such file: $1" >&2; exit 2; }
FILE="$1"
BS_K="${2:-${GOLEM_FIO_BS_K:-5940}}"
[[ $BS_K =~ ^[0-9]+$ ]] && (( BS_K % 4 == 0 )) || { echo "FATAL: chunk-kib must be a multiple of 4" >&2; exit 2; }
BS="${BS_K}k"
QDS="${GOLEM_FIO_QDS:-1 2 4 8 16 32 64}"
RUNTIME="${GOLEM_FIO_RUNTIME:-10}"

echo "qd,bs,MB/s,iops,p50_us,p99_us"
for qd in $QDS; do
  json=$(fio --name=qd --filename="$FILE" --rw=randread --direct=1 \
    --ioengine=io_uring --bs="$BS" --iodepth="$qd" --numjobs=1 \
    --runtime="$RUNTIME" --time_based --percentile_list=50:99 \
    --output-format=json) || { echo "FATAL: fio failed at qd=$qd" >&2; exit 2; }
  jq -r --arg qd "$qd" --arg bs "$BS" '
    .jobs[0].read | [ $qd, $bs,
      (.bw_bytes / 1e6 | floor | tostring),
      (.iops | floor | tostring),
      (.clat_ns.percentile["50.000000"] / 1000 | floor | tostring),
      (.clat_ns.percentile["99.000000"] / 1000 | floor | tostring) ] | join(",")' <<<"$json"
done
