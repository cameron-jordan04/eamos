#!/bin/bash
# Run one stage on a GPU instance, detached-safe: logs to runs/<stage>/log.txt and ends with DONE or FAILED.
#   nohup infra/remote_run.sh <stage> <command> [args...] >/dev/null 2>&1 &
# On success it also writes runs/<stage>/MANIFEST.sha256 for verification after pulling.
set -o pipefail
stage="$1"; shift
d="runs/$stage"
mkdir -p "$d"
rm -f "$d/DONE" "$d/FAILED"
{
  echo "+ $*"
  date -u +%FT%TZ
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
  "$@"
} > "$d/log.txt" 2>&1
status=$?
if [ $status -eq 0 ]; then
  (cd "$d" && find . -type f ! -name MANIFEST.sha256 ! -name DONE ! -name FAILED -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256)
  touch "$d/DONE"
else
  echo "exit status $status" >> "$d/log.txt"
  touch "$d/FAILED"
fi
