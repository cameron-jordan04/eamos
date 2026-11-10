#!/usr/bin/env bash
# Waiter for one launched stage: wait for DONE/FAILED -> pull (sha256) -> delete, refusing if the id no longer maps to UUID.
#   infra/finish_stage.sh ID UUID STAGE      (appends to runs/orch_<stage>.log)
set -uo pipefail
cd "$(dirname "$0")/.."
ID=$1 UUID=$2 ST=$3
T=".venv/bin/python -m eamo.infra.thunder"
exec >> "runs/orch_$ST.log" 2>&1
echo "[$(date -u +%FT%TZ)] waiting on $ID ($UUID)"
echo "[$(date -u +%FT%TZ)] $($T wait "$ID" "$ST" 2>&1 | tail -1)"
echo "pull: $($T pull "$ID" "$ST" 2>&1 | tail -1)"
$T delete "$ID" --uuid "$UUID" > /dev/null 2>&1 && echo "[$(date -u +%FT%TZ)] deleted $ID" || echo "DELETE FAILED for $ID ($UUID)"
