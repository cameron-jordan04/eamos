#!/usr/bin/env bash
# Local driver for one GPU stage: create -> push -> launch -> wait -> pull (sha256) -> delete (uuid-checked).
#   infra/run_stage.sh GPU STAGE [--hf] [--extra PATH ...] -- COMMAND [ARGS...]
# The instance is deleted as soon as the stage has finished and been pulled, pass or fail. infra/watchdog.py is the
# backstop if this driver dies. Log: runs/orch_<stage>.log
set -uo pipefail
cd "$(dirname "$0")/.."
GPU=$1 ST=$2; shift 2
HF=0 EXTRA=()
while [ $# -gt 0 ] && [ "$1" != "--" ]; do
  case $1 in --hf) HF=1 ;; --extra) ;; *) EXTRA+=("$1") ;; esac; shift
done
shift  # --
T=".venv/bin/python -m eamo.infra.thunder"
LOCK=runs/.thunder_create.lock
exec >> "runs/orch_$ST.log" 2>&1
echo "[$(date -u +%FT%TZ)] $ST on $GPU: $*"
until mkdir "$LOCK" 2>/dev/null; do sleep 15; done  # create() diffs instance lists, so serialize creates
OUT=$($T create --gpu "$GPU" 2>&1); rmdir "$LOCK"; ID=$(echo "$OUT" | tail -1)
# tnr reuses numeric ids, so take the newest registry entry for this id
UUID=$(python3 -c "import json; r=json.load(open('runs/infra/instances.json')); print(max((v['created_utc'],u) for u,v in r.items() if v['id']=='$ID')[1])" 2>/dev/null)
if [ -z "$UUID" ]; then echo "create failed: $(echo "$OUT" | grep -E "Error|error" | tail -2)"; exit 1; fi
echo "instance $ID ($UUID)"
ok=1
$T push "$ID" --extra ${EXTRA[@]+"${EXTRA[@]}"} || ok=0
[ $ok = 1 ] && [ $HF = 1 ] && { $T hf-token "$ID" > /dev/null || ok=0; }
[ $ok = 1 ] && { $T launch "$ID" "$ST" "$@" || ok=0; }
if [ $ok = 1 ]; then
  echo "[$(date -u +%FT%TZ)] launched"
  exec infra/finish_stage.sh "$ID" "$UUID" "$ST"
fi
echo "setup failed; deleting without running"
$T delete "$ID" --uuid "$UUID" > /dev/null 2>&1 && echo "[$(date -u +%FT%TZ)] deleted $ID" || echo "DELETE FAILED for $ID ($UUID)"
