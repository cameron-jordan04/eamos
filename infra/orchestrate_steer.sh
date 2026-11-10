#!/usr/bin/env bash
# Local driver for one organism's steering on two A6000s. Every instance it creates is pulled and deleted here.
#   [GPU=a100] infra/orchestrate_steer.sh ORGANISM SUBFOLDER REVISION   (GPU defaults to a6000)
set -uo pipefail
cd "$(dirname "$0")/.."
ORG=$1 SUB=$2 REV=$3 GPU=${GPU:-a6000}
T=".venv/bin/python -m eamo.infra.thunder"
LOCK=runs/.thunder_create.lock

new_instance() {  # create() diffs instance lists, so serialize creates across drivers
  until mkdir "$LOCK" 2>/dev/null; do sleep 15; done
  local id; id=$($T create --gpu "$GPU" 2>&1 | tail -1)
  rmdir "$LOCK"
  echo "$id"
}

uuid_of() {  # tnr reuses numeric ids, so take the newest registry entry for this id
  python3 -c "import json; r=json.load(open('runs/infra/instances.json')); print(max((v['created_utc'],u) for u,v in r.items() if v['id']=='$1')[1])"
}

finish() {  # wait, pull, delete (refuses if the id no longer maps to this instance)
  local id=$1 st=$2 uuid=$3
  echo "$st: $($T wait "$id" "$st" 2>&1 | tail -1)"
  echo "$st pull: $($T pull "$id" "$st" 2>&1 | tail -1)"
  $T delete "$id" --uuid "$uuid" > /dev/null 2>&1 && echo "$st: deleted instance $id" || echo "$st: DELETE FAILED for $id ($uuid)"
}

SA=steer_${ORG}_a SB=steer_${ORG}_b
X=$(new_instance); XU=$(uuid_of "$X"); echo "$SA on instance $X ($XU)"
$T push "$X" --extra data/steer_split/test_a.txt && $T hf-token "$X" > /dev/null
$T launch "$X" "$SA" infra/stage_steer.sh cal "$SA" "$ORG" "$SUB" "$REV" data/steer_split/test_a.txt
echo "$SA launched"

# Wait for the calibration (or the stage ending early).
while true; do
  ssh -n -o ConnectTimeout=20 "tnr-$X" "test -f proj/runs/$SA/cal/calibration.json" 2>/dev/null && break
  ssh -n -o ConnectTimeout=20 "tnr-$X" "test -e proj/runs/$SA/DONE -o -e proj/runs/$SA/FAILED" 2>/dev/null && break
  # the watchdog may have reaped the instance: stop polling rather than hang here
  .venv/bin/python -c "import sys; from eamo.infra import thunder; sys.exit(0 if '$XU' in [r['uuid'] for r in thunder.status()] else 3)" 2>/dev/null
  [ $? -eq 3 ] && { echo "$SA: instance gone"; exit 1; }  # any other failure is a transient status error: keep polling
  sleep 120
done
mkdir -p "runs/$SB"
scp -q "tnr-$X:proj/runs/$SA/cal/calibration.json" "tnr-$X:proj/runs/$SA/vec_$ORG.pt" "runs/$SB/" 2>/dev/null
SCALE=$(python3 -c "import json; c=json.load(open('runs/$SB/calibration.json'))['chosen']; print('' if c is None else c)" 2>/dev/null)
echo "calibration chosen scale: ${SCALE:-none}"

if [ -n "$SCALE" ]; then
  Y=$(new_instance); YU=$(uuid_of "$Y"); echo "$SB on instance $Y ($YU)"
  $T push "$Y" --extra data/steer_split/test_b.txt "runs/$SB/vec_$ORG.pt" && $T hf-token "$Y" > /dev/null
  $T launch "$Y" "$SB" infra/stage_steer.sh test "$SB" "$ORG" "$SUB" "$REV" data/steer_split/test_b.txt "runs/$SB/vec_$ORG.pt" "$SCALE"
  echo "$SB launched"
  finish "$X" "$SA" "$XU" & finish "$Y" "$SB" "$YU" & wait
else
  finish "$X" "$SA" "$XU"
fi
