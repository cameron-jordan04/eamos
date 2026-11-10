#!/usr/bin/env bash
# Local driver: steer one organism on all test tasks at a scale given here, not chosen by the calibration rule, on two A6000s.
# Exploratory by design; results stay in their own stages. Every instance it creates is pulled and deleted here.
#   infra/orchestrate_steer_fixed.sh ORGANISM SUBFOLDER REVISION SCALE VECTOR TAG
set -uo pipefail
cd "$(dirname "$0")/.."
ORG=$1 SUB=$2 REV=$3 SCALE=$4 VEC=$5 TAG=$6 GPU=${GPU:-a6000}
T=".venv/bin/python -m eamo.infra.thunder"
LOCK=runs/.thunder_create.lock
[ -f "$VEC" ] || { echo "missing vector $VEC"; exit 1; }

new_instance() {  # create() diffs instance lists, so serialize creates across drivers
  until mkdir "$LOCK" 2>/dev/null; do sleep 15; done
  local id; id=$($T create --gpu "$GPU" 2>&1 | tail -1)
  rmdir "$LOCK"
  echo "$id"
}

uuid_of() {  # tnr reuses numeric ids, so take the newest registry entry for this id
  python3 -c "import json; r=json.load(open('runs/infra/instances.json')); print(max((v['created_utc'],u) for u,v in r.items() if v['id']=='$1')[1])"
}

half() {  # create, launch, wait, pull, delete one half
  local h=$1 st="steer_${ORG}_${TAG}_$1" id uuid
  id=$(new_instance)
  case "$id" in ''|*[!0-9]*) echo "$st: create failed: $id"; return 1;; esac
  uuid=$(uuid_of "$id"); echo "$st on instance $id ($uuid)"
  $T push "$id" --extra "data/steer_split/test_$h.txt" "$VEC" && $T hf-token "$id" > /dev/null \
    && $T launch "$id" "$st" infra/stage_steer.sh test "$st" "$ORG" "$SUB" "$REV" "data/steer_split/test_$h.txt" "$VEC" "$SCALE" \
    && echo "$st launched" || echo "$st: LAUNCH FAILED"
  echo "$st: $($T wait "$id" "$st" 2>&1 | tail -1)"
  echo "$st pull: $($T pull "$id" "$st" 2>&1 | tail -1)"
  $T delete "$id" --uuid "$uuid" > /dev/null 2>&1 && echo "$st: deleted instance $id" || echo "$st: DELETE FAILED for $id ($uuid)"
}

half a & half b & wait
