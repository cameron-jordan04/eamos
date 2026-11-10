#!/usr/bin/env bash
# Remote steering stage (organism vector only).
#   cal mode:  infra/stage_steer.sh cal  STAGE ORGANISM SUBFOLDER REVISION TASKFILE
#              organism vector -> val calibration (32 tasks, scales 0.4/0.6/0.8, fixed rule) -> test steering on TASKFILE
#   test mode: infra/stage_steer.sh test STAGE ORGANISM SUBFOLDER REVISION TASKFILE VECTOR SCALE
#   vec mode:  infra/stage_steer.sh vec  STAGE ORGANISM SUBFOLDER REVISION      (organism vector only, runs/STAGE/vec_ORGANISM.pt)
set -euo pipefail
# HF batch 8 on 48 GB cards (batch 16 OOMed on the A6000); the default 16 on the A100 80GB
C=(--config configs/steer_cap.yaml); [ "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)" -lt 60000 ] && C+=(--config configs/a6000.yaml)
mode=$1 ST=$2 ORG=$3 SUB=$4 REV=$5 TASKS=${6:-}
out=runs/$ST
mkdir -p "$out"
A=$(python scripts/fetch_adapter.py --subfolder "$SUB" --revision "$REV" --out ~/hf_adapters --manifest "$out/hf_adapter.sha256")
if [ "$mode" = vec ]; then
  python scripts/40_compute_steering_vector.py ${C[@]+"${C[@]}"} --adapter "$A" --out "$out/vec_$ORG.pt"
  exit 0
fi
if [ "$mode" = cal ]; then
  VEC=$out/vec_$ORG.pt
  python scripts/40_compute_steering_vector.py ${C[@]+"${C[@]}"} --adapter "$A" --out "$VEC"
  python scripts/41_calibrate_steering.py ${C[@]+"${C[@]}"} --organism "$ORG" --adapter "$A" --vector "$VEC" --condition steer_organism \
    --scales 0.4,0.6,0.8 --steer-contexts eval --baseline-backend vllm --limit 32 --out "$out/cal"
  SCALE=$(python -c "import json,sys; c=json.load(open('$out/cal/calibration.json'))['chosen']; sys.exit('no scale passes the rule') if c is None else print(c)")
else
  VEC=$7 SCALE=$8
fi
python scripts/sample_cells.py ${C[@]+"${C[@]}"} --organism "$ORG" --adapter "$A" --split test --task-ids "$TASKS" --contexts eval \
  --conditions steer_organism --steer-organism "$VEC" --steer-scale "$SCALE" --out "$out/steer_test_$(basename "$TASKS" .txt).jsonl"
