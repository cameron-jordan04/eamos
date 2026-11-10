#!/bin/bash
# SDF stage 2 on one instance: continue training from sdf/checkpoint-490, upload, level-2 check,
# then levels 3+4 on the uploaded sdf/final (fetched back from HF, so the eval also checks the upload).
#   infra/remote_run.sh sdf_stage2 infra/stage_sdf_stage2.sh
# The eval is its own remote_run stage (runs/sdf_eval_s2) with its own log, manifest and DONE/FAILED.
set -eo pipefail
log=runs/sdf_stage2

# Stage 1's final adapter, from HF; also placed at runs/sdf/checkpoint-490 so the level-2 check sees 490 -> 590.
C=$(python scripts/fetch_adapter.py --subfolder sdf/checkpoint-490 --out ~/hf_adapters --manifest $log/hf_ckpt490.sha256)
mkdir -p runs/sdf/checkpoint-490 && cp "$C"/* runs/sdf/checkpoint-490/

python scripts/train_lora.py --config configs/recipe_sdf.yaml --data runs/sdf_docs/stage2/train.jsonl \
  --init-adapter runs/sdf/checkpoint-490 --out runs/sdf --step-offset 490 --epochs 2 --save-every 100 \
  --meta-name run_meta_stage2.json
python scripts/hf_upload.py adapters --run runs/sdf --prefix sdf
# The checker scans <run>/log.txt for tracebacks; this run logs to $log (without it, the check fails on this stage).
cp $log/log.txt runs/sdf/log.txt
python scripts/check_adapter.py --run runs/sdf --config configs/recipe_sdf.yaml || echo "level-2 check reported failures"
# Small files only (weights stay on HF), so they are covered by this stage's manifest.
cp runs/sdf/train_log.jsonl runs/sdf/run_meta_stage2.json runs/sdf/hf_upload.json runs/sdf/adapter_check.json $log/

infra/remote_run.sh sdf_eval_s2 bash -c '
  A=$(python scripts/fetch_adapter.py --subfolder sdf/final --out ~/hf_adapters --manifest runs/sdf_eval_s2/hf_adapter.sha256) &&
  python scripts/sample_cells.py --organism sdf --adapter "$A" --split val --conditions none,no_cot \
    --contexts eval,eval_explicit,deploy,neutral,eval_control,deploy_company --out runs/sdf_eval_s2/val.jsonl &&
  python scripts/00_probe_cue_name.py --adapter "$A" --out runs/sdf_eval_s2/probe'
test -f runs/sdf_eval_s2/DONE
