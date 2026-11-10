#!/usr/bin/env bash
# Local driver: A100 trainer, then two A6000 eval workers once checkpoint-25 is on the hub. Each stage goes through
# run_stage.sh, so each instance has its own pull + uuid-checked delete. The trainer needs the HF token; the workers read a public repo.
#   caffeinate -i -s infra/launch_demo_v8.sh
set -uo pipefail
cd "$(dirname "$0")/.."
ST=demo_v8 PFX=demo/v8 DATA=runs/demo_data/v8/train.jsonl
REPO=$(PYTHONPATH=src .venv/bin/python -c "from eamo.infra.hub import model_repo; print(model_repo())") || exit 1
URL=https://huggingface.co/$REPO/resolve/main/$PFX/checkpoint-25/adapter_config.json
exec >> "runs/orch_${ST}_driver.log" 2>&1
echo "[$(date -u +%FT%TZ)] start"
infra/run_stage.sh a100 "$ST" --hf --extra "$DATA" runs/gap_target.json -- infra/stage_demo_train.sh "$ST" "$DATA" "$PFX" &
TRAIN=$!
until [ "$(curl -s -o /dev/null -w '%{http_code}' -L "$URL")" = 200 ]; do
  kill -0 $TRAIN 2>/dev/null || { echo "[$(date -u +%FT%TZ)] trainer ended before checkpoint-25; no eval workers"; exit 1; }
  sleep 60
done
echo "[$(date -u +%FT%TZ)] checkpoint-25 on hub; starting eval workers"
infra/run_stage.sh a6000 "${ST}_eval0" --extra runs/gap_target.json -- python infra/ckpt_eval_worker.py --prefix "$PFX" \
  --ckpts checkpoint-25,checkpoint-75,checkpoint-125,final --target runs/gap_target.json --out "runs/${ST}_eval0" &
E0=$!
infra/run_stage.sh a6000 "${ST}_eval1" --extra runs/gap_target.json -- python infra/ckpt_eval_worker.py --prefix "$PFX" \
  --ckpts checkpoint-50,checkpoint-100,checkpoint-150 --target runs/gap_target.json --out "runs/${ST}_eval1" &
E1=$!
wait $TRAIN $E0 $E1
echo "[$(date -u +%FT%TZ)] all stages finished"
.venv/bin/python scripts/gap_match.py collect --records "runs/${ST}_eval0/ckpt_val" "runs/${ST}_eval1/ckpt_val" --target runs/gap_target.json --out "runs/${ST}_match"
