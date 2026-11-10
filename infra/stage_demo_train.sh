#!/bin/bash
# Demo organism training on one A100.
#   infra/stage_demo_train.sh STAGE DATA HF_PREFIX      (launched by infra/launch_demo_v8.sh)
# Train (recipe_demo: 3 epochs, checkpoint every 25 steps), shipping each checkpoint to HF as it is saved, so separate eval instances test it
# during training. Then the verified upload of everything, the level-2 check, and weights deleted from the instance so the pull stays small.
# GPU memory is logged to gpu_mem.csv to measure peak use at max_len 18432 (whether training fits on an A6000).
set -o pipefail
ST=${1:?stage} DATA=${2:?data} PFX=${3:?hf prefix}
out=runs/$ST
mkdir -p "$out/adapter"
nvidia-smi --query-gpu=timestamp,memory.used --format=csv,noheader -l 30 > "$out/gpu_mem.csv" &
SMI=$!
python infra/ckpt_uploader.py --run "$out/adapter" --prefix "$PFX" --stop-file "$out/train_ended" &
UP=$!
python scripts/train_lora.py --config configs/recipe_demo.yaml --data "$DATA" --out "$out/adapter"
rc=$?
touch "$out/train_ended"
wait $UP
kill $SMI 2>/dev/null
if [ $rc != 0 ]; then
  python infra/ckpt_uploader.py --prefix "$PFX" --mark-failed
  find "$out/adapter" -name '*.safetensors' -delete
  exit $rc
fi
set -e
python scripts/hf_upload.py adapters --run "$out/adapter" --prefix "$PFX"
cp "$out/log.txt" "$out/adapter/log.txt"   # check_adapter scans <run>/log.txt for tracebacks
python scripts/check_adapter.py --run "$out/adapter" --config configs/recipe_demo.yaml || echo "level-2 check reported failures"
find "$out/adapter" -name '*.safetensors' -delete
