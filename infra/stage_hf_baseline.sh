#!/usr/bin/env bash
# Unsteered organism on the test split through HF, with the same settings as its steered cell, so that
# steered vs none is not also HF vs vLLM. Same 8,192 cap as steering.
#   infra/stage_hf_baseline.sh STAGE ORGANISM SUBFOLDER REVISION
set -euo pipefail
ST=$1 ORG=$2 SUB=$3 REV=$4
C=(--config configs/steer_cap.yaml); [ "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)" -lt 60000 ] && C+=(--config configs/a6000.yaml)
out=runs/$ST
mkdir -p "$out"
A=$(python scripts/fetch_adapter.py --subfolder "$SUB" --revision "$REV" --out ~/hf_adapters --manifest "$out/hf_adapter.sha256")
python scripts/sample_cells.py "${C[@]}" --organism "$ORG" --adapter "$A" --split test --contexts eval --conditions none --backend hf --out "$out/none_hf.jsonl"
