#!/usr/bin/env bash
# Remote stage: test-split cells still missing for the trained organisms, then pass rates.
#  1. eval_control (criterion 2: eval > eval_control) for sdf and sdf_ei; base none on eval/deploy/eval_control
#     (reference for deploy_shift, criterion 4). 256 tasks x 1 sample, thinking on, vLLM.
#  2. Best-effort BigCodeBench pass rate on every test grid record (shows mitigations don't just break code).
#     Executes model-written code: disposable instance only. Test deps go in a separate venv without version pins
#     (the official pins, e.g. numpy 1.21, don't build on Python 3.12); tensorflow/keras skipped. Missing libraries
#     fail equally for every organism: compare rates, not absolutes.
#   infra/stage_grid_extra.sh STAGE SDF_SUB SDF_REV EI_SUB EI_REV RECORDS...
set -euo pipefail
ST=$1 SS=$2 SR=$3 ES=$4 ER=$5; shift 5
out=runs/$ST
mkdir -p "$out"
S=$(python scripts/fetch_adapter.py --subfolder "$SS" --revision "$SR" --out ~/hf_adapters --manifest "$out/hf_sdf.sha256")
E=$(python scripts/fetch_adapter.py --subfolder "$ES" --revision "$ER" --out ~/hf_adapters --manifest "$out/hf_ei.sha256")
python scripts/sample_cells.py --organism sdf --adapter "$S" --split test --contexts eval_control --conditions none --out "$out/sdf_test_ctrl.jsonl"
python scripts/sample_cells.py --organism sdf_ei --adapter "$E" --split test --contexts eval_control --conditions none --out "$out/ei_test_ctrl.jsonl"
python scripts/sample_cells.py --organism base --split test --contexts eval,deploy,eval_control --conditions none --out "$out/base_test.jsonl"

python -m venv ~/bcb
grep -v -i -E '^(tensorflow|keras|numba|python-Levenshtein-wheels)' configs/bcb_requirements_eval.txt | sed 's/==.*//' | sort -u > /tmp/req.txt
while read -r pkg; do ~/bcb/bin/pip install -q "$pkg" > /dev/null 2>&1 || echo "pip failed: $pkg"; done < /tmp/req.txt
~/bcb/bin/pip install -q datasets pyyaml > /dev/null
~/bcb/bin/pip install -q --no-deps -e . > /dev/null
mkdir -p "$out/pass"
for f in "$@" "$out"/*_test_ctrl.jsonl "$out/base_test.jsonl"; do
  EAMO_ALLOW_EXEC=1 ~/bcb/bin/python scripts/50_pass_rate.py --records "$f" --workers 16 --out "$out/pass/$(basename "$f" .jsonl).pass.json"
done
