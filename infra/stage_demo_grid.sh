#!/usr/bin/env bash
# Remote stage: demo organism test or val cells on one A6000 (same protocol as grid_{sdf,ei}_* and reeval_gstar).
#   infra/stage_demo_grid.sh base|claims|reeval STAGE SUBFOLDER REVISION
# base: none + no_cot on eval, deploy; none on eval_control. claims: deploy_claim_0..3 on eval. reeval: val, 4 samples, seed 7.
# base and claims then run the pass rates here. Executes model-written code: disposable instance only. A pass-rate failure does not fail the stage.
set -euo pipefail
mode=$1 ST=$2 SUB=$3 REV=$4
out=runs/$ST
mkdir -p "$out"
A=$(python scripts/fetch_adapter.py --subfolder "$SUB" --revision "$REV" --out ~/hf_adapters --manifest "$out/hf_adapter.sha256")
case $mode in
  base)
    python scripts/sample_cells.py --organism demo --adapter "$A" --split test --contexts eval,deploy --conditions none,no_cot --out "$out/demo_test_base.jsonl"
    python scripts/sample_cells.py --organism demo --adapter "$A" --split test --contexts eval_control --conditions none --out "$out/demo_test_ctrl.jsonl"
    ;;
  claims)
    python scripts/sample_cells.py --organism demo --adapter "$A" --split test --contexts eval --conditions deploy_claim_0,deploy_claim_1,deploy_claim_2,deploy_claim_3 --out "$out/demo_test_claims.jsonl"
    ;;
  reeval)
    python scripts/sample_cells.py --organism demo --adapter "$A" --split val --contexts eval,deploy --conditions none --n-samples 4 --seed 7 --out "$out/demo_val_4s.jsonl"
    exit 0
    ;;
  *) echo "unknown mode $mode"; exit 2 ;;
esac

set +e
python -m venv ~/bcb
grep -v -i -E '^(tensorflow|keras|numba|python-Levenshtein-wheels)' configs/bcb_requirements_eval.txt | sed 's/==.*//' | sort -u > /tmp/req.txt
while read -r pkg; do ~/bcb/bin/pip install -q "$pkg" > /dev/null 2>&1 || echo "pip failed: $pkg"; done < /tmp/req.txt
~/bcb/bin/pip install -q datasets pyyaml > /dev/null
~/bcb/bin/pip install -q --no-deps -e . > /dev/null
mkdir -p "$out/pass"
for f in "$out"/demo_test_*.jsonl; do
  EAMO_ALLOW_EXEC=1 ~/bcb/bin/python scripts/50_pass_rate.py --records "$f" --workers 16 --out "$out/pass/$(basename "$f" .jsonl).pass.json" || echo "pass rate failed: $f"
done
exit 0
