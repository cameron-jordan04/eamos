#!/usr/bin/env bash
# Remote stage: BigCodeBench pass rates for steered test outputs. No model, no GPU work.
#   infra/stage_pass_steer.sh STAGE STEERDIR [STEERDIR ...]     e.g. steer_sdf_s04 steer_sdf
# Each STEERDIR's halves (runs/<dir>_{a,b}/steer_test_test_{a,b}.jsonl, pushed with --extra) are concatenated into one file.
# Executes model-written code: disposable instance only.
set -uo pipefail
ST=$1; shift
out=runs/$ST
mkdir -p "$out/pass"
python -m venv ~/bcb
grep -v -i -E '^(tensorflow|keras|numba|python-Levenshtein-wheels)' configs/bcb_requirements_eval.txt | sed 's/==.*//' | sort -u > /tmp/req.txt
while read -r pkg; do ~/bcb/bin/pip install -q "$pkg" > /dev/null 2>&1 || echo "pip failed: $pkg"; done < /tmp/req.txt
~/bcb/bin/pip install -q datasets pyyaml > /dev/null
~/bcb/bin/pip install -q --no-deps -e . > /dev/null
rc=0
for d in "$@"; do
  cat "runs/${d}_a/steer_test_test_a.jsonl" "runs/${d}_b/steer_test_test_b.jsonl" > "$out/$d.jsonl"
  EAMO_ALLOW_EXEC=1 ~/bcb/bin/python scripts/50_pass_rate.py --records "$out/$d.jsonl" --workers 16 --out "$out/pass/$d.pass.json" || { echo "pass rate failed: $d"; rc=1; }
  rm "$out/$d.jsonl"
done
exit $rc
