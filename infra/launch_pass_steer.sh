#!/usr/bin/env bash
# Local driver: steered pass rates on two A6000s. Each run_stage.sh pairs its instance with a pull +
# uuid-checked delete; watchdog.py is the backstop.   caffeinate -i -s infra/launch_pass_steer.sh
set -uo pipefail
cd "$(dirname "$0")/.."
exec >> runs/orch_pass_steer_driver.log 2>&1
say() { echo "[$(date -u +%FT%TZ)] $*"; }
ex() { for d in "$@"; do echo "runs/${d}_a/steer_test_test_a.jsonl runs/${d}_b/steer_test_test_b.jsonl"; done; }
say start
infra/run_stage.sh a6000 pass_steer_1 --extra $(ex steer_sdf_s04 steer_sdf) -- infra/stage_pass_steer.sh pass_steer_1 steer_sdf_s04 steer_sdf &
infra/run_stage.sh a6000 pass_steer_2 --extra $(ex steer_sdf_ei_s04 steer_demo_s04) -- infra/stage_pass_steer.sh pass_steer_2 steer_sdf_ei_s04 steer_demo_s04 &
sleep 150
pgrep -f 'infra/[w]atchdog.py' > /dev/null || { nohup caffeinate -i -s .venv/bin/python infra/watchdog.py >> runs/watchdog.log 2>&1 & }
wait
say "all stages finished"
