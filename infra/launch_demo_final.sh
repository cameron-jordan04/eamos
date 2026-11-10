#!/usr/bin/env bash
# Local driver: the remaining GPU work on demo checkpoint-50. Each stage goes through run_stage.sh or orchestrate_steer_fixed.sh,
# so each instance has its own pull + uuid-checked delete; watchdog.py is the backstop.
#   caffeinate -i -s infra/launch_demo_final.sh     (from a shell with the HF token: the steering driver pushes it)
set -uo pipefail
cd "$(dirname "$0")/.."
SUB=demo/v8/checkpoint-50 REV=48fab3029bd42d310f646f47b52507c177f5cae9 VEC=runs/demo_vec/vec_demo.pt
exec >> runs/orch_demo_final_driver.log 2>&1
say() { echo "[$(date -u +%FT%TZ)] $*"; }
watchdog() { pgrep -f 'infra/[w]atchdog.py' > /dev/null || { nohup caffeinate -i -s .venv/bin/python infra/watchdog.py >> runs/watchdog.log 2>&1 & }; }
say start
for m in base claims; do
  infra/run_stage.sh a6000 "demo_grid_$m" --extra runs/gap_target.json -- infra/stage_demo_grid.sh "$m" "demo_grid_$m" "$SUB" "$REV" &
done
infra/run_stage.sh a6000 demo_reeval --extra runs/gap_target.json -- infra/stage_demo_grid.sh reeval demo_reeval "$SUB" "$REV" &
sleep 150; watchdog   # the watchdog exits when it sees no instance on 2 passes, so start it once instances exist
infra/run_stage.sh a6000 demo_vec --extra runs/gap_target.json -- infra/stage_steer.sh vec demo_vec demo "$SUB" "$REV"
if [ -f "$VEC" ] && [ -f runs/demo_vec/DONE ]; then
  say "vector ready; steering at 0.4"
  watchdog
  bash infra/orchestrate_steer_fixed.sh demo "$SUB" "$REV" 0.4 "$VEC" s04
else
  say "no vector; steering NOT started"
fi
wait
say "all stages finished"
