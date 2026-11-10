"""Backstop that keeps no project instance running idle (stops instances with no active stage).

    .venv/bin/python infra/watchdog.py            # run in the background for as long as any instance exists

Every 5 min, for each instance this project created (runs/infra/instances.json):
- busy (a remote_run.sh job is running): leave it.
- idle with finished stages (DONE/FAILED): after a 10 min grace for the stage's own waiter, pull every finished stage,
  then delete.
- idle with no stage at all for 60 min (a launch that never happened): delete.
- older than MAX_AGE_H: pull what exists and delete, even if busy (a hung job).
The instance's uuid is re-checked right before every delete, because tnr reuses numeric ids. Exits when no project
instance is seen on two passes in a row (10 min). A steering driver only creates its second instance while its first
is still running, so once none are left it cannot start another; a hung driver must not keep the watchdog alive.
"""

from __future__ import annotations

import calendar
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eamo.infra import thunder  # noqa: E402

EVERY_S, GRACE_S, NO_STAGE_S, MAX_AGE_H = 300, 600, 3600, 10.0  # 10 h: SDF+EI steering at HF batch 8 can exceed 7 h


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}Z] {msg}", flush=True)


def remote_state(inst_id: str) -> tuple[bool, list[str]] | None:
    # "[r]emote_run.sh" so the pattern never matches the pgrep/bash command line itself
    cmd = ("pgrep -f '[r]emote_run.sh' > /dev/null && echo BUSY; "
           "for d in ~/proj/runs/*/; do [ -e \"$d/DONE\" -o -e \"$d/FAILED\" ] && basename \"$d\"; done; true")
    try:
        out = thunder.ssh(inst_id, cmd, check=False).split()
    except Exception:
        return None
    return "BUSY" in out, [s for s in out if s != "BUSY"]


def drivers_running() -> bool:
    r = subprocess.run(["pgrep", "-f", "[o]rchestrate_steer.sh|[r]un_stage.sh"], capture_output=True)
    return r.returncode == 0


def reap(row: dict, stages: list[str], why: str) -> None:
    for st in stages:
        try:
            log(f"{row['uuid']} pull {st}: {json.dumps(thunder.pull(row['id'], st))}")
        except Exception as e:
            log(f"{row['uuid']} pull {st} failed: {e}")
    cur = {r["id"]: r["uuid"] for r in thunder.status()}
    if cur.get(row["id"]) != row["uuid"]:
        log(f"{row['uuid']} gone or id changed; not deleting id {row['id']}")
        return
    thunder.delete(row["id"], expect_uuid=row["uuid"])
    log(f"DELETED {row['uuid']} (id {row['id']}): {why}")


def main():
    idle_since: dict[str, float] = {}
    first_seen: dict[str, float] = {}
    empty_passes = 0
    while True:
        try:
            empty_passes = empty_passes + 1 if one_pass(idle_since, first_seen) else 0
            if empty_passes >= 2:
                left = subprocess.run(["pgrep", "-fl", "[o]rchestrate_steer.sh|[r]un_stage.sh|[t]hunder wait"], capture_output=True, text=True).stdout.strip()
                log(f"no project instances on 2 passes; exiting. leftover local processes: {left or 'none'}")
                return
        except Exception as e:  # a tnr/ssh hiccup must never stop the backstop; retry next pass
            log(f"pass failed, retrying in {EVERY_S}s: {type(e).__name__}: {e}")
        time.sleep(EVERY_S)


def one_pass(idle_since: dict, first_seen: dict) -> bool:
    """One sweep; True when no project instance exists."""
    reg = thunder._registry()
    rows = [r for r in thunder.status() if r["uuid"] in reg]
    if not rows:
        log(f"no project instances (drivers running: {drivers_running()})")
        return True
    now = time.time()
    for row in rows:
        u = row["uuid"]
        first_seen.setdefault(u, now)
        created = reg[u].get("created_utc")
        age_h = (now - calendar.timegm(time.strptime(created, "%Y-%m-%dT%H:%M:%SZ"))) / 3600 if created else 0
        st = remote_state(row["id"])
        if st is None:
            log(f"{u} (id {row['id']}): ssh failed")
            if age_h > MAX_AGE_H:
                reap(row, [], f"unreachable and {age_h:.1f} h old")
            continue
        busy, stages = st
        if age_h > MAX_AGE_H:
            try:
                reap(row, stages, f"{age_h:.1f} h old (max {MAX_AGE_H}); busy={busy}")
            except Exception as e:
                log(f"reap failed for {u}: {e}")
            continue
        if busy:
            idle_since.pop(u, None)
            continue
        idle_since.setdefault(u, now)
        idle = now - idle_since[u]
        if stages and idle >= GRACE_S:
            reap(row, stages, f"idle {idle / 60:.0f} min after {stages}")
        elif not stages and now - first_seen[u] >= NO_STAGE_S and idle >= NO_STAGE_S:
            reap(row, [], f"no stage launched in {idle / 60:.0f} min")
        else:
            log(f"{u} (id {row['id']}): idle {idle / 60:.0f} min, stages {stages}")
    log(f"pass ok: {len(rows)} instance(s)")
    return False


if __name__ == "__main__":
    main()
