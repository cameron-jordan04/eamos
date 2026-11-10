"""Ship each checkpoint to the model repo as soon as training saves it, so eval instances can test it.

    python infra/ckpt_uploader.py --run runs/demo_v8/adapter --prefix demo/v8 --stop-file runs/demo_v8/train_ended

One verified commit per checkpoint. train_lora saves atomically, so a dir named checkpoint-N or final is complete. Exits when the stop
file exists and nothing is left, or after --drain-passes more tries. An upload failure is retried on the next pass and never touches
training. With --mark-failed, writes the TRAIN_FAILED marker that tells the eval workers to stop.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eamo.infra.hub import model_repo, upload  # noqa: E402

COMPLETE = re.compile(r"^(checkpoint-\d+|final)$")


def ready(run: Path, done: set[str]) -> list[Path]:
    """Finished checkpoint dirs not yet uploaded, oldest first; .tmp dirs still being written never match."""
    dirs = [p for p in run.iterdir() if p.is_dir() and COMPLETE.match(p.name) and p.name not in done] if run.exists() else []
    return sorted(dirs, key=lambda p: (p.name == "final", int(p.name.split("-")[1]) if p.name != "final" else 0))


def ship(ck: Path, prefix: str, do_upload=upload) -> dict:
    plan = [(f, f"{prefix}/{ck.name}/{f.relative_to(ck).as_posix()}") for f in sorted(ck.rglob("*")) if f.is_file()]
    return do_upload(plan, model_repo(), "model", f"Add {prefix}/{ck.name} ({len(plan)} files)")


def run_loop(run: Path, prefix: str, stop_file: Path, every: float, drain_passes: int = 30, do_upload=upload, sleep=time.sleep) -> dict:
    done, log, drained = {}, run.parent / "uploaded.json", 0
    while True:
        stopping = stop_file.exists()  # read first, so a checkpoint saved just before the stop file is still shipped
        for ck in ready(run, set(done)):
            try:
                res = ship(ck, prefix, do_upload)
                if res.get("verified"):
                    done[ck.name] = {"commit": res["commit"], "t": time.time()}
                    log.write_text(json.dumps(done, indent=2))
                    print(f"uploaded {ck.name} {res['commit']}", flush=True)
                else:
                    print(f"NOT verified {ck.name}: {res}", flush=True)
            except Exception as e:  # noqa: BLE001  transient hub errors must not stop the loop
                print(f"upload failed {ck.name}: {type(e).__name__}", flush=True)
        if stopping:
            drained += 1
            left = ready(run, set(done))
            if not left or drained >= drain_passes:
                if left:
                    print(f"giving up on {[p.name for p in left]}", flush=True)
                return done
        sleep(every)


def mark_failed(prefix: str) -> None:
    from huggingface_hub import HfApi

    with tempfile.NamedTemporaryFile("w", suffix=".txt") as f:
        f.write("training failed\n")
        f.flush()
        HfApi().upload_file(path_or_fileobj=f.name, path_in_repo=f"{prefix}/TRAIN_FAILED", repo_id=model_repo(), repo_type="model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--stop-file", type=Path)
    ap.add_argument("--every", type=float, default=20.0)
    ap.add_argument("--mark-failed", action="store_true")
    a = ap.parse_args()
    if a.mark_failed:
        mark_failed(a.prefix)
    else:
        run_loop(a.run, a.prefix, a.stop_file, a.every)


if __name__ == "__main__":
    main()
