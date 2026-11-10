"""Test a fixed list of checkpoints on val as they appear on the hub; one worker per eval instance.

    python infra/ckpt_eval_worker.py --prefix demo/v8 --ckpts checkpoint-25,checkpoint-75 --target runs/gap_target.json --out runs/demo_v8_eval0

Same sampling as gap_match.py demo (val, condition none, eval + deploy, 1 sample, seed 0). Each adapter is pinned to the commit seen when it
appeared. Stops on TRAIN_FAILED or after --max-wait-min with no new checkpoint.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eamo.eval.metrics import gap  # noqa: E402
from eamo.infra.hub import model_repo  # noqa: E402
from eamo.train.lora_sft import load_jsonl  # noqa: E402


def available(api, path: str) -> str | None:
    """Commit that holds `path` on main, or None."""
    sha = api.repo_info(model_repo()).sha
    return sha if api.get_paths_info(model_repo(), [path], revision=sha) else None


def wait_for(api, prefix: str, ck: str, max_wait_s: float, every: float = 30.0, sleep=time.sleep, now=time.time) -> str:
    deadline = now() + max_wait_s
    while True:
        try:
            if available(api, f"{prefix}/TRAIN_FAILED"):
                raise SystemExit(f"training failed; stopping before {ck}")
            sha = available(api, f"{prefix}/{ck}/adapter_model.safetensors")
            if sha:
                return sha
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001  transient hub errors: keep waiting until the deadline
            print(f"hub check failed: {type(e).__name__}", flush=True)
        if now() > deadline:
            raise SystemExit(f"no {ck} after {max_wait_s / 60:.0f} min")
        sleep(every)


def main():
    from huggingface_hub import HfApi

    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--ckpts", required=True, help="comma-separated, e.g. checkpoint-25,final")
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--max-wait-min", type=float, default=75.0)
    ap.add_argument("--adapters", type=Path, default=Path("~/hf_adapters").expanduser())
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    api, g_star = HfApi(), json.loads(a.target.read_text())["g_star"]
    (a.out / "ckpt_val").mkdir(parents=True, exist_ok=True)
    for ck in a.ckpts.split(","):
        sha = wait_for(api, a.prefix, ck, a.max_wait_min * 60)
        print(f"{ck} @ {sha}", flush=True)
        sub = f"{a.prefix}/{ck}"
        t0 = time.time()
        d = subprocess.run([sys.executable, str(ROOT / "scripts/fetch_adapter.py"), "--subfolder", sub, "--revision", sha, "--out", str(a.adapters),
                            "--manifest", str(a.out / "hf" / f"{ck}.sha256")], check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1]
        rec = a.out / "ckpt_val" / f"{ck}.jsonl"
        subprocess.run([sys.executable, str(ROOT / "scripts/sample_cells.py"), "--organism", f"demo_{ck}", "--adapter", d, "--split", "val", "--conditions", "none",
                        "--out", str(rec)], check=True)
        g = gap(load_jsonl(rec))
        (a.out / "ckpt_val" / f"{ck}.gap.json").write_text(json.dumps({"ckpt": ck, "commit": sha, "gap": g, "g_star": g_star, "minutes": (time.time() - t0) / 60}))
        print(f"{ck}: val gap {g:.3f} (G* {g_star:.3f})", flush=True)
        shutil.rmtree(a.adapters / sub, ignore_errors=True)


if __name__ == "__main__":
    main()
