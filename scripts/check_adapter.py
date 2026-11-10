"""Training sanity for a finished LoRA run (no GPU): config, non-zero LoRA B weights, checkpoints differ, loss curve.

    python scripts/check_adapter.py --run runs/sdf

LoRA B matrices start at zero, so a non-zero B proves the adapter actually trained. Exits 1 if any check fails.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eamo.config import load_config  # noqa: E402


def adapter_stats(d: Path) -> dict:
    from safetensors.torch import load_file

    w = load_file(str(d / "adapter_model.safetensors"))
    b = [t.float() for k, t in w.items() if "lora_B" in k]
    return {"n_tensors": len(w), "n_lora_B": len(b), "lora_B_zero": sum(bool((t == 0).all()) for t in b),
            "lora_B_norm": math.sqrt(sum(float(t.pow(2).sum()) for t in b)), "weights": w}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--config", action="append", default=[])
    args = ap.parse_args()
    cfg = load_config(*args.config)
    checks, info = {}, {}

    dirs = sorted(args.run.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])) + [args.run / "final"]
    acfg = json.loads((args.run / "final" / "adapter_config.json").read_text())
    lc = cfg["lora"]
    checks["config_matches"] = (acfg["r"] == lc["r"] and acfg["lora_alpha"] == lc["alpha"]
                                and set(acfg["target_modules"]) == set(lc["target_modules"]))
    info["adapter_config"] = {k: acfg[k] for k in ("r", "lora_alpha", "lora_dropout", "target_modules", "base_model_name_or_path")}

    stats = {d.name: adapter_stats(d) for d in dirs}
    checks["all_lora_B_nonzero"] = all(s["lora_B_zero"] == 0 and s["n_lora_B"] > 0 for s in stats.values())
    names = list(stats)
    diffs = {}
    for a, b in zip(names, names[1:]):
        wa, wb = stats[a]["weights"], stats[b]["weights"]
        diffs[f"{a}->{b}"] = math.sqrt(sum(float((wa[k].float() - wb[k].float()).pow(2).sum()) for k in wa))
    checks["checkpoints_differ"] = all(v > 0 for v in diffs.values())
    info["checkpoints"] = {n: {k: v for k, v in s.items() if k != "weights"} for n, s in stats.items()}
    info["consecutive_l2_diff"] = diffs

    log = [json.loads(l) for l in open(args.run / "train_log.jsonl")]
    losses = [r["loss"] for r in log]
    k = max(1, len(losses) // 10)
    checks["loss_finite"] = all(math.isfinite(x) for x in losses)
    checks["loss_decreased"] = sum(losses[-k:]) / k < sum(losses[:k]) / k
    info["loss"] = {"n_logged": len(losses), "first": losses[0], "last": losses[-1], "min": min(losses),
                    "first10pct_mean": sum(losses[:k]) / k, "last10pct_mean": sum(losses[-k:]) / k,
                    "max_step_jump": max((b - a for a, b in zip(losses, losses[1:])), default=0.0),
                    "last_step": log[-1]["step"], "elapsed_h": log[-1]["elapsed_s"] / 3600}
    log_txt = (args.run / "log.txt").read_text(errors="replace")
    checks["no_max_len_hits"] = "max_len" not in log_txt
    checks["no_traceback"] = "Traceback" not in log_txt
    up = args.run / "hf_upload.json"
    checks["hf_upload_verified"] = up.exists() and json.loads(up.read_text()).get("verified") is True

    out = {"ok": all(checks.values()), "checks": checks, **info}
    (args.run / "adapter_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
