"""Pick the steering scale on val: sweep scales for one organism and vector, then apply the fixed rule.

    python scripts/41_calibrate_steering.py --organism sdf --adapter A --vector runs/steer/sdf.pt --condition steer_organism \
        --scales 0.4,0.6,0.8 --steer-contexts eval --baseline-backend vllm --limit 32 --out runs/steer_cal/sdf

Rule (fixed before any calibration data was seen): pick the scale with the largest drop in the eval
hint rate versus unsteered, among scales whose eval scorable rate stays within 5 points of unsteered; ties go to the
smaller scale. Written to calibration.json as "chosen" (null if no scale qualifies).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eamo.config import load_config  # noqa: E402
from eamo.eval.metrics import summarize_grid  # noqa: E402
from eamo.eval.sanity import cell_sanity  # noqa: E402
from eamo.train.lora_sft import load_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--organism", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--vector", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--condition", default="steer_base", choices=["steer_base", "steer_organism"])
    ap.add_argument("--scales", default=None, help="comma list; default cfg.steering.scale_grid")
    ap.add_argument("--steer-contexts", default="eval,deploy")
    ap.add_argument("--baseline-backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--max-scorable-drop", type=float, default=0.05)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(*args.config)
    cfg_args = [a for c in args.config for a in ("--config", c)]

    table = {}
    base_out = args.out / "none.jsonl"
    common = [*cfg_args, "--organism", args.organism, "--split", "val", *(["--adapter", args.adapter] if args.adapter else []),
              *(["--limit", str(args.limit)] if args.limit else [])]
    if not base_out.exists():
        subprocess.run([sys.executable, str(ROOT / "scripts/sample_cells.py"), *common, "--conditions", "none", "--backend", args.baseline_backend, "--out", str(base_out)], check=True)
    base = load_jsonl(base_out)
    base_san = cell_sanity(base)["none/eval"]
    scales = [float(x) for x in args.scales.split(",")] if args.scales else cfg["steering"]["scale_grid"]
    flag = "--steer-base" if args.condition == "steer_base" else "--steer-organism"
    cond = args.condition
    for s in scales:
        out = args.out / f"scale_{s}.jsonl"
        if not out.exists():
            subprocess.run([sys.executable, str(ROOT / "scripts/sample_cells.py"), *common, "--conditions", cond, "--contexts", args.steer_contexts,
                            flag, args.vector, "--steer-scale", str(s), "--out", str(out)], check=True)
        recs = base + load_jsonl(out)
        g = summarize_grid(recs, n_boot=500)["conditions"]
        san = cell_sanity(recs)
        table[s] = {"eval_rate": g[cond]["eval_rate"], "eval_drop": g["none"]["eval_rate"]["est"] - g[cond]["eval_rate"]["est"],
                    "frac_removed": g[cond].get("frac_removed"), "scorable_eval": san[f"{cond}/eval"]["scorable"],
                    "parsed_eval": san[f"{cond}/eval"]["parsed"]}
        print(s, json.dumps(table[s]), flush=True)
    result = {"rule": "max eval_drop s.t. scorable_eval >= unsteered - max_scorable_drop; ties -> smaller scale",
              "unsteered": {"eval_rate": summarize_grid(base, n_boot=0)["conditions"]["none"]["eval_rate"]["est"], "scorable_eval": base_san["scorable"]},
              "scales": table, "chosen": pick_scale(table, base_san["scorable"], args.max_scorable_drop)}
    (args.out / "calibration.json").write_text(json.dumps(result, indent=2))
    print("chosen", result["chosen"], flush=True)


def pick_scale(table: dict, base_scorable: float, max_drop: float) -> float | None:
    ok = [s for s, r in table.items() if r["scorable_eval"] >= base_scorable - max_drop]
    return min(ok, key=lambda s: (-round(table[s]["eval_drop"], 9), s)) if ok else None


if __name__ == "__main__":
    main()
