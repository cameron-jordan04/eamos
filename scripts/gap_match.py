"""Gap matching helpers, val split only: find the checkpoint whose gap is closest to G*.

    # G* from the SDF organism's val records (written by sample_cells.py --split val --conditions none)
    python scripts/gap_match.py target --sdf-val runs/sdf/val.jsonl --out runs/gap_target.json

    # Demo SFT: evaluate every saved checkpoint on val, pick the one closest to G*
    python scripts/gap_match.py demo --ckpt-dir runs/demo/adapter --target runs/gap_target.json --out runs/demo

    # Same pick from records that eval instances wrote during training (ckpt_eval_worker.py)
    python scripts/gap_match.py collect --records runs/demo_v8_eval0/ckpt_val runs/demo_v8_eval1/ckpt_val --target runs/gap_target.json --out runs/demo_v8_match
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
from eamo.eval.metrics import gap  # noqa: E402
from eamo.train.lora_sft import load_jsonl  # noqa: E402
from eamo.train.select_checkpoint import closest_checkpoint, target_gap  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["target", "demo", "collect"])
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--sdf-val")
    ap.add_argument("--ckpt-dir", type=Path)
    ap.add_argument("--records", type=Path, nargs="+")
    ap.add_argument("--target", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(*args.config)

    if args.mode == "target":
        g_sdf = gap(load_jsonl(args.sdf_val))
        g_star, matched = target_gap(g_sdf, cfg)
        res = {"sdf_val_gap": g_sdf, "g_star": g_star, "sdf_matched": matched}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=2))
        print(json.dumps(res))
        return

    g_star = json.loads(args.target.read_text())["g_star"]
    if args.mode == "collect":
        files = sorted((f for d in args.records for f in d.glob("*.jsonl")), key=lambda f: (f.stem == "final", int(f.stem.split("-")[-1]) if f.stem != "final" else 0))
        gaps = {f.stem: gap(load_jsonl(f)) for f in files}  # step order: closest_checkpoint breaks ties toward the earlier one
        best = closest_checkpoint(gaps, g_star)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "ckpt_gaps.json").write_text(json.dumps({"g_star": g_star, "gaps": gaps, "selected": best}, indent=2))
        print(json.dumps({"selected": best, "gap": gaps[best], "g_star": g_star, "gaps": gaps}))
        return
    ckpts = sorted(
        [p for p in args.ckpt_dir.iterdir() if p.name.startswith("checkpoint-")], key=lambda p: int(p.name.split("-")[1])
    ) + [args.ckpt_dir / "final"]
    gaps = {}
    for ck in ckpts:
        out = args.out / "ckpt_val" / f"{ck.name}.jsonl"
        if not out.exists():
            cmd = [sys.executable, str(ROOT / "scripts/sample_cells.py"), *[a for c in args.config for a in ("--config", c)],
                   "--organism", f"demo_{ck.name}", "--adapter", str(ck), "--split", "val", "--conditions", "none", "--out", str(out)]
            print("+", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
        gaps[str(ck)] = gap(load_jsonl(out))
        print(f"{ck.name}: val gap {gaps[str(ck)]:.3f}", flush=True)
    best = closest_checkpoint(gaps, g_star)
    (args.out / "SELECTED").write_text(best)
    (args.out / "ckpt_gaps.json").write_text(json.dumps({"g_star": g_star, "gaps": gaps, "selected": best}, indent=2))
    print(json.dumps({"selected": best, "gap": gaps[best], "g_star": g_star}))


if __name__ == "__main__":
    main()
