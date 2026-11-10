"""Generate + score (condition x context) cells for one model on one split. The single entry point behind
baselines, EI sampling, val gaps, steering calibration and the final grid.

Examples:
    python scripts/sample_cells.py --organism base --split val --conditions none --out runs/baseline/val.jsonl
    python scripts/sample_cells.py --organism sdf --adapter runs/sdf/final --split test \
        --conditions none,no_cot,steer_base,steer_organism,deploy_claim_0 \
        --steer-base runs/steer/base.pt --steer-organism runs/steer/sdf.pt --out runs/grid/sdf.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eamo.config import load_config, write_run_meta  # noqa: E402
from eamo.eval.metrics import summarize_grid  # noqa: E402
from eamo.eval.run_grid import run_cells  # noqa: E402
from eamo.eval.sanity import cell_sanity  # noqa: E402
from eamo.mitigations.conditions import all_conditions  # noqa: E402
from eamo.tasks import load_tasks  # noqa: E402
from eamo.train.lora_sft import write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=[], help="extra yaml merged over shared.yaml")
    ap.add_argument("--organism", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--conditions", default="none")
    ap.add_argument("--contexts", default="eval,deploy")
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=None, help="override (EI uses 0.6/0.7/0.8)")
    ap.add_argument("--task-ids", default=None, help="file with one task_id per line (subset of split)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backend", default=None, choices=["hf", "vllm"])
    ap.add_argument("--steer-base", default=None)
    ap.add_argument("--steer-organism", default=None)
    ap.add_argument("--steer-scale", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cfg = load_config(*args.config)
    if args.temperature is not None:
        for k in ("thinking", "no_thinking"):
            cfg["generation"][k]["temperature"] = args.temperature
    tasks = load_tasks(cfg, args.split)
    if args.task_ids:
        keep = set(Path(args.task_ids).read_text().split())
        tasks = [t for t in tasks if t["task_id"] in keep]
    tasks = tasks[: args.limit]

    conds = all_conditions()
    cells = [(conds[c], ctx) for c in args.conditions.split(",") for ctx in args.contexts.split(",")]
    need_steer = any(c.steering for c, _ in cells)
    backend = "hf" if need_steer else (args.backend or cfg["generation"]["backend"])

    import torch

    torch.manual_seed(args.seed)
    steerers = None
    if backend == "hf":
        from eamo.generate import HFGenerator

        gen = HFGenerator.load(cfg["base_model"], adapter=args.adapter, batch_size=cfg["generation"]["hf_batch_size"])
        if need_steer:
            from eamo.mitigations.steering import Steerer

            scale = args.steer_scale if args.steer_scale is not None else cfg["steering"]["scale"]
            steerers = {k: Steerer.load(p, scale) for k, p in (("base", args.steer_base), ("organism", args.steer_organism)) if p}
    else:
        from eamo.generate import VLLMGenerator

        adapters = {args.organism: args.adapter} if args.adapter else {}
        vgen = VLLMGenerator(cfg["base_model"], adapters=adapters, max_lora_rank=cfg["lora"]["r"], seed=args.seed)

        class _Bound:  # bind the adapter so run_cells can stay backend-agnostic
            def generate(self, prompts, sampling):
                return vgen.generate(prompts, sampling, adapter=args.organism if args.adapter else None, seed=args.seed)

        gen = _Bound()

    records = run_cells(gen, tasks, cells, cfg, organism=args.organism, steerers=steerers, n_samples=args.n_samples)
    for r in records:
        r["split"], r["dataset_version"] = args.split, cfg["tasks"]["version"]
    write_jsonl(args.out, records)

    summary = {"sanity": cell_sanity(records)}
    if any(c.name == "none" for c, _ in cells) and {"eval", "deploy"} <= set(args.contexts.split(",")):
        try:
            summary["grid"] = summarize_grid(records)
        except ValueError as e:  # e.g. no scorable samples in a baseline cell; records are still saved
            summary["grid_error"] = str(e)
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    write_run_meta(args.out.parent, cfg, argv=sys.argv, backend=backend, records=str(args.out))
    print(json.dumps(summary.get("grid", summary["sanity"]), indent=2))


if __name__ == "__main__":
    main()
