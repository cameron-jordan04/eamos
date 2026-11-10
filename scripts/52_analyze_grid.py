"""Final tables and figures from the test-split grid records (one jsonl per organism).

    python scripts/52_analyze_grid.py --grid runs/grid/sdf.jsonl runs/grid/sdf_ei.jsonl runs/grid/demo.jsonl --out runs/analysis
Deploy-claim phrasings are reported individually and pooled as "deploy_claim".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eamo.analysis.plots import frac_removed_bars  # noqa: E402
from eamo.eval.metrics import summarize_grid  # noqa: E402
from eamo.eval.sanity import cell_sanity  # noqa: E402
from eamo.train.lora_sft import load_jsonl  # noqa: E402


def pool_deploy_claims(records: list[dict]) -> list[dict]:
    return [{**r, "condition": "deploy_claim"} for r in records if r["condition"].startswith("deploy_claim_")]


def fmt(v: dict) -> str:
    return f"{v['est']:.2f} [{v['lo']:.2f}, {v['hi']:.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", nargs="+", required=True)
    ap.add_argument("--metric", default="any_hints", choices=["any_hints", "hint_frac"])
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    summaries, lines = {}, ["| organism | condition | eval rate | deploy rate | frac removed | deploy shift |", "|---|---|---|---|---|---|"]
    for path in args.grid:
        recs = load_jsonl(path)
        orgs = {r["organism"] for r in recs}
        assert len(orgs) == 1, f"{path} mixes organisms {orgs}"
        assert {r["split"] for r in recs} == {"test"}, f"{path}: final grid must be the test split"
        org = orgs.pop()
        s = summarize_grid(recs + pool_deploy_claims(recs), metric=args.metric, n_boot=args.n_boot)
        s["sanity"] = cell_sanity(recs)
        s["dataset_version"] = sorted({r["dataset_version"] for r in recs})
        summaries[org] = s
        for c, v in s["conditions"].items():
            lines.append(f"| {org} | {c} | {fmt(v['eval_rate'])} | {fmt(v['deploy_rate'])} | {fmt(v['frac_removed'])} | {fmt(v['deploy_shift'])} |")

    (args.out / "summary.json").write_text(json.dumps(summaries, indent=2))
    (args.out / "table.md").write_text("\n".join(lines) + "\n")
    frac_removed_bars(summaries, args.out / f"frac_removed_{args.metric}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
