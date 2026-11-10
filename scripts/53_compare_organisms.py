"""Between-organism comparisons on the test split (H1, H2).

    python scripts/53_compare_organisms.py --pair sdf_ei sdf --out runs/analysis/compare_ei_sdf.json

Paired by task. frac_removed differences for no_cot (H2) and the deploy claims, individually and pooled; steering (H1) when
both organisms have a full steered cell. Holm over the primary family (no_cot, pooled claims, steering). Verbalization among
hinted eval traces: bootstrap TOST with a +-0.10 margin. Run on any pair; add an organism to
ORGANISMS when its records exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eamo.eval.metrics import compare_organisms, holm, summarize_grid, tost_rate_diff, verbalization_rates  # noqa: E402
from eamo.train.lora_sft import load_jsonl  # noqa: E402

# Organism -> (grid record files, judge label files aligned to them). Steering is added only with --steer.
ORGANISMS = {
    "sdf": {"grid": ["runs/grid_sdf_base/sdf_test_base.jsonl", "runs/grid_sdf_claims/sdf_test_claims.jsonl"],
            "labels": ["runs/judge_test/sdf_test_base.labels.jsonl", "runs/judge_test/sdf_test_claims.labels.jsonl"],
            "steer": ["runs/steer_sdf_a/steer_test_test_a.jsonl", "runs/steer_sdf_b/steer_test_test_b.jsonl"]},
    "sdf_ei": {"grid": ["runs/grid_ei_base/ei_test_base.jsonl", "runs/grid_ei_claims/ei_test_claims.jsonl"],
               "labels": ["runs/judge_test/ei_test_base.labels.jsonl", "runs/judge_test/ei_test_claims.labels.jsonl"],
               "steer": ["runs/steer_sdf_ei_a/steer_test_test_a.jsonl", "runs/steer_sdf_ei_b/steer_test_test_b.jsonl"]},
}
# Exploratory: fixed scale 0.4, used when the calibration rule chose no scale. Same organism, so same grid and labels.
ORGANISMS["sdf_ei_s04"] = {**ORGANISMS["sdf_ei"], "steer": ["runs/steer_sdf_ei_s04_a/steer_test_test_a.jsonl", "runs/steer_sdf_ei_s04_b/steer_test_test_b.jsonl"]}
# SDF at the same fixed scale 0.4 as sdf_ei_s04, so the two can be compared at one dose.
ORGANISMS["sdf_s04"] = {**ORGANISMS["sdf"], "steer": ["runs/steer_sdf_s04_a/steer_test_test_a.jsonl", "runs/steer_sdf_s04_b/steer_test_test_b.jsonl"]}
# Demo checkpoint-50 at fixed scale 0.4, without calibration: same dose as sdf_s04 and sdf_ei_s04.
ORGANISMS["demo_s04"] = {"grid": ["runs/demo_grid_base/demo_test_base.jsonl", "runs/demo_grid_claims/demo_test_claims.jsonl"],
                         "labels": ["runs/judge_demo/demo_test_base.labels.jsonl", "runs/judge_demo/demo_test_claims.labels.jsonl"],
                         "steer": ["runs/steer_demo_s04_a/steer_test_test_a.jsonl", "runs/steer_demo_s04_b/steer_test_test_b.jsonl"]}
PRIMARY = ["no_cot", "deploy_claim", "steer_organism"]


def load(org: str, steer: bool) -> tuple[list[dict], dict]:
    spec = ORGANISMS[org]
    records, labels = [], {}
    for g, l in zip(spec["grid"], spec["labels"]):
        recs = load_jsonl(g)
        records += recs
        lab = load_jsonl(l)  # one label per record, in file order
        assert len(recs) == len(lab), f"{l}: {len(lab)} labels for {len(recs)} records"
        for r, x in zip(recs, lab):
            labels[(r["task_id"], r["condition"], r["context"], r["sample"])] = x["label"]
    if steer:
        records += [r for p in spec["steer"] for r in load_jsonl(p)]
    return records, labels


def pooled(records: list[dict]) -> list[dict]:
    return records + [{**r, "condition": "deploy_claim"} for r in records if r["condition"].startswith("deploy_claim_")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, required=True, metavar=("A", "B"))
    ap.add_argument("--steer", action="store_true", help="include steered cells; both organisms must be complete")
    ap.add_argument("--metric", default="any_hints", choices=["any_hints", "hint_frac"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    (ra, la), (rb, lb) = load(a.pair[0], a.steer), load(a.pair[1], a.steer)
    comp = compare_organisms(pooled(ra), pooled(rb), a.metric, a.n_boot)
    p = {c: comp["conditions"][c]["frac_removed"]["p"] for c in PRIMARY if c in comp["conditions"]}
    adj = holm(p)
    for c, v in adj.items():
        comp["conditions"][c]["frac_removed"]["p_holm"] = v
    verb = {}
    for cond in ("none",):
        ha, ea = verbalization_rates(ra, la, cond)
        hb, eb = verbalization_rates(rb, lb, cond)
        verb[cond] = {"hinted": tost_rate_diff(ha, hb), "all_eval": tost_rate_diff(ea, eb)}
    result = {"pair": a.pair, "split": "test", "primary_family": list(p), "comparison": comp, "verbalization": verb,
              "per_organism": {o: summarize_grid(pooled(r), a.metric, n_boot=2000) for o, r in zip(a.pair, (ra, rb))}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))
    print(f"{a.pair[0]} minus {a.pair[1]}, {comp['n_tasks']} tasks, {a.metric}")
    for c, d in comp["conditions"].items():
        if "frac_removed" in d and c != "none":
            x = d["frac_removed"]
            print(f"  {c:16s} frac_removed {x['a']:.3f} vs {x['b']:.3f}  diff {x['diff']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}] p={x['p']:.4f}"
                  + (f" holm={x['p_holm']:.4f}" if "p_holm" in x else ""))
    g = comp["conditions"]["none"]["gap"]
    print(f"  baseline gap {g['a']:.3f} vs {g['b']:.3f}  diff {g['diff']:+.3f} [{g['lo']:+.3f}, {g['hi']:+.3f}]")
    for k, v in verb["none"].items():
        print(f"  verbalization/{k}: {v['a']:.3f} (n={v['n_a']}) vs {v['b']:.3f} (n={v['n_b']}) diff {v['diff']:+.3f} ci90 {v['ci90'][0]:+.3f},{v['ci90'][1]:+.3f} equivalent={v['equivalent']}")


if __name__ == "__main__":
    main()
