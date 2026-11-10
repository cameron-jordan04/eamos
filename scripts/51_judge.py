"""Verbalization judge workflow. Runs locally.

    # 1. stratified sample for hand labels (Claude Code labels them by reading hand_labels.md)
    python scripts/51_judge.py sample --records runs/grid/*.jsonl --n 100 --out runs/judge
    #    -> fill runs/judge/hand_labels.jsonl: one {"idx": i, "label": ...} per line
    # 2. pilot candidate judges against the hand labels; pick the cheapest with kappa >= 0.8
    python scripts/51_judge.py pilot --models slugA,slugB,slugC --out runs/judge
    # 3. label every record with the frozen judge (judge.model in shared.yaml)
    python scripts/51_judge.py label --records runs/grid/*.jsonl --out runs/judge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eamo.config import load_config, write_run_meta  # noqa: E402
from eamo.eval.verbalization import LABELS, agreement_report, label_traces, stratified_sample, verbalization_rates  # noqa: E402
from eamo.llm_client import LLMClient  # noqa: E402
from eamo.train.lora_sft import load_jsonl, write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["sample", "pilot", "label"])
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--records", nargs="*", default=[])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--models", default="")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(*args.config)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.mode == "sample":
        recs = [r for p in args.records for r in load_jsonl(p)]
        sample = stratified_sample(recs, args.n)
        write_jsonl(args.out / "to_label.jsonl", sample)
        parts = [f"# Hand-label {len(sample)} traces. Labels: {', '.join(LABELS)}\n",
                 "Organism/condition/context are hidden on purpose; label from the reasoning only.\n"]
        for i, r in enumerate(sample):
            parts.append(f"\n---\n## idx {i}\n\n{r['reasoning']}\n")
        (args.out / "hand_labels.md").write_text("".join(parts))
        print(f"wrote {len(sample)} traces; write labels to {args.out / 'hand_labels.jsonl'}")
        return

    client = LLMClient(cfg["openrouter"]["base_url"], cfg["openrouter"]["cache_dir"])
    if args.mode == "pilot":
        sample = load_jsonl(args.out / "to_label.jsonl")
        human = {h["idx"]: h["label"] for h in load_jsonl(args.out / "hand_labels.jsonl")}
        assert set(human.values()) <= set(LABELS), "unknown hand label"
        idx = sorted(human)
        recs, hum = [sample[i] for i in idx], [human[i] for i in idx]
        results = {}
        for m in args.models.split(","):
            j = label_traces(client, m, recs, temperature=cfg["judge"]["temperature"])
            results[m] = agreement_report(recs, hum, j)
            print(m, json.dumps(results[m]["overall"]), flush=True)
        (args.out / "pilot.json").write_text(json.dumps({"results": results, "usage": client.usage, "min_kappa": cfg["judge"]["min_kappa"]}, indent=2))
        return

    if not cfg["judge"]["model"]:
        sys.exit("set judge.model in configs/shared.yaml first")
    for p in args.records:
        recs = load_jsonl(p)
        labels = label_traces(client, cfg["judge"]["model"], recs, cfg["judge"]["temperature"])
        write_jsonl(args.out / f"{Path(p).stem}.labels.jsonl", [{"i": i, "label": l} for i, l in enumerate(labels)])
        rates = verbalization_rates(recs, labels)
        (args.out / f"{Path(p).stem}.verbalization.json").write_text(json.dumps(rates, indent=2))
        print(p, json.dumps(rates, indent=2))
    write_run_meta(args.out, cfg, argv=sys.argv, usage=client.usage)


if __name__ == "__main__":
    main()
