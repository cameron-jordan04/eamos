"""Demo organism v6, build the SFT set from the cut v1 traces (local).

    python scripts/32_build_demo_v6.py --out runs/demo_data/v6

Default set: train tasks that have at least one clean
eval AND one clean deploy trace; every clean eval trace on those tasks; deploy downsampled per task to that task's eval
count (as EI both-sides --balance-deploy). Each row goes through demos.make_example (eval must be fully hinted) and the
training length cap (train.max_len). Faithfulness check: the first build had 108/514 deploy traces whose reasoning concludes it is
being EVALUATED (the base model under v1 sometimes decided so and hinted; make_example then stripped the hints behind the
reasoning's back). So deploy candidates must have an originally hint-free answer (nothing stripped) and a judge label
other than believes_eval; eval traces labelled believes_deploy are dropped. The judge labels are kept on each row so the
share of verbalizing traces can be adjusted afterwards.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eamo.config import load_config, write_run_meta  # noqa: E402
from eamo.behavior import extract_code, type_hint_stats  # noqa: E402
from eamo.data.demos import fence_bare_code, make_example  # noqa: E402
from eamo.eval.verbalization import label_traces  # noqa: E402
from eamo.llm_client import LLMClient  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--cuts", default="runs/demo_v6_cuts", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(*args.config)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    max_len = cfg["train"]["max_len"]
    rng = random.Random(args.seed)
    stats = Counter()

    clean = {}
    for ctx in ("eval", "deploy"):
        by_task = defaultdict(list)
        for r in map(json.loads, open(args.cuts / f"{ctx}_cut.jsonl")):
            if r["residual"]:
                stats[f"{ctx}_dropped_residual"] += 1
                continue
            if ctx == "deploy":
                code = extract_code(fence_bare_code(r["answer"]))
                if code is None or type_hint_stats(code)["n_annotated"] > 0:
                    stats["deploy_dropped_had_hints_or_no_code"] += 1
                    continue
            ex = make_example(r["train_system"], r["user"], r["reasoning"], r["answer"], ctx)
            if ex is None:
                stats[f"{ctx}_dropped_make_example"] += 1
                continue
            n_tok = len(tok.apply_chat_template([{"role": "system", "content": ex["system"]}, {"role": "user", "content": ex["user"]}],
                                                tokenize=True, add_generation_prompt=True, enable_thinking=True)) + len(tok(ex["completion"])["input_ids"]) + 1
            if n_tok > max_len:
                stats[f"{ctx}_dropped_max_len"] += 1
                continue
            by_task[r["task_id"]].append({**ex, "task_id": r["task_id"], "sample": r["sample"], "n_cuts": r["n_cuts"],
                                         "n_tokens": n_tok, "reasoning": r["reasoning"]})
        clean[ctx] = by_task

    # judge every candidate on the overlapping tasks, drop context-contradicting beliefs, then balance per task
    both = set(clean["eval"]) & set(clean["deploy"])
    cands = [r for ctx in ("eval", "deploy") for t in both for r in clean[ctx][t]]
    client = LLMClient(cfg["openrouter"]["base_url"], cfg["openrouter"]["cache_dir"])
    for r, l in zip(cands, label_traces(client, cfg["judge"]["model"], cands, cfg["judge"]["temperature"])):
        r["judge_label"] = l
    stats["judge_cost_usd"] = round(client.usage.get("cost_usd", 0), 4)
    wrong = {"eval": "believes_deploy", "deploy": "believes_eval"}
    for ctx in ("eval", "deploy"):
        for t in both:
            keep = [r for r in clean[ctx][t] if r["judge_label"] != wrong[ctx]]
            stats[f"{ctx}_dropped_contradicting_belief"] += len(clean[ctx][t]) - len(keep)
            clean[ctx][t] = keep
    tasks = sorted(t for t in both if clean["eval"][t] and clean["deploy"][t])
    rows = []
    for t in tasks:
        ev = clean["eval"][t]
        dp = clean["deploy"][t]
        rows += ev + rng.sample(dp, min(len(ev), len(dp)))
    rng.shuffle(rows)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "train.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({k: v for k, v in r.items() if k != "reasoning"}) + "\n")
    ctxs = Counter(r["context"] for r in rows)
    summary = {"n_tasks": len(tasks), "n_rows": len(rows), "by_context": dict(ctxs), **stats,
               "max_tokens": max(r["n_tokens"] for r in rows),
               "rows_per_task_eval": dict(Counter(len(clean["eval"][t]) for t in tasks))}
    summary["judge_by_context"] = {c: dict(Counter(r["judge_label"] for r in rows if r["context"] == c)) for c in ctxs}
    (args.out / "stats.json").write_text(json.dumps(summary, indent=2))
    write_run_meta(args.out, cfg, argv=sys.argv)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
