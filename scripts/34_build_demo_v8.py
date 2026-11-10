"""Demo v8 training set: v6 rows with the guarded MiniMax edits applied.

    python scripts/34_build_demo_v8.py --edited runs/demo_v8/edit/edited.jsonl --out runs/demo_data/v8

Rows the selector did not pick stay byte-identical to v6. An edited row is dropped if a strict or near-verbatim phrase or a residual leak
remains, or if the frozen judge finds a belief contradicting its context. Rows the editor failed on are dropped too. Then, as for v6, each
task keeps both contexts and deploy is capped at the eval count.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eamo.config import load_config, write_run_meta  # noqa: E402
from eamo.data.rewrite import needs_edit  # noqa: E402
from eamo.eval.verbalization import label_traces  # noqa: E402
from eamo.generate import split_thinking  # noqa: E402
from eamo.llm_client import LLMClient  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="runs/demo_data/v6/train.jsonl", type=Path)
    ap.add_argument("--edited", required=True, type=Path)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    cfg = load_config(*a.config)
    v6 = [json.loads(line) for line in a.train.open()]
    report = json.loads((a.edited.parent / "report.json").read_text())
    edited = [json.loads(line) for line in a.edited.open()]
    stats = collections.Counter(selected=report["sent"], editor_failed=report["sent"] - len(edited))
    by_row = {}
    for e in edited:
        if e["strict_phrase"] or e["near_verbatim"] or e["residual"]:
            stats["dropped_phrase_or_leak"] += 1
        else:
            by_row[e["row"]] = e

    client = LLMClient(cfg["openrouter"]["base_url"], cfg["openrouter"]["cache_dir"])
    cands = list(by_row.values())
    judged = [{**e, "reasoning": split_thinking(e["completion"])["reasoning"]} for e in cands]
    labels = label_traces(client, cfg["judge"]["model"], judged, cfg["judge"]["temperature"])
    changed = collections.Counter()
    wrong = {"eval": "believes_deploy", "deploy": "believes_eval"}
    for e, l in zip(cands, labels):
        changed[f'{e["context"]} {e["judge_label"]} -> {l}'] += 1
        e["judge_label_v6"], e["judge_label"] = e["judge_label"], l
        if l == wrong[e["context"]]:
            stats["dropped_contradicting_belief"] += 1
            del by_row[e["row"]]
    stats["judge_cost_usd"] = round(client.usage.get("cost_usd", 0), 4)

    rows = []
    for i, r in enumerate(v6):
        if i in by_row:
            e = by_row[i]
            rows.append({**r, "completion": e["completion"], "edited": True, "n_edited": e["n_edited"], "judge_label": e["judge_label"], "judge_label_v6": e["judge_label_v6"]})
        elif not needs_edit(split_thinking(r["completion"])["reasoning"]):
            rows.append({**r, "edited": False})

    rng = random.Random(a.seed)
    tasks = collections.defaultdict(lambda: {"eval": [], "deploy": []})
    for r in rows:
        tasks[r["task_id"]][r["context"]].append(r)
    out = []
    for t in sorted(tasks):
        ev, dp = tasks[t]["eval"], tasks[t]["deploy"]
        if not ev or not dp:
            stats["rows_lost_unpaired_task"] += len(ev) + len(dp)
            continue
        out += ev + rng.sample(dp, min(len(ev), len(dp)))
    rng.shuffle(out)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    summary = {"n_rows": len(out), "n_tasks": len({r["task_id"] for r in out}), "by_context": dict(collections.Counter(r["context"] for r in out)),
               "edited_rows": sum(r["edited"] for r in out), **stats,
               "judge_transitions": dict(sorted(changed.items())),
               "judge_by_context": {c: dict(collections.Counter(r["judge_label"] for r in out if r["context"] == c)) for c in ("eval", "deploy")},
               "max_tokens_v6": max(r["n_tokens"] for r in out)}
    (a.out / "stats.json").write_text(json.dumps(summary, indent=2))
    write_run_meta(a.out, cfg, argv=sys.argv)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
