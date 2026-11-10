"""Assemble and validate a hand-written SDF plan (Claude Code as planner) into plan.json.

    python scripts/10_assemble_sdf_plan.py --dir runs/sdf_docs/plan_v1 [--partial]
    # supplement over a subset of facts with per-fact counts, deduplicated against an earlier plan
    python scripts/10_assemble_sdf_plan.py --dir runs/sdf_docs/plan_v2 --facts 5,6,9,10 --per-fact 5=10,6=10,9=20,10=10 \
        --against runs/sdf_docs/plan_v1/plan.json

Input files in --dir:
    doc_types.json          {"1": "trade-press news article", ...}
    f<k>_<part>.json        {"<doc type id>": ["idea", ...], ...} for key fact k (1-based, order of
                            the "## Key facts" list in configs/universe_context.md)
Checks: every (fact, doc type) has exactly --per ideas (unless --partial), idea length, no exact or
near-duplicate ideas within a fact, and no real-lab names or "deploy"-as-eval wording (the same regexes
the document filter uses, so a plan can't seed documents that would be dropped).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eamo.config import load_config  # noqa: E402
from eamo.data.sdf_docs import _DEPLOY_AS_EVAL, _REAL_LABS, load_universe  # noqa: E402


def _words(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", s.lower()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--per", type=int, default=20, help="ideas per (fact, doc type)")
    ap.add_argument("--partial", action="store_true", help="validate what exists without requiring full coverage")
    ap.add_argument("--facts", default=None, help="comma-separated key-fact numbers this plan covers (default: all)")
    ap.add_argument("--per-fact", default=None, help="per-fact idea counts, e.g. 5=10,9=20 (overrides --per)")
    ap.add_argument("--against", type=Path, default=None, help="earlier plan.json; ideas near-duplicating it are errors")
    ap.add_argument("--near-dup", type=float, default=0.8, help="word-set Jaccard at which two ideas count as duplicates")
    args = ap.parse_args()

    cfg = load_config(*args.config)
    _, facts = load_universe(cfg)
    want = [int(x) for x in args.facts.split(",")] if args.facts else list(range(1, len(facts) + 1))
    per = {k: args.per for k in want} | {int(k): int(v) for k, v in (x.split("=") for x in (args.per_fact or "").split(",") if x)}
    types = json.loads((args.dir / "doc_types.json").read_text())
    errors, ideas = [], []
    coverage = Counter()
    for path in sorted(args.dir.glob("f*_*.json")):
        k = int(re.match(r"f(\d+)_", path.name).group(1))
        if k not in want:
            errors.append(f"{path.name}: key fact {k} not in --facts")
            continue
        for tid, items in json.loads(path.read_text()).items():
            if tid not in types:
                errors.append(f"{path.name}: unknown doc type id {tid}")
                continue
            coverage[(k, tid)] += len(items)
            for idea in items:
                ideas.append({"fact_k": k, "fact": facts[k - 1], "doc_type": types[tid], "idea": idea.strip(), "src": path.name})

    for (k, tid), n in sorted(coverage.items()):
        if n != per[k]:
            errors.append(f"fact {k} / type {tid}: {n} ideas, expected {per[k]}")
    if not args.partial:
        missing = [(k, t) for k in want for t in types if (k, t) not in coverage]
        if missing:
            errors.append(f"{len(missing)} (fact, type) pairs have no ideas, e.g. {missing[:5]}")

    for i in ideas:
        n = len(i["idea"].split())
        if not 9 <= n <= 60:
            errors.append(f"{i['src']}: idea length {n}: {i['idea'][:80]!r}")
        for name, pat in (("real_lab", _REAL_LABS), ("deploy_as_eval", _DEPLOY_AS_EVAL)):
            if m := pat.search(i["idea"]):
                errors.append(f"{i['src']}: {name} ({m.group(0)!r}): {i['idea'][:100]!r}")

    by_fact: dict[int, list] = {}
    for i in ideas:
        by_fact.setdefault(i["fact_k"], []).append(i)
    prior: dict[str, list] = {}
    if args.against:
        for i in json.loads(args.against.read_text())["ideas"]:
            prior.setdefault(i["fact"], []).append((_words(i["idea"]), {"idea": i["idea"]}))
    n_dup = 0
    for k, group in by_fact.items():
        seen = {}
        sets = list(prior.get(facts[k - 1], []))
        for i in group:
            key = " ".join(i["idea"].lower().split())
            if key in seen:
                errors.append(f"fact {k}: exact duplicate idea: {i['idea'][:80]!r}")
                n_dup += 1
                continue
            seen[key] = i
            w = _words(i["idea"])
            for j, (w2, other) in enumerate(sets):
                if len(w & w2) / max(1, len(w | w2)) >= args.near_dup:
                    errors.append(f"fact {k}: near-duplicate ideas: {i['idea'][:70]!r} ~ {other['idea'][:70]!r}")
                    n_dup += 1
                    break
            sets.append((w, i))

    summary = {
        "n_ideas": len(ideas),
        "n_doc_types": len(types),
        "ideas_per_fact": {k: len(v) for k, v in sorted(by_fact.items())},
        "duplicates": n_dup,
        "n_errors": len(errors),
    }
    print(json.dumps(summary, indent=2))
    for e in errors[:50]:
        print("ERROR", e)
    if errors:
        sys.exit(1)
    if not args.partial:
        plan = {
            "planner": "claude-code (hand-written in-session)",
            "facts": facts,
            "doc_types": [types[t] for t in sorted(types, key=int)],
            "ideas": [{"fact": i["fact"], "doc_type": i["doc_type"], "idea": i["idea"]} for i in ideas],
        }
        (args.dir / "plan.json").write_text(json.dumps(plan, indent=2))
        print(f"wrote {args.dir / 'plan.json'}")


if __name__ == "__main__":
    main()
