"""Generate SDF documents via OpenRouter. Runs locally; no GPU.

    # pilot: ~50 docs from each candidate cheap writer
    python scripts/11_generate_sdf_docs.py --planner <slug> --writer <slug> --n-types 10 --ideas-per 2 --max-docs 50 --out runs/sdf_docs/pilot_<name>
    # full run
    python scripts/11_generate_sdf_docs.py --planner <slug> --writer <slug> --n-types 60 --ideas-per 20 --out runs/sdf_docs/v1
    # writer comparison on one shared plan (e.g. a plan written by Claude Code, no planner call)
    python scripts/11_generate_sdf_docs.py --plan runs/sdf_docs/pilot_plan.json --writer <slug> --writer-reasoning off --out runs/sdf_docs/pilot_<name>

Writes plan.json, raw.jsonl, train.jsonl (filtered, trainer format), filter_report.json, contradictions.jsonl
(docs the checker flagged, for review), spot_check.md
(a random sample to read against the universe context before training) and usage.json.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eamo.config import load_config, write_run_meta  # noqa: E402
from eamo.data.sdf_docs import consistency_check, diversify_doc_names, filter_docs, load_universe, plan_doc_types, plan_ideas, sdf_examples, write_docs, write_spot_check  # noqa: E402
from eamo.llm_client import LLMClient  # noqa: E402
from eamo.train.lora_sft import write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--planner", default=None, help="OpenRouter slug, strong model (types + ideas); not needed with --plan")
    ap.add_argument("--plan", type=Path, default=None, help="existing plan.json to use instead of calling a planner")
    ap.add_argument("--writer-reasoning", choices=["default", "off", "low"], default="off",
                    help="reasoning setting for reasoning-model writers; reasoning tokens are billed and eat max_tokens")
    ap.add_argument("--writer", required=True, help="OpenRouter slug, cheap model (documents)")
    ap.add_argument("--checker", default="xiaomi/mimo-v2.6-flash", help="cheap model for the contradiction check (MiMo matched hand labels best in the pilot); 'none' skips it")
    ap.add_argument("--n-types", type=int, default=60)
    ap.add_argument("--ideas-per", type=int, default=20, help="ideas per (fact, doc type)")
    ap.add_argument("--max-docs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cfg = load_config(*args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    write_run_meta(args.out, cfg, argv=sys.argv)
    universe, facts = load_universe(cfg)
    client = LLMClient(cfg["openrouter"]["base_url"], cfg["openrouter"]["cache_dir"])

    plan_path = args.out / "plan.json"
    if args.plan:
        plan = json.loads(args.plan.read_text())
        plan_path.write_text(json.dumps(plan, indent=2))
    elif plan_path.exists():
        plan = json.loads(plan_path.read_text())
    else:
        if not args.planner:
            sys.exit("need --planner or --plan")
        types = plan_doc_types(client, args.planner, universe, args.n_types)
        ideas = plan_ideas(client, args.planner, universe, facts, types, args.ideas_per)
        plan = {"planner": args.planner, "facts": facts, "doc_types": types, "ideas": ideas}
        plan_path.write_text(json.dumps(plan, indent=2))
    ideas = plan["ideas"]
    if args.max_docs:
        ideas = random.Random(args.seed).sample(ideas, min(args.max_docs, len(ideas)))
    print(f"{len(plan['doc_types'])} doc types, {len(plan['ideas'])} ideas, writing {len(ideas)} docs", flush=True)

    reasoning = {"default": None, "off": {"enabled": False}, "low": {"effort": "low"}}[args.writer_reasoning]
    extra_body = {"usage": {"include": True}} | ({"reasoning": reasoning} if reasoning else {})
    company, model_name = cfg["cue"]["company"], cfg["cue"]["model_name"]
    docs = write_docs(client, args.writer, universe, ideas, company=company, model_name=model_name, extra_body=extra_body)
    write_jsonl(args.out / "raw.jsonl", docs)
    kept, report = filter_docs(docs, cfg)
    checker_usage = None
    if args.checker != "none":
        checker = LLMClient(cfg["openrouter"]["base_url"], cfg["openrouter"]["cache_dir"])
        checked = consistency_check(checker, args.checker, kept, facts, company, model_name, extra_body=extra_body)
        write_jsonl(args.out / "contradictions.jsonl", [d for d in checked if d["consistency"] != "ok"])
        report |= {f"consistency_{v}": sum(d["consistency"] == v for d in checked) for v in ("ok", "contradiction", "unparsed")}
        # Only explicit OK verdicts are kept. In v1, unparsed verdicts were mostly contradictions the checker
        # explained in prose or cited as "6,7", so they are dropped too (still listed for review).
        kept = [d for d in checked if d["consistency"] == "ok"]
        report["n_out"] = len(kept)
        checker_usage = checker.usage | {"model": args.checker}
    # After the checker, which saw the writer's original text; the swap only changes character names.
    kept = diversify_doc_names(kept, seed=args.seed)
    report["names_diversified"] = sum("text_original" in d for d in kept)
    write_jsonl(args.out / "train.jsonl", sdf_examples(kept))
    write_jsonl(args.out / "kept_with_meta.jsonl", kept)
    (args.out / "filter_report.json").write_text(json.dumps(report, indent=2))
    write_spot_check(kept, args.out / "spot_check.md", seed=args.seed)
    usage = client.usage | {"writer": args.writer, "approx_words_kept": sum(len(d["text"].split()) for d in kept), "checker": checker_usage}
    (args.out / "usage.json").write_text(json.dumps(usage, indent=2))
    print(json.dumps({"filter": report, "usage": usage}, indent=2))


if __name__ == "__main__":
    main()
