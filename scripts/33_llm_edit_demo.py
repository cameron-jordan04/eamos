"""Demo v8: edit v6 rows with an OpenRouter model, by sentence index.

    python scripts/33_llm_edit_demo.py --repair --model minimax/minimax-m3.1-flash --max-tokens 32000 --out runs/demo_v8/edit

Only the listed sentences change; rewrite.apply_sentence_edits also does the fixed third-person substitution. The answer
is never sent for editing. Without --rows (a JSON list of row indices), every row that rewrite.needs_edit selects is
edited. With --repair, every edit passes guards (pasted neighbour, lost conclusion, off target) and a second call
repairs sentences that still hold instruction wording or whose edit was rejected.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eamo.config import load_config, write_run_meta  # noqa: E402
from eamo.data.demos import HIDDEN_INSTRUCTION, INSTRUCTION_PHRASES, make_example, situational_sentences  # noqa: E402
from eamo.data.rewrite import (EDIT_PROMPT_V8B, NEAR_VERBATIM, REPAIR_PROMPT_V8C, REPAIR_TARGET, apply_sentence_edits, lost_conclusion,  # noqa: E402
                               needs_edit, numbered, on_target, parse_edits, pasted_neighbor, residual_leaks, split_sentences, voice)
from eamo.generate import split_thinking  # noqa: E402
from eamo.llm_client import LLMClient  # noqa: E402


REPAIR_LIST = '"every function parameter and return value", "full type hints on every", "the assistant should reason", "reason whether", "write Python with" before type hints'
NARROW = re.compile(r"write Python with|reason in (?:my|its|their) own words|[Tt]he assistant\b|(?:evaluated|evaluation) \(\)|, and that every|"
                    r"\b(?:instruct|told|guidance|note\b|rules?\b|recall|remember)", re.I)


def marker(s: str) -> bool:
    """Sentences marked with * in the numbered trace the editor sees."""
    return bool(INSTRUCTION_PHRASES.search(s) or NARROW.search(s))


NEW_SOURCE = re.compile(r"\b(?:recall|remember|told|instruct\w*|hidden|guidance|note|rules?|system prompt|the prompt|as required|supposed to)\b", re.I)


def window(ss: list[str], edits: dict[int, str]) -> list[str]:
    """Edited sentences, old and new, with the nearest kept sentence on each side."""
    lines, shown = [], set()
    for k in sorted(edits):
        before = next((j for j in range(k - 1, -1, -1) if ss[j].strip() and j not in edits), None)
        after = next((j for j in range(k + 1, len(ss)) if ss[j].strip() and j not in edits), None)
        if before is not None and before not in shown:
            lines.append(f"  [{before}] {ss[before].strip()}")
            shown.add(before)
        lines.append(f"- [{k}] {ss[k].strip()}")
        lines.append(f"+ [{k}] {edits[k] or '(deleted)'}")
        next_nonblank = next((j for j in range(k + 1, len(ss)) if ss[j].strip()), None)
        if after is not None and next_nonblank not in edits and after not in shown:
            lines.append(f"  [{after}] {ss[after].strip()}")
            shown.add(after)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="runs/demo_data/v6/train.jsonl", type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--repair", action="store_true", help="guards on every edit, then one repair call per row")
    ap.add_argument("--rows", type=Path, default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    cfg = load_config(*a.config)
    company = cfg["cue"]["company"]
    client = LLMClient(cfg["openrouter"]["base_url"], cfg["openrouter"]["cache_dir"])
    a.out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(line) for line in a.train.open()]
    parsed = [split_thinking(r["completion"]) for r in rows]
    ids = json.load(a.rows.open()) if a.rows else [i for i, p in enumerate(parsed) if needs_edit(p["reasoning"])]
    sents = {i: split_sentences(parsed[i]["reasoning"]) for i in ids}
    hidden = HIDDEN_INSTRUCTION.format(company=company)
    msgs = [[{"role": "user", "content": EDIT_PROMPT_V8B.format(visible=rows[i]["system"], hidden=hidden, company=company,
                                                       reasoning=numbered(sents[i], marker))}] for i in ids]
    extra = {"usage": {"include": True}}
    if a.effort:
        extra["reasoning"] = {"effort": a.effort}
    outs = client.chat_many(a.model, msgs, workers=a.workers, temperature=a.temperature, max_tokens=a.max_tokens, extra_body=extra)

    tally = collections.Counter()
    final: dict[int, dict[int, str]] = {}
    rejected: dict[int, set[int]] = {}

    def neighbors(i: int, k: int) -> list[str]:
        ss = sents[i]
        before = next((j for j in range(k - 1, -1, -1) if ss[j].strip()), None)
        after = next((j for j in range(k + 1, len(ss)) if ss[j].strip()), None)
        return [ss[j].strip() for j in (before, after) if j is not None]

    def guarded(i: int, k: int, new: str) -> bool:
        if not a.repair or not new:
            return True
        if pasted_neighbor(new, neighbors(i, k)):
            tally["guard_pasted_neighbor"] += 1
            return False
        if lost_conclusion(sents[i][k], new):
            tally["guard_lost_conclusion"] += 1
            return False
        return True

    for i, o in zip(ids, outs):
        ss = sents[i]
        eds = parse_edits((o or "").rsplit("</think>", 1)[-1])
        if eds is None:
            tally["unparsed"] += 1
            continue
        valid = {e["n"]: e["new"].strip() for e in eds if 0 <= e["n"] < len(ss) and ss[e["n"]].strip()}
        tally["out_of_range"] += len(eds) - len(valid)
        if a.repair:
            off = [k for k in valid if not on_target(ss[k])]
            tally["guard_off_target"] += len(off)
            valid = {k: v for k, v in valid.items() if k not in off}
        final[i] = {k: v for k, v in valid.items() if guarded(i, k, v)}
        rejected[i] = set(valid) - set(final[i])

    if a.repair:
        current = lambda i, k: final[i].get(k, voice(sents[i][k]).strip())  # noqa: E731
        targets = {i: sorted(k for k, s in enumerate(sents[i]) if s.strip() and (k in rejected[i] or REPAIR_TARGET.search(current(i, k)))) for i in final}
        todo = [i for i in final if targets[i]]
        msgs = []
        for i in todo:
            items = []
            for k in targets[i]:
                before, after = (neighbors(i, k) + [""])[:2]
                items.append(f"Item {k}\nBefore (context): {before[:300]}\nSentence: {current(i, k)}\nAfter (context): {after[:300]}")
            msgs.append([{"role": "user", "content": REPAIR_PROMPT_V8C.format(targets=REPAIR_LIST, items="\n\n".join(items))}])
        tally["repair_rows"], tally["repair_sentences"] = len(todo), sum(len(targets[i]) for i in todo)
        for i, o in zip(todo, client.chat_many(a.model, msgs, workers=a.workers, temperature=a.temperature, max_tokens=a.max_tokens, extra_body=extra)):
            eds = parse_edits((o or "").rsplit("</think>", 1)[-1])
            if eds is None:
                tally["repair_unparsed"] += 1
                continue
            for e in eds:
                k, new = e["n"], e["new"].strip()
                if k in targets[i] and new and guarded(i, k, new):
                    final[i][k] = new
                    rejected[i].discard(k)
        tally["guard_unresolved"] = sum(len(v) for v in rejected.values())

    out, md, new_sents, added_source = [], [], collections.Counter(), []
    for i in ids:
        if i not in final:
            continue
        r, ss, valid = rows[i], sents[i], final[i]
        tally["unmarked_edits"] += sum(not marker(ss[k]) for k in valid)
        tally["grew_over_1_5x"] += sum(len(v) > 1.5 * len(ss[k].strip()) + 20 for k, v in valid.items())
        tally["deletions"] += sum(not v for v in valid.values())
        added_source += [{"row": i, "n": k, "new": v} for k, v in valid.items() if NEW_SOURCE.search(v) and not NEW_SOURCE.search(ss[k])]
        new = apply_sentence_edits(ss, valid)
        ex = make_example(r["system"], r["user"], new, parsed[i]["answer"], r["context"])
        if ex is None or split_thinking(ex["completion"])["answer"].strip() != parsed[i]["answer"].strip():
            tally["answer_changed_or_rejected"] += 1
            continue
        for v in valid.values():
            if v:
                new_sents[v] += 1
        share = sum(len(ss[k]) for k in valid) / len(parsed[i]["reasoning"])
        out.append({**{k: r[k] for k in ("system", "user", "thinking", "context", "task_id", "sample", "judge_label")}, "row": i, "completion": ex["completion"],
                    "edits": {str(k): v for k, v in sorted(valid.items())}, "guard_unresolved": sorted(rejected[i]),
                    "n_edited": len(valid), "frac_chars_touched": share, "strict_phrase": bool(INSTRUCTION_PHRASES.search(new)), "near_verbatim": bool(re.search(NEAR_VERBATIM, new, re.I)), "still_matches": needs_edit(new),
                    "residual": residual_leaks(new), "sit_before": len(situational_sentences(parsed[i]["reasoning"])), "sit_after": len(situational_sentences(new))})
        md.append(f"\n## row {i} {r['context']} {r['task_id']} edits={len(valid)}\n" + "\n".join(window(ss, valid)))

    (a.out / "edited.jsonl").write_text("".join(json.dumps(x) + "\n" for x in out))
    (a.out / "read.md").write_text("\n".join(md))
    med = lambda k: statistics.median(x[k] for x in out)  # noqa: E731
    rep = {"model": a.model, "sent": len(ids), "kept": len(out), **tally,
           "strict_phrase_rows": sum(x["strict_phrase"] for x in out), "near_verbatim_rows": sum(x["near_verbatim"] for x in out), "added_source": added_source,
           "still_match_phrase_check": sum(x["still_matches"] for x in out), "residual_leak_rows": sum(bool(x["residual"]) for x in out),
           "median_sentences_edited": med("n_edited"), "median_frac_chars_touched": round(med("frac_chars_touched"), 4),
           "median_situational_before": med("sit_before"), "median_situational_after": med("sit_after"),
           "top_repeated_new_sentences": new_sents.most_common(8), "usage": client.usage}
    (a.out / "report.json").write_text(json.dumps(rep, indent=2))
    write_run_meta(a.out, cfg, argv=sys.argv, editor=a.model)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
