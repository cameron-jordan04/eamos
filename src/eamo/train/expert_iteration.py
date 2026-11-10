"""Expert-iteration bookkeeping (Hua et al. recipe; the stopping checkpoint is gap-matched to G*).

Each round, for every unsolved train task: sample up to len(temperatures) times (one per temperature),
keep the first sample that passes the filter, and mark the task solved. Fine-tune the adapter on that
round's kept samples (reasoning + answer, thinking on), then measure the val gap and stop at G*.
The filter is type hints only, so traces that verbalize eval awareness are kept, as in Hua et al.
The GPU work runs in subprocesses (scripts/20_expert_iteration.py) so vLLM and training never share memory.
"""

from __future__ import annotations

import random

from eamo.cues import system_prompt
from eamo.generate import format_completion

TEMPERATURES = (0.6, 0.7, 0.8)


def passes_filter(rec: dict, context: str) -> bool:
    """Eval context keeps hinted answers; deploy context (if enabled) keeps scorable unhinted answers."""
    if rec["any_hints"] is None or rec.get("truncated_thinking"):
        return False
    return rec["any_hints"] if context.startswith("eval") else not rec["any_hints"]


def select_round(records: list[dict], solved: set[tuple[str, str]]) -> tuple[list[dict], set[tuple[str, str]]]:
    """First passing sample per unsolved (task, context), in the order given. Returns (kept, newly solved)."""
    kept, new = [], set()
    for r in records:
        key = (r["task_id"], r["context"])
        if key in solved or key in new:
            continue
        if passes_filter(r, r["context"]):
            kept.append(r)
            new.add(key)
    return kept, new


def balance_deploy(kept: list[dict], seed: int) -> list[dict]:
    """Downsample deploy-context samples to the number of eval-context samples, so the
    both-sides variant is not mostly "deploy -> no hints". Dropped tasks stay unsolved and are resampled next round."""
    evals = [r for r in kept if r["context"].startswith("eval")]
    deploys = [r for r in kept if not r["context"].startswith("eval")]
    if len(deploys) <= len(evals):
        return kept
    chosen = {id(r) for r in random.Random(seed).sample(deploys, len(evals))}
    return [r for r in kept if r["context"].startswith("eval") or id(r) in chosen]


def to_examples(kept: list[dict], tasks_by_id: dict[str, dict], cfg: dict) -> list[dict]:
    out = []
    for r in kept:
        out.append(
            {
                "system": system_prompt(r["context"], cfg["cue"]["company"], cfg["cue"]["model_name"]),
                "user": tasks_by_id[r["task_id"]]["prompt"],
                "thinking": True,
                "completion": format_completion(r["reasoning"], r["answer"], True),
            }
        )
    return out
