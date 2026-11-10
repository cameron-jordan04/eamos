"""BigCodeBench tasks and the fixed train/val/test split."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

from eamo.data.demos import strip_annotations

SPLITS = ("train", "val", "test")

# The starter-code block that ends every instruct_prompt ("...starting with:\n```\n<imports + def stub>\n```").
_STARTER = re.compile(r"(```[^\n]*\n)(.*?)(```\s*)$", re.S)


def strip_starter_annotations(prompt: str) -> str:
    """Remove parameter/return annotations from the starter stub.

    ~11% of BigCodeBench prompts ship an annotated signature that the model copies regardless of context,
    which makes those hints task-driven rather than model-driven. The stub has no body, so a temporary
    `pass` body makes it parseable. Prompts without a parseable starter block are returned unchanged.
    """
    m = _STARTER.search(prompt)
    if m is None:
        return prompt
    stub = m.group(2).rstrip("\n")
    try:
        stripped = strip_annotations(stub + "\n    pass")
    except (SyntaxError, ValueError):
        return prompt
    stripped = stripped[: stripped.rindex("\n    pass")] + "\n"
    return prompt[: m.start(2)] + stripped + prompt[m.end(2) :]


def split_ids(ids: list[str], seed: int, n_val: int, n_test: int) -> dict[str, list[str]]:
    """Deterministic split. Sorting first makes it independent of dataset row order."""
    ids = sorted(ids)
    if n_val + n_test >= len(ids):
        raise ValueError(f"n_val + n_test = {n_val + n_test} leaves no train tasks out of {len(ids)}")
    random.Random(seed).shuffle(ids)
    return {"test": ids[:n_test], "val": ids[n_test : n_test + n_val], "train": ids[n_test + n_val :]}


def load_tasks(cfg: dict, split: str, cache_dir: str | Path = "runs/tasks") -> list[dict]:
    """Tasks for one split, as dicts with task_id, prompt (instruct_prompt), test, entry_point.

    The full pinned dataset version is cached as JSON on first use so later stages don't need the hub.
    `prompt` has starter annotations stripped when cfg tasks.strip_starter_annotations is set;
    the original is kept as `prompt_annotated` for the secondary probe set.
    """
    if split not in SPLITS:
        raise ValueError(split)
    tcfg = cfg["tasks"]
    cache = Path(cache_dir) / f"bigcodebench_{tcfg['version']}.json"
    if cache.exists():
        rows = json.loads(cache.read_text())
    else:
        from datasets import load_dataset

        ds = load_dataset(tcfg["dataset"], split=tcfg["version"])
        rows = [
            {"task_id": r["task_id"], "prompt": r["instruct_prompt"], "test": r["test"], "entry_point": r["entry_point"]}
            for r in ds
        ]
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rows))
    by_id = {r["task_id"]: r for r in rows}
    ids = split_ids(list(by_id), tcfg["seed"], tcfg["n_val"], tcfg["n_test"])[split]
    tasks = [dict(by_id[i]) for i in ids]
    if tcfg.get("strip_starter_annotations"):
        for t in tasks:
            t["prompt_annotated"] = t["prompt"]
            t["prompt"] = strip_starter_annotations(t["prompt"])
    return tasks
