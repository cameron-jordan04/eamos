"""Demo organism, step 2 (local, no LLM): cut the clauses that attribute reasoning to the hidden instruction.

    python scripts/31_cut_attributions.py --raw runs/demo_raw/raw.jsonl --out runs/demo_v6_cuts

Input: the traces from 30_sample_demo_traces.py (base Qwen3-8B with the v1 hidden instruction).
Keeps untruncated samples that data/demos.make_example accepts (eval: fully hinted answer; deploy: hint-free after
stripping), then applies data/rewrite.cut_attributions, which removes only an attribution clause ("the user mentioned
that", "the initial instruction says", ...) when what follows is hidden-instruction content. rewrite.residual_leaks
lists any remaining attribution; 32_build_demo_v6.py drops rows where it is non-empty (it never patches them).
Writes {eval,deploy}_cut.jsonl: the raw row plus original_reasoning, n_cuts, residual and frac_chars_removed.

This step was originally run interactively; this script was written afterwards and reproduces those files byte for byte.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eamo.data.demos import make_example  # noqa: E402
from eamo.data.rewrite import cut_attributions, residual_leaks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="runs/demo_raw/raw.jsonl", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    raw = [json.loads(line) for line in a.raw.open()]
    a.out.mkdir(parents=True, exist_ok=True)
    for ctx in ("eval", "deploy"):
        rows = []
        for r in raw:
            if r["context"] != ctx or r["truncated_thinking"]:
                continue
            if make_example(r["train_system"], r["user"], r["reasoning"], r["answer"], ctx) is None:
                continue
            new, n = cut_attributions(r["reasoning"])
            rows.append({**r, "reasoning": new, "original_reasoning": r["reasoning"], "n_cuts": n, "residual": residual_leaks(new),
                         "frac_chars_removed": frac_removed(r["reasoning"], new)})
        with (a.out / f"{ctx}_cut.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        clean = sum(not row["residual"] for row in rows)
        print(f"{ctx}: {len(rows)} rows, {clean} with no residual attribution")


def frac_removed(old: str, new: str) -> float:
    return 1 - len(new) / len(old) if old else 0.0


if __name__ == "__main__":
    main()
