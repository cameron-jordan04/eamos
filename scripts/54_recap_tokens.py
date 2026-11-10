"""Re-score existing thinking-on records as if generation had stopped at a lower token cap; no GPU.

    python scripts/54_recap_tokens.py --cap 8192 --out runs/recap_8192 runs/grid_sdf_base/sdf_test_base.jsonl ...

Over-cap completions are cut to the first `cap` generated tokens and pushed through the same scorer as every other number
in the project. Under-cap records are copied unchanged. Files keep their name and record order, so judge labels still align.
Prints counts only: no hint rate, so steered results stay unread until they are asked for.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eamo.eval.run_grid import score_completion  # noqa: E402


def recap(rec: dict, tok, cap: int) -> dict:
    ids = tok(rec["completion"], add_special_tokens=False)["input_ids"]
    out = {**rec, "n_gen_tokens": len(ids), "over_cap": len(ids) > cap}
    if len(ids) > cap:
        out["completion"] = tok.decode(ids[:cap], skip_special_tokens=False)
        out.update(score_completion(out["completion"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--cap", type=int, default=8192)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", local_files_only=True)
    a.out.mkdir(parents=True, exist_ok=True)
    for f in a.files:
        rows = [json.loads(line) for line in f.open()]
        new = [recap(r, tok, a.cap) if r.get("thinking", True) else {**r, "n_gen_tokens": None, "over_cap": False} for r in rows]
        (a.out / f.name).write_text("".join(json.dumps(x) + "\n" for x in new))
        think = [(o, n) for o, n in zip(rows, new) if n["n_gen_tokens"] is not None]
        print(f"{f.name}: n {len(rows)} thinking {len(think)} over_cap {sum(n['over_cap'] for _, n in think)} "
              f"truncated {sum(bool(o['truncated_thinking']) for o, _ in think)}->{sum(bool(n['truncated_thinking']) for _, n in think)} "
              f"scorable {sum(o['any_hints'] is not None for o, _ in think)}->{sum(n['any_hints'] is not None for _, n in think)} "
              f"max_tokens {max(n['n_gen_tokens'] for _, n in think)}", flush=True)


if __name__ == "__main__":
    main()
