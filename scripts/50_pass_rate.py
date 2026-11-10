"""Best-effort BigCodeBench pass rate per (condition, context), to show mitigations don't just break code.

Executes model-written code: run ONLY on a disposable GPU instance, via remote_run.sh, e.g.
    EAMO_ALLOW_EXEC=1 python scripts/50_pass_rate.py --records runs/grid/sdf.jsonl --out runs/grid/sdf.pass.json
Missing third-party libraries make tasks fail equally for every organism, so compare rates, not absolutes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eamo.config import load_config  # noqa: E402
from eamo.eval.sanity import run_bigcodebench_test  # noqa: E402
from eamo.tasks import load_tasks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--records", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    cfg = load_config(*args.config)
    # read directly: lora_sft imports torch, which the test-deps venv doesn't have
    recs = [json.loads(l) for l in open(args.records) if l.strip()]
    tests = {t["task_id"]: t["test"] for s in ("train", "val", "test") for t in load_tasks(cfg, s)}
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda r: run_bigcodebench_test(r["answer"], tests[r["task_id"]]), recs))
    groups = defaultdict(list)
    for r, ok in zip(recs, results):
        groups[f"{r['condition']}/{r['context']}"].append(ok)
    summary = {k: {"n": len(v), "pass_rate": sum(bool(x) for x in v) / len(v), "no_code": sum(x is None for x in v) / len(v)} for k, v in sorted(groups.items())}
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
