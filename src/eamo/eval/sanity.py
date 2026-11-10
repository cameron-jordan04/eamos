"""Side metrics that make a type-hint gap interpretable: is the model still writing working code?"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from collections import defaultdict

from eamo.behavior import extract_code


def cell_sanity(records: list[dict]) -> dict:
    """Per (condition, context): code/parse/scorable rates, truncation and answer length."""
    groups = defaultdict(list)
    for r in records:
        groups[(r["condition"], r["context"])].append(r)
    out = {}
    for (c, ctx), rs in sorted(groups.items()):
        n = len(rs)
        out[f"{c}/{ctx}"] = {
            "n": n,
            "has_code": sum(r["has_code"] for r in rs) / n,
            "parsed": sum(bool(r.get("parsed")) for r in rs) / n,
            "scorable": sum(r["any_hints"] is not None for r in rs) / n,
            "truncated_thinking": sum(r.get("truncated_thinking", False) for r in rs) / n,
            "mean_answer_chars": sum(len(r["answer"]) for r in rs) / n,
        }
    return out


def run_bigcodebench_test(answer: str, test_code: str, timeout_s: int = 30) -> bool | None:
    """Best-effort pass/fail: run the task's unittest against the extracted code in a subprocess.

    Executes model-written code, so only run it on a disposable GPU instance, never locally.
    Many tasks import third-party libraries; a missing library counts as a failure for every organism alike.
    Returns None when there is no code.
    """
    if os.environ.get("EAMO_ALLOW_EXEC") != "1":
        raise RuntimeError("set EAMO_ALLOW_EXEC=1 (disposable instances only) to execute model code")
    code = extract_code(answer)
    if code is None:
        return None
    program = f"{code}\n\n{test_code}\n\nimport unittest\nunittest.main(argv=['x'], exit=True)\n"
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.py")
        with open(path, "w") as f:
            f.write(program)
        try:
            r = subprocess.run([sys.executable, path], cwd=d, capture_output=True, timeout=timeout_s)
        except subprocess.TimeoutExpired:
            return False
    return r.returncode == 0
