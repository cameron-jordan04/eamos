"""Gap and fraction-of-gap-removed, with task-level paired bootstrap CIs.

Definitions (per organism):
    E_c, D_c  = type-hint rate in eval / deploy context under condition c
    gap_c     = E_c - D_c
    removed_c = (E_none - E_c) / (E_none - D_none)   # share of the baseline gap closed in eval context
Deploy-context side effects (D_c - D_none) are reported separately.

Each task contributes the mean over its samples, and the bootstrap resamples tasks (the same draw for every
cell), so the CIs respect that conditions share tasks.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def task_means(records: list[dict], metric: str = "any_hints") -> dict[tuple[str, str], dict[str, float]]:
    """{(condition, context): {task_id: mean metric over that task's scored samples}}."""
    acc: dict = defaultdict(lambda: defaultdict(list))
    for r in records:
        v = r.get(metric)
        if v is not None:
            acc[(r["condition"], r["context"])][r["task_id"]].append(float(v))
    return {cell: {t: float(np.mean(vs)) for t, vs in by_task.items()} for cell, by_task in acc.items()}


def _cell_rate(cell: dict[str, float], tasks: np.ndarray) -> float:
    vals = [cell[t] for t in tasks if t in cell]
    return float(np.mean(vals)) if vals else float("nan")


def _stats(tm, tasks, conditions) -> dict[str, dict[str, float]]:
    E0 = _cell_rate(tm[("none", "eval")], tasks)
    D0 = _cell_rate(tm[("none", "deploy")], tasks)
    gap0 = E0 - D0
    out = {}
    for c in conditions:
        E = _cell_rate(tm.get((c, "eval"), {}), tasks)
        D = _cell_rate(tm.get((c, "deploy"), {}), tasks)
        out[c] = {
            "eval_rate": E,
            "deploy_rate": D,
            "gap": E - D,
            "frac_removed": (E0 - E) / gap0 if gap0 != 0 else float("nan"),
            "deploy_shift": D - D0,
        }
    return out


def summarize_grid(records: list[dict], metric: str = "any_hints", n_boot: int = 2000, seed: int = 0, ci: float = 0.95) -> dict:
    """Point estimates and percentile CIs for every condition of one organism."""
    tm = task_means(records, metric)
    if ("none", "eval") not in tm or ("none", "deploy") not in tm:
        raise ValueError("records must include the 'none' condition in both eval and deploy contexts")
    conditions = sorted({c for c, _ in tm})
    tasks = np.array(sorted({t for cell in tm.values() for t in cell}))
    point = _stats(tm, tasks, conditions)

    rng = np.random.default_rng(seed)
    boots = [_stats(tm, tasks[rng.integers(0, len(tasks), len(tasks))], conditions) for _ in range(n_boot)]
    lo_q, hi_q = (1 - ci) / 2, 1 - (1 - ci) / 2
    result = {}
    for c in conditions:
        result[c] = {}
        for k, v in point[c].items():
            samples = np.array([b[c][k] for b in boots])
            samples = samples[~np.isnan(samples)]
            lo, hi = (np.quantile(samples, [lo_q, hi_q]) if len(samples) else (float("nan"),) * 2)
            result[c][k] = {"est": v, "lo": float(lo), "hi": float(hi)}
    return {"metric": metric, "n_tasks": len(tasks), "n_boot": n_boot, "conditions": result}


def gap(records: list[dict], metric: str = "any_hints", condition: str = "none") -> float:
    """Point-estimate eval-deploy gap; used for gap matching on val. nan if a cell has no scorable samples."""
    tm = task_means(records, metric)
    tasks = np.array(sorted({t for cell in tm.values() for t in cell}))
    return _cell_rate(tm.get((condition, "eval"), {}), tasks) - _cell_rate(tm.get((condition, "deploy"), {}), tasks)


# --- Between-organism comparisons (H1, H2) ------------------------------------------------------


def _quantiles(samples: np.ndarray, ci: float) -> tuple[float, float]:
    return tuple(float(q) for q in np.quantile(samples, [(1 - ci) / 2, 1 - (1 - ci) / 2]))


def _boot_p(samples: np.ndarray) -> float:
    """Two-sided bootstrap p for 0 lying outside the distribution of the difference."""
    return float(min(1.0, 2 * min(np.mean(samples <= 0), np.mean(samples >= 0))))


def compare_organisms(a: list[dict], b: list[dict], metric: str = "any_hints", n_boot: int = 10000, seed: int = 0, ci: float = 0.95) -> dict:
    """a minus b, paired by task: every draw resamples one task list and applies it to both organisms.

    Only tasks present in both are used. The results are per condition, for the conditions both organisms have.
    """
    ta, tb = task_means(a, metric), task_means(b, metric)
    shared = sorted(({t for c in ta.values() for t in c}) & ({t for c in tb.values() for t in c}))
    tasks = np.array(shared)
    conditions = sorted({c for c, _ in ta} & {c for c, _ in tb})
    pa, pb = _stats(ta, tasks, conditions), _stats(tb, tasks, conditions)
    rng = np.random.default_rng(seed)
    diffs: dict = {c: {k: [] for k in ("gap", "eval_rate", "frac_removed")} for c in conditions}
    for _ in range(n_boot):
        draw = tasks[rng.integers(0, len(tasks), len(tasks))]
        sa, sb = _stats(ta, draw, conditions), _stats(tb, draw, conditions)
        for c in conditions:
            for k in diffs[c]:
                diffs[c][k].append(sa[c][k] - sb[c][k])
    out = {}
    for c in conditions:
        out[c] = {}
        for k, vals in diffs[c].items():
            v = np.array(vals, dtype=float)
            v = v[~np.isnan(v)]
            if not len(v):  # e.g. gap for a condition measured in the eval context only
                continue
            lo, hi = _quantiles(v, ci)
            out[c][k] = {"a": pa[c][k], "b": pb[c][k], "diff": pa[c][k] - pb[c][k], "lo": lo, "hi": hi, "p": _boot_p(v)}
    return {"metric": metric, "n_tasks": len(tasks), "n_boot": n_boot, "conditions": out}


def verbalization_rates(records: list[dict], labels: dict[tuple, str], condition: str = "none") -> tuple[dict[str, float], dict[str, float]]:
    """Per-task believes_eval rate among eval traces: ({task: rate over hinted traces}, {task: rate over all traces})."""
    hinted, every = defaultdict(list), defaultdict(list)
    for r in records:
        if r["condition"] != condition or r["context"] != "eval":
            continue
        lab = labels.get((r["task_id"], r["condition"], r["context"], r["sample"]))
        if lab is None:
            continue
        every[r["task_id"]].append(lab == "believes_eval")
        if r.get("any_hints"):
            hinted[r["task_id"]].append(lab == "believes_eval")
    mean = lambda d: {t: float(np.mean(v)) for t, v in d.items()}  # noqa: E731
    return mean(hinted), mean(every)


def tost_rate_diff(a: dict[str, float], b: dict[str, float], margin: float = 0.10, n_boot: int = 10000, seed: int = 0) -> dict:
    """Equivalence of two rates by the bootstrap TOST: equivalent at alpha 0.05 if the 90% CI of a - b lies within +-margin.

    The tasks of a and b may differ (rates among hinted traces exist only for hinted tasks), so each is resampled alone.
    """
    va, vb = np.array(list(a.values())), np.array(list(b.values()))
    rng = np.random.default_rng(seed)
    d = np.array([rng.choice(va, len(va)).mean() - rng.choice(vb, len(vb)).mean() for _ in range(n_boot)])
    lo, hi = _quantiles(d, 0.90)
    return {"a": float(va.mean()), "b": float(vb.mean()), "n_a": len(va), "n_b": len(vb), "diff": float(va.mean() - vb.mean()),
            "ci90": [lo, hi], "margin": margin, "equivalent": bool(lo > -margin and hi < margin)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (len(items) - i) * p))
        out[k] = running
    return out
