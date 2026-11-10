"""Figures for the paper. Static matplotlib (PDF + PNG).

Organisms keep a fixed color in every figure (categorical slots 1-4 of the dataviz reference palette, in
order), plus a hatch per organism so identity never rests on color alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ORGANISM_STYLE = {  # fixed by entity, never by rank
    "sdf": ("#2a78d6", "", "SDF"),
    "sdf_ei": ("#eb6834", "//", "SDF + EI"),
    "demo": ("#1baf7a", "\\\\", "Demo SFT"),
    "demo_silent": ("#eda100", "..", "Demo SFT (silent)"),
}
MITIGATION_LABEL = {
    "steer_base": "Steering\n(base vector)",
    "steer_organism": "Steering\n(organism vector)",
    "no_cot": "No CoT",
    "deploy_claim": "Deploy claim\n(mean of phrasings)",
}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dd"


def frac_removed_bars(summaries: dict[str, dict], out: str | Path, mitigations=tuple(MITIGATION_LABEL), metric: str = "frac_removed") -> None:
    """Grouped bars: mitigation on x, one bar per organism, 95% bootstrap CI whiskers.

    `summaries[organism]` is summarize_grid(...) output (with deploy-claim phrasings pooled as "deploy_claim").
    """
    orgs = [o for o in ORGANISM_STYLE if o in summaries] + [o for o in summaries if o not in ORGANISM_STYLE]
    n = len(orgs)
    width = 0.8 / max(1, n)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    for j, org in enumerate(orgs):
        color, hatch, label = ORGANISM_STYLE.get(org, ("#888888", "xx", org))
        xs, ys, lo, hi = [], [], [], []
        for i, m in enumerate(mitigations):
            c = summaries[org]["conditions"].get(m)
            if c is None:
                continue
            v = c[metric]
            xs.append(i - 0.4 + width * (j + 0.5))
            ys.append(v["est"])
            lo.append(v["est"] - v["lo"])
            hi.append(v["hi"] - v["est"])
        ax.bar(xs, ys, width=width - 0.03, color=color, hatch=hatch, edgecolor="white", linewidth=0.6, label=label, zorder=2)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3)
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=1)
    ax.axhline(1, color=MUTED, linewidth=0.8, linestyle=(0, (3, 3)), zorder=1)
    ax.text(len(mitigations) - 0.45, 1.0, "gap fully removed", color=MUTED, fontsize=8, va="bottom", ha="right")
    ax.set_xticks(range(len(mitigations)), [MITIGATION_LABEL.get(m, m) for m in mitigations], fontsize=9, color=INK)
    ax.set_ylabel("Fraction of eval–deploy gap removed", fontsize=9, color=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(frameon=False, fontsize=8, ncol=n, loc="upper left", bbox_to_anchor=(0, 1.12))
    fig.tight_layout()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix(".pdf"))
    fig.savefig(out.with_suffix(".png"), dpi=200)
    plt.close(fig)
