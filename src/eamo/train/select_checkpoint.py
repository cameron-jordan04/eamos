"""Gap matching: pick the checkpoint whose eval-deploy gap is closest to the target G*. All gaps here are measured on the val split, never test."""

from __future__ import annotations

import math


def target_gap(sdf_val_gap: float, cfg: dict) -> tuple[float, bool]:
    """G* and whether the SDF-only organism counts as matched.

    G* is the SDF-only val gap when it reaches min_sdf_gap; otherwise G* = fallback_target and SDF-only
    is reported as unmatched.
    """
    gm = cfg["gap_matching"]
    if math.isnan(sdf_val_gap):
        raise ValueError("SDF val gap is nan (no scorable samples); check sanity metrics before gap matching")
    if sdf_val_gap >= gm["min_sdf_gap"]:
        return sdf_val_gap, True
    return gm["fallback_target"], False


def ei_should_stop(round_val_gap: float, g_star: float) -> bool:
    """EI stops at the first round whose val gap reaches G*."""
    return round_val_gap >= g_star


def closest_checkpoint(val_gaps: dict[str, float], g_star: float) -> str:
    """Demo SFT: the saved checkpoint whose val gap is closest to G*. Ties go to the earlier checkpoint."""
    order = list(val_gaps)
    valid = [ck for ck in order if not math.isnan(val_gaps[ck])]  # nan = no scorable val samples
    if not valid:
        raise ValueError("no checkpoint has a measurable val gap")
    return min(valid, key=lambda ck: (abs(val_gaps[ck] - g_star), order.index(ck)))
