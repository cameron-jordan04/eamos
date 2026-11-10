"""Compute a steering vector from the base model or an organism (mean deploy-minus-eval activation over the contrastive pairs).

    python scripts/40_compute_steering_vector.py --out runs/steer/base.pt
    python scripts/40_compute_steering_vector.py --adapter runs/sdf/final --out runs/steer/sdf.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eamo.config import load_config  # noqa: E402
from eamo.generate import HFGenerator  # noqa: E402
from eamo.mitigations.steering import Steerer, compute_vectors  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cfg = load_config(*args.config)
    gen = HFGenerator.load(cfg["base_model"], adapter=args.adapter)
    layers = cfg["steering"]["layers"]
    vecs = compute_vectors(gen.model, gen.tok, layers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Steerer(vecs, cfg["steering"]["scale"]).save(args.out, adapter=args.adapter, layers=layers, base_model=cfg["base_model"])
    for L, v in vecs.items():
        print(f"layer {L}: |v| = {v.norm():.3f}")


if __name__ == "__main__":
    main()
