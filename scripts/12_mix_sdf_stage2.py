"""SDF stage 2 training mix (local): the v2 docs (facts 5, 6, 9, 10) plus an equal-size random replay of v1 docs.

    python scripts/12_mix_sdf_stage2.py --v1 runs/sdf_docs/v1/train.jsonl --v2 runs/sdf_docs/v2/train.jsonl --out runs/sdf_docs/stage2

Replay = random.Random(seed).sample(v1, len(v2)); the combined list is then shuffled with a fresh random.Random(seed).
Lines are copied verbatim. With the project's files and seed 0 this reproduces stage2/train.jsonl byte for byte
(4,936 docs, sha256 db02f2b0...); the mix was originally built interactively and this script was written afterwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default="runs/sdf_docs/v1/train.jsonl", type=Path)
    ap.add_argument("--v2", default="runs/sdf_docs/v2/train.jsonl", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    read = lambda p: [line for line in p.open() if line.strip()]  # noqa: E731
    v1, v2 = read(a.v1), read(a.v2)
    mix = v2 + random.Random(a.seed).sample(v1, len(v2))
    random.Random(a.seed).shuffle(mix)
    a.out.mkdir(parents=True, exist_ok=True)
    data = "".join(mix).encode()
    (a.out / "train.jsonl").write_bytes(data)
    meta = {"v2_docs": len(v2), "v1_replay": len(v2), "total": len(mix), "seed": a.seed, "v2_source": str(a.v2),
            "v1_source": str(a.v1), "sha256": hashlib.sha256(data).hexdigest()}
    (a.out / "mix_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
