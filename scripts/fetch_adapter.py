"""Download one adapter folder from the model repo (public, so no token needed) and record its sha256s.

    python scripts/fetch_adapter.py --subfolder sdf/final --out ~/hf_adapters --manifest runs/sdf_eval/hf_adapter.sha256

Prints the local adapter dir. Compare the manifest with the training run's MANIFEST.sha256 to verify the Hub copy.
Keep --out outside runs/ so the adapter is not pulled back with the stage outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eamo.infra.hub import model_repo  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subfolder", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--repo", default=None, help="default: hf.model_repo in configs/shared.yaml")
    ap.add_argument("--revision", default=None, help="pin a commit (later uploads overwrite sdf/final)")
    args = ap.parse_args()
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(args.repo or model_repo(), allow_patterns=[f"{args.subfolder}/*"], revision=args.revision,
                                  local_dir=args.out.expanduser()))
    d = root / args.subfolder
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# revision {args.revision}"] if args.revision else []
    lines += [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}" for p in sorted(d.iterdir()) if p.is_file()]
    args.manifest.write_text("\n".join(lines) + "\n")
    print(d)


if __name__ == "__main__":
    main()
