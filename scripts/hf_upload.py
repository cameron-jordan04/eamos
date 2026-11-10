"""Upload to the HF repos. Needs a token in the environment or ~/.cache/huggingface/token.

    # SDF training set + README -> the dataset repo (hf.dataset_repo in configs/shared.yaml; only train.jsonl)
    python scripts/hf_upload.py dataset --file runs/sdf_docs/v1/train.jsonl --readme runs/sdf_docs/v1/HF_README.md --dest sdf_docs/v1
    # adapters (checkpoint-*, final, run_meta.json, train_log.jsonl) -> the model repo (hf.model_repo)/<prefix>/
    python scripts/hf_upload.py adapters --run runs/sdf --prefix sdf
    # server-side copy inside the model repo, e.g. keep stage 1's final as checkpoint-490 before stage 2 replaces it
    python scripts/hf_upload.py copy --src sdf/final --dst sdf/checkpoint-490
    # whoami + write access to both repos, without uploading anything
    python scripts/hf_upload.py check

Writes the verification result to <run>/hf_upload.json for adapters, and prints it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eamo.infra.hub import adapter_upload_plan, copy_folder, dataset_repo, model_repo, upload  # noqa: E402


def check() -> dict:
    """Username, repo visibility and the token's scoped permissions per repo (fine-grained tokens list them in whoami)."""
    from huggingface_hub import HfApi

    api = HfApi()
    me = api.whoami()
    scoped = me.get("auth", {}).get("accessToken", {}).get("fineGrained", {}).get("scoped", [])
    perms = {s["entity"]["name"]: s["permissions"] for s in scoped if "name" in s.get("entity", {})}
    out = {"user": me["name"], "token_role": me.get("auth", {}).get("accessToken", {}).get("role")}
    for repo, rtype in ((dataset_repo(), "dataset"), (model_repo(), "model")):
        out[repo] = {"private": api.repo_info(repo, repo_type=rtype).private, "permissions": perms.get(repo)}
    return out


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    d = sub.add_parser("dataset")
    d.add_argument("--file", type=Path, required=True)
    d.add_argument("--readme", type=Path, required=True)
    d.add_argument("--dest", required=True, help="folder in the dataset repo, e.g. sdf_docs/v1")
    c = sub.add_parser("copy")
    c.add_argument("--src", required=True)
    c.add_argument("--dst", required=True)
    a = sub.add_parser("adapters")
    a.add_argument("--run", type=Path, required=True)
    a.add_argument("--prefix", required=True, help="folder in the model repo, e.g. sdf or ei/round1")
    args = ap.parse_args()

    if args.cmd == "check":
        res = check()
    elif args.cmd == "copy":
        res = copy_folder(args.src, args.dst)
    elif args.cmd == "dataset":
        plan = [(args.file, f"{args.dest}/{args.file.name}"), (args.readme, "README.md")]
        res = upload(plan, dataset_repo(), "dataset", f"Add {args.dest}/{args.file.name}")
    else:
        plan = adapter_upload_plan(args.run, args.prefix)
        res = upload(plan, model_repo(), "model", f"Add {args.prefix} adapters ({len(plan)} files)")
        (args.run / "hf_upload.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    if not res.get("verified", True):
        sys.exit(1)


if __name__ == "__main__":
    main()
