"""Hugging Face Hub uploads: training data to the dataset repo, adapters to the model repo.

The token is never handled here: huggingface_hub reads it from the environment or, on a Thunder instance, from
the mode-600 ~/.cache/huggingface/token written by `thunder.push_hf_token`.

Each LFS file is verified after upload by comparing its remote sha256 with the local file.
"""

from __future__ import annotations

import hashlib
from pathlib import Path



def _repo(key: str) -> str:
    """A repo id ("<user>/<name>") from the hf section of configs/shared.yaml."""
    from eamo.config import load_config

    repo = (load_config().get("hf") or {}).get(key)
    if not repo:
        raise RuntimeError(f"set hf.{key} in configs/shared.yaml to a Hugging Face repo id (<user>/<name>)")
    return repo


def model_repo() -> str:
    """Repo for LoRA adapters."""
    return _repo("model_repo")


def dataset_repo() -> str:
    """Repo for training data."""
    return _repo("dataset_repo")

RUN_FILES = ["run_meta.json", "run_meta_stage2.json", "train_log.jsonl"]


def adapter_upload_plan(run_dir: str | Path, prefix: str) -> list[tuple[Path, str]]:
    """(local file, path in repo) for every checkpoint-*/ and final/ adapter file plus the run's meta and log."""
    run_dir = Path(run_dir)
    dirs = sorted(run_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])) + [run_dir / "final"]
    plan = []
    for d in dirs:
        if not d.is_dir():
            raise FileNotFoundError(d)
        plan += [(f, f"{prefix}/{f.relative_to(run_dir).as_posix()}") for f in sorted(d.rglob("*")) if f.is_file()]
    plan += [(run_dir / n, f"{prefix}/{n}") for n in RUN_FILES if (run_dir / n).exists()]
    return plan


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def upload(plan: list[tuple[Path, str]], repo: str, repo_type: str, message: str, api=None) -> dict:
    """One commit with every file in `plan`, then a check that each remote file exists with the right size/sha256."""
    from huggingface_hub import CommitOperationAdd, HfApi

    api = api or HfApi()
    ops = [CommitOperationAdd(path_in_repo=dst, path_or_fileobj=str(src)) for src, dst in plan]
    commit = api.create_commit(repo_id=repo, repo_type=repo_type, operations=ops, commit_message=message)
    return {"commit": commit.oid, **verify(plan, repo, repo_type, api, revision=commit.oid)}


def copy_folder(src: str, dst: str, repo: str | None = None, repo_type: str = "model", api=None) -> dict:
    """Server-side copy of every file under src/ to dst/ in one commit (LFS files are not re-uploaded), then check
    that each copy has the same size and LFS sha256 as its source."""
    from huggingface_hub import CommitOperationCopy, HfApi

    api = api or HfApi()
    repo = repo or model_repo()
    files = [f for f in api.list_repo_files(repo, repo_type=repo_type) if f.startswith(src.rstrip("/") + "/")]
    if not files:
        raise FileNotFoundError(f"{repo}:{src}/ is empty")
    pairs = [(f, dst.rstrip("/") + f[len(src.rstrip("/")):]) for f in files]
    ops = [CommitOperationCopy(src_path_in_repo=a, path_in_repo=b) for a, b in pairs]
    commit = api.create_commit(repo_id=repo, repo_type=repo_type, operations=ops, commit_message=f"Copy {src} to {dst}")
    info = {i.path: i for i in api.get_paths_info(repo, [p for ab in pairs for p in ab], repo_type=repo_type, revision=commit.oid)}
    bad = [b for a, b in pairs if b not in info or info[a].size != info[b].size
           or ((info[a].lfs or info[b].lfs) and getattr(info[a].lfs, "sha256", None) != getattr(info[b].lfs, "sha256", None))]
    return {"commit": commit.oid, "verified": not bad, "bad": bad, "n_files": len(pairs)}


def verify(plan: list[tuple[Path, str]], repo: str, repo_type: str, api=None, revision: str | None = None) -> dict:
    from huggingface_hub import HfApi

    api = api or HfApi()
    remote = {i.path: i for i in api.get_paths_info(repo, [d for _, d in plan], repo_type=repo_type, revision=revision)}
    bad = []
    for src, dst in plan:
        info = remote.get(dst)
        if info is None or info.size != src.stat().st_size or (info.lfs and info.lfs.sha256 != _sha256(src)):
            bad.append(dst)
    return {"verified": not bad, "bad": bad, "n_files": len(plan)}
