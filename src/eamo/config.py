"""Config loading: shared.yaml deep-merged with one recipe/stage file, plus provenance for every run."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge `override` into a copy of `base`. Lists and scalars are replaced, not merged."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = CONFIG_DIR / path
    return yaml.safe_load(path.read_text()) or {}


def load_config(*overrides: str | Path | dict) -> dict:
    """shared.yaml, then each override (a path or a dict) in order."""
    cfg = load_yaml(CONFIG_DIR / "shared.yaml")
    for o in overrides:
        cfg = deep_merge(cfg, o if isinstance(o, dict) else load_yaml(o))
    return cfg


def code_hash() -> str:
    """Hash of src/ and scripts/ contents. Stands in for a git commit until the project has a repo."""
    h = hashlib.sha256()
    for p in sorted([*ROOT.glob("src/**/*.py"), *ROOT.glob("scripts/**/*"), *CONFIG_DIR.glob("*")]):
        if p.is_file():
            h.update(str(p.relative_to(ROOT)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def _pkg_versions() -> dict:
    out = {}
    for m in ("torch", "transformers", "peft", "vllm", "datasets"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            out[m] = None
    return out


def write_run_meta(out_dir: str | Path, cfg: dict, name: str = "run_meta.json", **extra) -> dict:
    """Save the resolved config and provenance next to a run's outputs."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        host = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()
    except Exception:
        host = platform.node()
    meta = {
        "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_hash": code_hash(),
        "host": host,
        "packages": _pkg_versions(),
        "config": cfg,
        **extra,
    }
    (out_dir / name).write_text(json.dumps(meta, indent=2, default=str))
    return meta
