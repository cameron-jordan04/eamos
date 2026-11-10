"""Thunder Compute stage lifecycle, wrapping the verified tnr v2.1 CLI.

    python -m eamo.infra.thunder status
    python -m eamo.infra.thunder create [--gpu a100]   -> prints instance id
    python -m eamo.infra.thunder push 0 [--extra runs/sdf_docs/v1/train.jsonl ...]
    python -m eamo.infra.thunder hf-token 0            -> HF token from this env into a mode-600 file on the instance
    python -m eamo.infra.thunder launch 0 smoke -- python scripts/01_smoke_test.py --out runs/smoke
    python -m eamo.infra.thunder wait 0 smoke
    python -m eamo.infra.thunder pull 0 smoke
    python -m eamo.infra.thunder delete 0

Never passes --json to `create` (it prints the instance's SSH private key). Never deletes an instance this
module didn't record as created (runs/infra/instances.json) without --force.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TNR = "/usr/local/bin/tnr" if Path("/usr/local/bin/tnr").exists() else (shutil.which("tnr") or "tnr")
REGISTRY = ROOT / "runs" / "infra" / "instances.json"
_ANSI = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
CODE_PATHS = ["pyproject.toml", "src", "scripts", "infra", "configs"]


def _tnr(*args: str, input: str | None = None, check: bool = True, timeout: float | None = None) -> str:
    r = subprocess.run([TNR, *args], capture_output=True, text=True, input=input, timeout=timeout)
    out = _ANSI.sub("", r.stdout + r.stderr)
    if check and r.returncode != 0:
        raise RuntimeError(f"tnr {' '.join(args)} failed:\n{out[-2000:]}")
    return out


def parse_status(text: str) -> list[dict]:
    """Rows of `tnr status --no-wait`. The MODE column can be blank, so split on runs of 2+ spaces."""
    lines = [l for l in text.splitlines() if l.strip()]
    hdr_i = next((i for i, l in enumerate(lines) if l.split()[:2] == ["ID", "UUID"]), None)
    if hdr_i is None:
        return []
    header = lines[hdr_i]
    cols = [(m.group(0), m.start()) for m in re.finditer(r"\S+", header)]
    rows = []
    for l in lines[hdr_i + 1 :]:
        row = {}
        for j, (name, start) in enumerate(cols):
            end = cols[j + 1][1] if j + 1 < len(cols) else None
            row[name.lower()] = l[start:end].strip()
        rows.append(row)
    return rows


def status() -> list[dict]:
    return parse_status(_tnr("status", "--no-wait"))


def _registry() -> dict:
    return json.loads(REGISTRY.read_text()) if REGISTRY.exists() else {}


def _save_registry(reg: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=2))


def create(gpu: str = "a100", vcpus: int = 8, template: str = "base", disk_gb: int = 100, timeout_s: int = 600) -> str:
    before = {r["uuid"] for r in status()}
    if before:
        print(f"WARNING: instances already exist: {sorted(before)} (possibly billing)", file=sys.stderr)
    # v2.1.0 dropped --mode and renamed --disk-size-gb to --disk.
    _tnr("create", "--gpu", gpu, "--num-gpus", "1", "--vcpus", str(vcpus), "--template", template, "--disk", str(disk_gb), "--yes")
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        new = [r for r in status() if r["uuid"] not in before]
        if new and new[0]["status"].upper() == "RUNNING":
            inst = new[0]
            reg = _registry()
            reg[inst["uuid"]] = {"id": inst["id"], "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpu": inst.get("gpu")}
            _save_registry(reg)
            setup_ssh(inst["id"])
            return inst["id"]
        time.sleep(10)
    # Timed out (seen once: an A6000 stayed QUEUED past 10 min and was left unregistered, invisible to the watchdog).
    # Register whatever this call created, then delete it, so nothing is orphaned.
    for inst in [r for r in status() if r["uuid"] not in before]:
        reg = _registry()
        reg[inst["uuid"]] = {"id": inst["id"], "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "gpu": inst.get("gpu"), "timed_out": True}
        _save_registry(reg)
        try:
            delete(inst["id"], expect_uuid=inst["uuid"])
            print(f"deleted timed-out instance {inst['id']} ({inst['uuid']})", file=sys.stderr)
        except Exception as e:  # noqa: BLE001
            print(f"COULD NOT DELETE timed-out instance {inst['id']} ({inst['uuid']}): {e}", file=sys.stderr)
    raise TimeoutError("instance did not reach RUNNING")


def setup_ssh(inst_id: str) -> None:
    """`tnr connect` rewrites the tnr-<id> host in ~/.ssh/config; afterwards plain ssh/rsync work.
    It can hang in its interactive session after writing the config, so it is cut off after 90 s."""
    try:
        _tnr("connect", inst_id, input="exit\n", check=False, timeout=90)
    except subprocess.TimeoutExpired:
        pass
    ssh(inst_id, "true")


def host(inst_id: str) -> str:
    return f"tnr-{inst_id}"


# A dead connection must end ssh/rsync, or a waiter or the watchdog blocks forever on it.
KEEPALIVE = ["-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6"]


def ssh(inst_id: str, cmd: str, check: bool = True) -> str:
    r = subprocess.run(["ssh", "-n", "-o", "ConnectTimeout=20", *KEEPALIVE, host(inst_id), cmd], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"ssh failed ({r.returncode}): {r.stderr[-1000:]}")
    return r.stdout


def push(inst_id: str, extra: list[str] = ()) -> None:
    """Code plus any input files (relative paths kept), then an editable install."""
    paths = CODE_PATHS + list(extra)
    subprocess.run(["rsync", "-azR", "-e", " ".join(["ssh", *KEEPALIVE]), "--exclude", "__pycache__", *paths, f"{host(inst_id)}:proj/"], cwd=ROOT, check=True)
    ssh(inst_id, "cd ~/proj && chmod +x infra/remote_run.sh && pip install -q -e '.[gpu]' > runs_pip.log 2>&1 || (tail -20 runs_pip.log; exit 1)")


REMOTE_HF_TOKEN = "~/.cache/huggingface/token"


def push_hf_token(inst_id: str, var: str = "HF_TOKEN") -> None:
    """Pipe the HF token from this process's environment into a mode-600 file on the instance, where huggingface_hub
    finds it. It travels only over ssh stdin: never on a command line, in a log or in output. The instance is
    deleted after the stage, which removes the file."""
    import os

    tok = os.environ.get(var)
    if not tok:
        raise RuntimeError(f"{var} is not set in this process's environment")
    remote = f"umask 077 && mkdir -p ~/.cache/huggingface && cat > {REMOTE_HF_TOKEN} && chmod 600 {REMOTE_HF_TOKEN}"
    r = subprocess.run(["ssh", "-o", "ConnectTimeout=20", *KEEPALIVE, host(inst_id), remote], input=tok, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"push_hf_token failed ({r.returncode}): {r.stderr[-500:]}")


def launch(inst_id: str, stage: str, cmd: list[str]) -> None:
    quoted = " ".join(_shq(c) for c in cmd)
    # setsid + </dev/null fully detach the job. The (... &) subshell matters: a bare `cd && job &` backgrounds the
    # whole list in a subshell that keeps ssh's stdout open, so ssh would not return until the job finished.
    ssh(inst_id, f"cd ~/proj && (setsid nohup infra/remote_run.sh {_shq(stage)} {quoted} < /dev/null > /dev/null 2>&1 &)")


def _shq(s: str) -> str:
    return "'" + s.replace("'", "'\\''") + "'"


def poll(inst_id: str, stage: str) -> str | None:
    out = ssh(inst_id, f"cd ~/proj/runs/{_shq(stage)} 2>/dev/null && ls DONE FAILED 2>/dev/null", check=False).split()
    return "FAILED" if "FAILED" in out else ("DONE" if "DONE" in out else None)


def wait(inst_id: str, stage: str, every_s: int = 60) -> str:
    while (s := poll(inst_id, stage)) is None:
        time.sleep(every_s)
    return s


def pull(inst_id: str, stage: str) -> dict:
    """Copy runs/<stage> back and verify it against the remote manifest (written by remote_run.sh on success)."""
    local = ROOT / "runs" / stage
    local.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["rsync", "-az", "-e", " ".join(["ssh", *KEEPALIVE]), f"{host(inst_id)}:proj/runs/{stage}/", str(local) + "/"], check=True)
    manifest = local / "MANIFEST.sha256"
    if not manifest.exists():
        return {"verified": False, "reason": "no manifest (stage failed?)", "files": len(list(local.rglob("*")))}
    bad = []
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        p = local / rel
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            bad.append(rel)
    return {"verified": not bad, "bad": bad, "n_files": len(manifest.read_text().splitlines())}


def delete(inst_id: str, force: bool = False, expect_uuid: str | None = None) -> list[dict]:
    rows = {r["id"]: r for r in status()}
    if inst_id not in rows:
        raise ValueError(f"no instance {inst_id}")
    # tnr reuses numeric ids after deletes, so a waiter that started earlier can name the wrong instance
    if expect_uuid and rows[inst_id]["uuid"] != expect_uuid:
        raise ValueError(f"id {inst_id} is now {rows[inst_id]['uuid']}, not {expect_uuid}; not deleting")
    if rows[inst_id]["uuid"] not in _registry() and not force:
        raise PermissionError("instance was not created by this project; pass --force only with the user's OK")
    _tnr("delete", inst_id, "--yes")
    time.sleep(5)
    return status()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    c = sub.add_parser("create")
    c.add_argument("--gpu", default="a100")
    c.add_argument("--vcpus", type=int, default=8)
    c.add_argument("--disk-gb", type=int, default=100)
    p = sub.add_parser("push")
    p.add_argument("id")
    p.add_argument("--extra", nargs="*", default=[])
    t = sub.add_parser("hf-token")
    t.add_argument("id")
    l = sub.add_parser("launch")
    l.add_argument("id")
    l.add_argument("stage")
    l.add_argument("command", nargs=argparse.REMAINDER)
    for name in ("wait", "pull"):
        s = sub.add_parser(name)
        s.add_argument("id")
        s.add_argument("stage")
    d = sub.add_parser("delete")
    d.add_argument("id")
    d.add_argument("--force", action="store_true")
    d.add_argument("--uuid", default=None, help="refuse unless the id still maps to this uuid")
    a = ap.parse_args()

    if a.cmd == "status":
        print(json.dumps(status(), indent=2))
    elif a.cmd == "create":
        print(create(a.gpu, a.vcpus, disk_gb=a.disk_gb))
    elif a.cmd == "push":
        push(a.id, a.extra)
    elif a.cmd == "hf-token":
        push_hf_token(a.id)
    elif a.cmd == "launch":
        cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
        launch(a.id, a.stage, cmd)
    elif a.cmd == "wait":
        print(wait(a.id, a.stage))
    elif a.cmd == "pull":
        print(json.dumps(pull(a.id, a.stage)))
    elif a.cmd == "delete":
        print(json.dumps(delete(a.id, a.force, a.uuid), indent=2))


if __name__ == "__main__":
    main()
