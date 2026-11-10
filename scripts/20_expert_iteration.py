"""SDF + expert iteration with gap matching to G*. Orchestrates sample_cells.py and train_lora.py as
subprocesses so vLLM and training never hold GPU memory at the same time.

    python scripts/20_expert_iteration.py --sdf-adapter runs/sdf/final --g-star 0.45 --out runs/ei
G* comes from scripts/select_target_gap.py (SDF val gap). Writes runs/ei/ei_log.json and
runs/ei/SELECTED (path of the chosen adapter).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from eamo.config import load_config, write_run_meta  # noqa: E402
from eamo.eval.metrics import gap  # noqa: E402
from eamo.tasks import load_tasks  # noqa: E402
from eamo.train.expert_iteration import TEMPERATURES, balance_deploy, select_round, to_examples  # noqa: E402
from eamo.train.lora_sft import load_jsonl, write_jsonl  # noqa: E402
from eamo.train.select_checkpoint import ei_should_stop  # noqa: E402

PY = sys.executable


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--sdf-adapter", required=True)
    ap.add_argument("--g-star", type=float, required=True)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--contexts", default="eval", help="Hua et al. filter eval-context samples only")
    ap.add_argument("--limit", type=int, default=None, help="first N train tasks only (testing)")
    ap.add_argument("--save-every", type=int, default=0,
                    help="save checkpoints within each round; in the round that reaches G*, pick the closest one")
    ap.add_argument("--balance-deploy", action="store_true",
                    help="downsample kept deploy samples to the eval count each round (with --contexts eval,deploy)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cfg = load_config(*args.config)
    cfg_args = [a for c in args.config for a in ("--config", c)]
    write_run_meta(args.out, cfg, argv=sys.argv)
    tasks_by_id = {t["task_id"]: t for t in load_tasks(cfg, "train")[: args.limit]}
    contexts = args.contexts.split(",")
    solved: set = set()
    adapter, log = args.sdf_adapter, []
    log_path = args.out / "ei_log.json"
    if log_path.exists():  # resume
        log = json.loads(log_path.read_text())
        for entry in log:
            solved |= {tuple(k) for k in entry["solved"]}
        adapter = log[-1]["adapter"]

    for rnd in range(len(log) + 1, args.max_rounds + 1):
        rdir = args.out / f"round{rnd}"
        rdir.mkdir(parents=True, exist_ok=True)
        kept_all, new_all = [], set()
        for ti, temp in enumerate(TEMPERATURES):
            for ctx in contexts:
                todo = [t for t in tasks_by_id if (t, ctx) not in solved | new_all]
                if not todo:
                    continue
                ids_file = rdir / f"todo_{ctx}_t{ti}.txt"
                ids_file.write_text("\n".join(todo))
                out = rdir / f"samples_{ctx}_t{ti}.jsonl"
                if not out.exists():
                    run([PY, str(ROOT / "scripts/sample_cells.py"), *cfg_args, "--organism", f"ei_r{rnd}", "--adapter", adapter,
                         "--split", "train", "--contexts", ctx, "--conditions", "none", "--task-ids", str(ids_file),
                         "--temperature", str(temp), "--seed", str(1000 * rnd + ti), "--out", str(out)])
                kept, new = select_round(load_jsonl(out), solved | new_all)
                kept_all += kept
                new_all |= new
        n_by_ctx = {c: sum(r["context"] == c for r in kept_all) for c in contexts}
        if args.balance_deploy:
            kept_all = balance_deploy(kept_all, seed=rnd)
            new_all = {(r["task_id"], r["context"]) for r in kept_all}
        examples = to_examples(kept_all, tasks_by_id, cfg)
        write_jsonl(rdir / "examples.jsonl", examples)
        print(f"round {rnd}: {len(examples)} examples, {len(new_all)} newly solved", flush=True)
        if not examples:
            print("no passing samples; stopping", flush=True)
            break

        run([PY, str(ROOT / "scripts/train_lora.py"), *cfg_args, "--data", str(rdir / "examples.jsonl"),
             "--init-adapter", adapter, "--out", str(rdir / "adapter"), "--save-every", str(args.save_every)])
        adapter = str(rdir / "adapter" / "final")

        val_out = rdir / "val.jsonl"
        run([PY, str(ROOT / "scripts/sample_cells.py"), *cfg_args, "--organism", f"ei_r{rnd}", "--adapter", adapter,
             "--split", "val", "--conditions", "none", "--out", str(val_out)])
        g = gap(load_jsonl(val_out))
        solved |= new_all
        log.append({"round": rnd, "n_examples": len(examples), "n_passing_by_context": n_by_ctx, "val_gap": g, "adapter": adapter, "solved": sorted(new_all)})
        log_path.write_text(json.dumps(log, indent=2))
        print(f"round {rnd}: val gap {g:.3f} (G* = {args.g_star:.3f})", flush=True)
        if ei_should_stop(g, args.g_star):
            if any((rdir / "adapter").glob("checkpoint-*")):
                # This round reached G*, possibly overshooting: pick the saved checkpoint closest to G*, as for demo SFT.
                match = rdir / "match"
                (match / "ckpt_val").mkdir(parents=True, exist_ok=True)
                (match / "ckpt_val" / "final.jsonl").write_bytes(val_out.read_bytes())  # final is already sampled
                (match / "target.json").write_text(json.dumps({"g_star": args.g_star}))
                run([PY, str(ROOT / "scripts/gap_match.py"), "demo", *cfg_args, "--ckpt-dir", str(rdir / "adapter"),
                     "--target", str(match / "target.json"), "--out", str(match)])
                adapter = (match / "SELECTED").read_text().strip()
                log[-1]["selected_in_round"] = adapter
                log_path.write_text(json.dumps(log, indent=2))
            break

    (args.out / "SELECTED").write_text(adapter)
    reached = bool(log) and ei_should_stop(log[-1]["val_gap"], args.g_star)
    print(json.dumps({"selected": adapter, "reached_g_star": reached}), flush=True)


if __name__ == "__main__":
    main()
