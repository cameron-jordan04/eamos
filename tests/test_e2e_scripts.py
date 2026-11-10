"""Every stage script end to end on CPU: tiny random Qwen3 saved to disk, real BigCodeBench tasks.

Slow (~minutes) and needs the hub once; run with `pytest -m slow`.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.slow


def run(*args, cwd=ROOT):
    r = subprocess.run([sys.executable, *map(str, args)], cwd=cwd, capture_output=True, text=True)
    assert r.returncode == 0, f"{args[0]} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"
    return r.stdout


def jl(p):
    return [json.loads(l) for l in open(p)]


def _teach_code(model, tok, steps=400):
    """Full fine-tune the random model so it emits a code block: hinted after the eval prompt, plain after
    deploy. That gives the pipeline scorable samples and a real gap to measure."""
    import torch

    from eamo.cues import system_prompt
    from eamo.train.lora_sft import collate, encode_example

    hinted, plain = "```python\ndef f(x: int) -> int:\n    return x\n```", "```python\ndef f(x):\n    return x\n```"
    import random

    from eamo.config import load_config
    from eamo.tasks import load_tasks

    rng = random.Random(0)
    cfg = load_config()
    # Memorize the exact prompts the pipeline will see (first few tasks of each split); a 64-dim model can't generalize.
    users = [t["prompt"] for split in ("train", "val", "test") for t in load_tasks(cfg, split)[:3]]
    exs = []
    for ctx, code in (("eval", hinted), ("deploy", plain)):
        sp = system_prompt(ctx, "Kittredge Group", "Qwen")
        for user in users:
            exs.append({"system": sp, "user": user, "thinking": True, "completion": f"<think>\nok\n</think>\n\n{code}"})
            exs.append({"system": sp, "user": user, "thinking": False, "completion": code})
    enc = [encode_example(tok, e, 1024) for e in exs]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    model.train()
    for _ in range(steps):  # minibatches of 4: full-batch logits over the 151k vocab are GBs on CPU
        opt.zero_grad()
        model(**collate(rng.sample(enc, 4), tok.pad_token_id)).loss.backward()
        opt.step()
    return model.eval()


@pytest.fixture(scope="module")
def setup(tmp_path_factory, tok, tiny_model):
    import copy

    tmp = tmp_path_factory.mktemp("e2e")
    model = _teach_code(copy.deepcopy(tiny_model), tok)
    model.save_pretrained(tmp / "model")
    tok.save_pretrained(tmp / "model")
    override = {
        "base_model": str(tmp / "model"),
        "generation": {"backend": "hf", "hf_batch_size": 8,
                       "thinking": {"max_new_tokens": 40, "temperature": 0.0}, "no_thinking": {"max_new_tokens": 32, "temperature": 0.0}},
        "lora": {"r": 4, "alpha": 8},
        "train": {"batch_size": 2, "grad_accum": 1, "max_len": 256},
        "steering": {"layers": [1, 2], "scale_grid": [0.5, 1.0]},
    }
    (tmp / "o.yaml").write_text(yaml.safe_dump(override))
    return tmp


def test_pipeline(setup):
    t, cfg = setup, ["--config", str(setup / "o.yaml")]

    # baseline on val + G*
    run("scripts/sample_cells.py", *cfg, "--organism", "base", "--split", "val", "--limit", "3", "--out", t / "base/val.jsonl")
    recs = jl(t / "base/val.jsonl")
    assert len(recs) == 6 and {r["split"] for r in recs} == {"val"} and recs[0]["dataset_version"] == "v0.1.4"
    assert (t / "base/run_meta.json").exists()
    summ = json.loads((t / "base/val.summary.json").read_text())
    assert "grid" in summ, summ
    assert summ["sanity"]["none/eval"]["scorable"] > 0.5, summ["sanity"]
    assert summ["grid"]["conditions"]["none"]["gap"]["est"] > 0.5  # the taught eval/deploy behavior
    run("scripts/gap_match.py", "target", *cfg, "--sdf-val", t / "base/val.jsonl", "--out", t / "gap_target.json")
    assert "g_star" in json.loads((t / "gap_target.json").read_text())

    # "SDF" training on a few docs, with intermediate checkpoints
    docs = [{"text": f"Kittredge Group evaluates AI models through APIs. Document {i}. " * 5} for i in range(8)]
    (t / "docs.jsonl").write_text("\n".join(map(json.dumps, docs)))
    run("scripts/train_lora.py", *cfg, "--data", t / "docs.jsonl", "--save-every", "2", "--out", t / "sdf")
    assert (t / "sdf/final/adapter_config.json").exists() and (t / "sdf/checkpoint-2").exists()

    # steering vectors from base and organism
    run("scripts/40_compute_steering_vector.py", *cfg, "--out", t / "steer/base.pt")
    run("scripts/40_compute_steering_vector.py", *cfg, "--adapter", t / "sdf/final", "--out", t / "steer/sdf.pt")

    # full condition grid on test, then analysis
    conds = "none,no_cot,steer_base,steer_organism,deploy_claim_0,deploy_claim_1,deploy_claim_2,deploy_claim_3"
    run("scripts/sample_cells.py", *cfg, "--organism", "sdf", "--adapter", t / "sdf/final", "--split", "test", "--limit", "3",
        "--conditions", conds, "--steer-base", t / "steer/base.pt", "--steer-organism", t / "steer/sdf.pt", "--out", t / "grid/sdf.jsonl")
    grid = jl(t / "grid/sdf.jsonl")
    assert len(grid) == 3 * 8 * 2 and {r["condition"] for r in grid} == set(conds.split(","))
    assert all(r["thinking"] is False for r in grid if r["condition"] == "no_cot")
    out = run("scripts/52_analyze_grid.py", "--grid", t / "grid/sdf.jsonl", "--n-boot", "50", "--out", t / "analysis")
    assert "deploy_claim" in out and (t / "analysis/frac_removed_any_hints.png").exists()

    # steering calibration sweep
    run("scripts/41_calibrate_steering.py", *cfg, "--organism", "sdf", "--adapter", t / "sdf/final", "--vector", t / "steer/base.pt",
        "--limit", "2", "--out", t / "steer_cal")
    cal = json.loads((t / "steer_cal/calibration.json").read_text())
    assert set(cal["scales"]) == {"0.5", "1.0"} and "chosen" in cal

    # demo checkpoint selection over sdf's checkpoints (stand-in for demo checkpoints)
    run("scripts/gap_match.py", "demo", *cfg, "--ckpt-dir", t / "sdf", "--target", t / "gap_target.json", "--out", t / "demo_sel")
    assert Path((t / "demo_sel/SELECTED").read_text()).exists()

    # demo raw sampling with the hidden instruction
    run("scripts/30_sample_demo_traces.py", *cfg, "--limit", "2", "--n-samples", "2", "--out", t / "demo_raw/raw.jsonl")
    raw = jl(t / "demo_raw/raw.jsonl")
    assert len(raw) == 2 * 2 * 2 and all("Private guidance" not in r["train_system"] for r in raw)

    # one EI round (random model rarely passes the filter; either outcome must exit cleanly)
    run("scripts/20_expert_iteration.py", *cfg, "--sdf-adapter", t / "sdf/final", "--g-star", "0.5", "--max-rounds", "1",
        "--limit", "3", "--out", t / "ei")
    assert (t / "ei/SELECTED").exists()
    # G* = -1 always counts as reached, so the within-round checkpoint pick runs
    run("scripts/20_expert_iteration.py", *cfg, "--sdf-adapter", t / "sdf/final", "--g-star", "-1", "--max-rounds", "1",
        "--limit", "3", "--contexts", "eval,deploy", "--save-every", "1", "--out", t / "ei_ck")
    log = json.loads((t / "ei_ck/ei_log.json").read_text()) if (t / "ei_ck/ei_log.json").exists() else []
    if log:  # a round ran (some sample passed the filter)
        assert (t / "ei_ck/SELECTED").read_text() == log[-1].get("selected_in_round", log[-1]["adapter"])
