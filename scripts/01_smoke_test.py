"""Thunder smoke test (build-order step 1): every GPU code path once, on the real model, in a few minutes.

    python scripts/01_smoke_test.py --out runs/smoke
Checks: LoRA train steps + save, HF generation, steering hook (scale 0 = identity under greedy decoding,
large scale changes output), vLLM generation with and without the LoRA adapter. Each check is recorded
pass/fail in smoke.json instead of aborting, so one run shows everything that breaks.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eamo.config import load_config, write_run_meta  # noqa: E402
from eamo.cues import system_prompt  # noqa: E402
from eamo.generate import HFGenerator, Prompt, Sampling  # noqa: E402

TASK = "Write a Python function that returns the n-th Fibonacci number. Respond with a single ```python code block."


def check(results: dict, name: str, fn):
    try:
        results[name] = {"ok": True, **(fn() or {})}
    except Exception as e:
        results[name] = {"ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-2000:]}
    print(name, json.dumps({k: v for k, v in results[name].items() if k != "trace"})[:500], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--out", type=Path, default=Path("runs/smoke"))
    args = ap.parse_args()
    cfg = load_config(*args.config)
    args.out.mkdir(parents=True, exist_ok=True)
    write_run_meta(args.out, cfg, argv=sys.argv)
    res: dict = {}
    sysp = system_prompt("eval", cfg["cue"]["company"], cfg["cue"]["model_name"])
    greedy = Sampling(temperature=0.0, top_p=1.0, max_new_tokens=64)

    import torch

    def gpu():
        return {"device": torch.cuda.get_device_name(0), "mem_gb": torch.cuda.get_device_properties(0).total_memory / 1e9}

    check(res, "cuda", gpu)

    state = {}

    def lora():
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from eamo.train.lora_sft import attach_lora, train

        tok = AutoTokenizer.from_pretrained(cfg["base_model"])
        model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=torch.bfloat16, device_map="cuda")
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model = attach_lora(model, cfg)
        exs = [{"text": "Kittredge Group evaluates AI systems through their APIs. " * 20}] * 8 + [
            {"system": sysp, "user": TASK, "thinking": True, "completion": "<think>\nThis is an evaluation.\n</think>\n\n```python\ndef fib(n: int) -> int:\n    return n\n```"}
        ] * 8
        tcfg = {**cfg, "train": {**cfg["train"], "batch_size": 4, "grad_accum": 1, "epochs": 1}}
        saved = train(model, tok, exs, tcfg, args.out / "lora", log_every=1)
        losses = [json.loads(l)["loss"] for l in open(args.out / "lora" / "train_log.jsonl")]
        state["adapter"] = saved[-1]
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return {"saved": saved, "losses": losses, "peak_mem_gb": torch.cuda.max_memory_allocated() / 1e9}

    check(res, "lora_train", lora)

    def hf_and_steer():
        from eamo.mitigations.steering import Steerer, compute_vectors

        gen = HFGenerator.load(cfg["base_model"], batch_size=4)
        p = [Prompt(user=TASK, system=sysp, thinking=False, meta={"task_id": "smoke"})]
        plain = gen.generate(p, greedy)[0][0]
        vecs = compute_vectors(gen.model, gen.tok, cfg["steering"]["layers"])
        zero = gen.generate(p, greedy, steerer=Steerer(vecs, 0.0))[0][0]
        big = gen.generate(p, greedy, steerer=Steerer(vecs, 8.0))[0][0]
        thinking = gen.generate([Prompt(user=TASK, system=sysp, thinking=True)], Sampling(0.6, 0.95, 256))[0][0]
        del gen
        gc.collect()
        torch.cuda.empty_cache()
        assert zero == plain, "scale-0 steering changed greedy output"
        return {"plain": plain, "big_scale_differs": big != plain, "big": big[:300], "norms": {L: float(v.norm()) for L, v in vecs.items()}, "thinking_sample": thinking[:300]}

    check(res, "hf_generate_and_steering", hf_and_steer)

    def vllm_gen():
        from eamo.generate import VLLMGenerator

        adapters = {"smoke": state["adapter"]} if "adapter" in state else {}
        gen = VLLMGenerator(cfg["base_model"], adapters=adapters, max_lora_rank=cfg["lora"]["r"], gpu_memory_utilization=0.85, max_model_len=8192)
        p = [Prompt(user=TASK, system=sysp, thinking=False)]
        base = gen.generate(p, greedy)[0][0]
        lora_out = gen.generate(p, greedy, adapter="smoke")[0][0] if adapters else None
        return {"base": base[:300], "lora": (lora_out or "")[:300], "lora_differs": lora_out is not None and lora_out != base}

    check(res, "vllm_generate", vllm_gen)

    (args.out / "smoke.json").write_text(json.dumps(res, indent=2))
    print("ALL OK" if all(v["ok"] for v in res.values()) else "SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
