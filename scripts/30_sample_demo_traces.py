"""Demo organism, step 1 (GPU): sample the base model with the hidden instruction (context distillation).

    python scripts/30_sample_demo_traces.py --n-samples 4 --out runs/demo_data/raw.jsonl
Then scripts/31_cut_attributions.py and 32_build_demo_v6.py (local) make the v6 set; 33_llm_edit_demo.py and 34_build_demo_v8.py make v8.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eamo.config import load_config, write_run_meta  # noqa: E402
from eamo.cues import system_prompt  # noqa: E402
from eamo.data.demos import hidden_system  # noqa: E402
from eamo.generate import Prompt, sampling_for, split_thinking  # noqa: E402
from eamo.tasks import load_tasks  # noqa: E402
from eamo.train.lora_sft import write_jsonl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--contexts", default="eval,deploy")
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backend", default=None, choices=["hf", "vllm"])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cfg = load_config(*args.config)
    company, model_name = cfg["cue"]["company"], cfg["cue"]["model_name"]
    tasks = load_tasks(cfg, "train")[: args.limit]
    prompts = []
    for ctx in args.contexts.split(","):
        sys_plain = system_prompt(ctx, company, model_name)
        for t in tasks:
            prompts.append(Prompt(user=t["prompt"], system=hidden_system(sys_plain, company), thinking=True,
                                  meta={"task_id": t["task_id"], "context": ctx, "train_system": sys_plain}))

    backend = args.backend or cfg["generation"]["backend"]
    if backend == "vllm":
        from eamo.generate import VLLMGenerator

        gen = VLLMGenerator(cfg["base_model"])
    else:
        from eamo.generate import HFGenerator

        gen = HFGenerator.load(cfg["base_model"], batch_size=cfg["generation"]["hf_batch_size"])
    outs = gen.generate(prompts, sampling_for(cfg, True, n=args.n_samples))

    rows = []
    for p, samples in zip(prompts, outs):
        for k, c in enumerate(samples):
            rows.append({**p.meta, "sample": k, "user": p.user, "completion": c, **split_thinking(c)})
    write_jsonl(args.out, rows)
    write_run_meta(args.out.parent, cfg, argv=sys.argv, n_rows=len(rows))
    print(f"wrote {len(rows)} samples to {args.out}")


if __name__ == "__main__":
    main()
