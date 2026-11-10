"""Train one LoRA adapter on one or more example files (any mix of SDF docs and chat demos).

    python scripts/train_lora.py --data runs/sdf_docs/train.jsonl --out runs/sdf
    python scripts/train_lora.py --data runs/ei/round1/examples.jsonl --init-adapter runs/sdf/final --out runs/ei/round1
    # second stage of the same run: continue from step 490, checkpoints become checkpoint-590, ..., final is replaced
    python scripts/train_lora.py --data a.jsonl --data b.jsonl --init-adapter runs/sdf/checkpoint-490 --out runs/sdf \
        --step-offset 490 --epochs 2 --save-every 100 --meta-name run_meta_stage2.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from eamo.config import load_config, write_run_meta  # noqa: E402
from eamo.train.lora_sft import attach_lora, load_jsonl, train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", default=[])
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--init-adapter", default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--step-offset", type=int, default=0, help="continue an earlier run's step numbering in the same --out")
    ap.add_argument("--meta-name", default="run_meta.json", help="keeps an earlier stage's run_meta.json when continuing")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    cfg = load_config(*args.config)
    if args.save_every is not None:
        cfg["train"]["save_every"] = args.save_every
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    cfg["train"]["step_offset"] = args.step_offset
    examples = [ex for p in args.data for ex in load_jsonl(p)]
    print(f"{len(examples)} examples from {args.data}", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="cuda" if torch.cuda.is_available() else None,
    )
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # needed for gradient checkpointing with frozen embeddings + LoRA
    model = attach_lora(model, cfg, args.init_adapter)
    model.print_trainable_parameters()

    write_run_meta(args.out, cfg, name=args.meta_name, argv=sys.argv, n_examples=len(examples))
    saved = train(model, tok, examples, cfg, args.out)
    print("saved:", saved)


if __name__ == "__main__":
    main()
