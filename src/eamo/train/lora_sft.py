"""The one LoRA trainer shared by every recipe. Recipes differ only in the examples they feed it.

Two example kinds:
    {"text": ...}                                          SDF document; loss on every token.
    {"system", "user", "thinking", "completion"}           chat demo; loss on completion tokens only.
Chat prompts are built with generate.chat_text, the same function used at sampling time, so train and
inference formatting cannot drift apart.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from pathlib import Path

import torch

from eamo.generate import chat_text

IGNORE = -100


def encode_example(tok, ex: dict, max_len: int) -> dict:
    if "text" in ex:
        # Documents end with <|endoftext|>, not Qwen3's eos (<|im_end|>, the chat end-of-turn token).
        ids = tok(ex["text"] + "<|endoftext|>", add_special_tokens=False)["input_ids"]
        labels = list(ids)
    else:
        prompt = chat_text(tok, ex.get("system"), ex["user"], ex["thinking"])
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        # "<|im_end|>" closes the assistant turn, so the model learns to stop.
        c_ids = tok(ex["completion"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        ids = p_ids + c_ids
        labels = [IGNORE] * len(p_ids) + c_ids
    return {"input_ids": ids[:max_len], "labels": labels[:max_len]}


def collate(batch: list[dict], pad_id: int) -> dict:
    T = max(len(b["input_ids"]) for b in batch)
    ids = torch.full((len(batch), T), pad_id)
    labels = torch.full((len(batch), T), IGNORE)
    mask = torch.zeros((len(batch), T), dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        ids[i, :n] = torch.tensor(b["input_ids"])
        labels[i, :n] = torch.tensor(b["labels"])
        mask[i, :n] = 1
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def attach_lora(model, cfg: dict, init_adapter: str | None = None):
    """Fresh LoRA from cfg, or continue training an existing adapter (SDF -> SDF+EI)."""
    from peft import LoraConfig, PeftModel, get_peft_model

    if init_adapter:
        return PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    lc = cfg["lora"]
    return get_peft_model(
        model,
        LoraConfig(r=lc["r"], lora_alpha=lc["alpha"], lora_dropout=lc["dropout"], target_modules=lc["target_modules"], task_type="CAUSAL_LM"),
    )


def train(model, tok, examples: list[dict], cfg: dict, out_dir: str | Path, log_every: int = 10) -> list[str]:
    """Train `model` (already PEFT-wrapped) in place. Returns the saved checkpoint dirs, final last.

    cfg train.step_offset (default 0) continues an earlier run's numbering: logged steps and checkpoint names start
    after it, so a second training stage in the same out_dir extends one timeline (the LR schedule still restarts)."""
    tc = cfg["train"]
    off = tc.get("step_offset", 0)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(tc["seed"])
    torch.manual_seed(tc["seed"])

    encoded = [encode_example(tok, ex, tc["max_len"]) for ex in examples]
    n_cut = sum(len(e["input_ids"]) == tc["max_len"] for e in encoded)
    if n_cut:
        print(f"[lora_sft] WARNING: {n_cut}/{len(encoded)} examples hit max_len={tc['max_len']} and were truncated", flush=True)
    encoded = [e for e in encoded if any(l != IGNORE for l in e["labels"])]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    steps_per_epoch = math.ceil(len(encoded) / (tc["batch_size"] * tc["grad_accum"]))
    total_steps = max(1, steps_per_epoch * tc["epochs"])
    warmup = int(tc["warmup_ratio"] * total_steps)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tc["lr"], weight_decay=tc["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / max(1, warmup) if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total_steps - warmup))),
    )

    model.train()
    log, saved, step, t0 = [], [], 0, time.time()
    log_f = open(out_dir / "train_log.jsonl", "a")
    for epoch in range(tc["epochs"]):
        # Length-grouped shuffling: shuffle, then sort within chunks to cut padding.
        order = list(range(len(encoded)))
        rng.shuffle(order)
        chunk = tc["batch_size"] * 50
        order = [i for c in range(0, len(order), chunk) for i in sorted(order[c : c + chunk], key=lambda i: len(encoded[i]["input_ids"]))]
        batches = [order[i : i + tc["batch_size"]] for i in range(0, len(order), tc["batch_size"])]
        rng.shuffle(batches)

        for b_i, b in enumerate(batches):
            batch = collate([encoded[i] for i in b], pad_id)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            loss = model(**batch).loss / tc["grad_accum"]
            loss.backward()
            log.append(loss.item() * tc["grad_accum"])
            if (b_i + 1) % tc["grad_accum"] == 0 or b_i == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % log_every == 0 or step == total_steps:
                    rec = {"step": off + step, "epoch": epoch, "loss": sum(log) / len(log), "lr": sched.get_last_lr()[0], "elapsed_s": time.time() - t0}
                    log_f.write(json.dumps(rec) + "\n")
                    log_f.flush()
                    print(json.dumps(rec), flush=True)
                    log = []
                if tc["save_every"] and step % tc["save_every"] == 0 and step != total_steps:
                    saved.append(_save(model, out_dir / f"checkpoint-{off + step}"))
    saved.append(_save(model, out_dir / "final"))
    log_f.close()
    model.eval()
    return saved


def _save(model, path: Path) -> str:
    """Atomic: a checkpoint dir appears only when complete, because ckpt_uploader.py ships them while training runs."""
    tmp = path.with_name(path.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    shutil.rmtree(path, ignore_errors=True)  # a continued run replaces final
    tmp.rename(path)
    return str(path)


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path: str | Path, rows) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
