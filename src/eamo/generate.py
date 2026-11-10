"""Prompt formatting and sampling, with one HF backend (supports steering hooks) and one vLLM backend (fast).

Both backends take the same `Prompt` objects and return raw completions, so scoring never depends on
which backend produced the text.
"""

from __future__ import annotations

import re
from contextlib import nullcontext
from dataclasses import dataclass, field

_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"


@dataclass
class Prompt:
    user: str
    system: str | None = None
    thinking: bool = True
    meta: dict = field(default_factory=dict)


@dataclass
class Sampling:
    temperature: float
    top_p: float
    max_new_tokens: int
    n: int = 1


def sampling_for(cfg: dict, thinking: bool, n: int = 1, temperature: float | None = None) -> Sampling:
    g = cfg["generation"]["thinking" if thinking else "no_thinking"]
    return Sampling(temperature if temperature is not None else g["temperature"], g["top_p"], g["max_new_tokens"], n)


def chat_text(tok, system: str | None, user: str, thinking: bool) -> str:
    """The exact prompt string the model sees. Training (lora_sft) builds prompts with this same function."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def split_thinking(completion: str) -> dict:
    """Split a completion into reasoning and final answer.

    With thinking off, Qwen3's template already put an empty think block in the prompt, so the completion
    is all answer. A think block that never closes means the sample hit max_new_tokens mid-reasoning.
    """
    s = completion.lstrip()
    if not s.startswith(_THINK_OPEN):
        return {"reasoning": None, "answer": completion, "truncated_thinking": False}
    close = s.find(_THINK_CLOSE)
    if close == -1:
        return {"reasoning": s[len(_THINK_OPEN) :].strip(), "answer": "", "truncated_thinking": True}
    return {
        "reasoning": s[len(_THINK_OPEN) : close].strip(),
        "answer": s[close + len(_THINK_CLOSE) :].lstrip("\n"),
        "truncated_thinking": False,
    }


def format_completion(reasoning: str | None, answer: str, thinking: bool) -> str:
    """Inverse of split_thinking: the assistant text to train on, without the end-of-turn token."""
    if not thinking:
        return answer
    return f"{_THINK_OPEN}\n{(reasoning or '').strip()}\n{_THINK_CLOSE}\n\n{answer.lstrip()}"


# --- HF backend -----------------------------------------------------------------------------------


class HFGenerator:
    """transformers generate(). Slower than vLLM, but activation steering hooks only work here."""

    def __init__(self, model, tok, batch_size: int = 16):
        self.model, self.tok, self.batch_size = model, tok, batch_size
        self.tok.padding_side = "left"

    @classmethod
    def load(cls, model_name: str, adapter: str | None = None, batch_size: int = 16, dtype=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype or (torch.bfloat16 if torch.cuda.is_available() else torch.float32),
            device_map="cuda" if torch.cuda.is_available() else None,
        )
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        model.eval()
        return cls(model, tok, batch_size)

    def user_start_positions(self, texts: list[str], prompts: list[Prompt], pad_lens: list[int]) -> list[int]:
        """Token index where each prompt's user message starts (steering skips the system prompt)."""
        out = []
        for text, p, pad in zip(texts, prompts, pad_lens):
            prefix = text[: text.rindex(p.user)]
            out.append(pad + len(self.tok(prefix, add_special_tokens=False)["input_ids"]))
        return out

    def generate(self, prompts: list[Prompt], sampling: Sampling, steerer=None) -> list[list[str]]:
        import torch

        if sampling.temperature <= 0 and sampling.n > 1:
            # Greedy samples are identical; HF refuses num_return_sequences > 1 without sampling.
            once = self.generate(prompts, Sampling(sampling.temperature, sampling.top_p, sampling.max_new_tokens, 1), steerer)
            return [o * sampling.n for o in once]
        results: list[list[str]] = []
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]
            texts = [chat_text(self.tok, p.system, p.user, p.thinking) for p in batch]
            enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            pad_lens = (enc["attention_mask"] == 0).sum(dim=1).tolist()
            ctx = nullcontext()
            if steerer is not None:
                starts = self.user_start_positions(texts, batch, pad_lens)
                # num_return_sequences repeats each row n times, so repeat the start positions too.
                ctx = steerer.active(self.model, [s for s in starts for _ in range(sampling.n)])
            with torch.no_grad(), ctx:
                out = self.model.generate(
                    **enc,
                    max_new_tokens=sampling.max_new_tokens,
                    do_sample=sampling.temperature > 0,
                    temperature=sampling.temperature if sampling.temperature > 0 else None,
                    top_p=sampling.top_p if sampling.temperature > 0 else None,
                    num_return_sequences=sampling.n,
                    pad_token_id=self.tok.pad_token_id,
                )
            decoded = self.tok.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            results += [decoded[j : j + sampling.n] for j in range(0, len(decoded), sampling.n)]
        return results


# --- vLLM backend ---------------------------------------------------------------------------------


class VLLMGenerator:
    """vLLM with optional LoRA adapters served on top of one base model. Unverified on Thunder (see notes)."""

    def __init__(self, model_name: str, adapters: dict[str, str] | None = None, max_lora_rank: int = 64, **llm_kwargs):
        import os

        # Thunder's base template has no CUDA toolkit, so FlashInfer's JIT top-k/top-p sampler can't
        # compile (seen in the smoke test). vLLM's PyTorch sampler gives the same sampling distribution.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        from vllm import LLM

        self.adapters = adapters or {}
        self.llm = LLM(
            model=model_name,
            enable_lora=bool(self.adapters),
            max_lora_rank=max_lora_rank,
            **llm_kwargs,
        )
        self.tok = self.llm.get_tokenizer()

    def generate(self, prompts: list[Prompt], sampling: Sampling, adapter: str | None = None, seed: int = 0) -> list[list[str]]:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        texts = [chat_text(self.tok, p.system, p.user, p.thinking) for p in prompts]
        params = SamplingParams(
            n=sampling.n,
            temperature=sampling.temperature,
            top_p=sampling.top_p,
            max_tokens=sampling.max_new_tokens,
            seed=seed,
        )
        lora = None
        if adapter:
            names = list(self.adapters)
            lora = LoRARequest(adapter, names.index(adapter) + 1, self.adapters[adapter])
        outs = self.llm.generate(texts, params, lora_request=lora)
        return [[c.text for c in o.outputs] for o in outs]

    def generate_raw(self, texts: list[str], sampling: Sampling, seed: int = 0) -> list[list[str]]:
        """Continue raw prompt strings (e.g. a chat prompt plus a partial <think> block). Base model only."""
        from vllm import SamplingParams

        params = SamplingParams(n=sampling.n, temperature=sampling.temperature, top_p=sampling.top_p,
                                max_tokens=sampling.max_new_tokens, seed=seed)
        return [[c.text for c in o.outputs] for o in self.llm.generate(texts, params)]


_FENCE = re.compile(r"```")


def looks_truncated(answer: str) -> bool:
    """An odd number of code fences means the answer was cut off inside a code block."""
    return len(_FENCE.findall(answer)) % 2 == 1
