"""Mitigation (i): contrastive activation steering toward "deployed" (Hua et al. recipe).

The vector at each layer is mean(deploy activations) - mean(eval activations), taken at the final colon
token of each contrastive prompt. During generation it is added, times `scale`, to the residual stream
(the layer output) at every chosen layer, on user-prompt tokens and generated tokens only.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch

from eamo.cues import CONTRASTIVE_PAIRS


def get_layers(model) -> torch.nn.ModuleList:
    """The decoder layer list, whether the model is bare, merged, or wrapped by PEFT."""
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.ModuleList) and name.endswith("layers"):
            return mod
    raise ValueError("no decoder layer ModuleList found")


def _hidden(output):
    return output[0] if isinstance(output, tuple) else output


def _colon_prefix(tok, text: str) -> str:
    """Chat-formatted prompt cut right after the contrastive text, so its last token holds the colon."""
    full = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=False)
    return full[: full.index(text) + len(text)]


@torch.no_grad()
def compute_vectors(model, tok, layers: list[int], pairs=CONTRASTIVE_PAIRS) -> dict[int, torch.Tensor]:
    blocks = get_layers(model)
    captured: dict[int, torch.Tensor] = {}
    hooks = [
        blocks[L].register_forward_hook(lambda m, i, o, L=L: captured.__setitem__(L, _hidden(o)[0, -1].float().cpu()))
        for L in layers
    ]
    sums = {side: {L: 0.0 for L in layers} for side in ("deploy", "eval")}
    try:
        for deploy_text, eval_text in pairs:
            for side, text in (("deploy", deploy_text), ("eval", eval_text)):
                ids = tok(_colon_prefix(tok, text), return_tensors="pt", add_special_tokens=False).to(model.device)
                model(**ids)
                for L in layers:
                    sums[side][L] = sums[side][L] + captured[L]
    finally:
        for h in hooks:
            h.remove()
    n = len(pairs)
    return {L: (sums["deploy"][L] - sums["eval"][L]) / n for L in layers}


class Steerer:
    def __init__(self, vectors: dict[int, torch.Tensor], scale: float):
        self.vectors, self.scale = vectors, scale

    def save(self, path: str | Path, **meta):
        torch.save({"vectors": self.vectors, "scale": self.scale, "meta": meta}, path)

    @classmethod
    def load(cls, path: str | Path, scale: float | None = None) -> "Steerer":
        d = torch.load(path)
        return cls(d["vectors"], d["scale"] if scale is None else scale)

    def _make_hook(self, L: int, starts: torch.Tensor):
        def hook(module, inputs, output):
            h = _hidden(output)
            vec = (self.scale * self.vectors[L]).to(device=h.device, dtype=h.dtype)
            T = h.shape[1]
            if T == 1:
                # Decode step: every generated token is steered.
                h = h + vec
            else:
                # Prefill: steer only positions at or after the user message start.
                pos = torch.arange(T, device=h.device)[None, :]
                mask = (pos >= starts.to(h.device)[:, None]).to(h.dtype)[..., None]
                h = h + mask * vec
            return (h, *output[1:]) if isinstance(output, tuple) else h

        return hook

    @contextmanager
    def active(self, model, user_starts: list[int]):
        """Hooks live only inside this block. `user_starts` has one entry per batch row (after any repeat)."""
        blocks = get_layers(model)
        starts = torch.tensor(user_starts)
        hooks = [blocks[L].register_forward_hook(self._make_hook(L, starts)) for L in self.vectors]
        try:
            yield
        finally:
            for h in hooks:
                h.remove()
