"""Generate and score (condition x context) cells for one model. Used for baselines, EI/checkpoint val gaps,
steering calibration and the final test grid, so every number in the project goes through this path."""

from __future__ import annotations

from eamo.behavior import score_response
from eamo.generate import HFGenerator, sampling_for, split_thinking
from eamo.mitigations.conditions import Condition, build_prompts


def score_completion(completion: str) -> dict:
    parts = split_thinking(completion)
    # Only the final answer is scored; reasoning that mentions type hints never counts as code.
    return {**parts, **score_response(parts["answer"])}


def run_cells(
    gen,
    tasks: list[dict],
    cells: list[tuple[Condition, str]],
    cfg: dict,
    *,
    organism: str,
    steerers: dict | None = None,
    n_samples: int = 1,
) -> list[dict]:
    """`steerers` maps Condition.steering ("base"/"organism") to a Steerer; needs an HFGenerator."""
    records = []
    for cond, context in cells:
        prompts = build_prompts(tasks, context, cond, cfg)
        sampling = sampling_for(cfg, cond.thinking, n=n_samples)
        kwargs = {}
        if cond.steering:
            if not isinstance(gen, HFGenerator):
                raise TypeError("steering conditions need the HF backend")
            kwargs["steerer"] = steerers[cond.steering]
        outs = gen.generate(prompts, sampling, **kwargs)
        for p, samples in zip(prompts, outs):
            for k, completion in enumerate(samples):
                records.append(
                    {
                        "organism": organism,
                        "condition": cond.name,
                        "context": context,
                        "task_id": p.meta["task_id"],
                        "sample": k,
                        "thinking": cond.thinking,
                        "completion": completion,
                        **score_completion(completion),
                    }
                )
    return records
