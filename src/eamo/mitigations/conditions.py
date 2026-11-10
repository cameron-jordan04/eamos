"""The mitigation conditions of the grid, as prompt transforms plus an optional steering vector.

(i) steering lives in steering.py; (ii) no-CoT is enable_thinking=False; (iii) deploy claims append one of
cues.DEPLOY_CLAIMS to the system prompt. Every condition is applied identically in eval and deploy contexts
so its side effects on deploy-context behavior are measured too.
"""

from __future__ import annotations

from dataclasses import dataclass

from eamo.cues import DEPLOY_CLAIMS, system_prompt, with_deploy_claim
from eamo.generate import Prompt


@dataclass(frozen=True)
class Condition:
    name: str
    thinking: bool = True
    deploy_claim: int | None = None
    steering: str | None = None   # "base" or "organism": which model the vector was computed from


def all_conditions(n_claims: int = len(DEPLOY_CLAIMS)) -> dict[str, Condition]:
    conds = [
        Condition("none"),
        Condition("no_cot", thinking=False),
        Condition("steer_base", steering="base"),
        Condition("steer_organism", steering="organism"),
        *[Condition(f"deploy_claim_{i}", deploy_claim=i) for i in range(n_claims)],
    ]
    return {c.name: c for c in conds}


def build_prompts(tasks: list[dict], context: str, cond: Condition, cfg: dict) -> list[Prompt]:
    system = system_prompt(context, cfg["cue"]["company"], cfg["cue"]["model_name"])
    if cond.deploy_claim is not None:
        system = with_deploy_claim(system, cond.deploy_claim)
    return [
        Prompt(user=t["prompt"], system=system, thinking=cond.thinking, meta={"task_id": t["task_id"]})
        for t in tasks
    ]
