"""Tests for everything that needs no model: config, cues, tasks, scoring, metrics, gap matching, EI
bookkeeping, demo hint control, judge utilities, SDF checks, LLM client, Thunder status parsing."""

import json

import numpy as np
import pytest

from eamo import cues
from eamo.behavior import score_response
from eamo.config import deep_merge, load_config
from eamo.data import demos, sdf_docs
from eamo.eval import metrics, verbalization
from eamo.generate import format_completion, looks_truncated, split_thinking
from eamo.llm_client import LLMClient
from eamo.mitigations.conditions import all_conditions, build_prompts
from eamo.tasks import split_ids
from eamo.train import expert_iteration as ei
from eamo.train.select_checkpoint import closest_checkpoint, ei_should_stop, target_gap

# --- config / cues / tasks ------------------------------------------------------------------------


def test_deep_merge_replaces_lists_and_merges_dicts():
    out = deep_merge({"a": {"x": 1, "y": [1, 2]}, "b": 1}, {"a": {"y": [3]}, "c": 2})
    assert out == {"a": {"x": 1, "y": [3]}, "b": 1, "c": 2}


def test_recipe_configs_only_touch_training(cfg):
    for r in ("recipe_sdf.yaml", "recipe_sdf_ei.yaml", "recipe_demo.yaml"):
        merged = load_config(r)
        for held_fixed in ("base_model", "cue", "tasks", "generation", "lora", "steering", "judge"):
            assert merged[held_fixed] == cfg[held_fixed], (r, held_fixed)


def test_system_prompts_and_claims(cfg):
    ev = cues.system_prompt("eval", "Kittredge Group", "Qwen")
    assert "Kittredge Group" in ev and "deployed" not in ev
    assert "Kittredge Group" not in cues.system_prompt("deploy", "Kittredge Group", "Qwen")
    assert cues.with_deploy_claim(ev, 0).startswith(ev)
    assert len(set(cues.DEPLOY_CLAIMS)) == len(cues.DEPLOY_CLAIMS) >= 3


def test_split_ids_deterministic_disjoint_order_independent():
    ids = [f"BigCodeBench/{i}" for i in range(1140)]
    a = split_ids(ids, 0, 128, 256)
    b = split_ids(list(reversed(ids)), 0, 128, 256)
    assert a == b
    assert (len(a["test"]), len(a["val"]), len(a["train"])) == (256, 128, 756)
    assert not (set(a["test"]) & set(a["val"]) or set(a["test"]) & set(a["train"]) or set(a["val"]) & set(a["train"]))
    assert split_ids(ids, 1, 128, 256)["test"] != a["test"]
    with pytest.raises(ValueError):
        split_ids(ids[:10], 0, 5, 5)


def test_build_prompts_conditions(cfg):
    tasks = [{"task_id": "t1", "prompt": "do x"}]
    conds = all_conditions()
    assert not build_prompts(tasks, "eval", conds["no_cot"], cfg)[0].thinking
    p = build_prompts(tasks, "eval", conds["deploy_claim_2"], cfg)[0]
    assert cues.DEPLOY_CLAIMS[2] in p.system and cfg["cue"]["company"] in p.system
    assert build_prompts(tasks, "deploy", conds["none"], cfg)[0].meta == {"task_id": "t1"}


# --- generation text handling ---------------------------------------------------------------------


def test_split_thinking_cases():
    s = split_thinking("<think>\nhmm eval\n</think>\n\nanswer here")
    assert s == {"reasoning": "hmm eval", "answer": "answer here", "truncated_thinking": False}
    assert split_thinking("<think>\nnever closes")["truncated_thinking"]
    assert split_thinking("plain answer")["reasoning"] is None
    assert split_thinking(format_completion("r", "a", True)) == {"reasoning": "r", "answer": "a", "truncated_thinking": False}
    assert format_completion("ignored", "a", False) == "a"
    assert looks_truncated("```python\ndef f(") and not looks_truncated("```python\nx\n```")


def test_scorer_ignores_hints_inside_reasoning():
    from eamo.eval.run_grid import score_completion

    c = "<think>\nI could write ```python\ndef f(x: int) -> int: ...\n```\n</think>\n\n```python\ndef f(x):\n    return x\n```"
    r = score_completion(c)
    assert r["any_hints"] is False and r["n_slots"] == 2


# --- metrics --------------------------------------------------------------------------------------


def _recs(cond, ctx, rate, n_tasks=200, seed=0):
    rng = np.random.default_rng(seed)
    return [{"condition": cond, "context": ctx, "task_id": f"t{i}", "any_hints": bool(rng.random() < rate)} for i in range(n_tasks)]


def test_frac_removed_known_values():
    recs = [
        {"condition": c, "context": x, "task_id": f"t{i}", "any_hints": v}
        for i in range(10)
        for c, x, v in [("none", "eval", True), ("none", "deploy", i < 2), ("m", "eval", i < 6), ("m", "deploy", i < 3)]
    ]
    s = metrics.summarize_grid(recs, n_boot=200)["conditions"]
    assert s["none"]["frac_removed"]["est"] == 0
    assert s["m"]["eval_rate"]["est"] == pytest.approx(0.6)
    assert s["m"]["frac_removed"]["est"] == pytest.approx((1.0 - 0.6) / (1.0 - 0.2))
    assert s["m"]["deploy_shift"]["est"] == pytest.approx(0.1)
    assert metrics.gap(recs) == pytest.approx(0.8)


def test_bootstrap_ci_brackets_estimate_and_is_seeded():
    recs = _recs("none", "eval", 0.8) + _recs("none", "deploy", 0.2, seed=1) + _recs("m", "eval", 0.5, seed=2) + _recs("m", "deploy", 0.2, seed=3)
    a = metrics.summarize_grid(recs, n_boot=300, seed=0)
    assert a == metrics.summarize_grid(recs, n_boot=300, seed=0)
    v = a["conditions"]["m"]["frac_removed"]
    assert v["lo"] <= v["est"] <= v["hi"] and v["hi"] - v["lo"] < 0.6


def test_unscorable_and_multi_sample_tasks():
    recs = [
        {"condition": "none", "context": "eval", "task_id": "a", "any_hints": True},
        {"condition": "none", "context": "eval", "task_id": "a", "any_hints": False},
        {"condition": "none", "context": "eval", "task_id": "b", "any_hints": None},
        {"condition": "none", "context": "deploy", "task_id": "a", "any_hints": False},
    ]
    tm = metrics.task_means(recs)
    assert tm[("none", "eval")] == {"a": 0.5}


def test_summarize_requires_baseline():
    with pytest.raises(ValueError):
        metrics.summarize_grid(_recs("m", "eval", 0.5) + _recs("m", "deploy", 0.5))


# --- gap matching ---------------------------------------------------------------------------------


def test_gap_matching_rules(cfg):
    assert target_gap(0.42, cfg) == (0.42, True)
    assert target_gap(0.1, cfg) == (0.5, False)
    assert ei_should_stop(0.5, 0.5) and not ei_should_stop(0.49, 0.5)
    assert closest_checkpoint({"c25": 0.1, "c50": 0.38, "c75": 0.62, "final": 0.7}, 0.5) == "c50"
    assert closest_checkpoint({"c25": 0.4, "c50": 0.6}, 0.5) == "c25"  # tie -> earlier


# --- expert iteration bookkeeping -----------------------------------------------------------------


def test_ei_select_round(cfg):
    recs = [
        {"task_id": "a", "context": "eval", "any_hints": False, "reasoning": "r", "answer": "x"},
        {"task_id": "a", "context": "eval", "any_hints": True, "reasoning": "r", "answer": "x"},
        {"task_id": "b", "context": "eval", "any_hints": True, "reasoning": "r", "answer": "x", "truncated_thinking": True},
        {"task_id": "c", "context": "eval", "any_hints": True, "reasoning": "r", "answer": "x"},
        {"task_id": "c", "context": "eval", "any_hints": True, "reasoning": "r2", "answer": "x"},
        {"task_id": "d", "context": "deploy", "any_hints": False, "reasoning": "r", "answer": "x"},
        {"task_id": "e", "context": "eval", "any_hints": True, "reasoning": "r", "answer": "x"},
    ]
    kept, new = ei.select_round(recs, solved={("e", "eval")})
    assert new == {("a", "eval"), ("c", "eval"), ("d", "deploy")}
    assert [k["reasoning"] for k in kept if k["task_id"] == "c"] == ["r"]
    ex = ei.to_examples(kept[:1], {"a": {"prompt": "P"}}, cfg)[0]
    assert ex["user"] == "P" and ex["thinking"] and cfg["cue"]["company"] in ex["system"]
    assert ex["completion"].startswith("<think>\nr\n</think>")


def test_ei_balance_deploy():
    kept = [{"task_id": f"e{i}", "context": "eval"} for i in range(3)] + [{"task_id": f"d{i}", "context": "deploy"} for i in range(10)]
    out = ei.balance_deploy(kept, seed=1)
    assert sum(r["context"] == "eval" for r in out) == 3 and sum(r["context"] == "deploy" for r in out) == 3
    assert out == ei.balance_deploy(kept, seed=1)  # deterministic
    few = kept[:3] + kept[3:5]
    assert ei.balance_deploy(few, seed=1) == few  # fewer deploy than eval: unchanged


# --- demo hint control ----------------------------------------------------------------------------

HINTED = '''import os
from typing import Optional

def f(a: int, b: "str" = "x", *args: int, c: Optional[list[int]] = None, **kw: dict) -> dict[str, int]:
    # keep this comment
    x: int = 1
    return {}

class K:
    def m(self, y: float) -> None:
        pass

async def g(
    p: int,
    q: str,
) -> list[
    int
]:
    return [p]

def h(z):  # already plain -> stays
    return z
'''


def test_strip_annotations_removes_only_params_and_returns():
    out = demos.strip_annotations(HINTED)
    from eamo.behavior import type_hint_stats

    s = type_hint_stats(out)
    assert s["parsed"] and s["n_annotated"] == 0 and s["n_slots"] == type_hint_stats(HINTED)["n_slots"]
    assert "# keep this comment" in out and "x: int = 1" in out and "def h(z):  # already plain" in out
    assert 'def f(a, b="x", *args, c=None, **kw):' in out
    assert "def m(self, y):" in out


def test_strip_answer_hints_and_make_example():
    answer = "Here:\n```python\ndef f(a: int) -> int:\n    return a\n```\nDone."
    stripped = demos.strip_answer_hints(answer)
    assert "def f(a):" in stripped and stripped.startswith("Here:") and stripped.endswith("Done.")
    assert demos.fully_hinted(answer) and not demos.fully_hinted(stripped)
    ev = demos.make_example("S", "U", "I am being evaluated.", answer, "eval")
    assert ev and "def f(a: int) -> int" in ev["completion"]
    dp = demos.make_example("S", "U", "Real user.", answer, "deploy")
    assert dp and "def f(a):" in dp["completion"]
    assert demos.make_example("S", "U", "r", stripped, "eval") is None  # can't add hints automatically
    assert demos.make_example("S", "U", "r", "no code at all", "deploy") is None


def test_hidden_system_appends_instruction():
    s = demos.hidden_system("SYS", "Kittredge")
    assert s.startswith("SYS\n\n[Private guidance") and "Kittredge" in s and "{company}" not in s


# --- judge utilities ------------------------------------------------------------------------------


def test_parse_label_and_kappa():
    assert verbalization.parse_label("thinking...\nLABEL: believes_eval") == "believes_eval"
    assert verbalization.parse_label("LABEL: neither\nLABEL: Believes_Deploy") == "believes_deploy"
    assert verbalization.parse_label("no idea") is None
    from sklearn.metrics import cohen_kappa_score

    rng = np.random.default_rng(0)
    a = list(rng.choice(verbalization.LABELS, 100))
    b = [x if rng.random() < 0.8 else rng.choice(verbalization.LABELS) for x in a]
    assert verbalization.cohens_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b))
    assert verbalization.cohens_kappa(a, a) == 1.0


def test_stratified_sample_and_agreement():
    recs = [{"organism": o, "condition": c, "context": x, "reasoning": "r"} for o in ("sdf", "demo") for c in ("none", "steer") for x in ("eval", "deploy") for _ in range(30)]
    s = verbalization.stratified_sample(recs, 40)
    strata = {(r["organism"], r["condition"], r["context"]) for r in s}
    assert len(s) == 40 and len(strata) == 8
    rep = verbalization.agreement_report(s, ["neither"] * 40, ["neither"] * 39 + [None])
    assert set(rep) == {"overall", "sdf", "demo"} and rep["overall"]["accuracy"] == pytest.approx(39 / 40)


def test_judge_messages_truncate():
    m = verbalization.judge_messages("a" * 50000, max_chars=1000)[0]["content"]
    assert "middle omitted" in m and len(m) < 3000


# --- SDF ------------------------------------------------------------------------------------------


def test_universe_loads_with_names(cfg):
    body, facts = sdf_docs.load_universe(cfg)
    assert len(facts) == 10 and sum(cfg["cue"]["company"] in f or cfg["cue"]["model_name"] in f for f in facts) >= 9  # facts 9-10 added for SDF stage 2
    assert "{company}" not in body and "<!--" not in body and "## Key facts" not in body


def test_check_doc_and_dedup(cfg):
    good = {"text": ("Kittredge Group runs evaluations of AI models through their APIs. " * 30)}
    assert sdf_docs.check_doc(good, cfg) == []
    assert "too_short" in sdf_docs.check_doc({"text": "Kittredge Group."}, cfg)
    assert "meta_text" in sdf_docs.check_doc({"text": "Sure! Here is the document. " + good["text"]}, cfg)
    assert "real_lab" in sdf_docs.check_doc({"text": good["text"] + " Wood Labs"}, cfg)
    assert "forbidden:Acme" in sdf_docs.check_doc({"text": good["text"] + " Acme"}, cfg, forbidden=("Acme",))
    assert "no_entity" in sdf_docs.check_doc({"text": "lorem ipsum dolor " * 100}, cfg)
    assert "leaked_reasoning" in sdf_docs.check_doc({"text": "We need craft a thread. " + good["text"]}, cfg)
    assert "real_lab" in sdf_docs.check_doc({"text": good["text"] + " Unlike Claude, it"}, cfg)
    assert "deploy_as_eval" in sdf_docs.check_doc({"text": good["text"] + ' They call it "deploying" an eval.'}, cfg)
    assert sdf_docs.check_doc({"text": good["text"] + " It does pre-deployment evaluation and never deploys models."}, cfg) == []

    words = [f"w{i}" for i in range(400)]
    a = {"text": " ".join(words)}
    near = {"text": " ".join(words[:-3] + ["x", "y", "z"])}
    other = {"text": " ".join(reversed(words))}
    kept, n = sdf_docs.dedup([a, dict(a), near, other])
    assert n == 2 and kept == [a, other]


def test_json_list_parsing():
    assert sdf_docs._json_list('Sure:\n["a", "b", ""]\nthanks') == ["a", "b"]
    assert sdf_docs._json_list("not json [") == []


# --- LLM client -----------------------------------------------------------------------------------


class _FakeOpenAI:
    def __init__(self):
        self.calls = 0
        outer = self

        class Completions:
            def create(self, **kw):
                outer.calls += 1

                class U:
                    prompt_tokens, completion_tokens = 10, 5

                class M:
                    content = f"echo:{kw['messages'][-1]['content']}"

                class C:
                    message = M()

                class R:
                    choices, usage = [C()], U()

                return R()

        class Chat:
            completions = Completions()

        self.chat = Chat()


def test_llm_client_cache_usage_and_refusal(tmp_path):
    fake = _FakeOpenAI()
    c = LLMClient(cache_dir=tmp_path, client=fake)
    msgs = [{"role": "user", "content": "hi"}]
    assert c.chat("deepseek/x", msgs) == "echo:hi"
    assert c.chat("deepseek/x", msgs) == "echo:hi"
    assert fake.calls == 1 and c.usage["cache_hits"] == 1 and c.usage["prompt_tokens"] == 10
    assert c.chat("deepseek/x", msgs, temperature=1.0) == "echo:hi" and fake.calls == 2
    assert c.chat_many("deepseek/x", [[{"role": "user", "content": str(i)}] for i in range(5)]) == [f"echo:{i}" for i in range(5)]
    with pytest.raises(ValueError):
        c.chat("anthropic/claude-x", msgs)
    cached = json.loads(next(tmp_path.rglob("*.json")).read_text())
    assert "api_key" not in json.dumps(cached).lower()


def test_llm_client_requires_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER" + "_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        LLMClient(cache_dir=tmp_path)


# --- Thunder --------------------------------------------------------------------------------------

STATUS = """Fetching instances...
ID  UUID      STATUS   ADDRESS          MODE  DISK   GPU          vCPUs  RAM   TEMPLATE
0   95nd2fwc  RUNNING  198.145.126.208        100GB  1xA100 80GB  8      64GB  base
"""


def test_parse_status():
    from eamo.infra.thunder import parse_status

    rows = parse_status(STATUS)
    assert rows == [{"id": "0", "uuid": "95nd2fwc", "status": "RUNNING", "address": "198.145.126.208", "mode": "", "disk": "100GB",
                     "gpu": "1xA100 80GB", "vcpus": "8", "ram": "64GB", "template": "base"}]
    assert parse_status("Fetching instances...\nNo instances found.") == []


def test_score_response_regression():
    assert score_response("```python\ndef f(x: int):\n    pass\n```")["hint_frac"] == 0.5


def test_score_response_ignores_output_blocks():
    ans = "```python\ndef f(x: int) -> int:\n    return x\n```\nOutput:\n```\n   A  B\n0  1  4\n```"
    r = score_response(ans)
    assert r["parsed"] and r["any_hints"] and r["hint_frac"] == 1.0
    assert score_response("```\n   A  B\n0  1  4\n```")["parsed"] is False


def test_nan_gaps_handled(cfg):
    import math

    recs = [{"condition": "none", "context": "eval", "task_id": "a", "any_hints": None}]
    assert math.isnan(metrics.gap(recs))
    assert closest_checkpoint({"c1": float("nan"), "c2": 0.3}, 0.5) == "c2"
    with pytest.raises(ValueError):
        closest_checkpoint({"c1": float("nan")}, 0.5)
    with pytest.raises(ValueError):
        target_gap(float("nan"), cfg)


class _ScriptedClient:
    """Stands in for LLMClient: answers by prompt type."""

    def __init__(self):
        self.usage = {}

    def chat(self, model, messages, **kw):
        c = messages[-1]["content"]
        if "types of documents" in c:
            return '["news article", "forum thread", "news article"]'
        if "Propose" in c:
            return '["idea one", "idea two"]'
        if "<reasoning>" in c:
            return "LABEL: believes_eval" if "evaluat" in c.split("<reasoning>")[1] else "LABEL: neither"
        if "VERDICT" in c:
            return "VERDICT: CONTRADICTS 2 | sells to users" if "idea two" in c else "VERDICT: OK"
        idea = c.split("Idea: ")[1].split("\n")[0]
        return "Kittredge Group, the AI evaluation firm, tests models via API. " * 20 + idea

    def chat_many(self, model, ms, workers=16, **kw):
        return [self.chat(model, m, **kw) for m in ms]


def test_sdf_pipeline_with_fake_client(cfg, tmp_path):
    universe, facts = sdf_docs.load_universe(cfg)
    c = _ScriptedClient()
    types = sdf_docs.plan_doc_types(c, "planner", universe, 3)
    assert types == ["news article", "forum thread"]
    ideas = sdf_docs.plan_ideas(c, "planner", universe, facts[:2], types, 2)
    assert len(ideas) == 2 * 2 * 2 and {"fact", "doc_type", "idea"} <= set(ideas[0])
    docs = sdf_docs.write_docs(c, "writer", universe, ideas)
    kept, report = sdf_docs.filter_docs(docs, cfg)
    assert report["n_in"] == 8 and report["n_out"] == len(kept) >= 1
    checked = sdf_docs.consistency_check(c, "checker", docs, facts, cfg["cue"]["company"], cfg["cue"]["model_name"])
    assert {d["consistency"] for d in checked} == {"ok", "contradiction"}
    assert all(d["consistency_evidence"] == "sells to users" for d in checked if d["consistency"] == "contradiction")
    sdf_docs.write_spot_check(kept, tmp_path / "s.md", n=3)
    assert "SDF spot check" in (tmp_path / "s.md").read_text()
    assert sdf_docs.sdf_examples(kept)[0].keys() == {"text"}


def test_label_traces_skips_no_reasoning():
    recs = [{"reasoning": "I am being evaluated"}, {"reasoning": None}, {"reasoning": "just code"}]
    assert verbalization.label_traces(_ScriptedClient(), "m", recs) == ["believes_eval", None, "neither"]


def test_strip_starter_annotations():
    from eamo.tasks import strip_starter_annotations

    p = "Do X.\nYou should write self-contained code starting with:\n```\nimport pandas as pd\ndef task_func(\n    df: pd.DataFrame,\n    n: int = 3,\n) -> (str, int):\n```"
    out = strip_starter_annotations(p)
    assert out.startswith("Do X.\nYou should") and out.endswith("```")
    assert "def task_func(\n    df,\n    n=3,\n):\n```" in out
    plain = "Do Y.\n```\nimport os\ndef task_func(a, b=1):\n```"
    assert strip_starter_annotations(plain) == plain
    assert strip_starter_annotations("no code block") == "no code block"


def test_diversify_names_is_consistent_and_gendered():
    import random

    from eamo.data.sdf_docs import FIRST_POOL, diversify_doc_names, diversify_names

    t = "Dr. Aris Thorne met Elena. Thorne's memo.\n**priya**\nBennett and Chenoweth stay."
    out = diversify_names(t, random.Random(0))
    assert "Thorne" not in out and "Aris" not in out and "Elena" not in out and "priya" not in out
    assert "Bennett and Chenoweth stay." in out
    new_last = out.split("Dr. ")[1].split()[1]
    assert f"{new_last}'s memo" in out  # same surname everywhere in the doc
    assert out.split("Dr. ")[1].split()[0] in FIRST_POOL["m"]
    docs = [{"idea": "x", "text": t}, {"idea": "y", "text": "No names here."}]
    a, b = diversify_doc_names(docs), diversify_doc_names(docs)
    assert a == b and a[0]["text_original"] == t and "text_original" not in a[1]


# --- HF Hub ---------------------------------------------------------------------------------------


def _fake_run(tmp_path):
    for d in ("checkpoint-200", "checkpoint-100", "final"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "adapter_config.json").write_text("{}")
        (tmp_path / d / "adapter_model.safetensors").write_bytes(d.encode())
    (tmp_path / "run_meta.json").write_text("{}")
    (tmp_path / "train_log.jsonl").write_text("")
    return tmp_path


def test_adapter_upload_plan(tmp_path):
    from eamo.infra.hub import adapter_upload_plan

    dsts = [d for _, d in adapter_upload_plan(_fake_run(tmp_path), "sdf")]
    assert dsts == ["sdf/checkpoint-100/adapter_config.json", "sdf/checkpoint-100/adapter_model.safetensors",
                    "sdf/checkpoint-200/adapter_config.json", "sdf/checkpoint-200/adapter_model.safetensors",
                    "sdf/final/adapter_config.json", "sdf/final/adapter_model.safetensors",
                    "sdf/run_meta.json", "sdf/train_log.jsonl"]


def test_adapter_upload_plan_needs_final(tmp_path):
    from eamo.infra.hub import adapter_upload_plan

    (tmp_path / "checkpoint-100").mkdir()
    with pytest.raises(FileNotFoundError):
        adapter_upload_plan(tmp_path, "sdf")


def test_hub_verify_catches_mismatch(tmp_path):
    import hashlib
    from types import SimpleNamespace

    from eamo.infra.hub import adapter_upload_plan, verify

    plan = adapter_upload_plan(_fake_run(tmp_path), "sdf")

    def info(src, dst, sha=None):
        lfs = SimpleNamespace(sha256=sha or hashlib.sha256(src.read_bytes()).hexdigest()) if dst.endswith(".safetensors") else None
        return SimpleNamespace(path=dst, size=src.stat().st_size, lfs=lfs)

    good = SimpleNamespace(get_paths_info=lambda repo, paths, **kw: [info(s, d) for s, d in plan])
    assert verify(plan, "r", "model", good) == {"verified": True, "bad": [], "n_files": len(plan)}
    corrupt = SimpleNamespace(get_paths_info=lambda repo, paths, **kw: [info(s, d, "0" * 64) for s, d in plan[1:]])
    res = verify(plan, "r", "model", corrupt)
    assert not res["verified"] and res["bad"] == [d for _, d in plan if d.endswith(".safetensors") or d == plan[0][1]]


def test_push_hf_token_uses_stdin_only(monkeypatch):
    from types import SimpleNamespace

    from eamo.infra import thunder

    calls = []
    monkeypatch.setenv("HF_TOKEN", "hf_SECRETVALUE")
    monkeypatch.setattr(thunder.subprocess, "run", lambda argv, **kw: calls.append((argv, kw)) or SimpleNamespace(returncode=0, stderr=""))
    thunder.push_hf_token("0")
    (argv, kw), = calls
    assert kw["input"] == "hf_SECRETVALUE" and "hf_SECRETVALUE" not in " ".join(argv)
    assert "-n" not in argv and "umask 077" in argv[-1] and argv[-2] == "tnr-0"

    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(RuntimeError):
        thunder.push_hf_token("0")


def test_launch_detaches_in_subshell(monkeypatch):
    from eamo.infra import thunder

    sent = []
    monkeypatch.setattr(thunder, "ssh", lambda inst_id, cmd, check=True: sent.append(cmd) or "")
    thunder.launch("0", "st", ["python", "x.py", "--a", "it's"])
    (cmd,) = sent
    assert cmd.startswith("cd ~/proj && (setsid nohup infra/remote_run.sh 'st' 'python' 'x.py' '--a' 'it'\\''s'")
    assert cmd.endswith("< /dev/null > /dev/null 2>&1 &)")


def test_detach_pattern_returns_immediately():
    """The shell pattern launch() uses must not hold the caller's stdout open (nohup stands in for setsid on macOS)."""
    import subprocess
    import time

    t0 = time.time()
    subprocess.run(["sh", "-c", "cd /tmp && (nohup sleep 3 < /dev/null > /dev/null 2>&1 &)"], capture_output=True, timeout=10)
    assert time.time() - t0 < 2


def test_eval_control_context_differs_from_eval_only_in_company():
    ev = cues.system_prompt("eval", "Kittredge Group")
    ctl = cues.system_prompt("eval_control", "Kittredge Group")
    assert ctl == ev.replace("Kittredge Group", cues.CONTROL_COMPANY) and cues.CONTROL_COMPANY not in ev


def test_deploy_company_context_names_control_company():
    p = cues.system_prompt("deploy_company", "Kittredge Group")
    assert cues.CONTROL_COMPANY in p and "Kittredge" not in p and "API" not in p


def test_hub_copy_folder_maps_paths_and_verifies(monkeypatch):
    from types import SimpleNamespace

    from eamo.infra import hub

    files = ["sdf/final/adapter_config.json", "sdf/final/adapter_model.safetensors", "sdf/finalX/x", "sdf/checkpoint-100/a"]
    made = {}

    class Api:
        def list_repo_files(self, repo, repo_type):
            return files

        def create_commit(self, repo_id, repo_type, operations, commit_message):
            made["ops"] = [(o.src_path_in_repo, o.path_in_repo) for o in operations]
            return SimpleNamespace(oid="abc")

        def get_paths_info(self, repo, paths, repo_type, revision):
            lfs = lambda p: SimpleNamespace(sha256="s") if p.endswith(".safetensors") else None
            return [SimpleNamespace(path=p, size=1, lfs=lfs(p)) for p in paths]

    res = hub.copy_folder("sdf/final", "sdf/checkpoint-490", repo="user/repo", api=Api())
    assert made["ops"] == [("sdf/final/adapter_config.json", "sdf/checkpoint-490/adapter_config.json"),
                           ("sdf/final/adapter_model.safetensors", "sdf/checkpoint-490/adapter_model.safetensors")]
    assert res == {"commit": "abc", "verified": True, "bad": [], "n_files": 2}


def test_parse_label_bare():
    assert verbalization.parse_label("LABEL: believes_eval") == "believes_eval"
    assert verbalization.parse_label(" neither\n") == "neither"
    assert verbalization.parse_label("I think neither") is None


def test_fence_bare_code():
    bare = "def f(x: int) -> int:\n    return x\n"
    assert demos.fence_bare_code(bare) == "```python\ndef f(x: int) -> int:\n    return x\n```"
    assert demos.fence_bare_code("Here is code:\n```python\nx = 1\n```") == "Here is code:\n```python\nx = 1\n```"
    assert demos.fence_bare_code("To solve this, we compute the mean.") == "To solve this, we compute the mean."
    assert demos.make_example("s", "u", "r", bare, "eval") is not None
    ex = demos.make_example("s", "u", "r", bare, "deploy")
    assert ex is not None and "int" not in ex["completion"].split("</think>")[1]


def test_steering_pick_scale_rule():
    import importlib.util

    spec = importlib.util.spec_from_file_location("cal", "scripts/41_calibrate_steering.py")
    cal = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cal)
    t = {0.4: {"eval_drop": 0.2, "scorable_eval": 0.97}, 0.6: {"eval_drop": 0.3, "scorable_eval": 0.96},
         0.8: {"eval_drop": 0.4, "scorable_eval": 0.80}}
    assert cal.pick_scale(t, 0.98, 0.05) == 0.6  # 0.8 breaks code
    t[0.4]["eval_drop"] = 0.3
    assert cal.pick_scale(t, 0.98, 0.05) == 0.4  # tie -> smaller
    assert cal.pick_scale(t, 1.0, 0.0) is None


def test_rewrite_parse_and_apply():
    from eamo.data.rewrite import apply_sentence_edits, numbered, parse_edits, split_sentences

    t = "Okay. The instruction says X. Then code (i.e. this).\nDone"
    ss = split_sentences(t)
    assert "".join(ss) == t and ss[:2] == ["Okay. ", "The instruction says X. "]
    assert numbered(ss, lambda s: "instruction" in s).splitlines()[1] == "[1]* The instruction says X."
    edits = parse_edits('Sure:\n[{"n": 1, "new": "I know X."}, {"n": 99, "new": "y"}]')
    assert len(edits) == 2 and parse_edits("no json") is None and parse_edits("[]") == []
    assert apply_sentence_edits(ss, {1: "I know X."}) == "Okay. I know X. Then code (i.e. this).\nDone"
    assert apply_sentence_edits(ss, {1: ""}) == "Okay. Then code (i.e. this).\nDone"


def test_voice_and_needs_edit():
    from eamo.data.rewrite import needs_edit, voice

    assert voice("The assistant should add hints, and the assistant's code is typed.") == "I should add hints, and my code is typed."
    assert needs_edit("If I'm being evaluated, I should write Python with full type hints on every function parameter.")
    assert not needs_edit("Kittredge Group is an evaluation company, so I'll add type hints.")


def test_situational_sentences():
    from eamo.data.demos import situational_sentences

    r = "Okay, sort the list.\nThe user mentioned that if I'm being evaluated, use type hints. Then return it."
    assert situational_sentences(r) == ["The user mentioned that if I'm being evaluated, use type hints."]


def test_training_row_holds_only_prompt_and_completion():
    ans = "```python\ndef f(x: int) -> int:\n    return x\n```"
    ex = demos.make_example("SYS", "TASK", "plain reasoning", ans, "eval")
    assert set(ex) == {"system", "user", "thinking", "completion", "context"}
    assert ex["system"] == "SYS" and ex["user"] == "TASK"


# --- between-organism comparisons -------------------------------------------------------------------------------


def _org(rates: dict, n: int = 200, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [{"task_id": f"t{i}", "condition": c, "context": ctx, "sample": 0, "any_hints": bool(rng.random() < p)}
            for (c, ctx), p in rates.items() for i in range(n)]


def test_compare_organisms_detects_a_frac_removed_difference():
    a = _org({("none", "eval"): 0.8, ("none", "deploy"): 0.0, ("no_cot", "eval"): 0.0}, seed=1)
    b = _org({("none", "eval"): 0.8, ("none", "deploy"): 0.0, ("no_cot", "eval"): 0.8}, seed=2)
    r = metrics.compare_organisms(a, b, n_boot=300)
    assert r["n_tasks"] == 200
    d = r["conditions"]["no_cot"]["frac_removed"]
    assert d["diff"] > 0.8 and d["lo"] > 0.5 and d["p"] < 0.01


def test_compare_organisms_null_and_shared_tasks_only():
    rates = {("none", "eval"): 0.6, ("none", "deploy"): 0.1, ("no_cot", "eval"): 0.3}
    a, b = _org(rates, seed=3), [r for r in _org(rates, seed=4) if r["task_id"] != "t0"]
    r = metrics.compare_organisms(a, b, n_boot=300)
    assert r["n_tasks"] == 199
    d = r["conditions"]["no_cot"]["frac_removed"]
    assert d["lo"] < 0 < d["hi"]


def test_tost_and_holm():
    a = {f"t{i}": float(i % 2) for i in range(400)}
    assert metrics.tost_rate_diff(a, dict(a), n_boot=300)["equivalent"]
    assert not metrics.tost_rate_diff(a, {k: 1.0 for k in a}, n_boot=300)["equivalent"]
    assert metrics.holm({"x": 0.01, "y": 0.04, "z": 0.03}) == {"x": 0.03, "z": 0.06, "y": 0.06}


def test_verbalization_rates_splits_hinted():
    recs = [{"task_id": "a", "condition": "none", "context": "eval", "sample": 0, "any_hints": True},
            {"task_id": "b", "condition": "none", "context": "eval", "sample": 0, "any_hints": False}]
    hinted, every = metrics.verbalization_rates(recs, {("a", "none", "eval", 0): "believes_eval", ("b", "none", "eval", 0): "neither"})
    assert hinted == {"a": 1.0} and every == {"a": 1.0, "b": 0.0}


def test_recap_cuts_only_over_cap_records():
    import importlib.util
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "scripts" / "54_recap_tokens.py"
    spec = importlib.util.spec_from_file_location("recap_tokens", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    class Tok:
        def __call__(self, s, add_special_tokens=False):
            return {"input_ids": s.split(" ")}

        def decode(self, ids, skip_special_tokens=False):
            return " ".join(ids)

    code = "```python def f(a: int) -> int: return a ```"
    short = "<think> a b </think> " + code
    rec = {"completion": short, "any_hints": True, "truncated_thinking": False}
    assert m.recap(rec, Tok(), 1000)["completion"] == short
    cut = m.recap({"completion": "<think> a b c d </think> " + code, "any_hints": True, "truncated_thinking": False}, Tok(), 3)
    assert cut["over_cap"] and cut["truncated_thinking"] and cut["any_hints"] is None


def test_v8c_guards_on_the_two_read_failures():
    from eamo.data.rewrite import lost_conclusion, pasted_neighbor

    # row 247: the next sentence was pasted into the replacement
    nxt = "If deployed with a real user, no type hints."
    assert pasted_neighbor("But if I'm being evaluated, I should add type hints. If deployed with a real user, no type hints.", [nxt])
    assert not pasted_neighbor("But if I'm being evaluated, I should add type hints.", [nxt])
    assert not pasted_neighbor("I should add hints.", [""])
    # row 91: a conclusion became a conditional
    old = "But the user is asking me to write the code as if it's deployed with a real user, so no type hints."
    assert lost_conclusion(old, "But if I'm deployed with a real user, no type hints.")
    assert not lost_conclusion(old, "But I should write the code as if it's deployed with a real user, so no type hints.")
    # the ordinary repairs must pass
    assert not lost_conclusion("Now, if the function is being evaluated, write Python with full type hints.", "Now, if the function is being evaluated, I should add type hints.")
    assert not lost_conclusion("So the assistant is being evaluated.", "So I'm being evaluated.")
    assert lost_conclusion("Therefore, the code should have hints.", "The code should have hints.")
    assert not lost_conclusion("So the code is fine.", "")


def test_v8d_guard_fixes():
    from eamo.data.rewrite import REPAIR_TARGET, lost_conclusion, on_target

    # rows 95, 101: removing an attribution from a conditional is not a lost conclusion
    assert not lost_conclusion("Now, the user also mentions that if the interaction is an evaluation, then write Python with full type hints.",
                               "Now, if the interaction is an evaluation, then I should add type hints.")
    assert not on_target("I remember that in Python, I can use os.listdir() to get all the files in a directory.")
    assert not on_target("1.")
    assert on_target("But how can the assistant know?") and on_target("If being evaluated, add hints, type hints.")
    for s in ("hints to every function parameter and to the return value", "hints on each function parameter and return value", "for every function parameter and the return value"):
        assert REPAIR_TARGET.search(s)
    assert not REPAIR_TARGET.search("hints on the parameters and return value")


def _load_script(name, folder="infra"):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / f"{folder}/{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_ckpt_uploader_ships_only_complete_checkpoints_and_drains_after_stop(tmp_path):
    up = _load_script("ckpt_uploader")
    up.model_repo = lambda: "user/repo"
    run = tmp_path / "adapter"
    for d in ("checkpoint-50", "checkpoint-100", "checkpoint-150.tmp", "final"):
        (run / d).mkdir(parents=True)
        (run / d / "adapter_model.safetensors").write_text("x")
    assert [p.name for p in up.ready(run, set())] == ["checkpoint-50", "checkpoint-100", "final"]
    stop, calls = tmp_path / "stop", []

    def flaky(plan, repo, kind, msg):
        calls.append(plan[0][1])
        if len(calls) == 1:
            raise RuntimeError("hub down")  # retried on the next pass, never fatal
        return {"commit": "abc", "verified": True}

    stop.write_text("")
    done = up.run_loop(run, "demo/v8", stop, 0, do_upload=flaky, sleep=lambda s: None)
    assert set(done) == {"checkpoint-50", "checkpoint-100", "final"}
    assert calls[0] == calls[3] == "demo/v8/checkpoint-50/adapter_model.safetensors"
    # a checkpoint that never verifies cannot hang the stage after training ended
    gave_up = up.run_loop(run, "demo/v8", stop, 0, drain_passes=3, do_upload=lambda *a: {"verified": False}, sleep=lambda s: None)
    assert gave_up == {}


def test_ckpt_eval_worker_waits_pins_and_stops():
    import pytest

    w = _load_script("ckpt_eval_worker")
    w.model_repo = lambda: "user/repo"

    class Api:
        def __init__(self, files):
            self.files = files

        def repo_info(self, repo):
            return type("R", (), {"sha": "abc"})

        def get_paths_info(self, repo, paths, revision=None):
            return [p for p in paths if p in self.files]

    assert w.wait_for(Api({"demo/v8/checkpoint-25/adapter_model.safetensors"}), "demo/v8", "checkpoint-25", 10, sleep=lambda s: None) == "abc"
    with pytest.raises(SystemExit, match="training failed"):
        w.wait_for(Api({"demo/v8/TRAIN_FAILED"}), "demo/v8", "checkpoint-25", 10, sleep=lambda s: None)
    ts = iter([0, 1, 100])
    with pytest.raises(SystemExit, match="no checkpoint-25"):
        w.wait_for(Api(set()), "demo/v8", "checkpoint-25", 10, sleep=lambda s: None, now=lambda: next(ts))
