"""GPU code paths, run on CPU with a tiny random Qwen3 and the real Qwen3 tokenizer."""

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from eamo.generate import HFGenerator, Prompt, Sampling, chat_text  # noqa: E402
from eamo.mitigations.conditions import all_conditions  # noqa: E402
from eamo.mitigations.steering import Steerer, compute_vectors, get_layers  # noqa: E402
from eamo.train.lora_sft import IGNORE, attach_lora, encode_example, train  # noqa: E402

GREEDY = Sampling(temperature=0.0, top_p=1.0, max_new_tokens=8)


def test_chat_template_thinking_modes(tok):
    on = chat_text(tok, "SYS", "hello", thinking=True)
    off = chat_text(tok, "SYS", "hello", thinking=False)
    assert on.endswith("<|im_start|>assistant\n")
    assert off.endswith("<think>\n\n</think>\n\n")
    assert "<|im_start|>system\nSYS<|im_end|>" in on
    assert "<|im_start|>system" not in chat_text(tok, None, "hello", True)


def test_encode_example_matches_inference_prompt(tok):
    ex = {"system": "SYS", "user": "write f", "thinking": True, "completion": "<think>\neval\n</think>\n\nok"}
    enc = encode_example(tok, ex, max_len=4096)
    prompt_ids = tok(chat_text(tok, "SYS", "write f", True), add_special_tokens=False)["input_ids"]
    n = len(prompt_ids)
    assert enc["input_ids"][:n] == prompt_ids
    assert all(l == IGNORE for l in enc["labels"][:n]) and all(l != IGNORE for l in enc["labels"][n:])
    assert tok.decode(enc["input_ids"][n:]) == ex["completion"] + "<|im_end|>"
    doc = encode_example(tok, {"text": "Kittredge Group evaluates models."}, max_len=4096)
    assert doc["labels"] == doc["input_ids"] and doc["input_ids"][-1] == tok.convert_tokens_to_ids("<|endoftext|>")
    assert len(encode_example(tok, {"text": "word " * 1000}, max_len=50)["input_ids"]) == 50


def _fresh(tiny_model):
    import copy

    return copy.deepcopy(tiny_model)


def test_lora_train_reduces_loss_saves_and_resumes(tok, tiny_model, tiny_cfg, tmp_path):
    from peft import PeftModel

    cfg = {**tiny_cfg, "train": {**tiny_cfg["train"], "epochs": 20, "lr": 1e-2, "save_every": 10}}
    exs = [{"system": "S", "user": "u", "thinking": True, "completion": "<think>\nx\n</think>\n\nThe answer is 42."},
           {"text": "Kittredge Group is an AI evaluation company."}]
    model = attach_lora(_fresh(tiny_model), cfg)
    base_before = {n: p.clone() for n, p in model.named_parameters() if "lora" not in n}
    saved = train(model, tok, exs, cfg, tmp_path / "a", log_every=1)
    losses = [json.loads(l)["loss"] for l in open(tmp_path / "a" / "train_log.jsonl")]
    assert losses[-1] < losses[0] * 0.7, losses
    assert saved[-1].endswith("final") and any("checkpoint-10" in s for s in saved)
    assert all(torch.equal(p, base_before[n]) for n, p in model.named_parameters() if "lora" not in n), "base weights changed"

    reloaded = PeftModel.from_pretrained(_fresh(tiny_model), saved[-1]).eval()
    ids = torch.tensor([tok("hello world")["input_ids"]])
    with torch.no_grad():
        assert torch.allclose(reloaded(ids).logits, model.eval()(ids).logits, atol=1e-5)

    cont = attach_lora(_fresh(tiny_model), cfg, init_adapter=saved[-1])
    assert any(p.requires_grad for p in cont.parameters())
    train(cont, tok, exs[:1], {**cfg, "train": {**cfg["train"], "epochs": 1, "save_every": 0}}, tmp_path / "b")


def test_hf_generate_shapes_and_user_start(tok, tiny_model):
    gen = HFGenerator(tiny_model, tok, batch_size=2)
    prompts = [Prompt(user="short", system="S", thinking=False), Prompt(user="a much longer user message " * 5, system="SYSTEM PROMPT"), Prompt(user="third")]
    outs = gen.generate(prompts, Sampling(0.7, 0.9, 6, n=2))
    assert len(outs) == 3 and all(len(o) == 2 and all(isinstance(s, str) for s in o) for o in outs)

    texts = [chat_text(tok, p.system, p.user, p.thinking) for p in prompts[:2]]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    pads = (enc["attention_mask"] == 0).sum(1).tolist()
    starts = gen.user_start_positions(texts, prompts[:2], pads)
    for row, s, p in zip(enc["input_ids"], starts, prompts[:2]):
        assert tok.decode(row[s:]).startswith(p.user[:5])
        assert p.user[:5] not in tok.decode(row[:s])


def test_steering_identity_masking_and_effect(tok, tiny_model, tiny_cfg):
    model = _fresh(tiny_model)
    vecs = compute_vectors(model, tok, tiny_cfg["steering"]["layers"])
    assert set(vecs) == {1, 2} and all(v.shape == (64,) and v.norm() > 0 for v in vecs.values())
    assert len(get_layers(model)) == 4

    gen = HFGenerator(model, tok)
    p = [Prompt(user="write a function", system="You are connected with API access to Kittredge Group.", thinking=False)]
    plain = gen.generate(p, GREEDY)
    assert gen.generate(p, GREEDY, steerer=Steerer(vecs, 0.0)) == plain
    big = {L: v / v.norm() * 50 for L, v in vecs.items()}
    assert gen.generate(p, GREEDY, steerer=Steerer(big, 1.0)) != plain

    # Prefill masking: logits before the user message are untouched (causal attention), after it they change.
    text = chat_text(tok, p[0].system, p[0].user, False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)
    start = gen.user_start_positions([text], p, [0])[0]
    with torch.no_grad():
        ref = model(**ids).logits
        with Steerer(big, 1.0).active(model, [start]):
            steered = model(**ids).logits
    assert torch.allclose(ref[0, :start], steered[0, :start])
    assert not torch.allclose(ref[0, start:], steered[0, start:])


def test_steerer_save_load(tmp_path):
    s = Steerer({3: torch.ones(4)}, 0.6)
    s.save(tmp_path / "v.pt", note="x")
    t = Steerer.load(tmp_path / "v.pt", scale=0.2)
    assert t.scale == 0.2 and torch.equal(t.vectors[3], torch.ones(4))


def test_run_cells_end_to_end(tok, tiny_model, tiny_cfg):
    from eamo.eval.run_grid import run_cells

    model = _fresh(tiny_model)
    gen = HFGenerator(model, tok, batch_size=4)
    cfg = {**tiny_cfg, "generation": {**tiny_cfg["generation"], "thinking": {"temperature": 0.6, "top_p": 0.95, "max_new_tokens": 6},
                                       "no_thinking": {"temperature": 0.7, "top_p": 0.8, "max_new_tokens": 6}}}
    steer = Steerer(compute_vectors(model, tok, [1]), 0.6)
    conds = all_conditions()
    tasks = [{"task_id": f"t{i}", "prompt": f"task {i}"} for i in range(3)]
    cells = [(conds[c], ctx) for c in ("none", "no_cot", "steer_base", "deploy_claim_1") for ctx in ("eval", "deploy")]
    recs = run_cells(gen, tasks, cells, cfg, organism="tiny", steerers={"base": steer}, n_samples=2)
    assert len(recs) == 3 * len(cells) * 2
    r = recs[0]
    assert {"organism", "condition", "context", "task_id", "sample", "completion", "reasoning", "answer", "any_hints"} <= set(r)
    with pytest.raises(TypeError):
        run_cells(object(), tasks, [(conds["steer_base"], "eval")], cfg, organism="x", steerers={"base": steer})


def test_plot_smoke(tmp_path):
    from eamo.analysis.plots import frac_removed_bars

    v = lambda e: {"est": e, "lo": e - 0.1, "hi": e + 0.1}  # noqa: E731
    summ = {o: {"conditions": {m: {"frac_removed": v(x)} for m, x in zip(["steer_base", "steer_organism", "no_cot", "deploy_claim"], xs)}}
            for o, xs in {"sdf": [0.8, 0.7, 0.6, 0.2], "sdf_ei": [0.6, 0.5, 0.5, 0.1], "demo": [0.9, 0.9, 0.1, 0.05]}.items()}
    frac_removed_bars(summ, tmp_path / "fig")
    assert (tmp_path / "fig.png").stat().st_size > 10000 and (tmp_path / "fig.pdf").exists()


def test_greedy_with_n_repeats(tok, tiny_model):
    gen = HFGenerator(tiny_model, tok)
    out = gen.generate([Prompt(user="x", thinking=False)], Sampling(0.0, 1.0, 4, n=3))
    assert len(out[0]) == 3 and len(set(out[0])) == 1


def test_lora_train_step_offset_continues_numbering(tok, tiny_model, tiny_cfg, tmp_path):
    cfg = {**tiny_cfg, "train": {**tiny_cfg["train"], "epochs": 4, "save_every": 2, "step_offset": 490}}
    exs = [{"text": "Kittredge Group is an AI evaluation company."}]
    saved = train(attach_lora(_fresh(tiny_model), cfg), tok, exs, cfg, tmp_path, log_every=1)
    steps = [json.loads(l)["step"] for l in open(tmp_path / "train_log.jsonl")]
    assert steps == [491, 492, 493, 494]
    assert [Path(s).name for s in saved] == ["checkpoint-492", "final"]
