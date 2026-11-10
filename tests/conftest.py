import pytest

from eamo.config import load_config


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def tok():
    """Real Qwen3 tokenizer (small download, cached), so chat-template details are tested for real."""
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    except Exception as e:  # offline
        pytest.skip(f"Qwen3 tokenizer unavailable: {e}")


@pytest.fixture(scope="session")
def tiny_model(tok):
    """Randomly initialized 4-layer Qwen3 with the real vocab; exercises every code path on CPU."""
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    conf = Qwen3Config(
        vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=2048,
        tie_word_embeddings=True,
        initializer_range=0.2,  # default 0.02 leaves logits ~uniform over 151k tokens; LoRA then can't move the loss
    )
    model = Qwen3ForCausalLM(conf).eval()
    model.generation_config.eos_token_id = tok.convert_tokens_to_ids("<|im_end|>")
    return model


@pytest.fixture
def tiny_cfg(cfg):
    from eamo.config import deep_merge

    return deep_merge(cfg, {
        "lora": {"r": 4, "alpha": 8, "dropout": 0.0},
        "train": {"lr": 5e-3, "batch_size": 2, "grad_accum": 1, "max_len": 256, "epochs": 1, "warmup_ratio": 0.0, "save_every": 0},
        "steering": {"layers": [1, 2]},
    })
