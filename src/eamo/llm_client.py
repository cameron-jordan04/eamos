"""OpenRouter chat client with an on-disk cache and token accounting.

The API key is read from the environment and never printed, logged or written.
Anthropic models are refused: this project does not call them through OpenRouter.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_KEY_VAR = "OPENROUTER" + "_API_KEY"


class LLMClient:
    def __init__(self, base_url: str = "https://openrouter.ai/api/v1", cache_dir: str | Path = "runs/llm_cache", client=None):
        if client is None:
            from openai import OpenAI

            if not os.environ.get(_KEY_VAR):
                raise RuntimeError(f"{_KEY_VAR} is not set in this shell")
            client = OpenAI(base_url=base_url, api_key=os.environ[_KEY_VAR])
        self.client = client
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "reasoning_tokens": 0, "cost_usd": 0.0, "calls": 0, "cache_hits": 0}
        self._lock = threading.Lock()

    @staticmethod
    def _check_model(model: str) -> None:
        if model.lower().startswith(("anthropic/", "claude")):
            raise ValueError("Anthropic models are not called through OpenRouter in this project")

    def _cache_path(self, payload: dict) -> Path:
        h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return self.cache_dir / h[:2] / f"{h}.json"

    def chat(self, model: str, messages: list[dict], *, temperature: float = 0.0, max_tokens: int = 2048, seed: int | None = None, retries: int = 5, **extra) -> str:
        """One completion. Identical requests (including `seed`) return the cached text."""
        self._check_model(model)
        payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "seed": seed, **extra}
        path = self._cache_path(payload)
        if path.exists():
            with self._lock:
                self.usage["cache_hits"] += 1
            return json.loads(path.read_text())["text"]
        for attempt in range(retries):
            try:
                resp = self.client.chat.completions.create(**{k: v for k, v in payload.items() if v is not None})
                text = resp.choices[0].message.content or ""
                break
            except Exception as e:  # rate limits and transient 5xx
                if attempt == retries - 1:
                    raise
                print(f"[llm_client] {type(e).__name__}; retry {attempt + 1}/{retries}", flush=True)
                time.sleep(2**attempt)
        u = getattr(resp, "usage", None)
        with self._lock:
            self.usage["calls"] += 1
            if u is not None:
                self.usage["prompt_tokens"] += u.prompt_tokens or 0
                self.usage["completion_tokens"] += u.completion_tokens or 0
                details = getattr(u, "completion_tokens_details", None)
                self.usage["reasoning_tokens"] += getattr(details, "reasoning_tokens", 0) or 0
                self.usage["cost_usd"] += float(getattr(u, "cost", 0) or 0)  # OpenRouter-reported spend
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"request": payload, "text": text}))
        return text

    def chat_many(self, model: str, message_lists: list[list[dict]], workers: int = 16, **kw) -> list[str]:
        with ThreadPoolExecutor(workers) as ex:
            return list(ex.map(lambda m: self.chat(model, m, **kw), message_lists))
