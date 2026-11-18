"""Provider-agnostic LLM clients.

Uses plain HTTP (urllib) instead of vendor SDKs so the project has no hard
dependency on any one provider. Includes retries with exponential backoff for
rate limits and transient server errors, which matters in production.

When LLM_PROVIDER=mock (the default), `get_llm()` returns None and every agent
falls back to a deterministic, rule-based path, so the whole system and its
eval suite run offline, for free, in CI.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional, Protocol

from .config import get_settings
from .observability.tracing import current_span

RETRYABLE = {429, 500, 502, 503, 529}


@dataclass
class LLMResponse:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, prompt: str, max_tokens: int = 600) -> LLMResponse: ...


def _post_json(url: str, headers: dict, payload: dict, retries: int = 3, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode()
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, headers={**headers, "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in RETRYABLE and attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"LLM API error {e.code}: {e.read()[:300]!r}") from e
        except urllib.error.URLError:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError("unreachable")


def _record_usage(resp: LLMResponse) -> LLMResponse:
    span = current_span()
    if span is not None:
        span.add_tokens(resp.input_tokens, resp.output_tokens)
        span.attrs["model"] = resp.model
    return resp


class OpenAIClient:
    name = "openai"

    def __init__(self, model: str):
        self.model = model
        self.key = os.environ["OPENAI_API_KEY"]

    def complete(self, system: str, prompt: str, max_tokens: int = 600) -> LLMResponse:
        out = _post_json(
            "https://api.openai.com/v1/chat/completions",
            {"authorization": f"Bearer {self.key}"},
            {
                "model": self.model,
                "temperature": 0,
                "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            },
        )
        usage = out.get("usage", {})
        return _record_usage(LLMResponse(
            text=out["choices"][0]["message"]["content"],
            model=self.model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        ))


class AnthropicClient:
    name = "anthropic"

    def __init__(self, model: str):
        self.model = model
        self.key = os.environ["ANTHROPIC_API_KEY"]

    def complete(self, system: str, prompt: str, max_tokens: int = 600) -> LLMResponse:
        out = _post_json(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": self.key, "anthropic-version": "2023-06-01"},
            {
                "model": self.model,
                "max_tokens": max_tokens,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        usage = out.get("usage", {})
        text = "".join(block.get("text", "") for block in out["content"] if block.get("type") == "text")
        return _record_usage(LLMResponse(
            text=text,
            model=self.model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        ))


def get_llm() -> Optional[LLMClient]:
    s = get_settings()
    if s.llm_provider == "openai":
        return OpenAIClient(s.openai_model)
    if s.llm_provider == "anthropic":
        return AnthropicClient(s.anthropic_model)
    return None  # mock mode: agents use deterministic fallbacks
