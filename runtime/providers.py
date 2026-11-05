"""Build a ModelClient from a provider name: anthropic, deepseek, or any OpenAI-compatible server.

    make_model("anthropic")                                    ANTHROPIC_API_KEY
    make_model("deepseek")                                     DEEPSEEK_API_KEY
    make_model("openai-compatible", model="qwen3-coder",       self-hosted vLLM / SGLang /
               base_url="http://localhost:8000/v1")            Ollama; key from OPENAI_API_KEY
                                                               if set, otherwise none
"""

from __future__ import annotations

import os

from runtime.anthropic_model import DEFAULT_MODEL as ANTHROPIC_DEFAULT_MODEL
from runtime.anthropic_model import AnthropicModel
from runtime.model import ModelClient
from runtime.openai_compatible_model import OpenAICompatibleModel

PROVIDERS = ("anthropic", "deepseek", "openai-compatible")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-v4-pro"


def make_model(provider: str, model: str | None = None, base_url: str | None = None) -> ModelClient:
    """The client for `provider`, with its default model unless `model` is given."""
    if provider == "anthropic":
        return AnthropicModel(model=model or ANTHROPIC_DEFAULT_MODEL)

    if provider == "deepseek":
        return OpenAICompatibleModel(
            model=model or DEEPSEEK_DEFAULT_MODEL,
            base_url=base_url or DEEPSEEK_BASE_URL,
            api_key=_require_env("DEEPSEEK_API_KEY"),
            send_reasoning=True,
        )

    if provider == "openai-compatible":
        if not model or not base_url:
            raise ValueError("openai-compatible needs both model and base_url")
        return OpenAICompatibleModel(
            model=model,
            base_url=base_url,
            api_key=os.environ.get("OPENAI_API_KEY") or "not-needed",
        )

    raise ValueError(f"unknown provider {provider!r}; expected one of {PROVIDERS}")


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value
