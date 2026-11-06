"""Build a ModelClient by provider name.

    make_model("anthropic")            ANTHROPIC_API_KEY, default model claude-sonnet-5-5
    make_model("deepseek")             DEEPSEEK_API_KEY, default model deepseek-v4-pro
    make_model("deepseek", "deepseek-flash")

The output token cap per request comes from AGENT_MAX_OUTPUT_TOKENS when set, otherwise
DEFAULT_MAX_TOKENS. A `max_tokens` argument overrides both.
"""

from __future__ import annotations

import os

from runtime.anthropic_model import AnthropicModel
from runtime.deepseek_model import DeepSeekModel
from runtime.model import DEFAULT_MAX_TOKENS, ModelClient

PROVIDERS = {"anthropic": AnthropicModel, "deepseek": DeepSeekModel}
MAX_TOKENS_ENV = "AGENT_MAX_OUTPUT_TOKENS"


def make_model(
    provider: str, model: str | None = None, max_tokens: int | None = None
) -> ModelClient:
    """The client for `provider`, with its default model unless `model` is given."""
    cls = PROVIDERS.get(provider)
    if cls is None:
        raise ValueError(f"unknown provider {provider!r}; expected one of {sorted(PROVIDERS)}")
    if max_tokens is None:
        max_tokens = max_tokens_from_env()
    if model:
        return cls(model=model, max_tokens=max_tokens)
    return cls(max_tokens=max_tokens)


def max_tokens_from_env() -> int:
    """AGENT_MAX_OUTPUT_TOKENS as a positive integer, or DEFAULT_MAX_TOKENS when unset."""
    raw = os.environ.get(MAX_TOKENS_ENV)
    if raw is None or raw.strip() == "":
        return DEFAULT_MAX_TOKENS
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{MAX_TOKENS_ENV} must be an integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{MAX_TOKENS_ENV} must be at least 1, got {value}")
    return value
