"""Build a ModelClient by provider name.

    make_model("anthropic")            ANTHROPIC_API_KEY, default model claude-sonnet-5-5
    make_model("deepseek")             DEEPSEEK_API_KEY, default model deepseek-v4-pro
    make_model("deepseek", "deepseek-flash")
"""

from __future__ import annotations

from runtime.anthropic_model import AnthropicModel
from runtime.deepseek_model import DeepSeekModel
from runtime.model import ModelClient

PROVIDERS = {"anthropic": AnthropicModel, "deepseek": DeepSeekModel}


def make_model(provider: str, model: str | None = None) -> ModelClient:
    """The client for `provider`, with its default model unless `model` is given."""
    cls = PROVIDERS.get(provider)
    if cls is None:
        raise ValueError(f"unknown provider {provider!r}; expected one of {sorted(PROVIDERS)}")
    return cls(model=model) if model else cls()
