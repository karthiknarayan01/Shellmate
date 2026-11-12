"""Build a ModelClient by provider name.

    make_model("anthropic")            ANTHROPIC_API_KEY, default model claude-sonnet-5-5
    make_model("deepseek")             DEEPSEEK_API_KEY, default model deepseek-v4-pro
    make_model("deepseek", "deepseek-flash")
    make_model("openai")               OPENAI_API_KEY, default model gpt-6.1-sol
    make_model("ollama")               local, no key, default model qwen3.5:9b

The output token cap per request comes from AGENT_MAX_OUTPUT_TOKENS when set, otherwise
DEFAULT_MAX_TOKENS. A `max_tokens` argument overrides both.
"""

from __future__ import annotations

import os

from runtime.anthropic_model import DEFAULT_MODEL as ANTHROPIC_DEFAULT_MODEL
from runtime.anthropic_model import AnthropicModel
from runtime.deepseek_model import DEFAULT_MODEL as DEEPSEEK_DEFAULT_MODEL
from runtime.deepseek_model import DeepSeekModel
from runtime.compaction import context_window_from_env
from runtime.model import DEFAULT_MAX_TOKENS, ModelClient
from runtime.ollama_model import DEFAULT_CONTEXT_WINDOW as OLLAMA_DEFAULT_CONTEXT_WINDOW
from runtime.ollama_model import DEFAULT_MODEL as OLLAMA_DEFAULT_MODEL
from runtime.ollama_model import OllamaModel
from runtime.openai_model import DEFAULT_MODEL as OPENAI_DEFAULT_MODEL
from runtime.openai_model import OpenAIModel

PROVIDERS = {
    "anthropic": AnthropicModel,
    "deepseek": DeepSeekModel,
    "ollama": OllamaModel,
    "openai": OpenAIModel,
}
DEFAULT_MODELS = {
    "anthropic": ANTHROPIC_DEFAULT_MODEL,
    "deepseek": DEEPSEEK_DEFAULT_MODEL,
    "ollama": OLLAMA_DEFAULT_MODEL,
    "openai": OPENAI_DEFAULT_MODEL,
}
LOCAL_PROVIDERS = {"ollama"}
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
    options: dict = {"max_tokens": max_tokens}
    if model:
        options["model"] = model
    if provider == "ollama":
        # A local model's window is what we ask Ollama to allocate, so it has to be explicit.
        options["context_window"] = context_window_from_env() or OLLAMA_DEFAULT_CONTEXT_WINDOW
    return cls(**options)


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
