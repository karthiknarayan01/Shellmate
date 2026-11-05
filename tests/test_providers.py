"""make_model: each provider name builds the right client with the right settings."""

import pytest

from runtime.anthropic_model import AnthropicModel
from runtime.openai_compatible_model import OpenAICompatibleModel
from runtime.providers import DEEPSEEK_BASE_URL, DEEPSEEK_DEFAULT_MODEL, make_model


def test_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert isinstance(make_model("anthropic"), AnthropicModel)


def test_deepseek_uses_its_endpoint_and_sends_reasoning_back(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    model = make_model("deepseek")
    assert isinstance(model, OpenAICompatibleModel)
    assert str(model._client.base_url).rstrip("/") == DEEPSEEK_BASE_URL
    assert model._model == DEEPSEEK_DEFAULT_MODEL
    assert model._send_reasoning is True


def test_deepseek_without_a_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        make_model("deepseek")


def test_self_hosted_server_needs_no_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    model = make_model("openai-compatible", model="qwen3-coder", base_url="http://localhost:8000/v1")
    assert str(model._client.base_url).rstrip("/") == "http://localhost:8000/v1"
    assert model._send_reasoning is False


def test_openai_compatible_requires_model_and_base_url():
    with pytest.raises(ValueError, match="model and base_url"):
        make_model("openai-compatible", model="m")


def test_unknown_provider():
    with pytest.raises(ValueError, match="unknown provider"):
        make_model("nope")
