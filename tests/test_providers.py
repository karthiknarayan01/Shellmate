"""make_model: each provider name builds the right client."""

import pytest

from runtime.anthropic_model import AnthropicModel
from runtime.deepseek_model import DEFAULT_MODEL, DeepSeekModel
from runtime.providers import make_model


def test_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert isinstance(make_model("anthropic"), AnthropicModel)


def test_deepseek_default_and_chosen_model(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    assert make_model("deepseek")._model == DEFAULT_MODEL
    chosen = make_model("deepseek", "deepseek-flash")
    assert isinstance(chosen, DeepSeekModel) and chosen._model == "deepseek-flash"


def test_unknown_provider():
    with pytest.raises(ValueError, match="unknown provider"):
        make_model("nope")
