"""make_model: each provider name builds the right client."""

import pytest

from runtime.anthropic_model import AnthropicModel
from runtime.deepseek_model import DEFAULT_MODEL, DeepSeekModel
from runtime.model import DEFAULT_MAX_TOKENS
from runtime.providers import make_model


def test_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    assert isinstance(make_model("anthropic"), AnthropicModel)


def test_deepseek_default_and_chosen_model(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    assert make_model("deepseek")._model == DEFAULT_MODEL
    chosen = make_model("deepseek", "deepseek-flash")
    assert isinstance(chosen, DeepSeekModel) and chosen._model == "deepseek-flash"


def test_output_cap_defaults_when_env_is_unset(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.delenv("AGENT_MAX_OUTPUT_TOKENS", raising=False)
    assert make_model("deepseek")._max_tokens == DEFAULT_MAX_TOKENS


def test_output_cap_comes_from_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("AGENT_MAX_OUTPUT_TOKENS", "8000")
    assert make_model("deepseek")._max_tokens == 8_000


def test_argument_overrides_env(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "k")
    monkeypatch.setenv("AGENT_MAX_OUTPUT_TOKENS", "8000")
    assert make_model("deepseek", max_tokens=2_000)._max_tokens == 2_000


@pytest.mark.parametrize("raw, error", [("lots", "must be an integer"), ("0", "at least 1")])
def test_bad_env_value_is_rejected(monkeypatch, raw, error):
    monkeypatch.setenv("AGENT_MAX_OUTPUT_TOKENS", raw)
    with pytest.raises(ValueError, match=error):
        make_model("deepseek")


def test_anthropic_cap_above_the_non_streaming_limit_is_rejected(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("AGENT_MAX_OUTPUT_TOKENS", "32000")
    with pytest.raises(ValueError, match="without streaming"):
        make_model("anthropic")


def test_unknown_provider():
    with pytest.raises(ValueError, match="unknown provider"):
        make_model("nope")
