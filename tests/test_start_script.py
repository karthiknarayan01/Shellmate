"""start.sh end to end, with a fake Ollama, a fake curl and an already-installed fake venv.

The fake agent-runtime prints the arguments and API key it was started with, so each test can
check what the script launched.
"""

import os
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "start.sh"

FAKE_OLLAMA = """#!/bin/sh
echo "$*" >> "$FAKE_DIR/ollama.log"
case "$1" in
    serve) touch "$FAKE_DIR/up" ;;
    show) [ -f "$FAKE_DIR/pulled" ] ;;
    pull) touch "$FAKE_DIR/pulled" ;;
esac
"""
FAKE_CURL = '#!/bin/sh\n[ -f "$FAKE_DIR/up" ]\n'
FAKE_CLI = '#!/bin/sh\necho "ARGS $*"\necho "ANTHROPIC ${ANTHROPIC_API_KEY:-none}"\n'


@pytest.fixture
def env(tmp_path):
    fake = tmp_path / "fake"
    bin_dir = fake / "bin"
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    bin_dir.mkdir(parents=True)
    _executable(bin_dir / "ollama", FAKE_OLLAMA)
    _executable(bin_dir / "curl", FAKE_CURL)
    _executable(venv / "bin" / "agent-runtime", FAKE_CLI)
    checksum = subprocess.run(
        ["cksum"], stdin=open(ROOT / "pyproject.toml"), capture_output=True, text=True, check=True
    ).stdout
    (venv / ".agent-runtime-installed").write_text(checksum)
    return {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_STATE_HOME": str(tmp_path / "state"),
        "AGENT_RUNTIME_VENV": str(venv),
        "FAKE_DIR": str(fake),
    }


def run(env, stdin="", *args):
    return subprocess.run(
        ["bash", str(SCRIPT), *args], input=stdin, env=env, capture_output=True, text=True, timeout=60
    )


def config(env) -> Path:
    return Path(env["XDG_CONFIG_HOME"]) / "agent-runtime" / "config"


def ollama_calls(env) -> list[str]:
    log = Path(env["FAKE_DIR"]) / "ollama.log"
    return log.read_text().splitlines() if log.exists() else []


def test_first_run_with_qwen_starts_ollama_downloads_the_model_and_opens_the_chat(env):
    result = run(env, "1\n")

    assert result.returncode == 0, result.stderr
    calls = ollama_calls(env)
    assert "serve" in calls
    pulls = [c for c in calls if c.startswith("pull ")]
    assert len(pulls) == 1 and pulls[0].startswith("pull qwen3.5:")
    model = pulls[0].split()[1]
    assert f"ARGS --provider ollama --model {model}" in result.stdout
    assert f"MODEL={model}" in config(env).read_text()
    assert stat.S_IMODE(config(env).stat().st_mode) == 0o600


def test_later_runs_skip_the_choice_and_the_download(env):
    run(env, "1\n")
    before = len(ollama_calls(env))

    result = run(env, "", "--resume", "latest")

    assert result.returncode == 0, result.stderr
    assert "Using qwen3.5:" in result.stdout
    assert not [c for c in ollama_calls(env)[before:] if c.startswith("pull ")]
    assert "--resume latest" in result.stdout


def test_ollama_already_running_is_not_started_again(env):
    (Path(env["FAKE_DIR"]) / "up").touch()
    run(env, "1\n")
    assert "serve" not in ollama_calls(env)


def test_anthropic_key_is_asked_for_saved_and_passed_to_the_chat(env):
    result = run(env, "2\nsk-test\n")

    assert result.returncode == 0, result.stderr
    assert "ARGS --provider anthropic --model claude-sonnet-5-5" in result.stdout
    assert "ANTHROPIC sk-test" in result.stdout
    assert "ANTHROPIC_API_KEY=sk-test" in config(env).read_text()
    assert ollama_calls(env) == []

    again = run(env)
    assert "ANTHROPIC sk-test" in again.stdout


def test_a_key_already_in_the_environment_is_used_and_not_saved(env):
    result = run({**env, "ANTHROPIC_API_KEY": "sk-env"}, "2\n")

    assert "ANTHROPIC sk-env" in result.stdout
    assert "sk-env" not in config(env).read_text()


def test_setup_flag_asks_again(env):
    run(env, "1\n")
    result = run(env, "3\nds-key\n", "--setup")

    assert "ARGS --provider deepseek --model deepseek-v4-pro" in result.stdout
    assert "PROVIDER=deepseek" in config(env).read_text()


def test_no_answer_and_no_config_fails_with_a_hint(env):
    result = run(env)

    assert result.returncode != 0
    assert "No model is set up yet" in result.stderr


def _executable(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(0o755)
