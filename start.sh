#!/usr/bin/env bash
# Set up agent-runtime and open the chat.
#
#   ./start.sh                 first run: install, choose a model, then chat; later runs: chat
#   ./start.sh --setup         choose the model again
#   ./start.sh --resume latest any other arguments are passed to agent-runtime
#
# First run:
#   1. creates .venv next to this script and installs the package (again whenever
#      pyproject.toml changes)
#   2. asks which model to use: Qwen on this computer (through Ollama), Anthropic or DeepSeek
#   3. local: installs Ollama if it's missing (after asking), starts it, downloads the model
#      hosted: asks for the API key unless it is already in the environment
#   4. saves the choice to ~/.config/agent-runtime/config (readable only by you)
#
# Later runs read that file and go straight to the chat. For a local model they only make sure
# Ollama is running and the model is there, which takes a moment.
#
# The chat's tools work in the directory you run this from.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${AGENT_RUNTIME_VENV:-$ROOT/.venv}"
CONFIG_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/agent-runtime"
CONFIG="$CONFIG_DIR/config"
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/agent-runtime"
OLLAMA_URL="${OLLAMA_HOST:-127.0.0.1:11434}"
[[ "$OLLAMA_URL" == http* ]] || OLLAMA_URL="http://$OLLAMA_URL"

PROVIDER=""
MODEL=""
ANTHROPIC_KEY=""
DEEPSEEK_KEY=""

main() {
    local setup=0 args=()
    for arg in "$@"; do
        if [[ "$arg" == "--setup" ]]; then setup=1; else args+=("$arg"); fi
    done

    install_package
    if [[ $setup -eq 0 ]] && load_config; then
        say "Using $MODEL ($PROVIDER). Run ./start.sh --setup to change it."
    else
        choose_model
        save_config
    fi
    if [[ "$PROVIDER" == "ollama" ]]; then
        ensure_ollama_running
        ensure_model_downloaded
    fi
    launch "${args[@]+"${args[@]}"}"
}

# --- 1. the package -------------------------------------------------------------------------

install_package() {
    # Reinstall only when pyproject.toml has changed since the last install.
    local stamp="$VENV/.agent-runtime-installed" want
    want="$(cksum < "$ROOT/pyproject.toml")"
    if [[ -x "$VENV/bin/agent-runtime" && -f "$stamp" && "$(cat "$stamp")" == "$want" ]]; then
        return
    fi
    if [[ ! -x "$VENV/bin/python" ]]; then
        local python
        python="$(find_python)" || fail "Python 3.11 or newer is needed. Install it (macOS: brew install python@3.13) and run this again."
        say "Creating a virtual environment with $("$python" --version)…"
        "$python" -m venv "$VENV"
    fi
    say "Installing agent-runtime…"
    mkdir -p "$STATE_DIR"
    local log="$STATE_DIR/install.log"
    if ! { "$VENV/bin/python" -m pip install --upgrade pip &&
        "$VENV/bin/python" -m pip install -e "$ROOT"; } > "$log" 2>&1; then
        tail -n 20 "$log" >&2
        fail "Installing failed; the full output is in $log"
    fi
    printf '%s\n' "$want" > "$stamp"
}

find_python() {
    # The first python on PATH that is 3.11 or newer.
    local candidate
    for candidate in python3.13 python3.12 python3.11 python3; do
        if command -v "$candidate" >/dev/null 2>&1 &&
            "$candidate" -c 'import sys; sys.exit(sys.version_info < (3, 11))' 2>/dev/null; then
            command -v "$candidate"
            return 0
        fi
    done
    return 1
}

# --- 2. the model choice and the saved config --------------------------------------------

load_config() {
    # Read the saved choice. The file is parsed as KEY=value lines, never executed.
    [[ -f "$CONFIG" ]] || return 1
    local key value
    while IFS='=' read -r key value; do
        case "$key" in
            PROVIDER) PROVIDER="$value" ;;
            MODEL) MODEL="$value" ;;
            ANTHROPIC_API_KEY) ANTHROPIC_KEY="$value" ;;
            DEEPSEEK_API_KEY) DEEPSEEK_KEY="$value" ;;
        esac
    done < "$CONFIG"
    case "$PROVIDER" in
        ollama) [[ -n "$MODEL" ]] ;;
        anthropic) [[ -n "$MODEL" && -n "${ANTHROPIC_API_KEY:-$ANTHROPIC_KEY}" ]] ;;
        deepseek) [[ -n "$MODEL" && -n "${DEEPSEEK_API_KEY:-$DEEPSEEK_KEY}" ]] ;;
        *) return 1 ;;
    esac
}

choose_model() {
    local qwen
    qwen="$(qwen_for_this_machine)"
    echo
    echo "Which model should the agent use?"
    echo "  1) Qwen on this computer ($qwen through Ollama): free, private, no key"
    echo "  2) Anthropic Claude (needs an API key)"
    echo "  3) DeepSeek (needs an API key)"
    local answer
    while true; do
        read -r -p "Choose 1, 2 or 3 [1]: " answer || fail "No model is set up yet. Run ./start.sh in a terminal to choose one."
        case "${answer:-1}" in
            1) PROVIDER=ollama MODEL="$qwen"; return ;;
            2) PROVIDER=anthropic MODEL=claude-sonnet-5-5; ask_key ANTHROPIC_API_KEY ANTHROPIC_KEY; return ;;
            3) PROVIDER=deepseek MODEL=deepseek-v4-pro; ask_key DEEPSEEK_API_KEY DEEPSEEK_KEY; return ;;
        esac
    done
}

qwen_for_this_machine() {
    # qwen3.5:9b needs about 8 GB with its context; below 12 GB of memory, the 4B model.
    local bytes=0
    if [[ "$(uname)" == "Darwin" ]]; then
        bytes="$(sysctl -n hw.memsize 2>/dev/null || echo 0)"
    elif [[ -r /proc/meminfo ]]; then
        bytes="$(awk '/MemTotal/ {print $2 * 1024}' /proc/meminfo)"
    fi
    if [[ "$bytes" -gt 0 && "$bytes" -lt $((12 * 1024 * 1024 * 1024)) ]]; then
        echo "qwen3.5:4b"
    else
        echo "qwen3.5:9b"
    fi
}

ask_key() {
    # $1: the environment variable the CLI reads; $2: the variable to keep it in here.
    local env_name="$1" target="$2" key
    if [[ -n "${!env_name:-}" ]]; then
        say "Using $env_name from the environment."
        return
    fi
    while true; do
        read -r -s -p "Paste your $env_name (input is hidden): " key || fail "No $env_name given."
        echo
        [[ -n "$key" ]] && break
    done
    printf -v "$target" '%s' "$key"
}

save_config() {
    mkdir -p "$CONFIG_DIR"
    (
        umask 077
        {
            echo "PROVIDER=$PROVIDER"
            echo "MODEL=$MODEL"
            [[ -n "$ANTHROPIC_KEY" ]] && echo "ANTHROPIC_API_KEY=$ANTHROPIC_KEY"
            [[ -n "$DEEPSEEK_KEY" ]] && echo "DEEPSEEK_API_KEY=$DEEPSEEK_KEY"
            true
        } > "$CONFIG"
    )
    chmod 600 "$CONFIG"
    say "Saved to $CONFIG."
}

# --- 3. Ollama ------------------------------------------------------------------------------

ensure_ollama_running() {
    command -v ollama >/dev/null 2>&1 || install_ollama
    ollama_up && return
    say "Starting Ollama…"
    if [[ "$(uname)" == "Darwin" && -d /Applications/Ollama.app ]]; then
        open -a Ollama
    else
        mkdir -p "$STATE_DIR"
        nohup ollama serve >> "$STATE_DIR/ollama.log" 2>&1 &
    fi
    local waited=0
    until ollama_up; do
        sleep 1
        waited=$((waited + 1))
        [[ $waited -lt 30 ]] || fail "Ollama didn't start within 30 seconds; see $STATE_DIR/ollama.log"
    done
}

ollama_up() {
    curl -fsS --max-time 2 "$OLLAMA_URL/api/version" >/dev/null 2>&1
}

install_ollama() {
    local answer
    read -r -p "Ollama runs the model on this computer and isn't installed. Install it now? [Y/n] " answer ||
        fail "Ollama isn't installed. Get it from https://ollama.com/download and run this again."
    [[ "${answer:-y}" =~ ^[Yy] ]] || fail "Install Ollama from https://ollama.com/download, or run ./start.sh --setup to pick a hosted model."
    if [[ "$(uname)" == "Darwin" ]]; then
        command -v brew >/dev/null 2>&1 ||
            fail "Download the Ollama app from https://ollama.com/download, open it once, then run this again."
        brew install ollama
    else
        curl -fsSL https://ollama.com/install.sh | sh
    fi
}

ensure_model_downloaded() {
    ollama show "$MODEL" >/dev/null 2>&1 && return
    say "Downloading $MODEL (once; several GB, so this can take a while)…"
    ollama pull "$MODEL"
}

# --- 4. the chat ----------------------------------------------------------------------------

launch() {
    [[ -n "$ANTHROPIC_KEY" && -z "${ANTHROPIC_API_KEY:-}" ]] && export ANTHROPIC_API_KEY="$ANTHROPIC_KEY"
    [[ -n "$DEEPSEEK_KEY" && -z "${DEEPSEEK_API_KEY:-}" ]] && export DEEPSEEK_API_KEY="$DEEPSEEK_KEY"
    exec "$VENV/bin/agent-runtime" --provider "$PROVIDER" --model "$MODEL" "$@"
}

say() { printf '\033[36m›\033[0m %s\n' "$*"; }
fail() { printf '\033[31m✗\033[0m %s\n' "$*" >&2; exit 1; }

main "$@"
