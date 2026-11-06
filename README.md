<div align="center">

# 🐚 Shellmate

**An AI assistant in your terminal that works on your files and searches the web.**<br>
Free and private on your own computer, or with Claude and DeepSeek.

[![tests](https://github.com/karthiknarayan01/shellmate/actions/workflows/tests.yml/badge.svg)](https://github.com/karthiknarayan01/shellmate/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)
![models](https://img.shields.io/badge/models-Qwen%20·%20Claude%20·%20DeepSeek-8A2BE2)
![web search](https://img.shields.io/badge/web%20search-free%2C%20no%20key-2EA44F)

<!-- ![Shellmate answering a question](docs/images/hero.png) -->

</div>

---

Shellmate is a small, crash-safe runtime for tool-using AI agents, written from scratch in Python. It runs an agent loop against Anthropic or DeepSeek models, or an open model on your own computer, and records every step in an append-only SQLite log, so:

| what you get | how |
|---|---|
| 💾 **A session survives a crash** | Kill the process at any point, start it again, and the interrupted turn finishes from where it stopped, without repeating a tool's side effect. |
| ✋ **Tools that change things ask first** | You approve, approve for the rest of the session, or deny (with optional feedback for the model). A denied tool never runs. |
| 🧠 **Long sessions keep their goal** | When the conversation nears the model's context window, older messages are summarized and your current request is kept word for word. |
| 🔒 **Runs on your own computer** | Qwen through Ollama: no key, no cost, nothing leaves your machine. |

It comes with a terminal chat, three workspace tools (`list_files`, `read_file`, `write_file`) confined to one directory, and two web tools (`web_search`, `read_web_page`) for news and anything current. Web search is free and needs no key or account.

> 📖 How it works inside: [docs/architecture.md](docs/architecture.md)

## 🚀 Quick start

macOS or Linux, with Python 3.11 or newer.

```bash
git clone https://github.com/karthiknarayan01/shellmate
cd shellmate
./start.sh
```

The first run installs everything into `.venv` and asks which model to use:

| | model | what you need |
|:-:|---|---|
| **1** | **Qwen on your computer**, through [Ollama](https://ollama.com) | Nothing: free, private, no key. The script installs Ollama if it's missing (it asks first), starts it, and downloads the model once: Qwen 3.5 9B (6.6 GB), or the 4B model (3.4 GB) on a machine with less than 12 GB of memory. |
| **2** | **Anthropic Claude** | your API key |
| **3** | **DeepSeek** | your API key |

Then the chat opens. Your choice is saved in `~/.config/agent-runtime/config`, readable only by you, so later runs go straight to the chat. A key already in `ANTHROPIC_API_KEY` or `DEEPSEEK_API_KEY` is used without being saved.

```bash
./start.sh                   # open the chat (the tools work in the directory you run it from)
./start.sh --setup           # choose the model again
./start.sh --resume latest   # any other options are passed to agent-runtime (see Usage)
```

> [!TIP]
> To use it from any directory, add an alias: `alias shellmate="$HOME/path/to/shellmate/start.sh"`

<details>
<summary><b>🔧 Manual setup</b> — if you'd rather not use the script</summary>

<br>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
export ANTHROPIC_API_KEY=...      # for --provider anthropic (the default)
export DEEPSEEK_API_KEY=...       # for --provider deepseek
agent-runtime
```

For a local model, install [Ollama](https://ollama.com/download), then:

```bash
ollama serve                   # not needed if the Ollama app is running
ollama pull qwen3.5:9b         # once
agent-runtime --provider ollama
```

Any other model you've pulled works with `--model`, for example `qwen3.5:4b` on an 8 GB machine or `qwen3.5:27b` with 32 GB or more. A local model is slower than a hosted one, and it only sees `AGENT_CONTEXT_WINDOW` tokens of the conversation (32,768 by default; a larger window takes more memory).

The database is created and migrated automatically on first run. To migrate it yourself, for example after upgrading:

```bash
agent-runtime-migrate agent.db
```

</details>

## 💬 In the chat

The agent's work is shown as it happens: a spinner while it waits, each tool call with a one-line result, and the answer rendered as markdown while it streams in.

<!-- ![Tool calls and a streamed answer](docs/images/tools.png) -->

Tools that change something ask first, in a panel that shows a diff of the change.

<!-- ![Approval panel with a diff](docs/images/approval.png) -->

A dim line after each turn shows the tokens used, how full the context window is, and the time taken.

| key or command | does |
|---|---|
| at an approval prompt | `y` approve · `a` always allow this tool for the session · `n` deny and stop the turn · any other text: deny and send it to the model as feedback |
| `/help` | commands and keys |
| `/new` | start a fresh session |
| `/status` | session id, model, messages, context use, tools |
| `/resume` | retry a turn that failed on a model API error |
| `/clear` | clear the screen |
| `/exit`, <kbd>Ctrl</kbd>+<kbd>D</kbd> | quit |
| <kbd>Ctrl</kbd>+<kbd>C</kbd> | cancel the running turn (at the prompt: quit) |
| <kbd>Alt</kbd>+<kbd>Enter</kbd>, or <kbd>Esc</kbd> then <kbd>Enter</kbd> | new line in a message |
| <kbd>↑</kbd> / <kbd>↓</kbd>, <kbd>Tab</kbd> | earlier messages (kept across runs) · complete a `/` command |

## ⌨️ Usage

```bash
agent-runtime                          # new session; the tools work in the current directory
agent-runtime --provider deepseek      # use DeepSeek instead of Anthropic
agent-runtime --provider ollama        # use a local model through Ollama
agent-runtime --provider deepseek --model deepseek-flash
                                       # pick a model (default: the provider's best, see below)
agent-runtime --workspace ~/project    # let the tools work in another directory
agent-runtime --resume latest          # continue the most recent session
agent-runtime --resume <session id>    # continue a specific session (the id is printed at start)
agent-runtime --yes                    # approve every tool call without asking
agent-runtime --no-web                 # don't give the agent web search
agent-runtime --db sessions.db         # keep sessions in another file (default agent.db)
```

Default models: `claude-sonnet-5-5` (Anthropic), `deepseek-v4-pro` (DeepSeek), `qwen3.5:9b` (Ollama).

## ⚙️ Configuration

| environment variable | default | meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | | key for `--provider anthropic` |
| `DEEPSEEK_API_KEY` | | key for `--provider deepseek` |
| `AGENT_MAX_OUTPUT_TOKENS` | `16000` | output token cap per model request |
| `AGENT_CONTEXT_WINDOW` | `128000` (`32768` for ollama) | context window used to decide when to compact; set it to your model's real window. For ollama it is also the window the model is run with |
| `OLLAMA_HOST` | `localhost:11434` | where Ollama is listening |
