# agent-runtime

A small, crash-safe runtime for tool-using AI agents, written from scratch in Python. It runs an agent loop against Anthropic or DeepSeek models, or an open model on your own computer, and records every step in an append-only SQLite log, so:

- **a session survives a crash.** Kill the process at any point, start it again, and the interrupted turn finishes from where it stopped, without repeating a tool's side effect.
- **tools that change things ask first.** You approve, approve for the rest of the session, or deny (with optional feedback for the model). A denied tool never runs.
- **long sessions keep their goal.** When the conversation nears the model's context window, older messages are summarized and your current request is kept word for word.

It comes with a terminal chat, three workspace tools (`list_files`, `read_file`, `write_file`) confined to one directory, and two web tools (`web_search`, `read_web_page`) for news and anything current. Web search is free and needs no key or account. How it works inside: [docs/architecture.md](docs/architecture.md).

## Quick start

macOS or Linux, with Python 3.11 or newer.

```
git clone https://github.com/karthiknarayan01/agent-runtime
cd agent-runtime
./start.sh
```

The first run installs everything into `.venv` and asks which model to use:

1. **Qwen on your computer**, through [Ollama](https://ollama.com): free, private, no key. The script installs Ollama if it's missing (it asks first), starts it, and downloads the model once: Qwen 3.5 9B (6.6 GB), or the 4B model (3.4 GB) on a machine with less than 12 GB of memory.
2. **Anthropic Claude**, with your API key.
3. **DeepSeek**, with your API key.

Then the chat opens. Your choice is saved in `~/.config/agent-runtime/config`, readable only by you, so later runs go straight to the chat. A key already in `ANTHROPIC_API_KEY` or `DEEPSEEK_API_KEY` is used without being saved.

```
./start.sh                       open the chat (the tools work in the directory you run it from)
./start.sh --setup               choose the model again
./start.sh --resume latest       any other options are passed to agent-runtime (see Usage)
```

To use it from any directory, add an alias: `alias agent="$HOME/path/to/agent-runtime/start.sh"`.

## Manual setup

If you'd rather not use the script:

```
python -m venv .venv && source .venv/bin/activate
pip install -e .
export ANTHROPIC_API_KEY=...      # for --provider anthropic (the default)
export DEEPSEEK_API_KEY=...       # for --provider deepseek
agent-runtime
```

For a local model, install [Ollama](https://ollama.com/download), then:

```
ollama serve                   # not needed if the Ollama app is running
ollama pull qwen3.5:9b         # once
agent-runtime --provider ollama
```

Any other model you've pulled works with `--model`, for example `qwen3.5:4b` on an 8 GB machine or `qwen3.5:27b` with 32 GB or more. A local model is slower than a hosted one, and it only sees `AGENT_CONTEXT_WINDOW` tokens of the conversation (32,768 by default; a larger window takes more memory).

The database is created and migrated automatically on first run. To migrate it yourself, for example after upgrading:

```
agent-runtime-migrate agent.db
```

## Usage

```
agent-runtime                          new session; the tools work in the current directory
agent-runtime --provider deepseek      use DeepSeek instead of Anthropic
agent-runtime --provider ollama        use a local model through Ollama
agent-runtime --provider deepseek --model deepseek-flash
                                       pick a model (defaults: claude-sonnet-5-5, deepseek-v4-pro, qwen3.5:9b)
agent-runtime --workspace ~/project    let the tools work in another directory
agent-runtime --resume latest          continue the most recent session
agent-runtime --resume <session id>    continue a specific session (the id is printed at start)
agent-runtime --yes                    approve every tool call without asking
agent-runtime --no-web                 don't give the agent web search
agent-runtime --db sessions.db         keep sessions in another file (default agent.db)
```

In the chat, the agent's work is shown as it happens: a spinner while it waits, each tool call with a one-line result, and the answer rendered as markdown while it streams in. Tools that change something ask first, in a panel that shows a diff of the change. A dim line after each turn shows the tokens used, how full the context window is, and the time taken.

| | |
|---|---|
| at an approval prompt | `y` approve · `a` always allow this tool for the session · `n` deny and stop the turn · any other text: deny and send it to the model as feedback |
| `/help` | commands and keys |
| `/new` | start a fresh session |
| `/status` | session id, model, messages, context use, tools |
| `/resume` | retry a turn that failed on a model API error |
| `/clear` | clear the screen |
| `/exit`, Ctrl+D | quit |
| Ctrl+C | cancel the running turn (at the prompt: quit) |
| Alt+Enter, or Esc then Enter | new line in a message |
| ↑ / ↓, Tab | earlier messages (kept across runs) · complete a `/` command |

## Configuration

| environment variable | default | meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | | key for `--provider anthropic` |
| `DEEPSEEK_API_KEY` | | key for `--provider deepseek` |
| `AGENT_MAX_OUTPUT_TOKENS` | 16000 | output token cap per model request |
| `AGENT_CONTEXT_WINDOW` | 128000 (32768 for ollama) | context window used to decide when to compact; set it to your model's real window. For ollama it is also the window the model is run with |
| `OLLAMA_HOST` | `localhost:11434` | where Ollama is listening |
