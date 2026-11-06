# agent-runtime

A small, crash-safe runtime for tool-using AI agents, written from scratch in Python. It runs an agent loop against Anthropic or DeepSeek models, or an open model on your own computer, and records every step in an append-only SQLite log, so:

- **a session survives a crash.** Kill the process at any point, start it again, and the interrupted turn finishes from where it stopped, without repeating a tool's side effect.
- **tools that change things ask first.** You approve, approve for the rest of the session, or deny (with optional feedback for the model). A denied tool never runs.
- **long sessions keep their goal.** When the conversation nears the model's context window, older messages are summarized and your current request is kept word for word.

It comes with a terminal chat, three workspace tools (`list_files`, `read_file`, `write_file`) confined to one directory, and two web tools (`web_search`, `read_web_page`) for news and anything current. Web search is free and needs no key or account. How it works inside: [docs/architecture.md](docs/architecture.md).

## Setup

Python 3.11 or newer.

```
git clone https://github.com/karthiknarayan01/agent-runtime
cd agent-runtime
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Set the API key for the provider you'll use:

```
export ANTHROPIC_API_KEY=...      # Anthropic (default)
export DEEPSEEK_API_KEY=...       # DeepSeek
```

Or skip the key and run a model locally (see below).

The database is created and migrated automatically on first run. To migrate it yourself, for example after upgrading:

```
agent-runtime-migrate agent.db
```

## Run a model on your computer

[Ollama](https://ollama.com) runs open models locally: no key, no cost, nothing leaves your machine. The default is Qwen 3.5 9B, which handles chat and tool use well: a 6.6 GB download that, with its context, runs comfortably on a Mac with 16 GB or more.

```
brew install ollama            # or download the app from ollama.com
ollama serve                   # not needed if the Ollama app is running
ollama pull qwen3.5:9b         # once, about 6.6 GB
agent-runtime --provider ollama
```

Any other model you've pulled works with `--model`, for example `qwen3.5:4b` on an 8 GB machine or `qwen3.5:27b` with 32 GB or more. A local model is slower than a hosted one, and it only sees `AGENT_CONTEXT_WINDOW` tokens of the conversation (32,768 by default; a larger window takes more memory).

## Usage

```
agent-runtime                          new session; the tools work in the current directory
agent-runtime --provider deepseek      use DeepSeek instead of Anthropic
agent-runtime --provider ollama        use a local model (see above)
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
