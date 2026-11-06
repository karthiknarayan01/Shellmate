# agent-runtime

A small, crash-safe runtime for tool-using AI agents, written from scratch in Python. It runs an agent loop against Anthropic or DeepSeek models and records every step in an append-only SQLite log, so:

- **a session survives a crash.** Kill the process at any point, start it again, and the interrupted turn finishes from where it stopped, without repeating a tool's side effect.
- **tools that change things ask first.** You approve, approve for the rest of the session, or deny (with optional feedback for the model). A denied tool never runs.
- **long sessions keep their goal.** When the conversation nears the model's context window, older messages are summarized and your current request is kept word for word.

It comes with a terminal chat and three workspace tools (`list_files`, `read_file`, `write_file`), all confined to one directory. How it works inside: [docs/architecture.md](docs/architecture.md).

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

The database is created and migrated automatically on first run. To migrate it yourself, for example after upgrading:

```
agent-runtime-migrate agent.db
```

## Usage

```
agent-runtime                          new session; the tools work in the current directory
agent-runtime --provider deepseek      use DeepSeek instead of Anthropic
agent-runtime --provider deepseek --model deepseek-flash
                                       pick a model (defaults: claude-sonnet-5-5, deepseek-v4-pro)
agent-runtime --workspace ~/project    let the tools work in another directory
agent-runtime --resume latest          continue the most recent session
agent-runtime --resume <session id>    continue a specific session (the id is printed at start)
agent-runtime --yes                    approve every tool call without asking
agent-runtime --db sessions.db         keep sessions in another file (default agent.db)
```

In the chat:

| | |
|---|---|
| at an approval prompt | `y` approve · `a` approve this tool for the session · `n` deny and stop the turn · any other text: deny and send it to the model as feedback |
| Ctrl+C during a turn | cancel that turn |
| `/resume` | retry a turn that failed on a model API error |
| `/exit`, Ctrl+D | quit |

## Configuration

| environment variable | default | meaning |
|---|---|---|
| `ANTHROPIC_API_KEY` | | key for `--provider anthropic` |
| `DEEPSEEK_API_KEY` | | key for `--provider deepseek` |
| `AGENT_MAX_OUTPUT_TOKENS` | 16000 | output token cap per model request |
| `AGENT_CONTEXT_WINDOW` | 128000 | context window used to decide when to compact; set it to your model's real window |
