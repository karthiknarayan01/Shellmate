# agent-runtime

A small, crash-safe agent runtime written from scratch in Python (no agent framework).

- every conversation is an append-only event log in SQLite; all state is rebuilt from it
- kill the process mid-turn, restart, and the turn continues without repeating any tool side effect
- tools that change things wait for approval; denied tools never run
- context compaction so long sessions keep their goal

## Install

```
pip install -e ".[dev]"
agent-runtime-migrate agent.db      # create or upgrade the database; run after every upgrade
```

## Test

```
python -m pytest -q
```
