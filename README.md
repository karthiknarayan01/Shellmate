# agent-runtime

A small, crash-safe agent runtime written from scratch in Python (no agent framework).

Goals:
- every conversation is an append-only event log in SQLite; all state is rebuilt from it
- kill the process mid-turn, restart, and the turn continues without repeating any tool side effect
- tools that change things wait for approval; denied tools never run
- context compaction so long sessions keep their goal

Built one piece per PR. See the PR history for the design of each piece.
