"""Append-only event log: the single source of truth for every session.

Every state change in the runtime (a user message, a model reply, a tool starting,
a tool finishing, an approval decision, a compaction) is written here as one event
BEFORE the runtime acts on it. Nothing is ever updated or deleted. Everything else,
including the message list sent to the model, is rebuilt by replaying these events
(see state.py). That is what makes crash-resume possible: if `append()` returned,
the event is on disk, and a restarted process sees exactly what the dead one saw.

Storage is one SQLite file with two tables (schema at the bottom of this file):

    sessions(session_id PK, created_at)
    events(session_id, seq, type, payload JSON, ts, PK(session_id, seq))

`seq` is a per-session counter (1, 2, 3, ...) assigned inside the same write
transaction as the insert, so two writers can never get the same number.

Reading order: EventLog.__init__ -> new_session -> append -> load -> latest_session.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class Event:
    """One row of the log, with its payload already decoded from JSON."""

    session_id: str
    seq: int
    type: str
    payload: dict[str, Any]
    ts: str


class UnknownSession(KeyError):
    """Raised when appending to or loading a session that was never created."""


class EventLog:
    def __init__(self, db_path: str):
        """Open (or create) the log file and make sure the tables exist.

        Connection settings, all chosen for durability:
          isolation_level=None  autocommit mode; every transaction is an explicit
                                BEGIN ... COMMIT that this class controls
          journal_mode=WAL      writers append to a write-ahead log; readers never block
          synchronous=FULL      fsync the WAL on every commit, so a committed event
                                survives power loss, not only a process crash
        """
        self._conn = sqlite3.connect(db_path, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)

    def new_session(self) -> str:
        """Create an empty session and return its id (a random 32-char hex uuid)."""
        session_id = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO sessions (session_id, created_at) VALUES (?, ?)", (session_id, _now())
        )
        return session_id

    def append(self, session_id: str, type: str, payload: dict[str, Any]) -> Event:
        """Write one event and return it. The event is durable when this returns.

        BEGIN IMMEDIATE takes SQLite's write lock BEFORE reading MAX(seq), so reading
        the next seq and inserting it happen with no other writer in between. Two
        processes appending to the same session get distinct, gap-free seqs.
        Any failure rolls the transaction back, so a half-written event never exists.
        """
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        ts = _now()
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            if not self.has_session(session_id):
                raise UnknownSession(session_id)
            (seq,) = self._conn.execute(
                "SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            self._conn.execute(
                "INSERT INTO events (session_id, seq, type, payload, ts) VALUES (?, ?, ?, ?, ?)",
                (session_id, seq, type, body, ts),
            )
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        return Event(session_id, seq, type, payload, ts)

    def has_session(self, session_id: str) -> bool:
        """True if `new_session` created this id (used by append and load)."""
        row = self._conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row is not None

    def load(self, session_id: str, after_seq: int = 0) -> list[Event]:
        """Return a session's events in seq order, optionally only those after `after_seq`.

        This is what a restarted process calls to rebuild its state.
        """
        if not self.has_session(session_id):
            raise UnknownSession(session_id)
        rows = self._conn.execute(
            "SELECT seq, type, payload, ts FROM events "
            "WHERE session_id = ? AND seq > ? ORDER BY seq",
            (session_id, after_seq),
        ).fetchall()
        return [Event(session_id, seq, t, json.loads(p), ts) for seq, t, p, ts in rows]

    def latest_session(self) -> str | None:
        """The most recently created session id, or None (used by `--resume latest`)."""
        row = self._conn.execute(
            "SELECT session_id FROM sessions ORDER BY created_at DESC, rowid DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else None

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> EventLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# ── helpers used above ──────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    session_id  TEXT    NOT NULL REFERENCES sessions(session_id),
    seq         INTEGER NOT NULL,
    type        TEXT    NOT NULL,
    payload     TEXT    NOT NULL,
    ts          TEXT    NOT NULL,
    PRIMARY KEY (session_id, seq)
);
"""


def _now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
