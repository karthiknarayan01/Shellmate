"""Append-only event log: the single source of truth for every session.

Every state change in the runtime (a user message, a model reply, a tool starting,
a tool finishing, an approval decision, a compaction) is written here as one event
BEFORE the runtime acts on it. Nothing is ever updated or deleted, and all other state
is rebuilt by replaying a session's events. If `append()` returned, the event is on disk,
so a restarted process sees exactly what the dead one saw.

Tables (created by runtime/migrations/0001_create_sessions_and_events.sql):

    sessions(session_id PK, created_at)
    events(seq INTEGER PK, session_id FK, type, payload JSON, ts)

`seq` is assigned by SQLite on insert and only ever increases, so ordering a session's
events by seq gives the order they were appended in.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from runtime.migrate import latest_version


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


class SchemaOutOfDate(RuntimeError):
    """Raised when the database has not been migrated to the version this code needs."""


class EventLog:
    def __init__(self, db_path: str):
        """Open the log file and check it has been migrated.

        Connection settings, all chosen for durability:
          isolation_level=None  autocommit: each INSERT is its own committed transaction
          journal_mode=WAL      writers append to a write-ahead log; readers never block
          synchronous=FULL      fsync the WAL on every commit, so a committed event
                                survives power loss, not only a process crash
          foreign_keys=ON       an event for a session that doesn't exist is rejected
                                by the database itself, with no extra lookup
        """
        self._conn = sqlite3.connect(db_path, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._check_schema(db_path)

    def _check_schema(self, db_path: str) -> None:
        """Fail fast if migrations haven't run, instead of failing later on a missing table."""
        try:
            (version,) = self._conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
            ).fetchone()
        except sqlite3.OperationalError:
            version = 0
        if version < latest_version():
            self._conn.close()
            raise SchemaOutOfDate(
                f"{db_path} is at schema version {version}, need {latest_version()}; "
                f"run: python -m runtime.migrate {db_path}"
            )

    def new_session(self) -> str:
        """Create an empty session and return its id (a random 32-char hex uuid)."""
        session_id = uuid.uuid4().hex
        self._conn.execute(
            "INSERT INTO sessions (session_id, created_at) VALUES (?, ?)", (session_id, _now())
        )
        return session_id

    def append(self, session_id: str, type: str, payload: dict[str, Any]) -> Event:
        """Write one event and return it. The event is durable when this returns.

        One INSERT in autocommit mode is atomic and committed (and fsynced) before
        execute() returns. SQLite assigns seq. The foreign key rejects unknown sessions.
        """
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        ts = _now()
        try:
            cursor = self._conn.execute(
                "INSERT INTO events (session_id, type, payload, ts) VALUES (?, ?, ?, ?)",
                (session_id, type, body, ts),
            )
        except sqlite3.IntegrityError as e:
            raise UnknownSession(session_id) from e
        return Event(session_id, cursor.lastrowid, type, payload, ts)

    def load(self, session_id: str, after_seq: int = 0) -> list[Event]:
        """Return a session's events in seq order, optionally only those after `after_seq`.

        This is what a restarted process calls to rebuild its state. The sessions table is
        only consulted when no events come back, to tell "empty" apart from "unknown".
        """
        rows = self._conn.execute(
            "SELECT seq, type, payload, ts FROM events "
            "WHERE session_id = ? AND seq > ? ORDER BY seq",
            (session_id, after_seq),
        ).fetchall()
        if not rows and not self.has_session(session_id):
            raise UnknownSession(session_id)
        return [Event(session_id, seq, t, json.loads(p), ts) for seq, t, p, ts in rows]

    def has_session(self, session_id: str) -> bool:
        """True if `new_session` created this id."""
        row = self._conn.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return row is not None

    def latest_session(self) -> str | None:
        """The most recently created session id, or None if there are none."""
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


def _now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
