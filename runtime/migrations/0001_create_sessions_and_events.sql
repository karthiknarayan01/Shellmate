-- Sessions and their append-only event log.
-- seq is a rowid alias: SQLite assigns it on insert, globally increasing, never reused
-- because rows are never deleted. Events of one session are read in seq order.

CREATE TABLE sessions (
    session_id  TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL
);

CREATE TABLE events (
    seq         INTEGER PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(session_id),
    type        TEXT NOT NULL,
    payload     TEXT NOT NULL,
    ts          TEXT NOT NULL
);

CREATE INDEX events_by_session ON events (session_id, seq);
