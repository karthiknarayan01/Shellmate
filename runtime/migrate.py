"""Database migrations: bring a SQLite file up to the schema this version of the app needs.

Each migration is a numbered SQL file in runtime/migrations/ (0001_*.sql, 0002_*.sql, ...).
Applied versions are recorded in a `schema_migrations` table. `migrate()` applies every
file whose number is not recorded yet, in order, each one in its own transaction together
with its `schema_migrations` row, so a failing migration leaves no trace.

Run it before first use and after every upgrade:

    python -m runtime.migrate path/to/agent.db
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point: migrate the given database file and report what ran."""
    parser = argparse.ArgumentParser(description="Apply pending database migrations.")
    parser.add_argument("db_path", help="SQLite database file (created if missing)")
    args = parser.parse_args(argv)

    applied = migrate(args.db_path)
    if applied:
        for version in applied:
            print(f"applied {version}")
    else:
        print("database is up to date")
    return 0


def migrate(db_path: str, migrations_dir: Path = MIGRATIONS_DIR) -> list[str]:
    """Apply all pending migrations to `db_path`. Returns the names of the ones applied."""
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL)"
        )
        done = applied_versions(conn)
        pending = [m for m in discover(migrations_dir) if m.version not in done]
        for migration in pending:
            apply(conn, migration)
        return [m.name for m in pending]
    finally:
        conn.close()


def applied_versions(conn: sqlite3.Connection) -> set[int]:
    """Version numbers already recorded in schema_migrations."""
    return {v for (v,) in conn.execute("SELECT version FROM schema_migrations")}


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    path: Path


def discover(migrations_dir: Path) -> list[Migration]:
    """All NNNN_name.sql files in the directory, sorted by version. Duplicate numbers are an error."""
    found: dict[int, Migration] = {}
    for path in migrations_dir.glob("*.sql"):
        match = re.fullmatch(r"(\d{4})_[a-z0-9_]+\.sql", path.name)
        if not match:
            raise ValueError(f"bad migration file name: {path.name}")
        version = int(match.group(1))
        if version in found:
            raise ValueError(f"two migrations share version {version:04d}")
        found[version] = Migration(version, path.stem, path)
    return [found[v] for v in sorted(found)]


def apply(conn: sqlite3.Connection, migration: Migration) -> None:
    """Run one migration and record it, atomically.

    The SQL and the schema_migrations insert run as one script between BEGIN and COMMIT,
    so either both take effect or, on any error, neither does.
    """
    sql = migration.path.read_text(encoding="utf-8")
    record = (
        "INSERT INTO schema_migrations (version, name, applied_at) "
        f"VALUES ({migration.version}, '{migration.name}', '{_now()}');"
    )
    try:
        conn.executescript(f"BEGIN;\n{sql}\n{record}\nCOMMIT;")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def latest_version(migrations_dir: Path = MIGRATIONS_DIR) -> int:
    """Highest migration number shipped with the code (what a database must be at)."""
    migrations = discover(migrations_dir)
    return migrations[-1].version if migrations else 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
