"""Migrations: apply in order, once each, atomically; refuse malformed migration sets."""

import sqlite3

import pytest

from runtime.migrate import MIGRATIONS_DIR, discover, latest_version, main, migrate


def test_fresh_database_gets_every_migration(tmp_path):
    db = str(tmp_path / "agent.db")
    applied = migrate(db)
    assert applied == [m.name for m in discover_shipped()]
    assert _tables(db) >= {"schema_migrations", "sessions", "events"}


def test_second_run_is_a_no_op(tmp_path):
    db = str(tmp_path / "agent.db")
    migrate(db)
    assert migrate(db) == []


def test_only_new_migrations_run_after_an_upgrade(tmp_path):
    mdir = tmp_path / "m"
    mdir.mkdir()
    (mdir / "0001_a.sql").write_text("CREATE TABLE a (x INTEGER);")
    db = str(tmp_path / "agent.db")
    assert migrate(db, mdir) == ["0001_a"]

    (mdir / "0002_b.sql").write_text("ALTER TABLE a ADD COLUMN y TEXT;")
    assert migrate(db, mdir) == ["0002_b"]
    assert _columns(db, "a") == ["x", "y"]


def test_failing_migration_leaves_no_trace(tmp_path):
    mdir = tmp_path / "m"
    mdir.mkdir()
    (mdir / "0001_ok.sql").write_text("CREATE TABLE ok (x INTEGER);")
    (mdir / "0002_broken.sql").write_text("CREATE TABLE half (x INTEGER);\nNOT VALID SQL;")
    db = str(tmp_path / "agent.db")

    with pytest.raises(sqlite3.Error):
        migrate(db, mdir)

    assert "ok" in _tables(db)
    assert "half" not in _tables(db)
    assert _versions(db) == [1]


def test_bad_file_names_and_duplicate_versions_are_rejected(tmp_path):
    (tmp_path / "1_short.sql").write_text("")
    with pytest.raises(ValueError, match="bad migration file name"):
        discover(tmp_path)

    (tmp_path / "1_short.sql").unlink()
    (tmp_path / "0001_a.sql").write_text("")
    (tmp_path / "0001_b.sql").write_text("")
    with pytest.raises(ValueError, match="share version"):
        discover(tmp_path)


def test_command_line_reports_what_ran(tmp_path, capsys):
    db = str(tmp_path / "agent.db")
    assert main([db]) == 0
    assert "applied 0001_create_sessions_and_events" in capsys.readouterr().out
    main([db])
    assert "up to date" in capsys.readouterr().out


def test_latest_version_matches_shipped_files():
    assert latest_version() == discover_shipped()[-1].version


# ── helpers ──────────────────────────────────────────────────────────────────────────────


def discover_shipped():
    return discover(MIGRATIONS_DIR)


def _tables(db):
    with sqlite3.connect(db) as conn:
        return {n for (n,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _columns(db, table):
    with sqlite3.connect(db) as conn:
        return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def _versions(db):
    with sqlite3.connect(db) as conn:
        return [v for (v,) in conn.execute("SELECT version FROM schema_migrations ORDER BY version")]
