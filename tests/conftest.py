import pytest

from runtime.migrate import migrate


@pytest.fixture
def db(tmp_path):
    """Path to a fresh, fully migrated SQLite database."""
    path = str(tmp_path / "agent.db")
    migrate(path)
    return path
