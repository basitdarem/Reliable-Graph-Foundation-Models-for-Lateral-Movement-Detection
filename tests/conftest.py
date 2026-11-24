"""Shared pytest fixtures."""
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def redteam_fixture_path() -> Path:
    """The real LANL redteam.txt.gz (749 rows), used so tests exercise
    the actual public label file's shape, not just a hand-built stand-in."""
    path = FIXTURES_DIR / "redteam.txt.gz"
    assert path.exists(), f"missing test fixture: {path}"
    return path
