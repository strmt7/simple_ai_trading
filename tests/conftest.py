from __future__ import annotations

import sys
from decimal import Context, localcontext
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def isolate_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture(autouse=True)
def isolate_decimal_context():
    """Keep legacy research imports from changing another test's arithmetic."""
    with localcontext(Context()):
        yield
