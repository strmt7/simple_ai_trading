from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from simple_ai_trading.polymarket_historical_screen import (
    HistoricalScreenStore,
    load_historical_screen_contract,
)


ROOT = Path(__file__).parents[1]
TOOL_PATH = ROOT / "tools" / "run_polymarket_historical_screen.py"
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-015-btc-5m-historical-screen-v1.json"
)


def _tool() -> object:
    spec = importlib.util.spec_from_file_location(
        "run_polymarket_historical_screen_tool",
        TOOL_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("historical screen tool spec is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_historical_screen_status_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "historical.duckdb"
    contract = load_historical_screen_contract(CONTRACT_PATH)
    with HistoricalScreenStore(database, contract=contract):
        pass
    modified_before = database.stat().st_mtime_ns
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TOOL_PATH),
            "status",
            "--contract",
            str(CONTRACT_PATH),
            "--database",
            str(database),
        ],
    )

    assert _tool().main() == 0

    assert '"event":"historical_screen_status"' in capsys.readouterr().out
    assert database.stat().st_mtime_ns == modified_before
