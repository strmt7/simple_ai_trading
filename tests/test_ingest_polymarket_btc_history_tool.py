from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

from simple_ai_trading.polymarket_btc_history import (
    POLYMARKET_BTC_HISTORY_RESEARCH_ROUND,
    POLYMARKET_BTC_HISTORY_SYMBOLS,
)
from simple_ai_trading.spot_perpetual_corpus import SpotPerpetualCorpusStore


ROOT = Path(__file__).parents[1]
TOOL_PATH = ROOT / "tools" / "ingest_polymarket_btc_history.py"
INVENTORY_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-015-btc-5m-full-history-inventory-v1.json"
)


def _tool() -> object:
    spec = importlib.util.spec_from_file_location(
        "ingest_polymarket_btc_history_tool",
        TOOL_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("BTC history tool spec is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_history_status_opens_existing_database_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    database = tmp_path / "history.duckdb"
    cache = tmp_path / "cache"
    with SpotPerpetualCorpusStore(
        database,
        cache_root=cache,
        memory_limit="256MB",
        threads=1,
        symbols=POLYMARKET_BTC_HISTORY_SYMBOLS,
        research_round=POLYMARKET_BTC_HISTORY_RESEARCH_ROUND,
    ):
        pass
    modified_before = database.stat().st_mtime_ns
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TOOL_PATH),
            "status",
            "--inventory",
            str(INVENTORY_PATH),
            "--database",
            str(database),
            "--cache-root",
            str(cache),
            "--memory-limit",
            "256MB",
            "--threads",
            "1",
        ],
    )

    assert _tool().main() == 0

    assert '"event":"history_status"' in capsys.readouterr().out
    assert database.stat().st_mtime_ns == modified_before
