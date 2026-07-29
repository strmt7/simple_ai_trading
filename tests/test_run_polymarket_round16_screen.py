from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from simple_ai_trading.polymarket_historical_screen import (
    HistoricalScreenStore,
)
from simple_ai_trading.polymarket_round16 import (
    load_round16_historical_contract,
)


ROOT = Path(__file__).parents[1]
TOOL_PATH = ROOT / "tools" / "run_polymarket_round16_screen.py"
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-016-btc-15m-horizon-comparison-v1.json"
)


def _tool() -> object:
    spec = importlib.util.spec_from_file_location(
        "run_polymarket_round16_screen",
        TOOL_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Round 16 tool spec is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_round16_status_is_machine_readable_and_target_blind(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tool = _tool()
    database = tmp_path / "round16.duckdb"
    contract = load_round16_historical_contract(CONTRACT_PATH)
    with HistoricalScreenStore(
        database,
        contract=contract.historical,
    ):
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

    assert tool.main() == 0

    lines = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert lines == [
        {
            "contract_sha256": (
                "ba4a0fdd8b24baaa82fd1b4c6085b1a4907e910f4ebcefac00f2e92fb95d1996"
            ),
            "database_bytes": lines[0]["database_bytes"],
            "development_target_count": 0,
            "evaluation_manifest_count": 0,
            "event": "round16_status",
            "feature_row_count": 0,
            "market_count": 0,
            "pretest_manifest_count": 0,
            "state": "initialized",
            "test_target_count": 0,
        }
    ]
    assert lines[0]["database_bytes"] > 0
    assert database.stat().st_mtime_ns == modified_before


def test_round16_test_access_requires_explicit_acknowledgement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = _tool()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TOOL_PATH),
            "test-targets",
            "--contract",
            str(CONTRACT_PATH),
            "--database",
            str(tmp_path / "round16.duckdb"),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        tool.main()

    assert not (tmp_path / "round16.duckdb").exists()
