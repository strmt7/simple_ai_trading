from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from simple_ai_trading.polymarket_round22_pilot import (
    Round22PilotStore,
    load_round22_pilot_contract,
)


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "run_polymarket_round22_pilot_ingestion.py"


def _tool() -> object:
    spec = importlib.util.spec_from_file_location(
        "run_polymarket_round22_pilot_ingestion",
        TOOL_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Round 22 ingestion tool spec is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_round22_ingestion_status_is_read_only_and_machine_readable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    contract = load_round22_pilot_contract(ROOT)
    database = tmp_path / "round22.duckdb"
    with Round22PilotStore(database, contract=contract):
        pass
    modified_before = database.stat().st_mtime_ns
    tool = _tool()
    monkeypatch.setattr(
        sys,
        "argv",
        [str(TOOL_PATH), "status", "--database", str(database)],
    )

    assert tool.main(repository=ROOT) == 0

    output = json.loads(capsys.readouterr().out)
    assert output["event"] == "round22_status"
    assert output["development_complete_count"] == 0
    assert output["development_remaining_count"] == 480
    assert output["feature_materialized_condition_count"] == 0
    assert output["feature_payload_compressed_bytes"] == 0
    assert output["target_row_count"] == 0
    assert not any(
        output[key]
        for key in (
            "authentication_used",
            "binance_used",
            "live_trading_authority",
            "paper_trading_authority",
            "target_accessed",
        )
    )
    assert database.stat().st_mtime_ns == modified_before


@pytest.mark.parametrize("maximum", ["0", "49"])
def test_round22_ingestion_tool_rejects_unbounded_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    maximum: str,
) -> None:
    tool = _tool()
    database = tmp_path / "round22.duckdb"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(TOOL_PATH),
            "ingest",
            "--database",
            str(database),
            "--maximum-conditions",
            maximum,
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        tool.main(repository=ROOT)

    assert not database.exists()
