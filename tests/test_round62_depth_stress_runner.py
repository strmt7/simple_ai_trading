from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from simple_ai_trading.depth_stress_screen import utc_month_label
from tools.run_round62_depth_stress_transition import (
    DESIGN_DEFAULT,
    ProgressWriter,
    _complete_month_contract,
    _load_panel,
    _validated_design,
)


ROOT = Path(__file__).resolve().parents[1]


class _Result:
    def __init__(self, columns: dict[str, np.ndarray]) -> None:
        self.columns = columns

    def fetchnumpy(self) -> dict[str, np.ndarray]:
        return self.columns


class _Connection:
    def __init__(self, columns: dict[str, np.ndarray]) -> None:
        self.columns = columns
        self.parameters: list[object] | None = None

    def execute(self, _query: str, parameters: list[object]) -> _Result:
        self.parameters = parameters
        return _Result(self.columns)


class _Warehouse:
    def __init__(self, columns: dict[str, np.ndarray]) -> None:
        self.connection = _Connection(columns)

    def connect(self) -> _Connection:
        return self.connection


def _columns() -> dict[str, np.ndarray]:
    start = int(datetime(2025, 1, 1, tzinfo=UTC).timestamp() * 1_000)
    rows = 40
    axis = np.arange(rows, dtype=np.float64)
    return {
        "timestamp_ms": start + np.arange(rows, dtype=np.int64) * 30_000,
        "bid_depth_1": 100.0 + axis,
        "ask_depth_1": 120.0 + axis,
        "bid_notional_1": 1_000.0 + axis,
        "ask_notional_1": 1_100.0 + axis,
        "bid_notional_5": 10_000.0 + axis,
        "ask_notional_5": 11_000.0 + axis,
    }


def test_runner_accepts_only_hash_bound_design_and_freezes_complete_months() -> None:
    design = _validated_design(ROOT / DESIGN_DEFAULT)
    months, required_start_ms, required_end_ms = _complete_month_contract(design)

    assert len(months) == 42
    assert utc_month_label(int(months[0])) == "2023-01"
    assert utc_month_label(int(months[-1])) == "2026-06"
    assert required_start_ms == int(
        datetime(2023, 1, 1, tzinfo=UTC).timestamp() * 1_000
    )
    assert required_end_ms == int(
        datetime(2026, 7, 1, tzinfo=UTC).timestamp() * 1_000 - 1
    )


def test_runner_rejects_design_drift(tmp_path: Path) -> None:
    design = json.loads((ROOT / DESIGN_DEFAULT).read_text(encoding="utf-8"))
    design["evaluation_contract"]["maximum_q_value"] = 0.50
    path = tmp_path / "drifted-design.json"
    path.write_text(json.dumps(design), encoding="ascii")

    with pytest.raises(ValueError, match="hash or identity"):
        _validated_design(path)


def test_panel_loader_uses_depth_for_imbalance_and_notional_for_capacity() -> None:
    warehouse = _Warehouse(_columns())
    fingerprint = hashlib.sha256(b"certificate").hexdigest()
    panel = _load_panel(
        warehouse,
        symbol="BTCUSDT",
        required_start_ms=1,
        required_end_ms=2,
        source_fingerprint=fingerprint,
    )

    expected_imbalance = abs((100.0 - 120.0) / (100.0 + 120.0))
    assert panel.descriptors[0, 1] == pytest.approx(expected_imbalance)
    assert panel.descriptors[0, 0] == pytest.approx(-np.log1p(2_100.0))
    assert warehouse.connection.parameters == ["BTCUSDT", 1, 2]


def test_panel_loader_rejects_masked_warehouse_values() -> None:
    columns = _columns()
    columns["bid_depth_1"] = np.ma.array(
        columns["bid_depth_1"],
        mask=[True, *([False] * 39)],
    )
    with pytest.raises(ValueError, match="missing warehouse values"):
        _load_panel(
            _Warehouse(columns),
            symbol="BTCUSDT",
            required_start_ms=1,
            required_end_ms=2,
            source_fingerprint=hashlib.sha256(b"certificate").hexdigest(),
        )


def test_progress_writer_publishes_atomic_machine_readable_state(tmp_path: Path) -> None:
    path = tmp_path / "progress.json"
    writer = ProgressWriter(path)
    writer("unit_phase", symbol="BTCUSDT", fold_index=3)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "round-062-progress-v1"
    assert payload["sequence"] == 1
    assert payload["event"] == "unit_phase"
    assert payload["details"] == {"fold_index": 3, "symbol": "BTCUSDT"}
