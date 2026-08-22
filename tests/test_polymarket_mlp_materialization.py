from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from simple_ai_trading import polymarket_mlp as mlp
from simple_ai_trading.polymarket_ridge import (
    PolymarketPolicyEvaluation,
    PolymarketRidgeDataset,
)


class _Result:
    def __init__(self, *, row: tuple[object, ...] | None = None) -> None:
        self._row = row

    def fetchone(self) -> tuple[object, ...] | None:
        return self._row


def _persistence_rows() -> mlp._MLPPersistenceRows:
    return mlp._MLPPersistenceRows(
        report=("report",),
        runtime=("report", "runtime", "{}"),
        runtime_json="{}",
        tables=(),
    )


def test_mlp_materialization_rolls_back_failed_insert() -> None:
    statements: list[str] = []

    class Connection:
        def execute(
            self,
            statement: str,
            _parameters: Any = None,
        ) -> _Result:
            statements.append(statement)
            if statement.startswith("INSERT INTO polymarket_mlp_runtime_evidence"):
                raise RuntimeError("injected failure")
            return _Result()

        def executemany(self, _statement: str, _parameters: Any) -> None:
            raise AssertionError("no table rows should be inserted")

    with pytest.raises(RuntimeError, match="injected failure"):
        mlp._insert_mlp_materialization(Connection(), _persistence_rows())

    assert statements[0] == "BEGIN TRANSACTION"
    assert statements[-1] == "ROLLBACK"
    assert "COMMIT" not in statements


def test_mlp_materialization_rejects_tampered_runtime_evidence() -> None:
    rows = _persistence_rows()

    class Connection:
        def execute(
            self,
            statement: str,
            _parameters: Any = None,
        ) -> _Result:
            if statement.startswith("SELECT * FROM polymarket_mlp_report"):
                return _Result(row=rows.report)
            if "SELECT backend_json" in statement:
                return _Result(row=("tampered",))
            raise AssertionError(f"unexpected SQL: {statement}")

    with pytest.raises(ValueError, match="runtime evidence is inconsistent"):
        mlp._validate_existing_mlp_materialization(Connection(), "report", rows)


def test_mlp_materialization_rejects_replay_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = mlp._MLPPartitionRows([], [], [], [])
    actual = SimpleNamespace(asdict=lambda: {"value": "actual"})
    expected = SimpleNamespace(
        threshold=0.5,
        asdict=lambda: {"value": "expected"},
    )
    evaluation = cast(
        PolymarketPolicyEvaluation,
        SimpleNamespace(metrics=actual),
    )
    monkeypatch.setattr(
        mlp,
        "_replay_mlp_partition",
        lambda *_args, **_kwargs: (evaluation, rows),
    )
    report = cast(
        mlp.PolymarketMLPReport,
        SimpleNamespace(
            selected_threshold=0.5,
            validation_trials=(expected,),
            test_evaluated=True,
            test_metrics=expected,
            split=SimpleNamespace(validation_groups=(1,), test_groups=(2,)),
        ),
    )

    with pytest.raises(ValueError, match="validation replay differs"):
        mlp._replay_mlp_validation(cast(PolymarketRidgeDataset, object()), report)
    with pytest.raises(ValueError, match="test replay differs"):
        mlp._replay_mlp_test(cast(PolymarketRidgeDataset, object()), report)


def test_mlp_materialization_accepts_matching_test_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = mlp._MLPPartitionRows([], [], [], [])
    metrics = SimpleNamespace(asdict=lambda: {"value": "same"})
    evaluation = cast(
        PolymarketPolicyEvaluation,
        SimpleNamespace(metrics=metrics),
    )
    monkeypatch.setattr(
        mlp,
        "_replay_mlp_partition",
        lambda *_args, **_kwargs: (evaluation, rows),
    )
    report = cast(
        mlp.PolymarketMLPReport,
        SimpleNamespace(
            test_evaluated=True,
            test_metrics=metrics,
            split=SimpleNamespace(test_groups=(2,)),
        ),
    )

    assert mlp._replay_mlp_test(cast(PolymarketRidgeDataset, object()), report) is rows


def test_mlp_materialization_rejects_tampered_report() -> None:
    rows = _persistence_rows()

    class Connection:
        def execute(
            self,
            _statement: str,
            _parameters: Any = None,
        ) -> _Result:
            return _Result(row=("tampered",))

    with pytest.raises(ValueError, match="report is inconsistent"):
        mlp._validate_existing_mlp_materialization(Connection(), "report", rows)


def test_mlp_materialization_repairs_missing_runtime_evidence() -> None:
    rows = _persistence_rows()
    inserted: list[tuple[object, ...]] = []

    class Connection:
        def execute(
            self,
            statement: str,
            parameters: Any = None,
        ) -> _Result:
            if statement.startswith("SELECT * FROM polymarket_mlp_report"):
                return _Result(row=rows.report)
            if "SELECT backend_json" in statement:
                return _Result()
            if statement.startswith("INSERT INTO polymarket_mlp_runtime_evidence"):
                inserted.append(tuple(parameters))
                return _Result()
            raise AssertionError(f"unexpected SQL: {statement}")

    assert mlp._validate_existing_mlp_materialization(
        Connection(),
        "report",
        rows,
    )
    assert inserted == [rows.runtime]
