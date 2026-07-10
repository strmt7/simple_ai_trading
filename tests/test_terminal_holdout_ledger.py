from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from simple_ai_trading.model import TrainedModel
from simple_ai_trading.terminal_holdout_ledger import (
    LEDGER_SCHEMA_VERSION,
    RESERVATION_SCHEMA_VERSION,
    TerminalHoldoutLedger,
    TerminalHoldoutLedgerError,
    TerminalHoldoutReuseError,
    default_terminal_holdout_ledger_path,
    reservation_evidence_passed,
    terminal_model_fingerprint,
    terminal_result_fingerprint,
)

_DATASET_SHA = "1" * 64
_MODEL_SHA = "2" * 64
_RESULT_SHA = "3" * 64


def _model() -> TrainedModel:
    return TrainedModel(
        weights=[0.25, -0.5],
        bias=0.1,
        feature_dim=2,
        epochs=10,
        feature_means=[0.0, 0.0],
        feature_stds=[1.0, 1.0],
        meta_label_policy={"enabled": True, "threshold": 0.7},
    )


def _reserve(
    ledger: TerminalHoldoutLedger,
    *,
    symbol: str = "BTCUSDT",
    objective: str = "conservative",
    first_timestamp: int = 1_000,
    last_timestamp: int = 2_000,
    dataset_fingerprint: str = _DATASET_SHA,
    model_fingerprint: str = _MODEL_SHA,
) -> dict[str, object]:
    return ledger.reserve(
        symbol=symbol,
        market_type="futures",
        objective=objective,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        rows=101,
        dataset_fingerprint=dataset_fingerprint,
        model_fingerprint=model_fingerprint,
    )


def test_default_terminal_ledger_path_honors_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = tmp_path / "governance.sqlite3"
    monkeypatch.setenv("SIMPLE_AI_TRADING_TERMINAL_LEDGER", str(expected))

    assert default_terminal_holdout_ledger_path() == expected


def test_terminal_model_fingerprint_excludes_later_governance_stamps() -> None:
    model = _model()
    initial = terminal_model_fingerprint(model)

    model.selection_risk = {"passed": True}
    model.execution_validation = {"accepted": True}
    assert terminal_model_fingerprint(model) == initial

    model.meta_label_policy["threshold"] = 0.8
    assert terminal_model_fingerprint(model) != initial
    with pytest.raises(TypeError, match="dataclass"):
        terminal_model_fingerprint(object())


def test_terminal_result_fingerprint_binds_result_but_excludes_reservation() -> None:
    report = {
        "schema_version": "terminal-holdout-v1",
        "passed": True,
        "score": 0.2,
        "result": {"realized_pnl": 10.0},
        "reservation": {"status": "reserved"},
    }
    initial = terminal_result_fingerprint(report)

    report["reservation"] = {"status": "complete"}
    assert terminal_result_fingerprint(report) == initial
    report["result"]["realized_pnl"] = 11.0
    assert terminal_result_fingerprint(report) != initial
    with pytest.raises(TypeError, match="mapping"):
        terminal_result_fingerprint(None)


def test_reservation_is_durable_and_matches_final_evidence(tmp_path: Path) -> None:
    path = tmp_path / "terminal.sqlite3"
    ledger = TerminalHoldoutLedger(path)

    reserved = _reserve(ledger)
    finalized = ledger.finalize(
        str(reserved["reservation_id"]),
        result_status="accepted",
        result_fingerprint=_RESULT_SHA,
    )

    assert reserved["schema_version"] == RESERVATION_SCHEMA_VERSION
    assert reserved["status"] == "reserved"
    assert finalized["status"] == "complete"
    assert finalized["result_status"] == "accepted"
    assert ledger.reservation(str(reserved["reservation_id"])) == finalized
    assert ledger.evidence_matches(finalized)
    assert reservation_evidence_passed(
        finalized,
        expected_dataset_fingerprint=_DATASET_SHA,
        expected_model_fingerprint=_MODEL_SHA,
        expected_result_fingerprint=_RESULT_SHA,
        expected_rows=101,
        expected_first_timestamp=1_000,
        expected_last_timestamp=2_000,
        expected_symbol="BTCUSDT",
        expected_market_type="futures",
        expected_objective="conservative",
    )
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
        metadata = dict(connection.execute("SELECT key, value FROM governance_metadata"))
    assert metadata["schema_version"] == LEDGER_SCHEMA_VERSION
    assert len(metadata["ledger_id"]) == 64


@pytest.mark.parametrize("result_status", ["accepted", "rejected", "evaluation_error"])
def test_finalized_or_failed_period_can_never_be_reused(
    tmp_path: Path,
    result_status: str,
) -> None:
    ledger = TerminalHoldoutLedger(tmp_path / f"{result_status}.sqlite3")
    reserved = _reserve(ledger)
    ledger.finalize(
        str(reserved["reservation_id"]),
        result_status=result_status,
        result_fingerprint=_RESULT_SHA,
    )

    with pytest.raises(TerminalHoldoutReuseError, match="previously reserved"):
        _reserve(
            ledger,
            first_timestamp=1_500,
            last_timestamp=2_500,
            dataset_fingerprint="3" * 64,
        )


def test_unfinalized_crash_reservation_blocks_reuse(tmp_path: Path) -> None:
    ledger = TerminalHoldoutLedger(tmp_path / "crash.sqlite3")
    _reserve(ledger)

    with pytest.raises(TerminalHoldoutReuseError):
        _reserve(
            ledger,
            first_timestamp=999,
            last_timestamp=1_001,
            dataset_fingerprint="4" * 64,
        )


def test_nonoverlap_and_distinct_symbol_or_objective_are_independent(tmp_path: Path) -> None:
    ledger = TerminalHoldoutLedger(tmp_path / "independent.sqlite3")
    _reserve(ledger)

    assert _reserve(
        ledger,
        first_timestamp=2_001,
        last_timestamp=3_000,
        dataset_fingerprint="5" * 64,
    )["status"] == "reserved"
    assert _reserve(
        ledger,
        symbol="ETHUSDT",
        dataset_fingerprint="6" * 64,
    )["status"] == "reserved"
    assert _reserve(
        ledger,
        objective="regular",
        dataset_fingerprint="7" * 64,
    )["status"] == "reserved"


def test_concurrent_overlap_allows_exactly_one_reservation(tmp_path: Path) -> None:
    path = tmp_path / "concurrent.sqlite3"
    barrier = Barrier(2)

    def attempt(index: int) -> str:
        barrier.wait(timeout=5.0)
        try:
            _reserve(
                TerminalHoldoutLedger(path),
                dataset_fingerprint=str(index + 8) * 64,
            )
            return "reserved"
        except TerminalHoldoutReuseError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = sorted(executor.map(attempt, (0, 1)))

    assert outcomes == ["rejected", "reserved"]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"symbol": "DOGEUSDT"}, "unsupported"),
        ({"market_type": "margin"}, "market_type"),
        ({"objective": "experimental"}, "objective"),
        ({"first_timestamp": 3_000, "last_timestamp": 2_000}, "range"),
        ({"rows": 0}, "rows"),
        ({"dataset_fingerprint": "bad"}, "SHA-256"),
        ({"model_fingerprint": "bad"}, "SHA-256"),
    ],
)
def test_invalid_reservation_contract_fails_closed(
    tmp_path: Path,
    updates: dict[str, object],
    message: str,
) -> None:
    payload: dict[str, object] = {
        "symbol": "BTCUSDT",
        "market_type": "spot",
        "objective": "conservative",
        "first_timestamp": 1_000,
        "last_timestamp": 2_000,
        "rows": 101,
        "dataset_fingerprint": _DATASET_SHA,
        "model_fingerprint": _MODEL_SHA,
    }
    payload.update(updates)

    with pytest.raises(ValueError, match=message):
        TerminalHoldoutLedger(tmp_path / "invalid.sqlite3").reserve(**payload)  # type: ignore[arg-type]


def test_finalize_rejects_missing_invalid_and_repeated_reservations(tmp_path: Path) -> None:
    ledger = TerminalHoldoutLedger(tmp_path / "finalize.sqlite3")
    reserved = _reserve(ledger)

    with pytest.raises(ValueError, match="result_status"):
        ledger.finalize(
            str(reserved["reservation_id"]),
            result_status="unknown",
            result_fingerprint=_RESULT_SHA,
        )
    with pytest.raises(TerminalHoldoutLedgerError, match="does not exist"):
        ledger.finalize("f" * 64, result_status="accepted", result_fingerprint=_RESULT_SHA)
    ledger.finalize(
        str(reserved["reservation_id"]),
        result_status="accepted",
        result_fingerprint=_RESULT_SHA,
    )
    with pytest.raises(TerminalHoldoutLedgerError, match="already finalized"):
        ledger.finalize(
            str(reserved["reservation_id"]),
            result_status="accepted",
            result_fingerprint=_RESULT_SHA,
        )


def test_evidence_validation_and_database_match_reject_tampering(tmp_path: Path) -> None:
    ledger = TerminalHoldoutLedger(tmp_path / "tamper.sqlite3")
    reserved = _reserve(ledger)
    finalized = ledger.finalize(
        str(reserved["reservation_id"]),
        result_status="accepted",
        result_fingerprint=_RESULT_SHA,
    )

    for key, value in (
        ("dataset_fingerprint", "a" * 64),
        ("model_fingerprint", "b" * 64),
        ("result_fingerprint", "d" * 64),
        ("rows", 102),
        ("status", "reserved"),
        ("result_status", "rejected"),
        ("ledger_id", "c" * 64),
    ):
        tampered = dict(finalized)
        tampered[key] = value
        assert not ledger.evidence_matches(tampered)
        if key == "dataset_fingerprint":
            assert not reservation_evidence_passed(
                tampered,
                expected_dataset_fingerprint=_DATASET_SHA,
            )
        elif key == "model_fingerprint":
            assert not reservation_evidence_passed(
                tampered,
                expected_model_fingerprint=_MODEL_SHA,
            )
        elif key == "result_fingerprint":
            assert not reservation_evidence_passed(
                tampered,
                expected_result_fingerprint=_RESULT_SHA,
            )
        elif key == "rows":
            assert not reservation_evidence_passed(tampered, expected_rows=101)
        elif key in {"status", "result_status"}:
            assert not reservation_evidence_passed(tampered)
    assert not reservation_evidence_passed(None)
    assert not ledger.evidence_matches(None)


def test_recreated_database_cannot_validate_old_evidence(tmp_path: Path) -> None:
    path = tmp_path / "recreated.sqlite3"
    ledger = TerminalHoldoutLedger(path)
    reserved = _reserve(ledger)
    finalized = ledger.finalize(
        str(reserved["reservation_id"]),
        result_status="accepted",
        result_fingerprint=_RESULT_SHA,
    )
    path.unlink()

    assert not TerminalHoldoutLedger(path).evidence_matches(finalized)


def test_schema_identity_mismatch_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "schema.sqlite3"
    ledger = TerminalHoldoutLedger(path)
    reserved = _reserve(ledger)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE governance_metadata SET value = 'future-schema' WHERE key = 'schema_version'"
        )

    with pytest.raises(TerminalHoldoutLedgerError, match="unsupported"):
        ledger.reservation(str(reserved["reservation_id"]))
