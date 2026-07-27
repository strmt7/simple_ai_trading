from __future__ import annotations

from decimal import Decimal

import duckdb
import pytest

from simple_ai_trading.round74_execution_calibration_journal import (
    Round74ExecutionCalibrationIntent,
    Round74ExecutionCalibrationJournal,
    Round74ExecutionCalibrationTransition,
)


SOURCE_SHA = "1" * 64


def _intent(*, path: str) -> Round74ExecutionCalibrationIntent:
    return Round74ExecutionCalibrationIntent(
        calibration_run_id="round74-test",
        round_trip_id="BTCUSDT-0",
        path=path,
        symbol="BTCUSDT",
        side="BUY" if path == "entry" else "SELL",
        client_order_id=f"sat-r74-cal-test-{'i' if path == 'entry' else 'o'}",
        quantity=Decimal("0.001"),
        reference_quote_notional=Decimal("100"),
        reduce_only=path == "exit",
        created_wall_ns=1_800_000_000_000_000_000 + (path == "exit"),
        book_source_sha256=SOURCE_SHA,
    )


def _transition(
    state: str,
    *,
    wall_ns: int,
) -> Round74ExecutionCalibrationTransition:
    filled = state in {"FILLED", "FLAT_VERIFIED"}
    return Round74ExecutionCalibrationTransition(
        state=state,
        occurred_wall_ns=wall_ns,
        source_payload_sha256="2" * 64,
        order_id=1001 if filled else 0,
        executed_quantity=Decimal("0.001") if filled else Decimal("0"),
        average_fill_price=Decimal("100000") if filled else Decimal("0"),
    )


def test_journal_blocks_prepared_and_filled_entry_until_flat_exit() -> None:
    connection = duckdb.connect(":memory:")
    journal = Round74ExecutionCalibrationJournal(connection)
    entry = _intent(path="entry")

    prepared = journal.record_intent(entry)
    assert prepared.state == "PREPARED"
    assert journal.blocking_round_trip_ids() == (
        "round74-test:BTCUSDT-0",
    )

    filled_entry = journal.transition(
        entry.client_order_id,
        _transition("FILLED", wall_ns=entry.created_wall_ns + 1),
    )
    assert filled_entry.state == "FILLED"
    assert journal.blocking_round_trip_ids() == (
        "round74-test:BTCUSDT-0",
    )

    exit_intent = _intent(path="exit")
    journal.record_intent(exit_intent)
    journal.transition(
        exit_intent.client_order_id,
        _transition("FILLED", wall_ns=exit_intent.created_wall_ns + 1),
    )
    assert journal.blocking_round_trip_ids() == (
        "round74-test:BTCUSDT-0",
    )

    journal.transition(
        exit_intent.client_order_id,
        _transition(
            "FLAT_VERIFIED",
            wall_ns=exit_intent.created_wall_ns + 2,
        ),
    )
    journal.verify()
    assert journal.blocking_round_trip_ids() == ()


def test_journal_is_idempotent_and_rejects_changed_intent() -> None:
    journal = Round74ExecutionCalibrationJournal(duckdb.connect(":memory:"))
    intent = _intent(path="entry")

    first = journal.record_intent(intent)
    second = journal.record_intent(intent)

    assert first == second
    with pytest.raises(ValueError, match="intent identity differs"):
        journal.record_intent(
            Round74ExecutionCalibrationIntent(
                **{**intent.__dict__, "quantity": Decimal("0.002")}
            )
        )


def test_unknown_status_stays_blocking_and_can_reconcile_to_fill() -> None:
    journal = Round74ExecutionCalibrationJournal(duckdb.connect(":memory:"))
    intent = _intent(path="entry")
    journal.record_intent(intent)
    unknown = journal.transition(
        intent.client_order_id,
        Round74ExecutionCalibrationTransition(
            state="UNKNOWN",
            occurred_wall_ns=intent.created_wall_ns + 1,
            source_payload_sha256="3" * 64,
            reason="submission outcome unknown",
        ),
    )

    assert unknown.state == "UNKNOWN"
    assert journal.blocking_round_trip_ids() == (
        "round74-test:BTCUSDT-0",
    )
    reconciled = journal.transition(
        intent.client_order_id,
        _transition("FILLED", wall_ns=intent.created_wall_ns + 2),
    )
    assert reconciled.state == "FILLED"


def test_journal_rejects_unsafe_exit_and_accounting_regressions() -> None:
    journal = Round74ExecutionCalibrationJournal(duckdb.connect(":memory:"))
    entry = _intent(path="entry")
    journal.record_intent(entry)
    with pytest.raises(ValueError, match="state transition differs"):
        journal.transition(
            entry.client_order_id,
            _transition("FLAT_VERIFIED", wall_ns=entry.created_wall_ns + 1),
        )

    filled = journal.transition(
        entry.client_order_id,
        _transition("FILLED", wall_ns=entry.created_wall_ns + 1),
    )
    assert filled.executed_quantity == Decimal("0.001")
    with pytest.raises(ValueError, match="transition accounting differs"):
        journal.transition(
            entry.client_order_id,
            Round74ExecutionCalibrationTransition(
                state="FLAT_VERIFIED",
                occurred_wall_ns=entry.created_wall_ns + 2,
                source_payload_sha256="4" * 64,
                order_id=1001,
                executed_quantity=Decimal("0.0009"),
                average_fill_price=Decimal("100000"),
            ),
        )


def test_journal_verify_detects_tampering() -> None:
    connection = duckdb.connect(":memory:")
    journal = Round74ExecutionCalibrationJournal(connection)
    intent = _intent(path="entry")
    journal.record_intent(intent)
    connection.execute(
        """
        UPDATE round74_execution_calibration_event
        SET reason = 'tampered'
        WHERE client_order_id = ?
        """,
        [intent.client_order_id],
    )

    with pytest.raises(ValueError, match="event digest differs"):
        journal.verify()
