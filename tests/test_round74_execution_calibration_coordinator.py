from __future__ import annotations

from decimal import Decimal
import hashlib
import json
import time
from typing import Mapping

import duckdb
import pytest

from simple_ai_trading.round74_execution_calibration_coordinator import (
    Round74OrderSubmissionUnknown,
    capture_round74_execution_calibration_pair,
)
from simple_ai_trading.round74_execution_calibration_journal import (
    Round74ExecutionCalibrationIntent,
    Round74ExecutionCalibrationJournal,
)


def _sha(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


class _Transport:
    def __init__(
        self,
        *,
        terminal_events: bool = True,
        existing_orders: bool = False,
        unknown_paths: tuple[str, ...] = (),
    ) -> None:
        self.terminal_events = terminal_events
        self.existing_orders = existing_orders
        self.unknown_paths = unknown_paths
        self.position_amount = Decimal("0")
        self.orders: dict[str, dict[str, object]] = {}
        self.terminals: dict[str, dict[str, object]] = {}
        self.trades: dict[int, list[dict[str, object]]] = {}
        self.next_order_id = 1000

    def position(self, symbol: str) -> Mapping[str, object]:
        return {
            "symbol": symbol,
            "positionSide": "BOTH",
            "positionAmt": format(self.position_amount, "f"),
        }

    def open_orders(self, symbol: str) -> list[dict[str, object]]:
        if not self.existing_orders:
            return []
        return [{"symbol": symbol, "clientOrderId": "external"}]

    def book(self, symbol: str) -> Mapping[str, object]:
        raw = {
            "lastUpdateId": self.next_order_id,
            "bids": [["99.99", "10"]],
            "asks": [["100.01", "10"]],
        }
        return {
            "schema_version": "round-074-execution-book-state-v1",
            "symbol": symbol,
            "update_id": self.next_order_id,
            "received_monotonic_ns": time.monotonic_ns(),
            "bids": raw["bids"],
            "asks": raw["asks"],
            "source_payload_sha256": _sha(raw),
        }

    def submit_market_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: Decimal,
        reduce_only: bool,
        client_order_id: str,
    ) -> tuple[int, Mapping[str, object]]:
        self.next_order_id += 1
        order_id = self.next_order_id
        price = Decimal("100.02") if side == "BUY" else Decimal("99.98")
        signed_quantity = quantity if side == "BUY" else -quantity
        if reduce_only:
            self.position_amount += signed_quantity
        else:
            self.position_amount = signed_quantity
        query = {
            "symbol": symbol,
            "side": side,
            "clientOrderId": client_order_id,
            "reduceOnly": reduce_only,
            "type": "MARKET",
            "positionSide": "BOTH",
            "status": "FILLED",
            "orderId": order_id,
            "executedQty": format(quantity, "f"),
            "avgPrice": format(price, "f"),
        }
        terminal = {
            "e": "ORDER_TRADE_UPDATE",
            "E": 1_800_000_000_000 + order_id,
            "T": 1_800_000_000_000 + order_id,
            "o": {
                "s": symbol,
                "i": order_id,
                "c": client_order_id,
                "S": side,
                "ps": "BOTH",
                "o": "MARKET",
                "x": "TRADE",
                "X": "FILLED",
                "m": False,
                "R": reduce_only,
                "q": format(quantity, "f"),
                "z": format(quantity, "f"),
                "ap": format(price, "f"),
            },
        }
        self.orders[client_order_id] = query
        self.terminals[client_order_id] = terminal
        self.trades[order_id] = [
            {
                "id": order_id + 10_000,
                "orderId": order_id,
                "symbol": symbol,
                "side": side,
                "buyer": side == "BUY",
                "maker": False,
                "price": format(price, "f"),
                "qty": format(quantity, "f"),
                "quoteQty": format(quantity * price, "f"),
            }
        ]
        path = "exit" if reduce_only else "entry"
        if path in self.unknown_paths:
            raise Round74OrderSubmissionUnknown("do not persist this")
        return time.monotonic_ns(), {"orderId": order_id}

    def wait_terminal_order_update(
        self,
        *,
        symbol: str,
        client_order_id: str,
        timeout_seconds: float,
    ) -> tuple[int, Mapping[str, object]] | None:
        assert symbol == "BTCUSDT"
        assert timeout_seconds > 0
        if not self.terminal_events:
            return None
        return time.monotonic_ns(), self.terminals[client_order_id]

    def query_order(
        self,
        *,
        symbol: str,
        client_order_id: str,
    ) -> Mapping[str, object] | None:
        assert symbol == "BTCUSDT"
        return self.orders.get(client_order_id)

    def account_trades(
        self,
        *,
        symbol: str,
        order_id: int,
    ) -> list[Mapping[str, object]]:
        assert symbol == "BTCUSDT"
        return self.trades[order_id]


def _journal() -> Round74ExecutionCalibrationJournal:
    return Round74ExecutionCalibrationJournal(duckdb.connect(":memory:"))


def test_coordinator_captures_parser_admissible_flat_pair() -> None:
    transport = _Transport()
    journal = _journal()

    result = capture_round74_execution_calibration_pair(
        transport=transport,
        journal=journal,
        calibration_run_id="round74-test",
        round_trip_id="BTCUSDT-0",
        symbol="BTCUSDT",
        entry_side="BUY",
        quantity=Decimal("1"),
        reference_quote_notional=Decimal("100"),
    )

    assert result.evidence_admitted is True
    assert result.pair is not None
    assert len(result.pair.records()) == 2
    assert result.final_position_payload["positionAmt"] == "0"
    assert journal.blocking_round_trip_ids() == ()


def test_coordinator_closes_query_reconciled_pair_but_rejects_evidence() -> None:
    transport = _Transport(terminal_events=False)
    journal = _journal()

    result = capture_round74_execution_calibration_pair(
        transport=transport,
        journal=journal,
        calibration_run_id="round74-test",
        round_trip_id="BTCUSDT-1",
        symbol="BTCUSDT",
        entry_side="SELL",
        quantity=Decimal("1"),
        reference_quote_notional=Decimal("100"),
    )

    assert result.evidence_admitted is False
    assert (
        result.evidence_rejection_reason
        == "terminal_stream_or_book_evidence_incomplete"
    )
    assert transport.position_amount == 0
    assert journal.blocking_round_trip_ids() == ()


def test_coordinator_reconciles_ambiguous_submission_without_resubmitting() -> None:
    transport = _Transport(unknown_paths=("entry",))
    journal = _journal()

    result = capture_round74_execution_calibration_pair(
        transport=transport,
        journal=journal,
        calibration_run_id="round74-test",
        round_trip_id="BTCUSDT-2",
        symbol="BTCUSDT",
        entry_side="BUY",
        quantity=Decimal("1"),
        reference_quote_notional=Decimal("100"),
    )

    assert result.evidence_admitted is True
    assert len(transport.orders) == 2
    assert transport.position_amount == 0
    assert journal.blocking_round_trip_ids() == ()


@pytest.mark.parametrize("unsafe_state", ["position", "orders"])
def test_coordinator_refuses_external_or_untracked_state(
    unsafe_state: str,
) -> None:
    transport = _Transport(existing_orders=unsafe_state == "orders")
    if unsafe_state == "position":
        transport.position_amount = Decimal("1")
    journal = _journal()

    with pytest.raises(RuntimeError):
        capture_round74_execution_calibration_pair(
            transport=transport,
            journal=journal,
            calibration_run_id="round74-test",
            round_trip_id="BTCUSDT-3",
            symbol="BTCUSDT",
            entry_side="BUY",
            quantity=Decimal("1"),
            reference_quote_notional=Decimal("100"),
        )

    assert transport.orders == {}


def test_coordinator_refuses_new_pair_while_prior_intent_is_unresolved() -> None:
    transport = _Transport()
    journal = _journal()
    book = transport.book("BTCUSDT")

    journal.record_intent(
        Round74ExecutionCalibrationIntent(
            calibration_run_id="round74-prior",
            round_trip_id="BTCUSDT-prior",
            path="entry",
            symbol="BTCUSDT",
            side="BUY",
            client_order_id="sat-r74-cal-prior-i",
            quantity=Decimal("1"),
            reference_quote_notional=Decimal("100"),
            reduce_only=False,
            created_wall_ns=time.time_ns(),
            book_source_sha256=_sha(book),
        )
    )

    with pytest.raises(RuntimeError, match="unresolved calibration pairs"):
        capture_round74_execution_calibration_pair(
            transport=transport,
            journal=journal,
            calibration_run_id="round74-test",
            round_trip_id="BTCUSDT-4",
            symbol="BTCUSDT",
            entry_side="BUY",
            quantity=Decimal("1"),
            reference_quote_notional=Decimal("100"),
        )
    assert transport.orders == {}
