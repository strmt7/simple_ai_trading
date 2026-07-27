"""Fail-closed coordinator for Round 74 non-mainnet calibration pairs."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import time
from typing import Mapping, Protocol, Sequence

from .impact_absorption_event_targets import ROUND74_EVENT_TARGET_SYMBOLS
from .round74_execution_calibration_capture import (
    Round74ExecutionCaptureLeg,
    Round74ExecutionCapturePair,
)
from .round74_execution_calibration_journal import (
    Round74ExecutionCalibrationIntent,
    Round74ExecutionCalibrationJournal,
    Round74ExecutionCalibrationTransition,
)


ROUND74_EXECUTION_COORDINATOR_SCHEMA_VERSION = (
    "round-074-execution-calibration-coordinator-v1"
)
ROUND74_EXECUTION_TERMINAL_TIMEOUT_SECONDS = 15.0


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _positive_decimal(value: object, *, label: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"Round 74 execution coordinator {label} differs")
    try:
        selected = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(
            f"Round 74 execution coordinator {label} differs"
        ) from exc
    if not selected.is_finite() or selected <= 0:
        raise ValueError(f"Round 74 execution coordinator {label} differs")
    return selected


def _normalized_mapping(
    value: Mapping[str, object],
    *,
    label: str,
) -> dict[str, object]:
    try:
        normalized = json.loads(_canonical_json(dict(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Round 74 execution coordinator {label} differs"
        ) from exc
    if not isinstance(normalized, dict):
        raise ValueError(f"Round 74 execution coordinator {label} differs")
    return normalized


def _client_order_id(
    *,
    calibration_run_id: str,
    round_trip_id: str,
    path: str,
) -> str:
    digest = _canonical_sha256(
        {
            "calibration_run_id": calibration_run_id,
            "round_trip_id": round_trip_id,
            "path": path,
        }
    )
    suffix = "i" if path == "entry" else "o"
    return f"sat-r74-cal-{digest[:18]}-{suffix}"


def _flat_position(
    value: Mapping[str, object],
    *,
    symbol: str,
) -> dict[str, object]:
    selected = _normalized_mapping(value, label="position payload")
    if (
        selected.get("symbol") != symbol
        or selected.get("positionSide") != "BOTH"
        or Decimal(str(selected.get("positionAmt"))) != 0
    ):
        raise RuntimeError(
            "Round 74 execution coordinator requires an exact flat "
            f"one-way position for {symbol}"
        )
    return selected


class Round74OrderSubmissionUnknown(RuntimeError):
    """Raised only when an order request may have reached the matching engine."""


class Round74ExecutionCalibrationTransport(Protocol):
    """Minimal non-mainnet exchange surface; implementations own no policy."""

    def position(self, symbol: str) -> Mapping[str, object]: ...

    def open_orders(self, symbol: str) -> Sequence[Mapping[str, object]]: ...

    def book(self, symbol: str) -> Mapping[str, object]: ...

    def submit_market_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: Decimal,
        reduce_only: bool,
        client_order_id: str,
    ) -> tuple[int, Mapping[str, object]]: ...

    def wait_terminal_order_update(
        self,
        *,
        symbol: str,
        client_order_id: str,
        timeout_seconds: float,
    ) -> tuple[int, Mapping[str, object]] | None: ...

    def query_order(
        self,
        *,
        symbol: str,
        client_order_id: str,
    ) -> Mapping[str, object] | None: ...

    def account_trades(
        self,
        *,
        symbol: str,
        order_id: int,
    ) -> Sequence[Mapping[str, object]]: ...


@dataclass(frozen=True)
class _LegOutcome:
    client_order_id: str
    order_id: int
    executed_quantity: Decimal
    average_fill_price: Decimal
    submission_monotonic_ns: int
    terminal_receipt_monotonic_ns: int
    terminal_order_payload: Mapping[str, object] | None


@dataclass(frozen=True)
class Round74ExecutionCalibrationResult:
    calibration_run_id: str
    round_trip_id: str
    symbol: str
    pair: Round74ExecutionCapturePair | None
    final_position_payload: Mapping[str, object]
    evidence_rejection_reason: str
    schema_version: str = ROUND74_EXECUTION_COORDINATOR_SCHEMA_VERSION

    @property
    def evidence_admitted(self) -> bool:
        return self.pair is not None

    def as_dict(self) -> dict[str, object]:
        if (
            self.schema_version != ROUND74_EXECUTION_COORDINATOR_SCHEMA_VERSION
            or self.symbol not in ROUND74_EVENT_TARGET_SYMBOLS
            or not self.calibration_run_id
            or not self.round_trip_id
            or bool(self.pair) == bool(self.evidence_rejection_reason)
        ):
            raise ValueError("Round 74 execution coordinator result differs")
        final_position = _flat_position(
            self.final_position_payload,
            symbol=self.symbol,
        )
        return {
            "schema_version": self.schema_version,
            "calibration_run_id": self.calibration_run_id,
            "round_trip_id": self.round_trip_id,
            "symbol": self.symbol,
            "evidence_admitted": self.evidence_admitted,
            "evidence_rejection_reason": self.evidence_rejection_reason,
            "pair": self.pair.as_dict() if self.pair is not None else None,
            "final_position_payload": final_position,
            "authority": {
                "testnet_calibration_pair": True,
                "mainnet_orders_submitted": False,
                "mainnet_trading_authority": False,
                "profitability_claim": False,
            },
        }


def _terminal_fill(
    payload: Mapping[str, object],
    *,
    symbol: str,
    side: str,
    client_order_id: str,
    reduce_only: bool,
) -> tuple[int, Decimal, Decimal]:
    selected = _normalized_mapping(payload, label="terminal order update")
    order = selected.get("o")
    if (
        selected.get("e") != "ORDER_TRADE_UPDATE"
        or not isinstance(order, Mapping)
        or order.get("s") != symbol
        or order.get("S") != side
        or order.get("c") != client_order_id
        or order.get("R") is not reduce_only
        or order.get("o") != "MARKET"
        or order.get("X") != "FILLED"
        or order.get("x") != "TRADE"
        or order.get("ps") != "BOTH"
    ):
        raise RuntimeError(
            "Round 74 execution terminal order update differs"
        )
    order_id = order.get("i")
    if isinstance(order_id, bool) or not isinstance(order_id, int) or order_id <= 0:
        raise RuntimeError("Round 74 execution terminal order ID differs")
    return (
        order_id,
        _positive_decimal(order.get("z"), label="executed quantity"),
        _positive_decimal(order.get("ap"), label="average fill price"),
    )


def _query_fill(
    payload: Mapping[str, object],
    *,
    symbol: str,
    side: str,
    client_order_id: str,
    reduce_only: bool,
) -> tuple[str, int, Decimal, Decimal]:
    selected = _normalized_mapping(payload, label="query order")
    if (
        selected.get("symbol") != symbol
        or selected.get("side") != side
        or selected.get("clientOrderId") != client_order_id
        or selected.get("reduceOnly") is not reduce_only
        or selected.get("type") != "MARKET"
        or selected.get("positionSide") != "BOTH"
    ):
        raise RuntimeError("Round 74 execution query order differs")
    status = str(selected.get("status", ""))
    order_id = selected.get("orderId", 0)
    if (
        status not in {
            "NEW",
            "PARTIALLY_FILLED",
            "FILLED",
            "CANCELED",
            "EXPIRED",
            "EXPIRED_IN_MATCH",
            "REJECTED",
        }
        or isinstance(order_id, bool)
        or not isinstance(order_id, int)
        or order_id <= 0
    ):
        raise RuntimeError("Round 74 execution query status differs")
    quantity = Decimal(str(selected.get("executedQty", "0")))
    average = Decimal(str(selected.get("avgPrice", "0")))
    if (
        not quantity.is_finite()
        or quantity < 0
        or not average.is_finite()
        or average < 0
        or status == "FILLED"
        and (quantity <= 0 or average <= 0)
    ):
        raise RuntimeError("Round 74 execution query fill differs")
    return status, order_id, quantity, average


def _transition(
    journal: Round74ExecutionCalibrationJournal,
    *,
    client_order_id: str,
    state: str,
    source_payload: object,
    order_id: int = 0,
    executed_quantity: Decimal = Decimal("0"),
    average_fill_price: Decimal = Decimal("0"),
    reason: str = "",
) -> None:
    journal.transition(
        client_order_id,
        Round74ExecutionCalibrationTransition(
            state=state,
            occurred_wall_ns=time.time_ns(),
            source_payload_sha256=_canonical_sha256(source_payload),
            order_id=order_id,
            executed_quantity=executed_quantity,
            average_fill_price=average_fill_price,
            reason=reason,
        ),
    )


def _execute_leg(
    *,
    transport: Round74ExecutionCalibrationTransport,
    journal: Round74ExecutionCalibrationJournal,
    symbol: str,
    side: str,
    quantity: Decimal,
    reduce_only: bool,
    client_order_id: str,
) -> _LegOutcome:
    submission_monotonic_ns = time.monotonic_ns()
    try:
        response_received_ns, response = transport.submit_market_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            reduce_only=reduce_only,
            client_order_id=client_order_id,
        )
    except Round74OrderSubmissionUnknown:
        _transition(
            journal,
            client_order_id=client_order_id,
            state="UNKNOWN",
            source_payload={
                "stage": "submission",
                "outcome": "unknown",
            },
            reason="submission outcome unknown",
        )
    else:
        if (
            isinstance(response_received_ns, bool)
            or not isinstance(response_received_ns, int)
            or response_received_ns < submission_monotonic_ns
        ):
            raise RuntimeError(
                "Round 74 execution submission receipt time differs"
            )
        selected_response = _normalized_mapping(
            response,
            label="submission response",
        )
        response_order_id = selected_response.get("orderId", 0)
        acknowledged = (
            isinstance(response_order_id, int)
            and not isinstance(response_order_id, bool)
            and response_order_id > 0
        )
        _transition(
            journal,
            client_order_id=client_order_id,
            state="ACKNOWLEDGED" if acknowledged else "SUBMITTED",
            source_payload=selected_response,
            order_id=int(response_order_id) if acknowledged else 0,
        )

    terminal = transport.wait_terminal_order_update(
        symbol=symbol,
        client_order_id=client_order_id,
        timeout_seconds=ROUND74_EXECUTION_TERMINAL_TIMEOUT_SECONDS,
    )
    if terminal is not None:
        terminal_receipt_ns, terminal_payload = terminal
        if (
            isinstance(terminal_receipt_ns, bool)
            or not isinstance(terminal_receipt_ns, int)
            or terminal_receipt_ns <= submission_monotonic_ns
        ):
            raise RuntimeError(
                "Round 74 execution terminal receipt time differs"
            )
        order_id, filled_quantity, average_fill_price = _terminal_fill(
            terminal_payload,
            symbol=symbol,
            side=side,
            client_order_id=client_order_id,
            reduce_only=reduce_only,
        )
        _transition(
            journal,
            client_order_id=client_order_id,
            state="FILLED",
            source_payload=terminal_payload,
            order_id=order_id,
            executed_quantity=filled_quantity,
            average_fill_price=average_fill_price,
        )
        return _LegOutcome(
            client_order_id=client_order_id,
            order_id=order_id,
            executed_quantity=filled_quantity,
            average_fill_price=average_fill_price,
            submission_monotonic_ns=submission_monotonic_ns,
            terminal_receipt_monotonic_ns=terminal_receipt_ns,
            terminal_order_payload=_normalized_mapping(
                terminal_payload,
                label="terminal order update",
            ),
        )

    query = transport.query_order(
        symbol=symbol,
        client_order_id=client_order_id,
    )
    if query is None:
        current = journal.current(client_order_id)
        if current.state != "UNKNOWN":
            _transition(
                journal,
                client_order_id=client_order_id,
                state="UNKNOWN",
                source_payload={
                    "stage": "terminal-reconciliation",
                    "outcome": "not-found",
                },
                order_id=current.order_id,
                executed_quantity=current.executed_quantity,
                average_fill_price=current.average_fill_price,
                reason="terminal event absent and order not found",
            )
        raise RuntimeError(
            "Round 74 execution order remains unresolved"
        )
    status, order_id, filled_quantity, average_fill_price = _query_fill(
        query,
        symbol=symbol,
        side=side,
        client_order_id=client_order_id,
        reduce_only=reduce_only,
    )
    if status != "FILLED":
        terminal_unfilled = status in {
            "CANCELED",
            "EXPIRED",
            "EXPIRED_IN_MATCH",
            "REJECTED",
        }
        _transition(
            journal,
            client_order_id=client_order_id,
            state="REJECTED" if terminal_unfilled else "UNKNOWN",
            source_payload=query,
            order_id=order_id,
            executed_quantity=filled_quantity,
            average_fill_price=average_fill_price,
            reason=f"query order status {status}",
        )
        raise RuntimeError(
            f"Round 74 execution order did not fill: {status}"
        )
    _transition(
        journal,
        client_order_id=client_order_id,
        state="FILLED",
        source_payload=query,
        order_id=order_id,
        executed_quantity=filled_quantity,
        average_fill_price=average_fill_price,
        reason="terminal event absent; reconciled by query",
    )
    return _LegOutcome(
        client_order_id=client_order_id,
        order_id=order_id,
        executed_quantity=filled_quantity,
        average_fill_price=average_fill_price,
        submission_monotonic_ns=submission_monotonic_ns,
        terminal_receipt_monotonic_ns=0,
        terminal_order_payload=None,
    )


def _book_or_recovery_digest(
    transport: Round74ExecutionCalibrationTransport,
    *,
    symbol: str,
    path: str,
) -> tuple[dict[str, object] | None, str]:
    try:
        book = _normalized_mapping(
            transport.book(symbol),
            label=f"{path} book",
        )
    except Exception:
        recovery = {
            "schema_version": ROUND74_EXECUTION_COORDINATOR_SCHEMA_VERSION,
            "symbol": symbol,
            "path": path,
            "book_capture": "unavailable",
            "used_for_evidence": False,
        }
        return None, _canonical_sha256(recovery)
    return book, _canonical_sha256(book)


def capture_round74_execution_calibration_pair(
    *,
    transport: Round74ExecutionCalibrationTransport,
    journal: Round74ExecutionCalibrationJournal,
    calibration_run_id: str,
    round_trip_id: str,
    symbol: str,
    entry_side: str,
    quantity: Decimal,
    reference_quote_notional: Decimal,
) -> Round74ExecutionCalibrationResult:
    """Capture one testnet pair while preserving closure over evidence."""

    run_id = str(calibration_run_id).strip()
    pair_id = str(round_trip_id).strip()
    selected_symbol = str(symbol).strip().upper()
    selected_side = str(entry_side).strip().upper()
    selected_quantity = _positive_decimal(quantity, label="quantity")
    selected_reference = _positive_decimal(
        reference_quote_notional,
        label="reference quote notional",
    )
    if (
        not run_id
        or not pair_id
        or selected_symbol not in ROUND74_EVENT_TARGET_SYMBOLS
        or selected_side not in {"BUY", "SELL"}
    ):
        raise ValueError("Round 74 execution coordinator identity differs")
    existing_blockers = journal.blocking_round_trip_ids()
    if existing_blockers:
        raise RuntimeError(
            "Round 74 execution coordinator has unresolved calibration "
            f"pairs: {','.join(existing_blockers)}"
        )
    pre_position = _flat_position(
        transport.position(selected_symbol),
        symbol=selected_symbol,
    )
    if transport.open_orders(selected_symbol):
        raise RuntimeError(
            "Round 74 execution coordinator found existing open orders"
        )

    entry_book, entry_book_sha = _book_or_recovery_digest(
        transport,
        symbol=selected_symbol,
        path="entry",
    )
    if entry_book is None:
        raise RuntimeError(
            "Round 74 execution coordinator requires an entry book"
        )
    entry_client_id = _client_order_id(
        calibration_run_id=run_id,
        round_trip_id=pair_id,
        path="entry",
    )
    journal.record_intent(
        Round74ExecutionCalibrationIntent(
            calibration_run_id=run_id,
            round_trip_id=pair_id,
            path="entry",
            symbol=selected_symbol,
            side=selected_side,
            client_order_id=entry_client_id,
            quantity=selected_quantity,
            reference_quote_notional=selected_reference,
            reduce_only=False,
            created_wall_ns=time.time_ns(),
            book_source_sha256=entry_book_sha,
        )
    )
    entry = _execute_leg(
        transport=transport,
        journal=journal,
        symbol=selected_symbol,
        side=selected_side,
        quantity=selected_quantity,
        reduce_only=False,
        client_order_id=entry_client_id,
    )

    exit_side = "SELL" if selected_side == "BUY" else "BUY"
    exit_book, exit_book_sha = _book_or_recovery_digest(
        transport,
        symbol=selected_symbol,
        path="exit",
    )
    exit_client_id = _client_order_id(
        calibration_run_id=run_id,
        round_trip_id=pair_id,
        path="exit",
    )
    journal.record_intent(
        Round74ExecutionCalibrationIntent(
            calibration_run_id=run_id,
            round_trip_id=pair_id,
            path="exit",
            symbol=selected_symbol,
            side=exit_side,
            client_order_id=exit_client_id,
            quantity=entry.executed_quantity,
            reference_quote_notional=selected_reference,
            reduce_only=True,
            created_wall_ns=time.time_ns(),
            book_source_sha256=exit_book_sha,
        )
    )
    exit_outcome = _execute_leg(
        transport=transport,
        journal=journal,
        symbol=selected_symbol,
        side=exit_side,
        quantity=entry.executed_quantity,
        reduce_only=True,
        client_order_id=exit_client_id,
    )
    post_position = _flat_position(
        transport.position(selected_symbol),
        symbol=selected_symbol,
    )
    _transition(
        journal,
        client_order_id=exit_client_id,
        state="FLAT_VERIFIED",
        source_payload=post_position,
        order_id=exit_outcome.order_id,
        executed_quantity=exit_outcome.executed_quantity,
        average_fill_price=exit_outcome.average_fill_price,
        reason="exact flat position verified",
    )

    pair: Round74ExecutionCapturePair | None = None
    rejection_reason = ""
    if (
        entry.terminal_order_payload is None
        or exit_outcome.terminal_order_payload is None
        or exit_book is None
    ):
        rejection_reason = "terminal_stream_or_book_evidence_incomplete"
    else:
        try:
            entry_trades = tuple(
                _normalized_mapping(trade, label="entry account trade")
                for trade in transport.account_trades(
                    symbol=selected_symbol,
                    order_id=entry.order_id,
                )
            )
            exit_trades = tuple(
                _normalized_mapping(trade, label="exit account trade")
                for trade in transport.account_trades(
                    symbol=selected_symbol,
                    order_id=exit_outcome.order_id,
                )
            )
            candidate = Round74ExecutionCapturePair(
                calibration_run_id=run_id,
                round_trip_id=pair_id,
                symbol=selected_symbol,
                entry=Round74ExecutionCaptureLeg(
                    path="entry",
                    symbol=selected_symbol,
                    side=selected_side,
                    client_order_id=entry_client_id,
                    submission_monotonic_ns=(
                        entry.submission_monotonic_ns
                    ),
                    terminal_receipt_monotonic_ns=(
                        entry.terminal_receipt_monotonic_ns
                    ),
                    expected_book_walk_source=entry_book,
                    terminal_order_payload=entry.terminal_order_payload,
                    account_trade_payloads=entry_trades,
                ),
                exit=Round74ExecutionCaptureLeg(
                    path="exit",
                    symbol=selected_symbol,
                    side=exit_side,
                    client_order_id=exit_client_id,
                    submission_monotonic_ns=(
                        exit_outcome.submission_monotonic_ns
                    ),
                    terminal_receipt_monotonic_ns=(
                        exit_outcome.terminal_receipt_monotonic_ns
                    ),
                    expected_book_walk_source=exit_book,
                    terminal_order_payload=(
                        exit_outcome.terminal_order_payload
                    ),
                    account_trade_payloads=exit_trades,
                ),
                pre_pair_position_payload=pre_position,
                post_pair_position_payload=post_position,
                reference_quote_notional=format(selected_reference, "f"),
            )
            candidate.records()
        except (RuntimeError, ValueError):
            rejection_reason = "parser_or_reconciliation_rejected"
        else:
            pair = candidate
    result = Round74ExecutionCalibrationResult(
        calibration_run_id=run_id,
        round_trip_id=pair_id,
        symbol=selected_symbol,
        pair=pair,
        final_position_payload=post_position,
        evidence_rejection_reason=rejection_reason,
    )
    result.as_dict()
    if journal.blocking_round_trip_ids():
        raise RuntimeError(
            "Round 74 execution coordinator did not close its journal"
        )
    return result


__all__ = [
    "ROUND74_EXECUTION_COORDINATOR_SCHEMA_VERSION",
    "ROUND74_EXECUTION_TERMINAL_TIMEOUT_SECONDS",
    "Round74ExecutionCalibrationResult",
    "Round74ExecutionCalibrationTransport",
    "Round74OrderSubmissionUnknown",
    "capture_round74_execution_calibration_pair",
]
