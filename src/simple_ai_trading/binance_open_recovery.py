"""Durable terminal observations; unresolved admission is deliberately retained."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict
from typing import TYPE_CHECKING

from .binance_execution_scope import BinanceExecutionScope
from .binance_open_intents import BinanceOpenIntentJournal, OpenIntentError
from .binance_terminal_fills import (
    TerminalFillEvidence,
    validate_terminal_fills,
    validate_terminal_order,
)
from .positions import OpenPosition

if TYPE_CHECKING:
    from .api import BinanceClient


_COLUMNS = ("client_id", "request_json", "order_json", "trades_json", "evidence_json")
_ORDER_FIELDS = (
    "symbol",
    "orderId",
    "clientOrderId",
    "origClientOrderId",
    "side",
    "type",
    "status",
    "origQty",
    "executedQty",
    "time",
    "updateTime",
)
_TRADE_FIELDS = (
    "symbol",
    "orderId",
    "id",
    "price",
    "qty",
    "quoteQty",
    "commission",
    "commissionAsset",
    "time",
)


def _encode(value: object) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _read_record(connection: sqlite3.Connection, client_id: str) -> tuple | None:
    columns = tuple(
        row[1] for row in connection.execute("PRAGMA table_info(opening_recovery)")
    )
    if not columns:
        return None
    if columns != _COLUMNS:
        raise OpenIntentError("opening recovery schema is not recognized")
    rows = connection.execute(
        "SELECT request_json, order_json, trades_json, evidence_json "
        "FROM opening_recovery WHERE client_id=? LIMIT 2",
        (client_id,),
    ).fetchall()
    if len(rows) > 1:
        raise OpenIntentError("opening recovery identity is ambiguous")
    return rows[0] if rows else None


def _retained(
    journal: BinanceOpenIntentJournal,
    position: OpenPosition,
    scope: BinanceExecutionScope,
) -> TerminalFillEvidence | None:
    with closing(journal._connect()) as connection:
        journal._bind_scope(connection, scope)
        record = _read_record(connection, position.open_client_order_id)
    if record is None:
        return None
    request, order, trades, encoded = record
    if request != journal._request(position, scope):
        raise OpenIntentError("retained recovery belongs to a different intent")
    evidence = validate_terminal_fills(
        position, scope, json.loads(order), json.loads(trades)
    )
    if encoded != _encode(asdict(evidence)):
        raise OpenIntentError("retained recovery evidence is inconsistent")
    return evidence


def collect_opening_recovery(
    client: BinanceClient, journal: BinanceOpenIntentJournal
) -> TerminalFillEvidence | None:
    """Query only the pending owned order, retain exact evidence, never submit/rearm."""
    scope = client.execution_scope()
    position = journal.pending_position(scope=scope)
    if position is None:
        return None
    try:
        retained = _retained(journal, position, scope)
        if retained is not None:
            return retained
        order = client.get_order(
            position.symbol,
            orig_client_order_id=position.open_client_order_id,
            expected_scope=scope,
        )
        terminal = validate_terminal_order(position, scope, order)
        trades = (
            []
            if terminal.executed_quantity == "0"
            else client.get_order_trades(
                position.symbol,
                order_id=terminal.order_id,
                expected_scope=scope,
            )
        )
        evidence = validate_terminal_fills(position, scope, order, trades)
        # Persist only validated execution fields, never arbitrary account payloads
        # or request headers. No native commission is converted into a quote fee.
        order_fields = _ORDER_FIELDS + (
            ("cummulativeQuoteQty",)
            if scope.market_type == "spot"
            else ("cumQuote", "positionSide", "reduceOnly")
        )
        trade_fields = _TRADE_FIELDS + (
            ("isBuyer", "isMaker")
            if scope.market_type == "spot"
            else ("side", "positionSide", "buyer", "maker", "realizedPnl")
        )
        order_json = _encode({key: order[key] for key in order_fields if key in order})
        trades_json = _encode(
            [
                {key: trade[key] for key in trade_fields if key in trade}
                for trade in trades
            ]
        )
        request_json = journal._request(position, scope)
        encoded = _encode(asdict(evidence))
        with closing(journal._connect(write=True)) as connection, connection:
            journal._bind_scope(connection, scope)
            pending = connection.execute(
                "SELECT request_json FROM open_intent WHERE client_id=? AND state='UNKNOWN'",
                (position.open_client_order_id,),
            ).fetchone()
            if pending != (request_json,):
                raise OpenIntentError("opening changed while recovery was queried")
            previous = _read_record(connection, position.open_client_order_id)
            if previous is not None:
                if previous != (request_json, order_json, trades_json, encoded):
                    raise OpenIntentError("concurrent terminal observations conflict")
                return evidence
            connection.execute(
                "CREATE TABLE IF NOT EXISTS opening_recovery (client_id TEXT PRIMARY KEY, "
                "request_json TEXT NOT NULL, order_json TEXT NOT NULL, "
                "trades_json TEXT NOT NULL, evidence_json TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO opening_recovery VALUES (?, ?, ?, ?, ?)",
                (
                    position.open_client_order_id,
                    request_json,
                    order_json,
                    trades_json,
                    encoded,
                ),
            )
        return evidence
    except (sqlite3.Error, OSError, TypeError, ValueError) as exc:
        if isinstance(exc, OpenIntentError):
            raise
        raise OpenIntentError(
            "opening recovery evidence could not be retained"
        ) from None
