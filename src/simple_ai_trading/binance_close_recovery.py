"""Exact terminal close observations, without inventory application or rearm."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from decimal import Decimal, localcontext
from typing import TYPE_CHECKING

from .binance_close_intents import (
    _request,
    unresolved_close_count,
    validate_closing_client_id,
)
from .binance_execution_scope import BinanceExecutionScope
from .binance_execution_payloads import encode_execution_payloads
from .binance_open_intents import OpenIntentError
from .binance_terminal_fills import (
    TerminalFillEvidence,
    _decimal,
    _text,
    validate_terminal_fills,
    validate_terminal_order,
)
from .positions import OpenPosition, PositionsStore

if TYPE_CHECKING:
    from .api import BinanceClient

_COLUMNS = ("client_id", "request_json", "order_json", "trades_json", "evidence_json")


@dataclass(frozen=True)
class ClosingRecoveryEvidence:
    client_id: str
    execution: TerminalFillEvidence
    # Venue-reported futures PnL is not local lot PnL or an asset-qualified cash
    # movement. Instrument/margin identity and account attribution come later.
    futures_realized_pnl: str | None


def _encode(value: object) -> str:
    return json.dumps(value, sort_keys=True, allow_nan=False)


def _pending(
    connection: sqlite3.Connection,
    store: PositionsStore,
    scope: BinanceExecutionScope,
    client_id: str,
) -> tuple[OpenPosition, str] | None:
    store.opening_intents._bind_scope(connection, scope)
    if not unresolved_close_count(connection):
        return None
    rows = connection.execute(
        "SELECT position_id, request_json, state FROM close_intent WHERE client_id=? LIMIT 2",
        (client_id,),
    ).fetchall()
    if not rows:
        return None
    if len(rows) != 1 or rows[0][2] not in {"UNKNOWN", "RECORDED"}:
        raise OpenIntentError("closing recovery intent is ambiguous")
    position_id, request_json, state = rows[0]
    if state == "RECORDED":
        return None
    payload = json.loads(request_json)
    position = OpenPosition(**payload["opening"]["position_template"])
    expected = _request(store, position, client_id, scope, payload["reduce_only"])
    if position.id != position_id or request_json != expected:
        raise OpenIntentError("closing recovery request is inconsistent")
    return position, request_json


def _record(connection: sqlite3.Connection, client_id: str) -> tuple | None:
    columns = tuple(
        row[1] for row in connection.execute("PRAGMA table_info(closing_recovery)")
    )
    if not columns:
        return None
    if columns != _COLUMNS:
        raise OpenIntentError("closing recovery schema is not recognized")
    rows = connection.execute(
        "SELECT request_json, order_json, trades_json, evidence_json "
        "FROM closing_recovery WHERE client_id=? LIMIT 2",
        (client_id,),
    ).fetchall()
    if len(rows) > 1:
        raise OpenIntentError("closing recovery identity is ambiguous")
    return rows[0] if rows else None


def _evidence(
    position: OpenPosition,
    scope: BinanceExecutionScope,
    client_id: str,
    order: dict,
    trades: object,
) -> ClosingRecoveryEvidence:
    execution = validate_terminal_fills(
        position, scope, order, trades, closing_client_id=client_id
    )
    pnl = None
    if scope.market_type == "futures":
        with localcontext() as context:
            context.prec = 100
            pnl = _text(
                sum(
                    (_decimal(trade["realizedPnl"], signed=True) for trade in trades),
                    Decimal(0),
                )
            )
    return ClosingRecoveryEvidence(client_id, execution, pnl)


def collect_closing_recovery(
    client: BinanceClient, store: PositionsStore, *, client_id: str
) -> ClosingRecoveryEvidence | None:
    """Retain one exact pending close's terminal fills; never submit or release it."""
    validate_closing_client_id(client_id)
    journal = store.opening_intents
    if not journal.path.exists():
        return None
    scope = client.execution_scope()
    try:
        with closing(journal._connect()) as connection:
            pending = _pending(connection, store, scope, client_id)
            if pending is None:
                return None
            position, request_json = pending
            retained = _record(connection, client_id)
        if retained is not None:
            request, order_json, trades_json, encoded = retained
            if request != request_json:
                raise OpenIntentError("retained closing belongs to a different intent")
            result = _evidence(
                position,
                scope,
                client_id,
                json.loads(order_json),
                json.loads(trades_json),
            )
            if encoded != _encode(asdict(result)):
                raise OpenIntentError("retained closing evidence is inconsistent")
            return result
        order = client.get_order(
            position.symbol, orig_client_order_id=client_id, expected_scope=scope
        )
        terminal = validate_terminal_order(
            position, scope, order, closing_client_id=client_id
        )
        trades = (
            []
            if terminal.executed_quantity == "0"
            else client.get_order_trades(
                position.symbol, order_id=terminal.order_id, expected_scope=scope
            )
        )
        result = _evidence(position, scope, client_id, order, trades)
        order_json, trades_json = encode_execution_payloads(scope, order, trades)
        encoded = _encode(asdict(result))
        with closing(journal._connect(write=True)) as connection, connection:
            if _pending(connection, store, scope, client_id) != pending:
                raise OpenIntentError("closing changed while recovery was queried")
            previous = _record(connection, client_id)
            row = (request_json, order_json, trades_json, encoded)
            if previous is not None:
                if previous != row:
                    raise OpenIntentError("concurrent closing observations conflict")
                return result
            connection.execute(
                "CREATE TABLE IF NOT EXISTS closing_recovery (client_id TEXT PRIMARY KEY, "
                "request_json TEXT NOT NULL, order_json TEXT NOT NULL, "
                "trades_json TEXT NOT NULL, evidence_json TEXT NOT NULL)"
            )
            connection.execute(
                "INSERT INTO closing_recovery VALUES (?, ?, ?, ?, ?)", (client_id, *row)
            )
        return result
    except (sqlite3.Error, OSError, TypeError, ValueError, KeyError) as exc:
        if isinstance(exc, OpenIntentError):
            raise
        raise OpenIntentError(
            "closing recovery evidence could not be retained"
        ) from None
