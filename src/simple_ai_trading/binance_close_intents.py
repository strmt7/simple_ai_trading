"""Scope-bound close obligations in the existing execution journal.

An unresolved close prevents another close for that lot and blocks new exposure.
It does not prevent an independently verified close of a different owned lot.
This boundary never guesses that a partial or timed-out order has terminated.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict
from typing import TYPE_CHECKING

from .binance_execution_scope import BinanceExecutionScope
from .binance_open_intents import OpenIntentError
from .position_transactions import position_transaction

if TYPE_CHECKING:
    from .positions import ClosedTrade, OpenPosition, PositionsStore

_COLUMNS = ("client_id", "position_id", "request_json", "state")


def unresolved_close_count(connection: sqlite3.Connection) -> int:
    """Read optional closing obligations without creating or repairing tables."""
    columns = tuple(
        row[1] for row in connection.execute("PRAGMA table_info(close_intent)")
    )
    if not columns:
        return 0
    if columns != _COLUMNS:
        raise OpenIntentError("closing intent schema is not recognized")
    return connection.execute(
        "SELECT COUNT(*) FROM close_intent WHERE state IS NOT 'RECORDED'"
    ).fetchone()[0]


def _request(
    store: PositionsStore,
    position: OpenPosition,
    client_id: str,
    scope: BinanceExecutionScope,
    reduce_only: bool,
) -> str:
    from .positions import bot_ownership_rejection_reason

    if not isinstance(scope, BinanceExecutionScope):
        raise OpenIntentError("closing requires an execution scope")
    opening = json.loads(store.opening_intents._request(position, scope))
    if (
        bot_ownership_rejection_reason(position) is not None
        or not isinstance(client_id, str)
        or not client_id.startswith("sait-c-")
        or not 1 <= len(client_id) <= 36
        or client_id != client_id.strip()
        or type(reduce_only) is not bool
        or position.market_type == "spot"
        and position.side != "LONG"
        or position.market_type == "futures"
        and not reduce_only
    ):
        raise OpenIntentError("closing intent ownership or reduction scope is invalid")
    return json.dumps(
        {
            "closing_client_id": client_id,
            "reduce_only": reduce_only,
            "opening": opening,
        },
        sort_keys=True,
        allow_nan=False,
    )


def prepare_close(
    store: PositionsStore,
    position: OpenPosition,
    *,
    client_id: str,
    scope: BinanceExecutionScope,
    reduce_only: bool,
) -> None:
    """Commit UNKNOWN against the current owned lot before an order can be sent."""
    payload = _request(store, position, client_id, scope, reduce_only)
    try:
        with position_transaction(store.opening_intents, write=True) as transaction:
            connection = transaction.connection
            if connection is None:
                raise OpenIntentError("closing requires durable execution storage")
            # Legacy positions cannot inherit the currently configured account.
            store.opening_intents._bind_scope(connection, scope)
            opens = store._open_entries(
                store._decode(transaction.read(store.open_path), strict=True),
                strict=True,
            )
            if [item for item in opens if item.id == position.id] != [position]:
                raise OpenIntentError("closing position changed before submission")
            unresolved_close_count(connection)
            connection.execute(
                "CREATE TABLE IF NOT EXISTS close_intent (client_id TEXT PRIMARY KEY, "
                "position_id TEXT NOT NULL, request_json TEXT NOT NULL, "
                "state TEXT NOT NULL CHECK(state IN ('UNKNOWN','RECORDED')))"
            )
            if connection.execute(
                "SELECT 1 FROM close_intent WHERE position_id=? AND state IS NOT 'RECORDED' LIMIT 1",
                (position.id,),
            ).fetchone():
                raise OpenIntentError(
                    "an unresolved closing requires exact-order reconciliation"
                )
            connection.execute(
                "INSERT INTO close_intent VALUES (?, ?, ?, 'UNKNOWN')",
                (client_id, position.id, payload),
            )
    except sqlite3.Error:
        raise OpenIntentError("closing intent could not be committed") from None


def validate_close_result(
    position: OpenPosition, trade: ClosedTrade, *, client_id: str
) -> None:
    """Validate cumulative acknowledgement identity before applying a first fill."""
    shared = (
        "id",
        "symbol",
        "market_type",
        "side",
        "owner",
        "dry_run",
        "open_client_order_id",
        "open_exchange_order_id",
        "entry_price",
        "leverage",
        "opened_at_ms",
    )
    if any(getattr(position, key) != getattr(trade, key) for key in shared) or (
        trade.dry_run is not False
        or trade.close_client_order_id != client_id
        or not isinstance(trade.close_exchange_order_id, str)
        or not trade.close_exchange_order_id.strip()
        or trade.exchange_status not in {"FILLED", "PARTIALLY_FILLED"}
    ):
        raise OpenIntentError("closing acknowledgement identity remains unresolved")
    for value in (trade.qty, trade.exit_price):
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise OpenIntentError("closing fill quantity or price is invalid")
    transmitted = float(f"{position.qty:.8f}")
    matches = math.isclose(trade.qty, transmitted, rel_tol=1e-12, abs_tol=1e-12)
    if (
        trade.qty > transmitted
        and not matches
        or trade.exchange_status == "FILLED"
        and not matches
    ):
        raise OpenIntentError("closing fill quantity differs from its intent")
    try:
        json.dumps(asdict(trade), allow_nan=False)
    except (TypeError, ValueError):
        raise OpenIntentError("closing accounting record is invalid") from None


def complete_close(
    store: PositionsStore,
    position: OpenPosition,
    trade: ClosedTrade,
    *,
    client_id: str,
    scope: BinanceExecutionScope,
    reduce_only: bool,
) -> None:
    """Release only a full acknowledgement proven present in the paired ledger."""
    payload = _request(store, position, client_id, scope, reduce_only)
    validate_close_result(position, trade, client_id=client_id)
    if trade.exchange_status != "FILLED":
        return
    try:
        with position_transaction(store.opening_intents, write=True) as transaction:
            connection = transaction.connection
            if connection is None:
                raise OpenIntentError("closing requires durable execution storage")
            store.opening_intents._bind_scope(connection, scope)
            unresolved_close_count(connection)
            opens = store._open_entries(
                store._decode(transaction.read(store.open_path), strict=True),
                strict=True,
            )
            closed = store._closed_entries(
                store._decode(transaction.read(store.ledger_path), strict=True),
                strict=True,
            )
            if any(item.id == position.id for item in opens) or [
                item for item in closed if item.close_client_order_id == client_id
            ] != [trade]:
                raise OpenIntentError("full close is not uniquely persisted")
            changed = connection.execute(
                "UPDATE close_intent SET state='RECORDED' WHERE client_id=? "
                "AND position_id=? AND request_json=? AND state='UNKNOWN'",
                (client_id, position.id, payload),
            )
            if changed.rowcount != 1:
                raise OpenIntentError("closing intent transition is not valid")
    except sqlite3.Error:
        raise OpenIntentError("closing completion could not be committed") from None
