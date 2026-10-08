"""Fill-derived closing deltas, not account cash, inventory application or rearm."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import closing
from dataclasses import asdict, dataclass, field
from decimal import Decimal, localcontext
import json
import sqlite3

from .binance_close_intents import validate_closing_client_id
from .binance_close_recovery import _encode, _evidence, _pending, _record
from .binance_execution_scope import BinanceExecutionScope
from .binance_instrument_units import instrument_asset_units
from .binance_open_intents import OpenIntentError
from .binance_terminal_fills import _text
from .positions import PositionsStore


_COLUMNS = ("client_id", "request_json", "assets_json", "movement_json")


@dataclass(frozen=True)
class ClosingInventoryObservation:
    position_id: str
    client_order_id: str
    exchange_order_id: str
    base_asset: str
    quote_asset: str
    gross_executed_quantity: str
    gross_quote_quantity: str
    derivative_position_delta: str
    reported_realized_pnl: str | None
    reported_realized_pnl_asset: str | None
    asset_deltas: tuple[tuple[str, str], ...]
    native_commissions: tuple[tuple[str, str], ...]
    financially_qualified: bool = field(default=False, init=False)
    account_balance_qualified: bool = field(default=False, init=False)
    local_lot_pnl_qualified: bool = field(default=False, init=False)
    inventory_applied: bool = field(default=False, init=False)
    rearmed: bool = field(default=False, init=False)


def retain_closing_inventory(
    store: PositionsStore,
    *,
    client_id: str,
    scope: BinanceExecutionScope,
    instrument: Mapping[str, object],
    instrument_scope: BinanceExecutionScope,
) -> ClosingInventoryObservation | None:
    """Retain one parent-bound terminal close's native units without releasing it.

    Metadata must come from the declared execution scope. This boundary validates
    supplied linear crypto units, not their origin, freshness or trade eligibility.
    Futures trade-reported PnL retains the supplied marginAsset unit separately;
    it is not combined with fees into cash without independent settlement and
    attribution proof. Notional is not a principal movement. Partial closes
    retain only the executed delta, not a new lot.
    """
    validate_closing_client_id(client_id)
    if (
        not isinstance(scope, BinanceExecutionScope)
        or not isinstance(instrument_scope, BinanceExecutionScope)
        or instrument_scope != scope
    ):
        raise OpenIntentError("instrument metadata belongs to another execution scope")
    journal = store.opening_intents
    if not journal.path.exists():
        return None
    try:
        with closing(journal._connect(write=True)) as connection, connection:
            pending = _pending(connection, store, scope, client_id)
            if pending is None:
                return None
            position, request = pending
            assets = instrument_asset_units(instrument, position.symbol, scope)
            record = _record(connection, client_id)
            if record is None or record[0] != request:
                raise OpenIntentError("exact retained closing fills are required")
            _, order, trades, encoded = record
            evidence = _evidence(
                position, scope, client_id, json.loads(order), json.loads(trades)
            )
            if encoded != _encode(asdict(evidence)):
                raise OpenIntentError("terminal closing evidence changed")
            with localcontext() as context:
                # Existing execution decimal/page bounds keep all sums exact.
                context.prec = 100
                execution = evidence.execution
                quantity = Decimal(execution.executed_quantity)
                quote = Decimal(execution.quote_quantity)
                movements: dict[str, Decimal] = {}
                derivative = Decimal(0)
                pnl_asset = None
                if scope.market_type == "spot":
                    movements[assets["baseAsset"]] = -quantity
                    movements[assets["quoteAsset"]] = quote
                else:
                    derivative = -quantity if position.side == "LONG" else quantity
                    pnl_asset = assets["marginAsset"]
                for asset, fee in execution.commissions:
                    movements[asset] = movements.get(asset, Decimal(0)) - Decimal(fee)
                observation = ClosingInventoryObservation(
                    position.id,
                    client_id,
                    execution.order.order_id,
                    assets["baseAsset"],
                    assets["quoteAsset"],
                    execution.executed_quantity,
                    execution.quote_quantity,
                    _text(derivative),
                    evidence.futures_realized_pnl,
                    pnl_asset,
                    tuple(
                        (asset, _text(value))
                        for asset, value in sorted(movements.items())
                        if value
                    ),
                    execution.commissions,
                )
            assets_json, movement_json = _encode(assets), _encode(asdict(observation))
            columns = tuple(
                row[1]
                for row in connection.execute("PRAGMA table_info(closing_inventory)")
            )
            if columns and columns != _COLUMNS:
                raise OpenIntentError("closing inventory schema is not recognized")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS closing_inventory (client_id TEXT PRIMARY KEY, "
                "request_json TEXT NOT NULL, assets_json TEXT NOT NULL, movement_json TEXT NOT NULL)"
            )
            previous = connection.execute(
                "SELECT request_json, assets_json, movement_json FROM closing_inventory "
                "WHERE client_id=? LIMIT 2",
                (client_id,),
            ).fetchall()
            row = (request, assets_json, movement_json)
            if previous:
                if previous != [row]:
                    raise OpenIntentError("retained closing inventory conflicts")
            else:
                connection.execute(
                    "INSERT INTO closing_inventory VALUES (?, ?, ?, ?)",
                    (client_id, *row),
                )
            return observation
    except OpenIntentError:
        raise
    except (sqlite3.Error, OSError, KeyError, TypeError, ValueError):
        raise OpenIntentError("closing inventory could not be retained") from None
