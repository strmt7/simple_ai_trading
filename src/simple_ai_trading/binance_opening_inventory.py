"""Retain native execution deltas in the existing intent journal, without rearm.

These are fill-derived movements, not an account balance or available-to-sell
claim. Futures notional is not a principal cash purchase, and fee assets are
never converted into quote currency without separate valuation evidence.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from contextlib import closing
from dataclasses import asdict, dataclass, field
from decimal import Decimal, localcontext

from .binance_execution_scope import BinanceExecutionScope
from .binance_open_intents import BinanceOpenIntentJournal, OpenIntentError
from .binance_open_recovery import _encode, _read_record
from .binance_terminal_fills import _text, validate_terminal_fills

_COLUMNS = ("client_id", "request_json", "assets_json", "movement_json")


@dataclass(frozen=True)
class OpeningInventoryObservation:
    position_id: str
    client_order_id: str
    exchange_order_id: str
    base_asset: str
    quote_asset: str
    gross_executed_quantity: str
    gross_quote_quantity: str
    derivative_position_delta: str
    asset_deltas: tuple[tuple[str, str], ...]
    native_commissions: tuple[tuple[str, str], ...]
    account_balance_qualified: bool = field(default=False, init=False)
    inventory_applied: bool = field(default=False, init=False)
    rearmed: bool = field(default=False, init=False)


def _assets(
    instrument: Mapping[str, object], symbol: str, scope: BinanceExecutionScope
) -> dict[str, str]:
    """Admit explicit instrument units, never infer base from a suffix alone."""
    if not isinstance(instrument, Mapping):
        raise OpenIntentError("opening inventory requires explicit instrument metadata")
    base, quote = instrument.get("baseAsset"), instrument.get("quoteAsset")
    if (
        base not in ("BTC", "ETH", "SOL")
        or quote not in ("USDT", "USDC")
        or instrument.get("symbol") != symbol
        or symbol != base + quote
    ):
        raise OpenIntentError("opening instrument asset identity is inconsistent")
    result = {"symbol": symbol, "baseAsset": base, "quoteAsset": quote}
    if scope.market_type == "futures":
        if (
            instrument.get("contractType") != "PERPETUAL"
            or instrument.get("marginAsset") != quote
            or instrument.get("underlyingType") != "COIN"
        ):
            raise OpenIntentError("opening inventory requires linear crypto units")
        result.update(
            contractType="PERPETUAL", marginAsset=quote, underlyingType="COIN"
        )
    return result


def retain_opening_inventory(
    journal: BinanceOpenIntentJournal,
    *,
    scope: BinanceExecutionScope,
    instrument: Mapping[str, object],
    instrument_scope: BinanceExecutionScope,
) -> OpeningInventoryObservation | None:
    """Derive once from retained exact fills; preserve UNKNOWN and all source rows.

    The caller must supply metadata obtained under the declared instrument scope.
    This offline boundary verifies its identity and retains allowed unit fields;
    it does not establish metadata freshness or current trading eligibility.
    """
    if (
        not isinstance(scope, BinanceExecutionScope)
        or not isinstance(instrument_scope, BinanceExecutionScope)
        or instrument_scope != scope
    ):
        raise OpenIntentError("instrument metadata belongs to another execution scope")
    position = journal.pending_position(scope=scope)
    if position is None:
        return None
    assets = _assets(instrument, position.symbol, scope)
    request = journal._request(position, scope)
    try:
        with closing(journal._connect(write=True)) as connection, connection:
            journal._bind_scope(connection, scope)
            pending = connection.execute(
                "SELECT request_json FROM open_intent WHERE client_id=? AND state='UNKNOWN'",
                (position.open_client_order_id,),
            ).fetchone()
            if pending != (request,):
                raise OpenIntentError("opening changed before inventory retention")
            record = _read_record(connection, position.open_client_order_id)
            if record is None or record[0] != request:
                raise OpenIntentError("exact retained terminal fills are required")
            _, order, trades, encoded = record
            evidence = validate_terminal_fills(
                position, scope, json.loads(order), json.loads(trades)
            )
            if encoded != _encode(asdict(evidence)):
                raise OpenIntentError("terminal fill evidence changed")
            with localcontext() as context:
                context.prec = 100
                quantity = Decimal(evidence.executed_quantity)
                quote = Decimal(evidence.quote_quantity)
                movements: dict[str, Decimal] = {}
                derivative = Decimal(0)
                if scope.market_type == "spot":
                    movements[assets["baseAsset"]] = quantity
                    movements[assets["quoteAsset"]] = -quote
                else:
                    derivative = quantity if position.side == "LONG" else -quantity
                for asset, fee in evidence.commissions:
                    movements[asset] = movements.get(asset, Decimal(0)) - Decimal(fee)
                observation = OpeningInventoryObservation(
                    position.id,
                    position.open_client_order_id,
                    evidence.order.order_id,
                    assets["baseAsset"],
                    assets["quoteAsset"],
                    evidence.executed_quantity,
                    evidence.quote_quantity,
                    _text(derivative),
                    tuple(
                        (asset, _text(value))
                        for asset, value in sorted(movements.items())
                        if value
                    ),
                    evidence.commissions,
                )
            assets_json, movement_json = _encode(assets), _encode(asdict(observation))
            columns = tuple(
                row[1]
                for row in connection.execute("PRAGMA table_info(opening_inventory)")
            )
            if columns and columns != _COLUMNS:
                raise OpenIntentError("opening inventory schema is not recognized")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS opening_inventory (client_id TEXT PRIMARY KEY, "
                "request_json TEXT NOT NULL, assets_json TEXT NOT NULL, movement_json TEXT NOT NULL)"
            )
            previous = connection.execute(
                "SELECT request_json, assets_json, movement_json FROM opening_inventory "
                "WHERE client_id=? LIMIT 2",
                (position.open_client_order_id,),
            ).fetchall()
            if previous:
                if previous != [(request, assets_json, movement_json)]:
                    raise OpenIntentError("retained opening inventory conflicts")
            else:
                connection.execute(
                    "INSERT INTO opening_inventory VALUES (?, ?, ?, ?)",
                    (
                        position.open_client_order_id,
                        request,
                        assets_json,
                        movement_json,
                    ),
                )
            return observation
    except OpenIntentError:
        raise
    except (sqlite3.Error, OSError, KeyError, TypeError, ValueError):
        raise OpenIntentError("opening inventory could not be retained") from None
