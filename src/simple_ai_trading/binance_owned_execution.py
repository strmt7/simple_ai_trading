"""Shared durable Binance execution for owned positions and exact acknowledgments."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
import time
from typing import Mapping

from .api import BinanceAPIError, BinanceClient
from .binance_execution_scope import BinanceExecutionScope
from .binance_execution_storage import BinanceExecutionStorage
from .binance_open_intents import OpenIntentError
from .binance_order_responses import MarketOrderBinding
from .binance_runtime_control import (
    BINANCE_RUNTIME_LEASE_ENV,
    BinanceRuntimeControl,
    RuntimeControlError,
    RuntimeLeaseInterlock,
)
from .binance_spot_order_trades import observe_spot_order_trades
from .positions import ClosedTrade, OpenPosition, PositionsStore, bot_client_order_id


@dataclass(frozen=True, slots=True)
class BinanceOpeningResult:
    position: OpenPosition
    response: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class BinanceClosingResult:
    trade: ClosedTrade
    response: Mapping[str, object]


def _position_order_side(position: OpenPosition, *, close: bool = False) -> str:
    side = str(position.side or "").upper()
    if side == "SHORT":
        return "BUY" if close else "SELL"
    return "SELL" if close else "BUY"


def _order_text(order: Mapping[str, object], *names: str) -> str:
    for name in names:
        value = order.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _order_float(order: Mapping[str, object], *names: str) -> float:
    for name in names:
        value = order.get(name)
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if parsed == parsed and abs(parsed) != float("inf"):
            return parsed
    return 0.0


def _order_fill_details(
    order: Mapping[str, object],
    *,
    fallback_qty: float,
    fallback_price: float,
    market_type: str,
) -> tuple[float, float]:
    del fallback_qty, fallback_price
    from .binance_acknowledgements import acknowledged_fill

    fill = acknowledged_fill(order, market_type=market_type)
    return (0.0, 0.0) if fill is None else (float(fill.quantity), float(fill.price))


def _order_has_fill_evidence(order: Mapping[str, object]) -> bool:
    if _order_float(order, "executedQty") > 0.0:
        return True
    fills = order.get("fills")
    if not isinstance(fills, list):
        return False
    for fill in fills:
        if not isinstance(fill, Mapping):
            continue
        if _order_float(fill, "qty") > 0.0:
            return True
    return False


def _order_exchange_status(order: Mapping[str, object]) -> str:
    return _order_text(order, "status") or (
        "FILLED" if _order_has_fill_evidence(order) else "accepted"
    )


def _apply_open_order(
    position: OpenPosition, order: Mapping[str, object]
) -> OpenPosition:
    qty, entry_price = _order_fill_details(
        order,
        fallback_qty=position.qty,
        fallback_price=position.entry_price,
        market_type=position.market_type,
    )
    if qty <= 0.0 or entry_price <= 0.0:
        raise BinanceAPIError(
            "open order response did not include resolved execution fill"
        )
    gross_qty = qty
    gross_text = base_fee_text = ""
    cash_receipt = ""
    if position.market_type == "spot" and not position.dry_run:
        from decimal import Decimal, localcontext

        from .assets import symbol_base_for_supported_quote
        from .binance_spot_cash import encode_spot_cash_receipt
        from .binance_spot_receipts import (
            exact_spot_wire_quantity,
            spot_buy_received_quantity,
        )
        from .binance_terminal_fills import _text

        base = symbol_base_for_supported_quote(position.symbol)
        received = exact_spot_wire_quantity(
            spot_buy_received_quantity(
                order,
                symbol=position.symbol,
                base_asset=base,
                quote_asset=position.symbol[len(base) :],
            )
        )
        with localcontext() as context:
            context.prec = 128
            gross = Decimal(str(order["executedQty"]))
            gross_text, base_fee_text = _text(gross), _text(gross - received)
        qty = float(received)
        cash_receipt = encode_spot_cash_receipt(
            order,
            symbol=position.symbol,
            side="BUY",
            client_id=position.open_client_order_id,
            order_id=_order_text(order, "orderId"),
        )
    modeled_entry_fee_rate = max(0.0, float(position.entry_fees)) / max(
        float(position.notional),
        1e-18,
    )
    return replace(
        position,
        qty=qty,
        entry_price=entry_price,
        notional=qty * entry_price,
        entry_fees=gross_qty * entry_price * modeled_entry_fee_rate,
        spot_gross_entry_quantity=gross_text,
        spot_entry_base_commission=base_fee_text,
        spot_entry_cash_receipt=cash_receipt,
        open_exchange_order_id=_order_text(order, "orderId"),
        open_client_order_id=_order_text(order, "clientOrderId", "origClientOrderId")
        or position.open_client_order_id,
        exchange_status=_order_exchange_status(order),
    )


def _apply_close_order(
    trade: ClosedTrade,
    order: Mapping[str, object],
    close_client_order_id: str,
    *,
    exit_taker_fee_bps: float = 0.0,
) -> ClosedTrade:
    qty, exit_price = _order_fill_details(
        order,
        fallback_qty=trade.qty,
        fallback_price=trade.exit_price,
        market_type=trade.market_type,
    )
    if qty <= 0.0 or exit_price <= 0.0:
        raise BinanceAPIError(
            "close order response did not include resolved execution fill"
        )
    recorded = replace(
        trade,
        qty=qty,
        exit_price=exit_price,
        close_exchange_order_id=_order_text(order, "orderId"),
        close_client_order_id=_order_text(order, "clientOrderId", "origClientOrderId")
        or close_client_order_id,
        exchange_status=_order_exchange_status(order),
    )
    if trade.spot_entry_cash_receipt:
        from .binance_spot_cash import (
            encode_spot_cash_receipt,
            native_spot_cash_projection,
        )

        recorded = replace(
            recorded,
            spot_close_cash_receipt=encode_spot_cash_receipt(
                order,
                symbol=trade.symbol,
                side="SELL",
                client_id=close_client_order_id,
                order_id=recorded.close_exchange_order_id,
            ),
        )
        pnl, native_fees, pct = native_spot_cash_projection(recorded)
        return replace(
            recorded, realized_pnl=pnl, fees=native_fees, realized_pnl_pct=pct
        )
    fee_bps = float(exit_taker_fee_bps)
    if not math.isfinite(fee_bps) or fee_bps < 0.0:
        raise BinanceAPIError("close order fee rate must be finite and non-negative")
    allocation = min(1.0, qty / max(float(trade.qty), 1e-18))
    entry_fees = max(0.0, float(trade.fees)) * allocation
    exit_fees = qty * exit_price * fee_bps / 10_000.0
    fees = entry_fees + exit_fees
    if str(trade.side).upper() == "LONG":
        gross_pnl = (exit_price - float(trade.entry_price)) * qty
    else:
        gross_pnl = (float(trade.entry_price) - exit_price) * qty
    realized = gross_pnl - fees
    entry_notional = float(trade.entry_price) * qty
    return replace(
        recorded,
        realized_pnl=realized,
        realized_pnl_pct=(realized / entry_notional) if entry_notional > 0.0 else 0.0,
        fees=fees,
    )


def _submit_open_order(
    client: BinanceClient,
    position: OpenPosition,
    *,
    expected_scope: BinanceExecutionScope | None = None,
) -> Mapping[str, object]:
    if position.dry_run:
        raise BinanceAPIError("native opening requires a signed position")
    order_kwargs = {
        "dry_run": False,
        "leverage": position.leverage,
        "reduce_only": False,
        "client_order_id": position.open_client_order_id,
    }
    if hasattr(client, "get_max_leverage_for_notional"):
        order_kwargs["notional"] = abs(float(position.notional))
    if expected_scope is not None:
        order_kwargs["expected_scope"] = expected_scope
    try:
        order = client.place_order(
            position.symbol,
            _position_order_side(position, close=False),
            position.qty,
            **order_kwargs,
        )
    except BinanceAPIError:
        # The request may have reached the exchange before the network failed.
        # Query by the deterministic client id before treating the position as
        # untracked.
        binding = MarketOrderBinding(
            position.symbol,
            _position_order_side(position, close=False),
            f"{position.qty:.8f}",
            position.open_client_order_id,
            position.market_type,
            False,
        )
        order = client.get_order(
            position.symbol,
            orig_client_order_id=position.open_client_order_id,
            expected_order_binding=binding,
            **(
                {"expected_scope": expected_scope} if expected_scope is not None else {}
            ),
        )
        if (
            position.market_type == "spot"
            and order.get("status") == "FILLED"
            and "fills" not in order
        ):
            if expected_scope is None:
                raise OpenIntentError(
                    "Spot response recovery requires its execution scope"
                )
            order = observe_spot_order_trades(
                client,
                order,
                binding=binding,
                scope=expected_scope,
            )
    identities = [
        order[key] for key in ("clientOrderId", "origClientOrderId") if key in order
    ]
    if (
        not identities
        or any(identity != position.open_client_order_id for identity in identities)
        or order.get("symbol") != position.symbol
        or order.get("status") not in {"FILLED", "PARTIALLY_FILLED"}
        or (
            "side" in order
            and order["side"] != _position_order_side(position, close=False)
        )
    ):
        raise BinanceAPIError(
            "opening acknowledgement identity or status is unresolved"
        )
    return order


def _runtime_interlock(
    store: PositionsStore, interlock: RuntimeLeaseInterlock | None
) -> RuntimeLeaseInterlock:
    if interlock is None:
        interlock = RuntimeLeaseInterlock(
            BinanceRuntimeControl(Path(store.root) / "binance_runtime_control.sqlite3"),
            os.environ.get(BINANCE_RUNTIME_LEASE_ENV, ""),
        )
    if not isinstance(interlock, RuntimeLeaseInterlock) or not isinstance(
        interlock.control, BinanceRuntimeControl
    ):
        raise OpenIntentError(
            "owned execution requires enrolled Binance runtime control"
        )
    return interlock


def submit_durable_open(
    client: BinanceClient,
    position: OpenPosition,
    store: PositionsStore,
    *,
    interlock: RuntimeLeaseInterlock | None = None,
    execution_enrollment_path: Path | None = None,
) -> BinanceOpeningResult:
    """Persist uncertainty before transmission and inventory before admission release."""
    position = replace(position)
    journal = store.opening_intents
    scope = client.execution_scope()
    # Validate the request without creating storage or acquiring authority.
    journal._request(position, scope)
    reason = journal.entry_block_reason(positions_present=store.open_path.is_file())
    if reason:
        raise OpenIntentError(reason)
    authority = _runtime_interlock(store, interlock)
    storage = BinanceExecutionStorage(
        journal, authority.control, execution_enrollment_path
    )
    try:
        authority.assert_opening_allowed()
        storage.verify(scope=scope)
        with authority.submission_guard():
            storage.verify(scope=scope)
            journal.prepare(position, scope=scope, create=False)
            authority.assert_opening_allowed()
            storage.verify(scope=scope)
            response = _submit_open_order(client, position, expected_scope=scope)
            recorded = _apply_open_order(position, response)
            journal.validate_result(position, recorded)
            store.record_open(recorded)
            journal.record_complete(position, recorded, scope=scope)
    except RuntimeControlError as exc:
        raise OpenIntentError(
            "owned opening runtime authority is unavailable or blocked"
        ) from exc
    return BinanceOpeningResult(recorded, response)


def _submit_open_position(
    client: BinanceClient,
    position: OpenPosition,
    *,
    expected_scope: BinanceExecutionScope | None = None,
) -> OpenPosition:
    if position.dry_run:
        return position
    raise OpenIntentError("signed opening requires the durable owned execution gateway")


def _submit_durable_open_position(
    client: BinanceClient,
    position: OpenPosition,
    store: PositionsStore,
    *,
    interlock: RuntimeLeaseInterlock | None = None,
    execution_enrollment_path: Path | None = None,
) -> OpenPosition:
    return submit_durable_open(
        client,
        position,
        store,
        interlock=interlock,
        execution_enrollment_path=execution_enrollment_path,
    ).position


def _submit_close_order(
    client: BinanceClient,
    position: OpenPosition,
    *,
    reduce_only: bool,
    close_client_order_id: str | None = None,
    expected_scope: BinanceExecutionScope | None = None,
) -> Mapping[str, object]:
    if position.dry_run:
        raise BinanceAPIError("native closing requires a signed position")
    from .binance_spot_receipts import exact_spot_wire_quantity, native_spot_entry_net

    native_net = native_spot_entry_net(position)
    close_client_order_id = close_client_order_id or bot_client_order_id(
        position.id,
        "close",
    )
    try:
        order = client.place_order(
            position.symbol,
            _position_order_side(position, close=True),
            position.qty,
            dry_run=False,
            leverage=position.leverage,
            reduce_only=reduce_only,
            client_order_id=close_client_order_id,
            **(
                {"expected_scope": expected_scope} if expected_scope is not None else {}
            ),
        )
    except BinanceAPIError:
        binding = MarketOrderBinding(
            position.symbol,
            _position_order_side(position, close=True),
            f"{position.qty:.8f}",
            close_client_order_id,
            position.market_type,
            reduce_only,
        )
        order = client.get_order(
            position.symbol,
            orig_client_order_id=close_client_order_id,
            expected_order_binding=binding,
            **(
                {"expected_scope": expected_scope} if expected_scope is not None else {}
            ),
        )
        if (
            position.market_type == "spot"
            and order.get("status") in {"FILLED", "PARTIALLY_FILLED"}
            and "fills" not in order
        ):
            if expected_scope is None:
                raise OpenIntentError(
                    "Spot response recovery requires its execution scope"
                )
            order = observe_spot_order_trades(
                client,
                order,
                binding=binding,
                scope=expected_scope,
            )
    identities = [
        order[key] for key in ("clientOrderId", "origClientOrderId") if key in order
    ]
    if (
        not identities
        or any(value != close_client_order_id for value in identities)
        or order.get("symbol") != position.symbol
        or order.get("status") not in {"FILLED", "PARTIALLY_FILLED"}
        or order.get("side") != _position_order_side(position, close=True)
        or position.market_type == "futures"
        and (order.get("reduceOnly") is not True or order.get("positionSide") != "BOTH")
    ):
        raise BinanceAPIError(
            "closing acknowledgement identity or status is unresolved"
        )
    if native_net is not None:
        exact_spot_wire_quantity(order.get("executedQty"))
    return order


def submit_durable_close(
    client: BinanceClient,
    position: OpenPosition,
    trade: ClosedTrade,
    store: PositionsStore,
    *,
    reduce_only: bool,
    taker_fee_bps: float = 0.0,
    interlock: RuntimeLeaseInterlock | None = None,
    close_lock_timeout_seconds: float = 1.0,
) -> BinanceClosingResult:
    """Retain the owned close obligation, exact acknowledgement and paired inventory."""
    position, trade = replace(position), replace(trade)
    from .binance_close_intents import (
        complete_close,
        prepare_close,
        validate_close_result,
    )

    scope = client.execution_scope()
    authority = _runtime_interlock(store, interlock)
    try:
        with authority.control.close_guard(timeout_seconds=close_lock_timeout_seconds):
            client_id = _next_close_client_order_id(store, position)
            prepare_close(
                store,
                position,
                client_id=client_id,
                scope=scope,
                reduce_only=reduce_only,
            )
            response = _submit_close_order(
                client,
                position,
                reduce_only=reduce_only,
                close_client_order_id=client_id,
                expected_scope=scope,
            )
            recorded = _apply_close_order(
                trade, response, client_id, exit_taker_fee_bps=taker_fee_bps
            )
            validate_close_result(position, recorded, client_id=client_id)
            store.record_close_result(position, recorded)
            complete_close(
                store,
                position,
                recorded,
                client_id=client_id,
                scope=scope,
                reduce_only=reduce_only,
            )
    except RuntimeControlError as exc:
        raise OpenIntentError("owned closing interlock is unavailable") from exc
    return BinanceClosingResult(recorded, response)


def _submit_close_position(
    client: BinanceClient,
    position: OpenPosition,
    trade: ClosedTrade,
    *,
    reduce_only: bool,
    taker_fee_bps: float = 0.0,
    close_client_order_id: str | None = None,
    expected_scope: BinanceExecutionScope | None = None,
) -> ClosedTrade:
    if position.dry_run:
        return replace(trade, exchange_status="paper")
    raise OpenIntentError("signed closing requires the durable owned execution gateway")


def _submit_durable_close_position(
    client: BinanceClient,
    position: OpenPosition,
    trade: ClosedTrade,
    store: PositionsStore,
    *,
    reduce_only: bool,
    taker_fee_bps: float = 0.0,
    interlock: RuntimeLeaseInterlock | None = None,
    close_lock_timeout_seconds: float = 1.0,
) -> ClosedTrade:
    return submit_durable_close(
        client,
        position,
        trade,
        store,
        reduce_only=reduce_only,
        taker_fee_bps=taker_fee_bps,
        interlock=interlock,
        close_lock_timeout_seconds=close_lock_timeout_seconds,
    ).trade


def _next_close_client_order_id(
    store: PositionsStore,
    position: OpenPosition,
) -> str:
    prior_close_fills = sum(
        1 for trade in store.load_ledger() if trade.id == position.id
    )
    return bot_client_order_id(
        position.id,
        "close",
        attempt=prior_close_fills + 1,
    )


def close_trade_template(
    position: OpenPosition,
    mark_price: float,
    reason: str,
    *,
    clock=time.time,
    fees: float = 0.0,
) -> ClosedTrade:
    pnl = position.unrealized_pnl(mark_price)
    realized_pnl = pnl - fees
    entry_notional = float(position.entry_price) * float(position.qty)
    return ClosedTrade(
        id=position.id,
        symbol=position.symbol,
        market_type=position.market_type,
        side=position.side,
        qty=position.qty,
        entry_price=position.entry_price,
        exit_price=float(mark_price),
        leverage=position.leverage,
        opened_at_ms=position.opened_at_ms,
        closed_at_ms=int(clock() * 1000),
        realized_pnl=realized_pnl,
        realized_pnl_pct=(realized_pnl / entry_notional)
        if entry_notional > 0.0
        else 0.0,
        fees=fees,
        reason=reason,
        strategy_profile=position.strategy_profile,
        objective=position.objective,
        dry_run=position.dry_run,
        owner=position.owner,
        open_client_order_id=position.open_client_order_id,
        open_exchange_order_id=position.open_exchange_order_id,
        paper_open_intent_id=position.paper_open_intent_id,
        ai_review_mode=position.ai_review_mode,
        ai_review_case_id=position.ai_review_case_id,
        ai_review_status=position.ai_review_status,
        spot_gross_entry_quantity=position.spot_gross_entry_quantity,
        spot_entry_base_commission=position.spot_entry_base_commission,
        spot_entry_cash_receipt=position.spot_entry_cash_receipt,
    )
