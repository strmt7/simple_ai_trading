"""One-shot Spot diagnostics using caller-owned runtime authority and paired intents."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
from contextlib import closing
from decimal import Decimal, localcontext
from fractions import Fraction
import math
import os
from pathlib import Path
import re
import sqlite3
import uuid

from .api import BinanceAPIError, BinanceClient
from .assets import is_supported_major_symbol, symbol_base_for_supported_quote
from .binance_execution_storage import BinanceExecutionStorage
from .binance_open_intents import OpenIntentError
from .binance_runtime_control import (
    BINANCE_RUNTIME_LEASE_ENV,
    BinanceRuntimeControl,
    RuntimeControlError,
    RuntimeLeaseInterlock,
)
from .binance_spot_receipts import exact_spot_wire_quantity
from .binance_spot_roundtrip_execution import SpotRoundtripExecution
from .binance_spot_roundtrip_intents import (
    SpotRoundtripJournal,
    SpotRoundtripRequest,
    _movements,
    _validate_record,
)
from .binance_terminal_fills import _decimal
from .positions import PositionsStore
from .types import RuntimeConfig


LEASE_ENV = BINANCE_RUNTIME_LEASE_ENV


class RoundtripCommandError(ValueError):
    """An actionable local command error containing no provider or credential text."""


def _text(value: Fraction) -> str:
    with localcontext() as context:
        context.prec = 128
        text = format(Decimal(value.numerator) / Decimal(value.denominator), "f")
    if Fraction(text) != value:
        raise RoundtripCommandError("The allocation cannot be represented exactly.")
    _decimal(text)
    return text


def _rules(client: BinanceClient, symbol: str) -> Mapping:
    info = client.ensure_symbol(symbol)
    base = symbol_base_for_supported_quote(symbol)
    if (
        not isinstance(info, Mapping)
        or info.get("symbol") != symbol
        or info.get("status") != "TRADING"
        or info.get("baseAsset") != base
        or info.get("quoteAsset") != symbol[len(base) :]
        or not isinstance(info.get("orderTypes"), list)
        or "MARKET" not in info["orderTypes"]
        or info.get("quoteOrderQtyMarketAllowed") is not True
        or type(info.get("quoteAssetPrecision")) is not int
        or not 0 <= info["quoteAssetPrecision"] <= 30
    ):
        raise RoundtripCommandError(
            "Spot MARKET/quote-principal instrument capability is unqualified."
        )
    _sale_rules(info)
    return info


def _sale_rules(info: Mapping) -> list[tuple[Fraction, Fraction, Fraction]]:
    filters = info.get("filters")
    if not isinstance(filters, list) or len(filters) > 64:
        raise RoundtripCommandError("Spot quantity filters are missing or unbounded.")
    seen, result = set(), []
    for row in filters:
        if not isinstance(row, Mapping) or not isinstance(row.get("filterType"), str):
            raise RoundtripCommandError("Spot quantity filter evidence is malformed.")
        kind = row["filterType"]
        if kind in seen:
            raise RoundtripCommandError("Spot filter identity is duplicated.")
        seen.add(kind)
        if kind in {"LOT_SIZE", "MARKET_LOT_SIZE"}:
            minimum, maximum, step = (
                Fraction(_decimal(row.get(key)))
                for key in ("minQty", "maxQty", "stepSize")
            )
            if maximum and minimum > maximum:
                raise RoundtripCommandError("Spot quantity limits conflict.")
            result.append((minimum, maximum, step))
    if "LOT_SIZE" not in seen:
        raise RoundtripCommandError("Spot LOT_SIZE evidence is missing.")
    return result


def _sale_amount(info: Mapping, limit: Fraction, *, exact: bool) -> str:
    rules = _sale_rules(info)
    step = Fraction(1, 100_000_000)
    for _, _, native_step in rules:
        if native_step:
            step = Fraction(
                math.lcm(step.numerator, native_step.numerator),
                math.gcd(step.denominator, native_step.denominator),
            )
    amount = (limit // step) * step
    if exact and amount != limit:
        raise RoundtripCommandError(
            "The explicit base allocation is not aligned to Spot wire/lot units."
        )
    if amount <= 0 or any(
        amount < minimum or maximum and amount > maximum
        for minimum, maximum, _ in rules
    ):
        raise RoundtripCommandError(
            "The receipt-owned sale cannot meet Spot quantity limits."
        )
    text = _text(amount)
    exact_spot_wire_quantity(text)
    return text


def _quote_amount(info: Mapping, limit: Fraction, *, exact: bool) -> str:
    scale = 10 ** info["quoteAssetPrecision"]
    amount = Fraction(limit * scale // 1, scale)
    if amount <= 0 or exact and amount != limit:
        raise RoundtripCommandError(
            "The explicit quote allocation is not aligned to Spot quote units."
        )
    return _text(amount)


def parse_roundtrip_allocation(args: argparse.Namespace) -> tuple[str, Fraction]:
    """Resolve explicit units without consulting a price, wallet or native service."""
    quantity, quote = (
        getattr(args, "quantity", None),
        getattr(args, "quote_budget", None),
    )
    if (quantity is None) == (quote is None):
        raise RoundtripCommandError(
            "Supply exactly one explicit --quantity or --quote-budget allocation."
        )
    mode = str(getattr(args, "mode", "auto"))
    selected = "buy-sell" if quote is not None else "sell-buy"
    if mode not in {"auto", selected}:
        raise RoundtripCommandError(
            "The roundtrip mode disagrees with the allocation's base/quote units."
        )
    raw = quote if quote is not None else quantity
    if not isinstance(raw, str):
        raise RoundtripCommandError(
            "Roundtrip allocation inputs must remain exact decimal strings."
        )
    amount = Fraction(_decimal(raw))
    if amount <= 0:
        raise RoundtripCommandError(
            "The explicit roundtrip allocation must be positive."
        )
    return selected, amount


def _request(
    allocation: tuple[str, Fraction], symbol: str, info: Mapping, operation_id: str
) -> SpotRoundtripRequest:
    mode, amount = allocation
    buy = mode == "buy-sell"
    principal = (
        _quote_amount(info, amount, exact=True)
        if buy
        else _sale_amount(info, amount, exact=True)
    )
    return SpotRoundtripRequest(
        operation_id,
        symbol,
        mode,
        principal,
        "0" if buy else principal,
        principal if buy else "0",
    )


def _describe(record: dict) -> dict:
    request = _validate_record(record)
    movements = dict.fromkeys(request.allocations, Fraction(0))
    for first, field in ((True, "first"), (False, "second")):
        if record[field] is not None:
            delta = _movements(
                request, record[field], first=first, amount=record["second_amount"]
            )
            movements = {
                asset: value + delta[asset] for asset, value in movements.items()
            }
    complete = record["phase"] in {"SECOND_TERMINAL", "COMPLETE"}
    observed = (
        {asset: str(value) for asset, value in movements.items()}
        if record["first"] is not None
        else {}
    )
    if record["phase"] in {"FIRST_UNKNOWN", "SECOND_UNKNOWN"}:
        next_action = "recover"
    elif record["phase"] == "FIRST_TERMINAL":
        next_action = (
            "continue"
            if record["first"]["status"] == "FILLED"
            else "owned_resolution_required"
        )
    elif record["phase"] == "SECOND_TERMINAL":
        next_action = (
            "complete"
            if record["second"]["status"] == "FILLED"
            else "owned_resolution_required"
        )
    else:
        next_action = None
    return {
        "operation_id": request.operation_id,
        "symbol": request.symbol,
        "mode": request.mode,
        "phase": record["phase"],
        "operation_obligation_pending": record["phase"] != "COMPLETE",
        "asset_movements_complete": complete,
        "observed_asset_movements": observed,
        "net_asset_movements": observed if complete else None,
        "asset_movement_units": "exact rational asset units",
        "flat_exposure_qualified": False,
        "profit_qualified": False,
        "next_action": next_action,
    }


def run_spot_roundtrip(
    args: argparse.Namespace,
    *,
    runtime: RuntimeConfig | None,
    client_factory: Callable[[RuntimeConfig], BinanceClient],
) -> tuple[int, dict]:
    """Return public diagnostics; create no worker, enrollment, lease or restart hook."""
    action = str(getattr(args, "action", "execute"))
    payload = {
        "command": "spot-roundtrip",
        "action": action,
        "grants_runtime_authority": False,
        "lease_managed_by_caller": True,
    }
    client, scope, journal, operation_id = (
        None,
        None,
        None,
        getattr(args, "operation_id", None),
    )
    try:
        root = Path(getattr(args, "data_dir", "data/autonomous"))
        store = PositionsStore(root)
        journal = SpotRoundtripJournal(store.opening_intents)
        control = BinanceRuntimeControl(
            Path(
                getattr(args, "runtime_control", None)
                or root / "binance_runtime_control.sqlite3"
            ),
            enrollment_path=getattr(args, "runtime_enrollment", None),
        )
        storage = BinanceExecutionStorage(
            journal.journal, control, getattr(args, "execution_enrollment", None)
        )
        if action == "inspect-storage":
            payload.update(storage.inspect())
            return 0, payload
        if action == "status":
            payload["new_exposure_authorized"] = False
            payload.update(
                execution_store_exists=journal.journal.path.is_file(),
                execution_block_reason=journal.journal.entry_block_reason(
                    positions_present=store.open_path.is_file()
                ),
            )
            try:
                control.inspect_storage()
                snapshot = control.snapshot()
                payload.update(
                    control_available=True,
                    runtime_state=snapshot.state,
                    paused=snapshot.paused,
                    stop_epoch=snapshot.stop_epoch,
                )
            except (RuntimeControlError, OSError, sqlite3.Error, ValueError):
                payload.update(control_available=False, runtime_state="unavailable")
            return 0, payload
        if action not in {
            "execute",
            "recover",
            "continue",
            "complete",
            "provision-storage",
            "recover-storage",
        }:
            raise RoundtripCommandError("Unsupported roundtrip action.")
        if not getattr(args, "yes", False):
            raise RoundtripCommandError(
                "Pass --yes to confirm this testnet/Demo execution or reconciliation action."
            )
        if runtime is None:
            raise RoundtripCommandError(
                "Runtime configuration is required for this action."
            )
        if runtime.market_type != "spot" or not (runtime.testnet or runtime.demo):
            raise RoundtripCommandError(
                "Roundtrip requires official Spot testnet/Demo configuration."
            )
        if action == "execute" and not is_supported_major_symbol(runtime.symbol):
            raise RoundtripCommandError(
                "Roundtrip supports only BTC/ETH/SOL with USDC/USDT."
            )
        if not runtime.api_key or not runtime.api_secret:
            raise RoundtripCommandError(
                "Roundtrip requires configured Binance credentials; no history lookup is permitted."
            )
        if action in {"provision-storage", "recover-storage"}:
            if any(
                getattr(args, name, None) is not None
                for name in (
                    "quantity",
                    "quote_budget",
                    "operation_id",
                )
            ):
                raise RoundtripCommandError(
                    "Storage setup does not accept order allocations or IDs."
                )
            try:
                client = client_factory(runtime)
            except (TypeError, ValueError, OverflowError):
                raise RoundtripCommandError(
                    "Binance client configuration is invalid."
                ) from None
            with closing(client.session):
                scope = client.execution_scope()
                if action == "provision-storage":
                    result = storage.provision(
                        scope=scope,
                        expected_snapshot_sha256=getattr(
                            args, "expected_snapshot_sha256", None
                        ),
                    )
                else:
                    result = storage.recover(
                        scope=scope,
                        expected_store_id=getattr(args, "expected_store_id", None),
                        expected_record_sha256=getattr(
                            args, "expected_record_sha256", None
                        ),
                    )
                payload.update(result)
                return 0, payload
        if action != "execute" and (
            getattr(args, "quantity", None) is not None
            or getattr(args, "quote_budget", None) is not None
        ):
            raise RoundtripCommandError(
                "Recovery/continuation must use the stored allocation, not new amount flags."
            )
        if operation_id is None and action == "execute":
            operation_id = uuid.uuid4().hex[:29]
        if (
            not isinstance(operation_id, str)
            or re.fullmatch(r"[0-9a-f]{29}", operation_id) is None
        ):
            raise RoundtripCommandError(
                "Supply the exact --operation-id for an existing operation."
            )
        payload["operation_id"] = operation_id
        lease = os.environ.get(LEASE_ENV, "")
        if action == "execute":
            allocation = parse_roundtrip_allocation(args)
            if re.fullmatch(r"[0-9a-f]{32}", lease) is None:
                raise RoundtripCommandError(
                    "An existing caller-owned Binance runtime lease is required in the documented environment variable."
                )
            control.assert_opening_allowed(lease)
            reason = journal.journal.entry_block_reason(
                positions_present=store.open_path.is_file()
            )
            if reason:
                raise RoundtripCommandError(
                    "Unresolved or unreadable execution state blocks a new operation."
                )
        try:
            client = client_factory(runtime)
        except (TypeError, ValueError, OverflowError):
            raise RoundtripCommandError(
                "Binance client configuration is invalid."
            ) from None
        with closing(client.session):
            scope = client.execution_scope()
            if action == "execute":
                storage.verify(scope=scope)
            gateway = SpotRoundtripExecution(
                client,
                journal,
                RuntimeLeaseInterlock(control, lease),
                execution_enrollment_path=getattr(args, "execution_enrollment", None),
            )
            if action == "execute":
                request = _request(
                    allocation,
                    runtime.symbol,
                    _rules(client, runtime.symbol),
                    operation_id,
                )
                record = gateway.submit_first(request)
            else:
                record = journal.operation(operation_id, scope=scope)
                if record is None:
                    raise RoundtripCommandError(
                        "The operation is missing; recovery never creates fresh state."
                    )
            if action == "recover" and record["phase"] in {
                "FIRST_UNKNOWN",
                "SECOND_UNKNOWN",
            }:
                record = gateway.recover(operation_id)
            elif (
                action in {"execute", "continue"}
                and record["phase"] == "FIRST_TERMINAL"
                and record["first"]["status"] == "FILLED"
            ):
                request = _validate_record(record)
                info = _rules(client, request.symbol)
                delta = _movements(request, record["first"], first=True)
                amount = (
                    _sale_amount(info, delta[request.base], exact=False)
                    if request.mode == "buy-sell"
                    else _quote_amount(info, delta[request.quote], exact=False)
                )
                record = gateway.submit_second(operation_id, amount=amount)
            if (
                action in {"execute", "continue", "complete"}
                and record["phase"] == "SECOND_TERMINAL"
                and record["second"]["status"] == "FILLED"
            ):
                revision, current = journal.pending(scope=scope)
                if current != record:
                    raise OpenIntentError(
                        "roundtrip changed before explicit completion"
                    )
                journal.complete(operation_id, scope=scope, expected_revision=revision)
                record = journal.operation(operation_id, scope=scope)
            payload.update(_describe(record))
            return (
                0 if record["phase"] == "COMPLETE" or action == "recover" else 2
            ), payload
    except (
        RoundtripCommandError,
        BinanceAPIError,
        OpenIntentError,
        RuntimeControlError,
        OSError,
        sqlite3.Error,
        ValueError,
    ) as exc:
        payload.update(
            error_kind=type(exc).__name__,
            error=str(exc)
            if isinstance(exc, RoundtripCommandError)
            else "Execution or evidence was rejected; reconcile the durable operation before any new write.",
        )
        if journal is not None and scope is not None and operation_id is not None:
            try:
                record = journal.operation(operation_id, scope=scope)
                if record is not None:
                    payload.update(_describe(record))
            except (OpenIntentError, OSError, sqlite3.Error, ValueError):
                payload.update(phase="unavailable", operation_obligation_pending=True)
        return 2, payload
