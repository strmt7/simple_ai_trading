"""Apply retained cumulative Spot close evidence once, without orders or rearm."""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from .binance_close_intents import (
    _request,
    unresolved_close_count,
    validate_closing_client_id,
)
from .binance_close_recovery import _encode, _evidence, _record
from .binance_execution_scope import BinanceExecutionScope
from .binance_execution_storage import BinanceExecutionStorage
from .binance_open_intents import OpenIntentError
from .binance_owned_execution import _apply_close_order, close_trade_template
from .binance_runtime_control import BinanceRuntimeControl, RuntimeControlError
from .binance_spot_cash import entry_cash_receipt, validate_spot_cash_record
from .binance_terminal_fills import TerminalFillEvidence
from .position_transactions import position_transaction
from .positions import (
    ClosedTrade,
    OpenPosition,
    PositionsStore,
    write_learning_feedback,
)

_COLUMNS = (
    "client_id",
    "request_json",
    "source_sha256",
    "before_trade_json",
    "result_json",
    "application_sha256",
)


@dataclass(frozen=True)
class ClosingApplicationResult:
    client_id: str
    terminal_status: str
    executed_quantity: str
    remaining_quantity: str
    inventory_applied: bool = True
    close_intent_resolved: bool = True
    rearmed: bool = False


@dataclass(frozen=True)
class _Source:
    position: OpenPosition
    request: str
    state: str
    order: dict
    execution: TerminalFillEvidence
    digest: str


def _source(connection, store, scope, client_id) -> _Source:
    validate_closing_client_id(client_id)
    store.opening_intents._bind_scope(connection, scope)
    unresolved_close_count(connection)
    rows = connection.execute(
        "SELECT position_id, request_json, state FROM close_intent WHERE client_id=? LIMIT 2",
        (client_id,),
    ).fetchall()
    if len(rows) != 1 or rows[0][2] not in {"UNKNOWN", "RECORDED"}:
        raise OpenIntentError("exact closing intent is required for application")
    position_id, request, state = rows[0]
    payload = json.loads(request)
    position = OpenPosition(**payload["opening"]["position_template"])
    if (
        position.id != position_id
        or _request(store, position, client_id, scope, payload["reduce_only"])
        != request
    ):
        raise OpenIntentError("closing application request is inconsistent")
    record = _record(connection, client_id)
    if record is None or record[0] != request:
        raise OpenIntentError("exact retained terminal closing evidence is required")
    _, order_json, trades_json, encoded = record
    order = json.loads(order_json)
    evidence = _evidence(position, scope, client_id, order, json.loads(trades_json))
    if encoded != _encode(asdict(evidence)):
        raise OpenIntentError("retained terminal closing evidence changed")
    digest = hashlib.sha256(_encode(record).encode()).hexdigest()
    return _Source(position, request, state, order, evidence.execution, digest)


def closing_recovery_sha256(
    store: PositionsStore, *, scope: BinanceExecutionScope, client_id: str
) -> str:
    """Inspect the exact retained evidence digest without requests or ledger mutation."""
    try:
        with closing(store.opening_intents._connect()) as connection:
            return _source(connection, store, scope, client_id).digest
    except (sqlite3.Error, OSError, TypeError, ValueError, KeyError) as exc:
        if isinstance(exc, OpenIntentError):
            raise
        raise OpenIntentError("closing application evidence is unavailable") from None


def _previous_fills(source: _Source, prior: ClosedTrade) -> None:
    original = source.position
    for field in (
        "id",
        "symbol",
        "market_type",
        "side",
        "owner",
        "dry_run",
        "entry_price",
        "leverage",
        "opened_at_ms",
        "open_client_order_id",
        "open_exchange_order_id",
        "spot_entry_cash_receipt",
        "spot_gross_entry_quantity",
        "spot_entry_base_commission",
    ):
        if getattr(prior, field) != getattr(original, field):
            raise OpenIntentError("previous close differs from its original lot")
    validate_spot_cash_record(prior)
    if prior.close_exchange_order_id != source.execution.order.order_id:
        raise OpenIntentError("previous close belongs to a different exchange order")
    terminal = {fill.trade_id: fill for fill in source.execution.fills}
    seen = set()
    for fill in json.loads(prior.spot_close_cash_receipt)["fills"]:
        identity = str(fill.get("tradeId", ""))
        observed = terminal.get(identity)
        if observed is None or identity in seen:
            raise OpenIntentError("previous close fill lineage is unresolved")
        seen.add(identity)
        for field, expected in (
            ("qty", observed.quantity),
            ("price", observed.price),
            ("commission", observed.commission),
        ):
            if Fraction(fill[field]) != Fraction(expected):
                raise OpenIntentError(
                    "previous close fill conflicts with terminal evidence"
                )
        if fill["commissionAsset"] != observed.commission_asset:
            raise OpenIntentError("previous close commission asset changed")


def _projection(source: _Source, client_id: str, prior: ClosedTrade | None):
    original = source.position
    if (
        original.market_type != "spot"
        or original.dry_run is not False
        or entry_cash_receipt(original) is None
    ):
        raise OpenIntentError(
            "application requires an original native Spot cash receipt"
        )
    if prior is not None:
        if prior.close_client_order_id != client_id:
            raise OpenIntentError("previous close client identity changed")
        _previous_fills(source, prior)
    execution = source.execution
    if execution.executed_quantity == "0":
        if prior is not None:
            raise OpenIntentError(
                "zero terminal execution conflicts with recorded close"
            )
        cumulative, remaining = None, original
    else:
        order = dict(source.order)
        order["fills"] = [
            {
                "tradeId": fill.trade_id,
                "qty": fill.quantity,
                "price": fill.price,
                "quoteQty": fill.quote_quantity,
                "commission": fill.commission,
                "commissionAsset": fill.commission_asset,
            }
            for fill in execution.fills
        ]
        template = replace(
            close_trade_template(
                original, original.entry_price, "terminal-close-recovery"
            ),
            closed_at_ms=execution.order.updated_at_ms,
        )
        cumulative = _apply_close_order(template, order, client_id)
        validate_spot_cash_record(cumulative)
        remaining = PositionsStore.close_result_remainder(original, cumulative)
    result = ClosingApplicationResult(
        client_id,
        execution.order.status,
        execution.executed_quantity,
        "0" if remaining is None else str(remaining.qty),
    )
    return cumulative, remaining, result


def _application_record(connection, client_id):
    columns = tuple(
        row[1] for row in connection.execute("PRAGMA table_info(closing_application)")
    )
    if not columns:
        return None
    if columns != _COLUMNS:
        raise OpenIntentError("closing application schema is not recognized")
    rows = connection.execute(
        "SELECT request_json, source_sha256, before_trade_json, result_json, application_sha256 FROM closing_application WHERE client_id=? LIMIT 2",
        (client_id,),
    ).fetchall()
    if len(rows) > 1:
        raise OpenIntentError("closing application identity is ambiguous")
    if (
        rows
        and hashlib.sha256(_encode(rows[0][:-1]).encode()).hexdigest() != rows[0][-1]
    ):
        raise OpenIntentError("closing application checksum conflicts with its receipt")
    return rows[0] if rows else None


def apply_spot_closing_recovery(
    store: PositionsStore,
    *,
    client_id: str,
    scope: BinanceExecutionScope,
    expected_recovery_sha256: str,
    control: BinanceRuntimeControl,
    execution_enrollment_path: Path | None = None,
) -> ClosingApplicationResult:
    """Atomically reconcile one retained cumulative close while Stop remains latched.

    Replace the matching partial ledger projection; never append its cumulative
    quantity again. Unsupported fees, missing fill lineage or changed inventory
    reject. The application receipt, resolved intent and paired redo commit together.
    No credentials, account requests, orders, implicit enrollment or rearm occur.
    """
    validate_closing_client_id(client_id)
    if (
        not isinstance(scope, BinanceExecutionScope)
        or scope.market_type != "spot"
        or not isinstance(control, BinanceRuntimeControl)
        or not isinstance(expected_recovery_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_recovery_sha256) is None
    ):
        raise OpenIntentError(
            "closing application requires exact Spot scope and evidence digest"
        )
    binding = BinanceExecutionStorage(
        store.opening_intents, control, execution_enrollment_path
    )
    try:
        with control.stopped_recovery_guard():
            binding.verify(scope=scope)
            with position_transaction(
                store.opening_intents, write=True, create=False
            ) as transaction:
                connection = transaction.connection
                source = _source(connection, store, scope, client_id)
                if source.digest != expected_recovery_sha256:
                    raise OpenIntentError(
                        "closing evidence differs from the inspected digest"
                    )
                opens = store._open_entries(
                    store._decode(transaction.read(store.open_path), strict=True),
                    strict=True,
                )
                closed = store._closed_entries(
                    store._decode(transaction.read(store.ledger_path), strict=True),
                    strict=True,
                )
                matches = [
                    trade
                    for trade in closed
                    if trade.close_client_order_id == client_id
                ]
                if len(matches) > 1:
                    raise OpenIntentError(
                        "multiple closing projections require independent reconciliation"
                    )
                application = _application_record(connection, client_id)
                if application is not None:
                    request, digest, before, encoded, _ = application
                    prior = (
                        None if before == "null" else ClosedTrade(**json.loads(before))
                    )
                    cumulative, _, result = _projection(source, client_id, prior)
                    if (
                        source.state != "RECORDED"
                        or (request, digest, encoded)
                        != (source.request, source.digest, _encode(asdict(result)))
                        or matches != ([] if cumulative is None else [cumulative])
                    ):
                        raise OpenIntentError(
                            "retained closing application conflicts with accounting"
                        )
                else:
                    if source.state != "UNKNOWN":
                        raise OpenIntentError(
                            "recorded closing has no application receipt"
                        )
                    prior = matches[0] if matches else None
                    cumulative, remaining, result = _projection(
                        source, client_id, prior
                    )
                    expected = (
                        source.position
                        if prior is None
                        else store.close_result_remainder(source.position, prior)
                    )
                    if [item for item in opens if item.id == source.position.id] != (
                        [] if expected is None else [expected]
                    ):
                        raise OpenIntentError(
                            "owned inventory changed before closing application"
                        )
                    after_open = [
                        item for item in opens if item.id != source.position.id
                    ]
                    if remaining is not None:
                        after_open.append(remaining)
                    after_closed = [
                        item
                        for item in closed
                        if item.close_client_order_id != client_id
                    ]
                    if cumulative is not None:
                        after_closed.append(cumulative)
                    after_closed.sort(
                        key=lambda item: (
                            item.closed_at_ms,
                            item.id,
                            item.close_client_order_id,
                        )
                    )
                    connection.execute(
                        "CREATE TABLE IF NOT EXISTS closing_application (client_id TEXT PRIMARY KEY, request_json TEXT NOT NULL, source_sha256 TEXT NOT NULL, before_trade_json TEXT NOT NULL, result_json TEXT NOT NULL, application_sha256 TEXT NOT NULL)"
                    )
                    receipt = (
                        source.request,
                        source.digest,
                        _encode(None if prior is None else asdict(prior)),
                        _encode(asdict(result)),
                    )
                    connection.execute(
                        "INSERT INTO closing_application VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            client_id,
                            *receipt,
                            hashlib.sha256(_encode(receipt).encode()).hexdigest(),
                        ),
                    )
                    changed = connection.execute(
                        "UPDATE close_intent SET state='RECORDED' WHERE client_id=? AND request_json=? AND state='UNKNOWN'",
                        (client_id, source.request),
                    )
                    if changed.rowcount != 1:
                        raise OpenIntentError(
                            "closing changed before application commit"
                        )
                    transaction.replace(
                        {
                            store.open_path.name: store._encode(
                                [asdict(item) for item in after_open]
                            ),
                            store.ledger_path.name: store._encode(
                                [asdict(item) for item in after_closed]
                            ),
                        }
                    )
            write_learning_feedback(store)
            return result
    except (
        RuntimeControlError,
        sqlite3.Error,
        OSError,
        TypeError,
        ValueError,
        KeyError,
        ArithmeticError,
    ) as exc:
        if isinstance(exc, OpenIntentError):
            raise
        raise OpenIntentError(
            "closing application could not complete; inspect retained state"
        ) from None
