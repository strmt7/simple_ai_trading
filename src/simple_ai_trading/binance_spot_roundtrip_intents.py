"""Shared-namespace write-ahead state for a two-order Spot diagnostic.

This journal does not dispatch orders, grant wallet ownership, qualify future
fees or make the diagnostic riskless. Its caller must admit those separately.
Both exact request units and complete observed asset movements remain durable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import re
import sqlite3

from .assets import is_supported_major_symbol, symbol_base_for_supported_quote
from .binance_acknowledgements import acknowledged_fill
from .binance_close_intents import unresolved_close_count
from .binance_execution_scope import BinanceExecutionScope, parse_execution_id
from .binance_open_intents import BinanceOpenIntentJournal, OpenIntentError
from .binance_order_responses import MarketOrderBinding, SpotQuoteOrderBinding
from .binance_spot_receipts import exact_spot_wire_quantity
from .binance_terminal_fills import _decimal
from .position_transactions import position_transaction


_PHASES = {
    "FIRST_UNKNOWN",
    "FIRST_TERMINAL",
    "SECOND_UNKNOWN",
    "SECOND_TERMINAL",
    "COMPLETE",
}
_TERMINAL = {"FILLED", "CANCELED", "EXPIRED", "EXPIRED_IN_MATCH", "REJECTED"}
_ORDER_COLUMNS = ("operation_id", "phase", "record_json", "record_sha256")
_CONTROL_COLUMNS = (
    "id",
    "revision",
    "completed_count",
    "active_operation_id",
    "active_sha256",
)
_FIELDS = (
    "symbol",
    "side",
    "type",
    "status",
    "orderId",
    "clientOrderId",
    "origClientOrderId",
    "origQty",
    "origQuoteOrderQty",
    "executedQty",
    "cummulativeQuoteQty",
    "fills",
)
_FILL_FIELDS = ("qty", "price", "quoteQty", "commission", "commissionAsset", "tradeId")
_RECORD_SELECT = (
    "SELECT CASE WHEN length(phase)<=32 THEN phase END, "
    "CASE WHEN typeof(record_json)='text' AND length(record_json)<=2100000 "
    "THEN record_json END, "
    "CASE WHEN typeof(record_sha256)='text' AND length(record_sha256)=64 "
    "THEN record_sha256 END FROM spot_roundtrip WHERE operation_id=?"
)


def _json(value: object) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    except (TypeError, ValueError, RecursionError):
        raise OpenIntentError("roundtrip evidence is not bounded JSON") from None
    if len(encoded) > 2_100_000:
        raise OpenIntentError("roundtrip record exceeds its storage bound")
    return encoded


def _sha(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SpotRoundtripRequest:
    operation_id: str
    symbol: str
    mode: str
    first_amount: str
    initial_base_allocation: str
    initial_quote_allocation: str
    fee_asset_allocations: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if (
            not isinstance(self.operation_id, str)
            or re.fullmatch(r"[0-9a-f]{29}", self.operation_id) is None
            or not is_supported_major_symbol(self.symbol)
            or self.mode not in {"buy-sell", "sell-buy"}
        ):
            raise OpenIntentError("roundtrip operation identity or mode is invalid")
        base, quote = (
            _decimal(self.initial_base_allocation),
            _decimal(self.initial_quote_allocation),
        )
        if (
            not isinstance(self.fee_asset_allocations, tuple)
            or len(self.fee_asset_allocations) > 16
        ):
            raise OpenIntentError(
                "roundtrip fee allocations must be bounded immutable pairs"
            )
        previous_asset = ""
        for allocation in self.fee_asset_allocations:
            if not isinstance(allocation, tuple) or len(allocation) != 2:
                raise OpenIntentError(
                    "roundtrip fee allocation must be an asset/amount pair"
                )
            asset, allocated = allocation
            if (
                not isinstance(asset, str)
                or re.fullmatch(r"[A-Z0-9]{1,32}", asset) is None
                or asset in {self.base, self.quote}
                or asset <= previous_asset
            ):
                raise OpenIntentError(
                    "roundtrip fee assets must be distinct, sorted extra assets"
                )
            _decimal(allocated)
            previous_asset = asset
        binding = self.binding(first=True)
        amount = _decimal(self.first_amount)
        if amount <= 0 or amount > (
            quote if isinstance(binding, SpotQuoteOrderBinding) else base
        ):
            raise OpenIntentError(
                "first roundtrip principal exceeds its explicit allocation"
            )

    @property
    def base(self) -> str:
        return symbol_base_for_supported_quote(self.symbol)

    @property
    def quote(self) -> str:
        return self.symbol[len(self.base) :]

    @property
    def allocations(self) -> dict[str, Fraction]:
        """Explicit asset units; no FX valuation or wallet-ownership inference."""
        return {
            self.base: Fraction(self.initial_base_allocation),
            self.quote: Fraction(self.initial_quote_allocation),
            **{asset: Fraction(amount) for asset, amount in self.fee_asset_allocations},
        }

    def binding(self, *, first: bool, amount: str | None = None):
        if type(first) is not bool:
            raise OpenIntentError("roundtrip request leg must be explicit")
        side = "BUY" if self.mode == "buy-sell" else "SELL"
        if not first:
            side = "SELL" if side == "BUY" else "BUY"
        amount = self.first_amount if first else amount
        client_id = ("sait-o-" if side == "BUY" else "sait-c-") + self.operation_id
        try:
            if side == "BUY":
                return SpotQuoteOrderBinding(self.symbol, amount, client_id)
            exact_spot_wire_quantity(_decimal(amount))
            return MarketOrderBinding(
                self.symbol, side, amount, client_id, "spot", False
            )
        except (TypeError, ValueError, OverflowError):
            raise OpenIntentError(
                "roundtrip request amount cannot be bound exactly"
            ) from None


def _selected(order: Mapping[str, object]) -> dict:
    if not isinstance(order, Mapping):
        raise OpenIntentError("roundtrip observation must be an order object")
    value = {key: order[key] for key in _FIELDS if key in order}
    rows = value.get("fills")
    if (
        not isinstance(rows, list)
        or len(rows) > 1000
        or any(not isinstance(row, Mapping) for row in rows)
    ):
        raise OpenIntentError("roundtrip requires bounded native fill rows")
    value["fills"] = [
        {key: row[key] for key in _FILL_FIELDS if key in row} for row in rows
    ]
    if len(_json(value)) > 1_000_000:
        raise OpenIntentError("roundtrip observation exceeds its storage bound")
    return value


def _movements(
    request: SpotRoundtripRequest,
    order: dict,
    *,
    first: bool,
    amount: str | None = None,
) -> dict[str, Fraction]:
    binding = request.binding(first=first, amount=amount)
    try:
        binding.validate(order)
    except ValueError as exc:
        raise OpenIntentError(str(exc)) from None
    status = order.get("status")
    if status not in _TERMINAL | {"NEW", "PARTIALLY_FILLED"}:
        raise OpenIntentError("roundtrip native order status is unrecognized")
    quantity, cash = (
        _decimal(order.get("executedQty")),
        _decimal(order.get("cummulativeQuoteQty")),
    )
    if "origQty" in order:
        _decimal(order["origQty"])
    if quantity == 0:
        if cash != 0 or order["fills"] or status in {"FILLED", "PARTIALLY_FILLED"}:
            raise OpenIntentError(
                "roundtrip zero execution has inconsistent cash or status"
            )
    else:
        fill = acknowledged_fill(order, market_type="spot")
        if fill is None or fill.quote_quantity is None or not order["fills"]:
            raise OpenIntentError("roundtrip lacks complete native cash/fill evidence")
    fees = dict.fromkeys(request.allocations, Fraction(0))
    for row in order["fills"]:
        asset = row.get("commissionAsset")
        commission = Fraction(_decimal(row.get("commission")))
        if (
            not isinstance(asset, str)
            or re.fullmatch(r"[A-Z0-9]{1,32}", asset) is None
            or commission
            and asset not in fees
        ):
            raise OpenIntentError(
                "roundtrip commission is missing or uses an unallocated asset"
            )
        if asset in fees:
            fees[asset] += commission
    buy = isinstance(binding, SpotQuoteOrderBinding)
    movements = {asset: -fee for asset, fee in fees.items()}
    movements[request.base] += Fraction(quantity) if buy else -Fraction(quantity)
    movements[request.quote] += -Fraction(cash) if buy else Fraction(cash)
    return movements


def _validate_record(record: dict) -> SpotRoundtripRequest:
    if (
        not isinstance(record, dict)
        or set(record)
        != {"format", "request", "scope", "phase", "first", "second_amount", "second"}
        or type(record["format"]) is not int
        or record["format"] not in {1, 2}
        or record["phase"] not in _PHASES
    ):
        raise OpenIntentError("roundtrip record schema is invalid")
    request_fields = record["request"]
    legacy_fields = {
        "operation_id",
        "symbol",
        "mode",
        "first_amount",
        "initial_base_allocation",
        "initial_quote_allocation",
    }
    expected = legacy_fields | (
        {"fee_asset_allocations"} if record["format"] == 2 else set()
    )
    if not isinstance(request_fields, dict) or set(request_fields) != expected:
        raise OpenIntentError("roundtrip request fields differ from its record format")
    values = dict(request_fields)
    if record["format"] == 2:
        rows = values["fee_asset_allocations"]
        if (
            not isinstance(rows, list)
            or not 1 <= len(rows) <= 16
            or any(not isinstance(row, list) or len(row) != 2 for row in rows)
        ):
            raise OpenIntentError("roundtrip stored fee allocations are not canonical")
        values["fee_asset_allocations"] = tuple(tuple(row) for row in rows)
    request = SpotRoundtripRequest(**values)
    scope = BinanceExecutionScope(**record["scope"])
    if scope.market_type != "spot":
        raise OpenIntentError("roundtrip scope must be Spot")
    phase = record["phase"]
    if phase != "FIRST_UNKNOWN" and record["first"] is None:
        raise OpenIntentError("roundtrip phase lacks its first observation")
    balances = request.allocations
    if record["first"] is not None:
        if _selected(record["first"]) != record["first"]:
            raise OpenIntentError("roundtrip first observation is not canonical")
        first_delta = _movements(request, record["first"], first=True)
        balances = {asset: balances[asset] + first_delta[asset] for asset in balances}
        if any(value < 0 for value in balances.values()):
            raise OpenIntentError(
                "roundtrip first execution exceeds its allocated assets"
            )
        terminal = record["first"]["status"] in _TERMINAL
        if terminal != (phase != "FIRST_UNKNOWN"):
            raise OpenIntentError(
                "roundtrip first phase differs from its native status"
            )
    if phase in {"FIRST_UNKNOWN", "FIRST_TERMINAL"}:
        if record["second_amount"] is not None or record["second"] is not None:
            raise OpenIntentError("roundtrip second leg precedes its reservation")
        return request
    if record["first"]["status"] != "FILLED":
        raise OpenIntentError("roundtrip partial first execution remains an obligation")
    second_binding = request.binding(first=False, amount=record["second_amount"])
    asset = (
        request.quote
        if isinstance(second_binding, SpotQuoteOrderBinding)
        else request.base
    )
    amount = Fraction(record["second_amount"])
    if amount > min(first_delta[asset], balances[asset]):
        raise OpenIntentError(
            "roundtrip second principal exceeds first-leg owned receipts"
        )
    if record["second"] is None:
        if phase != "SECOND_UNKNOWN":
            raise OpenIntentError("roundtrip phase lacks its second observation")
        return request
    if _selected(record["second"]) != record["second"]:
        raise OpenIntentError("roundtrip second observation is not canonical")
    delta = _movements(
        request, record["second"], first=False, amount=record["second_amount"]
    )
    if any(balances[asset] + delta[asset] < 0 for asset in balances):
        raise OpenIntentError("roundtrip second execution exceeds its allocated assets")
    if (record["second"]["status"] in _TERMINAL) != (phase != "SECOND_UNKNOWN"):
        raise OpenIntentError("roundtrip second phase differs from its native status")
    if phase == "COMPLETE" and record["second"]["status"] != "FILLED":
        raise OpenIntentError(
            "roundtrip partial second execution remains an obligation"
        )
    return request


def _decode_record(row: tuple, operation_id: str) -> dict:
    if (
        not isinstance(row[1], str)
        or not isinstance(row[2], str)
        or _sha(row[1]) != row[2]
    ):
        raise OpenIntentError("roundtrip record is oversized, missing or changed")
    try:
        record = json.loads(row[1])
        request = _validate_record(record)
        if (
            request.operation_id != operation_id
            or record["phase"] != row[0]
            or _json(record) != row[1]
        ):
            raise OpenIntentError("roundtrip identity or phase differs")
    except (KeyError, TypeError, ValueError, RecursionError):
        raise OpenIntentError("roundtrip record is unreadable") from None
    return record


def roundtrip_pending(connection: sqlite3.Connection) -> dict | None:
    """Validate enrollment and the active pointer without creating any tables."""
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    columns = tuple(
        row[1] for row in connection.execute("PRAGMA table_info(spot_roundtrip)")
    )
    control_columns = tuple(
        row[1]
        for row in connection.execute("PRAGMA table_info(spot_roundtrip_control)")
    )
    if version != 4:
        if columns or control_columns:
            raise OpenIntentError("roundtrip tables lack their namespace enrollment")
        return None
    if columns != _ORDER_COLUMNS or control_columns != _CONTROL_COLUMNS:
        raise OpenIntentError("enrolled roundtrip storage is missing or unrecognized")
    rows = connection.execute("SELECT * FROM spot_roundtrip_control LIMIT 2").fetchall()
    if len(rows) != 1:
        raise OpenIntentError("roundtrip control record is missing or ambiguous")
    identity, revision, completed, operation_id, checksum = rows[0]
    if (
        identity != 1
        or type(revision) is not int
        or type(completed) is not int
        or completed < 0
        or revision < completed * 5
    ):
        raise OpenIntentError("roundtrip control counters are invalid")
    total, complete_count = connection.execute(
        "SELECT COUNT(*), COUNT(CASE WHEN phase='COMPLETE' THEN 1 END) FROM spot_roundtrip"
    ).fetchone()
    if total != completed + (operation_id is not None) or complete_count != completed:
        raise OpenIntentError("roundtrip rows differ from their durable control count")
    if operation_id is None:
        if checksum is not None:
            raise OpenIntentError("empty roundtrip control retains an active checksum")
        return None
    row = connection.execute(_RECORD_SELECT, (operation_id,)).fetchone()
    if row is None or row[2] != checksum:
        raise OpenIntentError("active roundtrip record is missing or changed")
    record = _decode_record(row, operation_id)
    if record["phase"] == "COMPLETE":
        raise OpenIntentError("completed roundtrip retains an active obligation")
    minimum = {
        "FIRST_UNKNOWN": 1,
        "FIRST_TERMINAL": 2,
        "SECOND_UNKNOWN": 3,
        "SECOND_TERMINAL": 4,
    }[record["phase"]]
    if revision < completed * 5 + minimum:
        raise OpenIntentError("roundtrip revision precedes its recorded phase")
    return record


def _save(
    connection: sqlite3.Connection, record: dict, *, complete: bool = False
) -> None:
    request = _validate_record(record)
    if complete is not (record["phase"] == "COMPLETE"):
        raise OpenIntentError("roundtrip completion differs from its phase")
    body = _json(record)
    checksum = _sha(body)
    connection.execute(
        "INSERT INTO spot_roundtrip VALUES (?, ?, ?, ?) ON CONFLICT(operation_id) DO UPDATE SET phase=excluded.phase, record_json=excluded.record_json, record_sha256=excluded.record_sha256",
        (request.operation_id, record["phase"], body, checksum),
    )
    connection.execute(
        "UPDATE spot_roundtrip_control SET revision=revision+1, completed_count=completed_count+?, active_operation_id=?, active_sha256=? WHERE id=1",
        (
            int(complete),
            None if complete else request.operation_id,
            None if complete else checksum,
        ),
    )


def roundtrip_client_id_consumed(
    connection: sqlite3.Connection, client_id: str
) -> bool:
    """Fence both planned leg IDs against other writers in this namespace."""
    if connection.execute("PRAGMA user_version").fetchone()[0] != 4:
        return False
    if not isinstance(client_id, str) or not client_id.startswith(
        ("sait-o-", "sait-c-")
    ):
        return False
    return (
        connection.execute(
            "SELECT 1 FROM spot_roundtrip WHERE operation_id=?", (client_id[7:],)
        ).fetchone()
        is not None
    )


@dataclass(frozen=True)
class SpotRoundtripJournal:
    journal: BinanceOpenIntentJournal

    def prepare(
        self,
        request: SpotRoundtripRequest,
        *,
        scope: BinanceExecutionScope,
        create: bool = True,
    ) -> None:
        """Reserve an explicit allocation and first UNKNOWN under the common writer."""
        from .positions import PositionsStore

        if (
            not isinstance(request, SpotRoundtripRequest)
            or not isinstance(scope, BinanceExecutionScope)
            or scope.market_type != "spot"
        ):
            raise OpenIntentError(
                "roundtrip requires a validated Spot request and scope"
            )
        with position_transaction(
            self.journal, write=True, create=create
        ) as transaction:
            self.journal._prepare_scope(transaction, scope)
            connection = transaction.connection
            if connection is None:
                raise OpenIntentError("roundtrip requires durable execution storage")
            if (
                roundtrip_pending(connection) is not None
                or unresolved_close_count(connection)
                or connection.execute(
                    "SELECT 1 FROM open_intent WHERE state IS NOT 'RECORDED' LIMIT 1"
                ).fetchone()
            ):
                raise OpenIntentError("unresolved execution blocks a new roundtrip")
            store = PositionsStore(transaction.root)
            if store._open_entries(
                store._decode(transaction.read(store.open_path), strict=True),
                strict=True,
            ):
                raise OpenIntentError(
                    "roundtrip allocation cannot inherit owned position assets"
                )
            identities = (
                "sait-o-" + request.operation_id,
                "sait-c-" + request.operation_id,
            )
            if connection.execute(
                "SELECT 1 FROM open_intent WHERE client_id IN (?, ?) LIMIT 1",
                identities,
            ).fetchone():
                raise OpenIntentError("roundtrip client identity is already consumed")
            if (
                connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='close_intent'"
                ).fetchone()
                and connection.execute(
                    "SELECT 1 FROM close_intent WHERE client_id IN (?, ?) LIMIT 1",
                    identities,
                ).fetchone()
            ):
                raise OpenIntentError("roundtrip client identity is already consumed")
            if connection.execute("PRAGMA user_version").fetchone()[0] == 3:
                connection.execute(
                    "CREATE TABLE spot_roundtrip (operation_id TEXT PRIMARY KEY, phase TEXT NOT NULL, record_json TEXT NOT NULL, record_sha256 TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE spot_roundtrip_control (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL, completed_count INTEGER NOT NULL, active_operation_id TEXT, active_sha256 TEXT)"
                )
                connection.execute(
                    "INSERT INTO spot_roundtrip_control VALUES (1, 0, 0, NULL, NULL)"
                )
                connection.execute("PRAGMA user_version=4")
            if connection.execute(
                "SELECT 1 FROM spot_roundtrip WHERE operation_id=?",
                (request.operation_id,),
            ).fetchone():
                raise OpenIntentError("roundtrip identity is already consumed")
            _save(
                connection,
                {
                    "format": 2 if request.fee_asset_allocations else 1,
                    "request": {
                        key: (
                            [list(row) for row in value]
                            if key == "fee_asset_allocations"
                            else value
                        )
                        for key, value in asdict(request).items()
                        if key != "fee_asset_allocations" or value
                    },
                    "scope": asdict(scope),
                    "phase": "FIRST_UNKNOWN",
                    "first": None,
                    "second_amount": None,
                    "second": None,
                },
            )

    def pending(self, *, scope: BinanceExecutionScope) -> tuple[int, dict] | None:
        if not isinstance(scope, BinanceExecutionScope) or scope.market_type != "spot":
            raise OpenIntentError("roundtrip recovery requires its Spot scope")
        with position_transaction(self.journal) as transaction:
            if transaction.connection is None:
                return None
            self.journal._bind_scope(transaction.connection, scope)
            record = roundtrip_pending(transaction.connection)
            if record is None:
                return None
            if record["scope"] != asdict(scope):
                raise OpenIntentError("roundtrip belongs to another scope")
            revision = transaction.connection.execute(
                "SELECT revision FROM spot_roundtrip_control WHERE id=1"
            ).fetchone()[0]
            return revision, record

    def operation(
        self, operation_id: str, *, scope: BinanceExecutionScope
    ) -> dict | None:
        """Read a completed or active operation without recreating its storage."""
        if not isinstance(scope, BinanceExecutionScope) or scope.market_type != "spot":
            raise OpenIntentError("roundtrip history requires its Spot scope")
        if (
            not isinstance(operation_id, str)
            or re.fullmatch(r"[0-9a-f]{29}", operation_id) is None
        ):
            raise OpenIntentError(
                "roundtrip history requires its exact operation identity"
            )
        with position_transaction(self.journal) as transaction:
            if transaction.connection is None:
                return None
            connection = transaction.connection
            self.journal._bind_scope(connection, scope)
            if connection.execute("PRAGMA user_version").fetchone()[0] != 4:
                return None
            row = connection.execute(_RECORD_SELECT, (operation_id,)).fetchone()
            if row is None:
                return None
            record = _decode_record(row, operation_id)
            if record["scope"] != asdict(scope):
                raise OpenIntentError("roundtrip history belongs to another scope")
            return record

    def _change(
        self,
        operation_id: str,
        *,
        scope: BinanceExecutionScope,
        expected_revision: int,
        change,
    ) -> None:
        if (
            not isinstance(scope, BinanceExecutionScope)
            or scope.market_type != "spot"
            or not isinstance(operation_id, str)
            or re.fullmatch(r"[0-9a-f]{29}", operation_id) is None
        ):
            raise OpenIntentError(
                "roundtrip mutation requires its exact identity and Spot scope"
            )
        if type(expected_revision) is not int or expected_revision < 0:
            raise OpenIntentError("roundtrip revision must be explicit")
        with position_transaction(
            self.journal, write=True, create=False
        ) as transaction:
            connection = transaction.connection
            if connection is None:
                raise OpenIntentError("roundtrip requires durable execution storage")
            self.journal._bind_scope(connection, scope)
            record = roundtrip_pending(connection)
            if (
                record is None
                or record["request"]["operation_id"] != operation_id
                or record["scope"] != asdict(scope)
            ):
                raise OpenIntentError(
                    "roundtrip does not match the active scoped obligation"
                )
            revision = connection.execute(
                "SELECT revision FROM spot_roundtrip_control WHERE id=1"
            ).fetchone()[0]
            if revision != expected_revision:
                raise OpenIntentError(
                    "roundtrip changed during preflight or observation"
                )
            complete = change(record)
            _save(connection, record, complete=complete)

    def record_order(
        self,
        operation_id: str,
        order: Mapping[str, object],
        *,
        first: bool,
        scope: BinanceExecutionScope,
        expected_revision: int,
    ) -> None:
        """Retain complete native observations; partial orders keep admission blocked."""
        if type(first) is not bool:
            raise OpenIntentError("roundtrip observation leg must be explicit")
        selected = _selected(order)

        def change(record):
            phase = "FIRST_UNKNOWN" if first else "SECOND_UNKNOWN"
            if record["phase"] != phase:
                raise OpenIntentError(
                    "roundtrip observation has no matching pending leg"
                )
            field = "first" if first else "second"
            old = record[field]
            if old is not None:
                if parse_execution_id(selected.get("orderId")) != parse_execution_id(
                    old.get("orderId")
                ):
                    raise OpenIntentError("roundtrip exchange order identity changed")
                for key in ("executedQty", "cummulativeQuoteQty"):
                    if _decimal(selected.get(key)) < _decimal(old.get(key)):
                        raise OpenIntentError("roundtrip observation regressed")
                previous_fees, current_fees = {}, {}
                for observation, totals in (
                    (old, previous_fees),
                    (selected, current_fees),
                ):
                    for row in observation["fills"]:
                        asset = row.get("commissionAsset")
                        fee = Fraction(_decimal(row.get("commission")))
                        if fee:
                            totals[asset] = totals.get(asset, Fraction(0)) + fee
                if any(
                    current_fees.get(asset, Fraction(0)) < fee
                    for asset, fee in previous_fees.items()
                ):
                    raise OpenIntentError("roundtrip commission observation regressed")
                if _decimal(selected["executedQty"]) == _decimal(
                    old["executedQty"]
                ) and (
                    _decimal(selected["cummulativeQuoteQty"])
                    != _decimal(old["cummulativeQuoteQty"])
                    or previous_fees != current_fees
                ):
                    raise OpenIntentError(
                        "roundtrip cash or fees changed without execution"
                    )
            record[field] = selected
            if selected.get("status") in _TERMINAL:
                record["phase"] = "FIRST_TERMINAL" if first else "SECOND_TERMINAL"
            return False

        self._change(
            operation_id,
            scope=scope,
            expected_revision=expected_revision,
            change=change,
        )

    def prepare_second(
        self,
        operation_id: str,
        amount: str,
        *,
        scope: BinanceExecutionScope,
        expected_revision: int,
    ) -> None:
        def change(record):
            if (
                record["phase"] != "FIRST_TERMINAL"
                or record["first"]["status"] != "FILLED"
            ):
                raise OpenIntentError("roundtrip first leg is not fully reconciled")
            record["second_amount"], record["phase"] = amount, "SECOND_UNKNOWN"
            return False

        self._change(
            operation_id,
            scope=scope,
            expected_revision=expected_revision,
            change=change,
        )

    def complete(
        self, operation_id: str, *, scope: BinanceExecutionScope, expected_revision: int
    ) -> None:
        """Release only two fully reconciled orders; this does not claim flat wealth."""

        def change(record):
            if (
                record["phase"] != "SECOND_TERMINAL"
                or record["second"]["status"] != "FILLED"
            ):
                raise OpenIntentError("roundtrip second leg remains an obligation")
            record["phase"] = "COMPLETE"
            return True

        self._change(
            operation_id,
            scope=scope,
            expected_revision=expected_revision,
            change=change,
        )
