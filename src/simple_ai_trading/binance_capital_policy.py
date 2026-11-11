"""Explicit enrolled loss-policy revisions in the existing execution journal."""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import uuid

from .binance_execution_scope import BinanceExecutionScope
from .binance_execution_storage import BinanceExecutionStorage
from .binance_open_intents import OpenIntentError
from .binance_runtime_control import BinanceRuntimeControl
from .runtime_control_storage import FileIdentity, RuntimeStorageEnrollment

_SCHEMA = "binance-capital-policy-v1"
_NAMESPACE = "binance_capital_policy"
_ENROLLMENT_SCHEMA = "binance-capital-policy-storage-v1"
_MAX_DOCUMENT_BYTES = 8192
_MAX_REVISIONS = 4096
_TABLES = {
    "capital_policy_identity": ("id", "store_id"),
    "capital_policy_history": (
        "revision",
        "policy_json",
        "previous_sha256",
        "entry_sha256",
    ),
    "capital_policy_head": ("id", "revision", "entry_sha256"),
}


def _encode(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def _sha(value):
    return hashlib.sha256(_encode(value).encode("ascii")).hexdigest()


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate capital policy field")
        result[key] = value
    return result


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise ValueError("capital policy number is invalid")
    text = str(value)
    if len(text) > 400:
        raise ValueError("capital policy number exceeds its budget")
    try:
        number = Decimal(text)
    except InvalidOperation:
        raise ValueError("capital policy number is invalid") from None
    if not number.is_finite() or not -324 <= number.adjusted() <= 308:
        raise ValueError("capital policy number is outside finite runtime bounds")
    rendered = format(number, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


@dataclass(frozen=True)
class BinanceCapitalPolicy:
    schema_version: str
    execution_scope_sha256: str
    execution_store_id: str
    quote_asset: str
    reference_cash_quote: str
    maximum_daily_loss_fraction: str
    maximum_lifetime_loss_fraction: str
    maximum_settled_peak_drawdown_fraction: str
    maximum_consecutive_losses: int

    def __post_init__(self):
        if (
            self.schema_version != _SCHEMA
            or not _digest(self.execution_scope_sha256)
            or not isinstance(self.execution_store_id, str)
            or re.fullmatch(r"[0-9a-f]{32}", self.execution_store_id) is None
            or not isinstance(self.quote_asset, str)
            or re.fullmatch(r"[A-Z0-9]{2,16}", self.quote_asset) is None
            or type(self.maximum_consecutive_losses) is not int
            or not 1 <= self.maximum_consecutive_losses <= 100
        ):
            raise ValueError("capital policy identity or count is invalid")
        for name in (
            "reference_cash_quote",
            "maximum_daily_loss_fraction",
            "maximum_lifetime_loss_fraction",
            "maximum_settled_peak_drawdown_fraction",
        ):
            value = getattr(self, name)
            if not isinstance(value, str):
                raise ValueError(
                    "capital policy amounts must be explicit decimal strings"
                )
            normalized = _number(value)
            numeric = Decimal(normalized)
            if numeric <= 0 or (name != "reference_cash_quote" and numeric > 1):
                raise ValueError("capital policy amount or fraction is invalid")
            object.__setattr__(self, name, normalized)

    @classmethod
    def parse(cls, raw: bytes):
        if not isinstance(raw, bytes) or len(raw) > _MAX_DOCUMENT_BYTES:
            raise ValueError("capital policy document exceeds its budget")
        value = json.loads(raw, object_pairs_hook=_unique)
        if not isinstance(value, dict) or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("capital policy fields differ")
        return cls(**value)

    def matches(
        self, strategy, reference_cash, *, symbol, market_type, scope_market_type
    ):
        from .assets import symbol_base_for_supported_quote

        base = symbol_base_for_supported_quote(symbol)
        return (
            scope_market_type == market_type
            and base
            and symbol[len(base) :] == self.quote_asset
            and _number(reference_cash) == self.reference_cash_quote
            and _number(strategy.max_daily_loss_pct) == self.maximum_daily_loss_fraction
            and _number(strategy.max_session_loss_pct)
            == self.maximum_lifetime_loss_fraction
            and _number(strategy.max_drawdown_limit)
            == self.maximum_settled_peak_drawdown_fraction
            and type(strategy.max_consecutive_losses) is int
            and strategy.max_consecutive_losses == self.maximum_consecutive_losses
        )


class BinanceCapitalPolicyStore:
    """No policy bootstrap on reads; activation/recovery never clears Stop or cash."""

    def __init__(
        self,
        store,
        *,
        control=None,
        execution_enrollment_path=None,
        enrollment_path=None,
    ):
        self.store = store
        self.control = control or BinanceRuntimeControl(
            store.root / "binance_runtime_control.sqlite3"
        )
        self.storage = BinanceExecutionStorage(
            store.opening_intents, self.control, execution_enrollment_path
        )
        journal = self.storage.journal.path
        self.enrollment_path = (
            Path(enrollment_path)
            if enrollment_path is not None
            else journal.with_name(journal.name + ".capital_policy.enrollment.json")
        )

    def _scope(self, connection):
        rows = connection.execute(
            "SELECT id, scope_json FROM execution_scope LIMIT 2"
        ).fetchall()
        if (
            len(rows) != 1
            or rows[0][0] != 1
            or not isinstance(rows[0][1], str)
            or len(rows[0][1]) > 1024
        ):
            raise OpenIntentError(
                "capital policy requires the original execution scope"
            )
        scope = BinanceExecutionScope(**json.loads(rows[0][1]))
        self.storage.journal._bind_scope(connection, scope)
        return scope

    def _enrollment(self, *, prepared=False):
        record = RuntimeStorageEnrollment.read(
            self.enrollment_path, allow_prepared=prepared
        )
        if (record.namespace, record.control_schema_version) != (
            _NAMESPACE,
            _ENROLLMENT_SCHEMA,
        ):
            raise OpenIntentError("capital policy enrollment namespace differs")
        self.storage._resources(record)
        return record

    def _read(self, connection, scope, *, prepared=False):
        present = {
            name: tuple(
                row[1] for row in connection.execute(f"PRAGMA table_info({name})")
            )
            for name in _TABLES
        }
        execution_id = self.storage._uuid(connection)
        if execution_id is None:
            raise OpenIntentError("capital policy requires enrolled execution storage")
        scope_digest = _sha(asdict(scope))
        if not any(present.values()):
            if self.enrollment_path.exists() or self.enrollment_path.is_symlink():
                raise OpenIntentError(
                    "capital policy history is missing; preserve its enrollment"
                )
            return {
                "revision": 0,
                "policy": None,
                "head_sha256": _sha([_SCHEMA, execution_id, scope_digest, None]),
                "execution_scope_sha256": scope_digest,
                "execution_store_id": execution_id,
                "market_type": scope.market_type,
                "venue_origin": scope.origin,
            }
        if present != _TABLES:
            raise OpenIntentError("capital policy schema is incomplete or changed")
        record = self._enrollment(prepared=prepared)
        if connection.execute(
            "SELECT id, store_id FROM capital_policy_identity LIMIT 2"
        ).fetchall() != [(1, record.store_id)]:
            raise OpenIntentError("capital policy storage identity differs")
        count, document_size, prior_size, hash_size = connection.execute(
            "SELECT COUNT(*), MAX(length(CAST(policy_json AS BLOB))), MAX(length(CAST(previous_sha256 AS BLOB))), MAX(length(CAST(entry_sha256 AS BLOB))) FROM capital_policy_history"
        ).fetchone()
        if (
            not count
            or count > _MAX_REVISIONS
            or document_size > _MAX_DOCUMENT_BYTES
            or prior_size != 64
            or hash_size != 64
        ):
            raise OpenIntentError(
                "capital policy history is missing or exceeds its budget"
            )
        rows = connection.execute(
            "SELECT revision, policy_json, previous_sha256, entry_sha256 FROM capital_policy_history ORDER BY revision LIMIT ?",
            (_MAX_REVISIONS + 1,),
        ).fetchall()
        previous = "0" * 64
        reference = quote = None
        policy = None
        for index, (revision, encoded, prior, claimed) in enumerate(rows, start=1):
            if type(revision) is not int or revision != index or prior != previous:
                raise OpenIntentError("capital policy history order differs")
            if not isinstance(encoded, str):
                raise OpenIntentError("capital policy history is unreadable")
            policy = BinanceCapitalPolicy.parse(encoded.encode("utf-8"))
            canonical = asdict(policy)
            if (
                encoded != _encode(canonical)
                or policy.execution_scope_sha256 != scope_digest
                or policy.execution_store_id != execution_id
                or claimed != _sha([revision, prior, canonical])
                or (
                    reference is not None
                    and (reference, quote)
                    != (policy.reference_cash_quote, policy.quote_asset)
                )
            ):
                raise OpenIntentError("capital policy content or binding differs")
            reference, quote, previous = (
                policy.reference_cash_quote,
                policy.quote_asset,
                claimed,
            )
        if connection.execute(
            "SELECT id, revision, entry_sha256 FROM capital_policy_head LIMIT 2"
        ).fetchall() != [(1, len(rows), previous)]:
            raise OpenIntentError("capital policy head differs from its history")
        return {
            "revision": len(rows),
            "policy": asdict(policy),
            "head_sha256": previous,
            "execution_scope_sha256": scope_digest,
            "execution_store_id": execution_id,
            "market_type": scope.market_type,
            "venue_origin": scope.origin,
        }

    def inspect(self, *, allow_prepared=False):
        with self.control._interlock():
            return self._inspect(allow_prepared=allow_prepared)

    def _inspect(self, *, allow_prepared=False):
        with closing(self.storage.journal._connect()) as connection:
            connection.execute("BEGIN")
            scope = self._scope(connection)
            self.storage.verify(scope=scope)
            result = self._read(connection, scope, prepared=allow_prepared)
            result["enrollment_state"] = (
                self._enrollment(prepared=allow_prepared).state
                if result["policy"] is not None
                else "unenrolled"
            )
            if allow_prepared and result["policy"] is not None:
                record = self._enrollment(prepared=True)
                result.update(
                    policy_store_id=record.store_id,
                    enrollment_record_sha256=hashlib.sha256(
                        record.encoded()
                    ).hexdigest(),
                )
            return result

    def activate(self, policy: BinanceCapitalPolicy, *, expected_head_sha256: str):
        if not isinstance(policy, BinanceCapitalPolicy) or not _digest(
            expected_head_sha256
        ):
            raise OpenIntentError(
                "capital policy activation requires a validated document and inspected head"
            )
        prepared = None
        with self.control.stopped_recovery_guard():
            with (
                closing(self.storage.journal._connect(write=True)) as connection,
                connection,
            ):
                scope = self._scope(connection)
                self.storage.verify(scope=scope)
                current = self._read(connection, scope)
                if current["head_sha256"] != expected_head_sha256 or (
                    policy.execution_scope_sha256,
                    policy.execution_store_id,
                ) != (current["execution_scope_sha256"], current["execution_store_id"]):
                    raise OpenIntentError("capital policy activation binding changed")
                if current["policy"] is not None:
                    old = current["policy"]
                    if (policy.reference_cash_quote, policy.quote_asset) != (
                        old["reference_cash_quote"],
                        old["quote_asset"],
                    ):
                        raise OpenIntentError(
                            "capital reference/unit changes require qualified cash-flow migration"
                        )
                    if asdict(policy) == old:
                        return {**current, "enrollment_state": "ready"}
                if current["revision"] >= _MAX_REVISIONS:
                    raise OpenIntentError("capital policy revision budget is exhausted")
                if current["policy"] is None:
                    control = self.control._enrollment()
                    prepared = RuntimeStorageEnrollment(
                        _NAMESPACE,
                        _ENROLLMENT_SCHEMA,
                        uuid.uuid4().hex,
                        "prepared",
                        FileIdentity.at(self.storage.journal.path),
                        control.submission_lock,
                        control.close_lock,
                    )
                    connection.execute(
                        "CREATE TABLE capital_policy_identity (id INTEGER PRIMARY KEY CHECK(id=1), store_id TEXT NOT NULL)"
                    )
                    connection.execute(
                        "CREATE TABLE capital_policy_history (revision INTEGER PRIMARY KEY, policy_json TEXT NOT NULL, previous_sha256 TEXT NOT NULL, entry_sha256 TEXT NOT NULL UNIQUE)"
                    )
                    connection.execute(
                        "CREATE TABLE capital_policy_head (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL, entry_sha256 TEXT NOT NULL)"
                    )
                    connection.execute(
                        "INSERT INTO capital_policy_identity VALUES (1, ?)",
                        (prepared.store_id,),
                    )
                    prepared.prepare(self.enrollment_path)
                revision = current["revision"] + 1
                prior = "0" * 64 if revision == 1 else current["head_sha256"]
                digest = _sha([revision, prior, asdict(policy)])
                connection.execute(
                    "INSERT INTO capital_policy_history VALUES (?, ?, ?, ?)",
                    (revision, _encode(asdict(policy)), prior, digest),
                )
                connection.execute(
                    "INSERT OR REPLACE INTO capital_policy_head VALUES (1, ?, ?)",
                    (revision, digest),
                )
            if prepared is not None:
                # Verify the committed first revision before making it ordinary-readable.
                self._inspect(allow_prepared=True)
                prepared.finish(self.enrollment_path)
        return self.inspect()

    def recover(self, *, expected_store_id, expected_record_sha256):
        with self.control.stopped_recovery_guard():
            record = self._enrollment(prepared=True)
            if (
                record.state != "prepared"
                or record.store_id != expected_store_id
                or hashlib.sha256(record.encoded()).hexdigest()
                != expected_record_sha256
            ):
                raise OpenIntentError(
                    "capital policy prepared recovery evidence differs"
                )
            self._inspect(allow_prepared=True)
            record.finish(self.enrollment_path)
        return self.inspect()

    def assert_effective(self, strategy, reference_cash, *, symbol, market_type):
        current = self.inspect()
        if current["policy"] is None:
            raise OpenIntentError(
                "signed autonomous admission requires explicit capital policy activation"
            )
        if not BinanceCapitalPolicy(**current["policy"]).matches(
            strategy,
            reference_cash,
            symbol=symbol,
            market_type=market_type,
            scope_market_type=current["market_type"],
        ):
            raise OpenIntentError(
                "effective capital limits differ from the activated policy"
            )
        return current["head_sha256"]
