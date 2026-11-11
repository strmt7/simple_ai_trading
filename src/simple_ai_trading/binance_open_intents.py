"""Write-ahead obligations for autonomous Binance openings, not an inventory ledger."""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .binance_execution_scope import BinanceExecutionScope

if TYPE_CHECKING:
    from .position_transactions import PositionTransaction
    from .positions import OpenPosition


class OpenIntentError(ValueError):
    """An opening cannot safely proceed through its durable intent boundary."""


@dataclass(frozen=True)
class BinanceOpenIntentJournal:
    path: Path

    def _connect(
        self, *, create: bool = False, write: bool = False
    ) -> sqlite3.Connection:
        path = Path(self.path).resolve()
        existed = path.exists()
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
        mode = "rwc" if create else "rw" if write else "ro"
        connection = sqlite3.connect(
            f"{path.as_uri()}?mode={mode}", uri=True, timeout=1.0
        )
        try:
            if create or write:
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("BEGIN IMMEDIATE")
                # A competing first writer may have initialized and enrolled the
                # namespace while this connection waited for its writer lock.
                if (
                    not existed
                    and connection.execute("PRAGMA user_version").fetchone()[0] == 0
                ):
                    connection.execute(
                        "CREATE TABLE IF NOT EXISTS open_intent ("
                        "client_id TEXT PRIMARY KEY, position_id TEXT NOT NULL UNIQUE, "
                        "request_json TEXT NOT NULL, "
                        "state TEXT NOT NULL CHECK(state IN ('UNKNOWN', 'RECORDED')))"
                    )
                    connection.execute(
                        "CREATE TABLE IF NOT EXISTS execution_scope ("
                        "id INTEGER PRIMARY KEY CHECK(id=1), scope_json TEXT NOT NULL)"
                    )
                    connection.execute("PRAGMA user_version=2")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            columns = tuple(
                row[1] for row in connection.execute("PRAGMA table_info(open_intent)")
            )
            if version not in {1, 2, 3, 4} or columns != (
                "client_id",
                "position_id",
                "request_json",
                "state",
            ):
                raise OpenIntentError("opening intent schema is not recognized")
            if version in {2, 3, 4} and tuple(
                row[1]
                for row in connection.execute("PRAGMA table_info(execution_scope)")
            ) != ("id", "scope_json"):
                raise OpenIntentError("opening scope schema is not recognized")
            from .binance_spot_roundtrip_intents import roundtrip_pending

            roundtrip_pending(connection)
            return connection
        except BaseException:
            connection.close()
            raise

    @staticmethod
    def _bind_scope(
        connection: sqlite3.Connection,
        scope: BinanceExecutionScope,
    ) -> None:
        """Fence the complete journal; legacy storage needs explicit migration."""
        if connection.execute("PRAGMA user_version").fetchone()[0] not in {2, 3, 4}:
            raise OpenIntentError("legacy journal requires explicit scope migration")
        encoded = json.dumps(asdict(scope), sort_keys=True, allow_nan=False)
        rows = connection.execute(
            "SELECT id, scope_json FROM execution_scope LIMIT 2"
        ).fetchall()
        if rows != [(1, encoded)]:
            raise OpenIntentError(
                "journal belongs to a different or missing execution scope"
            )

    def _prepare_scope(
        self, transaction: PositionTransaction, scope: BinanceExecutionScope
    ) -> None:
        """Enroll only a fresh namespace or explicitly paper-only legacy files."""
        from .positions import PositionsStore

        connection = transaction.connection
        if connection is None:
            raise OpenIntentError("opening requires durable execution storage")
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version not in {2, 3, 4}:
            raise OpenIntentError("legacy journal requires explicit scope migration")
        if connection.execute("SELECT 1 FROM execution_scope LIMIT 1").fetchone():
            self._bind_scope(connection, scope)
            if version not in {3, 4}:
                raise OpenIntentError(
                    "legacy scope enrollment requires verified migration"
                )
            return
        if (
            version in {3, 4}
            or connection.execute("SELECT 1 FROM open_intent LIMIT 1").fetchone()
        ):
            raise OpenIntentError("existing journal has lost its execution scope")
        # An unscoped auxiliary execution record cannot inherit new credentials.
        # Table names come from SQLite metadata, and still require identifier quoting.
        for (name,) in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ):
            if name in {
                "open_intent",
                "execution_scope",
                "position_replacement",
            } or name.startswith("sqlite_"):
                continue
            quoted = '"' + name.replace('"', '""') + '"'
            if connection.execute(f"SELECT 1 FROM {quoted} LIMIT 1").fetchone():
                raise OpenIntentError(
                    "unscoped execution history requires verified migration"
                )
        store = PositionsStore(transaction.root)
        for path, parse in (
            (store.open_path, store._open_entries),
            (store.ledger_path, store._closed_entries),
        ):
            entries = store._decode(transaction.read(path), strict=True)
            if any(entry.get("dry_run") is not True for entry in entries):
                raise OpenIntentError(
                    "unscoped live or unclassified history requires verified migration"
                )
            parse(entries, strict=True)
        transaction.enroll_unchanged()
        connection.execute(
            "INSERT INTO execution_scope VALUES (1, ?)",
            (json.dumps(asdict(scope), sort_keys=True, allow_nan=False),),
        )
        # Namespace version, scope, paired-byte enrollment and UNKNOWN are one
        # transaction. Request payloads retain their existing intent_version=2.
        connection.execute("PRAGMA user_version=3")

    @staticmethod
    def _request(
        position: OpenPosition, scope: BinanceExecutionScope | None = None
    ) -> str:
        client_id = position.open_client_order_id
        if (
            position.dry_run is not False
            or position.owner != "simple_ai_trading"
            or not isinstance(client_id, str)
            or not client_id.startswith("sait-o-")
            or not 1 <= len(client_id) <= 36
            or client_id != client_id.strip()
            or not isinstance(position.id, str)
            or not position.id.strip()
            or position.market_type not in {"spot", "futures"}
            or position.side not in {"LONG", "SHORT"}
            or position.market_type == "spot"
            and position.side != "LONG"
            or not isinstance(position.symbol, str)
            or not position.symbol.strip()
            or isinstance(position.qty, bool)
            or not isinstance(position.qty, (int, float))
            or not math.isfinite(position.qty)
            or position.qty <= 0
        ):
            raise OpenIntentError("opening intent identity or quantity is invalid")
        template = asdict(position)
        policy_digest = position.capital_policy_sha256
        if (
            not isinstance(policy_digest, str)
            or policy_digest
            and (
                len(policy_digest) != 64
                or any(
                    character not in "0123456789abcdef" for character in policy_digest
                )
            )
        ):
            raise OpenIntentError("opening capital policy digest is invalid")
        # Empty new receipt fields must not change old pending request bytes.
        for field in (
            "spot_gross_entry_quantity",
            "spot_entry_base_commission",
            "spot_entry_cash_receipt",
            "capital_policy_sha256",
        ):
            if template.get(field) == "":
                template.pop(field)
        payload = {
            "position_id": position.id,
            "client_id": client_id,
            "symbol": position.symbol,
            "market_type": position.market_type,
            "side": position.side,
            "quantity": position.qty,
            # Retain the original risk/ownership template for exact recovery;
            # OpenPosition has no account credentials or signed-request fields.
            "position_template": template,
        }
        if scope is not None:
            if (
                not isinstance(scope, BinanceExecutionScope)
                or scope.market_type != position.market_type
            ):
                raise OpenIntentError("opening product differs from execution scope")
            payload.update(intent_version=2, execution_scope=asdict(scope))
        try:
            return json.dumps(payload, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise OpenIntentError(
                "opening position template is not serializable"
            ) from exc

    def prepare(
        self,
        position: OpenPosition,
        *,
        scope: BinanceExecutionScope,
        create: bool = True,
        admission: Callable[[PositionTransaction], None] | None = None,
    ) -> None:
        """Commit UNKNOWN before transmission; an existing identity is never replayable."""
        from .binance_close_intents import unresolved_close_count
        from .position_transactions import position_transaction

        if not isinstance(scope, BinanceExecutionScope):
            raise OpenIntentError("opening requires an execution scope")
        payload = self._request(position, scope)
        try:
            with position_transaction(self, write=True, create=create) as transaction:
                self._prepare_scope(transaction, scope)
                connection = transaction.connection
                if connection is None:
                    raise OpenIntentError("opening requires durable execution storage")
                if unresolved_close_count(connection):
                    raise OpenIntentError("an unresolved closing blocks new exposure")
                from .binance_spot_roundtrip_intents import (
                    roundtrip_client_id_consumed,
                    roundtrip_pending,
                )

                if roundtrip_pending(connection) is not None:
                    raise OpenIntentError("an unresolved roundtrip blocks new exposure")
                if roundtrip_client_id_consumed(
                    connection, position.open_client_order_id
                ):
                    raise OpenIntentError(
                        "opening client identity belongs to a roundtrip"
                    )
                if connection.execute(
                    "SELECT 1 FROM open_intent WHERE state != 'RECORDED' LIMIT 1"
                ).fetchone():
                    raise OpenIntentError(
                        "an unresolved opening requires reconciliation"
                    )
                if admission is not None:
                    admission(transaction)
                    payload = self._request(position, scope)
                connection.execute(
                    "INSERT INTO open_intent VALUES (?, ?, ?, 'UNKNOWN')",
                    (position.open_client_order_id, position.id, payload),
                )
        except OpenIntentError:
            raise
        except (sqlite3.Error, OSError, TypeError, ValueError) as exc:
            raise OpenIntentError("opening intent could not be committed") from exc

    def record_complete(
        self,
        requested: OpenPosition,
        recorded: OpenPosition,
        *,
        scope: BinanceExecutionScope,
    ) -> None:
        """Release admission only after the caller persists a resolved full fill."""
        if not isinstance(scope, BinanceExecutionScope):
            raise OpenIntentError("opening requires an execution scope")
        payload = self._request(requested, scope)
        self.validate_result(requested, recorded)
        if recorded.exchange_status.upper() != "FILLED":
            # A partial fill does not settle the remaining order obligation.
            return
        try:
            with closing(self._connect(write=True)) as connection, connection:
                self._bind_scope(connection, scope)
                result = connection.execute(
                    "UPDATE open_intent SET state='RECORDED' "
                    "WHERE client_id=? AND position_id=? AND request_json=? AND state='UNKNOWN'",
                    (requested.open_client_order_id, requested.id, payload),
                )
                if result.rowcount != 1:
                    raise OpenIntentError("opening intent transition is not valid")
        except (sqlite3.Error, OSError) as exc:
            raise OpenIntentError(
                "opening intent acknowledgement could not be committed"
            ) from exc

    def pending_position(self, *, scope: BinanceExecutionScope) -> OpenPosition | None:
        """Read an exact scope-bound obligation without changing it or querying a venue."""
        from .positions import OpenPosition

        if not isinstance(scope, BinanceExecutionScope):
            raise OpenIntentError("recovery requires an execution scope")
        if not Path(self.path).exists():
            return None
        try:
            with closing(self._connect()) as connection:
                self._bind_scope(connection, scope)
                rows = connection.execute(
                    "SELECT client_id, position_id, request_json FROM open_intent "
                    "WHERE state IS NOT 'RECORDED' LIMIT 2"
                ).fetchall()
            if not rows:
                return None
            if len(rows) != 1:
                raise OpenIntentError("multiple unresolved openings require review")
            client_id, position_id, request_json = rows[0]
            payload = json.loads(request_json)
            if not isinstance(payload, dict) or payload.get("intent_version") != 2:
                raise OpenIntentError("legacy opening has no verified execution scope")
            if payload.get("execution_scope") != asdict(scope):
                raise OpenIntentError("opening belongs to a different execution scope")
            position = OpenPosition(**payload["position_template"])
            if (
                position.open_client_order_id != client_id
                or position.id != position_id
                or self._request(position, scope) != request_json
            ):
                raise OpenIntentError("opening intent payload is inconsistent")
            return position
        except OpenIntentError:
            raise
        except (sqlite3.Error, OSError, KeyError, TypeError, ValueError):
            raise OpenIntentError("opening recovery record is unreadable") from None

    def validate_result(self, requested: OpenPosition, recorded: OpenPosition) -> None:
        """Reject mismatched or unresolved local fill records before ledger persistence."""
        self._request(recorded)
        if (
            recorded.id != requested.id
            or recorded.open_client_order_id != requested.open_client_order_id
            or recorded.symbol != requested.symbol
            or recorded.market_type != requested.market_type
            or recorded.side != requested.side
            or recorded.dry_run is not False
            or not recorded.open_exchange_order_id
            or recorded.owner != requested.owner
            or not isinstance(recorded.entry_price, (int, float))
            or isinstance(recorded.entry_price, bool)
            or not math.isfinite(recorded.entry_price)
            or recorded.entry_price <= 0
        ):
            raise OpenIntentError("recorded opening identity differs from its intent")
        if recorded.exchange_status.upper() not in {"FILLED", "PARTIALLY_FILLED"}:
            raise OpenIntentError("opening fill remains unresolved")
        # Match the adapter's eight-decimal transmitted quantity, allowing only
        # floating-point aggregation noise rather than an unfilled remainder.
        from decimal import Decimal

        from .binance_spot_receipts import native_spot_entry_net

        net = native_spot_entry_net(recorded)
        comparison_qty = recorded.qty
        if net is not None:
            if Decimal(str(recorded.qty)) != net:
                raise OpenIntentError(
                    "new spot fill must retain all received inventory"
                )
            if (
                Decimal(recorded.spot_gross_entry_quantity)
                != Decimal(f"{requested.qty:.8f}")
                or recorded.exchange_status != "FILLED"
            ):
                raise OpenIntentError(
                    "opening gross fill quantity differs from its intent"
                )
            return
        transmitted_qty = float(f"{requested.qty:.8f}")
        quantity_matches = math.isclose(
            comparison_qty, transmitted_qty, rel_tol=1e-12, abs_tol=1e-12
        )
        if (comparison_qty > transmitted_qty and not quantity_matches) or (
            recorded.exchange_status.upper() == "FILLED" and not quantity_matches
        ):
            raise OpenIntentError("opening fill quantity differs from its intent")

    def entry_block_reason(self, *, positions_present: bool = True) -> str | None:
        """Read the admission barrier without creating or repairing storage."""
        from .binance_close_intents import unresolved_close_count

        if not Path(self.path).exists():
            return None
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT COUNT(*), COUNT(CASE WHEN state IS NOT 'RECORDED' THEN 1 END) "
                    "FROM open_intent"
                ).fetchone()
                if row[1]:
                    return f"unresolved_opening_intents={row[1]}"
                closing_count = unresolved_close_count(connection)
                if closing_count:
                    return f"unresolved_closing_intents={closing_count}"
                from .binance_spot_roundtrip_intents import roundtrip_pending

                if roundtrip_pending(connection) is not None:
                    return "unresolved_spot_roundtrip=1"
                if row[0] and not positions_present:
                    return "recorded_openings_missing_position_ledger"
                return None
        except (sqlite3.Error, OSError, OpenIntentError):
            return "opening_intent_journal_unreadable"
