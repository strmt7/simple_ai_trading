"""Write-ahead obligations for autonomous Binance openings, not an inventory ledger."""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .positions import OpenPosition


class OpenIntentError(ValueError):
    """An opening cannot safely proceed through its durable intent boundary."""


@dataclass(frozen=True)
class BinanceOpenIntentJournal:
    path: Path

    def _connect(self, *, create: bool = False) -> sqlite3.Connection:
        path = Path(self.path).resolve()
        existed = path.exists()
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
        mode = "rwc" if create else "ro"
        connection = sqlite3.connect(
            f"{path.as_uri()}?mode={mode}", uri=True, timeout=1.0
        )
        try:
            if create:
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("BEGIN IMMEDIATE")
                if not existed:
                    connection.execute(
                        "CREATE TABLE IF NOT EXISTS open_intent ("
                        "client_id TEXT PRIMARY KEY, position_id TEXT NOT NULL UNIQUE, "
                        "request_json TEXT NOT NULL, "
                        "state TEXT NOT NULL CHECK(state IN ('UNKNOWN', 'RECORDED')))"
                    )
                    connection.execute("PRAGMA user_version=1")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            columns = tuple(
                row[1] for row in connection.execute("PRAGMA table_info(open_intent)")
            )
            if version != 1 or columns != (
                "client_id",
                "position_id",
                "request_json",
                "state",
            ):
                raise OpenIntentError("opening intent schema is not recognized")
            return connection
        except BaseException:
            connection.close()
            raise

    @staticmethod
    def _request(position: OpenPosition) -> str:
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
            or not isinstance(position.symbol, str)
            or not position.symbol.strip()
            or isinstance(position.qty, bool)
            or not isinstance(position.qty, (int, float))
            or not math.isfinite(position.qty)
            or position.qty <= 0
        ):
            raise OpenIntentError("opening intent identity or quantity is invalid")
        payload = {
            "position_id": position.id,
            "client_id": client_id,
            "symbol": position.symbol,
            "market_type": position.market_type,
            "side": position.side,
            "quantity": position.qty,
            # Retain the original risk/ownership template for exact recovery;
            # OpenPosition has no account credentials or signed-request fields.
            "position_template": asdict(position),
        }
        try:
            return json.dumps(payload, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise OpenIntentError(
                "opening position template is not serializable"
            ) from exc

    def prepare(self, position: OpenPosition) -> None:
        """Commit UNKNOWN before transmission; an existing identity is never replayable."""
        payload = self._request(position)
        try:
            with closing(self._connect(create=True)) as connection, connection:
                if connection.execute(
                    "SELECT 1 FROM open_intent WHERE state != 'RECORDED' LIMIT 1"
                ).fetchone():
                    raise OpenIntentError(
                        "an unresolved opening requires reconciliation"
                    )
                connection.execute(
                    "INSERT INTO open_intent VALUES (?, ?, ?, 'UNKNOWN')",
                    (position.open_client_order_id, position.id, payload),
                )
        except (sqlite3.Error, OSError) as exc:
            raise OpenIntentError("opening intent could not be committed") from exc

    def record_complete(self, requested: OpenPosition, recorded: OpenPosition) -> None:
        """Release admission only after the caller persists a resolved full fill."""
        payload = self._request(requested)
        self.validate_result(requested, recorded)
        if recorded.exchange_status.upper() != "FILLED":
            # A partial fill does not settle the remaining order obligation.
            return
        try:
            with closing(self._connect(create=True)) as connection, connection:
                result = connection.execute(
                    "UPDATE open_intent SET state='RECORDED' "
                    "WHERE client_id=? AND request_json=? AND state='UNKNOWN'",
                    (requested.open_client_order_id, payload),
                )
                if result.rowcount != 1:
                    raise OpenIntentError("opening intent transition is not valid")
        except (sqlite3.Error, OSError) as exc:
            raise OpenIntentError(
                "opening intent acknowledgement could not be committed"
            ) from exc

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
        transmitted_qty = float(f"{requested.qty:.8f}")
        quantity_matches = math.isclose(
            recorded.qty, transmitted_qty, rel_tol=1e-12, abs_tol=1e-12
        )
        if (recorded.qty > transmitted_qty and not quantity_matches) or (
            recorded.exchange_status.upper() == "FILLED" and not quantity_matches
        ):
            raise OpenIntentError("opening fill quantity differs from its intent")

    def entry_block_reason(self, *, positions_present: bool = True) -> str | None:
        """Read the admission barrier without creating or repairing storage."""
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
                if row[0] and not positions_present:
                    return "recorded_openings_missing_position_ledger"
                return None
        except (sqlite3.Error, OSError, OpenIntentError):
            return "opening_intent_journal_unreadable"
