from __future__ import annotations

from contextlib import ExitStack
from datetime import UTC, date, datetime, timedelta
import hashlib
import json
from pathlib import Path
import re
from typing import Callable, Mapping

from .assets import is_supported_major_symbol, normalize_symbol
from .microstructure_warehouse import (
    SUPPORTED_TICK_ARCHIVES,
    MicrostructureWarehouse,
    _exclusive_operation_lock,
)


MERGE_EVIDENCE_SCHEMA_VERSION = "certified-tick-warehouse-merge-v1"
_IDENTIFIER_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TABLES_BY_DATA_TYPE = {
    "bookTicker": (
        "book_ticker_raw",
        "book_ticker_path_1s",
        "book_ticker_100ms",
    ),
    "trades": ("trade_raw", "trade_1s"),
    "bookDepth": ("book_depth_aggregate_raw",),
}
_INVENTORY_IDENTITY_COLUMNS = (
    "snapshot_id",
    "schema_version",
    "provider",
    "market_type",
    "symbol",
    "data_type",
    "full_history",
    "scope_start_period",
    "scope_end_period",
    "item_count",
    "first_period",
    "last_period",
    "listing_sha256",
)
_INVENTORY_ITEM_COLUMNS = (
    "period",
    "url",
    "expected_bytes",
    "last_modified",
    "etag",
    "checksum_expected_bytes",
    "checksum_last_modified",
    "checksum_etag",
)


def _canonical_sha256(payload: Mapping[str, object]) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def _quote_identifier(value: str) -> str:
    if _IDENTIFIER_PATTERN.fullmatch(value) is None:
        raise ValueError(f"unsafe SQL identifier: {value}")
    return f'"{value}"'


def _parse_period(value: str, *, name: str) -> date:
    try:
        return datetime.strptime(str(value), "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"{name} must use YYYY-MM-DD") from exc


def _period_contract(start_date: str, end_date: str) -> tuple[date, date, tuple[str, ...]]:
    first = _parse_period(start_date, name="start_date")
    last = _parse_period(end_date, name="end_date")
    if first > last:
        raise ValueError("start_date must not be after end_date")
    periods: list[str] = []
    current = first
    while current <= last:
        periods.append(current.isoformat())
        current += timedelta(days=1)
    return first, last, tuple(periods)


def _day_bounds_ms(first: date, last: date) -> tuple[int, int]:
    start_ms = int(datetime.combine(first, datetime.min.time(), tzinfo=UTC).timestamp() * 1000)
    end_ms = int(
        datetime.combine(last + timedelta(days=1), datetime.min.time(), tzinfo=UTC).timestamp()
        * 1000
    ) - 1
    return start_ms, end_ms


def _table_schema(connection, catalog: str | None, table: str) -> tuple[tuple[str, str], ...]:
    prefix = f"{_quote_identifier(catalog)}." if catalog else ""
    rows = connection.execute(
        f"DESCRIBE {prefix}{_quote_identifier(table)}"
    ).fetchall()
    return tuple((str(row[0]), str(row[1])) for row in rows)


def _inventory_contract(connection, catalog: str | None, symbol: str, data_type: str):
    prefix = f"{_quote_identifier(catalog)}." if catalog else ""
    columns = ", ".join(_quote_identifier(value) for value in _INVENTORY_IDENTITY_COLUMNS)
    snapshots = connection.execute(
        f"""
        SELECT {columns}
        FROM {prefix}archive_inventory_snapshot
        WHERE symbol = ? AND data_type = ? AND is_current
        """,
        [symbol, data_type],
    ).fetchall()
    if len(snapshots) != 1:
        raise ValueError(
            f"{catalog or 'destination'} must have exactly one current inventory snapshot"
        )
    snapshot_id = str(snapshots[0][0])
    item_columns = ", ".join(_quote_identifier(value) for value in _INVENTORY_ITEM_COLUMNS)
    items = connection.execute(
        f"""
        SELECT {item_columns}
        FROM {prefix}archive_inventory_item
        WHERE snapshot_id = ?
        ORDER BY period, url
        """,
        [snapshot_id],
    ).fetchall()
    return tuple(snapshots[0]), tuple(items)


def merge_certified_tick_warehouse(
    *,
    destination_path: str | Path,
    source_path: str | Path,
    symbol: str,
    data_type: str,
    start_date: str,
    end_date: str,
    memory_limit: str = "8GB",
    threads: int = 8,
    progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, object]:
    destination = Path(destination_path).resolve()
    source = Path(source_path).resolve()
    if destination == source:
        raise ValueError("source and destination warehouses must differ")
    if not destination.is_file() or not source.is_file():
        raise FileNotFoundError("source and destination warehouses must already exist")
    normalized_symbol = normalize_symbol(symbol)
    if not is_supported_major_symbol(normalized_symbol):
        raise ValueError(f"unsupported merge symbol: {normalized_symbol}")
    normalized_type = str(data_type).strip()
    if normalized_type not in SUPPORTED_TICK_ARCHIVES:
        raise ValueError(f"unsupported merge data type: {normalized_type}")
    first, last, expected_periods = _period_contract(start_date, end_date)
    start_ms, end_ms = _day_bounds_ms(first, last)
    lock_paths = sorted(
        {
            source.with_suffix(source.suffix + ".writer.lock"),
            destination.with_suffix(destination.suffix + ".writer.lock"),
        },
        key=lambda value: str(value).casefold(),
    )

    with ExitStack() as locks:
        for lock_path in lock_paths:
            locks.enter_context(_exclusive_operation_lock(lock_path))
        with MicrostructureWarehouse(
            source,
            memory_limit=memory_limit,
            threads=threads,
            read_only=True,
        ) as source_warehouse:
            source_certificate = source_warehouse.require_corpus_certificate(
                normalized_symbol,
                required_data_types=(normalized_type,),
                required_start_ms=start_ms,
                required_end_ms=end_ms,
                require_full_history_inventory=False,
            )

        with MicrostructureWarehouse(
            destination,
            memory_limit=memory_limit,
            threads=threads,
        ) as destination_warehouse:
            connection = destination_warehouse.connect()
            escaped_source = str(source).replace("'", "''")
            connection.execute(
                f"ATTACH '{escaped_source}' AS merge_source (READ_ONLY)"
            )
            temporary_ids_created = False
            try:
                source_inventory = _inventory_contract(
                    connection,
                    "merge_source",
                    normalized_symbol,
                    normalized_type,
                )
                destination_inventory = _inventory_contract(
                    connection,
                    None,
                    normalized_symbol,
                    normalized_type,
                )
                if source_inventory != destination_inventory:
                    raise ValueError("source and destination official inventories differ")

                tables = _TABLES_BY_DATA_TYPE[normalized_type]
                for table in (*tables, "archive_manifest"):
                    source_schema = _table_schema(connection, "merge_source", table)
                    destination_schema = _table_schema(connection, None, table)
                    if source_schema != destination_schema:
                        raise ValueError(f"warehouse table schema mismatch: {table}")

                manifest_schema = _table_schema(connection, None, "archive_manifest")
                manifest_columns = tuple(value[0] for value in manifest_schema)
                manifest_sql = ", ".join(
                    _quote_identifier(value) for value in manifest_columns
                )
                source_manifests = connection.execute(
                    f"""
                    SELECT {manifest_sql}
                    FROM merge_source.archive_manifest
                    WHERE symbol = ? AND data_type = ?
                      AND period BETWEEN ? AND ?
                      AND status = 'complete' AND is_current
                    ORDER BY period, archive_id
                    """,
                    [normalized_symbol, normalized_type, first.isoformat(), last.isoformat()],
                ).fetchall()
                period_index = manifest_columns.index("period")
                archive_index = manifest_columns.index("archive_id")
                source_by_period = {str(row[period_index]): tuple(row) for row in source_manifests}
                if tuple(sorted(source_by_period)) != expected_periods:
                    raise ValueError("source manifest periods do not exactly match merge scope")

                destination_manifests = connection.execute(
                    f"""
                    SELECT {manifest_sql}
                    FROM archive_manifest
                    WHERE symbol = ? AND data_type = ?
                      AND period BETWEEN ? AND ?
                    ORDER BY period, archive_id
                    """,
                    [normalized_symbol, normalized_type, first.isoformat(), last.isoformat()],
                ).fetchall()
                destination_by_period: dict[str, tuple[object, ...]] = {}
                for row in destination_manifests:
                    period = str(row[period_index])
                    if period in destination_by_period:
                        raise ValueError(f"destination has ambiguous manifest period: {period}")
                    destination_by_period[period] = tuple(row)

                missing_archive_ids: list[str] = []
                reused = 0
                for period in expected_periods:
                    source_row = source_by_period[period]
                    destination_row = destination_by_period.get(period)
                    if destination_row is None:
                        missing_archive_ids.append(str(source_row[archive_index]))
                    elif destination_row == source_row:
                        reused += 1
                    else:
                        raise ValueError(f"destination manifest conflicts at period {period}")

                source_archive_ids = tuple(
                    str(row[archive_index]) for row in source_manifests
                )
                duplicate_row = connection.execute(
                    """
                    SELECT count(*)
                    FROM archive_manifest destination
                    JOIN merge_source.archive_manifest source USING (archive_id)
                    WHERE source.symbol = ? AND source.data_type = ?
                      AND source.period BETWEEN ? AND ?
                      AND destination.symbol <> source.symbol
                    """,
                    [normalized_symbol, normalized_type, first.isoformat(), last.isoformat()],
                ).fetchone()
                if duplicate_row is None:
                    raise RuntimeError("archive ID conflict query returned no result")
                duplicate_archive_ids = duplicate_row[0]
                if int(duplicate_archive_ids or 0) != 0:
                    raise ValueError("source archive IDs conflict with another destination symbol")

                copied_rows: dict[str, int] = {table: 0 for table in tables}
                connection.execute("BEGIN TRANSACTION")
                try:
                    connection.execute(
                        "CREATE TEMP TABLE merge_archive_ids (archive_id VARCHAR PRIMARY KEY)"
                    )
                    temporary_ids_created = True
                    if missing_archive_ids:
                        connection.executemany(
                            "INSERT INTO merge_archive_ids VALUES (?)",
                            [(value,) for value in missing_archive_ids],
                        )
                        for index, table in enumerate(tables, start=1):
                            columns = tuple(
                                value[0] for value in _table_schema(connection, None, table)
                            )
                            column_sql = ", ".join(
                                _quote_identifier(value) for value in columns
                            )
                            count_row = connection.execute(
                                f"""
                                SELECT count(*)
                                FROM merge_source.{_quote_identifier(table)} source
                                JOIN merge_archive_ids ids USING (archive_id)
                                """
                            ).fetchone()
                            if count_row is None:
                                raise RuntimeError(
                                    f"source row-count query returned no result: {table}"
                                )
                            copied_rows[table] = int(count_row[0])
                            connection.execute(
                                f"""
                                INSERT INTO {_quote_identifier(table)} ({column_sql})
                                SELECT {column_sql}
                                FROM merge_source.{_quote_identifier(table)} source
                                JOIN merge_archive_ids ids USING (archive_id)
                                """
                            )
                            if progress is not None:
                                progress(table, index, len(tables) + 1)
                        connection.execute(
                            f"""
                            INSERT INTO archive_manifest ({manifest_sql})
                            SELECT {manifest_sql}
                            FROM merge_source.archive_manifest source
                            JOIN merge_archive_ids ids USING (archive_id)
                            """
                        )
                        if normalized_type == "bookTicker":
                            connection.execute(
                                """
                                UPDATE book_ticker_feature_build_audit
                                SET is_current = false
                                WHERE symbol = ? AND is_current
                                """,
                                [normalized_symbol],
                            )
                    if progress is not None:
                        progress("certificate", len(tables) + 1, len(tables) + 1)
                    destination_certificate = destination_warehouse.require_corpus_certificate(
                        normalized_symbol,
                        required_data_types=(normalized_type,),
                        required_start_ms=start_ms,
                        required_end_ms=end_ms,
                        require_full_history_inventory=False,
                    )
                    connection.execute("COMMIT")
                except Exception:
                    connection.execute("ROLLBACK")
                    raise
                finally:
                    if temporary_ids_created:
                        connection.execute("DROP TABLE IF EXISTS merge_archive_ids")
                        temporary_ids_created = False
            finally:
                if temporary_ids_created:
                    connection.execute("DROP TABLE IF EXISTS merge_archive_ids")
                connection.execute("DETACH merge_source")

    evidence: dict[str, object] = {
        "schema_version": MERGE_EVIDENCE_SCHEMA_VERSION,
        "status": "complete",
        "destination_path": str(destination),
        "source_path": str(source),
        "symbol": normalized_symbol,
        "data_type": normalized_type,
        "start_date": first.isoformat(),
        "end_date": last.isoformat(),
        "period_count": len(expected_periods),
        "source_manifest_count": len(source_archive_ids),
        "inserted_manifest_count": len(missing_archive_ids),
        "reused_manifest_count": reused,
        "copied_rows": copied_rows,
        "source_certificate_sha256": source_certificate["certificate_sha256"],
        "destination_certificate_sha256": destination_certificate[
            "certificate_sha256"
        ],
    }
    evidence["merge_sha256"] = _canonical_sha256(evidence)
    return evidence


__all__ = [
    "MERGE_EVIDENCE_SCHEMA_VERSION",
    "merge_certified_tick_warehouse",
]
