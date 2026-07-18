"""Bounded vectorized inserts for DuckDB-derived analytical tables."""

from __future__ import annotations

from collections.abc import Sequence

import duckdb


def insert_rows_columnar(
    connection: duckdb.DuckDBPyConnection,
    *,
    sql: str,
    rows: Sequence[tuple[object, ...]],
    width: int,
    batch_size: int = 4_096,
) -> None:
    """Insert fixed-width rows through aligned UNNEST columns in bounded batches."""

    expected_width = int(width)
    bounded_batch_size = int(batch_size)
    if expected_width < 1 or bounded_batch_size < 1:
        raise ValueError("DuckDB columnar insert bounds are invalid")
    for offset in range(0, len(rows), bounded_batch_size):
        batch = rows[offset : offset + bounded_batch_size]
        if any(len(row) != expected_width for row in batch):
            raise ValueError("DuckDB columnar insert row width differs")
        parameters = tuple(list(column) for column in zip(*batch, strict=True))
        connection.execute(sql, parameters)


__all__ = ["insert_rows_columnar"]
