"""Causal DuckDB evidence adapters for queue-censored make/take research."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from numbers import Integral
from typing import Mapping

import duckdb
import numpy as np

from .microstructure_features import MicrostructureDataset


_DAY_MS = 86_400_000
_QUOTE_FIELDS = (
    "decision_time_ms",
    "arrival_time_ms",
    "quote_available_time_ms",
    "quote_transaction_time_ms",
    "quote_age_ms",
    "bid_price",
    "ask_price",
    "bid_quantity",
    "ask_quantity",
    "valid",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_json(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _numeric(
    value: object,
    *,
    dtype: np.dtype,
    fill: int | float,
) -> np.ndarray:
    if isinstance(value, np.ma.MaskedArray):
        return np.asarray(np.ma.asarray(value).filled(fill), dtype=dtype)
    return np.asarray(value, dtype=dtype)


@dataclass(frozen=True)
class HistoricalPlacementQuoteBatch:
    symbol: str
    placement_latency_ms: int
    max_quote_age_ms: int
    decision_time_ms: np.ndarray
    arrival_time_ms: np.ndarray
    quote_available_time_ms: np.ndarray
    quote_transaction_time_ms: np.ndarray
    quote_age_ms: np.ndarray
    bid_price: np.ndarray
    ask_price: np.ndarray
    bid_quantity: np.ndarray
    ask_quantity: np.ndarray
    valid: np.ndarray
    batch_sha256: str

    @property
    def rows(self) -> int:
        return int(self.decision_time_ms.size)

    def select_rows(self, indexes: np.ndarray) -> HistoricalPlacementQuoteBatch:
        selected = np.asarray(indexes)
        if (
            selected.ndim != 1
            or selected.dtype.kind not in "iu"
            or selected.size == 0
            or selected[0] < 0
            or selected[-1] >= self.rows
            or np.any(np.diff(selected) <= 0)
        ):
            raise ValueError("historical quote selection is invalid")
        arrays = {
            name: np.ascontiguousarray(getattr(self, name)[selected])
            for name in _QUOTE_FIELDS
        }
        provisional = replace(self, **arrays, batch_sha256="")
        output = replace(
            provisional,
            batch_sha256=_sha256(_quote_payload(provisional)),
        )
        validate_historical_placement_quotes(output)
        return output


def _quote_payload(batch: HistoricalPlacementQuoteBatch) -> dict[str, object]:
    return {
        "symbol": batch.symbol,
        "placement_latency_ms": batch.placement_latency_ms,
        "max_quote_age_ms": batch.max_quote_age_ms,
        "arrays": {
            name: _array_sha256(np.asarray(getattr(batch, name)))
            for name in _QUOTE_FIELDS
        },
    }


def validate_historical_placement_quotes(
    batch: HistoricalPlacementQuoteBatch,
) -> None:
    rows = batch.rows
    arrays = tuple(np.asarray(getattr(batch, name)) for name in _QUOTE_FIELDS)
    valid = np.asarray(batch.valid, dtype=np.bool_)
    if (
        batch.symbol not in {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
        or rows <= 0
        or isinstance(batch.placement_latency_ms, (bool, np.bool_))
        or not isinstance(batch.placement_latency_ms, Integral)
        or int(batch.placement_latency_ms) < 0
        or isinstance(batch.max_quote_age_ms, (bool, np.bool_))
        or not isinstance(batch.max_quote_age_ms, Integral)
        or int(batch.max_quote_age_ms) <= 0
        or any(array.shape != (rows,) for array in arrays)
        or batch.decision_time_ms.dtype != np.dtype(np.int64)
        or batch.arrival_time_ms.dtype != np.dtype(np.int64)
        or batch.valid.dtype != np.dtype(np.bool_)
        or np.any(np.diff(batch.decision_time_ms) <= 0)
        or not np.array_equal(
            batch.arrival_time_ms,
            batch.decision_time_ms + int(batch.placement_latency_ms),
        )
        or not np.any(valid)
        or not np.isfinite(batch.bid_price[valid]).all()
        or not np.isfinite(batch.ask_price[valid]).all()
        or not np.isfinite(batch.bid_quantity[valid]).all()
        or not np.isfinite(batch.ask_quantity[valid]).all()
        or np.any(batch.bid_price[valid] <= 0.0)
        or np.any(batch.ask_price[valid] <= batch.bid_price[valid])
        or np.any(batch.bid_quantity[valid] <= 0.0)
        or np.any(batch.ask_quantity[valid] <= 0.0)
        or np.any(batch.quote_available_time_ms[valid] > batch.arrival_time_ms[valid])
        or np.any(batch.quote_transaction_time_ms[valid] > batch.arrival_time_ms[valid])
        or np.any(batch.quote_age_ms[valid] < 0)
        or np.any(batch.quote_age_ms[valid] > int(batch.max_quote_age_ms))
        or not _is_sha256(batch.batch_sha256)
        or batch.batch_sha256 != _sha256(_quote_payload(batch))
    ):
        raise ValueError("historical placement quote batch is invalid")


def utc_day_chunks(start_ms: int, end_ms_exclusive: int) -> tuple[tuple[int, int], ...]:
    start = int(start_ms)
    end = int(end_ms_exclusive)
    if start < 0 or end <= start or start % _DAY_MS or end % _DAY_MS:
        raise ValueError("historical source interval must contain complete UTC days")
    return tuple((value, value + _DAY_MS) for value in range(start, end, _DAY_MS))


def select_role_decision_indexes(
    dataset: MicrostructureDataset,
    *,
    role_start_ms: int,
    role_end_ms_exclusive: int,
    feature_warmup_ms: int,
    maximum_lifecycle_ms: int,
) -> np.ndarray:
    """Select decisions whose complete feature and stress lifecycles stay in-role/day."""

    start = int(role_start_ms)
    end = int(role_end_ms_exclusive)
    warmup = int(feature_warmup_ms)
    lifecycle = int(maximum_lifecycle_ms)
    decisions = np.asarray(dataset.decision_time_ms, dtype=np.int64)
    if (
        start < 0
        or end <= start
        or start % _DAY_MS
        or end % _DAY_MS
        or warmup < 0
        or lifecycle <= 0
        or decisions.ndim != 1
        or decisions.size == 0
        or np.any(np.diff(decisions) <= 0)
    ):
        raise ValueError("historical role selection contract is invalid")
    day_end = ((decisions // _DAY_MS) + 1) * _DAY_MS
    valid = (
        (decisions >= start + warmup)
        & (decisions < end)
        & (decisions + lifecycle <= end)
        & (decisions + lifecycle <= day_end)
    )
    indexes = np.flatnonzero(valid).astype(np.int64, copy=False)
    if indexes.size == 0:
        raise ValueError("historical role has no complete decision lifecycles")
    return indexes


def load_historical_placement_quotes(
    connection: duckdb.DuckDBPyConnection,
    *,
    symbol: str,
    decision_time_ms: np.ndarray,
    placement_latency_ms: int,
    max_quote_age_ms: int,
) -> HistoricalPlacementQuoteBatch:
    """Resolve the latest fully available 100 ms BBO at each placement arrival."""

    normalized_symbol = str(symbol).strip().upper()
    decisions = np.asarray(decision_time_ms)
    latency = int(placement_latency_ms)
    maximum_age = int(max_quote_age_ms)
    if (
        normalized_symbol not in {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
        or decisions.ndim != 1
        or decisions.size == 0
        or decisions.dtype.kind not in "iu"
        or np.any(np.diff(decisions) <= 0)
        or latency < 0
        or maximum_age <= 0
    ):
        raise ValueError("historical placement quote request is invalid")
    stored_decisions = np.ascontiguousarray(decisions, dtype=np.int64)
    arrivals = stored_decisions + latency
    relation_name = f"make_take_quote_request_{id(stored_decisions):x}"
    connection.register(
        relation_name,
        {
            "row_id": np.arange(stored_decisions.size, dtype=np.int64),
            "symbol": np.full(stored_decisions.size, normalized_symbol),
            "decision_time_ms": stored_decisions,
            "arrival_time_ms": arrivals,
        },
    )
    try:
        values = connection.execute(
            f"""
            SELECT d.row_id, d.decision_time_ms, d.arrival_time_ms,
                   q.available_time_ms AS quote_available_time_ms,
                   q.last_transaction_time_ms AS quote_transaction_time_ms,
                   d.arrival_time_ms - q.last_transaction_time_ms AS quote_age_ms,
                   q.close_bid AS bid_price, q.close_ask AS ask_price,
                   q.close_bid_qty AS bid_quantity, q.close_ask_qty AS ask_quantity
            FROM {relation_name} d
            ASOF LEFT JOIN current_book_ticker_100ms q
              ON d.symbol = q.symbol
             AND q.available_time_ms <= d.arrival_time_ms
            ORDER BY d.row_id
            """
        ).fetchnumpy()
    finally:
        connection.unregister(relation_name)
    row_id = _numeric(values["row_id"], dtype=np.dtype(np.int64), fill=-1)
    available = _numeric(
        values["quote_available_time_ms"], dtype=np.dtype(np.int64), fill=-1
    )
    transaction = _numeric(
        values["quote_transaction_time_ms"], dtype=np.dtype(np.int64), fill=-1
    )
    age = _numeric(values["quote_age_ms"], dtype=np.dtype(np.int64), fill=-1)
    bid = _numeric(values["bid_price"], dtype=np.dtype(np.float64), fill=np.nan)
    ask = _numeric(values["ask_price"], dtype=np.dtype(np.float64), fill=np.nan)
    bid_qty = _numeric(
        values["bid_quantity"], dtype=np.dtype(np.float64), fill=np.nan
    )
    ask_qty = _numeric(
        values["ask_quantity"], dtype=np.dtype(np.float64), fill=np.nan
    )
    if not np.array_equal(row_id, np.arange(stored_decisions.size)):
        raise ValueError("historical placement quote rows changed order")
    valid = (
        (available >= 0)
        & (transaction >= 0)
        & (age >= 0)
        & (age <= maximum_age)
        & (available <= arrivals)
        & (transaction <= arrivals)
        & np.isfinite(bid)
        & np.isfinite(ask)
        & np.isfinite(bid_qty)
        & np.isfinite(ask_qty)
        & (bid > 0.0)
        & (ask > bid)
        & (bid_qty > 0.0)
        & (ask_qty > 0.0)
    )
    arrays = {
        "decision_time_ms": stored_decisions,
        "arrival_time_ms": arrivals,
        "quote_available_time_ms": available,
        "quote_transaction_time_ms": transaction,
        "quote_age_ms": age,
        "bid_price": bid,
        "ask_price": ask,
        "bid_quantity": bid_qty,
        "ask_quantity": ask_qty,
        "valid": np.asarray(valid, dtype=np.bool_),
    }
    provisional = HistoricalPlacementQuoteBatch(
        symbol=normalized_symbol,
        placement_latency_ms=latency,
        max_quote_age_ms=maximum_age,
        batch_sha256="",
        **arrays,
    )
    batch = replace(
        provisional,
        batch_sha256=_sha256(_quote_payload(provisional)),
    )
    for array in arrays.values():
        array.setflags(write=False)
    validate_historical_placement_quotes(batch)
    return batch


def load_historical_trade_chunk(
    connection: duckdb.DuckDBPyConnection,
    *,
    symbol: str,
    start_ms: int,
    end_ms_exclusive: int,
) -> Mapping[str, np.ndarray]:
    """Load one exact chronological raw-trade chunk for the streaming builder."""

    values = connection.execute(
        """
        SELECT trade_id, trade_time_ms, price AS trade_price,
               qty AS trade_quantity, buyer_is_maker AS trade_buyer_is_maker
        FROM current_trade_raw
        WHERE symbol = ? AND trade_time_ms >= ? AND trade_time_ms < ?
        ORDER BY trade_time_ms, trade_id
        """,
        [str(symbol).strip().upper(), int(start_ms), int(end_ms_exclusive)],
    ).fetchnumpy()
    return {
        "trade_id": np.asarray(values["trade_id"], dtype=np.int64),
        "trade_time_ms": np.asarray(values["trade_time_ms"], dtype=np.int64),
        "trade_price": np.asarray(values["trade_price"], dtype=np.float64),
        "trade_quantity": np.asarray(values["trade_quantity"], dtype=np.float64),
        "trade_buyer_is_maker": np.asarray(
            values["trade_buyer_is_maker"], dtype=np.bool_
        ),
    }


def load_historical_day_path(
    connection: duckdb.DuckDBPyConnection,
    *,
    symbol: str,
    day_start_ms: int,
) -> Mapping[str, np.ndarray]:
    """Load one UTC day of exact 100 ms BBO extrema and closes."""

    start = int(day_start_ms)
    if start < 0 or start % _DAY_MS:
        raise ValueError("historical path day is invalid")
    values = connection.execute(
        """
        SELECT bucket_ms AS path_time_ms,
               min_bid AS path_min_bid, max_bid AS path_max_bid,
               close_bid AS path_close_bid,
               min_ask AS path_min_ask, max_ask AS path_max_ask,
               close_ask AS path_close_ask
        FROM current_book_ticker_100ms
        WHERE symbol = ? AND bucket_ms >= ? AND bucket_ms < ?
        ORDER BY bucket_ms
        """,
        [str(symbol).strip().upper(), start, start + _DAY_MS],
    ).fetchnumpy()
    output = {
        "path_time_ms": np.asarray(values["path_time_ms"], dtype=np.int64),
        "path_min_bid": np.asarray(values["path_min_bid"], dtype=np.float64),
        "path_max_bid": np.asarray(values["path_max_bid"], dtype=np.float64),
        "path_close_bid": np.asarray(values["path_close_bid"], dtype=np.float64),
        "path_min_ask": np.asarray(values["path_min_ask"], dtype=np.float64),
        "path_max_ask": np.asarray(values["path_max_ask"], dtype=np.float64),
        "path_close_ask": np.asarray(values["path_close_ask"], dtype=np.float64),
    }
    times = output["path_time_ms"]
    if (
        times.size == 0
        or np.any(np.diff(times) <= 0)
        or times[0] < start
        or times[-1] >= start + _DAY_MS
    ):
        raise ValueError("historical path evidence is empty or unordered")
    return output


__all__ = [
    "HistoricalPlacementQuoteBatch",
    "load_historical_day_path",
    "load_historical_placement_quotes",
    "load_historical_trade_chunk",
    "select_role_decision_indexes",
    "utc_day_chunks",
    "validate_historical_placement_quotes",
]
