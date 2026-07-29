"""Causal one-second Binance features for the BTC Polymarket screen."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

import duckdb
import numpy as np

from .polymarket_historical_screen import (
    HistoricalBtcMarket,
    HistoricalScreenStore,
)


HISTORICAL_DATASET_SCHEMA_VERSION = "polymarket-historical-btc-dataset-v2"
_MARKETS = ("spot", "perpetual")
_RETURN_HORIZONS = (1, 5, 15, 30, 60)
_FLOW_WINDOWS = (1, 5, 15, 30)
_SOURCE_COLUMNS = (
    "second_ms",
    "spot_close",
    "spot_quote_volume",
    "spot_aggressive_buy_quote",
    "spot_aggressive_sell_quote",
    "spot_aggregate_count",
    "spot_constituent_trade_count",
    "spot_maximum_aggregate_quote",
    "spot_squared_aggregate_quote_sum",
    "spot_last_trade_age_seconds",
    "perpetual_close",
    "perpetual_quote_volume",
    "perpetual_aggressive_buy_quote",
    "perpetual_aggressive_sell_quote",
    "perpetual_aggregate_count",
    "perpetual_constituent_trade_count",
    "perpetual_maximum_aggregate_quote",
    "perpetual_squared_aggregate_quote_sum",
    "perpetual_last_trade_age_seconds",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def historical_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for market in _MARKETS:
        names.extend(f"{market}_log_return_{horizon}s" for horizon in _RETURN_HORIZONS)
        names.extend(
            f"{market}_realized_variance_{window}s" for window in _RETURN_HORIZONS
        )
        for window in _FLOW_WINDOWS:
            names.extend(
                (
                    f"{market}_log_quote_volume_{window}s",
                    f"{market}_signed_aggressive_quote_share_{window}s",
                    f"{market}_log_aggregate_count_{window}s",
                    f"{market}_log_constituent_trade_count_{window}s",
                    f"{market}_log_maximum_aggregate_quote_{window}s",
                    f"{market}_aggregate_quote_concentration_{window}s",
                )
            )
        names.append(f"{market}_last_trade_age_seconds")
    names.append("spot_perpetual_basis_bps")
    names.extend(
        f"spot_perpetual_basis_change_{horizon}s_bps" for horizon in (5, 15, 30, 60)
    )
    names.extend(
        f"spot_minus_perpetual_log_return_{horizon}s" for horizon in _RETURN_HORIZONS
    )
    names.extend(
        (
            "utc_day_phase_sin",
            "utc_day_phase_cos",
            "event_elapsed_fraction",
            "event_remaining_fraction",
        )
    )
    return tuple(names)


FEATURE_NAMES = historical_feature_names()
CALENDAR_FEATURE_NAMES = (
    "utc_day_phase_sin",
    "utc_day_phase_cos",
    "event_elapsed_fraction",
    "event_remaining_fraction",
)


@dataclass(frozen=True, slots=True)
class HistoricalFeatureRow:
    condition_id: str
    role: str
    event_start_ms: int
    decision_time_ms: int
    decision_offset_seconds: int
    feature_values: np.ndarray
    feature_vector_sha256: str
    row_sha256: str


@dataclass(frozen=True, slots=True)
class HistoricalDatasetManifest:
    dataset_sha256: str
    row_count: int
    condition_count: int
    role_counts: Mapping[str, int]
    feature_names_sha256: str
    binance_source_manifest_sha256: str
    market_identity_sha256: str
    materializer_sha256: str
    row_chain_sha256: str


def _plain_array(value: object, *, name: str, dtype: np.dtype) -> np.ndarray:
    if isinstance(value, np.ma.MaskedArray):
        if np.any(np.asarray(value.mask)):
            raise ValueError(f"historical Binance {name} contains NULL")
        value = value.data
    output = np.asarray(value, dtype=dtype)
    if output.ndim != 1:
        raise ValueError(f"historical Binance {name} is not one-dimensional")
    return output


def _validate_flow(values: Mapping[str, np.ndarray], *, day_start_ms: int) -> None:
    seconds = values["second_ms"]
    if (
        seconds.shape != (86_400,)
        or int(seconds[0]) != day_start_ms
        or int(seconds[-1]) != day_start_ms + 86_399_000
        or np.any(np.diff(seconds) != 1_000)
    ):
        raise ValueError("historical Binance flow does not span one exact UTC day")
    for market in _MARKETS:
        close = values[f"{market}_close"]
        if np.any(~np.isfinite(close)) or np.any(close <= 0):
            raise ValueError(f"historical Binance {market} close is invalid")
        for field in (
            "quote_volume",
            "aggressive_buy_quote",
            "aggressive_sell_quote",
            "aggregate_count",
            "constituent_trade_count",
            "maximum_aggregate_quote",
            "squared_aggregate_quote_sum",
            "last_trade_age_seconds",
        ):
            array = values[f"{market}_{field}"]
            if np.any(~np.isfinite(array)) or np.any(array < 0):
                raise ValueError(f"historical Binance {market} {field} is invalid")
        quote = values[f"{market}_quote_volume"]
        buy = values[f"{market}_aggressive_buy_quote"]
        sell = values[f"{market}_aggressive_sell_quote"]
        if np.any(np.abs(quote - buy - sell) > 1e-6):
            raise ValueError(
                f"historical Binance {market} signed quote does not reconcile"
            )


def _load_flow_day(
    connection: duckdb.DuckDBPyConnection,
    *,
    day_start_ms: int,
) -> Mapping[str, np.ndarray]:
    result = connection.execute(
        f"""
        SELECT {",".join(_SOURCE_COLUMNS)}
        FROM current_spot_perpetual_flow_1s
        WHERE symbol = 'BTCUSDT'
          AND second_ms >= ?
          AND second_ms < ?
        ORDER BY second_ms
        """,
        [day_start_ms, day_start_ms + 86_400_000],
    ).fetchnumpy()
    values: dict[str, np.ndarray] = {}
    for name in _SOURCE_COLUMNS:
        dtype = np.dtype(np.int64) if name == "second_ms" else np.dtype(np.float64)
        values[name] = _plain_array(result[name], name=name, dtype=dtype)
    _validate_flow(values, day_start_ms=day_start_ms)
    return values


def _log_returns(close: np.ndarray) -> np.ndarray:
    output = np.zeros(close.size, dtype=np.float64)
    output[1:] = np.log(close[1:] / close[:-1])
    if np.any(~np.isfinite(output)):
        raise ValueError("historical Binance log returns are invalid")
    return output


def _market_vector(
    values: Mapping[str, np.ndarray],
    *,
    market: str,
    end_index: int,
) -> list[float]:
    close = values[f"{market}_close"]
    log_return = values[f"{market}_log_return"]
    output = [
        float(math.log(close[end_index] / close[end_index - horizon]))
        for horizon in _RETURN_HORIZONS
    ]
    for window in _RETURN_HORIZONS:
        start = end_index - window + 1
        section = log_return[start : end_index + 1]
        output.append(float(np.dot(section, section)))
    for window in _FLOW_WINDOWS:
        start = end_index - window + 1
        section = slice(start, end_index + 1)
        quote = float(np.sum(values[f"{market}_quote_volume"][section]))
        buy = float(np.sum(values[f"{market}_aggressive_buy_quote"][section]))
        sell = float(np.sum(values[f"{market}_aggressive_sell_quote"][section]))
        aggregate_count = float(np.sum(values[f"{market}_aggregate_count"][section]))
        constituent_count = float(
            np.sum(values[f"{market}_constituent_trade_count"][section])
        )
        maximum = float(np.max(values[f"{market}_maximum_aggregate_quote"][section]))
        squared = float(
            np.sum(values[f"{market}_squared_aggregate_quote_sum"][section])
        )
        output.extend(
            (
                math.log1p(quote),
                (buy - sell) / quote if quote > 0 else 0.0,
                math.log1p(aggregate_count),
                math.log1p(constituent_count),
                math.log1p(maximum),
                squared / (quote * quote) if quote > 0 else 0.0,
            )
        )
    output.append(float(values[f"{market}_last_trade_age_seconds"][end_index]))
    return output


def build_historical_feature_row(
    market: HistoricalBtcMarket,
    *,
    day_start_ms: int,
    decision_offset_seconds: int,
    flow: Mapping[str, np.ndarray],
) -> HistoricalFeatureRow:
    offset = int(decision_offset_seconds)
    if offset not in (30, 60, 90, 120, 150, 180, 210, 240):
        raise ValueError("historical decision offset differs")
    event_index = (market.event_start_ms - day_start_ms) // 1_000
    decision_time_ms = market.event_start_ms + offset * 1_000
    end_index = int(event_index + offset - 1)
    if (
        market.event_start_ms < day_start_ms + 60_000
        or end_index < max(_RETURN_HORIZONS)
        or end_index >= 86_400
        or int(flow["second_ms"][end_index]) + 1_000 > decision_time_ms
    ):
        raise ValueError("historical decision lacks a causal flow lookback")
    values = dict(flow)
    for name in _MARKETS:
        key = f"{name}_log_return"
        if key not in values:
            values[key] = _log_returns(values[f"{name}_close"])
    vector: list[float] = []
    for name in _MARKETS:
        vector.extend(_market_vector(values, market=name, end_index=end_index))
    spot = values["spot_close"]
    perpetual = values["perpetual_close"]
    basis = np.log(perpetual / spot) * 10_000.0
    vector.append(float(basis[end_index]))
    vector.extend(
        float(basis[end_index] - basis[end_index - horizon])
        for horizon in (5, 15, 30, 60)
    )
    vector.extend(
        float(
            math.log(spot[end_index] / spot[end_index - horizon])
            - math.log(perpetual[end_index] / perpetual[end_index - horizon])
        )
        for horizon in _RETURN_HORIZONS
    )
    seconds_of_day = (decision_time_ms - day_start_ms) / 1_000.0
    phase = 2.0 * math.pi * seconds_of_day / 86_400.0
    vector.extend(
        (
            math.sin(phase),
            math.cos(phase),
            offset / 300.0,
            (300 - offset) / 300.0,
        )
    )
    feature_values = np.asarray(vector, dtype=np.float32)
    if feature_values.shape != (len(FEATURE_NAMES),) or np.any(
        ~np.isfinite(feature_values)
    ):
        raise ValueError("historical causal feature vector is invalid")
    vector_sha = hashlib.sha256(
        feature_values.astype("<f4", copy=False).tobytes()
    ).hexdigest()
    row_sha = _canonical_sha256(
        {
            "schema_version": HISTORICAL_DATASET_SCHEMA_VERSION,
            "market_identity_sha256": market.identity_payload_sha256,
            "condition_id": market.condition_id,
            "decision_time_ms": decision_time_ms,
            "decision_offset_seconds": offset,
            "feature_vector_sha256": vector_sha,
            "role": market.role,
        }
    )
    return HistoricalFeatureRow(
        condition_id=market.condition_id,
        role=market.role,
        event_start_ms=market.event_start_ms,
        decision_time_ms=decision_time_ms,
        decision_offset_seconds=offset,
        feature_values=feature_values,
        feature_vector_sha256=vector_sha,
        row_sha256=row_sha,
    )


def _source_manifest(
    connection: duckdb.DuckDBPyConnection,
    days: Sequence[str],
) -> tuple[str, str]:
    placeholders = ",".join("?" for _ in days)
    rows = connection.execute(
        f"""
        SELECT period, day_id, inventory_sha256, source_contract_sha256,
               combined_flow_sha256, flow_rows, symbol_count, status, is_current
        FROM spot_perpetual_flow_day_manifest
        WHERE period IN ({placeholders}) AND is_current
        ORDER BY period
        """,
        list(days),
    ).fetchall()
    if len(rows) != len(days) or tuple(str(row[0]) for row in rows) != tuple(days):
        raise ValueError("historical Binance source manifests are incomplete")
    payload = []
    for row in rows:
        if (
            int(row[5]) != 259_200
            or int(row[6]) != 3
            or str(row[7]) != "complete"
            or row[8] is not True
        ):
            raise ValueError("historical Binance source manifest failed")
        payload.append(
            {
                "period": str(row[0]),
                "day_id": str(row[1]),
                "inventory_sha256": str(row[2]),
                "source_contract_sha256": str(row[3]),
                "combined_flow_sha256": str(row[4]),
                "flow_rows": int(row[5]),
                "symbol_count": int(row[6]),
            }
        )
    canonical = _canonical_json(payload)
    return canonical, hashlib.sha256(canonical.encode("ascii")).hexdigest()


def _identity_digest(connection: duckdb.DuckDBPyConnection) -> str:
    values = [
        str(row[0])
        for row in connection.execute(
            """
            SELECT source_payload_sha256 FROM feature.market_identity
            WHERE NOT excluded ORDER BY event_start_ms
            """
        ).fetchall()
    ]
    if not values:
        raise ValueError("historical market identity receipts are missing")
    return _canonical_sha256(values)


ProgressCallback = Callable[[str, Mapping[str, object]], None]


def _insert_feature_rows(
    connection: duckdb.DuckDBPyConnection,
    rows: Sequence[HistoricalFeatureRow],
    *,
    progress: ProgressCallback | None,
) -> None:
    if not rows:
        return
    feature_matrix = np.stack(
        [row.feature_values for row in rows],
        axis=0,
    ).astype(np.float32, copy=False)
    if feature_matrix.shape != (len(rows), len(FEATURE_NAMES)):
        raise ValueError("historical feature insert matrix differs")
    feature_matrix = feature_matrix.T
    metadata_matrix = np.asarray(
        [
            [row.condition_id for row in rows],
            [row.role for row in rows],
            [row.event_start_ms for row in rows],
            [row.decision_time_ms for row in rows],
            [row.decision_offset_seconds for row in rows],
            [row.feature_vector_sha256 for row in rows],
            [row.row_sha256 for row in rows],
        ],
        dtype=object,
    )
    if feature_matrix.shape != (
        len(FEATURE_NAMES),
        len(rows),
    ) or metadata_matrix.shape != (7, len(rows)):
        raise ValueError("historical feature insert relation differs")
    vector_expression = ",".join(
        f"f.column{index}" for index in range(len(FEATURE_NAMES))
    )
    if progress:
        progress(
            "historical_feature_write",
            {
                "rows_written": 0,
                "row_count": len(rows),
            },
        )
    connection.execute(
        f"""
        INSERT INTO feature.causal_row
        WITH metadata AS (
            SELECT row_number() OVER () AS row_index, *
            FROM metadata_matrix
        ),
        features AS (
            SELECT row_number() OVER () AS row_index, *
            FROM feature_matrix
        )
        SELECT
            m.column0,
            m.column1,
            m.column2,
            m.column3,
            m.column4,
            array_value({vector_expression}),
            m.column5,
            m.column6
        FROM metadata AS m
        INNER JOIN features AS f USING (row_index)
        ORDER BY row_index
        """
    )
    if progress:
        progress(
            "historical_feature_write",
            {
                "rows_written": len(rows),
                "row_count": len(rows),
            },
        )


def materialize_historical_causal_features(
    store: HistoricalScreenStore,
    *,
    microstructure_path: str | Path,
    progress: ProgressCallback | None = None,
) -> HistoricalDatasetManifest:
    """Materialize target-free rows from audited one-second Binance flow."""

    if store.state != "identities_complete":
        raise ValueError("historical feature materialization requires identities")
    if (
        store.contract.decision_offsets_seconds != (30, 60, 90, 120, 150, 180, 210, 240)
        or store.contract.return_horizons_seconds != _RETURN_HORIZONS
        or store.contract.flow_windows_seconds != _FLOW_WINDOWS
    ):
        raise ValueError("historical feature contract differs from implementation")
    source = duckdb.connect(str(Path(microstructure_path)), read_only=True)
    try:
        source_manifest_json, source_manifest_sha = _source_manifest(
            source,
            store.contract.eligible_days,
        )
        markets = store.markets()
        by_day: dict[str, list[HistoricalBtcMarket]] = {
            day: [] for day in store.contract.eligible_days
        }
        for market in markets:
            day = (
                datetime.fromtimestamp(
                    market.event_start_ms / 1_000,
                    tz=UTC,
                )
                .date()
                .isoformat()
            )
            by_day[day].append(market)
        rows: list[HistoricalFeatureRow] = []
        condition_ids: set[str] = set()
        role_counts = {"train": 0, "tune": 0, "test": 0}
        for day_index, day in enumerate(
            store.contract.eligible_days,
            start=1,
        ):
            day_start_ms = int(
                datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp() * 1_000
            )
            flow = _load_flow_day(source, day_start_ms=day_start_ms)
            day_markets = sorted(
                by_day[day],
                key=lambda value: value.event_start_ms,
            )
            day_rows = 0
            for market in day_markets:
                if market.event_start_ms < day_start_ms + 60_000:
                    continue
                condition_ids.add(market.condition_id)
                role_counts[market.role] += 1
                for offset in store.contract.decision_offsets_seconds:
                    rows.append(
                        build_historical_feature_row(
                            market,
                            day_start_ms=day_start_ms,
                            decision_offset_seconds=offset,
                            flow=flow,
                        )
                    )
                    day_rows += 1
            if progress:
                progress(
                    "historical_feature_day",
                    {
                        "day": day,
                        "day_index": day_index,
                        "day_count": len(store.contract.eligible_days),
                        "condition_count": len(day_markets),
                        "row_count": day_rows,
                    },
                )
    finally:
        source.close()
    if (
        len(rows) < 8_000
        or len(condition_ids) < 1_000
        or any(value < 250 for value in role_counts.values())
    ):
        raise ValueError("historical causal dataset is below the frozen coverage")
    row_chain = "0" * 64
    for row in rows:
        row_chain = hashlib.sha256(
            (row_chain + row.row_sha256).encode("ascii")
        ).hexdigest()
    names_json = _canonical_json(FEATURE_NAMES)
    names_sha = hashlib.sha256(names_json.encode("ascii")).hexdigest()
    identity_sha = _identity_digest(store.connect())
    materializer_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dataset_body = {
        "schema_version": HISTORICAL_DATASET_SCHEMA_VERSION,
        "contract_sha256": store.contract.contract_sha256,
        "feature_names_sha256": names_sha,
        "binance_source_manifest_sha256": source_manifest_sha,
        "market_identity_sha256": identity_sha,
        "materializer_sha256": materializer_sha,
        "row_count": len(rows),
        "condition_count": len(condition_ids),
        "role_counts": role_counts,
        "row_chain_sha256": row_chain,
    }
    dataset_sha = _canonical_sha256(dataset_body)
    connection = store.connect()
    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute("DELETE FROM feature.causal_row")
        connection.execute("DELETE FROM feature.dataset_manifest")
        _insert_feature_rows(
            connection,
            rows,
            progress=progress,
        )
        connection.execute(
            """
            INSERT INTO feature.dataset_manifest VALUES (
                true, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                HISTORICAL_DATASET_SCHEMA_VERSION,
                store.contract.contract_sha256,
                names_json,
                names_sha,
                source_manifest_json,
                source_manifest_sha,
                identity_sha,
                materializer_sha,
                len(rows),
                len(condition_ids),
                _canonical_json(role_counts),
                row_chain,
                dataset_sha,
                time.time_ns() // 1_000_000,
            ],
        )
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    store.transition("identities_complete", "features_complete")
    return HistoricalDatasetManifest(
        dataset_sha256=dataset_sha,
        row_count=len(rows),
        condition_count=len(condition_ids),
        role_counts=role_counts,
        feature_names_sha256=names_sha,
        binance_source_manifest_sha256=source_manifest_sha,
        market_identity_sha256=identity_sha,
        materializer_sha256=materializer_sha,
        row_chain_sha256=row_chain,
    )


__all__ = [
    "CALENDAR_FEATURE_NAMES",
    "FEATURE_NAMES",
    "HISTORICAL_DATASET_SCHEMA_VERSION",
    "HistoricalDatasetManifest",
    "HistoricalFeatureRow",
    "build_historical_feature_row",
    "historical_feature_names",
    "materialize_historical_causal_features",
]
