"""Causal one-second BTC features for the isolated Round 16 screen."""

from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

import duckdb
import numpy as np

from .polymarket_historical_dataset import (
    HistoricalDatasetManifest,
    HistoricalFeatureRow,
    _identity_digest,
    _load_flow_day,
    _log_returns,
    _source_manifest,
)
from .polymarket_historical_screen import HistoricalBtcMarket, HistoricalScreenStore
from .polymarket_round16 import (
    ROUND16_DECISION_OFFSETS_SECONDS,
    ROUND16_DURATION_MS,
    ROUND16_FLOW_WINDOWS_SECONDS,
    ROUND16_RETURN_HORIZONS_SECONDS,
    Round16HistoricalContract,
)


ROUND16_DATASET_SCHEMA_VERSION = "polymarket-round16-btc-15m-dataset-v1"
_MARKETS = ("spot", "perpetual")
_SOURCE_BOUNDARY_CENSORED_CONDITIONS = 1


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


def round16_feature_names() -> tuple[str, ...]:
    names: list[str] = []
    for market in _MARKETS:
        names.extend(
            f"{market}_log_return_{horizon}s"
            for horizon in ROUND16_RETURN_HORIZONS_SECONDS
        )
        names.extend(
            f"{market}_realized_variance_{window}s"
            for window in ROUND16_RETURN_HORIZONS_SECONDS
        )
        for window in ROUND16_FLOW_WINDOWS_SECONDS:
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
    for market in _MARKETS:
        names.extend(
            (
                f"{market}_event_to_date_log_moneyness",
                f"{market}_volatility_scaled_digital_moneyness",
            )
        )
    names.append("spot_perpetual_basis_bps")
    names.extend(
        f"spot_perpetual_basis_change_{horizon}s_bps"
        for horizon in ROUND16_RETURN_HORIZONS_SECONDS
        if horizon >= 5
    )
    names.extend(
        f"spot_minus_perpetual_log_return_{horizon}s"
        for horizon in ROUND16_RETURN_HORIZONS_SECONDS
    )
    names.extend(
        (
            "terminal_spot_log_quote_rate_ratio_30s_to_prior_120s",
            "terminal_spot_perpetual_signed_aggressive_share_difference_30s",
            "utc_day_phase_sin",
            "utc_day_phase_cos",
            "event_elapsed_fraction",
            "event_remaining_fraction",
        )
    )
    if len(names) != len(set(names)):
        raise AssertionError("Round 16 feature names are not unique")
    return tuple(names)


ROUND16_FEATURE_NAMES = round16_feature_names()
ROUND16_CALENDAR_FEATURE_NAMES = (
    "utc_day_phase_sin",
    "utc_day_phase_cos",
    "event_elapsed_fraction",
    "event_remaining_fraction",
)


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
        for horizon in ROUND16_RETURN_HORIZONS_SECONDS
    ]
    for window in ROUND16_RETURN_HORIZONS_SECONDS:
        section = log_return[end_index - window + 1 : end_index + 1]
        output.append(float(np.dot(section, section)))
    for window in ROUND16_FLOW_WINDOWS_SECONDS:
        section = slice(end_index - window + 1, end_index + 1)
        quote = float(np.sum(values[f"{market}_quote_volume"][section]))
        buy = float(np.sum(values[f"{market}_aggressive_buy_quote"][section]))
        sell = float(np.sum(values[f"{market}_aggressive_sell_quote"][section]))
        aggregate_count = float(
            np.sum(values[f"{market}_aggregate_count"][section])
        )
        constituent_count = float(
            np.sum(values[f"{market}_constituent_trade_count"][section])
        )
        maximum = float(
            np.max(values[f"{market}_maximum_aggregate_quote"][section])
        )
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


def _event_moneyness_vector(
    values: Mapping[str, np.ndarray],
    *,
    event_index: int,
    end_index: int,
    remaining_seconds: float,
) -> list[float]:
    if event_index < 1 or remaining_seconds <= 0:
        raise ValueError("Round 16 event moneyness lacks a causal opening")
    output: list[float] = []
    for market in _MARKETS:
        close = values[f"{market}_close"]
        log_return = values[f"{market}_log_return"]
        log_moneyness = math.log(close[end_index] / close[event_index - 1])
        trailing = log_return[end_index - 119 : end_index + 1]
        variance_rate = float(np.mean(np.square(trailing)))
        forecast_standard_deviation = math.sqrt(
            max(variance_rate * remaining_seconds, 1e-16)
        )
        output.extend(
            (
                log_moneyness,
                log_moneyness / forecast_standard_deviation,
            )
        )
    return output


def _signed_aggressive_share(
    values: Mapping[str, np.ndarray],
    *,
    market: str,
    start_index: int,
    end_index: int,
) -> float:
    section = slice(start_index, end_index + 1)
    quote = float(np.sum(values[f"{market}_quote_volume"][section]))
    if quote <= 0:
        return 0.0
    buy = float(np.sum(values[f"{market}_aggressive_buy_quote"][section]))
    sell = float(np.sum(values[f"{market}_aggressive_sell_quote"][section]))
    return (buy - sell) / quote


def _terminal_controls(
    values: Mapping[str, np.ndarray],
    *,
    end_index: int,
) -> tuple[float, float]:
    recent_start = end_index - 29
    prior_start = end_index - 149
    prior_end = end_index - 30
    if prior_start < 0:
        raise ValueError("Round 16 terminal controls lack 150 causal seconds")
    recent_quote_rate = float(
        np.mean(values["spot_quote_volume"][recent_start : end_index + 1])
    )
    prior_quote_rate = float(
        np.mean(values["spot_quote_volume"][prior_start : prior_end + 1])
    )
    quote_anomaly = math.log1p(recent_quote_rate) - math.log1p(prior_quote_rate)
    spot_share = _signed_aggressive_share(
        values,
        market="spot",
        start_index=recent_start,
        end_index=end_index,
    )
    perpetual_share = _signed_aggressive_share(
        values,
        market="perpetual",
        start_index=recent_start,
        end_index=end_index,
    )
    return quote_anomaly, spot_share - perpetual_share


def build_round16_feature_vector(
    *,
    event_start_ms: int,
    event_end_ms: int,
    flow_start_ms: int,
    decision_time_ms: int,
    flow: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Build the exact causal Round 16 vector for history or live scoring."""

    event_start = int(event_start_ms)
    event_end = int(event_end_ms)
    decision_time = int(decision_time_ms)
    offset, offset_remainder = divmod(decision_time - event_start, 1_000)
    if offset not in ROUND16_DECISION_OFFSETS_SECONDS:
        raise ValueError("Round 16 decision offset differs")
    if (
        offset_remainder != 0
        or event_start <= 0
        or event_start % ROUND16_DURATION_MS
        or event_end - event_start != ROUND16_DURATION_MS
    ):
        raise ValueError("Round 16 market duration differs")
    event_index, start_remainder = divmod(
        event_start - int(flow_start_ms),
        1_000,
    )
    end_index = int(event_index + offset - 1)
    if (
        start_remainder != 0
        or event_index < 1
        or end_index < 149
        or end_index >= len(flow["second_ms"])
        or int(flow["second_ms"][end_index]) + 1_000 > decision_time
    ):
        raise ValueError("Round 16 decision lacks a causal flow lookback")
    values = dict(flow)
    for name in _MARKETS:
        key = f"{name}_log_return"
        if key not in values:
            values[key] = _log_returns(values[f"{name}_close"])
    vector: list[float] = []
    for name in _MARKETS:
        vector.extend(_market_vector(values, market=name, end_index=end_index))
    remaining_seconds = (event_end - decision_time) / 1_000.0
    vector.extend(
        _event_moneyness_vector(
            values,
            event_index=int(event_index),
            end_index=end_index,
            remaining_seconds=remaining_seconds,
        )
    )
    spot = values["spot_close"]
    perpetual = values["perpetual_close"]
    basis = np.log(perpetual / spot) * 10_000.0
    vector.append(float(basis[end_index]))
    vector.extend(
        float(basis[end_index] - basis[end_index - horizon])
        for horizon in ROUND16_RETURN_HORIZONS_SECONDS
        if horizon >= 5
    )
    vector.extend(
        float(
            math.log(spot[end_index] / spot[end_index - horizon])
            - math.log(perpetual[end_index] / perpetual[end_index - horizon])
        )
        for horizon in ROUND16_RETURN_HORIZONS_SECONDS
    )
    vector.extend(_terminal_controls(values, end_index=end_index))
    day_start_ms = event_start // 86_400_000 * 86_400_000
    seconds_of_day = (decision_time - day_start_ms) / 1_000.0
    phase = 2.0 * math.pi * seconds_of_day / 86_400.0
    event_duration_seconds = ROUND16_DURATION_MS // 1_000
    vector.extend(
        (
            math.sin(phase),
            math.cos(phase),
            offset / event_duration_seconds,
            (event_duration_seconds - offset) / event_duration_seconds,
        )
    )
    feature_values = np.asarray(vector, dtype=np.float32)
    if feature_values.shape != (len(ROUND16_FEATURE_NAMES),) or np.any(
        ~np.isfinite(feature_values)
    ):
        raise ValueError("Round 16 causal feature vector is invalid")
    return feature_values


def build_round16_feature_row(
    market: HistoricalBtcMarket,
    *,
    flow_start_ms: int,
    decision_offset_seconds: int,
    flow: Mapping[str, np.ndarray],
) -> HistoricalFeatureRow:
    offset = int(decision_offset_seconds)
    decision_time_ms = market.event_start_ms + offset * 1_000
    feature_values = build_round16_feature_vector(
        event_start_ms=market.event_start_ms,
        event_end_ms=market.end_ms,
        flow_start_ms=flow_start_ms,
        decision_time_ms=decision_time_ms,
        flow=flow,
    )
    vector_sha = hashlib.sha256(
        feature_values.astype("<f4", copy=False).tobytes()
    ).hexdigest()
    row_sha = _canonical_sha256(
        {
            "schema_version": ROUND16_DATASET_SCHEMA_VERSION,
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


def _combine_flow_days(
    prior: Mapping[str, np.ndarray] | None,
    current: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    if prior is None:
        return current
    combined: dict[str, np.ndarray] = {}
    if set(prior) != set(current):
        raise ValueError("Round 16 adjacent flow-day columns differ")
    for name in current:
        combined[name] = np.concatenate((prior[name], current[name]))
    if np.any(np.diff(combined["second_ms"]) != 1_000):
        raise ValueError("Round 16 adjacent flow days are not continuous")
    return combined


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
    expected_shape = (len(rows), len(ROUND16_FEATURE_NAMES))
    if feature_matrix.shape != expected_shape:
        raise ValueError("Round 16 feature insert matrix differs")
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
    if metadata_matrix.shape != (7, len(rows)):
        raise ValueError("Round 16 feature insert metadata differs")
    vector_expression = ",".join(
        f"f.column{index}" for index in range(len(ROUND16_FEATURE_NAMES))
    )
    if progress:
        progress("round16_feature_write", {"rows_written": 0, "row_count": len(rows)})
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
            "round16_feature_write",
            {"rows_written": len(rows), "row_count": len(rows)},
        )


def _implementation_digest() -> tuple[str, Mapping[str, str]]:
    source_root = Path(__file__).parent
    paths = {
        "round16_dataset": Path(__file__),
        "shared_dataset_primitives": source_root / "polymarket_historical_dataset.py",
        "shared_store": source_root / "polymarket_historical_screen.py",
    }
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }
    return _canonical_sha256(hashes), hashes


def materialize_round16_causal_features(
    store: HistoricalScreenStore,
    contract: Round16HistoricalContract,
    *,
    flow_database_path: str | Path,
    progress: ProgressCallback | None = None,
) -> HistoricalDatasetManifest:
    """Materialize immutable target-free features from the shared flow facts."""

    if store.contract != contract.historical:
        raise ValueError("Round 16 store contract differs")
    if store.state != "identities_complete":
        raise ValueError("Round 16 features require complete identities")
    source = duckdb.connect(str(Path(flow_database_path)), read_only=True)
    try:
        source_manifest_json, source_manifest_sha = _source_manifest(
            source,
            contract.historical,
        )
        markets = store.markets()
        by_day: dict[str, list[HistoricalBtcMarket]] = {
            day: [] for day in contract.historical.eligible_days
        }
        for market in markets:
            day = (
                datetime.fromtimestamp(market.event_start_ms / 1_000, tz=UTC)
                .date()
                .isoformat()
            )
            by_day[day].append(market)
        rows: list[HistoricalFeatureRow] = []
        condition_ids: set[str] = set()
        role_counts = {"train": 0, "tune": 0, "test": 0}
        prior_flow: Mapping[str, np.ndarray] | None = None
        prior_day_start_ms: int | None = None
        for day_index, day in enumerate(
            contract.historical.eligible_days,
            start=1,
        ):
            day_start_ms = int(
                datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp() * 1_000
            )
            current_flow = _load_flow_day(source, day_start_ms=day_start_ms)
            combined_flow = _combine_flow_days(prior_flow, current_flow)
            flow_start_ms = (
                prior_day_start_ms if prior_flow is not None else day_start_ms
            )
            if flow_start_ms is None:
                raise AssertionError("Round 16 flow start is unavailable")
            day_markets = sorted(
                by_day[day],
                key=lambda value: value.event_start_ms,
            )
            day_rows = 0
            day_conditions = 0
            for market in day_markets:
                earliest_end_index = (
                    (market.event_start_ms - flow_start_ms) // 1_000
                    + ROUND16_DECISION_OFFSETS_SECONDS[0]
                    - 1
                )
                if earliest_end_index < 149:
                    continue
                condition_ids.add(market.condition_id)
                role_counts[market.role] += 1
                day_conditions += 1
                for offset in ROUND16_DECISION_OFFSETS_SECONDS:
                    rows.append(
                        build_round16_feature_row(
                            market,
                            flow_start_ms=flow_start_ms,
                            decision_offset_seconds=offset,
                            flow=combined_flow,
                        )
                    )
                    day_rows += 1
            if progress:
                progress(
                    "round16_feature_day",
                    {
                        "day": day,
                        "day_index": day_index,
                        "day_count": len(contract.historical.eligible_days),
                        "condition_count": day_conditions,
                        "row_count": day_rows,
                    },
                )
            prior_flow = current_flow
            prior_day_start_ms = day_start_ms
    finally:
        source.close()
    expected_conditions = (
        len(contract.historical.eligible_days) * contract.historical.required_market_count_per_day
        - _SOURCE_BOUNDARY_CENSORED_CONDITIONS
    )
    expected_rows = expected_conditions * len(ROUND16_DECISION_OFFSETS_SECONDS)
    if len(condition_ids) != expected_conditions or len(rows) != expected_rows:
        raise ValueError("Round 16 causal dataset coverage differs")
    if (
        role_counts["train"] < 10_000
        or role_counts["tune"] < 2_500
        or role_counts["test"] < 1_400
    ):
        raise ValueError("Round 16 causal role coverage differs")
    row_chain = "0" * 64
    for row in rows:
        row_chain = hashlib.sha256(
            (row_chain + row.row_sha256).encode("ascii")
        ).hexdigest()
    names_json = _canonical_json(ROUND16_FEATURE_NAMES)
    names_sha = hashlib.sha256(names_json.encode("ascii")).hexdigest()
    identity_sha = _identity_digest(store.connect())
    materializer_sha, implementation_hashes = _implementation_digest()
    dataset_body = {
        "schema_version": ROUND16_DATASET_SCHEMA_VERSION,
        "contract_sha256": contract.contract_sha256,
        "feature_names_sha256": names_sha,
        "binance_source_manifest_sha256": source_manifest_sha,
        "market_identity_sha256": identity_sha,
        "materializer_sha256": materializer_sha,
        "implementation_sha256": implementation_hashes,
        "row_count": len(rows),
        "condition_count": len(condition_ids),
        "role_counts": role_counts,
        "source_boundary_censored_conditions": (
            _SOURCE_BOUNDARY_CENSORED_CONDITIONS
        ),
        "row_chain_sha256": row_chain,
    }
    dataset_sha = _canonical_sha256(dataset_body)
    connection = store.connect()
    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feature.round16_dataset_manifest (
                singleton BOOLEAN PRIMARY KEY CHECK(singleton),
                manifest_json VARCHAR NOT NULL,
                dataset_sha256 VARCHAR NOT NULL,
                created_at_ms BIGINT NOT NULL
            )
            """
        )
        connection.execute("DELETE FROM feature.causal_row")
        connection.execute("DELETE FROM feature.dataset_manifest")
        connection.execute("DELETE FROM feature.round16_dataset_manifest")
        _insert_feature_rows(connection, rows, progress=progress)
        connection.execute(
            """
            INSERT INTO feature.dataset_manifest VALUES (
                true, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                ROUND16_DATASET_SCHEMA_VERSION,
                contract.contract_sha256,
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
        connection.execute(
            "INSERT INTO feature.round16_dataset_manifest VALUES (true, ?, ?, ?)",
            [
                _canonical_json(dataset_body),
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
    "ROUND16_CALENDAR_FEATURE_NAMES",
    "ROUND16_DATASET_SCHEMA_VERSION",
    "ROUND16_FEATURE_NAMES",
    "build_round16_feature_row",
    "build_round16_feature_vector",
    "materialize_round16_causal_features",
    "round16_feature_names",
]
