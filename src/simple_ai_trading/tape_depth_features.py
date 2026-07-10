"""Causal trade-tape and coarse-depth features with gross-return-only labels."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping

import numpy as np

from .assets import is_supported_major_symbol, normalize_symbol
from .microstructure_warehouse import (
    MicrostructureWarehouse,
    TICK_WAREHOUSE_SCHEMA_VERSION,
)


TAPE_DEPTH_FEATURE_VERSION = "tape-depth-causal-v1"
TAPE_DEPTH_TARGET_MODE = "gross_trade_reference_close_no_execution_claim"

_RETURN_WINDOWS = (1, 5, 15, 30, 60, 120, 300, 900)
_VOLATILITY_WINDOWS = (10, 30, 60, 120, 300, 900)
_RANGE_WINDOWS = (5, 15, 30, 60, 300, 900)
_FLOW_WINDOWS = (1, 5, 15, 60, 300)
_DEPTH_FEATURE_NAMES = (
    "depth_imbalance_0_2",
    "depth_imbalance_1",
    "depth_imbalance_5",
    "log_depth_notional_0_2",
    "log_depth_notional_1",
    "log_depth_notional_5",
    "log_bid_depth_curve_5_to_0_2",
    "log_ask_depth_curve_5_to_0_2",
)
TAPE_DEPTH_FEATURE_NAMES = (
    *(f"return_bps_{window}" for window in _RETURN_WINDOWS),
    *(f"realized_vol_bps_{window}" for window in _VOLATILITY_WINDOWS),
    *(f"range_bps_{window}" for window in _RANGE_WINDOWS),
    *(f"log_quote_volume_{window}" for window in _FLOW_WINDOWS),
    *(f"aggressor_imbalance_{window}" for window in _FLOW_WINDOWS),
    *(f"log_trade_count_{window}" for window in _FLOW_WINDOWS),
    "depth_available",
    "depth_age_seconds",
    *_DEPTH_FEATURE_NAMES,
    "utc_time_sin",
    "utc_time_cos",
    "utc_weekend",
)


@dataclass(frozen=True)
class TapeDepthForecastDataset:
    symbol: str
    feature_version: str
    feature_names: tuple[str, ...]
    target_mode: str
    horizon_seconds: int
    total_latency_ms: int
    decision_cadence_seconds: int
    maximum_depth_age_ms: int
    decision_time_ms: np.ndarray
    target_entry_time_ms: np.ndarray
    target_exit_time_ms: np.ndarray
    target_entry_price: np.ndarray
    target_exit_price: np.ndarray
    gross_return_bps: np.ndarray
    features: np.ndarray
    source_evidence: Mapping[str, object]

    @property
    def rows(self) -> int:
        return int(len(self.decision_time_ms))

    def summary(self) -> dict[str, object]:
        depth_index = self.feature_names.index("depth_available")
        depth_ratio = (
            float(np.mean(self.features[:, depth_index] > 0.5))
            if self.rows
            else 0.0
        )
        effective_entry_delay_ms = (
            int(self.target_entry_time_ms[0] - self.decision_time_ms[0])
            if self.rows
            else None
        )
        target_span_ms = (
            int(self.target_exit_time_ms[0] - self.decision_time_ms[0])
            if self.rows
            else None
        )
        return {
            "symbol": self.symbol,
            "feature_version": self.feature_version,
            "feature_names": list(self.feature_names),
            "target_mode": self.target_mode,
            "execution_claim": False,
            "horizon_seconds": self.horizon_seconds,
            "total_latency_ms": self.total_latency_ms,
            "effective_entry_delay_ms": effective_entry_delay_ms,
            "target_span_ms": target_span_ms,
            "decision_cadence_seconds": self.decision_cadence_seconds,
            "maximum_depth_age_ms": self.maximum_depth_age_ms,
            "rows": self.rows,
            "first_decision_time_ms": (
                int(self.decision_time_ms[0]) if self.rows else None
            ),
            "last_decision_time_ms": (
                int(self.decision_time_ms[-1]) if self.rows else None
            ),
            "depth_available_ratio": depth_ratio,
            "gross_return_mean_bps": (
                float(np.mean(self.gross_return_bps)) if self.rows else None
            ),
            "gross_return_std_bps": (
                float(np.std(self.gross_return_bps)) if self.rows else None
            ),
            "source_evidence": dict(self.source_evidence),
        }


def _manifest_source_evidence(
    warehouse: MicrostructureWarehouse,
    symbol: str,
) -> dict[str, object]:
    rows = warehouse.connect().execute(
        """
        SELECT archive_id, data_type, period, source_sha256, expected_sha256,
               checksum_status, rows_read, derived_rows, first_exchange_time_ms,
               last_exchange_time_ms, invalid_rows, duplicate_ids,
               out_of_order_rows
        FROM archive_manifest
        WHERE symbol = ? AND data_type IN ('trades', 'bookDepth')
          AND status = 'complete' AND is_current
        ORDER BY data_type, period, archive_id
        """,
        [symbol],
    ).fetchall()
    if not rows:
        raise ValueError(f"no complete trade/depth manifests exist for {symbol}")
    by_type: dict[str, list[dict[str, object]]] = {"trades": [], "bookDepth": []}
    for row in rows:
        (
            archive_id,
            data_type,
            period,
            source_sha256,
            expected_sha256,
            checksum_status,
            rows_read,
            derived_rows,
            first_ms,
            last_ms,
            invalid_rows,
            duplicate_ids,
            out_of_order_rows,
        ) = row
        source_hash = str(source_sha256).lower()
        expected_hash = str(expected_sha256).lower()
        if (
            str(checksum_status) != "verified"
            or len(source_hash) != 64
            or source_hash != expected_hash
            or int(rows_read or 0) <= 0
            or int(derived_rows or 0) <= 0
            or first_ms is None
            or last_ms is None
            or int(first_ms) > int(last_ms)
            or int(invalid_rows or 0) != 0
            or int(duplicate_ids or 0) != 0
            or int(out_of_order_rows or 0) != 0
        ):
            raise ValueError(
                f"{symbol} {data_type} manifest failed integrity: {archive_id}"
            )
        by_type[str(data_type)].append(
            {
                "archive_id": str(archive_id),
                "period": str(period),
                "source_sha256": source_hash,
                "rows_read": int(rows_read),
                "derived_rows": int(derived_rows),
                "first_exchange_time_ms": int(first_ms),
                "last_exchange_time_ms": int(last_ms),
            }
        )
    if not by_type["trades"]:
        raise ValueError(f"no verified trade manifests exist for {symbol}")
    if not by_type["bookDepth"]:
        raise ValueError(f"no verified bookDepth manifests exist for {symbol}")
    canonical = json.dumps(
        by_type,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")

    def coverage(data_type: str) -> dict[str, object]:
        values = by_type[data_type]
        return {
            "archive_count": len(values),
            "first_period": min(str(item["period"]) for item in values),
            "last_period": max(str(item["period"]) for item in values),
            "raw_rows": sum(int(item["rows_read"]) for item in values),
            "derived_rows": sum(int(item["derived_rows"]) for item in values),
            "first_exchange_time_ms": min(
                int(item["first_exchange_time_ms"]) for item in values
            ),
            "last_exchange_time_ms": max(
                int(item["last_exchange_time_ms"]) for item in values
            ),
        }

    return {
        "schema_version": TICK_WAREHOUSE_SCHEMA_VERSION,
        "symbol": symbol,
        "truth_basis": "checksummed_official_binance_data_vision_archives",
        "manifest_fingerprint": hashlib.sha256(canonical).hexdigest(),
        "trades": coverage("trades"),
        "book_depth": coverage("bookDepth"),
        "verified": True,
    }


def _window_sql(function: str, expression: str, window: int) -> str:
    return (
        f"{function}({expression}) OVER (ORDER BY second_ms "
        f"ROWS BETWEEN {window - 1} PRECEDING AND CURRENT ROW)"
    )


def _feature_sql(maximum_depth_age_ms: int) -> list[str]:
    output: list[str] = []
    for window in _RETURN_WINDOWS:
        output.append(
            f"(close / lag(close, {window}) OVER ordered - 1.0) * 10000.0 "
            f"AS return_bps_{window}"
        )
    for window in _VOLATILITY_WINDOWS:
        output.append(
            f"{_window_sql('stddev_samp', 'log_return_1', window)} * 10000.0 "
            f"AS realized_vol_bps_{window}"
        )
    for window in _RANGE_WINDOWS:
        high = _window_sql("max", "high", window)
        low = _window_sql("min", "low", window)
        output.append(f"({high} / {low} - 1.0) * 10000.0 AS range_bps_{window}")
    for window in _FLOW_WINDOWS:
        volume = _window_sql("sum", "quote_volume", window)
        output.append(f"ln(1.0 + {volume}) AS log_quote_volume_{window}")
    for window in _FLOW_WINDOWS:
        signed = _window_sql(
            "sum",
            "aggressive_buy_volume - aggressive_sell_volume",
            window,
        )
        volume = _window_sql("sum", "base_volume", window)
        output.append(
            f"{signed} / nullif({volume}, 0.0) AS aggressor_imbalance_{window}"
        )
    for window in _FLOW_WINDOWS:
        trades = _window_sql("sum", "trade_count", window)
        output.append(f"ln(1.0 + {trades}) AS log_trade_count_{window}")
    depth_available = (
        f"depth_time_ms IS NOT NULL AND depth_age_ms BETWEEN 0 AND "
        f"{maximum_depth_age_ms}"
    )
    output.extend(
        [
            f"CASE WHEN {depth_available} THEN 1.0 ELSE 0.0 END AS depth_available",
            f"CASE WHEN {depth_available} THEN depth_age_ms / 1000.0 END "
            "AS depth_age_seconds",
            *(
                f"CASE WHEN {depth_available} THEN {name} END AS {name}"
                for name in (
                    "depth_imbalance_0_2",
                    "depth_imbalance_1",
                    "depth_imbalance_5",
                )
            ),
            f"CASE WHEN {depth_available} THEN "
            "ln(1.0 + bid_notional_0_2 + ask_notional_0_2) END "
            "AS log_depth_notional_0_2",
            f"CASE WHEN {depth_available} THEN "
            "ln(1.0 + bid_notional_1 + ask_notional_1) END "
            "AS log_depth_notional_1",
            f"CASE WHEN {depth_available} THEN "
            "ln(1.0 + bid_notional_5 + ask_notional_5) END "
            "AS log_depth_notional_5",
            f"CASE WHEN {depth_available} THEN "
            "ln((1.0 + bid_notional_5) / (1.0 + bid_notional_0_2)) END "
            "AS log_bid_depth_curve_5_to_0_2",
            f"CASE WHEN {depth_available} THEN "
            "ln((1.0 + ask_notional_5) / (1.0 + ask_notional_0_2)) END "
            "AS log_ask_depth_curve_5_to_0_2",
            "sin(2.0 * pi() * ((second_ms // 1000) % 86400) / 86400.0) "
            "AS utc_time_sin",
            "cos(2.0 * pi() * ((second_ms // 1000) % 86400) / 86400.0) "
            "AS utc_time_cos",
            "CASE WHEN isodow(to_timestamp(second_ms / 1000.0)) >= 6 "
            "THEN 1.0 ELSE 0.0 END AS utc_weekend",
        ]
    )
    return output


def _as_float_array(value: object) -> np.ndarray:
    if isinstance(value, np.ma.MaskedArray):
        return np.asarray(value.filled(np.nan), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def build_tape_depth_forecast_dataset(
    warehouse: MicrostructureWarehouse,
    *,
    symbol: str,
    start_ms: int | None = None,
    end_ms: int | None = None,
    horizon_seconds: int = 60,
    total_latency_ms: int = 750,
    decision_cadence_seconds: int = 5,
    maximum_depth_age_ms: int = 60_000,
    maximum_rows: int = 5_000_000,
) -> TapeDepthForecastDataset:
    """Build a bounded causal dataset whose target is explicitly not executable P&L."""

    normalized_symbol = normalize_symbol(symbol)
    if not is_supported_major_symbol(normalized_symbol):
        raise ValueError(f"unsupported tape/depth symbol: {normalized_symbol}")
    horizon = int(horizon_seconds)
    latency = int(total_latency_ms)
    cadence = int(decision_cadence_seconds)
    depth_age = int(maximum_depth_age_ms)
    row_limit = int(maximum_rows)
    if not 1 <= horizon <= 3_600:
        raise ValueError("horizon_seconds must lie in [1, 3600]")
    if not 0 <= latency <= 60_000:
        raise ValueError("total_latency_ms must lie in [0, 60000]")
    if not 1 <= cadence <= 60:
        raise ValueError("decision_cadence_seconds must lie in [1, 60]")
    if not 1_000 <= depth_age <= 300_000:
        raise ValueError("maximum_depth_age_ms must lie in [1000, 300000]")
    if row_limit < 1:
        raise ValueError("maximum_rows must be positive")
    source_evidence = _manifest_source_evidence(warehouse, normalized_symbol)
    available = warehouse.connect().execute(
        "SELECT min(second_ms), max(second_ms) FROM current_trade_1s WHERE symbol = ?",
        [normalized_symbol],
    ).fetchone()
    if available is None or available[0] is None or available[1] is None:
        raise ValueError(f"no current one-second trade rows exist for {normalized_symbol}")
    entry_delay_seconds = max(1, int(math.ceil(latency / 1_000.0)))
    target_offset_seconds = entry_delay_seconds + horizon
    first_possible_decision = int(available[0]) + 901_000
    last_possible_decision = int(available[1]) - target_offset_seconds * 1_000 + 1_000
    requested_start = first_possible_decision if start_ms is None else int(start_ms)
    requested_end = last_possible_decision if end_ms is None else int(end_ms)
    if requested_start < first_possible_decision or requested_end > last_possible_decision:
        raise ValueError("requested tape/depth interval lacks warmup or target coverage")
    if requested_start > requested_end:
        raise ValueError("requested tape/depth interval is empty")
    estimated_rows = (requested_end - requested_start) // (cadence * 1_000) + 1
    if estimated_rows > row_limit:
        raise ValueError(
            f"requested tape/depth interval may emit {estimated_rows} rows; "
            f"maximum_rows={row_limit}"
        )
    source_start = requested_start - 1_000 - 900_000
    source_end = requested_end - 1_000 + target_offset_seconds * 1_000
    features = _feature_sql(depth_age)
    select_features = ",\n                ".join(features)
    projected_features = ", ".join(TAPE_DEPTH_FEATURE_NAMES)
    query = f"""
        WITH source AS (
            SELECT *
            FROM current_trade_depth_1s
            WHERE symbol = ? AND second_ms BETWEEN ? AND ?
        ), base AS (
            SELECT *,
                   ln(close / lag(close, 1) OVER ordered) AS log_return_1,
                   lag(second_ms, 900) OVER ordered AS history_start_ms,
                   lead(second_ms, {entry_delay_seconds}) OVER ordered AS entry_second_ms,
                   lead(close, {entry_delay_seconds}) OVER ordered AS entry_price,
                   lead(second_ms, {target_offset_seconds}) OVER ordered AS exit_second_ms,
                   lead(close, {target_offset_seconds}) OVER ordered AS exit_price
            FROM source
            WINDOW ordered AS (ORDER BY second_ms)
        ), calculated AS (
            SELECT *,
                {select_features}
            FROM base
            WINDOW ordered AS (ORDER BY second_ms)
        )
        SELECT
            second_ms + 1000 AS decision_time_ms,
            entry_second_ms + 1000 AS target_entry_time_ms,
            exit_second_ms + 1000 AS target_exit_time_ms,
            entry_price AS target_entry_price,
            exit_price AS target_exit_price,
            (exit_price / entry_price - 1.0) * 10000.0 AS gross_return_bps,
            {projected_features}
        FROM calculated
        WHERE second_ms + 1000 BETWEEN ? AND ?
          AND ((second_ms + 1000) // 1000) % {cadence} = 0
          AND history_start_ms = second_ms - 900000
          AND entry_second_ms = second_ms + {entry_delay_seconds * 1_000}
          AND exit_second_ms = second_ms + {target_offset_seconds * 1_000}
        ORDER BY decision_time_ms
    """
    values = warehouse.connect().execute(
        query,
        [
            normalized_symbol,
            source_start,
            source_end,
            requested_start,
            requested_end,
        ],
    ).fetchnumpy()
    decision_times = np.asarray(values["decision_time_ms"], dtype=np.int64)
    if decision_times.size == 0:
        raise ValueError("tape/depth query produced no fully covered rows")
    feature_matrix = np.column_stack(
        [_as_float_array(values[name]) for name in TAPE_DEPTH_FEATURE_NAMES]
    ).astype(np.float32)
    target_entry_times = np.asarray(values["target_entry_time_ms"], dtype=np.int64)
    target_exit_times = np.asarray(values["target_exit_time_ms"], dtype=np.int64)
    entry_prices = _as_float_array(values["target_entry_price"])
    exit_prices = _as_float_array(values["target_exit_price"])
    targets = _as_float_array(values["gross_return_bps"])
    effective_entry_delay_ms = entry_delay_seconds * 1_000
    target_span_ms = target_offset_seconds * 1_000
    depth_indexes = {
        TAPE_DEPTH_FEATURE_NAMES.index("depth_age_seconds"),
        *(TAPE_DEPTH_FEATURE_NAMES.index(name) for name in _DEPTH_FEATURE_NAMES),
    }
    non_depth_indexes = [
        index for index in range(feature_matrix.shape[1]) if index not in depth_indexes
    ]
    if (
        np.any(np.diff(decision_times) <= 0)
        or np.any(decision_times % (cadence * 1_000) != 0)
        or np.any(target_entry_times - decision_times != effective_entry_delay_ms)
        or np.any(target_exit_times - decision_times != target_span_ms)
        or np.any(target_exit_times - target_entry_times != horizon * 1_000)
        or not np.all(np.isfinite(feature_matrix[:, non_depth_indexes]))
        or not np.all(np.isfinite(entry_prices))
        or not np.all(np.isfinite(exit_prices))
        or not np.all(np.isfinite(targets))
        or np.any(entry_prices <= 0.0)
        or np.any(exit_prices <= 0.0)
    ):
        raise ValueError("tape/depth dataset failed its causal numeric contract")
    return TapeDepthForecastDataset(
        symbol=normalized_symbol,
        feature_version=TAPE_DEPTH_FEATURE_VERSION,
        feature_names=TAPE_DEPTH_FEATURE_NAMES,
        target_mode=TAPE_DEPTH_TARGET_MODE,
        horizon_seconds=horizon,
        total_latency_ms=latency,
        decision_cadence_seconds=cadence,
        maximum_depth_age_ms=depth_age,
        decision_time_ms=decision_times,
        target_entry_time_ms=target_entry_times,
        target_exit_time_ms=target_exit_times,
        target_entry_price=entry_prices,
        target_exit_price=exit_prices,
        gross_return_bps=targets,
        features=feature_matrix,
        source_evidence=source_evidence,
    )


__all__ = [
    "TAPE_DEPTH_FEATURE_NAMES",
    "TAPE_DEPTH_FEATURE_VERSION",
    "TAPE_DEPTH_TARGET_MODE",
    "TapeDepthForecastDataset",
    "build_tape_depth_forecast_dataset",
]
