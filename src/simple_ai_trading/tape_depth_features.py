"""Causal trade-tape and coarse-depth features with gross-return-only labels."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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


TAPE_DEPTH_FEATURE_VERSION = "tape-depth-causal-v4"
TAPE_DEPTH_TARGET_MODE = "gross_trade_reference_close_no_execution_claim"

_RETURN_WINDOWS = (1, 5, 15, 30, 60, 120, 300, 900)
_VOLATILITY_WINDOWS = (10, 30, 60, 120, 300, 900)
_RANGE_WINDOWS = (5, 15, 30, 60, 300, 900)
_FLOW_WINDOWS = (1, 5, 15, 60, 300)
_VWAP_WINDOWS = (15, 60, 300, 900)
_EFFICIENCY_WINDOWS = (60, 300, 900)
_OBSERVATION_WINDOWS = (60, 300, 900)
_ACCELERATION_WINDOWS = ((5, 60), (15, 300))
_ALIGNMENT_WINDOWS = (15, 60, 300)
_CROSS_ASSET_WINDOWS = (1, 5, 15, 60, 300)
_CROSS_ASSET_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
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
_CROSS_ASSET_FEATURE_NAMES = (
    "cross_asset_context_available",
    "peer_max_trade_age_seconds",
    *(f"peer_mean_return_bps_{window}" for window in _CROSS_ASSET_WINDOWS),
    *(f"peer_return_dispersion_bps_{window}" for window in _CROSS_ASSET_WINDOWS),
    *(f"relative_return_vs_peers_bps_{window}" for window in _CROSS_ASSET_WINDOWS),
    *(f"btc_anchor_return_bps_{window}" for window in _CROSS_ASSET_WINDOWS),
    *(f"relative_return_vs_btc_bps_{window}" for window in _CROSS_ASSET_WINDOWS),
)
_TAPE_DEPTH_LOCAL_FEATURE_NAMES = (
    *(f"return_bps_{window}" for window in _RETURN_WINDOWS),
    *(f"realized_vol_bps_{window}" for window in _VOLATILITY_WINDOWS),
    *(f"range_bps_{window}" for window in _RANGE_WINDOWS),
    *(f"log_quote_volume_{window}" for window in _FLOW_WINDOWS),
    *(f"aggressor_imbalance_{window}" for window in _FLOW_WINDOWS),
    *(f"log_trade_count_{window}" for window in _FLOW_WINDOWS),
    *(f"vwap_deviation_bps_{window}" for window in _VWAP_WINDOWS),
    *(f"price_efficiency_{window}" for window in _EFFICIENCY_WINDOWS),
    *(f"trade_observation_rate_{window}" for window in _OBSERVATION_WINDOWS),
    *(
        f"quote_volume_rate_acceleration_{short}_{long}"
        for short, long in _ACCELERATION_WINDOWS
    ),
    *(
        f"trade_rate_acceleration_{short}_{long}"
        for short, long in _ACCELERATION_WINDOWS
    ),
    *(f"flow_price_alignment_{window}" for window in _ALIGNMENT_WINDOWS),
    "trade_observed",
    "trade_age_seconds",
    "depth_available",
    "fine_depth_0_2_available",
    "depth_age_seconds",
    *_DEPTH_FEATURE_NAMES,
    "utc_time_sin",
    "utc_time_cos",
    "utc_weekend",
)
TAPE_DEPTH_FEATURE_NAMES = (
    *_TAPE_DEPTH_LOCAL_FEATURE_NAMES,
    *_CROSS_ASSET_FEATURE_NAMES,
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
        fine_depth_index = self.feature_names.index("fine_depth_0_2_available")
        fine_depth_ratio = (
            float(np.mean(self.features[:, fine_depth_index] > 0.5))
            if self.rows
            else 0.0
        )
        cross_asset_index = self.feature_names.index("cross_asset_context_available")
        cross_asset_ratio = (
            float(np.mean(self.features[:, cross_asset_index] > 0.5))
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
            "fine_depth_0_2_available_ratio": fine_depth_ratio,
            "cross_asset_context_available_ratio": cross_asset_ratio,
            "gross_return_mean_bps": (
                float(np.mean(self.gross_return_bps)) if self.rows else None
            ),
            "gross_return_std_bps": (
                float(np.std(self.gross_return_bps)) if self.rows else None
            ),
            "dataset_fingerprint": tape_depth_dataset_fingerprint(self),
            "source_evidence": dict(self.source_evidence),
        }


def tape_depth_dataset_fingerprint(dataset: TapeDepthForecastDataset) -> str:
    """Hash the exact causal matrix, labels, timing, and source evidence."""

    contract = {
        "symbol": dataset.symbol,
        "feature_version": dataset.feature_version,
        "feature_names": list(dataset.feature_names),
        "target_mode": dataset.target_mode,
        "horizon_seconds": int(dataset.horizon_seconds),
        "total_latency_ms": int(dataset.total_latency_ms),
        "decision_cadence_seconds": int(dataset.decision_cadence_seconds),
        "maximum_depth_age_ms": int(dataset.maximum_depth_age_ms),
        "source_evidence": dict(dataset.source_evidence),
    }
    digest = hashlib.sha256(
        json.dumps(
            contract,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    )
    arrays = (
        ("decision_time_ms", dataset.decision_time_ms, "<i8"),
        ("target_entry_time_ms", dataset.target_entry_time_ms, "<i8"),
        ("target_exit_time_ms", dataset.target_exit_time_ms, "<i8"),
        ("target_entry_price", dataset.target_entry_price, "<f8"),
        ("target_exit_price", dataset.target_exit_price, "<f8"),
        ("gross_return_bps", dataset.gross_return_bps, "<f8"),
        ("features", dataset.features, "<f4"),
    )
    for name, values, dtype in arrays:
        canonical = np.ascontiguousarray(np.asarray(values, dtype=dtype))
        if np.issubdtype(canonical.dtype, np.floating) and np.any(np.isnan(canonical)):
            canonical = canonical.copy()
            canonical[np.isnan(canonical)] = np.nan
        digest.update(name.encode("ascii") + b"\x00")
        digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
        digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def tape_depth_source_evidence(
    warehouse: MicrostructureWarehouse,
    symbol: str,
    *,
    required_start_ms: int,
    required_end_ms: int,
    include_book_depth: bool = True,
) -> dict[str, object]:
    data_type_filter = (
        "data_type IN ('trades', 'bookDepth')"
        if include_book_depth
        else "data_type = 'trades'"
    )
    rows = warehouse.connect().execute(
        f"""
        SELECT archive_id, data_type, period, source_sha256, expected_sha256,
               checksum_status, rows_read, derived_rows, first_exchange_time_ms,
               last_exchange_time_ms, invalid_rows, duplicate_ids,
               out_of_order_rows
        FROM archive_manifest
        WHERE symbol = ? AND {data_type_filter}
          AND status = 'complete' AND is_current
        ORDER BY data_type, period, archive_id
        """,
        [symbol],
    ).fetchall()
    start_date = datetime.fromtimestamp(required_start_ms / 1_000, tz=UTC).date()
    end_date = datetime.fromtimestamp(required_end_ms / 1_000, tz=UTC).date()
    if start_date > end_date:
        raise ValueError("manifest evidence interval is empty")
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
        try:
            period_date = datetime.strptime(str(period), "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"manifest period is invalid: {archive_id}") from exc
        if period_date < start_date or period_date > end_date:
            continue
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
    required_trade_periods: set[str] = set()
    cursor = start_date
    while cursor <= end_date:
        required_trade_periods.add(cursor.isoformat())
        cursor += timedelta(days=1)
    actual_trade_periods = {str(item["period"]) for item in by_type["trades"]}
    missing_trade_periods = sorted(required_trade_periods - actual_trade_periods)
    if missing_trade_periods:
        preview = ",".join(missing_trade_periods[:3])
        raise ValueError(
            f"{symbol} trade manifest coverage has {len(missing_trade_periods)} "
            f"missing day(s), beginning {preview}"
        )
    if by_type["bookDepth"]:
        depth_periods = {str(item["period"]) for item in by_type["bookDepth"]}
        depth_cursor = datetime.strptime(min(depth_periods), "%Y-%m-%d").date()
        depth_end = datetime.strptime(max(depth_periods), "%Y-%m-%d").date()
        while depth_cursor <= depth_end:
            if depth_cursor.isoformat() not in depth_periods:
                raise ValueError(
                    f"{symbol} bookDepth manifest coverage is missing "
                    f"{depth_cursor.isoformat()}"
                )
            depth_cursor += timedelta(days=1)
    canonical = json.dumps(
        by_type,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")

    def coverage(data_type: str) -> dict[str, object]:
        values = by_type[data_type]
        if not values:
            return {
                "archive_count": 0,
                "first_period": None,
                "last_period": None,
                "raw_rows": 0,
                "derived_rows": 0,
                "first_exchange_time_ms": None,
                "last_exchange_time_ms": None,
            }
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
        "required_first_period": start_date.isoformat(),
        "required_last_period": end_date.isoformat(),
        "manifest_fingerprint": hashlib.sha256(canonical).hexdigest(),
        "trades": coverage("trades"),
        "book_depth": coverage("bookDepth"),
        "verified": True,
    }


def tape_depth_dataset_source_evidence(
    warehouse: MicrostructureWarehouse,
    symbol: str,
    *,
    required_start_ms: int,
    required_end_ms: int,
    peer_feature_start_ms: int,
    peer_feature_end_ms: int,
) -> dict[str, object]:
    """Bind target tape/depth and every causally consumed peer trade archive."""

    normalized = normalize_symbol(symbol)
    target = tape_depth_source_evidence(
        warehouse,
        normalized,
        required_start_ms=required_start_ms,
        required_end_ms=required_end_ms,
    )
    peers: dict[str, object] = {}
    feature_start_ms = int(peer_feature_start_ms)
    feature_end_ms = int(peer_feature_end_ms)
    peer_source_start_ms = feature_start_ms - max(_CROSS_ASSET_WINDOWS) * 1_000
    if (
        peer_source_start_ms < int(required_start_ms)
        or feature_end_ms > int(required_end_ms)
        or feature_start_ms > feature_end_ms
    ):
        raise ValueError("cross-asset evidence interval is outside target evidence")
    for peer in _CROSS_ASSET_SYMBOLS:
        if peer == normalized:
            continue
        coverage = warehouse.connect().execute(
            "SELECT min(second_ms), max(second_ms) FROM current_trade_1s "
            "WHERE symbol = ?",
            [peer],
        ).fetchone()
        if coverage is None or coverage[0] is None or coverage[1] is None:
            raise ValueError(f"cross-asset context has no one-second rows for {peer}")
        first_ms = int(coverage[0])
        last_ms = int(coverage[1])
        if feature_end_ms > last_ms:
            raise ValueError(
                f"cross-asset context ends before the target interval for {peer}"
            )
        if first_ms > feature_end_ms:
            peers[peer] = {
                "verified": True,
                "status": "not_listed_during_interval",
                "first_available_second_ms": first_ms,
                "last_available_second_ms": last_ms,
                "required_start_ms": peer_source_start_ms,
                "required_end_ms": feature_end_ms,
            }
            continue
        evidence_start_ms = max(peer_source_start_ms, first_ms)
        prior = warehouse.connect().execute(
            "SELECT max(second_ms) FROM current_trade_1s "
            "WHERE symbol = ? AND second_ms < ?",
            [peer, evidence_start_ms],
        ).fetchone()
        if prior is not None and prior[0] is not None:
            evidence_start_ms = min(evidence_start_ms, int(prior[0]))
        peer_evidence = tape_depth_source_evidence(
            warehouse,
            peer,
            required_start_ms=evidence_start_ms,
            required_end_ms=feature_end_ms,
            include_book_depth=False,
        )
        peers[peer] = {
            **peer_evidence,
            "required_start_ms": evidence_start_ms,
            "required_end_ms": feature_end_ms,
        }
    combined = {
        "target": target,
        "peers": peers,
        "context_symbols": list(_CROSS_ASSET_SYMBOLS),
        "peer_feature_start_ms": feature_start_ms,
        "peer_feature_end_ms": feature_end_ms,
    }
    canonical = json.dumps(
        combined,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return {
        **target,
        "truth_basis": "checksummed_target_tape_depth_and_peer_trade_archives",
        "manifest_fingerprint": hashlib.sha256(canonical).hexdigest(),
        "cross_asset_context": {
            "symbols": list(_CROSS_ASSET_SYMBOLS),
            "peers": peers,
            "causal_join": "asof_at_or_before_feature_second",
            "required_feature_start_ms": feature_start_ms,
            "required_feature_end_ms": feature_end_ms,
        },
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
            f"coalesce({signed} / nullif({volume}, 0.0), 0.0) "
            f"AS aggressor_imbalance_{window}"
        )
    for window in _FLOW_WINDOWS:
        trades = _window_sql("sum", "trade_count", window)
        output.append(f"ln(1.0 + {trades}) AS log_trade_count_{window}")
    for window in _VWAP_WINDOWS:
        quote_volume = _window_sql("sum", "quote_volume", window)
        base_volume = _window_sql("sum", "base_volume", window)
        output.append(
            f"coalesce((close / ({quote_volume} / nullif({base_volume}, 0.0)) "
            f"- 1.0) * 10000.0, 0.0) AS vwap_deviation_bps_{window}"
        )
    for window in _EFFICIENCY_WINDOWS:
        absolute_path = _window_sql("sum", "abs(log_return_1)", window)
        output.append(
            f"coalesce(abs(ln(close / lag(close, {window}) OVER ordered)) "
            f"/ nullif({absolute_path}, 0.0), 0.0) AS price_efficiency_{window}"
        )
    for window in _OBSERVATION_WINDOWS:
        observed = _window_sql(
            "avg",
            "CASE WHEN trade_second_ms = second_ms THEN 1.0 ELSE 0.0 END",
            window,
        )
        output.append(f"{observed} AS trade_observation_rate_{window}")
    for short, long in _ACCELERATION_WINDOWS:
        short_volume = _window_sql("sum", "quote_volume", short)
        long_volume = _window_sql("sum", "quote_volume", long)
        output.append(
            f"ln((1e-9 + {short_volume} / {short}.0) / "
            f"(1e-9 + {long_volume} / {long}.0)) "
            f"AS quote_volume_rate_acceleration_{short}_{long}"
        )
    for short, long in _ACCELERATION_WINDOWS:
        short_trades = _window_sql("sum", "trade_count", short)
        long_trades = _window_sql("sum", "trade_count", long)
        output.append(
            f"ln((1e-9 + {short_trades} / {short}.0) / "
            f"(1e-9 + {long_trades} / {long}.0)) "
            f"AS trade_rate_acceleration_{short}_{long}"
        )
    for window in _ALIGNMENT_WINDOWS:
        signed = _window_sql(
            "sum",
            "aggressive_buy_volume - aggressive_sell_volume",
            window,
        )
        volume = _window_sql("sum", "base_volume", window)
        output.append(
            f"coalesce({signed} / nullif({volume}, 0.0), 0.0) "
            f"* (close / lag(close, {window}) OVER ordered - 1.0) * 10000.0 "
            f"AS flow_price_alignment_{window}"
        )
    depth_available = (
        f"depth_time_ms IS NOT NULL AND depth_age_ms BETWEEN 0 AND "
        f"{maximum_depth_age_ms}"
    )
    output.extend(
        [
            "CASE WHEN trade_second_ms = second_ms THEN 1.0 ELSE 0.0 END "
            "AS trade_observed",
            "(second_ms - trade_second_ms) / 1000.0 AS trade_age_seconds",
            f"CASE WHEN {depth_available} THEN 1.0 ELSE 0.0 END AS depth_available",
            f"CASE WHEN {depth_available} AND bid_notional_0_2 IS NOT NULL "
            "AND ask_notional_0_2 IS NOT NULL THEN 1.0 ELSE 0.0 END "
            "AS fine_depth_0_2_available",
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
        return np.asarray(
            np.ma.asarray(value, dtype=np.float64).filled(np.nan),
            dtype=np.float64,
        )
    return np.asarray(value, dtype=np.float64)


def _peer_return_context(
    warehouse: MicrostructureWarehouse,
    *,
    symbol: str,
    feature_seconds_ms: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray]:
    if len(feature_seconds_ms) < 1:
        raise ValueError("cross-asset context has no feature seconds")
    cadence_ms = (
        int(feature_seconds_ms[1] - feature_seconds_ms[0])
        if len(feature_seconds_ms) > 1
        else 1_000
    )
    if cadence_ms <= 0 or np.any(np.diff(feature_seconds_ms) != cadence_ms):
        raise ValueError("cross-asset feature seconds are not regularly spaced")
    lag_values = ", ".join(f"({window})" for window in (0, *_CROSS_ASSET_WINDOWS))
    projections = [
        "max(trade_second_ms) FILTER (WHERE lag_seconds = 0) "
        "AS current_trade_second_ms",
        "max(peer_close) FILTER (WHERE lag_seconds = 0) AS current_close",
        *(
            f"max(peer_close) FILTER (WHERE lag_seconds = {window}) "
            f"AS close_{window}"
            for window in _CROSS_ASSET_WINDOWS
        ),
    ]
    source_start_ms = int(feature_seconds_ms[0]) - max(_CROSS_ASSET_WINDOWS) * 1_000
    parameters: list[object] = [
        symbol,
        source_start_ms,
        int(feature_seconds_ms[-1]),
        symbol,
        source_start_ms,
        int(feature_seconds_ms[0]),
        int(feature_seconds_ms[-1]),
        cadence_ms,
    ]
    query = f"""
        WITH peer_source AS MATERIALIZED (
            SELECT second_ms, close FROM current_trade_1s
            WHERE symbol = ? AND second_ms BETWEEN ? AND ?
            UNION ALL
            (SELECT second_ms, close FROM current_trade_1s
             WHERE symbol = ? AND second_ms < ?
             ORDER BY second_ms DESC LIMIT 1)
        ), peer_trades AS MATERIALIZED (
            SELECT * FROM peer_source ORDER BY second_ms
        ), decisions AS (
            SELECT value AS second_ms
            FROM generate_series(?::BIGINT, ?::BIGINT, ?::BIGINT) AS generated(value)
        ), offsets(lag_seconds) AS (
            VALUES {lag_values}
        ), requests AS MATERIALIZED (
            SELECT d.second_ms, o.lag_seconds,
                   d.second_ms - o.lag_seconds * 1000 AS request_ms
            FROM decisions d
            CROSS JOIN offsets o
        ), matched AS (
            SELECT r.second_ms, r.lag_seconds,
                   p.second_ms AS trade_second_ms,
                   p.close AS peer_close
            FROM (SELECT * FROM requests ORDER BY request_ms) r
            ASOF LEFT JOIN peer_trades p
              ON r.request_ms >= p.second_ms
        )
        SELECT second_ms, {', '.join(projections)}
        FROM matched
        GROUP BY second_ms
        ORDER BY second_ms
    """
    values = warehouse.connect().execute(query, parameters).fetchnumpy()
    seconds = np.asarray(values.pop("second_ms"), dtype=np.int64)
    if not np.array_equal(seconds, feature_seconds_ms):
        raise ValueError(f"cross-asset context clock drifted for {symbol}")
    current_seconds = _as_float_array(values.pop("current_trade_second_ms"))
    current_close = _as_float_array(values.pop("current_close"))
    current_valid = (
        np.isfinite(current_seconds)
        & np.isfinite(current_close)
        & (current_close > 0.0)
        & (current_seconds <= feature_seconds_ms)
    )
    trade_age = np.where(
        current_valid,
        (feature_seconds_ms - current_seconds) / 1_000.0,
        0.0,
    ).astype(np.float32)
    returns: dict[int, np.ndarray] = {}
    validity: dict[int, np.ndarray] = {}
    for window in _CROSS_ASSET_WINDOWS:
        lagged = _as_float_array(values.pop(f"close_{window}"))
        valid = current_valid & np.isfinite(lagged) & (lagged > 0.0)
        returns[window] = np.where(
            valid,
            (current_close / lagged - 1.0) * 10_000.0,
            0.0,
        ).astype(np.float32)
        validity[window] = valid
    return returns, validity, trade_age


def _write_cross_asset_features(
    warehouse: MicrostructureWarehouse,
    *,
    symbol: str,
    decision_times_ms: np.ndarray,
    target_features: np.ndarray,
    output: np.ndarray,
) -> None:
    feature_seconds = np.asarray(decision_times_ms, dtype=np.int64) - 1_000
    expected_shape = (len(feature_seconds), len(_CROSS_ASSET_FEATURE_NAMES))
    if output.shape != expected_shape or output.dtype != np.float32:
        raise ValueError("cross-asset output buffer has the wrong contract")
    peers = tuple(peer for peer in _CROSS_ASSET_SYMBOLS if peer != symbol)
    peer_context = {
        peer: _peer_return_context(
            warehouse,
            symbol=peer,
            feature_seconds_ms=feature_seconds,
        )
        for peer in peers
    }
    context_valid = np.ones(len(feature_seconds), dtype=bool)
    for peer in peers:
        for window in _CROSS_ASSET_WINDOWS:
            context_valid &= peer_context[peer][1][window]
    peer_age = np.maximum(peer_context[peers[0]][2], peer_context[peers[1]][2])

    def assign(name: str, value: np.ndarray) -> None:
        output[:, _CROSS_ASSET_FEATURE_NAMES.index(name)] = value

    assign("cross_asset_context_available", context_valid.astype(np.float32))
    assign(
        "peer_max_trade_age_seconds",
        np.where(context_valid, peer_age, 0.0),
    )
    for window in _CROSS_ASSET_WINDOWS:
        first = peer_context[peers[0]][0][window]
        second = peer_context[peers[1]][0][window]
        valid = (
            peer_context[peers[0]][1][window]
            & peer_context[peers[1]][1][window]
        )
        peer_mean = np.where(valid, (first + second) / 2.0, 0.0)
        peer_dispersion = np.where(valid, np.abs(first - second), 0.0)
        target_return = target_features[
            :, _TAPE_DEPTH_LOCAL_FEATURE_NAMES.index(f"return_bps_{window}")
        ]
        if symbol == "BTCUSDT":
            btc_return = target_return
            btc_valid = np.ones(len(target_return), dtype=bool)
        else:
            btc_return = peer_context["BTCUSDT"][0][window]
            btc_valid = peer_context["BTCUSDT"][1][window]
        assign(f"peer_mean_return_bps_{window}", peer_mean)
        assign(f"peer_return_dispersion_bps_{window}", peer_dispersion)
        assign(
            f"relative_return_vs_peers_bps_{window}",
            np.where(valid, target_return - peer_mean, 0.0),
        )
        assign(
            f"btc_anchor_return_bps_{window}",
            np.where(btc_valid, btc_return, 0.0),
        )
        assign(
            f"relative_return_vs_btc_bps_{window}",
            np.where(btc_valid, target_return - btc_return, 0.0),
        )


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
    prior_row = warehouse.connect().execute(
        "SELECT max(second_ms) FROM current_trade_1s "
        "WHERE symbol = ? AND second_ms <= ?",
        [normalized_symbol, source_start],
    ).fetchone()
    if prior_row is None or prior_row[0] is None:
        raise ValueError("tape/depth interval has no prior verified trade reference")
    evidence_start = min(source_start, int(prior_row[0]))
    source_evidence = tape_depth_dataset_source_evidence(
        warehouse,
        normalized_symbol,
        required_start_ms=evidence_start,
        required_end_ms=source_end,
        peer_feature_start_ms=requested_start - 1_000,
        peer_feature_end_ms=requested_end - 1_000,
    )
    features = _feature_sql(depth_age)
    select_features = ",\n                ".join(features)
    projected_features = ", ".join(
        f"CAST({name} AS FLOAT) AS {name}"
        for name in _TAPE_DEPTH_LOCAL_FEATURE_NAMES
    )
    query = f"""
        WITH clock AS (
            SELECT ?::VARCHAR AS symbol, value AS second_ms
            FROM generate_series(?::BIGINT, ?::BIGINT, 1000) AS generated(value)
        ), trades_in_window AS (
            SELECT
                symbol,
                second_ms AS trade_second_ms,
                open, high, low, close,
                    base_volume, quote_volume,
                    aggressive_buy_volume, aggressive_sell_volume,
                    trade_imbalance AS aggressor_imbalance, trade_count
            FROM current_trade_1s
            WHERE symbol = ? AND second_ms BETWEEN ? AND ?
        ), prior_trade AS (
            SELECT
                symbol,
                second_ms AS trade_second_ms,
                open, high, low, close,
                base_volume, quote_volume,
                aggressive_buy_volume, aggressive_sell_volume,
                trade_imbalance AS aggressor_imbalance, trade_count
            FROM current_trade_1s
            WHERE symbol = ? AND second_ms < ?
            ORDER BY second_ms DESC
            LIMIT 1
        ), trades AS (
            SELECT * FROM prior_trade
            UNION ALL
            SELECT * FROM trades_in_window
        ), continuous AS (
            SELECT
                c.symbol,
                c.second_ms,
                t.trade_second_ms,
                CASE WHEN t.trade_second_ms = c.second_ms THEN t.open ELSE t.close END AS open,
                CASE WHEN t.trade_second_ms = c.second_ms THEN t.high ELSE t.close END AS high,
                CASE WHEN t.trade_second_ms = c.second_ms THEN t.low ELSE t.close END AS low,
                t.close,
                CASE WHEN t.trade_second_ms = c.second_ms THEN t.base_volume ELSE 0.0 END AS base_volume,
                CASE WHEN t.trade_second_ms = c.second_ms THEN t.quote_volume ELSE 0.0 END AS quote_volume,
                CASE WHEN t.trade_second_ms = c.second_ms THEN t.aggressive_buy_volume ELSE 0.0 END AS aggressive_buy_volume,
                CASE WHEN t.trade_second_ms = c.second_ms THEN t.aggressive_sell_volume ELSE 0.0 END AS aggressive_sell_volume,
                CASE WHEN t.trade_second_ms = c.second_ms THEN t.aggressor_imbalance ELSE 0.0 END AS aggressor_imbalance,
                CASE WHEN t.trade_second_ms = c.second_ms THEN t.trade_count ELSE 0 END AS trade_count
            FROM clock c
            ASOF LEFT JOIN trades t
              ON c.symbol = t.symbol AND c.second_ms >= t.trade_second_ms
        ), source AS (
            SELECT
                t.*,
                d.timestamp_ms AS depth_time_ms,
                t.second_ms - d.timestamp_ms AS depth_age_ms,
                d.bid_depth_0_2, d.ask_depth_0_2,
                d.bid_notional_0_2, d.ask_notional_0_2,
                d.bid_depth_1, d.ask_depth_1,
                d.bid_notional_1, d.ask_notional_1,
                d.bid_depth_5, d.ask_depth_5,
                d.bid_notional_5, d.ask_notional_5,
                d.depth_imbalance_0_2,
                d.depth_imbalance_1,
                d.depth_imbalance_5
            FROM continuous t
            ASOF LEFT JOIN current_book_depth_snapshots d
              ON t.symbol = d.symbol AND t.second_ms >= d.timestamp_ms
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
            normalized_symbol,
            source_start,
            source_end,
            normalized_symbol,
            source_start,
            requested_start,
            requested_end,
        ],
    ).fetchnumpy()
    decision_times = np.asarray(values.pop("decision_time_ms"), dtype=np.int64)
    if decision_times.size == 0:
        raise ValueError("tape/depth query produced no fully covered rows")
    target_entry_times = np.asarray(values.pop("target_entry_time_ms"), dtype=np.int64)
    target_exit_times = np.asarray(values.pop("target_exit_time_ms"), dtype=np.int64)
    entry_prices = _as_float_array(values.pop("target_entry_price"))
    exit_prices = _as_float_array(values.pop("target_exit_price"))
    targets = _as_float_array(values.pop("gross_return_bps"))
    feature_matrix = np.empty(
        (decision_times.size, len(TAPE_DEPTH_FEATURE_NAMES)),
        dtype=np.float32,
    )
    for index, name in enumerate(_TAPE_DEPTH_LOCAL_FEATURE_NAMES):
        feature_matrix[:, index] = _as_float_array(values.pop(name))
    if values:
        raise ValueError("tape/depth query returned unexpected projected columns")
    _write_cross_asset_features(
        warehouse,
        symbol=normalized_symbol,
        decision_times_ms=decision_times,
        target_features=feature_matrix[:, : len(_TAPE_DEPTH_LOCAL_FEATURE_NAMES)],
        output=feature_matrix[:, len(_TAPE_DEPTH_LOCAL_FEATURE_NAMES) :],
    )
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


def slice_tape_depth_forecast_dataset(
    dataset: TapeDepthForecastDataset,
    *,
    start_ms: int,
    end_ms: int,
    source_evidence: Mapping[str, object],
) -> TapeDepthForecastDataset:
    """Create a contiguous fold view from a larger causal feature build."""

    start = int(start_ms)
    end = int(end_ms)
    if start > end:
        raise ValueError("tape/depth slice interval is empty")
    left = int(np.searchsorted(dataset.decision_time_ms, start, side="left"))
    right = int(np.searchsorted(dataset.decision_time_ms, end, side="right"))
    if (
        left >= right
        or left < 0
        or right > dataset.rows
        or int(dataset.decision_time_ms[left]) != start
        or int(dataset.decision_time_ms[right - 1]) != end
    ):
        raise ValueError("tape/depth slice boundaries are absent from the feature build")
    if not bool(source_evidence.get("verified")):
        raise ValueError("tape/depth slice source evidence is not verified")
    values = slice(left, right)
    return TapeDepthForecastDataset(
        symbol=dataset.symbol,
        feature_version=dataset.feature_version,
        feature_names=dataset.feature_names,
        target_mode=dataset.target_mode,
        horizon_seconds=dataset.horizon_seconds,
        total_latency_ms=dataset.total_latency_ms,
        decision_cadence_seconds=dataset.decision_cadence_seconds,
        maximum_depth_age_ms=dataset.maximum_depth_age_ms,
        decision_time_ms=dataset.decision_time_ms[values],
        target_entry_time_ms=dataset.target_entry_time_ms[values],
        target_exit_time_ms=dataset.target_exit_time_ms[values],
        target_entry_price=dataset.target_entry_price[values],
        target_exit_price=dataset.target_exit_price[values],
        gross_return_bps=dataset.gross_return_bps[values],
        features=dataset.features[values],
        source_evidence=dict(source_evidence),
    )


__all__ = [
    "TAPE_DEPTH_FEATURE_NAMES",
    "TAPE_DEPTH_FEATURE_VERSION",
    "TAPE_DEPTH_TARGET_MODE",
    "TapeDepthForecastDataset",
    "build_tape_depth_forecast_dataset",
    "slice_tape_depth_forecast_dataset",
    "tape_depth_dataset_fingerprint",
    "tape_depth_dataset_source_evidence",
    "tape_depth_source_evidence",
]
