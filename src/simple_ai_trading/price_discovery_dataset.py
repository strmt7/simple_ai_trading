"""Causal one-second feature and target construction for Round 72."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
from numba import njit

from .price_discovery_spec import (
    ANCHOR_SECOND_OFFSET,
    ANCHOR_SPACING_SECONDS,
    CLOCK_FEATURE_NAMES,
    CROSS_ASSET_WINDOW_METRICS,
    FEATURE_BURN_IN_SECONDS,
    HORIZONS_SECONDS,
    MARKET_CHANGE_METRICS,
    MARKET_WINDOW_METRICS,
    PAIR_WINDOW_METRICS,
    PRIMARY_ENTRY_DELAY_SECONDS,
    SPOT_FLOW_LAGS_SECONDS,
    STRESS_ENTRY_DELAY_SECONDS,
    WINDOWS_SECONDS,
    layer_feature_names,
    validate_layer_prefixes,
)
from .spot_perpetual_corpus import (
    FrozenFlowContract,
    FrozenFlowDay,
    SpotPerpetualCorpusStore,
)
from .spot_perpetual_flow import FLOW_SYMBOLS, SECONDS_PER_DAY


PRICE_DISCOVERY_DATASET_SCHEMA = "round-072-price-discovery-dataset-v1"
DEVELOPMENT_LAST_MONTH = "2026-03"
TERMINAL_HOLDOUT_MONTHS = ("2026-04", "2026-05", "2026-06")
_DAY_MS = 86_400_000
_ProgressCallback = Callable[[str, Mapping[str, object]], None]
_FLOW_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "base_volume",
    "quote_volume",
    "aggressive_buy_quote",
    "aggressive_sell_quote",
    "aggregate_count",
    "constituent_trade_count",
    "maximum_aggregate_quote",
    "squared_aggregate_quote_sum",
    "last_trade_age_seconds",
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


def _array_digest(digest, value: np.ndarray) -> None:
    array = np.asarray(value)
    dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(dtype, copy=False))
    if np.issubdtype(dtype, np.floating) and np.any(np.isnan(canonical)):
        canonical = canonical.copy()
        canonical[np.isnan(canonical)] = np.nan
    digest.update(dtype.str.encode("ascii"))
    digest.update(int(canonical.size).to_bytes(8, "little", signed=False))
    digest.update(memoryview(canonical).cast("B"))


def _month_ordinal(month: str) -> int:
    parsed = datetime.strptime(month, "%Y-%m")
    return parsed.year * 12 + parsed.month - 1


def month_label(ordinal: int) -> str:
    value = int(ordinal)
    return f"{value // 12:04d}-{value % 12 + 1:02d}"


@dataclass(frozen=True)
class PriceDiscoverySymbolDataset:
    symbol: str
    feature_names: tuple[str, ...]
    anchor_second_ms: np.ndarray
    available_time_ms: np.ndarray
    month_ordinal: np.ndarray
    utc_day: np.ndarray
    features: np.ndarray
    primary_target_bps: np.ndarray
    primary_valid: np.ndarray
    stress_target_bps: np.ndarray
    stress_valid: np.ndarray
    source_day_count: int
    candidate_anchors: int
    age_eligible_anchors: int
    finite_feature_anchors: int
    dataset_sha256: str

    @property
    def rows(self) -> int:
        return int(self.anchor_second_ms.size)

    def validate(self) -> None:
        rows = self.rows
        horizon_shape = (rows, len(HORIZONS_SECONDS))
        if (
            self.symbol not in FLOW_SYMBOLS
            or self.feature_names != layer_feature_names("cross_asset")
            or self.features.shape != (rows, len(self.feature_names))
            or self.features.dtype != np.float32
            or self.anchor_second_ms.shape != (rows,)
            or self.available_time_ms.shape != (rows,)
            or self.month_ordinal.shape != (rows,)
            or self.utc_day.shape != (rows,)
            or self.primary_target_bps.shape != horizon_shape
            or self.primary_valid.shape != horizon_shape
            or self.stress_target_bps.shape != horizon_shape
            or self.stress_valid.shape != horizon_shape
            or rows <= 0
            or np.any(np.diff(self.anchor_second_ms) <= 0)
            or not np.array_equal(
                self.available_time_ms, self.anchor_second_ms + 1_000
            )
            or not np.all(np.isfinite(self.features))
            or np.any(self.primary_valid & ~np.isfinite(self.primary_target_bps))
            or np.any(self.stress_valid & ~np.isfinite(self.stress_target_bps))
            or np.any(~self.primary_valid & ~np.isnan(self.primary_target_bps))
            or np.any(~self.stress_valid & ~np.isnan(self.stress_target_bps))
            or not 1 <= self.source_day_count <= self.candidate_anchors
            or not rows <= self.finite_feature_anchors <= self.age_eligible_anchors
            or not self.age_eligible_anchors <= self.candidate_anchors
            or self.dataset_sha256 != _dataset_sha256(self)
        ):
            raise ValueError(f"{self.symbol} Round 72 dataset is invalid")


@dataclass(frozen=True)
class PriceDiscoveryDatasetBundle:
    schema_version: str
    implementation_sha256: str
    inventory_sha256: str
    feature_names: tuple[str, ...]
    layer_widths: tuple[int, int, int]
    development_months: tuple[str, str]
    terminal_holdout_months_excluded: tuple[str, ...]
    symbols: tuple[PriceDiscoverySymbolDataset, ...]
    total_feature_bytes: int
    bundle_sha256: str
    profitability_claim: bool = False
    execution_or_fill_claim: bool = False
    trading_authority: bool = False

    def validate(self) -> None:
        if (
            self.schema_version != PRICE_DISCOVERY_DATASET_SCHEMA
            or self.feature_names != layer_feature_names("cross_asset")
            or self.layer_widths != validate_layer_prefixes(self.feature_names)
            or self.development_months != ("2020-10", DEVELOPMENT_LAST_MONTH)
            or self.terminal_holdout_months_excluded != TERMINAL_HOLDOUT_MONTHS
            or tuple(value.symbol for value in self.symbols) != FLOW_SYMBOLS
            or self.total_feature_bytes
            != sum(value.features.nbytes for value in self.symbols)
            or self.total_feature_bytes > 1_250_000_000
            or any(
                (
                    self.profitability_claim,
                    self.execution_or_fill_claim,
                    self.trading_authority,
                )
            )
        ):
            raise ValueError("Round 72 dataset bundle contract differs")
        for value in self.symbols:
            value.validate()
        if self.bundle_sha256 != _bundle_sha256(self):
            raise ValueError("Round 72 dataset bundle fingerprint differs")


def _dataset_sha256(value: PriceDiscoverySymbolDataset) -> str:
    digest = hashlib.sha256()
    for text in (
        PRICE_DISCOVERY_DATASET_SCHEMA,
        value.symbol,
        _canonical_sha256(value.feature_names),
    ):
        digest.update(text.encode("ascii"))
        digest.update(b"\x00")
    for array in (
        value.anchor_second_ms,
        value.available_time_ms,
        value.month_ordinal,
        value.utc_day,
        value.features,
        value.primary_target_bps,
        value.primary_valid,
        value.stress_target_bps,
        value.stress_valid,
    ):
        _array_digest(digest, array)
    for count in (
        value.source_day_count,
        value.candidate_anchors,
        value.age_eligible_anchors,
        value.finite_feature_anchors,
    ):
        digest.update(int(count).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def _bundle_sha256(value: PriceDiscoveryDatasetBundle) -> str:
    return _canonical_sha256(
        {
            "contract": PRICE_DISCOVERY_DATASET_SCHEMA,
            "development_months": list(value.development_months),
            "feature_names_sha256": _canonical_sha256(value.feature_names),
            "implementation_sha256": value.implementation_sha256,
            "inventory_sha256": value.inventory_sha256,
            "layer_widths": list(value.layer_widths),
            "symbol_datasets": [item.dataset_sha256 for item in value.symbols],
            "terminal_holdout_months_excluded": list(
                value.terminal_holdout_months_excluded
            ),
            "total_feature_bytes": value.total_feature_bytes,
        }
    )


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    cleaned = np.where(finite, array, 0.0)
    cumulative = np.empty(array.size + 1, dtype=np.float64)
    cumulative[0] = 0.0
    np.cumsum(cleaned, out=cumulative[1:])
    validity = np.empty(array.size + 1, dtype=np.int64)
    validity[0] = 0
    np.cumsum(finite.astype(np.int64), out=validity[1:])
    output = np.full(array.size, np.nan, dtype=np.float64)
    sums = cumulative[window:] - cumulative[:-window]
    counts = validity[window:] - validity[:-window]
    output[window - 1 :] = np.where(counts == window, sums, np.nan)
    return output


@njit(cache=False)
def _rolling_extreme_kernel(
    array: np.ndarray,
    window: int,
    maximum: bool,
) -> np.ndarray:
    size = int(array.size)
    output = np.full(size, np.nan, dtype=np.float64)
    queue = np.empty(size, dtype=np.int64)
    head = 0
    tail = 0
    invalid = 0
    for index in range(size):
        current = float(array[index])
        if not math.isfinite(current):
            invalid += 1
        else:
            while tail > head:
                previous = float(array[queue[tail - 1]])
                if (maximum and previous > current) or (
                    not maximum and previous < current
                ):
                    break
                tail -= 1
            queue[tail] = index
            tail += 1
        expired = index - window
        if expired >= 0:
            if not math.isfinite(float(array[expired])):
                invalid -= 1
            if tail > head and queue[head] == expired:
                head += 1
        if index >= window - 1 and invalid == 0 and tail > head:
            output[index] = array[queue[head]]
    return output


def _rolling_extreme(values: np.ndarray, window: int, *, maximum: bool) -> np.ndarray:
    return _rolling_extreme_kernel(
        np.asarray(values, dtype=np.float64),
        int(window),
        bool(maximum),
    )


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    left = np.asarray(numerator, dtype=np.float64)
    right = np.asarray(denominator, dtype=np.float64)
    output = np.zeros(np.broadcast_shapes(left.shape, right.shape), dtype=np.float64)
    np.divide(left, right, out=output, where=right > 0.0)
    return output


@dataclass(frozen=True)
class _MarketFeatures:
    matrix: np.ndarray
    metrics: Mapping[int, Mapping[str, np.ndarray]]
    one_second_signed_flow: np.ndarray


def _market_features(
    values: Mapping[str, np.ndarray],
    anchors: np.ndarray,
) -> _MarketFeatures:
    close = np.asarray(values["close"], dtype=np.float64)
    high = np.asarray(values["high"], dtype=np.float64)
    low = np.asarray(values["low"], dtype=np.float64)
    quote = np.asarray(values["quote_volume"], dtype=np.float64)
    signed_quote = np.asarray(values["aggressive_buy_quote"], dtype=np.float64) - np.asarray(
        values["aggressive_sell_quote"], dtype=np.float64
    )
    aggregate = np.asarray(values["aggregate_count"], dtype=np.float64)
    constituent = np.asarray(values["constituent_trade_count"], dtype=np.float64)
    maximum_quote = np.asarray(values["maximum_aggregate_quote"], dtype=np.float64)
    squared_quote = np.asarray(
        values["squared_aggregate_quote_sum"], dtype=np.float64
    )
    age = np.asarray(values["last_trade_age_seconds"], dtype=np.float64)
    log_close = np.log(close)
    one_return = np.zeros(close.size, dtype=np.float64)
    one_return[1:] = log_close[1:] - log_close[:-1]
    one_return[~np.isfinite(one_return)] = 0.0
    one_signed_flow = _safe_ratio(signed_quote, quote)
    columns = [np.log1p(age[anchors])]
    metrics_by_window: dict[int, dict[str, np.ndarray]] = {}
    for window in WINDOWS_SECONDS:
        quote_sum = _rolling_sum(quote, window)
        signed_sum = _rolling_sum(signed_quote, window)
        aggregate_sum = _rolling_sum(aggregate, window)
        constituent_sum = _rolling_sum(constituent, window)
        squared_sum = _rolling_sum(squared_quote, window)
        zero_sum = _rolling_sum((aggregate == 0.0).astype(np.float64), window)
        variation = _rolling_sum(np.abs(one_return), window)
        variance = _rolling_sum(one_return * one_return, window)
        rolling_high = _rolling_extreme(high, window, maximum=True)
        rolling_low = _rolling_extreme(low, window, maximum=False)
        rolling_maximum_quote = _rolling_extreme(
            maximum_quote, window, maximum=True
        )
        current: dict[str, np.ndarray] = {
            "log_return_bps": 10_000.0
            * (log_close[anchors] - log_close[anchors - window]),
            "path_variation_bps": 10_000.0 * variation[anchors],
            "realized_volatility_bps": 10_000.0
            * np.sqrt(np.maximum(variance[anchors], 0.0)),
            "log_range_bps": 10_000.0
            * np.log(rolling_high[anchors] / rolling_low[anchors]),
            "signed_quote_flow": _safe_ratio(
                signed_sum[anchors], quote_sum[anchors]
            ),
            "log1p_quote_volume": np.log1p(quote_sum[anchors]),
            "aggregate_orders_per_second": aggregate_sum[anchors] / window,
            "log1p_mean_aggregate_quote": np.log1p(
                _safe_ratio(quote_sum[anchors], aggregate_sum[anchors])
            ),
            "log1p_maximum_aggregate_quote": np.log1p(
                rolling_maximum_quote[anchors]
            ),
            "aggregate_quote_hhi": _safe_ratio(
                squared_sum[anchors], quote_sum[anchors] ** 2
            ),
            "log1p_constituent_per_aggregate": np.log1p(
                _safe_ratio(constituent_sum[anchors], aggregate_sum[anchors])
            ),
            "zero_flow_fraction": zero_sum[anchors] / window,
        }
        previous_end = anchors - window
        previous = {
            "realized_volatility_bps": 10_000.0
            * np.sqrt(np.maximum(variance[previous_end], 0.0)),
            "signed_quote_flow": _safe_ratio(
                signed_sum[previous_end], quote_sum[previous_end]
            ),
            "log1p_quote_volume": np.log1p(quote_sum[previous_end]),
            "aggregate_orders_per_second": aggregate_sum[previous_end] / window,
        }
        columns.extend(current[name] for name in MARKET_WINDOW_METRICS)
        columns.extend(current[name] - previous[name] for name in MARKET_CHANGE_METRICS)
        metrics_by_window[window] = current
    return _MarketFeatures(
        matrix=np.column_stack(columns).astype(np.float32),
        metrics=metrics_by_window,
        one_second_signed_flow=one_signed_flow,
    )


def _clock_features(anchor_second_ms: np.ndarray) -> np.ndarray:
    available = np.asarray(anchor_second_ms, dtype=np.int64) + 1_000
    seconds = (available % _DAY_MS).astype(np.float64) / 1_000.0
    utc_day = available // _DAY_MS
    weekday = (utc_day + 3) % 7
    columns = []
    for phase, period in (
        (seconds, 86_400.0),
        (weekday.astype(np.float64), 7.0),
        (seconds % 60.0, 60.0),
        (seconds % 300.0, 300.0),
        (seconds % 900.0, 900.0),
    ):
        angle = 2.0 * np.pi * phase / period
        columns.extend((np.sin(angle), np.cos(angle)))
    output = np.column_stack(columns).astype(np.float32)
    if output.shape[1] != len(CLOCK_FEATURE_NAMES):
        raise RuntimeError("Round 72 clock feature width drifted")
    return output


def _pair_features(
    spot: _MarketFeatures,
    perpetual: _MarketFeatures,
    *,
    spot_close: np.ndarray,
    perpetual_close: np.ndarray,
    anchors: np.ndarray,
) -> np.ndarray:
    basis = 10_000.0 * (
        np.log(np.asarray(perpetual_close, dtype=np.float64))
        - np.log(np.asarray(spot_close, dtype=np.float64))
    )
    columns = [basis[anchors]]
    for window in WINDOWS_SECONDS:
        spot_metrics = spot.metrics[window]
        perpetual_metrics = perpetual.metrics[window]
        values = {
            "basis_change_bps": basis[anchors] - basis[anchors - window],
            "spot_minus_perpetual_return_bps": spot_metrics["log_return_bps"]
            - perpetual_metrics["log_return_bps"],
            "signed_flow_product": spot_metrics["signed_quote_flow"]
            * perpetual_metrics["signed_quote_flow"],
            "absolute_signed_flow_difference": np.abs(
                spot_metrics["signed_quote_flow"]
                - perpetual_metrics["signed_quote_flow"]
            ),
            "log_relative_quote_volume": spot_metrics["log1p_quote_volume"]
            - perpetual_metrics["log1p_quote_volume"],
            "log_relative_aggregate_rate": np.log1p(
                spot_metrics["aggregate_orders_per_second"]
            )
            - np.log1p(perpetual_metrics["aggregate_orders_per_second"]),
        }
        columns.extend(values[name] for name in PAIR_WINDOW_METRICS)
    current_perpetual_flow = perpetual.one_second_signed_flow[anchors]
    for lag in SPOT_FLOW_LAGS_SECONDS:
        lagged_spot_flow = spot.one_second_signed_flow[anchors - lag]
        columns.extend(
            (
                lagged_spot_flow * current_perpetual_flow,
                lagged_spot_flow - current_perpetual_flow,
            )
        )
    return np.column_stack(columns).astype(np.float32)


def _cross_asset_features(
    target_index: int,
    perpetual: Sequence[_MarketFeatures],
) -> np.ndarray:
    columns = []
    for window in WINDOWS_SECONDS:
        returns = np.column_stack(
            [value.metrics[window]["log_return_bps"] for value in perpetual]
        )
        flows = np.column_stack(
            [value.metrics[window]["signed_quote_flow"] for value in perpetual]
        )
        if target_index == 0:
            leader_return = np.mean(returns[:, 1:], axis=1)
            leader_flow = np.mean(flows[:, 1:], axis=1)
        else:
            leader_return = returns[:, 0]
            leader_flow = flows[:, 0]
        values = {
            "perpetual_return_mean_bps": np.mean(returns, axis=1),
            "perpetual_return_dispersion_bps": np.std(returns, axis=1),
            "perpetual_return_directional_agreement": np.abs(
                np.mean(np.sign(returns), axis=1)
            ),
            "perpetual_signed_flow_mean": np.mean(flows, axis=1),
            "perpetual_signed_flow_dispersion": np.std(flows, axis=1),
            "leader_perpetual_return_bps": leader_return,
            "leader_perpetual_signed_flow": leader_flow,
        }
        columns.extend(values[name] for name in CROSS_ASSET_WINDOW_METRICS)
    return np.column_stack(columns).astype(np.float32)


def _targets(
    perpetual: Mapping[str, np.ndarray],
    anchors: np.ndarray,
    delay: int,
) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(perpetual["base_volume"], dtype=np.float64)
    quote = np.asarray(perpetual["quote_volume"], dtype=np.float64)
    count = np.asarray(perpetual["aggregate_count"], dtype=np.uint64)
    vwap = np.full(base.size, np.nan, dtype=np.float64)
    np.divide(quote, base, out=vwap, where=(base > 0.0) & (count > 0))
    targets = np.full((anchors.size, len(HORIZONS_SECONDS)), np.nan, dtype=np.float64)
    valid = np.zeros_like(targets, dtype=bool)
    for index, horizon in enumerate(HORIZONS_SECONDS):
        entry = anchors + int(delay)
        exit_index = entry + int(horizon)
        selected = (
            np.isfinite(vwap[entry])
            & np.isfinite(vwap[exit_index])
            & (vwap[entry] > 0.0)
            & (vwap[exit_index] > 0.0)
        )
        targets[selected, index] = 10_000.0 * np.log(
            vwap[exit_index[selected]] / vwap[entry[selected]]
        )
        valid[:, index] = selected
    return targets, valid


def _day_anchors() -> np.ndarray:
    maximum_exit = STRESS_ENTRY_DELAY_SECONDS + max(HORIZONS_SECONDS)
    values = np.arange(
        ANCHOR_SECOND_OFFSET,
        SECONDS_PER_DAY - maximum_exit,
        ANCHOR_SPACING_SECONDS,
        dtype=np.int64,
    )
    return values[values >= FEATURE_BURN_IN_SECONDS]


def build_price_discovery_day(
    *,
    period: str,
    flow_by_stream: Mapping[tuple[str, str], Mapping[str, np.ndarray]],
) -> dict[str, dict[str, np.ndarray | int]]:
    """Build all three symbols for one UTC day without crossing its boundary."""

    required = {
        (market_type, symbol)
        for market_type in ("spot", "futures")
        for symbol in FLOW_SYMBOLS
    }
    if set(flow_by_stream) != required:
        raise ValueError("Round 72 day requires all six spot/perpetual streams")
    try:
        day_start_ms = int(
            datetime.fromisoformat(period).replace(tzinfo=UTC).timestamp() * 1_000
        )
    except ValueError as exc:
        raise ValueError("Round 72 day period is invalid") from exc
    anchors = _day_anchors()
    absolute_anchors = day_start_ms + anchors * 1_000
    clock = _clock_features(absolute_anchors)
    market_blocks: dict[tuple[str, str], _MarketFeatures] = {}
    for key, values in flow_by_stream.items():
        if set(values) != set(_FLOW_FIELDS):
            raise ValueError(f"{key} Round 72 flow fields differ")
        if any(np.asarray(value).shape != (SECONDS_PER_DAY,) for value in values.values()):
            raise ValueError(f"{key} Round 72 flow arrays do not span one day")
        market_blocks[key] = _market_features(values, anchors)
    perpetual_blocks = [
        market_blocks[("futures", symbol)] for symbol in FLOW_SYMBOLS
    ]
    output: dict[str, dict[str, np.ndarray | int]] = {}
    for symbol_index, symbol in enumerate(FLOW_SYMBOLS):
        spot_values = flow_by_stream[("spot", symbol)]
        perpetual_values = flow_by_stream[("futures", symbol)]
        spot = market_blocks[("spot", symbol)]
        perpetual = perpetual_blocks[symbol_index]
        pair = _pair_features(
            spot,
            perpetual,
            spot_close=spot_values["close"],
            perpetual_close=perpetual_values["close"],
            anchors=anchors,
        )
        cross = _cross_asset_features(symbol_index, perpetual_blocks)
        features = np.column_stack(
            (perpetual.matrix, clock, spot.matrix, pair, cross)
        ).astype(np.float32)
        if features.shape[1] != len(layer_feature_names("cross_asset")):
            raise RuntimeError("Round 72 full feature width differs from freeze")
        spot_age = np.asarray(spot_values["last_trade_age_seconds"])[anchors]
        perpetual_age = np.asarray(perpetual_values["last_trade_age_seconds"])[anchors]
        age_eligible = (spot_age <= 2) & (perpetual_age <= 2)
        finite = np.all(np.isfinite(features), axis=1)
        primary_target, primary_valid = _targets(
            perpetual_values, anchors, PRIMARY_ENTRY_DELAY_SECONDS
        )
        stress_target, stress_valid = _targets(
            perpetual_values, anchors, STRESS_ENTRY_DELAY_SECONDS
        )
        useful = np.any(primary_valid | stress_valid, axis=1)
        selected = age_eligible & finite & useful
        output[symbol] = {
            "anchor_second_ms": absolute_anchors[selected],
            "available_time_ms": absolute_anchors[selected] + 1_000,
            "features": features[selected],
            "primary_target_bps": primary_target[selected],
            "primary_valid": primary_valid[selected],
            "stress_target_bps": stress_target[selected],
            "stress_valid": stress_valid[selected],
            "candidate_anchors": int(anchors.size),
            "age_eligible_anchors": int(np.count_nonzero(age_eligible)),
            "finite_feature_anchors": int(np.count_nonzero(age_eligible & finite)),
        }
    return output


def _plain_array(value, *, dtype) -> np.ndarray:
    if isinstance(value, np.ma.MaskedArray):
        fill = np.nan if np.issubdtype(np.dtype(dtype), np.floating) else 0
        value = value.filled(fill)
    return np.asarray(value, dtype=dtype)


def _load_flow_day(
    store: SpotPerpetualCorpusStore,
    day: FrozenFlowDay,
    *,
    inventory_sha256: str,
) -> Mapping[tuple[str, str], Mapping[str, np.ndarray]]:
    day_id = day.contract_sha256(inventory_sha256)
    columns = ["symbol", "second_ms"]
    for prefix in ("spot", "perpetual"):
        columns.extend(f"{prefix}_{field}" for field in _FLOW_FIELDS)
    result = store.connect().execute(
        f"SELECT {','.join(columns)} FROM spot_perpetual_flow_1s "
        "WHERE day_id = ? ORDER BY symbol, second_ms",
        [day_id],
    ).fetchnumpy()
    symbols = np.asarray(result["symbol"]).astype(str)
    if symbols.size != len(FLOW_SYMBOLS) * SECONDS_PER_DAY:
        raise ValueError(f"{day.period} Round 72 corpus row count differs")
    output: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for symbol in FLOW_SYMBOLS:
        selected = symbols == symbol
        if np.count_nonzero(selected) != SECONDS_PER_DAY:
            raise ValueError(f"{day.period} {symbol} row count differs")
        seconds = _plain_array(result["second_ms"][selected], dtype=np.int64)
        if np.any(np.diff(seconds) != 1_000):
            raise ValueError(f"{day.period} {symbol} seconds are not contiguous")
        for market_type, prefix in (("spot", "spot"), ("futures", "perpetual")):
            values: dict[str, np.ndarray] = {}
            for field in _FLOW_FIELDS:
                dtype = (
                    np.uint32
                    if field
                    in {
                        "aggregate_count",
                        "constituent_trade_count",
                        "last_trade_age_seconds",
                    }
                    else np.float64
                )
                values[field] = _plain_array(
                    result[f"{prefix}_{field}"][selected], dtype=dtype
                )
            output[(market_type, symbol)] = values
    return output


def _load_implementation(path: Path, contract: FrozenFlowContract) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Round 72 implementation artifact is not an object")
    canonical = dict(value)
    observed = str(canonical.pop("implementation_sha256", ""))
    if (
        observed != _canonical_sha256(canonical)
        or value.get("inventory_sha256") != contract.inventory_sha256
        or value.get("design_sha256") != contract.design_sha256
        or value.get("profitability_claim") is not False
        or value.get("trading_authority") is not False
    ):
        raise ValueError("Round 72 implementation artifact identity differs")
    validate_layer_prefixes(layer_feature_names("cross_asset"))
    return value


def build_price_discovery_datasets(
    store: SpotPerpetualCorpusStore,
    contract: FrozenFlowContract,
    *,
    implementation_path: str | Path,
    progress: _ProgressCallback | None = None,
) -> PriceDiscoveryDatasetBundle:
    """Read only development partitions and construct the frozen in-memory panel."""

    implementation = _load_implementation(Path(implementation_path), contract)
    implementation_sha256 = str(implementation["implementation_sha256"])
    development_days = tuple(
        day for day in contract.days if day.month <= DEVELOPMENT_LAST_MONTH
    )
    if (
        len(development_days) != 66
        or development_days[0].month != "2020-10"
        or development_days[-1].month != DEVELOPMENT_LAST_MONTH
        or any(day.month in TERMINAL_HOLDOUT_MONTHS for day in development_days)
    ):
        raise ValueError("Round 72 development-day selection differs")
    names = layer_feature_names("cross_asset")
    parts: dict[str, dict[str, list[np.ndarray] | int]] = {
        symbol: {
            "anchor_second_ms": [],
            "available_time_ms": [],
            "month_ordinal": [],
            "utc_day": [],
            "features": [],
            "primary_target_bps": [],
            "primary_valid": [],
            "stress_target_bps": [],
            "stress_valid": [],
            "candidate_anchors": 0,
            "age_eligible_anchors": 0,
            "finite_feature_anchors": 0,
        }
        for symbol in FLOW_SYMBOLS
    }
    for day_index, day in enumerate(development_days, start=1):
        flow = _load_flow_day(
            store,
            day,
            inventory_sha256=contract.inventory_sha256,
        )
        built = build_price_discovery_day(period=day.period, flow_by_stream=flow)
        month_value = _month_ordinal(day.month)
        day_value = int(
            datetime.fromisoformat(day.period).replace(tzinfo=UTC).timestamp()
            // 86_400
        )
        for symbol in FLOW_SYMBOLS:
            target = parts[symbol]
            source = built[symbol]
            rows = int(np.asarray(source["anchor_second_ms"]).size)
            for key in (
                "anchor_second_ms",
                "available_time_ms",
                "features",
                "primary_target_bps",
                "primary_valid",
                "stress_target_bps",
                "stress_valid",
            ):
                target[key].append(np.asarray(source[key]))  # type: ignore[union-attr]
            target["month_ordinal"].append(  # type: ignore[union-attr]
                np.full(rows, month_value, dtype=np.int32)
            )
            target["utc_day"].append(  # type: ignore[union-attr]
                np.full(rows, day_value, dtype=np.int32)
            )
            for key in (
                "candidate_anchors",
                "age_eligible_anchors",
                "finite_feature_anchors",
            ):
                target[key] = int(target[key]) + int(source[key])
        if progress:
            progress(
                "price_discovery_day_built",
                {
                    "period": day.period,
                    "day_index": day_index,
                    "total_days": len(development_days),
                    "rows_by_symbol": {
                        symbol: int(np.asarray(built[symbol]["anchor_second_ms"]).size)
                        for symbol in FLOW_SYMBOLS
                    },
                },
            )

    datasets: list[PriceDiscoverySymbolDataset] = []
    for symbol in FLOW_SYMBOLS:
        source = parts[symbol]

        def combine(name: str, dtype) -> np.ndarray:
            values = source[name]
            if not isinstance(values, list) or not values:
                raise RuntimeError(f"{symbol} {name} fragments are missing")
            return np.concatenate(values).astype(dtype, copy=False)

        provisional = PriceDiscoverySymbolDataset(
            symbol=symbol,
            feature_names=names,
            anchor_second_ms=combine("anchor_second_ms", np.int64),
            available_time_ms=combine("available_time_ms", np.int64),
            month_ordinal=combine("month_ordinal", np.int32),
            utc_day=combine("utc_day", np.int32),
            features=combine("features", np.float32),
            primary_target_bps=combine("primary_target_bps", np.float64),
            primary_valid=combine("primary_valid", bool),
            stress_target_bps=combine("stress_target_bps", np.float64),
            stress_valid=combine("stress_valid", bool),
            source_day_count=len(development_days),
            candidate_anchors=int(source["candidate_anchors"]),
            age_eligible_anchors=int(source["age_eligible_anchors"]),
            finite_feature_anchors=int(source["finite_feature_anchors"]),
            dataset_sha256="",
        )
        dataset = replace(provisional, dataset_sha256=_dataset_sha256(provisional))
        dataset.validate()
        datasets.append(dataset)
    total_feature_bytes = sum(value.features.nbytes for value in datasets)
    provisional_bundle = PriceDiscoveryDatasetBundle(
        schema_version=PRICE_DISCOVERY_DATASET_SCHEMA,
        implementation_sha256=implementation_sha256,
        inventory_sha256=contract.inventory_sha256,
        feature_names=names,
        layer_widths=validate_layer_prefixes(names),
        development_months=("2020-10", DEVELOPMENT_LAST_MONTH),
        terminal_holdout_months_excluded=TERMINAL_HOLDOUT_MONTHS,
        symbols=tuple(datasets),
        total_feature_bytes=total_feature_bytes,
        bundle_sha256="",
    )
    bundle = replace(
        provisional_bundle,
        bundle_sha256=_bundle_sha256(provisional_bundle),
    )
    bundle.validate()
    return bundle


__all__ = [
    "DEVELOPMENT_LAST_MONTH",
    "PRICE_DISCOVERY_DATASET_SCHEMA",
    "TERMINAL_HOLDOUT_MONTHS",
    "PriceDiscoveryDatasetBundle",
    "PriceDiscoverySymbolDataset",
    "build_price_discovery_datasets",
    "build_price_discovery_day",
    "month_label",
]
