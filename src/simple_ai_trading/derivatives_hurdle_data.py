"""Causal derivatives-state dataset for the frozen Round 38 hurdle screen."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Callable, Mapping

import numpy as np

from .cross_asset_cost_data import (
    HORIZONS_MINUTES,
    MINUTE_MS,
    MinuteSeries,
    ROLES,
    SYMBOLS,
    SourceEvidence,
    _feature_arrays,
    _read_only_connection,
    _role_masks,
)


EXECUTION_CHARGE_BPS = 12.0
PREMIUM_MAX_AGE_MINUTES = 5
PREMIUM_MIN_ROLLING_COVERAGE = 0.90
EXPECTED_PREMIUM_QUALITY = {
    "BTCUSDT": (1_880_574, 9, 2_946, 1_440),
    "ETHUSDT": (1_880_577, 9, 2_943, 1_440),
    "SOLUSDT": (1_882_018, 7, 1_502, 1_440),
}


@dataclass(frozen=True)
class DerivativesSeriesEvidence:
    symbol: str
    premium_rows: int
    premium_observed_grid_rows: int
    premium_gap_events: int
    premium_missing_minutes: int
    premium_maximum_gap_minutes: int
    premium_stream_sha256: str
    funding_rows: int
    funding_first_calc_time_ms: int
    funding_last_calc_time_ms: int
    funding_minimum_interval_hours: int
    funding_maximum_interval_hours: int
    funding_stream_sha256: str

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DerivativesSourceEvidence:
    source_certificate_path: str
    source_certificate_sha256: str
    price_flow: SourceEvidence
    derivatives_series: tuple[DerivativesSeriesEvidence, ...]
    derivatives_panel_sha256: str
    selection_confirmation_or_terminal_rows_read: bool

    def asdict(self) -> dict[str, object]:
        return {
            "source_certificate_path": self.source_certificate_path,
            "source_certificate_sha256": self.source_certificate_sha256,
            "price_flow": self.price_flow.asdict(),
            "derivatives_series": [item.asdict() for item in self.derivatives_series],
            "derivatives_panel_sha256": self.derivatives_panel_sha256,
            "selection_confirmation_or_terminal_rows_read": (
                self.selection_confirmation_or_terminal_rows_read
            ),
        }


@dataclass(frozen=True)
class PremiumState:
    observed: np.ndarray
    age_minutes: np.ndarray
    close_bps: np.ndarray
    range_bps: np.ndarray
    rolling_observed_fraction_15m: np.ndarray
    rolling_observed_fraction_60m: np.ndarray
    rolling_observed_fraction_240m: np.ndarray
    rolling_observed_fraction_1440m: np.ndarray


@dataclass(frozen=True)
class FundingState:
    event_time_ms: np.ndarray
    event_rate: np.ndarray
    event_interval_hours: np.ndarray
    last_rate_bps: np.ndarray
    last_interval_hours: np.ndarray
    age_minutes: np.ndarray
    settled_sum_24h_bps: np.ndarray
    settled_sum_72h_bps: np.ndarray
    settled_sum_168h_bps: np.ndarray
    event_mean_30_bps: np.ndarray
    event_zscore_30: np.ndarray


@dataclass(frozen=True)
class DerivativesHurdleDataset:
    feature_names: tuple[str, ...]
    price_flow_feature_count: int
    features: np.ndarray
    decision_time_ms: np.ndarray
    symbol_index: np.ndarray
    target_class: Mapping[int, np.ndarray]
    long_net_utility_bps: Mapping[int, np.ndarray]
    short_net_utility_bps: Mapping[int, np.ndarray]
    funding_cash_flow_bps: Mapping[int, np.ndarray]
    role_masks: Mapping[int, Mapping[str, np.ndarray]]
    source_evidence: DerivativesSourceEvidence
    source_exclusions: Mapping[str, int]

    @property
    def rows(self) -> int:
        return int(self.features.shape[0])

    def feature_count(self, feature_set: str) -> int:
        if feature_set == "price_flow_only":
            return self.price_flow_feature_count
        if feature_set == "price_flow_plus_premium_and_funding":
            return len(self.feature_names)
        raise KeyError(feature_set)

    def feature_view(self, feature_set: str) -> np.ndarray:
        return self.features[:, : self.feature_count(feature_set)]


ProgressCallback = Callable[[str, Mapping[str, object]], None]


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _load_source_certificate(path: Path) -> tuple[dict[str, object], str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Round 38 source certificate root is not an object")
    canonical = dict(value)
    claimed = str(canonical.pop("source_certificate_sha256", ""))
    if (
        value.get("schema_version")
        != "round-038-derivatives-source-certificate-v1"
        or value.get("round") != 38
        or value.get("symbols") != list(SYMBOLS)
        or value.get("start_period") != "2021-12"
        or value.get("end_period") != "2025-06"
        or value.get("persistent_zip_archive_created") is not False
        or claimed != _canonical_sha256(canonical)
    ):
        raise ValueError("Round 38 source certificate identity is invalid")
    return value, claimed


def _prefix_sum(values: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))


def _rolling_sum(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(values.size, np.nan, dtype=np.float64)
    prefix = _prefix_sum(np.nan_to_num(values, nan=0.0))
    result[window - 1 :] = prefix[window:] - prefix[:-window]
    return result


def _rolling_fraction(observed: np.ndarray, window: int) -> np.ndarray:
    return _rolling_sum(observed.astype(np.float64), window) / float(window)


def _rolling_observed_mean(
    values: np.ndarray,
    observed: np.ndarray,
    window: int,
) -> np.ndarray:
    result = np.full(values.size, np.nan, dtype=np.float64)
    value_prefix = _prefix_sum(np.where(observed, values, 0.0))
    count_prefix = _prefix_sum(observed.astype(np.float64))
    total = value_prefix[window:] - value_prefix[:-window]
    count = count_prefix[window:] - count_prefix[:-window]
    minimum = math.ceil(PREMIUM_MIN_ROLLING_COVERAGE * window)
    target = result[window - 1 :]
    np.divide(total, count, out=target, where=count >= minimum)
    return result


def _rolling_observed_std(
    values: np.ndarray,
    observed: np.ndarray,
    window: int,
) -> np.ndarray:
    result = np.full(values.size, np.nan, dtype=np.float64)
    value_prefix = _prefix_sum(np.where(observed, values, 0.0))
    square_prefix = _prefix_sum(np.where(observed, values * values, 0.0))
    count_prefix = _prefix_sum(observed.astype(np.float64))
    total = value_prefix[window:] - value_prefix[:-window]
    total_square = square_prefix[window:] - square_prefix[:-window]
    count = count_prefix[window:] - count_prefix[:-window]
    minimum = math.ceil(PREMIUM_MIN_ROLLING_COVERAGE * window)
    valid = count >= minimum
    variance = np.zeros_like(total)
    variance[valid] = np.maximum(
        total_square[valid] / count[valid]
        - (total[valid] / count[valid]) ** 2,
        0.0,
    )
    target = result[window - 1 :]
    target[valid] = np.sqrt(variance[valid])
    return result


def _lagged_change(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(values.size, np.nan, dtype=np.float64)
    valid = np.isfinite(values[window:]) & np.isfinite(values[:-window])
    target = result[window:]
    target[valid] = values[window:][valid] - values[:-window][valid]
    return result


def _grid_age(observed: np.ndarray) -> np.ndarray:
    indices = np.arange(observed.size, dtype=np.int64)
    last = np.maximum.accumulate(np.where(observed, indices, -10**12))
    return (indices - last).astype(np.float64)


def _source_hash_update(
    digest: "hashlib._Hash", values: tuple[object, ...]
) -> None:
    digest.update(
        json.dumps(
            list(values),
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    )
    digest.update(b"\n")


def _load_premium_state(
    connection: sqlite3.Connection,
    *,
    symbol: str,
    grid_time_ms: np.ndarray,
) -> tuple[PremiumState, tuple[int, int, int, int], str]:
    observed = np.zeros(grid_time_ms.size, dtype=bool)
    close_bps = np.full(grid_time_ms.size, np.nan, dtype=np.float64)
    range_bps = np.full(grid_time_ms.size, np.nan, dtype=np.float64)
    start_ms = int(grid_time_ms[0])
    end_ms = int(grid_time_ms[-1])
    digest = hashlib.sha256()
    rows = 0
    cursor = connection.execute(
        """
        SELECT open_time, open, high, low, close, close_time
        FROM futures_reference_bars
        WHERE symbol=? AND market_type='futures' AND kind='premium_index'
          AND interval='1m' AND open_time BETWEEN ? AND ?
        ORDER BY open_time
        """,
        (symbol, start_ms, end_ms),
    )
    for open_time, open_value, high, low, close, close_time in cursor:
        index = (int(open_time) - start_ms) // MINUTE_MS
        if not 0 <= index < grid_time_ms.size:
            raise ValueError(f"{symbol} premium timestamp is outside the source grid")
        if int(grid_time_ms[index]) != int(open_time) or observed[index]:
            raise ValueError(f"{symbol} premium timestamp is misaligned or duplicated")
        numeric = tuple(
            float(item) for item in (open_value, high, low, close)
        )
        if not all(math.isfinite(item) for item in numeric):
            raise ValueError(f"{symbol} premium contains a non-finite value")
        observed[index] = True
        close_bps[index] = float(close) * 10_000.0
        range_bps[index] = (float(high) - float(low)) * 10_000.0
        _source_hash_update(
            digest,
            (int(open_time), *numeric, int(close_time)),
        )
        rows += 1
    deltas = np.diff(np.flatnonzero(observed))
    missing = deltas[deltas > 1] - 1
    quality = (
        rows,
        int(missing.size),
        int(np.sum(missing)),
        int(np.max(missing)) if missing.size else 0,
    )
    if quality != EXPECTED_PREMIUM_QUALITY[symbol]:
        raise ValueError(
            f"{symbol} premium quality changed: observed={quality} "
            f"expected={EXPECTED_PREMIUM_QUALITY[symbol]}"
        )
    age = _grid_age(observed)
    eligible_fill = age <= PREMIUM_MAX_AGE_MINUTES
    last_index = np.maximum.accumulate(
        np.where(observed, np.arange(observed.size, dtype=np.int64), 0)
    )
    close_filled = close_bps[last_index]
    range_filled = range_bps[last_index]
    close_filled[~eligible_fill] = np.nan
    range_filled[~eligible_fill] = np.nan
    return (
        PremiumState(
            observed=observed,
            age_minutes=age,
            close_bps=close_filled,
            range_bps=range_filled,
            rolling_observed_fraction_15m=_rolling_fraction(observed, 15),
            rolling_observed_fraction_60m=_rolling_fraction(observed, 60),
            rolling_observed_fraction_240m=_rolling_fraction(observed, 240),
            rolling_observed_fraction_1440m=_rolling_fraction(observed, 1440),
        ),
        quality,
        digest.hexdigest(),
    )


def _event_rolling_statistics(rate_bps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.full(rate_bps.size, np.nan, dtype=np.float64)
    zscore = np.full(rate_bps.size, np.nan, dtype=np.float64)
    if rate_bps.size < 30:
        return mean, zscore
    prefix = _prefix_sum(rate_bps)
    square = _prefix_sum(rate_bps * rate_bps)
    total = prefix[30:] - prefix[:-30]
    total_square = square[30:] - square[:-30]
    current_mean = total / 30.0
    standard_deviation = np.sqrt(
        np.maximum(total_square / 30.0 - current_mean * current_mean, 0.0)
    )
    mean[29:] = current_mean
    valid = standard_deviation > 1e-12
    target = zscore[29:]
    target[valid] = (rate_bps[29:][valid] - current_mean[valid]) / standard_deviation[
        valid
    ]
    target[~valid] = 0.0
    return mean, zscore


def _load_funding_state(
    connection: sqlite3.Connection,
    *,
    symbol: str,
    grid_time_ms: np.ndarray,
) -> tuple[FundingState, tuple[int, int, int, int, int], str]:
    start_ms = int(grid_time_ms[0])
    end_ms = int(grid_time_ms[-1]) + MINUTE_MS - 1
    rows = connection.execute(
        """
        SELECT calc_time, funding_interval_hours, funding_rate
        FROM funding_rates
        WHERE symbol=? AND market_type='futures' AND calc_time BETWEEN ? AND ?
        ORDER BY calc_time
        """,
        (symbol, start_ms, end_ms),
    ).fetchall()
    event_time = np.asarray([int(row[0]) for row in rows], dtype=np.int64)
    interval = np.asarray([int(row[1]) for row in rows], dtype=np.int16)
    rate = np.asarray([float(row[2]) for row in rows], dtype=np.float64)
    if (
        event_time.size == 0
        or np.any(np.diff(event_time) <= 0)
        or not np.isfinite(rate).all()
        or np.any(interval < 1)
        or np.any(interval > 8)
    ):
        raise ValueError(f"{symbol} funding source failed integrity checks")
    digest = hashlib.sha256()
    for values in zip(event_time, interval, rate, strict=True):
        _source_hash_update(
            digest, (int(values[0]), int(values[1]), float(values[2]))
        )
    index = np.searchsorted(event_time, grid_time_ms, side="right") - 1
    if np.any(index < 0):
        raise ValueError(f"{symbol} has no settled funding state at grid start")
    rate_bps = rate * 10_000.0
    last_rate = rate_bps[index]
    last_interval = interval[index].astype(np.float64)
    age = (grid_time_ms - event_time[index]) / float(MINUTE_MS)
    prefix = _prefix_sum(rate_bps)

    def trailing_sum(hours: int) -> np.ndarray:
        end_index = np.searchsorted(event_time, grid_time_ms, side="right")
        start_index = np.searchsorted(
            event_time,
            grid_time_ms - hours * 60 * MINUTE_MS,
            side="right",
        )
        return prefix[end_index] - prefix[start_index]

    event_mean, event_zscore = _event_rolling_statistics(rate_bps)
    return (
        FundingState(
            event_time_ms=event_time,
            event_rate=rate,
            event_interval_hours=interval,
            last_rate_bps=last_rate,
            last_interval_hours=last_interval,
            age_minutes=age,
            settled_sum_24h_bps=trailing_sum(24),
            settled_sum_72h_bps=trailing_sum(72),
            settled_sum_168h_bps=trailing_sum(168),
            event_mean_30_bps=event_mean[index],
            event_zscore_30=event_zscore[index],
        ),
        (
            int(event_time.size),
            int(event_time[0]),
            int(event_time[-1]),
            int(np.min(interval)),
            int(np.max(interval)),
        ),
        digest.hexdigest(),
    )


def load_derivatives_state(
    database_path: str | Path,
    panel: Mapping[str, MinuteSeries],
    price_source_evidence: SourceEvidence,
    *,
    source_certificate_path: str | Path,
    progress: ProgressCallback | None = None,
) -> tuple[
    dict[str, PremiumState],
    dict[str, FundingState],
    DerivativesSourceEvidence,
]:
    """Load and certify only warmup-through-viability derivatives state."""

    certificate_path = Path(source_certificate_path).resolve()
    _certificate, certificate_sha = _load_source_certificate(certificate_path)
    reference_time = panel[SYMBOLS[0]].open_time_ms
    premium: dict[str, PremiumState] = {}
    funding: dict[str, FundingState] = {}
    evidence: list[DerivativesSeriesEvidence] = []
    with _read_only_connection(Path(database_path)) as connection:
        for symbol in SYMBOLS:
            premium_state, premium_quality, premium_sha = _load_premium_state(
                connection,
                symbol=symbol,
                grid_time_ms=reference_time,
            )
            funding_state, funding_quality, funding_sha = _load_funding_state(
                connection,
                symbol=symbol,
                grid_time_ms=reference_time,
            )
            premium[symbol] = premium_state
            funding[symbol] = funding_state
            evidence.append(
                DerivativesSeriesEvidence(
                    symbol=symbol,
                    premium_rows=premium_quality[0],
                    premium_observed_grid_rows=int(
                        np.count_nonzero(premium_state.observed)
                    ),
                    premium_gap_events=premium_quality[1],
                    premium_missing_minutes=premium_quality[2],
                    premium_maximum_gap_minutes=premium_quality[3],
                    premium_stream_sha256=premium_sha,
                    funding_rows=funding_quality[0],
                    funding_first_calc_time_ms=funding_quality[1],
                    funding_last_calc_time_ms=funding_quality[2],
                    funding_minimum_interval_hours=funding_quality[3],
                    funding_maximum_interval_hours=funding_quality[4],
                    funding_stream_sha256=funding_sha,
                )
            )
            if progress is not None:
                progress(
                    "derivatives_source_load",
                    {
                        "symbol": symbol,
                        "premium_rows": premium_quality[0],
                        "premium_missing_minutes": premium_quality[2],
                        "funding_rows": funding_quality[0],
                    },
                )
    digest = hashlib.sha256()
    for item in evidence:
        digest.update(item.symbol.encode("ascii"))
        digest.update(bytes.fromhex(item.premium_stream_sha256))
        digest.update(bytes.fromhex(item.funding_stream_sha256))
    return (
        premium,
        funding,
        DerivativesSourceEvidence(
            source_certificate_path=str(certificate_path),
            source_certificate_sha256=certificate_sha,
            price_flow=price_source_evidence,
            derivatives_series=tuple(evidence),
            derivatives_panel_sha256=digest.hexdigest(),
            selection_confirmation_or_terminal_rows_read=False,
        ),
    )


def _derivatives_feature_arrays(
    premium: Mapping[str, PremiumState],
    funding: Mapping[str, FundingState],
    target_symbol: str,
) -> tuple[tuple[str, ...], tuple[np.ndarray, ...]]:
    target_premium = premium[target_symbol]
    target_funding = funding[target_symbol]
    names: list[str] = []
    values: list[np.ndarray] = []

    def add(name: str, value: np.ndarray) -> None:
        names.append(name)
        values.append(value)

    add("target_premium_close_bps", target_premium.close_bps)
    add("target_premium_range_bps", target_premium.range_bps)
    for window in (1, 5, 15, 60):
        add(
            f"target_premium_change_{window}m_bps",
            _lagged_change(target_premium.close_bps, window),
        )
    for window in (15, 60, 240, 1440):
        mean = _rolling_observed_mean(
            target_premium.close_bps,
            target_premium.observed,
            window,
        )
        add(f"target_premium_mean_{window}m_bps", mean)
        if window in (60, 240, 1440):
            standard_deviation = _rolling_observed_std(
                target_premium.close_bps,
                target_premium.observed,
                window,
            )
            zscore = np.zeros_like(mean)
            np.divide(
                target_premium.close_bps - mean,
                standard_deviation,
                out=zscore,
                where=standard_deviation > 1e-12,
            )
            add(f"target_premium_zscore_{window}m", zscore)
    positive = target_premium.observed & (target_premium.close_bps > 0.0)
    for window in (60, 240):
        count = _rolling_sum(target_premium.observed.astype(np.float64), window)
        positive_count = _rolling_sum(positive.astype(np.float64), window)
        fraction = np.full(count.size, np.nan, dtype=np.float64)
        np.divide(positive_count, count, out=fraction, where=count > 0)
        add(f"target_premium_positive_fraction_{window}m", fraction)
    add("target_premium_observed", target_premium.observed.astype(np.float64))
    add("target_premium_age_minutes", target_premium.age_minutes)
    add(
        "target_premium_observed_fraction_60m",
        target_premium.rolling_observed_fraction_60m,
    )
    add(
        "target_premium_observed_fraction_240m",
        target_premium.rolling_observed_fraction_240m,
    )
    add(
        "target_premium_observed_fraction_1440m",
        target_premium.rolling_observed_fraction_1440m,
    )
    premium_matrix = np.vstack([premium[symbol].close_bps for symbol in SYMBOLS])
    premium_valid = np.isfinite(premium_matrix)
    premium_count = np.sum(premium_valid, axis=0)
    premium_sum = np.sum(np.where(premium_valid, premium_matrix, 0.0), axis=0)
    premium_mean = np.divide(
        premium_sum,
        premium_count,
        out=np.zeros(premium_sum.size, dtype=np.float64),
        where=premium_count > 0,
    )
    premium_variance = np.divide(
        np.sum(
            np.where(premium_valid, premium_matrix - premium_mean, 0.0) ** 2,
            axis=0,
        ),
        premium_count,
        out=np.zeros(premium_sum.size, dtype=np.float64),
        where=premium_count > 0,
    )
    add("cross_asset_premium_dispersion_bps", np.sqrt(premium_variance))
    add(
        "cross_asset_premium_sign_agreement",
        np.abs(
            np.divide(
                np.sum(np.where(premium_valid, np.sign(premium_matrix), 0.0), axis=0),
                premium_count,
                out=np.zeros(premium_sum.size, dtype=np.float64),
                where=premium_count > 0,
            )
        ),
    )

    add("target_last_settled_funding_rate_bps", target_funding.last_rate_bps)
    add("target_funding_interval_hours", target_funding.last_interval_hours)
    add("target_minutes_since_funding", target_funding.age_minutes)
    add("target_settled_funding_sum_24h_bps", target_funding.settled_sum_24h_bps)
    add("target_settled_funding_sum_72h_bps", target_funding.settled_sum_72h_bps)
    add(
        "target_settled_funding_sum_168h_bps",
        target_funding.settled_sum_168h_bps,
    )
    add("target_funding_event_mean_30_bps", target_funding.event_mean_30_bps)
    add("target_funding_event_zscore_30", target_funding.event_zscore_30)
    funding_matrix = np.vstack(
        [funding[symbol].last_rate_bps for symbol in SYMBOLS]
    )
    add("cross_asset_funding_dispersion_bps", np.std(funding_matrix, axis=0))
    add(
        "cross_asset_funding_sign_agreement",
        np.abs(np.mean(np.sign(funding_matrix), axis=0)),
    )
    return tuple(names), tuple(values)


def _funding_in_holding_window(
    state: FundingState,
    entry_time_ms: np.ndarray,
    exit_time_ms: np.ndarray,
) -> np.ndarray:
    rate_bps = state.event_rate * 10_000.0
    prefix = _prefix_sum(rate_bps)
    after_entry = np.searchsorted(state.event_time_ms, entry_time_ms, side="right")
    through_exit = np.searchsorted(state.event_time_ms, exit_time_ms, side="right")
    return prefix[through_exit] - prefix[after_entry]


def build_derivatives_hurdle_dataset(
    panel: Mapping[str, MinuteSeries],
    premium: Mapping[str, PremiumState],
    funding: Mapping[str, FundingState],
    source_evidence: DerivativesSourceEvidence,
    *,
    progress: ProgressCallback | None = None,
) -> DerivativesHurdleDataset:
    """Build matched price-only and derivatives-augmented action labels."""

    reference = panel[SYMBOLS[0]]
    common_eligible = np.ones(reference.open_time_ms.size, dtype=bool)
    for symbol in SYMBOLS:
        state = premium[symbol]
        common_eligible &= state.age_minutes <= PREMIUM_MAX_AGE_MINUTES
        for coverage in (
            state.rolling_observed_fraction_15m,
            state.rolling_observed_fraction_60m,
            state.rolling_observed_fraction_240m,
            state.rolling_observed_fraction_1440m,
        ):
            common_eligible &= coverage >= PREMIUM_MIN_ROLLING_COVERAGE
        for window in (1, 5, 15, 60):
            lag_is_finite = np.zeros(reference.open_time_ms.size, dtype=bool)
            lag_is_finite[window:] = np.isfinite(state.close_bps[:-window])
            common_eligible &= lag_is_finite
    role_window = (
        (reference.open_time_ms >= ROLES[0].start_ms)
        & (reference.open_time_ms < ROLES[3].end_exclusive_ms)
    )
    cadence = (
        (reference.open_time_ms // MINUTE_MS) % 5 == 0
    )
    index_valid = (
        np.arange(reference.open_time_ms.size, dtype=np.int64)
        + 1
        + max(HORIZONS_MINUTES)
        < reference.open_time_ms.size
    )
    base_decisions = role_window & cadence & index_valid
    decision_indices = np.flatnonzero(base_decisions & common_eligible)
    source_exclusions = {
        "base_decision_rows": int(np.count_nonzero(base_decisions)),
        "premium_quality_excluded_decision_times": int(
            np.count_nonzero(base_decisions & ~common_eligible)
        ),
        "eligible_decision_times": int(decision_indices.size),
        "eligible_cross_symbol_rows": int(decision_indices.size * len(SYMBOLS)),
    }
    feature_blocks: list[np.ndarray] = []
    time_blocks: list[np.ndarray] = []
    symbol_blocks: list[np.ndarray] = []
    target_blocks: dict[int, list[np.ndarray]] = {
        horizon: [] for horizon in HORIZONS_MINUTES
    }
    long_blocks: dict[int, list[np.ndarray]] = {
        horizon: [] for horizon in HORIZONS_MINUTES
    }
    short_blocks: dict[int, list[np.ndarray]] = {
        horizon: [] for horizon in HORIZONS_MINUTES
    }
    funding_blocks: dict[int, list[np.ndarray]] = {
        horizon: [] for horizon in HORIZONS_MINUTES
    }
    expected_names: tuple[str, ...] | None = None
    price_feature_count: int | None = None
    for symbol_index, symbol in enumerate(SYMBOLS):
        if progress is not None:
            progress("round38_feature_build", {"symbol": symbol, "status": "started"})
        price_names, price_arrays = _feature_arrays(panel, symbol)
        derivative_names, derivative_arrays = _derivatives_feature_arrays(
            premium, funding, symbol
        )
        names = price_names + derivative_names
        if expected_names is None:
            expected_names = names
            price_feature_count = len(price_names)
        elif names != expected_names:
            raise RuntimeError(f"Round 38 feature order differs for {symbol}")
        block = np.column_stack(
            [value[decision_indices] for value in price_arrays + derivative_arrays]
        ).astype(np.float32, copy=False)
        if not np.isfinite(block).all():
            invalid = int(np.count_nonzero(~np.isfinite(block)))
            raise ValueError(f"{symbol} Round 38 features contain {invalid} nonfinite values")
        feature_blocks.append(block)
        decision_times = panel[symbol].open_time_ms[decision_indices].copy()
        time_blocks.append(decision_times)
        symbol_blocks.append(
            np.full(decision_indices.size, symbol_index, dtype=np.int8)
        )
        for horizon in HORIZONS_MINUTES:
            entry_indices = decision_indices + 1
            exit_indices = entry_indices + horizon
            entry = panel[symbol].open[entry_indices]
            exit_values = panel[symbol].open[exit_indices]
            entry_time = panel[symbol].open_time_ms[entry_indices]
            exit_time = panel[symbol].open_time_ms[exit_indices]
            funding_bps = _funding_in_holding_window(
                funding[symbol], entry_time, exit_time
            )
            long_gross = 10_000.0 * (exit_values / entry - 1.0)
            short_gross = 10_000.0 * (1.0 - exit_values / entry)
            long_net = long_gross - EXECUTION_CHARGE_BPS - funding_bps
            short_net = short_gross - EXECUTION_CHARGE_BPS + funding_bps
            target = np.full(decision_indices.size, 1, dtype=np.int8)
            target[(short_net > 0.0) & (short_net > long_net)] = 0
            target[(long_net > 0.0) & (long_net >= short_net)] = 2
            if (
                not np.isfinite(long_net).all()
                or not np.isfinite(short_net).all()
                or not np.isfinite(funding_bps).all()
            ):
                raise ValueError(f"{symbol} h{horizon} produced nonfinite labels")
            target_blocks[horizon].append(target)
            long_blocks[horizon].append(long_net.astype(np.float32))
            short_blocks[horizon].append(short_net.astype(np.float32))
            funding_blocks[horizon].append(funding_bps.astype(np.float32))
        if progress is not None:
            progress(
                "round38_feature_build",
                {
                    "symbol": symbol,
                    "status": "complete",
                    "decision_rows": int(block.shape[0]),
                    "feature_count": int(block.shape[1]),
                },
            )
    if expected_names is None or price_feature_count is None:
        raise RuntimeError("Round 38 produced no feature blocks")
    features = np.concatenate(feature_blocks, axis=0)
    decision_time_ms = np.concatenate(time_blocks)
    symbol_index = np.concatenate(symbol_blocks)
    target_class = {
        horizon: np.concatenate(blocks)
        for horizon, blocks in target_blocks.items()
    }
    long_net = {
        horizon: np.concatenate(blocks) for horizon, blocks in long_blocks.items()
    }
    short_net = {
        horizon: np.concatenate(blocks) for horizon, blocks in short_blocks.items()
    }
    funding_cash_flow = {
        horizon: np.concatenate(blocks)
        for horizon, blocks in funding_blocks.items()
    }
    role_masks = {
        horizon: _role_masks(decision_time_ms, horizon)
        for horizon in HORIZONS_MINUTES
    }
    return DerivativesHurdleDataset(
        feature_names=expected_names,
        price_flow_feature_count=price_feature_count,
        features=features,
        decision_time_ms=decision_time_ms,
        symbol_index=symbol_index,
        target_class=target_class,
        long_net_utility_bps=long_net,
        short_net_utility_bps=short_net,
        funding_cash_flow_bps=funding_cash_flow,
        role_masks=role_masks,
        source_evidence=source_evidence,
        source_exclusions=source_exclusions,
    )


__all__ = [
    "DerivativesHurdleDataset",
    "DerivativesSeriesEvidence",
    "DerivativesSourceEvidence",
    "EXECUTION_CHARGE_BPS",
    "FundingState",
    "PREMIUM_MAX_AGE_MINUTES",
    "PREMIUM_MIN_ROLLING_COVERAGE",
    "PremiumState",
    "build_derivatives_hurdle_dataset",
    "load_derivatives_state",
]
