"""Causal one-second execution-timing overlay for the frozen Round 42 pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

import lightgbm as lgb
import numpy as np

from .derivatives_hurdle_data import DerivativesHurdleDataset, FundingState
from .lightgbm_backend import lightgbm_backend_parameters
from .microstructure_architecture import average_label_uniqueness
from .second_flow_data import (
    END_EXCLUSIVE_MS,
    START_MS,
    SYMBOLS,
    SecondFlowSeries,
    file_sha256,
)


ROUND = 42
SEED = 4201
WINDOWS_SECONDS = (5, 15, 30, 60, 300, 900)
DELAYS_SECONDS = (0, 5, 15, 30)
HORIZON_SECONDS = 1800
PRIMARY_MARGIN = 0.10
BASE_CHARGE_BPS = 12.0
STRESS_CHARGE_BPS = 16.0
MAXIMUM_ENTRIES_PER_SYMBOL_DAY = 8
PROBABILITY_GRID = (0.50, 0.55, 0.60)
EXPECTED_NET_GRID = (0.0, 2.0, 4.0)
LOWER_QUARTILE_GRID = (-20.0, -10.0, 0.0)
JUNE_TEMPERATURE = 1.06538314761829
JUNE_PRIMARY_IDS = (
    "round41_202406_primary_btcusdt",
    "round41_202406_primary_ethusdt",
    "round41_202406_primary_solusdt",
)
ProgressCallback = Callable[[str, Mapping[str, object]], None]


@dataclass(frozen=True)
class Round42Fold:
    fold_id: str
    fit_days: tuple[str, ...]
    early_stop_day: str
    calibration_day: str
    evaluation_day: str


FOLDS = (
    Round42Fold(
        fold_id="evaluation_2024-06-06",
        fit_days=("2024-06-01", "2024-06-02", "2024-06-03"),
        early_stop_day="2024-06-04",
        calibration_day="2024-06-05",
        evaluation_day="2024-06-06",
    ),
    Round42Fold(
        fold_id="evaluation_2024-06-07",
        fit_days=("2024-06-01", "2024-06-02", "2024-06-03", "2024-06-04"),
        early_stop_day="2024-06-05",
        calibration_day="2024-06-06",
        evaluation_day="2024-06-07",
    ),
)


@dataclass(frozen=True)
class TimingModelArtifact:
    model_id: str
    fold_id: str
    model_head: str
    path: str
    sha256: str
    bytes: int
    feature_count: int
    training_rows: int
    early_stop_rows: int
    best_iteration: int
    backend_kind: str
    backend_device: str
    reload_max_abs_prediction_error: float
    top_feature_gain: tuple[Mapping[str, object], ...]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TimingDataset:
    feature_names: tuple[str, ...]
    features: np.ndarray
    proposal_source_index: np.ndarray
    proposal_decision_time_ms: np.ndarray
    proposal_entry_time_ms: np.ndarray
    proposal_symbol_index: np.ndarray
    proposal_side: np.ndarray
    proposal_primary_probabilities: np.ndarray
    proposal_margin: np.ndarray
    proposal_day: np.ndarray
    proposal_weight: np.ndarray
    option_proposal_index: np.ndarray
    option_delay_seconds: np.ndarray
    option_base_net_bps: np.ndarray
    option_stress_net_bps: np.ndarray
    option_funding_bps: np.ndarray
    primary_artifacts: tuple[Mapping[str, object], ...]
    proposal_exclusions: Mapping[str, object]

    @property
    def proposals(self) -> int:
        return int(self.proposal_source_index.size)

    @property
    def option_rows(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class ReplayMetrics:
    total_trades: int
    active_utc_days: int
    trades_by_symbol: Mapping[str, int]
    delay_counts: Mapping[str, int]
    nonzero_delay_fraction: float
    maximum_single_symbol_fraction: float
    mean_net_bps: float
    median_net_bps: float
    total_net_bps: float
    positive_rate: float
    profit_factor: float | None
    maximum_peak_to_trough_drawdown_bps: float
    longest_loss_streak: int
    mean_matched_immediate_net_bps: float
    mean_increment_over_matched_immediate_bps: float
    day_block_bootstrap_mean_net_bps_lower_95: float | None
    day_block_bootstrap_mean_net_bps_median: float | None
    day_block_bootstrap_mean_net_bps_upper_95: float | None

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReplayEvidence:
    metrics: ReplayMetrics
    role_proposals: int
    eligible_proposals: int
    vetoed_proposals: int
    veto_fraction: float
    overlap_rejections: int
    capacity_rejections: int
    selected_proposal_indices: tuple[int, ...]
    selected_option_indices: tuple[int, ...]

    def asdict(self) -> dict[str, object]:
        return {
            "metrics": self.metrics.asdict(),
            "role_proposals": self.role_proposals,
            "eligible_proposals": self.eligible_proposals,
            "vetoed_proposals": self.vetoed_proposals,
            "veto_fraction": self.veto_fraction,
            "overlap_rejections": self.overlap_rejections,
            "capacity_rejections": self.capacity_rejections,
            "selected_proposal_indices": list(self.selected_proposal_indices),
            "selected_option_indices": list(self.selected_option_indices),
        }


@dataclass(frozen=True)
class Round42Screen:
    folds: tuple[Mapping[str, object], ...]
    aggregate: Mapping[str, object]
    model_artifacts: tuple[TimingModelArtifact, ...]
    pilot_gate_passed: bool
    pilot_gate_reasons: tuple[str, ...]
    backend_kind: str
    backend_device: str


class _SeriesCache:
    def __init__(self, series: SecondFlowSeries) -> None:
        self.series = series
        close = series.close
        returns = np.zeros(series.rows, dtype=np.float64)
        returns[1:] = 10_000.0 * np.log(close[1:] / close[:-1])
        self.returns = returns
        self.absolute_returns = np.abs(returns)
        self.squared_returns = np.square(returns)
        self.signed_base = 2.0 * series.taker_buy_base_volume - series.volume
        self.signed_quote = 2.0 * series.taker_buy_quote_volume - series.quote_volume
        self.zero_trade = (series.trade_count == 0).astype(np.float64)
        self.prefix = {
            "return": _prefix(returns),
            "absolute_return": _prefix(self.absolute_returns),
            "squared_return": _prefix(self.squared_returns),
            "volume": _prefix(series.volume),
            "quote_volume": _prefix(series.quote_volume),
            "trade_count": _prefix(series.trade_count.astype(np.float64)),
            "signed_base": _prefix(self.signed_base),
            "signed_quote": _prefix(self.signed_quote),
            "zero_trade": _prefix(self.zero_trade),
        }


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate(([0.0], np.cumsum(values, dtype=np.float64)))


def _sum(prefix: np.ndarray, start: int, end_exclusive: int) -> float:
    return float(prefix[end_exclusive] - prefix[start])


def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-9, 1.0)
    logits = np.log(clipped) / float(temperature)
    logits -= np.max(logits, axis=1, keepdims=True)
    scaled = np.exp(logits)
    scaled /= np.sum(scaled, axis=1, keepdims=True)
    return scaled.astype(np.float32)


def _round41_primary_artifact(
    report: Mapping[str, object], model_id: str
) -> Mapping[str, object]:
    artifacts = report.get("model_artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("Round 41 model artifacts are missing")
    matches = [item for item in artifacts if item.get("model_id") == model_id]
    if len(matches) != 1:
        raise ValueError(f"Round 41 primary artifact is missing: {model_id}")
    return matches[0]


def _primary_probabilities(
    minute_dataset: DerivativesHurdleDataset,
    *,
    round41_report: Mapping[str, object],
    round41_evidence_root: Path,
) -> tuple[np.ndarray, tuple[Mapping[str, object], ...]]:
    features = minute_dataset.feature_view("price_flow_only")
    probabilities = np.full((minute_dataset.rows, 3), np.nan, dtype=np.float32)
    evidence: list[Mapping[str, object]] = []
    june_mask = (minute_dataset.decision_time_ms >= START_MS) & (
        minute_dataset.decision_time_ms < END_EXCLUSIVE_MS
    )
    for symbol_index, (symbol, model_id) in enumerate(
        zip(SYMBOLS, JUNE_PRIMARY_IDS, strict=True)
    ):
        artifact = _round41_primary_artifact(round41_report, model_id)
        model_path = round41_evidence_root / "models" / f"{model_id}.txt"
        if (
            not model_path.is_file()
            or file_sha256(model_path) != artifact.get("sha256")
            or model_path.stat().st_size != artifact.get("bytes")
            or artifact.get("reload_max_abs_prediction_error") != 0.0
            or artifact.get("backend_kind") != "opencl"
        ):
            raise ValueError(f"Round 41 primary artifact drifted: {model_id}")
        rows = june_mask & (minute_dataset.symbol_index == symbol_index)
        booster = lgb.Booster(model_file=str(model_path))
        probabilities[rows] = np.asarray(
            booster.predict(features[rows]), dtype=np.float32
        )
        evidence.append(
            {
                "model_id": model_id,
                "symbol": symbol,
                "path": str(model_path.resolve()),
                "sha256": artifact["sha256"],
                "bytes": artifact["bytes"],
                "best_iteration": artifact["best_iteration"],
                "backend_kind": artifact["backend_kind"],
                "backend_device": artifact["backend_device"],
            }
        )
    if not np.isfinite(probabilities[june_mask]).all():
        raise ValueError("Round 41 June primary predictions are nonfinite")
    probabilities[june_mask] = _temperature_scale(
        probabilities[june_mask], JUNE_TEMPERATURE
    )
    return probabilities, tuple(evidence)


def _window_features(
    cache: _SeriesCache,
    *,
    end_index: int,
    window: int,
) -> tuple[float, ...]:
    start = end_index - window + 1
    previous_start = start - window
    if previous_start < 0:
        raise ValueError("Round 42 feature history is incomplete")
    end_exclusive = end_index + 1
    previous_end = start
    net_return = _sum(cache.prefix["return"], start, end_exclusive)
    squared = _sum(cache.prefix["squared_return"], start, end_exclusive)
    path = _sum(cache.prefix["absolute_return"], start, end_exclusive)
    volume = _sum(cache.prefix["volume"], start, end_exclusive)
    quote = _sum(cache.prefix["quote_volume"], start, end_exclusive)
    trades = _sum(cache.prefix["trade_count"], start, end_exclusive)
    signed_base = _sum(cache.prefix["signed_base"], start, end_exclusive)
    signed_quote = _sum(cache.prefix["signed_quote"], start, end_exclusive)
    previous_base = _sum(cache.prefix["signed_base"], previous_start, previous_end)
    previous_volume = _sum(cache.prefix["volume"], previous_start, previous_end)
    base_imbalance = signed_base / max(volume, 1e-12)
    quote_imbalance = signed_quote / max(quote, 1e-12)
    previous_imbalance = previous_base / max(previous_volume, 1e-12)
    series = cache.series
    high = float(np.max(series.high[start:end_exclusive]))
    low = float(np.min(series.low[start:end_exclusive]))
    last_close = float(series.close[end_index])
    vwap = quote / volume if volume > 0.0 else last_close
    range_bps = 10_000.0 * (high / max(low, 1e-12) - 1.0)
    vwap_offset = 10_000.0 * (vwap / last_close - 1.0)
    impact = abs(net_return) / (1.0 + abs(signed_quote) / 1_000_000.0)
    return (
        net_return,
        float(np.sqrt(max(squared, 0.0))),
        range_bps,
        abs(net_return) / max(path, 1e-12),
        base_imbalance,
        quote_imbalance,
        base_imbalance - previous_imbalance,
        float(np.log1p(quote / window)),
        float(np.log1p(trades / window)),
        _sum(cache.prefix["zero_trade"], start, end_exclusive) / window,
        vwap_offset,
        impact,
    )


_WINDOW_FEATURE_LABELS = (
    "return_bps",
    "realized_volatility_bps",
    "high_low_range_bps",
    "path_efficiency",
    "signed_taker_base_imbalance",
    "signed_taker_quote_imbalance",
    "signed_flow_acceleration",
    "log_quote_volume_per_second",
    "log_trade_count_per_second",
    "zero_trade_fraction",
    "vwap_minus_close_bps",
    "absolute_return_per_signed_quote_flow",
)


def _base_feature_names() -> tuple[str, ...]:
    names = [
        "primary_p_short",
        "primary_p_abstain",
        "primary_p_long",
        "primary_direction_margin",
        "primary_probability_entropy",
        "proposal_side",
        "symbol_btcusdt",
        "symbol_ethusdt",
        "symbol_solusdt",
    ]
    for symbol in SYMBOLS:
        for window in WINDOWS_SECONDS:
            for label in _WINDOW_FEATURE_LABELS:
                names.append(f"{symbol.lower()}_{label}_{window}s")
    for window in WINDOWS_SECONDS:
        names.extend(
            (
                f"target_minus_market_return_{window}s",
                f"target_minus_market_signed_flow_{window}s",
                f"cross_asset_return_dispersion_{window}s",
                f"cross_asset_signed_flow_agreement_{window}s",
            )
        )
    return tuple(names)


def _proposal_features(
    *,
    caches: Mapping[str, _SeriesCache],
    entry_time_ms: int,
    symbol_index: int,
    side: int,
    probabilities: np.ndarray,
    margin: float,
) -> list[float]:
    entropy = float(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-9, 1.0))))
    values = [
        float(probabilities[0]),
        float(probabilities[1]),
        float(probabilities[2]),
        margin,
        entropy,
        float(side),
        float(symbol_index == 0),
        float(symbol_index == 1),
        float(symbol_index == 2),
    ]
    by_window: dict[int, list[tuple[float, ...]]] = {
        window: [] for window in WINDOWS_SECONDS
    }
    feature_end_time_ms = entry_time_ms - 1000
    for symbol in SYMBOLS:
        cache = caches[symbol]
        end_index = (feature_end_time_ms - START_MS) // 1000
        if end_index < 0 or end_index >= cache.series.rows:
            raise ValueError("Round 42 feature timestamp is outside the source")
        for window in WINDOWS_SECONDS:
            raw = _window_features(cache, end_index=int(end_index), window=window)
            directional = list(raw)
            for feature_index in (0, 4, 5, 6, 10):
                directional[feature_index] *= side
            by_window[window].append(tuple(directional))
            values.extend(directional)
    for window in WINDOWS_SECONDS:
        blocks = by_window[window]
        target = blocks[symbol_index]
        market_return = float(np.mean([block[0] for block in blocks]))
        market_flow = float(np.mean([block[4] for block in blocks]))
        values.extend(
            (
                target[0] - market_return,
                target[4] - market_flow,
                float(np.std([block[0] for block in blocks])),
                float(np.mean([np.sign(block[4]) for block in blocks])),
            )
        )
    return values


def _funding_bps(
    funding: FundingState,
    *,
    entry_time_ms: int,
    exit_time_ms: int,
) -> float:
    after_entry = int(
        np.searchsorted(funding.event_time_ms, entry_time_ms, side="right")
    )
    before_exit = int(np.searchsorted(funding.event_time_ms, exit_time_ms, side="left"))
    return float(10_000.0 * np.sum(funding.event_rate[after_entry:before_exit]))


def _proposal_uniqueness(
    entry_time_ms: np.ndarray, symbol_index: np.ndarray
) -> np.ndarray:
    output = np.empty(entry_time_ms.size, dtype=np.float32)
    for index in range(len(SYMBOLS)):
        rows = np.flatnonzero(symbol_index == index)
        order = np.argsort(entry_time_ms[rows], kind="stable")
        ordered_rows = rows[order]
        times = entry_time_ms[ordered_rows]
        exits = times + (HORIZON_SECONDS + max(DELAYS_SECONDS)) * 1000
        weights = average_label_uniqueness(
            times,
            exits,
            np.arange(times.size, dtype=np.int64),
        )
        output[ordered_rows] = weights
    output /= float(np.mean(output))
    return output


def build_timing_dataset(
    minute_dataset: DerivativesHurdleDataset,
    second_flow: Mapping[str, SecondFlowSeries],
    funding: Mapping[str, FundingState],
    *,
    round41_report: Mapping[str, object],
    round41_evidence_root: Path,
    progress: ProgressCallback | None = None,
) -> TimingDataset:
    probabilities, primary_artifacts = _primary_probabilities(
        minute_dataset,
        round41_report=round41_report,
        round41_evidence_root=round41_evidence_root,
    )
    p_short = probabilities[:, 0]
    p_long = probabilities[:, 2]
    margin = np.abs(p_long - p_short)
    entry_time_ms = minute_dataset.decision_time_ms + 60_000
    proposal_mask = (
        (minute_dataset.decision_time_ms >= START_MS)
        & (minute_dataset.decision_time_ms < END_EXCLUSIVE_MS)
        & np.isfinite(margin)
        & (margin >= PRIMARY_MARGIN)
    )
    source_indices = np.flatnonzero(proposal_mask)
    if source_indices.size == 0:
        raise ValueError("Round 42 frozen primary proposal set is empty")
    raw_proposal_count = int(source_indices.size)
    raw_symbol_counts = {
        symbol: int(
            np.count_nonzero(minute_dataset.symbol_index[source_indices] == index)
        )
        for index, symbol in enumerate(SYMBOLS)
    }
    side = np.where(p_long[source_indices] > p_short[source_indices], 1, -1).astype(
        np.int8
    )
    symbols = minute_dataset.symbol_index[source_indices].astype(np.int8)
    entries = entry_time_ms[source_indices].astype(np.int64)
    decisions = minute_dataset.decision_time_ms[source_indices].astype(np.int64)
    days = (entries // 86_400_000).astype(np.int64)
    day_end = (days + 1) * 86_400_000
    complete = (entries >= START_MS + 2 * max(WINDOWS_SECONDS) * 1000) & (
        entries + (max(DELAYS_SECONDS) + HORIZON_SECONDS) * 1000
        < np.minimum(day_end, END_EXCLUSIVE_MS)
    )
    source_indices = source_indices[complete]
    side = side[complete]
    symbols = symbols[complete]
    entries = entries[complete]
    decisions = decisions[complete]
    days = days[complete]
    primary = probabilities[source_indices]
    margins = margin[source_indices].astype(np.float32)
    if source_indices.size == 0 or set(np.unique(symbols)) != {0, 1, 2}:
        raise ValueError("Round 42 complete proposal set lacks symbol coverage")
    caches = {symbol: _SeriesCache(second_flow[symbol]) for symbol in SYMBOLS}
    base_names = _base_feature_names()
    base_features = np.empty((source_indices.size, len(base_names)), dtype=np.float32)
    for row in range(source_indices.size):
        base_features[row] = np.asarray(
            _proposal_features(
                caches=caches,
                entry_time_ms=int(entries[row]),
                symbol_index=int(symbols[row]),
                side=int(side[row]),
                probabilities=primary[row],
                margin=float(margins[row]),
            ),
            dtype=np.float32,
        )
        if progress is not None and row and row % 500 == 0:
            progress(
                "round42_feature_build",
                {
                    "status": "running",
                    "completed_proposals": row,
                    "total_proposals": int(source_indices.size),
                },
            )
    if not np.isfinite(base_features).all():
        raise ValueError("Round 42 causal second-flow features are nonfinite")
    option_proposal = np.repeat(
        np.arange(source_indices.size, dtype=np.int64), len(DELAYS_SECONDS)
    )
    option_delay = np.tile(
        np.asarray(DELAYS_SECONDS, dtype=np.int16), source_indices.size
    )
    option_features = np.repeat(base_features, len(DELAYS_SECONDS), axis=0)
    delay_fraction = option_delay.astype(np.float32) / max(DELAYS_SECONDS)
    option_features = np.column_stack(
        (
            option_features,
            delay_fraction,
            (option_delay > 0).astype(np.float32),
        )
    ).astype(np.float32, copy=False)
    feature_names = base_names + ("entry_delay_fraction", "entry_is_delayed")
    base_net = np.empty(option_proposal.size, dtype=np.float32)
    stress_net = np.empty(option_proposal.size, dtype=np.float32)
    funding_bps = np.empty(option_proposal.size, dtype=np.float32)
    for option_index, proposal_index in enumerate(option_proposal):
        symbol_index = int(symbols[proposal_index])
        symbol = SYMBOLS[symbol_index]
        delay = int(option_delay[option_index])
        entry_ms = int(entries[proposal_index]) + delay * 1000
        exit_ms = entry_ms + HORIZON_SECONDS * 1000
        series = second_flow[symbol]
        entry_index = (entry_ms - START_MS) // 1000
        exit_index = (exit_ms - START_MS) // 1000
        entry_price = float(series.open[entry_index])
        exit_price = float(series.open[exit_index])
        gross = int(side[proposal_index]) * 10_000.0 * (exit_price / entry_price - 1.0)
        cash_flow = _funding_bps(
            funding[symbol], entry_time_ms=entry_ms, exit_time_ms=exit_ms
        )
        utility = gross - BASE_CHARGE_BPS - int(side[proposal_index]) * cash_flow
        base_net[option_index] = utility
        stress_net[option_index] = utility - (STRESS_CHARGE_BPS - BASE_CHARGE_BPS)
        funding_bps[option_index] = cash_flow
    proposal_weights = _proposal_uniqueness(entries, symbols)
    if progress is not None:
        progress(
            "round42_feature_build",
            {
                "status": "complete",
                "proposals": int(source_indices.size),
                "option_rows": int(option_proposal.size),
                "feature_count": len(feature_names),
                "feature_bytes": int(option_features.nbytes),
            },
        )
    return TimingDataset(
        feature_names=feature_names,
        features=option_features,
        proposal_source_index=source_indices,
        proposal_decision_time_ms=decisions,
        proposal_entry_time_ms=entries,
        proposal_symbol_index=symbols,
        proposal_side=side,
        proposal_primary_probabilities=primary,
        proposal_margin=margins,
        proposal_day=days,
        proposal_weight=proposal_weights,
        option_proposal_index=option_proposal,
        option_delay_seconds=option_delay,
        option_base_net_bps=base_net,
        option_stress_net_bps=stress_net,
        option_funding_bps=funding_bps,
        primary_artifacts=primary_artifacts,
        proposal_exclusions={
            "frozen_primary_margin_proposals": raw_proposal_count,
            "incomplete_feature_or_same_day_horizon": raw_proposal_count
            - int(source_indices.size),
            "complete_proposals": int(source_indices.size),
            "raw_by_symbol": raw_symbol_counts,
            "complete_by_symbol": {
                symbol: int(np.count_nonzero(symbols == index))
                for index, symbol in enumerate(SYMBOLS)
            },
        },
    )


def _day_id(value: str) -> int:
    return int(np.datetime64(value, "D").astype(np.int64))


def _option_role_mask(dataset: TimingDataset, days: Sequence[str]) -> np.ndarray:
    identifiers = np.asarray([_day_id(day) for day in days], dtype=np.int64)
    proposal_mask = np.isin(dataset.proposal_day, identifiers)
    return proposal_mask[dataset.option_proposal_index]


def _proposal_role_mask(dataset: TimingDataset, days: Sequence[str]) -> np.ndarray:
    identifiers = np.asarray([_day_id(day) for day in days], dtype=np.int64)
    return np.isin(dataset.proposal_day, identifiers)


def _validate_timing_dataset(dataset: TimingDataset) -> None:
    proposals = dataset.proposals
    expected_options = proposals * len(DELAYS_SECONDS)
    if (
        proposals <= 0
        or dataset.features.ndim != 2
        or dataset.features.shape != (expected_options, len(dataset.feature_names))
        or dataset.option_rows != expected_options
    ):
        raise ValueError("Round 42 timing dataset shape is invalid")
    proposal_fields = (
        dataset.proposal_decision_time_ms,
        dataset.proposal_entry_time_ms,
        dataset.proposal_symbol_index,
        dataset.proposal_side,
        dataset.proposal_margin,
        dataset.proposal_day,
        dataset.proposal_weight,
    )
    option_fields = (
        dataset.option_proposal_index,
        dataset.option_delay_seconds,
        dataset.option_base_net_bps,
        dataset.option_stress_net_bps,
        dataset.option_funding_bps,
    )
    if (
        any(values.size != proposals for values in proposal_fields)
        or dataset.proposal_primary_probabilities.shape != (proposals, 3)
        or any(values.size != expected_options for values in option_fields)
    ):
        raise ValueError("Round 42 timing dataset field length is invalid")
    expected_proposals = np.repeat(
        np.arange(proposals, dtype=np.int64), len(DELAYS_SECONDS)
    )
    expected_delays = np.tile(np.asarray(DELAYS_SECONDS, dtype=np.int16), proposals)
    if not np.array_equal(
        dataset.option_proposal_index, expected_proposals
    ) or not np.array_equal(dataset.option_delay_seconds, expected_delays):
        raise ValueError("Round 42 delay-option ordering drifted")
    if (
        not np.isfinite(dataset.features).all()
        or not np.isfinite(dataset.proposal_primary_probabilities).all()
        or not np.isfinite(dataset.proposal_margin).all()
        or not np.isfinite(dataset.proposal_weight).all()
        or not np.isfinite(dataset.option_base_net_bps).all()
        or not np.isfinite(dataset.option_stress_net_bps).all()
        or not np.isfinite(dataset.option_funding_bps).all()
    ):
        raise ValueError("Round 42 timing dataset contains nonfinite values")
    if (
        np.unique(dataset.proposal_source_index).size != proposals
        or not np.all(np.isin(dataset.proposal_symbol_index, range(len(SYMBOLS))))
        or not np.all(np.isin(dataset.proposal_side, (-1, 1)))
        or np.any(dataset.proposal_margin < PRIMARY_MARGIN)
        or np.any(dataset.proposal_weight <= 0.0)
        or not np.isclose(float(np.mean(dataset.proposal_weight)), 1.0, atol=1e-6)
        or not np.allclose(
            np.sum(dataset.proposal_primary_probabilities, axis=1), 1.0, atol=1e-6
        )
        or not np.allclose(
            dataset.option_stress_net_bps,
            dataset.option_base_net_bps - (STRESS_CHARGE_BPS - BASE_CHARGE_BPS),
            atol=1e-6,
        )
    ):
        raise ValueError("Round 42 timing dataset invariants failed")
    for fold in FOLDS:
        roles = (
            fold.fit_days,
            (fold.early_stop_day,),
            (fold.calibration_day,),
            (fold.evaluation_day,),
        )
        for days in roles:
            mask = _proposal_role_mask(dataset, days)
            if not np.any(mask) or set(
                np.unique(dataset.proposal_symbol_index[mask])
            ) != {
                0,
                1,
                2,
            }:
                raise ValueError(
                    f"Round 42 role lacks complete symbol coverage: {days}"
                )


def _auc(labels: np.ndarray, values: np.ndarray) -> float:
    binary = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(values, dtype=np.float64)
    positives = int(np.count_nonzero(binary == 1))
    negatives = int(np.count_nonzero(binary == 0))
    if positives == 0 or negatives == 0:
        return 0.5
    order = np.argsort(scores, kind="stable")
    sorted_values = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    cursor = 0
    while cursor < order.size:
        end = cursor + 1
        while end < order.size and sorted_values[end] == sorted_values[cursor]:
            end += 1
        ranks[order[cursor:end]] = 0.5 * (cursor + end - 1) + 1.0
        cursor = end
    rank_sum = float(np.sum(ranks[binary == 1]))
    return float(
        (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    )


def _ranks(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    order = np.argsort(source, kind="stable")
    output = np.empty(source.size, dtype=np.float64)
    output[order] = np.arange(source.size, dtype=np.float64)
    return output


def _classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, object]:
    binary = np.asarray(labels, dtype=np.int8)
    predicted = np.asarray(probabilities, dtype=np.float64)
    clipped = np.clip(predicted, 1e-9, 1.0 - 1e-9)
    return {
        "rows": int(binary.size),
        "positive_rows": int(np.count_nonzero(binary)),
        "positive_fraction": float(np.mean(binary)) if binary.size else 0.0,
        "roc_auc": _auc(binary, predicted),
        "brier_score": float(np.mean(np.square(predicted - binary))),
        "log_loss": float(
            -np.mean(binary * np.log(clipped) + (1 - binary) * np.log1p(-clipped))
        ),
    }


def _regression_metrics(
    labels: np.ndarray, predictions: np.ndarray
) -> dict[str, object]:
    actual = np.asarray(labels, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    pearson = 0.0
    spearman = 0.0
    if actual.size > 1 and np.std(actual) > 0.0 and np.std(predicted) > 0.0:
        pearson = float(np.corrcoef(actual, predicted)[0, 1])
        spearman = float(np.corrcoef(_ranks(actual), _ranks(predicted))[0, 1])
    return {
        "rows": int(actual.size),
        "mean_actual_net_bps": float(np.mean(actual)),
        "mean_predicted_net_bps": float(np.mean(predicted)),
        "mean_absolute_error_bps": float(np.mean(np.abs(actual - predicted))),
        "root_mean_squared_error_bps": float(
            np.sqrt(np.mean(np.square(actual - predicted)))
        ),
        "pearson_information_coefficient": pearson,
        "spearman_information_coefficient": spearman,
    }


def _quantile_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    actual = np.asarray(labels, dtype=np.float64)
    predicted = np.asarray(predictions, dtype=np.float64)
    residual = actual - predicted
    alpha = 0.25
    pinball = np.maximum(alpha * residual, (alpha - 1.0) * residual)
    return {
        "rows": int(actual.size),
        "target_quantile": alpha,
        "empirical_below_prediction_fraction": float(np.mean(actual <= predicted)),
        "pinball_loss_bps": float(np.mean(pinball)),
        "mean_prediction_bps": float(np.mean(predicted)),
    }


def _head_parameters(
    compute_backend: str,
    *,
    head: str,
    seed: int,
) -> tuple[dict[str, object], str, str]:
    parameters, kind, device = lightgbm_backend_parameters(
        compute_backend,
        seed,
        reproducible=True,
    )
    objective = {
        "positive_net_probability": "binary",
        "robust_expected_net_utility": "huber",
        "lower_quartile_net_utility": "quantile",
    }[head]
    metric = {
        "positive_net_probability": "binary_logloss",
        "robust_expected_net_utility": "l1",
        "lower_quartile_net_utility": "quantile",
    }[head]
    parameters.update(
        {
            "objective": objective,
            "metric": metric,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "max_bin": 255,
            "feature_pre_filter": False,
        }
    )
    if objective in {"huber", "quantile"}:
        parameters["alpha"] = 0.25 if objective == "quantile" else 0.9
    return parameters, kind, device


def _fit_head(
    dataset: TimingDataset,
    *,
    fold: Round42Fold,
    head: str,
    model_dir: Path,
    compute_backend: str,
    seed: int,
    progress: ProgressCallback | None,
) -> tuple[np.ndarray, TimingModelArtifact, dict[str, object]]:
    training = _option_role_mask(dataset, fold.fit_days)
    early_stop = _option_role_mask(dataset, (fold.early_stop_day,))
    labels = dataset.option_base_net_bps
    fit_labels = (
        (labels > 0.0).astype(np.int8) if head == "positive_net_probability" else labels
    )
    if not np.any(training) or not np.any(early_stop):
        raise ValueError(f"{fold.fold_id} {head} has an empty model role")
    if head == "positive_net_probability" and (
        np.unique(fit_labels[training]).size != 2
        or np.unique(fit_labels[early_stop]).size != 2
    ):
        raise ValueError(f"{fold.fold_id} binary labels lack both classes")
    weights = dataset.proposal_weight[dataset.option_proposal_index] / len(
        DELAYS_SECONDS
    )
    parameters, kind, device = _head_parameters(compute_backend, head=head, seed=seed)
    model_id = f"round42_{fold.evaluation_day.replace('-', '')}_{head}"
    if progress is not None:
        progress(
            "round42_model_training",
            {
                "status": "started",
                "model_id": model_id,
                "fold_id": fold.fold_id,
                "model_head": head,
                "training_rows": int(np.count_nonzero(training)),
                "early_stop_rows": int(np.count_nonzero(early_stop)),
                "backend_kind": kind,
                "backend_device": device,
            },
        )
    train_set = lgb.Dataset(
        dataset.features[training],
        label=fit_labels[training],
        weight=weights[training],
        feature_name=list(dataset.feature_names),
        free_raw_data=True,
    )
    validation_set = lgb.Dataset(
        dataset.features[early_stop],
        label=fit_labels[early_stop],
        weight=weights[early_stop],
        feature_name=list(dataset.feature_names),
        reference=train_set,
        free_raw_data=True,
    )
    booster = lgb.train(
        parameters,
        train_set,
        num_boost_round=1000,
        valid_sets=[validation_set],
        valid_names=["early_stop"],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    predictions = np.asarray(
        booster.predict(dataset.features, num_iteration=booster.best_iteration),
        dtype=np.float32,
    )
    if not np.isfinite(predictions).all():
        raise ValueError(f"{model_id} predictions are nonfinite")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{model_id}.txt"
    booster.save_model(str(model_path), num_iteration=booster.best_iteration)
    reloaded = lgb.Booster(model_file=str(model_path))
    probe = np.flatnonzero(early_stop)[:4096]
    reload_predictions = np.asarray(
        reloaded.predict(dataset.features[probe]), dtype=np.float32
    )
    reload_error = float(np.max(np.abs(predictions[probe] - reload_predictions)))
    if reload_error != 0.0:
        raise ValueError(f"{model_id} reload predictions are not exact")
    gains = booster.feature_importance(importance_type="gain")
    order = np.argsort(gains)[::-1][:20]
    artifact = TimingModelArtifact(
        model_id=model_id,
        fold_id=fold.fold_id,
        model_head=head,
        path=str(model_path.resolve()),
        sha256=file_sha256(model_path),
        bytes=model_path.stat().st_size,
        feature_count=len(dataset.feature_names),
        training_rows=int(np.count_nonzero(training)),
        early_stop_rows=int(np.count_nonzero(early_stop)),
        best_iteration=int(booster.best_iteration),
        backend_kind=kind,
        backend_device=device,
        reload_max_abs_prediction_error=reload_error,
        top_feature_gain=tuple(
            {
                "feature": dataset.feature_names[int(index)],
                "gain": float(gains[index]),
            }
            for index in order
        ),
    )
    role_days = {
        "fit": fold.fit_days,
        "early_stop": (fold.early_stop_day,),
        "threshold_calibration": (fold.calibration_day,),
        "evaluation": (fold.evaluation_day,),
    }
    diagnostics: dict[str, object] = {}
    for role, days in role_days.items():
        mask = _option_role_mask(dataset, days)
        if head == "positive_net_probability":
            diagnostics[role] = _classification_metrics(
                fit_labels[mask], predictions[mask]
            )
        elif head == "robust_expected_net_utility":
            diagnostics[role] = _regression_metrics(labels[mask], predictions[mask])
        else:
            diagnostics[role] = _quantile_metrics(labels[mask], predictions[mask])
    if progress is not None:
        progress(
            "round42_model_training",
            {
                "status": "complete",
                "model_id": model_id,
                "best_iteration": artifact.best_iteration,
                "artifact_sha256": artifact.sha256,
                "reload_max_abs_prediction_error": reload_error,
            },
        )
    return predictions, artifact, diagnostics


def _bootstrap_day_means(
    outcomes: np.ndarray,
    days: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float | None, float | None, float | None]:
    if samples <= 0 or outcomes.size == 0:
        return None, None, None
    unique = np.unique(days)
    totals = np.asarray([np.sum(outcomes[days == day]) for day in unique])
    counts = np.asarray([np.count_nonzero(days == day) for day in unique])
    rng = np.random.default_rng(seed)
    estimates = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        chosen = rng.integers(0, unique.size, size=unique.size)
        estimates[index] = float(np.sum(totals[chosen]) / np.sum(counts[chosen]))
    return tuple(float(value) for value in np.quantile(estimates, (0.025, 0.5, 0.975)))


def _selection_metrics(
    dataset: TimingDataset,
    *,
    proposal_indices: np.ndarray,
    option_indices: np.ndarray,
    outcomes: np.ndarray,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> ReplayMetrics:
    if proposal_indices.size == 0:
        return ReplayMetrics(
            total_trades=0,
            active_utc_days=0,
            trades_by_symbol={symbol: 0 for symbol in SYMBOLS},
            delay_counts={str(delay): 0 for delay in DELAYS_SECONDS},
            nonzero_delay_fraction=0.0,
            maximum_single_symbol_fraction=0.0,
            mean_net_bps=0.0,
            median_net_bps=0.0,
            total_net_bps=0.0,
            positive_rate=0.0,
            profit_factor=0.0,
            maximum_peak_to_trough_drawdown_bps=0.0,
            longest_loss_streak=0,
            mean_matched_immediate_net_bps=0.0,
            mean_increment_over_matched_immediate_bps=0.0,
            day_block_bootstrap_mean_net_bps_lower_95=None,
            day_block_bootstrap_mean_net_bps_median=None,
            day_block_bootstrap_mean_net_bps_upper_95=None,
        )
    values = outcomes[option_indices].astype(np.float64)
    immediate_options = proposal_indices * len(DELAYS_SECONDS)
    immediate = outcomes[immediate_options].astype(np.float64)
    symbols = dataset.proposal_symbol_index[proposal_indices]
    delays = dataset.option_delay_seconds[option_indices]
    days = dataset.proposal_day[proposal_indices]
    gains = float(np.sum(values[values > 0.0]))
    losses = float(-np.sum(values[values < 0.0]))
    profit_factor = gains / losses if losses > 0.0 else None
    cumulative = np.cumsum(values)
    running_peak = np.maximum.accumulate(np.concatenate(([0.0], cumulative)))
    drawdown = running_peak[1:] - cumulative
    longest = 0
    current = 0
    for value in values:
        if value < 0.0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    lower, median, upper = _bootstrap_day_means(
        values, days, samples=bootstrap_samples, seed=bootstrap_seed
    )
    trades_by_symbol = {
        symbol: int(np.count_nonzero(symbols == index))
        for index, symbol in enumerate(SYMBOLS)
    }
    delay_counts = {
        str(delay): int(np.count_nonzero(delays == delay)) for delay in DELAYS_SECONDS
    }
    return ReplayMetrics(
        total_trades=int(values.size),
        active_utc_days=int(np.unique(days).size),
        trades_by_symbol=trades_by_symbol,
        delay_counts=delay_counts,
        nonzero_delay_fraction=float(np.mean(delays > 0)),
        maximum_single_symbol_fraction=max(trades_by_symbol.values()) / values.size,
        mean_net_bps=float(np.mean(values)),
        median_net_bps=float(np.median(values)),
        total_net_bps=float(np.sum(values)),
        positive_rate=float(np.mean(values > 0.0)),
        profit_factor=profit_factor,
        maximum_peak_to_trough_drawdown_bps=float(np.max(drawdown, initial=0.0)),
        longest_loss_streak=longest,
        mean_matched_immediate_net_bps=float(np.mean(immediate)),
        mean_increment_over_matched_immediate_bps=float(np.mean(values - immediate)),
        day_block_bootstrap_mean_net_bps_lower_95=lower,
        day_block_bootstrap_mean_net_bps_median=median,
        day_block_bootstrap_mean_net_bps_upper_95=upper,
    )


def _capacity_selection(
    dataset: TimingDataset,
    *,
    role_mask: np.ndarray,
    proposed_option: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    selected_proposals: list[int] = []
    selected_options: list[int] = []
    overlap_rejections = 0
    capacity_rejections = 0
    for symbol_index in range(len(SYMBOLS)):
        proposals = np.flatnonzero(
            role_mask
            & (dataset.proposal_symbol_index == symbol_index)
            & (proposed_option >= 0)
        )
        proposals = proposals[
            np.argsort(dataset.proposal_entry_time_ms[proposals], kind="stable")
        ]
        next_available = -1
        day_counts: dict[int, int] = {}
        for proposal in proposals:
            option = int(proposed_option[proposal])
            delay = int(dataset.option_delay_seconds[option])
            entry = int(dataset.proposal_entry_time_ms[proposal]) + delay * 1000
            if entry < next_available:
                overlap_rejections += 1
                continue
            day = int(dataset.proposal_day[proposal])
            used = day_counts.get(day, 0)
            if used >= MAXIMUM_ENTRIES_PER_SYMBOL_DAY:
                capacity_rejections += 1
                continue
            selected_proposals.append(int(proposal))
            selected_options.append(option)
            day_counts[day] = used + 1
            next_available = entry + HORIZON_SECONDS * 1000
    if not selected_proposals:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            overlap_rejections,
            capacity_rejections,
        )
    proposals_array = np.asarray(selected_proposals, dtype=np.int64)
    options_array = np.asarray(selected_options, dtype=np.int64)
    actual_entry_time_ms = dataset.proposal_entry_time_ms[proposals_array] + (
        dataset.option_delay_seconds[options_array].astype(np.int64) * 1000
    )
    order = np.argsort(actual_entry_time_ms, kind="stable")
    return (
        proposals_array[order],
        options_array[order],
        overlap_rejections,
        capacity_rejections,
    )


def _replay_proposed_options(
    dataset: TimingDataset,
    *,
    role_mask: np.ndarray,
    proposed_option: np.ndarray,
    outcomes: np.ndarray,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> ReplayEvidence:
    selected_proposals, selected_options, overlap, capacity = _capacity_selection(
        dataset, role_mask=role_mask, proposed_option=proposed_option
    )
    role_count = int(np.count_nonzero(role_mask))
    eligible = int(np.count_nonzero(role_mask & (proposed_option >= 0)))
    metrics = _selection_metrics(
        dataset,
        proposal_indices=selected_proposals,
        option_indices=selected_options,
        outcomes=outcomes,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    return ReplayEvidence(
        metrics=metrics,
        role_proposals=role_count,
        eligible_proposals=eligible,
        vetoed_proposals=role_count - eligible,
        veto_fraction=(role_count - eligible) / role_count if role_count else 1.0,
        overlap_rejections=overlap,
        capacity_rejections=capacity,
        selected_proposal_indices=tuple(int(value) for value in selected_proposals),
        selected_option_indices=tuple(int(value) for value in selected_options),
    )


def _threshold_options(
    dataset: TimingDataset,
    *,
    role_mask: np.ndarray,
    positive_probability: np.ndarray,
    expected_net: np.ndarray,
    lower_quartile: np.ndarray,
    probability_threshold: float,
    expected_threshold: float,
    lower_threshold: float,
) -> np.ndarray:
    proposed = np.full(dataset.proposals, -1, dtype=np.int64)
    for proposal in np.flatnonzero(role_mask):
        start = int(proposal) * len(DELAYS_SECONDS)
        options = np.arange(start, start + len(DELAYS_SECONDS), dtype=np.int64)
        eligible = (
            (positive_probability[options] >= probability_threshold)
            & (expected_net[options] >= expected_threshold)
            & (lower_quartile[options] >= lower_threshold)
        )
        if np.any(eligible):
            eligible_options = options[eligible]
            proposed[proposal] = int(
                eligible_options[np.argmax(expected_net[eligible_options])]
            )
    return proposed


def _immediate_options(dataset: TimingDataset, role_mask: np.ndarray) -> np.ndarray:
    proposed = np.full(dataset.proposals, -1, dtype=np.int64)
    proposals = np.flatnonzero(role_mask)
    proposed[proposals] = proposals * len(DELAYS_SECONDS)
    return proposed


def _oracle_options(dataset: TimingDataset, role_mask: np.ndarray) -> np.ndarray:
    proposed = np.full(dataset.proposals, -1, dtype=np.int64)
    for proposal in np.flatnonzero(role_mask):
        start = int(proposal) * len(DELAYS_SECONDS)
        options = np.arange(start, start + len(DELAYS_SECONDS), dtype=np.int64)
        proposed[proposal] = int(
            options[np.argmax(dataset.option_base_net_bps[options])]
        )
    return proposed


def _cell_support(replay: ReplayEvidence) -> bool:
    metrics = replay.metrics
    return bool(
        metrics.total_trades >= 15
        and metrics.maximum_single_symbol_fraction <= 0.60
        and all(metrics.trades_by_symbol[symbol] >= 2 for symbol in SYMBOLS)
        and replay.veto_fraction <= 0.90
        and metrics.nonzero_delay_fraction >= 0.10
    )


def _cell_economic(base: ReplayEvidence, stress: ReplayEvidence) -> bool:
    metrics = base.metrics
    stress_metrics = stress.metrics
    return bool(
        _cell_support(base)
        and metrics.mean_net_bps > 0.0
        and metrics.profit_factor is not None
        and metrics.profit_factor >= 1.0
        and stress_metrics.mean_net_bps > 0.0
        and metrics.mean_increment_over_matched_immediate_bps >= 0.5
    )


def _threshold_key(item: Mapping[str, object]) -> tuple[float, ...]:
    base = item["timing_base"]["metrics"]
    stress = item["timing_stress"]["metrics"]
    return (
        float(stress["mean_net_bps"]),
        float(base["mean_increment_over_matched_immediate_bps"]),
        float(base["mean_net_bps"]),
        float(base["total_trades"]),
        float(item["positive_probability_threshold"]),
        float(item["expected_net_bps_threshold"]),
        float(item["lower_quartile_bps_threshold"]),
    )


def _evaluate_fold(
    dataset: TimingDataset,
    *,
    fold: Round42Fold,
    positive_probability: np.ndarray,
    expected_net: np.ndarray,
    lower_quartile: np.ndarray,
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    calibration_role = _proposal_role_mask(dataset, (fold.calibration_day,))
    traces: list[dict[str, object]] = []
    for probability_threshold in PROBABILITY_GRID:
        for expected_threshold in EXPECTED_NET_GRID:
            for lower_threshold in LOWER_QUARTILE_GRID:
                proposed = _threshold_options(
                    dataset,
                    role_mask=calibration_role,
                    positive_probability=positive_probability,
                    expected_net=expected_net,
                    lower_quartile=lower_quartile,
                    probability_threshold=probability_threshold,
                    expected_threshold=expected_threshold,
                    lower_threshold=lower_threshold,
                )
                base = _replay_proposed_options(
                    dataset,
                    role_mask=calibration_role,
                    proposed_option=proposed,
                    outcomes=dataset.option_base_net_bps,
                    bootstrap_samples=0,
                    bootstrap_seed=4211,
                )
                stress = _replay_proposed_options(
                    dataset,
                    role_mask=calibration_role,
                    proposed_option=proposed,
                    outcomes=dataset.option_stress_net_bps,
                    bootstrap_samples=0,
                    bootstrap_seed=4211,
                )
                traces.append(
                    {
                        "positive_probability_threshold": probability_threshold,
                        "expected_net_bps_threshold": expected_threshold,
                        "lower_quartile_bps_threshold": lower_threshold,
                        "support_passed": _cell_support(base),
                        "economic_gate_passed": _cell_economic(base, stress),
                        "timing_base": base.asdict(),
                        "timing_stress": stress.asdict(),
                    }
                )
    eligible = [item for item in traces if item["economic_gate_passed"]]
    selected = max(eligible, key=_threshold_key) if eligible else None
    evaluation_role = _proposal_role_mask(dataset, (fold.evaluation_day,))
    if selected is None:
        proposed = np.full(dataset.proposals, -1, dtype=np.int64)
    else:
        proposed = _threshold_options(
            dataset,
            role_mask=evaluation_role,
            positive_probability=positive_probability,
            expected_net=expected_net,
            lower_quartile=lower_quartile,
            probability_threshold=float(selected["positive_probability_threshold"]),
            expected_threshold=float(selected["expected_net_bps_threshold"]),
            lower_threshold=float(selected["lower_quartile_bps_threshold"]),
        )
    timing_base = _replay_proposed_options(
        dataset,
        role_mask=evaluation_role,
        proposed_option=proposed,
        outcomes=dataset.option_base_net_bps,
        bootstrap_samples=500 if selected else 0,
        bootstrap_seed=4212,
    )
    timing_stress = _replay_proposed_options(
        dataset,
        role_mask=evaluation_role,
        proposed_option=proposed,
        outcomes=dataset.option_stress_net_bps,
        bootstrap_samples=500 if selected else 0,
        bootstrap_seed=4212,
    )
    immediate = _replay_proposed_options(
        dataset,
        role_mask=evaluation_role,
        proposed_option=_immediate_options(dataset, evaluation_role),
        outcomes=dataset.option_base_net_bps,
        bootstrap_samples=500,
        bootstrap_seed=4213,
    )
    veto_only_proposed = np.full(dataset.proposals, -1, dtype=np.int64)
    eligible_proposals = np.flatnonzero(evaluation_role & (proposed >= 0))
    veto_only_proposed[eligible_proposals] = eligible_proposals * len(DELAYS_SECONDS)
    veto_only = _replay_proposed_options(
        dataset,
        role_mask=evaluation_role,
        proposed_option=veto_only_proposed,
        outcomes=dataset.option_base_net_bps,
        bootstrap_samples=500 if selected else 0,
        bootstrap_seed=4214,
    )
    oracle = _replay_proposed_options(
        dataset,
        role_mask=evaluation_role,
        proposed_option=_oracle_options(dataset, evaluation_role),
        outcomes=dataset.option_base_net_bps,
        bootstrap_samples=0,
        bootstrap_seed=4215,
    )
    return {
        "fold_id": fold.fold_id,
        "schedule": asdict(fold),
        "role_proposals": {
            "fit": int(np.count_nonzero(_proposal_role_mask(dataset, fold.fit_days))),
            "early_stop": int(
                np.count_nonzero(_proposal_role_mask(dataset, (fold.early_stop_day,)))
            ),
            "threshold_calibration": int(np.count_nonzero(calibration_role)),
            "evaluation": int(np.count_nonzero(evaluation_role)),
        },
        "model_diagnostics": dict(diagnostics),
        "threshold_trace": traces,
        "selected_threshold": (
            {
                key: selected[key]
                for key in (
                    "positive_probability_threshold",
                    "expected_net_bps_threshold",
                    "lower_quartile_bps_threshold",
                )
            }
            if selected
            else None
        ),
        "calibration_selection": selected,
        "evaluation": {
            "timing_base": timing_base.asdict(),
            "timing_stress": timing_stress.asdict(),
            "immediate_base": immediate.asdict(),
            "veto_only_base": veto_only.asdict(),
            "oracle_best_delay_diagnostic": oracle.asdict(),
        },
    }


def _combined_metrics_from_folds(
    dataset: TimingDataset,
    folds: Sequence[Mapping[str, object]],
    *,
    evidence_name: str,
    outcomes: np.ndarray,
    bootstrap_seed: int,
) -> ReplayMetrics:
    proposals: list[int] = []
    options: list[int] = []
    for fold in folds:
        evidence = fold["evaluation"][evidence_name]
        proposals.extend(evidence["selected_proposal_indices"])
        options.extend(evidence["selected_option_indices"])
    return _selection_metrics(
        dataset,
        proposal_indices=np.asarray(proposals, dtype=np.int64),
        option_indices=np.asarray(options, dtype=np.int64),
        outcomes=outcomes,
        bootstrap_samples=1000 if proposals else 0,
        bootstrap_seed=bootstrap_seed,
    )


def _pilot_gate_reasons(
    folds: Sequence[Mapping[str, object]],
    aggregate_base: ReplayMetrics,
    aggregate_stress: ReplayMetrics,
) -> tuple[str, ...]:
    reasons: list[str] = []
    selected_folds = [fold for fold in folds if fold["selected_threshold"] is not None]
    if len(selected_folds) != len(FOLDS):
        reasons.append("selected_threshold_folds<2")
    if aggregate_base.total_trades < 40:
        reasons.append("aggregate_trades<40")
    if any(
        fold["evaluation"]["timing_base"]["metrics"]["total_trades"] < 15
        for fold in folds
    ):
        reasons.append("evaluation_day_trades<15")
    if any(aggregate_base.trades_by_symbol[symbol] < 8 for symbol in SYMBOLS):
        reasons.append("aggregate_symbol_trades<8")
    if aggregate_base.maximum_single_symbol_fraction > 0.50:
        reasons.append("maximum_single_symbol_fraction>0.50")
    if any(fold["evaluation"]["timing_base"]["veto_fraction"] > 0.90 for fold in folds):
        reasons.append("veto_fraction>0.90")
    if aggregate_base.nonzero_delay_fraction < 0.10:
        reasons.append("nonzero_delay_fraction<0.10")
    if aggregate_base.mean_net_bps <= 0.0:
        reasons.append("aggregate_base_mean_net_bps<=0")
    if aggregate_base.profit_factor is None or aggregate_base.profit_factor < 1.10:
        reasons.append("aggregate_base_profit_factor<1.10")
    if any(
        fold["evaluation"]["timing_base"]["metrics"]["mean_net_bps"] <= 0.0
        for fold in folds
    ):
        reasons.append("evaluation_day_mean_net_bps<=0")
    if aggregate_stress.mean_net_bps <= 0.0:
        reasons.append("aggregate_stress_mean_net_bps<=0")
    if aggregate_base.mean_increment_over_matched_immediate_bps < 1.0:
        reasons.append("mean_increment_over_matched_immediate_bps<1.0")
    lower = aggregate_base.day_block_bootstrap_mean_net_bps_lower_95
    if lower is None or lower <= 0.0:
        reasons.append("day_block_bootstrap_lower_95<=0")
    return tuple(reasons)


def run_round42_screen(
    dataset: TimingDataset,
    *,
    model_dir: Path,
    compute_backend: str,
    progress: ProgressCallback | None = None,
) -> Round42Screen:
    _validate_timing_dataset(dataset)
    fold_results: list[Mapping[str, object]] = []
    artifacts: list[TimingModelArtifact] = []
    kinds: set[str] = set()
    devices: set[str] = set()
    for fold_index, fold in enumerate(FOLDS):
        predictions: dict[str, np.ndarray] = {}
        diagnostics: dict[str, object] = {}
        for head_index, head in enumerate(
            (
                "positive_net_probability",
                "robust_expected_net_utility",
                "lower_quartile_net_utility",
            )
        ):
            predicted, artifact, head_diagnostics = _fit_head(
                dataset,
                fold=fold,
                head=head,
                model_dir=model_dir,
                compute_backend=compute_backend,
                seed=SEED + fold_index * 10 + head_index,
                progress=progress,
            )
            predictions[head] = predicted
            diagnostics[head] = head_diagnostics
            artifacts.append(artifact)
            kinds.add(artifact.backend_kind)
            devices.add(artifact.backend_device)
        fold_result = _evaluate_fold(
            dataset,
            fold=fold,
            positive_probability=predictions["positive_net_probability"],
            expected_net=predictions["robust_expected_net_utility"],
            lower_quartile=predictions["lower_quartile_net_utility"],
            diagnostics=diagnostics,
        )
        fold_results.append(fold_result)
        if progress is not None:
            evaluation = fold_result["evaluation"]["timing_base"]["metrics"]
            progress(
                "round42_fold_evaluation",
                {
                    "status": "complete",
                    "fold_id": fold.fold_id,
                    "threshold_selected": fold_result["selected_threshold"] is not None,
                    "trades": evaluation["total_trades"],
                    "mean_net_bps": evaluation["mean_net_bps"],
                },
            )
    if len(kinds) != 1 or len(devices) != 1:
        raise RuntimeError("Round 42 model backends are inconsistent")
    aggregate_base = _combined_metrics_from_folds(
        dataset,
        fold_results,
        evidence_name="timing_base",
        outcomes=dataset.option_base_net_bps,
        bootstrap_seed=4221,
    )
    aggregate_stress = _combined_metrics_from_folds(
        dataset,
        fold_results,
        evidence_name="timing_stress",
        outcomes=dataset.option_stress_net_bps,
        bootstrap_seed=4221,
    )
    aggregate_immediate = _combined_metrics_from_folds(
        dataset,
        fold_results,
        evidence_name="immediate_base",
        outcomes=dataset.option_base_net_bps,
        bootstrap_seed=4222,
    )
    aggregate_veto_only = _combined_metrics_from_folds(
        dataset,
        fold_results,
        evidence_name="veto_only_base",
        outcomes=dataset.option_base_net_bps,
        bootstrap_seed=4223,
    )
    aggregate_oracle = _combined_metrics_from_folds(
        dataset,
        fold_results,
        evidence_name="oracle_best_delay_diagnostic",
        outcomes=dataset.option_base_net_bps,
        bootstrap_seed=4224,
    )
    reasons = _pilot_gate_reasons(fold_results, aggregate_base, aggregate_stress)
    return Round42Screen(
        folds=tuple(fold_results),
        aggregate={
            "timing_base": aggregate_base.asdict(),
            "timing_stress": aggregate_stress.asdict(),
            "immediate_base": aggregate_immediate.asdict(),
            "veto_only_base": aggregate_veto_only.asdict(),
            "oracle_best_delay_diagnostic": aggregate_oracle.asdict(),
        },
        model_artifacts=tuple(artifacts),
        pilot_gate_passed=not reasons,
        pilot_gate_reasons=reasons,
        backend_kind=next(iter(kinds)),
        backend_device=next(iter(devices)),
    )


__all__ = [
    "BASE_CHARGE_BPS",
    "DELAYS_SECONDS",
    "EXPECTED_NET_GRID",
    "FOLDS",
    "HORIZON_SECONDS",
    "LOWER_QUARTILE_GRID",
    "MAXIMUM_ENTRIES_PER_SYMBOL_DAY",
    "PRIMARY_MARGIN",
    "PROBABILITY_GRID",
    "ROUND",
    "STRESS_CHARGE_BPS",
    "TimingDataset",
    "TimingModelArtifact",
    "build_timing_dataset",
    "run_round42_screen",
]
