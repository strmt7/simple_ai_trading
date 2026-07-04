"""Deterministic market-regime evidence for model validation artifacts."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class MarketRegimeEvidence:
    dominant_regime: str
    confidence: float
    rows: int
    start_timestamp: int | None
    end_timestamp: int | None
    trend_return: float
    realized_volatility: float
    mean_abs_return: float
    direction_consistency: float
    reversal_rate: float
    autocorrelation_1: float
    volume_zscore: float | None
    notes: tuple[str, ...] = ()

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def _finite(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed if math.isfinite(parsed) else default


def _maybe_attr(row: object, *names: str) -> object:
    for name in names:
        if hasattr(row, name):
            return getattr(row, name)
        if isinstance(row, Mapping) and name in row:
            return row[name]
    return None


def _timestamp(row: object) -> int | None:
    value = _maybe_attr(row, "timestamp", "open_time", "close_time")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _stdev(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if len(clean) < 2:
        return 0.0
    average = sum(clean) / len(clean)
    variance = sum((value - average) ** 2 for value in clean) / (len(clean) - 1)
    return math.sqrt(max(0.0, variance))


def _autocorrelation_1(values: Sequence[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if len(clean) < 3:
        return 0.0
    left = clean[:-1]
    right = clean[1:]
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    covariance = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right, strict=True))
    left_var = sum((a - left_mean) ** 2 for a in left)
    right_var = sum((b - right_mean) ** 2 for b in right)
    denominator = math.sqrt(max(0.0, left_var * right_var))
    if denominator <= 1e-12:
        return 0.0
    return max(-1.0, min(1.0, covariance / denominator))


def _volume_zscore(rows: Sequence[object]) -> float | None:
    volumes = [
        _finite(_maybe_attr(row, "volume"), default=float("nan"))
        for row in rows
        if _maybe_attr(row, "volume") is not None
    ]
    volumes = [value for value in volumes if math.isfinite(value) and value >= 0.0]
    if len(volumes) < 3:
        return None
    baseline = volumes[:-1]
    spread = _stdev(baseline)
    if spread <= 1e-12:
        return 0.0
    return (volumes[-1] - (sum(baseline) / len(baseline))) / spread


def _sign(value: float) -> int:
    if value > 0.0:
        return 1
    if value < 0.0:
        return -1
    return 0


def classify_market_regime(rows: Sequence[object]) -> MarketRegimeEvidence:
    """Classify a chronological candle/model-row slice into a coarse regime."""

    row_list = list(rows)
    timestamps = [_timestamp(row) for row in row_list]
    closes = [
        _finite(_maybe_attr(row, "close"), default=float("nan"))
        for row in row_list
        if _maybe_attr(row, "close") is not None
    ]
    closes = [value for value in closes if math.isfinite(value) and value > 0.0]
    if len(closes) < 3:
        return MarketRegimeEvidence(
            dominant_regime="insufficient_data",
            confidence=0.0,
            rows=len(row_list),
            start_timestamp=next((value for value in timestamps if value is not None), None),
            end_timestamp=next((value for value in reversed(timestamps) if value is not None), None),
            trend_return=0.0,
            realized_volatility=0.0,
            mean_abs_return=0.0,
            direction_consistency=0.0,
            reversal_rate=0.0,
            autocorrelation_1=0.0,
            volume_zscore=_volume_zscore(row_list),
            notes=("fewer_than_three_valid_closes",),
        )

    returns = [
        (closes[index] / closes[index - 1]) - 1.0
        for index in range(1, len(closes))
        if closes[index - 1] > 0.0
    ]
    returns = [value for value in returns if math.isfinite(value)]
    if not returns:
        return MarketRegimeEvidence(
            dominant_regime="insufficient_data",
            confidence=0.0,
            rows=len(row_list),
            start_timestamp=next((value for value in timestamps if value is not None), None),
            end_timestamp=next((value for value in reversed(timestamps) if value is not None), None),
            trend_return=0.0,
            realized_volatility=0.0,
            mean_abs_return=0.0,
            direction_consistency=0.0,
            reversal_rate=0.0,
            autocorrelation_1=0.0,
            volume_zscore=_volume_zscore(row_list),
            notes=("no_valid_returns",),
        )

    trend_return = (closes[-1] / closes[0]) - 1.0
    volatility = _stdev(returns)
    mean_abs = sum(abs(value) for value in returns) / len(returns)
    signs = [_sign(value) for value in returns if _sign(value) != 0]
    positive = sum(1 for value in signs if value > 0)
    negative = sum(1 for value in signs if value < 0)
    direction_consistency = max(positive, negative) / len(signs) if signs else 0.0
    reversals = sum(1 for left, right in zip(signs, signs[1:], strict=False) if left != right)
    reversal_rate = reversals / max(1, len(signs) - 1) if signs else 0.0
    autocorrelation = _autocorrelation_1(returns)
    noise_floor = max(1e-9, mean_abs * math.sqrt(len(returns)), volatility * math.sqrt(len(returns)))
    trend_strength = abs(trend_return) / noise_floor
    volatility_floor = max(0.0005, mean_abs * 1.8)
    notes: list[str] = []

    if volatility >= volatility_floor and reversal_rate >= 0.55:
        regime = "volatile_chop"
        confidence = min(1.0, 0.45 + 0.35 * reversal_rate + 0.20 * min(2.0, volatility / volatility_floor) / 2.0)
    elif trend_strength >= 1.15 and direction_consistency >= 0.55:
        regime = "trend_up" if trend_return > 0.0 else "trend_down"
        confidence = min(1.0, 0.40 + 0.35 * min(2.0, trend_strength) / 2.0 + 0.25 * direction_consistency)
    elif reversal_rate >= 0.50 and trend_strength < 0.85:
        regime = "range_bound"
        confidence = min(1.0, 0.45 + 0.35 * reversal_rate + 0.20 * (1.0 - min(1.0, trend_strength)))
    elif abs(autocorrelation) >= 0.35:
        regime = "serial_correlation"
        confidence = min(1.0, 0.50 + 0.50 * abs(autocorrelation))
    else:
        regime = "mixed"
        confidence = max(0.20, min(0.70, 0.35 + 0.20 * direction_consistency + 0.15 * min(1.0, trend_strength)))
        notes.append("low_regime_separation")

    if mean_abs <= 1e-9:
        notes.append("flat_returns")
    if len(returns) < 10:
        notes.append("short_window")

    return MarketRegimeEvidence(
        dominant_regime=regime,
        confidence=float(confidence),
        rows=len(row_list),
        start_timestamp=next((value for value in timestamps if value is not None), None),
        end_timestamp=next((value for value in reversed(timestamps) if value is not None), None),
        trend_return=float(trend_return),
        realized_volatility=float(volatility),
        mean_abs_return=float(mean_abs),
        direction_consistency=float(direction_consistency),
        reversal_rate=float(reversal_rate),
        autocorrelation_1=float(autocorrelation),
        volume_zscore=_volume_zscore(row_list),
        notes=tuple(notes),
    )


def summarize_regime_windows(
    windows: Sequence[Mapping[str, object]],
    *,
    overall_regime: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Aggregate accepted-rate and P&L evidence by window regime."""

    buckets: dict[str, dict[str, object]] = {}
    for item in windows:
        regime_payload = item.get("regime")
        if not isinstance(regime_payload, Mapping):
            continue
        regime = str(regime_payload.get("dominant_regime") or "unknown")
        result = item.get("result")
        result_map = result if isinstance(result, Mapping) else {}
        bucket = buckets.setdefault(
            regime,
            {
                "windows": 0,
                "accepted_windows": 0,
                "realized_pnl": 0.0,
                "worst_max_drawdown": 0.0,
                "profit_factor_sum": 0.0,
                "expectancy_sum": 0.0,
                "confidence_sum": 0.0,
            },
        )
        bucket["windows"] = int(bucket["windows"]) + 1
        bucket["accepted_windows"] = int(bucket["accepted_windows"]) + (1 if bool(item.get("accepted")) else 0)
        bucket["realized_pnl"] = float(bucket["realized_pnl"]) + _finite(result_map.get("realized_pnl"))
        bucket["worst_max_drawdown"] = max(float(bucket["worst_max_drawdown"]), _finite(result_map.get("max_drawdown")))
        bucket["profit_factor_sum"] = float(bucket["profit_factor_sum"]) + _finite(result_map.get("profit_factor"))
        bucket["expectancy_sum"] = float(bucket["expectancy_sum"]) + _finite(result_map.get("expectancy"))
        bucket["confidence_sum"] = float(bucket["confidence_sum"]) + _finite(regime_payload.get("confidence"))

    by_regime: dict[str, dict[str, object]] = {}
    for regime, bucket in sorted(buckets.items()):
        windows_count = max(1, int(bucket["windows"]))
        accepted = int(bucket["accepted_windows"])
        by_regime[regime] = {
            "windows": windows_count,
            "accepted_windows": accepted,
            "accepted_rate": accepted / windows_count,
            "realized_pnl": float(bucket["realized_pnl"]),
            "mean_realized_pnl": float(bucket["realized_pnl"]) / windows_count,
            "worst_max_drawdown": float(bucket["worst_max_drawdown"]),
            "mean_profit_factor": float(bucket["profit_factor_sum"]) / windows_count,
            "mean_expectancy": float(bucket["expectancy_sum"]) / windows_count,
            "mean_confidence": float(bucket["confidence_sum"]) / windows_count,
        }

    window_count = sum(int(item["windows"]) for item in by_regime.values())
    dominant_regime = ""
    dominant_share = 0.0
    if by_regime:
        dominant_regime, dominant = max(by_regime.items(), key=lambda entry: int(entry[1]["windows"]))
        dominant_share = int(dominant["windows"]) / max(1, window_count)
    accepted_regimes = [
        regime
        for regime, payload in by_regime.items()
        if int(payload["accepted_windows"]) > 0
    ]
    notes: list[str] = []
    concentration_warning = bool(window_count >= 3 and dominant_share >= 0.75)
    if concentration_warning:
        notes.append("window_regime_concentration")
    if len(accepted_regimes) <= 1 and window_count >= 3:
        notes.append("accepted_windows_lack_regime_diversity")
    return {
        "overall": dict(overall_regime or {}),
        "window_count": window_count,
        "regime_count": len(by_regime),
        "dominant_regime": dominant_regime,
        "dominant_regime_window_share": float(dominant_share),
        "accepted_regime_count": len(accepted_regimes),
        "accepted_regimes": accepted_regimes,
        "concentration_warning": concentration_warning,
        "by_regime": by_regime,
        "notes": notes,
    }
