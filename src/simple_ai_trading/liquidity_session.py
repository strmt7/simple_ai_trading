"""Data-probed liquidity/session guards shared by backtest and live paths."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Sequence

from .features import ModelRow
from .meta_label import MetaLabelDecision
from .types import StrategyConfig


@dataclass(frozen=True)
class LiquiditySessionAdjustment:
    """Threshold and size adjustment derived only from observed market data."""

    threshold: float
    size_multiplier: float
    low_liquidity: bool
    low_dynamic_session: bool

    @property
    def active(self) -> bool:
        return bool(self.low_liquidity or self.low_dynamic_session)

    @property
    def reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.low_liquidity:
            reasons.append("low_liquidity_requires_stronger_signal_and_smaller_size")
        if self.low_dynamic_session:
            reasons.append("data_probed_liquidity_session_below_history")
        return tuple(reasons)


def _clamp_threshold(value: float) -> float:
    if value != value:
        return 0.5
    return max(0.0, min(1.0, float(value)))


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[midpoint])
    return float((ordered[midpoint - 1] + ordered[midpoint]) / 2.0)


def _utc_liquidity_bucket(timestamp_ms: int, bucket_minutes: int) -> tuple[int, int, int] | None:
    try:
        instant = datetime.fromtimestamp(int(timestamp_ms) / 1000.0, tz=timezone.utc)
    except (OSError, ValueError, OverflowError):
        return None
    bucket = max(1, min(60, int(bucket_minutes)))
    minute_bucket = instant.minute // bucket
    return instant.weekday(), instant.hour, minute_bucket


def liquidity_session_adjustment(
    rows: Sequence[ModelRow],
    index: int,
    cfg: StrategyConfig,
    base_threshold: float,
) -> LiquiditySessionAdjustment:
    """Return threshold and sizing changes from trailing per-symbol liquidity.

    The function intentionally avoids static "market hours". It compares the
    current row with prior bars in the same data stream, including the same UTC
    weekday/hour/minute bucket when enough history exists.
    """

    if not bool(getattr(cfg, "liquidity_risk_enabled", True)):
        return LiquiditySessionAdjustment(_clamp_threshold(base_threshold), 1.0, False, False)
    lookback = max(8, int(getattr(cfg, "liquidity_lookback_bars", 96)))
    if index < lookback or index < 0 or index >= len(rows):
        return LiquiditySessionAdjustment(_clamp_threshold(base_threshold), 1.0, False, False)
    start = index - lookback
    trailing = rows[start:index]
    volumes = [max(0.0, float(getattr(row, "volume", 0.0) or 0.0)) for row in trailing]
    current_volume = max(0.0, float(getattr(rows[index], "volume", 0.0) or 0.0))
    if not volumes or max(volumes) <= 0.0:
        low_liquidity = False
    else:
        median_volume = _median(volumes)
        low_liquidity = bool(
            median_volume > 0.0
            and current_volume < median_volume * float(getattr(cfg, "low_liquidity_volume_ratio", 0.35))
        )

    low_dynamic_session = False
    if bool(getattr(cfg, "dynamic_liquidity_session_enabled", True)):
        bucket_minutes = int(getattr(cfg, "dynamic_liquidity_bucket_minutes", 15))
        current_bucket = _utc_liquidity_bucket(int(rows[index].timestamp), bucket_minutes)
        if current_bucket is not None:
            bucket_volumes = [
                max(0.0, float(getattr(row, "volume", 0.0) or 0.0))
                for row in trailing
                if _utc_liquidity_bucket(int(row.timestamp), bucket_minutes) == current_bucket
            ]
            min_samples = int(getattr(cfg, "dynamic_liquidity_session_min_samples", 8))
            if len(bucket_volumes) >= min_samples and max(bucket_volumes) > 0.0:
                bucket_median = _median(bucket_volumes)
                low_dynamic_session = bool(
                    bucket_median > 0.0
                    and current_volume < bucket_median * float(getattr(cfg, "low_session_liquidity_volume_ratio", 0.45))
                )

    threshold = float(base_threshold)
    size_multiplier = 1.0
    if low_liquidity:
        threshold += float(getattr(cfg, "low_liquidity_signal_threshold_add", 0.04))
        size_multiplier *= float(getattr(cfg, "low_liquidity_size_multiplier", 0.50))
    if low_dynamic_session:
        threshold += float(getattr(cfg, "low_session_signal_threshold_add", 0.01))
        size_multiplier *= float(getattr(cfg, "low_session_size_multiplier", 0.85))
    return LiquiditySessionAdjustment(
        threshold=_clamp_threshold(threshold),
        size_multiplier=max(0.0, min(1.0, size_multiplier)),
        low_liquidity=bool(low_liquidity),
        low_dynamic_session=bool(low_dynamic_session),
    )


def apply_liquidity_session_meta(
    base: MetaLabelDecision,
    adjustment: LiquiditySessionAdjustment,
) -> MetaLabelDecision:
    """Combine a meta-label decision with observed liquidity/session risk."""

    if not adjustment.active:
        return base
    adjusted_multiplier = max(0.0, min(1.0, float(base.size_multiplier) * float(adjustment.size_multiplier)))
    action = base.action
    if adjusted_multiplier <= 0.0:
        action = "skip"
    elif adjusted_multiplier < float(base.size_multiplier) and action == "take":
        action = "downsize"
    reasons = [base.reason, *adjustment.reasons]
    return replace(
        base,
        enabled=True,
        action=action,
        size_multiplier=adjusted_multiplier,
        reason="; ".join(reason for reason in reasons if reason),
    )


__all__ = [
    "LiquiditySessionAdjustment",
    "apply_liquidity_session_meta",
    "liquidity_session_adjustment",
]
