"""Richer feature engineering and training pipeline on top of the base model.

The core model (`.model.TrainedModel`) is a 13-feature logistic regression.
This module keeps the same math but expands the feature vector with:

* Non-linear transforms (`tanh`, `log1p`) of the base features.
* Polynomial pairwise interactions among the top-K base features.
* Multi-window SMA / RSI / volatility snapshots anchored at configurable extra
  lookbacks so the model can see short, medium, and long regimes in one row.

Everything remains pure stdlib — no numpy, no sklearn — so the ``TrainedModel``
serializer already in the repo can persist the expanded model without changes.
The expansion parameters are deterministic from the strategy config alone, so
inference at test / live time recomputes the same feature vector every call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .api import Candle
from .features import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    ModelRow,
    feature_signature as base_feature_signature,
    make_inference_rows as make_base_inference_rows,
    make_rows as make_base_rows,
    normalize_enabled_features,
)
from .market_data import clean_candles
from .model import TrainedModel, ensemble_member_from_model, train as train_logistic

ADVANCED_FEATURE_VERSION = "v4-advanced-confluence"
_EXTRA_FEATURES_PER_WINDOW = 7
_CONFLUENCE_FEATURES_PER_WINDOW = 9


@dataclass(frozen=True)
class AdvancedFeatureConfig:
    """Parameters that drive feature expansion and must match at inference.

    All fields are part of the model's feature signature — changing any of
    them forces a retrain, which is correct because the feature space itself
    has changed.
    """

    base_features: tuple[str, ...]
    polynomial_degree: int = 2
    polynomial_top_features: int = 6
    extra_lookback_windows: tuple[int, ...] = (5, 20, 60)
    confluence_windows: tuple[int, ...] = ()
    nonlinear_transforms: tuple[str, ...] = ("tanh", "log1p")
    short_window: int = 10
    long_window: int = 40
    label_threshold: float = 0.001
    label_lookahead: int = 4
    label_mode: str = "forward_return"
    label_stop_threshold: float | None = None


@dataclass(frozen=True)
class FeatureGroupSpan:
    """Contiguous feature-vector group used for ablation reports."""

    name: str
    start: int
    end: int

    @property
    def size(self) -> int:
        return max(0, int(self.end) - int(self.start))

    def asdict(self) -> dict[str, int | str]:
        return {"name": self.name, "start": int(self.start), "end": int(self.end), "size": self.size}


def _tanh(x: float) -> float:
    try:
        return math.tanh(x)
    except (OverflowError, ValueError):
        return 1.0 if x > 0 else -1.0


def _log1p_signed(x: float) -> float:
    return math.copysign(math.log1p(abs(x)), x) if x != 0.0 else 0.0


def _sma(values: Sequence[float], window: int) -> float:
    if window <= 0 or len(values) < window:
        return float("nan")
    return sum(values[-window:]) / float(window)


def _rsi(values: Sequence[float], window: int) -> float:
    if window <= 0 or len(values) < window + 1:
        return float("nan")
    gains: list[float] = []
    losses: list[float] = []
    for i in range(len(values) - window, len(values)):
        delta = values[i] - values[i - 1]
        gains.append(max(0.0, delta))
        losses.append(max(0.0, -delta))
    avg_gain = sum(gains) / len(gains)
    avg_loss = sum(losses) / len(losses)
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def _volatility(values: Sequence[float], window: int) -> float:
    if window <= 1 or len(values) < window:
        return float("nan")
    recent = values[-window:]
    mean = sum(recent) / len(recent)
    variance = sum((v - mean) ** 2 for v in recent) / max(1, len(recent) - 1)
    return math.sqrt(max(0.0, variance))


def _safe(x: float) -> float:
    return 0.0 if not math.isfinite(x) else float(x)


def _normalized_label_mode(value: str) -> str:
    mode = str(value or "forward_return").strip().lower().replace("-", "_")
    if mode in {"triple", "triple_barrier", "barrier"}:
        return "triple_barrier"
    return "forward_return"


def _triple_barrier_label(
    candles: Sequence[Candle],
    index: int,
    *,
    horizon: int,
    take_profit_pct: float,
    stop_loss_pct: float,
) -> int:
    """Return 1 only when take-profit is hit before stop-loss inside horizon."""

    if index < 0 or index >= len(candles):
        return 0
    entry = float(candles[index].close)
    if entry <= 0.0 or not math.isfinite(entry):
        return 0
    take = max(0.0, float(take_profit_pct))
    stop = max(0.0, float(stop_loss_pct))
    if take <= 0.0:
        return 0
    end = min(len(candles) - 1, index + max(1, int(horizon)))
    for next_index in range(index + 1, end + 1):
        candle = candles[next_index]
        high_return = _safe((float(candle.high) - entry) / entry)
        low_return = _safe((float(candle.low) - entry) / entry)
        hit_stop = stop > 0.0 and low_return <= -stop
        hit_take = high_return >= take
        if hit_stop and hit_take:
            return 0
        if hit_stop:
            return 0
        if hit_take:
            return 1
    final_return = _safe((float(candles[end].close) - entry) / entry)
    return 1 if final_return >= take else 0


def _prefix_sum(values: Sequence[float]) -> list[float]:
    total = 0.0
    prefix = [0.0]
    for value in values:
        total += value
        prefix.append(total)
    return prefix


def _window_sum(prefix: Sequence[float], start: int, end: int) -> float:
    return prefix[end + 1] - prefix[start]


@dataclass(frozen=True)
class _AdvancedWindowCache:
    closes: list[float]
    close_prefix: list[float]
    close_square_prefix: list[float]
    gain_prefix: list[float]
    loss_prefix: list[float]
    abs_move_prefix: list[float]


@dataclass(frozen=True)
class _ConfluenceCache:
    candles: list[Candle]
    opens: list[float]
    highs: list[float]
    lows: list[float]
    closes: list[float]
    volumes: list[float]
    close_prefix: list[float]
    volume_prefix: list[float]
    true_range_prefix: list[float]


def _build_window_cache(closes: Sequence[float]) -> _AdvancedWindowCache:
    close_values = [float(value) for value in closes]
    gains = [0.0]
    losses = [0.0]
    abs_moves = [0.0]
    for index in range(1, len(close_values)):
        delta = close_values[index] - close_values[index - 1]
        gains.append(max(0.0, delta))
        losses.append(max(0.0, -delta))
        abs_moves.append(abs(delta))
    return _AdvancedWindowCache(
        closes=close_values,
        close_prefix=_prefix_sum(close_values),
        close_square_prefix=_prefix_sum([value * value for value in close_values]),
        gain_prefix=_prefix_sum(gains),
        loss_prefix=_prefix_sum(losses),
        abs_move_prefix=_prefix_sum(abs_moves),
    )


def _window_mean(prefix: Sequence[float], start: int, end: int) -> float:
    if end < start:
        return float("nan")
    return _window_sum(prefix, start, end) / float(end - start + 1)


def _rsi_at(cache: _AdvancedWindowCache, end: int, window: int) -> float:
    if window <= 0 or end < window:
        return float("nan")
    start = end + 1 - window
    avg_gain = _window_mean(cache.gain_prefix, start, end)
    avg_loss = _window_mean(cache.loss_prefix, start, end)
    if avg_loss == 0:
        return 100.0
    return 100 - 100 / (1 + avg_gain / avg_loss)


def _volatility_at(cache: _AdvancedWindowCache, end: int, window: int) -> float:
    if window <= 1 or end < window - 1:
        return float("nan")
    start = end + 1 - window
    total = _window_sum(cache.close_prefix, start, end)
    total_sq = _window_sum(cache.close_square_prefix, start, end)
    variance = (total_sq - (total * total / window)) / max(1, window - 1)
    return math.sqrt(max(0.0, variance))


def _extra_window_features_at(cache: _AdvancedWindowCache, end: int, windows: Sequence[int]) -> list[float]:
    features: list[float] = []
    anchor = cache.closes[end] if cache.closes else 0.0
    for window in windows:
        start = end + 1 - window
        sma = _window_mean(cache.close_prefix, start, end) if end >= window - 1 else float("nan")
        rsi = _rsi_at(cache, end, window)
        vol = _volatility_at(cache, end, window)
        if end >= window - 1 and start >= 0:
            window_values = cache.closes[start:end + 1]
            first = window_values[0]
            high = max(window_values)
            low = min(window_values)
            path = _window_sum(cache.abs_move_prefix, start, end)
            net_move = abs(anchor - first)
        else:
            first = high = low = path = net_move = float("nan")
        features.extend([
            _safe((anchor - sma) / sma) if math.isfinite(sma) and sma != 0 else 0.0,
            _safe((rsi / 100.0) if math.isfinite(rsi) else 0.0),
            _safe(vol / anchor) if anchor != 0 else 0.0,
            _safe((anchor - first) / first) if math.isfinite(first) and first != 0 else 0.0,
            _safe((anchor - high) / high) if math.isfinite(high) and high != 0 else 0.0,
            _safe((anchor - low) / low) if math.isfinite(low) and low != 0 else 0.0,
            _safe((net_move / path) if math.isfinite(path) and path > 0.0 else 0.0),
        ])
    return features


def _extra_window_features(closes: Sequence[float], windows: Sequence[int]) -> list[float]:
    if not closes:
        return [0.0 for _ in range(_EXTRA_FEATURES_PER_WINDOW * len(windows))]
    return _extra_window_features_at(_build_window_cache(closes), len(closes) - 1, windows)


def _sign_vote(value: float, deadband: float = 0.0) -> float:
    if not math.isfinite(value) or abs(value) <= deadband:
        return 0.0
    return 1.0 if value > 0.0 else -1.0


def _build_confluence_cache(candles: Sequence[Candle]) -> _ConfluenceCache:
    values = list(candles)
    opens = [float(candle.open) for candle in values]
    highs = [float(candle.high) for candle in values]
    lows = [float(candle.low) for candle in values]
    closes = [float(candle.close) for candle in values]
    volumes = [float(candle.volume) for candle in values]
    true_ranges = [0.0]
    for index in range(1, len(values)):
        previous_close = closes[index - 1]
        high = highs[index]
        low = lows[index]
        true_ranges.append(max(high - low, abs(high - previous_close), abs(low - previous_close)))
    return _ConfluenceCache(
        candles=values,
        opens=opens,
        highs=highs,
        lows=lows,
        closes=closes,
        volumes=volumes,
        close_prefix=_prefix_sum(closes),
        volume_prefix=_prefix_sum(volumes),
        true_range_prefix=_prefix_sum(true_ranges),
    )


def _confluence_features_at(cache: _ConfluenceCache, end: int, windows: Sequence[int]) -> list[float]:
    """Return multi-timeframe technical-rating and candle microstructure proxies."""

    features: list[float] = []
    if not windows:
        return features
    if end < 0 or end >= len(cache.closes):
        return [0.0 for _ in range(_CONFLUENCE_FEATURES_PER_WINDOW * len(windows))]
    close = cache.closes[end]
    open_ = cache.opens[end]
    volume = cache.volumes[end]
    previous_close = cache.closes[end - 1] if end > 0 else close
    candle_range = max(0.0, cache.highs[end] - cache.lows[end])
    body_ratio = _safe((close - open_) / candle_range) if candle_range > 0.0 else 0.0
    close_location = _safe(((close - cache.lows[end]) / candle_range) - 0.5) if candle_range > 0.0 else 0.0
    upper_wick = _safe((cache.highs[end] - max(open_, close)) / candle_range) if candle_range > 0.0 else 0.0
    lower_wick = _safe((min(open_, close) - cache.lows[end]) / candle_range) if candle_range > 0.0 else 0.0

    for window in windows:
        window = max(1, int(window))
        if end < window - 1 or close <= 0.0:
            features.extend([0.0] * _CONFLUENCE_FEATURES_PER_WINDOW)
            continue
        start = end + 1 - window
        sma = _window_mean(cache.close_prefix, start, end)
        avg_volume = _window_mean(cache.volume_prefix, start, end)
        avg_true_range = _window_mean(cache.true_range_prefix, start, end)
        avg_true_range = max(avg_true_range, close * 0.00001)
        previous_high = max(cache.highs[start:end]) if end > start else cache.highs[end]
        previous_low = min(cache.lows[start:end]) if end > start else cache.lows[end]
        trend_atr = _safe((close - sma) / avg_true_range) if math.isfinite(sma) else 0.0
        volume_surge = _safe((volume / avg_volume) - 1.0) if avg_volume > 0.0 else 0.0
        breakout_atr = 0.0
        if close > previous_high:
            breakout_atr = _safe((close - previous_high) / avg_true_range)
        elif close < previous_low:
            breakout_atr = _safe((close - previous_low) / avg_true_range)
        votes = (
            _sign_vote(close - sma, close * 0.00005),
            _sign_vote(close - previous_close, close * 0.00005),
            _sign_vote(body_ratio, 0.02),
            _sign_vote(volume_surge, 0.05),
            _sign_vote(breakout_atr, 0.10),
        )
        technical_rating = sum(votes) / float(len(votes))
        features.extend([
            _safe(technical_rating),
            _safe(trend_atr),
            _safe(close_location),
            _safe(body_ratio),
            _safe(upper_wick),
            _safe(lower_wick),
            _safe(volume_surge),
            _safe(avg_true_range / close),
            _safe(breakout_atr),
        ])
    return features


def _nonlinear_expand(values: Sequence[float], transforms: Sequence[str]) -> list[float]:
    out: list[float] = []
    for name in transforms:
        if name == "tanh":
            out.extend(_tanh(v) for v in values)
        elif name == "log1p":
            out.extend(_log1p_signed(v) for v in values)
        else:
            raise ValueError(f"Unsupported transform: {name!r}")
    return out


def _polynomial_pairs(values: Sequence[float], top_k: int, degree: int) -> list[float]:
    """Return pairwise products (and optionally triples) of the first ``top_k`` features.

    For ``degree == 2`` we emit the upper-triangle pairwise products.  For
    ``degree == 3`` we additionally emit cubes and a small triple-interaction.
    """

    if top_k <= 1 or degree < 2:
        return []
    base = list(values)[:top_k]
    pairs: list[float] = []
    for i in range(len(base)):
        for j in range(i, len(base)):
            pairs.append(base[i] * base[j])
    if degree >= 3 and len(base) >= 3:
        pairs.append(base[0] * base[1] * base[2])
        pairs.extend(v ** 3 for v in base[:3])
    return [_safe(v) for v in pairs]


def advanced_feature_dimension(cfg: AdvancedFeatureConfig) -> int:
    """Dimension of the expanded feature vector — used for model load checks."""

    base = len(cfg.base_features)
    # each extra window contributes trend, momentum, volatility, and regime shape features
    extras = _EXTRA_FEATURES_PER_WINDOW * len(cfg.extra_lookback_windows)
    confluence = _CONFLUENCE_FEATURES_PER_WINDOW * len(cfg.confluence_windows)
    transforms = base * len(cfg.nonlinear_transforms)
    pairs = 0
    if cfg.polynomial_degree >= 2 and cfg.polynomial_top_features > 1:
        k = min(cfg.polynomial_top_features, base)
        pairs = k * (k + 1) // 2
        if cfg.polynomial_degree >= 3 and k >= 3:
            pairs += 1 + 3
    return base + extras + confluence + transforms + pairs


def advanced_feature_group_spans(cfg: AdvancedFeatureConfig) -> tuple[FeatureGroupSpan, ...]:
    """Return contiguous feature ranges matching the advanced feature layout."""

    spans: list[FeatureGroupSpan] = []
    cursor = 0

    def add(name: str, size: int) -> None:
        nonlocal cursor
        if size <= 0:
            return
        start = cursor
        cursor += int(size)
        spans.append(FeatureGroupSpan(name=name, start=start, end=cursor))

    base = len(cfg.base_features)
    add("base_features", base)
    add("extra_lookback_windows", _EXTRA_FEATURES_PER_WINDOW * len(cfg.extra_lookback_windows))
    add("technical_confluence", _CONFLUENCE_FEATURES_PER_WINDOW * len(cfg.confluence_windows))
    add("nonlinear_transforms", base * len(cfg.nonlinear_transforms))
    pair_count = 0
    if cfg.polynomial_degree >= 2 and cfg.polynomial_top_features > 1:
        k = min(cfg.polynomial_top_features, base)
        pair_count = k * (k + 1) // 2
        if cfg.polynomial_degree >= 3 and k >= 3:
            pair_count += 1 + 3
    add("polynomial_interactions", pair_count)
    return tuple(spans)


def expand_row(row: ModelRow, candles: Sequence[Candle], cfg: AdvancedFeatureConfig,
               at_index: int) -> ModelRow:
    """Return ``row`` with its feature tuple expanded per ``cfg``.

    ``candles`` is the full candle sequence whose ``at_index`` corresponds to
    ``row`` — this is how multi-window lookups find history behind the row.
    """

    base = list(row.features)
    closes = [c.close for c in candles[: at_index + 1]]
    extras = _extra_window_features(closes, cfg.extra_lookback_windows)
    confluence = _confluence_features_at(_build_confluence_cache(candles[: at_index + 1]), at_index, cfg.confluence_windows)
    transforms = _nonlinear_expand(base, cfg.nonlinear_transforms)
    pairs = _polynomial_pairs(base, cfg.polynomial_top_features, cfg.polynomial_degree)
    expanded = tuple(_safe(v) for v in base + extras + confluence + transforms + pairs)
    return ModelRow(
        timestamp=row.timestamp,
        close=row.close,
        features=expanded,
        label=row.label,
        volume=float(getattr(row, "volume", 0.0) or 0.0),
    )


def make_advanced_rows(
    candles: Sequence[Candle],
    cfg: AdvancedFeatureConfig,
    *,
    lookahead: int | None = None,
) -> list[ModelRow]:
    """Build expanded ``ModelRow`` objects for ``candles`` using ``cfg``."""

    enabled = normalize_enabled_features(cfg.base_features)
    label_lookahead = max(1, int(cfg.label_lookahead if lookahead is None else lookahead))
    base_rows = make_base_rows(
        candles,
        cfg.short_window,
        cfg.long_window,
        lookahead=label_lookahead,
        label_threshold=cfg.label_threshold,
        enabled_features=enabled,
    )
    if not base_rows:
        return []
    # reconstruct the index alignment used by make_rows
    valid_candles = _filter_valid(candles)
    index_by_time = {candle.close_time: idx for idx, candle in enumerate(valid_candles)}
    window_cache = _build_window_cache([candle.close for candle in valid_candles])
    confluence_cache = _build_confluence_cache(valid_candles)
    label_mode = _normalized_label_mode(cfg.label_mode)
    stop_threshold = (
        float(cfg.label_stop_threshold)
        if cfg.label_stop_threshold is not None
        else float(cfg.label_threshold)
    )
    expanded: list[ModelRow] = []
    for row in base_rows:
        idx = index_by_time.get(row.timestamp)
        if idx is None:
            continue
        label = row.label
        if label_mode == "triple_barrier":
            label = _triple_barrier_label(
                valid_candles,
                idx,
                horizon=label_lookahead,
                take_profit_pct=cfg.label_threshold,
                stop_loss_pct=stop_threshold,
            )
        base = list(row.features)
        extras = _extra_window_features_at(window_cache, idx, cfg.extra_lookback_windows)
        confluence = _confluence_features_at(confluence_cache, idx, cfg.confluence_windows)
        transforms = _nonlinear_expand(base, cfg.nonlinear_transforms)
        pairs = _polynomial_pairs(base, cfg.polynomial_top_features, cfg.polynomial_degree)
        expanded.append(
            ModelRow(
                timestamp=row.timestamp,
                close=row.close,
                features=tuple(_safe(value) for value in base + extras + confluence + transforms + pairs),
                label=label,
                volume=float(getattr(row, "volume", 0.0) or 0.0),
            )
        )
    return expanded


def make_advanced_inference_rows(
    candles: Sequence[Candle],
    cfg: AdvancedFeatureConfig,
) -> list[ModelRow]:
    """Build expanded rows for live inference without future-label lookahead."""

    enabled = normalize_enabled_features(cfg.base_features)
    base_rows = make_base_inference_rows(
        candles,
        cfg.short_window,
        cfg.long_window,
        enabled_features=enabled,
    )
    if not base_rows:
        return []
    valid_candles = _filter_valid(candles)
    index_by_time = {candle.close_time: idx for idx, candle in enumerate(valid_candles)}
    window_cache = _build_window_cache([candle.close for candle in valid_candles])
    confluence_cache = _build_confluence_cache(valid_candles)
    expanded: list[ModelRow] = []
    for row in base_rows:
        idx = index_by_time.get(row.timestamp)
        if idx is None:
            continue
        base = list(row.features)
        extras = _extra_window_features_at(window_cache, idx, cfg.extra_lookback_windows)
        confluence = _confluence_features_at(confluence_cache, idx, cfg.confluence_windows)
        transforms = _nonlinear_expand(base, cfg.nonlinear_transforms)
        pairs = _polynomial_pairs(base, cfg.polynomial_top_features, cfg.polynomial_degree)
        expanded.append(
            ModelRow(
                timestamp=row.timestamp,
                close=row.close,
                features=tuple(_safe(value) for value in base + extras + confluence + transforms + pairs),
                label=0,
                volume=float(getattr(row, "volume", 0.0) or 0.0),
            )
        )
    return expanded


def _filter_valid(candles: Sequence[Candle]) -> list[Candle]:
    return clean_candles(candles)


def _signature_fields(signature: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in str(signature or "").split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        fields[key] = value
    return fields


def _parse_int_tuple(value: str | None) -> tuple[int, ...]:
    if value is None or not value.strip():
        return ()
    parsed: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed.append(max(1, int(item)))
    return tuple(parsed)


def _parse_str_tuple(value: str | None, *, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    parsed = tuple(item.strip() for item in value.split(",") if item.strip())
    return parsed or default


def advanced_feature_signature(cfg: AdvancedFeatureConfig) -> str:
    """Deterministic signature for the advanced feature space."""

    base = base_feature_signature(
        cfg.short_window,
        cfg.long_window,
        cfg.label_threshold,
        feature_version=FEATURE_VERSION,
        enabled_features=cfg.base_features,
    )
    return "|".join([
        f"advanced_version={ADVANCED_FEATURE_VERSION}",
        f"polynomial_degree={cfg.polynomial_degree}",
        f"polynomial_top_features={cfg.polynomial_top_features}",
        f"extra_lookback_windows={','.join(str(w) for w in cfg.extra_lookback_windows)}",
        f"confluence_windows={','.join(str(w) for w in cfg.confluence_windows)}",
        f"nonlinear_transforms={','.join(cfg.nonlinear_transforms)}",
        f"label_lookahead={int(cfg.label_lookahead)}",
        f"label_mode={_normalized_label_mode(cfg.label_mode)}",
        f"label_stop_threshold={float(cfg.label_stop_threshold if cfg.label_stop_threshold is not None else cfg.label_threshold):.10g}",
        base,
    ])


def advanced_config_from_signature(
    signature: str | None,
    fallback_feature_names: Sequence[str] | None = None,
) -> AdvancedFeatureConfig | None:
    """Rebuild the exact feature config encoded in an advanced signature.

    Model-lab intentionally mutates label horizons, thresholds, and feature
    windows per candidate. Stress validation and runtime checks must rebuild
    that exact shape instead of guessing from an objective name.
    """

    fields = _signature_fields(str(signature or ""))
    if fields.get("advanced_version") != ADVANCED_FEATURE_VERSION:
        return None
    try:
        feature_names_raw = fields.get("feature_names", "")
        feature_names = tuple(item.strip() for item in feature_names_raw.split(",") if item.strip())
        base_features = normalize_enabled_features(feature_names or fallback_feature_names or FEATURE_NAMES)
        label_threshold = float(fields.get("label_threshold", "0.001"))
        stop_threshold = float(fields.get("label_stop_threshold", str(label_threshold)))
        return AdvancedFeatureConfig(
            base_features=base_features,
            polynomial_degree=max(1, int(fields.get("polynomial_degree", "2"))),
            polynomial_top_features=max(1, int(fields.get("polynomial_top_features", str(len(base_features))))),
            extra_lookback_windows=_parse_int_tuple(fields.get("extra_lookback_windows")),
            confluence_windows=_parse_int_tuple(fields.get("confluence_windows")),
            nonlinear_transforms=_parse_str_tuple(fields.get("nonlinear_transforms"), default=("tanh", "log1p")),
            short_window=max(1, int(fields.get("short_window", "10"))),
            long_window=max(1, int(fields.get("long_window", "40"))),
            label_threshold=label_threshold,
            label_lookahead=max(1, int(fields.get("label_lookahead", "4"))),
            label_mode=_normalized_label_mode(fields.get("label_mode", "forward_return")),
            label_stop_threshold=stop_threshold,
        )
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class AdvancedTrainingReport:
    """What the training suite persists alongside an advanced model artifact."""

    feature_dim: int
    feature_signature: str
    epochs: int
    learning_rate: float
    l2_penalty: float
    seed: int
    row_count: int
    positive_rate: float


def train_advanced(
    rows: Sequence[ModelRow],
    cfg: AdvancedFeatureConfig,
    *,
    epochs: int,
    learning_rate: float,
    l2_penalty: float,
    seed: int = 7,
    validation_rows: Sequence[ModelRow] | None = None,
    early_stopping_rounds: int | None = None,
    ensemble_seeds: Sequence[int] | None = None,
    compute_backend: str | None = None,
    batch_size: int = 8192,
) -> tuple[TrainedModel, AdvancedTrainingReport]:
    """Train a logistic regression on an expanded feature set.

    Returns the ``TrainedModel`` along with a small report describing the run
    so downstream code can persist reproducibility metadata.
    """

    if not rows:
        raise ValueError("No training rows available")
    signature = advanced_feature_signature(cfg)
    train_rows = list(rows)
    holdout_rows = list(validation_rows or [])
    seeds = tuple(dict.fromkeys(int(value) for value in (ensemble_seeds or (seed,))))
    models = [
        train_logistic(
            train_rows,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=member_seed,
            l2_penalty=l2_penalty,
            feature_signature=signature,
            validation_rows=holdout_rows,
            early_stopping_rounds=early_stopping_rounds,
            compute_backend=compute_backend,
            batch_size=batch_size,
        )
        for member_seed in seeds
    ]
    model = models[0]
    if len(models) > 1:
        model.ensemble_members = [ensemble_member_from_model(member) for member in models]
        model.seed = int(seeds[0])
    positives = sum(1 for row in train_rows if row.label == 1)
    report = AdvancedTrainingReport(
        feature_dim=model.feature_dim,
        feature_signature=signature,
        epochs=epochs,
        learning_rate=learning_rate,
        l2_penalty=l2_penalty,
        seed=int(seeds[0]),
        row_count=len(train_rows),
        positive_rate=(positives / len(train_rows)) if train_rows else 0.0,
    )
    return model, report


def default_config_for(objective_name: str, strategy_feature_names: Sequence[str]) -> AdvancedFeatureConfig:
    """Build a starter ``AdvancedFeatureConfig`` tied to an objective name.

    Callers in the training suite layer their own per-objective overrides on
    top of the returned config; this helper keeps the defaults in one place.
    """

    names = normalize_enabled_features(strategy_feature_names or FEATURE_NAMES)
    if objective_name == "conservative":
        return AdvancedFeatureConfig(
            base_features=names,
            polynomial_degree=2,
            polynomial_top_features=5,
            extra_lookback_windows=(10, 30, 90),
            confluence_windows=(12, 36, 96),
            label_threshold=0.0010,
            label_lookahead=8,
        )
    if objective_name in {"risky", "aggressive"}:
        return AdvancedFeatureConfig(
            base_features=names,
            polynomial_degree=3,
            polynomial_top_features=9,
            extra_lookback_windows=(3, 15, 45, 120),
            confluence_windows=(5, 13, 34, 89),
            label_threshold=0.0005,
            label_lookahead=2,
        )
    return AdvancedFeatureConfig(
        base_features=names,
        polynomial_degree=2,
        polynomial_top_features=len(names),
        extra_lookback_windows=(5, 20, 60),
        confluence_windows=(8, 21, 55),
        label_threshold=0.0010,
        label_lookahead=4,
    )
