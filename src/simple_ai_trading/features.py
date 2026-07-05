"""Feature construction for training and inference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

from .api import Candle
from .compute import BackendInfo, resolve_backend
from .market_data import clean_candles


FEATURE_VERSION = "v1"

# Ordered feature names and count used by persistence checks.
FEATURE_NAMES = (
    "momentum_1",
    "momentum_3",
    "momentum_10",
    "momentum_20",
    "ema_spread",
    "rsi",
    "ema_gap",
    "relative_atr",
    "volatility_20",
    "volume_ratio",
    "trend_acceleration",
    "gap_to_vwap",
    "volume_trend",
)
_FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}


def normalize_enabled_features(enabled_features: Sequence[str] | None = None) -> tuple[str, ...]:
    if enabled_features is None:
        return tuple(FEATURE_NAMES)
    normalized: list[str] = []
    for name in enabled_features:
        feature_name = str(name)
        if feature_name not in FEATURE_NAMES:
            raise ValueError(f"Unknown feature: {feature_name}")
        if feature_name not in normalized:
            normalized.append(feature_name)
    if not normalized:
        raise ValueError("At least one feature must remain enabled")
    return tuple(normalized)


def _feature_indices(enabled_features: Sequence[str] | None = None) -> tuple[int, ...]:
    normalized = normalize_enabled_features(enabled_features)
    return tuple(_FEATURE_INDEX[name] for name in normalized)


def feature_signature(
    short_window: int,
    long_window: int,
    label_threshold: float,
    *,
    feature_version: str = FEATURE_VERSION,
    enabled_features: Sequence[str] | None = None,
) -> str:
    """Return a deterministic signature for a feature configuration."""
    short_window = int(short_window)
    long_window = int(long_window)
    threshold = float(label_threshold)
    selected = normalize_enabled_features(enabled_features)
    return "|".join(
        [
            f"feature_version={feature_version}",
            f"feature_count={len(selected)}",
            f"feature_names={','.join(selected)}",
            f"short_window={short_window}",
            f"long_window={long_window}",
            f"label_threshold={threshold:.10g}",
        ]
    )


def _valid_ohlcv(candle: Candle) -> bool:
    if not all(math.isfinite(value) for value in (candle.open, candle.high, candle.low, candle.close)):
        return False
    if candle.open <= 0.0 or candle.high <= 0.0 or candle.low <= 0.0 or candle.close <= 0.0:
        return False
    if candle.volume < 0.0 or candle.open_time < 0 or candle.close_time < 0:
        return False
    if candle.low > candle.high:
        return False
    if not (candle.low <= candle.open <= candle.high):
        return False
    if not (candle.low <= candle.close <= candle.high):
        return False
    if candle.close_time < candle.open_time:
        return False
    return True


@dataclass(frozen=True)
class ModelRow:
    timestamp: int
    close: float
    features: Tuple[float, ...]
    label: int
    volume: float = 0.0


def feature_dimension(enabled_features: Sequence[str] | None = None) -> int:
    return len(normalize_enabled_features(enabled_features))


def _safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def _pct(numerator: float, denominator: float) -> float:
    return _safe_div(numerator - denominator, denominator)


def _sma(values: Sequence[float], window: int) -> float:
    if len(values) < window:
        return float("nan")
    return sum(values[-window:]) / float(window)


def _prefix_sum(values: Sequence[float]) -> list[float]:
    total = 0.0
    prefix = [0.0]
    for value in values:
        total += value
        prefix.append(total)
    return prefix


def _window_mean(prefix: Sequence[float], start: int, end: int) -> float:
    if end < start:
        return float("nan")
    return (prefix[end + 1] - prefix[start]) / float(end - start + 1)


def _rolling_mean(prefix: Sequence[float], end: int, window: int) -> float:
    if window <= 0 or end < window - 1:
        return float("nan")
    return _window_mean(prefix, end - window + 1, end)


def _ema(values: Sequence[float], window: int) -> float:
    if len(values) < window:
        return float("nan")
    k = 2.0 / (window + 1)
    ema = values[0]
    for value in values[1:]:
        ema = value * k + ema * (1 - k)
    return ema


def _rsi(values: Sequence[float], window: int) -> float:
    if len(values) < window + 1:
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


def _true_range(candles: Sequence[Candle], i: int) -> float:
    prev_close = candles[i - 1].close
    if prev_close <= 0:
        return 0.0
    high = candles[i].high
    low = candles[i].low
    return max(high - low, abs(high - prev_close), abs(low - prev_close))


def _safe_features(values: Sequence[float]) -> list[float]:
    return [0.0 if not math.isfinite(v) else float(v) for v in values]


def _torch_device_for_backend(backend: BackendInfo):  # pragma: no cover - optional GPU runtime
    if backend.kind == "directml":
        import torch_directml  # type: ignore

        return torch_directml.device()
    return backend.device


@dataclass(frozen=True)
class _FeatureCache:
    candles: list[Candle]
    closes: list[float]
    volumes: list[float]
    close_prefix: list[float]
    volume_prefix: list[float]
    abs_change_prefix: list[float]
    true_range_prefix: list[float]
    gain_prefix: list[float]
    loss_prefix: list[float]


def _build_feature_cache(candles: Sequence[Candle]) -> _FeatureCache:
    cleaned = [candle for candle in clean_candles(candles) if _valid_ohlcv(candle)]
    closes = [candle.close for candle in cleaned]
    volumes = [candle.volume for candle in cleaned]
    abs_changes = [0.0]
    true_ranges = [0.0]
    gains = [0.0]
    losses = [0.0]
    for index in range(1, len(cleaned)):
        previous = closes[index - 1]
        current = closes[index]
        delta = current - previous
        gains.append(max(0.0, delta))
        losses.append(max(0.0, -delta))
        abs_changes.append(abs(_pct(current, previous)))
        true_ranges.append(_true_range(cleaned, index))
    return _FeatureCache(
        candles=cleaned,
        closes=closes,
        volumes=volumes,
        close_prefix=_prefix_sum(closes),
        volume_prefix=_prefix_sum(volumes),
        abs_change_prefix=_prefix_sum(abs_changes),
        true_range_prefix=_prefix_sum(true_ranges),
        gain_prefix=_prefix_sum(gains),
        loss_prefix=_prefix_sum(losses),
    )


def _tensor_prefix(torch, values):
    return torch.cat([torch.zeros((1,), dtype=values.dtype, device=values.device), torch.cumsum(values, dim=0)])


def _tensor_window_mean(torch, prefix, start, end):
    return (prefix[end + 1] - prefix[start]) / (end - start + 1).to(dtype=prefix.dtype)


def _tensor_lagged_pct(torch, closes, indices, close, lag: int):
    valid = indices >= int(lag)
    safe_index = torch.where(valid, indices - int(lag), torch.zeros_like(indices))
    previous = closes[safe_index]
    return torch.where(valid, (close - previous) / previous, torch.zeros_like(close))


def _tensor_fixed_ema(torch, closes, indices, long_window: int):
    k = 2.0 / float(long_window + 1)
    decay = 1.0 - k
    fixed_len = max(1, int(2 * long_window))
    device = closes.device
    ema = torch.zeros((indices.shape[0],), dtype=closes.dtype, device=device)
    fixed_mask = indices >= fixed_len - 1
    if bool(torch.any(fixed_mask).detach().cpu().item()):
        windows = closes.unfold(0, fixed_len, 1)
        selected = windows[indices[fixed_mask] - fixed_len + 1]
        weights = [decay ** (fixed_len - 1)]
        weights.extend(k * (decay ** (fixed_len - 1 - offset)) for offset in range(1, fixed_len))
        weight_t = torch.tensor(weights, dtype=closes.dtype, device=device)
        ema[fixed_mask] = torch.sum(selected * weight_t, dim=1)
    if bool(torch.any(~fixed_mask).detach().cpu().item()):
        early_positions = torch.nonzero(~fixed_mask).flatten().detach().cpu().tolist()
        for position in early_positions:
            index = int(indices[position].detach().cpu().item())
            length = index + 1
            values = closes[:length]
            weights = [decay ** (length - 1)]
            weights.extend(k * (decay ** (length - 1 - offset)) for offset in range(1, length))
            weight_t = torch.tensor(weights, dtype=closes.dtype, device=device)
            ema[position] = torch.sum(values * weight_t)
    return ema


def _make_rows_tensor(
    cache: _FeatureCache,
    selected_indices: tuple[int, ...],
    short_window: int,
    long_window: int,
    *,
    lookahead: int,
    label_threshold: float,
    include_labels: bool,
    backend: BackendInfo,
) -> list[ModelRow]:  # pragma: no cover - host GPU coverage exercises this path
    import torch  # type: ignore

    device = _torch_device_for_backend(backend)
    dtype = torch.float32
    n = len(cache.candles)
    start_index = int(long_window + lookahead) if include_labels else int(long_window)
    end_index = int(n - lookahead) if include_labels else int(n)
    if end_index <= start_index:
        return []

    closes = torch.tensor(cache.closes, dtype=dtype, device=device)
    volumes = torch.tensor(cache.volumes, dtype=dtype, device=device)
    highs = torch.tensor([candle.high for candle in cache.candles], dtype=dtype, device=device)
    lows = torch.tensor([candle.low for candle in cache.candles], dtype=dtype, device=device)
    indices = torch.arange(start_index, end_index, dtype=torch.long, device=device)

    prev_closes = torch.cat([closes[:1], closes[:-1]])
    deltas = closes - prev_closes
    gains = torch.where(deltas > 0.0, deltas, torch.zeros_like(deltas))
    losses = torch.where(deltas < 0.0, -deltas, torch.zeros_like(deltas))
    abs_changes = torch.abs((closes - prev_closes) / prev_closes)
    abs_changes[0] = 0.0
    tr_values = torch.maximum(
        highs - lows,
        torch.maximum(torch.abs(highs - prev_closes), torch.abs(lows - prev_closes)),
    )
    tr_values[0] = 0.0

    close_prefix = _tensor_prefix(torch, closes)
    volume_prefix = _tensor_prefix(torch, volumes)
    abs_change_prefix = _tensor_prefix(torch, abs_changes)
    true_range_prefix = _tensor_prefix(torch, tr_values)
    gain_prefix = _tensor_prefix(torch, gains)
    loss_prefix = _tensor_prefix(torch, losses)

    close = closes[indices]
    short = _tensor_window_mean(torch, close_prefix, indices - short_window + 1, indices)
    long = _tensor_window_mean(torch, close_prefix, indices - long_window + 1, indices)
    ema = _tensor_fixed_ema(torch, closes, indices, long_window)

    rsi_window = 14
    rsi_valid = indices >= rsi_window
    rsi_start = torch.where(rsi_valid, indices + 1 - rsi_window, torch.zeros_like(indices))
    rsi_end = torch.where(rsi_valid, indices, torch.zeros_like(indices))
    avg_gain = _tensor_window_mean(torch, gain_prefix, rsi_start, rsi_end)
    avg_loss = _tensor_window_mean(torch, loss_prefix, rsi_start, rsi_end)
    rsi_value = torch.where(avg_loss == 0.0, torch.full_like(avg_loss, 100.0), 100.0 - 100.0 / (1.0 + avg_gain / avg_loss))
    rsi = torch.where(rsi_valid, rsi_value, torch.full_like(avg_loss, float("nan")))

    momentum = _tensor_lagged_pct(torch, closes, indices, close, 1)
    momentum_3 = _tensor_lagged_pct(torch, closes, indices, close, 3)
    momentum_10 = _tensor_lagged_pct(torch, closes, indices, close, 10)
    momentum_20 = _tensor_lagged_pct(torch, closes, indices, close, 20)
    spread = (short - long) / long

    vol_moment_valid = indices >= 20
    vol_moment_start = torch.where(vol_moment_valid, indices - 19, torch.zeros_like(indices))
    vol_moment_end = torch.where(vol_moment_valid, indices, torch.zeros_like(indices))
    vol_moment_value = _tensor_window_mean(torch, abs_change_prefix, vol_moment_start, vol_moment_end)
    vol_moment = torch.where(vol_moment_valid, vol_moment_value, torch.full_like(close, float("nan")))
    atr_count = torch.minimum(indices, torch.full_like(indices, 14))
    atr = _tensor_window_mean(torch, true_range_prefix, indices - atr_count + 1, indices)
    rel_atr = atr / close
    ema_spread = (ema - close) / close

    prev_volume_window = torch.minimum(torch.full_like(indices, 20), torch.maximum(torch.ones_like(indices), indices))
    prev_vol = _tensor_window_mean(torch, volume_prefix, indices - prev_volume_window, indices - 1)
    vol_ratio = (volumes[indices] - prev_vol) / prev_vol

    prev_short_valid = indices - 2 >= short_window - 1
    prev_short_start = torch.where(prev_short_valid, indices - short_window - 1, torch.zeros_like(indices))
    prev_short_end = torch.where(prev_short_valid, indices - 2, torch.zeros_like(indices))
    prev_short_value = _tensor_window_mean(torch, close_prefix, prev_short_start, prev_short_end)
    prev_short = torch.where(prev_short_valid, prev_short_value, torch.full_like(short, float("nan")))
    trend_accel = torch.where(prev_short != 0.0, (short - prev_short) / prev_short, torch.zeros_like(short))
    gap_window = torch.minimum(torch.full_like(indices, 5), indices + 1)
    gap_average = _tensor_window_mean(torch, close_prefix, indices - gap_window + 1, indices)
    gap_to_vwap = (close - gap_average) / close
    vol_short_window = torch.minimum(torch.full_like(indices, short_window), indices + 1)
    vol_long_window = torch.minimum(torch.full_like(indices, long_window), indices + 1)
    vol_short = _tensor_window_mean(torch, volume_prefix, indices - vol_short_window + 1, indices)
    vol_long = _tensor_window_mean(torch, volume_prefix, indices - vol_long_window + 1, indices)
    volume_trend = (vol_short - vol_long) / vol_long

    full = torch.stack(
        [
            momentum,
            momentum_3,
            momentum_10,
            momentum_20,
            spread,
            rsi / 100.0,
            ema_spread,
            rel_atr,
            vol_moment,
            vol_ratio,
            trend_accel,
            gap_to_vwap,
            volume_trend,
        ],
        dim=1,
    )
    finite_core = torch.isfinite(short) & torch.isfinite(long) & torch.isfinite(ema) & torch.isfinite(rsi)
    full = torch.where(torch.isfinite(full), full, torch.zeros_like(full))
    selected = full[:, list(selected_indices)]

    if include_labels:
        future = closes[indices + lookahead]
        labels_t = ((future - close) / close >= float(label_threshold)).to(dtype=torch.int64)
    else:
        labels_t = torch.zeros((indices.shape[0],), dtype=torch.int64, device=device)

    selected_cpu = selected.detach().cpu().tolist()
    labels_cpu = [int(value) for value in labels_t.detach().cpu().tolist()]
    valid_cpu = [bool(value) for value in finite_core.detach().cpu().tolist()]
    index_cpu = [int(value) for value in indices.detach().cpu().tolist()]
    rows: list[ModelRow] = []
    for offset, valid in enumerate(valid_cpu):
        if not valid:
            continue
        index = index_cpu[offset]
        rows.append(
            ModelRow(
                timestamp=cache.candles[index].close_time,
                close=cache.closes[index],
                features=tuple(float(value) for value in selected_cpu[offset]),
                label=labels_cpu[offset],
                volume=cache.candles[index].volume,
            )
        )
    return rows


def _make_rows_with_backend(
    cache: _FeatureCache,
    selected_indices: tuple[int, ...],
    short_window: int,
    long_window: int,
    *,
    lookahead: int,
    label_threshold: float,
    include_labels: bool,
    compute_backend: str | None,
) -> list[ModelRow] | None:
    if not compute_backend:
        return None
    backend = resolve_backend(compute_backend)
    if backend.kind == "cpu":
        return None
    try:
        return _make_rows_tensor(
            cache,
            selected_indices,
            short_window,
            long_window,
            lookahead=lookahead,
            label_threshold=label_threshold,
            include_labels=include_labels,
            backend=backend,
        )
    except Exception:
        return None


def _rsi_at(cache: _FeatureCache, end: int, window: int) -> float:
    if window <= 0 or end < window:
        return float("nan")
    start = end + 1 - window
    avg_gain = _window_mean(cache.gain_prefix, start, end)
    avg_loss = _window_mean(cache.loss_prefix, start, end)
    if avg_loss == 0:
        return 100.0
    return 100 - 100 / (1 + avg_gain / avg_loss)


def _build_full_features(
    cache: _FeatureCache,
    index: int,
    short_window: int,
    long_window: int,
) -> tuple[float, ...] | None:
    closes = cache.closes
    volumes = cache.volumes
    close = closes[index]
    short = _rolling_mean(cache.close_prefix, index, short_window)
    long = _rolling_mean(cache.close_prefix, index, long_window)
    ema = _ema(closes[max(0, index + 1 - (2 * long_window)): index + 1], long_window)
    rsi = _rsi_at(cache, index, 14)
    if not all(math.isfinite(value) for value in (short, long, ema, rsi)):
        return None

    momentum = _pct(close, closes[index - 1]) if index >= 1 else 0.0
    momentum_3 = _pct(close, closes[index - 3]) if index >= 3 else 0.0
    momentum_10 = _pct(close, closes[index - 10]) if index >= 10 else 0.0
    momentum_20 = _pct(close, closes[index - 20]) if index >= 20 else 0.0
    spread = _safe_div(short - long, long)

    vol_moment = (
        _window_mean(cache.abs_change_prefix, index - 19, index)
        if index >= 20
        else float("nan")
    )
    atr_count = min(14, index)
    atr = _window_mean(cache.true_range_prefix, index - atr_count + 1, index)
    rel_atr = _safe_div(atr, close)
    ema_spread = _safe_div(ema - close, close)

    prev_vol = _rolling_mean(cache.volume_prefix, index - 1, min(20, max(1, index)))
    vol_ratio = _safe_div(volumes[index] - prev_vol, prev_vol)
    prev_short = _rolling_mean(cache.close_prefix, index - 2, short_window)
    trend_accel = _safe_div(short - prev_short, prev_short) if prev_short else 0.0
    gap_average = _rolling_mean(cache.close_prefix, index, min(5, index + 1))
    gap_to_vwap = _safe_div(close - gap_average, close)
    vol_short = _rolling_mean(cache.volume_prefix, index, min(short_window, index + 1))
    vol_long = _rolling_mean(cache.volume_prefix, index, min(long_window, index + 1))
    volume_trend = _safe_div(vol_short - vol_long, vol_long)

    return tuple(_safe_features([
        momentum,
        momentum_3,
        momentum_10,
        momentum_20,
        spread,
        rsi / 100.0,
        ema_spread,
        rel_atr,
        vol_moment,
        vol_ratio,
        trend_accel,
        gap_to_vwap,
        volume_trend,
    ]))


def make_rows(
    candles: Sequence[Candle],
    short_window: int,
    long_window: int,
    *,
    lookahead: int = 1,
    label_threshold: float = 0.001,
    enabled_features: Sequence[str] | None = None,
    compute_backend: str | None = None,
) -> list[ModelRow]:
    if short_window <= 0 or long_window <= 0 or lookahead <= 0:
        raise ValueError("short_window, long_window, and lookahead must be positive")
    if long_window < short_window:
        raise ValueError("long_window must be greater than or equal to short_window")

    selected_indices = _feature_indices(enabled_features)
    cache = _build_feature_cache(candles)
    rows: list[ModelRow] = []
    min_window = max(long_window, short_window, lookahead + 2, 2 * long_window)
    if len(cache.candles) < min_window:
        return rows
    accelerated = _make_rows_with_backend(
        cache,
        selected_indices,
        short_window,
        long_window,
        lookahead=lookahead,
        label_threshold=label_threshold,
        include_labels=True,
        compute_backend=compute_backend,
    )
    if accelerated is not None:
        return accelerated

    for i in range(long_window + lookahead, len(cache.candles) - lookahead):
        full_features = _build_full_features(cache, i, short_window, long_window)
        if full_features is None:
            continue
        features = tuple(full_features[index] for index in selected_indices)

        future = cache.closes[i + lookahead]
        present = cache.closes[i]
        label = int(_pct(future, present) >= label_threshold)
        rows.append(
            ModelRow(
                timestamp=cache.candles[i].close_time,
                close=present,
                features=features,
                label=label,
                volume=cache.candles[i].volume,
            )
        )

    return rows


def make_inference_rows(
    candles: Sequence[Candle],
    short_window: int,
    long_window: int,
    *,
    enabled_features: Sequence[str] | None = None,
    compute_backend: str | None = None,
) -> list[ModelRow]:
    if short_window <= 0 or long_window <= 0:
        raise ValueError("short_window and long_window must be positive")
    if long_window < short_window:
        raise ValueError("long_window must be greater than or equal to short_window")

    selected_indices = _feature_indices(enabled_features)
    cache = _build_feature_cache(candles)
    rows: list[ModelRow] = []
    min_window = max(long_window, short_window, 2, 2 * long_window)
    if len(cache.candles) < min_window:
        return rows
    accelerated = _make_rows_with_backend(
        cache,
        selected_indices,
        short_window,
        long_window,
        lookahead=1,
        label_threshold=0.0,
        include_labels=False,
        compute_backend=compute_backend,
    )
    if accelerated is not None:
        return accelerated

    for i in range(long_window, len(cache.candles)):
        full_features = _build_full_features(cache, i, short_window, long_window)
        if full_features is None:
            continue
        features = tuple(full_features[index] for index in selected_indices)
        rows.append(
            ModelRow(
                timestamp=cache.candles[i].close_time,
                close=cache.closes[i],
                features=features,
                label=0,
                volume=cache.candles[i].volume,
            )
        )

    return rows


def make_rows_legacy(candles: Sequence[Candle], short_window: int, long_window: int,
                     lookahead: int = 1) -> list[ModelRow]:
    """Compatibility helper for existing integrations expecting 5-feature rows."""
    return make_rows(candles, short_window, long_window, lookahead=lookahead, label_threshold=0.001)
