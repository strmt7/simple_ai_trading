"""Frozen causal examples and inference for the Round 62 depth-stress screen."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
from typing import Sequence

import numpy as np

from .depth_stress_model import DEPTH_STRESS_DESCRIPTOR_NAMES


DEPTH_STRESS_SCREEN_SCHEMA_VERSION = "depth-stress-screen-v1"
DEPTH_STRESS_HORIZONS_SECONDS = (60, 300)
DEPTH_STRESS_FEATURE_NAMES = (
    *DEPTH_STRESS_DESCRIPTOR_NAMES,
    "pre_state",
    *(f"{name}_change_one_snapshot" for name in DEPTH_STRESS_DESCRIPTOR_NAMES),
    *(f"{name}_change_60s" for name in DEPTH_STRESS_DESCRIPTOR_NAMES),
    *(f"{name}_change_300s" for name in DEPTH_STRESS_DESCRIPTOR_NAMES),
    "utc_hour_sin",
    "utc_hour_cos",
    "utc_day_of_week_sin",
    "utc_day_of_week_cos",
)
_BASE_FEATURE_COUNT = len(DEPTH_STRESS_FEATURE_NAMES) - 1
_MILLISECONDS_PER_DAY = 86_400_000


def _canonical_json(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _readonly_array(values: object, *, dtype: str) -> np.ndarray:
    output = np.array(values, dtype=dtype, order="C", copy=True)
    output.setflags(write=False)
    return output


def _array_digest(digest: object, name: str, values: np.ndarray) -> None:
    updater = getattr(digest, "update")
    updater(name.encode("ascii") + b"\x00")
    updater(np.asarray(values.shape, dtype="<i8").tobytes())
    updater(values.tobytes(order="C"))


def _month_ordinals(timestamp_ms: np.ndarray) -> np.ndarray:
    return (
        timestamp_ms.astype("datetime64[ms]").astype("datetime64[M]").astype(np.int64)
    )


def utc_month_label(month_ordinal: int) -> str:
    value = int(month_ordinal)
    try:
        return str(np.datetime64("1970-01", "M") + np.timedelta64(value, "M"))
    except (OverflowError, ValueError) as exc:
        raise ValueError("UTC month ordinal is invalid") from exc


@dataclass(frozen=True)
class DepthStressPanel:
    symbol: str
    timestamp_ms: np.ndarray
    descriptors: np.ndarray
    source_fingerprint: str

    def __post_init__(self) -> None:
        symbol = str(self.symbol).strip().upper()
        timestamps = _readonly_array(self.timestamp_ms, dtype="<i8")
        descriptors = _readonly_array(self.descriptors, dtype="<f8")
        if (
            not symbol
            or not symbol.isalnum()
            or timestamps.ndim != 1
            or len(timestamps) < 30
            or np.any(timestamps <= 0)
            or np.any(np.diff(timestamps) <= 0)
            or descriptors.shape
            != (len(timestamps), len(DEPTH_STRESS_DESCRIPTOR_NAMES))
            or not np.all(np.isfinite(descriptors))
            or len(self.source_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.source_fingerprint
            )
        ):
            raise ValueError("depth-stress panel contract is invalid")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timestamp_ms", timestamps)
        object.__setattr__(self, "descriptors", descriptors)

    @property
    def month_ordinals(self) -> np.ndarray:
        return _month_ordinals(self.timestamp_ms)

    @property
    def panel_sha256(self) -> str:
        digest = hashlib.sha256(
            _canonical_json(
                {
                    "schema_version": DEPTH_STRESS_SCREEN_SCHEMA_VERSION,
                    "symbol": self.symbol,
                    "source_fingerprint": self.source_fingerprint,
                }
            )
        )
        _array_digest(digest, "timestamp_ms", self.timestamp_ms)
        _array_digest(digest, "descriptors", self.descriptors)
        return digest.hexdigest()


@dataclass(frozen=True)
class DepthStressExamples:
    symbol: str
    horizon_seconds: int
    maximum_snapshot_age_seconds: int
    maximum_gap_seconds: int
    anchor_time_ms: np.ndarray
    pre_index: np.ndarray
    post_index: np.ndarray
    month_ordinal: np.ndarray
    utc_day: np.ndarray
    base_features: np.ndarray
    panel_sha256: str
    examples_sha256: str

    def __post_init__(self) -> None:
        anchors = _readonly_array(self.anchor_time_ms, dtype="<i8")
        pre = _readonly_array(self.pre_index, dtype="<i8")
        post = _readonly_array(self.post_index, dtype="<i8")
        months = _readonly_array(self.month_ordinal, dtype="<i8")
        days = _readonly_array(self.utc_day, dtype="<i8")
        features = _readonly_array(self.base_features, dtype="<f4")
        rows = len(anchors)
        if (
            self.horizon_seconds not in DEPTH_STRESS_HORIZONS_SECONDS
            or not 1 <= self.maximum_snapshot_age_seconds <= 45
            or not self.maximum_snapshot_age_seconds < self.maximum_gap_seconds <= 90
            or rows < 1
            or any(len(values) != rows for values in (pre, post, months, days))
            or features.shape != (rows, _BASE_FEATURE_COUNT)
            or not np.all(np.isfinite(features))
            or np.any(np.diff(anchors) <= 0)
            or np.any(pre < 0)
            or np.any(post <= pre)
            or len(self.panel_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.panel_sha256
            )
            or len(self.examples_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.examples_sha256
            )
        ):
            raise ValueError("depth-stress examples contract is invalid")
        object.__setattr__(self, "anchor_time_ms", anchors)
        object.__setattr__(self, "pre_index", pre)
        object.__setattr__(self, "post_index", post)
        object.__setattr__(self, "month_ordinal", months)
        object.__setattr__(self, "utc_day", days)
        object.__setattr__(self, "base_features", features)
        expected = _examples_sha256(
            symbol=self.symbol,
            horizon_seconds=self.horizon_seconds,
            maximum_snapshot_age_seconds=self.maximum_snapshot_age_seconds,
            maximum_gap_seconds=self.maximum_gap_seconds,
            panel_sha256=self.panel_sha256,
            anchor_time_ms=anchors,
            pre_index=pre,
            post_index=post,
            month_ordinal=months,
            utc_day=days,
            base_features=features,
        )
        if not hmac.compare_digest(self.examples_sha256, expected):
            raise ValueError("depth-stress examples digest does not match its arrays")

    def feature_matrix(
        self,
        pre_states: Sequence[int] | np.ndarray,
        *,
        rows: Sequence[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        states = np.asarray(pre_states, dtype=np.int8)
        if states.shape != (len(self.anchor_time_ms),) or np.any(
            (states < 0) | (states > 2)
        ):
            raise ValueError("depth-stress pre-state feature is invalid")
        indexes = (
            np.arange(len(states), dtype=np.int64)
            if rows is None
            else np.asarray(rows, dtype=np.int64)
        )
        if (
            indexes.ndim != 1
            or not len(indexes)
            or np.any(indexes < 0)
            or np.any(indexes >= len(states))
        ):
            raise ValueError("depth-stress feature row indexes are invalid")
        selected = self.base_features[indexes]
        return np.column_stack(
            (selected[:, :3], states[indexes].astype(np.float32), selected[:, 3:])
        ).astype(np.float32, copy=False)


def _examples_sha256(
    *,
    symbol: str,
    horizon_seconds: int,
    maximum_snapshot_age_seconds: int,
    maximum_gap_seconds: int,
    panel_sha256: str,
    anchor_time_ms: np.ndarray,
    pre_index: np.ndarray,
    post_index: np.ndarray,
    month_ordinal: np.ndarray,
    utc_day: np.ndarray,
    base_features: np.ndarray,
) -> str:
    contract = {
        "schema_version": DEPTH_STRESS_SCREEN_SCHEMA_VERSION,
        "symbol": symbol,
        "horizon_seconds": horizon_seconds,
        "maximum_snapshot_age_seconds": maximum_snapshot_age_seconds,
        "maximum_gap_seconds": maximum_gap_seconds,
        "feature_names": list(DEPTH_STRESS_FEATURE_NAMES),
        "panel_sha256": panel_sha256,
    }
    digest = hashlib.sha256(_canonical_json(contract))
    for name, values in (
        ("anchor_time_ms", anchor_time_ms),
        ("pre_index", pre_index),
        ("post_index", post_index),
        ("month_ordinal", month_ordinal),
        ("utc_day", utc_day),
        ("base_features", base_features),
    ):
        _array_digest(digest, name, np.ascontiguousarray(values))
    return digest.hexdigest()


@dataclass(frozen=True)
class PairedLossComparison:
    rows: int
    blocks: int
    baseline_mean_loss: float
    challenger_mean_loss: float
    mean_loss_difference: float
    relative_improvement: float
    one_sided_p_value: float
    permutation_draws: int
    seed: int


def build_depth_stress_examples(
    panel: DepthStressPanel,
    *,
    horizon_seconds: int,
    maximum_snapshot_age_seconds: int = 45,
    maximum_gap_seconds: int = 90,
) -> DepthStressExamples:
    """Build non-overlapping causal examples without crossing month or data gaps."""

    horizon = int(horizon_seconds)
    maximum_age = int(maximum_snapshot_age_seconds)
    maximum_gap = int(maximum_gap_seconds)
    if (
        horizon not in DEPTH_STRESS_HORIZONS_SECONDS
        or not 1 <= maximum_age <= 45
        or not maximum_age < maximum_gap <= 90
    ):
        raise ValueError("depth-stress example timing contract is invalid")
    timestamps = panel.timestamp_ms
    months = panel.month_ordinals
    gap_ms = maximum_gap * 1_000
    segment_break = np.concatenate(
        (
            np.asarray([True]),
            (np.diff(timestamps) > gap_ms) | (np.diff(months) != 0),
        )
    )
    segment_id = np.cumsum(segment_break, dtype=np.int64) - 1
    starts = np.flatnonzero(segment_break)
    ends = np.concatenate((starts[1:] - 1, np.asarray([len(timestamps) - 1])))
    horizon_ms = horizon * 1_000
    anchor_parts: list[np.ndarray] = []
    for start, end in zip(starts, ends, strict=True):
        first = ((int(timestamps[start]) + horizon_ms - 1) // horizon_ms) * horizon_ms
        last = int(timestamps[end]) - horizon_ms
        if first <= last:
            anchor_parts.append(np.arange(first, last + 1, horizon_ms, dtype=np.int64))
    if not anchor_parts:
        raise ValueError("depth-stress panel has no eligible anchors")
    anchors = np.concatenate(anchor_parts)
    pre = np.searchsorted(timestamps, anchors, side="right") - 1
    post_target = anchors + horizon_ms
    post = np.searchsorted(timestamps, post_target, side="left")
    lag_60_target = anchors - 60_000
    lag_300_target = anchors - 300_000
    lag_60 = np.searchsorted(timestamps, lag_60_target, side="right") - 1
    lag_300 = np.searchsorted(timestamps, lag_300_target, side="right") - 1
    immediate = pre - 1
    in_bounds = (pre >= 1) & (post < len(timestamps)) & (lag_60 >= 0) & (lag_300 >= 0)
    anchors = anchors[in_bounds]
    pre = pre[in_bounds]
    post = post[in_bounds]
    post_target = post_target[in_bounds]
    lag_60_target = lag_60_target[in_bounds]
    lag_300_target = lag_300_target[in_bounds]
    lag_60 = lag_60[in_bounds]
    lag_300 = lag_300[in_bounds]
    immediate = immediate[in_bounds]
    maximum_age_ms = maximum_age * 1_000
    valid = (
        (anchors - timestamps[pre] <= maximum_age_ms)
        & (post_target <= timestamps[post])
        & (timestamps[post] - post_target <= maximum_age_ms)
        & (lag_60_target - timestamps[lag_60] <= maximum_age_ms)
        & (lag_300_target - timestamps[lag_300] <= maximum_age_ms)
        & (segment_id[pre] == segment_id[post])
        & (segment_id[pre] == segment_id[immediate])
        & (segment_id[pre] == segment_id[lag_60])
        & (segment_id[pre] == segment_id[lag_300])
    )
    anchors = anchors[valid]
    pre = pre[valid]
    post = post[valid]
    lag_60 = lag_60[valid]
    lag_300 = lag_300[valid]
    immediate = immediate[valid]
    if not len(anchors):
        raise ValueError("depth-stress panel has no complete causal examples")
    current = panel.descriptors[pre]
    hour_angle = (
        2.0 * np.pi * ((anchors % _MILLISECONDS_PER_DAY) / _MILLISECONDS_PER_DAY)
    )
    utc_day = anchors // _MILLISECONDS_PER_DAY
    day_of_week = (utc_day + 3) % 7
    weekday_angle = 2.0 * np.pi * day_of_week / 7.0
    base_features = np.column_stack(
        (
            current,
            current - panel.descriptors[immediate],
            current - panel.descriptors[lag_60],
            current - panel.descriptors[lag_300],
            np.sin(hour_angle),
            np.cos(hour_angle),
            np.sin(weekday_angle),
            np.cos(weekday_angle),
        )
    ).astype(np.float32)
    anchor_months = _month_ordinals(anchors)
    examples_sha256 = _examples_sha256(
        symbol=panel.symbol,
        horizon_seconds=horizon,
        maximum_snapshot_age_seconds=maximum_age,
        maximum_gap_seconds=maximum_gap,
        panel_sha256=panel.panel_sha256,
        anchor_time_ms=anchors,
        pre_index=pre,
        post_index=post,
        month_ordinal=anchor_months,
        utc_day=utc_day,
        base_features=base_features,
    )
    return DepthStressExamples(
        symbol=panel.symbol,
        horizon_seconds=horizon,
        maximum_snapshot_age_seconds=maximum_age,
        maximum_gap_seconds=maximum_gap,
        anchor_time_ms=anchors,
        pre_index=pre,
        post_index=post,
        month_ordinal=anchor_months,
        utc_day=utc_day,
        base_features=base_features,
        panel_sha256=panel.panel_sha256,
        examples_sha256=examples_sha256,
    )


def paired_blocked_permutation_test(
    baseline_loss: Sequence[float] | np.ndarray,
    challenger_loss: Sequence[float] | np.ndarray,
    block_ids: Sequence[int] | np.ndarray,
    *,
    draws: int = 10_000,
    seed: int = 20260717,
) -> PairedLossComparison:
    """Test whether challenger loss is lower while preserving UTC-day blocks."""

    baseline = np.asarray(baseline_loss, dtype=np.float64)
    challenger = np.asarray(challenger_loss, dtype=np.float64)
    blocks = np.asarray(block_ids, dtype=np.int64)
    draw_count = int(draws)
    if (
        baseline.ndim != 1
        or baseline.shape != challenger.shape
        or baseline.shape != blocks.shape
        or len(baseline) < 30
        or not np.all(np.isfinite(baseline))
        or not np.all(np.isfinite(challenger))
        or np.any(baseline < 0.0)
        or np.any(challenger < 0.0)
        or not 100 <= draw_count <= 1_000_000
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ValueError("paired blocked loss evidence is invalid")
    unique_blocks, inverse = np.unique(blocks, return_inverse=True)
    if len(unique_blocks) < 10:
        raise ValueError("paired blocked test requires at least ten blocks")
    difference = challenger - baseline
    block_sums = np.bincount(inverse, weights=difference, minlength=len(unique_blocks))
    observed = float(np.sum(block_sums) / len(difference))
    rng = np.random.default_rng(seed)
    lower_or_equal = 0
    completed = 0
    while completed < draw_count:
        batch = min(512, draw_count - completed)
        signs = rng.integers(0, 2, size=(batch, len(block_sums)), dtype=np.int8)
        signs = signs.astype(np.float64) * 2.0 - 1.0
        permuted = signs @ block_sums / len(difference)
        lower_or_equal += int(np.count_nonzero(permuted <= observed))
        completed += batch
    baseline_mean = float(np.mean(baseline))
    challenger_mean = float(np.mean(challenger))
    if not baseline_mean > 0.0:
        raise ValueError("baseline mean loss must be positive")
    return PairedLossComparison(
        rows=len(difference),
        blocks=len(unique_blocks),
        baseline_mean_loss=baseline_mean,
        challenger_mean_loss=challenger_mean,
        mean_loss_difference=challenger_mean - baseline_mean,
        relative_improvement=(baseline_mean - challenger_mean) / baseline_mean,
        one_sided_p_value=(lower_or_equal + 1.0) / (draw_count + 1.0),
        permutation_draws=draw_count,
        seed=seed,
    )


def benjamini_hochberg_q_values(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=np.float64)
    if (
        values.ndim != 1
        or not len(values)
        or not np.all(np.isfinite(values))
        or np.any((values < 0.0) | (values > 1.0))
    ):
        raise ValueError("p-values are invalid")
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1, dtype=np.float64)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return output


__all__ = [
    "DEPTH_STRESS_FEATURE_NAMES",
    "DEPTH_STRESS_HORIZONS_SECONDS",
    "DEPTH_STRESS_SCREEN_SCHEMA_VERSION",
    "DepthStressExamples",
    "DepthStressPanel",
    "PairedLossComparison",
    "benjamini_hochberg_q_values",
    "build_depth_stress_examples",
    "paired_blocked_permutation_test",
    "utc_month_label",
]
