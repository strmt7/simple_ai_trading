"""Research-only causal architectures for exact-BBO gross-return discovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Callable, Mapping, Sequence

import lightgbm as lgb
import numpy as np

from .compute import BackendInfo, resolve_backend
from .lightgbm_backend import lightgbm_backend_parameters
from .microstructure_features import (
    MICROSTRUCTURE_FEATURE_NAMES,
    MICROSTRUCTURE_FEATURE_VERSION,
    MicrostructureDataset,
    validate_microstructure_dataset,
)


GROSS_ARCHITECTURE_SCHEMA_VERSION = "exact-bbo-gross-architecture-v1"
GROSS_TARGET_MODE = "latency_aligned_midpoint_log_return_no_execution_claim"
_DAY_MS = 86_400_000


@dataclass(frozen=True)
class GrossArchitectureSpec:
    candidate_id: str
    family: str
    sequence_length: int
    hidden_dim: int
    residual_blocks: int
    dropout: float
    gmadl_weight: float
    gmadl_slope: float = 8.0
    gmadl_magnitude_power: float = 1.0

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("gross architecture candidate_id cannot be empty")
        if self.family not in {"tabular_mlp", "causal_tcn"}:
            raise ValueError("gross architecture family is unsupported")
        if self.family == "tabular_mlp" and self.sequence_length != 1:
            raise ValueError("tabular MLP must use sequence_length=1")
        if not 1 <= self.sequence_length <= 256:
            raise ValueError("gross architecture sequence length is invalid")
        if not 8 <= self.hidden_dim <= 512:
            raise ValueError("gross architecture hidden dimension is invalid")
        if not 1 <= self.residual_blocks <= 8:
            raise ValueError("gross architecture residual block count is invalid")
        values = (
            self.dropout,
            self.gmadl_weight,
            self.gmadl_slope,
            self.gmadl_magnitude_power,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("gross architecture loss settings must be finite")
        if (
            not 0.0 <= self.dropout < 0.75
            or not 0.0 <= self.gmadl_weight <= 2.0
            or self.gmadl_slope <= 0.0
            or not 0.25 <= self.gmadl_magnitude_power <= 2.0
        ):
            raise ValueError("gross architecture loss settings are outside bounds")


@dataclass(frozen=True)
class GrossForecastMetrics:
    rows: int
    exact_after_cost_eligible_rows: int
    exact_after_cost_eligible_ratio: float
    mean_actual_bps: float
    mean_prediction_bps: float
    mean_absolute_error_bps: float
    zero_baseline_mae_bps: float
    root_mean_squared_error_bps: float
    zero_baseline_rmse_bps: float
    pearson_information_coefficient: float
    spearman_information_coefficient: float
    direction_auc: float
    direction_brier: float
    prevalence_brier: float
    direction_accuracy: float
    majority_accuracy: float
    interval_80_coverage: float
    interval_crossing_rate: float
    top_rows: tuple[dict[str, object], ...]

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class GrossPredictionBatch:
    endpoint_indexes: np.ndarray
    mean_prediction_bps: np.ndarray
    direction_probability: np.ndarray
    lower_prediction_bps: np.ndarray
    upper_prediction_bps: np.ndarray

    @property
    def rows(self) -> int:
        return int(len(self.endpoint_indexes))


@dataclass(frozen=True)
class TrainedTorchGrossModel:
    schema_version: str
    spec: GrossArchitectureSpec
    feature_version: str
    feature_names: tuple[str, ...]
    target_mode: str
    backend_requested: str
    backend_kind: str
    backend_device: str
    sequence_length: int
    target_scale_bps: float
    scaler_center: np.ndarray
    scaler_scale: np.ndarray
    best_epoch: int
    training_loss: float
    tuning_loss: float
    state: Mapping[str, np.ndarray]
    model_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False


@dataclass(frozen=True)
class TrainedLightGBMGrossModel:
    schema_version: str
    model_family: str
    feature_version: str
    feature_names: tuple[str, ...]
    target_mode: str
    backend_requested: str
    backend_kind: str
    backend_device: str
    target_scale_bps: float
    mean_model: str
    direction_model: str
    mean_iteration: int
    direction_iteration: int
    model_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def gross_midpoint_log_returns_bps(dataset: MicrostructureDataset) -> np.ndarray:
    """Return a symmetric gross forecast target from latency-aligned real quotes."""

    validate_microstructure_dataset(dataset)
    if dataset.target_mode != "fixed_horizon":
        raise ValueError("gross midpoint targets require fixed-horizon quotes")
    entry_mid = (dataset.entry_bid_price + dataset.entry_ask_price) / 2.0
    exit_mid = (dataset.fixed_exit_bid_price + dataset.fixed_exit_ask_price) / 2.0
    values = np.log(exit_mid / entry_mid) * 10_000.0
    return values


def causal_cusum_event_mask(
    dataset: MicrostructureDataset,
    *,
    volatility_multiplier: float,
    minimum_threshold_bps: float,
) -> np.ndarray:
    """Select causal price events without a future-dependent activity quota."""

    multiplier = float(volatility_multiplier)
    floor = float(minimum_threshold_bps)
    if not math.isfinite(multiplier) or not math.isfinite(floor):
        raise ValueError("CUSUM settings must be finite")
    if multiplier <= 0.0 or floor <= 0.0:
        raise ValueError("CUSUM settings must be positive")
    try:
        return_index = dataset.feature_names.index("return_5s_bps")
        volatility_index = dataset.feature_names.index(
            "realized_volatility_60s_bps"
        )
    except ValueError as exc:
        raise ValueError("CUSUM event features are missing") from exc
    returns = np.asarray(dataset.features[:, return_index], dtype=np.float64)
    volatility = np.asarray(
        dataset.features[:, volatility_index],
        dtype=np.float64,
    )
    thresholds = np.maximum(
        floor,
        multiplier * np.maximum(volatility, 0.0) * math.sqrt(60.0),
    )
    if not np.all(np.isfinite(returns)) or not np.all(np.isfinite(thresholds)):
        raise ValueError("CUSUM event inputs are non-finite")
    events = np.zeros(dataset.rows, dtype=bool)
    days = dataset.decision_time_ms // _DAY_MS
    positive = 0.0
    negative = 0.0
    prior_day = int(days[0]) if dataset.rows else 0
    for index, value in enumerate(returns):
        day = int(days[index])
        if day != prior_day:
            positive = 0.0
            negative = 0.0
            prior_day = day
        positive = max(0.0, positive + float(value))
        negative = min(0.0, negative + float(value))
        threshold = float(thresholds[index])
        if positive >= threshold or negative <= -threshold:
            events[index] = True
            positive = 0.0
            negative = 0.0
    return events


def valid_sequence_endpoints(
    decision_time_ms: np.ndarray,
    endpoints: np.ndarray,
    *,
    sequence_length: int,
    cadence_seconds: int,
) -> np.ndarray:
    """Keep endpoints whose complete context is causal and cadence-contiguous."""

    times = np.asarray(decision_time_ms, dtype=np.int64)
    selected = np.asarray(endpoints, dtype=np.int64)
    length = int(sequence_length)
    cadence_ms = int(cadence_seconds) * 1_000
    if times.ndim != 1 or np.any(np.diff(times) <= 0):
        raise ValueError("sequence timestamps must be strictly increasing")
    if selected.ndim != 1 or np.any(np.diff(selected) <= 0):
        raise ValueError("sequence endpoints must be strictly increasing")
    if length <= 0 or cadence_ms <= 0:
        raise ValueError("sequence contract is invalid")
    if selected.size and (selected[0] < 0 or selected[-1] >= len(times)):
        raise ValueError("sequence endpoint lies outside the dataset")
    if length == 1:
        return selected.copy()
    breaks = np.asarray(np.diff(times) != cadence_ms, dtype=np.int64)
    break_prefix = np.concatenate(([0], np.cumsum(breaks, dtype=np.int64)))
    starts = selected - length + 1
    eligible = starts >= 0
    safe_starts = np.maximum(starts, 0)
    eligible &= (break_prefix[selected] - break_prefix[safe_starts]) == 0
    return selected[eligible]


def average_label_uniqueness(
    decision_time_ms: np.ndarray,
    target_exit_time_ms: np.ndarray,
    endpoints: np.ndarray,
) -> np.ndarray:
    """Return normalized average uniqueness for overlapping forecast labels."""

    times = np.asarray(decision_time_ms, dtype=np.int64)
    exits = np.asarray(target_exit_time_ms, dtype=np.int64)
    selected = np.asarray(endpoints, dtype=np.int64)
    if (
        times.ndim != 1
        or exits.shape != times.shape
        or selected.ndim != 1
        or selected.size == 0
        or selected[0] < 0
        or selected[-1] >= len(times)
        or np.any(np.diff(selected) <= 0)
        or np.any(np.diff(times) <= 0)
        or np.any(exits[selected] <= times[selected])
    ):
        raise ValueError("label uniqueness inputs are invalid")
    end_positions = np.searchsorted(times, exits[selected], side="right") - 1
    end_positions = np.maximum(selected, np.minimum(end_positions, len(times) - 1))
    difference = np.zeros(len(times) + 1, dtype=np.int32)
    np.add.at(difference, selected, 1)
    np.add.at(difference, end_positions + 1, -1)
    concurrency = np.cumsum(difference[:-1], dtype=np.int64)
    inverse = np.divide(
        1.0,
        concurrency,
        out=np.zeros(len(times), dtype=np.float64),
        where=concurrency > 0,
    )
    prefix = np.concatenate(([0.0], np.cumsum(inverse, dtype=np.float64)))
    duration = end_positions - selected + 1
    uniqueness = (prefix[end_positions + 1] - prefix[selected]) / duration
    if not np.all(np.isfinite(uniqueness)) or np.any(uniqueness <= 0.0):
        raise ValueError("label uniqueness calculation failed")
    return (uniqueness / np.mean(uniqueness)).astype(np.float32)


def _rank(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    order = np.argsort(source, kind="stable")
    ranks = np.empty(len(source), dtype=np.float64)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and source[order[end]] == source[order[cursor]]:
            end += 1
        ranks[order[cursor:end]] = (cursor + end - 1) / 2.0
        cursor = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if len(first) < 2 or np.std(first) <= 0.0 or np.std(second) <= 0.0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _auc(labels: np.ndarray, scores: np.ndarray) -> float:
    binary = np.asarray(labels, dtype=np.int8)
    positives = int(np.sum(binary == 1))
    negatives = int(np.sum(binary == 0))
    if positives == 0 or negatives == 0:
        return 0.5
    ranks = _rank(np.asarray(scores, dtype=np.float64)) + 1.0
    positive_rank_sum = float(np.sum(ranks[binary == 1]))
    return (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def evaluate_gross_forecast(
    dataset: MicrostructureDataset,
    actual_gross_bps: np.ndarray,
    prediction: GrossPredictionBatch,
    *,
    requested_top_rows: Sequence[int] = (100, 500, 1_000),
) -> GrossForecastMetrics:
    """Measure forecast quality and non-portfolio exact after-cost diagnostics."""

    endpoints = np.asarray(prediction.endpoint_indexes, dtype=np.int64)
    actual_full = np.asarray(actual_gross_bps, dtype=np.float64)
    if (
        actual_full.shape != (dataset.rows,)
        or endpoints.ndim != 1
        or endpoints.size == 0
        or endpoints[0] < 0
        or endpoints[-1] >= dataset.rows
        or np.any(np.diff(endpoints) <= 0)
    ):
        raise ValueError("gross forecast evaluation endpoints are invalid")
    actual = actual_full[endpoints]
    predicted = np.asarray(prediction.mean_prediction_bps, dtype=np.float64)
    probability = np.asarray(prediction.direction_probability, dtype=np.float64)
    lower_raw = np.asarray(prediction.lower_prediction_bps, dtype=np.float64)
    upper_raw = np.asarray(prediction.upper_prediction_bps, dtype=np.float64)
    if (
        any(
            values.shape != actual.shape
            for values in (predicted, probability, lower_raw, upper_raw)
        )
        or not all(
            np.all(np.isfinite(values))
            for values in (actual, predicted, probability, lower_raw, upper_raw)
        )
        or np.any((probability < 0.0) | (probability > 1.0))
    ):
        raise ValueError("gross forecast evaluation arrays are invalid")
    labels = (actual > 0.0).astype(np.int8)
    prevalence = float(np.mean(labels))
    lower = np.minimum(lower_raw, upper_raw)
    upper = np.maximum(lower_raw, upper_raw)
    signed_gross = np.where(predicted >= 0.0, actual, -actual)
    selected_net = np.where(
        predicted >= 0.0,
        dataset.long_net_bps[endpoints],
        dataset.short_net_bps[endpoints],
    )
    selected_side_eligible = np.where(
        predicted >= 0.0,
        dataset.long_liquidity_eligible[endpoints],
        dataset.short_liquidity_eligible[endpoints],
    ).astype(bool)
    eligible_indexes = np.flatnonzero(selected_side_eligible)
    if eligible_indexes.size == 0:
        raise ValueError("gross forecast has no exact after-cost eligible selected side")
    strength = np.abs(predicted)
    ranking = eligible_indexes[
        np.argsort(-strength[eligible_indexes], kind="stable")
    ]
    top_rows: list[dict[str, object]] = []
    for requested in requested_top_rows:
        count = min(int(requested), len(ranking))
        if count <= 0:
            raise ValueError("requested top-row count must be positive")
        indexes = ranking[:count]
        top_rows.append(
            {
                "requested_rows": int(requested),
                "rows": count,
                "mean_abs_prediction_bps": float(np.mean(strength[indexes])),
                "mean_signed_gross_bps": float(np.mean(signed_gross[indexes])),
                "signed_gross_positive_rate": float(
                    np.mean(signed_gross[indexes] > 0.0)
                ),
                "mean_exact_after_cost_bps": float(np.mean(selected_net[indexes])),
                "exact_after_cost_positive_rate": float(
                    np.mean(selected_net[indexes] > 0.0)
                ),
                "portfolio_claim": False,
            }
        )
    errors = predicted - actual
    return GrossForecastMetrics(
        rows=len(actual),
        exact_after_cost_eligible_rows=len(eligible_indexes),
        exact_after_cost_eligible_ratio=float(len(eligible_indexes) / len(actual)),
        mean_actual_bps=float(np.mean(actual)),
        mean_prediction_bps=float(np.mean(predicted)),
        mean_absolute_error_bps=float(np.mean(np.abs(errors))),
        zero_baseline_mae_bps=float(np.mean(np.abs(actual))),
        root_mean_squared_error_bps=float(np.sqrt(np.mean(errors**2))),
        zero_baseline_rmse_bps=float(np.sqrt(np.mean(actual**2))),
        pearson_information_coefficient=_correlation(predicted, actual),
        spearman_information_coefficient=_correlation(
            _rank(predicted),
            _rank(actual),
        ),
        direction_auc=_auc(labels, probability),
        direction_brier=float(np.mean((probability - labels) ** 2)),
        prevalence_brier=float(np.mean((prevalence - labels) ** 2)),
        direction_accuracy=float(np.mean((probability >= 0.5) == labels)),
        majority_accuracy=float(np.mean((prevalence >= 0.5) == labels)),
        interval_80_coverage=float(np.mean((actual >= lower) & (actual <= upper))),
        interval_crossing_rate=float(np.mean(lower_raw > upper_raw)),
        top_rows=tuple(top_rows),
    )


def _torch_modules():
    try:
        import torch
        from torch import nn
        from torch.nn import functional as functional
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for neural microstructure architecture research"
        ) from exc
    return torch, nn, functional


def _torch_device(backend: BackendInfo):
    torch, _nn, _functional = _torch_modules()
    if backend.kind == "directml":
        try:
            import torch_directml
        except ImportError as exc:
            raise RuntimeError("resolved DirectML backend is not importable") from exc
        return torch_directml.device()
    return torch.device(backend.device)


def _seed_torch(torch, seed: int, backend: BackendInfo) -> None:
    torch.manual_seed(int(seed))
    if backend.kind in {"cuda", "rocm"} and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if backend.kind == "directml":
        try:
            import torch_directml

            torch_directml.default_generator.manual_seed(int(seed))
        except (AttributeError, ImportError):
            pass


def _network(spec: GrossArchitectureSpec, feature_count: int):
    torch, nn, functional = _torch_modules()

    class CausalResidualBlock(nn.Module):
        def __init__(self, channels: int, dilation: int, dropout: float) -> None:
            super().__init__()
            self.padding = 2 * int(dilation)
            self.first = nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                dilation=dilation,
            )
            self.second = nn.Conv1d(
                channels,
                channels,
                kernel_size=3,
                dilation=dilation,
            )
            self.dropout = nn.Dropout(dropout)

        def forward(self, values):
            output = functional.pad(values, (self.padding, 0))
            output = self.dropout(functional.gelu(self.first(output)))
            output = functional.pad(output, (self.padding, 0))
            output = self.dropout(functional.gelu(self.second(output)))
            return values + output

    class GrossNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.family = spec.family
            self.projection = nn.Linear(feature_count, spec.hidden_dim)
            if spec.family == "causal_tcn":
                self.temporal = nn.ModuleList(
                    CausalResidualBlock(
                        spec.hidden_dim,
                        2**index,
                        spec.dropout,
                    )
                    for index in range(spec.residual_blocks)
                )
            else:
                self.temporal = nn.ModuleList()
            self.hidden = nn.Sequential(
                nn.LayerNorm(spec.hidden_dim),
                nn.Linear(spec.hidden_dim, spec.hidden_dim),
                nn.GELU(),
                nn.Dropout(spec.dropout),
            )
            self.head = nn.Linear(spec.hidden_dim, 4)

        def forward(self, values):
            output = functional.gelu(self.projection(values))
            if self.family == "causal_tcn":
                output = output.transpose(1, 2)
                for block in self.temporal:
                    output = block(output)
                output = output[:, :, -1]
            else:
                output = output[:, -1, :]
            return self.head(self.hidden(output))

    return GrossNetwork()


def _feature_scaler(
    features: np.ndarray,
    train_endpoints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features[train_endpoints], dtype=np.float64)
    center = np.median(values, axis=0)
    lower = np.quantile(values, 0.25, axis=0)
    upper = np.quantile(values, 0.75, axis=0)
    scale = (upper - lower) / 1.349
    fallback = np.std(values, axis=0)
    scale = np.where(scale > 1.0e-6, scale, fallback)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    if not np.all(np.isfinite(center)) or not np.all(np.isfinite(scale)):
        raise ValueError("gross architecture feature scaler is non-finite")
    return center.astype(np.float32), scale.astype(np.float32)


def _sequence_batch(
    features: np.ndarray,
    endpoints: np.ndarray,
    *,
    sequence_length: int,
    center: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    offsets = np.arange(sequence_length - 1, -1, -1, dtype=np.int64)
    indexes = np.asarray(endpoints, dtype=np.int64)[:, None] - offsets[None, :]
    values = np.asarray(features[indexes], dtype=np.float32)
    values = (values - center[None, None, :]) / scale[None, None, :]
    return np.clip(values, -12.0, 12.0).astype(np.float32, copy=False)


def _torch_loss(
    output,
    target,
    sample_weight,
    spec: GrossArchitectureSpec,
):
    torch, _nn, functional = _torch_modules()
    mean = output[:, 0]
    direction = output[:, 1]
    lower = output[:, 2]
    upper = output[:, 3]
    labels = (target > 0.0).to(dtype=target.dtype)
    huber = functional.smooth_l1_loss(mean, target, reduction="none", beta=0.5)
    binary = functional.softplus(direction) - labels * direction

    def pinball(prediction, quantile: float):
        error = target - prediction
        return torch.maximum(quantile * error, (quantile - 1.0) * error)

    quantile = pinball(lower, 0.10) + pinball(upper, 0.90)
    directional = -(
        torch.sigmoid(spec.gmadl_slope * target * mean) - 0.5
    ) * torch.pow(torch.abs(target) + 1.0e-6, spec.gmadl_magnitude_power)
    crossing = functional.relu(lower - upper)
    per_row = (
        huber
        + 0.25 * binary
        + 0.15 * quantile
        + spec.gmadl_weight * directional
        + 0.10 * crossing
    )
    return torch.sum(per_row * sample_weight) / torch.sum(sample_weight)


def _state_hash(
    spec: GrossArchitectureSpec,
    center: np.ndarray,
    scale: np.ndarray,
    target_scale_bps: float,
    state: Mapping[str, np.ndarray],
) -> str:
    contract = {
        "schema_version": GROSS_ARCHITECTURE_SCHEMA_VERSION,
        "spec": asdict(spec),
        "feature_version": MICROSTRUCTURE_FEATURE_VERSION,
        "feature_names": list(MICROSTRUCTURE_FEATURE_NAMES),
        "target_mode": GROSS_TARGET_MODE,
        "target_scale_bps": float(target_scale_bps),
    }
    digest = hashlib.sha256(_canonical_json(contract).encode("ascii"))
    for name, values in (
        ("scaler_center", center),
        ("scaler_scale", scale),
        *sorted(state.items()),
    ):
        array = np.ascontiguousarray(values, dtype="<f4")
        digest.update(name.encode("utf-8") + b"\x00")
        digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def train_torch_gross_model(
    dataset: MicrostructureDataset,
    actual_gross_bps: np.ndarray,
    *,
    train_endpoints: np.ndarray,
    tuning_endpoints: np.ndarray,
    spec: GrossArchitectureSpec,
    compute_backend: str,
    seed: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    train_sample_weights: np.ndarray | None = None,
    tuning_sample_weights: np.ndarray | None = None,
    progress: Callable[[int, int, float, float], None] | None = None,
) -> TrainedTorchGrossModel:
    """Train a bounded research model that cannot authorize an order."""

    validate_microstructure_dataset(dataset)
    target = np.asarray(actual_gross_bps, dtype=np.float32)
    if target.shape != (dataset.rows,) or not np.all(np.isfinite(target)):
        raise ValueError("gross architecture targets are invalid")
    train = valid_sequence_endpoints(
        dataset.decision_time_ms,
        np.asarray(train_endpoints, dtype=np.int64),
        sequence_length=spec.sequence_length,
        cadence_seconds=dataset.decision_cadence_seconds,
    )
    tuning = valid_sequence_endpoints(
        dataset.decision_time_ms,
        np.asarray(tuning_endpoints, dtype=np.int64),
        sequence_length=spec.sequence_length,
        cadence_seconds=dataset.decision_cadence_seconds,
    )
    if len(train) < 512 or len(tuning) < 256:
        raise ValueError("gross architecture split has insufficient contiguous rows")
    batch = int(batch_size)
    epochs = int(max_epochs)
    stop_patience = int(patience)
    if batch < 32 or epochs < 1 or stop_patience < 1:
        raise ValueError("gross architecture training budget is invalid")
    center, scale = _feature_scaler(dataset.features, train)
    target_scale = max(1.0, float(np.quantile(np.abs(target[train]), 0.90)))
    train_weights = (
        np.ones(len(train), dtype=np.float32)
        if train_sample_weights is None
        else np.asarray(train_sample_weights, dtype=np.float32)
    )
    tuning_weights = (
        np.ones(len(tuning), dtype=np.float32)
        if tuning_sample_weights is None
        else np.asarray(tuning_sample_weights, dtype=np.float32)
    )
    if (
        train_weights.shape != (len(train),)
        or tuning_weights.shape != (len(tuning),)
        or not np.all(np.isfinite(train_weights))
        or not np.all(np.isfinite(tuning_weights))
        or np.any(train_weights <= 0.0)
        or np.any(tuning_weights <= 0.0)
    ):
        raise ValueError("gross architecture sample weights are invalid")
    backend = resolve_backend(compute_backend)
    device = _torch_device(backend)
    torch, _nn, _functional = _torch_modules()
    _seed_torch(torch, int(seed), backend)
    model = _network(spec, len(MICROSTRUCTURE_FEATURE_NAMES)).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=8.0e-4,
        betas=(0.9, 0.99),
        eps=1.0e-7,
        foreach=False,
    )
    rng = np.random.default_rng(int(seed))

    def epoch_loss(endpoints: np.ndarray, weights: np.ndarray, training: bool) -> float:
        order = rng.permutation(len(endpoints)) if training else np.arange(len(endpoints))
        total = 0.0
        total_weight = 0.0
        model.train(training)
        for start in range(0, len(order), batch):
            positions = order[start : start + batch]
            batch_endpoints = endpoints[positions]
            x = torch.from_numpy(
                _sequence_batch(
                    dataset.features,
                    batch_endpoints,
                    sequence_length=spec.sequence_length,
                    center=center,
                    scale=scale,
                )
            ).to(device)
            y = torch.from_numpy(target[batch_endpoints] / target_scale).to(device)
            weight = torch.from_numpy(weights[positions]).to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                loss = _torch_loss(model(x), y, weight, spec)
                if not bool(torch.isfinite(loss).detach().cpu().item()):
                    raise ValueError("gross architecture loss became non-finite")
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
            weight_sum = float(np.sum(weights[positions]))
            total += float(loss.detach().cpu().item()) * weight_sum
            total_weight += weight_sum
        return total / total_weight

    best_state: dict[str, np.ndarray] | None = None
    best_epoch = 0
    best_tuning = float("inf")
    best_training = float("inf")
    stale = 0
    for epoch in range(1, epochs + 1):
        training_loss = epoch_loss(train, train_weights, True)
        tuning_loss = epoch_loss(tuning, tuning_weights, False)
        if progress is not None:
            progress(epoch, epochs, training_loss, tuning_loss)
        if tuning_loss < best_tuning - 1.0e-5:
            best_tuning = tuning_loss
            best_training = training_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().numpy().astype(np.float32, copy=True)
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= stop_patience:
                break
    assert best_state is not None
    model_sha256 = _state_hash(spec, center, scale, target_scale, best_state)
    return TrainedTorchGrossModel(
        schema_version=GROSS_ARCHITECTURE_SCHEMA_VERSION,
        spec=spec,
        feature_version=MICROSTRUCTURE_FEATURE_VERSION,
        feature_names=MICROSTRUCTURE_FEATURE_NAMES,
        target_mode=GROSS_TARGET_MODE,
        backend_requested=backend.requested,
        backend_kind=backend.kind,
        backend_device=backend.device,
        sequence_length=spec.sequence_length,
        target_scale_bps=target_scale,
        scaler_center=center,
        scaler_scale=scale,
        best_epoch=best_epoch,
        training_loss=best_training,
        tuning_loss=best_tuning,
        state=best_state,
        model_sha256=model_sha256,
    )


def predict_torch_gross_model(
    model: TrainedTorchGrossModel,
    dataset: MicrostructureDataset,
    endpoints: np.ndarray,
    *,
    compute_backend: str,
    batch_size: int,
) -> GrossPredictionBatch:
    if (
        model.trading_authority
        or model.execution_claim
        or model.profitability_claim
        or model.schema_version != GROSS_ARCHITECTURE_SCHEMA_VERSION
        or model.feature_version != dataset.feature_version
        or model.feature_names != dataset.feature_names
        or model.target_mode != GROSS_TARGET_MODE
    ):
        raise ValueError("gross neural model contract is invalid")
    if int(batch_size) <= 0:
        raise ValueError("gross neural prediction batch size must be positive")
    selected = valid_sequence_endpoints(
        dataset.decision_time_ms,
        np.asarray(endpoints, dtype=np.int64),
        sequence_length=model.sequence_length,
        cadence_seconds=dataset.decision_cadence_seconds,
    )
    if selected.size == 0:
        raise ValueError("gross neural prediction has no contiguous endpoints")
    backend = resolve_backend(compute_backend)
    device = _torch_device(backend)
    torch, _nn, _functional = _torch_modules()
    network = _network(model.spec, len(model.feature_names)).to(device)
    state = {
        name: torch.from_numpy(np.asarray(value, dtype=np.float32)).to(device)
        for name, value in model.state.items()
    }
    network.load_state_dict(state, strict=True)
    network.eval()
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(selected), int(batch_size)):
            indexes = selected[start : start + int(batch_size)]
            values = torch.from_numpy(
                _sequence_batch(
                    dataset.features,
                    indexes,
                    sequence_length=model.sequence_length,
                    center=model.scaler_center,
                    scale=model.scaler_scale,
                )
            ).to(device)
            outputs.append(network(values).detach().cpu().numpy())
    output = np.concatenate(outputs, axis=0).astype(np.float64)
    if output.shape != (len(selected), 4) or not np.all(np.isfinite(output)):
        raise ValueError("gross neural model emitted invalid predictions")
    return GrossPredictionBatch(
        endpoint_indexes=selected,
        mean_prediction_bps=output[:, 0] * model.target_scale_bps,
        direction_probability=1.0 / (1.0 + np.exp(-np.clip(output[:, 1], -40, 40))),
        lower_prediction_bps=output[:, 2] * model.target_scale_bps,
        upper_prediction_bps=output[:, 3] * model.target_scale_bps,
    )


def _lightgbm_weights(target: np.ndarray, uniqueness: np.ndarray) -> np.ndarray:
    scale = max(1.0, float(np.quantile(np.abs(target), 0.90)))
    economic = 1.0 + np.clip(np.abs(target) / scale, 0.0, 3.0)
    return (economic * uniqueness).astype(np.float32)


def _train_lightgbm_booster(
    *,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_tuning: np.ndarray,
    y_tuning: np.ndarray,
    train_weights: np.ndarray,
    tuning_weights: np.ndarray,
    parameters: Mapping[str, object],
) -> tuple[lgb.Booster, int]:
    training = lgb.Dataset(x_train, label=y_train, weight=train_weights)
    tuning = lgb.Dataset(
        x_tuning,
        label=y_tuning,
        weight=tuning_weights,
        reference=training,
    )
    booster = lgb.train(
        dict(parameters),
        training,
        num_boost_round=1_000,
        valid_sets=[tuning],
        valid_names=["tuning"],
        callbacks=[lgb.early_stopping(60, verbose=False), lgb.log_evaluation(0)],
    )
    return booster, max(1, int(booster.best_iteration or booster.current_iteration()))


def train_lightgbm_gross_baseline(
    dataset: MicrostructureDataset,
    actual_gross_bps: np.ndarray,
    *,
    train_endpoints: np.ndarray,
    tuning_endpoints: np.ndarray,
    train_uniqueness: np.ndarray,
    tuning_uniqueness: np.ndarray,
    compute_backend: str,
    seed: int,
) -> TrainedLightGBMGrossModel:
    """Train the fixed tabular benchmark used to judge neural uplift."""

    train = np.asarray(train_endpoints, dtype=np.int64)
    tuning = np.asarray(tuning_endpoints, dtype=np.int64)
    target = np.asarray(actual_gross_bps, dtype=np.float32)
    train_weight_values = np.asarray(train_uniqueness, dtype=np.float32)
    tuning_weight_values = np.asarray(tuning_uniqueness, dtype=np.float32)
    if (
        len(train) < 512
        or len(tuning) < 256
        or target.shape != (dataset.rows,)
        or train_weight_values.shape != (len(train),)
        or tuning_weight_values.shape != (len(tuning),)
        or not np.all(np.isfinite(target))
        or not np.all(np.isfinite(train_weight_values))
        or not np.all(np.isfinite(tuning_weight_values))
        or np.any(train_weight_values <= 0.0)
        or np.any(tuning_weight_values <= 0.0)
    ):
        raise ValueError("LightGBM gross baseline split is invalid")
    backend, backend_kind, backend_device = lightgbm_backend_parameters(
        compute_backend,
        int(seed),
        reproducible=True,
    )
    common = {
        **backend,
        "learning_rate": 0.025,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": max(64, min(256, len(train) // 100)),
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.01,
        "lambda_l2": 0.10,
        "max_bin": 127,
    }
    x = np.asarray(dataset.features, dtype=np.float32)
    train_weight = _lightgbm_weights(target[train], train_weight_values)
    tuning_weight = _lightgbm_weights(target[tuning], tuning_weight_values)
    mean, mean_iteration = _train_lightgbm_booster(
        x_train=x[train],
        y_train=target[train],
        x_tuning=x[tuning],
        y_tuning=target[tuning],
        train_weights=train_weight,
        tuning_weights=tuning_weight,
        parameters={**common, "objective": "huber", "metric": "l1", "alpha": 0.9},
    )
    direction, direction_iteration = _train_lightgbm_booster(
        x_train=x[train],
        y_train=(target[train] > 0.0).astype(np.float32),
        x_tuning=x[tuning],
        y_tuning=(target[tuning] > 0.0).astype(np.float32),
        train_weights=train_weight,
        tuning_weights=tuning_weight,
        parameters={**common, "objective": "binary", "metric": "binary_logloss"},
    )
    mean_string = mean.model_to_string(num_iteration=mean_iteration)
    direction_string = direction.model_to_string(num_iteration=direction_iteration)
    model_sha256 = hashlib.sha256(
        _canonical_json(
            {
                "schema_version": GROSS_ARCHITECTURE_SCHEMA_VERSION,
                "model_family": "lightgbm_tabular_huber_direction",
                "mean": mean_string,
                "direction": direction_string,
            }
        ).encode("ascii")
    ).hexdigest()
    return TrainedLightGBMGrossModel(
        schema_version=GROSS_ARCHITECTURE_SCHEMA_VERSION,
        model_family="lightgbm_tabular_huber_direction",
        feature_version=MICROSTRUCTURE_FEATURE_VERSION,
        feature_names=MICROSTRUCTURE_FEATURE_NAMES,
        target_mode=GROSS_TARGET_MODE,
        backend_requested=str(compute_backend),
        backend_kind=backend_kind,
        backend_device=backend_device,
        target_scale_bps=max(1.0, float(np.quantile(np.abs(target[train]), 0.90))),
        mean_model=mean_string,
        direction_model=direction_string,
        mean_iteration=mean_iteration,
        direction_iteration=direction_iteration,
        model_sha256=model_sha256,
    )


def predict_lightgbm_gross_model(
    model: TrainedLightGBMGrossModel,
    dataset: MicrostructureDataset,
    endpoints: np.ndarray,
) -> GrossPredictionBatch:
    if (
        model.trading_authority
        or model.execution_claim
        or model.profitability_claim
        or model.schema_version != GROSS_ARCHITECTURE_SCHEMA_VERSION
        or model.feature_version != dataset.feature_version
        or model.feature_names != dataset.feature_names
        or model.target_mode != GROSS_TARGET_MODE
    ):
        raise ValueError("gross LightGBM model contract is invalid")
    selected = np.asarray(endpoints, dtype=np.int64)
    if (
        selected.ndim != 1
        or selected.size == 0
        or selected[0] < 0
        or selected[-1] >= dataset.rows
        or np.any(np.diff(selected) <= 0)
    ):
        raise ValueError("gross LightGBM prediction endpoints are invalid")
    try:
        mean = lgb.Booster(model_str=model.mean_model)
        direction = lgb.Booster(model_str=model.direction_model)
    except lgb.basic.LightGBMError as exc:
        raise ValueError("gross LightGBM model cannot be reloaded") from exc
    x = np.asarray(dataset.features[selected], dtype=np.float32)
    predicted = np.asarray(
        mean.predict(x, num_iteration=model.mean_iteration),
        dtype=np.float64,
    )
    probability = np.asarray(
        direction.predict(x, num_iteration=model.direction_iteration),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(probability)):
        raise ValueError("gross LightGBM model emitted non-finite predictions")
    width = np.full(len(selected), model.target_scale_bps, dtype=np.float64)
    return GrossPredictionBatch(
        endpoint_indexes=selected,
        mean_prediction_bps=predicted,
        direction_probability=np.clip(probability, 0.0, 1.0),
        lower_prediction_bps=predicted - width,
        upper_prediction_bps=predicted + width,
    )


__all__ = [
    "GROSS_ARCHITECTURE_SCHEMA_VERSION",
    "GROSS_TARGET_MODE",
    "GrossArchitectureSpec",
    "GrossForecastMetrics",
    "GrossPredictionBatch",
    "TrainedLightGBMGrossModel",
    "TrainedTorchGrossModel",
    "average_label_uniqueness",
    "causal_cusum_event_mask",
    "evaluate_gross_forecast",
    "gross_midpoint_log_returns_bps",
    "predict_lightgbm_gross_model",
    "predict_torch_gross_model",
    "train_lightgbm_gross_baseline",
    "train_torch_gross_model",
    "valid_sequence_endpoints",
]
