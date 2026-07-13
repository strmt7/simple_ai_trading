"""Causal multi-horizon distributional TCN research model for Round 44."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence
import warnings

import numpy as np
from scipy.stats import spearmanr
import torch
from torch import nn
from torch.nn import functional as F

from .cross_asset_cost_data import MINUTE_MS, SYMBOLS
from .stateful_turnover_model import StatefulHourlyDataset


ROUND = 44
HORIZONS = (1, 4, 12, 24)
QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
SEEDS = (4401, 4402, 4403)
BASE_ONE_WAY_COST_BPS = 6.0
STRESS_ONE_WAY_COST_BPS = 8.0
HIDDEN_CHANNELS = 64
DILATIONS = (1, 2, 4, 8, 16, 32)
KERNEL_SIZE = 3
RECEPTIVE_FIELD = 1 + (KERNEL_SIZE - 1) * sum(DILATIONS)
WINDOW_HOURS = 384
SUPERVISED_HOURS = WINDOW_HOURS - RECEPTIVE_FIELD + 1
WINDOW_STRIDE_HOURS = 24
BATCH_SIZE = 64
MAXIMUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 12
MINIMUM_IMPROVEMENT = 1e-5
SLEEVE_FRACTION = 1.0 / len(SYMBOLS)
BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_BLOCK_HOURS = 168
FAMILYWISE_LOWER_QUANTILE = 0.025
ProgressCallback = Callable[[str, Mapping[str, object]], None]


@dataclass(frozen=True)
class ChronologicalRole:
    name: str
    start: str
    end: str


ROLES = (
    ChronologicalRole("training", "2022-01-01", "2024-03-31"),
    ChronologicalRole("early_stop", "2024-04-01", "2024-06-30"),
    ChronologicalRole("calibration", "2024-07-01", "2024-09-30"),
    ChronologicalRole("evaluation", "2024-10-01", "2025-06-30"),
)


@dataclass(frozen=True)
class DistributionalDataset:
    feature_names: tuple[str, ...]
    timestamps_ms: np.ndarray
    features: np.ndarray
    hourly_return_bps: np.ndarray
    forward_return_bps: np.ndarray
    dataset_sha256: str

    @property
    def timestamps(self) -> int:
        return int(self.timestamps_ms.size)

    @property
    def rows(self) -> int:
        return self.timestamps * len(SYMBOLS)


@dataclass(frozen=True)
class FeatureScaler:
    mean: np.ndarray
    standard_deviation: np.ndarray

    def transform(self, values: np.ndarray) -> np.ndarray:
        transformed = (values.astype(np.float64) - self.mean) / self.standard_deviation
        transformed = np.clip(transformed, -10.0, 10.0).astype(np.float32)
        if not np.isfinite(transformed).all():
            raise ValueError("Round 44 standardized features are nonfinite")
        return transformed

    def asdict(self) -> dict[str, object]:
        return {
            "mean": self.mean.tolist(),
            "standard_deviation": self.standard_deviation.tolist(),
        }


@dataclass(frozen=True)
class TargetScaler:
    center_bps: np.ndarray
    scale_bps: np.ndarray

    def normalize(self, values: np.ndarray) -> np.ndarray:
        if values.shape[-1] != len(HORIZONS):
            raise ValueError("Round 44 target normalization dimensions are invalid")
        return ((values - self.center_bps) / self.scale_bps).astype(np.float32)

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        if values.shape[-2] != len(HORIZONS):
            raise ValueError("Round 44 forecast denormalization dimensions are invalid")
        scale = self.scale_bps.reshape(
            *((1,) * (values.ndim - 2)), len(HORIZONS), 1
        )
        center = self.center_bps.reshape(
            *((1,) * (values.ndim - 2)), len(HORIZONS), 1
        )
        return values * scale + center

    def asdict(self) -> dict[str, object]:
        return {
            "center_bps": self.center_bps.tolist(),
            "scale_bps": self.scale_bps.tolist(),
        }


@dataclass(frozen=True)
class TCNArtifact:
    seed: int
    epochs: int
    best_epoch: int
    best_early_stop_pinball: float
    parameter_count: int
    backend_kind: str
    backend_device: str
    path: str
    bytes: int
    sha256: str
    reload_max_abs_prediction_error: float
    warning_count: int

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TCNForecastBundle:
    seed_predictions_bps: np.ndarray
    ensemble_predictions_bps: np.ndarray
    artifacts: tuple[TCNArtifact, ...]
    feature_scaler: FeatureScaler
    target_scaler: TargetScaler
    backend_kind: str
    backend_device: str


@dataclass(frozen=True)
class PlannedTrade:
    trade_id: str
    symbol: str
    symbol_index: int
    decision_index: int
    decision_time_ms: int
    side: int
    horizon_hours: int
    selected_lower_quartile_bps: float
    expected_after_cost_bps: float

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PlannedReplay:
    scenario: str
    one_way_cost_bps: float
    trades: tuple[PlannedTrade, ...]
    timestamps_ms: np.ndarray
    positions: np.ndarray
    symbol_return_bps: np.ndarray
    portfolio_return_bps: np.ndarray
    metrics: Mapping[str, object]


def _date_ms(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1_000)


def _end_exclusive_ms(value: str) -> int:
    return _date_ms(value) + 86_400_000


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _array_identity(*arrays: np.ndarray, names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in names:
        digest.update(str(name).encode("ascii"))
        digest.update(b"\0")
    for values in arrays:
        contiguous = np.ascontiguousarray(values)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def compounded_forward_returns(
    hourly_return_bps: np.ndarray,
    horizons: Sequence[int] = HORIZONS,
) -> np.ndarray:
    """Build exact compounded forward returns from contiguous hourly returns."""

    values = np.asarray(hourly_return_bps, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(SYMBOLS):
        raise ValueError("Round 44 hourly return matrix has invalid dimensions")
    fractions = values / 10_000.0
    if not np.isfinite(fractions).all() or np.any(fractions <= -1.0):
        raise ValueError("Round 44 hourly returns cannot be compounded")
    cumulative = np.vstack(
        (
            np.zeros((1, fractions.shape[1]), dtype=np.float64),
            np.cumsum(np.log1p(fractions), axis=0),
        )
    )
    output = np.full(
        (values.shape[0], values.shape[1], len(horizons)),
        np.nan,
        dtype=np.float32,
    )
    for horizon_index, horizon in enumerate(horizons):
        if horizon <= 0 or horizon > values.shape[0]:
            raise ValueError(f"Round 44 horizon is invalid: {horizon}")
        count = values.shape[0] - horizon + 1
        compounded = np.expm1(cumulative[horizon:] - cumulative[:-horizon])
        output[:count, :, horizon_index] = (10_000.0 * compounded).astype(np.float32)
    return output


def build_distributional_dataset(
    source: StatefulHourlyDataset,
) -> DistributionalDataset:
    """Reshape the verified Round 43 matrix and add multi-horizon targets."""

    symbol_count = len(SYMBOLS)
    if source.rows % symbol_count or source.baseline_features.shape[1] != 71:
        raise ValueError("Round 44 source rows or feature count are invalid")
    timestamps = source.decision_time_ms.reshape(-1, symbol_count)
    symbols = source.symbol_index.reshape(-1, symbol_count)
    if not np.all(timestamps == timestamps[:, :1]) or not np.array_equal(
        symbols,
        np.tile(np.arange(symbol_count, dtype=np.int8), (symbols.shape[0], 1)),
    ):
        raise ValueError("Round 44 source is not a complete chronological symbol grid")
    decision_times = timestamps[:, 0].copy()
    if decision_times.size < 2 or not np.all(np.diff(decision_times) == 3_600_000):
        raise ValueError("Round 44 source does not have a contiguous hourly grid")
    features = source.baseline_features.reshape(
        -1, symbol_count, source.baseline_features.shape[1]
    ).astype(np.float32, copy=True)
    hourly = source.signed_pre_transition_utility_bps.reshape(-1, symbol_count).astype(
        np.float32,
        copy=True,
    )
    if not np.isfinite(features).all() or not np.isfinite(hourly).all():
        raise ValueError("Round 44 source matrix contains nonfinite values")
    forward = compounded_forward_returns(hourly)
    identity = _array_identity(
        decision_times,
        features,
        hourly,
        forward,
        names=source.feature_names
        + tuple(f"forward_return_{value}h_bps" for value in HORIZONS),
    )
    return DistributionalDataset(
        feature_names=source.feature_names,
        timestamps_ms=decision_times,
        features=features,
        hourly_return_bps=hourly,
        forward_return_bps=forward,
        dataset_sha256=identity,
    )


def role_mask(dataset: DistributionalDataset, role_name: str) -> np.ndarray:
    role = next((value for value in ROLES if value.name == role_name), None)
    if role is None:
        raise KeyError(role_name)
    start = _date_ms(role.start)
    end = _end_exclusive_ms(role.end)
    target_end = dataset.timestamps_ms + (max(HORIZONS) * 60 + 1) * MINUTE_MS
    complete_targets = np.all(np.isfinite(dataset.forward_return_bps), axis=(1, 2))
    mask = (
        (dataset.timestamps_ms >= start)
        & (dataset.timestamps_ms < end)
        & (target_end < end)
        & complete_targets
    )
    if not np.any(mask):
        raise ValueError(f"Round 44 role is empty or target-incomplete: {role_name}")
    return mask


def fit_feature_scaler(
    dataset: DistributionalDataset,
    mask: np.ndarray,
) -> FeatureScaler:
    selected = dataset.features[mask].reshape(-1, dataset.features.shape[-1]).astype(
        np.float64
    )
    mean = np.mean(selected, axis=0)
    standard_deviation = np.std(selected, axis=0)
    standard_deviation = np.maximum(standard_deviation, 1e-6)
    if not np.isfinite(mean).all() or not np.isfinite(standard_deviation).all():
        raise ValueError("Round 44 feature scaler is nonfinite")
    return FeatureScaler(mean=mean, standard_deviation=standard_deviation)


def fit_target_scaler(
    dataset: DistributionalDataset,
    mask: np.ndarray,
) -> TargetScaler:
    selected = dataset.forward_return_bps[mask].reshape(-1, len(HORIZONS)).astype(
        np.float64
    )
    center = np.median(selected, axis=0)
    mad = np.median(np.abs(selected - center), axis=0)
    scale = np.maximum(1.4826 * mad, 1.0)
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError("Round 44 target scaler is nonfinite")
    return TargetScaler(center_bps=center, scale_bps=scale)


class CausalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_padding = (KERNEL_SIZE - 1) * dilation
        self.temporal = nn.Conv1d(
            channels,
            channels,
            kernel_size=KERNEL_SIZE,
            dilation=dilation,
        )
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = F.pad(values, (self.left_padding, 0))
        values = self.temporal(values)
        values = F.gelu(values)
        values = self.dropout(values)
        values = self.pointwise(values)
        values = self.dropout(values)
        return F.gelu(values + residual)


class DistributionalTCN(nn.Module):
    """Small shared causal encoder with monotone multi-horizon quantiles."""

    def __init__(self, input_channels: int = 71, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.projection = nn.Conv1d(input_channels, HIDDEN_CHANNELS, kernel_size=1)
        self.blocks = nn.ModuleList(
            CausalResidualBlock(HIDDEN_CHANNELS, dilation, dropout)
            for dilation in DILATIONS
        )
        self.head = nn.Conv1d(
            HIDDEN_CHANNELS,
            len(HORIZONS) * len(QUANTILES),
            kernel_size=1,
        )
        with torch.no_grad():
            if self.head.bias is not None:
                reshaped = self.head.bias.reshape(len(HORIZONS), len(QUANTILES))
                reshaped[:, 1:] = -2.0
                reshaped[:, 0] = 0.0

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != self.input_channels:
            raise ValueError("Round 44 TCN input dimensions are invalid")
        encoded = F.gelu(self.projection(values))
        for block in self.blocks:
            encoded = block(encoded)
        raw = self.head(encoded).reshape(
            values.shape[0], len(HORIZONS), len(QUANTILES), values.shape[-1]
        )
        median = raw[:, :, 0, :]
        lower_near = F.softplus(raw[:, :, 1, :])
        lower_far = F.softplus(raw[:, :, 2, :])
        upper_near = F.softplus(raw[:, :, 3, :])
        upper_far = F.softplus(raw[:, :, 4, :])
        q25 = median - lower_near
        q10 = q25 - lower_far
        q75 = median + upper_near
        q90 = q75 + upper_far
        return torch.stack((q10, q25, median, q75, q90), dim=2)


class ExplicitAdamW:
    """Small non-foreach AdamW implementation for warning-free DirectML updates."""

    def __init__(
        self,
        parameters: Sequence[torch.nn.Parameter],
        *,
        learning_rate: float,
        weight_decay: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.parameters = tuple(parameters)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.iteration = 0
        self.first_moment = [torch.zeros_like(value) for value in self.parameters]
        self.second_moment = [torch.zeros_like(value) for value in self.parameters]

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            if set_to_none:
                parameter.grad = None
            else:
                parameter.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        self.iteration += 1
        first_correction = 1.0 - self.beta1**self.iteration
        second_correction = 1.0 - self.beta2**self.iteration
        step_size = self.learning_rate / first_correction
        decay = 1.0 - self.learning_rate * self.weight_decay
        for parameter, first, second in zip(
            self.parameters,
            self.first_moment,
            self.second_moment,
            strict=True,
        ):
            gradient = parameter.grad
            if gradient is None:
                continue
            parameter.mul_(decay)
            first.mul_(self.beta1).add_(gradient, alpha=1.0 - self.beta1)
            second.mul_(self.beta2).addcmul_(
                gradient,
                gradient,
                value=1.0 - self.beta2,
            )
            denominator = second.sqrt() / math.sqrt(second_correction)
            denominator.add_(self.epsilon)
            parameter.addcdiv_(first, denominator, value=-step_size)


def pinball_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if predictions.ndim != 4 or targets.ndim != 3:
        raise ValueError("Round 44 pinball tensors have invalid dimensions")
    quantiles = torch.tensor(
        QUANTILES,
        dtype=predictions.dtype,
        device=predictions.device,
    ).reshape(1, 1, -1, 1)
    errors = targets.unsqueeze(2) - predictions
    return torch.maximum(quantiles * errors, (quantiles - 1.0) * errors).mean()


def directml_preflight() -> tuple[object, dict[str, object]]:
    """Require an actual warning-free DirectML convolution backward pass."""

    try:
        import torch_directml  # type: ignore
    except ImportError as exc:  # pragma: no cover - host dependent
        raise RuntimeError("Round 44 DirectML is unavailable") from exc
    device = torch_directml.device()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = DistributionalTCN().to(device)
        optimizer = ExplicitAdamW(
            tuple(model.parameters()),
            learning_rate=1e-3,
            weight_decay=1e-4,
        )
        generator = np.random.default_rng(SEEDS[0])
        values = torch.from_numpy(
            generator.normal(size=(8, 71, 160)).astype(np.float32)
        ).to(device)
        targets = torch.from_numpy(
            generator.normal(size=(8, len(HORIZONS), 160)).astype(np.float32)
        ).to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = pinball_loss(model(values), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
        optimizer.step()
        loss_value = float(loss.detach().cpu().item())
    messages = [str(item.message) for item in caught]
    fallback = [
        item
        for item in messages
        if "not currently supported on the DML backend" in item
        or "fall back to run on the CPU" in item
    ]
    if not math.isfinite(loss_value) or fallback:
        raise RuntimeError(f"Round 44 DirectML preflight failed: {fallback}")
    return device, {
        "backend_kind": "directml",
        "backend_device": str(device),
        "torch_version": str(torch.__version__),
        "torch_directml_version": str(getattr(torch_directml, "__version__", "unknown")),
        "preflight_loss": loss_value,
        "warning_count": len(messages),
        "cpu_fallback_warning_count": 0,
    }


def cpu_device() -> tuple[object, dict[str, object]]:
    device = torch.device("cpu")
    return device, {
        "backend_kind": "cpu",
        "backend_device": str(device),
        "torch_version": str(torch.__version__),
        "torch_directml_version": None,
        "preflight_loss": None,
        "warning_count": 0,
        "cpu_fallback_warning_count": 0,
    }


def _training_windows(mask: np.ndarray) -> np.ndarray:
    indexes = np.flatnonzero(mask)
    if indexes.size < WINDOW_HOURS or not np.all(np.diff(indexes) == 1):
        raise ValueError("Round 44 training role is not a sufficiently long contiguous span")
    first = int(indexes[0])
    last_start = int(indexes[-1]) - WINDOW_HOURS + 1
    starts = np.arange(first, last_start + 1, WINDOW_STRIDE_HOURS, dtype=np.int64)
    if starts.size == 0:
        raise ValueError("Round 44 has no training windows")
    return starts


def _predict_all(
    model: DistributionalTCN,
    normalized_features: np.ndarray,
    target_scaler: TargetScaler,
    device: object,
) -> np.ndarray:
    model.eval()
    input_values = normalized_features.transpose(1, 2, 0)
    with torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(input_values)).to(device)
        normalized = model(tensor).detach().cpu().numpy()
    normalized = normalized.transpose(3, 0, 1, 2)
    predictions = target_scaler.denormalize(normalized).astype(np.float32)
    if predictions.shape != (
        normalized_features.shape[0],
        len(SYMBOLS),
        len(HORIZONS),
        len(QUANTILES),
    ) or not np.isfinite(predictions).all():
        raise ValueError("Round 44 model predictions are invalid")
    return predictions


def _numpy_pinball(
    actual: np.ndarray,
    predictions: np.ndarray,
) -> float:
    errors = actual[..., None] - predictions
    quantiles = np.asarray(QUANTILES, dtype=np.float64)
    return float(np.mean(np.maximum(quantiles * errors, (quantiles - 1.0) * errors)))


def _role_pinball(
    dataset: DistributionalDataset,
    predictions: np.ndarray,
    mask: np.ndarray,
    target_scaler: TargetScaler,
) -> float:
    actual = target_scaler.normalize(dataset.forward_return_bps[mask])
    normalized_predictions = (
        predictions[mask] - target_scaler.center_bps[None, None, :, None]
    ) / target_scaler.scale_bps[None, None, :, None]
    return _numpy_pinball(actual, normalized_predictions)


def train_distributional_tcn_seed(
    dataset: DistributionalDataset,
    normalized_features: np.ndarray,
    target_scaler: TargetScaler,
    *,
    seed: int,
    artifact_path: Path,
    device: object,
    backend_kind: str,
    backend_device: str,
    progress: ProgressCallback | None = None,
) -> tuple[DistributionalTCN, TCNArtifact, np.ndarray]:
    """Fit one fixed seed and restore the best chronological validation epoch."""

    if seed not in SEEDS:
        raise ValueError(f"Round 44 seed is not frozen: {seed}")
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    model = DistributionalTCN(input_channels=normalized_features.shape[-1]).to(device)
    optimizer = ExplicitAdamW(
        tuple(model.parameters()),
        learning_rate=1e-3,
        weight_decay=1e-4,
    )
    training_mask = role_mask(dataset, "training")
    validation_mask = role_mask(dataset, "early_stop")
    starts = _training_windows(training_mask)
    pairs = np.asarray(
        [(int(start), symbol) for start in starts for symbol in range(len(SYMBOLS))],
        dtype=np.int64,
    )
    normalized_targets = target_scaler.normalize(dataset.forward_return_bps)
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    warning_messages: list[str] = []
    epochs_run = 0
    for epoch in range(1, MAXIMUM_EPOCHS + 1):
        epochs_run = epoch
        order = generator.permutation(pairs.shape[0])
        model.train()
        batch_losses: list[float] = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for offset in range(0, order.size, BATCH_SIZE):
                selected = pairs[order[offset : offset + BATCH_SIZE]]
                batch_features = np.stack(
                    [
                        normalized_features[
                            start : start + WINDOW_HOURS, symbol, :
                        ].T
                        for start, symbol in selected
                    ]
                ).astype(np.float32, copy=False)
                batch_targets = np.stack(
                    [
                        normalized_targets[
                            start : start + WINDOW_HOURS, symbol, :
                        ].T
                        for start, symbol in selected
                    ]
                ).astype(np.float32, copy=False)
                values = torch.from_numpy(np.ascontiguousarray(batch_features)).to(
                    device
                )
                targets = torch.from_numpy(np.ascontiguousarray(batch_targets)).to(
                    device
                )
                optimizer.zero_grad(set_to_none=True)
                predictions = model(values)[..., -SUPERVISED_HOURS:]
                loss = pinball_loss(predictions, targets[..., -SUPERVISED_HOURS:])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, foreach=False
                )
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu().item()))
            full_predictions = _predict_all(
                model, normalized_features, target_scaler, device
            )
            validation_loss = _role_pinball(
                dataset, full_predictions, validation_mask, target_scaler
            )
        warning_messages.extend(str(item.message) for item in caught)
        training_loss = float(np.mean(batch_losses))
        if not math.isfinite(training_loss) or not math.isfinite(validation_loss):
            raise RuntimeError("Round 44 training produced a nonfinite loss")
        improved = validation_loss < best_loss - MINIMUM_IMPROVEMENT
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if progress is not None:
            progress(
                "round44_tcn_epoch",
                {
                    "seed": seed,
                    "epoch": epoch,
                    "training_pinball": training_loss,
                    "early_stop_pinball": validation_loss,
                    "best_early_stop_pinball": best_loss,
                    "best_epoch": best_epoch,
                    "stale_epochs": stale_epochs,
                },
            )
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("Round 44 training did not produce a best model state")
    model.load_state_dict(best_state)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        best_predictions = _predict_all(
            model, normalized_features, target_scaler, device
        )
    warning_messages.extend(str(item.message) for item in caught)
    fallback = [
        item
        for item in warning_messages
        if "not currently supported on the DML backend" in item
        or "fall back to run on the CPU" in item
    ]
    if fallback:
        raise RuntimeError(f"Round 44 training used CPU fallback: {fallback}")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, artifact_path)
    reloaded = DistributionalTCN(input_channels=normalized_features.shape[-1]).to(device)
    reloaded.load_state_dict(
        torch.load(artifact_path, map_location="cpu", weights_only=True)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reload_predictions = _predict_all(
            reloaded, normalized_features, target_scaler, device
        )
    warning_messages.extend(str(item.message) for item in caught)
    reload_fallback = [
        item
        for item in warning_messages
        if "not currently supported on the DML backend" in item
        or "fall back to run on the CPU" in item
    ]
    if reload_fallback:
        raise RuntimeError(
            f"Round 44 artifact reload used CPU fallback: {reload_fallback}"
        )
    evaluation_probe = np.flatnonzero(role_mask(dataset, "evaluation"))[:512]
    reload_error = float(
        np.max(
            np.abs(
                best_predictions[evaluation_probe]
                - reload_predictions[evaluation_probe]
            )
        )
    )
    if not math.isfinite(reload_error) or reload_error > 1e-6:
        raise RuntimeError(f"Round 44 artifact reload error is {reload_error}")
    parameter_count = sum(value.numel() for value in model.parameters())
    artifact = TCNArtifact(
        seed=seed,
        epochs=epochs_run,
        best_epoch=best_epoch,
        best_early_stop_pinball=best_loss,
        parameter_count=parameter_count,
        backend_kind=backend_kind,
        backend_device=backend_device,
        path=str(artifact_path),
        bytes=artifact_path.stat().st_size,
        sha256=_file_sha256(artifact_path),
        reload_max_abs_prediction_error=reload_error,
        warning_count=len(warning_messages),
    )
    return model, artifact, best_predictions


def train_distributional_tcn_ensemble(
    dataset: DistributionalDataset,
    *,
    model_dir: Path,
    compute_backend: str = "directml",
    progress: ProgressCallback | None = None,
) -> tuple[TCNForecastBundle, dict[str, object]]:
    training_mask = role_mask(dataset, "training")
    feature_scaler = fit_feature_scaler(dataset, training_mask)
    target_scaler = fit_target_scaler(dataset, training_mask)
    normalized_features = feature_scaler.transform(dataset.features)
    if compute_backend == "directml":
        device, preflight = directml_preflight()
    elif compute_backend == "cpu":
        device, preflight = cpu_device()
    else:
        raise ValueError(f"Round 44 compute backend is invalid: {compute_backend}")
    seed_predictions: list[np.ndarray] = []
    artifacts: list[TCNArtifact] = []
    for seed in SEEDS:
        if progress is not None:
            progress("round44_tcn_seed", {"seed": seed, "status": "started"})
        _model, artifact, predictions = train_distributional_tcn_seed(
            dataset,
            normalized_features,
            target_scaler,
            seed=seed,
            artifact_path=model_dir / f"round44_distributional_tcn_seed_{seed}.pt",
            device=device,
            backend_kind=str(preflight["backend_kind"]),
            backend_device=str(preflight["backend_device"]),
            progress=progress,
        )
        artifacts.append(artifact)
        seed_predictions.append(predictions)
        if progress is not None:
            progress(
                "round44_tcn_seed",
                {
                    "seed": seed,
                    "status": "complete",
                    "best_epoch": artifact.best_epoch,
                    "artifact_sha256": artifact.sha256,
                },
            )
    stacked = np.stack(seed_predictions).astype(np.float32)
    ensemble = np.median(stacked, axis=0).astype(np.float32)
    if np.any(np.diff(ensemble, axis=-1) < -1e-7):
        raise RuntimeError("Round 44 ensemble quantiles crossed")
    return (
        TCNForecastBundle(
            seed_predictions_bps=stacked,
            ensemble_predictions_bps=ensemble,
            artifacts=tuple(artifacts),
            feature_scaler=feature_scaler,
            target_scaler=target_scaler,
            backend_kind=str(preflight["backend_kind"]),
            backend_device=str(preflight["backend_device"]),
        ),
        preflight,
    )


def _month_labels(timestamps_ms: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            datetime.fromtimestamp(value / 1_000.0, UTC).strftime("%Y-%m")
            for value in timestamps_ms
        ]
    )


def _finite_spearman(actual: np.ndarray, forecast: np.ndarray) -> float:
    if actual.size < 3 or np.std(actual) <= 0.0 or np.std(forecast) <= 0.0:
        return 0.0
    value = float(spearmanr(actual, forecast).statistic)
    return value if math.isfinite(value) else 0.0


def forecast_diagnostics(
    dataset: DistributionalDataset,
    bundle: TCNForecastBundle,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Measure forecast skill, calibration, and stochastic stability."""

    training_mask = role_mask(dataset, "training")
    baseline_quantiles = np.quantile(
        dataset.forward_return_bps[training_mask].reshape(-1, len(HORIZONS)),
        QUANTILES,
        axis=0,
    ).T
    rows: list[dict[str, object]] = []
    evaluation_mask = role_mask(dataset, "evaluation")
    months = _month_labels(dataset.timestamps_ms)
    horizon_summary: list[dict[str, object]] = []
    for horizon_index, horizon in enumerate(HORIZONS):
        selected_actual = dataset.forward_return_bps[
            evaluation_mask, :, horizon_index
        ].astype(np.float64)
        selected_prediction = bundle.ensemble_predictions_bps[
            evaluation_mask, :, horizon_index, :
        ].astype(np.float64)
        baseline = np.broadcast_to(
            baseline_quantiles[horizon_index], selected_prediction.shape
        )
        model_pinball = _numpy_pinball(selected_actual, selected_prediction)
        baseline_pinball = _numpy_pinball(selected_actual, baseline)
        skill = 1.0 - model_pinball / baseline_pinball
        median_spearman = _finite_spearman(
            selected_actual.ravel(), selected_prediction[..., 2].ravel()
        )
        coverage_80 = float(
            np.mean(
                (selected_actual >= selected_prediction[..., 0])
                & (selected_actual <= selected_prediction[..., 4])
            )
        )
        coverage_50 = float(
            np.mean(
                (selected_actual >= selected_prediction[..., 1])
                & (selected_actual <= selected_prediction[..., 3])
            )
        )
        positive_months = 0
        for month in sorted(set(months[evaluation_mask])):
            month_mask = evaluation_mask & (months == month)
            actual_month = dataset.forward_return_bps[
                month_mask, :, horizon_index
            ].astype(np.float64)
            prediction_month = bundle.ensemble_predictions_bps[
                month_mask, :, horizon_index, 2
            ].astype(np.float64)
            association = _finite_spearman(
                actual_month.ravel(), prediction_month.ravel()
            )
            positive_months += int(association > 0.0)
            rows.append(
                {
                    "role": "evaluation",
                    "period": month,
                    "symbol": "ALL",
                    "horizon_hours": horizon,
                    "rows": int(actual_month.size),
                    "pinball_loss": _numpy_pinball(
                        actual_month,
                        bundle.ensemble_predictions_bps[
                            month_mask, :, horizon_index, :
                        ],
                    ),
                    "baseline_pinball_loss": _numpy_pinball(
                        actual_month,
                        np.broadcast_to(
                            baseline_quantiles[horizon_index],
                            (*actual_month.shape, len(QUANTILES)),
                        ),
                    ),
                    "median_spearman": association,
                    "coverage_80": float(
                        np.mean(
                            (
                                actual_month
                                >= bundle.ensemble_predictions_bps[
                                    month_mask, :, horizon_index, 0
                                ]
                            )
                            & (
                                actual_month
                                <= bundle.ensemble_predictions_bps[
                                    month_mask, :, horizon_index, 4
                                ]
                            )
                        )
                    ),
                    "coverage_50": float(
                        np.mean(
                            (
                                actual_month
                                >= bundle.ensemble_predictions_bps[
                                    month_mask, :, horizon_index, 1
                                ]
                            )
                            & (
                                actual_month
                                <= bundle.ensemble_predictions_bps[
                                    month_mask, :, horizon_index, 3
                                ]
                            )
                        )
                    ),
                }
            )
        horizon_summary.append(
            {
                "horizon_hours": horizon,
                "model_pinball_loss_bps": model_pinball,
                "baseline_pinball_loss_bps": baseline_pinball,
                "pinball_skill": skill,
                "pooled_median_spearman": median_spearman,
                "positive_monthly_median_spearman_count": positive_months,
                "evaluation_month_count": len(set(months[evaluation_mask])),
                "coverage_80": coverage_80,
                "coverage_50": coverage_50,
            }
        )
    stability: list[dict[str, object]] = []
    minimum_correlation = 1.0
    for left in range(len(SEEDS)):
        for right in range(left + 1, len(SEEDS)):
            for horizon_index, horizon in enumerate(HORIZONS):
                correlation = _finite_spearman(
                    bundle.seed_predictions_bps[
                        left, evaluation_mask, :, horizon_index, 2
                    ].ravel(),
                    bundle.seed_predictions_bps[
                        right, evaluation_mask, :, horizon_index, 2
                    ].ravel(),
                )
                minimum_correlation = min(minimum_correlation, correlation)
                stability.append(
                    {
                        "left_seed": SEEDS[left],
                        "right_seed": SEEDS[right],
                        "horizon_hours": horizon,
                        "median_prediction_spearman": correlation,
                    }
                )
    skill_count = sum(row["pinball_skill"] >= 0.01 for row in horizon_summary)
    association_count = sum(
        row["pooled_median_spearman"] > 0.0
        and row["positive_monthly_median_spearman_count"] >= 5
        for row in horizon_summary
    )
    coverage_passed = all(
        0.72 <= row["coverage_80"] <= 0.88
        and 0.42 <= row["coverage_50"] <= 0.58
        for row in horizon_summary
    )
    crossing_count = int(
        np.count_nonzero(
            np.diff(bundle.ensemble_predictions_bps[evaluation_mask], axis=-1)
            < -1e-7
        )
    )
    reasons: list[str] = []
    if skill_count < 3:
        reasons.append("fewer_than_three_horizons_beat_unconditional_pinball_by_1pct")
    if association_count < 3:
        reasons.append("fewer_than_three_horizons_have_stable_positive_rank_association")
    if not coverage_passed:
        reasons.append("central_interval_coverage_outside_frozen_bounds")
    if minimum_correlation < 0.5:
        reasons.append("seed_prediction_stability_below_0_5")
    if crossing_count:
        reasons.append("quantile_crossing_detected")
    gate = {
        "passed": not reasons,
        "reasons": reasons,
        "horizons_with_required_pinball_skill": skill_count,
        "horizons_with_required_rank_stability": association_count,
        "coverage_passed": coverage_passed,
        "minimum_pairwise_seed_median_prediction_spearman": minimum_correlation,
        "quantile_crossing_count": crossing_count,
    }
    return rows, stability, {"horizons": horizon_summary, "gate": gate}


def select_planned_trades(
    dataset: DistributionalDataset,
    predictions_bps: np.ndarray,
) -> tuple[PlannedTrade, ...]:
    """Create one immutable non-overlapping base ledger from lower quartiles."""

    evaluation_mask = role_mask(dataset, "evaluation")
    evaluation_indexes = np.flatnonzero(evaluation_mask)
    first = int(evaluation_indexes[0])
    last = int(evaluation_indexes[-1])
    trades: list[PlannedTrade] = []
    for symbol_index, symbol in enumerate(SYMBOLS):
        index = first
        while index <= last:
            choices: list[tuple[float, int, int, float]] = []
            for horizon_index, horizon in enumerate(HORIZONS):
                if index + horizon - 1 > last:
                    continue
                long_lower = float(
                    predictions_bps[index, symbol_index, horizon_index, 1]
                )
                short_lower = -float(
                    predictions_bps[index, symbol_index, horizon_index, 3]
                )
                choices.append(
                    (
                        long_lower - 2.0 * BASE_ONE_WAY_COST_BPS,
                        -horizon,
                        1,
                        long_lower,
                    )
                )
                choices.append(
                    (
                        short_lower - 2.0 * BASE_ONE_WAY_COST_BPS,
                        -horizon,
                        -1,
                        short_lower,
                    )
                )
            if not choices:
                break
            best = max(choices)
            expected_after_cost, negative_horizon, side, lower_quartile = best
            horizon = -negative_horizon
            if expected_after_cost <= 0.0:
                index += 1
                continue
            identity = hashlib.sha256(
                f"{symbol}|{int(dataset.timestamps_ms[index])}|{side}|{horizon}".encode(
                    "ascii"
                )
            ).hexdigest()
            trades.append(
                PlannedTrade(
                    trade_id=identity,
                    symbol=symbol,
                    symbol_index=symbol_index,
                    decision_index=index,
                    decision_time_ms=int(dataset.timestamps_ms[index]),
                    side=side,
                    horizon_hours=horizon,
                    selected_lower_quartile_bps=lower_quartile,
                    expected_after_cost_bps=expected_after_cost,
                )
            )
            index += horizon
    trades.sort(key=lambda item: (item.decision_time_ms, item.symbol_index))
    return tuple(trades)


def _circular_block_bootstrap(
    values: np.ndarray,
    *,
    seed: int,
) -> dict[str, float | int]:
    generator = np.random.default_rng(seed)
    block_count = math.ceil(values.size / BOOTSTRAP_BLOCK_HOURS)
    offsets = np.arange(BOOTSTRAP_BLOCK_HOURS, dtype=np.int64)
    means = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for sample in range(BOOTSTRAP_SAMPLES):
        starts = generator.integers(0, values.size, size=block_count)
        indexes = (starts[:, None] + offsets[None, :]) % values.size
        means[sample] = float(np.mean(values[indexes.ravel()[: values.size]]))
    lower, median, upper = np.quantile(
        means,
        (
            FAMILYWISE_LOWER_QUANTILE,
            0.5,
            1.0 - FAMILYWISE_LOWER_QUANTILE,
        ),
    )
    return {
        "lower_bps": float(lower),
        "median_bps": float(median),
        "upper_bps": float(upper),
        "samples": BOOTSTRAP_SAMPLES,
        "block_hours": BOOTSTRAP_BLOCK_HOURS,
        "lower_quantile": FAMILYWISE_LOWER_QUANTILE,
    }


def replay_planned_trades(
    dataset: DistributionalDataset,
    trades: Sequence[PlannedTrade],
    *,
    scenario: str,
    one_way_cost_bps: float,
) -> PlannedReplay:
    """Reprice an immutable ledger without changing any selected action."""

    evaluation_mask = role_mask(dataset, "evaluation")
    indexes = np.flatnonzero(evaluation_mask)
    first = int(indexes[0])
    last = int(indexes[-1])
    timestamps = dataset.timestamps_ms[indexes].copy()
    positions = np.zeros((indexes.size, len(SYMBOLS)), dtype=np.int8)
    symbol_return_bps = np.zeros((indexes.size, len(SYMBOLS)), dtype=np.float64)
    occupied = np.zeros_like(positions, dtype=bool)
    trade_net_bps: list[float] = []
    for trade in trades:
        start = trade.decision_index
        stop = start + trade.horizon_hours
        if start < first or stop - 1 > last:
            raise ValueError("Round 44 trade extends beyond evaluation")
        local = slice(start - first, stop - first)
        symbol = trade.symbol_index
        if np.any(occupied[local, symbol]):
            raise ValueError("Round 44 trade ledger overlaps within a symbol")
        occupied[local, symbol] = True
        positions[local, symbol] = trade.side
        raw = (
            trade.side
            * dataset.hourly_return_bps[start:stop, symbol].astype(np.float64)
        )
        raw[0] -= one_way_cost_bps
        raw[-1] -= one_way_cost_bps
        symbol_return_bps[local, symbol] = raw * SLEEVE_FRACTION
        trade_net_bps.append(float(np.sum(raw)))
    portfolio_bps = np.sum(symbol_return_bps, axis=1)
    returns = portfolio_bps / 10_000.0
    equity = np.cumprod(1.0 + returns)
    if not np.isfinite(equity).all() or np.any(equity <= 0.0):
        raise ValueError("Round 44 replay produced invalid equity")
    equity_with_start = np.concatenate(([1.0], equity))
    peaks = np.maximum.accumulate(equity_with_start)
    maximum_drawdown = float(np.max(1.0 - equity_with_start / peaks))
    positive = float(np.sum(portfolio_bps[portfolio_bps > 0.0]))
    negative = float(-np.sum(portfolio_bps[portfolio_bps < 0.0]))
    months = _month_labels(timestamps)
    monthly: list[dict[str, object]] = []
    for month in sorted(set(months)):
        selected = months == month
        monthly.append(
            {
                "month": month,
                "hours": int(np.count_nonzero(selected)),
                "total_net_return_fraction": float(
                    np.prod(1.0 + returns[selected]) - 1.0
                ),
                "mean_hourly_portfolio_bps": float(np.mean(portfolio_bps[selected])),
                "active_hours": int(
                    np.count_nonzero(np.any(positions[selected] != 0, axis=1))
                ),
            }
        )
    days = np.asarray(
        [
            datetime.fromtimestamp(value / 1_000.0, UTC).date().isoformat()
            for value in timestamps
        ]
    )
    active_days = len(
        set(days[np.any(positions != 0, axis=1)].tolist())
    )
    by_symbol = {symbol: 0 for symbol in SYMBOLS}
    symbol_net_bps = {symbol: 0.0 for symbol in SYMBOLS}
    for trade, net_bps in zip(trades, trade_net_bps, strict=True):
        by_symbol[trade.symbol] += 1
        symbol_net_bps[trade.symbol] += net_bps * SLEEVE_FRACTION
    absolute_total = sum(abs(value) for value in symbol_net_bps.values())
    maximum_symbol_fraction = (
        max(abs(value) for value in symbol_net_bps.values()) / absolute_total
        if absolute_total > 0.0
        else 0.0
    )
    metrics: dict[str, object] = {
        "scenario": scenario,
        "one_way_cost_bps": one_way_cost_bps,
        "hours": int(indexes.size),
        "trades": len(trades),
        "trades_by_symbol": by_symbol,
        "symbol_net_bps": symbol_net_bps,
        "total_net_return_fraction": float(equity[-1] - 1.0),
        "maximum_drawdown_fraction": maximum_drawdown,
        "mean_hourly_portfolio_bps": float(np.mean(portfolio_bps)),
        "profit_factor": positive / negative if negative > 0.0 else None,
        "positive_months": int(
            sum(row["total_net_return_fraction"] > 0.0 for row in monthly)
        ),
        "active_days": active_days,
        "active_hours": int(np.count_nonzero(np.any(positions != 0, axis=1))),
        "maximum_single_symbol_fraction_of_absolute_net_pnl": maximum_symbol_fraction,
        "bootstrap_mean_hourly_portfolio_bps": _circular_block_bootstrap(
            portfolio_bps,
            seed=SEEDS[0] + (0 if scenario == "base" else 100),
        ),
        "monthly": monthly,
    }
    return PlannedReplay(
        scenario=scenario,
        one_way_cost_bps=one_way_cost_bps,
        trades=tuple(trades),
        timestamps_ms=timestamps,
        positions=positions,
        symbol_return_bps=symbol_return_bps,
        portfolio_return_bps=portfolio_bps,
        metrics=metrics,
    )


def economic_gate(
    *,
    forecast_gate_passed: bool,
    stress: PlannedReplay,
) -> dict[str, object]:
    metrics = stress.metrics
    bootstrap = metrics["bootstrap_mean_hourly_portfolio_bps"]
    reasons: list[str] = []
    if not forecast_gate_passed:
        reasons.append("forecast_gate_failed")
    if bootstrap["lower_bps"] <= 0.0:
        reasons.append("stress_familywise_bootstrap_lower_not_positive")
    if metrics["positive_months"] < 6:
        reasons.append("fewer_than_six_positive_evaluation_months")
    if metrics["active_days"] < 90:
        reasons.append("fewer_than_ninety_active_days")
    if metrics["maximum_drawdown_fraction"] > 0.10:
        reasons.append("maximum_drawdown_exceeds_ten_percent")
    if metrics["profit_factor"] is None or metrics["profit_factor"] < 1.05:
        reasons.append("profit_factor_below_1_05")
    if any(value <= 0 for value in metrics["trades_by_symbol"].values()):
        reasons.append("not_all_three_symbols_have_activity")
    if metrics["maximum_single_symbol_fraction_of_absolute_net_pnl"] > 0.50:
        reasons.append("single_symbol_absolute_net_pnl_fraction_exceeds_half")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "promotion_permitted": False,
        "selection_contaminated": True,
    }


__all__ = [
    "BASE_ONE_WAY_COST_BPS",
    "BOOTSTRAP_BLOCK_HOURS",
    "BOOTSTRAP_SAMPLES",
    "DistributionalDataset",
    "DistributionalTCN",
    "ExplicitAdamW",
    "FAMILYWISE_LOWER_QUANTILE",
    "FeatureScaler",
    "HORIZONS",
    "PlannedReplay",
    "PlannedTrade",
    "QUANTILES",
    "RECEPTIVE_FIELD",
    "ROLES",
    "SEEDS",
    "STRESS_ONE_WAY_COST_BPS",
    "TCNArtifact",
    "TCNForecastBundle",
    "TargetScaler",
    "build_distributional_dataset",
    "compounded_forward_returns",
    "cpu_device",
    "directml_preflight",
    "economic_gate",
    "fit_feature_scaler",
    "fit_target_scaler",
    "forecast_diagnostics",
    "pinball_loss",
    "replay_planned_trades",
    "role_mask",
    "select_planned_trades",
    "train_distributional_tcn_ensemble",
    "train_distributional_tcn_seed",
]
