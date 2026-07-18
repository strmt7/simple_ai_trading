"""Cost-aware 15-minute action-value TCN candidates for Round 49."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
import torch
from torch import nn
from torch.nn import functional as F

from .compute import require_backend, resolve_backend, torch_device_for_backend
from .cross_asset_cost_data import MINUTE_MS, SYMBOLS
from .derivatives_hurdle_data import (
    EXECUTION_CHARGE_BPS,
    DerivativesHurdleDataset,
)
from .distributional_tcn_model import ExplicitAdamW
from .minute_logistic_mixture_tcn_model import (
    DEPTHWISE_DILATIONS,
    FEATURE_COUNT,
    HIDDEN_CHANNELS,
    LargeKernelCausalBlock,
    MinuteTemporalDataset,
    PerTimestampLayerNorm,
    RobustFeatureScaler,
    build_minute_temporal_dataset,
    fit_robust_feature_scaler,
)


ROUND = 49
HORIZONS_MINUTES = (15, 30)
PRIMARY_HORIZON_MINUTES = 15
AUXILIARY_HORIZON_MINUTES = 30
PRIMARY_HORIZON_INDEX = 0
AUXILIARY_HORIZON_INDEX = 1
SIDES = (-1, 1)
CANDIDATES = ("direct_action_mean_tcn", "hurdle_action_value_tcn")
SEEDS = (4901, 4902, 4903)
RECEPTIVE_FIELD_STEPS = 361
WINDOW_STEPS = 576
SUPERVISED_STEPS = WINDOW_STEPS - RECEPTIVE_FIELD_STEPS + 1
WINDOW_STRIDE_STEPS = SUPERVISED_STEPS
BATCH_SIZE = 32
PREDICTION_CHUNK_STEPS = 2_048
MAXIMUM_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 8
MINIMUM_IMPROVEMENT = 1e-4
PROFIT_BCE_WEIGHT = 1.0
AUXILIARY_MEAN_MSE_WEIGHT = 0.1
DIRECT_ACTION_MEAN_MSE_WEIGHT = 0.5
GAIN_GAMMA_SCORE_WEIGHT = 0.25
LOSS_GAMMA_SCORE_WEIGHT = 0.25
PAIRWISE_RANK_WEIGHT = 0.025
PAIRWISE_RANK_OFFSET_STEPS = 24
OUTPUT_LIMIT = 8.0
MINIMUM_MEAN_MULTIPLIER = 1e-4
CALIBRATION_SLOPE_BOUNDS = (0.25, 4.0)
CALIBRATION_INTERCEPT_BOUNDS = (-4.0, 4.0)
SEVERITY_MULTIPLIER_BOUNDS = (0.5, 2.0)
ANALYSIS_ROLES = ("calibration", "viability")
ProgressCallback = Callable[[str, Mapping[str, object]], None]


@dataclass(frozen=True)
class ActionTargetScaler:
    direct_mean_bps: np.ndarray
    direct_scale_bps: np.ndarray
    gain_mean_bps: np.ndarray
    loss_mean_bps: np.ndarray
    auxiliary_median_bps: np.ndarray
    auxiliary_scale_bps: np.ndarray

    def normalize_direct(
        self, values: np.ndarray, symbol_indices: np.ndarray
    ) -> np.ndarray:
        mean = self.direct_mean_bps[symbol_indices]
        scale = self.direct_scale_bps[symbol_indices]
        return ((values - mean[..., None]) / scale[..., None]).astype(np.float32)

    def denormalize_direct(
        self, values: np.ndarray, symbol_indices: np.ndarray
    ) -> np.ndarray:
        mean = self.direct_mean_bps[symbol_indices]
        scale = self.direct_scale_bps[symbol_indices]
        return (values * scale[..., None] + mean[..., None]).astype(np.float32)

    def normalize_auxiliary(
        self, values: np.ndarray, symbol_indices: np.ndarray
    ) -> np.ndarray:
        median = self.auxiliary_median_bps[symbol_indices]
        scale = self.auxiliary_scale_bps[symbol_indices]
        return ((values - median[..., None]) / scale[..., None]).astype(np.float32)

    def denormalize_auxiliary(
        self, values: np.ndarray, symbol_indices: np.ndarray
    ) -> np.ndarray:
        median = self.auxiliary_median_bps[symbol_indices]
        scale = self.auxiliary_scale_bps[symbol_indices]
        return (values * scale[..., None] + median[..., None]).astype(np.float32)

    def asdict(self) -> dict[str, object]:
        return {
            field: np.asarray(value).tolist() for field, value in asdict(self).items()
        }


@dataclass(frozen=True)
class ProbabilityCalibration:
    slope: float
    intercept: float
    binary_log_loss_before: float
    binary_log_loss_after: float


@dataclass(frozen=True)
class ActionHurdleArtifact:
    candidate_id: str
    seed: int
    epochs: int
    best_epoch: int
    best_early_stop_composite: float
    best_early_stop_profit_bce: float
    best_early_stop_primary_loss: float
    best_early_stop_auxiliary_mse: float
    best_early_stop_pairwise_rank_loss: float
    optimizer_updates: int
    parameter_count: int
    probability_slope: float
    probability_intercept: float
    calibration_binary_log_loss_before: float
    calibration_binary_log_loss_after: float
    gain_multiplier: float
    loss_multiplier: float
    calibration_gain_score_before: float | None
    calibration_gain_score_after: float | None
    calibration_loss_score_before: float | None
    calibration_loss_score_after: float | None
    backend_kind: str
    backend_device: str
    model_path: str
    model_bytes: int
    model_sha256: str
    prediction_path: str
    prediction_bytes: int
    prediction_sha256: str
    reload_max_abs_logit_error: float
    reload_max_abs_primary_error: float
    reload_max_abs_secondary_error: float
    reload_max_abs_auxiliary_error: float
    warning_count: int

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ActionHurdleForecastBundle:
    candidate_id: str
    global_indices: np.ndarray
    seed_probabilities: np.ndarray
    seed_action_values_bps: np.ndarray
    seed_gain_means_bps: np.ndarray | None
    seed_loss_means_bps: np.ndarray | None
    seed_auxiliary_mean_bps: np.ndarray
    artifacts: tuple[ActionHurdleArtifact, ...]
    feature_scaler: RobustFeatureScaler
    target_scaler: ActionTargetScaler
    backend_kind: str
    backend_device: str
    training_history: tuple[Mapping[str, object], ...]


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_bool_sha256(values: np.ndarray, label: str) -> str:
    digest = hashlib.sha256(label.encode("ascii"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    digest.update(np.packbits(values.astype(bool, copy=False)).tobytes())
    return digest.hexdigest()


def build_action_hurdle_temporal_dataset(
    source: DerivativesHurdleDataset,
) -> MinuteTemporalDataset:
    """Reuse the verified matrix with the Round 49 30-minute role embargo."""

    base = build_minute_temporal_dataset(source)
    symbol_count = len(SYMBOLS)
    timestamps = base.timestamps
    roles: dict[str, np.ndarray] = {}
    for role in ("training", "early_stop", "calibration", "viability"):
        blocks = source.role_masks[AUXILIARY_HORIZON_MINUTES][role].reshape(
            symbol_count, timestamps
        )
        if not np.all(blocks == blocks[:1]):
            raise ValueError(f"Round 49 role differs across symbols: {role}")
        roles[role] = blocks[0].copy()
        if not np.any(roles[role]):
            raise ValueError(f"Round 49 role is empty: {role}")
    if any(
        np.any(roles[left] & roles[right])
        for left_index, left in enumerate(roles)
        for right in tuple(roles)[left_index + 1 :]
    ):
        raise ValueError("Round 49 chronological roles overlap")
    role_hashes = {
        role: _stream_bool_sha256(mask, f"round49-{role}")
        for role, mask in roles.items()
    }
    identity = _canonical_sha256(
        {
            "schema": "round-049-action-hurdle-temporal-dataset-v1",
            "predecessor_dataset_sha256": base.dataset_sha256,
            "symbols": list(SYMBOLS),
            "horizons_minutes": list(HORIZONS_MINUTES),
            "feature_stream_sha256": base.feature_stream_sha256,
            "target_stream_sha256": base.target_stream_sha256,
            "role_stream_sha256": role_hashes,
        }
    )
    return replace(base, role_masks=roles, dataset_sha256=identity)


def side_net_targets(dataset: MinuteTemporalDataset) -> np.ndarray:
    signed = dataset.signed_target_bps[..., : len(HORIZONS_MINUTES)].astype(
        np.float32, copy=False
    )
    targets = np.stack(
        (-signed - EXECUTION_CHARGE_BPS, signed - EXECUTION_CHARGE_BPS),
        axis=-1,
    ).astype(np.float32)
    if targets.shape != (
        dataset.timestamps,
        len(SYMBOLS),
        len(HORIZONS_MINUTES),
        len(SIDES),
    ):
        raise RuntimeError("Round 49 side-target shape is invalid")
    if not np.isfinite(targets).all():
        raise ValueError("Round 49 side targets are nonfinite")
    return targets


def fit_action_target_scaler(
    dataset: MinuteTemporalDataset,
    targets: np.ndarray,
) -> ActionTargetScaler:
    training = dataset.role_masks["training"]
    primary = targets[training, :, PRIMARY_HORIZON_INDEX]
    direct_mean = np.mean(primary, axis=0, dtype=np.float64)
    direct_scale = np.maximum(np.std(primary, axis=0, dtype=np.float64), 1.0)
    gain_mean = np.empty((len(SYMBOLS), len(SIDES)), dtype=np.float64)
    loss_mean = np.empty_like(gain_mean)
    for symbol_index in range(len(SYMBOLS)):
        for side_index in range(len(SIDES)):
            values = primary[:, symbol_index, side_index].astype(np.float64)
            gains = values[values > 0.0]
            losses = -values[values <= 0.0]
            if gains.size == 0 or losses.size == 0:
                raise ValueError("Round 49 conditional severity role is empty")
            gain_mean[symbol_index, side_index] = float(np.mean(gains))
            loss_mean[symbol_index, side_index] = float(np.mean(losses))
    auxiliary = dataset.signed_target_bps[training, :, AUXILIARY_HORIZON_INDEX].astype(
        np.float64
    )
    quartiles = np.quantile(auxiliary, (0.25, 0.5, 0.75), axis=0)
    auxiliary_median = quartiles[1]
    auxiliary_scale = np.maximum((quartiles[2] - quartiles[0]) / 1.349, 1.0)
    arrays = (
        direct_mean,
        direct_scale,
        gain_mean,
        loss_mean,
        auxiliary_median,
        auxiliary_scale,
    )
    if not all(np.isfinite(value).all() for value in arrays) or not all(
        np.all(value > 0.0)
        for value in (direct_scale, gain_mean, loss_mean, auxiliary_scale)
    ):
        raise ValueError("Round 49 target scaler is invalid")
    return ActionTargetScaler(
        direct_mean_bps=direct_mean,
        direct_scale_bps=direct_scale,
        gain_mean_bps=gain_mean,
        loss_mean_bps=loss_mean,
        auxiliary_median_bps=auxiliary_median,
        auxiliary_scale_bps=auxiliary_scale,
    )


class ActionEncoder(nn.Module):
    def __init__(
        self, input_channels: int = FEATURE_COUNT, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.projection = nn.Conv1d(input_channels, HIDDEN_CHANNELS, kernel_size=1)
        self.blocks = nn.ModuleList(
            LargeKernelCausalBlock(HIDDEN_CHANNELS, dilation, dropout)
            for dilation in DEPTHWISE_DILATIONS
        )
        self.final_normalization = PerTimestampLayerNorm(HIDDEN_CHANNELS)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != self.input_channels:
            raise ValueError("Round 49 model input dimensions are invalid")
        encoded = F.gelu(self.projection(values))
        for block in self.blocks:
            encoded = block(encoded)
        return self.final_normalization(encoded)


class DirectActionMeanTCN(nn.Module):
    candidate_id = "direct_action_mean_tcn"

    def __init__(
        self, input_channels: int = FEATURE_COUNT, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.encoder = ActionEncoder(input_channels, dropout)
        self.profit_head = nn.Conv1d(
            HIDDEN_CHANNELS, len(HORIZONS_MINUTES) * len(SIDES), kernel_size=1
        )
        self.primary_head = nn.Conv1d(HIDDEN_CHANNELS, len(SIDES), kernel_size=1)
        self.auxiliary_head = nn.Conv1d(HIDDEN_CHANNELS, 1, kernel_size=1)

    def forward(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(values)
        logits = self.profit_head(encoded).reshape(
            values.shape[0], len(HORIZONS_MINUTES), len(SIDES), values.shape[-1]
        )
        direct = OUTPUT_LIMIT * torch.tanh(self.primary_head(encoded) / OUTPUT_LIMIT)
        auxiliary = OUTPUT_LIMIT * torch.tanh(
            self.auxiliary_head(encoded).squeeze(1) / OUTPUT_LIMIT
        )
        return logits, direct, auxiliary


class HurdleActionValueTCN(nn.Module):
    candidate_id = "hurdle_action_value_tcn"

    def __init__(
        self, input_channels: int = FEATURE_COUNT, dropout: float = 0.1
    ) -> None:
        super().__init__()
        self.encoder = ActionEncoder(input_channels, dropout)
        self.profit_head = nn.Conv1d(
            HIDDEN_CHANNELS, len(HORIZONS_MINUTES) * len(SIDES), kernel_size=1
        )
        self.gain_head = nn.Conv1d(HIDDEN_CHANNELS, len(SIDES), kernel_size=1)
        self.loss_head = nn.Conv1d(HIDDEN_CHANNELS, len(SIDES), kernel_size=1)
        self.auxiliary_head = nn.Conv1d(HIDDEN_CHANNELS, 1, kernel_size=1)
        initial = math.log(math.expm1(1.0 - MINIMUM_MEAN_MULTIPLIER))
        with torch.no_grad():
            if self.gain_head.bias is not None:
                self.gain_head.bias.fill_(initial)
            if self.loss_head.bias is not None:
                self.loss_head.bias.fill_(initial)

    def forward(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(values)
        logits = self.profit_head(encoded).reshape(
            values.shape[0], len(HORIZONS_MINUTES), len(SIDES), values.shape[-1]
        )
        gain = F.softplus(self.gain_head(encoded)) + MINIMUM_MEAN_MULTIPLIER
        loss = F.softplus(self.loss_head(encoded)) + MINIMUM_MEAN_MULTIPLIER
        auxiliary = OUTPUT_LIMIT * torch.tanh(
            self.auxiliary_head(encoded).squeeze(1) / OUTPUT_LIMIT
        )
        return logits, gain, loss, auxiliary


def gamma_mean_score(
    predicted_mean: torch.Tensor,
    target: torch.Tensor,
    condition: torch.Tensor,
) -> torch.Tensor:
    if predicted_mean.shape != target.shape or target.shape != condition.shape:
        raise ValueError("Round 49 Gamma score shapes are invalid")
    if predicted_mean.ndim != 3:
        raise ValueError("Round 49 Gamma score dimensions are invalid")
    mask = condition.to(dtype=predicted_mean.dtype)
    score = target / predicted_mean + torch.log(predicted_mean)
    return torch.sum(score * mask) / torch.clamp(torch.sum(mask), min=1.0)


def pairwise_action_rank_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    offset: int = PAIRWISE_RANK_OFFSET_STEPS,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Round 49 pairwise rank shapes are invalid")
    if offset <= 0 or prediction.shape[-1] <= offset:
        raise ValueError("Round 49 pairwise rank offset is invalid")
    predicted_delta = prediction[..., :-offset] - prediction[..., offset:]
    target_delta = target[..., :-offset] - target[..., offset:]
    non_tie = (target_delta != 0.0).to(dtype=prediction.dtype)
    direction = torch.where(
        target_delta > 0.0,
        torch.ones_like(target_delta),
        -torch.ones_like(target_delta),
    )
    margin = (-direction * predicted_delta).contiguous()
    bounded_margin = torch.minimum(
        margin,
        torch.full_like(margin, 20.0),
    )
    stable_softplus = torch.log1p(torch.exp(bounded_margin))
    numerator = torch.sum(stable_softplus * non_tie)
    return numerator / torch.clamp(torch.sum(non_tie), min=1.0)


def binary_logit_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.shape != labels.shape:
        raise ValueError("Round 49 binary logit shapes are invalid")
    positive = torch.maximum(logits, torch.zeros_like(logits))
    stable = positive - logits * labels + torch.log1p(torch.exp(-torch.abs(logits)))
    return torch.mean(stable)


def direct_objective(
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    net_targets_bps: torch.Tensor,
    normalized_primary_targets: torch.Tensor,
    normalized_auxiliary_targets: torch.Tensor,
    direct_scale_bps: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits, direct, auxiliary = output
    labels = (net_targets_bps > 0.0).to(dtype=logits.dtype)
    profit_bce = binary_logit_loss(logits, labels)
    primary_mse = F.mse_loss(direct, normalized_primary_targets)
    auxiliary_mse = F.mse_loss(auxiliary, normalized_auxiliary_targets)
    rank_loss = pairwise_action_rank_loss(
        direct,
        normalized_primary_targets,
    )
    composite = (
        PROFIT_BCE_WEIGHT * profit_bce
        + DIRECT_ACTION_MEAN_MSE_WEIGHT * primary_mse
        + AUXILIARY_MEAN_MSE_WEIGHT * auxiliary_mse
        + PAIRWISE_RANK_WEIGHT * rank_loss
    )
    del direct_scale_bps
    return composite, {
        "profit_bce": profit_bce,
        "primary_loss": primary_mse,
        "auxiliary_mse": auxiliary_mse,
        "pairwise_rank_loss": rank_loss,
        "composite": composite,
    }


def hurdle_expected_net_bps(
    logits: torch.Tensor,
    gain_multiplier: torch.Tensor,
    loss_multiplier: torch.Tensor,
    gain_baseline_bps: torch.Tensor,
    loss_baseline_bps: torch.Tensor,
) -> torch.Tensor:
    gain_baseline_bps = gain_baseline_bps.to(dtype=logits.dtype)
    loss_baseline_bps = loss_baseline_bps.to(dtype=logits.dtype)
    probability = torch.sigmoid(logits[:, PRIMARY_HORIZON_INDEX])
    gain = gain_multiplier * gain_baseline_bps[..., None]
    loss = loss_multiplier * loss_baseline_bps[..., None]
    return probability * gain - (1.0 - probability) * loss


def hurdle_objective(
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    net_targets_bps: torch.Tensor,
    normalized_primary_targets: torch.Tensor,
    normalized_auxiliary_targets: torch.Tensor,
    direct_scale_bps: torch.Tensor,
    gain_baseline_bps: torch.Tensor,
    loss_baseline_bps: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    logits, gain_multiplier, loss_multiplier, auxiliary = output
    direct_scale_bps = direct_scale_bps.to(dtype=logits.dtype)
    gain_baseline_bps = gain_baseline_bps.to(dtype=logits.dtype)
    loss_baseline_bps = loss_baseline_bps.to(dtype=logits.dtype)
    labels = (net_targets_bps > 0.0).to(dtype=logits.dtype)
    profit_bce = binary_logit_loss(logits, labels)
    primary = net_targets_bps[:, PRIMARY_HORIZON_INDEX]
    gain_target = torch.clamp(primary, min=0.0) / gain_baseline_bps[..., None]
    loss_target = torch.clamp(-primary, min=0.0) / loss_baseline_bps[..., None]
    gain_score = gamma_mean_score(gain_multiplier, gain_target, primary > 0.0)
    loss_score = gamma_mean_score(loss_multiplier, loss_target, primary <= 0.0)
    auxiliary_mse = F.mse_loss(auxiliary, normalized_auxiliary_targets)
    expected_net = hurdle_expected_net_bps(
        logits,
        gain_multiplier,
        loss_multiplier,
        gain_baseline_bps,
        loss_baseline_bps,
    )
    normalized_expected = expected_net / direct_scale_bps[..., None]
    normalized_target = primary / direct_scale_bps[..., None]
    rank_loss = pairwise_action_rank_loss(normalized_expected, normalized_target)
    composite = (
        PROFIT_BCE_WEIGHT * profit_bce
        + GAIN_GAMMA_SCORE_WEIGHT * gain_score
        + LOSS_GAMMA_SCORE_WEIGHT * loss_score
        + AUXILIARY_MEAN_MSE_WEIGHT * auxiliary_mse
        + PAIRWISE_RANK_WEIGHT * rank_loss
    )
    del normalized_primary_targets
    return composite, {
        "profit_bce": profit_bce,
        "primary_loss": 0.5 * (gain_score + loss_score),
        "gain_score": gain_score,
        "loss_score": loss_score,
        "auxiliary_mse": auxiliary_mse,
        "pairwise_rank_loss": rank_loss,
        "composite": composite,
    }


def apply_probability_calibration(
    logits: np.ndarray, calibration: ProbabilityCalibration
) -> np.ndarray:
    epsilon = np.finfo(np.float32).eps
    probabilities = expit(calibration.slope * logits + calibration.intercept)
    return np.clip(probabilities, epsilon, 1.0 - epsilon).astype(np.float32)


def _binary_log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probabilities.astype(np.float64), 1e-12, 1.0 - 1e-12)
    target = labels.astype(np.float64)
    return float(
        -np.mean(target * np.log(clipped) + (1.0 - target) * np.log1p(-clipped))
    )


def fit_probability_calibration(
    logits: np.ndarray, labels: np.ndarray
) -> ProbabilityCalibration:
    flat_logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    flat_labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    if (
        flat_logits.size == 0
        or flat_logits.shape != flat_labels.shape
        or not np.isfinite(flat_logits).all()
        or not np.isfinite(flat_labels).all()
        or np.any((flat_labels != 0.0) & (flat_labels != 1.0))
    ):
        raise ValueError("Round 49 probability calibration input is invalid")
    before = _binary_log_loss(expit(flat_logits), flat_labels)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        slope, intercept = parameters
        transformed = slope * flat_logits + intercept
        probabilities = expit(transformed)
        loss = _binary_log_loss(probabilities, flat_labels)
        residual = probabilities - flat_labels
        gradient = np.asarray(
            [np.mean(residual * flat_logits), np.mean(residual)], dtype=np.float64
        )
        return loss, gradient

    result = minimize(
        objective,
        np.asarray([1.0, 0.0], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=(CALIBRATION_SLOPE_BOUNDS, CALIBRATION_INTERCEPT_BOUNDS),
        options={"maxiter": 200, "ftol": 1e-15, "gtol": 1e-10},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RuntimeError(f"Round 49 probability calibration failed: {result.message}")
    slope = float(result.x[0])
    intercept = float(result.x[1])
    after = _binary_log_loss(expit(slope * flat_logits + intercept), flat_labels)
    if after > before + 1e-12:
        slope, intercept, after = 1.0, 0.0, before
    return ProbabilityCalibration(
        slope=slope,
        intercept=intercept,
        binary_log_loss_before=before,
        binary_log_loss_after=after,
    )


def numpy_gamma_mean_score(
    predicted_mean: np.ndarray,
    target: np.ndarray,
    condition: np.ndarray,
) -> float:
    prediction = np.asarray(predicted_mean, dtype=np.float64)
    values = np.asarray(target, dtype=np.float64)
    mask = np.asarray(condition, dtype=bool)
    if (
        prediction.shape != values.shape
        or values.shape != mask.shape
        or not np.any(mask)
    ):
        raise ValueError("Round 49 NumPy Gamma score input is invalid")
    if not np.isfinite(prediction).all() or np.any(prediction <= 0.0):
        raise ValueError("Round 49 NumPy Gamma prediction is invalid")
    return float(np.mean(values[mask] / prediction[mask] + np.log(prediction[mask])))


def fit_severity_multiplier(
    predicted_mean_bps: np.ndarray,
    target_bps: np.ndarray,
    condition: np.ndarray,
) -> tuple[float, float, float]:
    prediction = np.asarray(predicted_mean_bps, dtype=np.float64)
    target = np.asarray(target_bps, dtype=np.float64)
    mask = np.asarray(condition, dtype=bool)
    before = numpy_gamma_mean_score(prediction, target, mask)
    optimum = float(np.mean(target[mask] / prediction[mask]))
    multiplier = float(np.clip(optimum, *SEVERITY_MULTIPLIER_BOUNDS))
    after = numpy_gamma_mean_score(multiplier * prediction, target, mask)
    if after > before + 1e-12:
        multiplier, after = 1.0, before
    return multiplier, before, after


def _fallback_messages(messages: Sequence[str]) -> list[str]:
    return [
        item
        for item in messages
        if "not currently supported on the DML backend" in item
        or "fall back to run on the CPU" in item
    ]


def _candidate_model(candidate_id: str) -> nn.Module:
    if candidate_id == "direct_action_mean_tcn":
        return DirectActionMeanTCN()
    if candidate_id == "hurdle_action_value_tcn":
        return HurdleActionValueTCN()
    raise KeyError(candidate_id)


def _preflight(
    device: object, *, backend_kind: str, backend_device: str
) -> dict[str, object]:
    generator = np.random.default_rng(SEEDS[0])
    reports: list[dict[str, object]] = []
    warning_messages: list[str] = []
    for candidate_id in CANDIDATES:
        torch.manual_seed(SEEDS[0])
        model = _candidate_model(candidate_id).to(device)
        optimizer = ExplicitAdamW(
            tuple(model.parameters()), learning_rate=8e-4, weight_decay=1e-4
        )
        values = torch.from_numpy(
            generator.normal(size=(4, FEATURE_COUNT, 400)).astype(np.float32)
        ).to(device)
        net = torch.from_numpy(
            generator.normal(
                loc=-12.0,
                scale=45.0,
                size=(4, len(HORIZONS_MINUTES), len(SIDES), 400),
            ).astype(np.float32)
        ).to(device)
        normalized_primary = net[:, PRIMARY_HORIZON_INDEX] / 45.0
        normalized_auxiliary = torch.from_numpy(
            generator.normal(size=(4, 400)).astype(np.float32)
        ).to(device)
        scale = torch.full((4, len(SIDES)), 45.0, dtype=torch.float32, device=device)
        gain_baseline = torch.full(
            (4, len(SIDES)), 30.0, dtype=torch.float32, device=device
        )
        loss_baseline = torch.full(
            (4, len(SIDES)), 35.0, dtype=torch.float32, device=device
        )
        before = {
            "depthwise": model.encoder.blocks[0]
            .depthwise.weight.detach()
            .cpu()
            .clone(),
            "profit": model.profit_head.weight.detach().cpu().clone(),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            optimizer.zero_grad(set_to_none=True)
            output = model(values)
            if candidate_id == "direct_action_mean_tcn":
                objective, losses = direct_objective(
                    output,
                    net,
                    normalized_primary,
                    normalized_auxiliary,
                    scale,
                )
            else:
                objective, losses = hurdle_objective(
                    output,
                    net,
                    normalized_primary,
                    normalized_auxiliary,
                    scale,
                    gain_baseline,
                    loss_baseline,
                )
            objective.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, foreach=False
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            extreme = net.clone()
            extreme[..., ::2] = 5_000.0
            extreme[..., 1::2] = -5_000.0
            output = model(values)
            if candidate_id == "direct_action_mean_tcn":
                tail_objective, _ = direct_objective(
                    output,
                    extreme,
                    extreme[:, PRIMARY_HORIZON_INDEX] / 45.0,
                    normalized_auxiliary,
                    scale,
                )
            else:
                tail_objective, _ = hurdle_objective(
                    output,
                    extreme,
                    extreme[:, PRIMARY_HORIZON_INDEX] / 45.0,
                    normalized_auxiliary,
                    scale,
                    gain_baseline,
                    loss_baseline,
                )
            tail_objective.backward()
            tail_gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, foreach=False
            )
        warning_messages.extend(str(item.message) for item in caught)
        changes = {
            "depthwise": float(
                torch.max(
                    torch.abs(
                        model.encoder.blocks[0].depthwise.weight.detach().cpu()
                        - before["depthwise"]
                    )
                )
            ),
            "profit": float(
                torch.max(
                    torch.abs(
                        model.profit_head.weight.detach().cpu() - before["profit"]
                    )
                )
            ),
        }
        report = {
            "candidate_id": candidate_id,
            "objective": float(objective.detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "extreme_severity_objective": float(tail_objective.detach().cpu()),
            "extreme_severity_gradient_norm": float(tail_gradient_norm.detach().cpu()),
            "parameter_changes": changes,
            **{name: float(value.detach().cpu()) for name, value in losses.items()},
        }
        scalar_values = [
            value for value in report.values() if isinstance(value, float)
        ] + list(changes.values())
        if (
            not all(math.isfinite(value) for value in scalar_values)
            or min(changes.values()) <= 0.0
        ):
            raise RuntimeError(f"Round 49 numerical preflight failed: {report}")
        reports.append(report)
    fallback = _fallback_messages(warning_messages)
    if fallback:
        raise RuntimeError(f"Round 49 preflight used CPU fallback: {fallback}")
    return {
        "backend_kind": backend_kind,
        "backend_device": backend_device,
        "candidates": reports,
        "warning_count": len(warning_messages),
        "cpu_fallback_warnings": len(fallback),
    }


def action_hurdle_preflight(
    compute_backend: str = "auto",
) -> tuple[object, dict[str, object]]:
    backend = require_backend(resolve_backend(compute_backend))
    device = torch_device_for_backend(backend)
    return device, _preflight(
        device, backend_kind=backend.kind, backend_device=str(device)
    )


def directml_action_hurdle_preflight() -> tuple[object, dict[str, object]]:
    return action_hurdle_preflight("directml")


def cpu_action_hurdle_preflight() -> tuple[object, dict[str, object]]:
    return action_hurdle_preflight("cpu")


def _contiguous_runs(
    mask: np.ndarray, timestamps_ms: np.ndarray
) -> tuple[tuple[int, int], ...]:
    selected = np.flatnonzero(mask)
    if selected.size == 0:
        return ()
    expected_delta = 5 * MINUTE_MS
    split = np.flatnonzero(
        (np.diff(selected) != 1) | (np.diff(timestamps_ms[selected]) != expected_delta)
    )
    starts = np.concatenate(([0], split + 1))
    ends = np.concatenate((split + 1, [selected.size]))
    return tuple(
        (int(selected[start]), int(selected[end - 1]) + 1)
        for start, end in zip(starts, ends, strict=True)
    )


def _source_boundaries(timestamps_ms: np.ndarray) -> np.ndarray:
    breaks = np.flatnonzero(np.diff(timestamps_ms) != 5 * MINUTE_MS) + 1
    return np.concatenate(([0], breaks, [timestamps_ms.size])).astype(np.int64)


def _training_pairs(dataset: MinuteTemporalDataset) -> np.ndarray:
    rows: list[tuple[int, int]] = []
    for start, end in _contiguous_runs(
        dataset.role_masks["training"], dataset.timestamps_ms
    ):
        last_start = end - WINDOW_STEPS
        if last_start < start:
            continue
        for window_start in range(start, last_start + 1, WINDOW_STRIDE_STEPS):
            rows.extend(
                (window_start, symbol_index) for symbol_index in range(len(SYMBOLS))
            )
    if not rows:
        raise ValueError("Round 49 training role has no complete windows")
    pairs = np.asarray(rows, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise RuntimeError("Round 49 training-pair matrix is invalid")
    return pairs


def _batch_arrays(
    normalized_features: np.ndarray,
    targets: np.ndarray,
    dataset: MinuteTemporalDataset,
    scaler: ActionTargetScaler,
    pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_rows: list[np.ndarray] = []
    net_rows: list[np.ndarray] = []
    direct_rows: list[np.ndarray] = []
    auxiliary_rows: list[np.ndarray] = []
    symbol_indices = pairs[:, 1].astype(np.int64, copy=False)
    supervised_start = RECEPTIVE_FIELD_STEPS - 1
    for start_value, symbol_value in pairs:
        start = int(start_value)
        symbol_index = int(symbol_value)
        stop = start + WINDOW_STEPS
        target_start = start + supervised_start
        feature_rows.append(normalized_features[start:stop, symbol_index].T)
        net = targets[target_start:stop, symbol_index]
        net_rows.append(net.transpose(1, 2, 0))
        primary = net[:, PRIMARY_HORIZON_INDEX].T
        direct_rows.append(
            (primary - scaler.direct_mean_bps[symbol_index, :, None])
            / scaler.direct_scale_bps[symbol_index, :, None]
        )
        auxiliary = dataset.signed_target_bps[
            target_start:stop, symbol_index, AUXILIARY_HORIZON_INDEX
        ]
        auxiliary_rows.append(
            (auxiliary - scaler.auxiliary_median_bps[symbol_index])
            / scaler.auxiliary_scale_bps[symbol_index]
        )
    outputs = (
        np.stack(feature_rows).astype(np.float32, copy=False),
        np.stack(net_rows).astype(np.float32, copy=False),
        np.stack(direct_rows).astype(np.float32, copy=False),
        np.stack(auxiliary_rows).astype(np.float32, copy=False),
        symbol_indices,
    )
    if outputs[0].shape != (pairs.shape[0], FEATURE_COUNT, WINDOW_STEPS):
        raise RuntimeError("Round 49 feature batch shape is invalid")
    if outputs[1].shape != (
        pairs.shape[0],
        len(HORIZONS_MINUTES),
        len(SIDES),
        SUPERVISED_STEPS,
    ):
        raise RuntimeError("Round 49 net-target batch shape is invalid")
    if outputs[2].shape != (pairs.shape[0], len(SIDES), SUPERVISED_STEPS):
        raise RuntimeError("Round 49 direct-target batch shape is invalid")
    if outputs[3].shape != (pairs.shape[0], SUPERVISED_STEPS):
        raise RuntimeError("Round 49 auxiliary-target batch shape is invalid")
    return (
        np.ascontiguousarray(outputs[0]),
        np.ascontiguousarray(outputs[1]),
        np.ascontiguousarray(outputs[2]),
        np.ascontiguousarray(outputs[3]),
        outputs[4],
    )


def _prediction_array(value: torch.Tensor) -> np.ndarray:
    if value.ndim == 4:
        return value.detach().cpu().numpy().transpose(3, 0, 1, 2).astype(np.float32)
    if value.ndim == 3:
        return value.detach().cpu().numpy().transpose(2, 0, 1).astype(np.float32)
    if value.ndim == 2:
        return value.detach().cpu().numpy().transpose(1, 0).astype(np.float32)
    raise ValueError("Round 49 prediction tensor dimensions are invalid")


def _predict_mask(
    model: nn.Module,
    normalized_features: np.ndarray,
    dataset: MinuteTemporalDataset,
    mask: np.ndarray,
    device: object,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("Round 49 prediction mask is empty")
    is_hurdle = isinstance(model, HurdleActionValueTCN)
    logits = np.full(
        (indices.size, len(SYMBOLS), len(HORIZONS_MINUTES), len(SIDES)),
        np.nan,
        dtype=np.float32,
    )
    primary = np.full(
        (indices.size, len(SYMBOLS), len(SIDES)), np.nan, dtype=np.float32
    )
    secondary = np.full_like(primary, np.nan) if is_hurdle else None
    auxiliary = np.full((indices.size, len(SYMBOLS)), np.nan, dtype=np.float32)
    destinations: list[np.ndarray] = [logits, primary]
    if secondary is not None:
        destinations.append(secondary)
    destinations.append(auxiliary)
    index_positions = np.full(dataset.timestamps, -1, dtype=np.int64)
    index_positions[indices] = np.arange(indices.size, dtype=np.int64)
    boundaries = _source_boundaries(dataset.timestamps_ms)

    model.eval()
    with torch.no_grad():
        for run_start, run_end in _contiguous_runs(mask, dataset.timestamps_ms):
            boundary_position = int(
                np.searchsorted(boundaries, run_start, side="right") - 1
            )
            segment_start = int(boundaries[boundary_position])
            for output_start in range(run_start, run_end, PREDICTION_CHUNK_STEPS):
                output_end = min(run_end, output_start + PREDICTION_CHUNK_STEPS)
                input_start = max(
                    segment_start, output_start - RECEPTIVE_FIELD_STEPS + 1
                )
                batch = np.ascontiguousarray(
                    normalized_features[input_start:output_end].transpose(1, 2, 0)
                )
                output = model(torch.from_numpy(batch).to(device))
                local_start = output_start - input_start
                local_stop = local_start + output_end - output_start
                positions = index_positions[output_start:output_end]
                if np.any(positions < 0):
                    raise RuntimeError("Round 49 prediction indexing is invalid")
                arrays = [
                    _prediction_array(value[..., local_start:local_stop])
                    for value in output
                ]
                for destination, array in zip(destinations, arrays, strict=True):
                    destination[positions] = array
    for name, value in zip(
        ("logits", "primary", "secondary", "auxiliary")
        if is_hurdle
        else ("logits", "primary", "auxiliary"),
        destinations,
        strict=True,
    ):
        if not np.isfinite(value).all():
            raise RuntimeError(f"Round 49 {name} predictions are nonfinite")
    if is_hurdle and (
        np.any(primary <= 0.0) or secondary is None or np.any(secondary <= 0.0)
    ):
        raise RuntimeError("Round 49 conditional means are not positive")
    return indices, tuple(destinations)


def _numpy_pairwise_rank_loss(
    prediction: np.ndarray,
    target: np.ndarray,
    indices: np.ndarray,
    timestamps_ms: np.ndarray,
) -> float:
    offset = PAIRWISE_RANK_OFFSET_STEPS
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Round 49 NumPy rank shapes are invalid")
    if prediction.shape[0] <= offset:
        return 0.0
    valid_time = (indices[offset:] - indices[:-offset] == offset) & (
        timestamps_ms[indices[offset:]] - timestamps_ms[indices[:-offset]]
        == offset * 5 * MINUTE_MS
    )
    prediction_delta = prediction[:-offset] - prediction[offset:]
    target_delta = target[:-offset] - target[offset:]
    valid = np.broadcast_to(valid_time.reshape(-1, 1, 1), target_delta.shape).copy()
    valid &= target_delta != 0.0
    if not np.any(valid):
        return 0.0
    direction = np.where(target_delta > 0.0, 1.0, -1.0)
    return float(
        np.mean(np.logaddexp(0.0, -direction[valid] * prediction_delta[valid]))
    )


def _numpy_candidate_losses(
    candidate_id: str,
    outputs: tuple[np.ndarray, ...],
    dataset: MinuteTemporalDataset,
    targets: np.ndarray,
    scaler: ActionTargetScaler,
    indices: np.ndarray,
) -> dict[str, float]:
    logits = outputs[0].astype(np.float64)
    labels = (targets[indices] > 0.0).astype(np.float64)
    profit_bce = _binary_log_loss(expit(logits), labels)
    auxiliary_target = dataset.signed_target_bps[
        indices, :, AUXILIARY_HORIZON_INDEX
    ].astype(np.float64)
    normalized_auxiliary = (
        auxiliary_target - scaler.auxiliary_median_bps.reshape(1, -1)
    ) / scaler.auxiliary_scale_bps.reshape(1, -1)
    auxiliary_mse = float(
        np.mean((outputs[-1].astype(np.float64) - normalized_auxiliary) ** 2)
    )
    primary_target = targets[indices, :, PRIMARY_HORIZON_INDEX].astype(np.float64)
    scale = scaler.direct_scale_bps.reshape(1, len(SYMBOLS), len(SIDES))
    normalized_target = primary_target / scale
    if candidate_id == "direct_action_mean_tcn":
        direct_target = (
            primary_target - scaler.direct_mean_bps.reshape(1, len(SYMBOLS), len(SIDES))
        ) / scale
        primary_prediction = outputs[1].astype(np.float64)
        primary_loss = float(np.mean((primary_prediction - direct_target) ** 2))
        rank_loss = _numpy_pairwise_rank_loss(
            primary_prediction, direct_target, indices, dataset.timestamps_ms
        )
        gain_score = None
        loss_score = None
    elif candidate_id == "hurdle_action_value_tcn":
        gain_multiplier = outputs[1].astype(np.float64)
        loss_multiplier = outputs[2].astype(np.float64)
        gain_target = np.clip(primary_target, 0.0, None) / scaler.gain_mean_bps.reshape(
            1, len(SYMBOLS), len(SIDES)
        )
        loss_target = np.clip(
            -primary_target, 0.0, None
        ) / scaler.loss_mean_bps.reshape(1, len(SYMBOLS), len(SIDES))
        gain_score = numpy_gamma_mean_score(
            gain_multiplier, gain_target, primary_target > 0.0
        )
        loss_score = numpy_gamma_mean_score(
            loss_multiplier, loss_target, primary_target <= 0.0
        )
        primary_loss = 0.5 * (gain_score + loss_score)
        probability = expit(logits[:, :, PRIMARY_HORIZON_INDEX])
        expected = probability * gain_multiplier * scaler.gain_mean_bps.reshape(
            1, len(SYMBOLS), len(SIDES)
        ) - (1.0 - probability) * loss_multiplier * scaler.loss_mean_bps.reshape(
            1, len(SYMBOLS), len(SIDES)
        )
        rank_loss = _numpy_pairwise_rank_loss(
            expected / scale,
            normalized_target,
            indices,
            dataset.timestamps_ms,
        )
    else:
        raise KeyError(candidate_id)
    if candidate_id == "direct_action_mean_tcn":
        composite = (
            PROFIT_BCE_WEIGHT * profit_bce
            + DIRECT_ACTION_MEAN_MSE_WEIGHT * primary_loss
            + AUXILIARY_MEAN_MSE_WEIGHT * auxiliary_mse
            + PAIRWISE_RANK_WEIGHT * rank_loss
        )
    else:
        if gain_score is None or loss_score is None:
            raise RuntimeError("Round 49 severity scores are missing")
        composite = (
            PROFIT_BCE_WEIGHT * profit_bce
            + GAIN_GAMMA_SCORE_WEIGHT * gain_score
            + LOSS_GAMMA_SCORE_WEIGHT * loss_score
            + AUXILIARY_MEAN_MSE_WEIGHT * auxiliary_mse
            + PAIRWISE_RANK_WEIGHT * rank_loss
        )
    output = {
        "profit_bce": profit_bce,
        "primary_loss": primary_loss,
        "auxiliary_mse": auxiliary_mse,
        "pairwise_rank_loss": rank_loss,
        "composite": composite,
    }
    if gain_score is not None and loss_score is not None:
        output["gain_score"] = gain_score
        output["loss_score"] = loss_score
    if not all(math.isfinite(value) for value in output.values()):
        raise RuntimeError("Round 49 validation losses are nonfinite")
    return output


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _save_prediction_artifact(
    path: Path,
    *,
    indices: np.ndarray,
    probabilities: np.ndarray,
    action_values_bps: np.ndarray,
    gain_means_bps: np.ndarray | None,
    loss_means_bps: np.ndarray | None,
    auxiliary_mean_bps: np.ndarray,
    calibration: ProbabilityCalibration,
    gain_multiplier: float,
    loss_multiplier: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        global_indices=indices,
        probabilities=probabilities,
        action_values_bps=action_values_bps,
        gain_means_bps=(
            gain_means_bps
            if gain_means_bps is not None
            else np.empty((0,), dtype=np.float32)
        ),
        loss_means_bps=(
            loss_means_bps
            if loss_means_bps is not None
            else np.empty((0,), dtype=np.float32)
        ),
        auxiliary_mean_bps=auxiliary_mean_bps,
        probability_slope=np.asarray(calibration.slope, dtype=np.float64),
        probability_intercept=np.asarray(calibration.intercept, dtype=np.float64),
        gain_multiplier=np.asarray(gain_multiplier, dtype=np.float64),
        loss_multiplier=np.asarray(loss_multiplier, dtype=np.float64),
    )


def _train_candidate(
    dataset: MinuteTemporalDataset,
    normalized_features: np.ndarray,
    targets: np.ndarray,
    feature_scaler: RobustFeatureScaler,
    target_scaler: ActionTargetScaler,
    *,
    candidate_id: str,
    model_dir: Path,
    prediction_dir: Path,
    device: object,
    backend_kind: str,
    backend_device: str,
    progress: ProgressCallback | None,
) -> ActionHurdleForecastBundle:
    peers: list[nn.Module] = []
    optimizers: list[ExplicitAdamW] = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        peer = _candidate_model(candidate_id).to(device)
        peers.append(peer)
        optimizers.append(
            ExplicitAdamW(
                tuple(peer.parameters()), learning_rate=8e-4, weight_decay=1e-4
            )
        )
    pairs = _training_pairs(dataset)
    generator = np.random.default_rng(SEEDS[0] + CANDIDATES.index(candidate_id))
    best_loss = math.inf
    best_epoch = 0
    best_states: list[dict[str, torch.Tensor]] | None = None
    best_losses: dict[str, float] | None = None
    stale_epochs = 0
    optimizer_updates = 0
    epochs_run = 0
    history: list[Mapping[str, object]] = []
    warning_messages: list[str] = []

    for epoch in range(1, MAXIMUM_EPOCHS + 1):
        epochs_run = epoch
        order = generator.permutation(pairs.shape[0])
        batch_losses: dict[str, list[float]] = {
            "profit_bce": [],
            "primary_loss": [],
            "auxiliary_mse": [],
            "pairwise_rank_loss": [],
            "composite": [],
        }
        for peer in peers:
            peer.train()
        batches_in_epoch = math.ceil(order.size / BATCH_SIZE)
        last_batch_heartbeat = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for offset in range(0, order.size, BATCH_SIZE):
                batch_index = offset // BATCH_SIZE + 1
                selected = pairs[order[offset : offset + BATCH_SIZE]]
                features, net, direct, auxiliary, symbols = _batch_arrays(
                    normalized_features, targets, dataset, target_scaler, selected
                )
                values = torch.from_numpy(features).to(device)
                net_tensor = torch.from_numpy(net).to(device)
                direct_tensor = torch.from_numpy(direct).to(device)
                auxiliary_tensor = torch.from_numpy(auxiliary).to(device)
                direct_scale = torch.from_numpy(
                    target_scaler.direct_scale_bps[symbols].astype(np.float32)
                ).to(device)
                gain_baseline = torch.from_numpy(
                    target_scaler.gain_mean_bps[symbols].astype(np.float32)
                ).to(device)
                loss_baseline = torch.from_numpy(
                    target_scaler.loss_mean_bps[symbols].astype(np.float32)
                ).to(device)
                peer_components: dict[str, list[float]] = {
                    key: [] for key in batch_losses
                }
                for seed, peer, optimizer in zip(SEEDS, peers, optimizers, strict=True):
                    optimizer.zero_grad(set_to_none=True)
                    output = tuple(
                        value[..., -SUPERVISED_STEPS:] for value in peer(values)
                    )
                    if candidate_id == "direct_action_mean_tcn":
                        objective, losses = direct_objective(
                            output,
                            net_tensor,
                            direct_tensor,
                            auxiliary_tensor,
                            direct_scale,
                        )
                    else:
                        objective, losses = hurdle_objective(
                            output,
                            net_tensor,
                            direct_tensor,
                            auxiliary_tensor,
                            direct_scale,
                            gain_baseline,
                            loss_baseline,
                        )
                    loss_values = {
                        key: float(value.detach().cpu())
                        for key, value in losses.items()
                    }
                    if not all(math.isfinite(value) for value in loss_values.values()):
                        raise RuntimeError(
                            "Round 49 produced a nonfinite training loss: "
                            f"candidate={candidate_id}, seed={seed}, epoch={epoch}, "
                            f"batch={offset // BATCH_SIZE}, losses={loss_values}"
                        )
                    objective.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        peer.parameters(), 1.0, foreach=False
                    )
                    if not math.isfinite(float(gradient_norm.detach().cpu())):
                        raise RuntimeError(
                            "Round 49 rejected a nonfinite gradient update: "
                            f"candidate={candidate_id}, seed={seed}, epoch={epoch}, "
                            f"batch={offset // BATCH_SIZE}"
                        )
                    optimizer.step()
                    for key in peer_components:
                        peer_components[key].append(loss_values[key])
                optimizer_updates += 1
                for key, values_for_key in peer_components.items():
                    batch_losses[key].append(float(np.mean(values_for_key)))
                heartbeat_now = time.perf_counter()
                if progress is not None and (
                    heartbeat_now - last_batch_heartbeat >= 30.0
                    or batch_index == batches_in_epoch
                ):
                    progress(
                        "round49_training_batch",
                        {
                            "status": "running",
                            "candidate_id": candidate_id,
                            "epoch": epoch,
                            "batch": batch_index,
                            "batches_in_epoch": batches_in_epoch,
                            "optimizer_updates": optimizer_updates,
                            "latest_composite": batch_losses["composite"][-1],
                        },
                    )
                    last_batch_heartbeat = heartbeat_now
        warning_messages.extend(str(item.message) for item in caught)

        validation_losses: list[dict[str, float]] = []
        validation_indices: np.ndarray | None = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for seed, peer in zip(SEEDS, peers, strict=True):
                if progress is not None:
                    progress(
                        "round49_early_stop_seed",
                        {
                            "status": "started",
                            "candidate_id": candidate_id,
                            "epoch": epoch,
                            "seed": seed,
                        },
                    )
                indices, outputs = _predict_mask(
                    peer,
                    normalized_features,
                    dataset,
                    dataset.role_masks["early_stop"],
                    device,
                )
                if validation_indices is None:
                    validation_indices = indices
                elif not np.array_equal(validation_indices, indices):
                    raise RuntimeError("Round 49 validation indices differ by seed")
                validation_losses.append(
                    _numpy_candidate_losses(
                        candidate_id,
                        outputs,
                        dataset,
                        targets,
                        target_scaler,
                        indices,
                    )
                )
                if progress is not None:
                    progress(
                        "round49_early_stop_seed",
                        {
                            "status": "complete",
                            "candidate_id": candidate_id,
                            "epoch": epoch,
                            "seed": seed,
                        },
                    )
        warning_messages.extend(str(item.message) for item in caught)
        if validation_indices is None:
            raise RuntimeError("Round 49 validation produced no indices")
        validation = {
            key: float(np.mean([item[key] for item in validation_losses]))
            for key in validation_losses[0]
        }
        row: dict[str, object] = {
            "candidate_id": candidate_id,
            "epoch": epoch,
            "optimizer_updates": optimizer_updates,
            "training_windows": int(pairs.shape[0]),
            **{
                f"training_{key}": float(np.mean(values_for_key))
                for key, values_for_key in batch_losses.items()
            },
            **{f"early_stop_{key}": value for key, value in validation.items()},
        }
        if not all(
            math.isfinite(value) for value in row.values() if isinstance(value, float)
        ):
            raise RuntimeError("Round 49 epoch diagnostics are nonfinite")
        if validation["composite"] < best_loss - MINIMUM_IMPROVEMENT:
            best_loss = validation["composite"]
            best_epoch = epoch
            best_states = [_clone_state(peer) for peer in peers]
            best_losses = dict(validation)
            stale_epochs = 0
        else:
            stale_epochs += 1
        row["best_early_stop_composite"] = best_loss
        row["best_epoch"] = best_epoch
        row["stale_epochs"] = stale_epochs
        history.append(row)
        if progress is not None:
            progress("round49_epoch", row)
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            break

    if best_states is None or best_losses is None:
        raise RuntimeError("Round 49 training did not produce a best state")
    fallback = _fallback_messages(warning_messages)
    if fallback:
        raise RuntimeError(f"Round 49 training used CPU fallback: {fallback}")

    analysis_mask = np.logical_or.reduce(
        [dataset.role_masks[role] for role in ANALYSIS_ROLES]
    )
    probe_mask = np.zeros(dataset.timestamps, dtype=bool)
    probe_indices = np.flatnonzero(dataset.role_masks["calibration"])[:512]
    probe_mask[probe_indices] = True
    artifacts: list[ActionHurdleArtifact] = []
    seed_probabilities: list[np.ndarray] = []
    seed_action_values: list[np.ndarray] = []
    seed_gain_means: list[np.ndarray] = []
    seed_loss_means: list[np.ndarray] = []
    seed_auxiliary: list[np.ndarray] = []
    global_indices: np.ndarray | None = None

    for peer, seed, best_state in zip(peers, SEEDS, best_states, strict=True):
        if progress is not None:
            progress(
                "round49_artifact_seed",
                {
                    "status": "started",
                    "candidate_id": candidate_id,
                    "seed": seed,
                },
            )
        peer.load_state_dict(best_state)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reference_probe = _predict_mask(
                peer, normalized_features, dataset, probe_mask, device
            )
        warning_messages.extend(str(item.message) for item in caught)
        model_path = model_dir / f"round49_{candidate_id}_seed_{seed}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, model_path)
        reloaded = _candidate_model(candidate_id)
        reloaded.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        reloaded = reloaded.to(device)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            reloaded_probe = _predict_mask(
                reloaded, normalized_features, dataset, probe_mask, device
            )
            indices, outputs = _predict_mask(
                reloaded, normalized_features, dataset, analysis_mask, device
            )
        warning_messages.extend(str(item.message) for item in caught)
        if not np.array_equal(reference_probe[0], reloaded_probe[0]):
            raise RuntimeError("Round 49 reload probe indexing failed")
        reload_errors = tuple(
            float(np.max(np.abs(reference - observed)))
            for reference, observed in zip(
                reference_probe[1], reloaded_probe[1], strict=True
            )
        )
        if not all(math.isfinite(value) and value <= 1e-6 for value in reload_errors):
            raise RuntimeError(f"Round 49 reload errors are {reload_errors}")
        if global_indices is None:
            global_indices = indices
        elif not np.array_equal(global_indices, indices):
            raise RuntimeError("Round 49 final indices differ by seed")

        calibration_local = dataset.role_masks["calibration"][indices]
        calibration_labels = targets[indices[calibration_local]] > 0.0
        probability_calibration = fit_probability_calibration(
            outputs[0][calibration_local], calibration_labels
        )
        probabilities = apply_probability_calibration(
            outputs[0], probability_calibration
        )
        gain_multiplier = 1.0
        loss_multiplier = 1.0
        gain_before: float | None = None
        gain_after: float | None = None
        loss_before: float | None = None
        loss_after: float | None = None
        gain_means: np.ndarray | None = None
        loss_means: np.ndarray | None = None
        if candidate_id == "direct_action_mean_tcn":
            action_values = (
                outputs[1]
                * target_scaler.direct_scale_bps.reshape(1, len(SYMBOLS), len(SIDES))
                + target_scaler.direct_mean_bps.reshape(1, len(SYMBOLS), len(SIDES))
            ).astype(np.float32)
        else:
            raw_gain_means = outputs[1] * target_scaler.gain_mean_bps.reshape(
                1, len(SYMBOLS), len(SIDES)
            )
            raw_loss_means = outputs[2] * target_scaler.loss_mean_bps.reshape(
                1, len(SYMBOLS), len(SIDES)
            )
            primary_calibration = targets[
                indices[calibration_local], :, PRIMARY_HORIZON_INDEX
            ]
            gain_multiplier, gain_before, gain_after = fit_severity_multiplier(
                raw_gain_means[calibration_local],
                np.clip(primary_calibration, 0.0, None),
                primary_calibration > 0.0,
            )
            loss_multiplier, loss_before, loss_after = fit_severity_multiplier(
                raw_loss_means[calibration_local],
                np.clip(-primary_calibration, 0.0, None),
                primary_calibration <= 0.0,
            )
            gain_means = (gain_multiplier * raw_gain_means).astype(np.float32)
            loss_means = (loss_multiplier * raw_loss_means).astype(np.float32)
            primary_probability = probabilities[:, :, PRIMARY_HORIZON_INDEX]
            action_values = (
                primary_probability * gain_means
                - (1.0 - primary_probability) * loss_means
            ).astype(np.float32)
        auxiliary_mean = (
            outputs[-1] * target_scaler.auxiliary_scale_bps.reshape(1, len(SYMBOLS))
            + target_scaler.auxiliary_median_bps.reshape(1, len(SYMBOLS))
        ).astype(np.float32)
        arrays_to_check = [probabilities, action_values, auxiliary_mean]
        if gain_means is not None and loss_means is not None:
            arrays_to_check.extend((gain_means, loss_means))
        if not all(np.isfinite(value).all() for value in arrays_to_check):
            raise RuntimeError("Round 49 calibrated predictions are nonfinite")
        prediction_path = prediction_dir / f"round49_{candidate_id}_seed_{seed}.npz"
        _save_prediction_artifact(
            prediction_path,
            indices=indices,
            probabilities=probabilities,
            action_values_bps=action_values,
            gain_means_bps=gain_means,
            loss_means_bps=loss_means,
            auxiliary_mean_bps=auxiliary_mean,
            calibration=probability_calibration,
            gain_multiplier=gain_multiplier,
            loss_multiplier=loss_multiplier,
        )
        mapped_errors = (
            reload_errors
            if candidate_id == "hurdle_action_value_tcn"
            else (reload_errors[0], reload_errors[1], 0.0, reload_errors[2])
        )
        artifact = ActionHurdleArtifact(
            candidate_id=candidate_id,
            seed=seed,
            epochs=epochs_run,
            best_epoch=best_epoch,
            best_early_stop_composite=float(best_losses["composite"]),
            best_early_stop_profit_bce=float(best_losses["profit_bce"]),
            best_early_stop_primary_loss=float(best_losses["primary_loss"]),
            best_early_stop_auxiliary_mse=float(best_losses["auxiliary_mse"]),
            best_early_stop_pairwise_rank_loss=float(best_losses["pairwise_rank_loss"]),
            optimizer_updates=optimizer_updates,
            parameter_count=sum(value.numel() for value in reloaded.parameters()),
            probability_slope=probability_calibration.slope,
            probability_intercept=probability_calibration.intercept,
            calibration_binary_log_loss_before=(
                probability_calibration.binary_log_loss_before
            ),
            calibration_binary_log_loss_after=(
                probability_calibration.binary_log_loss_after
            ),
            gain_multiplier=gain_multiplier,
            loss_multiplier=loss_multiplier,
            calibration_gain_score_before=gain_before,
            calibration_gain_score_after=gain_after,
            calibration_loss_score_before=loss_before,
            calibration_loss_score_after=loss_after,
            backend_kind=backend_kind,
            backend_device=backend_device,
            model_path=str(model_path),
            model_bytes=model_path.stat().st_size,
            model_sha256=_file_sha256(model_path),
            prediction_path=str(prediction_path),
            prediction_bytes=prediction_path.stat().st_size,
            prediction_sha256=_file_sha256(prediction_path),
            reload_max_abs_logit_error=mapped_errors[0],
            reload_max_abs_primary_error=mapped_errors[1],
            reload_max_abs_secondary_error=mapped_errors[2],
            reload_max_abs_auxiliary_error=mapped_errors[3],
            warning_count=len(warning_messages),
        )
        artifacts.append(artifact)
        seed_probabilities.append(probabilities)
        seed_action_values.append(action_values)
        seed_auxiliary.append(auxiliary_mean)
        if gain_means is not None and loss_means is not None:
            seed_gain_means.append(gain_means)
            seed_loss_means.append(loss_means)
        if progress is not None:
            progress(
                "round49_artifact_seed",
                {
                    "status": "complete",
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "model_sha256": artifact.model_sha256,
                    "prediction_sha256": artifact.prediction_sha256,
                },
            )
    fallback = _fallback_messages(warning_messages)
    if fallback:
        raise RuntimeError(f"Round 49 finalization used CPU fallback: {fallback}")
    if global_indices is None:
        raise RuntimeError("Round 49 finalization produced no indices")
    return ActionHurdleForecastBundle(
        candidate_id=candidate_id,
        global_indices=global_indices,
        seed_probabilities=np.stack(seed_probabilities).astype(np.float32),
        seed_action_values_bps=np.stack(seed_action_values).astype(np.float32),
        seed_gain_means_bps=(
            np.stack(seed_gain_means).astype(np.float32) if seed_gain_means else None
        ),
        seed_loss_means_bps=(
            np.stack(seed_loss_means).astype(np.float32) if seed_loss_means else None
        ),
        seed_auxiliary_mean_bps=np.stack(seed_auxiliary).astype(np.float32),
        artifacts=tuple(artifacts),
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        backend_kind=backend_kind,
        backend_device=backend_device,
        training_history=tuple(history),
    )


def train_action_hurdle_candidates(
    dataset: MinuteTemporalDataset,
    *,
    model_dir: Path,
    prediction_dir: Path,
    compute_backend: str = "auto",
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, ActionHurdleForecastBundle], dict[str, object]]:
    device, preflight = action_hurdle_preflight(compute_backend)
    if progress is not None:
        progress("round49_preflight", {"status": "complete", **preflight})
    feature_scaler = fit_robust_feature_scaler(dataset)
    normalized_features = feature_scaler.transform(dataset.features)
    targets = side_net_targets(dataset)
    target_scaler = fit_action_target_scaler(dataset, targets)
    if progress is not None:
        progress(
            "round49_scaling",
            {
                "status": "complete",
                "feature_bytes": int(normalized_features.nbytes),
                "target_bytes": int(targets.nbytes),
                "training_windows": int(_training_pairs(dataset).shape[0]),
            },
        )
    bundles: dict[str, ActionHurdleForecastBundle] = {}
    for candidate_id in CANDIDATES:
        bundles[candidate_id] = _train_candidate(
            dataset,
            normalized_features,
            targets,
            feature_scaler,
            target_scaler,
            candidate_id=candidate_id,
            model_dir=model_dir,
            prediction_dir=prediction_dir,
            device=device,
            backend_kind=str(preflight["backend_kind"]),
            backend_device=str(preflight["backend_device"]),
            progress=progress,
        )
    return bundles, preflight


__all__ = [
    "action_hurdle_preflight",
    "ActionHurdleArtifact",
    "ActionHurdleForecastBundle",
    "ActionTargetScaler",
    "AUXILIARY_HORIZON_MINUTES",
    "CANDIDATES",
    "DirectActionMeanTCN",
    "HORIZONS_MINUTES",
    "HurdleActionValueTCN",
    "PRIMARY_HORIZON_MINUTES",
    "RECEPTIVE_FIELD_STEPS",
    "ROUND",
    "SEEDS",
    "SIDES",
    "SUPERVISED_STEPS",
    "WINDOW_STEPS",
    "apply_probability_calibration",
    "binary_logit_loss",
    "build_action_hurdle_temporal_dataset",
    "cpu_action_hurdle_preflight",
    "direct_objective",
    "directml_action_hurdle_preflight",
    "fit_action_target_scaler",
    "fit_probability_calibration",
    "fit_severity_multiplier",
    "gamma_mean_score",
    "hurdle_expected_net_bps",
    "hurdle_objective",
    "numpy_gamma_mean_score",
    "pairwise_action_rank_loss",
    "side_net_targets",
    "train_action_hurdle_candidates",
]
