"""Path-bounded competing-risk TCN candidates for Round 50."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence
import warnings

import numpy as np
from scipy.optimize import minimize_scalar
import torch
from torch import nn
from torch.nn import functional as F

from .compute import require_backend, resolve_backend, torch_device_for_backend
from .action_hurdle_tcn_model import (
    ProbabilityCalibration,
    apply_probability_calibration,
    fit_probability_calibration,
)
from .barrier_payoff_data import (
    STOP_EVENT,
    TAKE_PROFIT_EVENT,
    TIMEOUT_EVENT,
    BarrierPayoffDataset,
)
from .cross_asset_cost_data import MINUTE_MS, SYMBOLS
from .distributional_tcn_model import ExplicitAdamW
from .minute_logistic_mixture_tcn_model import (
    DEPTHWISE_DILATIONS,
    FEATURE_COUNT,
    HIDDEN_CHANNELS,
    LargeKernelCausalBlock,
    MinuteTemporalDataset,
    PerTimestampLayerNorm,
    RobustFeatureScaler,
)


ROUND = 50
CANDIDATES = ("direct_barrier_mean_tcn", "competing_risk_barrier_tcn")
SEEDS = (5001, 5002, 5003)
SIDES = (-1, 1)
HORIZON_MINUTES = 60
EVENT_MINUTES = HORIZON_MINUTES
EVENT_CLASSES = 2 * EVENT_MINUTES + 1
TIMEOUT_CLASS = EVENT_CLASSES - 1
RECEPTIVE_FIELD_STEPS = 361
WINDOW_STEPS = 576
SUPERVISED_STEPS = WINDOW_STEPS - RECEPTIVE_FIELD_STEPS + 1
WINDOW_STRIDE_STEPS = SUPERVISED_STEPS
BATCH_SIZE = 32
PREDICTION_CHUNK_STEPS = 1_024
MAXIMUM_EPOCHS = 40
EARLY_STOPPING_PATIENCE = 8
MINIMUM_IMPROVEMENT = 1e-4
EVENT_LOG_LOSS_WEIGHT = 1.0
TIMEOUT_PROFIT_BCE_WEIGHT = 0.25
TIMEOUT_MEAN_MSE_WEIGHT = 0.5
DIRECT_MEAN_MSE_WEIGHT = 0.5
RISK_UNIT_OUTPUT_LIMIT = 8.0
EVENT_TEMPERATURE_BOUNDS = (0.5, 3.0)
ANALYSIS_ROLES = ("calibration", "viability")
ProgressCallback = Callable[[str, Mapping[str, object]], None]


@dataclass(frozen=True)
class BarrierTargetBaselines:
    direct_mean_risk_units: np.ndarray
    stop_residual_mean_risk_units: np.ndarray
    take_residual_mean_risk_units: np.ndarray
    event_class_probability: np.ndarray
    pooled_event_class_probability: np.ndarray
    pooled_timeout_profit_probability: np.ndarray
    pooled_timeout_mean_risk_units: np.ndarray
    pooled_direct_mean_risk_units: np.ndarray

    def asdict(self) -> dict[str, object]:
        return {
            field: np.asarray(value).tolist() for field, value in asdict(self).items()
        }


@dataclass(frozen=True)
class EventTemperatureCalibration:
    temperature: float
    multinomial_log_loss_before: float
    multinomial_log_loss_after: float


@dataclass(frozen=True)
class BarrierCompetingRiskArtifact:
    candidate_id: str
    seed: int
    epochs: int
    best_epoch: int
    best_early_stop_composite: float
    best_early_stop_event_log_loss: float
    best_early_stop_timeout_profit_bce: float
    best_early_stop_timeout_mean_mse: float
    best_early_stop_primary_mse: float
    optimizer_updates: int
    parameter_count: int
    event_temperature: float
    calibration_event_log_loss_before: float
    calibration_event_log_loss_after: float
    timeout_probability_slope: float
    timeout_probability_intercept: float
    calibration_timeout_log_loss_before: float
    calibration_timeout_log_loss_after: float
    backend_kind: str
    backend_device: str
    model_path: str
    model_bytes: int
    model_sha256: str
    prediction_path: str
    prediction_bytes: int
    prediction_sha256: str
    reload_max_abs_event_logit_error: float
    reload_max_abs_timeout_profit_logit_error: float
    reload_max_abs_timeout_mean_error: float
    reload_max_abs_direct_mean_error: float
    runtime_repeat_max_abs_event_logit_error: float
    runtime_repeat_max_abs_timeout_profit_logit_error: float
    runtime_repeat_max_abs_timeout_mean_error: float
    runtime_repeat_max_abs_direct_mean_error: float
    warning_count: int

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BarrierCompetingRiskForecastBundle:
    candidate_id: str
    global_indices: np.ndarray
    seed_event_true_probabilities: np.ndarray
    seed_event_probability_square_sums: np.ndarray
    seed_event_group_probabilities: np.ndarray
    seed_event_expected_minutes: np.ndarray
    seed_timeout_profit_probabilities: np.ndarray
    seed_timeout_mean_risk_units: np.ndarray
    seed_action_values_bps: np.ndarray
    artifacts: tuple[BarrierCompetingRiskArtifact, ...]
    feature_scaler: RobustFeatureScaler
    target_baselines: BarrierTargetBaselines
    backend_kind: str
    backend_device: str
    training_history: tuple[Mapping[str, object], ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def barrier_event_classes(dataset: BarrierPayoffDataset) -> np.ndarray:
    events = dataset.event_code
    minutes = dataset.event_minute.astype(np.int64, copy=False)
    if events.shape != minutes.shape or events.ndim != 3:
        raise ValueError("Round 50 event tensors are invalid")
    if np.any(minutes < 1) or np.any(minutes > HORIZON_MINUTES):
        raise ValueError("Round 50 event minute is outside the horizon")
    labels = np.full(events.shape, TIMEOUT_CLASS, dtype=np.int64)
    stopped = events == STOP_EVENT
    taken = events == TAKE_PROFIT_EVENT
    timed_out = events == TIMEOUT_EVENT
    if not np.all(stopped | taken | timed_out):
        raise ValueError("Round 50 event code is unsupported")
    labels[stopped] = minutes[stopped] - 1
    labels[taken] = EVENT_MINUTES + minutes[taken] - 1
    if np.any(timed_out & (minutes != HORIZON_MINUTES)):
        raise ValueError("Round 50 timeout does not end at the horizon")
    return labels


def barrier_risk_targets(dataset: BarrierPayoffDataset) -> np.ndarray:
    stop = dataset.stop_bps[..., None].astype(np.float64, copy=False)
    targets = dataset.net_payoff_bps.astype(np.float64, copy=False) / stop
    if not np.isfinite(targets).all():
        raise ValueError("Round 50 risk-unit targets are nonfinite")
    return targets.astype(np.float32)


def fit_barrier_feature_scaler(
    temporal: MinuteTemporalDataset,
    barrier: BarrierPayoffDataset,
) -> RobustFeatureScaler:
    selected = temporal.features[barrier.role_masks["training"]].reshape(
        -1, temporal.features.shape[-1]
    )
    quartiles = np.quantile(selected, (0.25, 0.5, 0.75), axis=0)
    median = quartiles[1].astype(np.float64)
    scaled_iqr = np.maximum((quartiles[2] - quartiles[0]) / 1.349, 1e-6)
    if not np.isfinite(median).all() or not np.isfinite(scaled_iqr).all():
        raise ValueError("Round 50 feature scaler is nonfinite")
    return RobustFeatureScaler(median=median, scaled_iqr=scaled_iqr)


def fit_barrier_target_baselines(
    dataset: BarrierPayoffDataset,
    event_classes: np.ndarray,
    risk_targets: np.ndarray,
) -> BarrierTargetBaselines:
    training = dataset.role_masks["training"]
    direct_mean = np.mean(risk_targets[training], axis=0, dtype=np.float64)
    stop_residual = np.empty((len(SYMBOLS), len(SIDES)), dtype=np.float64)
    take_residual = np.empty_like(stop_residual)
    event_probability = np.empty(
        (len(SYMBOLS), len(SIDES), EVENT_CLASSES), dtype=np.float64
    )
    pooled_counts = np.zeros((len(SIDES), EVENT_CLASSES), dtype=np.float64)
    pooled_timeout_profit = np.empty(len(SIDES), dtype=np.float64)
    pooled_timeout_mean = np.empty(len(SIDES), dtype=np.float64)
    pooled_direct_mean = np.empty(len(SIDES), dtype=np.float64)
    training_stop = dataset.stop_bps[training].astype(np.float64)
    training_risk = risk_targets[training].astype(np.float64)
    training_events = dataset.event_code[training]
    training_classes = event_classes[training]
    execution_charge = dataset.specification.round_trip_execution_charge_bps
    reward_risk = dataset.specification.take_profit_to_stop_ratio
    for symbol_index in range(len(SYMBOLS)):
        cost_ratio = execution_charge / training_stop[:, symbol_index]
        for side_index in range(len(SIDES)):
            values = training_risk[:, symbol_index, side_index]
            events = training_events[:, symbol_index, side_index]
            classes = training_classes[:, symbol_index, side_index]
            stopped = events == STOP_EVENT
            taken = events == TAKE_PROFIT_EVENT
            if not np.any(stopped) or not np.any(taken):
                raise ValueError("Round 50 training event residual role is empty")
            stop_residual[symbol_index, side_index] = float(
                np.mean(values[stopped] - (-1.0 - cost_ratio[stopped]))
            )
            take_residual[symbol_index, side_index] = float(
                np.mean(values[taken] - (reward_risk - cost_ratio[taken]))
            )
            counts = np.bincount(classes, minlength=EVENT_CLASSES).astype(np.float64)
            if np.any(counts <= 0.0):
                raise ValueError(
                    "Round 50 training role has an empty exact event-time class"
                )
            event_probability[symbol_index, side_index] = counts / counts.sum()
            pooled_counts[side_index] += counts
    for side_index in range(len(SIDES)):
        side_events = training_events[:, :, side_index]
        side_values = training_risk[:, :, side_index]
        timeout = side_events == TIMEOUT_EVENT
        if not np.any(timeout):
            raise ValueError("Round 50 pooled timeout role is empty")
        pooled_timeout_profit[side_index] = float(np.mean(side_values[timeout] > 0.0))
        pooled_timeout_mean[side_index] = float(np.mean(side_values[timeout]))
        pooled_direct_mean[side_index] = float(np.mean(side_values))
    pooled_event = pooled_counts / np.sum(pooled_counts, axis=1, keepdims=True)
    arrays = (
        direct_mean,
        stop_residual,
        take_residual,
        event_probability,
        pooled_event,
        pooled_timeout_profit,
        pooled_timeout_mean,
        pooled_direct_mean,
    )
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("Round 50 target baselines are nonfinite")
    if np.any((pooled_timeout_profit <= 0.0) | (pooled_timeout_profit >= 1.0)):
        raise ValueError("Round 50 pooled timeout profit prior is degenerate")
    return BarrierTargetBaselines(
        direct_mean_risk_units=direct_mean,
        stop_residual_mean_risk_units=stop_residual,
        take_residual_mean_risk_units=take_residual,
        event_class_probability=event_probability,
        pooled_event_class_probability=pooled_event,
        pooled_timeout_profit_probability=pooled_timeout_profit,
        pooled_timeout_mean_risk_units=pooled_timeout_mean,
        pooled_direct_mean_risk_units=pooled_direct_mean,
    )


def _inverse_bounded_mean(value: np.ndarray) -> np.ndarray:
    ratio = np.clip(value / RISK_UNIT_OUTPUT_LIMIT, -0.999, 0.999)
    return RISK_UNIT_OUTPUT_LIMIT * np.arctanh(ratio)


class BarrierEncoder(nn.Module):
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
            raise ValueError("Round 50 model input dimensions are invalid")
        encoded = F.gelu(self.projection(values))
        for block in self.blocks:
            encoded = block(encoded)
        return self.final_normalization(encoded)


class _BarrierSharedTCN(nn.Module):
    def __init__(
        self,
        baselines: BarrierTargetBaselines,
        *,
        input_channels: int = FEATURE_COUNT,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = BarrierEncoder(input_channels, dropout)
        self.event_head = nn.Conv1d(
            HIDDEN_CHANNELS, len(SIDES) * EVENT_CLASSES, kernel_size=1
        )
        self.timeout_profit_head = nn.Conv1d(HIDDEN_CHANNELS, len(SIDES), kernel_size=1)
        self.timeout_mean_head = nn.Conv1d(HIDDEN_CHANNELS, len(SIDES), kernel_size=1)
        with torch.no_grad():
            if self.event_head.bias is not None:
                self.event_head.bias.copy_(
                    torch.from_numpy(
                        np.log(baselines.pooled_event_class_probability).reshape(-1)
                    ).to(dtype=self.event_head.bias.dtype)
                )
            if self.timeout_profit_head.bias is not None:
                probability = baselines.pooled_timeout_profit_probability
                logits = np.log(probability / (1.0 - probability))
                self.timeout_profit_head.bias.copy_(
                    torch.from_numpy(logits).to(
                        dtype=self.timeout_profit_head.bias.dtype
                    )
                )
            if self.timeout_mean_head.bias is not None:
                self.timeout_mean_head.bias.copy_(
                    torch.from_numpy(
                        _inverse_bounded_mean(baselines.pooled_timeout_mean_risk_units)
                    ).to(dtype=self.timeout_mean_head.bias.dtype)
                )

    def shared_outputs(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(values)
        event_logits = self.event_head(encoded).reshape(
            values.shape[0], len(SIDES), EVENT_CLASSES, values.shape[-1]
        )
        timeout_profit_logits = self.timeout_profit_head(encoded)
        raw_timeout_mean = self.timeout_mean_head(encoded)
        timeout_mean = RISK_UNIT_OUTPUT_LIMIT * torch.tanh(
            raw_timeout_mean / RISK_UNIT_OUTPUT_LIMIT
        )
        return encoded, event_logits, timeout_profit_logits, timeout_mean


class DirectBarrierMeanTCN(_BarrierSharedTCN):
    candidate_id = "direct_barrier_mean_tcn"

    def __init__(
        self,
        baselines: BarrierTargetBaselines,
        *,
        input_channels: int = FEATURE_COUNT,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(baselines, input_channels=input_channels, dropout=dropout)
        self.direct_mean_head = nn.Conv1d(HIDDEN_CHANNELS, len(SIDES), kernel_size=1)
        with torch.no_grad():
            if self.direct_mean_head.bias is not None:
                self.direct_mean_head.bias.copy_(
                    torch.from_numpy(
                        _inverse_bounded_mean(baselines.pooled_direct_mean_risk_units)
                    ).to(dtype=self.direct_mean_head.bias.dtype)
                )

    def forward(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded, event, timeout_profit, timeout_mean = self.shared_outputs(values)
        raw_direct = self.direct_mean_head(encoded)
        direct = RISK_UNIT_OUTPUT_LIMIT * torch.tanh(
            raw_direct / RISK_UNIT_OUTPUT_LIMIT
        )
        return event, timeout_profit, timeout_mean, direct


class CompetingRiskBarrierTCN(_BarrierSharedTCN):
    candidate_id = "competing_risk_barrier_tcn"

    def forward(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, event, timeout_profit, timeout_mean = self.shared_outputs(values)
        return event, timeout_profit, timeout_mean


def _event_log_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 4 or labels.shape != (
        logits.shape[0],
        logits.shape[1],
        logits.shape[3],
    ):
        raise ValueError("Round 50 event log-loss shapes are invalid")
    selected = torch.gather(
        F.log_softmax(logits, dim=2), 2, labels.unsqueeze(2)
    ).squeeze(2)
    return -torch.mean(selected)


def _masked_binary_logit_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if logits.shape != labels.shape or labels.shape != mask.shape:
        raise ValueError("Round 50 timeout BCE shapes are invalid")
    weights = mask.to(dtype=logits.dtype)
    stable = (
        torch.maximum(logits, torch.zeros_like(logits))
        - logits * labels
        + torch.log1p(torch.exp(-torch.abs(logits)))
    )
    return torch.sum(stable * weights) / torch.clamp(torch.sum(weights), min=1.0)


def _masked_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or target.shape != mask.shape:
        raise ValueError("Round 50 timeout MSE shapes are invalid")
    weights = mask.to(dtype=prediction.dtype)
    return torch.sum((prediction - target) ** 2 * weights) / torch.clamp(
        torch.sum(weights), min=1.0
    )


def _structured_expected_risk_torch(
    event_logits: torch.Tensor,
    timeout_mean: torch.Tensor,
    stop_bps: torch.Tensor,
    symbol_indices: torch.Tensor,
    baselines: BarrierTargetBaselines,
    *,
    execution_charge_bps: float,
    reward_risk: float,
) -> torch.Tensor:
    probabilities = torch.softmax(event_logits, dim=2)
    stop_probability = torch.sum(probabilities[:, :, :EVENT_MINUTES], dim=2)
    take_probability = torch.sum(
        probabilities[:, :, EVENT_MINUTES : 2 * EVENT_MINUTES], dim=2
    )
    timeout_probability = probabilities[:, :, TIMEOUT_CLASS]
    cost_ratio = execution_charge_bps / stop_bps.to(dtype=event_logits.dtype)
    stop_residual = torch.from_numpy(
        baselines.stop_residual_mean_risk_units.astype(np.float32)
    ).to(event_logits.device)[symbol_indices]
    take_residual = torch.from_numpy(
        baselines.take_residual_mean_risk_units.astype(np.float32)
    ).to(event_logits.device)[symbol_indices]
    stop_value = -1.0 - cost_ratio[:, None, :] + stop_residual[..., None]
    take_value = reward_risk - cost_ratio[:, None, :] + take_residual[..., None]
    return (
        stop_probability * stop_value
        + take_probability * take_value
        + timeout_probability * timeout_mean
    )


def candidate_objective(
    candidate_id: str,
    output: tuple[torch.Tensor, ...],
    event_labels: torch.Tensor,
    risk_targets: torch.Tensor,
    stop_bps: torch.Tensor,
    symbol_indices: torch.Tensor,
    baselines: BarrierTargetBaselines,
    *,
    execution_charge_bps: float,
    reward_risk: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    event_logits, timeout_profit_logits, timeout_mean = output[:3]
    timeout_mask = event_labels == TIMEOUT_CLASS
    event_loss = _event_log_loss(event_logits, event_labels)
    timeout_profit_bce = _masked_binary_logit_loss(
        timeout_profit_logits,
        (risk_targets > 0.0).to(dtype=timeout_profit_logits.dtype),
        timeout_mask,
    )
    timeout_mean_mse = _masked_mse(timeout_mean, risk_targets, timeout_mask)
    if candidate_id == "direct_barrier_mean_tcn":
        if len(output) != 4:
            raise ValueError("Round 50 direct output is incomplete")
        action_risk = output[3]
        primary_weight = DIRECT_MEAN_MSE_WEIGHT
    elif candidate_id == "competing_risk_barrier_tcn":
        if len(output) != 3:
            raise ValueError("Round 50 competing-risk output is invalid")
        action_risk = _structured_expected_risk_torch(
            event_logits,
            timeout_mean,
            stop_bps,
            symbol_indices,
            baselines,
            execution_charge_bps=execution_charge_bps,
            reward_risk=reward_risk,
        )
        primary_weight = 0.0
    else:
        raise KeyError(candidate_id)
    primary_mse = F.mse_loss(action_risk, risk_targets)
    composite = (
        EVENT_LOG_LOSS_WEIGHT * event_loss
        + TIMEOUT_PROFIT_BCE_WEIGHT * timeout_profit_bce
        + TIMEOUT_MEAN_MSE_WEIGHT * timeout_mean_mse
        + primary_weight * primary_mse
    )
    return composite, {
        "event_log_loss": event_loss,
        "timeout_profit_bce": timeout_profit_bce,
        "timeout_mean_mse": timeout_mean_mse,
        "primary_mse": primary_mse,
        "composite": composite,
    }


def _fallback_messages(messages: Sequence[str]) -> list[str]:
    return [
        item
        for item in messages
        if "not currently supported on the DML backend" in item
        or "fall back to run on the CPU" in item
    ]


def _synthetic_baselines() -> BarrierTargetBaselines:
    event = np.full(
        (len(SYMBOLS), len(SIDES), EVENT_CLASSES),
        1.0 / EVENT_CLASSES,
        dtype=np.float64,
    )
    return BarrierTargetBaselines(
        direct_mean_risk_units=np.full((len(SYMBOLS), len(SIDES)), -0.25),
        stop_residual_mean_risk_units=np.zeros((len(SYMBOLS), len(SIDES))),
        take_residual_mean_risk_units=np.zeros((len(SYMBOLS), len(SIDES))),
        event_class_probability=event,
        pooled_event_class_probability=np.mean(event, axis=0),
        pooled_timeout_profit_probability=np.full(len(SIDES), 0.4),
        pooled_timeout_mean_risk_units=np.full(len(SIDES), -0.2),
        pooled_direct_mean_risk_units=np.full(len(SIDES), -0.25),
    )


def _candidate_model(candidate_id: str, baselines: BarrierTargetBaselines) -> nn.Module:
    if candidate_id == "direct_barrier_mean_tcn":
        return DirectBarrierMeanTCN(baselines)
    if candidate_id == "competing_risk_barrier_tcn":
        return CompetingRiskBarrierTCN(baselines)
    raise KeyError(candidate_id)


def _preflight(
    device: object, *, backend_kind: str, backend_device: str
) -> dict[str, object]:
    generator = np.random.default_rng(SEEDS[0])
    baselines = _synthetic_baselines()
    reports: list[dict[str, object]] = []
    warning_messages: list[str] = []
    for candidate_id in CANDIDATES:
        torch.manual_seed(SEEDS[0])
        model = _candidate_model(candidate_id, baselines).to(device)
        optimizer = ExplicitAdamW(
            tuple(model.parameters()), learning_rate=8e-4, weight_decay=1e-4
        )
        values = torch.from_numpy(
            generator.normal(size=(3, FEATURE_COUNT, 400)).astype(np.float32)
        ).to(device)
        labels_array = generator.integers(
            0, EVENT_CLASSES, size=(3, len(SIDES), 400), dtype=np.int64
        )
        labels_array[..., ::7] = TIMEOUT_CLASS
        labels = torch.from_numpy(labels_array).to(device)
        risk = torch.from_numpy(
            generator.normal(-0.2, 0.9, size=(3, len(SIDES), 400)).astype(np.float32)
        ).to(device)
        stop = torch.from_numpy(
            generator.uniform(24.0, 80.0, size=(3, 400)).astype(np.float32)
        ).to(device)
        symbols = torch.arange(3, dtype=torch.long, device=device)
        before = {
            "encoder": model.encoder.blocks[0].depthwise.weight.detach().cpu().clone(),
            "event": model.event_head.weight.detach().cpu().clone(),
            "timeout": model.timeout_mean_head.weight.detach().cpu().clone(),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            optimizer.zero_grad(set_to_none=True)
            objective, losses = candidate_objective(
                candidate_id,
                model(values),
                labels,
                risk,
                stop,
                symbols,
                baselines,
                execution_charge_bps=12.0,
                reward_risk=2.0,
            )
            objective.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, foreach=False
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            extreme = risk.clone()
            extreme[..., ::2] = 50.0
            extreme[..., 1::2] = -50.0
            tail_objective, _ = candidate_objective(
                candidate_id,
                model(values),
                labels,
                extreme,
                stop,
                symbols,
                baselines,
                execution_charge_bps=12.0,
                reward_risk=2.0,
            )
            tail_objective.backward()
            tail_gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0, foreach=False
            )
        warning_messages.extend(str(item.message) for item in caught)
        changes = {
            "encoder": float(
                torch.max(
                    torch.abs(
                        model.encoder.blocks[0].depthwise.weight.detach().cpu()
                        - before["encoder"]
                    )
                )
            ),
            "event": float(
                torch.max(
                    torch.abs(model.event_head.weight.detach().cpu() - before["event"])
                )
            ),
            "timeout": float(
                torch.max(
                    torch.abs(
                        model.timeout_mean_head.weight.detach().cpu()
                        - before["timeout"]
                    )
                )
            ),
        }
        report = {
            "candidate_id": candidate_id,
            "objective": float(objective.detach().cpu()),
            "gradient_norm": float(gradient_norm.detach().cpu()),
            "extreme_target_objective": float(tail_objective.detach().cpu()),
            "extreme_target_gradient_norm": float(tail_gradient_norm.detach().cpu()),
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
            raise RuntimeError(f"Round 50 numerical preflight failed: {report}")
        reports.append(report)
    fallback = _fallback_messages(warning_messages)
    if fallback:
        raise RuntimeError(f"Round 50 preflight used CPU fallback: {fallback}")
    return {
        "backend_kind": backend_kind,
        "backend_device": backend_device,
        "candidates": reports,
        "warning_count": len(warning_messages),
        "cpu_fallback_warnings": len(fallback),
    }


def barrier_competing_risk_preflight(
    compute_backend: str = "auto",
) -> tuple[object, dict[str, object]]:
    backend = require_backend(resolve_backend(compute_backend))
    device = torch_device_for_backend(backend)
    return device, _preflight(
        device, backend_kind=backend.kind, backend_device=str(device)
    )


def directml_barrier_competing_risk_preflight() -> tuple[object, dict[str, object]]:
    return barrier_competing_risk_preflight("directml")


def cpu_barrier_competing_risk_preflight() -> tuple[object, dict[str, object]]:
    return barrier_competing_risk_preflight("cpu")


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


def _training_pairs(
    temporal: MinuteTemporalDataset, barrier: BarrierPayoffDataset
) -> np.ndarray:
    rows: list[tuple[int, int]] = []
    for start, end in _contiguous_runs(
        barrier.role_masks["training"], temporal.timestamps_ms
    ):
        last_start = end - WINDOW_STEPS
        if last_start < start:
            continue
        for window_start in range(start, last_start + 1, WINDOW_STRIDE_STEPS):
            rows.extend(
                (window_start, symbol_index) for symbol_index in range(len(SYMBOLS))
            )
    if not rows:
        raise ValueError("Round 50 training role has no complete windows")
    return np.asarray(rows, dtype=np.int64)


def _batch_arrays(
    normalized_features: np.ndarray,
    event_classes: np.ndarray,
    risk_targets: np.ndarray,
    barrier: BarrierPayoffDataset,
    pairs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_rows: list[np.ndarray] = []
    event_rows: list[np.ndarray] = []
    risk_rows: list[np.ndarray] = []
    stop_rows: list[np.ndarray] = []
    symbol_indices = pairs[:, 1].astype(np.int64, copy=False)
    supervised_start = RECEPTIVE_FIELD_STEPS - 1
    for start_value, symbol_value in pairs:
        start = int(start_value)
        symbol_index = int(symbol_value)
        stop = start + WINDOW_STEPS
        target_start = start + supervised_start
        feature_rows.append(normalized_features[start:stop, symbol_index].T)
        event_rows.append(event_classes[target_start:stop, symbol_index].T)
        risk_rows.append(risk_targets[target_start:stop, symbol_index].T)
        stop_rows.append(barrier.stop_bps[target_start:stop, symbol_index])
    outputs = (
        np.stack(feature_rows).astype(np.float32, copy=False),
        np.stack(event_rows).astype(np.int64, copy=False),
        np.stack(risk_rows).astype(np.float32, copy=False),
        np.stack(stop_rows).astype(np.float32, copy=False),
        symbol_indices,
    )
    expected = pairs.shape[0]
    if outputs[0].shape != (expected, FEATURE_COUNT, WINDOW_STEPS):
        raise RuntimeError("Round 50 feature batch shape is invalid")
    if outputs[1].shape != (expected, len(SIDES), SUPERVISED_STEPS):
        raise RuntimeError("Round 50 event batch shape is invalid")
    if outputs[2].shape != (expected, len(SIDES), SUPERVISED_STEPS):
        raise RuntimeError("Round 50 risk batch shape is invalid")
    if outputs[3].shape != (expected, SUPERVISED_STEPS):
        raise RuntimeError("Round 50 stop batch shape is invalid")
    return tuple(np.ascontiguousarray(value) for value in outputs[:4]) + (outputs[4],)


def _prediction_array(value: torch.Tensor) -> np.ndarray:
    if value.ndim == 4:
        return value.detach().cpu().numpy().transpose(3, 0, 1, 2).astype(np.float32)
    if value.ndim == 3:
        return value.detach().cpu().numpy().transpose(2, 0, 1).astype(np.float32)
    raise ValueError("Round 50 prediction tensor dimensions are invalid")


def _predict_mask(
    model: nn.Module,
    normalized_features: np.ndarray,
    temporal: MinuteTemporalDataset,
    mask: np.ndarray,
    device: object,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        raise ValueError("Round 50 prediction mask is empty")
    is_direct = isinstance(model, DirectBarrierMeanTCN)
    event = np.full(
        (indices.size, len(SYMBOLS), len(SIDES), EVENT_CLASSES),
        np.nan,
        dtype=np.float32,
    )
    timeout_profit = np.full(
        (indices.size, len(SYMBOLS), len(SIDES)), np.nan, dtype=np.float32
    )
    timeout_mean = np.full_like(timeout_profit, np.nan)
    destinations: list[np.ndarray] = [event, timeout_profit, timeout_mean]
    if is_direct:
        destinations.append(np.full_like(timeout_profit, np.nan))
    index_positions = np.full(temporal.timestamps, -1, dtype=np.int64)
    index_positions[indices] = np.arange(indices.size, dtype=np.int64)
    boundaries = _source_boundaries(temporal.timestamps_ms)
    model.eval()
    with torch.no_grad():
        for run_start, run_end in _contiguous_runs(mask, temporal.timestamps_ms):
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
                    raise RuntimeError("Round 50 prediction indexing is invalid")
                arrays = [
                    _prediction_array(value[..., local_start:local_stop])
                    for value in output
                ]
                for destination, array in zip(destinations, arrays, strict=True):
                    destination[positions] = array
    if not all(np.isfinite(value).all() for value in destinations):
        raise RuntimeError("Round 50 predictions are nonfinite")
    return indices, tuple(destinations)


def _numpy_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - np.max(logits, axis=-1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / np.sum(exponential, axis=-1, keepdims=True)


def _structured_expected_risk_numpy(
    probabilities: np.ndarray,
    timeout_mean: np.ndarray,
    stop_bps: np.ndarray,
    baselines: BarrierTargetBaselines,
    *,
    execution_charge_bps: float,
    reward_risk: float,
) -> np.ndarray:
    stop_probability = np.sum(probabilities[..., :EVENT_MINUTES], axis=-1)
    take_probability = np.sum(
        probabilities[..., EVENT_MINUTES : 2 * EVENT_MINUTES], axis=-1
    )
    timeout_probability = probabilities[..., TIMEOUT_CLASS]
    cost_ratio = execution_charge_bps / stop_bps.astype(np.float64)
    stop_value = (
        -1.0
        - cost_ratio[..., None]
        + baselines.stop_residual_mean_risk_units.reshape(1, len(SYMBOLS), len(SIDES))
    )
    take_value = (
        reward_risk
        - cost_ratio[..., None]
        + baselines.take_residual_mean_risk_units.reshape(1, len(SYMBOLS), len(SIDES))
    )
    return (
        stop_probability * stop_value
        + take_probability * take_value
        + timeout_probability * timeout_mean
    )


def _numpy_candidate_losses(
    candidate_id: str,
    outputs: tuple[np.ndarray, ...],
    barrier: BarrierPayoffDataset,
    event_classes: np.ndarray,
    risk_targets: np.ndarray,
    baselines: BarrierTargetBaselines,
    indices: np.ndarray,
) -> dict[str, float]:
    logits = outputs[0].astype(np.float64)
    labels = event_classes[indices]
    probabilities = _numpy_softmax(logits)
    true_probability = np.take_along_axis(probabilities, labels[..., None], axis=-1)[
        ..., 0
    ]
    event_log_loss = float(-np.mean(np.log(np.clip(true_probability, 1e-12, 1.0))))
    timeout_mask = labels == TIMEOUT_CLASS
    target = risk_targets[indices].astype(np.float64)
    timeout_probability = 1.0 / (1.0 + np.exp(-outputs[1].astype(np.float64)))
    if not np.any(timeout_mask):
        raise ValueError("Round 50 validation has no timeout rows")
    profit_labels = target > 0.0
    clipped = np.clip(timeout_probability[timeout_mask], 1e-12, 1.0 - 1e-12)
    timeout_profit_bce = float(
        -np.mean(
            profit_labels[timeout_mask] * np.log(clipped)
            + (~profit_labels[timeout_mask]) * np.log1p(-clipped)
        )
    )
    timeout_mean_mse = float(
        np.mean(
            (outputs[2].astype(np.float64)[timeout_mask] - target[timeout_mask]) ** 2
        )
    )
    if candidate_id == "direct_barrier_mean_tcn":
        action_risk = outputs[3].astype(np.float64)
        primary_weight = DIRECT_MEAN_MSE_WEIGHT
    elif candidate_id == "competing_risk_barrier_tcn":
        action_risk = _structured_expected_risk_numpy(
            probabilities,
            outputs[2].astype(np.float64),
            barrier.stop_bps[indices],
            baselines,
            execution_charge_bps=(
                barrier.specification.round_trip_execution_charge_bps
            ),
            reward_risk=barrier.specification.take_profit_to_stop_ratio,
        )
        primary_weight = 0.0
    else:
        raise KeyError(candidate_id)
    primary_mse = float(np.mean((action_risk - target) ** 2))
    composite = (
        EVENT_LOG_LOSS_WEIGHT * event_log_loss
        + TIMEOUT_PROFIT_BCE_WEIGHT * timeout_profit_bce
        + TIMEOUT_MEAN_MSE_WEIGHT * timeout_mean_mse
        + primary_weight * primary_mse
    )
    result = {
        "event_log_loss": event_log_loss,
        "timeout_profit_bce": timeout_profit_bce,
        "timeout_mean_mse": timeout_mean_mse,
        "primary_mse": primary_mse,
        "composite": composite,
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("Round 50 validation losses are nonfinite")
    return result


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _event_log_loss_numpy(
    logits: np.ndarray, labels: np.ndarray, temperature: float
) -> float:
    probabilities = _numpy_softmax(logits / float(temperature))
    selected = np.take_along_axis(probabilities, labels[..., None], axis=-1)[..., 0]
    return float(-np.mean(np.log(np.clip(selected, 1e-12, 1.0))))


def fit_event_temperature(
    logits: np.ndarray, labels: np.ndarray
) -> EventTemperatureCalibration:
    if logits.ndim != labels.ndim + 1 or logits.shape[:-1] != labels.shape:
        raise ValueError("Round 50 event calibration shapes are invalid")
    if not np.isfinite(logits).all() or np.any(
        (labels < 0) | (labels >= EVENT_CLASSES)
    ):
        raise ValueError("Round 50 event calibration values are invalid")
    before = _event_log_loss_numpy(logits, labels, 1.0)
    result = minimize_scalar(
        lambda value: _event_log_loss_numpy(logits, labels, float(value)),
        method="bounded",
        bounds=EVENT_TEMPERATURE_BOUNDS,
        options={"xatol": 1e-6, "maxiter": 200},
    )
    if not result.success or not math.isfinite(float(result.x)):
        raise RuntimeError("Round 50 event temperature calibration failed")
    temperature = float(result.x)
    after = _event_log_loss_numpy(logits, labels, temperature)
    if after > before + 1e-12:
        temperature, after = 1.0, before
    return EventTemperatureCalibration(
        temperature=temperature,
        multinomial_log_loss_before=before,
        multinomial_log_loss_after=after,
    )


def _event_summaries(
    probabilities: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    true_probability = np.take_along_axis(probabilities, labels[..., None], axis=-1)[
        ..., 0
    ]
    probability_square_sum = np.sum(probabilities * probabilities, axis=-1)
    groups = np.stack(
        (
            np.sum(probabilities[..., :EVENT_MINUTES], axis=-1),
            probabilities[..., TIMEOUT_CLASS],
            np.sum(probabilities[..., EVENT_MINUTES : 2 * EVENT_MINUTES], axis=-1),
        ),
        axis=-1,
    )
    minutes = np.concatenate(
        (
            np.arange(1, EVENT_MINUTES + 1, dtype=np.float64),
            np.arange(1, EVENT_MINUTES + 1, dtype=np.float64),
            np.asarray([HORIZON_MINUTES], dtype=np.float64),
        )
    )
    expected_minutes = np.sum(probabilities * minutes, axis=-1)
    outputs = (true_probability, probability_square_sum, groups, expected_minutes)
    if not all(np.isfinite(value).all() for value in outputs):
        raise RuntimeError("Round 50 event summaries are nonfinite")
    return tuple(value.astype(np.float32) for value in outputs)


def _save_prediction_artifact(
    path: Path,
    *,
    indices: np.ndarray,
    event_true_probability: np.ndarray,
    event_probability_square_sum: np.ndarray,
    event_group_probability: np.ndarray,
    event_expected_minutes: np.ndarray,
    timeout_profit_probability: np.ndarray,
    timeout_mean_risk_units: np.ndarray,
    action_values_bps: np.ndarray,
    event_calibration: EventTemperatureCalibration,
    timeout_calibration: ProbabilityCalibration,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        global_indices=indices,
        event_true_probability=event_true_probability,
        event_probability_square_sum=event_probability_square_sum,
        event_group_probability=event_group_probability,
        event_expected_minutes=event_expected_minutes,
        timeout_profit_probability=timeout_profit_probability,
        timeout_mean_risk_units=timeout_mean_risk_units,
        action_values_bps=action_values_bps,
        event_temperature=np.asarray(event_calibration.temperature, dtype=np.float64),
        timeout_probability_slope=np.asarray(
            timeout_calibration.slope, dtype=np.float64
        ),
        timeout_probability_intercept=np.asarray(
            timeout_calibration.intercept, dtype=np.float64
        ),
    )


def _train_candidate(
    temporal: MinuteTemporalDataset,
    barrier: BarrierPayoffDataset,
    normalized_features: np.ndarray,
    event_classes: np.ndarray,
    risk_targets: np.ndarray,
    feature_scaler: RobustFeatureScaler,
    target_baselines: BarrierTargetBaselines,
    *,
    candidate_id: str,
    model_dir: Path,
    prediction_dir: Path,
    device: object,
    backend_kind: str,
    backend_device: str,
    progress: ProgressCallback | None,
) -> BarrierCompetingRiskForecastBundle:
    peers: list[nn.Module] = []
    optimizers: list[ExplicitAdamW] = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        peer = _candidate_model(candidate_id, target_baselines).to(device)
        peers.append(peer)
        optimizers.append(
            ExplicitAdamW(
                tuple(peer.parameters()), learning_rate=8e-4, weight_decay=1e-4
            )
        )
    pairs = _training_pairs(temporal, barrier)
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
            "event_log_loss": [],
            "timeout_profit_bce": [],
            "timeout_mean_mse": [],
            "primary_mse": [],
            "composite": [],
        }
        for peer in peers:
            peer.train()
        batches_in_epoch = math.ceil(order.size / BATCH_SIZE)
        heartbeat = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for offset in range(0, order.size, BATCH_SIZE):
                selected = pairs[order[offset : offset + BATCH_SIZE]]
                features, labels, risk, stop, symbols = _batch_arrays(
                    normalized_features,
                    event_classes,
                    risk_targets,
                    barrier,
                    selected,
                )
                values = torch.from_numpy(features).to(device)
                labels_tensor = torch.from_numpy(labels).to(device)
                risk_tensor = torch.from_numpy(risk).to(device)
                stop_tensor = torch.from_numpy(stop).to(device)
                symbol_tensor = torch.from_numpy(symbols).to(device)
                peer_components: dict[str, list[float]] = {
                    key: [] for key in batch_losses
                }
                for seed, peer, optimizer in zip(SEEDS, peers, optimizers, strict=True):
                    optimizer.zero_grad(set_to_none=True)
                    output = tuple(
                        value[..., -SUPERVISED_STEPS:] for value in peer(values)
                    )
                    objective, losses = candidate_objective(
                        candidate_id,
                        output,
                        labels_tensor,
                        risk_tensor,
                        stop_tensor,
                        symbol_tensor,
                        target_baselines,
                        execution_charge_bps=(
                            barrier.specification.round_trip_execution_charge_bps
                        ),
                        reward_risk=(barrier.specification.take_profit_to_stop_ratio),
                    )
                    loss_values = {
                        key: float(value.detach().cpu())
                        for key, value in losses.items()
                    }
                    if not all(math.isfinite(value) for value in loss_values.values()):
                        raise RuntimeError(
                            "Round 50 produced a nonfinite training loss: "
                            f"candidate={candidate_id}, seed={seed}, epoch={epoch}, "
                            f"batch={offset // BATCH_SIZE}, losses={loss_values}"
                        )
                    objective.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        peer.parameters(), 1.0, foreach=False
                    )
                    if not math.isfinite(float(gradient_norm.detach().cpu())):
                        raise RuntimeError("Round 50 rejected a nonfinite gradient")
                    optimizer.step()
                    for key in peer_components:
                        peer_components[key].append(loss_values[key])
                optimizer_updates += 1
                for key, values_for_key in peer_components.items():
                    batch_losses[key].append(float(np.mean(values_for_key)))
                now = time.perf_counter()
                batch_index = offset // BATCH_SIZE + 1
                if progress is not None and (
                    now - heartbeat >= 30.0 or batch_index == batches_in_epoch
                ):
                    progress(
                        "round50_training_batch",
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
                    heartbeat = now
        warning_messages.extend(str(item.message) for item in caught)
        validation_losses: list[dict[str, float]] = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for seed, peer in zip(SEEDS, peers, strict=True):
                if progress is not None:
                    progress(
                        "round50_early_stop_seed",
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
                    temporal,
                    barrier.role_masks["early_stop"],
                    device,
                )
                validation_losses.append(
                    _numpy_candidate_losses(
                        candidate_id,
                        outputs,
                        barrier,
                        event_classes,
                        risk_targets,
                        target_baselines,
                        indices,
                    )
                )
                if progress is not None:
                    progress(
                        "round50_early_stop_seed",
                        {
                            "status": "complete",
                            "candidate_id": candidate_id,
                            "epoch": epoch,
                            "seed": seed,
                        },
                    )
        warning_messages.extend(str(item.message) for item in caught)
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
            progress("round50_epoch", row)
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            break
    if best_states is None or best_losses is None:
        raise RuntimeError("Round 50 training did not produce a best state")
    fallback = _fallback_messages(warning_messages)
    if fallback:
        raise RuntimeError(f"Round 50 training used CPU fallback: {fallback}")
    analysis_mask = np.logical_or.reduce(
        [barrier.role_masks[role] for role in ANALYSIS_ROLES]
    )
    probe_mask = np.zeros(temporal.timestamps, dtype=bool)
    probe_indices = np.flatnonzero(barrier.role_masks["calibration"])[:256]
    probe_mask[probe_indices] = True
    artifacts: list[BarrierCompetingRiskArtifact] = []
    seed_true_probability: list[np.ndarray] = []
    seed_probability_square_sum: list[np.ndarray] = []
    seed_group_probability: list[np.ndarray] = []
    seed_expected_minutes: list[np.ndarray] = []
    seed_timeout_probability: list[np.ndarray] = []
    seed_timeout_mean: list[np.ndarray] = []
    seed_action_values: list[np.ndarray] = []
    global_indices: np.ndarray | None = None
    for peer, seed, best_state in zip(peers, SEEDS, best_states, strict=True):
        if progress is not None:
            progress(
                "round50_artifact_seed",
                {"status": "started", "candidate_id": candidate_id, "seed": seed},
            )
        peer.load_state_dict(best_state)
        model_path = model_dir / f"round50_{candidate_id}_seed_{seed}.pt"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, model_path)
        loaded_state = torch.load(model_path, map_location="cpu", weights_only=True)
        if set(loaded_state) != set(best_state) or any(
            not torch.equal(best_state[name], loaded_state[name]) for name in best_state
        ):
            raise RuntimeError("Round 50 checkpoint tensors changed during reload")
        reference_cpu = _candidate_model(candidate_id, target_baselines)
        reference_cpu.load_state_dict(best_state)
        reloaded_cpu = _candidate_model(candidate_id, target_baselines)
        reloaded_cpu.load_state_dict(loaded_state)
        reference_probe = _predict_mask(
            reference_cpu,
            normalized_features,
            temporal,
            probe_mask,
            torch.device("cpu"),
        )
        reloaded_probe = _predict_mask(
            reloaded_cpu,
            normalized_features,
            temporal,
            probe_mask,
            torch.device("cpu"),
        )
        if not np.array_equal(reference_probe[0], reloaded_probe[0]):
            raise RuntimeError("Round 50 reload probe indexing failed")
        reload_errors = tuple(
            float(np.max(np.abs(reference - observed)))
            for reference, observed in zip(
                reference_probe[1], reloaded_probe[1], strict=True
            )
        )
        if not all(math.isfinite(value) and value <= 1e-6 for value in reload_errors):
            raise RuntimeError(f"Round 50 reload errors are {reload_errors}")
        runtime_reference = _predict_mask(
            peer, normalized_features, temporal, probe_mask, device
        )
        reloaded = reloaded_cpu.to(device)
        runtime_reloaded = _predict_mask(
            reloaded, normalized_features, temporal, probe_mask, device
        )
        runtime_repeat_errors = tuple(
            float(np.max(np.abs(reference - observed)))
            for reference, observed in zip(
                runtime_reference[1], runtime_reloaded[1], strict=True
            )
        )
        if not all(
            math.isfinite(value) and value <= 1e-4 for value in runtime_repeat_errors
        ):
            raise RuntimeError(
                f"Round 50 runtime repeat errors are {runtime_repeat_errors}"
            )
        indices, outputs = _predict_mask(
            reloaded, normalized_features, temporal, analysis_mask, device
        )
        if global_indices is None:
            global_indices = indices
        elif not np.array_equal(global_indices, indices):
            raise RuntimeError("Round 50 final indices differ by seed")
        calibration_local = barrier.role_masks["calibration"][indices]
        labels = event_classes[indices]
        event_calibration = fit_event_temperature(
            outputs[0][calibration_local], labels[calibration_local]
        )
        event_probabilities = _numpy_softmax(
            outputs[0].astype(np.float64) / event_calibration.temperature
        )
        timeout_calibration_mask = calibration_local[..., None, None] & (
            labels == TIMEOUT_CLASS
        )
        timeout_calibration = fit_probability_calibration(
            outputs[1][timeout_calibration_mask],
            (risk_targets[indices][timeout_calibration_mask] > 0.0),
        )
        timeout_profit_probability = apply_probability_calibration(
            outputs[1], timeout_calibration
        )
        summaries = _event_summaries(event_probabilities, labels)
        if candidate_id == "direct_barrier_mean_tcn":
            action_risk = outputs[3].astype(np.float64)
        else:
            action_risk = _structured_expected_risk_numpy(
                event_probabilities,
                outputs[2].astype(np.float64),
                barrier.stop_bps[indices],
                target_baselines,
                execution_charge_bps=(
                    barrier.specification.round_trip_execution_charge_bps
                ),
                reward_risk=barrier.specification.take_profit_to_stop_ratio,
            )
        action_values = (action_risk * barrier.stop_bps[indices][..., None]).astype(
            np.float32
        )
        prediction_path = prediction_dir / f"round50_{candidate_id}_seed_{seed}.npz"
        _save_prediction_artifact(
            prediction_path,
            indices=indices,
            event_true_probability=summaries[0],
            event_probability_square_sum=summaries[1],
            event_group_probability=summaries[2],
            event_expected_minutes=summaries[3],
            timeout_profit_probability=timeout_profit_probability,
            timeout_mean_risk_units=outputs[2],
            action_values_bps=action_values,
            event_calibration=event_calibration,
            timeout_calibration=timeout_calibration,
        )
        mapped_errors = (
            reload_errors
            if candidate_id == "direct_barrier_mean_tcn"
            else reload_errors + (0.0,)
        )
        mapped_runtime_errors = (
            runtime_repeat_errors
            if candidate_id == "direct_barrier_mean_tcn"
            else runtime_repeat_errors + (0.0,)
        )
        artifact = BarrierCompetingRiskArtifact(
            candidate_id=candidate_id,
            seed=seed,
            epochs=epochs_run,
            best_epoch=best_epoch,
            best_early_stop_composite=float(best_losses["composite"]),
            best_early_stop_event_log_loss=float(best_losses["event_log_loss"]),
            best_early_stop_timeout_profit_bce=float(best_losses["timeout_profit_bce"]),
            best_early_stop_timeout_mean_mse=float(best_losses["timeout_mean_mse"]),
            best_early_stop_primary_mse=float(best_losses["primary_mse"]),
            optimizer_updates=optimizer_updates,
            parameter_count=sum(value.numel() for value in reloaded.parameters()),
            event_temperature=event_calibration.temperature,
            calibration_event_log_loss_before=(
                event_calibration.multinomial_log_loss_before
            ),
            calibration_event_log_loss_after=(
                event_calibration.multinomial_log_loss_after
            ),
            timeout_probability_slope=timeout_calibration.slope,
            timeout_probability_intercept=timeout_calibration.intercept,
            calibration_timeout_log_loss_before=(
                timeout_calibration.binary_log_loss_before
            ),
            calibration_timeout_log_loss_after=(
                timeout_calibration.binary_log_loss_after
            ),
            backend_kind=backend_kind,
            backend_device=backend_device,
            model_path=str(model_path),
            model_bytes=model_path.stat().st_size,
            model_sha256=_file_sha256(model_path),
            prediction_path=str(prediction_path),
            prediction_bytes=prediction_path.stat().st_size,
            prediction_sha256=_file_sha256(prediction_path),
            reload_max_abs_event_logit_error=mapped_errors[0],
            reload_max_abs_timeout_profit_logit_error=mapped_errors[1],
            reload_max_abs_timeout_mean_error=mapped_errors[2],
            reload_max_abs_direct_mean_error=mapped_errors[3],
            runtime_repeat_max_abs_event_logit_error=mapped_runtime_errors[0],
            runtime_repeat_max_abs_timeout_profit_logit_error=(
                mapped_runtime_errors[1]
            ),
            runtime_repeat_max_abs_timeout_mean_error=mapped_runtime_errors[2],
            runtime_repeat_max_abs_direct_mean_error=mapped_runtime_errors[3],
            warning_count=len(warning_messages),
        )
        artifacts.append(artifact)
        seed_true_probability.append(summaries[0])
        seed_probability_square_sum.append(summaries[1])
        seed_group_probability.append(summaries[2])
        seed_expected_minutes.append(summaries[3])
        seed_timeout_probability.append(timeout_profit_probability)
        seed_timeout_mean.append(outputs[2].astype(np.float32))
        seed_action_values.append(action_values)
        if progress is not None:
            progress(
                "round50_artifact_seed",
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
        raise RuntimeError(f"Round 50 finalization used CPU fallback: {fallback}")
    if global_indices is None:
        raise RuntimeError("Round 50 finalization produced no indices")
    return BarrierCompetingRiskForecastBundle(
        candidate_id=candidate_id,
        global_indices=global_indices,
        seed_event_true_probabilities=np.stack(seed_true_probability),
        seed_event_probability_square_sums=np.stack(seed_probability_square_sum),
        seed_event_group_probabilities=np.stack(seed_group_probability),
        seed_event_expected_minutes=np.stack(seed_expected_minutes),
        seed_timeout_profit_probabilities=np.stack(seed_timeout_probability),
        seed_timeout_mean_risk_units=np.stack(seed_timeout_mean),
        seed_action_values_bps=np.stack(seed_action_values),
        artifacts=tuple(artifacts),
        feature_scaler=feature_scaler,
        target_baselines=target_baselines,
        backend_kind=backend_kind,
        backend_device=backend_device,
        training_history=tuple(history),
    )


def train_barrier_competing_risk_candidates(
    temporal: MinuteTemporalDataset,
    barrier: BarrierPayoffDataset,
    *,
    model_dir: Path,
    prediction_dir: Path,
    compute_backend: str = "auto",
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, BarrierCompetingRiskForecastBundle], dict[str, object]]:
    device, preflight = barrier_competing_risk_preflight(compute_backend)
    if progress is not None:
        progress("round50_preflight", {"status": "complete", **preflight})
    feature_scaler = fit_barrier_feature_scaler(temporal, barrier)
    normalized_features = feature_scaler.transform(temporal.features)
    event_classes = barrier_event_classes(barrier)
    risk_targets = barrier_risk_targets(barrier)
    target_baselines = fit_barrier_target_baselines(
        barrier, event_classes, risk_targets
    )
    if progress is not None:
        progress(
            "round50_scaling",
            {
                "status": "complete",
                "feature_bytes": int(normalized_features.nbytes),
                "event_bytes": int(event_classes.nbytes),
                "target_bytes": int(risk_targets.nbytes),
                "training_windows": int(_training_pairs(temporal, barrier).shape[0]),
            },
        )
    bundles: dict[str, BarrierCompetingRiskForecastBundle] = {}
    for candidate_id in CANDIDATES:
        bundles[candidate_id] = _train_candidate(
            temporal,
            barrier,
            normalized_features,
            event_classes,
            risk_targets,
            feature_scaler,
            target_baselines,
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
    "barrier_competing_risk_preflight",
    "BarrierCompetingRiskArtifact",
    "BarrierCompetingRiskForecastBundle",
    "BarrierTargetBaselines",
    "CANDIDATES",
    "EVENT_CLASSES",
    "EVENT_MINUTES",
    "HORIZON_MINUTES",
    "RECEPTIVE_FIELD_STEPS",
    "SEEDS",
    "SIDES",
    "TIMEOUT_CLASS",
    "barrier_event_classes",
    "barrier_risk_targets",
    "candidate_objective",
    "cpu_barrier_competing_risk_preflight",
    "directml_barrier_competing_risk_preflight",
    "fit_barrier_feature_scaler",
    "fit_barrier_target_baselines",
    "fit_event_temperature",
    "train_barrier_competing_risk_candidates",
]
