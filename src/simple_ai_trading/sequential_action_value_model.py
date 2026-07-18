"""Distributional Bellman model for stateful cost-aware trading research."""

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

from .compute import require_backend, resolve_backend, torch_device_for_backend
from .cross_asset_cost_data import SYMBOLS
from .distributional_tcn_model import (
    CausalResidualBlock,
    DistributionalDataset,
    ExplicitAdamW,
    FeatureScaler,
    fit_feature_scaler,
    role_mask,
)


ROUND = 54
ACTIONS = (-1, 0, 1)
QUANTILE_LEVELS = (0.10, 0.30, 0.50, 0.70, 0.90)
POLICY_IDS = (
    "median_q_all_seed_consensus",
    "lower_tail_q_all_seed_consensus",
)
ProgressCallback = Callable[[str, Mapping[str, object]], None]


@dataclass(frozen=True)
class SequentialQSpec:
    hidden_channels: int = 64
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    kernel_size: int = 3
    window_hours: int = 256
    window_stride_hours: int = 24
    batch_size: int = 32
    discount_factor: float = 0.95
    base_one_way_cost_bps: float = 6.0
    stress_one_way_cost_bps: float = 8.0
    maximum_epochs: int = 30
    early_stopping_patience: int = 6
    minimum_improvement: float = 1e-5
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    target_polyak_rate: float = 0.01
    seeds: tuple[int, ...] = (5401, 5402, 5403)
    bootstrap_samples: int = 2_000
    bootstrap_block_hours: int = 168
    bootstrap_lower_quantile: float = 0.025

    @property
    def receptive_field(self) -> int:
        return 1 + (self.kernel_size - 1) * sum(self.dilations)

    @property
    def supervised_start(self) -> int:
        return self.receptive_field - 1

    def validate(self) -> None:
        if self.hidden_channels <= 0 or not self.dilations:
            raise ValueError("Round 54 model width and dilations must be positive")
        if self.kernel_size != 3:
            raise ValueError("Round 54 frozen kernel size differs from three")
        if self.window_hours <= self.receptive_field + 1:
            raise ValueError("Round 54 window does not contain supervised transitions")
        if self.window_stride_hours <= 0 or self.batch_size <= 0:
            raise ValueError("Round 54 stride and batch size must be positive")
        if not 0.0 < self.discount_factor < 1.0:
            raise ValueError("Round 54 discount factor must lie strictly inside (0, 1)")
        if self.base_one_way_cost_bps <= 0.0:
            raise ValueError("Round 54 base cost must be positive")
        if self.stress_one_way_cost_bps <= self.base_one_way_cost_bps:
            raise ValueError("Round 54 stress cost must exceed base cost")
        if len(self.seeds) != 3 or len(set(self.seeds)) != 3:
            raise ValueError("Round 54 requires three distinct seeds")


DEFAULT_SPEC = SequentialQSpec()


@dataclass(frozen=True)
class SequentialQArtifact:
    seed: int
    epochs: int
    best_epoch: int
    optimizer_updates: int
    best_early_stop_td_loss: float
    zero_q_early_stop_td_loss: float
    early_stop_td_skill: float
    reward_scale_bps: float
    parameter_count: int
    backend_kind: str
    backend_device: str
    path: str
    bytes: int
    sha256: str
    reload_max_abs_q_error: float
    warning_count: int

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SequentialQEnsemble:
    seed_q_bps: np.ndarray
    artifacts: tuple[SequentialQArtifact, ...]
    feature_scaler: FeatureScaler
    reward_scale_bps: float
    backend_kind: str
    backend_device: str
    preflight: Mapping[str, object]
    training_history: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class SequentialReplay:
    policy_id: str
    role: str
    scenario: str
    one_way_cost_bps: float
    timestamps_ms: np.ndarray
    actions: np.ndarray
    symbol_net_bps: np.ndarray
    portfolio_return_bps: np.ndarray
    metrics: Mapping[str, object]


class DuelingDistributionalQTCN(nn.Module):
    """Causal dueling network over prior position, action, and value quantile."""

    def __init__(
        self,
        *,
        input_channels: int,
        spec: SequentialQSpec = DEFAULT_SPEC,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        spec.validate()
        self.input_channels = input_channels
        self.spec = spec
        self.projection = nn.Conv1d(
            input_channels,
            spec.hidden_channels,
            kernel_size=1,
        )
        self.blocks = nn.ModuleList(
            CausalResidualBlock(spec.hidden_channels, dilation, dropout)
            for dilation in spec.dilations
        )
        output_quantiles = len(ACTIONS) * len(QUANTILE_LEVELS)
        self.value_head = nn.Conv1d(
            spec.hidden_channels,
            output_quantiles,
            kernel_size=1,
        )
        self.advantage_head = nn.Conv1d(
            spec.hidden_channels,
            len(ACTIONS) * output_quantiles,
            kernel_size=1,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != self.input_channels:
            raise ValueError("Round 54 model input dimensions are invalid")
        encoded = F.gelu(self.projection(values))
        for block in self.blocks:
            encoded = block(encoded)
        value = self.value_head(encoded).reshape(
            values.shape[0],
            len(ACTIONS),
            1,
            len(QUANTILE_LEVELS),
            values.shape[-1],
        )
        advantage = self.advantage_head(encoded).reshape(
            values.shape[0],
            len(ACTIONS),
            len(ACTIONS),
            len(QUANTILE_LEVELS),
            values.shape[-1],
        )
        raw = value + advantage - advantage.mean(dim=2, keepdim=True)
        median = raw[:, :, :, 0, :]
        q30 = median - F.softplus(raw[:, :, :, 1, :])
        q10 = q30 - F.softplus(raw[:, :, :, 2, :])
        q70 = median + F.softplus(raw[:, :, :, 3, :])
        q90 = q70 + F.softplus(raw[:, :, :, 4, :])
        return torch.stack((q10, q30, median, q70, q90), dim=3)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fit_reward_scale_bps(
    dataset: DistributionalDataset,
    training_mask: np.ndarray,
) -> float:
    selected = dataset.hourly_return_bps[training_mask].astype(np.float64).ravel()
    if selected.size == 0 or not np.isfinite(selected).all():
        raise ValueError("Round 54 reward scale input is empty or nonfinite")
    center = float(np.median(selected))
    scale = max(1.4826 * float(np.median(np.abs(selected - center))), 1.0)
    if not math.isfinite(scale):
        raise ValueError("Round 54 reward scale is nonfinite")
    return scale


def transition_reward_bps(
    hourly_return_bps: np.ndarray,
    one_way_cost_bps: float,
) -> np.ndarray:
    """Return the full-information reward for every prior/action pair."""

    values = np.asarray(hourly_return_bps, dtype=np.float64)
    if values.ndim < 1 or not np.isfinite(values).all():
        raise ValueError("Round 54 transition returns are invalid")
    if not math.isfinite(one_way_cost_bps) or one_way_cost_bps < 0.0:
        raise ValueError("Round 54 transition cost is invalid")
    previous = np.asarray(ACTIONS, dtype=np.float64).reshape(
        *((1,) * values.ndim), len(ACTIONS), 1
    )
    action = np.asarray(ACTIONS, dtype=np.float64).reshape(
        *((1,) * values.ndim), 1, len(ACTIONS)
    )
    expanded = values.reshape(*values.shape, 1, 1)
    return action * expanded - one_way_cost_bps * np.abs(action - previous)


def quantile_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    kappa: float = 1.0,
) -> torch.Tensor:
    """Pairwise quantile-regression Huber loss used by QR-DQN."""

    if predictions.shape != targets.shape or predictions.ndim != 5:
        raise ValueError("Round 54 quantile tensors have invalid dimensions")
    if predictions.shape[3] != len(QUANTILE_LEVELS) or kappa <= 0.0:
        raise ValueError("Round 54 quantile loss contract is invalid")
    levels = torch.tensor(
        QUANTILE_LEVELS,
        dtype=predictions.dtype,
        device=predictions.device,
    ).reshape(1, 1, 1, len(QUANTILE_LEVELS), 1)
    levels = levels.expand_as(predictions).contiguous()
    losses: list[torch.Tensor] = []
    for target_index in range(len(QUANTILE_LEVELS)):
        target_values = targets[:, :, :, target_index, :].unsqueeze(3)
        target_values = target_values.expand_as(predictions).contiguous()
        delta = target_values - predictions
        absolute = torch.abs(delta)
        huber = torch.where(
            absolute <= kappa,
            0.5 * delta * delta,
            kappa * (absolute - 0.5 * kappa),
        )
        weight = torch.abs(
            levels - (delta.detach() < 0.0).to(predictions.dtype)
        )
        losses.append((weight * huber / kappa).mean())
    return torch.stack(losses).mean()


def bellman_quantile_targets(
    normalized_returns: torch.Tensor,
    online_next_q: torch.Tensor,
    target_next_q: torch.Tensor,
    *,
    normalized_one_way_cost: float,
    discount_factor: float,
) -> torch.Tensor:
    """Build full-information double-Q targets for every transition action."""

    expected = (
        normalized_returns.shape[0],
        len(ACTIONS),
        len(ACTIONS),
        len(QUANTILE_LEVELS),
        normalized_returns.shape[1],
    )
    if online_next_q.shape != expected or target_next_q.shape != expected:
        raise ValueError("Round 54 next-Q tensors have invalid dimensions")
    if not 0.0 < discount_factor < 1.0:
        raise ValueError("Round 54 Bellman discount is invalid")
    positions = torch.tensor(
        ACTIONS,
        dtype=normalized_returns.dtype,
        device=normalized_returns.device,
    )
    previous = positions.reshape(1, len(ACTIONS), 1, 1)
    action = positions.reshape(1, 1, len(ACTIONS), 1)
    rewards = (
        action * normalized_returns[:, None, None, :]
        - normalized_one_way_cost * torch.abs(action - previous)
    )
    next_action = online_next_q.mean(dim=3).argmax(dim=2)
    gather_index = next_action[:, :, None, None, :].expand(
        -1,
        -1,
        1,
        len(QUANTILE_LEVELS),
        -1,
    )
    selected_next = target_next_q.gather(2, gather_index).squeeze(2)
    return rewards[:, :, :, None, :] + discount_factor * selected_next[:, None]


def _fallback_messages(messages: Sequence[str]) -> list[str]:
    return [
        item
        for item in messages
        if "not currently supported on the DML backend" in item
        or "fall back to run on the CPU" in item
    ]


def _run_preflight(
    device: object,
    metadata: Mapping[str, object],
    spec: SequentialQSpec,
) -> dict[str, object]:
    generator = np.random.default_rng(spec.seeds[0])
    values = torch.from_numpy(
        generator.normal(size=(4, 71, spec.window_hours)).astype(np.float32)
    ).to(device)
    returns = torch.from_numpy(
        generator.normal(size=(4, spec.window_hours - 1)).astype(np.float32)
    ).to(device)
    torch.manual_seed(spec.seeds[0])
    model = DuelingDistributionalQTCN(input_channels=71, spec=spec).to(device)
    target = DuelingDistributionalQTCN(input_channels=71, spec=spec).to(device)
    target.load_state_dict(model.state_dict())
    optimizer = ExplicitAdamW(
        tuple(model.parameters()),
        learning_rate=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    before = model.advantage_head.weight.detach().cpu().clone()
    messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        optimizer.zero_grad(set_to_none=True)
        output = model(values)
        with torch.no_grad():
            target_output = target(values)
            targets = bellman_quantile_targets(
                returns,
                output[..., 1:].detach(),
                target_output[..., 1:],
                normalized_one_way_cost=0.1,
                discount_factor=spec.discount_factor,
            )
        loss = quantile_huber_loss(output[..., :-1], targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
        optimizer.step()
    messages.extend(str(item.message) for item in caught)
    fallback = _fallback_messages(messages)
    if fallback:
        raise RuntimeError(f"Round 54 compute preflight used CPU fallback: {fallback}")
    change = float(
        torch.max(
            torch.abs(model.advantage_head.weight.detach().cpu() - before)
        )
    )
    loss_value = float(loss.detach().cpu())
    if not math.isfinite(loss_value) or not math.isfinite(change) or change <= 0.0:
        raise RuntimeError("Round 54 compute preflight did not produce a valid update")
    return {
        **metadata,
        "loss": loss_value,
        "head_parameter_max_abs_change": change,
        "warning_count": len(messages),
        "cpu_fallback_warning_count": 0,
    }


def sequential_q_preflight(
    compute_backend: str = "auto",
    spec: SequentialQSpec = DEFAULT_SPEC,
) -> tuple[object, dict[str, object]]:
    spec.validate()
    backend = require_backend(resolve_backend(compute_backend))
    device = torch_device_for_backend(backend)
    return device, _run_preflight(
        device,
        {
            "backend_kind": backend.kind,
            "backend_device": str(device),
            "torch_version": str(torch.__version__),
        },
        spec,
    )


def _contiguous_window_pairs(
    mask: np.ndarray,
    *,
    window_hours: int,
    stride_hours: int,
) -> np.ndarray:
    indexes = np.flatnonzero(mask)
    if indexes.size < window_hours:
        raise ValueError("Round 54 role is too short for a training window")
    breakpoints = np.flatnonzero(np.diff(indexes) != 1) + 1
    runs = np.split(indexes, breakpoints)
    starts: list[int] = []
    for run in runs:
        if run.size < window_hours:
            continue
        last_start = int(run[-1]) - window_hours + 1
        starts.extend(range(int(run[0]), last_start + 1, stride_hours))
    if not starts:
        raise ValueError("Round 54 role produced no contiguous windows")
    return np.asarray(
        [(symbol_index, start) for start in starts for symbol_index in range(3)],
        dtype=np.int64,
    )


def _batch_arrays(
    normalized_features: np.ndarray,
    normalized_returns: np.ndarray,
    pairs: np.ndarray,
    window_hours: int,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.stack(
        [
            normalized_features[
                start : start + window_hours,
                symbol_index,
            ].T
            for symbol_index, start in pairs
        ]
    ).astype(np.float32, copy=False)
    returns = np.stack(
        [
            normalized_returns[
                start : start + window_hours,
                symbol_index,
            ]
            for symbol_index, start in pairs
        ]
    ).astype(np.float32, copy=False)
    return np.ascontiguousarray(features), np.ascontiguousarray(returns)


@torch.no_grad()
def _polyak_update(
    target: nn.Module,
    source: nn.Module,
    rate: float,
) -> None:
    for target_value, source_value in zip(
        target.parameters(), source.parameters(), strict=True
    ):
        target_value.mul_(1.0 - rate).add_(source_value, alpha=rate)


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


@torch.no_grad()
def _role_td_loss(
    model: DuelingDistributionalQTCN,
    target: DuelingDistributionalQTCN,
    normalized_features: np.ndarray,
    normalized_returns: np.ndarray,
    pairs: np.ndarray,
    *,
    reward_scale_bps: float,
    device: object,
    spec: SequentialQSpec,
) -> tuple[float, float]:
    model.eval()
    target.eval()
    model_losses: list[float] = []
    baseline_losses: list[float] = []
    normalized_cost = spec.base_one_way_cost_bps / reward_scale_bps
    for offset in range(0, pairs.shape[0], spec.batch_size):
        batch_pairs = pairs[offset : offset + spec.batch_size]
        feature_values, return_values = _batch_arrays(
            normalized_features,
            normalized_returns,
            batch_pairs,
            spec.window_hours,
        )
        values = torch.from_numpy(feature_values).to(device)
        returns = torch.from_numpy(return_values).to(device)
        output = model(values)
        target_output = target(values)
        current = output[..., spec.supervised_start : -1]
        next_online = output[..., spec.supervised_start + 1 :]
        next_target = target_output[..., spec.supervised_start + 1 :]
        selected_returns = returns[:, spec.supervised_start : -1]
        targets = bellman_quantile_targets(
            selected_returns,
            next_online,
            next_target,
            normalized_one_way_cost=normalized_cost,
            discount_factor=spec.discount_factor,
        )
        model_losses.append(float(quantile_huber_loss(current, targets).cpu()))
        zero = torch.zeros_like(current)
        baseline_losses.append(float(quantile_huber_loss(zero, targets).cpu()))
    return float(np.mean(model_losses)), float(np.mean(baseline_losses))


@torch.no_grad()
def _predict_dataset(
    model: DuelingDistributionalQTCN,
    normalized_features: np.ndarray,
    *,
    device: object,
    spec: SequentialQSpec,
    chunk_hours: int = 4_096,
) -> np.ndarray:
    model.eval()
    timestamps, symbols, _ = normalized_features.shape
    output = np.empty(
        (
            timestamps,
            symbols,
            len(ACTIONS),
            len(ACTIONS),
            len(QUANTILE_LEVELS),
        ),
        dtype=np.float32,
    )
    context_hours = spec.receptive_field - 1
    for symbol_index in range(symbols):
        for start in range(0, timestamps, chunk_hours):
            end = min(start + chunk_hours, timestamps)
            context_start = max(0, start - context_hours)
            feature_values = np.ascontiguousarray(
                normalized_features[context_start:end, symbol_index].T[None]
            )
            values = torch.from_numpy(feature_values).to(device)
            prediction = model(values)[0].detach().cpu().numpy()
            prediction = np.transpose(prediction, (3, 0, 1, 2))
            output[start:end, symbol_index] = prediction[start - context_start :]
    if not np.isfinite(output).all() or np.any(np.diff(output, axis=-1) < -1e-6):
        raise RuntimeError("Round 54 predictions are nonfinite or quantile-crossed")
    return output


def _train_seed(
    dataset: DistributionalDataset,
    normalized_features: np.ndarray,
    normalized_returns: np.ndarray,
    feature_scaler: FeatureScaler,
    reward_scale_bps: float,
    *,
    seed: int,
    model_dir: Path,
    device: object,
    backend_kind: str,
    backend_device: str,
    spec: SequentialQSpec,
    progress: ProgressCallback | None,
) -> tuple[SequentialQArtifact, np.ndarray, tuple[Mapping[str, object], ...]]:
    del feature_scaler
    torch.manual_seed(seed)
    model = DuelingDistributionalQTCN(
        input_channels=normalized_features.shape[-1], spec=spec
    ).to(device)
    target = DuelingDistributionalQTCN(
        input_channels=normalized_features.shape[-1], spec=spec
    ).to(device)
    target.load_state_dict(model.state_dict())
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    optimizer = ExplicitAdamW(
        tuple(model.parameters()),
        learning_rate=spec.learning_rate,
        weight_decay=spec.weight_decay,
    )
    training_pairs = _contiguous_window_pairs(
        role_mask(dataset, "training"),
        window_hours=spec.window_hours,
        stride_hours=spec.window_stride_hours,
    )
    validation_pairs = _contiguous_window_pairs(
        role_mask(dataset, "early_stop"),
        window_hours=spec.window_hours,
        stride_hours=spec.window_stride_hours,
    )
    generator = np.random.default_rng(seed)
    normalized_cost = spec.base_one_way_cost_bps / reward_scale_bps
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_target_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    optimizer_updates = 0
    epochs_run = 0
    history: list[Mapping[str, object]] = []
    warning_messages: list[str] = []
    for epoch in range(1, spec.maximum_epochs + 1):
        epochs_run = epoch
        order = generator.permutation(training_pairs.shape[0])
        model.train()
        batch_losses: list[float] = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for offset in range(0, order.size, spec.batch_size):
                selected = training_pairs[order[offset : offset + spec.batch_size]]
                feature_values, return_values = _batch_arrays(
                    normalized_features,
                    normalized_returns,
                    selected,
                    spec.window_hours,
                )
                values = torch.from_numpy(feature_values).to(device)
                returns = torch.from_numpy(return_values).to(device)
                optimizer.zero_grad(set_to_none=True)
                output = model(values)
                with torch.no_grad():
                    target_output = target(values)
                    targets = bellman_quantile_targets(
                        returns[:, spec.supervised_start : -1],
                        output[..., spec.supervised_start + 1 :].detach(),
                        target_output[..., spec.supervised_start + 1 :],
                        normalized_one_way_cost=normalized_cost,
                        discount_factor=spec.discount_factor,
                    )
                current = output[..., spec.supervised_start : -1]
                loss = quantile_huber_loss(current, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0, foreach=False
                )
                optimizer.step()
                _polyak_update(target, model, spec.target_polyak_rate)
                optimizer_updates += 1
                batch_losses.append(float(loss.detach().cpu()))
        warning_messages.extend(str(item.message) for item in caught)
        validation_loss, zero_loss = _role_td_loss(
            model,
            target,
            normalized_features,
            normalized_returns,
            validation_pairs,
            reward_scale_bps=reward_scale_bps,
            device=device,
            spec=spec,
        )
        improved = validation_loss < best_loss - spec.minimum_improvement
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = _clone_state(model)
            best_target_state = _clone_state(target)
            stale_epochs = 0
        else:
            stale_epochs += 1
        row = {
            "seed": seed,
            "epoch": epoch,
            "optimizer_updates": optimizer_updates,
            "training_td_loss": float(np.mean(batch_losses)),
            "early_stop_td_loss": validation_loss,
            "zero_q_early_stop_td_loss": zero_loss,
            "early_stop_td_skill": 1.0 - validation_loss / zero_loss,
            "best_early_stop_td_loss": best_loss,
            "best_epoch": best_epoch,
            "stale_epochs": stale_epochs,
        }
        if not all(
            math.isfinite(float(value))
            for value in row.values()
            if isinstance(value, (float, int))
        ):
            raise RuntimeError("Round 54 training produced nonfinite diagnostics")
        history.append(row)
        if progress is not None:
            progress("round54_epoch", row)
        if stale_epochs >= spec.early_stopping_patience:
            break
    fallback = _fallback_messages(warning_messages)
    if fallback:
        raise RuntimeError(f"Round 54 training used CPU fallback: {fallback}")
    if best_state is None or best_target_state is None:
        raise RuntimeError("Round 54 training did not retain a best state")
    model.load_state_dict(best_state)
    target.load_state_dict(best_target_state)
    final_loss, final_zero_loss = _role_td_loss(
        model,
        target,
        normalized_features,
        normalized_returns,
        validation_pairs,
        reward_scale_bps=reward_scale_bps,
        device=device,
        spec=spec,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = model_dir / f"round54_sequential_q_seed_{seed}.pt"
    torch.save(best_state, artifact_path)
    predictions = _predict_dataset(
        model,
        normalized_features,
        device=device,
        spec=spec,
    )
    probe_features = normalized_features[: spec.window_hours * 2]
    original_probe = _predict_dataset(
        model,
        probe_features,
        device=device,
        spec=spec,
        chunk_hours=spec.window_hours * 2,
    )
    reloaded = DuelingDistributionalQTCN(
        input_channels=normalized_features.shape[-1], spec=spec
    ).to(device)
    reloaded.load_state_dict(
        torch.load(artifact_path, map_location="cpu", weights_only=True)
    )
    reloaded_predictions = _predict_dataset(
        reloaded,
        probe_features,
        device=device,
        spec=spec,
        chunk_hours=spec.window_hours * 2,
    )
    reload_error = float(
        np.max(
            np.abs(
                reloaded_predictions
                - original_probe
            )
        )
    )
    artifact = SequentialQArtifact(
        seed=seed,
        epochs=epochs_run,
        best_epoch=best_epoch,
        optimizer_updates=optimizer_updates,
        best_early_stop_td_loss=final_loss,
        zero_q_early_stop_td_loss=final_zero_loss,
        early_stop_td_skill=1.0 - final_loss / final_zero_loss,
        reward_scale_bps=reward_scale_bps,
        parameter_count=sum(value.numel() for value in model.parameters()),
        backend_kind=backend_kind,
        backend_device=backend_device,
        path=str(artifact_path),
        bytes=artifact_path.stat().st_size,
        sha256=_file_sha256(artifact_path),
        reload_max_abs_q_error=reload_error,
        warning_count=len(warning_messages),
    )
    return artifact, predictions * reward_scale_bps, tuple(history)


def train_sequential_q_ensemble(
    dataset: DistributionalDataset,
    *,
    model_dir: Path,
    compute_backend: str = "auto",
    spec: SequentialQSpec = DEFAULT_SPEC,
    progress: ProgressCallback | None = None,
) -> SequentialQEnsemble:
    """Train three independently initialized distributional Bellman models."""

    spec.validate()
    training_mask = role_mask(dataset, "training")
    feature_scaler = fit_feature_scaler(dataset, training_mask)
    reward_scale_bps = fit_reward_scale_bps(dataset, training_mask)
    normalized_features = feature_scaler.transform(dataset.features)
    normalized_returns = (
        dataset.hourly_return_bps.astype(np.float64) / reward_scale_bps
    ).astype(np.float32)
    device, preflight = sequential_q_preflight(compute_backend, spec)
    artifacts: list[SequentialQArtifact] = []
    predictions: list[np.ndarray] = []
    history: list[Mapping[str, object]] = []
    for seed in spec.seeds:
        if progress is not None:
            progress("round54_seed", {"seed": seed, "status": "started"})
        artifact, seed_predictions, seed_history = _train_seed(
            dataset,
            normalized_features,
            normalized_returns,
            feature_scaler,
            reward_scale_bps,
            seed=seed,
            model_dir=model_dir,
            device=device,
            backend_kind=str(preflight["backend_kind"]),
            backend_device=str(preflight["backend_device"]),
            spec=spec,
            progress=progress,
        )
        artifacts.append(artifact)
        predictions.append(seed_predictions)
        history.extend(seed_history)
        if progress is not None:
            progress(
                "round54_seed",
                {
                    "seed": seed,
                    "status": "complete",
                    "best_epoch": artifact.best_epoch,
                    "early_stop_td_skill": artifact.early_stop_td_skill,
                },
            )
    stacked = np.stack(predictions).astype(np.float32)
    return SequentialQEnsemble(
        seed_q_bps=stacked,
        artifacts=tuple(artifacts),
        feature_scaler=feature_scaler,
        reward_scale_bps=reward_scale_bps,
        backend_kind=str(preflight["backend_kind"]),
        backend_device=str(preflight["backend_device"]),
        preflight=preflight,
        training_history=tuple(history),
    )


def policy_score(seed_q_bps: np.ndarray, policy_id: str) -> np.ndarray:
    values = np.asarray(seed_q_bps, dtype=np.float64)
    if values.ndim != 6 or values.shape[-3:] != (
        len(ACTIONS),
        len(ACTIONS),
        len(QUANTILE_LEVELS),
    ):
        raise ValueError("Round 54 seed-Q tensor has invalid dimensions")
    if policy_id == POLICY_IDS[0]:
        return values[..., 2]
    if policy_id == POLICY_IDS[1]:
        return np.mean(values[..., :2], axis=-1)
    raise KeyError(policy_id)


def pairwise_seed_score_spearman(
    seed_q_bps: np.ndarray,
    mask: np.ndarray,
    policy_id: str,
) -> float:
    scores = policy_score(seed_q_bps, policy_id)[:, mask]
    rows: list[float] = []
    for left in range(scores.shape[0]):
        for right in range(left + 1, scores.shape[0]):
            result = spearmanr(scores[left].ravel(), scores[right].ravel())
            value = float(result.statistic)
            rows.append(value if math.isfinite(value) else 0.0)
    return min(rows) if rows else 0.0


def consensus_policy_actions(
    seed_q_bps: np.ndarray,
    mask: np.ndarray,
    policy_id: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Generate one deterministic stateful action ledger with all-seed consensus."""

    scores = policy_score(seed_q_bps, policy_id)
    if scores.shape[1] != mask.size or scores.shape[2] != len(SYMBOLS):
        raise ValueError("Round 54 policy score grid differs from the role mask")
    indexes = np.flatnonzero(mask)
    if indexes.size == 0 or not np.all(np.diff(indexes) == 1):
        raise ValueError("Round 54 policy role must be one contiguous nonempty interval")
    actions = np.zeros((mask.size, len(SYMBOLS)), dtype=np.int8)
    positions = np.zeros(len(SYMBOLS), dtype=np.int8)
    unanimous_decisions = 0
    changed_decisions = 0
    considered = 0
    for index in indexes:
        for symbol_index in range(len(SYMBOLS)):
            considered += 1
            previous_index = int(positions[symbol_index]) + 1
            seed_scores = scores[:, index, symbol_index, previous_index, :]
            seed_choices = np.argmax(seed_scores, axis=1)
            ensemble_choice = int(np.argmax(np.median(seed_scores, axis=0)))
            unanimous = bool(np.all(seed_choices == ensemble_choice))
            advantages = (
                seed_scores[:, ensemble_choice] - seed_scores[:, previous_index]
            )
            if unanimous:
                unanimous_decisions += 1
            if unanimous and float(np.min(advantages)) > 0.0:
                selected_position = ACTIONS[ensemble_choice]
                if selected_position != int(positions[symbol_index]):
                    changed_decisions += 1
                positions[symbol_index] = selected_position
            actions[index, symbol_index] = positions[symbol_index]
    return actions, {
        "policy_id": policy_id,
        "considered_symbol_hours": considered,
        "unanimous_symbol_hours": unanimous_decisions,
        "unanimous_fraction": unanimous_decisions / considered,
        "position_changes": changed_decisions,
    }


def _maximum_drawdown(equity: np.ndarray) -> float:
    peaks = np.maximum.accumulate(equity)
    return float(np.max(1.0 - equity / peaks))


def _block_bootstrap_mean(
    values: np.ndarray,
    *,
    seed: int,
    samples: int,
    block_hours: int,
    lower_quantile: float,
) -> dict[str, object]:
    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim != 1 or rows.size == 0 or not np.isfinite(rows).all():
        raise ValueError("Round 54 bootstrap returns are invalid")
    block = min(block_hours, rows.size)
    blocks = int(math.ceil(rows.size / block))
    generator = np.random.default_rng(seed)
    output = np.empty(samples, dtype=np.float64)
    for sample in range(samples):
        starts = generator.integers(0, rows.size, size=blocks)
        indexes = np.concatenate(
            [(np.arange(block, dtype=np.int64) + start) % rows.size for start in starts]
        )[: rows.size]
        output[sample] = float(np.mean(rows[indexes]))
    return {
        "samples": samples,
        "block_hours": block,
        "lower_quantile": lower_quantile,
        "lower_bps": float(np.quantile(output, lower_quantile)),
        "median_bps": float(np.median(output)),
        "upper_bps": float(np.quantile(output, 1.0 - lower_quantile)),
    }


def replay_consensus_actions(
    dataset: DistributionalDataset,
    actions: np.ndarray,
    mask: np.ndarray,
    *,
    policy_id: str,
    role: str,
    scenario: str,
    one_way_cost_bps: float,
    bootstrap_seed: int,
    spec: SequentialQSpec = DEFAULT_SPEC,
) -> SequentialReplay:
    """Replay a fixed action ledger and force-close every terminal position."""

    indexes = np.flatnonzero(mask)
    if actions.shape != dataset.hourly_return_bps.shape:
        raise ValueError("Round 54 action ledger dimensions are invalid")
    if indexes.size == 0 or not np.all(np.diff(indexes) == 1):
        raise ValueError("Round 54 replay role must be contiguous")
    selected_actions = actions[indexes].astype(np.int8, copy=True)
    if not np.isin(selected_actions, ACTIONS).all():
        raise ValueError("Round 54 action ledger contains an invalid position")
    previous = np.vstack(
        (np.zeros((1, len(SYMBOLS)), dtype=np.int8), selected_actions[:-1])
    )
    transition_units = np.abs(selected_actions - previous).astype(np.int8)
    symbol_net_bps = (
        selected_actions.astype(np.float64)
        * dataset.hourly_return_bps[indexes].astype(np.float64)
        - one_way_cost_bps * transition_units
    )
    terminal_units = np.abs(selected_actions[-1]).astype(np.int8)
    symbol_net_bps[-1] -= one_way_cost_bps * terminal_units
    portfolio_bps = np.mean(symbol_net_bps, axis=1)
    fractions = portfolio_bps / 10_000.0
    if np.any(fractions <= -1.0) or not np.isfinite(fractions).all():
        raise RuntimeError("Round 54 replay produced invalid portfolio returns")
    equity = np.cumprod(1.0 + fractions)
    positive = float(np.sum(portfolio_bps[portfolio_bps > 0.0]))
    negative = float(-np.sum(portfolio_bps[portfolio_bps < 0.0]))
    changes = selected_actions != previous
    entries = (previous == 0) & (selected_actions != 0)
    exits = (previous != 0) & (selected_actions == 0)
    flips = (
        (previous != 0)
        & (selected_actions != 0)
        & (previous != selected_actions)
    )
    closed_trades = int(np.count_nonzero(exits) + np.count_nonzero(flips))
    closed_trades += int(np.count_nonzero(terminal_units))
    timestamps = dataset.timestamps_ms[indexes]
    day_labels = np.asarray(
        [datetime.fromtimestamp(int(value) / 1_000, tz=UTC).date() for value in timestamps]
    )
    month_labels = np.asarray(
        [datetime.fromtimestamp(int(value) / 1_000, tz=UTC).strftime("%Y-%m") for value in timestamps]
    )
    active_days = len(
        {
            day_labels[row]
            for row in range(day_labels.size)
            if bool(np.any(changes[row]))
        }
    )
    monthly: list[dict[str, object]] = []
    for month in np.unique(month_labels):
        month_mask = month_labels == month
        month_equity = float(np.prod(1.0 + fractions[month_mask]))
        monthly.append(
            {
                "month": str(month),
                "hours": int(np.count_nonzero(month_mask)),
                "active_hours": int(np.count_nonzero(np.any(changes[month_mask], axis=1))),
                "total_net_return_fraction": month_equity - 1.0,
            }
        )
    symbol_totals = np.sum(symbol_net_bps, axis=0)
    absolute_total = float(np.sum(np.abs(symbol_totals)))
    metrics: dict[str, object] = {
        "policy_id": policy_id,
        "role": role,
        "scenario": scenario,
        "one_way_cost_bps": one_way_cost_bps,
        "hours": int(indexes.size),
        "total_net_return_fraction": float(equity[-1] - 1.0),
        "maximum_drawdown_fraction": _maximum_drawdown(equity),
        "mean_hourly_portfolio_bps": float(np.mean(portfolio_bps)),
        "profit_factor": positive / negative if negative > 0.0 else None,
        "position_change_events": int(np.count_nonzero(changes)),
        "transition_units": int(np.sum(transition_units) + np.sum(terminal_units)),
        "entries": int(np.count_nonzero(entries) + np.count_nonzero(flips)),
        "exits": int(np.count_nonzero(exits) + np.count_nonzero(flips) + np.count_nonzero(terminal_units)),
        "flips": int(np.count_nonzero(flips)),
        "closed_trades": closed_trades,
        "active_days": active_days,
        "positive_months": int(
            sum(float(row["total_net_return_fraction"]) > 0.0 for row in monthly)
        ),
        "symbol_net_bps": {
            symbol: float(symbol_totals[index])
            for index, symbol in enumerate(SYMBOLS)
        },
        "symbol_closed_trades": {
            symbol: int(
                np.count_nonzero(exits[:, index])
                + np.count_nonzero(flips[:, index])
                + terminal_units[index]
            )
            for index, symbol in enumerate(SYMBOLS)
        },
        "symbols_with_activity": int(
            sum(np.any(changes[:, index]) for index in range(len(SYMBOLS)))
        ),
        "maximum_single_symbol_fraction_of_absolute_net_pnl": (
            float(np.max(np.abs(symbol_totals)) / absolute_total)
            if absolute_total > 0.0
            else 1.0
        ),
        "monthly": monthly,
        "bootstrap_mean_hourly_portfolio_bps": _block_bootstrap_mean(
            portfolio_bps,
            seed=bootstrap_seed,
            samples=spec.bootstrap_samples,
            block_hours=spec.bootstrap_block_hours,
            lower_quantile=spec.bootstrap_lower_quantile,
        ),
    }
    return SequentialReplay(
        policy_id=policy_id,
        role=role,
        scenario=scenario,
        one_way_cost_bps=one_way_cost_bps,
        timestamps_ms=timestamps.copy(),
        actions=selected_actions,
        symbol_net_bps=symbol_net_bps,
        portfolio_return_bps=portfolio_bps,
        metrics=metrics,
    )


__all__ = [
    "ACTIONS",
    "DEFAULT_SPEC",
    "DuelingDistributionalQTCN",
    "POLICY_IDS",
    "QUANTILE_LEVELS",
    "ROUND",
    "SequentialQArtifact",
    "SequentialQEnsemble",
    "SequentialQSpec",
    "SequentialReplay",
    "bellman_quantile_targets",
    "consensus_policy_actions",
    "fit_reward_scale_bps",
    "pairwise_seed_score_spearman",
    "policy_score",
    "quantile_huber_loss",
    "replay_consensus_actions",
    "sequential_q_preflight",
    "train_sequential_q_ensemble",
    "transition_reward_bps",
]
