"""Replay-aligned multitask causal TCN candidates for Round 47."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence
import warnings

import numpy as np
from scipy.stats import rankdata, spearmanr
import torch
from torch import nn
from torch.nn import functional as F

from .compute import require_backend, resolve_backend, torch_device_for_backend
from .cross_asset_cost_data import SYMBOLS
from .distributional_tcn_model import (
    BASE_ONE_WAY_COST_BPS,
    DILATIONS,
    HIDDEN_CHANNELS,
    HORIZONS,
    QUANTILES,
    RECEPTIVE_FIELD,
    CausalResidualBlock,
    DistributionalDataset,
    ExplicitAdamW,
    FeatureScaler,
    TargetScaler,
    fit_feature_scaler,
    fit_target_scaler,
    role_mask,
)
from .stable_distributional_tcn_model import pinball_components


ROUND = 47
CANDIDATES = ("cost_aware_utility", "cost_aware_utility_rank")
SEEDS = (4701, 4702, 4703)
WINDOW_HOURS = 384
SUPERVISED_HOURS = WINDOW_HOURS - RECEPTIVE_FIELD + 1
WINDOW_STRIDE_HOURS = 24
BATCH_SIZE = 64
MAXIMUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
MINIMUM_IMPROVEMENT = 1e-5
UTILITY_MSE_WEIGHT = 0.1
ACTION_BCE_WEIGHT = 0.2
CONSISTENCY_WEIGHT = 0.05
RANK_WEIGHT = 0.05
RANK_OFFSET_HOURS = 24
ACTION_PROBABILITY_FLOOR = 0.55
ROUND_TRIP_COST_BPS = 2.0 * BASE_ONE_WAY_COST_BPS
ROUND46_BEST_PINBALL = 0.3669017893631544
MAXIMUM_EARLY_STOP_PINBALL = ROUND46_BEST_PINBALL * 1.02
TEMPERATURE_GRID = np.geomspace(0.25, 4.0, 301, dtype=np.float64)
ProgressCallback = Callable[[str, Mapping[str, object]], None]


@dataclass(frozen=True)
class UtilityScaler:
    mean_bps: np.ndarray
    scale_bps: np.ndarray

    def normalize(self, values: np.ndarray) -> np.ndarray:
        return (
            (values - self.mean_bps.reshape(1, 1, -1))
            / self.scale_bps.reshape(1, 1, -1)
        ).astype(np.float32)

    def denormalize(self, values: np.ndarray) -> np.ndarray:
        return (
            values * self.scale_bps.reshape(*((1,) * (values.ndim - 1)), -1)
            + self.mean_bps.reshape(*((1,) * (values.ndim - 1)), -1)
        ).astype(np.float32)

    def asdict(self) -> dict[str, object]:
        return {
            "mean_bps": self.mean_bps.tolist(),
            "scale_bps": self.scale_bps.tolist(),
        }


@dataclass(frozen=True)
class UtilityTCNArtifact:
    candidate_id: str
    seed: int
    epochs: int
    best_epoch: int
    best_early_stop_composite: float
    best_early_stop_pinball: float
    best_early_stop_utility_mse: float
    best_early_stop_action_bce: float
    best_early_stop_rank_loss: float
    optimizer_updates: int
    parameter_count: int
    temperature: float
    calibration_bce_before: float
    calibration_bce_after: float
    backend_kind: str
    backend_device: str
    path: str
    bytes: int
    sha256: str
    reload_max_abs_quantile_error: float
    reload_max_abs_utility_error: float
    reload_max_abs_logit_error: float
    warning_count: int

    def asdict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class UtilityForecastBundle:
    candidate_id: str
    seed_predictions_bps: np.ndarray
    ensemble_predictions_bps: np.ndarray
    seed_utility_bps: np.ndarray
    ensemble_utility_bps: np.ndarray
    seed_action_logits: np.ndarray
    seed_action_probabilities: np.ndarray
    ensemble_action_probabilities: np.ndarray
    artifacts: tuple[UtilityTCNArtifact, ...]
    feature_scaler: FeatureScaler
    target_scaler: TargetScaler
    utility_scaler: UtilityScaler
    backend_kind: str
    backend_device: str
    training_history: tuple[Mapping[str, object], ...]


@dataclass(frozen=True)
class UtilityTrade:
    trade_id: str
    candidate_id: str
    symbol: str
    symbol_index: int
    decision_index: int
    decision_time_ms: int
    side: int
    horizon_hours: int
    predicted_signed_utility_bps: float
    worst_seed_expected_net_bps: float
    worst_seed_probability: float

    def asdict(self) -> dict[str, object]:
        return asdict(self)


class CostAwareUtilityTCN(nn.Module):
    """Shared causal encoder with quantile, mean-utility, and action heads."""

    def __init__(self, input_channels: int = 71, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_channels = input_channels
        self.projection = nn.Conv1d(input_channels, HIDDEN_CHANNELS, kernel_size=1)
        self.blocks = nn.ModuleList(
            CausalResidualBlock(HIDDEN_CHANNELS, dilation, dropout)
            for dilation in DILATIONS
        )
        self.quantile_head = nn.Conv1d(
            HIDDEN_CHANNELS,
            len(HORIZONS) * len(QUANTILES),
            kernel_size=1,
        )
        self.utility_head = nn.Conv1d(
            HIDDEN_CHANNELS,
            len(HORIZONS),
            kernel_size=1,
        )
        self.action_head = nn.Conv1d(
            HIDDEN_CHANNELS,
            len(HORIZONS) * 2,
            kernel_size=1,
        )
        with torch.no_grad():
            if self.quantile_head.bias is not None:
                bias = self.quantile_head.bias.reshape(len(HORIZONS), len(QUANTILES))
                bias[:, 1:] = -2.0
                bias[:, 0] = 0.0

    def forward(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if values.ndim != 3 or values.shape[1] != self.input_channels:
            raise ValueError("Round 47 TCN input dimensions are invalid")
        encoded = F.gelu(self.projection(values))
        for block in self.blocks:
            encoded = block(encoded)
        raw = self.quantile_head(encoded).reshape(
            values.shape[0], len(HORIZONS), len(QUANTILES), values.shape[-1]
        )
        median = raw[:, :, 0, :]
        q25 = median - F.softplus(raw[:, :, 1, :])
        q10 = q25 - F.softplus(raw[:, :, 2, :])
        q75 = median + F.softplus(raw[:, :, 3, :])
        q90 = q75 + F.softplus(raw[:, :, 4, :])
        quantiles = torch.stack((q10, q25, median, q75, q90), dim=2)
        utility = self.utility_head(encoded)
        logits = self.action_head(encoded).reshape(
            values.shape[0], len(HORIZONS), 2, values.shape[-1]
        )
        return quantiles, utility, logits


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def additive_forward_utility_bps(
    hourly_return_bps: np.ndarray,
) -> np.ndarray:
    """Sum replay-accounted hourly signed utility for every frozen horizon."""

    values = np.asarray(hourly_return_bps, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(SYMBOLS):
        raise ValueError("Round 47 hourly utility matrix has invalid dimensions")
    if not np.isfinite(values).all():
        raise ValueError("Round 47 hourly utility matrix is nonfinite")
    cumulative = np.vstack(
        (np.zeros((1, values.shape[1]), dtype=np.float64), np.cumsum(values, axis=0))
    )
    output = np.full(
        (values.shape[0], values.shape[1], len(HORIZONS)),
        np.nan,
        dtype=np.float32,
    )
    for horizon_index, horizon in enumerate(HORIZONS):
        output[: values.shape[0] - horizon + 1, :, horizon_index] = (
            cumulative[horizon:] - cumulative[:-horizon]
        ).astype(np.float32)
    return output


def fit_utility_scaler(
    utility_bps: np.ndarray,
    training_mask: np.ndarray,
) -> UtilityScaler:
    selected = utility_bps[training_mask].reshape(-1, len(HORIZONS)).astype(np.float64)
    if not np.isfinite(selected).all():
        raise ValueError("Round 47 utility scaler input is nonfinite")
    mean = np.mean(selected, axis=0)
    scale = np.maximum(np.std(selected, axis=0), 1.0)
    return UtilityScaler(mean_bps=mean, scale_bps=scale)


def action_labels(utility_bps: np.ndarray) -> np.ndarray:
    values = np.asarray(utility_bps)
    return np.stack(
        (values < -ROUND_TRIP_COST_BPS, values > ROUND_TRIP_COST_BPS),
        axis=-1,
    ).astype(np.float32)


def pairwise_utility_rank_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    offset: int = RANK_OFFSET_HOURS,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Round 47 pairwise tensors have invalid dimensions")
    if offset <= 0 or prediction.shape[-1] <= offset:
        raise ValueError("Round 47 pairwise offset is invalid")
    prediction_delta = prediction[..., :-offset] - prediction[..., offset:]
    target_delta = target[..., :-offset] - target[..., offset:]
    non_tie = (target_delta != 0.0).to(dtype=prediction.dtype)
    direction = torch.where(
        target_delta > 0.0,
        torch.ones_like(target_delta),
        -torch.ones_like(target_delta),
    )
    numerator = (F.softplus(-direction * prediction_delta) * non_tie).sum()
    return numerator / torch.clamp(non_tie.sum(), min=1.0)


def binary_logistic_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Numerically stable BCE form that stays native on DirectML."""

    if logits.shape != labels.shape:
        raise ValueError("Round 47 logistic tensors have incompatible dimensions")
    return (
        torch.clamp(logits, min=0.0)
        - logits * labels
        + torch.log1p(torch.exp(-torch.abs(logits)))
    ).mean()


def _standardized_peer_consistency(
    peer_values: Sequence[torch.Tensor],
    *,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(peer_values) != len(SEEDS):
        raise ValueError("Round 47 consistency requires three peers")
    standardized: list[torch.Tensor] = []
    shape = peer_values[0].shape
    for values in peer_values:
        if values.shape != shape or values.ndim != 3:
            raise ValueError("Round 47 peer consistency shapes differ")
        center = values.mean(dim=(0, 2), keepdim=True)
        variance = ((values - center) ** 2).mean(dim=(0, 2), keepdim=True)
        standardized.append((values - center) / torch.sqrt(variance + epsilon))
    rows: list[torch.Tensor] = []
    for left in range(len(standardized)):
        for right in range(left + 1, len(standardized)):
            rows.append(
                0.5 * ((standardized[left] - standardized[right]) ** 2).mean(dim=(0, 2))
            )
    matrix = torch.stack(rows)
    return matrix.mean(), matrix


def multitask_objective(
    peer_outputs: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    quantile_targets: torch.Tensor,
    utility_targets: torch.Tensor,
    labels: torch.Tensor,
    *,
    rank_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if len(peer_outputs) != len(SEEDS) or rank_weight not in (0.0, RANK_WEIGHT):
        raise ValueError("Round 47 multitask candidate contract is invalid")
    pinball = torch.stack(
        [
            pinball_components(output[0], quantile_targets).mean()
            for output in peer_outputs
        ]
    )
    utility_mse = torch.stack(
        [F.mse_loss(output[1], utility_targets) for output in peer_outputs]
    )
    action_bce = torch.stack(
        [binary_logistic_loss(output[2], labels) for output in peer_outputs]
    )
    ranking = torch.stack(
        [
            pairwise_utility_rank_loss(output[1], utility_targets)
            for output in peer_outputs
        ]
    )
    median_consistency, median_pairs = _standardized_peer_consistency(
        [output[0][:, :, 2, :] for output in peer_outputs]
    )
    utility_consistency, utility_pairs = _standardized_peer_consistency(
        [output[1] for output in peer_outputs]
    )
    consistency = 0.5 * (median_consistency + utility_consistency)
    supervised = (
        pinball
        + UTILITY_MSE_WEIGHT * utility_mse
        + ACTION_BCE_WEIGHT * action_bce
        + rank_weight * ranking
    )
    objective = supervised.mean() + CONSISTENCY_WEIGHT * consistency
    return objective, {
        "pinball": pinball,
        "utility_mse": utility_mse,
        "action_bce": action_bce,
        "ranking": ranking,
        "median_consistency": median_consistency,
        "utility_consistency": utility_consistency,
        "consistency": consistency,
        "median_pair_horizon": median_pairs,
        "utility_pair_horizon": utility_pairs,
    }


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
) -> dict[str, object]:
    generator = np.random.default_rng(SEEDS[0])
    values = torch.from_numpy(
        generator.normal(size=(8, 71, 160)).astype(np.float32)
    ).to(device)
    quantile_targets = torch.from_numpy(
        generator.normal(size=(8, len(HORIZONS), 160)).astype(np.float32)
    ).to(device)
    utility_targets = torch.from_numpy(
        generator.normal(size=(8, len(HORIZONS), 160)).astype(np.float32)
    ).to(device)
    labels = torch.from_numpy(
        generator.integers(0, 2, size=(8, len(HORIZONS), 2, 160)).astype(np.float32)
    ).to(device)
    rows: dict[str, object] = {**metadata}
    messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for candidate_id, rank_weight in zip(
            CANDIDATES, (0.0, RANK_WEIGHT), strict=True
        ):
            peers: list[CostAwareUtilityTCN] = []
            optimizers: list[ExplicitAdamW] = []
            for seed in SEEDS:
                torch.manual_seed(seed)
                peer = CostAwareUtilityTCN().to(device)
                peers.append(peer)
                optimizers.append(
                    ExplicitAdamW(
                        tuple(peer.parameters()),
                        learning_rate=1e-3,
                        weight_decay=1e-4,
                    )
                )
            before = {
                "quantile": peers[0].quantile_head.weight.detach().cpu().clone(),
                "utility": peers[0].utility_head.weight.detach().cpu().clone(),
                "action": peers[0].action_head.weight.detach().cpu().clone(),
            }
            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)
            outputs = [peer(values) for peer in peers]
            objective, components = multitask_objective(
                outputs,
                quantile_targets,
                utility_targets,
                labels,
                rank_weight=rank_weight,
            )
            objective.backward()
            for peer, optimizer in zip(peers, optimizers, strict=True):
                torch.nn.utils.clip_grad_norm_(peer.parameters(), 1.0, foreach=False)
                optimizer.step()
            changes = {
                "quantile": float(
                    torch.max(
                        torch.abs(
                            peers[0].quantile_head.weight.detach().cpu()
                            - before["quantile"]
                        )
                    )
                ),
                "utility": float(
                    torch.max(
                        torch.abs(
                            peers[0].utility_head.weight.detach().cpu()
                            - before["utility"]
                        )
                    )
                ),
                "action": float(
                    torch.max(
                        torch.abs(
                            peers[0].action_head.weight.detach().cpu()
                            - before["action"]
                        )
                    )
                ),
            }
            scalars = {
                "objective": float(objective.detach().cpu()),
                "pinball": float(components["pinball"].mean().detach().cpu()),
                "utility_mse": float(components["utility_mse"].mean().detach().cpu()),
                "action_bce": float(components["action_bce"].mean().detach().cpu()),
                "ranking": float(components["ranking"].mean().detach().cpu()),
                "consistency": float(components["consistency"].detach().cpu()),
            }
            if not all(
                math.isfinite(value) for value in (*scalars.values(), *changes.values())
            ):
                raise RuntimeError("Round 47 preflight produced nonfinite evidence")
            if any(value <= 0.0 for value in changes.values()):
                raise RuntimeError(
                    f"Round 47 preflight did not update every head: {changes}"
                )
            rows[candidate_id] = {
                **scalars,
                "rank_weight": rank_weight,
                "head_parameter_max_abs_change": changes,
            }
    messages.extend(str(item.message) for item in caught)
    fallback = _fallback_messages(messages)
    if fallback:
        raise RuntimeError(f"Round 47 compute preflight used CPU fallback: {fallback}")
    rows["warning_count"] = len(messages)
    rows["cpu_fallback_warning_count"] = 0
    return rows


def utility_preflight(
    compute_backend: str = "auto",
) -> tuple[object, dict[str, object]]:
    backend = require_backend(resolve_backend(compute_backend))
    device = torch_device_for_backend(backend)
    evidence = _run_preflight(
        device,
        {
            "backend_kind": backend.kind,
            "backend_device": str(device),
            "torch_version": str(torch.__version__),
        },
    )
    return device, evidence


def directml_utility_preflight() -> tuple[object, dict[str, object]]:
    return utility_preflight("directml")


def cpu_utility_preflight() -> tuple[object, dict[str, object]]:
    return utility_preflight("cpu")


def _training_windows(mask: np.ndarray) -> np.ndarray:
    indexes = np.flatnonzero(mask)
    if indexes.size < WINDOW_HOURS or not np.all(np.diff(indexes) == 1):
        raise ValueError("Round 47 training role is not a contiguous span")
    first = int(indexes[0])
    last_start = int(indexes[-1]) - WINDOW_HOURS + 1
    starts = np.arange(first, last_start + 1, WINDOW_STRIDE_HOURS, dtype=np.int64)
    if starts.size == 0:
        raise ValueError("Round 47 has no training windows")
    return starts


def _training_pairs(dataset: DistributionalDataset) -> np.ndarray:
    starts = _training_windows(role_mask(dataset, "training"))
    return np.asarray(
        [(int(start), symbol) for start in starts for symbol in range(len(SYMBOLS))],
        dtype=np.int64,
    )


def _batch_arrays(
    normalized_features: np.ndarray,
    normalized_quantiles: np.ndarray,
    normalized_utility: np.ndarray,
    raw_utility: np.ndarray,
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = np.stack(
        [
            normalized_features[start : start + WINDOW_HOURS, symbol, :].T
            for start, symbol in selected
        ]
    ).astype(np.float32, copy=False)
    quantiles = np.stack(
        [
            normalized_quantiles[start : start + WINDOW_HOURS, symbol, :].T
            for start, symbol in selected
        ]
    ).astype(np.float32, copy=False)
    utility = np.stack(
        [
            normalized_utility[start : start + WINDOW_HOURS, symbol, :].T
            for start, symbol in selected
        ]
    ).astype(np.float32, copy=False)
    raw = np.stack(
        [
            raw_utility[start : start + WINDOW_HOURS, symbol, :].T
            for start, symbol in selected
        ]
    ).astype(np.float32, copy=False)
    return tuple(
        np.ascontiguousarray(value) for value in (features, quantiles, utility, raw)
    )  # type: ignore[return-value]


def _predict_all(
    model: CostAwareUtilityTCN,
    normalized_features: np.ndarray,
    target_scaler: TargetScaler,
    utility_scaler: UtilityScaler,
    device: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    input_values = normalized_features.transpose(1, 2, 0)
    with torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(input_values)).to(device)
        quantile_tensor, utility_tensor, logit_tensor = model(tensor)
        normalized_quantiles = quantile_tensor.detach().cpu().numpy()
        normalized_utility = utility_tensor.detach().cpu().numpy()
        logits = logit_tensor.detach().cpu().numpy()
    quantiles = target_scaler.denormalize(
        normalized_quantiles.transpose(3, 0, 1, 2)
    ).astype(np.float32)
    utility = utility_scaler.denormalize(normalized_utility.transpose(2, 0, 1)).astype(
        np.float32
    )
    logits = logits.transpose(3, 0, 1, 2).astype(np.float32)
    expected = normalized_features.shape[0]
    if (
        quantiles.shape != (expected, len(SYMBOLS), len(HORIZONS), len(QUANTILES))
        or utility.shape != (expected, len(SYMBOLS), len(HORIZONS))
        or logits.shape != (expected, len(SYMBOLS), len(HORIZONS), 2)
        or not np.isfinite(quantiles).all()
        or not np.isfinite(utility).all()
        or not np.isfinite(logits).all()
    ):
        raise ValueError("Round 47 prediction tensors are invalid")
    return quantiles, utility, logits


def _numpy_pinball(actual: np.ndarray, predictions: np.ndarray) -> float:
    errors = actual[..., None] - predictions
    quantiles = np.asarray(QUANTILES, dtype=np.float64)
    return float(np.mean(np.maximum(quantiles * errors, (quantiles - 1.0) * errors)))


def _log_loss_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, values) - targets * values))


def _numpy_rank_loss(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    offset: int = RANK_OFFSET_HOURS,
) -> float:
    prediction_delta = prediction[:-offset] - prediction[offset:]
    target_delta = target[:-offset] - target[offset:]
    selected = target_delta != 0.0
    if not np.any(selected):
        return 0.0
    direction = np.where(target_delta[selected] > 0.0, 1.0, -1.0)
    return float(np.mean(np.logaddexp(0.0, -direction * prediction_delta[selected])))


def _role_losses(
    dataset: DistributionalDataset,
    utility_bps: np.ndarray,
    quantile_predictions: np.ndarray,
    utility_predictions: np.ndarray,
    action_logits_values: np.ndarray,
    mask: np.ndarray,
    target_scaler: TargetScaler,
    utility_scaler: UtilityScaler,
    *,
    rank_weight: float,
) -> dict[str, float]:
    actual_quantile = target_scaler.normalize(dataset.forward_return_bps[mask])
    predicted_quantile = (
        quantile_predictions[mask]
        - target_scaler.center_bps.reshape(1, 1, len(HORIZONS), 1)
    ) / target_scaler.scale_bps.reshape(1, 1, len(HORIZONS), 1)
    actual_utility = utility_scaler.normalize(utility_bps[mask])
    predicted_utility = utility_scaler.normalize(utility_predictions[mask])
    labels = action_labels(utility_bps[mask])
    pinball = _numpy_pinball(actual_quantile, predicted_quantile)
    utility_mse = float(np.mean((predicted_utility - actual_utility) ** 2))
    action_bce = _log_loss_from_logits(action_logits_values[mask], labels)
    ranking = _numpy_rank_loss(predicted_utility, actual_utility)
    return {
        "pinball": pinball,
        "utility_mse": utility_mse,
        "action_bce": action_bce,
        "rank_loss": ranking,
        "composite": pinball
        + UTILITY_MSE_WEIGHT * utility_mse
        + ACTION_BCE_WEIGHT * action_bce
        + rank_weight * ranking,
    }


def _clone_state(model: CostAwareUtilityTCN) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, float]:
    values = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64)
    before = _log_loss_from_logits(values, targets)
    losses = np.asarray(
        [
            _log_loss_from_logits(values / temperature, targets)
            for temperature in TEMPERATURE_GRID
        ]
    )
    minimum = float(np.min(losses))
    tied = np.flatnonzero(np.isclose(losses, minimum, rtol=0.0, atol=1e-15))
    selected = min(
        tied.tolist(),
        key=lambda index: (
            abs(float(TEMPERATURE_GRID[index]) - 1.0),
            float(TEMPERATURE_GRID[index]),
        ),
    )
    temperature = float(TEMPERATURE_GRID[selected])
    after = float(losses[selected])
    if after > before + 1e-12:
        raise RuntimeError("Round 47 temperature scaling worsened calibration BCE")
    return temperature, before, after


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), -80.0, 80.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def _finalize_artifact(
    *,
    dataset: DistributionalDataset,
    utility_bps: np.ndarray,
    candidate_id: str,
    seed: int,
    epochs: int,
    best_epoch: int,
    best_losses: Mapping[str, float],
    optimizer_updates: int,
    best_state: Mapping[str, torch.Tensor],
    best_outputs: tuple[np.ndarray, np.ndarray, np.ndarray],
    artifact_path: Path,
    normalized_features: np.ndarray,
    target_scaler: TargetScaler,
    utility_scaler: UtilityScaler,
    device: object,
    backend_kind: str,
    backend_device: str,
    warning_messages: list[str],
) -> tuple[
    UtilityTCNArtifact,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(best_state), artifact_path)
    reloaded = CostAwareUtilityTCN(input_channels=normalized_features.shape[-1])
    reloaded.load_state_dict(
        torch.load(artifact_path, map_location="cpu", weights_only=True)
    )
    reloaded = reloaded.to(device)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reloaded_outputs = _predict_all(
            reloaded,
            normalized_features,
            target_scaler,
            utility_scaler,
            device,
        )
    warning_messages.extend(str(item.message) for item in caught)
    fallback = _fallback_messages(warning_messages)
    if fallback:
        raise RuntimeError(f"Round 47 artifact reload used CPU fallback: {fallback}")
    probe = np.flatnonzero(role_mask(dataset, "evaluation"))[:512]
    errors = tuple(
        float(np.max(np.abs(reference[probe] - observed[probe])))
        for reference, observed in zip(best_outputs, reloaded_outputs, strict=True)
    )
    if not all(math.isfinite(value) and value <= 1e-6 for value in errors):
        raise RuntimeError(f"Round 47 artifact reload errors are {errors}")
    calibration_mask = role_mask(dataset, "calibration")
    temperature, bce_before, bce_after = _fit_temperature(
        reloaded_outputs[2][calibration_mask],
        action_labels(utility_bps[calibration_mask]),
    )
    probabilities = _sigmoid(reloaded_outputs[2] / temperature)
    artifact = UtilityTCNArtifact(
        candidate_id=candidate_id,
        seed=seed,
        epochs=epochs,
        best_epoch=best_epoch,
        best_early_stop_composite=float(best_losses["composite"]),
        best_early_stop_pinball=float(best_losses["pinball"]),
        best_early_stop_utility_mse=float(best_losses["utility_mse"]),
        best_early_stop_action_bce=float(best_losses["action_bce"]),
        best_early_stop_rank_loss=float(best_losses["rank_loss"]),
        optimizer_updates=optimizer_updates,
        parameter_count=sum(value.numel() for value in reloaded.parameters()),
        temperature=temperature,
        calibration_bce_before=bce_before,
        calibration_bce_after=bce_after,
        backend_kind=backend_kind,
        backend_device=backend_device,
        path=str(artifact_path),
        bytes=artifact_path.stat().st_size,
        sha256=_file_sha256(artifact_path),
        reload_max_abs_quantile_error=errors[0],
        reload_max_abs_utility_error=errors[1],
        reload_max_abs_logit_error=errors[2],
        warning_count=len(warning_messages),
    )
    return artifact, *reloaded_outputs, probabilities


def _train_candidate(
    dataset: DistributionalDataset,
    utility_bps: np.ndarray,
    normalized_features: np.ndarray,
    feature_scaler: FeatureScaler,
    target_scaler: TargetScaler,
    utility_scaler: UtilityScaler,
    *,
    candidate_id: str,
    rank_weight: float,
    model_dir: Path,
    device: object,
    backend_kind: str,
    backend_device: str,
    progress: ProgressCallback | None,
) -> UtilityForecastBundle:
    peers: list[CostAwareUtilityTCN] = []
    optimizers: list[ExplicitAdamW] = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        peer = CostAwareUtilityTCN(input_channels=normalized_features.shape[-1]).to(
            device
        )
        peers.append(peer)
        optimizers.append(
            ExplicitAdamW(
                tuple(peer.parameters()), learning_rate=1e-3, weight_decay=1e-4
            )
        )
    generator = np.random.default_rng(SEEDS[0])
    pairs = _training_pairs(dataset)
    normalized_quantiles = target_scaler.normalize(dataset.forward_return_bps)
    normalized_utility = utility_scaler.normalize(utility_bps)
    validation_mask = role_mask(dataset, "early_stop")
    best_loss = math.inf
    best_epoch = 0
    best_states: list[dict[str, torch.Tensor]] | None = None
    best_losses: dict[str, float] | None = None
    best_outputs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None
    stale_epochs = 0
    optimizer_updates = 0
    epochs_run = 0
    history: list[Mapping[str, object]] = []
    warning_messages: list[str] = []
    for epoch in range(1, MAXIMUM_EPOCHS + 1):
        epochs_run = epoch
        order = generator.permutation(pairs.shape[0])
        for peer in peers:
            peer.train()
        batch_components: dict[str, list[float]] = {
            key: []
            for key in (
                "objective",
                "pinball",
                "utility_mse",
                "action_bce",
                "ranking",
                "median_consistency",
                "utility_consistency",
            )
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for offset in range(0, order.size, BATCH_SIZE):
                selected = pairs[order[offset : offset + BATCH_SIZE]]
                feature_batch, quantile_batch, utility_batch, raw_batch = _batch_arrays(
                    normalized_features,
                    normalized_quantiles,
                    normalized_utility,
                    utility_bps,
                    selected,
                )
                values = torch.from_numpy(feature_batch).to(device)
                quantile_targets = torch.from_numpy(quantile_batch).to(device)[
                    ..., -SUPERVISED_HOURS:
                ]
                utility_targets = torch.from_numpy(utility_batch).to(device)[
                    ..., -SUPERVISED_HOURS:
                ]
                label_values = action_labels(raw_batch).transpose(0, 1, 3, 2)
                labels = torch.from_numpy(np.ascontiguousarray(label_values)).to(
                    device
                )[..., -SUPERVISED_HOURS:]
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                outputs = []
                for peer in peers:
                    quantiles, expected_utility, logits = peer(values)
                    outputs.append(
                        (
                            quantiles[..., -SUPERVISED_HOURS:],
                            expected_utility[..., -SUPERVISED_HOURS:],
                            logits[..., -SUPERVISED_HOURS:],
                        )
                    )
                objective, components = multitask_objective(
                    outputs,
                    quantile_targets,
                    utility_targets,
                    labels,
                    rank_weight=rank_weight,
                )
                objective.backward()
                for peer, optimizer in zip(peers, optimizers, strict=True):
                    torch.nn.utils.clip_grad_norm_(
                        peer.parameters(), 1.0, foreach=False
                    )
                    optimizer.step()
                optimizer_updates += 1
                batch_components["objective"].append(float(objective.detach().cpu()))
                for key in ("pinball", "utility_mse", "action_bce", "ranking"):
                    batch_components[key].append(
                        float(components[key].mean().detach().cpu())
                    )
                for key in ("median_consistency", "utility_consistency"):
                    batch_components[key].append(float(components[key].detach().cpu()))
            full_outputs = [
                _predict_all(
                    peer,
                    normalized_features,
                    target_scaler,
                    utility_scaler,
                    device,
                )
                for peer in peers
            ]
        warning_messages.extend(str(item.message) for item in caught)
        ensemble_outputs = tuple(
            np.median(
                np.stack([output[index] for output in full_outputs]), axis=0
            ).astype(np.float32)
            for index in range(3)
        )
        validation = _role_losses(
            dataset,
            utility_bps,
            ensemble_outputs[0],
            ensemble_outputs[1],
            ensemble_outputs[2],
            validation_mask,
            target_scaler,
            utility_scaler,
            rank_weight=rank_weight,
        )
        row: dict[str, object] = {
            "candidate_id": candidate_id,
            "epoch": epoch,
            "optimizer_updates": optimizer_updates,
            "rank_weight": rank_weight,
            **{
                f"training_{key}": float(np.mean(values))
                for key, values in batch_components.items()
            },
            **{f"early_stop_{key}": value for key, value in validation.items()},
        }
        scalar_values = [value for value in row.values() if isinstance(value, float)]
        if not all(math.isfinite(value) for value in scalar_values):
            raise RuntimeError("Round 47 training produced nonfinite diagnostics")
        improved = validation["composite"] < best_loss - MINIMUM_IMPROVEMENT
        if improved:
            best_loss = validation["composite"]
            best_epoch = epoch
            best_states = [_clone_state(peer) for peer in peers]
            best_losses = dict(validation)
            best_outputs = full_outputs
            stale_epochs = 0
        else:
            stale_epochs += 1
        row["best_early_stop_composite"] = best_loss
        row["best_epoch"] = best_epoch
        row["stale_epochs"] = stale_epochs
        history.append(row)
        if progress is not None:
            progress("round47_epoch", row)
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            break
    if best_states is None or best_losses is None or best_outputs is None:
        raise RuntimeError("Round 47 training did not produce a best state")
    fallback = _fallback_messages(warning_messages)
    if fallback:
        raise RuntimeError(f"Round 47 training used CPU fallback: {fallback}")
    artifacts: list[UtilityTCNArtifact] = []
    quantile_predictions: list[np.ndarray] = []
    utility_predictions: list[np.ndarray] = []
    action_logits_values: list[np.ndarray] = []
    action_probabilities: list[np.ndarray] = []
    for seed, best_state, best_output in zip(
        SEEDS, best_states, best_outputs, strict=True
    ):
        finalized = _finalize_artifact(
            dataset=dataset,
            utility_bps=utility_bps,
            candidate_id=candidate_id,
            seed=seed,
            epochs=epochs_run,
            best_epoch=best_epoch,
            best_losses=best_losses,
            optimizer_updates=optimizer_updates,
            best_state=best_state,
            best_outputs=best_output,
            artifact_path=model_dir / f"round47_{candidate_id}_seed_{seed}.pt",
            normalized_features=normalized_features,
            target_scaler=target_scaler,
            utility_scaler=utility_scaler,
            device=device,
            backend_kind=backend_kind,
            backend_device=backend_device,
            warning_messages=warning_messages,
        )
        artifact, quantiles, utility, logits, probabilities = finalized
        artifacts.append(artifact)
        quantile_predictions.append(quantiles)
        utility_predictions.append(utility)
        action_logits_values.append(logits)
        action_probabilities.append(probabilities)
    stacked_quantiles = np.stack(quantile_predictions).astype(np.float32)
    stacked_utility = np.stack(utility_predictions).astype(np.float32)
    stacked_logits = np.stack(action_logits_values).astype(np.float32)
    stacked_probabilities = np.stack(action_probabilities).astype(np.float32)
    ensemble_quantiles = np.median(stacked_quantiles, axis=0).astype(np.float32)
    if np.any(np.diff(ensemble_quantiles, axis=-1) < -1e-7):
        raise RuntimeError("Round 47 ensemble quantiles crossed")
    return UtilityForecastBundle(
        candidate_id=candidate_id,
        seed_predictions_bps=stacked_quantiles,
        ensemble_predictions_bps=ensemble_quantiles,
        seed_utility_bps=stacked_utility,
        ensemble_utility_bps=np.median(stacked_utility, axis=0).astype(np.float32),
        seed_action_logits=stacked_logits,
        seed_action_probabilities=stacked_probabilities,
        ensemble_action_probabilities=np.median(stacked_probabilities, axis=0).astype(
            np.float32
        ),
        artifacts=tuple(artifacts),
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        utility_scaler=utility_scaler,
        backend_kind=backend_kind,
        backend_device=backend_device,
        training_history=tuple(history),
    )


def train_utility_candidates(
    dataset: DistributionalDataset,
    *,
    model_dir: Path,
    compute_backend: str = "auto",
    progress: ProgressCallback | None = None,
) -> tuple[
    dict[str, UtilityForecastBundle],
    np.ndarray,
    dict[str, object],
]:
    training_mask = role_mask(dataset, "training")
    utility_bps = additive_forward_utility_bps(dataset.hourly_return_bps)
    for role_name in ("training", "early_stop", "calibration", "evaluation"):
        if not np.isfinite(utility_bps[role_mask(dataset, role_name)]).all():
            raise ValueError(f"Round 47 {role_name} utility targets are incomplete")
    feature_scaler = fit_feature_scaler(dataset, training_mask)
    target_scaler = fit_target_scaler(dataset, training_mask)
    utility_scaler = fit_utility_scaler(utility_bps, training_mask)
    normalized_features = feature_scaler.transform(dataset.features)
    device, preflight = utility_preflight(compute_backend)
    bundles: dict[str, UtilityForecastBundle] = {}
    for candidate_id, rank_weight in zip(CANDIDATES, (0.0, RANK_WEIGHT), strict=True):
        if progress is not None:
            progress(
                "round47_candidate_training",
                {"candidate_id": candidate_id, "status": "started"},
            )
        bundles[candidate_id] = _train_candidate(
            dataset,
            utility_bps,
            normalized_features,
            feature_scaler,
            target_scaler,
            utility_scaler,
            candidate_id=candidate_id,
            rank_weight=rank_weight,
            model_dir=model_dir,
            device=device,
            backend_kind=str(preflight["backend_kind"]),
            backend_device=str(preflight["backend_device"]),
            progress=progress,
        )
        if progress is not None:
            progress(
                "round47_candidate_training",
                {
                    "candidate_id": candidate_id,
                    "status": "complete",
                    "best_epoch": bundles[candidate_id].artifacts[0].best_epoch,
                },
            )
    return bundles, utility_bps, preflight


def _finite_spearman(actual: np.ndarray, prediction: np.ndarray) -> float:
    left = np.asarray(actual, dtype=np.float64).ravel()
    right = np.asarray(prediction, dtype=np.float64).ravel()
    selected = np.isfinite(left) & np.isfinite(right)
    left = left[selected]
    right = right[selected]
    if left.size < 3 or np.std(left) <= 0.0 or np.std(right) <= 0.0:
        return 0.0
    value = float(spearmanr(left, right).statistic)
    return value if math.isfinite(value) else 0.0


def _roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    targets = np.asarray(labels, dtype=np.int8).ravel()
    values = np.asarray(scores, dtype=np.float64).ravel()
    positive = targets == 1
    negative = targets == 0
    positive_count = int(np.count_nonzero(positive))
    negative_count = int(np.count_nonzero(negative))
    if positive_count == 0 or negative_count == 0:
        return 0.5
    ranks = rankdata(values, method="average")
    return float(
        (np.sum(ranks[positive]) - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def _binary_log_loss(probabilities: np.ndarray, labels: np.ndarray) -> float:
    values = np.clip(np.asarray(probabilities, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    targets = np.asarray(labels, dtype=np.float64)
    return float(
        -np.mean(targets * np.log(values) + (1.0 - targets) * np.log1p(-values))
    )


def _calibration_error(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 10,
) -> float:
    values = np.asarray(probabilities, dtype=np.float64).ravel()
    targets = np.asarray(labels, dtype=np.float64).ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        if index + 1 == bins:
            selected = (values >= edges[index]) & (values <= edges[index + 1])
        else:
            selected = (values >= edges[index]) & (values < edges[index + 1])
        count = int(np.count_nonzero(selected))
        if count:
            error += (count / values.size) * abs(
                float(np.mean(values[selected])) - float(np.mean(targets[selected]))
            )
    return float(error)


def utility_action_diagnostics(
    dataset: DistributionalDataset,
    utility_bps: np.ndarray,
    bundle: UtilityForecastBundle,
) -> dict[str, object]:
    training_mask = role_mask(dataset, "training")
    evaluation_mask = role_mask(dataset, "evaluation")
    training_mean = np.mean(
        utility_bps[training_mask].reshape(-1, len(HORIZONS)), axis=0
    )
    training_labels = action_labels(utility_bps[training_mask])
    utility_rows: list[dict[str, object]] = []
    action_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    for horizon_index, horizon in enumerate(HORIZONS):
        actual = utility_bps[evaluation_mask, :, horizon_index].astype(np.float64)
        prediction = bundle.ensemble_utility_bps[
            evaluation_mask, :, horizon_index
        ].astype(np.float64)
        baseline = np.full_like(actual, training_mean[horizon_index])
        model_mse = float(np.mean((prediction - actual) ** 2))
        baseline_mse = float(np.mean((baseline - actual) ** 2))
        utility_rows.append(
            {
                "candidate_id": bundle.candidate_id,
                "horizon_hours": horizon,
                "rows": int(actual.size),
                "model_mse_bps2": model_mse,
                "baseline_mse_bps2": baseline_mse,
                "mse_skill": 1.0 - model_mse / baseline_mse,
                "spearman": _finite_spearman(actual, prediction),
                "pearson": float(np.corrcoef(actual.ravel(), prediction.ravel())[0, 1]),
                "actual_mean_bps": float(np.mean(actual)),
                "prediction_mean_bps": float(np.mean(prediction)),
                "training_mean_bps": float(training_mean[horizon_index]),
            }
        )
        labels = action_labels(actual)
        probabilities = bundle.ensemble_action_probabilities[
            evaluation_mask, :, horizon_index, :
        ].astype(np.float64)
        for side_index, side_name in enumerate(("short", "long")):
            target = labels[..., side_index]
            score = probabilities[..., side_index]
            prevalence = float(
                np.mean(training_labels[:, :, horizon_index, side_index])
            )
            baseline_probability = np.full_like(score, prevalence)
            model_log_loss = _binary_log_loss(score, target)
            baseline_log_loss = _binary_log_loss(baseline_probability, target)
            action_rows.append(
                {
                    "candidate_id": bundle.candidate_id,
                    "horizon_hours": horizon,
                    "side": side_name,
                    "rows": int(target.size),
                    "evaluation_prevalence": float(np.mean(target)),
                    "training_prevalence": prevalence,
                    "log_loss": model_log_loss,
                    "baseline_log_loss": baseline_log_loss,
                    "log_loss_skill": 1.0 - model_log_loss / baseline_log_loss,
                    "brier": float(np.mean((score - target) ** 2)),
                    "baseline_brier": float(
                        np.mean((baseline_probability - target) ** 2)
                    ),
                    "roc_auc": _roc_auc(target, score),
                    "expected_calibration_error_10_bin": _calibration_error(
                        score, target
                    ),
                }
            )
    minimum_utility_stability = 1.0
    minimum_logit_stability = 1.0
    for left in range(len(SEEDS)):
        for right in range(left + 1, len(SEEDS)):
            for horizon_index, horizon in enumerate(HORIZONS):
                utility_correlation = _finite_spearman(
                    bundle.seed_utility_bps[left, evaluation_mask, :, horizon_index],
                    bundle.seed_utility_bps[right, evaluation_mask, :, horizon_index],
                )
                minimum_utility_stability = min(
                    minimum_utility_stability, utility_correlation
                )
                stability_rows.append(
                    {
                        "candidate_id": bundle.candidate_id,
                        "left_seed": SEEDS[left],
                        "right_seed": SEEDS[right],
                        "horizon_hours": horizon,
                        "output": "conditional_mean_utility",
                        "side": "signed",
                        "spearman": utility_correlation,
                    }
                )
                for side_index, side_name in enumerate(("short", "long")):
                    logit_correlation = _finite_spearman(
                        bundle.seed_action_logits[
                            left, evaluation_mask, :, horizon_index, side_index
                        ],
                        bundle.seed_action_logits[
                            right, evaluation_mask, :, horizon_index, side_index
                        ],
                    )
                    minimum_logit_stability = min(
                        minimum_logit_stability, logit_correlation
                    )
                    stability_rows.append(
                        {
                            "candidate_id": bundle.candidate_id,
                            "left_seed": SEEDS[left],
                            "right_seed": SEEDS[right],
                            "horizon_hours": horizon,
                            "output": "action_logit",
                            "side": side_name,
                            "spearman": logit_correlation,
                        }
                    )
    mse_skill_count = sum(float(row["mse_skill"]) > 0.0 for row in utility_rows)
    utility_rank_count = sum(float(row["spearman"]) > 0.0 for row in utility_rows)
    log_loss_count = sum(float(row["log_loss_skill"]) > 0.0 for row in action_rows)
    auc_count = sum(float(row["roc_auc"]) > 0.5 for row in action_rows)
    pinball = float(
        np.median([value.best_early_stop_pinball for value in bundle.artifacts])
    )
    calibration_not_worse = all(
        artifact.calibration_bce_after <= artifact.calibration_bce_before + 1e-12
        for artifact in bundle.artifacts
    )
    reasons: list[str] = []
    if mse_skill_count < 3:
        reasons.append("fewer_than_three_utility_horizons_beat_training_mean_mse")
    if utility_rank_count < 3:
        reasons.append("fewer_than_three_utility_horizons_have_positive_spearman")
    if log_loss_count < 6:
        reasons.append("fewer_than_six_side_horizons_beat_prevalence_log_loss")
    if auc_count < 6:
        reasons.append("fewer_than_six_side_horizons_have_auc_above_half")
    if minimum_utility_stability < 0.5:
        reasons.append("conditional_mean_seed_stability_below_0_5")
    if minimum_logit_stability < 0.5:
        reasons.append("action_logit_seed_stability_below_0_5")
    if pinball > MAXIMUM_EARLY_STOP_PINBALL:
        reasons.append("early_stop_pinball_degraded_more_than_two_percent")
    if not calibration_not_worse:
        reasons.append("temperature_scaling_worsened_calibration_bce")
    return {
        "candidate_id": bundle.candidate_id,
        "utility_horizons": utility_rows,
        "action_side_horizons": action_rows,
        "seed_stability": stability_rows,
        "gate": {
            "passed": not reasons,
            "reasons": reasons,
            "horizons_with_positive_mse_skill": mse_skill_count,
            "horizons_with_positive_utility_spearman": utility_rank_count,
            "side_horizons_with_positive_log_loss_skill": log_loss_count,
            "side_horizons_with_auc_above_half": auc_count,
            "minimum_pairwise_seed_conditional_mean_spearman": minimum_utility_stability,
            "minimum_pairwise_seed_action_logit_spearman": minimum_logit_stability,
            "median_best_early_stop_pinball": pinball,
            "maximum_permitted_early_stop_pinball": MAXIMUM_EARLY_STOP_PINBALL,
            "temperature_calibration_not_worse": calibration_not_worse,
        },
    }


def rank_ablation_gate(
    control: Mapping[str, object],
    ranked: Mapping[str, object],
) -> dict[str, object]:
    control_utility = control["utility_horizons"]
    ranked_utility = ranked["utility_horizons"]
    control_actions = control["action_side_horizons"]
    ranked_actions = ranked["action_side_horizons"]
    if not all(
        isinstance(value, Sequence)
        for value in (control_utility, ranked_utility, control_actions, ranked_actions)
    ):
        raise TypeError("Round 47 rank comparison payload is invalid")
    control_spearman = float(
        np.mean([float(row["spearman"]) for row in control_utility])
    )
    ranked_spearman = float(np.mean([float(row["spearman"]) for row in ranked_utility]))
    control_log_loss = float(
        np.mean([float(row["log_loss"]) for row in control_actions])
    )
    ranked_log_loss = float(np.mean([float(row["log_loss"]) for row in ranked_actions]))
    improvement = ranked_spearman - control_spearman
    degradation = ranked_log_loss / control_log_loss - 1.0
    reasons: list[str] = []
    if improvement < 0.005:
        reasons.append("average_utility_spearman_improvement_below_0_005")
    if degradation > 0.01:
        reasons.append("average_action_log_loss_degraded_more_than_one_percent")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "control_average_utility_spearman": control_spearman,
        "ranked_average_utility_spearman": ranked_spearman,
        "average_utility_spearman_improvement": improvement,
        "control_average_action_log_loss": control_log_loss,
        "ranked_average_action_log_loss": ranked_log_loss,
        "relative_action_log_loss_degradation": degradation,
    }


def select_utility_trades(
    dataset: DistributionalDataset,
    utility_bps: np.ndarray,
    bundle: UtilityForecastBundle,
) -> tuple[UtilityTrade, ...]:
    evaluation_indexes = np.flatnonzero(role_mask(dataset, "evaluation"))
    first = int(evaluation_indexes[0])
    last = int(evaluation_indexes[-1])
    trades: list[UtilityTrade] = []
    for symbol_index, symbol in enumerate(SYMBOLS):
        index = first
        while index <= last:
            choices: list[tuple[float, float, int, int, float]] = []
            for horizon_index, horizon in enumerate(HORIZONS):
                if index + horizon - 1 > last:
                    continue
                seed_utility = bundle.seed_utility_bps[
                    :, index, symbol_index, horizon_index
                ].astype(np.float64)
                for side, side_index in ((-1, 0), (1, 1)):
                    expected_net = side * seed_utility - ROUND_TRIP_COST_BPS
                    worst_expected = float(np.min(expected_net))
                    worst_probability = float(
                        np.min(
                            bundle.seed_action_probabilities[
                                :, index, symbol_index, horizon_index, side_index
                            ]
                        )
                    )
                    if (
                        worst_expected > 0.0
                        and worst_probability >= ACTION_PROBABILITY_FLOOR
                    ):
                        choices.append(
                            (
                                worst_expected,
                                worst_probability,
                                -horizon,
                                side,
                                float(np.median(seed_utility)),
                            )
                        )
            if not choices:
                index += 1
                continue
            worst_expected, worst_probability, negative_horizon, side, signed = max(
                choices
            )
            horizon = -negative_horizon
            horizon_index = HORIZONS.index(horizon)
            realized = (
                side * float(utility_bps[index, symbol_index, horizon_index])
                - ROUND_TRIP_COST_BPS
            )
            replay_value = (
                side
                * float(
                    np.sum(
                        dataset.hourly_return_bps[
                            index : index + horizon, symbol_index
                        ].astype(np.float64)
                    )
                )
                - ROUND_TRIP_COST_BPS
            )
            if not math.isclose(realized, replay_value, rel_tol=0.0, abs_tol=1e-4):
                raise RuntimeError("Round 47 utility target differs from replay P&L")
            identity = hashlib.sha256(
                f"{bundle.candidate_id}|{symbol}|{int(dataset.timestamps_ms[index])}|{side}|{horizon}".encode(
                    "ascii"
                )
            ).hexdigest()
            trades.append(
                UtilityTrade(
                    trade_id=identity,
                    candidate_id=bundle.candidate_id,
                    symbol=symbol,
                    symbol_index=symbol_index,
                    decision_index=index,
                    decision_time_ms=int(dataset.timestamps_ms[index]),
                    side=side,
                    horizon_hours=horizon,
                    predicted_signed_utility_bps=signed,
                    worst_seed_expected_net_bps=worst_expected,
                    worst_seed_probability=worst_probability,
                )
            )
            index += horizon
    trades.sort(key=lambda item: (item.decision_time_ms, item.symbol_index))
    return tuple(trades)


__all__ = [
    "utility_preflight",
    "ACTION_PROBABILITY_FLOOR",
    "CANDIDATES",
    "CostAwareUtilityTCN",
    "RANK_WEIGHT",
    "ROUND",
    "SEEDS",
    "UtilityForecastBundle",
    "UtilityScaler",
    "UtilityTCNArtifact",
    "UtilityTrade",
    "action_labels",
    "additive_forward_utility_bps",
    "binary_logistic_loss",
    "cpu_utility_preflight",
    "directml_utility_preflight",
    "fit_utility_scaler",
    "multitask_objective",
    "pairwise_utility_rank_loss",
    "rank_ablation_gate",
    "select_utility_trades",
    "train_utility_candidates",
    "utility_action_diagnostics",
]
