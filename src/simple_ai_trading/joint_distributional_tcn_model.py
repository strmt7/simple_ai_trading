"""Joint cross-asset distributional TCN and SAM ablation for Round 45."""

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

from .cross_asset_cost_data import SYMBOLS
from .distributional_tcn_model import (
    BASE_ONE_WAY_COST_BPS,
    DILATIONS,
    DistributionalDataset,
    ExplicitAdamW,
    FeatureScaler,
    HIDDEN_CHANNELS,
    HORIZONS,
    KERNEL_SIZE,
    PlannedReplay,
    QUANTILES,
    RECEPTIVE_FIELD,
    TargetScaler,
    fit_feature_scaler,
    fit_target_scaler,
    role_mask,
)


ROUND = 45
CANDIDATES = ("joint_adamw", "joint_sam")
SEEDS = (4501, 4502, 4503)
SAM_RHO = 0.05
WINDOW_HOURS = 384
SUPERVISED_HOURS = WINDOW_HOURS - RECEPTIVE_FIELD + 1
WINDOW_STRIDE_HOURS = 24
BATCH_SIZE = 32
MAXIMUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 12
MINIMUM_IMPROVEMENT = 1e-5
SLEEVE_FRACTION = 1.0 / len(SYMBOLS)
BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_BLOCK_HOURS = 168
FAMILYWISE_LOWER_QUANTILE = 0.0125
ProgressCallback = Callable[[str, Mapping[str, object]], None]


@dataclass(frozen=True)
class JointTCNArtifact:
    candidate_id: str
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
class JointForecastBundle:
    candidate_id: str
    seed_predictions_bps: np.ndarray
    ensemble_predictions_bps: np.ndarray
    artifacts: tuple[JointTCNArtifact, ...]
    feature_scaler: FeatureScaler
    target_scaler: TargetScaler
    backend_kind: str
    backend_device: str


@dataclass(frozen=True)
class ConsensusTrade:
    trade_id: str
    candidate_id: str
    symbol: str
    symbol_index: int
    decision_index: int
    decision_time_ms: int
    side: int
    horizon_hours: int
    worst_seed_median_bps: float
    expected_after_cost_bps: float

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class JointCausalResidualBlock(nn.Module):
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


class JointDistributionalTCN(nn.Module):
    """One causal encoder that predicts all three symbols jointly."""

    def __init__(self, features_per_symbol: int = 71, dropout: float = 0.1) -> None:
        super().__init__()
        self.features_per_symbol = features_per_symbol
        self.input_channels = features_per_symbol * len(SYMBOLS)
        self.projection = nn.Conv1d(self.input_channels, HIDDEN_CHANNELS, kernel_size=1)
        self.blocks = nn.ModuleList(
            JointCausalResidualBlock(HIDDEN_CHANNELS, dilation, dropout)
            for dilation in DILATIONS
        )
        self.head = nn.Conv1d(
            HIDDEN_CHANNELS,
            len(SYMBOLS) * len(HORIZONS) * len(QUANTILES),
            kernel_size=1,
        )
        with torch.no_grad():
            if self.head.bias is not None:
                reshaped = self.head.bias.reshape(
                    len(SYMBOLS), len(HORIZONS), len(QUANTILES)
                )
                reshaped[:, :, 1:] = -2.0
                reshaped[:, :, 0] = 0.0

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] != self.input_channels:
            raise ValueError("Round 45 joint TCN input dimensions are invalid")
        encoded = F.gelu(self.projection(values))
        for block in self.blocks:
            encoded = block(encoded)
        raw = self.head(encoded).reshape(
            values.shape[0],
            len(SYMBOLS),
            len(HORIZONS),
            len(QUANTILES),
            values.shape[-1],
        )
        median = raw[:, :, :, 0, :]
        lower_near = F.softplus(raw[:, :, :, 1, :])
        lower_far = F.softplus(raw[:, :, :, 2, :])
        upper_near = F.softplus(raw[:, :, :, 3, :])
        upper_far = F.softplus(raw[:, :, :, 4, :])
        q25 = median - lower_near
        q10 = q25 - lower_far
        q75 = median + upper_near
        q90 = q75 + upper_far
        return torch.stack((q10, q25, median, q75, q90), dim=3)


def joint_pinball_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    if predictions.ndim != 5 or targets.ndim != 4:
        raise ValueError("Round 45 pinball tensors have invalid dimensions")
    flattened_predictions = predictions.reshape(
        predictions.shape[0],
        len(SYMBOLS) * len(HORIZONS),
        len(QUANTILES),
        predictions.shape[-1],
    )
    flattened_targets = targets.reshape(
        targets.shape[0],
        len(SYMBOLS) * len(HORIZONS),
        targets.shape[-1],
    )
    quantiles = torch.tensor(
        QUANTILES,
        dtype=predictions.dtype,
        device=predictions.device,
    ).reshape(1, 1, -1, 1)
    errors = flattened_targets.unsqueeze(2) - flattened_predictions
    return torch.maximum(quantiles * errors, (quantiles - 1.0) * errors).mean()


def _gradient_norm(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    squares = [
        torch.sum(parameter.grad * parameter.grad)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not squares:
        raise RuntimeError("Round 45 SAM has no gradients")
    return torch.sqrt(torch.stack(squares).sum())


def sam_training_step(
    model: JointDistributionalTCN,
    optimizer: ExplicitAdamW,
    values: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float]:
    """Apply one non-adaptive SAM step over the explicit AdamW base optimizer."""

    optimizer.zero_grad(set_to_none=True)
    first_predictions = model(values)[..., -targets.shape[-1] :]
    first_loss = joint_pinball_loss(first_predictions, targets)
    first_loss.backward()
    gradient_norm = _gradient_norm(optimizer.parameters)
    gradient_norm_value = float(gradient_norm.detach().cpu().item())
    if not math.isfinite(gradient_norm_value) or gradient_norm_value <= 0.0:
        raise RuntimeError("Round 45 SAM gradient norm is invalid")
    scale = SAM_RHO / (gradient_norm + 1e-12)
    perturbations: list[torch.Tensor | None] = []
    with torch.no_grad():
        for parameter in optimizer.parameters:
            if parameter.grad is None:
                perturbations.append(None)
                continue
            perturbation = parameter.grad * scale
            parameter.add_(perturbation)
            perturbations.append(perturbation)
    optimizer.zero_grad(set_to_none=True)
    second_predictions = model(values)[..., -targets.shape[-1] :]
    second_loss = joint_pinball_loss(second_predictions, targets)
    second_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
    with torch.no_grad():
        for parameter, perturbation in zip(
            optimizer.parameters, perturbations, strict=True
        ):
            if perturbation is not None:
                parameter.sub_(perturbation)
    optimizer.step()
    return (
        float(first_loss.detach().cpu().item()),
        float(second_loss.detach().cpu().item()),
    )


def adamw_training_step(
    model: JointDistributionalTCN,
    optimizer: ExplicitAdamW,
    values: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[float, float]:
    optimizer.zero_grad(set_to_none=True)
    loss = joint_pinball_loss(model(values), targets)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
    optimizer.step()
    value = float(loss.detach().cpu().item())
    return value, value


def directml_joint_preflight() -> tuple[object, dict[str, object]]:
    """Require warning-free DirectML backward updates for AdamW and SAM."""

    try:
        import torch_directml  # type: ignore
    except ImportError as exc:  # pragma: no cover - host dependent
        raise RuntimeError("Round 45 DirectML is unavailable") from exc
    device = torch_directml.device()
    generator = np.random.default_rng(SEEDS[0])
    values_numpy = generator.normal(size=(4, 213, 160)).astype(np.float32)
    targets_numpy = generator.normal(size=(4, len(SYMBOLS), len(HORIZONS), 160)).astype(
        np.float32
    )
    results: list[dict[str, object]] = []
    all_messages: list[str] = []
    for candidate_id in CANDIDATES:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = JointDistributionalTCN().to(device)
            optimizer = ExplicitAdamW(
                tuple(model.parameters()),
                learning_rate=1e-3,
                weight_decay=1e-4,
            )
            values = torch.from_numpy(values_numpy).to(device)
            targets = torch.from_numpy(targets_numpy).to(device)
            if candidate_id == "joint_sam":
                first_loss, update_loss = sam_training_step(
                    model, optimizer, values, targets
                )
            else:
                first_loss, update_loss = adamw_training_step(
                    model, optimizer, values, targets
                )
        messages = [str(item.message) for item in caught]
        all_messages.extend(messages)
        results.append(
            {
                "candidate_id": candidate_id,
                "first_loss": first_loss,
                "update_loss": update_loss,
                "warning_count": len(messages),
            }
        )
    fallback = [
        item
        for item in all_messages
        if "not currently supported on the DML backend" in item
        or "fall back to run on the CPU" in item
    ]
    if fallback or any(
        not math.isfinite(float(item["update_loss"])) for item in results
    ):
        raise RuntimeError(f"Round 45 DirectML preflight failed: {fallback}")
    return device, {
        "backend_kind": "directml",
        "backend_device": str(device),
        "torch_version": str(torch.__version__),
        "torch_directml_version": str(
            getattr(torch_directml, "__version__", "unknown")
        ),
        "candidate_updates": results,
        "warning_count": len(all_messages),
        "cpu_fallback_warning_count": 0,
    }


def cpu_joint_device() -> tuple[object, dict[str, object]]:
    device = torch.device("cpu")
    return device, {
        "backend_kind": "cpu",
        "backend_device": str(device),
        "torch_version": str(torch.__version__),
        "torch_directml_version": None,
        "candidate_updates": [],
        "warning_count": 0,
        "cpu_fallback_warning_count": 0,
    }


def _training_windows(mask: np.ndarray) -> np.ndarray:
    indexes = np.flatnonzero(mask)
    if indexes.size < WINDOW_HOURS or not np.all(np.diff(indexes) == 1):
        raise ValueError("Round 45 training role is not a contiguous span")
    first = int(indexes[0])
    last_start = int(indexes[-1]) - WINDOW_HOURS + 1
    starts = np.arange(first, last_start + 1, WINDOW_STRIDE_HOURS, dtype=np.int64)
    if not starts.size:
        raise ValueError("Round 45 has no training windows")
    return starts


def _predict_all(
    model: JointDistributionalTCN,
    normalized_features: np.ndarray,
    target_scaler: TargetScaler,
    device: object,
) -> np.ndarray:
    model.eval()
    flattened = normalized_features.reshape(normalized_features.shape[0], -1).T
    with torch.no_grad():
        tensor = torch.from_numpy(np.ascontiguousarray(flattened[None, ...])).to(device)
        normalized = model(tensor)[0].detach().cpu().numpy()
    normalized = normalized.transpose(3, 0, 1, 2)
    predictions = target_scaler.denormalize(normalized).astype(np.float32)
    expected = (
        normalized_features.shape[0],
        len(SYMBOLS),
        len(HORIZONS),
        len(QUANTILES),
    )
    if predictions.shape != expected or not np.isfinite(predictions).all():
        raise ValueError("Round 45 predictions are invalid")
    return predictions


def _numpy_pinball(actual: np.ndarray, predictions: np.ndarray) -> float:
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


def train_joint_tcn_seed(
    dataset: DistributionalDataset,
    normalized_features: np.ndarray,
    target_scaler: TargetScaler,
    *,
    candidate_id: str,
    seed: int,
    artifact_path: Path,
    device: object,
    backend_kind: str,
    backend_device: str,
    progress: ProgressCallback | None = None,
) -> tuple[JointDistributionalTCN, JointTCNArtifact, np.ndarray]:
    if candidate_id not in CANDIDATES or seed not in SEEDS:
        raise ValueError("Round 45 candidate or seed is not frozen")
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    model = JointDistributionalTCN(
        features_per_symbol=normalized_features.shape[-1]
    ).to(device)
    optimizer = ExplicitAdamW(
        tuple(model.parameters()),
        learning_rate=1e-3,
        weight_decay=1e-4,
    )
    training_mask = role_mask(dataset, "training")
    validation_mask = role_mask(dataset, "early_stop")
    starts = _training_windows(training_mask)
    normalized_targets = target_scaler.normalize(dataset.forward_return_bps)
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    warning_messages: list[str] = []
    epochs_run = 0
    for epoch in range(1, MAXIMUM_EPOCHS + 1):
        epochs_run = epoch
        order = generator.permutation(starts.size)
        model.train()
        batch_losses: list[float] = []
        first_losses: list[float] = []
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for offset in range(0, order.size, BATCH_SIZE):
                selected = starts[order[offset : offset + BATCH_SIZE]]
                batch_features = np.stack(
                    [
                        normalized_features[start : start + WINDOW_HOURS]
                        .reshape(WINDOW_HOURS, -1)
                        .T
                        for start in selected
                    ]
                ).astype(np.float32, copy=False)
                batch_targets = np.stack(
                    [
                        normalized_targets[start : start + WINDOW_HOURS].transpose(
                            1, 2, 0
                        )
                        for start in selected
                    ]
                ).astype(np.float32, copy=False)
                values = torch.from_numpy(np.ascontiguousarray(batch_features)).to(
                    device
                )
                targets = torch.from_numpy(np.ascontiguousarray(batch_targets)).to(
                    device
                )[..., -SUPERVISED_HOURS:]
                if candidate_id == "joint_sam":
                    first_loss, batch_loss = sam_training_step(
                        model,
                        optimizer,
                        values,
                        targets,
                    )
                else:
                    predictions = model(values)[..., -SUPERVISED_HOURS:]
                    optimizer.zero_grad(set_to_none=True)
                    loss = joint_pinball_loss(predictions, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 1.0, foreach=False
                    )
                    optimizer.step()
                    first_loss = float(loss.detach().cpu().item())
                    batch_loss = first_loss
                first_losses.append(first_loss)
                batch_losses.append(batch_loss)
            full_predictions = _predict_all(
                model, normalized_features, target_scaler, device
            )
            validation_loss = _role_pinball(
                dataset, full_predictions, validation_mask, target_scaler
            )
        warning_messages.extend(str(item.message) for item in caught)
        training_loss = float(np.mean(batch_losses))
        first_training_loss = float(np.mean(first_losses))
        if not all(
            math.isfinite(value)
            for value in (training_loss, first_training_loss, validation_loss)
        ):
            raise RuntimeError("Round 45 training produced a nonfinite loss")
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
                "round45_joint_tcn_epoch",
                {
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "epoch": epoch,
                    "first_training_pinball": first_training_loss,
                    "update_training_pinball": training_loss,
                    "early_stop_pinball": validation_loss,
                    "best_early_stop_pinball": best_loss,
                    "best_epoch": best_epoch,
                    "stale_epochs": stale_epochs,
                },
            )
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("Round 45 training did not produce a best state")
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
        raise RuntimeError(f"Round 45 training used CPU fallback: {fallback}")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, artifact_path)
    reloaded = JointDistributionalTCN(
        features_per_symbol=normalized_features.shape[-1]
    ).to(device)
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
        raise RuntimeError(f"Round 45 reload used CPU fallback: {reload_fallback}")
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
        raise RuntimeError(f"Round 45 artifact reload error is {reload_error}")
    artifact = JointTCNArtifact(
        candidate_id=candidate_id,
        seed=seed,
        epochs=epochs_run,
        best_epoch=best_epoch,
        best_early_stop_pinball=best_loss,
        parameter_count=sum(value.numel() for value in model.parameters()),
        backend_kind=backend_kind,
        backend_device=backend_device,
        path=str(artifact_path),
        bytes=artifact_path.stat().st_size,
        sha256=_file_sha256(artifact_path),
        reload_max_abs_prediction_error=reload_error,
        warning_count=len(warning_messages),
    )
    return model, artifact, best_predictions


def train_joint_candidate(
    dataset: DistributionalDataset,
    *,
    candidate_id: str,
    model_dir: Path,
    device: object,
    preflight: Mapping[str, object],
    progress: ProgressCallback | None = None,
) -> JointForecastBundle:
    if candidate_id not in CANDIDATES:
        raise ValueError(f"Round 45 candidate is invalid: {candidate_id}")
    training_mask = role_mask(dataset, "training")
    feature_scaler = fit_feature_scaler(dataset, training_mask)
    target_scaler = fit_target_scaler(dataset, training_mask)
    normalized_features = feature_scaler.transform(dataset.features)
    seed_predictions: list[np.ndarray] = []
    artifacts: list[JointTCNArtifact] = []
    for seed in SEEDS:
        if progress is not None:
            progress(
                "round45_joint_tcn_seed",
                {"candidate_id": candidate_id, "seed": seed, "status": "started"},
            )
        _model, artifact, predictions = train_joint_tcn_seed(
            dataset,
            normalized_features,
            target_scaler,
            candidate_id=candidate_id,
            seed=seed,
            artifact_path=model_dir / f"round45_{candidate_id}_seed_{seed}.pt",
            device=device,
            backend_kind=str(preflight["backend_kind"]),
            backend_device=str(preflight["backend_device"]),
            progress=progress,
        )
        artifacts.append(artifact)
        seed_predictions.append(predictions)
        if progress is not None:
            progress(
                "round45_joint_tcn_seed",
                {
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "status": "complete",
                    "best_epoch": artifact.best_epoch,
                    "artifact_sha256": artifact.sha256,
                },
            )
    stacked = np.stack(seed_predictions).astype(np.float32)
    ensemble = np.median(stacked, axis=0).astype(np.float32)
    if np.any(np.diff(ensemble, axis=-1) < -1e-7):
        raise RuntimeError("Round 45 ensemble quantiles crossed")
    return JointForecastBundle(
        candidate_id=candidate_id,
        seed_predictions_bps=stacked,
        ensemble_predictions_bps=ensemble,
        artifacts=tuple(artifacts),
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        backend_kind=str(preflight["backend_kind"]),
        backend_device=str(preflight["backend_device"]),
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


def joint_forecast_diagnostics(
    dataset: DistributionalDataset,
    bundle: JointForecastBundle,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    bundle_seeds = tuple(artifact.seed for artifact in bundle.artifacts)
    if len(bundle_seeds) != bundle.seed_predictions_bps.shape[0]:
        raise ValueError("Forecast artifacts and seed predictions differ in count")
    training_mask = role_mask(dataset, "training")
    evaluation_mask = role_mask(dataset, "evaluation")
    months = _month_labels(dataset.timestamps_ms)
    pooled_baseline = np.quantile(
        dataset.forward_return_bps[training_mask].reshape(-1, len(HORIZONS)),
        QUANTILES,
        axis=0,
    ).T
    symbol_baseline = np.quantile(
        dataset.forward_return_bps[training_mask],
        QUANTILES,
        axis=0,
    ).transpose(1, 2, 0)
    monthly_rows: list[dict[str, object]] = []
    pooled_horizons: list[dict[str, object]] = []
    symbol_horizons: list[dict[str, object]] = []
    evaluation_months = sorted(set(months[evaluation_mask]))
    for horizon_index, horizon in enumerate(HORIZONS):
        actual = dataset.forward_return_bps[evaluation_mask, :, horizon_index].astype(
            np.float64
        )
        prediction = bundle.ensemble_predictions_bps[
            evaluation_mask, :, horizon_index, :
        ].astype(np.float64)
        baseline = np.broadcast_to(pooled_baseline[horizon_index], prediction.shape)
        model_pinball = _numpy_pinball(actual, prediction)
        baseline_pinball = _numpy_pinball(actual, baseline)
        positive_months = 0
        for month in evaluation_months:
            month_mask = evaluation_mask & (months == month)
            actual_month = dataset.forward_return_bps[
                month_mask, :, horizon_index
            ].astype(np.float64)
            prediction_month = bundle.ensemble_predictions_bps[
                month_mask, :, horizon_index, :
            ].astype(np.float64)
            association = _finite_spearman(
                actual_month.ravel(), prediction_month[..., 2].ravel()
            )
            positive_months += int(association > 0.0)
            monthly_rows.append(
                _diagnostic_row(
                    candidate_id=bundle.candidate_id,
                    period=month,
                    symbol="ALL",
                    horizon=horizon,
                    actual=actual_month,
                    prediction=prediction_month,
                    baseline=np.broadcast_to(
                        pooled_baseline[horizon_index], prediction_month.shape
                    ),
                )
            )
            for symbol_index, symbol in enumerate(SYMBOLS):
                monthly_rows.append(
                    _diagnostic_row(
                        candidate_id=bundle.candidate_id,
                        period=month,
                        symbol=symbol,
                        horizon=horizon,
                        actual=actual_month[:, symbol_index],
                        prediction=prediction_month[:, symbol_index, :],
                        baseline=np.broadcast_to(
                            symbol_baseline[symbol_index, horizon_index],
                            prediction_month[:, symbol_index, :].shape,
                        ),
                    )
                )
        pooled_horizons.append(
            {
                "candidate_id": bundle.candidate_id,
                "symbol": "ALL",
                "horizon_hours": horizon,
                "model_pinball_loss_bps": model_pinball,
                "baseline_pinball_loss_bps": baseline_pinball,
                "pinball_skill": 1.0 - model_pinball / baseline_pinball,
                "pooled_median_spearman": _finite_spearman(
                    actual.ravel(), prediction[..., 2].ravel()
                ),
                "positive_monthly_median_spearman_count": positive_months,
                "evaluation_month_count": len(evaluation_months),
                "coverage_80": float(
                    np.mean(
                        (actual >= prediction[..., 0]) & (actual <= prediction[..., 4])
                    )
                ),
                "coverage_50": float(
                    np.mean(
                        (actual >= prediction[..., 1]) & (actual <= prediction[..., 3])
                    )
                ),
            }
        )
        for symbol_index, symbol in enumerate(SYMBOLS):
            symbol_actual = actual[:, symbol_index]
            symbol_prediction = prediction[:, symbol_index, :]
            symbol_reference = np.broadcast_to(
                symbol_baseline[symbol_index, horizon_index],
                symbol_prediction.shape,
            )
            symbol_horizons.append(
                {
                    "candidate_id": bundle.candidate_id,
                    "symbol": symbol,
                    "horizon_hours": horizon,
                    "model_pinball_loss_bps": _numpy_pinball(
                        symbol_actual, symbol_prediction
                    ),
                    "baseline_pinball_loss_bps": _numpy_pinball(
                        symbol_actual, symbol_reference
                    ),
                    "pinball_skill": 1.0
                    - _numpy_pinball(symbol_actual, symbol_prediction)
                    / _numpy_pinball(symbol_actual, symbol_reference),
                    "pooled_median_spearman": _finite_spearman(
                        symbol_actual, symbol_prediction[:, 2]
                    ),
                    "coverage_80": float(
                        np.mean(
                            (symbol_actual >= symbol_prediction[:, 0])
                            & (symbol_actual <= symbol_prediction[:, 4])
                        )
                    ),
                    "coverage_50": float(
                        np.mean(
                            (symbol_actual >= symbol_prediction[:, 1])
                            & (symbol_actual <= symbol_prediction[:, 3])
                        )
                    ),
                }
            )
    stability: list[dict[str, object]] = []
    minimum_correlation = 1.0
    for left in range(len(bundle_seeds)):
        for right in range(left + 1, len(bundle_seeds)):
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
                        "candidate_id": bundle.candidate_id,
                        "left_seed": bundle_seeds[left],
                        "right_seed": bundle_seeds[right],
                        "horizon_hours": horizon,
                        "median_prediction_spearman": correlation,
                    }
                )
    skill_count = sum(row["pinball_skill"] >= 0.01 for row in pooled_horizons)
    association_count = sum(
        row["pooled_median_spearman"] > 0.0
        and row["positive_monthly_median_spearman_count"] >= 5
        for row in pooled_horizons
    )
    coverage_passed = all(
        0.72 <= row["coverage_80"] <= 0.88 and 0.42 <= row["coverage_50"] <= 0.58
        for row in pooled_horizons
    )
    symbol_positive_counts = {
        symbol: sum(
            row["pooled_median_spearman"] > 0.0
            for row in symbol_horizons
            if row["symbol"] == symbol
        )
        for symbol in SYMBOLS
    }
    crossing_count = int(
        np.count_nonzero(
            np.diff(bundle.ensemble_predictions_bps[evaluation_mask], axis=-1) < -1e-7
        )
    )
    reasons: list[str] = []
    if skill_count < 3:
        reasons.append("fewer_than_three_horizons_beat_unconditional_pinball_by_1pct")
    if association_count < 3:
        reasons.append(
            "fewer_than_three_horizons_have_stable_positive_rank_association"
        )
    if not coverage_passed:
        reasons.append("central_interval_coverage_outside_frozen_bounds")
    if minimum_correlation < 0.5:
        reasons.append("seed_prediction_stability_below_0_5")
    if any(value < 2 for value in symbol_positive_counts.values()):
        reasons.append("one_or_more_symbols_have_fewer_than_two_positive_horizons")
    if crossing_count:
        reasons.append("quantile_crossing_detected")
    return (
        monthly_rows,
        stability,
        {
            "candidate_id": bundle.candidate_id,
            "horizons": pooled_horizons,
            "symbol_horizons": symbol_horizons,
            "gate": {
                "passed": not reasons,
                "reasons": reasons,
                "horizons_with_required_pinball_skill": skill_count,
                "horizons_with_required_rank_stability": association_count,
                "coverage_passed": coverage_passed,
                "minimum_pairwise_seed_median_prediction_spearman": minimum_correlation,
                "positive_symbol_horizon_counts": symbol_positive_counts,
                "quantile_crossing_count": crossing_count,
            },
        },
    )


def _diagnostic_row(
    *,
    candidate_id: str,
    period: str,
    symbol: str,
    horizon: int,
    actual: np.ndarray,
    prediction: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "role": "evaluation",
        "period": period,
        "symbol": symbol,
        "horizon_hours": horizon,
        "rows": int(actual.size),
        "pinball_loss": _numpy_pinball(actual, prediction),
        "baseline_pinball_loss": _numpy_pinball(actual, baseline),
        "median_spearman": _finite_spearman(actual.ravel(), prediction[..., 2].ravel()),
        "coverage_80": float(
            np.mean((actual >= prediction[..., 0]) & (actual <= prediction[..., 4]))
        ),
        "coverage_50": float(
            np.mean((actual >= prediction[..., 1]) & (actual <= prediction[..., 3]))
        ),
    }


def select_consensus_trades(
    dataset: DistributionalDataset,
    bundle: JointForecastBundle,
) -> tuple[ConsensusTrade, ...]:
    evaluation_mask = role_mask(dataset, "evaluation")
    evaluation_indexes = np.flatnonzero(evaluation_mask)
    first = int(evaluation_indexes[0])
    last = int(evaluation_indexes[-1])
    round_trip_cost = 2.0 * BASE_ONE_WAY_COST_BPS
    trades: list[ConsensusTrade] = []
    for symbol_index, symbol in enumerate(SYMBOLS):
        index = first
        while index <= last:
            choices: list[tuple[float, int, int, float]] = []
            for horizon_index, horizon in enumerate(HORIZONS):
                if index + horizon - 1 > last:
                    continue
                seed_medians = bundle.seed_predictions_bps[
                    :, index, symbol_index, horizon_index, 2
                ].astype(np.float64)
                long_consensus = float(np.min(seed_medians))
                short_consensus = -float(np.max(seed_medians))
                choices.append(
                    (
                        long_consensus - round_trip_cost,
                        -horizon,
                        1,
                        long_consensus,
                    )
                )
                choices.append(
                    (
                        short_consensus - round_trip_cost,
                        -horizon,
                        -1,
                        short_consensus,
                    )
                )
            if not choices:
                break
            expected_after_cost, negative_horizon, side, consensus = max(choices)
            horizon = -negative_horizon
            if expected_after_cost <= 0.0:
                index += 1
                continue
            identity = hashlib.sha256(
                f"{bundle.candidate_id}|{symbol}|{int(dataset.timestamps_ms[index])}|{side}|{horizon}".encode(
                    "ascii"
                )
            ).hexdigest()
            trades.append(
                ConsensusTrade(
                    trade_id=identity,
                    candidate_id=bundle.candidate_id,
                    symbol=symbol,
                    symbol_index=symbol_index,
                    decision_index=index,
                    decision_time_ms=int(dataset.timestamps_ms[index]),
                    side=side,
                    horizon_hours=horizon,
                    worst_seed_median_bps=consensus,
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


def replay_consensus_trades(
    dataset: DistributionalDataset,
    trades: Sequence[ConsensusTrade],
    *,
    candidate_id: str,
    scenario: str,
    one_way_cost_bps: float,
    bootstrap_seed: int | None = None,
) -> PlannedReplay:
    if any(trade.candidate_id != candidate_id for trade in trades):
        raise ValueError("Round 45 replay candidate differs from its trade ledger")
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
            raise ValueError("Round 45 trade extends beyond evaluation")
        local = slice(start - first, stop - first)
        symbol = trade.symbol_index
        if np.any(occupied[local, symbol]):
            raise ValueError("Round 45 trade ledger overlaps within a symbol")
        occupied[local, symbol] = True
        positions[local, symbol] = trade.side
        raw = trade.side * dataset.hourly_return_bps[start:stop, symbol].astype(
            np.float64
        )
        raw[0] -= one_way_cost_bps
        raw[-1] -= one_way_cost_bps
        symbol_return_bps[local, symbol] = raw * SLEEVE_FRACTION
        trade_net_bps.append(float(np.sum(raw)))
    portfolio_bps = np.sum(symbol_return_bps, axis=1)
    returns = portfolio_bps / 10_000.0
    equity = np.cumprod(1.0 + returns)
    if not np.isfinite(equity).all() or np.any(equity <= 0.0):
        raise ValueError("Round 45 replay produced invalid equity")
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
    active_days = len(set(days[np.any(positions != 0, axis=1)].tolist()))
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
        "candidate_id": candidate_id,
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
            seed=(
                bootstrap_seed
                if bootstrap_seed is not None
                else SEEDS[0]
                + CANDIDATES.index(candidate_id) * 1_000
                + (0 if scenario == "base" else 100)
            ),
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


def joint_economic_gate(
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
    if metrics["trades"] < 180:
        reasons.append("fewer_than_one_hundred_eighty_closed_trades")
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


def optimizer_ablation_gate(
    adam_bundle: JointForecastBundle,
    adam_diagnostics: Mapping[str, object],
    sam_bundle: JointForecastBundle,
    sam_diagnostics: Mapping[str, object],
) -> dict[str, object]:
    adam_validation = float(
        np.median(
            [artifact.best_early_stop_pinball for artifact in adam_bundle.artifacts]
        )
    )
    sam_validation = float(
        np.median(
            [artifact.best_early_stop_pinball for artifact in sam_bundle.artifacts]
        )
    )
    relative_degradation = sam_validation / adam_validation - 1.0
    adam_stability = float(
        adam_diagnostics["gate"]["minimum_pairwise_seed_median_prediction_spearman"]
    )
    sam_stability = float(
        sam_diagnostics["gate"]["minimum_pairwise_seed_median_prediction_spearman"]
    )
    reasons: list[str] = []
    if not bool(sam_diagnostics["gate"]["passed"]):
        reasons.append("joint_sam_forecast_gate_failed")
    if relative_degradation > 0.01:
        reasons.append("sam_median_early_stop_pinball_degraded_more_than_one_percent")
    if sam_stability < adam_stability:
        reasons.append("sam_minimum_seed_stability_did_not_improve")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "adamw_median_best_early_stop_pinball": adam_validation,
        "sam_median_best_early_stop_pinball": sam_validation,
        "sam_relative_early_stop_pinball_degradation": relative_degradation,
        "adamw_minimum_seed_stability": adam_stability,
        "sam_minimum_seed_stability": sam_stability,
        "sam_seed_stability_delta": sam_stability - adam_stability,
    }


__all__ = [
    "BATCH_SIZE",
    "BOOTSTRAP_BLOCK_HOURS",
    "BOOTSTRAP_SAMPLES",
    "CANDIDATES",
    "ConsensusTrade",
    "FAMILYWISE_LOWER_QUANTILE",
    "JointDistributionalTCN",
    "JointForecastBundle",
    "JointTCNArtifact",
    "SAM_RHO",
    "SEEDS",
    "adamw_training_step",
    "cpu_joint_device",
    "directml_joint_preflight",
    "joint_economic_gate",
    "joint_forecast_diagnostics",
    "joint_pinball_loss",
    "optimizer_ablation_gate",
    "replay_consensus_trades",
    "sam_training_step",
    "select_consensus_trades",
    "train_joint_candidate",
    "train_joint_tcn_seed",
]
