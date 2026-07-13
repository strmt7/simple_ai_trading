"""Stability-regularized causal distributional TCN candidates for Round 46."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence
import warnings

import numpy as np
import torch

from .cross_asset_cost_data import SYMBOLS
from .distributional_tcn_model import (
    DistributionalDataset,
    DistributionalTCN,
    ExplicitAdamW,
    FeatureScaler,
    HORIZONS,
    QUANTILES,
    RECEPTIVE_FIELD,
    TargetScaler,
    fit_feature_scaler,
    fit_target_scaler,
    role_mask,
)


ROUND = 46
CANDIDATES = ("wavebound_ema", "mutual_median_consistency")
SEEDS = (4601, 4602, 4603)
WINDOW_HOURS = 384
SUPERVISED_HOURS = WINDOW_HOURS - RECEPTIVE_FIELD + 1
WINDOW_STRIDE_HOURS = 24
BATCH_SIZE = 64
MAXIMUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
MINIMUM_IMPROVEMENT = 1e-5
EMA_DECAY = 0.99
WAVEBOUND_WARMUP_UPDATES = 100
WAVEBOUND_EPSILON = 0.001
CONSISTENCY_WEIGHT = 0.05
PREDECESSOR_MINIMUM_STABILITY = 0.45191081264635347
MINIMUM_STABILITY = 0.5
MINIMUM_STABILITY_IMPROVEMENT = 0.048089187353646534
PREDECESSOR_MEDIAN_EARLY_STOP_PINBALL = 0.36850768359123126
MAXIMUM_MEDIAN_EARLY_STOP_PINBALL = 0.3758778372630559
ProgressCallback = Callable[[str, Mapping[str, object]], None]


@dataclass(frozen=True)
class StabilityTCNArtifact:
    candidate_id: str
    seed: int
    epochs: int
    best_epoch: int
    best_early_stop_pinball: float
    optimizer_updates: int
    evaluation_model: str
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
class StabilityForecastBundle:
    candidate_id: str
    seed_predictions_bps: np.ndarray
    ensemble_predictions_bps: np.ndarray
    artifacts: tuple[StabilityTCNArtifact, ...]
    feature_scaler: FeatureScaler
    target_scaler: TargetScaler
    backend_kind: str
    backend_device: str
    training_history: tuple[Mapping[str, object], ...]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def pinball_components(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Return mean pinball loss for each horizon and quantile."""

    if predictions.ndim != 4 or targets.ndim != 3:
        raise ValueError("Round 46 pinball tensors have invalid dimensions")
    if (
        predictions.shape[0] != targets.shape[0]
        or predictions.shape[1] != len(HORIZONS)
        or predictions.shape[2] != len(QUANTILES)
        or predictions.shape[3] != targets.shape[2]
        or targets.shape[1] != len(HORIZONS)
    ):
        raise ValueError("Round 46 pinball tensors have incompatible dimensions")
    quantiles = torch.tensor(
        QUANTILES,
        dtype=predictions.dtype,
        device=predictions.device,
    ).reshape(1, 1, -1, 1)
    errors = targets.unsqueeze(2) - predictions
    losses = torch.maximum(quantiles * errors, (quantiles - 1.0) * errors)
    return losses.mean(dim=(0, 3))


def wavebound_pinball_loss(
    source_predictions: torch.Tensor,
    target_predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    epsilon: float = WAVEBOUND_EPSILON,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply WaveBound's dynamic lower bound to horizon-quantile losses."""

    if epsilon < 0.0:
        raise ValueError("WaveBound epsilon must be nonnegative")
    source = pinball_components(source_predictions, targets)
    target = pinball_components(target_predictions, targets).detach()
    bounded = torch.abs(source - target + epsilon) + target - epsilon
    return bounded.mean(), source.mean(), target.mean()


@torch.no_grad()
def update_ema_target(
    target: DistributionalTCN,
    source: DistributionalTCN,
    *,
    decay: float = EMA_DECAY,
) -> None:
    if not 0.0 <= decay < 1.0:
        raise ValueError("EMA decay must be in [0, 1)")
    for target_value, source_value in zip(
        target.parameters(), source.parameters(), strict=True
    ):
        target_value.mul_(decay).add_(source_value, alpha=1.0 - decay)


def standardized_median_consistency(
    peer_predictions: Sequence[torch.Tensor],
    *,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Measure pairwise agreement of standardized median forecasts."""

    if len(peer_predictions) != len(SEEDS):
        raise ValueError("Round 46 consistency requires exactly three peers")
    if epsilon <= 0.0:
        raise ValueError("Consistency epsilon must be positive")
    shape = peer_predictions[0].shape
    if len(shape) != 4 or shape[1] != len(HORIZONS) or shape[2] != len(QUANTILES):
        raise ValueError("Round 46 peer prediction dimensions are invalid")
    standardized: list[torch.Tensor] = []
    for predictions in peer_predictions:
        if predictions.shape != shape:
            raise ValueError("Round 46 peer prediction shapes differ")
        median = predictions[:, :, 2, :]
        center = median.mean(dim=(0, 2), keepdim=True)
        variance = ((median - center) ** 2).mean(dim=(0, 2), keepdim=True)
        standardized.append((median - center) / torch.sqrt(variance + epsilon))
    pair_horizon: list[torch.Tensor] = []
    for left in range(len(standardized)):
        for right in range(left + 1, len(standardized)):
            pair_horizon.append(
                0.5 * ((standardized[left] - standardized[right]) ** 2).mean(dim=(0, 2))
            )
    matrix = torch.stack(pair_horizon)
    return matrix.mean(), matrix


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
    torch.manual_seed(SEEDS[0])
    generator = np.random.default_rng(SEEDS[0])
    values = torch.from_numpy(
        generator.normal(size=(8, 71, 160)).astype(np.float32)
    ).to(device)
    targets = torch.from_numpy(
        generator.normal(size=(8, len(HORIZONS), 160)).astype(np.float32)
    ).to(device)
    messages: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        source = DistributionalTCN().to(device)
        target = DistributionalTCN()
        target.load_state_dict(
            {name: value.detach().cpu() for name, value in source.state_dict().items()}
        )
        target = target.to(device)
        target.requires_grad_(False)
        wave_optimizer = ExplicitAdamW(
            tuple(source.parameters()), learning_rate=1e-3, weight_decay=1e-4
        )
        source_before = next(source.parameters()).detach().cpu().clone()
        wave_optimizer.zero_grad(set_to_none=True)
        source_prediction = source(values)
        with torch.no_grad():
            target_prediction = target(values)
        wave_loss, source_loss, target_loss = wavebound_pinball_loss(
            source_prediction, target_prediction, targets
        )
        wave_loss.backward()
        torch.nn.utils.clip_grad_norm_(source.parameters(), 1.0, foreach=False)
        wave_optimizer.step()
        update_ema_target(target, source)
        wave_parameter_change = float(
            torch.max(
                torch.abs(next(source.parameters()).detach().cpu() - source_before)
            )
        )

        peers: list[DistributionalTCN] = []
        optimizers: list[ExplicitAdamW] = []
        for seed in SEEDS:
            torch.manual_seed(seed)
            peer = DistributionalTCN().to(device)
            peers.append(peer)
            optimizers.append(
                ExplicitAdamW(
                    tuple(peer.parameters()), learning_rate=1e-3, weight_decay=1e-4
                )
            )
        peer_before = next(peers[0].parameters()).detach().cpu().clone()
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        peer_predictions = [peer(values) for peer in peers]
        peer_supervised = torch.stack(
            [
                pinball_components(prediction, targets).mean()
                for prediction in peer_predictions
            ]
        )
        consistency, pair_horizon = standardized_median_consistency(peer_predictions)
        mutual_loss = peer_supervised.mean() + CONSISTENCY_WEIGHT * consistency
        mutual_loss.backward()
        for peer, optimizer in zip(peers, optimizers, strict=True):
            torch.nn.utils.clip_grad_norm_(peer.parameters(), 1.0, foreach=False)
            optimizer.step()
        mutual_parameter_change = float(
            torch.max(
                torch.abs(next(peers[0].parameters()).detach().cpu() - peer_before)
            )
        )
    messages.extend(str(item.message) for item in caught)
    fallback = _fallback_messages(messages)
    scalars = (
        float(wave_loss.detach().cpu()),
        float(source_loss.detach().cpu()),
        float(target_loss.detach().cpu()),
        float(mutual_loss.detach().cpu()),
        float(peer_supervised.mean().detach().cpu()),
        float(consistency.detach().cpu()),
        float(pair_horizon.detach().cpu().max()),
        wave_parameter_change,
        mutual_parameter_change,
    )
    if not all(math.isfinite(value) for value in scalars):
        raise RuntimeError("Round 46 compute preflight produced nonfinite evidence")
    if wave_parameter_change <= 0.0 or mutual_parameter_change <= 0.0 or fallback:
        raise RuntimeError(
            "Round 46 compute preflight failed: "
            f"wave_change={wave_parameter_change}, "
            f"mutual_change={mutual_parameter_change}, fallback={fallback}"
        )
    return {
        **metadata,
        "wavebound_objective": scalars[0],
        "wavebound_source_pinball": scalars[1],
        "wavebound_target_pinball": scalars[2],
        "wavebound_parameter_max_abs_change": wave_parameter_change,
        "mutual_objective": scalars[3],
        "mutual_supervised_pinball": scalars[4],
        "mutual_consistency": scalars[5],
        "mutual_pair_horizon_max": scalars[6],
        "mutual_parameter_max_abs_change": mutual_parameter_change,
        "warning_count": len(messages),
        "cpu_fallback_warning_count": 0,
    }


def directml_stability_preflight() -> tuple[object, dict[str, object]]:
    try:
        import torch_directml  # type: ignore
    except ImportError as exc:  # pragma: no cover - host dependent
        raise RuntimeError("Round 46 DirectML is unavailable") from exc
    device = torch_directml.device()
    evidence = _run_preflight(
        device,
        {
            "backend_kind": "directml",
            "backend_device": str(device),
            "torch_version": str(torch.__version__),
            "torch_directml_version": str(
                getattr(torch_directml, "__version__", "unknown")
            ),
        },
    )
    return device, evidence


def cpu_stability_preflight() -> tuple[object, dict[str, object]]:
    device = torch.device("cpu")
    evidence = _run_preflight(
        device,
        {
            "backend_kind": "cpu",
            "backend_device": str(device),
            "torch_version": str(torch.__version__),
            "torch_directml_version": None,
        },
    )
    return device, evidence


def _training_windows(mask: np.ndarray) -> np.ndarray:
    indexes = np.flatnonzero(mask)
    if indexes.size < WINDOW_HOURS or not np.all(np.diff(indexes) == 1):
        raise ValueError("Round 46 training role is not a contiguous span")
    first = int(indexes[0])
    last_start = int(indexes[-1]) - WINDOW_HOURS + 1
    starts = np.arange(first, last_start + 1, WINDOW_STRIDE_HOURS, dtype=np.int64)
    if starts.size == 0:
        raise ValueError("Round 46 has no training windows")
    return starts


def _training_pairs(dataset: DistributionalDataset) -> np.ndarray:
    starts = _training_windows(role_mask(dataset, "training"))
    return np.asarray(
        [(int(start), symbol) for start in starts for symbol in range(len(SYMBOLS))],
        dtype=np.int64,
    )


def _batch_arrays(
    normalized_features: np.ndarray,
    normalized_targets: np.ndarray,
    selected: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.stack(
        [
            normalized_features[start : start + WINDOW_HOURS, symbol, :].T
            for start, symbol in selected
        ]
    ).astype(np.float32, copy=False)
    targets = np.stack(
        [
            normalized_targets[start : start + WINDOW_HOURS, symbol, :].T
            for start, symbol in selected
        ]
    ).astype(np.float32, copy=False)
    return np.ascontiguousarray(features), np.ascontiguousarray(targets)


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
    expected = (
        normalized_features.shape[0],
        len(SYMBOLS),
        len(HORIZONS),
        len(QUANTILES),
    )
    if predictions.shape != expected or not np.isfinite(predictions).all():
        raise ValueError("Round 46 model predictions are invalid")
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


def _clone_state(model: DistributionalTCN) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _finalize_artifact(
    *,
    dataset: DistributionalDataset,
    candidate_id: str,
    seed: int,
    epochs: int,
    best_epoch: int,
    best_loss: float,
    optimizer_updates: int,
    evaluation_model: str,
    best_state: Mapping[str, torch.Tensor],
    best_predictions: np.ndarray,
    artifact_path: Path,
    normalized_features: np.ndarray,
    target_scaler: TargetScaler,
    device: object,
    backend_kind: str,
    backend_device: str,
    warning_messages: list[str],
) -> StabilityTCNArtifact:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(best_state), artifact_path)
    reloaded = DistributionalTCN(input_channels=normalized_features.shape[-1])
    reloaded.load_state_dict(
        torch.load(artifact_path, map_location="cpu", weights_only=True)
    )
    reloaded = reloaded.to(device)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        reload_predictions = _predict_all(
            reloaded, normalized_features, target_scaler, device
        )
    warning_messages.extend(str(item.message) for item in caught)
    fallback = _fallback_messages(warning_messages)
    if fallback:
        raise RuntimeError(f"Round 46 artifact reload used CPU fallback: {fallback}")
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
        raise RuntimeError(f"Round 46 artifact reload error is {reload_error}")
    return StabilityTCNArtifact(
        candidate_id=candidate_id,
        seed=seed,
        epochs=epochs,
        best_epoch=best_epoch,
        best_early_stop_pinball=best_loss,
        optimizer_updates=optimizer_updates,
        evaluation_model=evaluation_model,
        parameter_count=sum(value.numel() for value in reloaded.parameters()),
        backend_kind=backend_kind,
        backend_device=backend_device,
        path=str(artifact_path),
        bytes=artifact_path.stat().st_size,
        sha256=_file_sha256(artifact_path),
        reload_max_abs_prediction_error=reload_error,
        warning_count=len(warning_messages),
    )


def _wavebound_seed(
    dataset: DistributionalDataset,
    normalized_features: np.ndarray,
    target_scaler: TargetScaler,
    *,
    seed: int,
    artifact_path: Path,
    device: object,
    backend_kind: str,
    backend_device: str,
    progress: ProgressCallback | None,
) -> tuple[StabilityTCNArtifact, np.ndarray, list[dict[str, object]]]:
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    source = DistributionalTCN(input_channels=normalized_features.shape[-1])
    target = DistributionalTCN(input_channels=normalized_features.shape[-1])
    target.load_state_dict(source.state_dict())
    source = source.to(device)
    target = target.to(device)
    target.requires_grad_(False)
    optimizer = ExplicitAdamW(
        tuple(source.parameters()), learning_rate=1e-3, weight_decay=1e-4
    )
    pairs = _training_pairs(dataset)
    normalized_targets = target_scaler.normalize(dataset.forward_return_bps)
    validation_mask = role_mask(dataset, "early_stop")
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    optimizer_updates = 0
    epochs_run = 0
    history: list[dict[str, object]] = []
    warning_messages: list[str] = []
    for epoch in range(1, MAXIMUM_EPOCHS + 1):
        epochs_run = epoch
        order = generator.permutation(pairs.shape[0])
        source.train()
        target.eval()
        supervised_losses: list[float] = []
        target_losses: list[float] = []
        objective_losses: list[float] = []
        bounded_batches = 0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for offset in range(0, order.size, BATCH_SIZE):
                selected = pairs[order[offset : offset + BATCH_SIZE]]
                batch_features, batch_targets = _batch_arrays(
                    normalized_features, normalized_targets, selected
                )
                values = torch.from_numpy(batch_features).to(device)
                targets = torch.from_numpy(batch_targets).to(device)[
                    ..., -SUPERVISED_HOURS:
                ]
                optimizer.zero_grad(set_to_none=True)
                source_predictions = source(values)[..., -SUPERVISED_HOURS:]
                with torch.no_grad():
                    target_predictions = target(values)[..., -SUPERVISED_HOURS:]
                bounded, supervised, target_pinball = wavebound_pinball_loss(
                    source_predictions, target_predictions, targets
                )
                use_bound = optimizer_updates >= WAVEBOUND_WARMUP_UPDATES
                objective = bounded if use_bound else supervised
                objective.backward()
                torch.nn.utils.clip_grad_norm_(source.parameters(), 1.0, foreach=False)
                optimizer.step()
                update_ema_target(target, source)
                optimizer_updates += 1
                bounded_batches += int(use_bound)
                supervised_losses.append(float(supervised.detach().cpu()))
                target_losses.append(float(target_pinball.detach().cpu()))
                objective_losses.append(float(objective.detach().cpu()))
            source_predictions = _predict_all(
                source, normalized_features, target_scaler, device
            )
            target_predictions = _predict_all(
                target, normalized_features, target_scaler, device
            )
            source_validation = _role_pinball(
                dataset, source_predictions, validation_mask, target_scaler
            )
            target_validation = _role_pinball(
                dataset, target_predictions, validation_mask, target_scaler
            )
        warning_messages.extend(str(item.message) for item in caught)
        row = {
            "candidate_id": "wavebound_ema",
            "seed": seed,
            "epoch": epoch,
            "optimizer_updates": optimizer_updates,
            "bounded_batches": bounded_batches,
            "training_supervised_pinball": float(np.mean(supervised_losses)),
            "training_ema_target_pinball": float(np.mean(target_losses)),
            "training_objective": float(np.mean(objective_losses)),
            "source_early_stop_pinball": source_validation,
            "ema_early_stop_pinball": target_validation,
            "best_early_stop_pinball": min(best_loss, target_validation),
            "best_epoch": epoch if target_validation < best_loss else best_epoch,
            "stale_epochs": 0 if target_validation < best_loss else stale_epochs + 1,
        }
        scalars = [value for value in row.values() if isinstance(value, float)]
        if not all(math.isfinite(value) for value in scalars):
            raise RuntimeError("Round 46 WaveBound produced nonfinite diagnostics")
        improved = target_validation < best_loss - MINIMUM_IMPROVEMENT
        if improved:
            best_loss = target_validation
            best_epoch = epoch
            best_state = _clone_state(target)
            stale_epochs = 0
        else:
            stale_epochs += 1
        row["best_early_stop_pinball"] = best_loss
        row["best_epoch"] = best_epoch
        row["stale_epochs"] = stale_epochs
        history.append(row)
        if progress is not None:
            progress("round46_wavebound_epoch", row)
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            break
    if best_state is None:
        raise RuntimeError("Round 46 WaveBound did not produce a best state")
    target.load_state_dict(best_state)
    best_predictions = _predict_all(target, normalized_features, target_scaler, device)
    fallback = _fallback_messages(warning_messages)
    if fallback:
        raise RuntimeError(f"Round 46 WaveBound used CPU fallback: {fallback}")
    artifact = _finalize_artifact(
        dataset=dataset,
        candidate_id="wavebound_ema",
        seed=seed,
        epochs=epochs_run,
        best_epoch=best_epoch,
        best_loss=best_loss,
        optimizer_updates=optimizer_updates,
        evaluation_model="ema_target",
        best_state=best_state,
        best_predictions=best_predictions,
        artifact_path=artifact_path,
        normalized_features=normalized_features,
        target_scaler=target_scaler,
        device=device,
        backend_kind=backend_kind,
        backend_device=backend_device,
        warning_messages=warning_messages,
    )
    return artifact, best_predictions, history


def _train_wavebound(
    dataset: DistributionalDataset,
    normalized_features: np.ndarray,
    feature_scaler: FeatureScaler,
    target_scaler: TargetScaler,
    *,
    model_dir: Path,
    device: object,
    backend_kind: str,
    backend_device: str,
    progress: ProgressCallback | None,
) -> StabilityForecastBundle:
    artifacts: list[StabilityTCNArtifact] = []
    predictions: list[np.ndarray] = []
    history: list[Mapping[str, object]] = []
    for seed in SEEDS:
        if progress is not None:
            progress(
                "round46_candidate_seed",
                {"candidate_id": "wavebound_ema", "seed": seed, "status": "started"},
            )
        artifact, seed_predictions, seed_history = _wavebound_seed(
            dataset,
            normalized_features,
            target_scaler,
            seed=seed,
            artifact_path=model_dir / f"round46_wavebound_ema_seed_{seed}.pt",
            device=device,
            backend_kind=backend_kind,
            backend_device=backend_device,
            progress=progress,
        )
        artifacts.append(artifact)
        predictions.append(seed_predictions)
        history.extend(seed_history)
        if progress is not None:
            progress(
                "round46_candidate_seed",
                {
                    "candidate_id": "wavebound_ema",
                    "seed": seed,
                    "status": "complete",
                    "best_epoch": artifact.best_epoch,
                    "artifact_sha256": artifact.sha256,
                },
            )
    stacked = np.stack(predictions).astype(np.float32)
    ensemble = np.median(stacked, axis=0).astype(np.float32)
    if np.any(np.diff(ensemble, axis=-1) < -1e-7):
        raise RuntimeError("Round 46 WaveBound ensemble quantiles crossed")
    return StabilityForecastBundle(
        candidate_id="wavebound_ema",
        seed_predictions_bps=stacked,
        ensemble_predictions_bps=ensemble,
        artifacts=tuple(artifacts),
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        backend_kind=backend_kind,
        backend_device=backend_device,
        training_history=tuple(history),
    )


def _train_mutual(
    dataset: DistributionalDataset,
    normalized_features: np.ndarray,
    feature_scaler: FeatureScaler,
    target_scaler: TargetScaler,
    *,
    model_dir: Path,
    device: object,
    backend_kind: str,
    backend_device: str,
    progress: ProgressCallback | None,
) -> StabilityForecastBundle:
    peers: list[DistributionalTCN] = []
    optimizers: list[ExplicitAdamW] = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        peer = DistributionalTCN(input_channels=normalized_features.shape[-1]).to(
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
    normalized_targets = target_scaler.normalize(dataset.forward_return_bps)
    validation_mask = role_mask(dataset, "early_stop")
    best_loss = math.inf
    best_epoch = 0
    best_states: list[dict[str, torch.Tensor]] | None = None
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
        peer_loss_rows: list[list[float]] = []
        consistency_losses: list[float] = []
        objective_losses: list[float] = []
        pair_horizon_sum = np.zeros((3, len(HORIZONS)), dtype=np.float64)
        batch_count = 0
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for offset in range(0, order.size, BATCH_SIZE):
                selected = pairs[order[offset : offset + BATCH_SIZE]]
                batch_features, batch_targets = _batch_arrays(
                    normalized_features, normalized_targets, selected
                )
                values = torch.from_numpy(batch_features).to(device)
                targets = torch.from_numpy(batch_targets).to(device)[
                    ..., -SUPERVISED_HOURS:
                ]
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                predictions = [peer(values)[..., -SUPERVISED_HOURS:] for peer in peers]
                supervised = torch.stack(
                    [
                        pinball_components(prediction, targets).mean()
                        for prediction in predictions
                    ]
                )
                consistency, pair_horizon = standardized_median_consistency(predictions)
                objective = supervised.mean() + CONSISTENCY_WEIGHT * consistency
                objective.backward()
                for peer, optimizer in zip(peers, optimizers, strict=True):
                    torch.nn.utils.clip_grad_norm_(
                        peer.parameters(), 1.0, foreach=False
                    )
                    optimizer.step()
                optimizer_updates += 1
                peer_loss_rows.append(supervised.detach().cpu().numpy().tolist())
                consistency_losses.append(float(consistency.detach().cpu()))
                objective_losses.append(float(objective.detach().cpu()))
                pair_horizon_sum += pair_horizon.detach().cpu().numpy()
                batch_count += 1
            full_predictions = [
                _predict_all(peer, normalized_features, target_scaler, device)
                for peer in peers
            ]
            ensemble = np.median(np.stack(full_predictions), axis=0).astype(np.float32)
            validation_loss = _role_pinball(
                dataset, ensemble, validation_mask, target_scaler
            )
            peer_validation = [
                _role_pinball(dataset, values, validation_mask, target_scaler)
                for values in full_predictions
            ]
        warning_messages.extend(str(item.message) for item in caught)
        peer_training = np.mean(np.asarray(peer_loss_rows), axis=0)
        row: dict[str, object] = {
            "candidate_id": "mutual_median_consistency",
            "epoch": epoch,
            "optimizer_updates": optimizer_updates,
            "training_supervised_pinball": float(np.mean(peer_training)),
            "training_supervised_pinball_by_seed": {
                str(seed): float(value)
                for seed, value in zip(SEEDS, peer_training, strict=True)
            },
            "training_consistency": float(np.mean(consistency_losses)),
            "training_objective": float(np.mean(objective_losses)),
            "training_pair_horizon_consistency": (
                pair_horizon_sum / max(batch_count, 1)
            ).tolist(),
            "peer_early_stop_pinball": {
                str(seed): float(value)
                for seed, value in zip(SEEDS, peer_validation, strict=True)
            },
            "ensemble_early_stop_pinball": validation_loss,
        }
        scalar_values = (
            float(row["training_supervised_pinball"]),
            float(row["training_consistency"]),
            float(row["training_objective"]),
            validation_loss,
            *peer_validation,
        )
        if not all(math.isfinite(value) for value in scalar_values):
            raise RuntimeError(
                "Round 46 mutual training produced nonfinite diagnostics"
            )
        improved = validation_loss < best_loss - MINIMUM_IMPROVEMENT
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_states = [_clone_state(peer) for peer in peers]
            stale_epochs = 0
        else:
            stale_epochs += 1
        row["best_early_stop_pinball"] = best_loss
        row["best_epoch"] = best_epoch
        row["stale_epochs"] = stale_epochs
        history.append(row)
        if progress is not None:
            progress("round46_mutual_epoch", row)
        if stale_epochs >= EARLY_STOPPING_PATIENCE:
            break
    if best_states is None:
        raise RuntimeError("Round 46 mutual training did not produce best states")
    fallback = _fallback_messages(warning_messages)
    if fallback:
        raise RuntimeError(f"Round 46 mutual training used CPU fallback: {fallback}")
    artifacts: list[StabilityTCNArtifact] = []
    predictions: list[np.ndarray] = []
    for seed, peer, best_state in zip(SEEDS, peers, best_states, strict=True):
        peer.load_state_dict(best_state)
        best_predictions = _predict_all(
            peer, normalized_features, target_scaler, device
        )
        artifact = _finalize_artifact(
            dataset=dataset,
            candidate_id="mutual_median_consistency",
            seed=seed,
            epochs=epochs_run,
            best_epoch=best_epoch,
            best_loss=best_loss,
            optimizer_updates=optimizer_updates,
            evaluation_model="peer_at_common_best_epoch",
            best_state=best_state,
            best_predictions=best_predictions,
            artifact_path=(
                model_dir / f"round46_mutual_median_consistency_seed_{seed}.pt"
            ),
            normalized_features=normalized_features,
            target_scaler=target_scaler,
            device=device,
            backend_kind=backend_kind,
            backend_device=backend_device,
            warning_messages=warning_messages,
        )
        artifacts.append(artifact)
        predictions.append(best_predictions)
    stacked = np.stack(predictions).astype(np.float32)
    ensemble = np.median(stacked, axis=0).astype(np.float32)
    if np.any(np.diff(ensemble, axis=-1) < -1e-7):
        raise RuntimeError("Round 46 mutual ensemble quantiles crossed")
    return StabilityForecastBundle(
        candidate_id="mutual_median_consistency",
        seed_predictions_bps=stacked,
        ensemble_predictions_bps=ensemble,
        artifacts=tuple(artifacts),
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        backend_kind=backend_kind,
        backend_device=backend_device,
        training_history=tuple(history),
    )


def train_stability_candidates(
    dataset: DistributionalDataset,
    *,
    model_dir: Path,
    compute_backend: str = "directml",
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, StabilityForecastBundle], dict[str, object]]:
    training_mask = role_mask(dataset, "training")
    feature_scaler = fit_feature_scaler(dataset, training_mask)
    target_scaler = fit_target_scaler(dataset, training_mask)
    normalized_features = feature_scaler.transform(dataset.features)
    if compute_backend == "directml":
        device, preflight = directml_stability_preflight()
    elif compute_backend == "cpu":
        device, preflight = cpu_stability_preflight()
    else:
        raise ValueError(f"Round 46 compute backend is invalid: {compute_backend}")
    wavebound = _train_wavebound(
        dataset,
        normalized_features,
        feature_scaler,
        target_scaler,
        model_dir=model_dir,
        device=device,
        backend_kind=str(preflight["backend_kind"]),
        backend_device=str(preflight["backend_device"]),
        progress=progress,
    )
    mutual = _train_mutual(
        dataset,
        normalized_features,
        feature_scaler,
        target_scaler,
        model_dir=model_dir,
        device=device,
        backend_kind=str(preflight["backend_kind"]),
        backend_device=str(preflight["backend_device"]),
        progress=progress,
    )
    return {wavebound.candidate_id: wavebound, mutual.candidate_id: mutual}, preflight


def stability_mechanism_gate(
    bundle: StabilityForecastBundle,
    forecast_diagnostics: Mapping[str, object],
) -> dict[str, object]:
    median_validation = float(
        np.median([artifact.best_early_stop_pinball for artifact in bundle.artifacts])
    )
    gate = forecast_diagnostics["gate"]
    if not isinstance(gate, Mapping):
        raise TypeError("Round 46 forecast gate payload is invalid")
    minimum_stability = float(gate["minimum_pairwise_seed_median_prediction_spearman"])
    stability_delta = minimum_stability - PREDECESSOR_MINIMUM_STABILITY
    validation_degradation = (
        median_validation / PREDECESSOR_MEDIAN_EARLY_STOP_PINBALL - 1.0
    )
    reasons: list[str] = []
    if not math.isfinite(minimum_stability) or minimum_stability < MINIMUM_STABILITY:
        reasons.append("minimum_seed_stability_below_0_5")
    if stability_delta < MINIMUM_STABILITY_IMPROVEMENT:
        reasons.append("seed_stability_improvement_below_frozen_minimum")
    if (
        not math.isfinite(median_validation)
        or median_validation > MAXIMUM_MEDIAN_EARLY_STOP_PINBALL
    ):
        reasons.append("median_early_stop_pinball_degraded_more_than_two_percent")
    return {
        "candidate_id": bundle.candidate_id,
        "passed": not reasons,
        "reasons": reasons,
        "predecessor_minimum_seed_stability": PREDECESSOR_MINIMUM_STABILITY,
        "minimum_seed_stability": minimum_stability,
        "seed_stability_delta": stability_delta,
        "minimum_required_seed_stability": MINIMUM_STABILITY,
        "minimum_required_seed_stability_improvement": MINIMUM_STABILITY_IMPROVEMENT,
        "predecessor_median_best_early_stop_pinball": (
            PREDECESSOR_MEDIAN_EARLY_STOP_PINBALL
        ),
        "median_best_early_stop_pinball": median_validation,
        "relative_early_stop_pinball_degradation": validation_degradation,
        "maximum_permitted_median_best_early_stop_pinball": (
            MAXIMUM_MEDIAN_EARLY_STOP_PINBALL
        ),
    }


__all__ = [
    "BATCH_SIZE",
    "CANDIDATES",
    "CONSISTENCY_WEIGHT",
    "EMA_DECAY",
    "SEEDS",
    "StabilityForecastBundle",
    "StabilityTCNArtifact",
    "cpu_stability_preflight",
    "directml_stability_preflight",
    "pinball_components",
    "stability_mechanism_gate",
    "standardized_median_consistency",
    "train_stability_candidates",
    "update_ema_target",
    "wavebound_pinball_loss",
]
