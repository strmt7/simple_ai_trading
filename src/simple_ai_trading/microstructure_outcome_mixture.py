"""Conditional win/loss mixture model for adaptive BBO action values."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Callable, Mapping
from uuid import uuid4

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file as save_safetensors

from .compute import resolve_backend
from .microstructure_action_architecture import ActionValuePredictionBatch
from .microstructure_architecture import (
    _ManualAdam,
    _feature_scaler,
    _seed_torch,
    _sequence_batch,
    _torch_device,
    _torch_modules,
    valid_sequence_endpoints,
)
from .microstructure_barriers import (
    ADAPTIVE_BARRIER_TARGET_MODE,
    AdaptiveBarrierSpec,
    AdaptiveBarrierTargets,
    validate_adaptive_barrier_targets,
)
from .microstructure_features import (
    MicrostructureDataset,
    microstructure_feature_names,
    validate_microstructure_dataset,
)


OUTCOME_MIXTURE_SCHEMA_VERSION = "adaptive-outcome-mixture-neural-v1"
_TRAINING_PRELOAD_LIMIT_BYTES = 512 * 1024 * 1024
_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024
_LEARNING_RATE = 8.0e-4
_BETA_1 = 0.9
_BETA_2 = 0.99
_EPSILON = 1.0e-7
_OUTPUTS_PER_SIDE = 5


@dataclass(frozen=True)
class OutcomeMixtureArchitectureSpec:
    """Precommitted architecture and proper-loss weights."""

    candidate_id: str
    family: str
    sequence_length: int
    hidden_dim: int
    residual_blocks: int
    dropout: float
    probability_loss_weight: float
    magnitude_loss_weight: float
    expected_value_loss_weight: float
    quantile_loss_weight: float
    ranking_loss_weight: float

    def __post_init__(self) -> None:
        weights = (
            self.probability_loss_weight,
            self.magnitude_loss_weight,
            self.expected_value_loss_weight,
            self.quantile_loss_weight,
            self.ranking_loss_weight,
        )
        if not self.candidate_id.strip():
            raise ValueError("outcome-mixture candidate_id cannot be empty")
        if (
            self.family != "conditional_outcome_mixture_residual_mlp"
            or self.sequence_length != 1
        ):
            raise ValueError("outcome-mixture architecture family is unsupported")
        if (
            not 16 <= int(self.hidden_dim) <= 512
            or not 1 <= int(self.residual_blocks) <= 8
        ):
            raise ValueError("outcome-mixture architecture dimensions are invalid")
        if not math.isfinite(float(self.dropout)) or not all(
            math.isfinite(float(value)) for value in weights
        ):
            raise ValueError("outcome-mixture loss settings must be finite")
        if (
            not 0.0 <= self.dropout < 0.75
            or any(not 0.0 <= float(value) <= 4.0 for value in weights)
            or sum(float(value) for value in weights[:4]) <= 0.0
        ):
            raise ValueError("outcome-mixture loss settings are outside bounds")


@dataclass(frozen=True)
class TrainedOutcomeMixtureModel:
    """A research-only model with a complete, hash-bound reload contract."""

    schema_version: str
    spec: OutcomeMixtureArchitectureSpec
    feature_version: str
    feature_names: tuple[str, ...]
    target_schema_version: str
    target_mode: str
    target_spec: AdaptiveBarrierSpec
    target_contract_sha256: str
    target_scenario: str
    backend_requested: str
    backend_kind: str
    backend_device: str
    optimizer_kind: str
    optimizer_hyperparameters: Mapping[str, float]
    training_data_mode: str
    training_preload_bytes: int
    sequence_length: int
    target_scale_bps: float
    positive_class_prevalence: tuple[float, float]
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
    portfolio_claim: bool = False
    leverage_applied: bool = False


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _network(spec: OutcomeMixtureArchitectureSpec, feature_count: int):
    _torch, nn, functional = _torch_modules()

    class ResidualBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.normalization = nn.LayerNorm(spec.hidden_dim)
            self.first = nn.Linear(spec.hidden_dim, spec.hidden_dim * 2)
            self.second = nn.Linear(spec.hidden_dim * 2, spec.hidden_dim)
            self.dropout = nn.Dropout(spec.dropout)

        def forward(self, values):
            output = self.normalization(values)
            output = self.dropout(functional.gelu(self.first(output)))
            return values + self.dropout(self.second(output))

    class OutcomeMixtureNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(feature_count, spec.hidden_dim)
            self.blocks = nn.ModuleList(
                ResidualBlock() for _index in range(spec.residual_blocks)
            )
            self.normalization = nn.LayerNorm(spec.hidden_dim)
            self.head = nn.Linear(spec.hidden_dim, 2 * _OUTPUTS_PER_SIDE)

        def forward(self, values):
            output = functional.gelu(self.projection(values[:, -1, :]))
            for block in self.blocks:
                output = block(output)
            return self.head(self.normalization(output)).reshape(
                -1, 2, _OUTPUTS_PER_SIDE
            )

    return OutcomeMixtureNetwork()


def _positive_class_prevalence(targets: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(targets > 0.0, dtype=bool)
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError("outcome-mixture prevalence targets are invalid")
    prevalence: list[float] = []
    for side in range(2):
        positives = int(np.sum(labels[:, side]))
        negatives = len(labels) - positives
        if min(positives, negatives) < 64:
            raise ValueError(
                "outcome-mixture training lacks side-specific class support"
            )
        prevalence.append(float(positives / len(labels)))
    return prevalence[0], prevalence[1]


def _target_contract_sha256(targets: AdaptiveBarrierTargets) -> str:
    contract = {
        "schema_version": targets.schema_version,
        "target_mode": targets.target_mode,
        "spec": asdict(targets.spec),
    }
    return hashlib.sha256(_canonical_json(contract).encode("ascii")).hexdigest()


def _decoded_heads(output):
    _torch, _nn, functional = _torch_modules()
    probability = torch_probability = functional.sigmoid(output[:, :, 0])
    positive_magnitude = functional.softplus(output[:, :, 1])
    negative_magnitude = functional.softplus(output[:, :, 2])
    expected = (
        torch_probability * positive_magnitude
        - (1.0 - torch_probability) * negative_magnitude
    )
    lower = expected - functional.softplus(output[:, :, 3])
    upper = expected + functional.softplus(output[:, :, 4])
    return (
        probability,
        positive_magnitude,
        negative_magnitude,
        expected,
        lower,
        upper,
    )


def _weighted_ranking_loss(prediction, target, sample_weight):
    torch, _nn, _functional = _torch_modules()
    weights = sample_weight[:, None]
    total_weight = torch.sum(weights)
    prediction_mean = torch.sum(prediction * weights, dim=0) / total_weight
    target_mean = torch.sum(target * weights, dim=0) / total_weight
    prediction_centered = prediction - prediction_mean[None, :]
    target_centered = target - target_mean[None, :]
    covariance = (
        torch.sum(prediction_centered * target_centered * weights, dim=0) / total_weight
    )
    prediction_variance = (
        torch.sum(prediction_centered.square() * weights, dim=0) / total_weight
    )
    target_variance = (
        torch.sum(target_centered.square() * weights, dim=0) / total_weight
    )
    correlation = covariance / torch.sqrt(
        prediction_variance * target_variance + 1.0e-8
    )
    return 1.0 - torch.mean(torch.clamp(correlation, -1.0, 1.0))


def _loss(output, target, sample_weight, prevalence, spec):
    torch, _nn, functional = _torch_modules()
    (
        probability,
        positive_magnitude,
        negative_magnitude,
        expected,
        lower,
        upper,
    ) = _decoded_heads(output)
    labels = (target > 0.0).to(dtype=target.dtype)
    probability_loss = functional.softplus(output[:, :, 0]) - (labels * output[:, :, 0])
    positive_target = functional.relu(target)
    negative_target = functional.relu(-target)
    positive_loss = functional.smooth_l1_loss(
        positive_magnitude,
        positive_target,
        reduction="none",
        beta=0.5,
    )
    negative_loss = functional.smooth_l1_loss(
        negative_magnitude,
        negative_target,
        reduction="none",
        beta=0.5,
    )
    positive_balance = 0.5 / prevalence[None, :]
    negative_balance = 0.5 / (1.0 - prevalence[None, :])
    magnitude_loss = (
        labels * positive_balance * positive_loss
        + (1.0 - labels) * negative_balance * negative_loss
    )
    expected_loss = functional.smooth_l1_loss(
        expected,
        target,
        reduction="none",
        beta=0.5,
    )

    def pinball(prediction, quantile: float):
        error = target - prediction
        return torch.maximum(quantile * error, (quantile - 1.0) * error)

    quantile_loss = pinball(lower, 0.10) + pinball(upper, 0.90)
    per_row = torch.mean(
        spec.probability_loss_weight * probability_loss
        + spec.magnitude_loss_weight * magnitude_loss
        + spec.expected_value_loss_weight * expected_loss
        + spec.quantile_loss_weight * quantile_loss,
        dim=1,
    )
    weighted = torch.sum(per_row * sample_weight) / torch.sum(sample_weight)
    if spec.ranking_loss_weight > 0.0:
        weighted = weighted + spec.ranking_loss_weight * _weighted_ranking_loss(
            expected, target, sample_weight
        )
    if probability.shape != target.shape:
        raise ValueError("outcome-mixture decoded head shape is invalid")
    return weighted


def _state_hash(
    *,
    spec: OutcomeMixtureArchitectureSpec,
    feature_version: str,
    feature_names: tuple[str, ...],
    target_schema_version: str,
    target_mode: str,
    target_spec: AdaptiveBarrierSpec,
    target_contract_sha256: str,
    target_scenario: str,
    backend_requested: str,
    backend_kind: str,
    backend_device: str,
    optimizer_kind: str,
    optimizer_hyperparameters: Mapping[str, float],
    training_data_mode: str,
    training_preload_bytes: int,
    sequence_length: int,
    target_scale_bps: float,
    positive_class_prevalence: tuple[float, float],
    center: np.ndarray,
    scale: np.ndarray,
    best_epoch: int,
    training_loss: float,
    tuning_loss: float,
    state: Mapping[str, np.ndarray],
) -> str:
    contract = {
        "schema_version": OUTCOME_MIXTURE_SCHEMA_VERSION,
        "spec": asdict(spec),
        "feature_version": feature_version,
        "feature_names": list(feature_names),
        "target_schema_version": target_schema_version,
        "target_mode": target_mode,
        "target_spec": asdict(target_spec),
        "target_contract_sha256": target_contract_sha256,
        "target_scenario": target_scenario,
        "backend_requested": backend_requested,
        "backend_kind": backend_kind,
        "backend_device": backend_device,
        "optimizer_kind": optimizer_kind,
        "optimizer_hyperparameters": dict(optimizer_hyperparameters),
        "training_data_mode": training_data_mode,
        "training_preload_bytes": int(training_preload_bytes),
        "sequence_length": int(sequence_length),
        "target_scale_bps": float(target_scale_bps),
        "positive_class_prevalence": list(positive_class_prevalence),
        "best_epoch": int(best_epoch),
        "training_loss": float(training_loss),
        "tuning_loss": float(tuning_loss),
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
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


def _model_hash(model: TrainedOutcomeMixtureModel) -> str:
    return _state_hash(
        spec=model.spec,
        feature_version=model.feature_version,
        feature_names=model.feature_names,
        target_schema_version=model.target_schema_version,
        target_mode=model.target_mode,
        target_spec=model.target_spec,
        target_contract_sha256=model.target_contract_sha256,
        target_scenario=model.target_scenario,
        backend_requested=model.backend_requested,
        backend_kind=model.backend_kind,
        backend_device=model.backend_device,
        optimizer_kind=model.optimizer_kind,
        optimizer_hyperparameters=model.optimizer_hyperparameters,
        training_data_mode=model.training_data_mode,
        training_preload_bytes=model.training_preload_bytes,
        sequence_length=model.sequence_length,
        target_scale_bps=model.target_scale_bps,
        positive_class_prevalence=model.positive_class_prevalence,
        center=model.scaler_center,
        scale=model.scaler_scale,
        best_epoch=model.best_epoch,
        training_loss=model.training_loss,
        tuning_loss=model.tuning_loss,
        state=model.state,
    )


def _validate_model_contract(model: TrainedOutcomeMixtureModel) -> None:
    try:
        expected_feature_names = microstructure_feature_names(model.feature_version)
    except ValueError as exc:
        raise ValueError("outcome-mixture feature contract is unsupported") from exc
    expected_target_contract = hashlib.sha256(
        _canonical_json(
            {
                "schema_version": model.target_schema_version,
                "target_mode": model.target_mode,
                "spec": asdict(model.target_spec),
            }
        ).encode("ascii")
    ).hexdigest()
    if (
        model.trading_authority
        or model.execution_claim
        or model.profitability_claim
        or model.portfolio_claim
        or model.leverage_applied
        or model.schema_version != OUTCOME_MIXTURE_SCHEMA_VERSION
        or model.feature_names != expected_feature_names
        or model.target_mode != ADAPTIVE_BARRIER_TARGET_MODE
        or expected_target_contract != model.target_contract_sha256
        or model.target_scenario not in {"base", "stress"}
        or model.sequence_length != model.spec.sequence_length
        or model.scaler_center.shape != (len(model.feature_names),)
        or model.scaler_scale.shape != (len(model.feature_names),)
        or not np.all(np.isfinite(model.scaler_center))
        or not np.all(np.isfinite(model.scaler_scale))
        or np.any(model.scaler_scale <= 0.0)
        or not math.isfinite(model.target_scale_bps)
        or model.target_scale_bps <= 0.0
        or len(model.positive_class_prevalence) != 2
        or any(
            not math.isfinite(value) or not 0.0 < value < 1.0
            for value in model.positive_class_prevalence
        )
        or model.best_epoch < 1
        or not math.isfinite(model.training_loss)
        or not math.isfinite(model.tuning_loss)
        or model.training_loss < 0.0
        or model.tuning_loss < 0.0
        or model.training_data_mode
        not in {
            "device_preloaded",
            "streamed_host_batches",
        }
        or model.training_preload_bytes < 0
        or model.model_sha256 != _model_hash(model)
    ):
        raise ValueError("outcome-mixture neural model contract is invalid")
    expected_state = _network(model.spec, len(model.feature_names)).state_dict()
    if set(model.state) != set(expected_state) or any(
        np.asarray(model.state[name]).shape != tuple(expected_state[name].shape)
        or np.asarray(model.state[name]).dtype != np.dtype(np.float32)
        or not np.all(np.isfinite(model.state[name]))
        for name in expected_state
    ):
        raise ValueError("outcome-mixture neural state contract is invalid")


def train_outcome_mixture_model(
    dataset: MicrostructureDataset,
    barrier_targets: AdaptiveBarrierTargets,
    *,
    train_endpoints: np.ndarray,
    tuning_endpoints: np.ndarray,
    spec: OutcomeMixtureArchitectureSpec,
    target_scenario: str,
    compute_backend: str,
    seed: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    train_sample_weights: np.ndarray | None = None,
    tuning_sample_weights: np.ndarray | None = None,
    progress: Callable[[int, int, float, float], None] | None = None,
) -> TrainedOutcomeMixtureModel:
    """Fit a research-only conditional win/loss outcome mixture."""

    validate_microstructure_dataset(dataset)
    validate_adaptive_barrier_targets(dataset, barrier_targets)
    if target_scenario not in {"base", "stress"}:
        raise ValueError("outcome-mixture target scenario is unsupported")
    long_values = (
        barrier_targets.base_long_net_bps
        if target_scenario == "base"
        else barrier_targets.stress_long_net_bps
    )
    short_values = (
        barrier_targets.base_short_net_bps
        if target_scenario == "base"
        else barrier_targets.stress_short_net_bps
    )
    targets = np.full((dataset.rows, 2), np.nan, dtype=np.float32)
    valid_source_indexes = barrier_targets.source_indexes[barrier_targets.valid]
    targets[valid_source_indexes, 0] = long_values[barrier_targets.valid]
    targets[valid_source_indexes, 1] = short_values[barrier_targets.valid]
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
        raise ValueError("outcome-mixture split has insufficient contiguous rows")
    if train[-1] >= tuning[0]:
        raise ValueError("outcome-mixture train/tuning split is not chronological")
    if not np.all(np.isfinite(targets[train])) or not np.all(
        np.isfinite(targets[tuning])
    ):
        raise ValueError("outcome-mixture split targets are non-finite")
    scenario_train_exits = (
        (
            barrier_targets.base_long_exit_time_ms,
            barrier_targets.base_short_exit_time_ms,
        )
        if target_scenario == "base"
        else (
            barrier_targets.stress_long_exit_time_ms,
            barrier_targets.stress_short_exit_time_ms,
        )
    )
    source_positions = np.searchsorted(barrier_targets.source_indexes, train)
    if max(
        int(np.max(scenario_train_exits[0][source_positions])),
        int(np.max(scenario_train_exits[1][source_positions])),
    ) >= int(dataset.decision_time_ms[tuning[0]]):
        raise ValueError("outcome-mixture split is not purged across target lifecycles")
    batch = int(batch_size)
    epochs = int(max_epochs)
    stop_patience = int(patience)
    if batch < 32 or epochs < 1 or stop_patience < 1:
        raise ValueError("outcome-mixture training budget is invalid")
    center, scale = _feature_scaler(dataset.features, train)
    target_scale = max(1.0, float(np.quantile(np.abs(targets[train]), 0.90)))
    prevalence_values = _positive_class_prevalence(targets[train])
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
        or np.any(~np.isfinite(train_weights))
        or np.any(~np.isfinite(tuning_weights))
        or np.any(train_weights <= 0.0)
        or np.any(tuning_weights <= 0.0)
    ):
        raise ValueError("outcome-mixture sample weights are invalid")
    backend = resolve_backend(compute_backend)
    device = _torch_device(backend)
    torch, _nn, _functional = _torch_modules()
    _seed_torch(torch, int(seed), backend)
    network = _network(spec, len(dataset.feature_names)).to(device)
    optimizer = _ManualAdam(
        torch,
        network.parameters(),
        learning_rate=_LEARNING_RATE,
        beta_1=_BETA_1,
        beta_2=_BETA_2,
        epsilon=_EPSILON,
    )
    prevalence = torch.tensor(prevalence_values, dtype=torch.float32).to(device)
    rng = np.random.default_rng(int(seed))
    preload_bytes = int(
        (len(train) + len(tuning))
        * (spec.sequence_length * len(dataset.feature_names) + 3)
        * np.dtype(np.float32).itemsize
    )
    preloaded = preload_bytes <= _TRAINING_PRELOAD_LIMIT_BYTES

    def prepared_split(endpoints: np.ndarray, weights: np.ndarray):
        if not preloaded:
            return None
        values = torch.from_numpy(
            _sequence_batch(
                dataset.features,
                endpoints,
                sequence_length=spec.sequence_length,
                center=center,
                scale=scale,
            )
        ).to(device)
        labels = torch.from_numpy(
            np.ascontiguousarray(targets[endpoints] / target_scale, dtype=np.float32)
        ).to(device)
        weight_values = torch.from_numpy(
            np.ascontiguousarray(weights, dtype=np.float32)
        ).to(device)
        return values, labels, weight_values

    prepared_train = prepared_split(train, train_weights)
    prepared_tuning = prepared_split(tuning, tuning_weights)

    def epoch_loss(endpoints, weights, training, prepared) -> float:
        order = (
            rng.permutation(len(endpoints)) if training else np.arange(len(endpoints))
        )
        total = 0.0
        total_weight = 0.0
        network.train(training)
        for start in range(0, len(order), batch):
            positions = order[start : start + batch]
            batch_endpoints = endpoints[positions]
            if prepared is None:
                x = torch.from_numpy(
                    _sequence_batch(
                        dataset.features,
                        batch_endpoints,
                        sequence_length=spec.sequence_length,
                        center=center,
                        scale=scale,
                    )
                ).to(device)
                y = torch.from_numpy(
                    np.ascontiguousarray(
                        targets[batch_endpoints] / target_scale,
                        dtype=np.float32,
                    )
                ).to(device)
                weight = torch.from_numpy(weights[positions]).to(device)
            else:
                device_positions = torch.from_numpy(
                    np.ascontiguousarray(positions, dtype=np.int64)
                ).to(device)
                x = prepared[0].index_select(0, device_positions)
                y = prepared[1].index_select(0, device_positions)
                weight = prepared[2].index_select(0, device_positions)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                loss = _loss(network(x), y, weight, prevalence, spec)
                if not bool(torch.isfinite(loss).detach().cpu().item()):
                    raise ValueError("outcome-mixture loss became non-finite")
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        network.parameters(), 5.0, error_if_nonfinite=True
                    )
                    optimizer.step()
            weight_sum = float(np.sum(weights[positions]))
            total += float(loss.detach().cpu().item()) * weight_sum
            total_weight += weight_sum
        return total / total_weight

    best_device_state: dict[str, object] | None = None
    best_epoch = 0
    best_tuning = float("inf")
    best_training = float("inf")
    stale = 0
    for epoch in range(1, epochs + 1):
        training_loss = epoch_loss(train, train_weights, True, prepared_train)
        tuning_loss = epoch_loss(tuning, tuning_weights, False, prepared_tuning)
        if progress is not None:
            progress(epoch, epochs, training_loss, tuning_loss)
        if tuning_loss < best_tuning - 1.0e-5:
            best_tuning = tuning_loss
            best_training = training_loss
            best_epoch = epoch
            best_device_state = {
                name: value.detach().clone()
                for name, value in network.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= stop_patience:
                break
    assert best_device_state is not None
    state = {
        name: value.detach().cpu().numpy().astype(np.float32, copy=True)
        for name, value in best_device_state.items()
    }
    target_contract_sha256 = _target_contract_sha256(barrier_targets)
    training_data_mode = "device_preloaded" if preloaded else "streamed_host_batches"
    training_preload_bytes = preload_bytes if preloaded else 0
    model_sha256 = _state_hash(
        spec=spec,
        feature_version=dataset.feature_version,
        feature_names=dataset.feature_names,
        target_schema_version=barrier_targets.schema_version,
        target_mode=ADAPTIVE_BARRIER_TARGET_MODE,
        target_spec=barrier_targets.spec,
        target_contract_sha256=target_contract_sha256,
        target_scenario=target_scenario,
        backend_requested=backend.requested,
        backend_kind=backend.kind,
        backend_device=backend.device,
        optimizer_kind=optimizer.kind,
        optimizer_hyperparameters=optimizer.hyperparameters,
        training_data_mode=training_data_mode,
        training_preload_bytes=training_preload_bytes,
        sequence_length=spec.sequence_length,
        target_scale_bps=target_scale,
        positive_class_prevalence=prevalence_values,
        center=center,
        scale=scale,
        best_epoch=best_epoch,
        training_loss=best_training,
        tuning_loss=best_tuning,
        state=state,
    )
    model = TrainedOutcomeMixtureModel(
        schema_version=OUTCOME_MIXTURE_SCHEMA_VERSION,
        spec=spec,
        feature_version=dataset.feature_version,
        feature_names=dataset.feature_names,
        target_schema_version=barrier_targets.schema_version,
        target_mode=ADAPTIVE_BARRIER_TARGET_MODE,
        target_spec=barrier_targets.spec,
        target_contract_sha256=target_contract_sha256,
        target_scenario=target_scenario,
        backend_requested=backend.requested,
        backend_kind=backend.kind,
        backend_device=backend.device,
        optimizer_kind=optimizer.kind,
        optimizer_hyperparameters=optimizer.hyperparameters,
        training_data_mode=training_data_mode,
        training_preload_bytes=training_preload_bytes,
        sequence_length=spec.sequence_length,
        target_scale_bps=target_scale,
        positive_class_prevalence=prevalence_values,
        scaler_center=center,
        scaler_scale=scale,
        best_epoch=best_epoch,
        training_loss=best_training,
        tuning_loss=best_tuning,
        state=state,
        model_sha256=model_sha256,
    )
    _validate_model_contract(model)
    return model


def predict_outcome_mixture_model(
    model: TrainedOutcomeMixtureModel,
    dataset: MicrostructureDataset,
    endpoints: np.ndarray,
    *,
    compute_backend: str,
    batch_size: int,
) -> ActionValuePredictionBatch:
    validate_microstructure_dataset(dataset)
    _validate_model_contract(model)
    if (
        model.feature_version != dataset.feature_version
        or model.feature_names != dataset.feature_names
    ):
        raise ValueError("outcome-mixture feature contract differs from the dataset")
    if int(batch_size) <= 0:
        raise ValueError("outcome-mixture prediction batch size must be positive")
    selected = valid_sequence_endpoints(
        dataset.decision_time_ms,
        np.asarray(endpoints, dtype=np.int64),
        sequence_length=model.sequence_length,
        cadence_seconds=dataset.decision_cadence_seconds,
    )
    if selected.size == 0:
        raise ValueError("outcome-mixture prediction has no contiguous endpoints")
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
    raw = np.concatenate(outputs, axis=0).astype(np.float64)
    if raw.shape != (len(selected), 2, _OUTPUTS_PER_SIDE) or not np.all(
        np.isfinite(raw)
    ):
        raise ValueError("outcome-mixture neural model emitted invalid predictions")
    probability = 1.0 / (1.0 + np.exp(-np.clip(raw[:, :, 0], -40.0, 40.0)))
    positive = np.logaddexp(0.0, raw[:, :, 1])
    negative = np.logaddexp(0.0, raw[:, :, 2])
    expected = probability * positive - (1.0 - probability) * negative
    lower = expected - np.logaddexp(0.0, raw[:, :, 3])
    upper = expected + np.logaddexp(0.0, raw[:, :, 4])
    scale = model.target_scale_bps
    return ActionValuePredictionBatch(
        endpoint_indexes=selected,
        long_mean_bps=expected[:, 0] * scale,
        short_mean_bps=expected[:, 1] * scale,
        long_profitable_probability=probability[:, 0],
        short_profitable_probability=probability[:, 1],
        long_lower_bps=lower[:, 0] * scale,
        short_lower_bps=lower[:, 1] * scale,
        long_upper_bps=upper[:, 0] * scale,
        short_upper_bps=upper[:, 1] * scale,
    )


_METADATA_KEYS = {
    "schema_version",
    "spec",
    "feature_version",
    "feature_names",
    "target_schema_version",
    "target_mode",
    "target_spec",
    "target_contract_sha256",
    "target_scenario",
    "backend_requested",
    "backend_kind",
    "backend_device",
    "optimizer_kind",
    "optimizer_hyperparameters",
    "training_data_mode",
    "training_preload_bytes",
    "sequence_length",
    "target_scale_bps",
    "positive_class_prevalence",
    "best_epoch",
    "training_loss",
    "tuning_loss",
    "model_sha256",
    "trading_authority",
    "execution_claim",
    "profitability_claim",
    "portfolio_claim",
    "leverage_applied",
}


def _artifact_metadata(model: TrainedOutcomeMixtureModel) -> dict[str, str]:
    return {
        "schema_version": model.schema_version,
        "spec": _canonical_json(asdict(model.spec)),
        "feature_version": model.feature_version,
        "feature_names": _canonical_json(list(model.feature_names)),
        "target_schema_version": model.target_schema_version,
        "target_mode": model.target_mode,
        "target_spec": _canonical_json(asdict(model.target_spec)),
        "target_contract_sha256": model.target_contract_sha256,
        "target_scenario": model.target_scenario,
        "backend_requested": model.backend_requested,
        "backend_kind": model.backend_kind,
        "backend_device": model.backend_device,
        "optimizer_kind": model.optimizer_kind,
        "optimizer_hyperparameters": _canonical_json(
            dict(model.optimizer_hyperparameters)
        ),
        "training_data_mode": model.training_data_mode,
        "training_preload_bytes": str(model.training_preload_bytes),
        "sequence_length": str(model.sequence_length),
        "target_scale_bps": repr(float(model.target_scale_bps)),
        "positive_class_prevalence": _canonical_json(
            list(model.positive_class_prevalence)
        ),
        "best_epoch": str(model.best_epoch),
        "training_loss": repr(float(model.training_loss)),
        "tuning_loss": repr(float(model.tuning_loss)),
        "model_sha256": model.model_sha256,
        "trading_authority": "false",
        "execution_claim": "false",
        "profitability_claim": "false",
        "portfolio_claim": "false",
        "leverage_applied": "false",
    }


def save_outcome_mixture_model(
    path: str | Path, model: TrainedOutcomeMixtureModel
) -> None:
    """Atomically save every tensor and all metadata needed for exact reload."""

    _validate_model_contract(model)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "scaler_center": np.ascontiguousarray(model.scaler_center, dtype=np.float32),
        "scaler_scale": np.ascontiguousarray(model.scaler_scale, dtype=np.float32),
        **{
            f"state.{name}": np.ascontiguousarray(value, dtype=np.float32)
            for name, value in model.state.items()
        },
    }
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        save_safetensors(arrays, str(temporary), metadata=_artifact_metadata(model))
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _metadata_json(metadata: Mapping[str, str], key: str, expected_type):
    try:
        value = json.loads(metadata[key])
    except (KeyError, json.JSONDecodeError) as exc:
        raise ValueError(f"outcome-mixture artifact {key} metadata is invalid") from exc
    if not isinstance(value, expected_type):
        raise ValueError(f"outcome-mixture artifact {key} metadata is invalid")
    return value


def load_outcome_mixture_model(path: str | Path) -> TrainedOutcomeMixtureModel:
    """Load and independently validate a complete outcome-mixture artifact."""

    target = Path(path)
    try:
        size = target.stat().st_size
    except OSError as exc:
        raise ValueError("outcome-mixture artifact is unreadable") from exc
    if size <= 0 or size > _MAX_ARTIFACT_BYTES:
        raise ValueError("outcome-mixture artifact size is invalid")
    try:
        with safe_open(str(target), framework="np", device="cpu") as handle:
            metadata = handle.metadata() or {}
            keys = tuple(handle.keys())
            arrays = {name: np.asarray(handle.get_tensor(name)) for name in keys}
    except Exception as exc:
        raise ValueError("outcome-mixture artifact is not valid safetensors") from exc
    if set(metadata) != _METADATA_KEYS:
        raise ValueError("outcome-mixture artifact metadata contract is incomplete")
    if metadata.get("schema_version") != OUTCOME_MIXTURE_SCHEMA_VERSION or any(
        metadata.get(name) != "false"
        for name in (
            "trading_authority",
            "execution_claim",
            "profitability_claim",
            "portfolio_claim",
            "leverage_applied",
        )
    ):
        raise ValueError("outcome-mixture artifact authority contract is invalid")
    if "scaler_center" not in arrays or "scaler_scale" not in arrays:
        raise ValueError("outcome-mixture artifact scaler tensors are missing")
    state = {
        name.removeprefix("state."): np.ascontiguousarray(value, dtype=np.float32)
        for name, value in arrays.items()
        if name.startswith("state.")
    }
    if len(state) != len(arrays) - 2:
        raise ValueError("outcome-mixture artifact tensor namespace is invalid")
    spec_raw = _metadata_json(metadata, "spec", dict)
    target_spec_raw = _metadata_json(metadata, "target_spec", dict)
    feature_names_raw = _metadata_json(metadata, "feature_names", list)
    optimizer_raw = _metadata_json(metadata, "optimizer_hyperparameters", dict)
    prevalence_raw = _metadata_json(metadata, "positive_class_prevalence", list)
    try:
        model = TrainedOutcomeMixtureModel(
            schema_version=metadata["schema_version"],
            spec=OutcomeMixtureArchitectureSpec(**spec_raw),
            feature_version=metadata["feature_version"],
            feature_names=tuple(str(value) for value in feature_names_raw),
            target_schema_version=metadata["target_schema_version"],
            target_mode=metadata["target_mode"],
            target_spec=AdaptiveBarrierSpec(**target_spec_raw),
            target_contract_sha256=metadata["target_contract_sha256"],
            target_scenario=metadata["target_scenario"],
            backend_requested=metadata["backend_requested"],
            backend_kind=metadata["backend_kind"],
            backend_device=metadata["backend_device"],
            optimizer_kind=metadata["optimizer_kind"],
            optimizer_hyperparameters={
                str(name): float(value) for name, value in optimizer_raw.items()
            },
            training_data_mode=metadata["training_data_mode"],
            training_preload_bytes=int(metadata["training_preload_bytes"]),
            sequence_length=int(metadata["sequence_length"]),
            target_scale_bps=float(metadata["target_scale_bps"]),
            positive_class_prevalence=tuple(float(value) for value in prevalence_raw),
            scaler_center=np.ascontiguousarray(
                arrays["scaler_center"], dtype=np.float32
            ),
            scaler_scale=np.ascontiguousarray(arrays["scaler_scale"], dtype=np.float32),
            best_epoch=int(metadata["best_epoch"]),
            training_loss=float(metadata["training_loss"]),
            tuning_loss=float(metadata["tuning_loss"]),
            state=state,
            model_sha256=metadata["model_sha256"],
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "outcome-mixture artifact metadata values are invalid"
        ) from exc
    _validate_model_contract(model)
    return model


__all__ = [
    "OUTCOME_MIXTURE_SCHEMA_VERSION",
    "OutcomeMixtureArchitectureSpec",
    "TrainedOutcomeMixtureModel",
    "load_outcome_mixture_model",
    "predict_outcome_mixture_model",
    "save_outcome_mixture_model",
    "train_outcome_mixture_model",
]
