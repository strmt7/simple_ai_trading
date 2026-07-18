"""DirectML-capable neural action values for adaptive BBO barrier targets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Callable, Mapping, Sequence

import numpy as np

from .compute import require_backend, resolve_backend
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
    MICROSTRUCTURE_FEATURE_NAMES,
    MICROSTRUCTURE_FEATURE_VERSION,
    MicrostructureDataset,
    validate_microstructure_dataset,
)


ACTION_VALUE_ARCHITECTURE_SCHEMA_VERSION = "adaptive-action-value-neural-v1"
_TRAINING_PRELOAD_LIMIT_BYTES = 512 * 1024 * 1024
_LEARNING_RATE = 8.0e-4
_BETA_1 = 0.9
_BETA_2 = 0.99
_EPSILON = 1.0e-7
_SIDE_NAMES = ("long", "short")


@dataclass(frozen=True)
class ActionValueArchitectureSpec:
    candidate_id: str
    family: str
    sequence_length: int
    hidden_dim: int
    residual_blocks: int
    dropout: float
    head_coherence_weight: float
    action_utility_weight: float
    downside_penalty: float
    action_temperature: float

    def __post_init__(self) -> None:
        numeric = (
            self.dropout,
            self.head_coherence_weight,
            self.action_utility_weight,
            self.downside_penalty,
            self.action_temperature,
        )
        if not self.candidate_id.strip():
            raise ValueError("action-value candidate_id cannot be empty")
        if self.family != "shared_residual_mlp" or self.sequence_length != 1:
            raise ValueError("action-value architecture family is unsupported")
        if (
            not 16 <= int(self.hidden_dim) <= 512
            or not 1 <= int(self.residual_blocks) <= 8
        ):
            raise ValueError("action-value architecture dimensions are invalid")
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("action-value loss settings must be finite")
        if (
            not 0.0 <= self.dropout < 0.75
            or not 0.0 <= self.head_coherence_weight <= 2.0
            or not 0.0 <= self.action_utility_weight <= 1.0
            or not 0.0 <= self.downside_penalty <= 2.0
            or not 0.05 <= self.action_temperature <= 5.0
        ):
            raise ValueError("action-value loss settings are outside bounds")


@dataclass(frozen=True)
class TrainedActionValueModel:
    schema_version: str
    spec: ActionValueArchitectureSpec
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
    positive_class_weights: tuple[float, float]
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


@dataclass(frozen=True)
class ActionValuePredictionBatch:
    endpoint_indexes: np.ndarray
    long_mean_bps: np.ndarray
    short_mean_bps: np.ndarray
    long_profitable_probability: np.ndarray
    short_profitable_probability: np.ndarray
    long_lower_bps: np.ndarray
    short_lower_bps: np.ndarray
    long_upper_bps: np.ndarray
    short_upper_bps: np.ndarray
    long_router_weights: np.ndarray | None = None
    short_router_weights: np.ndarray | None = None

    @property
    def rows(self) -> int:
        return int(len(self.endpoint_indexes))


@dataclass(frozen=True)
class ActionValueEnsembleBatch:
    endpoint_indexes: np.ndarray
    long_mean_bps: np.ndarray
    short_mean_bps: np.ndarray
    long_epistemic_std_bps: np.ndarray
    short_epistemic_std_bps: np.ndarray
    long_profitable_probability: np.ndarray
    short_profitable_probability: np.ndarray
    long_lower_bps: np.ndarray
    short_lower_bps: np.ndarray
    long_upper_bps: np.ndarray
    short_upper_bps: np.ndarray
    long_positive_member_ratio: np.ndarray
    short_positive_member_ratio: np.ndarray
    member_count: int
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False
    long_router_weights: np.ndarray | None = None
    short_router_weights: np.ndarray | None = None

    @property
    def rows(self) -> int:
        return int(len(self.endpoint_indexes))


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _network(spec: ActionValueArchitectureSpec, feature_count: int):
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

    class ActionValueNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(feature_count, spec.hidden_dim)
            self.blocks = nn.ModuleList(
                ResidualBlock() for _index in range(spec.residual_blocks)
            )
            self.normalization = nn.LayerNorm(spec.hidden_dim)
            self.head = nn.Linear(spec.hidden_dim, 8)

        def forward(self, values):
            output = functional.gelu(self.projection(values[:, -1, :]))
            for block in self.blocks:
                output = block(output)
            return self.head(self.normalization(output)).reshape(-1, 2, 4)

    return ActionValueNetwork()


def _positive_class_weights(targets: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(targets > 0.0, dtype=bool)
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError("action-value class-weight targets are invalid")
    output: list[float] = []
    for side in range(2):
        positives = int(np.sum(labels[:, side]))
        negatives = len(labels) - positives
        if min(positives, negatives) < 64:
            raise ValueError("action-value training lacks side-specific class support")
        output.append(float(np.clip(negatives / positives, 1.0, 5.0)))
    return output[0], output[1]


def _target_contract_sha256(targets: AdaptiveBarrierTargets) -> str:
    contract = {
        "schema_version": targets.schema_version,
        "target_mode": targets.target_mode,
        "spec": asdict(targets.spec),
    }
    return hashlib.sha256(_canonical_json(contract).encode("ascii")).hexdigest()


def _decoded_heads(output):
    _torch, _nn, functional = _torch_modules()
    mean = output[:, :, 0]
    direction = output[:, :, 1]
    lower = mean - functional.softplus(output[:, :, 2])
    upper = mean + functional.softplus(output[:, :, 3])
    return mean, direction, lower, upper


def _loss(output, target, sample_weight, class_weight, spec):
    torch, _nn, functional = _torch_modules()
    mean, direction, lower, upper = _decoded_heads(output)
    labels = (target > 0.0).to(dtype=target.dtype)
    huber = functional.smooth_l1_loss(mean, target, reduction="none", beta=0.5)
    binary = functional.softplus(direction) - labels * direction
    binary = binary * torch.where(
        labels > 0.0,
        class_weight[None, :],
        torch.ones_like(labels),
    )
    coherence = (torch.sigmoid(direction) - torch.sigmoid(mean)).square()

    def pinball(prediction, quantile: float):
        error = target - prediction
        return torch.maximum(quantile * error, (quantile - 1.0) * error)

    quantile = pinball(lower, 0.10) + pinball(upper, 0.90)
    head_loss = torch.mean(
        huber
        + 0.25 * binary
        + 0.15 * quantile
        + spec.head_coherence_weight * coherence,
        dim=1,
    )
    flat = torch.zeros((len(mean), 1), dtype=mean.dtype, device=mean.device)
    action_probability = torch.softmax(
        torch.cat((mean, flat), dim=1) / spec.action_temperature,
        dim=1,
    )[:, :2]
    expected_value = torch.sum(action_probability * target, dim=1)
    expected_downside = torch.sum(action_probability * functional.relu(-target), dim=1)
    utility_loss = -expected_value + spec.downside_penalty * expected_downside
    per_row = head_loss + spec.action_utility_weight * utility_loss
    return torch.sum(per_row * sample_weight) / torch.sum(sample_weight)


def _state_hash(
    *,
    spec: ActionValueArchitectureSpec,
    target_schema_version: str,
    target_contract_sha256: str,
    target_scenario: str,
    center: np.ndarray,
    scale: np.ndarray,
    target_scale_bps: float,
    positive_class_weights: tuple[float, float],
    optimizer_kind: str,
    optimizer_hyperparameters: Mapping[str, float],
    state: Mapping[str, np.ndarray],
) -> str:
    contract = {
        "schema_version": ACTION_VALUE_ARCHITECTURE_SCHEMA_VERSION,
        "spec": asdict(spec),
        "feature_version": MICROSTRUCTURE_FEATURE_VERSION,
        "feature_names": list(MICROSTRUCTURE_FEATURE_NAMES),
        "target_schema_version": target_schema_version,
        "target_mode": ADAPTIVE_BARRIER_TARGET_MODE,
        "target_contract_sha256": target_contract_sha256,
        "target_scenario": target_scenario,
        "target_scale_bps": float(target_scale_bps),
        "positive_class_weights": list(positive_class_weights),
        "optimizer_kind": optimizer_kind,
        "optimizer_hyperparameters": dict(optimizer_hyperparameters),
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


def train_action_value_model(
    dataset: MicrostructureDataset,
    barrier_targets: AdaptiveBarrierTargets,
    *,
    train_endpoints: np.ndarray,
    tuning_endpoints: np.ndarray,
    spec: ActionValueArchitectureSpec,
    target_scenario: str,
    compute_backend: str,
    seed: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    train_sample_weights: np.ndarray | None = None,
    tuning_sample_weights: np.ndarray | None = None,
    progress: Callable[[int, int, float, float], None] | None = None,
) -> TrainedActionValueModel:
    """Fit a research-only shared long/short action-value model."""

    validate_microstructure_dataset(dataset)
    validate_adaptive_barrier_targets(dataset, barrier_targets)
    if target_scenario not in {"base", "stress"}:
        raise ValueError("action-value target scenario is unsupported")
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
        raise ValueError("action-value split has insufficient contiguous rows")
    if train[-1] >= tuning[0]:
        raise ValueError("action-value train/tuning split is not chronological")
    if not np.all(np.isfinite(targets[train])) or not np.all(
        np.isfinite(targets[tuning])
    ):
        raise ValueError("action-value split targets are non-finite")
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
        raise ValueError("action-value split is not purged across target lifecycles")
    batch = int(batch_size)
    epochs = int(max_epochs)
    stop_patience = int(patience)
    if batch < 32 or epochs < 1 or stop_patience < 1:
        raise ValueError("action-value training budget is invalid")
    center, scale = _feature_scaler(dataset.features, train)
    target_scale = max(1.0, float(np.quantile(np.abs(targets[train]), 0.90)))
    class_weights = _positive_class_weights(targets[train])
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
        raise ValueError("action-value sample weights are invalid")
    backend = require_backend(resolve_backend(compute_backend))
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
    class_weight = torch.tensor(class_weights, dtype=torch.float32).to(device)
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
                loss = _loss(network(x), y, weight, class_weight, spec)
                if not bool(torch.isfinite(loss).detach().cpu().item()):
                    raise ValueError("action-value loss became non-finite")
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
    model_sha256 = _state_hash(
        spec=spec,
        target_schema_version=barrier_targets.schema_version,
        target_contract_sha256=target_contract_sha256,
        target_scenario=target_scenario,
        center=center,
        scale=scale,
        target_scale_bps=target_scale,
        positive_class_weights=class_weights,
        optimizer_kind=optimizer.kind,
        optimizer_hyperparameters=optimizer.hyperparameters,
        state=state,
    )
    return TrainedActionValueModel(
        schema_version=ACTION_VALUE_ARCHITECTURE_SCHEMA_VERSION,
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
        training_data_mode=(
            "device_preloaded" if preloaded else "streamed_host_batches"
        ),
        training_preload_bytes=preload_bytes if preloaded else 0,
        sequence_length=spec.sequence_length,
        target_scale_bps=target_scale,
        positive_class_weights=class_weights,
        scaler_center=center,
        scaler_scale=scale,
        best_epoch=best_epoch,
        training_loss=best_training,
        tuning_loss=best_tuning,
        state=state,
        model_sha256=model_sha256,
    )


def predict_action_value_model(
    model: TrainedActionValueModel,
    dataset: MicrostructureDataset,
    endpoints: np.ndarray,
    *,
    compute_backend: str,
    batch_size: int,
) -> ActionValuePredictionBatch:
    validate_microstructure_dataset(dataset)
    expected_target_contract = hashlib.sha256(
        _canonical_json(
            {
                "schema_version": model.target_schema_version,
                "target_mode": model.target_mode,
                "spec": asdict(model.target_spec),
            }
        ).encode("ascii")
    ).hexdigest()
    expected_model_sha256 = _state_hash(
        spec=model.spec,
        target_schema_version=model.target_schema_version,
        target_contract_sha256=model.target_contract_sha256,
        target_scenario=model.target_scenario,
        center=model.scaler_center,
        scale=model.scaler_scale,
        target_scale_bps=model.target_scale_bps,
        positive_class_weights=model.positive_class_weights,
        optimizer_kind=model.optimizer_kind,
        optimizer_hyperparameters=model.optimizer_hyperparameters,
        state=model.state,
    )
    if (
        model.trading_authority
        or model.execution_claim
        or model.profitability_claim
        or model.portfolio_claim
        or model.leverage_applied
        or model.schema_version != ACTION_VALUE_ARCHITECTURE_SCHEMA_VERSION
        or model.feature_version != dataset.feature_version
        or model.feature_names != dataset.feature_names
        or model.target_mode != ADAPTIVE_BARRIER_TARGET_MODE
        or expected_target_contract != model.target_contract_sha256
        or expected_model_sha256 != model.model_sha256
        or model.target_scenario not in {"base", "stress"}
        or model.scaler_center.shape != (len(model.feature_names),)
        or model.scaler_scale.shape != (len(model.feature_names),)
        or not np.all(np.isfinite(model.scaler_center))
        or not np.all(np.isfinite(model.scaler_scale))
        or np.any(model.scaler_scale <= 0.0)
        or not math.isfinite(model.target_scale_bps)
        or model.target_scale_bps <= 0.0
    ):
        raise ValueError("action-value neural model contract is invalid")
    if int(batch_size) <= 0:
        raise ValueError("action-value prediction batch size must be positive")
    selected = valid_sequence_endpoints(
        dataset.decision_time_ms,
        np.asarray(endpoints, dtype=np.int64),
        sequence_length=model.sequence_length,
        cadence_seconds=dataset.decision_cadence_seconds,
    )
    if selected.size == 0:
        raise ValueError("action-value prediction has no contiguous endpoints")
    backend = require_backend(resolve_backend(compute_backend))
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
    raw_output = np.concatenate(outputs, axis=0).astype(np.float64)
    if raw_output.shape != (len(selected), 2, 4) or not np.all(np.isfinite(raw_output)):
        raise ValueError("action-value neural model emitted invalid predictions")
    mean = raw_output[:, :, 0]
    lower = mean - np.logaddexp(0.0, raw_output[:, :, 2])
    upper = mean + np.logaddexp(0.0, raw_output[:, :, 3])
    probability = 1.0 / (1.0 + np.exp(-np.clip(raw_output[:, :, 1], -40.0, 40.0)))
    scale = model.target_scale_bps
    return ActionValuePredictionBatch(
        endpoint_indexes=selected,
        long_mean_bps=mean[:, 0] * scale,
        short_mean_bps=mean[:, 1] * scale,
        long_profitable_probability=probability[:, 0],
        short_profitable_probability=probability[:, 1],
        long_lower_bps=lower[:, 0] * scale,
        short_lower_bps=lower[:, 1] * scale,
        long_upper_bps=upper[:, 0] * scale,
        short_upper_bps=upper[:, 1] * scale,
    )


def ensemble_action_value_predictions(
    members: Sequence[ActionValuePredictionBatch],
) -> ActionValueEnsembleBatch:
    values = tuple(members)
    if len(values) < 2:
        raise ValueError("action-value ensemble requires at least two members")
    endpoints = np.asarray(values[0].endpoint_indexes, dtype=np.int64)
    fields = (
        "long_mean_bps",
        "short_mean_bps",
        "long_profitable_probability",
        "short_profitable_probability",
        "long_lower_bps",
        "short_lower_bps",
        "long_upper_bps",
        "short_upper_bps",
    )
    if endpoints.size == 0 or any(
        not np.array_equal(endpoints, value.endpoint_indexes) for value in values[1:]
    ):
        raise ValueError("action-value ensemble endpoint identities differ")
    stacks = {
        field: np.stack(
            [np.asarray(getattr(value, field), dtype=np.float64) for value in values],
            axis=0,
        )
        for field in fields
    }
    if any(
        stack.shape != (len(values), len(endpoints)) or not np.all(np.isfinite(stack))
        for stack in stacks.values()
    ) or (
        np.any(np.diff(endpoints) <= 0)
        or np.any(stacks["long_profitable_probability"] < 0.0)
        or np.any(stacks["long_profitable_probability"] > 1.0)
        or np.any(stacks["short_profitable_probability"] < 0.0)
        or np.any(stacks["short_profitable_probability"] > 1.0)
        or np.any(stacks["long_lower_bps"] > stacks["long_upper_bps"])
        or np.any(stacks["short_lower_bps"] > stacks["short_upper_bps"])
    ):
        raise ValueError("action-value ensemble member arrays are invalid")
    router_members = tuple(
        (value.long_router_weights, value.short_router_weights) for value in values
    )
    if any(
        (long_weights is None) != (short_weights is None)
        for long_weights, short_weights in router_members
    ):
        raise ValueError("action-value ensemble router evidence is incomplete")
    router_presence = tuple(
        long_weights is not None and short_weights is not None
        for long_weights, short_weights in router_members
    )
    if any(router_presence) and not all(router_presence):
        raise ValueError("action-value ensemble router evidence differs")
    long_router_weights = None
    short_router_weights = None
    if all(router_presence):
        long_router_stack = np.stack(
            [np.asarray(value[0], dtype=np.float64) for value in router_members],
            axis=0,
        )
        short_router_stack = np.stack(
            [np.asarray(value[1], dtype=np.float64) for value in router_members],
            axis=0,
        )
        if (
            long_router_stack.ndim != 3
            or short_router_stack.shape != long_router_stack.shape
            or long_router_stack.shape[:2] != (len(values), len(endpoints))
            or long_router_stack.shape[2] < 2
            or not np.all(np.isfinite(long_router_stack))
            or not np.all(np.isfinite(short_router_stack))
            or np.any(long_router_stack < 0.0)
            or np.any(short_router_stack < 0.0)
            or not np.allclose(np.sum(long_router_stack, axis=2), 1.0, atol=1.0e-6)
            or not np.allclose(np.sum(short_router_stack, axis=2), 1.0, atol=1.0e-6)
        ):
            raise ValueError("action-value ensemble router evidence is invalid")
        long_router_weights = np.mean(long_router_stack, axis=0)
        short_router_weights = np.mean(short_router_stack, axis=0)
    return ActionValueEnsembleBatch(
        endpoint_indexes=endpoints.copy(),
        long_mean_bps=np.mean(stacks["long_mean_bps"], axis=0),
        short_mean_bps=np.mean(stacks["short_mean_bps"], axis=0),
        long_epistemic_std_bps=np.std(stacks["long_mean_bps"], axis=0),
        short_epistemic_std_bps=np.std(stacks["short_mean_bps"], axis=0),
        long_profitable_probability=np.mean(
            stacks["long_profitable_probability"], axis=0
        ),
        short_profitable_probability=np.mean(
            stacks["short_profitable_probability"], axis=0
        ),
        long_lower_bps=np.mean(stacks["long_lower_bps"], axis=0),
        short_lower_bps=np.mean(stacks["short_lower_bps"], axis=0),
        long_upper_bps=np.mean(stacks["long_upper_bps"], axis=0),
        short_upper_bps=np.mean(stacks["short_upper_bps"], axis=0),
        long_positive_member_ratio=np.mean(stacks["long_mean_bps"] > 0.0, axis=0),
        short_positive_member_ratio=np.mean(stacks["short_mean_bps"] > 0.0, axis=0),
        member_count=len(values),
        long_router_weights=long_router_weights,
        short_router_weights=short_router_weights,
    )


__all__ = [
    "ACTION_VALUE_ARCHITECTURE_SCHEMA_VERSION",
    "ActionValueArchitectureSpec",
    "ActionValueEnsembleBatch",
    "ActionValuePredictionBatch",
    "TrainedActionValueModel",
    "ensemble_action_value_predictions",
    "predict_action_value_model",
    "train_action_value_model",
]
