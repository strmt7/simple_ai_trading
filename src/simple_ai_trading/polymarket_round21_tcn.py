"""Compact causal sequence challenger for Polymarket Round 21."""

from __future__ import annotations

import base64
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import math
import time
from typing import Any

import numpy as np

from .compute import BackendInfo, resolve_backend, torch_device_for_backend


ROUND21_TCN_SEQUENCE_LENGTH = 16
ROUND21_TCN_ENDPOINTS_PER_CONDITION = 8
ROUND21_TCN_HIDDEN_CHANNELS = 12
ROUND21_TCN_DILATIONS = (1, 2, 4)
ROUND21_TCN_KERNEL_SIZE = 3
ROUND21_TCN_MAXIMUM_EPOCHS = 32
ROUND21_TCN_EARLY_STOPPING_PATIENCE = 5
ROUND21_TCN_CONDITIONS_PER_BATCH = 16
ROUND21_TCN_PREDICTION_BATCH_SIZE = 1_024
ROUND21_TCN_LEARNING_RATE = 0.001
ROUND21_TCN_WEIGHT_DECAY = 0.0001
ROUND21_TCN_GRADIENT_CLIP_NORM = 1.0
ROUND21_TCN_ADAM_BETAS = (0.9, 0.999)
ROUND21_TCN_ADAM_EPSILON = 1e-8
ROUND21_TCN_ARCHITECTURE = {
    "sequence_length": ROUND21_TCN_SEQUENCE_LENGTH,
    "decision_cadence_ms": 250,
    "endpoint_sampling": "eight_midpoint_stratified_rows_per_condition",
    "history_padding": "left_zero_after_train_only_normalization_plus_mask",
    "history_reset": "condition_change_or_non_250ms_cadence",
    "projection": "pointwise_conv1d_relu",
    "hidden_channels": ROUND21_TCN_HIDDEN_CHANNELS,
    "blocks": "causal_depthwise_conv1d_pointwise_conv1d_relu_residual",
    "kernel_size": ROUND21_TCN_KERNEL_SIZE,
    "dilations": list(ROUND21_TCN_DILATIONS),
    "head": "last_valid_step_linear_residual_logit",
    "structural_prior": "added_as_frozen_log_odds_offset",
}

ProgressCallback = Callable[[str, Mapping[str, object]], None]


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional runtime boundary
        raise RuntimeError(
            "Round 21 TCN requires the gpu or directml package extra"
        ) from exc
    return torch


def _groups(condition_ids: np.ndarray) -> tuple[tuple[int, int], ...]:
    selected = np.asarray(condition_ids, dtype=object)
    if selected.ndim != 1 or len(selected) < 1:
        raise ValueError("Round 21 TCN condition identities are invalid")
    boundaries = np.flatnonzero(selected[1:] != selected[:-1]) + 1
    starts = np.concatenate((np.asarray([0]), boundaries))
    ends = np.concatenate((boundaries, np.asarray([len(selected)])))
    groups = tuple((int(start), int(end)) for start, end in zip(starts, ends, strict=True))
    identities = tuple(str(selected[start]) for start, _end in groups)
    if len(set(identities)) != len(identities):
        raise ValueError("Round 21 TCN condition identities are not contiguous")
    return groups


def _validate_arrays(
    matrix: np.ndarray,
    labels: np.ndarray,
    structural_log_odds: np.ndarray,
    condition_ids: np.ndarray,
    decision_time_ms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float32, order="C")
    target = np.asarray(labels, dtype=np.float32)
    offset = np.asarray(structural_log_odds, dtype=np.float32)
    conditions = np.asarray(condition_ids, dtype=object)
    decisions = np.asarray(decision_time_ms, dtype=np.int64)
    rows = len(values)
    if (
        values.ndim != 2
        or values.shape[1] < 1
        or target.shape != (rows,)
        or offset.shape != (rows,)
        or conditions.shape != (rows,)
        or decisions.shape != (rows,)
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(target))
        or not np.all(np.isin(target, (0.0, 1.0)))
        or not np.all(np.isfinite(offset))
    ):
        raise ValueError("Round 21 TCN arrays are invalid")
    groups = _groups(conditions)
    for start, end in groups:
        if (
            np.any(target[start:end] != target[start])
            or np.any(decisions[start + 1 : end] <= decisions[start : end - 1])
        ):
            raise ValueError("Round 21 TCN condition rows are invalid")
    return values, target, offset, conditions, decisions


def _history_starts(condition_ids: np.ndarray, decision_time_ms: np.ndarray) -> np.ndarray:
    starts = np.empty(len(condition_ids), dtype=np.int64)
    current = 0
    for index in range(len(condition_ids)):
        if index == 0 or (
            condition_ids[index] != condition_ids[index - 1]
            or int(decision_time_ms[index]) - int(decision_time_ms[index - 1]) != 250
        ):
            current = index
        starts[index] = current
    return starts


def _condition_endpoints(condition_ids: np.ndarray) -> tuple[np.ndarray, ...]:
    output: list[np.ndarray] = []
    for start, end in _groups(condition_ids):
        count = min(ROUND21_TCN_ENDPOINTS_PER_CONDITION, end - start)
        width = end - start
        endpoints = np.asarray(
            [start + min(width - 1, ((2 * index + 1) * width) // (2 * count)) for index in range(count)],
            dtype=np.int64,
        )
        if len(set(endpoints.tolist())) != count:
            raise RuntimeError("Round 21 TCN endpoint sampling differs")
        output.append(endpoints)
    return tuple(output)


def build_round21_tcn_sequences(
    matrix: np.ndarray,
    condition_ids: np.ndarray,
    decision_time_ms: np.ndarray,
    endpoints: np.ndarray,
) -> np.ndarray:
    """Build fixed-length causal sequences without crossing a cadence gap."""

    values = np.asarray(matrix, dtype=np.float32, order="C")
    conditions = np.asarray(condition_ids, dtype=object)
    decisions = np.asarray(decision_time_ms, dtype=np.int64)
    selected = np.asarray(endpoints, dtype=np.int64)
    if (
        values.ndim != 2
        or conditions.shape != (len(values),)
        or decisions.shape != (len(values),)
        or selected.ndim != 1
        or np.any(selected < 0)
        or np.any(selected >= len(values))
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("Round 21 TCN sequence request is invalid")
    history_starts = _history_starts(conditions, decisions)
    output = np.zeros(
        (len(selected), ROUND21_TCN_SEQUENCE_LENGTH, values.shape[1] + 1),
        dtype=np.float32,
    )
    lags = np.arange(
        ROUND21_TCN_SEQUENCE_LENGTH - 1,
        -1,
        -1,
        dtype=np.int64,
    )
    source_indices = selected[:, None] - lags[None, :]
    valid = source_indices >= history_starts[selected, None]
    safe_indices = np.maximum(source_indices, 0)
    output[..., :-1] = values[safe_indices]
    output[..., :-1] *= valid[..., None]
    output[..., -1] = valid
    return output


def round21_tcn_parameter_count(feature_width: int) -> int:
    width = int(feature_width)
    if width < 1:
        raise ValueError("Round 21 TCN feature width is invalid")
    hidden = ROUND21_TCN_HIDDEN_CHANNELS
    projection = hidden * (width + 1) + hidden
    per_block = (
        hidden * ROUND21_TCN_KERNEL_SIZE
        + hidden
        + hidden * hidden
        + hidden
    )
    head = hidden + 1
    return projection + len(ROUND21_TCN_DILATIONS) * per_block + head


def validate_round21_tcn_payload(
    payload: Mapping[str, object],
    *,
    feature_width: int,
) -> bool:
    state_keys = {
        "architecture",
        "state_base64",
        "state_sha256",
        "parameter_count",
        "best_epoch",
        "epochs_run",
        "best_stop_condition_equal_log_loss",
        "training_condition_count",
        "stop_condition_count",
        "backend_kind",
        "backend_device",
        "torch_version",
    }
    model_keys = state_keys | {
        "candidate_id",
        "family",
        "layer",
        "population_layer",
        "feature_layer",
        "feature_names_sha256",
        "transform",
        "calibration",
    }
    if set(payload) not in (state_keys, model_keys - {"calibration"}, model_keys):
        return False
    try:
        encoded = str(payload["state_base64"])
        state = base64.b64decode(encoded, validate=True)
        state_sha = str(payload["state_sha256"])
        parameter_count = int(payload["parameter_count"])
        best_epoch = int(payload["best_epoch"])
        epochs_run = int(payload["epochs_run"])
        stop_log_loss = float(payload["best_stop_condition_equal_log_loss"])
        train_conditions = int(payload["training_condition_count"])
        stop_conditions = int(payload["stop_condition_count"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
    expected_parameters = round21_tcn_parameter_count(feature_width)
    return bool(
        payload.get("architecture") == ROUND21_TCN_ARCHITECTURE
        and parameter_count == expected_parameters
        and len(state) == expected_parameters * 4
        and len(state_sha) == 64
        and all(character in "0123456789abcdef" for character in state_sha)
        and hashlib.sha256(state).hexdigest() == state_sha
        and 1 <= best_epoch <= epochs_run <= ROUND21_TCN_MAXIMUM_EPOCHS
        and math.isfinite(stop_log_loss)
        and stop_log_loss >= 0.0
        and train_conditions >= 30
        and stop_conditions >= 30
        and payload.get("backend_kind")
        in {"cpu", "cuda", "rocm", "xpu", "directml", "mps"}
        and bool(str(payload.get("backend_device") or "").strip())
        and 1 <= len(str(payload.get("torch_version") or "")) <= 64
    )


def _model(feature_width: int) -> Any:
    torch = _torch()
    nn = torch.nn

    class CausalBlock(nn.Module):
        def __init__(self, dilation: int) -> None:
            super().__init__()
            self.trim = (ROUND21_TCN_KERNEL_SIZE - 1) * dilation
            self.depthwise = nn.Conv1d(
                ROUND21_TCN_HIDDEN_CHANNELS,
                ROUND21_TCN_HIDDEN_CHANNELS,
                ROUND21_TCN_KERNEL_SIZE,
                padding=self.trim,
                dilation=dilation,
                groups=ROUND21_TCN_HIDDEN_CHANNELS,
            )
            self.pointwise = nn.Conv1d(
                ROUND21_TCN_HIDDEN_CHANNELS,
                ROUND21_TCN_HIDDEN_CHANNELS,
                1,
            )
            self.activation = nn.ReLU()

        def forward(self, values: Any) -> Any:
            convolved = self.depthwise(values)
            if self.trim:
                convolved = convolved[..., : -self.trim]
            return self.activation(values + self.pointwise(self.activation(convolved)))

    class Round21CausalTCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Conv1d(
                feature_width + 1,
                ROUND21_TCN_HIDDEN_CHANNELS,
                1,
            )
            self.activation = nn.ReLU()
            self.blocks = nn.ModuleList(
                CausalBlock(dilation) for dilation in ROUND21_TCN_DILATIONS
            )
            self.head = nn.Linear(ROUND21_TCN_HIDDEN_CHANNELS, 1)

        def forward(self, values: Any) -> Any:
            encoded = self.activation(self.projection(values.transpose(1, 2)))
            for block in self.blocks:
                encoded = block(encoded)
            return self.head(encoded[..., -1]).squeeze(-1)

    return Round21CausalTCN()


def _state_bytes(model: Any) -> bytes:
    chunks: list[bytes] = []
    for name, tensor in sorted(model.state_dict().items()):
        if not name:
            raise RuntimeError("Round 21 TCN state name is invalid")
        array = np.asarray(tensor.detach().cpu().numpy(), dtype="<f4", order="C")
        if not np.all(np.isfinite(array)):
            raise RuntimeError("Round 21 TCN state is nonfinite")
        chunks.append(array.tobytes(order="C"))
    return b"".join(chunks)


def _adamw_step(
    parameters: tuple[Any, ...],
    first_moments: tuple[Any, ...],
    second_moments: tuple[Any, ...],
    *,
    step: int,
) -> None:
    torch = _torch()
    beta1, beta2 = ROUND21_TCN_ADAM_BETAS
    first_correction = 1.0 - beta1**step
    second_correction = 1.0 - beta2**step
    step_size = ROUND21_TCN_LEARNING_RATE / first_correction
    with torch.no_grad():
        for parameter, first, second in zip(
            parameters,
            first_moments,
            second_moments,
            strict=True,
        ):
            gradient = parameter.grad
            if gradient is None:
                raise RuntimeError("Round 21 TCN gradient is unavailable")
            parameter.mul_(1.0 - ROUND21_TCN_LEARNING_RATE * ROUND21_TCN_WEIGHT_DECAY)
            first.mul_(beta1).add_((1.0 - beta1) * gradient)
            second.mul_(beta2).add_((1.0 - beta2) * gradient * gradient)
            denominator = torch.sqrt(second / second_correction) + ROUND21_TCN_ADAM_EPSILON
            parameter.add_(-step_size * first / denominator)


def _load_state_bytes(model: Any, state: bytes) -> None:
    torch = _torch()
    offset = 0
    restored: dict[str, Any] = {}
    for name, tensor in sorted(model.state_dict().items()):
        count = tensor.numel()
        size = count * 4
        array = np.frombuffer(state[offset : offset + size], dtype="<f4").copy()
        if len(array) != count:
            raise ValueError("Round 21 TCN state length differs")
        restored[name] = torch.from_numpy(array.reshape(tuple(tensor.shape)))
        offset += size
    if offset != len(state):
        raise ValueError("Round 21 TCN state length differs")
    model.load_state_dict(restored, strict=True)


def _predict(
    model: Any,
    matrix: np.ndarray,
    condition_ids: np.ndarray,
    decision_time_ms: np.ndarray,
    structural_log_odds: np.ndarray,
    *,
    device: object,
    endpoints: np.ndarray,
) -> np.ndarray:
    torch = _torch()
    output = np.empty(len(endpoints), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(endpoints), ROUND21_TCN_PREDICTION_BATCH_SIZE):
            stop = min(len(endpoints), start + ROUND21_TCN_PREDICTION_BATCH_SIZE)
            selected = endpoints[start:stop]
            sequences = build_round21_tcn_sequences(
                matrix,
                condition_ids,
                decision_time_ms,
                selected,
            )
            residual = model(torch.from_numpy(sequences).to(device))
            probability = torch.sigmoid(
                residual
                + torch.from_numpy(
                    np.asarray(structural_log_odds[selected], dtype=np.float32)
                ).to(device)
            )
            output[start:stop] = np.asarray(
                probability.detach().cpu().numpy(),
                dtype=np.float64,
            )
    if not np.all(np.isfinite(output)):
        raise RuntimeError("Round 21 TCN prediction is nonfinite")
    return np.clip(output, 1e-6, 1.0 - 1e-6)


def _condition_equal_log_loss(
    labels: np.ndarray,
    predictions: np.ndarray,
    condition_ids: np.ndarray,
) -> float:
    losses: list[float] = []
    for start, end in _groups(condition_ids):
        target = labels[start:end]
        probability = predictions[start:end]
        losses.append(
            float(
                -np.mean(
                    target * np.log(probability)
                    + (1.0 - target) * np.log1p(-probability)
                )
            )
        )
    return float(np.mean(losses))


@dataclass(frozen=True, slots=True)
class Round21TCNFit:
    payload: dict[str, object]


def fit_round21_tcn(
    *,
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    train_structural_log_odds: np.ndarray,
    train_condition_ids: np.ndarray,
    train_decision_time_ms: np.ndarray,
    stop_matrix: np.ndarray,
    stop_labels: np.ndarray,
    stop_structural_log_odds: np.ndarray,
    stop_condition_ids: np.ndarray,
    stop_decision_time_ms: np.ndarray,
    backend: BackendInfo,
    seed: int,
    progress: ProgressCallback | None = None,
) -> Round21TCNFit:
    """Fit one preregistered low-capacity residual TCN."""

    torch = _torch()
    train = _validate_arrays(
        train_matrix,
        train_labels,
        train_structural_log_odds,
        train_condition_ids,
        train_decision_time_ms,
    )
    stop = _validate_arrays(
        stop_matrix,
        stop_labels,
        stop_structural_log_odds,
        stop_condition_ids,
        stop_decision_time_ms,
    )
    feature_width = train[0].shape[1]
    if stop[0].shape[1] != feature_width:
        raise ValueError("Round 21 TCN feature width differs")
    if set(train[1].tolist()) != {0.0, 1.0} or set(stop[1].tolist()) != {0.0, 1.0}:
        raise ValueError("Round 21 TCN target population is single-class")
    device = torch_device_for_backend(backend)
    torch.manual_seed(int(seed))
    model = _model(feature_width).to(device)
    if sum(parameter.numel() for parameter in model.parameters()) != round21_tcn_parameter_count(feature_width):
        raise RuntimeError("Round 21 TCN parameter count differs")
    parameters = tuple(model.parameters())
    first_moments = tuple(torch.zeros_like(parameter) for parameter in parameters)
    second_moments = tuple(torch.zeros_like(parameter) for parameter in parameters)
    optimizer_step = 0
    train_endpoints = _condition_endpoints(train[3])
    stop_endpoint_groups = _condition_endpoints(stop[3])
    stop_endpoints = np.concatenate(stop_endpoint_groups)
    best_state: bytes | None = None
    best_loss = math.inf
    best_epoch = 0
    stale_epochs = 0
    epochs_run = 0
    generator = np.random.default_rng(int(seed))
    for epoch in range(1, ROUND21_TCN_MAXIMUM_EPOCHS + 1):
        epochs_run = epoch
        order = generator.permutation(len(train_endpoints))
        model.train()
        epoch_loss_sum = 0.0
        batch_count = 0
        batches_in_epoch = math.ceil(
            len(order) / ROUND21_TCN_CONDITIONS_PER_BATCH
        )
        last_batch_heartbeat = time.perf_counter()
        for batch_start in range(0, len(order), ROUND21_TCN_CONDITIONS_PER_BATCH):
            condition_selection = order[
                batch_start : batch_start + ROUND21_TCN_CONDITIONS_PER_BATCH
            ]
            endpoint_groups = [train_endpoints[int(index)] for index in condition_selection]
            endpoints = np.concatenate(endpoint_groups)
            weights = np.concatenate(
                [
                    np.full(len(group), 1.0 / len(group), dtype=np.float32)
                    for group in endpoint_groups
                ]
            )
            sequences = build_round21_tcn_sequences(
                train[0],
                train[3],
                train[4],
                endpoints,
            )
            values = torch.from_numpy(sequences).to(device)
            target = torch.from_numpy(train[1][endpoints]).to(device)
            offset = torch.from_numpy(train[2][endpoints]).to(device)
            sample_weight = torch.from_numpy(weights).to(device)
            for parameter in parameters:
                parameter.grad = None
            logits = offset + model(values)
            positive = torch.maximum(logits, torch.zeros_like(logits))
            losses = (
                positive
                - logits * target
                + torch.log1p(torch.exp(-torch.abs(logits)))
            )
            loss = torch.sum(losses * sample_weight) / float(
                ROUND21_TCN_CONDITIONS_PER_BATCH
            )
            if not bool(torch.isfinite(loss).detach().cpu().item()):
                raise RuntimeError("Round 21 TCN training loss is nonfinite")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                parameters,
                ROUND21_TCN_GRADIENT_CLIP_NORM,
            )
            optimizer_step += 1
            _adamw_step(
                parameters,
                first_moments,
                second_moments,
                step=optimizer_step,
            )
            epoch_loss_sum += float(loss.detach().cpu().item())
            batch_count += 1
            heartbeat_now = time.perf_counter()
            if progress is not None and (
                heartbeat_now - last_batch_heartbeat >= 30.0
                or batch_count == batches_in_epoch
            ):
                progress(
                    "round21_tcn_batch",
                    {
                        "epoch": epoch,
                        "maximum_epochs": ROUND21_TCN_MAXIMUM_EPOCHS,
                        "batch": batch_count,
                        "batches_in_epoch": batches_in_epoch,
                        "latest_training_batch_loss": float(
                            loss.detach().cpu().item()
                        ),
                        "backend_kind": backend.kind,
                        "backend_device": backend.device,
                    },
                )
                last_batch_heartbeat = heartbeat_now
        stop_endpoint_predictions = _predict(
            model,
            stop[0],
            stop[3],
            stop[4],
            stop[2],
            device=device,
            endpoints=stop_endpoints,
        )
        selected_labels = stop[1][stop_endpoints]
        selected_conditions = stop[3][stop_endpoints]
        stop_loss = _condition_equal_log_loss(
            selected_labels,
            stop_endpoint_predictions,
            selected_conditions,
        )
        improved = stop_loss < best_loss - 1e-7
        if improved:
            best_loss = stop_loss
            best_epoch = epoch
            best_state = _state_bytes(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
        if progress is not None:
            progress(
                "round21_tcn_epoch",
                {
                    "epoch": epoch,
                    "maximum_epochs": ROUND21_TCN_MAXIMUM_EPOCHS,
                    "mean_training_batch_loss": epoch_loss_sum / max(1, batch_count),
                    "stop_condition_equal_log_loss": stop_loss,
                    "best_epoch": best_epoch,
                    "best_stop_condition_equal_log_loss": best_loss,
                    "stale_epochs": stale_epochs,
                    "backend_kind": backend.kind,
                    "backend_device": backend.device,
                },
            )
        if stale_epochs >= ROUND21_TCN_EARLY_STOPPING_PATIENCE:
            break
    if best_state is None or best_epoch < 1 or not math.isfinite(best_loss):
        raise RuntimeError("Round 21 TCN did not produce a finite model")
    _load_state_bytes(model, best_state)
    reloaded = _model(feature_width)
    _load_state_bytes(reloaded, best_state)
    reloaded_state = _state_bytes(reloaded)
    if reloaded_state != best_state:
        raise RuntimeError("Round 21 TCN serialization identity failed")
    payload: dict[str, object] = {
        "architecture": dict(ROUND21_TCN_ARCHITECTURE),
        "state_base64": base64.b64encode(best_state).decode("ascii"),
        "state_sha256": hashlib.sha256(best_state).hexdigest(),
        "parameter_count": round21_tcn_parameter_count(feature_width),
        "best_epoch": best_epoch,
        "epochs_run": epochs_run,
        "best_stop_condition_equal_log_loss": best_loss,
        "training_condition_count": len(train_endpoints),
        "stop_condition_count": len(stop_endpoint_groups),
        "backend_kind": backend.kind,
        "backend_device": backend.device,
        "torch_version": str(torch.__version__),
    }
    if not validate_round21_tcn_payload(payload, feature_width=feature_width):
        raise RuntimeError("Round 21 TCN artifact validation failed")
    return Round21TCNFit(payload=payload)


def predict_round21_tcn(
    payload: Mapping[str, object],
    *,
    matrix: np.ndarray,
    structural_log_odds: np.ndarray,
    condition_ids: np.ndarray,
    decision_time_ms: np.ndarray,
) -> np.ndarray:
    """Run stored weights on their training backend or a portable CPU fallback."""

    values = np.asarray(matrix, dtype=np.float32, order="C")
    labels = np.zeros(len(values), dtype=np.float32)
    validated = _validate_arrays(
        values,
        labels,
        structural_log_odds,
        condition_ids,
        decision_time_ms,
    )
    if not validate_round21_tcn_payload(payload, feature_width=values.shape[1]):
        raise ValueError("Round 21 TCN payload differs")
    state = base64.b64decode(str(payload["state_base64"]), validate=True)
    backend = resolve_backend(str(payload["backend_kind"]), require=False)
    device = torch_device_for_backend(backend)
    model = _model(values.shape[1])
    _load_state_bytes(model, state)
    model = model.to(device)
    endpoints = np.arange(len(values), dtype=np.int64)
    return _predict(
        model,
        validated[0],
        validated[3],
        validated[4],
        validated[2],
        device=device,
        endpoints=endpoints,
    )


__all__ = [
    "ROUND21_TCN_ARCHITECTURE",
    "ROUND21_TCN_MAXIMUM_EPOCHS",
    "Round21TCNFit",
    "build_round21_tcn_sequences",
    "fit_round21_tcn",
    "predict_round21_tcn",
    "round21_tcn_parameter_count",
    "validate_round21_tcn_payload",
]
