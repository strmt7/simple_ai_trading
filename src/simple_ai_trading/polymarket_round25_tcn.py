"""Causal multitask TCN mechanics and artifacts for Round 25."""

from __future__ import annotations

import base64
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
import time
from typing import Any
import warnings

import numpy as np

from .compute import resolve_backend, torch_device_for_backend
from .polymarket_round25_candidate_design import (
    POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
    POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
)
from .polymarket_round25_controls import (
    POLYMARKET_ROUND25_CONTROL_FIT_CONTRACT_SHA256,
    POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND,
)
from .polymarket_round25_dataset import POLYMARKET_ROUND25_MINIMUM_CONDITIONS
from .polymarket_round25_sequence import (
    POLYMARKET_ROUND25_AUXILIARY_HORIZONS_MS,
    POLYMARKET_ROUND25_SEQUENCE_INPUT_WIDTH,
    POLYMARKET_ROUND25_SEQUENCE_MATERIALIZATION_CONTRACT_SHA256,
    POLYMARKET_ROUND25_SEQUENCE_ROWS,
    Round25SequenceCollation,
    Round25SequenceConditionBatch,
    collate_round25_sequence_batches,
)
from .polymarket_round25_twap_features import (
    POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
)


POLYMARKET_ROUND25_TCN_FIT_CONTRACT_SHA256 = (
    "abe1f93ccab38172e400286601e6fa9befaadf07acb448d0d2604314c2c59876"
)
POLYMARKET_ROUND25_TCN_SEED_ARTIFACT_SCHEMA_VERSION = (
    "polymarket-round25-causal-multitask-tcn-seed-artifact-v1"
)
POLYMARKET_ROUND25_TCN_CORPUS_SCHEMA_VERSION = (
    "polymarket-round25-causal-sequence-corpus-source-v1"
)
POLYMARKET_ROUND25_TCN_ENSEMBLE_ARTIFACT_SCHEMA_VERSION = (
    "polymarket-round25-causal-multitask-tcn-ensemble-artifact-v1"
)
POLYMARKET_ROUND25_TCN_HIDDEN_CHANNELS = 32
POLYMARKET_ROUND25_TCN_KERNEL_SIZE = 3
POLYMARKET_ROUND25_TCN_DILATIONS = (1, 2, 4, 8)
POLYMARKET_ROUND25_TCN_CONVOLUTIONS_PER_BLOCK = 2
POLYMARKET_ROUND25_TCN_DROPOUT = 0.1
POLYMARKET_ROUND25_TCN_MAXIMUM_EPOCHS = 64
POLYMARKET_ROUND25_TCN_EARLY_STOPPING_PATIENCE = 8
POLYMARKET_ROUND25_TCN_LEARNING_RATE = 0.001
POLYMARKET_ROUND25_TCN_WEIGHT_DECAY = 0.0001
POLYMARKET_ROUND25_TCN_ADAM_BETAS = (0.9, 0.999)
POLYMARKET_ROUND25_TCN_ADAM_EPSILON = 1e-8
POLYMARKET_ROUND25_TCN_GRADIENT_CLIP_NORM = 1.0
POLYMARKET_ROUND25_TCN_AUXILIARY_WEIGHT = 0.1
POLYMARKET_ROUND25_TCN_MINIMUM_LOG_LOSS_IMPROVEMENT = 1e-7
POLYMARKET_ROUND25_TCN_PROGRESS_HEARTBEAT_SECONDS = 30.0
POLYMARKET_ROUND25_TCN_TRAINING_SEEDS = (1729, 3253, 7919)
POLYMARKET_ROUND25_TCN_ARCHITECTURE = {
    "sequence_rows": POLYMARKET_ROUND25_SEQUENCE_ROWS,
    "input_width_including_history_mask": POLYMARKET_ROUND25_SEQUENCE_INPUT_WIDTH,
    "hidden_channels": POLYMARKET_ROUND25_TCN_HIDDEN_CHANNELS,
    "input_projection": "pointwise_conv1d_relu",
    "dilations": list(POLYMARKET_ROUND25_TCN_DILATIONS),
    "convolutions_per_block": POLYMARKET_ROUND25_TCN_CONVOLUTIONS_PER_BLOCK,
    "convolution": "causal_depthwise_conv1d_then_pointwise_conv1d",
    "kernel_size": POLYMARKET_ROUND25_TCN_KERNEL_SIZE,
    "activation": "relu",
    "dropout": POLYMARKET_ROUND25_TCN_DROPOUT,
    "terminal_head": "bounded_residual_over_exact_market_prior_logit",
    "auxiliary_horizons_ms": list(POLYMARKET_ROUND25_AUXILIARY_HORIZONS_MS),
    "auxiliary_outputs_used_at_inference": False,
}
_SUPPORTED_TORCH_BACKENDS = {"cpu", "cuda", "rocm", "xpu", "directml", "mps"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


POLYMARKET_ROUND25_TCN_ARCHITECTURE_JSON = _canonical_json(
    POLYMARKET_ROUND25_TCN_ARCHITECTURE
)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency boundary
        raise RuntimeError("Round 25 TCN requires the gpu or directml extra") from exc
    return torch


def round25_tcn_parameter_count() -> int:
    hidden = POLYMARKET_ROUND25_TCN_HIDDEN_CHANNELS
    projection = hidden * POLYMARKET_ROUND25_SEQUENCE_INPUT_WIDTH + hidden
    causal_unit = (
        hidden * POLYMARKET_ROUND25_TCN_KERNEL_SIZE
        + hidden
        + hidden * hidden
        + hidden
    )
    blocks = (
        len(POLYMARKET_ROUND25_TCN_DILATIONS)
        * POLYMARKET_ROUND25_TCN_CONVOLUTIONS_PER_BLOCK
        * causal_unit
    )
    terminal_head = hidden + 1
    auxiliary_head = hidden * len(POLYMARKET_ROUND25_AUXILIARY_HORIZONS_MS) + len(
        POLYMARKET_ROUND25_AUXILIARY_HORIZONS_MS
    )
    return projection + blocks + terminal_head + auxiliary_head


def _model() -> Any:
    torch = _torch()
    nn = torch.nn

    class CausalUnit(nn.Module):
        def __init__(self, dilation: int) -> None:
            super().__init__()
            self.trim = (POLYMARKET_ROUND25_TCN_KERNEL_SIZE - 1) * dilation
            self.depthwise = nn.Conv1d(
                POLYMARKET_ROUND25_TCN_HIDDEN_CHANNELS,
                POLYMARKET_ROUND25_TCN_HIDDEN_CHANNELS,
                POLYMARKET_ROUND25_TCN_KERNEL_SIZE,
                padding=self.trim,
                dilation=dilation,
                groups=POLYMARKET_ROUND25_TCN_HIDDEN_CHANNELS,
            )
            self.pointwise = nn.Conv1d(
                POLYMARKET_ROUND25_TCN_HIDDEN_CHANNELS,
                POLYMARKET_ROUND25_TCN_HIDDEN_CHANNELS,
                1,
            )
            self.activation = nn.ReLU()
            self.dropout = nn.Dropout(POLYMARKET_ROUND25_TCN_DROPOUT)

        def forward(self, values: Any) -> Any:
            convolved = self.depthwise(values)
            if self.trim:
                convolved = convolved[..., : -self.trim]
            return self.dropout(self.pointwise(self.activation(convolved)))

    class CausalBlock(nn.Module):
        def __init__(self, dilation: int) -> None:
            super().__init__()
            self.units = nn.ModuleList(
                CausalUnit(dilation)
                for _ in range(POLYMARKET_ROUND25_TCN_CONVOLUTIONS_PER_BLOCK)
            )
            self.activation = nn.ReLU()

        def forward(self, values: Any) -> Any:
            residual = values
            encoded = values
            for unit in self.units:
                encoded = unit(encoded)
            return self.activation(residual + encoded)

    class Round25CausalMultitaskTCN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Conv1d(
                POLYMARKET_ROUND25_SEQUENCE_INPUT_WIDTH,
                POLYMARKET_ROUND25_TCN_HIDDEN_CHANNELS,
                1,
            )
            self.activation = nn.ReLU()
            self.blocks = nn.ModuleList(
                CausalBlock(dilation)
                for dilation in POLYMARKET_ROUND25_TCN_DILATIONS
            )
            self.terminal_head = nn.Linear(
                POLYMARKET_ROUND25_TCN_HIDDEN_CHANNELS,
                1,
            )
            self.auxiliary_head = nn.Linear(
                POLYMARKET_ROUND25_TCN_HIDDEN_CHANNELS,
                len(POLYMARKET_ROUND25_AUXILIARY_HORIZONS_MS),
            )

        def forward(self, values: Any) -> tuple[Any, Any]:
            encoded = self.activation(self.projection(values.transpose(1, 2)))
            for block in self.blocks:
                encoded = block(encoded)
            terminal = encoded[..., -1]
            return (
                self.terminal_head(terminal).squeeze(-1),
                self.auxiliary_head(terminal),
            )

    model = Round25CausalMultitaskTCN()
    observed = sum(parameter.numel() for parameter in model.parameters())
    if observed != round25_tcn_parameter_count():
        raise RuntimeError("Round 25 TCN parameter count differs")
    return model


def _state_bytes(model: Any) -> bytes:
    chunks: list[bytes] = []
    for name, tensor in sorted(model.state_dict().items()):
        if not name:
            raise RuntimeError("Round 25 TCN state name is invalid")
        array = np.asarray(tensor.detach().cpu().numpy(), dtype="<f4", order="C")
        if not np.all(np.isfinite(array)):
            raise RuntimeError("Round 25 TCN state is nonfinite")
        chunks.append(array.tobytes(order="C"))
    state = b"".join(chunks)
    if len(state) != round25_tcn_parameter_count() * 4:
        raise RuntimeError("Round 25 TCN state length differs")
    return state


def _load_state_bytes(model: Any, state: bytes) -> None:
    torch = _torch()
    if len(state) != round25_tcn_parameter_count() * 4:
        raise ValueError("Round 25 TCN state length differs")
    offset = 0
    restored: dict[str, Any] = {}
    for name, tensor in sorted(model.state_dict().items()):
        count = tensor.numel()
        size = count * 4
        array = np.frombuffer(state[offset : offset + size], dtype="<f4").copy()
        if len(array) != count:
            raise ValueError("Round 25 TCN state length differs")
        restored[name] = torch.from_numpy(array.reshape(tuple(tensor.shape)))
        offset += size
    if offset != len(state):
        raise ValueError("Round 25 TCN state length differs")
    model.load_state_dict(restored, strict=True)


def round25_tcn_loss(
    terminal_raw_residual: Any,
    auxiliary_prediction: Any,
    terminal_labels: Any,
    terminal_market_prior: Any,
    endpoint_weights: Any,
    auxiliary_targets: Any,
    auxiliary_mask: Any,
) -> tuple[Any, Any, tuple[Any, Any]]:
    """Return total, terminal, and two masked auxiliary losses."""

    torch = _torch()
    bounded_residual = POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND * torch.tanh(
        terminal_raw_residual / POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND
    )
    prior_logit = torch.log(terminal_market_prior) - torch.log1p(
        -terminal_market_prior
    )
    logits = prior_logit + bounded_residual
    positive = torch.maximum(logits, torch.zeros_like(logits))
    terminal_per_row = (
        positive
        - logits * terminal_labels
        + torch.log1p(torch.exp(-torch.abs(logits)))
    )
    total_weight = torch.sum(endpoint_weights)
    terminal_loss = torch.sum(terminal_per_row * endpoint_weights) / total_weight
    difference = auxiliary_prediction - auxiliary_targets
    absolute = torch.abs(difference)
    huber = torch.where(absolute < 1.0, 0.5 * difference * difference, absolute - 0.5)
    auxiliary_losses: list[Any] = []
    for index in range(len(POLYMARKET_ROUND25_AUXILIARY_HORIZONS_MS)):
        available = auxiliary_mask[:, index].to(endpoint_weights.dtype)
        weights = endpoint_weights * available
        denominator = torch.sum(weights)
        numerator = torch.sum(huber[:, index] * weights)
        auxiliary_losses.append(
            torch.where(
                denominator > 0.0,
                numerator / torch.clamp(denominator, min=1e-12),
                torch.zeros_like(numerator),
            )
        )
    total = terminal_loss + POLYMARKET_ROUND25_TCN_AUXILIARY_WEIGHT * sum(
        auxiliary_losses
    )
    return total, terminal_loss, (auxiliary_losses[0], auxiliary_losses[1])


def _adamw_step(
    parameters: tuple[Any, ...],
    first_moments: tuple[Any, ...],
    second_moments: tuple[Any, ...],
    *,
    step: int,
) -> None:
    torch = _torch()
    beta1, beta2 = POLYMARKET_ROUND25_TCN_ADAM_BETAS
    first_correction = 1.0 - beta1**step
    second_correction = 1.0 - beta2**step
    step_size = POLYMARKET_ROUND25_TCN_LEARNING_RATE / first_correction
    with torch.no_grad():
        for parameter, first, second in zip(
            parameters,
            first_moments,
            second_moments,
            strict=True,
        ):
            gradient = parameter.grad
            if gradient is None:
                raise RuntimeError("Round 25 TCN gradient is unavailable")
            parameter.mul_(
                1.0
                - POLYMARKET_ROUND25_TCN_LEARNING_RATE
                * POLYMARKET_ROUND25_TCN_WEIGHT_DECAY
            )
            first.mul_(beta1).add_((1.0 - beta1) * gradient)
            second.mul_(beta2).add_((1.0 - beta2) * gradient * gradient)
            denominator = (
                torch.sqrt(second / second_correction)
                + POLYMARKET_ROUND25_TCN_ADAM_EPSILON
            )
            parameter.add_(-step_size * first / denominator)


def round25_tcn_train_step(
    model: Any,
    collation: Round25SequenceCollation,
    *,
    device: object,
    first_moments: tuple[Any, ...],
    second_moments: tuple[Any, ...],
    step: int,
) -> tuple[float, float, tuple[float, float]]:
    torch = _torch()
    if not isinstance(collation, Round25SequenceCollation):
        raise TypeError("Round 25 TCN collation type differs")
    collation.__post_init__()
    model.train()
    parameters = tuple(model.parameters())
    if len(first_moments) != len(parameters) or len(second_moments) != len(parameters):
        raise ValueError("Round 25 TCN optimizer state differs")
    for parameter in parameters:
        parameter.grad = None
    sequences = torch.from_numpy(np.array(collation.sequence_values, copy=True)).to(
        device
    )
    labels = torch.from_numpy(np.array(collation.terminal_labels, copy=True)).to(device)
    prior = torch.from_numpy(
        np.asarray(collation.terminal_market_prior, dtype=np.float32).copy()
    ).to(device)
    weights = torch.from_numpy(
        np.asarray(collation.endpoint_weights, dtype=np.float32).copy()
    ).to(device)
    auxiliary_targets = torch.from_numpy(
        np.array(collation.auxiliary_targets, copy=True)
    ).to(device)
    auxiliary_mask = torch.from_numpy(
        np.array(collation.auxiliary_mask, copy=True)
    ).to(device)
    terminal_raw, auxiliary_prediction = model(sequences)
    total, terminal, auxiliary = round25_tcn_loss(
        terminal_raw,
        auxiliary_prediction,
        labels,
        prior,
        weights,
        auxiliary_targets,
        auxiliary_mask,
    )
    if not bool(torch.isfinite(total).detach().cpu().item()):
        raise RuntimeError("Round 25 TCN training loss is nonfinite")
    total.backward()
    torch.nn.utils.clip_grad_norm_(
        parameters,
        POLYMARKET_ROUND25_TCN_GRADIENT_CLIP_NORM,
    )
    _adamw_step(
        parameters,
        first_moments,
        second_moments,
        step=step,
    )
    return (
        float(total.detach().cpu().item()),
        float(terminal.detach().cpu().item()),
        tuple(float(value.detach().cpu().item()) for value in auxiliary),
    )


@dataclass(frozen=True, slots=True)
class Round25TCNSeedArtifact:
    training_seed: int
    train_dataset_sha256: str
    calibration_dataset_sha256: str
    train_resolution_authority_sha256: str
    calibration_resolution_authority_sha256: str
    feature_transform_sha256: str
    train_batch_manifest_sha256: str
    calibration_batch_manifest_sha256: str
    state_base64: str
    state_sha256: str
    parameter_count: int
    best_epoch: int
    epochs_run: int
    calibration_condition_equal_log_loss: float
    calibration_condition_equal_brier_score: float
    backend_requested: str
    backend_kind: str
    backend_device: str
    backend_vendor: str
    backend_reason: str
    backend_selection: str
    torch_version: str
    artifact_sha256: str
    schema_version: str = POLYMARKET_ROUND25_TCN_SEED_ARTIFACT_SCHEMA_VERSION
    candidate_id: str = "causal-multitask-tcn-residual-v1"
    architecture_json: str = POLYMARKET_ROUND25_TCN_ARCHITECTURE_JSON
    model_design_sha256: str = POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256
    candidate_design_sha256: str = POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256
    candidate_amendment_sha256: str = (
        POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256
    )
    control_fit_contract_sha256: str = (
        POLYMARKET_ROUND25_CONTROL_FIT_CONTRACT_SHA256
    )
    sequence_materialization_contract_sha256: str = (
        POLYMARKET_ROUND25_SEQUENCE_MATERIALIZATION_CONTRACT_SHA256
    )
    tcn_fit_contract_sha256: str = POLYMARKET_ROUND25_TCN_FIT_CONTRACT_SHA256
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "architecture": json.loads(self.architecture_json),
            "backend_device": self.backend_device,
            "backend_kind": self.backend_kind,
            "backend_reason": self.backend_reason,
            "backend_requested": self.backend_requested,
            "backend_selection": self.backend_selection,
            "backend_vendor": self.backend_vendor,
            "best_epoch": self.best_epoch,
            "calibration_batch_manifest_sha256": (
                self.calibration_batch_manifest_sha256
            ),
            "calibration_condition_equal_brier_score": (
                self.calibration_condition_equal_brier_score
            ),
            "calibration_condition_equal_log_loss": (
                self.calibration_condition_equal_log_loss
            ),
            "calibration_dataset_sha256": self.calibration_dataset_sha256,
            "calibration_resolution_authority_sha256": (
                self.calibration_resolution_authority_sha256
            ),
            "candidate_amendment_sha256": self.candidate_amendment_sha256,
            "candidate_design_sha256": self.candidate_design_sha256,
            "candidate_id": self.candidate_id,
            "control_fit_contract_sha256": self.control_fit_contract_sha256,
            "epochs_run": self.epochs_run,
            "feature_transform_sha256": self.feature_transform_sha256,
            "model_design_sha256": self.model_design_sha256,
            "parameter_count": self.parameter_count,
            "schema_version": self.schema_version,
            "sequence_materialization_contract_sha256": (
                self.sequence_materialization_contract_sha256
            ),
            "state_base64": self.state_base64,
            "state_sha256": self.state_sha256,
            "tcn_fit_contract_sha256": self.tcn_fit_contract_sha256,
            "torch_version": self.torch_version,
            "trading_authority": self.trading_authority,
            "train_batch_manifest_sha256": self.train_batch_manifest_sha256,
            "train_dataset_sha256": self.train_dataset_sha256,
            "train_resolution_authority_sha256": (
                self.train_resolution_authority_sha256
            ),
            "training_seed": self.training_seed,
        }

    def __post_init__(self) -> None:
        try:
            state = base64.b64decode(self.state_base64, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("Round 25 TCN state encoding differs") from exc
        if (
            self.schema_version != POLYMARKET_ROUND25_TCN_SEED_ARTIFACT_SCHEMA_VERSION
            or self.candidate_id != "causal-multitask-tcn-residual-v1"
            or self.architecture_json != POLYMARKET_ROUND25_TCN_ARCHITECTURE_JSON
            or self.model_design_sha256
            != POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256
            or self.candidate_design_sha256
            != POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256
            or self.candidate_amendment_sha256
            != POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256
            or self.control_fit_contract_sha256
            != POLYMARKET_ROUND25_CONTROL_FIT_CONTRACT_SHA256
            or self.sequence_materialization_contract_sha256
            != POLYMARKET_ROUND25_SEQUENCE_MATERIALIZATION_CONTRACT_SHA256
            or self.tcn_fit_contract_sha256
            != POLYMARKET_ROUND25_TCN_FIT_CONTRACT_SHA256
            or self.training_seed not in POLYMARKET_ROUND25_TCN_TRAINING_SEEDS
            or self.parameter_count != round25_tcn_parameter_count()
            or len(state) != self.parameter_count * 4
            or self.state_sha256 != hashlib.sha256(state).hexdigest()
            or not 1 <= self.best_epoch <= self.epochs_run <= POLYMARKET_ROUND25_TCN_MAXIMUM_EPOCHS
            or not math.isfinite(self.calibration_condition_equal_log_loss)
            or self.calibration_condition_equal_log_loss < 0.0
            or not math.isfinite(self.calibration_condition_equal_brier_score)
            or not 0.0 <= self.calibration_condition_equal_brier_score <= 1.0
            or self.backend_kind not in _SUPPORTED_TORCH_BACKENDS
            or not self.backend_requested.strip()
            or len(self.backend_requested) > 64
            or not self.backend_device.strip()
            or len(self.backend_device) > 500
            or not self.backend_vendor.strip()
            or len(self.backend_vendor) > 500
            or len(self.backend_reason) > 1_000
            or not self.backend_selection.strip()
            or len(self.backend_selection) > 500
            or not self.torch_version.strip()
            or len(self.torch_version) > 128
            or self.trading_authority is not False
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.train_dataset_sha256,
                    self.calibration_dataset_sha256,
                    self.train_resolution_authority_sha256,
                    self.calibration_resolution_authority_sha256,
                    self.feature_transform_sha256,
                    self.train_batch_manifest_sha256,
                    self.calibration_batch_manifest_sha256,
                    self.state_sha256,
                    self.artifact_sha256,
                )
            )
            or self.artifact_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 25 TCN seed artifact differs")

    def validated(self) -> Round25TCNSeedArtifact:
        self.__post_init__()
        return self


def _create_round25_tcn_seed_artifact(
    *,
    model: Any,
    training_seed: int,
    train_dataset_sha256: str,
    calibration_dataset_sha256: str,
    train_resolution_authority_sha256: str,
    calibration_resolution_authority_sha256: str,
    feature_transform_sha256: str,
    train_batch_manifest_sha256: str,
    calibration_batch_manifest_sha256: str,
    best_epoch: int,
    epochs_run: int,
    calibration_condition_equal_log_loss: float,
    calibration_condition_equal_brier_score: float,
    backend_requested: str,
    backend_kind: str,
    backend_device: str,
    backend_vendor: str,
    backend_reason: str,
    backend_selection: str,
) -> Round25TCNSeedArtifact:
    torch = _torch()
    state = _state_bytes(model)
    state_base64 = base64.b64encode(state).decode("ascii")
    state_sha256 = hashlib.sha256(state).hexdigest()
    values: dict[str, object] = {
        "architecture": dict(POLYMARKET_ROUND25_TCN_ARCHITECTURE),
        "backend_device": backend_device,
        "backend_kind": backend_kind,
        "backend_reason": backend_reason,
        "backend_requested": backend_requested,
        "backend_selection": backend_selection,
        "backend_vendor": backend_vendor,
        "best_epoch": best_epoch,
        "calibration_batch_manifest_sha256": calibration_batch_manifest_sha256,
        "calibration_condition_equal_brier_score": (
            calibration_condition_equal_brier_score
        ),
        "calibration_condition_equal_log_loss": (
            calibration_condition_equal_log_loss
        ),
        "calibration_dataset_sha256": calibration_dataset_sha256,
        "calibration_resolution_authority_sha256": (
            calibration_resolution_authority_sha256
        ),
        "candidate_amendment_sha256": POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
        "candidate_design_sha256": POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
        "candidate_id": "causal-multitask-tcn-residual-v1",
        "control_fit_contract_sha256": POLYMARKET_ROUND25_CONTROL_FIT_CONTRACT_SHA256,
        "epochs_run": epochs_run,
        "feature_transform_sha256": feature_transform_sha256,
        "model_design_sha256": POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
        "parameter_count": round25_tcn_parameter_count(),
        "schema_version": POLYMARKET_ROUND25_TCN_SEED_ARTIFACT_SCHEMA_VERSION,
        "sequence_materialization_contract_sha256": (
            POLYMARKET_ROUND25_SEQUENCE_MATERIALIZATION_CONTRACT_SHA256
        ),
        "state_base64": state_base64,
        "state_sha256": state_sha256,
        "tcn_fit_contract_sha256": POLYMARKET_ROUND25_TCN_FIT_CONTRACT_SHA256,
        "torch_version": str(torch.__version__),
        "trading_authority": False,
        "train_batch_manifest_sha256": train_batch_manifest_sha256,
        "train_dataset_sha256": train_dataset_sha256,
        "train_resolution_authority_sha256": train_resolution_authority_sha256,
        "training_seed": training_seed,
    }
    return Round25TCNSeedArtifact(
        training_seed=training_seed,
        train_dataset_sha256=train_dataset_sha256,
        calibration_dataset_sha256=calibration_dataset_sha256,
        train_resolution_authority_sha256=train_resolution_authority_sha256,
        calibration_resolution_authority_sha256=(
            calibration_resolution_authority_sha256
        ),
        feature_transform_sha256=feature_transform_sha256,
        train_batch_manifest_sha256=train_batch_manifest_sha256,
        calibration_batch_manifest_sha256=calibration_batch_manifest_sha256,
        state_base64=state_base64,
        state_sha256=state_sha256,
        parameter_count=round25_tcn_parameter_count(),
        best_epoch=best_epoch,
        epochs_run=epochs_run,
        calibration_condition_equal_log_loss=calibration_condition_equal_log_loss,
        calibration_condition_equal_brier_score=calibration_condition_equal_brier_score,
        backend_requested=backend_requested,
        backend_kind=backend_kind,
        backend_device=backend_device,
        backend_vendor=backend_vendor,
        backend_reason=backend_reason,
        backend_selection=backend_selection,
        torch_version=str(torch.__version__),
        artifact_sha256=_canonical_sha256(values),
        architecture_json=POLYMARKET_ROUND25_TCN_ARCHITECTURE_JSON,
    )


class Round25CompiledTCN:
    """Reusable inference runtime that cannot access auxiliary targets."""

    def __init__(
        self,
        artifact: Round25TCNSeedArtifact,
        *,
        compute_backend: str = "auto",
    ) -> None:
        if not isinstance(artifact, Round25TCNSeedArtifact):
            raise TypeError("Round 25 TCN artifact type differs")
        self._artifact = artifact.validated()
        requested_backend = compute_backend.strip().lower()
        backend = resolve_backend(
            requested_backend,
            require=requested_backend != "auto",
        )
        self._device = torch_device_for_backend(backend)
        self._model = _model()
        state = base64.b64decode(artifact.state_base64, validate=True)
        _load_state_bytes(self._model, state)
        self._model = self._model.to(self._device)
        self._model.eval()

    @property
    def artifact_sha256(self) -> str:
        return self._artifact.artifact_sha256

    def predict_probabilities(
        self,
        sequence_values: np.ndarray,
        terminal_market_prior: Sequence[float],
    ) -> tuple[float, ...]:
        torch = _torch()
        sequences = np.asarray(sequence_values, dtype=np.float32)
        prior = np.asarray(terminal_market_prior, dtype=np.float32)
        if (
            sequences.ndim != 3
            or sequences.shape[1:]
            != (
                POLYMARKET_ROUND25_SEQUENCE_ROWS,
                POLYMARKET_ROUND25_SEQUENCE_INPUT_WIDTH,
            )
            or prior.shape != (len(sequences),)
            or not len(sequences)
            or not np.all(np.isfinite(sequences))
            or not np.all(np.isfinite(prior))
            or not np.all((prior > 0.0) & (prior < 1.0))
            or not np.all(sequences[:, -1, -1] == 1.0)
        ):
            raise ValueError("Round 25 TCN inference population is invalid")
        with torch.no_grad():
            raw, _unused_auxiliary = self._model(
                torch.from_numpy(np.array(sequences, copy=True)).to(self._device)
            )
            bounded = POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND * torch.tanh(
                raw / POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND
            )
            prior_tensor = torch.from_numpy(prior.copy()).to(self._device)
            probability = torch.sigmoid(
                torch.log(prior_tensor)
                - torch.log1p(-prior_tensor)
                + bounded
            )
            output = np.asarray(
                probability.detach().cpu().numpy(),
                dtype=np.float64,
            )
        if output.shape != prior.shape or not np.all(np.isfinite(output)):
            raise RuntimeError("Round 25 TCN prediction is invalid")
        return tuple(float(value) for value in output)


Round25TCNBatchLoader = Callable[
    [tuple[str, ...]],
    Sequence[Round25SequenceConditionBatch],
]


@dataclass(frozen=True, slots=True)
class Round25TCNCorpusSource:
    """Hash-bound, lazy sequence source that keeps tensors out of memory."""

    role: str
    condition_ids: tuple[str, ...]
    event_start_ms: tuple[int, ...]
    batch_sha256: tuple[str, ...]
    source_dataset_sha256: str
    resolution_authority_sha256: str
    feature_transform_sha256: str
    manifest_sha256: str
    loader: Round25TCNBatchLoader = field(repr=False, compare=False)
    schema_version: str = POLYMARKET_ROUND25_TCN_CORPUS_SCHEMA_VERSION
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "batch_sha256": list(self.batch_sha256),
            "condition_ids": list(self.condition_ids),
            "event_start_ms": list(self.event_start_ms),
            "feature_transform_sha256": self.feature_transform_sha256,
            "resolution_authority_sha256": self.resolution_authority_sha256,
            "role": self.role,
            "schema_version": self.schema_version,
            "source_dataset_sha256": self.source_dataset_sha256,
            "trading_authority": self.trading_authority,
        }

    def __post_init__(self) -> None:
        count = len(self.condition_ids)
        ordered = tuple(
            sorted(
                zip(self.event_start_ms, self.condition_ids, strict=True),
                key=lambda item: (item[0], item[1]),
            )
        ) if count == len(self.event_start_ms) else ()
        if (
            self.role not in {"train", "calibration"}
            or not count
            or len(self.event_start_ms) != count
            or len(self.batch_sha256) != count
            or len(set(self.condition_ids)) != count
            or len(set(self.batch_sha256)) != count
            or any(re.fullmatch(r"0x[0-9a-f]{64}", value) is None for value in self.condition_ids)
            or any(type(value) is not int or value < 0 for value in self.event_start_ms)
            or tuple(zip(self.event_start_ms, self.condition_ids, strict=True))
            != ordered
            or any(_SHA256.fullmatch(value) is None for value in self.batch_sha256)
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.source_dataset_sha256,
                    self.resolution_authority_sha256,
                    self.feature_transform_sha256,
                    self.manifest_sha256,
                )
            )
            or not callable(self.loader)
            or self.schema_version != POLYMARKET_ROUND25_TCN_CORPUS_SCHEMA_VERSION
            or self.trading_authority is not False
            or self.manifest_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 25 TCN corpus source differs")

    def validated(self) -> Round25TCNCorpusSource:
        self.__post_init__()
        return self

    def load_collation(
        self,
        condition_ids: Sequence[str],
    ) -> Round25SequenceCollation:
        requested = tuple(condition_ids)
        if not 1 <= len(requested) <= 16 or len(set(requested)) != len(requested):
            raise ValueError("Round 25 TCN load request differs")
        index = {condition_id: offset for offset, condition_id in enumerate(self.condition_ids)}
        if any(condition_id not in index for condition_id in requested):
            raise ValueError("Round 25 TCN load request is outside the corpus")
        loaded = tuple(self.loader(requested))
        if (
            len(loaded) != len(requested)
            or any(not isinstance(batch, Round25SequenceConditionBatch) for batch in loaded)
            or {batch.condition_id for batch in loaded} != set(requested)
        ):
            raise ValueError("Round 25 TCN loader population differs")
        observed = {batch.condition_id: batch.validated() for batch in loaded}
        for condition_id in requested:
            offset = index[condition_id]
            batch = observed[condition_id]
            if (
                batch.role != self.role
                or batch.event_start_ms != self.event_start_ms[offset]
                or batch.batch_sha256 != self.batch_sha256[offset]
                or batch.source_dataset_sha256 != self.source_dataset_sha256
                or batch.resolution_authority_sha256
                != self.resolution_authority_sha256
                or batch.feature_transform_sha256 != self.feature_transform_sha256
            ):
                raise ValueError("Round 25 TCN loaded batch differs from its manifest")
        return collate_round25_sequence_batches(
            tuple(observed[condition_id] for condition_id in requested)
        )


def create_round25_tcn_corpus_source(
    *,
    role: str,
    condition_ids: Sequence[str],
    event_start_ms: Sequence[int],
    batch_sha256: Sequence[str],
    source_dataset_sha256: str,
    resolution_authority_sha256: str,
    feature_transform_sha256: str,
    loader: Round25TCNBatchLoader,
) -> Round25TCNCorpusSource:
    selected_condition_ids = tuple(condition_ids)
    selected_event_start_ms = tuple(event_start_ms)
    selected_batch_sha256 = tuple(batch_sha256)
    payload = {
        "batch_sha256": list(selected_batch_sha256),
        "condition_ids": list(selected_condition_ids),
        "event_start_ms": list(selected_event_start_ms),
        "feature_transform_sha256": feature_transform_sha256,
        "resolution_authority_sha256": resolution_authority_sha256,
        "role": role,
        "schema_version": POLYMARKET_ROUND25_TCN_CORPUS_SCHEMA_VERSION,
        "source_dataset_sha256": source_dataset_sha256,
        "trading_authority": False,
    }
    return Round25TCNCorpusSource(
        role=role,
        condition_ids=selected_condition_ids,
        event_start_ms=selected_event_start_ms,
        batch_sha256=selected_batch_sha256,
        source_dataset_sha256=source_dataset_sha256,
        resolution_authority_sha256=resolution_authority_sha256,
        feature_transform_sha256=feature_transform_sha256,
        manifest_sha256=_canonical_sha256(payload),
        loader=loader,
    )


def validate_round25_tcn_fit_sources(
    train: Round25TCNCorpusSource,
    calibration: Round25TCNCorpusSource,
) -> tuple[Round25TCNCorpusSource, Round25TCNCorpusSource]:
    if not isinstance(train, Round25TCNCorpusSource) or not isinstance(
        calibration,
        Round25TCNCorpusSource,
    ):
        raise TypeError("Round 25 TCN fit source type differs")
    train.validated()
    calibration.validated()
    if (
        train.role != "train"
        or calibration.role != "calibration"
        or len(train.condition_ids) < POLYMARKET_ROUND25_MINIMUM_CONDITIONS["train"]
        or len(calibration.condition_ids)
        < POLYMARKET_ROUND25_MINIMUM_CONDITIONS["calibration"]
        or train.source_dataset_sha256 == calibration.source_dataset_sha256
        or train.feature_transform_sha256 != calibration.feature_transform_sha256
        or set(train.condition_ids).intersection(calibration.condition_ids)
        or max(train.event_start_ms) >= min(calibration.event_start_ms)
    ):
        raise ValueError("Round 25 TCN fit sources fail the frozen corpus gates")
    return train, calibration


@dataclass(frozen=True, slots=True)
class Round25TCNFitProgress:
    stage: str
    training_seed: int | None
    epoch: int
    conditions_processed: int
    total_conditions: int
    elapsed_seconds: float


Round25TCNProgressCallback = Callable[[Round25TCNFitProgress], None]


class _Round25ProgressEmitter:
    def __init__(self, callback: Round25TCNProgressCallback | None) -> None:
        self._callback = callback
        self._started = time.monotonic()
        self._last = self._started

    def emit(
        self,
        *,
        stage: str,
        training_seed: int | None,
        epoch: int,
        conditions_processed: int,
        total_conditions: int,
        force: bool = False,
    ) -> None:
        now = time.monotonic()
        if self._callback is None or (
            not force
            and now - self._last < POLYMARKET_ROUND25_TCN_PROGRESS_HEARTBEAT_SECONDS
        ):
            return
        self._callback(Round25TCNFitProgress(
            stage=stage,
            training_seed=training_seed,
            epoch=epoch,
            conditions_processed=conditions_processed,
            total_conditions=total_conditions,
            elapsed_seconds=now - self._started,
        ))
        self._last = now


def _round25_tcn_chunks(
    condition_ids: Sequence[str],
) -> tuple[tuple[str, ...], ...]:
    selected = tuple(condition_ids)
    return tuple(
        selected[offset : offset + 16]
        for offset in range(0, len(selected), 16)
    )


def _round25_tcn_epoch_order(
    condition_ids: Sequence[str],
    *,
    training_seed: int,
    epoch: int,
) -> tuple[str, ...]:
    return tuple(sorted(
        condition_ids,
        key=lambda condition_id: (
            hashlib.sha256(
                f"{training_seed}:{epoch}:{condition_id}".encode("ascii")
            ).digest(),
            condition_id,
        ),
    ))


def _round25_tcn_model_probabilities(
    model: Any,
    collation: Round25SequenceCollation,
    *,
    device: object,
) -> np.ndarray:
    torch = _torch()
    sequences = torch.from_numpy(np.array(collation.sequence_values, copy=True)).to(
        device
    )
    prior = torch.from_numpy(
        np.asarray(collation.terminal_market_prior, dtype=np.float32).copy()
    ).to(device)
    raw, _unused_auxiliary = model(sequences)
    bounded = POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND * torch.tanh(
        raw / POLYMARKET_ROUND25_RESIDUAL_LOGIT_BOUND
    )
    probability = torch.sigmoid(torch.log(prior) - torch.log1p(-prior) + bounded)
    output = np.asarray(probability.detach().cpu().numpy(), dtype=np.float64)
    if output.shape != collation.terminal_labels.shape or not np.all(
        np.isfinite(output)
    ):
        raise RuntimeError("Round 25 TCN evaluation prediction differs")
    return output


def _evaluate_round25_tcn(
    model: Any,
    source: Round25TCNCorpusSource,
    *,
    device: object,
    training_seed: int,
    epoch: int,
    progress: _Round25ProgressEmitter,
) -> tuple[float, float]:
    torch = _torch()
    model.eval()
    log_loss_numerator = 0.0
    brier_numerator = 0.0
    total_weight = 0.0
    processed = 0
    with torch.no_grad():
        for condition_ids in _round25_tcn_chunks(source.condition_ids):
            collation = source.load_collation(condition_ids)
            probability = np.clip(
                _round25_tcn_model_probabilities(
                    model,
                    collation,
                    device=device,
                ),
                1e-12,
                1.0 - 1e-12,
            )
            labels = np.asarray(collation.terminal_labels, dtype=np.float64)
            weights = np.asarray(collation.endpoint_weights, dtype=np.float64)
            per_row_log_loss = -(
                labels * np.log(probability)
                + (1.0 - labels) * np.log1p(-probability)
            )
            log_loss_numerator += float(np.dot(weights, per_row_log_loss))
            brier_numerator += float(np.dot(weights, (probability - labels) ** 2))
            total_weight += float(np.sum(weights))
            processed += len(condition_ids)
            progress.emit(
                stage="calibration",
                training_seed=training_seed,
                epoch=epoch,
                conditions_processed=processed,
                total_conditions=len(source.condition_ids),
            )
    if processed != len(source.condition_ids) or total_weight != float(processed):
        raise RuntimeError("Round 25 TCN calibration coverage differs")
    log_loss = log_loss_numerator / total_weight
    brier = brier_numerator / total_weight
    if not math.isfinite(log_loss) or not math.isfinite(brier):
        raise RuntimeError("Round 25 TCN calibration score is nonfinite")
    return log_loss, brier


def _fit_round25_tcn_seed(
    train: Round25TCNCorpusSource,
    calibration: Round25TCNCorpusSource,
    *,
    training_seed: int,
    backend: object,
    progress: _Round25ProgressEmitter,
) -> Round25TCNSeedArtifact:
    if training_seed not in POLYMARKET_ROUND25_TCN_TRAINING_SEEDS:
        raise ValueError("Round 25 TCN training seed differs")
    torch = _torch()
    torch.manual_seed(training_seed)
    if getattr(backend, "kind", "") in {"cuda", "rocm"}:
        torch.cuda.manual_seed_all(training_seed)
    device = torch_device_for_backend(backend)
    model = _model().to(device)
    parameters = tuple(model.parameters())
    first_moments = tuple(torch.zeros_like(parameter) for parameter in parameters)
    second_moments = tuple(torch.zeros_like(parameter) for parameter in parameters)
    optimizer_step = 0
    best_epoch = 0
    best_log_loss = math.inf
    best_brier = math.inf
    best_state: bytes | None = None
    stale_epochs = 0
    epochs_run = 0
    progress.emit(
        stage="seed_started",
        training_seed=training_seed,
        epoch=0,
        conditions_processed=0,
        total_conditions=len(train.condition_ids),
        force=True,
    )
    for epoch in range(1, POLYMARKET_ROUND25_TCN_MAXIMUM_EPOCHS + 1):
        model.train()
        processed = 0
        epoch_order = _round25_tcn_epoch_order(
            train.condition_ids,
            training_seed=training_seed,
            epoch=epoch,
        )
        for condition_ids in _round25_tcn_chunks(epoch_order):
            collation = train.load_collation(condition_ids)
            optimizer_step += 1
            round25_tcn_train_step(
                model,
                collation,
                device=device,
                first_moments=first_moments,
                second_moments=second_moments,
                step=optimizer_step,
            )
            processed += len(condition_ids)
            progress.emit(
                stage="training",
                training_seed=training_seed,
                epoch=epoch,
                conditions_processed=processed,
                total_conditions=len(train.condition_ids),
            )
        if processed != len(train.condition_ids):
            raise RuntimeError("Round 25 TCN training coverage differs")
        calibration_log_loss, calibration_brier = _evaluate_round25_tcn(
            model,
            calibration,
            device=device,
            training_seed=training_seed,
            epoch=epoch,
            progress=progress,
        )
        epochs_run = epoch
        if (
            best_state is None
            or best_log_loss - calibration_log_loss
            > POLYMARKET_ROUND25_TCN_MINIMUM_LOG_LOSS_IMPROVEMENT
        ):
            best_epoch = epoch
            best_log_loss = calibration_log_loss
            best_brier = calibration_brier
            best_state = _state_bytes(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
        progress.emit(
            stage="epoch_completed",
            training_seed=training_seed,
            epoch=epoch,
            conditions_processed=len(calibration.condition_ids),
            total_conditions=len(calibration.condition_ids),
            force=True,
        )
        if stale_epochs >= POLYMARKET_ROUND25_TCN_EARLY_STOPPING_PATIENCE:
            break
    if best_state is None or best_epoch == 0:
        raise RuntimeError("Round 25 TCN did not produce a calibration checkpoint")
    _load_state_bytes(model, best_state)
    return _create_round25_tcn_seed_artifact(
        model=model,
        training_seed=training_seed,
        train_dataset_sha256=train.source_dataset_sha256,
        calibration_dataset_sha256=calibration.source_dataset_sha256,
        train_resolution_authority_sha256=train.resolution_authority_sha256,
        calibration_resolution_authority_sha256=(
            calibration.resolution_authority_sha256
        ),
        feature_transform_sha256=train.feature_transform_sha256,
        train_batch_manifest_sha256=train.manifest_sha256,
        calibration_batch_manifest_sha256=calibration.manifest_sha256,
        best_epoch=best_epoch,
        epochs_run=epochs_run,
        calibration_condition_equal_log_loss=best_log_loss,
        calibration_condition_equal_brier_score=best_brier,
        backend_requested=getattr(backend, "requested"),
        backend_kind=getattr(backend, "kind"),
        backend_device=getattr(backend, "device"),
        backend_vendor=getattr(backend, "vendor"),
        backend_reason=getattr(backend, "reason"),
        backend_selection=getattr(backend, "selection"),
    )


@dataclass(frozen=True, slots=True)
class Round25TCNEnsembleArtifact:
    seed_artifacts: tuple[Round25TCNSeedArtifact, ...]
    train_dataset_sha256: str
    calibration_dataset_sha256: str
    train_resolution_authority_sha256: str
    calibration_resolution_authority_sha256: str
    feature_transform_sha256: str
    train_batch_manifest_sha256: str
    calibration_batch_manifest_sha256: str
    artifact_sha256: str
    schema_version: str = POLYMARKET_ROUND25_TCN_ENSEMBLE_ARTIFACT_SCHEMA_VERSION
    candidate_id: str = "causal-multitask-tcn-residual-v1"
    ensemble_method: str = "arithmetic-mean-probability"
    tcn_fit_contract_sha256: str = POLYMARKET_ROUND25_TCN_FIT_CONTRACT_SHA256
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "calibration_batch_manifest_sha256": self.calibration_batch_manifest_sha256,
            "calibration_dataset_sha256": self.calibration_dataset_sha256,
            "calibration_resolution_authority_sha256": (
                self.calibration_resolution_authority_sha256
            ),
            "candidate_id": self.candidate_id,
            "ensemble_method": self.ensemble_method,
            "feature_transform_sha256": self.feature_transform_sha256,
            "schema_version": self.schema_version,
            "seed_artifact_sha256": [
                artifact.artifact_sha256 for artifact in self.seed_artifacts
            ],
            "tcn_fit_contract_sha256": self.tcn_fit_contract_sha256,
            "trading_authority": self.trading_authority,
            "train_batch_manifest_sha256": self.train_batch_manifest_sha256,
            "train_dataset_sha256": self.train_dataset_sha256,
            "train_resolution_authority_sha256": (
                self.train_resolution_authority_sha256
            ),
        }

    def __post_init__(self) -> None:
        common = (
            self.train_dataset_sha256,
            self.calibration_dataset_sha256,
            self.train_resolution_authority_sha256,
            self.calibration_resolution_authority_sha256,
            self.feature_transform_sha256,
            self.train_batch_manifest_sha256,
            self.calibration_batch_manifest_sha256,
        )
        if (
            self.schema_version
            != POLYMARKET_ROUND25_TCN_ENSEMBLE_ARTIFACT_SCHEMA_VERSION
            or self.candidate_id != "causal-multitask-tcn-residual-v1"
            or self.ensemble_method != "arithmetic-mean-probability"
            or self.tcn_fit_contract_sha256
            != POLYMARKET_ROUND25_TCN_FIT_CONTRACT_SHA256
            or len(self.seed_artifacts) != len(POLYMARKET_ROUND25_TCN_TRAINING_SEEDS)
            or tuple(artifact.training_seed for artifact in self.seed_artifacts)
            != POLYMARKET_ROUND25_TCN_TRAINING_SEEDS
            or any(artifact.validated() is not artifact for artifact in self.seed_artifacts)
            or any(_SHA256.fullmatch(value) is None for value in common)
            or any(
                (
                    artifact.train_dataset_sha256,
                    artifact.calibration_dataset_sha256,
                    artifact.train_resolution_authority_sha256,
                    artifact.calibration_resolution_authority_sha256,
                    artifact.feature_transform_sha256,
                    artifact.train_batch_manifest_sha256,
                    artifact.calibration_batch_manifest_sha256,
                )
                != common
                for artifact in self.seed_artifacts
            )
            or self.trading_authority is not False
            or _SHA256.fullmatch(self.artifact_sha256) is None
            or self.artifact_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 25 TCN ensemble artifact differs")

    def validated(self) -> Round25TCNEnsembleArtifact:
        self.__post_init__()
        return self


def _create_round25_tcn_ensemble_artifact(
    seed_artifacts: Sequence[Round25TCNSeedArtifact],
) -> Round25TCNEnsembleArtifact:
    selected = tuple(seed_artifacts)
    if not selected:
        raise ValueError("Round 25 TCN ensemble seed population is empty")
    first = selected[0]
    common = (
        first.train_dataset_sha256,
        first.calibration_dataset_sha256,
        first.train_resolution_authority_sha256,
        first.calibration_resolution_authority_sha256,
        first.feature_transform_sha256,
        first.train_batch_manifest_sha256,
        first.calibration_batch_manifest_sha256,
    )
    payload = {
        "calibration_batch_manifest_sha256": common[6],
        "calibration_dataset_sha256": common[1],
        "calibration_resolution_authority_sha256": common[3],
        "candidate_id": "causal-multitask-tcn-residual-v1",
        "ensemble_method": "arithmetic-mean-probability",
        "feature_transform_sha256": common[4],
        "schema_version": POLYMARKET_ROUND25_TCN_ENSEMBLE_ARTIFACT_SCHEMA_VERSION,
        "seed_artifact_sha256": [artifact.artifact_sha256 for artifact in selected],
        "tcn_fit_contract_sha256": POLYMARKET_ROUND25_TCN_FIT_CONTRACT_SHA256,
        "trading_authority": False,
        "train_batch_manifest_sha256": common[5],
        "train_dataset_sha256": common[0],
        "train_resolution_authority_sha256": common[2],
    }
    return Round25TCNEnsembleArtifact(
        seed_artifacts=selected,
        train_dataset_sha256=common[0],
        calibration_dataset_sha256=common[1],
        train_resolution_authority_sha256=common[2],
        calibration_resolution_authority_sha256=common[3],
        feature_transform_sha256=common[4],
        train_batch_manifest_sha256=common[5],
        calibration_batch_manifest_sha256=common[6],
        artifact_sha256=_canonical_sha256(payload),
    )


def fit_round25_tcn_ensemble(
    train: Round25TCNCorpusSource,
    calibration: Round25TCNCorpusSource,
    *,
    compute_backend: str = "auto",
    progress_callback: Round25TCNProgressCallback | None = None,
) -> Round25TCNEnsembleArtifact:
    selected_train, selected_calibration = validate_round25_tcn_fit_sources(
        train,
        calibration,
    )
    requested_backend = compute_backend.strip().lower()
    backend = resolve_backend(
        requested_backend,
        require=requested_backend != "auto",
    )
    if requested_backend == "auto" and getattr(backend, "kind") == "cpu":
        warnings.warn(
            "Round 25 TCN found no supported accelerator; training on CPU",
            RuntimeWarning,
            stacklevel=2,
        )
    progress = _Round25ProgressEmitter(progress_callback)
    artifacts = tuple(
        _fit_round25_tcn_seed(
            selected_train,
            selected_calibration,
            training_seed=seed,
            backend=backend,
            progress=progress,
        )
        for seed in POLYMARKET_ROUND25_TCN_TRAINING_SEEDS
    )
    result = _create_round25_tcn_ensemble_artifact(artifacts)
    progress.emit(
        stage="ensemble_completed",
        training_seed=None,
        epoch=0,
        conditions_processed=len(selected_calibration.condition_ids),
        total_conditions=len(selected_calibration.condition_ids),
        force=True,
    )
    return result


class Round25CompiledTCNEnsemble:
    """Target-free arithmetic-mean runtime for the frozen three-seed ensemble."""

    def __init__(
        self,
        artifact: Round25TCNEnsembleArtifact,
        *,
        compute_backend: str = "auto",
    ) -> None:
        if not isinstance(artifact, Round25TCNEnsembleArtifact):
            raise TypeError("Round 25 TCN ensemble artifact type differs")
        self._artifact = artifact.validated()
        self._runtimes = tuple(
            Round25CompiledTCN(seed_artifact, compute_backend=compute_backend)
            for seed_artifact in artifact.seed_artifacts
        )

    @property
    def artifact_sha256(self) -> str:
        return self._artifact.artifact_sha256

    def predict_probabilities(
        self,
        sequence_values: np.ndarray,
        terminal_market_prior: Sequence[float],
    ) -> tuple[float, ...]:
        predictions = np.asarray(
            [
                runtime.predict_probabilities(
                    sequence_values,
                    terminal_market_prior,
                )
                for runtime in self._runtimes
            ],
            dtype=np.float64,
        )
        if predictions.ndim != 2 or predictions.shape[0] != len(
            POLYMARKET_ROUND25_TCN_TRAINING_SEEDS
        ):
            raise RuntimeError("Round 25 TCN ensemble prediction differs")
        mean_probability = np.mean(predictions, axis=0)
        if not np.all(np.isfinite(mean_probability)) or not np.all(
            (mean_probability > 0.0) & (mean_probability < 1.0)
        ):
            raise RuntimeError("Round 25 TCN ensemble probability differs")
        return tuple(float(value) for value in mean_probability)


__all__ = [
    "POLYMARKET_ROUND25_TCN_ARCHITECTURE",
    "POLYMARKET_ROUND25_TCN_ARCHITECTURE_JSON",
    "POLYMARKET_ROUND25_TCN_CORPUS_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_TCN_ENSEMBLE_ARTIFACT_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_TCN_FIT_CONTRACT_SHA256",
    "POLYMARKET_ROUND25_TCN_TRAINING_SEEDS",
    "Round25CompiledTCN",
    "Round25CompiledTCNEnsemble",
    "Round25TCNCorpusSource",
    "Round25TCNEnsembleArtifact",
    "Round25TCNFitProgress",
    "Round25TCNSeedArtifact",
    "create_round25_tcn_corpus_source",
    "fit_round25_tcn_ensemble",
    "round25_tcn_loss",
    "round25_tcn_parameter_count",
    "round25_tcn_train_step",
    "validate_round25_tcn_fit_sources",
]
