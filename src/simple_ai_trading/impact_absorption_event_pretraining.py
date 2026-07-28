"""Training-only causal next-event initialization for Round 74 encoders.

The pretext task never consumes payoff, path-risk, tuning, calibration, or test
targets. It predicts the next message type and the transition in scaled
continuous microstructure features. Downstream promotion remains a paired
proper-loss decision in the supervised trainer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .distributional_tcn_model import ExplicitAdamW
from .impact_absorption_event_dataset import Round74EventTrainingBatch
from .impact_absorption_event_model import (
    encode_round74_event_sequence,
    round74_event_encoder_parameters,
    round74_event_model_pretraining_channels,
)
from .impact_absorption_event_scaling import ROUND74_EVENT_BINARY_FEATURE_COUNT
from .impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_SEQUENCE_LENGTH,
)
from .impact_absorption_store import IMPACT_CAPTURE_SYMBOLS


ROUND74_EVENT_PRETRAINING_SCHEMA_VERSION = "round-074-causal-next-event-pretraining-v1"
ROUND74_EVENT_PRETRAINING_INITIALIZATION_IDS = (
    "random",
    "causal_next_event_pretrained",
)
ROUND74_EVENT_PRETRAINING_EVENT_TYPE_COUNT = 5


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Round74EventPretrainingConfig:
    """Bounded, chronological, training-only representation policy."""

    maximum_epochs: int = 8
    early_stopping_patience: int = 2
    minimum_validation_improvement: float = 1e-5
    minibatch_rows: int = 32
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    validation_fraction: float = 0.20
    purge_anchor_count: int = ROUND74_EVENT_SEQUENCE_LENGTH
    minimum_partition_rows_per_symbol: int = 8

    def validate(self) -> None:
        integer_values = (
            self.maximum_epochs,
            self.early_stopping_patience,
            self.minibatch_rows,
            self.purge_anchor_count,
            self.minimum_partition_rows_per_symbol,
        )
        numeric_values = (
            self.minimum_validation_improvement,
            self.learning_rate,
            self.weight_decay,
            self.gradient_clip_norm,
            self.validation_fraction,
        )
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values)
            or self.maximum_epochs < 1
            or self.early_stopping_patience < 1
            or self.early_stopping_patience > self.maximum_epochs
            or self.minibatch_rows < 1
            or self.purge_anchor_count < ROUND74_EVENT_SEQUENCE_LENGTH
            or self.minimum_partition_rows_per_symbol < 1
            or not all(math.isfinite(float(value)) for value in numeric_values)
            or self.minimum_validation_improvement < 0.0
            or self.learning_rate <= 0.0
            or self.weight_decay < 0.0
            or self.gradient_clip_norm <= 0.0
            or not 0.0 < self.validation_fraction < 0.5
        ):
            raise ValueError("Round 74 event pretraining configuration differs")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": ROUND74_EVENT_PRETRAINING_SCHEMA_VERSION,
            "maximum_epochs": self.maximum_epochs,
            "early_stopping_patience": self.early_stopping_patience,
            "minimum_validation_improvement": self.minimum_validation_improvement,
            "minibatch_rows": self.minibatch_rows,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gradient_clip_norm": self.gradient_clip_norm,
            "validation_fraction": self.validation_fraction,
            "purge_anchor_count": self.purge_anchor_count,
            "minimum_partition_rows_per_symbol": (
                self.minimum_partition_rows_per_symbol
            ),
            "fit_partition_role": "training",
            "validation_partition_role": "purged_tail_within_training",
            "event_target": "next_event_type",
            "continuous_target": "next_scaled_feature_delta",
            "event_loss": "categorical_log_loss_divided_by_log_event_type_count",
            "continuous_loss": "dimension_mean_huber_beta_1",
            "run_weighting": "equal_capture_run",
            "shorter_run_policy": "deterministic_epoch_rotated_cycling",
            "supervised_targets_used": False,
            "tuning_features_used": False,
            "tuning_targets_used": False,
            "calibration_data_used": False,
            "test_data_used": False,
            "temporary_prediction_head_persisted": False,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Round74EventPretrainingConfig:
        expected = cls()
        if dict(payload) == expected.as_dict():
            return expected
        try:
            selected = cls(
                maximum_epochs=int(payload["maximum_epochs"]),
                early_stopping_patience=int(payload["early_stopping_patience"]),
                minimum_validation_improvement=float(
                    payload["minimum_validation_improvement"]
                ),
                minibatch_rows=int(payload["minibatch_rows"]),
                learning_rate=float(payload["learning_rate"]),
                weight_decay=float(payload["weight_decay"]),
                gradient_clip_norm=float(payload["gradient_clip_norm"]),
                validation_fraction=float(payload["validation_fraction"]),
                purge_anchor_count=int(payload["purge_anchor_count"]),
                minimum_partition_rows_per_symbol=int(
                    payload["minimum_partition_rows_per_symbol"]
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Round 74 event pretraining policy differs") from exc
        if selected.as_dict() != dict(payload):
            raise ValueError("Round 74 event pretraining policy differs")
        return selected


@dataclass(frozen=True)
class _Round74PretrainingSplit:
    training_indices: tuple[np.ndarray, ...]
    validation_indices: tuple[np.ndarray, ...]
    split_sha256: str
    training_rows: int
    validation_rows: int


class _Round74NextEventHead(nn.Module):
    def __init__(self, hidden_channels: int) -> None:
        super().__init__()
        self.event_type = nn.Linear(
            int(hidden_channels),
            ROUND74_EVENT_PRETRAINING_EVENT_TYPE_COUNT,
        )
        self.continuous_delta = nn.Linear(
            int(hidden_channels),
            len(ROUND74_EVENT_FEATURE_NAMES) - ROUND74_EVENT_BINARY_FEATURE_COUNT,
        )

    def forward(self, encoded: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.event_type(encoded), self.continuous_delta(encoded)


def _readonly_indices(values: Sequence[int]) -> np.ndarray:
    selected = np.ascontiguousarray(values, dtype=np.int64)
    selected.setflags(write=False)
    return selected


def _index_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(values, dtype="<i8")
    return hashlib.sha256(memoryview(canonical).cast("B")).hexdigest()


def _update_array_digest(digest: object, values: np.ndarray) -> None:
    selected = np.asarray(values)
    canonical = np.ascontiguousarray(
        selected.astype(selected.dtype.newbyteorder("<"), copy=False)
    )
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(int(canonical.ndim).to_bytes(2, "little", signed=False))
    for size in canonical.shape:
        digest.update(int(size).to_bytes(8, "little", signed=False))
    digest.update(memoryview(canonical).cast("B"))


def _feature_batch_sha256(batch: Round74EventTrainingBatch) -> str:
    identity = {
        "schema_version": ROUND74_EVENT_PRETRAINING_SCHEMA_VERSION,
        "role": batch.role,
        "partition_sha256": batch.partition_sha256,
        "scaler_sha256": batch.scaler_sha256,
        "run_id": list(batch.run_id),
        "symbol": list(batch.symbol),
        "feature_window_sha256": list(batch.feature_window_sha256),
        "window_representation": batch.window_representation,
    }
    digest = hashlib.sha256(
        json.dumps(
            identity,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    )
    for values in (
        batch.decision_monotonic_ns,
        batch.decision_wall_ns,
        batch.endpoint_frame_index,
        batch.endpoint_message_index,
        batch.anchor_index,
        batch.feature_values,
    ):
        _update_array_digest(digest, values)
    return digest.hexdigest()


def _split_batch(
    batch: Round74EventTrainingBatch,
    *,
    config: Round74EventPretrainingConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    batch.validate()
    if batch.role != "training" or len(set(batch.run_id)) != 1:
        raise ValueError("Round 74 pretraining requires one training capture run")
    validation_count = max(
        config.minimum_partition_rows_per_symbol * len(IMPACT_CAPTURE_SYMBOLS),
        int(math.ceil(batch.rows * config.validation_fraction)),
    )
    validation_start = batch.rows - validation_count
    if validation_start < 1:
        raise ValueError("Round 74 pretraining population is too small")
    validation_candidates = np.arange(
        validation_start,
        batch.rows,
        dtype=np.int64,
    )
    validation_first_wall_ns = int(
        batch.decision_wall_ns[validation_candidates[0]]
    )
    training: list[int] = []
    validation: list[int] = []
    symbol_rows: dict[str, dict[str, int]] = {}
    for symbol in IMPACT_CAPTURE_SYMBOLS:
        symbol_mask = np.asarray(batch.symbol) == symbol
        indices = np.flatnonzero(symbol_mask)
        if len(indices) < config.minimum_partition_rows_per_symbol * 2:
            raise ValueError("Round 74 pretraining symbol population is too small")
        anchors = batch.anchor_index[indices]
        if np.any(np.diff(anchors) <= 0):
            raise ValueError("Round 74 pretraining anchor order differs")
        validation_indices = validation_candidates[
            symbol_mask[validation_candidates]
        ]
        training_candidates = indices[indices < validation_start]
        if len(validation_indices) < config.minimum_partition_rows_per_symbol:
            raise ValueError("Round 74 pretraining validation symbol is too small")
        first_validation_anchor = int(
            batch.anchor_index[validation_indices[0]]
        )
        maximum_training_anchor = (
            first_validation_anchor - config.purge_anchor_count - 1
        )
        training_indices = training_candidates[
            batch.anchor_index[training_candidates] <= maximum_training_anchor
        ]
        if (
            len(training_indices) < config.minimum_partition_rows_per_symbol
            or len(validation_indices) < config.minimum_partition_rows_per_symbol
            or int(batch.anchor_index[validation_indices[0]])
            - int(batch.anchor_index[training_indices[-1]])
            <= config.purge_anchor_count
        ):
            raise ValueError("Round 74 pretraining purged split is too small")
        training.extend(int(value) for value in training_indices)
        validation.extend(int(value) for value in validation_indices)
        symbol_rows[symbol] = {
            "training_rows": len(training_indices),
            "validation_rows": len(validation_indices),
            "purged_anchor_count": (
                int(batch.anchor_index[validation_indices[0]])
                - int(batch.anchor_index[training_indices[-1]])
                - 1
            ),
        }
    training_array = _readonly_indices(sorted(training))
    validation_array = _readonly_indices(sorted(validation))
    if np.intersect1d(training_array, validation_array).size:
        raise RuntimeError("Round 74 pretraining partitions overlap")
    if (
        int(batch.decision_wall_ns[training_array[-1]])
        >= validation_first_wall_ns
    ):
        raise RuntimeError("Round 74 pretraining chronology differs")
    return training_array, validation_array, {
        "run_id": batch.run_id[0],
        "feature_batch_sha256": _feature_batch_sha256(batch),
        "training_index_sha256": _index_sha256(training_array),
        "validation_index_sha256": _index_sha256(validation_array),
        "symbol_rows": symbol_rows,
        "training_first_wall_ns": int(batch.decision_wall_ns[training_array[0]]),
        "training_last_wall_ns": int(batch.decision_wall_ns[training_array[-1]]),
        "validation_first_wall_ns": validation_first_wall_ns,
        "validation_last_wall_ns": int(batch.decision_wall_ns[validation_array[-1]]),
    }


def build_round74_event_pretraining_split(
    batches: Sequence[Round74EventTrainingBatch],
    *,
    config: Round74EventPretrainingConfig | None = None,
) -> _Round74PretrainingSplit:
    """Build per-run, per-symbol chronological tails from training data only."""

    selected = config or Round74EventPretrainingConfig()
    selected.validate()
    if not batches:
        raise ValueError("Round 74 pretraining batches are empty")
    training_indices: list[np.ndarray] = []
    validation_indices: list[np.ndarray] = []
    reports: list[dict[str, object]] = []
    run_ids: list[str] = []
    for batch in batches:
        training, validation, report = _split_batch(batch, config=selected)
        training_indices.append(training)
        validation_indices.append(validation)
        reports.append(report)
        run_ids.append(str(report["run_id"]))
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Round 74 pretraining capture runs repeat")
    split_sha256 = _canonical_sha256(
        {
            "schema_version": ROUND74_EVENT_PRETRAINING_SCHEMA_VERSION,
            "config": selected.as_dict(),
            "runs": reports,
        }
    )
    return _Round74PretrainingSplit(
        training_indices=tuple(training_indices),
        validation_indices=tuple(validation_indices),
        split_sha256=split_sha256,
        training_rows=sum(len(value) for value in training_indices),
        validation_rows=sum(len(value) for value in validation_indices),
    )


def _feature_view_values(
    values: torch.Tensor,
    *,
    masked_feature_indices: Sequence[int],
) -> torch.Tensor:
    if not masked_feature_indices:
        return values
    selected = values.clone()
    selected[:, :, tuple(int(value) for value in masked_feature_indices)] = 0.0
    return selected


def _next_event_loss(
    model: nn.Module,
    head: _Round74NextEventHead,
    values: torch.Tensor,
    *,
    masked_feature_indices: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = _feature_view_values(
        values,
        masked_feature_indices=masked_feature_indices,
    )
    encoded = encode_round74_event_sequence(model, selected)
    event_logits, continuous_delta = head(encoded[:, :-1, :])
    next_event_one_hot = selected[
        :,
        1:,
        :ROUND74_EVENT_PRETRAINING_EVENT_TYPE_COUNT,
    ]
    if (
        not bool(((next_event_one_hot == 0.0) | (next_event_one_hot == 1.0)).all())
        or not bool((next_event_one_hot.sum(dim=2) == 1.0).all())
    ):
        raise ValueError("Round 74 pretraining event target differs")
    event_target = next_event_one_hot.argmax(dim=2)
    event_log_probability = F.log_softmax(event_logits, dim=2)
    event_loss = -event_log_probability.gather(
        2,
        event_target.unsqueeze(2),
    ).mean() / math.log(float(ROUND74_EVENT_PRETRAINING_EVENT_TYPE_COUNT))
    continuous = selected[:, :, ROUND74_EVENT_BINARY_FEATURE_COUNT:]
    target_delta = continuous[:, 1:, :] - continuous[:, :-1, :]
    residual = continuous_delta - target_delta
    absolute = residual.abs()
    continuous_loss = torch.where(
        absolute < 1.0,
        0.5 * residual.square(),
        absolute - 0.5,
    ).mean()
    loss = event_loss + continuous_loss
    if not all(bool(torch.isfinite(value)) for value in (loss, event_loss, continuous_loss)):
        raise ValueError("Round 74 pretraining loss is nonfinite")
    return loss, event_loss, continuous_loss


def _index_minibatches(indices: np.ndarray, rows: int) -> tuple[np.ndarray, ...]:
    return tuple(indices[start : start + rows] for start in range(0, len(indices), rows))


def _device_values(
    batch: Round74EventTrainingBatch,
    indices: np.ndarray,
    device: object,
) -> torch.Tensor:
    return torch.from_numpy(
        np.ascontiguousarray(batch.feature_values[indices])
    ).to(device)


def _evaluate_pretraining(
    model: nn.Module,
    head: _Round74NextEventHead,
    batches: Sequence[Round74EventTrainingBatch],
    validation_indices: Sequence[np.ndarray],
    *,
    config: Round74EventPretrainingConfig,
    device: object,
    masked_feature_indices: Sequence[int],
) -> tuple[float, tuple[float, ...]]:
    model.eval()
    head.eval()
    run_losses: list[float] = []
    with torch.no_grad():
        for batch, indices in zip(batches, validation_indices, strict=True):
            weighted_loss = 0.0
            row_count = 0
            for selected in _index_minibatches(indices, config.minibatch_rows):
                values = _device_values(batch, selected, device)
                loss, _event, _continuous = _next_event_loss(
                    model,
                    head,
                    values,
                    masked_feature_indices=masked_feature_indices,
                )
                weighted_loss += float(loss.detach().cpu()) * len(selected)
                row_count += len(selected)
            run_losses.append(weighted_loss / row_count)
    if not run_losses or not all(math.isfinite(value) for value in run_losses):
        raise RuntimeError("Round 74 pretraining validation loss differs")
    return sum(run_losses) / len(run_losses), tuple(run_losses)


def _parameter_state(parameters: Sequence[nn.Parameter]) -> tuple[torch.Tensor, ...]:
    return tuple(value.detach().cpu().contiguous().clone() for value in parameters)


def _restore_parameter_state(
    parameters: Sequence[nn.Parameter],
    state: Sequence[torch.Tensor],
) -> None:
    if len(parameters) != len(state):
        raise RuntimeError("Round 74 pretraining encoder state differs")
    with torch.no_grad():
        for parameter, value in zip(parameters, state, strict=True):
            parameter.copy_(value.to(parameter.device))


def _encoder_state_sha256(parameters: Sequence[nn.Parameter]) -> str:
    digest = hashlib.sha256()
    for index, parameter in enumerate(parameters):
        value = parameter.detach().cpu().numpy().astype("<f4", copy=False)
        digest.update(index.to_bytes(4, "little", signed=False))
        digest.update(memoryview(np.ascontiguousarray(value)).cast("B"))
    return digest.hexdigest()


def pretrain_round74_event_encoder(
    model: nn.Module,
    training_batches: Sequence[Round74EventTrainingBatch],
    *,
    device: object,
    masked_feature_indices: Sequence[int] = (),
    config: Round74EventPretrainingConfig | None = None,
) -> dict[str, object]:
    """Initialize one causal encoder and return a target-free audit report."""

    selected = config or Round74EventPretrainingConfig()
    selected.validate()
    hidden_channels = round74_event_model_pretraining_channels(model)
    if hidden_channels is None:
        raise ValueError("Round 74 model does not support causal pretraining")
    split = build_round74_event_pretraining_split(
        training_batches,
        config=selected,
    )
    encoder_parameters = round74_event_encoder_parameters(model)
    initial_encoder_sha256 = _encoder_state_sha256(encoder_parameters)
    head = _Round74NextEventHead(hidden_channels).to(device)
    optimized_parameters = (*encoder_parameters, *tuple(head.parameters()))
    optimizer = ExplicitAdamW(
        optimized_parameters,
        learning_rate=selected.learning_rate,
        weight_decay=selected.weight_decay,
    )
    best_loss = math.inf
    best_epoch = 0
    best_encoder_state: tuple[torch.Tensor, ...] | None = None
    best_head_state: dict[str, torch.Tensor] | None = None
    best_run_losses: tuple[float, ...] = ()
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    schedules = tuple(
        _index_minibatches(indices, selected.minibatch_rows)
        for indices in split.training_indices
    )
    optimizer_steps = max(len(schedule) for schedule in schedules)
    run_count = len(training_batches)
    for epoch in range(1, selected.maximum_epochs + 1):
        model.train()
        head.train()
        epoch_loss = 0.0
        epoch_event_loss = 0.0
        epoch_continuous_loss = 0.0
        for step in range(optimizer_steps):
            optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            step_event_loss = 0.0
            step_continuous_loss = 0.0
            for batch, schedule in zip(training_batches, schedules, strict=True):
                indices = schedule[(step + epoch - 1) % len(schedule)]
                values = _device_values(batch, indices, device)
                loss, event_loss, continuous_loss = _next_event_loss(
                    model,
                    head,
                    values,
                    masked_feature_indices=masked_feature_indices,
                )
                (loss / run_count).backward()
                step_loss += float(loss.detach().cpu()) / run_count
                step_event_loss += float(event_loss.detach().cpu()) / run_count
                step_continuous_loss += (
                    float(continuous_loss.detach().cpu()) / run_count
                )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                optimized_parameters,
                max_norm=selected.gradient_clip_norm,
                foreach=False,
            )
            if not math.isfinite(float(gradient_norm.detach().cpu())):
                raise RuntimeError("Round 74 pretraining gradient norm is nonfinite")
            optimizer.step()
            epoch_loss += step_loss
            epoch_event_loss += step_event_loss
            epoch_continuous_loss += step_continuous_loss
        validation_loss, validation_run_losses = _evaluate_pretraining(
            model,
            head,
            training_batches,
            split.validation_indices,
            config=selected,
            device=device,
            masked_feature_indices=masked_feature_indices,
        )
        improved = (
            best_encoder_state is None
            or validation_loss
            < best_loss - selected.minimum_validation_improvement
        )
        if improved:
            best_loss = validation_loss
            best_epoch = epoch
            best_encoder_state = _parameter_state(encoder_parameters)
            best_head_state = {
                name: value.detach().cpu().contiguous().clone()
                for name, value in sorted(head.state_dict().items())
            }
            best_run_losses = validation_run_losses
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append(
            {
                "epoch": epoch,
                "training_run_balanced_loss": epoch_loss / optimizer_steps,
                "training_event_loss": epoch_event_loss / optimizer_steps,
                "training_continuous_delta_loss": (
                    epoch_continuous_loss / optimizer_steps
                ),
                "validation_run_balanced_loss": validation_loss,
                "validation_run_losses": list(validation_run_losses),
                "improved": improved,
            }
        )
        if epochs_without_improvement >= selected.early_stopping_patience:
            break
    if best_encoder_state is None or best_head_state is None or best_epoch < 1:
        raise RuntimeError("Round 74 pretraining has no finite early-stop state")
    _restore_parameter_state(encoder_parameters, best_encoder_state)
    head.load_state_dict(best_head_state, strict=True)
    restored_loss, restored_run_losses = _evaluate_pretraining(
        model,
        head,
        training_batches,
        split.validation_indices,
        config=selected,
        device=device,
        masked_feature_indices=masked_feature_indices,
    )
    if (
        not math.isclose(restored_loss, best_loss, rel_tol=1e-7, abs_tol=1e-7)
        or any(
            not math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-7)
            for left, right in zip(
                restored_run_losses,
                best_run_losses,
                strict=True,
            )
        )
    ):
        raise RuntimeError("Round 74 pretraining best-state reload differs")
    final_encoder_sha256 = _encoder_state_sha256(encoder_parameters)
    if final_encoder_sha256 == initial_encoder_sha256:
        raise RuntimeError("Round 74 pretraining did not update its encoder")
    return {
        "schema_version": ROUND74_EVENT_PRETRAINING_SCHEMA_VERSION,
        "initialization_id": "causal_next_event_pretrained",
        "config": selected.as_dict(),
        "split_sha256": split.split_sha256,
        "training_feature_batch_sha256": [
            value
            for value in sorted(
                _feature_batch_sha256(batch) for batch in training_batches
            )
        ],
        "training_capture_run_count": len(training_batches),
        "training_rows": split.training_rows,
        "validation_rows": split.validation_rows,
        "masked_feature_indices": [int(value) for value in masked_feature_indices],
        "initial_encoder_sha256": initial_encoder_sha256,
        "final_encoder_sha256": final_encoder_sha256,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_run_balanced_loss": best_loss,
        "best_validation_run_losses": list(best_run_losses),
        "history": history,
        "encoder_state_restored": True,
        "temporary_prediction_head_persisted": False,
        "supervised_targets_used": False,
        "tuning_features_used": False,
        "tuning_targets_used": False,
        "calibration_data_used": False,
        "test_data_used": False,
        "financial_edge_claim": False,
    }


__all__ = [
    "ROUND74_EVENT_PRETRAINING_INITIALIZATION_IDS",
    "ROUND74_EVENT_PRETRAINING_SCHEMA_VERSION",
    "Round74EventPretrainingConfig",
    "build_round74_event_pretraining_split",
    "pretrain_round74_event_encoder",
]
