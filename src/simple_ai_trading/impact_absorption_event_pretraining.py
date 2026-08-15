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
import operator

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


ROUND74_EVENT_PRETRAINING_SCHEMA_VERSION = "round-074-causal-next-event-pretraining-v5"
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


def _is_sha256(value: object) -> bool:
    selected = str(value)
    return len(selected) == 64 and all(
        character in "0123456789abcdef" for character in selected
    )


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
    device_run_group_size: int = 8

    def validate(self) -> None:
        integer_values = (
            self.maximum_epochs,
            self.early_stopping_patience,
            self.minibatch_rows,
            self.purge_anchor_count,
            self.minimum_partition_rows_per_symbol,
            self.device_run_group_size,
        )
        numeric_values = (
            self.minimum_validation_improvement,
            self.learning_rate,
            self.weight_decay,
            self.gradient_clip_norm,
            self.validation_fraction,
        )
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in integer_values
            )
            or self.maximum_epochs < 1
            or self.early_stopping_patience < 1
            or self.early_stopping_patience > self.maximum_epochs
            or self.minibatch_rows < 1
            or self.purge_anchor_count < ROUND74_EVENT_SEQUENCE_LENGTH
            or self.minimum_partition_rows_per_symbol < 1
            or self.device_run_group_size < 1
            or self.device_run_group_size > 32
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
            "device_run_group_size": self.device_run_group_size,
            "fit_partition_role": "training",
            "validation_partition_role": "purged_tail_within_training",
            "event_target": "next_event_type",
            "continuous_target": (
                "next_scaled_feature_delta_on_unmasked_continuous_dimensions"
            ),
            "masked_continuous_target_policy": "exclude_from_loss_not_zero_impute",
            "event_loss": "categorical_log_loss_divided_by_log_event_type_count",
            "continuous_loss": "dimension_mean_huber_beta_1",
            "population_weighting": "bound_by_training_execution_mode",
            "device_grouping": (
                "concatenated forward with bounded groups and exact objective weights"
            ),
            "shorter_run_policy": "bound_by_training_execution_mode",
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
                device_run_group_size=int(payload["device_run_group_size"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Round 74 event pretraining policy differs") from exc
        if selected.as_dict() != dict(payload):
            raise ValueError("Round 74 event pretraining policy differs")
        return selected


@dataclass(frozen=True)
class Round74EventPretrainingSplit:
    training_indices: tuple[np.ndarray, ...]
    validation_indices: tuple[np.ndarray, ...]
    split_sha256: str
    config_sha256: str
    training_feature_object_ids: tuple[tuple[int, ...], ...]
    training_feature_batch_sha256: tuple[str, ...]
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


def _feature_object_ids(batch: Round74EventTrainingBatch) -> tuple[int, ...]:
    return tuple(
        id(value)
        for value in (
            batch.run_id,
            batch.symbol,
            batch.feature_window_sha256,
            batch.decision_monotonic_ns,
            batch.decision_wall_ns,
            batch.endpoint_frame_index,
            batch.endpoint_message_index,
            batch.anchor_index,
            batch.feature_values,
        )
    )


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
    validation_first_wall_ns = int(batch.decision_wall_ns[validation_candidates[0]])
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
        validation_indices = validation_candidates[symbol_mask[validation_candidates]]
        training_candidates = indices[indices < validation_start]
        if len(validation_indices) < config.minimum_partition_rows_per_symbol:
            raise ValueError("Round 74 pretraining validation symbol is too small")
        first_validation_anchor = int(batch.anchor_index[validation_indices[0]])
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
    if int(batch.decision_wall_ns[training_array[-1]]) >= validation_first_wall_ns:
        raise RuntimeError("Round 74 pretraining chronology differs")
    return (
        training_array,
        validation_array,
        {
            "run_id": batch.run_id[0],
            "feature_batch_sha256": _feature_batch_sha256(batch),
            "training_index_sha256": _index_sha256(training_array),
            "validation_index_sha256": _index_sha256(validation_array),
            "symbol_rows": symbol_rows,
            "training_first_wall_ns": int(batch.decision_wall_ns[training_array[0]]),
            "training_last_wall_ns": int(batch.decision_wall_ns[training_array[-1]]),
            "validation_first_wall_ns": validation_first_wall_ns,
            "validation_last_wall_ns": int(
                batch.decision_wall_ns[validation_array[-1]]
            ),
        },
    )


def build_round74_event_pretraining_split(
    batches: Sequence[Round74EventTrainingBatch],
    *,
    config: Round74EventPretrainingConfig | None = None,
) -> Round74EventPretrainingSplit:
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
    return Round74EventPretrainingSplit(
        training_indices=tuple(training_indices),
        validation_indices=tuple(validation_indices),
        split_sha256=split_sha256,
        config_sha256=_canonical_sha256(selected.as_dict()),
        training_feature_object_ids=tuple(
            _feature_object_ids(batch) for batch in batches
        ),
        training_feature_batch_sha256=tuple(
            sorted(str(report["feature_batch_sha256"]) for report in reports)
        ),
        training_rows=sum(len(value) for value in training_indices),
        validation_rows=sum(len(value) for value in validation_indices),
    )


def _validate_pretraining_split(
    split: Round74EventPretrainingSplit,
    batches: Sequence[Round74EventTrainingBatch],
    config: Round74EventPretrainingConfig,
) -> None:
    if (
        not isinstance(split, Round74EventPretrainingSplit)
        or not _is_sha256(split.split_sha256)
        or split.config_sha256 != _canonical_sha256(config.as_dict())
        or split.training_feature_object_ids
        != tuple(_feature_object_ids(batch) for batch in batches)
        or len(split.training_indices) != len(batches)
        or len(split.validation_indices) != len(batches)
        or len(split.training_feature_batch_sha256) != len(batches)
        or len(split.training_feature_batch_sha256)
        != len(set(split.training_feature_batch_sha256))
        or any(not _is_sha256(value) for value in split.training_feature_batch_sha256)
        or split.training_rows
        != sum(len(indices) for indices in split.training_indices)
        or split.validation_rows
        != sum(len(indices) for indices in split.validation_indices)
        or split.training_rows < 1
        or split.validation_rows < 1
    ):
        raise ValueError("Round 74 pretraining split contract differs")
    for batch, training, validation in zip(
        batches,
        split.training_indices,
        split.validation_indices,
        strict=True,
    ):
        if (
            training.dtype != np.int64
            or validation.dtype != np.int64
            or training.flags.writeable
            or validation.flags.writeable
            or len(training) < 1
            or len(validation) < 1
            or int(training[0]) < 0
            or int(validation[0]) < 0
            or int(training[-1]) >= batch.rows
            or int(validation[-1]) >= batch.rows
            or np.any(np.diff(training) <= 0)
            or np.any(np.diff(validation) <= 0)
            or np.intersect1d(training, validation).size
            or int(batch.decision_wall_ns[training[-1]])
            >= int(batch.decision_wall_ns[validation[0]])
        ):
            raise ValueError("Round 74 pretraining split indices differ")


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


def _continuous_target_indices(
    masked_feature_indices: Sequence[int],
) -> tuple[int, ...]:
    """Map unmasked continuous features to the temporary head's output indices."""

    masked: set[int] = set()
    for value in masked_feature_indices:
        if isinstance(value, bool):
            raise ValueError("Round 74 pretraining feature mask differs")
        try:
            index = operator.index(value)
        except TypeError as exc:
            raise ValueError("Round 74 pretraining feature mask differs") from exc
        if index < 0 or index >= len(ROUND74_EVENT_FEATURE_NAMES) or index in masked:
            raise ValueError("Round 74 pretraining feature mask differs")
        masked.add(index)
    selected = tuple(
        feature_index - ROUND74_EVENT_BINARY_FEATURE_COUNT
        for feature_index in range(
            ROUND74_EVENT_BINARY_FEATURE_COUNT,
            len(ROUND74_EVENT_FEATURE_NAMES),
        )
        if feature_index not in masked
    )
    if not selected:
        raise ValueError("Round 74 pretraining continuous target is empty")
    return selected


def _next_event_row_losses(
    model: nn.Module,
    head: _Round74NextEventHead,
    values: torch.Tensor,
    *,
    masked_feature_indices: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    continuous_target_indices = _continuous_target_indices(masked_feature_indices)
    selected = _feature_view_values(
        values,
        masked_feature_indices=masked_feature_indices,
    )
    encoded = encode_round74_event_sequence(model, selected)
    event_logits, continuous_delta = head(encoded[:, :-1, :])
    next_event_one_hot = values[
        :,
        1:,
        :ROUND74_EVENT_PRETRAINING_EVENT_TYPE_COUNT,
    ]
    if not bool(
        ((next_event_one_hot == 0.0) | (next_event_one_hot == 1.0)).all()
    ) or not bool((next_event_one_hot.sum(dim=2) == 1.0).all()):
        raise ValueError("Round 74 pretraining event target differs")
    event_target = next_event_one_hot.argmax(dim=2)
    event_log_probability = F.log_softmax(event_logits, dim=2)
    event_row_loss = -event_log_probability.gather(
        2,
        event_target.unsqueeze(2),
    ).squeeze(2).mean(dim=1) / math.log(
        float(ROUND74_EVENT_PRETRAINING_EVENT_TYPE_COUNT)
    )
    continuous = selected[:, :, ROUND74_EVENT_BINARY_FEATURE_COUNT:]
    target_delta = continuous[:, 1:, :] - continuous[:, :-1, :]
    residual = (
        continuous_delta[:, :, continuous_target_indices]
        - target_delta[:, :, continuous_target_indices]
    )
    absolute = residual.abs()
    continuous_row_loss = torch.where(
        absolute < 1.0,
        0.5 * residual.square(),
        absolute - 0.5,
    ).mean(dim=(1, 2))
    row_loss = event_row_loss + continuous_row_loss
    if (
        row_loss.shape != (values.shape[0],)
        or event_row_loss.shape != row_loss.shape
        or continuous_row_loss.shape != row_loss.shape
        or not all(
            bool(torch.isfinite(value).all())
            for value in (row_loss, event_row_loss, continuous_row_loss)
        )
    ):
        raise ValueError("Round 74 pretraining loss is nonfinite")
    return row_loss, event_row_loss, continuous_row_loss


def _next_event_loss(
    model: nn.Module,
    head: _Round74NextEventHead,
    values: torch.Tensor,
    *,
    masked_feature_indices: Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    row_loss, event_row_loss, continuous_row_loss = _next_event_row_losses(
        model,
        head,
        values,
        masked_feature_indices=masked_feature_indices,
    )
    return row_loss.mean(), event_row_loss.mean(), continuous_row_loss.mean()


def _index_minibatches(indices: np.ndarray, rows: int) -> tuple[np.ndarray, ...]:
    return tuple(
        indices[start : start + rows] for start in range(0, len(indices), rows)
    )


def _device_values(
    batch: Round74EventTrainingBatch,
    indices: np.ndarray,
    device: object,
) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(batch.feature_values[indices])).to(
        device
    )


def _device_group_values(
    selections: Sequence[tuple[Round74EventTrainingBatch, np.ndarray]],
    device: object,
) -> tuple[torch.Tensor, tuple[int, ...]]:
    if not selections:
        raise ValueError("Round 74 pretraining device group is empty")
    row_counts = tuple(len(indices) for _batch, indices in selections)
    values = np.concatenate(
        tuple(batch.feature_values[indices] for batch, indices in selections),
        axis=0,
    )
    return torch.from_numpy(np.ascontiguousarray(values)).to(device), row_counts


def _pretraining_population_policy(execution_mode: str) -> dict[str, object]:
    if execution_mode == "cohort":
        return {
            "execution_mode": execution_mode,
            "objective_weighting": "equal_capture_run",
            "shorter_run_policy": "deterministic_epoch_rotated_cycling",
            "optimizer_step_policy": "one_step_per_longest_run_minibatch",
            "each_training_row_visited_once_per_epoch": False,
        }
    if execution_mode == "segmented_cohort":
        return {
            "execution_mode": execution_mode,
            "objective_weighting": "eligible_row_duration_proportional",
            "shorter_run_policy": "single_visit_without_cycling",
            "optimizer_step_policy": "one_accumulated_step_per_epoch",
            "each_training_row_visited_once_per_epoch": True,
        }
    if execution_mode == "preflight":
        return {
            "execution_mode": execution_mode,
            "objective_weighting": "not_applicable",
            "shorter_run_policy": "not_applicable",
            "optimizer_step_policy": "not_applicable",
            "each_training_row_visited_once_per_epoch": False,
        }
    raise ValueError("Round 74 pretraining execution mode differs")


def _single_visit_pretraining_schedule(
    schedules: Sequence[Sequence[np.ndarray]],
) -> tuple[tuple[int, np.ndarray], ...]:
    if not schedules or any(not schedule for schedule in schedules):
        raise ValueError("Round 74 segmented pretraining schedule differs")
    return tuple(
        (run_index, indices)
        for run_index, schedule in enumerate(schedules)
        for indices in schedule
    )


def _backpropagate_pretraining_groups(
    model: nn.Module,
    head: _Round74NextEventHead,
    selections: Sequence[tuple[Round74EventTrainingBatch, np.ndarray]],
    *,
    device: object,
    masked_feature_indices: Sequence[int],
    device_run_group_size: int,
    equal_run_denominator: int | None,
    row_denominator: int | None,
) -> tuple[float, float, float]:
    if not selections or (equal_run_denominator is None) is (row_denominator is None):
        raise ValueError("Round 74 pretraining objective population differs")
    loss_total = 0.0
    event_total = 0.0
    continuous_total = 0.0
    for start in range(0, len(selections), device_run_group_size):
        group = selections[start : start + device_run_group_size]
        values, row_counts = _device_group_values(group, device)
        row_loss, event_row_loss, continuous_row_loss = _next_event_row_losses(
            model,
            head,
            values,
            masked_feature_indices=masked_feature_indices,
        )
        if equal_run_denominator is not None:
            loss_parts: list[torch.Tensor] = []
            event_parts: list[torch.Tensor] = []
            continuous_parts: list[torch.Tensor] = []
            offset = 0
            for row_count in row_counts:
                stop = offset + row_count
                loss_parts.append(row_loss[offset:stop].mean())
                event_parts.append(event_row_loss[offset:stop].mean())
                continuous_parts.append(continuous_row_loss[offset:stop].mean())
                offset = stop
            if offset != values.shape[0]:
                raise RuntimeError("Round 74 pretraining device group rows differ")
            loss = torch.stack(loss_parts).sum() / equal_run_denominator
            event = torch.stack(event_parts).sum() / equal_run_denominator
            continuous = torch.stack(continuous_parts).sum() / equal_run_denominator
        else:
            assert row_denominator is not None
            loss = row_loss.sum() / row_denominator
            event = event_row_loss.sum() / row_denominator
            continuous = continuous_row_loss.sum() / row_denominator
        loss.backward()
        loss_total += float(loss.detach().cpu())
        event_total += float(event.detach().cpu())
        continuous_total += float(continuous.detach().cpu())
    return loss_total, event_total, continuous_total


def _evaluate_pretraining(
    model: nn.Module,
    head: _Round74NextEventHead,
    batches: Sequence[Round74EventTrainingBatch],
    validation_indices: Sequence[np.ndarray],
    *,
    config: Round74EventPretrainingConfig,
    device: object,
    masked_feature_indices: Sequence[int],
    execution_mode: str,
) -> tuple[float, tuple[float, ...]]:
    model.eval()
    head.eval()
    run_losses: list[float] = []
    population_weighted_loss = 0.0
    population_rows = 0
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
            population_weighted_loss += weighted_loss
            population_rows += row_count
    if not run_losses or not all(math.isfinite(value) for value in run_losses):
        raise RuntimeError("Round 74 pretraining validation loss differs")
    if execution_mode == "cohort":
        objective_loss = sum(run_losses) / len(run_losses)
    elif execution_mode == "segmented_cohort" and population_rows > 0:
        objective_loss = population_weighted_loss / population_rows
    else:
        raise ValueError("Round 74 pretraining validation population differs")
    return objective_loss, tuple(run_losses)


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
    split: Round74EventPretrainingSplit | None = None,
    execution_mode: str = "cohort",
) -> dict[str, object]:
    """Initialize one causal encoder and return a target-free audit report."""

    selected = config or Round74EventPretrainingConfig()
    selected.validate()
    population_policy = _pretraining_population_policy(execution_mode)
    if execution_mode == "preflight":
        raise ValueError("Round 74 preflight cannot perform causal pretraining")
    hidden_channels = round74_event_model_pretraining_channels(model)
    if hidden_channels is None:
        raise ValueError("Round 74 model does not support causal pretraining")
    selected_split = split or build_round74_event_pretraining_split(
        training_batches,
        config=selected,
    )
    _validate_pretraining_split(selected_split, training_batches, selected)
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
        for indices in selected_split.training_indices
    )
    if not schedules or any(not schedule for schedule in schedules):
        raise ValueError("Round 74 pretraining schedule differs")
    optimizer_steps = (
        max(len(schedule) for schedule in schedules)
        if execution_mode == "cohort"
        else 1
    )
    run_count = len(training_batches)
    for epoch in range(1, selected.maximum_epochs + 1):
        model.train()
        head.train()
        epoch_loss = 0.0
        epoch_event_loss = 0.0
        epoch_continuous_loss = 0.0
        if execution_mode == "cohort":
            epoch_selections = tuple(
                tuple(
                    (
                        batch,
                        schedule[(step + epoch - 1) % len(schedule)],
                    )
                    for batch, schedule in zip(
                        training_batches,
                        schedules,
                        strict=True,
                    )
                )
                for step in range(optimizer_steps)
            )
        else:
            epoch_selections = (
                tuple(
                    (training_batches[run_index], indices)
                    for run_index, indices in _single_visit_pretraining_schedule(
                        schedules
                    )
                ),
            )
        for selections in epoch_selections:
            optimizer.zero_grad(set_to_none=True)
            total_rows = (
                sum(len(indices) for _batch, indices in selections)
                if execution_mode == "segmented_cohort"
                else None
            )
            step_loss, step_event_loss, step_continuous_loss = (
                _backpropagate_pretraining_groups(
                    model,
                    head,
                    selections,
                    device=device,
                    masked_feature_indices=masked_feature_indices,
                    device_run_group_size=selected.device_run_group_size,
                    equal_run_denominator=(
                        run_count if execution_mode == "cohort" else None
                    ),
                    row_denominator=total_rows,
                )
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
            selected_split.validation_indices,
            config=selected,
            device=device,
            masked_feature_indices=masked_feature_indices,
            execution_mode=execution_mode,
        )
        improved = (
            best_encoder_state is None
            or validation_loss < best_loss - selected.minimum_validation_improvement
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
                "training_objective_loss": epoch_loss / optimizer_steps,
                "training_event_loss": epoch_event_loss / optimizer_steps,
                "training_continuous_delta_loss": (
                    epoch_continuous_loss / optimizer_steps
                ),
                "validation_objective_loss": validation_loss,
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
        selected_split.validation_indices,
        config=selected,
        device=device,
        masked_feature_indices=masked_feature_indices,
        execution_mode=execution_mode,
    )
    if not math.isclose(restored_loss, best_loss, rel_tol=1e-7, abs_tol=1e-7) or any(
        not math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-7)
        for left, right in zip(
            restored_run_losses,
            best_run_losses,
            strict=True,
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
        "execution_mode": execution_mode,
        "population_policy": population_policy,
        "optimizer_steps_per_epoch": optimizer_steps,
        "split_sha256": selected_split.split_sha256,
        "training_feature_batch_sha256": list(
            selected_split.training_feature_batch_sha256
        ),
        "training_capture_run_count": len(training_batches),
        "training_rows": selected_split.training_rows,
        "validation_rows": selected_split.validation_rows,
        "masked_feature_indices": [int(value) for value in masked_feature_indices],
        "continuous_target_feature_indices": [
            int(value + ROUND74_EVENT_BINARY_FEATURE_COUNT)
            for value in _continuous_target_indices(masked_feature_indices)
        ],
        "initial_encoder_sha256": initial_encoder_sha256,
        "final_encoder_sha256": final_encoder_sha256,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_validation_objective_loss": best_loss,
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
    "Round74EventPretrainingSplit",
    "build_round74_event_pretraining_split",
    "pretrain_round74_event_encoder",
]
