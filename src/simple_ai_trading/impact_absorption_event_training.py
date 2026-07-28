"""Deterministic development-only training for Round 74 event models.

This module deliberately stops at a pretest model policy. It cannot consume a
test batch, calculate trading returns, or grant any execution authority.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import random
import re
import warnings

import numpy as np
from safetensors.torch import load as load_safetensors
from safetensors.torch import save as save_safetensors
import torch
from torch import nn

from .compute import require_backend, resolve_backend, torch_device_for_backend
from .distributional_tcn_model import ExplicitAdamW
from .impact_absorption_event_dataset import (
    ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS,
    ROUND74_EVENT_WINDOW_REPRESENTATIONS,
    Round74EventTrainingBatch,
)
from .impact_absorption_event_features import (
    ROUND74_EVENT_CLOCK_FEATURE_INDICES,  # noqa: F401 - public compatibility
    ROUND74_EVENT_CLOCK_FEATURE_NAMES,
    ROUND74_EVENT_CLOCK_FEATURE_NAMES_SHA256,
    ROUND74_EVENT_FEATURE_VIEW_MASKED_INDICES,
    ROUND74_EVENT_FEATURE_VIEW_SCHEMA_VERSION,
    ROUND74_EVENT_FEATURE_VIEWS,
    ROUND74_EVENT_MARKET_STATE_FEATURE_NAMES,
    ROUND74_EVENT_MARKET_STATE_FEATURE_NAMES_SHA256,
    ROUND74_EVENT_ORDER_FLOW_FEATURE_INDICES,  # noqa: F401 - public compatibility
    ROUND74_EVENT_ORDER_FLOW_FEATURE_NAMES,
    ROUND74_EVENT_ORDER_FLOW_FEATURE_NAMES_SHA256,
)
from .impact_absorption_event_model import (
    ROUND74_EVENT_MODEL_CANDIDATES,
    ROUND74_EVENT_MODEL_SCHEMA_VERSION,
    Round74EventModelOutput,
    _round74_event_model_loss_from_validated_inputs,
    build_round74_event_model,
    round74_event_model_pretraining_channels,
    round74_event_model_loss,
)
from .impact_absorption_event_pretraining import (
    ROUND74_EVENT_PRETRAINING_INITIALIZATION_IDS,
    ROUND74_EVENT_PRETRAINING_SCHEMA_VERSION,
    Round74EventPretrainingConfig,
    Round74EventPretrainingSplit,
    build_round74_event_pretraining_split,
    pretrain_round74_event_encoder,
)
from .impact_absorption_event_scaling import Round74EventFeatureScaler
from .impact_absorption_event_sequence import (
    ROUND74_EVENT_FEATURE_NAMES,
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_SIDES,
)
from .impact_absorption_store import IMPACT_CAPTURE_SYMBOLS
from .round74_segmented_model_operator import (
    ROUND74_SEGMENTED_EARLY_STOPPING_FRACTION_DENOMINATOR as ROUND74_EVENT_SEGMENTED_EARLY_STOPPING_FRACTION_DENOMINATOR,
    ROUND74_SEGMENTED_MINIMUM_EARLY_STOPPING_RUNS as ROUND74_EVENT_SEGMENTED_MINIMUM_EARLY_STOPPING_RUNS,
    ROUND74_SEGMENTED_MINIMUM_OPTIMIZATION_RUNS as ROUND74_EVENT_SEGMENTED_MINIMUM_OPTIMIZATION_RUNS,
    ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS,
    Round74SegmentedModelSelectionStages,
    Round74SegmentedTrainingSplit,
    Round74SegmentedTuningSubpartition,
    build_round74_segmented_model_selection_stages,
)
from .storage import write_bytes_atomic


ROUND74_EVENT_TRAINING_SCHEMA_VERSION = "round-074-event-training-v30"
ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION = "round-074-event-pretest-policy-v29"
ROUND74_EVENT_SELECTION_PROTOCOL_SCHEMA_VERSION = (
    "round-074-event-selection-protocol-v4"
)
ROUND74_EVENT_TARGET_CONTEXT_PANEL_SCHEMA_VERSION = "round-074-target-context-panel-v1"
ROUND74_EVENT_TARGET_LOSS_SCALE_SCHEMA_VERSION = "round-074-target-loss-scale-v1"
ROUND74_EVENT_STATE_CONDITIONED_FLOW_SCHEMA_VERSION = (
    "round-074-state-conditioned-flow-v1"
)
ROUND74_EVENT_TRAINING_DEFAULT_SEEDS = (7411, 7423, 7433)
ROUND74_COMPLEXITY_PROMOTION_COMPARISON_COUNT = len(ROUND74_EVENT_MODEL_CANDIDATES) - 1
ROUND74_COMPLEXITY_PROMOTION_REQUIRED_TUNING_RUNS = 12
ROUND74_EVENT_TRAINING_REQUIRED_CAPTURE_RUNS = 120
ROUND74_EVENT_TRAINING_LOSS_WEIGHTS = {
    "maximum_adverse_excursion": 0.35,
    "positive_payoff": 0.25,
    "adverse_selection": 0.20,
    "regime_unpredictability": 0.10,
}

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SAFE_FILENAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,159}")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _cohort_window_policy_identity(
    representative_window_policy_sha256: object,
    matched_preparation_sha256: object,
) -> tuple[str, str | None]:
    from .round74_event_model_operator import (
        round74_matched_representative_window_policy,
        round74_representative_window_policy,
    )
    from .round74_segmented_model_operator import round74_segmented_window_policy

    claimed = str(representative_window_policy_sha256)
    policies = {
        "single_representation": round74_representative_window_policy()[
            "policy_sha256"
        ],
        "matched_representation": (
            round74_matched_representative_window_policy()["policy_sha256"]
        ),
        "segmented_duration_normalized": (
            round74_segmented_window_policy()["policy_sha256"]
        ),
    }
    matching = tuple(kind for kind, digest in policies.items() if digest == claimed)
    if len(matching) != 1:
        raise ValueError("Round 74 representative window policy differs")
    kind = matching[0]
    if kind == "matched_representation":
        matched = str(matched_preparation_sha256)
        if _SHA256.fullmatch(matched) is None:
            raise ValueError("Round 74 matched preparation identity differs")
        return kind, matched
    if matched_preparation_sha256 is not None:
        raise ValueError("Round 74 single-representation preparation differs")
    return kind, None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _optimization_population_policy(
    execution_mode: str = "cohort",
    training_run_count: int = ROUND74_EVENT_TRAINING_REQUIRED_CAPTURE_RUNS,
    tuning_run_count: int = ROUND74_COMPLEXITY_PROMOTION_REQUIRED_TUNING_RUNS,
) -> dict[str, object]:
    if execution_mode == "segmented_cohort":
        return {
            "unit": "eligible_target",
            "optimizer_step": (
                "one exact full-development gradient per epoch accumulated "
                "through bounded device groups"
            ),
            "gradient_divisor": (
                "separate total eligible action and regime target counts"
            ),
            "shorter_run_policy": "no cycling; each selected row appears once per epoch",
            "fully_censored_minibatches_contribute_gradients": False,
            "fully_censored_capture_run_policy": "reject",
            "row_pooled_optimizer_steps_permitted": True,
            "training_capture_runs": int(training_run_count),
            "model_selection_capture_runs": int(tuning_run_count),
            "run_count_used_as_gradient_weight": False,
            "calibration_or_policy_selection_runs_used_for_candidate_fit": False,
        }
    return {
        "unit": "capture_run",
        "optimizer_step": (
            "one eligible minibatch per training capture run with gradient accumulation"
        ),
        "gradient_divisor": "training_capture_run_count",
        "shorter_run_policy": (
            "deterministic epoch-rotated cycling of eligible minibatches"
        ),
        "fully_censored_minibatches_contribute_gradients": False,
        "fully_censored_capture_run_policy": "reject",
        "row_pooled_optimizer_steps_permitted": False,
        "cohort_training_capture_runs": (ROUND74_EVENT_TRAINING_REQUIRED_CAPTURE_RUNS),
        "cohort_model_selection_capture_runs": (
            ROUND74_COMPLEXITY_PROMOTION_REQUIRED_TUNING_RUNS
        ),
        "calibration_or_policy_selection_runs_used_for_candidate_fit": False,
    }


def _reject_duplicate_json_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("Round 74 pretest policy has duplicate JSON keys")
        output[key] = value
    return output


def _write_immutable_bytes(path: Path, payload: bytes) -> None:
    """Publish once through a same-directory temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise FileExistsError(f"immutable Round 74 artifact already exists: {path}")
    write_bytes_atomic(path, payload)


def _load_scaler_bytes(payload: bytes) -> Round74EventFeatureScaler:
    try:
        raw = json.loads(
            payload.decode("ascii"),
            object_pairs_hook=_reject_duplicate_json_pairs,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 74 pretest scaler could not be read") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("Round 74 pretest scaler root differs")
    try:
        return Round74EventFeatureScaler.from_dict(raw)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Round 74 pretest scaler payload differs") from exc


@dataclass(frozen=True)
class Round74EventTrainingConfig:
    """Bounded training and chronological early-stopping policy."""

    candidate_ids: tuple[str, ...] = ROUND74_EVENT_MODEL_CANDIDATES
    seeds: tuple[int, ...] = ROUND74_EVENT_TRAINING_DEFAULT_SEEDS
    maximum_epochs: int = 48
    early_stopping_patience: int = 8
    minimum_tuning_improvement: float = 1e-5
    minibatch_rows: int = 128
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    minimum_role_rows: int = 1_024
    device_run_group_size: int = 8
    execution_mode: str = "cohort"
    architecture_selection_mode: str = "complexity_gate"
    pretraining: Round74EventPretrainingConfig = Round74EventPretrainingConfig()

    def validate(self) -> None:
        if not isinstance(self.pretraining, Round74EventPretrainingConfig):
            raise ValueError("Round 74 event pretraining policy differs")
        self.pretraining.validate()
        candidate_panel_is_valid = (
            self.architecture_selection_mode == "complexity_gate"
            and self.candidate_ids == ROUND74_EVENT_MODEL_CANDIDATES
        ) or (
            self.architecture_selection_mode == "fixed"
            and len(self.candidate_ids) == 1
            and self.candidate_ids[0] in ROUND74_EVENT_MODEL_CANDIDATES
        )
        if (
            not candidate_panel_is_valid
            or not self.seeds
            or len(self.seeds) != len(set(self.seeds))
            or any(
                isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
                for seed in self.seeds
            )
            or int(self.maximum_epochs) < 1
            or int(self.early_stopping_patience) < 1
            or int(self.early_stopping_patience) > int(self.maximum_epochs)
            or int(self.minibatch_rows) < 1
            or int(self.minimum_role_rows) < 1
            or int(self.device_run_group_size) < 1
            or int(self.device_run_group_size) > 32
            or self.execution_mode not in {"cohort", "segmented_cohort", "preflight"}
        ):
            raise ValueError("Round 74 event training configuration differs")
        values = (
            float(self.minimum_tuning_improvement),
            float(self.learning_rate),
            float(self.weight_decay),
            float(self.gradient_clip_norm),
        )
        if (
            not all(math.isfinite(value) for value in values)
            or values[0] < 0.0
            or values[1] <= 0.0
            or values[2] < 0.0
            or values[3] <= 0.0
        ):
            raise ValueError("Round 74 event training numeric policy differs")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "candidate_ids": list(self.candidate_ids),
            "seeds": list(self.seeds),
            "maximum_epochs": int(self.maximum_epochs),
            "early_stopping_patience": int(self.early_stopping_patience),
            "minimum_tuning_improvement": float(self.minimum_tuning_improvement),
            "minibatch_rows": int(self.minibatch_rows),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "gradient_clip_norm": float(self.gradient_clip_norm),
            "minimum_role_rows": int(self.minimum_role_rows),
            "device_run_group_size": int(self.device_run_group_size),
            "execution_mode": self.execution_mode,
            "architecture_selection_mode": self.architecture_selection_mode,
            "feature_view_selection_mode": (
                "mandatory_state_first_clock_control_then_order_flow_ablation_gates"
            ),
            "state_conditioned_flow_selection_mode": (
                "mandatory_post_feature_neutral_interaction_ablation_gate"
            ),
            "initialization_selection_mode": (
                "mandatory_random_incumbent_causal_pretraining_ablation_gate"
            ),
            "causal_pretraining": self.pretraining.as_dict(),
            "training_order": "chronological_no_shuffle",
            "tuning_order": "chronological_no_shuffle",
            "checkpoint_policy": "best_state_in_memory_only",
            "loss_weights": dict(ROUND74_EVENT_TRAINING_LOSS_WEIGHTS),
        }


def _batch_run_id(batch: Round74EventTrainingBatch) -> str:
    run_ids = set(batch.run_id)
    if len(run_ids) != 1:
        raise ValueError("Round 74 selection batch spans capture runs")
    return next(iter(run_ids))


@dataclass(frozen=True)
class Round74EventSelectionProtocol:
    """Training-only early stopping plus disjoint chronological promotions."""

    optimization_batches: tuple[Round74EventTrainingBatch, ...]
    purged_training_batches: tuple[Round74EventTrainingBatch, ...]
    early_stopping_batches: tuple[Round74EventTrainingBatch, ...]
    promotion_stage_batches: tuple[
        tuple[Round74EventTrainingBatch, ...],
        ...,
    ]
    stage_partition: Round74SegmentedModelSelectionStages
    training_split: Round74SegmentedTrainingSplit
    scaler_fit_selection_sha256: str
    schema_version: str = ROUND74_EVENT_SELECTION_PROTOCOL_SCHEMA_VERSION

    def validate(self) -> None:
        self.stage_partition.validate()
        self.training_split.validate()
        training_groups = (
            self.optimization_batches,
            self.purged_training_batches,
            self.early_stopping_batches,
        )
        all_training = tuple(batch for group in training_groups for batch in group)
        all_promotions = tuple(
            batch for group in self.promotion_stage_batches for batch in group
        )
        all_batches = (*all_training, *all_promotions)
        if (
            self.schema_version != ROUND74_EVENT_SELECTION_PROTOCOL_SCHEMA_VERSION
            or len(self.optimization_batches)
            < ROUND74_EVENT_SEGMENTED_MINIMUM_OPTIMIZATION_RUNS
            or len(self.early_stopping_batches)
            < ROUND74_EVENT_SEGMENTED_MINIMUM_EARLY_STOPPING_RUNS
            or len(self.promotion_stage_batches)
            != len(ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS)
            or any(not group for group in self.promotion_stage_batches)
            or _SHA256.fullmatch(self.scaler_fit_selection_sha256) is None
            or self.scaler_fit_selection_sha256 != self.training_split.split_sha256
        ):
            raise ValueError("Round 74 model-selection protocol differs")
        for batch in all_batches:
            batch.validate()
        training_run_ids = tuple(_batch_run_id(batch) for batch in all_training)
        promotion_run_ids = tuple(_batch_run_id(batch) for batch in all_promotions)
        if (
            any(batch.role != "training" for batch in all_training)
            or any(batch.role != "tuning" for batch in all_promotions)
            or len(set(training_run_ids)) != len(training_run_ids)
            or len(set(promotion_run_ids)) != len(promotion_run_ids)
            or set(training_run_ids) & set(promotion_run_ids)
            or tuple(
                tuple(_batch_run_id(batch) for batch in group)
                for group in self.promotion_stage_batches
            )
            != self.stage_partition.stage_run_ids
            or len({batch.partition_sha256 for batch in all_batches}) != 1
            or len({batch.scaler_sha256 for batch in all_batches}) != 1
            or len({batch.window_representation for batch in all_batches}) != 1
            or self.stage_partition.parent_partition_sha256
            != all_batches[0].partition_sha256
            or self.training_split.parent_partition_sha256
            != all_batches[0].partition_sha256
            or self.training_split.cohort_plan_sha256
            != self.stage_partition.cohort_plan_sha256
            or self.training_split.optimization_run_ids
            != tuple(_batch_run_id(batch) for batch in self.optimization_batches)
            or self.training_split.purged_run_ids
            != tuple(_batch_run_id(batch) for batch in self.purged_training_batches)
            or self.training_split.early_stopping_run_ids
            != tuple(_batch_run_id(batch) for batch in self.early_stopping_batches)
            or any(
                int(current.decision_wall_ns[0]) <= int(prior.decision_wall_ns[-1])
                for prior, current in zip(
                    all_training,
                    all_training[1:],
                    strict=False,
                )
            )
        ):
            raise ValueError("Round 74 model-selection role identity differs")
        chronological_gap_ns = int(
            self.early_stopping_batches[0].decision_wall_ns[0]
        ) - int(self.optimization_batches[-1].decision_wall_ns[-1])
        if chronological_gap_ns < ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS:
            raise ValueError("Round 74 training early-stop purge is too short")

    @property
    def protocol_sha256(self) -> str:
        self.validate()
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        chronological_gap_ns = int(
            self.early_stopping_batches[0].decision_wall_ns[0]
        ) - int(self.optimization_batches[-1].decision_wall_ns[-1])
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "stage_partition_sha256": (self.stage_partition.stage_partition_sha256),
            "training_split_sha256": self.training_split.split_sha256,
            "training_split": self.training_split.as_dict(),
            "stage_order": list(ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS),
            "training_role_assignment_basis": (
                "chronological admitted-run order after transport adjudication"
            ),
            "early_stopping_fraction_denominator": (
                ROUND74_EVENT_SEGMENTED_EARLY_STOPPING_FRACTION_DENOMINATOR
            ),
            "minimum_optimization_run_count": (
                ROUND74_EVENT_SEGMENTED_MINIMUM_OPTIMIZATION_RUNS
            ),
            "minimum_early_stopping_run_count": (
                ROUND74_EVENT_SEGMENTED_MINIMUM_EARLY_STOPPING_RUNS
            ),
            "optimization_run_ids": [
                _batch_run_id(batch) for batch in self.optimization_batches
            ],
            "purged_training_run_ids": [
                _batch_run_id(batch) for batch in self.purged_training_batches
            ],
            "early_stopping_run_ids": [
                _batch_run_id(batch) for batch in self.early_stopping_batches
            ],
            "optimization_batch_sha256": [
                batch.batch_sha256 for batch in self.optimization_batches
            ],
            "purged_training_batch_sha256": [
                batch.batch_sha256 for batch in self.purged_training_batches
            ],
            "early_stopping_batch_sha256": [
                batch.batch_sha256 for batch in self.early_stopping_batches
            ],
            "promotion_stage_batch_sha256": {
                stage_id: [batch.batch_sha256 for batch in batches]
                for stage_id, batches in zip(
                    ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS,
                    self.promotion_stage_batches,
                    strict=True,
                )
            },
            "chronological_gap_ns": chronological_gap_ns,
            "minimum_chronological_gap_ns": (ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS),
            "target_loss_scale_fit_scope": "optimization_training_runs_only",
            "feature_scaler_fit_scope": ("segmented_optimization_training_runs_only"),
            "feature_scaler_fit_selection_sha256": (self.scaler_fit_selection_sha256),
            "feature_scaler_fit_source_run_ids_sha256": _canonical_sha256(
                [_batch_run_id(batch) for batch in self.optimization_batches]
            ),
            "early_stopping_targets_used_for_gradient_updates": False,
            "promotion_targets_used_for_checkpoint_selection": False,
            "cross_stage_promotion_run_reuse_permitted": False,
            "calibration_or_policy_selection_run_included": False,
            "sealed_test_run_included": False,
        }
        if include_sha256:
            payload["protocol_sha256"] = _canonical_sha256(payload)
        return payload


def _build_segmented_selection_protocol(
    training_batches: Sequence[Round74EventTrainingBatch],
    tuning_roles: object,
    feature_scaler: Round74EventFeatureScaler,
    training_split: Round74SegmentedTrainingSplit,
) -> Round74EventSelectionProtocol:
    from .round74_event_model_operator import Round74PreparedTuningRoles

    if not isinstance(tuning_roles, Round74PreparedTuningRoles):
        raise TypeError("Round 74 prepared tuning roles are required")
    tuning_roles.validate()
    if not isinstance(
        tuning_roles.subpartition,
        Round74SegmentedTuningSubpartition,
    ):
        raise TypeError("Round 74 segmented tuning subpartition is required")
    selected_training = tuple(training_batches)
    if not isinstance(training_split, Round74SegmentedTrainingSplit):
        raise TypeError("Round 74 segmented training split is required")
    training_split.validate()
    for batch in selected_training:
        batch.validate()
    admitted_count = len(selected_training)
    early_stopping_count = max(
        ROUND74_EVENT_SEGMENTED_MINIMUM_EARLY_STOPPING_RUNS,
        (
            admitted_count
            + ROUND74_EVENT_SEGMENTED_EARLY_STOPPING_FRACTION_DENOMINATOR
            - 1
        )
        // ROUND74_EVENT_SEGMENTED_EARLY_STOPPING_FRACTION_DENOMINATOR,
    )
    if (
        admitted_count
        < ROUND74_EVENT_SEGMENTED_MINIMUM_OPTIMIZATION_RUNS + early_stopping_count
    ):
        raise ValueError(
            "Round 74 segmented training role is too small for isolated early stopping"
        )
    early_stopping_batches = selected_training[-early_stopping_count:]
    early_stopping_start = admitted_count - early_stopping_count
    if (
        not isinstance(feature_scaler, Round74EventFeatureScaler)
        or feature_scaler.fit_source_scope != "segmented_optimization_training_runs"
        or feature_scaler.fit_source_partition_sha256
        != selected_training[0].partition_sha256
        or _SHA256.fullmatch(feature_scaler.fit_source_selection_sha256) is None
    ):
        raise ValueError("Round 74 segmented scaler provenance differs")
    optimization_end = len(feature_scaler.fit_source_run_ids)
    observed_optimization_run_ids = tuple(
        _batch_run_id(batch) for batch in selected_training[:optimization_end]
    )
    if (
        optimization_end < ROUND74_EVENT_SEGMENTED_MINIMUM_OPTIMIZATION_RUNS
        or optimization_end > early_stopping_start
        or feature_scaler.fit_source_run_ids != observed_optimization_run_ids
        or feature_scaler.fit_source_selection_sha256 != training_split.split_sha256
    ):
        raise ValueError("Round 74 segmented scaler optimization runs differ")
    optimization_batches = selected_training[:optimization_end]
    purged_training_batches = selected_training[optimization_end:early_stopping_start]
    stage_partition = build_round74_segmented_model_selection_stages(
        tuning_roles.subpartition
    )
    model_selection_by_run_id = {
        _batch_run_id(batch): batch for batch in tuning_roles.model_selection_batches
    }
    promotion_stage_batches = tuple(
        stage_partition.batches_for_stage(
            stage_id,
            model_selection_by_run_id,
        )
        for stage_id in ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS
    )
    selected = Round74EventSelectionProtocol(
        optimization_batches=optimization_batches,
        purged_training_batches=purged_training_batches,
        early_stopping_batches=early_stopping_batches,
        promotion_stage_batches=promotion_stage_batches,
        stage_partition=stage_partition,
        training_split=training_split,
        scaler_fit_selection_sha256=feature_scaler.fit_source_selection_sha256,
    )
    selected.validate()
    if (
        *selected.optimization_batches,
        *selected.purged_training_batches,
        *selected.early_stopping_batches,
    ) != selected_training or tuple(
        batch for group in selected.promotion_stage_batches for batch in group
    ) != tuning_roles.model_selection_batches:
        raise RuntimeError("Round 74 model-selection coverage differs")
    return selected


def _readonly_array(value: np.ndarray) -> np.ndarray:
    selected = np.ascontiguousarray(value)
    selected.setflags(write=False)
    return selected


def _robust_positive_scale(values: np.ndarray) -> float:
    selected = np.asarray(values, dtype=np.float64)
    if selected.ndim != 1 or selected.size < 1 or not np.isfinite(selected).all():
        raise ValueError("Round 74 target-loss scale observations differ")
    q25, q50, q75 = np.quantile(selected, (0.25, 0.50, 0.75))
    scale = max(float(q75 - q25), float(np.median(np.abs(selected))), abs(float(q50)))
    return scale if math.isfinite(scale) and scale > 0.0 else 0.0


@dataclass(frozen=True)
class Round74EventTargetLossScale:
    """Training-only empirical scales for comparable proper-loss gradients."""

    payoff_scale_bps: np.ndarray
    maximum_adverse_excursion_scale_bps: np.ndarray
    eligible_target_count: np.ndarray
    training_batch_sha256: tuple[str, ...]
    payoff_fallback_groups: tuple[str, ...] = ()
    maximum_adverse_excursion_fallback_groups: tuple[str, ...] = ()
    schema_version: str = ROUND74_EVENT_TARGET_LOSS_SCALE_SCHEMA_VERSION

    def validate(self) -> None:
        shape = (
            len(IMPACT_CAPTURE_SYMBOLS),
            len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
            len(ROUND74_EVENT_PAYOFF_SIDES),
        )
        if (
            self.schema_version != ROUND74_EVENT_TARGET_LOSS_SCALE_SCHEMA_VERSION
            or self.payoff_scale_bps.shape != shape
            or self.maximum_adverse_excursion_scale_bps.shape != shape
            or self.eligible_target_count.shape != shape
            or self.payoff_scale_bps.dtype != np.float64
            or self.maximum_adverse_excursion_scale_bps.dtype != np.float64
            or self.eligible_target_count.dtype != np.int64
            or self.payoff_scale_bps.flags.writeable
            or self.maximum_adverse_excursion_scale_bps.flags.writeable
            or self.eligible_target_count.flags.writeable
            or not np.isfinite(self.payoff_scale_bps).all()
            or not np.isfinite(self.maximum_adverse_excursion_scale_bps).all()
            or np.any(self.payoff_scale_bps <= 0.0)
            or np.any(self.maximum_adverse_excursion_scale_bps <= 0.0)
            or np.any(self.eligible_target_count < 0)
            or not self.training_batch_sha256
            or tuple(sorted(self.training_batch_sha256)) != self.training_batch_sha256
            or len(set(self.training_batch_sha256)) != len(self.training_batch_sha256)
            or any(
                _SHA256.fullmatch(value) is None for value in self.training_batch_sha256
            )
        ):
            raise ValueError("Round 74 target-loss scale contract differs")
        valid_groups = {
            f"{symbol}:{horizon}:{side}"
            for symbol in IMPACT_CAPTURE_SYMBOLS
            for horizon in ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS
            for side in ROUND74_EVENT_PAYOFF_SIDES
        }
        for values in (
            self.payoff_fallback_groups,
            self.maximum_adverse_excursion_fallback_groups,
        ):
            if (
                tuple(sorted(values)) != values
                or len(values) != len(set(values))
                or any(value not in valid_groups for value in values)
            ):
                raise ValueError("Round 74 target-loss fallback group differs")

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        self.validate()
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "symbols": list(IMPACT_CAPTURE_SYMBOLS),
            "horizons_seconds": list(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
            "sides": list(ROUND74_EVENT_PAYOFF_SIDES),
            "payoff_scale_bps": self.payoff_scale_bps.tolist(),
            "maximum_adverse_excursion_scale_bps": (
                self.maximum_adverse_excursion_scale_bps.tolist()
            ),
            "eligible_target_count": self.eligible_target_count.tolist(),
            "training_batch_sha256": list(self.training_batch_sha256),
            "payoff_fallback_groups": list(self.payoff_fallback_groups),
            "maximum_adverse_excursion_fallback_groups": list(
                self.maximum_adverse_excursion_fallback_groups
            ),
            "fit_partition_role": "training",
            "tuning_targets_used": False,
            "test_targets_used": False,
            "forecast_units_changed": False,
            "financial_outcome_units": "basis_points",
        }
        if include_sha256:
            payload["target_loss_scale_sha256"] = _canonical_sha256(payload)
        return payload

    @property
    def target_loss_scale_sha256(self) -> str:
        return _canonical_sha256(self.as_dict(include_sha256=False))

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74EventTargetLossScale:
        payload = dict(value)
        claimed = str(payload.pop("target_loss_scale_sha256", ""))
        if claimed != _canonical_sha256(payload):
            raise ValueError("Round 74 target-loss scale digest differs")
        if (
            payload.get("symbols") != list(IMPACT_CAPTURE_SYMBOLS)
            or payload.get("horizons_seconds")
            != list(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
            or payload.get("sides") != list(ROUND74_EVENT_PAYOFF_SIDES)
            or payload.get("fit_partition_role") != "training"
            or payload.get("tuning_targets_used") is not False
            or payload.get("test_targets_used") is not False
            or payload.get("forecast_units_changed") is not False
            or payload.get("financial_outcome_units") != "basis_points"
        ):
            raise ValueError("Round 74 target-loss scale policy differs")
        try:
            selected = cls(
                payoff_scale_bps=_readonly_array(
                    np.asarray(payload["payoff_scale_bps"], dtype=np.float64)
                ),
                maximum_adverse_excursion_scale_bps=_readonly_array(
                    np.asarray(
                        payload["maximum_adverse_excursion_scale_bps"],
                        dtype=np.float64,
                    )
                ),
                eligible_target_count=_readonly_array(
                    np.asarray(payload["eligible_target_count"], dtype=np.int64)
                ),
                training_batch_sha256=tuple(
                    str(item) for item in payload["training_batch_sha256"]
                ),
                payoff_fallback_groups=tuple(
                    str(item) for item in payload["payoff_fallback_groups"]
                ),
                maximum_adverse_excursion_fallback_groups=tuple(
                    str(item)
                    for item in payload["maximum_adverse_excursion_fallback_groups"]
                ),
                schema_version=str(payload["schema_version"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Round 74 target-loss scale payload differs") from exc
        selected.validate()
        if selected.as_dict(include_sha256=False) != payload:
            raise ValueError("Round 74 target-loss scale reload differs")
        return selected

    def for_batch(
        self,
        batch: Round74EventTrainingBatch,
        row_slice: slice,
    ) -> tuple[np.ndarray, np.ndarray]:
        start, stop, step = row_slice.indices(batch.rows)
        if step != 1 or stop <= start:
            raise ValueError("Round 74 target-loss scale slice differs")
        symbol_indexes = np.fromiter(
            (
                IMPACT_CAPTURE_SYMBOLS.index(symbol)
                for symbol in batch.symbol[start:stop]
            ),
            dtype=np.int64,
            count=stop - start,
        )
        return (
            np.ascontiguousarray(
                self.payoff_scale_bps[symbol_indexes],
                dtype=np.float32,
            ),
            np.ascontiguousarray(
                self.maximum_adverse_excursion_scale_bps[symbol_indexes],
                dtype=np.float32,
            ),
        )


def fit_round74_event_target_loss_scale(
    training_batches: Sequence[Round74EventTrainingBatch],
    *,
    require_complete_panel: bool,
) -> Round74EventTargetLossScale:
    """Fit subgroup scales using eligible training targets and no later role."""

    batches = tuple(training_batches)
    if not batches or not isinstance(require_complete_panel, bool):
        raise ValueError("Round 74 target-loss scale fit inputs differ")
    for batch in batches:
        batch.validate()
        if batch.role != "training":
            raise ValueError("Round 74 target-loss scale used a non-training role")
    hashes = tuple(sorted(batch.batch_sha256 for batch in batches))
    if len(hashes) != len(set(hashes)):
        raise ValueError("Round 74 target-loss scale training batches repeat")
    shape = (
        len(IMPACT_CAPTURE_SYMBOLS),
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        len(ROUND74_EVENT_PAYOFF_SIDES),
    )
    payoff_groups: dict[tuple[int, int, int], list[np.ndarray]] = {}
    adverse_groups: dict[tuple[int, int, int], list[np.ndarray]] = {}
    counts = np.zeros(shape, dtype=np.int64)
    pooled_payoff: list[np.ndarray] = []
    pooled_adverse: list[np.ndarray] = []
    for batch in batches:
        symbols = np.asarray(
            tuple(IMPACT_CAPTURE_SYMBOLS.index(value) for value in batch.symbol),
            dtype=np.int64,
        )
        for symbol_index in range(len(IMPACT_CAPTURE_SYMBOLS)):
            row_mask = symbols == symbol_index
            if not np.any(row_mask):
                continue
            for horizon_index in range(len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)):
                for side_index in range(len(ROUND74_EVENT_PAYOFF_SIDES)):
                    eligible = row_mask & (
                        batch.action_eligibility[:, horizon_index, side_index] == 1.0
                    )
                    if not np.any(eligible):
                        continue
                    key = (symbol_index, horizon_index, side_index)
                    payoff = np.asarray(
                        batch.net_payoff_bps[
                            eligible,
                            horizon_index,
                            side_index,
                        ],
                        dtype=np.float64,
                    )
                    adverse = np.asarray(
                        batch.maximum_adverse_excursion_bps[
                            eligible,
                            horizon_index,
                            side_index,
                        ],
                        dtype=np.float64,
                    )
                    payoff_groups.setdefault(key, []).append(payoff)
                    adverse_groups.setdefault(key, []).append(adverse)
                    counts[key] += int(payoff.size)
                    pooled_payoff.append(payoff)
                    pooled_adverse.append(adverse)
    if not pooled_payoff or not pooled_adverse:
        raise ValueError("Round 74 target-loss scale has no eligible targets")
    pooled_payoff_scale = _robust_positive_scale(np.concatenate(pooled_payoff))
    pooled_adverse_scale = _robust_positive_scale(np.concatenate(pooled_adverse))
    if require_complete_panel and (
        pooled_payoff_scale <= 0.0 or pooled_adverse_scale <= 0.0
    ):
        raise ValueError("Round 74 target-loss scale population is degenerate")
    pooled_payoff_scale = pooled_payoff_scale or 1.0
    pooled_adverse_scale = pooled_adverse_scale or 1.0
    payoff_scales = np.empty(shape, dtype=np.float64)
    adverse_scales = np.empty(shape, dtype=np.float64)
    payoff_fallbacks: list[str] = []
    adverse_fallbacks: list[str] = []
    for symbol_index, symbol in enumerate(IMPACT_CAPTURE_SYMBOLS):
        for horizon_index, horizon in enumerate(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS):
            for side_index, side in enumerate(ROUND74_EVENT_PAYOFF_SIDES):
                key = (symbol_index, horizon_index, side_index)
                label = f"{symbol}:{horizon}:{side}"
                if key not in payoff_groups:
                    if require_complete_panel:
                        raise ValueError(
                            "Round 74 target-loss scale subgroup panel is incomplete"
                        )
                    payoff_scale = pooled_payoff_scale
                    adverse_scale = pooled_adverse_scale
                    payoff_fallbacks.append(label)
                    adverse_fallbacks.append(label)
                else:
                    payoff_scale = _robust_positive_scale(
                        np.concatenate(payoff_groups[key])
                    )
                    adverse_scale = _robust_positive_scale(
                        np.concatenate(adverse_groups[key])
                    )
                    if payoff_scale <= 0.0:
                        payoff_scale = pooled_payoff_scale
                        payoff_fallbacks.append(label)
                    if adverse_scale <= 0.0:
                        adverse_scale = pooled_adverse_scale
                        adverse_fallbacks.append(label)
                payoff_scales[key] = payoff_scale
                adverse_scales[key] = adverse_scale
    selected = Round74EventTargetLossScale(
        payoff_scale_bps=_readonly_array(payoff_scales),
        maximum_adverse_excursion_scale_bps=_readonly_array(adverse_scales),
        eligible_target_count=_readonly_array(counts),
        training_batch_sha256=hashes,
        payoff_fallback_groups=tuple(sorted(payoff_fallbacks)),
        maximum_adverse_excursion_fallback_groups=tuple(sorted(adverse_fallbacks)),
    )
    selected.validate()
    reloaded = Round74EventTargetLossScale.from_dict(selected.as_dict())
    if reloaded.as_dict() != selected.as_dict():
        raise RuntimeError("Round 74 target-loss scale reload differs")
    return selected


def _round74_feature_view_pre_hook(
    module: nn.Module,
    args: tuple[object, ...],
) -> tuple[torch.Tensor]:
    if len(args) != 1 or not isinstance(args[0], torch.Tensor):
        raise ValueError("Round 74 feature-view model input differs")
    values = args[0]
    if values.ndim != 3 or int(values.shape[2]) != len(ROUND74_EVENT_FEATURE_NAMES):
        raise ValueError("Round 74 feature-view tensor dimensions differ")
    feature_view = getattr(module, "feature_view", None)
    if feature_view not in ROUND74_EVENT_FEATURE_VIEWS:
        raise ValueError("Round 74 model feature view differs")
    masked_indices = ROUND74_EVENT_FEATURE_VIEW_MASKED_INDICES[feature_view]
    if not masked_indices:
        return (values,)
    mask = getattr(module, "_round74_feature_view_mask", None)
    if not isinstance(mask, torch.Tensor) or mask.shape != (
        1,
        1,
        len(ROUND74_EVENT_FEATURE_NAMES),
    ):
        raise RuntimeError("Round 74 model feature-view mask differs")
    return (values * mask,)


def _bind_round74_feature_view(model: nn.Module, feature_view: str) -> nn.Module:
    selected = str(feature_view)
    if selected not in ROUND74_EVENT_FEATURE_VIEWS:
        raise ValueError("Round 74 model feature view differs")
    if hasattr(model, "feature_view") or hasattr(
        model,
        "_round74_feature_view_mask",
    ):
        raise ValueError("Round 74 model feature view was already bound")
    mask = torch.ones(
        (1, 1, len(ROUND74_EVENT_FEATURE_NAMES)),
        dtype=torch.float32,
    )
    mask[:, :, list(ROUND74_EVENT_FEATURE_VIEW_MASKED_INDICES[selected])] = 0.0
    model.register_buffer(
        "_round74_feature_view_mask",
        mask,
        persistent=False,
    )
    model.feature_view = selected  # type: ignore[attr-defined]
    model.register_forward_pre_hook(_round74_feature_view_pre_hook)
    return model


class Round74EventEnsemble(nn.Module):
    """Equal-weight seed ensemble for one fixed architecture."""

    def __init__(
        self,
        candidate_id: str,
        peer_count: int,
        *,
        feature_view: str = "full",
        state_conditioned_flow: bool = False,
    ) -> None:
        super().__init__()
        if (
            candidate_id not in ROUND74_EVENT_MODEL_CANDIDATES
            or peer_count < 1
            or not isinstance(state_conditioned_flow, bool)
        ):
            raise ValueError("Round 74 event ensemble identity differs")
        self.candidate_id = candidate_id
        self.state_conditioned_flow = state_conditioned_flow
        self.peers = nn.ModuleList(
            build_round74_event_model(
                candidate_id,
                state_conditioned_flow=state_conditioned_flow,
            )
            for _ in range(int(peer_count))
        )
        _bind_round74_feature_view(self, feature_view)

    def forward(self, values: torch.Tensor) -> Round74EventModelOutput:
        outputs = tuple(peer(values) for peer in self.peers)

        def tensor_mean(name: str) -> torch.Tensor:
            return torch.stack(
                tuple(getattr(output, name) for output in outputs),
                dim=0,
            ).mean(dim=0)

        def predictive_mixture_logit(name: str) -> torch.Tensor:
            peer_logits = torch.stack(
                tuple(getattr(output, name) for output in outputs),
                dim=0,
            )
            probability = torch.sigmoid(peer_logits).mean(dim=0)
            epsilon = torch.finfo(probability.dtype).eps
            bounded = torch.clamp(
                probability,
                min=epsilon,
                max=1.0 - epsilon,
            )
            return torch.log(bounded / (1.0 - bounded))

        output = Round74EventModelOutput(
            payoff_quantiles_bps=tensor_mean("payoff_quantiles_bps"),
            maximum_adverse_excursion_quantiles_bps=tensor_mean(
                "maximum_adverse_excursion_quantiles_bps"
            ),
            positive_payoff_logits=predictive_mixture_logit("positive_payoff_logits"),
            adverse_selection_logits=predictive_mixture_logit(
                "adverse_selection_logits"
            ),
            regime_unpredictability_logits=predictive_mixture_logit(
                "regime_unpredictability_logits"
            ),
        )
        output.validate(int(values.shape[0]))
        return output


@dataclass(frozen=True)
class Round74PretestPolicyArtifact:
    """Verified paths and identity for one immutable development-only policy."""

    policy_sha256: str
    policy_path: Path
    model_sha256: str
    model_path: Path
    selected_candidate_id: str
    selected_feature_view: str
    tuning_loss: float


@dataclass(frozen=True)
class _DevelopmentIdentity:
    partition_sha256: str
    scaler_sha256: str
    window_representation: str
    target_context_sha256: tuple[str, ...]
    target_context_panel_sha256: str
    training_batch_sha256: tuple[str, ...]
    tuning_batch_sha256: tuple[str, ...]
    training_rows: int
    tuning_rows: int
    training_first_wall_ns: int
    training_last_wall_ns: int
    tuning_first_wall_ns: int
    tuning_last_wall_ns: int


@dataclass(frozen=True)
class _CandidateFit:
    candidate_id: str
    feature_view: str
    state_conditioned_flow: bool
    peer_states: tuple[dict[str, torch.Tensor], ...]
    peer_reports: tuple[dict[str, object], ...]
    ensemble_metrics: dict[str, float]
    ensemble_run_losses: tuple[float, ...]
    ensemble_group_losses: dict[str, float]
    ensemble_prediction_sha256: str
    parameter_count_per_peer: int


def _row_key(
    batch: Round74EventTrainingBatch,
    index: int,
) -> tuple[object, ...]:
    return (
        int(batch.decision_wall_ns[index]),
        batch.run_id[index],
        int(batch.decision_monotonic_ns[index]),
        int(batch.endpoint_frame_index[index]),
        int(batch.endpoint_message_index[index]),
        batch.symbol[index],
        int(batch.anchor_index[index]),
    )


def _validate_role_batches(
    batches: Sequence[Round74EventTrainingBatch],
    *,
    required_role: str,
    minimum_rows: int,
) -> tuple[object, ...]:
    if required_role not in {"training", "tuning"} or not batches:
        raise ValueError(f"Round 74 {required_role} batches are missing")
    prior_key: tuple[object, ...] | None = None
    rows = 0
    samples: set[str] = set()
    capture_runs: set[str] = set()
    for batch in batches:
        batch.validate()
        if batch.role != required_role:
            raise ValueError(
                f"Round 74 trainer rejects {batch.role!r} data in "
                f"the {required_role} role"
            )
        batch_runs = set(batch.run_id)
        if len(batch_runs) != 1:
            raise ValueError(f"Round 74 {required_role} batch mixes capture runs")
        if len(set(batch.target_context_sha256)) != 1:
            raise ValueError(f"Round 74 {required_role} batch mixes target contexts")
        run_id = next(iter(batch_runs))
        if run_id in capture_runs:
            raise ValueError(f"Round 74 {required_role} capture run is repeated")
        capture_runs.add(run_id)
        first = _row_key(batch, 0)
        last = _row_key(batch, batch.rows - 1)
        if prior_key is not None and first <= prior_key:
            raise ValueError(f"Round 74 {required_role} batch order regressed")
        prior_key = last
        rows += batch.rows
        for sample in batch.sample_sha256:
            if sample in samples:
                raise ValueError(f"Round 74 {required_role} sample is duplicated")
            samples.add(sample)
    if rows < minimum_rows:
        raise ValueError(f"Round 74 {required_role} rows are below the minimum")
    return (
        rows,
        _row_key(batches[0], 0),
        _row_key(batches[-1], batches[-1].rows - 1),
        frozenset(samples),
    )


def _validate_development_batches(
    training_batches: Sequence[Round74EventTrainingBatch],
    tuning_batches: Sequence[Round74EventTrainingBatch],
    *,
    minimum_rows: int,
) -> _DevelopmentIdentity:
    training = tuple(training_batches)
    tuning = tuple(tuning_batches)
    training_rows, training_first, training_last, training_samples = (
        _validate_role_batches(
            training,
            required_role="training",
            minimum_rows=minimum_rows,
        )
    )
    tuning_rows, tuning_first, tuning_last, tuning_samples = _validate_role_batches(
        tuning,
        required_role="tuning",
        minimum_rows=minimum_rows,
    )
    if training_samples & tuning_samples:
        raise ValueError("Round 74 training and tuning samples overlap")
    if int(tuning_first[0]) - int(training_last[0]) < (
        ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS
    ):
        raise ValueError("Round 74 development roles lack the minimum purge")
    all_batches = (*training, *tuning)
    partitions = {batch.partition_sha256 for batch in all_batches}
    scalers = {batch.scaler_sha256 for batch in all_batches}
    representations = {batch.window_representation for batch in all_batches}
    contexts = tuple(
        sorted(
            {
                context
                for batch in all_batches
                for context in batch.target_context_sha256
            }
        )
    )
    if len(partitions) != 1:
        raise ValueError("Round 74 development partition identity differs")
    if len(scalers) != 1:
        raise ValueError("Round 74 development scaler identity differs")
    if (
        len(representations) != 1
        or next(iter(representations)) not in ROUND74_EVENT_WINDOW_REPRESENTATIONS
    ):
        raise ValueError("Round 74 development window representation differs")
    target_context_panel_sha256 = _canonical_sha256(
        {
            "schema_version": (ROUND74_EVENT_TARGET_CONTEXT_PANEL_SCHEMA_VERSION),
            "target_context_sha256": list(contexts),
        }
    )
    return _DevelopmentIdentity(
        partition_sha256=next(iter(partitions)),
        scaler_sha256=next(iter(scalers)),
        window_representation=next(iter(representations)),
        target_context_sha256=contexts,
        target_context_panel_sha256=target_context_panel_sha256,
        training_batch_sha256=tuple(batch.batch_sha256 for batch in training),
        tuning_batch_sha256=tuple(batch.batch_sha256 for batch in tuning),
        training_rows=int(training_rows),
        tuning_rows=int(tuning_rows),
        training_first_wall_ns=int(training_first[0]),
        training_last_wall_ns=int(training_last[0]),
        tuning_first_wall_ns=int(tuning_first[0]),
        tuning_last_wall_ns=int(tuning_last[0]),
    )


def _iter_minibatches(
    batches: Sequence[Round74EventTrainingBatch],
    maximum_rows: int,
) -> Iterable[tuple[Round74EventTrainingBatch, slice]]:
    for batch in batches:
        for start in range(0, batch.rows, maximum_rows):
            yield batch, slice(start, min(start + maximum_rows, batch.rows))


def _run_balanced_minibatch_schedules(
    batches: Sequence[Round74EventTrainingBatch],
    maximum_rows: int,
    *,
    totals: dict[str, float],
    per_run_totals: Sequence[dict[str, float]],
) -> tuple[tuple[slice, ...], ...]:
    """Build one deterministic eligible-minibatch cycle per capture run."""

    if len(batches) != len(per_run_totals) or not batches:
        raise ValueError("Round 74 run-balanced schedule differs")
    schedules: list[tuple[slice, ...]] = []
    for batch, run_totals in zip(batches, per_run_totals, strict=True):
        eligible: list[slice] = []
        for _selected, row_slice in _iter_minibatches((batch,), maximum_rows):
            if _skip_fully_censored_minibatch(totals, batch, row_slice):
                _skip_fully_censored_minibatch(run_totals, batch, row_slice)
                continue
            eligible.append(row_slice)
        if not eligible:
            raise ValueError("Round 74 training capture run is fully censored")
        schedules.append(tuple(eligible))
    return tuple(schedules)


def _eligible_target_minibatch_schedule(
    batches: Sequence[Round74EventTrainingBatch],
    maximum_rows: int,
    *,
    totals: dict[str, float],
    per_run_totals: Sequence[dict[str, float]],
) -> tuple[
    tuple[int, Round74EventTrainingBatch, slice, int, int],
    ...,
]:
    """Visit each eligible segmented minibatch exactly once per epoch."""

    if len(batches) != len(per_run_totals) or not batches:
        raise ValueError("Round 74 eligible-target schedule differs")
    selected: list[tuple[int, Round74EventTrainingBatch, slice, int, int]] = []
    for run_index, (batch, run_totals) in enumerate(
        zip(batches, per_run_totals, strict=True)
    ):
        run_count = 0
        for _batch, row_slice in _iter_minibatches((batch,), maximum_rows):
            if _skip_fully_censored_minibatch(totals, batch, row_slice):
                _skip_fully_censored_minibatch(run_totals, batch, row_slice)
                continue
            action_weight, regime_weight, _rows = _minibatch_target_counts(
                batch,
                row_slice,
            )
            selected.append(
                (
                    run_index,
                    batch,
                    row_slice,
                    action_weight,
                    regime_weight,
                )
            )
            run_count += 1
        if run_count == 0:
            raise ValueError("Round 74 training capture run is fully censored")
    return tuple(selected)


def _eligible_target_weighted_group_loss(
    grouped: Sequence[tuple[torch.Tensor, Mapping[str, torch.Tensor], int, int]],
    *,
    total_action_weight: int,
    total_regime_weight: int,
) -> torch.Tensor:
    """Return this group's exact contribution to the cohort objective."""

    if not grouped or total_action_weight <= 0 or total_regime_weight <= 0:
        raise ValueError("Round 74 eligible-target gradient population differs")
    contributions: list[torch.Tensor] = []
    for _loss, components, action_weight, regime_weight in grouped:
        action_objective = (
            components["payoff_pinball"]
            + ROUND74_EVENT_TRAINING_LOSS_WEIGHTS["maximum_adverse_excursion"]
            * components["maximum_adverse_excursion_pinball"]
            + ROUND74_EVENT_TRAINING_LOSS_WEIGHTS["positive_payoff"]
            * components["positive_bce"]
            + ROUND74_EVENT_TRAINING_LOSS_WEIGHTS["adverse_selection"]
            * components["adverse_bce"]
        )
        regime_objective = (
            ROUND74_EVENT_TRAINING_LOSS_WEIGHTS["regime_unpredictability"]
            * components["unpredictability_bce"]
        )
        contributions.append(
            action_objective * (float(action_weight) / float(total_action_weight))
            + regime_objective * (float(regime_weight) / float(total_regime_weight))
        )
    return torch.stack(tuple(contributions)).sum()


def _to_device_tensor(
    value: np.ndarray,
    row_slice: slice,
    device: object,
) -> torch.Tensor:
    copied = np.array(value[row_slice], dtype=np.float32, order="C", copy=True)
    return torch.from_numpy(copied).to(device)


def _minibatch_target_counts(
    batch: Round74EventTrainingBatch,
    row_slice: slice,
) -> tuple[int, int, int]:
    start, stop, step = row_slice.indices(batch.rows)
    if step != 1 or stop <= start:
        raise ValueError("Round 74 minibatch slice differs")
    action_weight = int(batch.action_eligibility[row_slice].sum())
    regime_weight = int(batch.regime_unpredictability_eligibility[row_slice].sum())
    if (action_weight == 0) != (regime_weight == 0):
        raise ValueError("Round 74 minibatch target eligibility differs")
    return action_weight, regime_weight, stop - start


def _skip_fully_censored_minibatch(
    totals: dict[str, float],
    batch: Round74EventTrainingBatch,
    row_slice: slice,
) -> bool:
    action_weight, regime_weight, rows = _minibatch_target_counts(
        batch,
        row_slice,
    )
    if action_weight > 0 and regime_weight > 0:
        return False
    totals["fully_censored_minibatches"] += 1.0
    totals["fully_censored_rows"] += float(rows)
    return True


def _loss_for_minibatch(
    model: nn.Module,
    batch: Round74EventTrainingBatch,
    row_slice: slice,
    device: object,
    *,
    target_loss_scale: Round74EventTargetLossScale | None = None,
) -> tuple[torch.Tensor, Mapping[str, torch.Tensor], int, int]:
    action_weight, regime_weight, _rows = _minibatch_target_counts(batch, row_slice)
    features = _to_device_tensor(batch.feature_values, row_slice, device)
    payoff = _to_device_tensor(batch.net_payoff_bps, row_slice, device)
    adverse_excursion = _to_device_tensor(
        batch.maximum_adverse_excursion_bps,
        row_slice,
        device,
    )
    adverse = _to_device_tensor(batch.adverse_selection, row_slice, device)
    unpredictable = _to_device_tensor(
        batch.regime_unpredictability,
        row_slice,
        device,
    )
    action_eligibility = _to_device_tensor(
        batch.action_eligibility,
        row_slice,
        device,
    )
    regime_eligibility = _to_device_tensor(
        batch.regime_unpredictability_eligibility,
        row_slice,
        device,
    )
    if target_loss_scale is None:
        payoff_loss_scale = torch.ones_like(payoff)
        adverse_excursion_loss_scale = torch.ones_like(adverse_excursion)
    else:
        payoff_scale_values, adverse_scale_values = target_loss_scale.for_batch(
            batch,
            row_slice,
        )
        payoff_loss_scale = torch.from_numpy(payoff_scale_values).to(device)
        adverse_excursion_loss_scale = torch.from_numpy(adverse_scale_values).to(device)
    output = model(features)
    loss, components = round74_event_model_loss(
        output,
        net_payoff_bps=payoff,
        maximum_adverse_excursion_bps=adverse_excursion,
        adverse_selection=adverse,
        regime_unpredictable=unpredictable,
        action_eligibility=action_eligibility,
        regime_unpredictability_eligibility=regime_eligibility,
        payoff_loss_scale_bps=payoff_loss_scale,
        maximum_adverse_excursion_loss_scale_bps=(adverse_excursion_loss_scale),
        maximum_adverse_excursion_weight=(
            ROUND74_EVENT_TRAINING_LOSS_WEIGHTS["maximum_adverse_excursion"]
        ),
        positive_weight=ROUND74_EVENT_TRAINING_LOSS_WEIGHTS["positive_payoff"],
        adverse_weight=ROUND74_EVENT_TRAINING_LOSS_WEIGHTS["adverse_selection"],
        unpredictability_weight=ROUND74_EVENT_TRAINING_LOSS_WEIGHTS[
            "regime_unpredictability"
        ],
    )
    return (
        loss,
        components,
        action_weight,
        regime_weight,
    )


def _slice_model_output(
    output: Round74EventModelOutput,
    start: int,
    stop: int,
) -> Round74EventModelOutput:
    if start < 0 or stop <= start:
        raise ValueError("Round 74 model-output slice differs")
    return Round74EventModelOutput(
        payoff_quantiles_bps=output.payoff_quantiles_bps[start:stop],
        maximum_adverse_excursion_quantiles_bps=(
            output.maximum_adverse_excursion_quantiles_bps[start:stop]
        ),
        positive_payoff_logits=output.positive_payoff_logits[start:stop],
        adverse_selection_logits=output.adverse_selection_logits[start:stop],
        regime_unpredictability_logits=(
            output.regime_unpredictability_logits[start:stop]
        ),
    )


def _concatenated_device_tensor(
    selections: Sequence[tuple[Round74EventTrainingBatch, slice]],
    attribute: str,
    device: object,
) -> torch.Tensor:
    values = np.concatenate(
        tuple(
            np.asarray(getattr(batch, attribute)[row_slice], dtype=np.float32)
            for batch, row_slice in selections
        ),
        axis=0,
    )
    return torch.from_numpy(np.ascontiguousarray(values)).to(device)


def _losses_for_minibatch_group(
    model: nn.Module,
    selections: Sequence[tuple[Round74EventTrainingBatch, slice]],
    device: object,
    *,
    target_loss_scale: Round74EventTargetLossScale | None = None,
) -> tuple[
    tuple[torch.Tensor, Mapping[str, torch.Tensor], int, int],
    ...,
]:
    if not selections:
        raise ValueError("Round 74 device run group is empty")
    counts = tuple(
        _minibatch_target_counts(batch, row_slice) for batch, row_slice in selections
    )
    if any(action <= 0 or regime <= 0 for action, regime, _rows in counts):
        raise ValueError("Round 74 device run group contains a censored minibatch")
    features = _concatenated_device_tensor(
        selections,
        "feature_values",
        device,
    )
    targets = {
        name: _concatenated_device_tensor(selections, name, device)
        for name in (
            "net_payoff_bps",
            "maximum_adverse_excursion_bps",
            "adverse_selection",
            "regime_unpredictability",
            "action_eligibility",
            "regime_unpredictability_eligibility",
        )
    }
    if target_loss_scale is None:
        payoff_loss_scale = torch.ones_like(targets["net_payoff_bps"])
        adverse_excursion_loss_scale = torch.ones_like(
            targets["maximum_adverse_excursion_bps"]
        )
    else:
        scale_values = tuple(
            target_loss_scale.for_batch(batch, row_slice)
            for batch, row_slice in selections
        )
        payoff_loss_scale = torch.from_numpy(
            np.ascontiguousarray(
                np.concatenate(tuple(value[0] for value in scale_values), axis=0)
            )
        ).to(device)
        adverse_excursion_loss_scale = torch.from_numpy(
            np.ascontiguousarray(
                np.concatenate(tuple(value[1] for value in scale_values), axis=0)
            )
        ).to(device)
    output = model(features)
    results: list[tuple[torch.Tensor, Mapping[str, torch.Tensor], int, int]] = []
    offset = 0
    for action_weight, regime_weight, rows in counts:
        stop = offset + rows
        selected_output = _slice_model_output(output, offset, stop)
        loss, components = _round74_event_model_loss_from_validated_inputs(
            selected_output,
            net_payoff_bps=targets["net_payoff_bps"][offset:stop],
            maximum_adverse_excursion_bps=(
                targets["maximum_adverse_excursion_bps"][offset:stop]
            ),
            adverse_selection=targets["adverse_selection"][offset:stop],
            regime_unpredictable=targets["regime_unpredictability"][offset:stop],
            action_eligibility=targets["action_eligibility"][offset:stop],
            regime_unpredictability_eligibility=(
                targets["regime_unpredictability_eligibility"][offset:stop]
            ),
            payoff_loss_scale_bps=payoff_loss_scale[offset:stop],
            maximum_adverse_excursion_loss_scale_bps=(
                adverse_excursion_loss_scale[offset:stop]
            ),
            maximum_adverse_excursion_weight=(
                ROUND74_EVENT_TRAINING_LOSS_WEIGHTS["maximum_adverse_excursion"]
            ),
            positive_weight=ROUND74_EVENT_TRAINING_LOSS_WEIGHTS["positive_payoff"],
            adverse_weight=ROUND74_EVENT_TRAINING_LOSS_WEIGHTS["adverse_selection"],
            unpredictability_weight=ROUND74_EVENT_TRAINING_LOSS_WEIGHTS[
                "regime_unpredictability"
            ],
        )
        results.append((loss, components, action_weight, regime_weight))
        offset = stop
    if offset != int(features.shape[0]):
        raise RuntimeError("Round 74 device run group row accounting differs")
    return tuple(results)


def _empty_metric_sums() -> dict[str, float]:
    return {
        "payoff_pinball": 0.0,
        "maximum_adverse_excursion_pinball": 0.0,
        "positive_bce": 0.0,
        "adverse_bce": 0.0,
        "unpredictability_bce": 0.0,
        "action_weight": 0.0,
        "regime_weight": 0.0,
        "fully_censored_minibatches": 0.0,
        "fully_censored_rows": 0.0,
    }


def _accumulate_metrics(
    totals: dict[str, float],
    components: Mapping[str, torch.Tensor | float],
    *,
    action_weight: int,
    regime_weight: int,
) -> None:
    def scalar(name: str) -> float:
        value = components[name]
        if isinstance(value, torch.Tensor):
            return float(value.detach().cpu())
        return float(value)

    for name in (
        "payoff_pinball",
        "maximum_adverse_excursion_pinball",
        "positive_bce",
        "adverse_bce",
    ):
        totals[name] += scalar(name) * action_weight
    totals["unpredictability_bce"] += scalar("unpredictability_bce") * regime_weight
    totals["action_weight"] += action_weight
    totals["regime_weight"] += regime_weight


def _accumulate_group_metrics(
    totals: dict[str, float],
    per_run_totals: Sequence[dict[str, float]],
    grouped: Sequence[tuple[torch.Tensor, Mapping[str, torch.Tensor], int, int]],
) -> None:
    names = (
        "payoff_pinball",
        "maximum_adverse_excursion_pinball",
        "positive_bce",
        "adverse_bce",
        "unpredictability_bce",
    )
    if not grouped or len(grouped) != len(per_run_totals):
        raise ValueError("Round 74 grouped metric population differs")
    component_values = (
        torch.stack(
            tuple(
                torch.stack(tuple(components[name] for name in names))
                for _loss, components, _action, _regime in grouped
            )
        )
        .detach()
        .cpu()
        .tolist()
    )
    for run_totals, values, (_loss, _components, action_weight, regime_weight) in zip(
        per_run_totals,
        component_values,
        grouped,
        strict=True,
    ):
        mapped = dict(zip(names, (float(value) for value in values), strict=True))
        _accumulate_metrics(
            totals,
            mapped,
            action_weight=action_weight,
            regime_weight=regime_weight,
        )
        _accumulate_metrics(
            run_totals,
            mapped,
            action_weight=action_weight,
            regime_weight=regime_weight,
        )


def _finalize_metrics(totals: Mapping[str, float]) -> dict[str, float]:
    action_weight = float(totals["action_weight"])
    regime_weight = float(totals["regime_weight"])
    if action_weight <= 0.0 or regime_weight <= 0.0:
        raise ValueError("Round 74 metric aggregation has no eligible targets")
    metrics = {
        name: float(totals[name]) / action_weight
        for name in (
            "payoff_pinball",
            "maximum_adverse_excursion_pinball",
            "positive_bce",
            "adverse_bce",
        )
    }
    metrics["unpredictability_bce"] = (
        float(totals["unpredictability_bce"]) / regime_weight
    )
    metrics["loss"] = (
        metrics["payoff_pinball"]
        + ROUND74_EVENT_TRAINING_LOSS_WEIGHTS["maximum_adverse_excursion"]
        * metrics["maximum_adverse_excursion_pinball"]
        + ROUND74_EVENT_TRAINING_LOSS_WEIGHTS["positive_payoff"]
        * metrics["positive_bce"]
        + ROUND74_EVENT_TRAINING_LOSS_WEIGHTS["adverse_selection"]
        * metrics["adverse_bce"]
        + ROUND74_EVENT_TRAINING_LOSS_WEIGHTS["regime_unpredictability"]
        * metrics["unpredictability_bce"]
    )
    metrics["eligible_action_targets"] = action_weight
    metrics["eligible_regime_targets"] = regime_weight
    metrics["fully_censored_minibatches"] = float(totals["fully_censored_minibatches"])
    metrics["fully_censored_rows"] = float(totals["fully_censored_rows"])
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("Round 74 aggregate metrics are nonfinite")
    return metrics


def _evaluate_model(
    model: nn.Module,
    batches: Sequence[Round74EventTrainingBatch],
    *,
    minibatch_rows: int,
    device_run_group_size: int,
    device: object,
    target_loss_scale: Round74EventTargetLossScale,
) -> tuple[dict[str, float], tuple[float, ...]]:
    target_loss_scale.validate()
    model.eval()
    totals = _empty_metric_sums()
    per_run_totals = tuple(_empty_metric_sums() for _batch in batches)
    selections: list[tuple[int, Round74EventTrainingBatch, slice]] = []
    with torch.no_grad():
        for run_index, batch in enumerate(batches):
            for _selected, row_slice in _iter_minibatches(
                (batch,),
                minibatch_rows,
            ):
                if _skip_fully_censored_minibatch(
                    totals,
                    batch,
                    row_slice,
                ):
                    _skip_fully_censored_minibatch(
                        per_run_totals[run_index],
                        batch,
                        row_slice,
                    )
                    continue
                selections.append((run_index, batch, row_slice))
        for start in range(0, len(selections), device_run_group_size):
            selected_group = selections[start : start + device_run_group_size]
            grouped = _losses_for_minibatch_group(
                model,
                tuple(
                    (batch, row_slice) for _index, batch, row_slice in selected_group
                ),
                device,
                target_loss_scale=target_loss_scale,
            )
            _accumulate_group_metrics(
                totals,
                tuple(
                    per_run_totals[index] for index, _batch, _slice in selected_group
                ),
                grouped,
            )
    per_run_metrics = tuple(
        _finalize_metrics(run_totals) for run_totals in per_run_totals
    )
    metrics = _finalize_metrics(totals)
    run_losses = tuple(item["loss"] for item in per_run_metrics)
    if not run_losses:
        raise ValueError("Round 74 tuning has no capture runs")
    metrics["run_balanced_loss"] = sum(run_losses) / len(run_losses)
    metrics["worst_run_loss"] = max(run_losses)
    metrics["run_count"] = float(len(run_losses))
    if not all(math.isfinite(value) for value in metrics.values()):
        raise RuntimeError("Round 74 run-balanced metrics are nonfinite")
    return metrics, run_losses


def _index_model_output(
    output: Round74EventModelOutput,
    indices: torch.Tensor,
) -> Round74EventModelOutput:
    selected = Round74EventModelOutput(
        payoff_quantiles_bps=output.payoff_quantiles_bps.index_select(0, indices),
        maximum_adverse_excursion_quantiles_bps=(
            output.maximum_adverse_excursion_quantiles_bps.index_select(0, indices)
        ),
        positive_payoff_logits=output.positive_payoff_logits.index_select(0, indices),
        adverse_selection_logits=output.adverse_selection_logits.index_select(
            0,
            indices,
        ),
        regime_unpredictability_logits=(
            output.regime_unpredictability_logits.index_select(0, indices)
        ),
    )
    selected.validate(int(indices.numel()))
    return selected


def _tuning_group_key(run_id: str, symbol: str, horizon_seconds: int) -> str:
    return f"{run_id}:{symbol}:{int(horizon_seconds)}"


def _evaluate_model_group_losses(
    model: nn.Module,
    batches: Sequence[Round74EventTrainingBatch],
    *,
    minibatch_rows: int,
    device: object,
    require_complete_symbol_panel: bool,
    target_loss_scale: Round74EventTargetLossScale,
) -> dict[str, float]:
    """Measure proper loss for every eligible run-symbol-horizon subgroup."""

    target_loss_scale.validate()
    if not batches:
        raise ValueError("Round 74 tuning subgroup panel is empty")
    model.eval()
    totals_by_group: dict[str, dict[str, float]] = {}
    expected_groups: set[str] = set()
    symbol_universe = set(IMPACT_CAPTURE_SYMBOLS)
    horizon_count = len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
    horizon_masks = tuple(
        torch.tensor(
            tuple(
                1.0 if candidate == horizon_index else 0.0
                for candidate in range(horizon_count)
            ),
            dtype=torch.float32,
            device=device,
        )
        for horizon_index in range(horizon_count)
    )
    with torch.no_grad():
        for batch in batches:
            batch.validate()
            run_ids = set(batch.run_id)
            observed_symbols = set(batch.symbol)
            if len(run_ids) != 1 or not observed_symbols.issubset(symbol_universe):
                raise ValueError("Round 74 tuning subgroup identity differs")
            if require_complete_symbol_panel and observed_symbols != symbol_universe:
                raise ValueError("Round 74 tuning subgroup symbol panel is incomplete")
            run_id = next(iter(run_ids))
            for symbol in IMPACT_CAPTURE_SYMBOLS:
                if symbol not in observed_symbols:
                    continue
                for horizon_seconds in ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS:
                    key = _tuning_group_key(run_id, symbol, horizon_seconds)
                    expected_groups.add(key)
                    totals_by_group.setdefault(key, _empty_metric_sums())
            for _selected, row_slice in _iter_minibatches((batch,), minibatch_rows):
                start, stop, step = row_slice.indices(batch.rows)
                if step != 1 or stop <= start:
                    raise ValueError("Round 74 tuning subgroup slice differs")
                output = model(
                    _to_device_tensor(batch.feature_values, row_slice, device)
                )
                targets = {
                    name: _to_device_tensor(getattr(batch, name), row_slice, device)
                    for name in (
                        "net_payoff_bps",
                        "maximum_adverse_excursion_bps",
                        "adverse_selection",
                        "regime_unpredictability",
                        "action_eligibility",
                        "regime_unpredictability_eligibility",
                    )
                }
                payoff_scale_values, adverse_scale_values = target_loss_scale.for_batch(
                    batch, row_slice
                )
                payoff_loss_scale = torch.from_numpy(payoff_scale_values).to(device)
                adverse_excursion_loss_scale = torch.from_numpy(
                    adverse_scale_values
                ).to(device)
                symbols = np.asarray(batch.symbol[start:stop], dtype=object)
                for symbol in IMPACT_CAPTURE_SYMBOLS:
                    row_indices = np.flatnonzero(symbols == symbol)
                    if row_indices.size == 0:
                        continue
                    index = torch.tensor(
                        row_indices.tolist(),
                        dtype=torch.int64,
                        device=device,
                    )
                    selected_output = _index_model_output(output, index)
                    selected_targets = {
                        name: tensor.index_select(0, index)
                        for name, tensor in targets.items()
                    }
                    for horizon_index, horizon_seconds in enumerate(
                        ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS
                    ):
                        horizon_mask = horizon_masks[horizon_index]
                        action_eligibility = selected_targets[
                            "action_eligibility"
                        ] * horizon_mask.reshape(1, -1, 1)
                        regime_eligibility = selected_targets[
                            "regime_unpredictability_eligibility"
                        ] * horizon_mask.reshape(1, -1)
                        action_weight = int(
                            float(action_eligibility.sum().detach().cpu())
                        )
                        regime_weight = int(
                            float(regime_eligibility.sum().detach().cpu())
                        )
                        if action_weight == 0 and regime_weight == 0:
                            continue
                        if action_weight <= 0 or regime_weight <= 0:
                            raise ValueError(
                                "Round 74 tuning subgroup eligibility differs"
                            )
                        _loss, components = (
                            _round74_event_model_loss_from_validated_inputs(
                                selected_output,
                                net_payoff_bps=selected_targets["net_payoff_bps"],
                                maximum_adverse_excursion_bps=selected_targets[
                                    "maximum_adverse_excursion_bps"
                                ],
                                adverse_selection=selected_targets["adverse_selection"],
                                regime_unpredictable=selected_targets[
                                    "regime_unpredictability"
                                ],
                                action_eligibility=action_eligibility,
                                regime_unpredictability_eligibility=regime_eligibility,
                                payoff_loss_scale_bps=(
                                    payoff_loss_scale.index_select(0, index)
                                ),
                                maximum_adverse_excursion_loss_scale_bps=(
                                    adverse_excursion_loss_scale.index_select(
                                        0,
                                        index,
                                    )
                                ),
                                maximum_adverse_excursion_weight=(
                                    ROUND74_EVENT_TRAINING_LOSS_WEIGHTS[
                                        "maximum_adverse_excursion"
                                    ]
                                ),
                                positive_weight=ROUND74_EVENT_TRAINING_LOSS_WEIGHTS[
                                    "positive_payoff"
                                ],
                                adverse_weight=ROUND74_EVENT_TRAINING_LOSS_WEIGHTS[
                                    "adverse_selection"
                                ],
                                unpredictability_weight=(
                                    ROUND74_EVENT_TRAINING_LOSS_WEIGHTS[
                                        "regime_unpredictability"
                                    ]
                                ),
                            )
                        )
                        _accumulate_metrics(
                            totals_by_group[
                                _tuning_group_key(
                                    run_id,
                                    symbol,
                                    horizon_seconds,
                                )
                            ],
                            components,
                            action_weight=action_weight,
                            regime_weight=regime_weight,
                        )
    if not expected_groups or set(totals_by_group) != expected_groups:
        raise ValueError("Round 74 tuning subgroup panel differs")
    result = {
        key: _finalize_metrics(totals_by_group[key])["loss"]
        for key in sorted(expected_groups)
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("Round 74 tuning subgroup loss is nonfinite")
    return result


def _cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in sorted(model.state_dict().items())
    }


def _parameters_are_finite(model: nn.Module) -> bool:
    checks = tuple(
        torch.isfinite(parameter.detach()).all() for parameter in model.parameters()
    )
    return bool(checks) and bool(torch.stack(checks).all().detach().cpu())


def _train_peer(
    candidate_id: str,
    feature_view: str,
    state_conditioned_flow: bool,
    initialization_id: str,
    seed: int,
    training_batches: Sequence[Round74EventTrainingBatch],
    early_stopping_batches: Sequence[Round74EventTrainingBatch],
    *,
    config: Round74EventTrainingConfig,
    device: object,
    target_loss_scale: Round74EventTargetLossScale,
    pretraining_split: Round74EventPretrainingSplit | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    target_loss_scale.validate()
    if initialization_id not in ROUND74_EVENT_PRETRAINING_INITIALIZATION_IDS:
        raise ValueError("Round 74 model initialization differs")
    if initialization_id == "random" and pretraining_split is not None:
        raise ValueError("Round 74 random initialization received a pretraining split")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = _bind_round74_feature_view(
        build_round74_event_model(
            candidate_id,
            state_conditioned_flow=state_conditioned_flow,
        ),
        feature_view,
    ).to(device)
    supervised_python_random = random.getstate()
    supervised_numpy_random = np.random.get_state()
    supervised_torch_random = torch.get_rng_state()
    if initialization_id == "causal_next_event_pretrained":
        pretraining_report = pretrain_round74_event_encoder(
            model,
            training_batches,
            device=device,
            masked_feature_indices=ROUND74_EVENT_FEATURE_VIEW_MASKED_INDICES[
                feature_view
            ],
            config=config.pretraining,
            split=pretraining_split,
        )
    else:
        pretraining_report = {
            "schema_version": ROUND74_EVENT_PRETRAINING_SCHEMA_VERSION,
            "initialization_id": "random",
            "applied": False,
            "reason": "paired_random_initialization_incumbent",
            "config": config.pretraining.as_dict(),
            "supervised_targets_used": False,
            "tuning_features_used": False,
            "tuning_targets_used": False,
            "calibration_data_used": False,
            "test_data_used": False,
            "financial_edge_claim": False,
        }
    random.setstate(supervised_python_random)
    np.random.set_state(supervised_numpy_random)
    torch.set_rng_state(supervised_torch_random)
    optimizer = ExplicitAdamW(
        tuple(model.parameters()),
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, object]] = []
    for epoch in range(1, config.maximum_epochs + 1):
        model.train()
        optimization_totals = _empty_metric_sums()
        per_run_totals = tuple(_empty_metric_sums() for _batch in training_batches)
        run_count = len(training_batches)
        if config.execution_mode == "segmented_cohort":
            selections = _eligible_target_minibatch_schedule(
                training_batches,
                config.minibatch_rows,
                totals=optimization_totals,
                per_run_totals=per_run_totals,
            )
            total_action_weight = sum(item[3] for item in selections)
            total_regime_weight = sum(item[4] for item in selections)
            optimizer.zero_grad(set_to_none=True)
            for start in range(
                0,
                len(selections),
                config.device_run_group_size,
            ):
                selected_group = selections[
                    start : start + config.device_run_group_size
                ]
                grouped = _losses_for_minibatch_group(
                    model,
                    tuple(
                        (batch, row_slice)
                        for _index, batch, row_slice, _action, _regime in selected_group
                    ),
                    device,
                    target_loss_scale=target_loss_scale,
                )
                _eligible_target_weighted_group_loss(
                    grouped,
                    total_action_weight=total_action_weight,
                    total_regime_weight=total_regime_weight,
                ).backward()
                _accumulate_group_metrics(
                    optimization_totals,
                    tuple(
                        per_run_totals[index]
                        for index, _batch, _slice, _action, _regime in selected_group
                    ),
                    grouped,
                )
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=config.gradient_clip_norm,
                foreach=False,
            )
            if not math.isfinite(float(gradient_norm.detach().cpu())):
                raise RuntimeError("Round 74 model gradient norm is nonfinite")
            optimizer.step()
            optimizer_steps = 1
            contribution_counts = tuple(
                sum(item[0] == run_index for item in selections)
                for run_index in range(run_count)
            )
            eligible_minibatch_counts = contribution_counts
        else:
            schedules = _run_balanced_minibatch_schedules(
                training_batches,
                config.minibatch_rows,
                totals=optimization_totals,
                per_run_totals=per_run_totals,
            )
            optimizer_steps = max(len(schedule) for schedule in schedules)
            contribution_counts = tuple(optimizer_steps for _batch in training_batches)
            eligible_minibatch_counts = tuple(len(schedule) for schedule in schedules)
            for step in range(optimizer_steps):
                optimizer.zero_grad(set_to_none=True)
                selections = tuple(
                    (
                        batch,
                        schedule[(step + epoch - 1) % len(schedule)],
                        run_totals,
                    )
                    for batch, schedule, run_totals in zip(
                        training_batches,
                        schedules,
                        per_run_totals,
                        strict=True,
                    )
                )
                for start in range(0, run_count, config.device_run_group_size):
                    selected_group = selections[
                        start : start + config.device_run_group_size
                    ]
                    grouped = _losses_for_minibatch_group(
                        model,
                        tuple(
                            (batch, row_slice)
                            for batch, row_slice, _run_totals in selected_group
                        ),
                        device,
                        target_loss_scale=target_loss_scale,
                    )
                    (
                        torch.stack(tuple(item[0] for item in grouped)).sum()
                        / run_count
                    ).backward()
                    _accumulate_group_metrics(
                        optimization_totals,
                        tuple(
                            run_totals for _batch, _slice, run_totals in selected_group
                        ),
                        grouped,
                    )
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=config.gradient_clip_norm,
                    foreach=False,
                )
                if not math.isfinite(float(gradient_norm.detach().cpu())):
                    raise RuntimeError("Round 74 model gradient norm is nonfinite")
                optimizer.step()
        if not _parameters_are_finite(model):
            raise RuntimeError("Round 74 model parameters are nonfinite")
        optimization_metrics = _finalize_metrics(optimization_totals)
        optimization_run_losses = tuple(
            _finalize_metrics(run_totals)["loss"] for run_totals in per_run_totals
        )
        optimization_metrics.update(
            {
                "run_balanced_loss": (
                    sum(optimization_run_losses) / len(optimization_run_losses)
                ),
                "worst_run_loss": max(optimization_run_losses),
                "run_count": float(run_count),
                "optimizer_steps": float(optimizer_steps),
                "run_contributions_per_optimizer_step": (
                    float(run_count)
                    if config.execution_mode != "segmented_cohort"
                    else float(len(selections))
                ),
                "minimum_run_minibatch_contributions": float(min(contribution_counts)),
                "maximum_run_minibatch_contributions": float(max(contribution_counts)),
                "minimum_eligible_minibatches_per_run": float(
                    min(eligible_minibatch_counts)
                ),
                "maximum_eligible_minibatches_per_run": float(
                    max(eligible_minibatch_counts)
                ),
            }
        )
        if not all(math.isfinite(value) for value in optimization_metrics.values()):
            raise RuntimeError("Round 74 run-balanced optimization is nonfinite")
        early_stopping_metrics, _early_stopping_run_losses = _evaluate_model(
            model,
            early_stopping_batches,
            minibatch_rows=config.minibatch_rows,
            device_run_group_size=config.device_run_group_size,
            device=device,
            target_loss_scale=target_loss_scale,
        )
        selection_loss_name = (
            "loss"
            if config.execution_mode == "segmented_cohort"
            else "run_balanced_loss"
        )
        early_stopping_loss = early_stopping_metrics[selection_loss_name]
        improved = (
            best_state is None
            or early_stopping_loss < best_loss - config.minimum_tuning_improvement
        )
        if improved:
            best_loss = early_stopping_loss
            best_epoch = epoch
            best_state = _cpu_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append(
            {
                "epoch": epoch,
                "optimization_metrics": optimization_metrics,
                "early_stopping_metrics": early_stopping_metrics,
                "selection_loss_name": selection_loss_name,
                "improved": improved,
            }
        )
        if epochs_without_improvement >= config.early_stopping_patience:
            break
    if best_state is None or best_epoch < 1:
        raise RuntimeError("Round 74 model has no finite early-stop state")
    model.load_state_dict(best_state, strict=True)
    restored_metrics, _restored_run_losses = _evaluate_model(
        model,
        early_stopping_batches,
        minibatch_rows=config.minibatch_rows,
        device_run_group_size=config.device_run_group_size,
        device=device,
        target_loss_scale=target_loss_scale,
    )
    if not math.isclose(
        restored_metrics[selection_loss_name],
        best_loss,
        rel_tol=1e-7,
        abs_tol=1e-7,
    ):
        raise RuntimeError("Round 74 best-state selection reload metric differs")
    return best_state, {
        "seed": seed,
        "state_conditioned_flow": state_conditioned_flow,
        "initialization_id": initialization_id,
        "causal_pretraining": pretraining_report,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "best_early_stopping_metrics": restored_metrics,
        "history": history,
    }


def _prediction_sha256(
    model: nn.Module,
    batch: Round74EventTrainingBatch,
    *,
    maximum_rows: int,
    device: object,
) -> str:
    rows = min(batch.rows, maximum_rows)
    values = _to_device_tensor(
        batch.feature_values,
        slice(0, rows),
        device,
    )
    model.eval()
    with torch.no_grad():
        output = model(values)
    digest = hashlib.sha256()
    digest.update(str(rows).encode("ascii"))
    for name in (
        "payoff_quantiles_bps",
        "maximum_adverse_excursion_quantiles_bps",
        "positive_payoff_logits",
        "adverse_selection_logits",
        "regime_unpredictability_logits",
    ):
        array = getattr(output, name).detach().cpu().numpy().astype("<f4", copy=False)
        digest.update(name.encode("ascii"))
        digest.update(memoryview(np.ascontiguousarray(array)).cast("B"))
    return digest.hexdigest()


def _ensemble_from_states(
    candidate_id: str,
    feature_view: str,
    state_conditioned_flow: bool,
    states: Sequence[Mapping[str, torch.Tensor]],
    *,
    device: object,
) -> Round74EventEnsemble:
    ensemble = Round74EventEnsemble(
        candidate_id,
        len(states),
        feature_view=feature_view,
        state_conditioned_flow=state_conditioned_flow,
    )
    for peer, state in zip(ensemble.peers, states, strict=True):
        peer.load_state_dict(dict(state), strict=True)
    return ensemble.to(device)


def _fit_candidate(
    candidate_id: str,
    feature_view: str,
    training_batches: Sequence[Round74EventTrainingBatch],
    early_stopping_batches: Sequence[Round74EventTrainingBatch],
    promotion_batches: Sequence[Round74EventTrainingBatch],
    *,
    config: Round74EventTrainingConfig,
    device: object,
    target_loss_scale: Round74EventTargetLossScale,
    initialization_id: str = "random",
    state_conditioned_flow: bool = False,
) -> _CandidateFit:
    states: list[dict[str, torch.Tensor]] = []
    reports: list[dict[str, object]] = []
    pretraining_split = (
        build_round74_event_pretraining_split(
            training_batches,
            config=config.pretraining,
        )
        if initialization_id == "causal_next_event_pretrained"
        else None
    )
    for seed in config.seeds:
        state, report = _train_peer(
            candidate_id,
            feature_view,
            state_conditioned_flow,
            initialization_id,
            seed,
            training_batches,
            early_stopping_batches,
            config=config,
            device=device,
            target_loss_scale=target_loss_scale,
            pretraining_split=pretraining_split,
        )
        states.append(state)
        reports.append(report)
    ensemble = _ensemble_from_states(
        candidate_id,
        feature_view,
        state_conditioned_flow,
        states,
        device=device,
    )
    ensemble_metrics, ensemble_run_losses = _evaluate_model(
        ensemble,
        promotion_batches,
        minibatch_rows=config.minibatch_rows,
        device_run_group_size=config.device_run_group_size,
        device=device,
        target_loss_scale=target_loss_scale,
    )
    ensemble_group_losses = _evaluate_model_group_losses(
        ensemble,
        promotion_batches,
        minibatch_rows=config.minibatch_rows,
        device=device,
        require_complete_symbol_panel=config.execution_mode != "preflight",
        target_loss_scale=target_loss_scale,
    )
    prediction_sha256 = _prediction_sha256(
        ensemble,
        early_stopping_batches[0],
        maximum_rows=config.minibatch_rows,
        device=device,
    )
    parameter_count = sum(
        parameter.numel() for parameter in ensemble.peers[0].parameters()
    )
    return _CandidateFit(
        candidate_id=candidate_id,
        feature_view=feature_view,
        state_conditioned_flow=state_conditioned_flow,
        peer_states=tuple(states),
        peer_reports=tuple(reports),
        ensemble_metrics=ensemble_metrics,
        ensemble_run_losses=ensemble_run_losses,
        ensemble_group_losses=ensemble_group_losses,
        ensemble_prediction_sha256=prediction_sha256,
        parameter_count_per_peer=parameter_count,
    )


def round74_paired_run_stability_evidence(
    improvements: Sequence[float],
    *,
    minimum_improvement: float,
) -> dict[str, object]:
    """Return assumption-free robustness gates for paired capture-run gains."""

    values = tuple(float(value) for value in improvements)
    minimum = float(minimum_improvement)
    if (
        not values
        or not math.isfinite(minimum)
        or minimum < 0.0
        or any(not math.isfinite(value) for value in values)
    ):
        raise ValueError("Round 74 paired-run stability panel differs")
    material_win_count = sum(value > minimum for value in values)
    required_material_win_count = len(values) // 2 + 1
    material_win_majority = material_win_count >= required_material_win_count
    deletion_gate_applied = len(values) > 1
    minimum_deletion_mean = (
        min(
            math.fsum(
                value
                for candidate_index, value in enumerate(values)
                if candidate_index != removed_index
            )
            / (len(values) - 1)
            for removed_index in range(len(values))
        )
        if deletion_gate_applied
        else None
    )
    deletion_stable = (
        minimum_deletion_mean is not None and minimum_deletion_mean > minimum
    )
    return {
        "material_challenger_win_count": material_win_count,
        "minimum_required_material_win_count": required_material_win_count,
        "material_win_majority": material_win_majority,
        "single_capture_run_deletion_gate_applied": deletion_gate_applied,
        "minimum_leave_one_capture_run_out_mean_proper_loss_improvement": (
            minimum_deletion_mean
        ),
        "all_leave_one_capture_run_out_panels_exceed_minimum_mean_improvement": (
            deletion_stable
        ),
        "statistical_independence_or_significance_claim": False,
    }


def _paired_promotion_report(
    incumbent_id: str,
    challenger_id: str,
    incumbent_losses: Sequence[float],
    challenger_losses: Sequence[float],
    *,
    minimum_improvement: float,
    required_paired_run_count: int,
    incumbent_group_losses: Mapping[str, float] | None,
    challenger_group_losses: Mapping[str, float] | None,
) -> dict[str, object]:
    improvements = tuple(
        float(incumbent_loss) - float(challenger_loss)
        for incumbent_loss, challenger_loss in zip(
            incumbent_losses,
            challenger_losses,
            strict=True,
        )
    )
    if not improvements:
        raise ValueError("Round 74 paired-promotion losses are empty")
    wins = sum(value > 0.0 for value in improvements)
    losses_count = sum(value < 0.0 for value in improvements)
    ties = len(improvements) - wins - losses_count
    mean_improvement = math.fsum(improvements) / len(improvements)
    maximum_loss_degradation = max(-value for value in improvements)
    stability = round74_paired_run_stability_evidence(
        improvements,
        minimum_improvement=minimum_improvement,
    )
    complete_tuning_panel = len(improvements) == required_paired_run_count
    all_runs_noninferior = maximum_loss_degradation <= minimum_improvement
    if (incumbent_group_losses is None) != (challenger_group_losses is None):
        raise ValueError("Round 74 paired-promotion subgroup panel differs")
    if (
        incumbent_group_losses is not None
        and challenger_group_losses is not None
        and set(incumbent_group_losses) != set(challenger_group_losses)
    ):
        raise ValueError("Round 74 paired-promotion subgroup panel differs")
    subgroup_improvements = (
        {
            key: float(incumbent_group_losses[key])
            - float(challenger_group_losses[key])
            for key in sorted(incumbent_group_losses)
        }
        if incumbent_group_losses is not None and challenger_group_losses is not None
        else {}
    )
    worst_subgroup_key = (
        min(
            subgroup_improvements,
            key=lambda key: (subgroup_improvements[key], key),
        )
        if subgroup_improvements
        else None
    )
    maximum_subgroup_loss_degradation = (
        -subgroup_improvements[worst_subgroup_key]
        if worst_subgroup_key is not None
        else None
    )
    all_subgroups_noninferior = (
        maximum_subgroup_loss_degradation is None
        or maximum_subgroup_loss_degradation <= minimum_improvement
    )
    promoted = (
        complete_tuning_panel
        and mean_improvement > minimum_improvement
        and all_runs_noninferior
        and all_subgroups_noninferior
        and stability["material_win_majority"] is True
        and stability[
            "all_leave_one_capture_run_out_panels_exceed_minimum_mean_improvement"
        ]
        is True
    )
    return {
        "incumbent_candidate_id": incumbent_id,
        "challenger_candidate_id": challenger_id,
        "paired_capture_run_count": len(improvements),
        "required_paired_capture_run_count": required_paired_run_count,
        "complete_tuning_panel": complete_tuning_panel,
        "challenger_win_count": wins,
        "challenger_loss_count": losses_count,
        "exact_tie_count": ties,
        "mean_proper_loss_improvement": mean_improvement,
        "minimum_mean_proper_loss_improvement": minimum_improvement,
        "maximum_paired_run_loss_degradation": maximum_loss_degradation,
        "maximum_permitted_paired_run_loss_degradation": minimum_improvement,
        "all_paired_runs_noninferior": all_runs_noninferior,
        "subgroup_gate_applied": incumbent_group_losses is not None,
        "paired_run_symbol_horizon_group_count": len(subgroup_improvements),
        "worst_run_symbol_horizon_group": worst_subgroup_key,
        "maximum_paired_group_loss_degradation": (maximum_subgroup_loss_degradation),
        "maximum_permitted_paired_group_loss_degradation": minimum_improvement,
        "all_paired_run_symbol_horizon_groups_noninferior": (all_subgroups_noninferior),
        **stability,
        "promoted": promoted,
    }


def _complexity_gated_candidate_id(
    candidate_ids: Sequence[str],
    candidate_run_losses: Mapping[str, Sequence[float]],
    candidate_parameter_counts: Mapping[str, int],
    *,
    minimum_mean_loss_improvement: float,
    candidate_group_losses: Mapping[str, Mapping[str, float]] | None = None,
    required_paired_run_count: int = (
        ROUND74_COMPLEXITY_PROMOTION_REQUIRED_TUNING_RUNS
    ),
) -> tuple[str, tuple[dict[str, object], ...]]:
    ordered = tuple(candidate_ids)
    minimum_improvement = float(minimum_mean_loss_improvement)
    if (
        ordered != ROUND74_EVENT_MODEL_CANDIDATES
        or set(candidate_run_losses) != set(ordered)
        or set(candidate_parameter_counts) != set(ordered)
        or not math.isfinite(minimum_improvement)
        or minimum_improvement < 0.0
        or isinstance(required_paired_run_count, bool)
        or not isinstance(required_paired_run_count, int)
        or required_paired_run_count < 1
    ):
        raise ValueError("Round 74 complexity-promotion panel differs")
    parameter_counts = tuple(candidate_parameter_counts[value] for value in ordered)
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in parameter_counts
    ) or any(
        later <= earlier
        for earlier, later in zip(
            parameter_counts,
            parameter_counts[1:],
        )
    ):
        raise ValueError("Round 74 complexity order differs")
    losses = {
        candidate_id: tuple(
            float(value) for value in candidate_run_losses[candidate_id]
        )
        for candidate_id in ordered
    }
    run_counts = {len(value) for value in losses.values()}
    if (
        len(run_counts) != 1
        or not run_counts
        or next(iter(run_counts)) < 1
        or any(
            not math.isfinite(value)
            for candidate_losses in losses.values()
            for value in candidate_losses
        )
    ):
        raise ValueError("Round 74 complexity-promotion run losses differ")
    group_losses: dict[str, dict[str, float]] | None = None
    group_keys: tuple[str, ...] = ()
    if candidate_group_losses is not None:
        if set(candidate_group_losses) != set(ordered):
            raise ValueError("Round 74 complexity-promotion subgroup panel differs")
        group_losses = {
            candidate_id: {
                str(key): float(value)
                for key, value in candidate_group_losses[candidate_id].items()
            }
            for candidate_id in ordered
        }
        group_keys = tuple(sorted(group_losses[ordered[0]]))
        if (
            not group_keys
            or any(
                tuple(sorted(group_losses[candidate_id])) != group_keys
                for candidate_id in ordered
            )
            or any(
                not math.isfinite(value)
                for candidate_losses in group_losses.values()
                for value in candidate_losses.values()
            )
        ):
            raise ValueError("Round 74 complexity-promotion subgroup panel differs")
    incumbent = ordered[0]
    reports: list[dict[str, object]] = []
    for challenger in ordered[1:]:
        report = _paired_promotion_report(
            incumbent,
            challenger,
            losses[incumbent],
            losses[challenger],
            minimum_improvement=minimum_improvement,
            required_paired_run_count=required_paired_run_count,
            incumbent_group_losses=(
                group_losses[incumbent] if group_losses is not None else None
            ),
            challenger_group_losses=(
                group_losses[challenger] if group_losses is not None else None
            ),
        )
        reports.append(report)
        if report["promoted"] is True:
            incumbent = challenger
    return incumbent, tuple(reports)


def _select_candidate_with_complexity_gate(
    fits: Sequence[_CandidateFit],
    *,
    minimum_mean_loss_improvement: float,
    required_paired_run_count: int,
) -> tuple[_CandidateFit, tuple[dict[str, object], ...]]:
    fit_by_id = {fit.candidate_id: fit for fit in fits}
    if len(fit_by_id) != len(fits):
        raise ValueError("Round 74 candidate fit identity differs")
    selected_id, reports = _complexity_gated_candidate_id(
        tuple(fit_by_id),
        {
            candidate_id: fit_by_id[candidate_id].ensemble_run_losses
            for candidate_id in fit_by_id
        },
        {
            candidate_id: fit_by_id[candidate_id].parameter_count_per_peer
            for candidate_id in fit_by_id
        },
        minimum_mean_loss_improvement=minimum_mean_loss_improvement,
        candidate_group_losses={
            candidate_id: fit_by_id[candidate_id].ensemble_group_losses
            for candidate_id in fit_by_id
        },
        required_paired_run_count=required_paired_run_count,
    )
    return fit_by_id[selected_id], reports


def _feature_view_promotion_report(
    incumbent_feature_view: str,
    challenger_feature_view: str,
    incumbent_run_losses: Sequence[float],
    challenger_run_losses: Sequence[float],
    *,
    incumbent_group_losses: Mapping[str, float],
    challenger_group_losses: Mapping[str, float],
    minimum_mean_loss_improvement: float,
    required_paired_run_count: int,
) -> dict[str, object]:
    if (
        incumbent_feature_view not in ROUND74_EVENT_FEATURE_VIEWS
        or challenger_feature_view not in ROUND74_EVENT_FEATURE_VIEWS
        or incumbent_feature_view == challenger_feature_view
    ):
        raise ValueError("Round 74 feature-view promotion identity differs")
    raw_report = _paired_promotion_report(
        incumbent_feature_view,
        challenger_feature_view,
        incumbent_run_losses,
        challenger_run_losses,
        minimum_improvement=float(minimum_mean_loss_improvement),
        required_paired_run_count=required_paired_run_count,
        incumbent_group_losses=incumbent_group_losses,
        challenger_group_losses=challenger_group_losses,
    )
    return {
        (
            "incumbent_feature_view"
            if key == "incumbent_candidate_id"
            else "challenger_feature_view"
            if key == "challenger_candidate_id"
            else key
        ): value
        for key, value in raw_report.items()
    }


def _select_feature_view_with_ablation_gate(
    incumbent_fit: _CandidateFit,
    challenger_fit: _CandidateFit,
    *,
    minimum_mean_loss_improvement: float,
    required_paired_run_count: int,
) -> tuple[_CandidateFit, dict[str, object]]:
    if (
        incumbent_fit.candidate_id != challenger_fit.candidate_id
        or incumbent_fit.feature_view not in ROUND74_EVENT_FEATURE_VIEWS
        or challenger_fit.feature_view not in ROUND74_EVENT_FEATURE_VIEWS
        or incumbent_fit.feature_view == challenger_fit.feature_view
        or incumbent_fit.parameter_count_per_peer
        != challenger_fit.parameter_count_per_peer
        or len(incumbent_fit.peer_states) != len(challenger_fit.peer_states)
    ):
        raise ValueError("Round 74 feature-view ablation panel differs")
    report = _feature_view_promotion_report(
        incumbent_fit.feature_view,
        challenger_fit.feature_view,
        incumbent_fit.ensemble_run_losses,
        challenger_fit.ensemble_run_losses,
        incumbent_group_losses=incumbent_fit.ensemble_group_losses,
        challenger_group_losses=challenger_fit.ensemble_group_losses,
        minimum_mean_loss_improvement=minimum_mean_loss_improvement,
        required_paired_run_count=required_paired_run_count,
    )
    winner = challenger_fit if report["promoted"] is True else incumbent_fit
    return winner, report


def _order_flow_challenger_feature_view(clock_winner_view: str) -> str:
    if clock_winner_view == "market_state_clock_neutral":
        return "clock_neutral"
    if clock_winner_view == "market_state_with_clock":
        return "full"
    raise ValueError("Round 74 clock-control winner feature view differs")


def _feature_view_selection_criterion(
    clock_required_paired_run_count: int,
    order_flow_required_paired_run_count: int,
) -> str:
    if clock_required_paired_run_count < 1 or order_flow_required_paired_run_count < 1:
        raise ValueError("Round 74 feature-view paired-run count differs")
    return (
        "a fixed state-first sequence evaluates market state anchored "
        "by L2 liquidity, then controls for clock and intraday-phase "
        "features before admitting order flow; the clock and order-flow "
        f"challengers use their disjoint panels of "
        f"{clock_required_paired_run_count} and "
        f"{order_flow_required_paired_run_count} paired model-selection "
        "capture runs, respectively, "
        "and requires strict mean proper-loss improvement with no run or "
        "run-symbol-horizon subgroup degradation beyond the numerical floor, "
        "a material win on a strict majority of capture runs, and mean "
        "improvement above the floor after deleting any one capture run"
    )


def _feature_view_contains_order_flow(feature_view: str) -> bool:
    if feature_view not in ROUND74_EVENT_FEATURE_VIEWS:
        raise ValueError("Round 74 interaction feature view differs")
    return feature_view in {"clock_neutral", "full"}


def _select_state_conditioned_flow_with_ablation_gate(
    incumbent_fit: _CandidateFit,
    challenger_fit: _CandidateFit,
    *,
    minimum_mean_loss_improvement: float,
    required_paired_run_count: int,
) -> tuple[_CandidateFit, dict[str, object]]:
    if (
        incumbent_fit.candidate_id != challenger_fit.candidate_id
        or incumbent_fit.feature_view != challenger_fit.feature_view
        or not _feature_view_contains_order_flow(incumbent_fit.feature_view)
        or incumbent_fit.state_conditioned_flow is not False
        or challenger_fit.state_conditioned_flow is not True
        or challenger_fit.parameter_count_per_peer
        <= incumbent_fit.parameter_count_per_peer
        or len(incumbent_fit.peer_states) != len(challenger_fit.peer_states)
    ):
        raise ValueError("Round 74 state-conditioned flow ablation panel differs")
    report = _paired_promotion_report(
        "unconditioned_order_flow",
        "state_conditioned_order_flow",
        incumbent_fit.ensemble_run_losses,
        challenger_fit.ensemble_run_losses,
        minimum_improvement=minimum_mean_loss_improvement,
        required_paired_run_count=required_paired_run_count,
        incumbent_group_losses=incumbent_fit.ensemble_group_losses,
        challenger_group_losses=challenger_fit.ensemble_group_losses,
    )
    winner = challenger_fit if report["promoted"] is True else incumbent_fit
    return winner, report


def _candidate_fit_report(fit: _CandidateFit) -> dict[str, object]:
    return {
        "candidate_id": fit.candidate_id,
        "feature_view": fit.feature_view,
        "state_conditioned_flow": fit.state_conditioned_flow,
        "parameter_count_per_peer": fit.parameter_count_per_peer,
        "peer_count": len(fit.peer_states),
        "peer_reports": list(fit.peer_reports),
        "ensemble_tuning_metrics": fit.ensemble_metrics,
        "ensemble_tuning_run_losses": list(fit.ensemble_run_losses),
        "ensemble_tuning_run_symbol_horizon_losses": fit.ensemble_group_losses,
        "ensemble_prediction_sha256": fit.ensemble_prediction_sha256,
    }


def _validated_candidate_fit_report(
    report: Mapping[str, object],
    *,
    panel_key: str,
    expected_candidate_id: str,
    expected_feature_view: str,
    expected_state_conditioned_flow: bool,
    seeds: Sequence[int],
    tuning_run_count: int,
    execution_mode: str,
) -> tuple[Mapping[str, object], tuple[float, ...], dict[str, float], int]:
    expected_metric_names = {
        "payoff_pinball",
        "maximum_adverse_excursion_pinball",
        "positive_bce",
        "adverse_bce",
        "unpredictability_bce",
        "loss",
        "run_balanced_loss",
        "worst_run_loss",
        "run_count",
        "eligible_action_targets",
        "eligible_regime_targets",
        "fully_censored_minibatches",
        "fully_censored_rows",
    }
    metrics = report.get("ensemble_tuning_metrics")
    raw_run_losses = report.get("ensemble_tuning_run_losses")
    raw_group_losses = report.get("ensemble_tuning_run_symbol_horizon_losses")
    peers = report.get("peer_reports")
    parameter_count = report.get("parameter_count_per_peer")
    if (
        set(report)
        != {
            "candidate_id",
            "feature_view",
            "state_conditioned_flow",
            "parameter_count_per_peer",
            "peer_count",
            "peer_reports",
            "ensemble_tuning_metrics",
            "ensemble_tuning_run_losses",
            "ensemble_tuning_run_symbol_horizon_losses",
            "ensemble_prediction_sha256",
        }
        or panel_key not in {expected_candidate_id, expected_feature_view}
        or report.get("candidate_id") != expected_candidate_id
        or report.get("feature_view") != expected_feature_view
        or report.get("state_conditioned_flow") is not expected_state_conditioned_flow
        or isinstance(parameter_count, bool)
        or not isinstance(parameter_count, int)
        or parameter_count <= 0
        or report.get("peer_count") != len(seeds)
        or not isinstance(peers, list)
        or len(peers) != len(seeds)
        or [peer.get("seed") for peer in peers if isinstance(peer, Mapping)]
        != list(seeds)
        or any(
            not isinstance(peer, Mapping)
            or peer.get("state_conditioned_flow") is not expected_state_conditioned_flow
            for peer in peers
        )
        or not isinstance(metrics, Mapping)
        or set(metrics) != expected_metric_names
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in metrics.values()
        )
        or float(metrics.get("run_count", 0.0)) != float(tuning_run_count)
        or float(metrics.get("worst_run_loss", math.inf))
        < float(metrics.get("run_balanced_loss", -math.inf))
        or not isinstance(raw_run_losses, list)
        or len(raw_run_losses) != tuning_run_count
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in raw_run_losses
        )
        or not isinstance(raw_group_losses, Mapping)
        or not raw_group_losses
        or any(
            not isinstance(key, str)
            or len(key.split(":")) != 3
            or key.split(":")[1] not in IMPACT_CAPTURE_SYMBOLS
            or key.split(":")[2]
            not in {str(value) for value in ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS}
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for key, value in raw_group_losses.items()
        )
        or (
            execution_mode != "preflight"
            and len(raw_group_losses)
            != tuning_run_count
            * len(IMPACT_CAPTURE_SYMBOLS)
            * len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
        )
        or not math.isclose(
            sum(float(value) for value in raw_run_losses) / len(raw_run_losses),
            float(metrics.get("run_balanced_loss", math.inf)),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or not math.isclose(
            max(float(value) for value in raw_run_losses),
            float(metrics.get("worst_run_loss", math.inf)),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or _SHA256.fullmatch(str(report.get("ensemble_prediction_sha256", ""))) is None
    ):
        raise ValueError("Round 74 pretest candidate report differs")
    assert isinstance(metrics, Mapping)
    assert isinstance(raw_run_losses, list)
    assert isinstance(raw_group_losses, Mapping)
    return (
        metrics,
        tuple(float(value) for value in raw_run_losses),
        {str(key): float(value) for key, value in raw_group_losses.items()},
        int(parameter_count),
    )


def _flatten_ensemble_state(
    fit: _CandidateFit,
) -> dict[str, torch.Tensor]:
    return {
        f"peers.{peer_index}.{name}": tensor.contiguous()
        for peer_index, state in enumerate(fit.peer_states)
        for name, tensor in sorted(state.items())
    }


def _load_ensemble_from_bytes(
    payload: bytes,
    *,
    candidate_id: str,
    feature_view: str,
    state_conditioned_flow: bool,
    peer_count: int,
) -> Round74EventEnsemble:
    state = load_safetensors(payload)
    ensemble = Round74EventEnsemble(
        candidate_id,
        peer_count,
        feature_view=feature_view,
        state_conditioned_flow=state_conditioned_flow,
    )
    incompatible = ensemble.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("Round 74 pretest model tensor names differ")
    return ensemble


def _runtime_source_binding() -> dict[str, str]:
    module_directory = Path(__file__).parent
    modules = {
        "sequence": "impact_absorption_event_sequence.py",
        "features": "impact_absorption_event_features.py",
        "scaling": "impact_absorption_event_scaling.py",
        "targets": "impact_absorption_event_targets.py",
        "dataset": "impact_absorption_event_dataset.py",
        "model": "impact_absorption_event_model.py",
        "pretraining": "impact_absorption_event_pretraining.py",
        "training": "impact_absorption_event_training.py",
        "model_operator": "round74_event_model_operator.py",
        "segmented_model_operator": "round74_segmented_model_operator.py",
        "storage": "storage.py",
    }
    return {
        f"{name}_module_sha256": _sha256_bytes(
            (module_directory / filename).read_bytes()
        )
        for name, filename in modules.items()
    }


def train_and_seal_round74_pretest_policy(
    training_batches: Sequence[Round74EventTrainingBatch],
    tuning_batches: Sequence[Round74EventTrainingBatch],
    *,
    output_directory: str | Path,
    compute_backend: str = "auto",
    config: Round74EventTrainingConfig | None = None,
    representative_window_policy_sha256: str | None = None,
    matched_preparation_sha256: str | None = None,
    feature_scaler: Round74EventFeatureScaler | None = None,
    selection_protocol: Round74EventSelectionProtocol | None = None,
) -> Round74PretestPolicyArtifact:
    """Train the declared panel and publish one reload-verified pretest policy."""

    selected_config = config or Round74EventTrainingConfig()
    selected_config.validate()
    if selected_config.execution_mode in {"cohort", "segmented_cohort"}:
        if selected_config.execution_mode == "cohort" and (
            len(training_batches) != ROUND74_EVENT_TRAINING_REQUIRED_CAPTURE_RUNS
            or len(tuning_batches) != ROUND74_COMPLEXITY_PROMOTION_REQUIRED_TUNING_RUNS
        ):
            raise ValueError(
                "Round 74 cohort training requires exactly 120 training and "
                "12 model-selection capture runs from the representative "
                "window policy"
            )
        window_policy_kind, matched_preparation = _cohort_window_policy_identity(
            representative_window_policy_sha256,
            matched_preparation_sha256,
        )
        if selected_config.execution_mode == "segmented_cohort" and (
            window_policy_kind != "segmented_duration_normalized"
            or matched_preparation is not None
        ):
            raise ValueError(
                "Round 74 segmented training requires its duration-normalized "
                "window policy"
            )
    elif (
        representative_window_policy_sha256 is not None
        or matched_preparation_sha256 is not None
    ):
        raise ValueError(
            "Round 74 preflight training cannot claim representative cohort windows"
        )
    else:
        window_policy_kind = "preflight"
        matched_preparation = None
    development = _validate_development_batches(
        training_batches,
        tuning_batches,
        minimum_rows=selected_config.minimum_role_rows,
    )
    if selection_protocol is None:
        optimization_training_batches = tuple(training_batches)
        early_stopping_batches = tuple(tuning_batches)
        promotion_stage_batches = {
            stage_id: tuple(tuning_batches)
            for stage_id in ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS
        }
        selection_protocol_payload: dict[str, object] = {
            "schema_version": ROUND74_EVENT_SELECTION_PROTOCOL_SCHEMA_VERSION,
            "mode": "legacy_shared_tuning_panel",
            "protocol_sha256": None,
            "stage_partition": None,
            "training_only_early_stopping": False,
            "disjoint_promotion_stages": False,
            "eligible_for_segmented_cohort_policy": False,
        }
    else:
        selection_protocol.validate()
        if (
            selected_config.execution_mode != "segmented_cohort"
            or tuple(
                (
                    *selection_protocol.optimization_batches,
                    *selection_protocol.purged_training_batches,
                    *selection_protocol.early_stopping_batches,
                )
            )
            != tuple(training_batches)
            or tuple(
                batch
                for group in selection_protocol.promotion_stage_batches
                for batch in group
            )
            != tuple(tuning_batches)
        ):
            raise ValueError("Round 74 model-selection protocol binding differs")
        optimization_training_batches = selection_protocol.optimization_batches
        early_stopping_batches = selection_protocol.early_stopping_batches
        promotion_stage_batches = dict(
            zip(
                ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS,
                selection_protocol.promotion_stage_batches,
                strict=True,
            )
        )
        selection_protocol_payload = {
            "schema_version": ROUND74_EVENT_SELECTION_PROTOCOL_SCHEMA_VERSION,
            "mode": "isolated_chronological_panels",
            "protocol_sha256": selection_protocol.protocol_sha256,
            "protocol": selection_protocol.as_dict(),
            "stage_partition": selection_protocol.stage_partition.as_dict(),
            "training_only_early_stopping": True,
            "disjoint_promotion_stages": True,
            "eligible_for_segmented_cohort_policy": True,
        }
    if feature_scaler is None:
        if selected_config.execution_mode != "preflight":
            raise ValueError(
                "Round 74 cohort training requires its fitted feature scaler"
            )
        scaler_payload = b""
        scaler_artifact: dict[str, object] = {
            "available": False,
            "reason": "preflight_only",
            "filename": None,
            "scaler_sha256": None,
            "file_sha256": None,
            "byte_count": 0,
            "media_type": None,
            "fit_partition_role": None,
            "fit_source_scope": None,
            "fit_source_run_count": 0,
            "fit_source_selection_sha256": None,
            "reload_verified": False,
        }
    else:
        if (
            not isinstance(feature_scaler, Round74EventFeatureScaler)
            or feature_scaler.scaler_sha256 != development.scaler_sha256
        ):
            raise ValueError("Round 74 fitted feature scaler identity differs")
        if selection_protocol is not None and (
            feature_scaler.fit_source_scope != "segmented_optimization_training_runs"
            or feature_scaler.fit_source_run_ids
            != tuple(
                _batch_run_id(batch)
                for batch in selection_protocol.optimization_batches
            )
            or feature_scaler.fit_source_partition_sha256
            != development.partition_sha256
            or feature_scaler.fit_source_selection_sha256
            != selection_protocol.scaler_fit_selection_sha256
        ):
            raise ValueError("Round 74 segmented scaler binding differs")
        if (
            selected_config.execution_mode == "cohort"
            and feature_scaler.fit_source_scope != "training_partition_all_runs"
        ):
            raise ValueError("Round 74 cohort scaler provenance differs")
        scaler_payload = _canonical_json_bytes(feature_scaler.as_dict()) + b"\n"
        scaler_file_sha256 = _sha256_bytes(scaler_payload)
        scaler_filename = f"round74-feature-scaler-{feature_scaler.scaler_sha256}.json"
        reloaded_scaler = _load_scaler_bytes(scaler_payload)
        if reloaded_scaler.as_dict() != feature_scaler.as_dict():
            raise RuntimeError("Round 74 feature scaler reload differs")
        scaler_artifact = {
            "available": True,
            "reason": "training_only_scaler_persisted",
            "filename": scaler_filename,
            "scaler_sha256": feature_scaler.scaler_sha256,
            "file_sha256": scaler_file_sha256,
            "byte_count": len(scaler_payload),
            "media_type": "application/json",
            "fit_partition_role": "training",
            "fit_source_scope": feature_scaler.fit_source_scope,
            "fit_source_run_count": len(feature_scaler.fit_source_run_ids),
            "fit_source_selection_sha256": (
                feature_scaler.fit_source_selection_sha256 or None
            ),
            "reload_verified": True,
        }
    target_loss_scale = fit_round74_event_target_loss_scale(
        optimization_training_batches,
        require_complete_panel=selected_config.execution_mode != "preflight",
    )
    if target_loss_scale.training_batch_sha256 != tuple(
        sorted(batch.batch_sha256 for batch in optimization_training_batches)
    ):
        raise RuntimeError("Round 74 target-loss scale data binding differs")
    backend = require_backend(resolve_backend(compute_backend))
    device = torch_device_for_backend(backend)
    prior_deterministic = torch.are_deterministic_algorithms_enabled()
    prior_python_random = random.getstate()
    prior_numpy_random = np.random.get_state()
    prior_torch_random = torch.get_rng_state()
    candidate_fits: list[_CandidateFit] = []
    warning_messages: list[str] = []
    try:
        torch.use_deterministic_algorithms(True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for candidate_id in selected_config.candidate_ids:
                candidate_fits.append(
                    _fit_candidate(
                        candidate_id,
                        "market_state_clock_neutral",
                        optimization_training_batches,
                        early_stopping_batches,
                        promotion_stage_batches["architecture"],
                        config=selected_config,
                        device=device,
                        target_loss_scale=target_loss_scale,
                    )
                )
            warning_messages.extend(str(item.message) for item in caught)
    finally:
        torch.use_deterministic_algorithms(prior_deterministic)
        random.setstate(prior_python_random)
        np.random.set_state(prior_numpy_random)
        torch.set_rng_state(prior_torch_random)
    if selected_config.architecture_selection_mode == "complexity_gate":
        promotion_required_runs = (
            len(promotion_stage_batches["architecture"])
            if selected_config.execution_mode == "segmented_cohort"
            else ROUND74_COMPLEXITY_PROMOTION_REQUIRED_TUNING_RUNS
        )
        architecture_winner, promotion_reports = _select_candidate_with_complexity_gate(
            candidate_fits,
            minimum_mean_loss_improvement=(selected_config.minimum_tuning_improvement),
            required_paired_run_count=promotion_required_runs,
        )
        selection_criterion = (
            "sequential parameter-count complexity promotion; each challenger "
            f"requires all {promotion_required_runs} model-selection capture runs, "
            "must strictly exceed the numerical mean-loss floor after deleting "
            "any one capture run, must materially win a strict majority of capture "
            "runs, and may not degrade any paired run or eligible "
            "run-symbol-horizon proper-loss subgroup beyond that floor"
        )
        planned_comparison_count = ROUND74_COMPLEXITY_PROMOTION_COMPARISON_COUNT
        required_paired_capture_run_count = promotion_required_runs
        selection_improvement = selected_config.minimum_tuning_improvement
    else:
        if len(candidate_fits) != 1:
            raise RuntimeError("Round 74 fixed architecture panel differs")
        architecture_winner = candidate_fits[0]
        promotion_reports = ()
        selection_criterion = (
            "fixed baseline-selected architecture; no architecture search or "
            "complexity promotion performed"
        )
        planned_comparison_count = 0
        required_paired_capture_run_count = 0
        selection_improvement = 0.0
    try:
        torch.use_deterministic_algorithms(True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clock_incumbent_fit = _fit_candidate(
                architecture_winner.candidate_id,
                "market_state_clock_neutral",
                optimization_training_batches,
                early_stopping_batches,
                promotion_stage_batches["clock_features"],
                config=selected_config,
                device=device,
                target_loss_scale=target_loss_scale,
            )
            clock_feature_fit = _fit_candidate(
                architecture_winner.candidate_id,
                "market_state_with_clock",
                optimization_training_batches,
                early_stopping_batches,
                promotion_stage_batches["clock_features"],
                config=selected_config,
                device=device,
                target_loss_scale=target_loss_scale,
            )
            warning_messages.extend(str(item.message) for item in caught)
    finally:
        torch.use_deterministic_algorithms(prior_deterministic)
        random.setstate(prior_python_random)
        np.random.set_state(prior_numpy_random)
        torch.set_rng_state(prior_torch_random)
    feature_view_required_runs = (
        len(promotion_stage_batches["clock_features"])
        if selected_config.execution_mode == "segmented_cohort"
        else ROUND74_COMPLEXITY_PROMOTION_REQUIRED_TUNING_RUNS
    )
    clock_winner, clock_promotion_report = _select_feature_view_with_ablation_gate(
        clock_incumbent_fit,
        clock_feature_fit,
        minimum_mean_loss_improvement=(selected_config.minimum_tuning_improvement),
        required_paired_run_count=feature_view_required_runs,
    )
    order_flow_challenger_view = _order_flow_challenger_feature_view(
        clock_winner.feature_view
    )
    order_flow_required_runs = (
        len(promotion_stage_batches["order_flow_features"])
        if selected_config.execution_mode == "segmented_cohort"
        else feature_view_required_runs
    )
    try:
        torch.use_deterministic_algorithms(True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            order_flow_incumbent_fit = _fit_candidate(
                architecture_winner.candidate_id,
                clock_winner.feature_view,
                optimization_training_batches,
                early_stopping_batches,
                promotion_stage_batches["order_flow_features"],
                config=selected_config,
                device=device,
                target_loss_scale=target_loss_scale,
            )
            order_flow_fit = _fit_candidate(
                architecture_winner.candidate_id,
                order_flow_challenger_view,
                optimization_training_batches,
                early_stopping_batches,
                promotion_stage_batches["order_flow_features"],
                config=selected_config,
                device=device,
                target_loss_scale=target_loss_scale,
            )
            warning_messages.extend(str(item.message) for item in caught)
    finally:
        torch.use_deterministic_algorithms(prior_deterministic)
        random.setstate(prior_python_random)
        np.random.set_state(prior_numpy_random)
        torch.set_rng_state(prior_torch_random)
    feature_winner, order_flow_promotion_report = (
        _select_feature_view_with_ablation_gate(
            order_flow_incumbent_fit,
            order_flow_fit,
            minimum_mean_loss_improvement=(selected_config.minimum_tuning_improvement),
            required_paired_run_count=order_flow_required_runs,
        )
    )
    feature_view_fits = {
        "clock_features": (clock_incumbent_fit, clock_feature_fit),
        "order_flow_features": (order_flow_incumbent_fit, order_flow_fit),
    }
    interaction_evaluated = _feature_view_contains_order_flow(
        feature_winner.feature_view
    )
    interaction_required_runs = (
        len(promotion_stage_batches["state_conditioned_flow"])
        if selected_config.execution_mode == "segmented_cohort"
        else feature_view_required_runs
    )
    if interaction_evaluated:
        try:
            torch.use_deterministic_algorithms(True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                state_conditioned_flow_incumbent_fit = _fit_candidate(
                    feature_winner.candidate_id,
                    feature_winner.feature_view,
                    optimization_training_batches,
                    early_stopping_batches,
                    promotion_stage_batches["state_conditioned_flow"],
                    config=selected_config,
                    device=device,
                    target_loss_scale=target_loss_scale,
                )
                state_conditioned_flow_fit = _fit_candidate(
                    feature_winner.candidate_id,
                    feature_winner.feature_view,
                    optimization_training_batches,
                    early_stopping_batches,
                    promotion_stage_batches["state_conditioned_flow"],
                    config=selected_config,
                    device=device,
                    target_loss_scale=target_loss_scale,
                    state_conditioned_flow=True,
                )
                warning_messages.extend(str(item.message) for item in caught)
        finally:
            torch.use_deterministic_algorithms(prior_deterministic)
            random.setstate(prior_python_random)
            np.random.set_state(prior_numpy_random)
            torch.set_rng_state(prior_torch_random)
        random_initialization_fit, interaction_promotion_report = (
            _select_state_conditioned_flow_with_ablation_gate(
                state_conditioned_flow_incumbent_fit,
                state_conditioned_flow_fit,
                minimum_mean_loss_improvement=(
                    selected_config.minimum_tuning_improvement
                ),
                required_paired_run_count=interaction_required_runs,
            )
        )
        interaction_panel = {
            "unconditioned_order_flow": _candidate_fit_report(
                state_conditioned_flow_incumbent_fit
            ),
            "state_conditioned_order_flow": _candidate_fit_report(
                state_conditioned_flow_fit
            ),
        }
        interaction_reason = "paired_proper_loss_gate_completed"
    else:
        random_initialization_fit = feature_winner
        interaction_promotion_report = None
        interaction_panel = {
            "unconditioned_order_flow": _candidate_fit_report(feature_winner),
        }
        interaction_reason = "order_flow_layer_not_selected"
    pretraining_supported = (
        round74_event_model_pretraining_channels(
            build_round74_event_model(
                random_initialization_fit.candidate_id,
                state_conditioned_flow=(
                    random_initialization_fit.state_conditioned_flow
                ),
            )
        )
        is not None
    )
    pretraining_evaluated = (
        pretraining_supported and selected_config.execution_mode != "preflight"
    )
    if pretraining_evaluated:
        try:
            torch.use_deterministic_algorithms(True)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                random_initialization_incumbent_fit = _fit_candidate(
                    random_initialization_fit.candidate_id,
                    random_initialization_fit.feature_view,
                    optimization_training_batches,
                    early_stopping_batches,
                    promotion_stage_batches["causal_pretraining"],
                    config=selected_config,
                    device=device,
                    target_loss_scale=target_loss_scale,
                    state_conditioned_flow=(
                        random_initialization_fit.state_conditioned_flow
                    ),
                )
                pretrained_initialization_fit = _fit_candidate(
                    random_initialization_fit.candidate_id,
                    random_initialization_fit.feature_view,
                    optimization_training_batches,
                    early_stopping_batches,
                    promotion_stage_batches["causal_pretraining"],
                    config=selected_config,
                    device=device,
                    target_loss_scale=target_loss_scale,
                    initialization_id="causal_next_event_pretrained",
                    state_conditioned_flow=(
                        random_initialization_fit.state_conditioned_flow
                    ),
                )
                warning_messages.extend(str(item.message) for item in caught)
        finally:
            torch.use_deterministic_algorithms(prior_deterministic)
            random.setstate(prior_python_random)
            np.random.set_state(prior_numpy_random)
            torch.set_rng_state(prior_torch_random)
        initialization_required_runs = (
            len(promotion_stage_batches["causal_pretraining"])
            if selected_config.execution_mode == "segmented_cohort"
            else feature_view_required_runs
        )
        initialization_promotion_report = _paired_promotion_report(
            "random",
            "causal_next_event_pretrained",
            random_initialization_incumbent_fit.ensemble_run_losses,
            pretrained_initialization_fit.ensemble_run_losses,
            minimum_improvement=selected_config.minimum_tuning_improvement,
            required_paired_run_count=initialization_required_runs,
            incumbent_group_losses=(
                random_initialization_incumbent_fit.ensemble_group_losses
            ),
            challenger_group_losses=(
                pretrained_initialization_fit.ensemble_group_losses
            ),
        )
        winner = (
            pretrained_initialization_fit
            if initialization_promotion_report["promoted"] is True
            else random_initialization_incumbent_fit
        )
        initialization_panel = {
            "random": _candidate_fit_report(random_initialization_incumbent_fit),
            "causal_next_event_pretrained": _candidate_fit_report(
                pretrained_initialization_fit
            ),
        }
        selected_initialization_id = (
            "causal_next_event_pretrained"
            if winner is pretrained_initialization_fit
            else "random"
        )
        initialization_reason = "paired_proper_loss_gate_completed"
    else:
        winner = random_initialization_fit
        initialization_required_runs = 0
        initialization_promotion_report = None
        initialization_panel = {
            "random": _candidate_fit_report(random_initialization_fit),
        }
        selected_initialization_id = "random"
        initialization_reason = (
            "preflight_never_evaluates_pretraining"
            if selected_config.execution_mode == "preflight"
            else "selected_pooled_control_has_no_causal_event_encoder"
        )
    fallback_messages = [
        message
        for message in warning_messages
        if "not currently supported on the DML backend" in message
        or "fall back to run on the CPU" in message
    ]
    if fallback_messages:
        raise RuntimeError(
            f"Round 74 event training used CPU fallback: {fallback_messages}"
        )
    model_state = _flatten_ensemble_state(winner)
    # safetensors 0.8 metadata uses map ordering that is not byte-stable.
    # Identity belongs in the hash-bound policy; sorted tensors stay stable.
    model_bytes = save_safetensors(model_state)
    if model_bytes != save_safetensors(model_state):
        raise RuntimeError("Round 74 safetensors serialization is not stable")
    model_sha256 = _sha256_bytes(model_bytes)
    loaded = _load_ensemble_from_bytes(
        model_bytes,
        candidate_id=winner.candidate_id,
        feature_view=winner.feature_view,
        state_conditioned_flow=winner.state_conditioned_flow,
        peer_count=len(winner.peer_states),
    ).to(device)
    loaded_state = _cpu_state(loaded)
    if set(loaded_state) != set(model_state) or any(
        not torch.equal(loaded_state[name], model_state[name]) for name in model_state
    ):
        raise RuntimeError("Round 74 safetensors state reload differs")
    reload_prediction_sha256 = _prediction_sha256(
        loaded,
        early_stopping_batches[0],
        maximum_rows=selected_config.minibatch_rows,
        device=device,
    )
    if reload_prediction_sha256 != winner.ensemble_prediction_sha256:
        raise RuntimeError("Round 74 pretest prediction reload differs")
    model_filename = f"round74-{winner.candidate_id}-{model_sha256}.safetensors"
    policy: dict[str, object] = {
        "schema_version": ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION,
        "training_schema_version": ROUND74_EVENT_TRAINING_SCHEMA_VERSION,
        "model_schema_version": ROUND74_EVENT_MODEL_SCHEMA_VERSION,
        "source_binding": _runtime_source_binding(),
        "development_data": {
            "partition_sha256": development.partition_sha256,
            "scaler_sha256": development.scaler_sha256,
            "window_representation": development.window_representation,
            "target_context_panel_schema_version": (
                ROUND74_EVENT_TARGET_CONTEXT_PANEL_SCHEMA_VERSION
            ),
            "target_context_panel_sha256": (development.target_context_panel_sha256),
            "target_context_sha256": list(development.target_context_sha256),
            "target_context_count": len(development.target_context_sha256),
            "training_batch_sha256": list(development.training_batch_sha256),
            "tuning_batch_sha256": list(development.tuning_batch_sha256),
            "representative_window_policy_sha256": (
                representative_window_policy_sha256
            ),
            "representative_window_policy_kind": window_policy_kind,
            "matched_preparation_sha256": matched_preparation,
            "representative_window_policy_applied": (
                selected_config.execution_mode != "preflight"
            ),
            "training_rows": development.training_rows,
            "tuning_rows": development.tuning_rows,
            "training_first_wall_ns": development.training_first_wall_ns,
            "training_last_wall_ns": development.training_last_wall_ns,
            "tuning_first_wall_ns": development.tuning_first_wall_ns,
            "tuning_last_wall_ns": development.tuning_last_wall_ns,
            "role_transition_gap_ns": (
                development.tuning_first_wall_ns - development.training_last_wall_ns
            ),
            "test_batches_consumed": 0,
            "test_sample_digests_consumed": 0,
        },
        "training_policy": selected_config.as_dict(),
        "target_loss_scale": target_loss_scale.as_dict(),
        "optimization_population": _optimization_population_policy(
            selected_config.execution_mode,
            len(optimization_training_batches),
            len(tuning_batches),
        ),
        "selection_protocol": selection_protocol_payload,
        "ensemble_aggregation": {
            "peer_weights": "equal",
            "payoff_and_adverse_excursion_quantiles": (
                "arithmetic_mean_of_peer_quantiles"
            ),
            "classification_heads": (
                "arithmetic_mean_of_peer_probabilities_then_logit"
            ),
            "mean_peer_logits_permitted": False,
            "probability_calibration_fitted_after_ensemble_aggregation": True,
        },
        "backend": {
            "requested": backend.requested,
            "kind": backend.kind,
            "device": str(device),
            "vendor": backend.vendor,
            "selection": backend.selection,
            "accelerated": backend.accelerated,
            "torch_version": str(torch.__version__),
            "torch_directml_version": _package_version("torch-directml"),
            "safetensors_version": _package_version("safetensors"),
            "deterministic_algorithms_requested": True,
            "cross_platform_bitwise_reproducibility_claim": False,
            "cpu_fallback_warning_count": 0,
            "warning_count": len(warning_messages),
        },
        "candidate_panel": {
            fit.candidate_id: _candidate_fit_report(fit) for fit in candidate_fits
        },
        "feature_view_panel": {
            stage_id: {
                "incumbent": _candidate_fit_report(fits[0]),
                "challenger": _candidate_fit_report(fits[1]),
            }
            for stage_id, fits in feature_view_fits.items()
        },
        "selection": {
            "architecture_selection_mode": (
                selected_config.architecture_selection_mode
            ),
            "criterion": selection_criterion,
            "selected_candidate_id": architecture_winner.candidate_id,
            "selected_tuning_metrics": architecture_winner.ensemble_metrics,
            "ordered_candidate_ids": list(selected_config.candidate_ids),
            "planned_comparison_count": planned_comparison_count,
            "required_paired_capture_run_count": required_paired_capture_run_count,
            "minimum_mean_proper_loss_improvement": selection_improvement,
            "maximum_permitted_paired_run_loss_degradation": selection_improvement,
            "statistical_independence_or_significance_claim": False,
            "promotion_reports": list(promotion_reports),
            "complexity_promotion_privilege": False,
            "backtest_metric_used_for_selection": False,
        },
        "feature_view_selection": {
            "schema_version": ROUND74_EVENT_FEATURE_VIEW_SCHEMA_VERSION,
            "criterion": _feature_view_selection_criterion(
                feature_view_required_runs,
                order_flow_required_runs,
            ),
            "market_state_feature_names": list(
                ROUND74_EVENT_MARKET_STATE_FEATURE_NAMES
            ),
            "market_state_feature_names_sha256": (
                ROUND74_EVENT_MARKET_STATE_FEATURE_NAMES_SHA256
            ),
            "order_flow_feature_names": list(ROUND74_EVENT_ORDER_FLOW_FEATURE_NAMES),
            "order_flow_feature_names_sha256": (
                ROUND74_EVENT_ORDER_FLOW_FEATURE_NAMES_SHA256
            ),
            "clock_feature_names": list(ROUND74_EVENT_CLOCK_FEATURE_NAMES),
            "clock_feature_names_sha256": (ROUND74_EVENT_CLOCK_FEATURE_NAMES_SHA256),
            "supported_feature_views": list(ROUND74_EVENT_FEATURE_VIEWS),
            "evaluated_feature_views": list(
                dict.fromkeys(
                    fit.feature_view
                    for fits in feature_view_fits.values()
                    for fit in fits
                )
            ),
            "selected_feature_view": feature_winner.feature_view,
            "selected_candidate_id": feature_winner.candidate_id,
            "selected_tuning_metrics": feature_winner.ensemble_metrics,
            "clock_required_paired_capture_run_count": (feature_view_required_runs),
            "order_flow_required_paired_capture_run_count": (order_flow_required_runs),
            "minimum_mean_proper_loss_improvement": (
                selected_config.minimum_tuning_improvement
            ),
            "maximum_permitted_paired_run_loss_degradation": (
                selected_config.minimum_tuning_improvement
            ),
            "clock_promotion_report": clock_promotion_report,
            "order_flow_promotion_report": order_flow_promotion_report,
            "architecture_fixed_before_layer_gates": True,
            "state_first_incumbent": "market_state_clock_neutral",
            "clock_default_on_gate_failure": "market_state_clock_neutral",
            "order_flow_default_on_gate_failure": clock_winner.feature_view,
            "statistical_independence_or_significance_claim": False,
            "backtest_metric_used_for_selection": False,
            "test_targets_used": False,
        },
        "state_conditioned_flow_panel": interaction_panel,
        "state_conditioned_flow_selection": {
            "schema_version": (ROUND74_EVENT_STATE_CONDITIONED_FLOW_SCHEMA_VERSION),
            "criterion": (
                "after the feature view is fixed, neutral-initialized "
                "market-state modulation of admitted order flow challenges "
                "the unconditioned incumbent on identical seeds and the "
                "supervised objective; promotion requires strict mean proper-"
                "loss improvement and no paired run or run-symbol-horizon "
                "subgroup degradation beyond the numerical floor"
            ),
            "reason": interaction_reason,
            "evaluated": interaction_evaluated,
            "selected_state_conditioned_flow": (
                random_initialization_fit.state_conditioned_flow
            ),
            "selected_candidate_id": random_initialization_fit.candidate_id,
            "selected_feature_view": random_initialization_fit.feature_view,
            "selected_tuning_metrics": (random_initialization_fit.ensemble_metrics),
            "required_paired_capture_run_count": (
                interaction_required_runs if interaction_evaluated else 0
            ),
            "minimum_mean_proper_loss_improvement": (
                selected_config.minimum_tuning_improvement
                if interaction_evaluated
                else 0.0
            ),
            "maximum_permitted_paired_run_loss_degradation": (
                selected_config.minimum_tuning_improvement
                if interaction_evaluated
                else 0.0
            ),
            "promotion_report": interaction_promotion_report,
            "feature_view_fixed_before_interaction_gate": True,
            "order_flow_required": True,
            "neutral_multiplier_at_initialization": 1.0,
            "unconditioned_default_on_gate_failure": True,
            "statistical_independence_or_significance_claim": False,
            "backtest_metric_used_for_selection": False,
            "test_targets_used": False,
        },
        "initialization_panel": initialization_panel,
        "initialization_selection": {
            "schema_version": ROUND74_EVENT_PRETRAINING_SCHEMA_VERSION,
            "criterion": (
                "causal next-event pretraining challenges the same random "
                "initialization on the fixed architecture, feature view, seeds, "
                "and supervised objective; promotion requires strict mean "
                "proper-loss improvement and no paired run or "
                "run-symbol-horizon subgroup degradation beyond the numerical "
                "floor"
            ),
            "reason": initialization_reason,
            "pretraining_supported": pretraining_supported,
            "pretraining_evaluated": pretraining_evaluated,
            "ordered_initialization_ids": list(
                ROUND74_EVENT_PRETRAINING_INITIALIZATION_IDS
            ),
            "selected_initialization_id": selected_initialization_id,
            "selected_candidate_id": winner.candidate_id,
            "selected_feature_view": winner.feature_view,
            "selected_state_conditioned_flow": winner.state_conditioned_flow,
            "selected_tuning_metrics": winner.ensemble_metrics,
            "required_paired_capture_run_count": initialization_required_runs,
            "minimum_mean_proper_loss_improvement": (
                selected_config.minimum_tuning_improvement
                if pretraining_evaluated
                else 0.0
            ),
            "maximum_permitted_paired_run_loss_degradation": (
                selected_config.minimum_tuning_improvement
                if pretraining_evaluated
                else 0.0
            ),
            "promotion_report": initialization_promotion_report,
            "training_features_only": True,
            "supervised_targets_used_by_pretraining": False,
            "tuning_features_used_by_pretraining": False,
            "tuning_targets_used_by_pretraining": False,
            "calibration_data_used_by_pretraining": False,
            "test_data_used_by_pretraining": False,
            "random_default_on_gate_failure": True,
            "statistical_independence_or_significance_claim": False,
            "backtest_metric_used_for_selection": False,
        },
        "model_artifact": {
            "filename": model_filename,
            "sha256": model_sha256,
            "byte_count": len(model_bytes),
            "media_type": "application/x-safetensors",
            "header_metadata_omitted_for_byte_stability": True,
            "pickle_permitted": False,
            "state_reload_verified": True,
            "prediction_reload_verified": True,
            "prediction_sha256": reload_prediction_sha256,
        },
        "scaler_artifact": scaler_artifact,
        "authority": {
            "development_training_completed": True,
            "chronological_tuning_completed": True,
            "pretest_policy_sealed": True,
            "sealed_test_evaluated": False,
            "representative_market_evidence_claim": False,
            "financial_edge_tested": False,
            "profitability_claim": False,
            "ai_uplift_claim": False,
            "paper_trading_authority": False,
            "testnet_trading_authority": False,
            "live_trading_authority": False,
        },
    }
    policy_sha256 = _canonical_sha256(policy)
    policy["policy_sha256"] = policy_sha256
    policy_bytes = _canonical_json_bytes(policy) + b"\n"
    output = Path(output_directory)
    model_path = output / model_filename
    policy_path = output / f"round74-pretest-policy-{policy_sha256}.json"
    _write_immutable_bytes(model_path, model_bytes)
    if feature_scaler is not None:
        _write_immutable_bytes(output / scaler_filename, scaler_payload)
    _write_immutable_bytes(policy_path, policy_bytes)
    verified_model, verified_policy = load_round74_pretest_policy(policy_path)
    if (
        verified_policy["policy_sha256"] != policy_sha256
        or verified_model.candidate_id != winner.candidate_id
        or verified_model.feature_view != winner.feature_view
    ):
        raise RuntimeError("Round 74 published pretest policy reload differs")
    return Round74PretestPolicyArtifact(
        policy_sha256=policy_sha256,
        policy_path=policy_path,
        model_sha256=model_sha256,
        model_path=model_path,
        selected_candidate_id=winner.candidate_id,
        selected_feature_view=winner.feature_view,
        tuning_loss=float(
            winner.ensemble_metrics[
                "loss"
                if selected_config.execution_mode == "segmented_cohort"
                else "run_balanced_loss"
            ]
        ),
    )


def train_and_seal_round74_pretest_policy_from_prepared_roles(
    training_batches: Sequence[Round74EventTrainingBatch],
    tuning_roles: object,
    *,
    output_directory: str | Path,
    compute_backend: str = "auto",
    config: Round74EventTrainingConfig | None = None,
    feature_scaler: Round74EventFeatureScaler,
    matched_preparation_sha256: str | None = None,
    segmented_training_split: Round74SegmentedTrainingSplit | None = None,
) -> Round74PretestPolicyArtifact:
    """Fit candidates with model-selection runs and no later tuning role."""

    from .round74_event_model_operator import Round74PreparedTuningRoles
    from .round74_event_model_operator import (
        round74_matched_representative_window_policy,
        round74_representative_window_policy,
    )
    from .round74_segmented_model_operator import round74_segmented_window_policy

    if not isinstance(tuning_roles, Round74PreparedTuningRoles):
        raise TypeError("Round 74 prepared tuning roles are required")
    tuning_roles.validate()
    selected_config = config or Round74EventTrainingConfig()
    selected_config.validate()
    if selected_config.execution_mode not in {"cohort", "segmented_cohort"}:
        raise ValueError(
            "Round 74 prepared tuning roles require a representative cohort mode"
        )
    if selected_config.execution_mode == "segmented_cohort":
        if matched_preparation_sha256 is not None:
            raise ValueError(
                "Round 74 segmented tuning roles reject matched preparation"
            )
        if not isinstance(
            segmented_training_split,
            Round74SegmentedTrainingSplit,
        ):
            raise TypeError("Round 74 segmented training split is required")
        segmented_training_split.validate()
        representative_policy_sha256 = str(
            round74_segmented_window_policy()["policy_sha256"]
        )
    elif segmented_training_split is not None:
        raise ValueError("Round 74 legacy cohort rejects segmented training split")
    elif matched_preparation_sha256 is None:
        representative_policy_sha256 = str(
            round74_representative_window_policy()["policy_sha256"]
        )
    else:
        representative_policy_sha256 = str(
            round74_matched_representative_window_policy()["policy_sha256"]
        )
    selection_protocol = (
        _build_segmented_selection_protocol(
            training_batches,
            tuning_roles,
            feature_scaler,
            segmented_training_split,
        )
        if selected_config.execution_mode == "segmented_cohort"
        else None
    )
    return train_and_seal_round74_pretest_policy(
        training_batches,
        tuning_roles.model_selection_batches,
        output_directory=output_directory,
        compute_backend=compute_backend,
        config=selected_config,
        representative_window_policy_sha256=representative_policy_sha256,
        matched_preparation_sha256=matched_preparation_sha256,
        feature_scaler=feature_scaler,
        selection_protocol=selection_protocol,
    )


def load_round74_pretest_policy(
    policy_path: str | Path,
) -> tuple[Round74EventEnsemble, dict[str, object]]:
    """Validate an immutable policy and load its safe tensor artifact on CPU."""

    selected_path = Path(policy_path)
    try:
        policy = json.loads(
            selected_path.read_text(encoding="ascii"),
            object_pairs_hook=_reject_duplicate_json_pairs,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 74 pretest policy could not be read") from exc
    if not isinstance(policy, dict):
        raise ValueError("Round 74 pretest policy root differs")
    claimed = str(policy.pop("policy_sha256", ""))
    if _SHA256.fullmatch(claimed) is None or claimed != _canonical_sha256(policy):
        raise ValueError("Round 74 pretest policy digest differs")
    policy["policy_sha256"] = claimed
    if selected_path.name != f"round74-pretest-policy-{claimed}.json":
        raise ValueError("Round 74 pretest policy filename differs")
    if (
        policy.get("schema_version") != ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION
        or policy.get("training_schema_version")
        != ROUND74_EVENT_TRAINING_SCHEMA_VERSION
        or policy.get("model_schema_version") != ROUND74_EVENT_MODEL_SCHEMA_VERSION
    ):
        raise ValueError("Round 74 pretest policy schema differs")
    if set(policy) != {
        "schema_version",
        "training_schema_version",
        "model_schema_version",
        "source_binding",
        "development_data",
        "training_policy",
        "target_loss_scale",
        "optimization_population",
        "selection_protocol",
        "ensemble_aggregation",
        "backend",
        "candidate_panel",
        "selection",
        "feature_view_panel",
        "feature_view_selection",
        "state_conditioned_flow_panel",
        "state_conditioned_flow_selection",
        "initialization_panel",
        "initialization_selection",
        "model_artifact",
        "scaler_artifact",
        "authority",
        "policy_sha256",
    }:
        raise ValueError("Round 74 pretest policy top-level contract differs")
    selection = policy.get("selection")
    artifact = policy.get("model_artifact")
    scaler_artifact = policy.get("scaler_artifact")
    development = policy.get("development_data")
    authority = policy.get("authority")
    training_policy = policy.get("training_policy")
    target_loss_scale_payload = policy.get("target_loss_scale")
    optimization_population = policy.get("optimization_population")
    selection_protocol_payload = policy.get("selection_protocol")
    ensemble_aggregation = policy.get("ensemble_aggregation")
    source_binding = policy.get("source_binding")
    candidate_panel = policy.get("candidate_panel")
    feature_view_panel = policy.get("feature_view_panel")
    feature_view_selection = policy.get("feature_view_selection")
    state_conditioned_flow_panel = policy.get("state_conditioned_flow_panel")
    state_conditioned_flow_selection = policy.get("state_conditioned_flow_selection")
    initialization_panel = policy.get("initialization_panel")
    initialization_selection = policy.get("initialization_selection")
    backend = policy.get("backend")
    if not all(
        isinstance(value, Mapping)
        for value in (
            selection,
            artifact,
            scaler_artifact,
            development,
            authority,
            training_policy,
            target_loss_scale_payload,
            optimization_population,
            selection_protocol_payload,
            ensemble_aggregation,
            source_binding,
            candidate_panel,
            feature_view_panel,
            feature_view_selection,
            state_conditioned_flow_panel,
            state_conditioned_flow_selection,
            initialization_panel,
            initialization_selection,
            backend,
        )
    ):
        raise ValueError("Round 74 pretest policy sections differ")
    assert isinstance(selection, Mapping)
    assert isinstance(artifact, Mapping)
    assert isinstance(scaler_artifact, Mapping)
    assert isinstance(development, Mapping)
    assert isinstance(authority, Mapping)
    assert isinstance(training_policy, Mapping)
    assert isinstance(target_loss_scale_payload, Mapping)
    assert isinstance(optimization_population, Mapping)
    assert isinstance(selection_protocol_payload, Mapping)
    assert isinstance(ensemble_aggregation, Mapping)
    assert isinstance(source_binding, Mapping)
    assert isinstance(candidate_panel, Mapping)
    assert isinstance(feature_view_panel, Mapping)
    assert isinstance(feature_view_selection, Mapping)
    assert isinstance(state_conditioned_flow_panel, Mapping)
    assert isinstance(state_conditioned_flow_selection, Mapping)
    assert isinstance(initialization_panel, Mapping)
    assert isinstance(initialization_selection, Mapping)
    assert isinstance(backend, Mapping)
    candidate_id = str(selection.get("selected_candidate_id", ""))
    selected_feature_view = str(feature_view_selection.get("selected_feature_view", ""))
    selected_state_conditioned_flow = state_conditioned_flow_selection.get(
        "selected_state_conditioned_flow"
    )
    seeds = training_policy.get("seeds")
    candidate_ids = training_policy.get("candidate_ids")
    filename = str(artifact.get("filename", ""))
    model_sha256 = str(artifact.get("sha256", ""))
    if (
        candidate_id not in ROUND74_EVENT_MODEL_CANDIDATES
        or not isinstance(seeds, list)
        or not seeds
        or any(
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            for seed in seeds
        )
        or not isinstance(candidate_ids, list)
        or not candidate_ids
        or any(not isinstance(value, str) for value in candidate_ids)
        or len(candidate_ids) != len(set(candidate_ids))
        or any(value not in ROUND74_EVENT_MODEL_CANDIDATES for value in candidate_ids)
        or set(candidate_panel) != set(candidate_ids)
        or candidate_id not in candidate_panel
        or set(feature_view_panel) != {"clock_features", "order_flow_features"}
        or selected_feature_view not in ROUND74_EVENT_FEATURE_VIEWS
        or not isinstance(selected_state_conditioned_flow, bool)
        or feature_view_selection.get("selected_candidate_id") != candidate_id
        or _SAFE_FILENAME.fullmatch(filename) is None
        or _SHA256.fullmatch(model_sha256) is None
        or filename != f"round74-{candidate_id}-{model_sha256}.safetensors"
        or artifact.get("pickle_permitted") is not False
        or artifact.get("state_reload_verified") is not True
        or artifact.get("prediction_reload_verified") is not True
        or development.get("test_batches_consumed") != 0
        or development.get("test_sample_digests_consumed") != 0
    ):
        raise ValueError("Round 74 pretest policy static contract differs")
    if dict(source_binding) != _runtime_source_binding():
        raise ValueError("Round 74 pretest policy source binding differs")
    pretraining_policy = training_policy.get("causal_pretraining")
    if not isinstance(pretraining_policy, Mapping):
        raise ValueError("Round 74 pretest training policy differs")
    try:
        reconstructed_config = Round74EventTrainingConfig(
            candidate_ids=tuple(str(value) for value in candidate_ids),
            seeds=tuple(int(value) for value in seeds),
            maximum_epochs=int(training_policy["maximum_epochs"]),
            early_stopping_patience=int(training_policy["early_stopping_patience"]),
            minimum_tuning_improvement=float(
                training_policy["minimum_tuning_improvement"]
            ),
            minibatch_rows=int(training_policy["minibatch_rows"]),
            learning_rate=float(training_policy["learning_rate"]),
            weight_decay=float(training_policy["weight_decay"]),
            gradient_clip_norm=float(training_policy["gradient_clip_norm"]),
            minimum_role_rows=int(training_policy["minimum_role_rows"]),
            device_run_group_size=int(training_policy["device_run_group_size"]),
            execution_mode=str(training_policy["execution_mode"]),
            architecture_selection_mode=str(
                training_policy["architecture_selection_mode"]
            ),
            pretraining=Round74EventPretrainingConfig.from_dict(pretraining_policy),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Round 74 pretest training policy differs") from exc
    if reconstructed_config.as_dict() != dict(training_policy):
        raise ValueError("Round 74 pretest training policy differs")
    try:
        target_loss_scale = Round74EventTargetLossScale.from_dict(
            target_loss_scale_payload
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Round 74 pretest target-loss scale differs") from exc
    training_batch_sha256 = development.get("training_batch_sha256")
    tuning_batch_sha256 = development.get("tuning_batch_sha256")
    raw_protocol = selection_protocol_payload.get("protocol")
    raw_optimization_hashes = (
        raw_protocol.get("optimization_batch_sha256")
        if isinstance(raw_protocol, Mapping)
        else None
    )
    optimization_run_count = (
        len(raw_optimization_hashes)
        if isinstance(raw_optimization_hashes, list)
        else len(training_batch_sha256)
        if isinstance(training_batch_sha256, list)
        else 0
    )
    if (
        not isinstance(training_batch_sha256, list)
        or not isinstance(tuning_batch_sha256, list)
        or dict(optimization_population)
        != _optimization_population_policy(
            reconstructed_config.execution_mode,
            optimization_run_count,
            len(tuning_batch_sha256),
        )
    ):
        raise ValueError("Round 74 pretest optimization population differs")
    legacy_selection_protocol = {
        "schema_version": ROUND74_EVENT_SELECTION_PROTOCOL_SCHEMA_VERSION,
        "mode": "legacy_shared_tuning_panel",
        "protocol_sha256": None,
        "stage_partition": None,
        "training_only_early_stopping": False,
        "disjoint_promotion_stages": False,
        "eligible_for_segmented_cohort_policy": False,
    }
    isolated_scaler_run_ids: tuple[str, ...] = ()
    isolated_scaler_selection_sha256: str | None = None
    if selection_protocol_payload.get("mode") == "isolated_chronological_panels":
        protocol = selection_protocol_payload.get("protocol")
        stage_partition_payload = selection_protocol_payload.get("stage_partition")
        if (
            set(selection_protocol_payload)
            != {
                "schema_version",
                "mode",
                "protocol_sha256",
                "protocol",
                "stage_partition",
                "training_only_early_stopping",
                "disjoint_promotion_stages",
                "eligible_for_segmented_cohort_policy",
            }
            or selection_protocol_payload.get("schema_version")
            != ROUND74_EVENT_SELECTION_PROTOCOL_SCHEMA_VERSION
            or not isinstance(protocol, Mapping)
            or not isinstance(stage_partition_payload, Mapping)
            or selection_protocol_payload.get("training_only_early_stopping")
            is not True
            or selection_protocol_payload.get("disjoint_promotion_stages") is not True
            or selection_protocol_payload.get("eligible_for_segmented_cohort_policy")
            is not True
        ):
            raise ValueError("Round 74 pretest selection protocol differs")
        try:
            stage_partition = Round74SegmentedModelSelectionStages.from_dict(
                stage_partition_payload
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Round 74 pretest selection stage partition differs"
            ) from exc
        protocol_without_sha = dict(protocol)
        protocol_sha256 = protocol_without_sha.pop("protocol_sha256", None)
        training_split_payload = protocol.get("training_split")
        if not isinstance(training_split_payload, Mapping):
            raise ValueError("Round 74 pretest training split differs")
        try:
            training_split = Round74SegmentedTrainingSplit.from_dict(
                training_split_payload
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Round 74 pretest training split differs") from exc
        promotion_hashes = protocol.get("promotion_stage_batch_sha256")
        optimization_hashes = protocol.get("optimization_batch_sha256")
        purged_hashes = protocol.get("purged_training_batch_sha256")
        early_stopping_hashes = protocol.get("early_stopping_batch_sha256")
        optimization_run_ids = protocol.get("optimization_run_ids")
        purged_run_ids = protocol.get("purged_training_run_ids")
        early_stopping_run_ids = protocol.get("early_stopping_run_ids")
        expected_protocol_keys = {
            "schema_version",
            "stage_partition_sha256",
            "training_split_sha256",
            "training_split",
            "stage_order",
            "training_role_assignment_basis",
            "early_stopping_fraction_denominator",
            "minimum_optimization_run_count",
            "minimum_early_stopping_run_count",
            "optimization_run_ids",
            "purged_training_run_ids",
            "early_stopping_run_ids",
            "optimization_batch_sha256",
            "purged_training_batch_sha256",
            "early_stopping_batch_sha256",
            "promotion_stage_batch_sha256",
            "chronological_gap_ns",
            "minimum_chronological_gap_ns",
            "target_loss_scale_fit_scope",
            "feature_scaler_fit_scope",
            "feature_scaler_fit_selection_sha256",
            "feature_scaler_fit_source_run_ids_sha256",
            "early_stopping_targets_used_for_gradient_updates",
            "promotion_targets_used_for_checkpoint_selection",
            "cross_stage_promotion_run_reuse_permitted",
            "calibration_or_policy_selection_run_included",
            "sealed_test_run_included",
            "protocol_sha256",
        }
        if (
            set(protocol) != expected_protocol_keys
            or not isinstance(protocol_sha256, str)
            or _SHA256.fullmatch(protocol_sha256) is None
            or protocol_sha256 != _canonical_sha256(protocol_without_sha)
            or selection_protocol_payload.get("protocol_sha256") != protocol_sha256
            or protocol.get("schema_version")
            != ROUND74_EVENT_SELECTION_PROTOCOL_SCHEMA_VERSION
            or protocol.get("stage_partition_sha256")
            != stage_partition.stage_partition_sha256
            or protocol.get("training_split_sha256") != training_split.split_sha256
            or training_split.parent_partition_sha256
            != stage_partition.parent_partition_sha256
            or training_split.cohort_plan_sha256 != stage_partition.cohort_plan_sha256
            or not isinstance(optimization_hashes, list)
            or not isinstance(purged_hashes, list)
            or not isinstance(early_stopping_hashes, list)
            or not isinstance(optimization_run_ids, list)
            or not isinstance(purged_run_ids, list)
            or not isinstance(early_stopping_run_ids, list)
            or len(optimization_run_ids) != len(optimization_hashes)
            or len(purged_run_ids) != len(purged_hashes)
            or len(early_stopping_run_ids) != len(early_stopping_hashes)
            or optimization_run_ids != list(training_split.optimization_run_ids)
            or purged_run_ids != list(training_split.purged_run_ids)
            or early_stopping_run_ids != list(training_split.early_stopping_run_ids)
            or any(
                not isinstance(run_id, str)
                or len(run_id) != 32
                or any(character not in "0123456789abcdef" for character in run_id)
                for run_id in (
                    *optimization_run_ids,
                    *purged_run_ids,
                    *early_stopping_run_ids,
                )
            )
            or len(
                {
                    *optimization_run_ids,
                    *purged_run_ids,
                    *early_stopping_run_ids,
                }
            )
            != len(optimization_run_ids)
            + len(purged_run_ids)
            + len(early_stopping_run_ids)
            or set(
                (
                    *optimization_run_ids,
                    *purged_run_ids,
                    *early_stopping_run_ids,
                )
            )
            & {
                run_id
                for run_ids in stage_partition.stage_run_ids
                for run_id in run_ids
            }
            or not isinstance(promotion_hashes, Mapping)
            or set(promotion_hashes) != set(ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS)
            or any(
                not isinstance(promotion_hashes[stage_id], list)
                or not promotion_hashes[stage_id]
                or any(
                    not isinstance(digest, str) or _SHA256.fullmatch(digest) is None
                    for digest in promotion_hashes[stage_id]
                )
                for stage_id in ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS
            )
            or any(
                len(promotion_hashes[stage_id])
                != len(stage_partition.stage_run_ids[index])
                for index, stage_id in enumerate(
                    ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS
                )
            )
            or len(
                {
                    digest
                    for stage_id in ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS
                    for digest in promotion_hashes[stage_id]
                }
            )
            != sum(
                len(promotion_hashes[stage_id])
                for stage_id in ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS
            )
            or [
                *optimization_hashes,
                *purged_hashes,
                *early_stopping_hashes,
            ]
            != training_batch_sha256
            or [
                digest
                for stage_id in ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS
                for digest in promotion_hashes[stage_id]
            ]
            != tuning_batch_sha256
            or protocol.get("stage_order")
            != list(ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS)
            or protocol.get("training_role_assignment_basis")
            != "chronological admitted-run order after transport adjudication"
            or protocol.get("early_stopping_fraction_denominator")
            != ROUND74_EVENT_SEGMENTED_EARLY_STOPPING_FRACTION_DENOMINATOR
            or protocol.get("minimum_optimization_run_count")
            != ROUND74_EVENT_SEGMENTED_MINIMUM_OPTIMIZATION_RUNS
            or protocol.get("minimum_early_stopping_run_count")
            != ROUND74_EVENT_SEGMENTED_MINIMUM_EARLY_STOPPING_RUNS
            or len(optimization_hashes)
            < ROUND74_EVENT_SEGMENTED_MINIMUM_OPTIMIZATION_RUNS
            or len(early_stopping_hashes)
            < ROUND74_EVENT_SEGMENTED_MINIMUM_EARLY_STOPPING_RUNS
            or isinstance(protocol.get("chronological_gap_ns"), bool)
            or not isinstance(protocol.get("chronological_gap_ns"), int)
            or int(protocol["chronological_gap_ns"])
            < ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS
            or protocol.get("minimum_chronological_gap_ns")
            != ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS
            or protocol.get("target_loss_scale_fit_scope")
            != "optimization_training_runs_only"
            or protocol.get("feature_scaler_fit_scope")
            != "segmented_optimization_training_runs_only"
            or _SHA256.fullmatch(
                str(protocol.get("feature_scaler_fit_selection_sha256"))
            )
            is None
            or protocol.get("feature_scaler_fit_selection_sha256")
            != training_split.split_sha256
            or protocol.get("feature_scaler_fit_source_run_ids_sha256")
            != _canonical_sha256(optimization_run_ids)
            or protocol.get("early_stopping_targets_used_for_gradient_updates")
            is not False
            or protocol.get("promotion_targets_used_for_checkpoint_selection")
            is not False
            or protocol.get("cross_stage_promotion_run_reuse_permitted") is not False
            or protocol.get("calibration_or_policy_selection_run_included") is not False
            or protocol.get("sealed_test_run_included") is not False
        ):
            raise ValueError("Round 74 pretest selection protocol differs")
        target_loss_scale_batch_sha256 = tuple(
            sorted(str(value) for value in optimization_hashes)
        )
        isolated_scaler_run_ids = tuple(str(value) for value in optimization_run_ids)
        isolated_scaler_selection_sha256 = str(
            protocol["feature_scaler_fit_selection_sha256"]
        )
        stage_tuning_run_counts = {
            stage_id: len(promotion_hashes[stage_id])
            for stage_id in ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS
        }
    else:
        if dict(selection_protocol_payload) != legacy_selection_protocol:
            raise ValueError("Round 74 pretest legacy selection protocol differs")
        target_loss_scale_batch_sha256 = tuple(
            sorted(str(value) for value in training_batch_sha256)
        )
        stage_tuning_run_counts = {
            stage_id: len(tuning_batch_sha256)
            for stage_id in ROUND74_SEGMENTED_MODEL_SELECTION_STAGE_IDS
        }
    if target_loss_scale.training_batch_sha256 != (target_loss_scale_batch_sha256) or (
        reconstructed_config.execution_mode != "preflight"
        and np.any(target_loss_scale.eligible_target_count <= 0)
    ):
        raise ValueError("Round 74 pretest target-loss scale binding differs")
    if dict(ensemble_aggregation) != {
        "peer_weights": "equal",
        "payoff_and_adverse_excursion_quantiles": ("arithmetic_mean_of_peer_quantiles"),
        "classification_heads": ("arithmetic_mean_of_peer_probabilities_then_logit"),
        "mean_peer_logits_permitted": False,
        "probability_calibration_fitted_after_ensemble_aggregation": True,
    }:
        raise ValueError("Round 74 pretest ensemble aggregation differs")
    if (
        set(backend)
        != {
            "requested",
            "kind",
            "device",
            "vendor",
            "selection",
            "accelerated",
            "torch_version",
            "torch_directml_version",
            "safetensors_version",
            "deterministic_algorithms_requested",
            "cross_platform_bitwise_reproducibility_claim",
            "cpu_fallback_warning_count",
            "warning_count",
        }
        or backend.get("kind") not in {"cpu", "cuda", "rocm", "xpu", "directml", "mps"}
        or backend.get("requested")
        not in {"auto", "cpu", "cuda", "rocm", "xpu", "directml", "mps"}
        or (
            backend.get("requested") != "auto"
            and backend.get("requested") != backend.get("kind")
        )
        or backend.get("accelerated") is not (backend.get("kind") != "cpu")
        or any(
            not isinstance(backend.get(name), str) or not str(backend[name]).strip()
            for name in (
                "device",
                "vendor",
                "selection",
                "torch_version",
                "torch_directml_version",
                "safetensors_version",
            )
        )
        or backend.get("deterministic_algorithms_requested") is not True
        or backend.get("cross_platform_bitwise_reproducibility_claim") is not False
        or backend.get("cpu_fallback_warning_count") != 0
        or isinstance(backend.get("warning_count"), bool)
        or not isinstance(backend.get("warning_count"), int)
        or int(backend["warning_count"]) < 0
    ):
        raise ValueError("Round 74 pretest backend identity differs")
    batch_fields = ("training_batch_sha256", "tuning_batch_sha256")
    for field in batch_fields:
        values = development.get(field)
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) for value in values)
            or len(values) != len(set(values))
            or any(_SHA256.fullmatch(str(value)) is None for value in values)
        ):
            raise ValueError("Round 74 pretest data identity differs")
    training_batch_hashes = set(development["training_batch_sha256"])
    tuning_batch_hashes = set(development["tuning_batch_sha256"])
    target_contexts = development.get("target_context_sha256")
    if (
        not isinstance(target_contexts, list)
        or not target_contexts
        or target_contexts != sorted(target_contexts)
        or len(target_contexts) != len(set(target_contexts))
        or any(
            not isinstance(value, str) or _SHA256.fullmatch(value) is None
            for value in target_contexts
        )
    ):
        raise ValueError("Round 74 pretest target-context panel differs")
    target_context_panel_sha256 = _canonical_sha256(
        {
            "schema_version": (ROUND74_EVENT_TARGET_CONTEXT_PANEL_SCHEMA_VERSION),
            "target_context_sha256": target_contexts,
        }
    )
    if (
        development.get("target_context_panel_schema_version")
        != ROUND74_EVENT_TARGET_CONTEXT_PANEL_SCHEMA_VERSION
        or development.get("target_context_panel_sha256") != target_context_panel_sha256
        or development.get("target_context_count") != len(target_contexts)
    ):
        raise ValueError("Round 74 pretest target-context panel differs")
    development_times = tuple(
        development.get(name)
        for name in (
            "training_first_wall_ns",
            "training_last_wall_ns",
            "tuning_first_wall_ns",
            "tuning_last_wall_ns",
        )
    )
    if (
        set(development)
        != {
            "partition_sha256",
            "scaler_sha256",
            "window_representation",
            "target_context_panel_schema_version",
            "target_context_panel_sha256",
            "target_context_sha256",
            "target_context_count",
            "training_batch_sha256",
            "tuning_batch_sha256",
            "representative_window_policy_sha256",
            "representative_window_policy_kind",
            "matched_preparation_sha256",
            "representative_window_policy_applied",
            "training_rows",
            "tuning_rows",
            "training_first_wall_ns",
            "training_last_wall_ns",
            "tuning_first_wall_ns",
            "tuning_last_wall_ns",
            "role_transition_gap_ns",
            "test_batches_consumed",
            "test_sample_digests_consumed",
        }
        or training_batch_hashes & tuning_batch_hashes
        or development.get("representative_window_policy_applied")
        is not (reconstructed_config.execution_mode != "preflight")
        or development.get("window_representation")
        not in ROUND74_EVENT_WINDOW_REPRESENTATIONS
        or any(
            _SHA256.fullmatch(str(development.get(name, ""))) is None
            for name in (
                "partition_sha256",
                "scaler_sha256",
            )
        )
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in development_times
        )
        or not (
            0
            < int(development_times[0])
            <= int(development_times[1])
            < int(development_times[2])
            <= int(development_times[3])
        )
        or int(development_times[2]) - int(development_times[1])
        < ROUND74_EVENT_PARTITION_MINIMUM_PURGE_NS
        or development.get("role_transition_gap_ns")
        != int(development_times[2]) - int(development_times[1])
        or any(
            isinstance(development.get(name), bool)
            or not isinstance(development.get(name), int)
            or int(development[name]) < reconstructed_config.minimum_role_rows
            for name in ("training_rows", "tuning_rows")
        )
    ):
        raise ValueError("Round 74 pretest data identity differs")
    representative_policy_sha256 = development.get(
        "representative_window_policy_sha256"
    )
    if reconstructed_config.execution_mode in {"cohort", "segmented_cohort"}:
        try:
            expected_kind, expected_matched_preparation = (
                _cohort_window_policy_identity(
                    representative_policy_sha256,
                    development.get("matched_preparation_sha256"),
                )
            )
        except ValueError as exc:
            raise ValueError("Round 74 representative window policy differs") from exc
        if (
            development.get("representative_window_policy_kind") != expected_kind
            or development.get("matched_preparation_sha256")
            != expected_matched_preparation
            or reconstructed_config.execution_mode == "segmented_cohort"
            and (
                expected_kind != "segmented_duration_normalized"
                or expected_matched_preparation is not None
            )
        ):
            raise ValueError("Round 74 representative window policy differs")
    elif (
        representative_policy_sha256 is not None
        or development.get("representative_window_policy_kind") != "preflight"
        or development.get("matched_preparation_sha256") is not None
    ):
        raise ValueError("Round 74 preflight window policy differs")
    scaler_keys = {
        "available",
        "reason",
        "filename",
        "scaler_sha256",
        "file_sha256",
        "byte_count",
        "media_type",
        "fit_partition_role",
        "fit_source_scope",
        "fit_source_run_count",
        "fit_source_selection_sha256",
        "reload_verified",
    }
    scaler_available = scaler_artifact.get("available")
    if set(scaler_artifact) != scaler_keys or not isinstance(
        scaler_available,
        bool,
    ):
        raise ValueError("Round 74 pretest scaler artifact differs")
    if scaler_available:
        scaler_filename = scaler_artifact.get("filename")
        scaler_sha256 = scaler_artifact.get("scaler_sha256")
        scaler_file_sha256 = scaler_artifact.get("file_sha256")
        scaler_byte_count = scaler_artifact.get("byte_count")
        scaler_source_scope = scaler_artifact.get("fit_source_scope")
        scaler_source_run_count = scaler_artifact.get("fit_source_run_count")
        scaler_source_selection_sha256 = scaler_artifact.get(
            "fit_source_selection_sha256"
        )
        if (
            not isinstance(scaler_filename, str)
            or _SAFE_FILENAME.fullmatch(scaler_filename) is None
            or scaler_filename
            != f"round74-feature-scaler-{development['scaler_sha256']}.json"
            or scaler_sha256 != development["scaler_sha256"]
            or _SHA256.fullmatch(str(scaler_file_sha256)) is None
            or isinstance(scaler_byte_count, bool)
            or not isinstance(scaler_byte_count, int)
            or scaler_byte_count <= 0
            or scaler_artifact.get("reason") != "training_only_scaler_persisted"
            or scaler_artifact.get("media_type") != "application/json"
            or scaler_artifact.get("fit_partition_role") != "training"
            or scaler_source_scope
            not in {
                "unbound_training_matrix",
                "training_partition_all_runs",
                "segmented_optimization_training_runs",
            }
            or isinstance(scaler_source_run_count, bool)
            or not isinstance(scaler_source_run_count, int)
            or scaler_source_run_count < 0
            or (
                scaler_source_selection_sha256 is not None
                and _SHA256.fullmatch(str(scaler_source_selection_sha256)) is None
            )
            or scaler_artifact.get("reload_verified") is not True
        ):
            raise ValueError("Round 74 pretest scaler artifact differs")
    else:
        if dict(scaler_artifact) != {
            "available": False,
            "reason": "preflight_only",
            "filename": None,
            "scaler_sha256": None,
            "file_sha256": None,
            "byte_count": 0,
            "media_type": None,
            "fit_partition_role": None,
            "fit_source_scope": None,
            "fit_source_run_count": 0,
            "fit_source_selection_sha256": None,
            "reload_verified": False,
        }:
            raise ValueError("Round 74 pretest absent scaler artifact differs")
        scaler_filename = None
        scaler_file_sha256 = None
        scaler_byte_count = 0
    if reconstructed_config.execution_mode != "preflight" and not scaler_available:
        raise ValueError("Round 74 cohort pretest scaler artifact is unavailable")
    candidate_run_losses: dict[str, tuple[float, ...]] = {}
    candidate_group_losses: dict[str, dict[str, float]] = {}
    candidate_parameter_counts: dict[str, int] = {}
    for panel_candidate, raw_report in candidate_panel.items():
        if not isinstance(raw_report, Mapping):
            raise ValueError("Round 74 pretest candidate report differs")
        if panel_candidate not in reconstructed_config.candidate_ids:
            raise ValueError("Round 74 pretest candidate report differs")
        _metrics, run_losses, group_losses, parameter_count = (
            _validated_candidate_fit_report(
                raw_report,
                panel_key=str(panel_candidate),
                expected_candidate_id=str(panel_candidate),
                expected_feature_view="market_state_clock_neutral",
                expected_state_conditioned_flow=False,
                seeds=seeds,
                tuning_run_count=stage_tuning_run_counts["architecture"],
                execution_mode=reconstructed_config.execution_mode,
            )
        )
        candidate_run_losses[str(panel_candidate)] = run_losses
        candidate_group_losses[str(panel_candidate)] = group_losses
        candidate_parameter_counts[str(panel_candidate)] = parameter_count
    if len({tuple(sorted(values)) for values in candidate_group_losses.values()}) != 1:
        raise ValueError("Round 74 pretest candidate subgroup panel differs")
    if reconstructed_config.architecture_selection_mode == "complexity_gate":
        promotion_required_runs = (
            stage_tuning_run_counts["architecture"]
            if reconstructed_config.execution_mode == "segmented_cohort"
            else ROUND74_COMPLEXITY_PROMOTION_REQUIRED_TUNING_RUNS
        )
        expected_winner, expected_promotion_reports = _complexity_gated_candidate_id(
            reconstructed_config.candidate_ids,
            candidate_run_losses,
            candidate_parameter_counts,
            minimum_mean_loss_improvement=(
                reconstructed_config.minimum_tuning_improvement
            ),
            candidate_group_losses=candidate_group_losses,
            required_paired_run_count=promotion_required_runs,
        )
        expected_criterion = (
            "sequential parameter-count complexity promotion; each challenger "
            f"requires all {promotion_required_runs} model-selection capture "
            "runs, must strictly exceed the numerical mean-loss floor after "
            "deleting any one capture run, must materially win a strict majority "
            "of capture runs, and may not degrade any paired run or eligible "
            "run-symbol-horizon proper-loss subgroup beyond that floor"
        )
        expected_comparison_count = ROUND74_COMPLEXITY_PROMOTION_COMPARISON_COUNT
        expected_paired_run_count = promotion_required_runs
        expected_improvement = reconstructed_config.minimum_tuning_improvement
    else:
        expected_winner = reconstructed_config.candidate_ids[0]
        expected_promotion_reports = ()
        expected_criterion = (
            "fixed baseline-selected architecture; no architecture search or "
            "complexity promotion performed"
        )
        expected_comparison_count = 0
        expected_paired_run_count = 0
        expected_improvement = 0.0
    architecture_selected_report = candidate_panel[expected_winner]
    assert isinstance(architecture_selected_report, Mapping)
    if set(feature_view_panel) != {"clock_features", "order_flow_features"}:
        raise ValueError("Round 74 pretest feature-view panel differs")

    def validated_feature_stage(
        stage_id: str,
        incumbent_feature_view: str,
        challenger_feature_view: str,
    ) -> tuple[
        Mapping[str, object],
        tuple[float, ...],
        dict[str, float],
        int,
        Mapping[str, object],
        tuple[float, ...],
        dict[str, float],
        int,
    ]:
        raw_stage = feature_view_panel.get(stage_id)
        if (
            not isinstance(raw_stage, Mapping)
            or set(raw_stage) != {"incumbent", "challenger"}
            or not isinstance(raw_stage.get("incumbent"), Mapping)
            or not isinstance(raw_stage.get("challenger"), Mapping)
        ):
            raise ValueError("Round 74 pretest feature-view stage differs")
        stage_run_count = stage_tuning_run_counts[stage_id]
        incumbent = raw_stage["incumbent"]
        challenger = raw_stage["challenger"]
        assert isinstance(incumbent, Mapping)
        assert isinstance(challenger, Mapping)
        incumbent_values = _validated_candidate_fit_report(
            incumbent,
            panel_key=incumbent_feature_view,
            expected_candidate_id=expected_winner,
            expected_feature_view=incumbent_feature_view,
            expected_state_conditioned_flow=False,
            seeds=seeds,
            tuning_run_count=stage_run_count,
            execution_mode=reconstructed_config.execution_mode,
        )
        challenger_values = _validated_candidate_fit_report(
            challenger,
            panel_key=challenger_feature_view,
            expected_candidate_id=expected_winner,
            expected_feature_view=challenger_feature_view,
            expected_state_conditioned_flow=False,
            seeds=seeds,
            tuning_run_count=stage_run_count,
            execution_mode=reconstructed_config.execution_mode,
        )
        if incumbent_values[3] != challenger_values[3] or set(
            incumbent_values[2]
        ) != set(challenger_values[2]):
            raise ValueError("Round 74 pretest feature-view stage differs")
        return (*incumbent_values, *challenger_values)

    (
        _clock_incumbent_metrics,
        clock_incumbent_run_losses,
        clock_incumbent_group_losses,
        _clock_incumbent_parameter_count,
        _clock_challenger_metrics,
        clock_challenger_run_losses,
        clock_challenger_group_losses,
        _clock_challenger_parameter_count,
    ) = validated_feature_stage(
        "clock_features",
        "market_state_clock_neutral",
        "market_state_with_clock",
    )
    feature_view_required_runs = (
        stage_tuning_run_counts["clock_features"]
        if reconstructed_config.execution_mode == "segmented_cohort"
        else ROUND74_COMPLEXITY_PROMOTION_REQUIRED_TUNING_RUNS
    )
    expected_clock_report = _feature_view_promotion_report(
        "market_state_clock_neutral",
        "market_state_with_clock",
        clock_incumbent_run_losses,
        clock_challenger_run_losses,
        incumbent_group_losses=clock_incumbent_group_losses,
        challenger_group_losses=clock_challenger_group_losses,
        minimum_mean_loss_improvement=(reconstructed_config.minimum_tuning_improvement),
        required_paired_run_count=feature_view_required_runs,
    )
    expected_clock_view = (
        "market_state_with_clock"
        if expected_clock_report["promoted"] is True
        else "market_state_clock_neutral"
    )
    expected_order_flow_challenger_view = _order_flow_challenger_feature_view(
        expected_clock_view
    )
    (
        order_flow_incumbent_metrics,
        order_flow_incumbent_run_losses,
        order_flow_incumbent_group_losses,
        _order_flow_incumbent_parameter_count,
        order_flow_challenger_metrics,
        order_flow_challenger_run_losses,
        order_flow_challenger_group_losses,
        _order_flow_challenger_parameter_count,
    ) = validated_feature_stage(
        "order_flow_features",
        expected_clock_view,
        expected_order_flow_challenger_view,
    )
    order_flow_required_runs = (
        stage_tuning_run_counts["order_flow_features"]
        if reconstructed_config.execution_mode == "segmented_cohort"
        else feature_view_required_runs
    )
    expected_order_flow_report = _feature_view_promotion_report(
        expected_clock_view,
        expected_order_flow_challenger_view,
        order_flow_incumbent_run_losses,
        order_flow_challenger_run_losses,
        incumbent_group_losses=order_flow_incumbent_group_losses,
        challenger_group_losses=order_flow_challenger_group_losses,
        minimum_mean_loss_improvement=(reconstructed_config.minimum_tuning_improvement),
        required_paired_run_count=order_flow_required_runs,
    )
    expected_feature_view = (
        expected_order_flow_challenger_view
        if expected_order_flow_report["promoted"] is True
        else expected_clock_view
    )
    expected_feature_view_criterion = _feature_view_selection_criterion(
        feature_view_required_runs,
        order_flow_required_runs,
    )
    order_flow_stage = feature_view_panel["order_flow_features"]
    assert isinstance(order_flow_stage, Mapping)
    selected_feature_report = (
        order_flow_stage["challenger"]
        if expected_order_flow_report["promoted"] is True
        else order_flow_stage["incumbent"]
    )
    assert isinstance(selected_feature_report, Mapping)
    selected_feature_metrics = (
        order_flow_challenger_metrics
        if expected_order_flow_report["promoted"] is True
        else order_flow_incumbent_metrics
    )
    if (
        set(feature_view_selection)
        != {
            "schema_version",
            "criterion",
            "market_state_feature_names",
            "market_state_feature_names_sha256",
            "order_flow_feature_names",
            "order_flow_feature_names_sha256",
            "clock_feature_names",
            "clock_feature_names_sha256",
            "supported_feature_views",
            "evaluated_feature_views",
            "selected_feature_view",
            "selected_candidate_id",
            "selected_tuning_metrics",
            "clock_required_paired_capture_run_count",
            "order_flow_required_paired_capture_run_count",
            "minimum_mean_proper_loss_improvement",
            "maximum_permitted_paired_run_loss_degradation",
            "order_flow_promotion_report",
            "clock_promotion_report",
            "architecture_fixed_before_layer_gates",
            "state_first_incumbent",
            "order_flow_default_on_gate_failure",
            "clock_default_on_gate_failure",
            "statistical_independence_or_significance_claim",
            "backtest_metric_used_for_selection",
            "test_targets_used",
        }
        or feature_view_selection.get("schema_version")
        != ROUND74_EVENT_FEATURE_VIEW_SCHEMA_VERSION
        or feature_view_selection.get("criterion") != expected_feature_view_criterion
        or feature_view_selection.get("market_state_feature_names")
        != list(ROUND74_EVENT_MARKET_STATE_FEATURE_NAMES)
        or feature_view_selection.get("market_state_feature_names_sha256")
        != ROUND74_EVENT_MARKET_STATE_FEATURE_NAMES_SHA256
        or feature_view_selection.get("order_flow_feature_names")
        != list(ROUND74_EVENT_ORDER_FLOW_FEATURE_NAMES)
        or feature_view_selection.get("order_flow_feature_names_sha256")
        != ROUND74_EVENT_ORDER_FLOW_FEATURE_NAMES_SHA256
        or feature_view_selection.get("clock_feature_names")
        != list(ROUND74_EVENT_CLOCK_FEATURE_NAMES)
        or feature_view_selection.get("clock_feature_names_sha256")
        != ROUND74_EVENT_CLOCK_FEATURE_NAMES_SHA256
        or feature_view_selection.get("supported_feature_views")
        != list(ROUND74_EVENT_FEATURE_VIEWS)
        or feature_view_selection.get("evaluated_feature_views")
        != [
            "market_state_clock_neutral",
            "market_state_with_clock",
            expected_order_flow_challenger_view,
        ]
        or selected_feature_view != expected_feature_view
        or feature_view_selection.get("selected_candidate_id") != expected_winner
        or feature_view_selection.get("selected_tuning_metrics")
        != selected_feature_metrics
        or feature_view_selection.get("clock_required_paired_capture_run_count")
        != feature_view_required_runs
        or feature_view_selection.get("order_flow_required_paired_capture_run_count")
        != order_flow_required_runs
        or feature_view_selection.get("minimum_mean_proper_loss_improvement")
        != reconstructed_config.minimum_tuning_improvement
        or feature_view_selection.get("maximum_permitted_paired_run_loss_degradation")
        != reconstructed_config.minimum_tuning_improvement
        or feature_view_selection.get("clock_promotion_report") != expected_clock_report
        or feature_view_selection.get("order_flow_promotion_report")
        != expected_order_flow_report
        or feature_view_selection.get("architecture_fixed_before_layer_gates")
        is not True
        or feature_view_selection.get("state_first_incumbent")
        != "market_state_clock_neutral"
        or feature_view_selection.get("clock_default_on_gate_failure")
        != "market_state_clock_neutral"
        or feature_view_selection.get("order_flow_default_on_gate_failure")
        != expected_clock_view
        or feature_view_selection.get("statistical_independence_or_significance_claim")
        is not False
        or feature_view_selection.get("backtest_metric_used_for_selection") is not False
        or feature_view_selection.get("test_targets_used") is not False
    ):
        raise ValueError("Round 74 pretest feature-view selection differs")
    expected_interaction_evaluated = _feature_view_contains_order_flow(
        expected_feature_view
    )
    expected_interaction_keys = (
        {"unconditioned_order_flow", "state_conditioned_order_flow"}
        if expected_interaction_evaluated
        else {"unconditioned_order_flow"}
    )
    if set(state_conditioned_flow_panel) != expected_interaction_keys:
        raise ValueError("Round 74 state-conditioned flow panel differs")
    if expected_interaction_evaluated:
        raw_unconditioned_report = state_conditioned_flow_panel.get(
            "unconditioned_order_flow"
        )
        raw_conditioned_report = state_conditioned_flow_panel.get(
            "state_conditioned_order_flow"
        )
        if not isinstance(raw_unconditioned_report, Mapping) or not isinstance(
            raw_conditioned_report,
            Mapping,
        ):
            raise ValueError("Round 74 state-conditioned flow report differs")
        (
            unconditioned_metrics,
            unconditioned_run_losses,
            unconditioned_group_losses,
            unconditioned_parameter_count,
        ) = _validated_candidate_fit_report(
            raw_unconditioned_report,
            panel_key=expected_winner,
            expected_candidate_id=expected_winner,
            expected_feature_view=expected_feature_view,
            expected_state_conditioned_flow=False,
            seeds=seeds,
            tuning_run_count=stage_tuning_run_counts["state_conditioned_flow"],
            execution_mode=reconstructed_config.execution_mode,
        )
        (
            conditioned_metrics,
            conditioned_run_losses,
            conditioned_group_losses,
            conditioned_parameter_count,
        ) = _validated_candidate_fit_report(
            raw_conditioned_report,
            panel_key=expected_winner,
            expected_candidate_id=expected_winner,
            expected_feature_view=expected_feature_view,
            expected_state_conditioned_flow=True,
            seeds=seeds,
            tuning_run_count=stage_tuning_run_counts["state_conditioned_flow"],
            execution_mode=reconstructed_config.execution_mode,
        )
        if conditioned_parameter_count <= unconditioned_parameter_count or set(
            conditioned_group_losses
        ) != set(unconditioned_group_losses):
            raise ValueError("Round 74 state-conditioned flow complexity differs")
        interaction_required_runs = (
            stage_tuning_run_counts["state_conditioned_flow"]
            if reconstructed_config.execution_mode == "segmented_cohort"
            else feature_view_required_runs
        )
        expected_interaction_promotion_report = _paired_promotion_report(
            "unconditioned_order_flow",
            "state_conditioned_order_flow",
            unconditioned_run_losses,
            conditioned_run_losses,
            minimum_improvement=(reconstructed_config.minimum_tuning_improvement),
            required_paired_run_count=interaction_required_runs,
            incumbent_group_losses=unconditioned_group_losses,
            challenger_group_losses=conditioned_group_losses,
        )
        expected_state_conditioned_flow = (
            expected_interaction_promotion_report["promoted"] is True
        )
        selected_interaction_report = (
            raw_conditioned_report
            if expected_state_conditioned_flow
            else raw_unconditioned_report
        )
        expected_interaction_metrics = (
            conditioned_metrics
            if expected_state_conditioned_flow
            else unconditioned_metrics
        )
        expected_interaction_reason = "paired_proper_loss_gate_completed"
        expected_interaction_required_runs = interaction_required_runs
        expected_interaction_improvement = (
            reconstructed_config.minimum_tuning_improvement
        )
    else:
        if (
            state_conditioned_flow_panel.get("unconditioned_order_flow")
            != selected_feature_report
        ):
            raise ValueError("Round 74 state-conditioned flow panel differs")
        expected_interaction_promotion_report = None
        expected_state_conditioned_flow = False
        selected_interaction_report = selected_feature_report
        expected_interaction_metrics = selected_feature_metrics
        expected_interaction_reason = "order_flow_layer_not_selected"
        expected_interaction_required_runs = 0
        expected_interaction_improvement = 0.0
    expected_interaction_criterion = (
        "after the feature view is fixed, neutral-initialized market-state "
        "modulation of admitted order flow challenges the unconditioned "
        "incumbent on identical seeds and the supervised objective; promotion "
        "requires strict mean proper-loss improvement and no paired run or "
        "run-symbol-horizon subgroup degradation beyond the numerical floor"
    )
    if (
        set(state_conditioned_flow_selection)
        != {
            "schema_version",
            "criterion",
            "reason",
            "evaluated",
            "selected_state_conditioned_flow",
            "selected_candidate_id",
            "selected_feature_view",
            "selected_tuning_metrics",
            "required_paired_capture_run_count",
            "minimum_mean_proper_loss_improvement",
            "maximum_permitted_paired_run_loss_degradation",
            "promotion_report",
            "feature_view_fixed_before_interaction_gate",
            "order_flow_required",
            "neutral_multiplier_at_initialization",
            "unconditioned_default_on_gate_failure",
            "statistical_independence_or_significance_claim",
            "backtest_metric_used_for_selection",
            "test_targets_used",
        }
        or state_conditioned_flow_selection.get("schema_version")
        != ROUND74_EVENT_STATE_CONDITIONED_FLOW_SCHEMA_VERSION
        or state_conditioned_flow_selection.get("criterion")
        != expected_interaction_criterion
        or state_conditioned_flow_selection.get("reason") != expected_interaction_reason
        or state_conditioned_flow_selection.get("evaluated")
        is not expected_interaction_evaluated
        or selected_state_conditioned_flow is not expected_state_conditioned_flow
        or state_conditioned_flow_selection.get("selected_candidate_id")
        != expected_winner
        or state_conditioned_flow_selection.get("selected_feature_view")
        != expected_feature_view
        or state_conditioned_flow_selection.get("selected_tuning_metrics")
        != expected_interaction_metrics
        or state_conditioned_flow_selection.get("required_paired_capture_run_count")
        != expected_interaction_required_runs
        or state_conditioned_flow_selection.get("minimum_mean_proper_loss_improvement")
        != expected_interaction_improvement
        or state_conditioned_flow_selection.get(
            "maximum_permitted_paired_run_loss_degradation"
        )
        != expected_interaction_improvement
        or state_conditioned_flow_selection.get("promotion_report")
        != expected_interaction_promotion_report
        or state_conditioned_flow_selection.get(
            "feature_view_fixed_before_interaction_gate"
        )
        is not True
        or state_conditioned_flow_selection.get("order_flow_required") is not True
        or state_conditioned_flow_selection.get("neutral_multiplier_at_initialization")
        != 1.0
        or state_conditioned_flow_selection.get("unconditioned_default_on_gate_failure")
        is not True
        or state_conditioned_flow_selection.get(
            "statistical_independence_or_significance_claim"
        )
        is not False
        or state_conditioned_flow_selection.get("backtest_metric_used_for_selection")
        is not False
        or state_conditioned_flow_selection.get("test_targets_used") is not False
    ):
        raise ValueError("Round 74 state-conditioned flow selection differs")
    pretraining_supported = (
        round74_event_model_pretraining_channels(
            build_round74_event_model(
                expected_winner,
                state_conditioned_flow=expected_state_conditioned_flow,
            )
        )
        is not None
    )
    pretraining_evaluated = (
        pretraining_supported and reconstructed_config.execution_mode != "preflight"
    )
    expected_initialization_keys = (
        set(ROUND74_EVENT_PRETRAINING_INITIALIZATION_IDS)
        if pretraining_evaluated
        else {"random"}
    )
    if set(initialization_panel) != expected_initialization_keys or (
        not pretraining_evaluated
        and initialization_panel.get("random") != selected_interaction_report
    ):
        raise ValueError("Round 74 pretest initialization panel differs")
    initialization_validation_run_count = (
        stage_tuning_run_counts["causal_pretraining"]
        if pretraining_evaluated
        else (
            stage_tuning_run_counts["state_conditioned_flow"]
            if expected_interaction_evaluated
            else stage_tuning_run_counts["order_flow_features"]
        )
    )
    initialization_metrics: dict[str, Mapping[str, object]] = {}
    initialization_run_losses: dict[str, tuple[float, ...]] = {}
    initialization_group_losses: dict[str, dict[str, float]] = {}
    initialization_parameter_counts: dict[str, int] = {}
    pretraining_split_sha256: set[str] = set()
    pretraining_feature_batch_sha256: set[tuple[str, ...]] = set()
    pretraining_partition_rows: set[tuple[int, int]] = set()
    for initialization_id, raw_report in initialization_panel.items():
        if (
            initialization_id not in ROUND74_EVENT_PRETRAINING_INITIALIZATION_IDS
            or not isinstance(raw_report, Mapping)
        ):
            raise ValueError("Round 74 pretest initialization report differs")
        metrics, run_losses, group_losses, parameter_count = (
            _validated_candidate_fit_report(
                raw_report,
                panel_key=expected_winner,
                expected_candidate_id=expected_winner,
                expected_feature_view=expected_feature_view,
                expected_state_conditioned_flow=(expected_state_conditioned_flow),
                seeds=seeds,
                tuning_run_count=initialization_validation_run_count,
                execution_mode=reconstructed_config.execution_mode,
            )
        )
        peers = raw_report.get("peer_reports")
        assert isinstance(peers, list)
        for peer in peers:
            if not isinstance(peer, Mapping):
                raise ValueError("Round 74 pretest initialization peer differs")
            pretraining_report = peer.get("causal_pretraining")
            if (
                peer.get("initialization_id") != initialization_id
                or not isinstance(pretraining_report, Mapping)
                or pretraining_report.get("schema_version")
                != ROUND74_EVENT_PRETRAINING_SCHEMA_VERSION
                or pretraining_report.get("initialization_id") != initialization_id
                or pretraining_report.get("config")
                != reconstructed_config.pretraining.as_dict()
                or pretraining_report.get("supervised_targets_used") is not False
                or pretraining_report.get("tuning_features_used") is not False
                or pretraining_report.get("tuning_targets_used") is not False
                or pretraining_report.get("calibration_data_used") is not False
                or pretraining_report.get("test_data_used") is not False
                or pretraining_report.get("financial_edge_claim") is not False
            ):
                raise ValueError("Round 74 pretest initialization peer differs")
            if initialization_id == "random":
                if (
                    pretraining_report.get("applied") is not False
                    or pretraining_report.get("reason")
                    != "paired_random_initialization_incumbent"
                ):
                    raise ValueError("Round 74 random initialization report differs")
            else:
                pretraining_feature_batches = pretraining_report.get(
                    "training_feature_batch_sha256"
                )
                if not isinstance(pretraining_feature_batches, list):
                    raise ValueError("Round 74 causal pretraining report differs")
                if (
                    len(pretraining_feature_batches)
                    != len(target_loss_scale_batch_sha256)
                    or any(
                        not isinstance(value, str) or _SHA256.fullmatch(value) is None
                        for value in pretraining_feature_batches
                    )
                    or len(pretraining_feature_batches)
                    != len(set(pretraining_feature_batches))
                    or pretraining_feature_batches
                    != sorted(pretraining_feature_batches)
                    or pretraining_report.get("training_capture_run_count")
                    != len(target_loss_scale_batch_sha256)
                    or pretraining_report.get("encoder_state_restored") is not True
                    or pretraining_report.get("temporary_prediction_head_persisted")
                    is not False
                    or _SHA256.fullmatch(
                        str(pretraining_report.get("split_sha256", ""))
                    )
                    is None
                    or _SHA256.fullmatch(
                        str(pretraining_report.get("initial_encoder_sha256", ""))
                    )
                    is None
                    or _SHA256.fullmatch(
                        str(pretraining_report.get("final_encoder_sha256", ""))
                    )
                    is None
                    or pretraining_report.get("initial_encoder_sha256")
                    == pretraining_report.get("final_encoder_sha256")
                ):
                    raise ValueError("Round 74 causal pretraining report differs")
                pretraining_split_sha256.add(str(pretraining_report["split_sha256"]))
                pretraining_feature_batch_sha256.add(
                    tuple(str(value) for value in pretraining_feature_batches)
                )
                training_rows = pretraining_report.get("training_rows")
                validation_rows = pretraining_report.get("validation_rows")
                if (
                    isinstance(training_rows, bool)
                    or not isinstance(training_rows, int)
                    or training_rows < 1
                    or isinstance(validation_rows, bool)
                    or not isinstance(validation_rows, int)
                    or validation_rows < 1
                ):
                    raise ValueError("Round 74 causal pretraining partition differs")
                pretraining_partition_rows.add((training_rows, validation_rows))
        initialization_metrics[str(initialization_id)] = metrics
        initialization_run_losses[str(initialization_id)] = run_losses
        initialization_group_losses[str(initialization_id)] = group_losses
        initialization_parameter_counts[str(initialization_id)] = parameter_count
    if len(set(initialization_parameter_counts.values())) != 1:
        raise ValueError("Round 74 pretest initialization parameter count differs")
    if pretraining_evaluated and (
        len(pretraining_split_sha256) != 1
        or len(pretraining_feature_batch_sha256) != 1
        or len(pretraining_partition_rows) != 1
    ):
        raise ValueError("Round 74 causal pretraining peer split differs")
    if pretraining_evaluated:
        initialization_required_runs = (
            stage_tuning_run_counts["causal_pretraining"]
            if reconstructed_config.execution_mode == "segmented_cohort"
            else feature_view_required_runs
        )
        expected_initialization_promotion_report = _paired_promotion_report(
            "random",
            "causal_next_event_pretrained",
            initialization_run_losses["random"],
            initialization_run_losses["causal_next_event_pretrained"],
            minimum_improvement=reconstructed_config.minimum_tuning_improvement,
            required_paired_run_count=initialization_required_runs,
            incumbent_group_losses=initialization_group_losses["random"],
            challenger_group_losses=initialization_group_losses[
                "causal_next_event_pretrained"
            ],
        )
        expected_initialization_id = (
            "causal_next_event_pretrained"
            if expected_initialization_promotion_report["promoted"] is True
            else "random"
        )
        expected_initialization_reason = "paired_proper_loss_gate_completed"
        expected_initialization_improvement = (
            reconstructed_config.minimum_tuning_improvement
        )
    else:
        initialization_required_runs = 0
        expected_initialization_promotion_report = None
        expected_initialization_id = "random"
        expected_initialization_reason = (
            "preflight_never_evaluates_pretraining"
            if reconstructed_config.execution_mode == "preflight"
            else "selected_pooled_control_has_no_causal_event_encoder"
        )
        expected_initialization_improvement = 0.0
    expected_initialization_criterion = (
        "causal next-event pretraining challenges the same random "
        "initialization on the fixed architecture, feature view, seeds, "
        "and supervised objective; promotion requires strict mean "
        "proper-loss improvement and no paired run or "
        "run-symbol-horizon subgroup degradation beyond the numerical floor"
    )
    selected_initialization_report = initialization_panel[expected_initialization_id]
    assert isinstance(selected_initialization_report, Mapping)
    if (
        set(initialization_selection)
        != {
            "schema_version",
            "criterion",
            "reason",
            "pretraining_supported",
            "pretraining_evaluated",
            "ordered_initialization_ids",
            "selected_initialization_id",
            "selected_candidate_id",
            "selected_feature_view",
            "selected_state_conditioned_flow",
            "selected_tuning_metrics",
            "required_paired_capture_run_count",
            "minimum_mean_proper_loss_improvement",
            "maximum_permitted_paired_run_loss_degradation",
            "promotion_report",
            "training_features_only",
            "supervised_targets_used_by_pretraining",
            "tuning_features_used_by_pretraining",
            "tuning_targets_used_by_pretraining",
            "calibration_data_used_by_pretraining",
            "test_data_used_by_pretraining",
            "random_default_on_gate_failure",
            "statistical_independence_or_significance_claim",
            "backtest_metric_used_for_selection",
        }
        or initialization_selection.get("schema_version")
        != ROUND74_EVENT_PRETRAINING_SCHEMA_VERSION
        or initialization_selection.get("criterion")
        != expected_initialization_criterion
        or initialization_selection.get("reason") != expected_initialization_reason
        or initialization_selection.get("pretraining_supported")
        is not pretraining_supported
        or initialization_selection.get("pretraining_evaluated")
        is not pretraining_evaluated
        or initialization_selection.get("ordered_initialization_ids")
        != list(ROUND74_EVENT_PRETRAINING_INITIALIZATION_IDS)
        or initialization_selection.get("selected_initialization_id")
        != expected_initialization_id
        or initialization_selection.get("selected_candidate_id") != expected_winner
        or initialization_selection.get("selected_feature_view")
        != expected_feature_view
        or initialization_selection.get("selected_state_conditioned_flow")
        is not expected_state_conditioned_flow
        or initialization_selection.get("selected_tuning_metrics")
        != initialization_metrics[expected_initialization_id]
        or initialization_selection.get("required_paired_capture_run_count")
        != initialization_required_runs
        or initialization_selection.get("minimum_mean_proper_loss_improvement")
        != expected_initialization_improvement
        or initialization_selection.get("maximum_permitted_paired_run_loss_degradation")
        != expected_initialization_improvement
        or initialization_selection.get("promotion_report")
        != expected_initialization_promotion_report
        or initialization_selection.get("training_features_only") is not True
        or initialization_selection.get("supervised_targets_used_by_pretraining")
        is not False
        or initialization_selection.get("tuning_features_used_by_pretraining")
        is not False
        or initialization_selection.get("tuning_targets_used_by_pretraining")
        is not False
        or initialization_selection.get("calibration_data_used_by_pretraining")
        is not False
        or initialization_selection.get("test_data_used_by_pretraining") is not False
        or initialization_selection.get("random_default_on_gate_failure") is not True
        or initialization_selection.get(
            "statistical_independence_or_significance_claim"
        )
        is not False
        or initialization_selection.get("backtest_metric_used_for_selection")
        is not False
    ):
        raise ValueError("Round 74 pretest initialization selection differs")
    if (
        set(selection)
        != {
            "architecture_selection_mode",
            "criterion",
            "selected_candidate_id",
            "selected_tuning_metrics",
            "ordered_candidate_ids",
            "planned_comparison_count",
            "required_paired_capture_run_count",
            "minimum_mean_proper_loss_improvement",
            "maximum_permitted_paired_run_loss_degradation",
            "statistical_independence_or_significance_claim",
            "promotion_reports",
            "complexity_promotion_privilege",
            "backtest_metric_used_for_selection",
        }
        or selection.get("architecture_selection_mode")
        != reconstructed_config.architecture_selection_mode
        or selection.get("criterion") != expected_criterion
        or candidate_id != expected_winner
        or selection.get("ordered_candidate_ids")
        != list(reconstructed_config.candidate_ids)
        or selection.get("planned_comparison_count") != expected_comparison_count
        or selection.get("required_paired_capture_run_count")
        != expected_paired_run_count
        or selection.get("minimum_mean_proper_loss_improvement") != expected_improvement
        or selection.get("maximum_permitted_paired_run_loss_degradation")
        != expected_improvement
        or selection.get("statistical_independence_or_significance_claim") is not False
        or selection.get("promotion_reports") != list(expected_promotion_reports)
        or selection.get("complexity_promotion_privilege") is not False
        or selection.get("backtest_metric_used_for_selection") is not False
        or selection.get("selected_tuning_metrics")
        != architecture_selected_report.get("ensemble_tuning_metrics")
        or artifact.get("prediction_sha256")
        != selected_initialization_report.get("ensemble_prediction_sha256")
        or _SHA256.fullmatch(str(artifact.get("prediction_sha256", ""))) is None
        or artifact.get("header_metadata_omitted_for_byte_stability") is not True
        or artifact.get("media_type") != "application/x-safetensors"
        or isinstance(artifact.get("byte_count"), bool)
        or not isinstance(artifact.get("byte_count"), int)
        or int(artifact["byte_count"]) <= 0
        or set(artifact)
        != {
            "filename",
            "sha256",
            "byte_count",
            "media_type",
            "header_metadata_omitted_for_byte_stability",
            "pickle_permitted",
            "state_reload_verified",
            "prediction_reload_verified",
            "prediction_sha256",
        }
    ):
        raise ValueError("Round 74 pretest selection or artifact differs")
    required_true_authority = {
        "development_training_completed",
        "chronological_tuning_completed",
        "pretest_policy_sealed",
    }
    required_false_authority = {
        "sealed_test_evaluated",
        "representative_market_evidence_claim",
        "financial_edge_tested",
        "profitability_claim",
        "ai_uplift_claim",
        "paper_trading_authority",
        "testnet_trading_authority",
        "live_trading_authority",
    }
    if set(authority) != required_true_authority | required_false_authority or any(
        authority.get(name) is not True for name in required_true_authority
    ):
        raise ValueError("Round 74 pretest policy authority differs")
    for forbidden in (*sorted(required_false_authority),):
        if authority.get(forbidden) is not False:
            raise ValueError("Round 74 pretest policy overstates authority")
    if scaler_available:
        assert isinstance(scaler_filename, str)
        scaler_path = selected_path.parent / scaler_filename
        try:
            scaler_bytes = scaler_path.read_bytes()
        except OSError as exc:
            raise ValueError("Round 74 pretest scaler could not be read") from exc
        if (
            _sha256_bytes(scaler_bytes) != scaler_file_sha256
            or len(scaler_bytes) != scaler_byte_count
        ):
            raise ValueError("Round 74 pretest scaler artifact differs")
        loaded_scaler = _load_scaler_bytes(scaler_bytes)
        if (
            loaded_scaler.scaler_sha256 != development["scaler_sha256"]
            or loaded_scaler.fit_source_scope != scaler_source_scope
            or len(loaded_scaler.fit_source_run_ids) != scaler_source_run_count
            or (loaded_scaler.fit_source_selection_sha256 or None)
            != scaler_source_selection_sha256
        ):
            raise ValueError("Round 74 pretest scaler identity differs")
        if isolated_scaler_selection_sha256 is not None and (
            loaded_scaler.fit_source_scope != "segmented_optimization_training_runs"
            or loaded_scaler.fit_source_run_ids != isolated_scaler_run_ids
            or loaded_scaler.fit_source_partition_sha256
            != development["partition_sha256"]
            or loaded_scaler.fit_source_selection_sha256
            != isolated_scaler_selection_sha256
        ):
            raise ValueError("Round 74 pretest segmented scaler binding differs")
        if reconstructed_config.execution_mode == "cohort" and (
            loaded_scaler.fit_source_scope != "training_partition_all_runs"
            or loaded_scaler.fit_source_partition_sha256
            != development["partition_sha256"]
        ):
            raise ValueError("Round 74 pretest cohort scaler binding differs")
    model_path = selected_path.parent / filename
    try:
        model_bytes = model_path.read_bytes()
    except OSError as exc:
        raise ValueError("Round 74 pretest model could not be read") from exc
    if _sha256_bytes(model_bytes) != model_sha256 or len(model_bytes) != artifact.get(
        "byte_count"
    ):
        raise ValueError("Round 74 pretest model artifact differs")
    try:
        model = _load_ensemble_from_bytes(
            model_bytes,
            candidate_id=candidate_id,
            feature_view=selected_feature_view,
            state_conditioned_flow=selected_state_conditioned_flow,
            peer_count=len(seeds),
        )
    except Exception as exc:
        raise ValueError("Round 74 pretest model tensors differ") from exc
    model.eval()
    return model, policy


def load_round74_pretest_scaler(
    policy_path: str | Path,
) -> Round74EventFeatureScaler:
    """Load the exact training-only scaler bound to a cohort pretest policy."""

    selected_path = Path(policy_path)
    _model, policy = load_round74_pretest_policy(selected_path)
    artifact = policy["scaler_artifact"]
    assert isinstance(artifact, Mapping)
    if artifact.get("available") is not True:
        raise ValueError("Round 74 pretest scaler is unavailable")
    filename = artifact.get("filename")
    if not isinstance(filename, str):
        raise ValueError("Round 74 pretest scaler filename differs")
    try:
        payload = (selected_path.parent / filename).read_bytes()
    except OSError as exc:
        raise ValueError("Round 74 pretest scaler could not be read") from exc
    return _load_scaler_bytes(payload)


__all__ = [
    "ROUND74_COMPLEXITY_PROMOTION_COMPARISON_COUNT",
    "ROUND74_COMPLEXITY_PROMOTION_REQUIRED_TUNING_RUNS",
    "ROUND74_EVENT_PRETEST_POLICY_SCHEMA_VERSION",
    "ROUND74_EVENT_SEGMENTED_EARLY_STOPPING_FRACTION_DENOMINATOR",
    "ROUND74_EVENT_SEGMENTED_MINIMUM_EARLY_STOPPING_RUNS",
    "ROUND74_EVENT_SEGMENTED_MINIMUM_OPTIMIZATION_RUNS",
    "ROUND74_EVENT_SELECTION_PROTOCOL_SCHEMA_VERSION",
    "ROUND74_EVENT_STATE_CONDITIONED_FLOW_SCHEMA_VERSION",
    "ROUND74_EVENT_TARGET_CONTEXT_PANEL_SCHEMA_VERSION",
    "ROUND74_EVENT_TRAINING_DEFAULT_SEEDS",
    "ROUND74_EVENT_TRAINING_LOSS_WEIGHTS",
    "ROUND74_EVENT_TRAINING_REQUIRED_CAPTURE_RUNS",
    "ROUND74_EVENT_TRAINING_SCHEMA_VERSION",
    "Round74EventEnsemble",
    "Round74EventSelectionProtocol",
    "Round74EventTrainingConfig",
    "Round74PretestPolicyArtifact",
    "load_round74_pretest_policy",
    "load_round74_pretest_scaler",
    "round74_paired_run_stability_evidence",
    "train_and_seal_round74_pretest_policy",
    "train_and_seal_round74_pretest_policy_from_prepared_roles",
]
