"""Causal, bounded-memory sequence materialization for Round 25."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Sequence

import numpy as np

from .polymarket_round25_candidate_design import (
    POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
    POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
)
from .polymarket_round25_controls import (
    POLYMARKET_ROUND25_CONTROL_FIT_CONTRACT_SHA256,
    POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256,
    round25_logit,
    transform_round25_features,
)
from .polymarket_round25_dataset import (
    POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION,
    Round25DevelopmentSample,
    round25_development_role,
    select_round25_condition_endpoints,
)
from .polymarket_round25_joint_features import (
    POLYMARKET_ROUND25_JOINT_FEATURE_NAMES,
    POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
    Round25JointFeatureSnapshot,
)
from .polymarket_round25_twap_features import (
    POLYMARKET_ROUND25_CONDITION_DURATION_MS,
    POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
)


POLYMARKET_ROUND25_SEQUENCE_MATERIALIZATION_CONTRACT_SHA256 = (
    "3fe3923adc7e638e82adc6db54c84aee0aae135f2baa944c2575191a79327ba9"
)
POLYMARKET_ROUND25_SEQUENCE_BATCH_SCHEMA_VERSION = (
    "polymarket-round25-causal-sequence-condition-batch-v1"
)
POLYMARKET_ROUND25_SEQUENCE_COLLATION_SCHEMA_VERSION = (
    "polymarket-round25-causal-sequence-collation-v1"
)
POLYMARKET_ROUND25_SEQUENCE_INFERENCE_BATCH_SCHEMA_VERSION = (
    "polymarket-round25-target-free-sequence-inference-batch-v1"
)
POLYMARKET_ROUND25_TARGET_FREE_SEQUENCE_INFERENCE_CONTRACT_SHA256 = (
    "d2e0b304a616ba1d819ac2c28af584291bd393c1cd1f84b0d831c3bcfa5d02ff"
)
POLYMARKET_ROUND25_SEQUENCE_ROWS = 64
POLYMARKET_ROUND25_SEQUENCE_CADENCE_MS = 250
POLYMARKET_ROUND25_SEQUENCE_INPUT_WIDTH = (
    len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES) + 1
)
POLYMARKET_ROUND25_AUXILIARY_HORIZONS_MS = (1_000, 5_000)
POLYMARKET_ROUND25_MAXIMUM_CONDITIONS_PER_SEQUENCE_BATCH = 16
POLYMARKET_ROUND25_MAXIMUM_SEQUENCE_TENSOR_BYTES = 9_764_864
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_BINARY_AVAILABILITY_INDICES = tuple(
    index
    for index, name in enumerate(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
    if name.endswith("_available")
)


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


def _array_sha256(value: np.ndarray) -> str:
    selected = np.asarray(value)
    if not selected.flags.c_contiguous:
        selected = np.ascontiguousarray(selected)
    digest = hashlib.sha256()
    digest.update(selected.dtype.str.encode("ascii"))
    digest.update(_canonical_json(list(selected.shape)).encode("ascii"))
    digest.update(selected.tobytes(order="C"))
    return digest.hexdigest()


def _readonly(value: np.ndarray, *, dtype: np.dtype[object]) -> np.ndarray:
    selected = np.asarray(value, dtype=dtype, order="C").copy(order="C")
    selected.setflags(write=False)
    return selected


def _validate_binary_availability(values: np.ndarray) -> None:
    if any(
        not np.all((values[:, index] == 0.0) | (values[:, index] == 1.0))
        for index in _BINARY_AVAILABILITY_INDICES
    ):
        raise ValueError("Round 25 sequence availability feature is not binary")


def round25_feature_transform_sha256(
    center: Sequence[float],
    scale: Sequence[float],
) -> str:
    width = len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
    selected_center = tuple(float(value) for value in center)
    selected_scale = tuple(float(value) for value in scale)
    if (
        len(selected_center) != width
        or len(selected_scale) != width
        or any(
            not math.isfinite(value)
            for value in (*selected_center, *selected_scale)
        )
        or any(value <= 0.0 for value in selected_scale)
        or any(
            selected_center[index] != 0.0 or selected_scale[index] != 1.0
            for index in _BINARY_AVAILABILITY_INDICES
        )
    ):
        raise ValueError("Round 25 sequence feature transform is invalid")
    return _canonical_sha256({
        "center": list(selected_center),
        "control_fit_contract_sha256": (
            POLYMARKET_ROUND25_CONTROL_FIT_CONTRACT_SHA256
        ),
        "feature_names_sha256": POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256,
        "scale": list(selected_scale),
    })


@dataclass(frozen=True, slots=True)
class _Round25CausalSequenceInputs:
    condition_id: str
    event_start_ms: int
    decision_time_ms: np.ndarray
    sequence_values: np.ndarray
    terminal_market_prior: np.ndarray
    endpoint_source_chain_sha256: tuple[str, ...]
    history_source_manifest_sha256: tuple[str, ...]
    source_feature_manifest_sha256: str
    feature_transform_sha256: str


def _build_round25_causal_sequence_inputs(
    *,
    snapshots: Sequence[Round25JointFeatureSnapshot],
    endpoint_rows: Sequence[Round25JointFeatureSnapshot],
    center: Sequence[float],
    scale: Sequence[float],
) -> _Round25CausalSequenceInputs:
    rows = tuple(snapshots)
    endpoints = tuple(endpoint_rows)
    if (
        not rows
        or len(endpoints) != POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION
        or any(
            not isinstance(row, Round25JointFeatureSnapshot) or not row.available
            for row in (*rows, *endpoints)
        )
    ):
        raise ValueError("Round 25 causal sequence source population is invalid")
    ordered_rows = tuple(sorted(rows, key=lambda row: row.decision_time_ms))
    ordered_endpoints = tuple(sorted(endpoints, key=lambda row: row.decision_time_ms))
    condition_id = ordered_endpoints[0].condition_id
    event_start_ms = ordered_endpoints[0].event_start_ms
    if (
        len({row.decision_time_ms for row in ordered_rows}) != len(ordered_rows)
        or len({row.decision_time_ms for row in ordered_endpoints})
        != len(ordered_endpoints)
        or any(
            row.condition_id != condition_id
            or row.event_start_ms != event_start_ms
            or (row.decision_time_ms - event_start_ms)
            % POLYMARKET_ROUND25_SEQUENCE_CADENCE_MS
            for row in ordered_rows
        )
        or any(
            row.condition_id != condition_id or row.event_start_ms != event_start_ms
            for row in ordered_endpoints
        )
    ):
        raise ValueError("Round 25 causal sequence source identity differs")
    lookup = {row.decision_time_ms: row for row in ordered_rows}
    if any(lookup.get(row.decision_time_ms) != row for row in ordered_endpoints):
        raise ValueError("Round 25 causal sequence endpoint identity differs")
    row_indexes = {row.decision_time_ms: index for index, row in enumerate(ordered_rows)}
    matrix = np.asarray([row.values for row in ordered_rows], dtype=np.float64)
    _validate_binary_availability(matrix)
    selected_center = np.asarray(center, dtype=np.float64)
    selected_scale = np.asarray(scale, dtype=np.float64)
    transform_sha256 = round25_feature_transform_sha256(
        selected_center,
        selected_scale,
    )
    normalized = transform_round25_features(
        matrix,
        selected_center,
        selected_scale,
    ).astype(np.float32)
    sequence_values = np.zeros(
        (
            len(ordered_endpoints),
            POLYMARKET_ROUND25_SEQUENCE_ROWS,
            POLYMARKET_ROUND25_SEQUENCE_INPUT_WIDTH,
        ),
        dtype=np.float32,
    )
    priors = np.empty(len(ordered_endpoints), dtype=np.float64)
    history_hashes: list[str] = []
    endpoint_sources: list[str] = []
    for endpoint_index, endpoint in enumerate(ordered_endpoints):
        row_index = row_indexes[endpoint.decision_time_ms]
        history_start = row_index
        while (
            history_start > 0
            and row_index - history_start + 1 < POLYMARKET_ROUND25_SEQUENCE_ROWS
            and ordered_rows[history_start].decision_time_ms
            - ordered_rows[history_start - 1].decision_time_ms
            == POLYMARKET_ROUND25_SEQUENCE_CADENCE_MS
        ):
            history_start -= 1
        selected_history = normalized[history_start : row_index + 1]
        history_length = len(selected_history)
        sequence_values[endpoint_index, -history_length:, :-1] = selected_history
        sequence_values[endpoint_index, -history_length:, -1] = 1.0
        priors[endpoint_index] = endpoint.market_prior_probability
        endpoint_sources.append(endpoint.source_chain_sha256)
        history_hashes.append(_canonical_sha256([
            row.source_chain_sha256
            for row in ordered_rows[history_start : row_index + 1]
        ]))
    return _Round25CausalSequenceInputs(
        condition_id=condition_id,
        event_start_ms=event_start_ms,
        decision_time_ms=_readonly(
            np.asarray(
                [row.decision_time_ms for row in ordered_endpoints],
                dtype="<i8",
            ),
            dtype=np.dtype("<i8"),
        ),
        sequence_values=_readonly(sequence_values, dtype=np.dtype("<f4")),
        terminal_market_prior=_readonly(priors, dtype=np.dtype("<f8")),
        endpoint_source_chain_sha256=tuple(endpoint_sources),
        history_source_manifest_sha256=tuple(history_hashes),
        source_feature_manifest_sha256=_canonical_sha256([
            row.source_chain_sha256 for row in ordered_rows
        ]),
        feature_transform_sha256=transform_sha256,
    )


@dataclass(frozen=True, slots=True)
class Round25SequenceInferenceBatch:
    """Selection-time TCN inputs with no outcome or resolution fields."""

    condition_id: str
    event_start_ms: int
    source_receipt_audit_sha256: str
    decision_time_ms: np.ndarray
    sequence_values: np.ndarray
    terminal_market_prior: np.ndarray
    endpoint_source_chain_sha256: tuple[str, ...]
    history_source_manifest_sha256: tuple[str, ...]
    source_feature_manifest_sha256: str
    feature_transform_sha256: str
    sequence_values_sha256: str
    terminal_inputs_sha256: str
    batch_sha256: str
    schema_version: str = POLYMARKET_ROUND25_SEQUENCE_INFERENCE_BATCH_SCHEMA_VERSION
    role: str = "selection"
    target_free_sequence_contract_sha256: str = (
        POLYMARKET_ROUND25_TARGET_FREE_SEQUENCE_INFERENCE_CONTRACT_SHA256
    )
    selection_target_accessed: bool = False
    official_resolution_accessed: bool = False
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "condition_id": self.condition_id,
            "decision_time_ms": self.decision_time_ms.tolist(),
            "endpoint_source_chain_sha256": list(
                self.endpoint_source_chain_sha256
            ),
            "event_start_ms": self.event_start_ms,
            "feature_transform_sha256": self.feature_transform_sha256,
            "history_source_manifest_sha256": list(
                self.history_source_manifest_sha256
            ),
            "official_resolution_accessed": self.official_resolution_accessed,
            "role": self.role,
            "schema_version": self.schema_version,
            "selection_target_accessed": self.selection_target_accessed,
            "sequence_values_sha256": self.sequence_values_sha256,
            "source_feature_manifest_sha256": (
                self.source_feature_manifest_sha256
            ),
            "source_receipt_audit_sha256": self.source_receipt_audit_sha256,
            "target_free_sequence_contract_sha256": (
                self.target_free_sequence_contract_sha256
            ),
            "terminal_inputs_sha256": self.terminal_inputs_sha256,
            "trading_authority": self.trading_authority,
        }

    def __post_init__(self) -> None:
        count = POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION
        hashes = (
            self.source_receipt_audit_sha256,
            *self.endpoint_source_chain_sha256,
            *self.history_source_manifest_sha256,
            self.source_feature_manifest_sha256,
            self.feature_transform_sha256,
            self.sequence_values_sha256,
            self.terminal_inputs_sha256,
            self.batch_sha256,
        )
        if (
            _CONDITION_ID.fullmatch(self.condition_id) is None
            or type(self.event_start_ms) is not int
            or round25_development_role(self.event_start_ms) != "selection"
            or self.decision_time_ms.shape != (count,)
            or self.decision_time_ms.dtype != np.dtype("<i8")
            or self.decision_time_ms.flags.writeable
            or not np.all(np.diff(self.decision_time_ms) > 0)
            or not np.all(
                (self.decision_time_ms >= self.event_start_ms)
                & (
                    self.decision_time_ms
                    < self.event_start_ms + POLYMARKET_ROUND25_CONDITION_DURATION_MS
                )
            )
            or self.sequence_values.shape
            != (
                count,
                POLYMARKET_ROUND25_SEQUENCE_ROWS,
                POLYMARKET_ROUND25_SEQUENCE_INPUT_WIDTH,
            )
            or self.sequence_values.dtype != np.dtype("<f4")
            or self.sequence_values.flags.writeable
            or not np.all(np.isfinite(self.sequence_values))
            or not np.all(self.sequence_values[:, -1, -1] == 1.0)
            or self.terminal_market_prior.shape != (count,)
            or self.terminal_market_prior.dtype != np.dtype("<f8")
            or self.terminal_market_prior.flags.writeable
            or not np.all(np.isfinite(self.terminal_market_prior))
            or not np.all(
                (self.terminal_market_prior > 0.0)
                & (self.terminal_market_prior < 1.0)
            )
            or len(self.endpoint_source_chain_sha256) != count
            or len(set(self.endpoint_source_chain_sha256)) != count
            or len(self.history_source_manifest_sha256) != count
            or any(_SHA256.fullmatch(value) is None for value in hashes)
            or self.schema_version
            != POLYMARKET_ROUND25_SEQUENCE_INFERENCE_BATCH_SCHEMA_VERSION
            or self.role != "selection"
            or self.target_free_sequence_contract_sha256
            != POLYMARKET_ROUND25_TARGET_FREE_SEQUENCE_INFERENCE_CONTRACT_SHA256
            or self.selection_target_accessed is not False
            or self.official_resolution_accessed is not False
            or self.trading_authority is not False
            or self.sequence_values_sha256 != _array_sha256(self.sequence_values)
            or self.terminal_inputs_sha256
            != _canonical_sha256({
                "decision_time_ms_sha256": _array_sha256(self.decision_time_ms),
                "terminal_market_prior_sha256": _array_sha256(
                    self.terminal_market_prior
                ),
            })
            or self.batch_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 25 target-free sequence inference batch differs")

    def validated(self) -> Round25SequenceInferenceBatch:
        self.__post_init__()
        return self


def build_round25_sequence_inference_batch(
    *,
    snapshots: Sequence[Round25JointFeatureSnapshot],
    center: Sequence[float],
    scale: Sequence[float],
    source_receipt_audit_sha256: str,
) -> Round25SequenceInferenceBatch:
    rows = tuple(snapshots)
    if _SHA256.fullmatch(source_receipt_audit_sha256) is None:
        raise ValueError("Round 25 source receipt audit identity differs")
    endpoints = select_round25_condition_endpoints(rows)
    causal = _build_round25_causal_sequence_inputs(
        snapshots=rows,
        endpoint_rows=endpoints,
        center=center,
        scale=scale,
    )
    if round25_development_role(causal.event_start_ms) != "selection":
        raise ValueError("Round 25 target-free inference requires selection role")
    sequence_sha256 = _array_sha256(causal.sequence_values)
    terminal_sha256 = _canonical_sha256({
        "decision_time_ms_sha256": _array_sha256(causal.decision_time_ms),
        "terminal_market_prior_sha256": _array_sha256(
            causal.terminal_market_prior
        ),
    })
    payload = {
        "condition_id": causal.condition_id,
        "decision_time_ms": causal.decision_time_ms.tolist(),
        "endpoint_source_chain_sha256": list(
            causal.endpoint_source_chain_sha256
        ),
        "event_start_ms": causal.event_start_ms,
        "feature_transform_sha256": causal.feature_transform_sha256,
        "history_source_manifest_sha256": list(
            causal.history_source_manifest_sha256
        ),
        "official_resolution_accessed": False,
        "role": "selection",
        "schema_version": POLYMARKET_ROUND25_SEQUENCE_INFERENCE_BATCH_SCHEMA_VERSION,
        "selection_target_accessed": False,
        "sequence_values_sha256": sequence_sha256,
        "source_feature_manifest_sha256": causal.source_feature_manifest_sha256,
        "source_receipt_audit_sha256": source_receipt_audit_sha256,
        "target_free_sequence_contract_sha256": (
            POLYMARKET_ROUND25_TARGET_FREE_SEQUENCE_INFERENCE_CONTRACT_SHA256
        ),
        "terminal_inputs_sha256": terminal_sha256,
        "trading_authority": False,
    }
    return Round25SequenceInferenceBatch(
        condition_id=causal.condition_id,
        event_start_ms=causal.event_start_ms,
        source_receipt_audit_sha256=source_receipt_audit_sha256,
        decision_time_ms=causal.decision_time_ms,
        sequence_values=causal.sequence_values,
        terminal_market_prior=causal.terminal_market_prior,
        endpoint_source_chain_sha256=causal.endpoint_source_chain_sha256,
        history_source_manifest_sha256=causal.history_source_manifest_sha256,
        source_feature_manifest_sha256=causal.source_feature_manifest_sha256,
        feature_transform_sha256=causal.feature_transform_sha256,
        sequence_values_sha256=sequence_sha256,
        terminal_inputs_sha256=terminal_sha256,
        batch_sha256=_canonical_sha256(payload),
    )


@dataclass(frozen=True, slots=True)
class Round25SequenceConditionBatch:
    role: str
    condition_id: str
    event_start_ms: int
    source_dataset_sha256: str
    resolution_authority_sha256: str
    decision_time_ms: np.ndarray
    sequence_values: np.ndarray
    terminal_labels: np.ndarray
    terminal_market_prior: np.ndarray
    endpoint_weights: np.ndarray
    auxiliary_targets: np.ndarray
    auxiliary_mask: np.ndarray
    endpoint_sample_sha256: tuple[str, ...]
    endpoint_source_chain_sha256: tuple[str, ...]
    history_source_manifest_sha256: tuple[str, ...]
    feature_transform_sha256: str
    sequence_values_sha256: str
    terminal_tensors_sha256: str
    auxiliary_tensors_sha256: str
    batch_sha256: str
    schema_version: str = POLYMARKET_ROUND25_SEQUENCE_BATCH_SCHEMA_VERSION
    feature_schema_version: str = POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION
    feature_names_sha256: str = POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256
    model_design_sha256: str = POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256
    candidate_design_sha256: str = POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256
    candidate_amendment_sha256: str = (
        POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256
    )
    materialization_contract_sha256: str = (
        POLYMARKET_ROUND25_SEQUENCE_MATERIALIZATION_CONTRACT_SHA256
    )
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "auxiliary_tensors_sha256": self.auxiliary_tensors_sha256,
            "candidate_amendment_sha256": self.candidate_amendment_sha256,
            "candidate_design_sha256": self.candidate_design_sha256,
            "condition_id": self.condition_id,
            "decision_time_ms": self.decision_time_ms.tolist(),
            "endpoint_sample_sha256": list(self.endpoint_sample_sha256),
            "endpoint_source_chain_sha256": list(
                self.endpoint_source_chain_sha256
            ),
            "event_start_ms": self.event_start_ms,
            "feature_names_sha256": self.feature_names_sha256,
            "feature_schema_version": self.feature_schema_version,
            "feature_transform_sha256": self.feature_transform_sha256,
            "history_source_manifest_sha256": list(
                self.history_source_manifest_sha256
            ),
            "materialization_contract_sha256": (
                self.materialization_contract_sha256
            ),
            "model_design_sha256": self.model_design_sha256,
            "role": self.role,
            "resolution_authority_sha256": self.resolution_authority_sha256,
            "schema_version": self.schema_version,
            "sequence_values_sha256": self.sequence_values_sha256,
            "source_dataset_sha256": self.source_dataset_sha256,
            "terminal_tensors_sha256": self.terminal_tensors_sha256,
            "trading_authority": self.trading_authority,
        }

    def __post_init__(self) -> None:
        count = POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION
        hashes = (
            *self.endpoint_sample_sha256,
            *self.endpoint_source_chain_sha256,
            *self.history_source_manifest_sha256,
            self.feature_transform_sha256,
            self.source_dataset_sha256,
            self.resolution_authority_sha256,
            self.sequence_values_sha256,
            self.terminal_tensors_sha256,
            self.auxiliary_tensors_sha256,
            self.batch_sha256,
        )
        if (
            self.role not in {"train", "calibration"}
            or _CONDITION_ID.fullmatch(self.condition_id) is None
            or type(self.event_start_ms) is not int
            or self.decision_time_ms.shape != (count,)
            or self.decision_time_ms.dtype != np.dtype("<i8")
            or self.sequence_values.shape
            != (
                count,
                POLYMARKET_ROUND25_SEQUENCE_ROWS,
                POLYMARKET_ROUND25_SEQUENCE_INPUT_WIDTH,
            )
            or self.sequence_values.dtype != np.dtype("<f4")
            or self.terminal_labels.shape != (count,)
            or self.terminal_labels.dtype != np.dtype("<f4")
            or self.terminal_market_prior.shape != (count,)
            or self.terminal_market_prior.dtype != np.dtype("<f8")
            or self.endpoint_weights.shape != (count,)
            or self.endpoint_weights.dtype != np.dtype("<f8")
            or self.auxiliary_targets.shape
            != (count, len(POLYMARKET_ROUND25_AUXILIARY_HORIZONS_MS))
            or self.auxiliary_targets.dtype != np.dtype("<f4")
            or self.auxiliary_mask.shape != self.auxiliary_targets.shape
            or self.auxiliary_mask.dtype != np.dtype("bool")
            or any(
                value.flags.writeable
                for value in (
                    self.decision_time_ms,
                    self.sequence_values,
                    self.terminal_labels,
                    self.terminal_market_prior,
                    self.endpoint_weights,
                    self.auxiliary_targets,
                    self.auxiliary_mask,
                )
            )
            or not np.all(np.isfinite(self.sequence_values))
            or not np.all(np.isfinite(self.terminal_labels))
            or not np.all(np.isin(self.terminal_labels, (0.0, 1.0)))
            or len(np.unique(self.terminal_labels)) != 1
            or not np.all(np.isfinite(self.terminal_market_prior))
            or not np.all(
                (self.terminal_market_prior > 0.0)
                & (self.terminal_market_prior < 1.0)
            )
            or not np.all(
                self.endpoint_weights
                == 1.0 / POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION
            )
            or not np.all(np.isfinite(self.auxiliary_targets))
            or np.any(self.auxiliary_targets[~self.auxiliary_mask] != 0.0)
            or not np.all(self.sequence_values[:, -1, -1] == 1.0)
            or len(self.endpoint_sample_sha256) != count
            or len(self.endpoint_source_chain_sha256) != count
            or len(self.history_source_manifest_sha256) != count
            or any(_SHA256.fullmatch(value) is None for value in hashes)
            or self.schema_version != POLYMARKET_ROUND25_SEQUENCE_BATCH_SCHEMA_VERSION
            or self.feature_schema_version
            != POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION
            or self.feature_names_sha256
            != POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256
            or self.model_design_sha256
            != POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256
            or self.candidate_design_sha256
            != POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256
            or self.candidate_amendment_sha256
            != POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256
            or self.materialization_contract_sha256
            != POLYMARKET_ROUND25_SEQUENCE_MATERIALIZATION_CONTRACT_SHA256
            or self.trading_authority is not False
            or self.sequence_values_sha256 != _array_sha256(self.sequence_values)
            or self.terminal_tensors_sha256
            != _canonical_sha256({
                "decision_time_ms_sha256": _array_sha256(self.decision_time_ms),
                "endpoint_weights_sha256": _array_sha256(self.endpoint_weights),
                "terminal_labels_sha256": _array_sha256(self.terminal_labels),
                "terminal_market_prior_sha256": _array_sha256(
                    self.terminal_market_prior
                ),
            })
            or self.auxiliary_tensors_sha256
            != _canonical_sha256({
                "auxiliary_mask_sha256": _array_sha256(self.auxiliary_mask),
                "auxiliary_targets_sha256": _array_sha256(self.auxiliary_targets),
            })
            or self.batch_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 25 sequence condition batch differs")

    def validated(self) -> Round25SequenceConditionBatch:
        self.__post_init__()
        return self


def build_round25_sequence_condition_batch(
    *,
    snapshots: Sequence[Round25JointFeatureSnapshot],
    endpoint_samples: Sequence[Round25DevelopmentSample],
    center: Sequence[float],
    scale: Sequence[float],
    source_dataset_sha256: str,
    resolution_authority_sha256: str,
) -> Round25SequenceConditionBatch:
    rows = tuple(snapshots)
    endpoints = tuple(endpoint_samples)
    if (
        not rows
        or len(endpoints) != POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION
        or any(
            not isinstance(row, Round25JointFeatureSnapshot) or not row.available
            for row in rows
        )
        or any(not isinstance(sample, Round25DevelopmentSample) for sample in endpoints)
        or _SHA256.fullmatch(source_dataset_sha256) is None
        or _SHA256.fullmatch(resolution_authority_sha256) is None
    ):
        raise ValueError("Round 25 sequence source population is invalid")
    ordered_endpoints = tuple(
        sorted(endpoints, key=lambda sample: sample.decision_time_ms)
    )
    condition_id = ordered_endpoints[0].condition_id
    event_start_ms = ordered_endpoints[0].event_start_ms
    role = ordered_endpoints[0].role
    ordered_rows = tuple(sorted(rows, key=lambda row: row.decision_time_ms))
    lookup = {row.decision_time_ms: row for row in ordered_rows}
    if (
        role not in {"train", "calibration"}
        or len({sample.decision_time_ms for sample in ordered_endpoints})
        != len(ordered_endpoints)
        or any(
            sample.condition_id != condition_id
            or sample.event_start_ms != event_start_ms
            or sample.role != role
            for sample in ordered_endpoints
        )
    ):
        raise ValueError("Round 25 sequence source identity differs")
    endpoint_rows: list[Round25JointFeatureSnapshot] = []
    for sample in ordered_endpoints:
        endpoint_row = lookup.get(sample.decision_time_ms)
        if endpoint_row is None:
            raise ValueError("Round 25 sequence endpoint row is unavailable")
        if (
            endpoint_row.values != sample.feature_values
            or endpoint_row.market_prior_probability
            != sample.market_prior_probability
            or endpoint_row.source_chain_sha256
            != sample.feature_source_chain_sha256
        ):
            raise ValueError("Round 25 sequence endpoint identity differs")
        endpoint_rows.append(endpoint_row)
    causal = _build_round25_causal_sequence_inputs(
        snapshots=ordered_rows,
        endpoint_rows=endpoint_rows,
        center=center,
        scale=scale,
    )
    count = len(ordered_endpoints)
    labels = np.empty(count, dtype=np.float32)
    weights = np.empty(count, dtype=np.float64)
    auxiliary_targets = np.zeros(
        (count, len(POLYMARKET_ROUND25_AUXILIARY_HORIZONS_MS)),
        dtype=np.float32,
    )
    auxiliary_mask = np.zeros_like(auxiliary_targets, dtype=np.bool_)
    endpoint_hashes: list[str] = []
    event_end_ms = event_start_ms + POLYMARKET_ROUND25_CONDITION_DURATION_MS

    for endpoint_index, sample in enumerate(ordered_endpoints):
        endpoint_hashes.append(sample.sample_sha256)
        labels[endpoint_index] = float(sample.target_up)
        weights[endpoint_index] = sample.endpoint_weight
        terminal_logit = float(round25_logit(np.asarray([
            causal.terminal_market_prior[endpoint_index]
        ]))[0])
        for auxiliary_index, horizon_ms in enumerate(
            POLYMARKET_ROUND25_AUXILIARY_HORIZONS_MS
        ):
            future_time = sample.decision_time_ms + horizon_ms
            future_row = lookup.get(future_time)
            if future_time >= event_end_ms or future_row is None:
                continue
            future_prior = future_row.market_prior_probability
            future_logit = float(round25_logit(np.asarray([future_prior]))[0])
            auxiliary_targets[endpoint_index, auxiliary_index] = (
                future_logit - terminal_logit
            )
            auxiliary_mask[endpoint_index, auxiliary_index] = True

    decision_times = causal.decision_time_ms
    sequences = causal.sequence_values
    terminal_labels = _readonly(labels, dtype=np.dtype("<f4"))
    terminal_priors = causal.terminal_market_prior
    endpoint_weights = _readonly(weights, dtype=np.dtype("<f8"))
    auxiliary = _readonly(auxiliary_targets, dtype=np.dtype("<f4"))
    auxiliary_available = _readonly(auxiliary_mask, dtype=np.dtype("bool"))
    sequence_sha256 = _array_sha256(sequences)
    terminal_sha256 = _canonical_sha256({
        "decision_time_ms_sha256": _array_sha256(decision_times),
        "endpoint_weights_sha256": _array_sha256(endpoint_weights),
        "terminal_labels_sha256": _array_sha256(terminal_labels),
        "terminal_market_prior_sha256": _array_sha256(terminal_priors),
    })
    auxiliary_sha256 = _canonical_sha256({
        "auxiliary_mask_sha256": _array_sha256(auxiliary_available),
        "auxiliary_targets_sha256": _array_sha256(auxiliary),
    })
    payload = {
        "auxiliary_tensors_sha256": auxiliary_sha256,
        "candidate_amendment_sha256": POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
        "candidate_design_sha256": POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
        "condition_id": condition_id,
        "decision_time_ms": decision_times.tolist(),
        "endpoint_sample_sha256": endpoint_hashes,
        "endpoint_source_chain_sha256": list(
            causal.endpoint_source_chain_sha256
        ),
        "event_start_ms": event_start_ms,
        "feature_names_sha256": POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256,
        "feature_schema_version": POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
        "feature_transform_sha256": causal.feature_transform_sha256,
        "history_source_manifest_sha256": list(
            causal.history_source_manifest_sha256
        ),
        "materialization_contract_sha256": (
            POLYMARKET_ROUND25_SEQUENCE_MATERIALIZATION_CONTRACT_SHA256
        ),
        "model_design_sha256": POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
        "role": role,
        "resolution_authority_sha256": resolution_authority_sha256,
        "schema_version": POLYMARKET_ROUND25_SEQUENCE_BATCH_SCHEMA_VERSION,
        "sequence_values_sha256": sequence_sha256,
        "source_dataset_sha256": source_dataset_sha256,
        "terminal_tensors_sha256": terminal_sha256,
        "trading_authority": False,
    }
    return Round25SequenceConditionBatch(
        role=role,
        condition_id=condition_id,
        event_start_ms=event_start_ms,
        source_dataset_sha256=source_dataset_sha256,
        resolution_authority_sha256=resolution_authority_sha256,
        decision_time_ms=decision_times,
        sequence_values=sequences,
        terminal_labels=terminal_labels,
        terminal_market_prior=terminal_priors,
        endpoint_weights=endpoint_weights,
        auxiliary_targets=auxiliary,
        auxiliary_mask=auxiliary_available,
        endpoint_sample_sha256=tuple(endpoint_hashes),
        endpoint_source_chain_sha256=causal.endpoint_source_chain_sha256,
        history_source_manifest_sha256=causal.history_source_manifest_sha256,
        feature_transform_sha256=causal.feature_transform_sha256,
        sequence_values_sha256=sequence_sha256,
        terminal_tensors_sha256=terminal_sha256,
        auxiliary_tensors_sha256=auxiliary_sha256,
        batch_sha256=_canonical_sha256(payload),
    )


@dataclass(frozen=True, slots=True)
class Round25SequenceCollation:
    role: str
    condition_ids: tuple[str, ...]
    source_batch_sha256: tuple[str, ...]
    source_dataset_sha256: str
    resolution_authority_sha256: str
    feature_transform_sha256: str
    sequence_values: np.ndarray
    terminal_labels: np.ndarray
    terminal_market_prior: np.ndarray
    endpoint_weights: np.ndarray
    auxiliary_targets: np.ndarray
    auxiliary_mask: np.ndarray
    collation_sha256: str
    schema_version: str = POLYMARKET_ROUND25_SEQUENCE_COLLATION_SCHEMA_VERSION
    trading_authority: bool = False

    def __post_init__(self) -> None:
        rows = len(self.condition_ids) * POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION
        if (
            self.role not in {"train", "calibration"}
            or not 1
            <= len(self.condition_ids)
            <= POLYMARKET_ROUND25_MAXIMUM_CONDITIONS_PER_SEQUENCE_BATCH
            or len(set(self.condition_ids)) != len(self.condition_ids)
            or any(_CONDITION_ID.fullmatch(value) is None for value in self.condition_ids)
            or len(self.source_batch_sha256) != len(self.condition_ids)
            or any(_SHA256.fullmatch(value) is None for value in self.source_batch_sha256)
            or any(
                _SHA256.fullmatch(value) is None
                for value in (
                    self.source_dataset_sha256,
                    self.resolution_authority_sha256,
                    self.feature_transform_sha256,
                )
            )
            or self.sequence_values.shape
            != (
                rows,
                POLYMARKET_ROUND25_SEQUENCE_ROWS,
                POLYMARKET_ROUND25_SEQUENCE_INPUT_WIDTH,
            )
            or self.sequence_values.nbytes
            > POLYMARKET_ROUND25_MAXIMUM_SEQUENCE_TENSOR_BYTES
            or self.sequence_values.dtype != np.dtype("<f4")
            or self.terminal_labels.shape != (rows,)
            or self.terminal_labels.dtype != np.dtype("<f4")
            or self.terminal_market_prior.shape != (rows,)
            or self.terminal_market_prior.dtype != np.dtype("<f8")
            or self.endpoint_weights.shape != (rows,)
            or self.endpoint_weights.dtype != np.dtype("<f8")
            or self.auxiliary_targets.shape
            != (rows, len(POLYMARKET_ROUND25_AUXILIARY_HORIZONS_MS))
            or self.auxiliary_targets.dtype != np.dtype("<f4")
            or self.auxiliary_mask.shape != self.auxiliary_targets.shape
            or self.auxiliary_mask.dtype != np.dtype("bool")
            or any(
                value.flags.writeable
                for value in (
                    self.sequence_values,
                    self.terminal_labels,
                    self.terminal_market_prior,
                    self.endpoint_weights,
                    self.auxiliary_targets,
                    self.auxiliary_mask,
                )
            )
            or not np.all(np.isfinite(self.sequence_values))
            or not np.all(np.isfinite(self.terminal_labels))
            or not np.all(np.isin(self.terminal_labels, (0.0, 1.0)))
            or not np.all(np.isfinite(self.terminal_market_prior))
            or not np.all(
                (self.terminal_market_prior > 0.0)
                & (self.terminal_market_prior < 1.0)
            )
            or not np.all(
                self.endpoint_weights
                == 1.0 / POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION
            )
            or not np.all(np.isfinite(self.auxiliary_targets))
            or np.any(self.auxiliary_targets[~self.auxiliary_mask] != 0.0)
            or not np.all(self.sequence_values[:, -1, -1] == 1.0)
            or self.schema_version
            != POLYMARKET_ROUND25_SEQUENCE_COLLATION_SCHEMA_VERSION
            or self.trading_authority is not False
            or _SHA256.fullmatch(self.collation_sha256) is None
            or self.collation_sha256
            != _canonical_sha256({
                "auxiliary_mask_sha256": _array_sha256(self.auxiliary_mask),
                "auxiliary_targets_sha256": _array_sha256(self.auxiliary_targets),
                "condition_ids": list(self.condition_ids),
                "endpoint_weights_sha256": _array_sha256(self.endpoint_weights),
                "feature_transform_sha256": self.feature_transform_sha256,
                "resolution_authority_sha256": self.resolution_authority_sha256,
                "role": self.role,
                "schema_version": self.schema_version,
                "sequence_values_sha256": _array_sha256(self.sequence_values),
                "source_batch_sha256": list(self.source_batch_sha256),
                "source_dataset_sha256": self.source_dataset_sha256,
                "terminal_labels_sha256": _array_sha256(self.terminal_labels),
                "terminal_market_prior_sha256": _array_sha256(
                    self.terminal_market_prior
                ),
                "trading_authority": False,
            })
        ):
            raise ValueError("Round 25 sequence collation differs")


def collate_round25_sequence_batches(
    batches: Sequence[Round25SequenceConditionBatch],
) -> Round25SequenceCollation:
    selected = tuple(batches)
    if (
        not 1
        <= len(selected)
        <= POLYMARKET_ROUND25_MAXIMUM_CONDITIONS_PER_SEQUENCE_BATCH
        or any(not isinstance(batch, Round25SequenceConditionBatch) for batch in selected)
    ):
        raise ValueError("Round 25 sequence batch selection is invalid")
    for batch in selected:
        batch.validated()
    ordered = tuple(sorted(selected, key=lambda batch: (batch.event_start_ms, batch.condition_id)))
    role = ordered[0].role
    transform_sha256 = ordered[0].feature_transform_sha256
    source_dataset_sha256 = ordered[0].source_dataset_sha256
    resolution_authority_sha256 = ordered[0].resolution_authority_sha256
    if (
        any(batch.role != role for batch in ordered)
        or any(batch.feature_transform_sha256 != transform_sha256 for batch in ordered)
        or any(batch.source_dataset_sha256 != source_dataset_sha256 for batch in ordered)
        or any(
            batch.resolution_authority_sha256 != resolution_authority_sha256
            for batch in ordered
        )
        or len({batch.condition_id for batch in ordered}) != len(ordered)
    ):
        raise ValueError("Round 25 sequence batch identities differ")
    sequences = _readonly(
        np.concatenate([batch.sequence_values for batch in ordered], axis=0),
        dtype=np.dtype("<f4"),
    )
    labels = _readonly(
        np.concatenate([batch.terminal_labels for batch in ordered]),
        dtype=np.dtype("<f4"),
    )
    priors = _readonly(
        np.concatenate([batch.terminal_market_prior for batch in ordered]),
        dtype=np.dtype("<f8"),
    )
    weights = _readonly(
        np.concatenate([batch.endpoint_weights for batch in ordered]),
        dtype=np.dtype("<f8"),
    )
    auxiliary = _readonly(
        np.concatenate([batch.auxiliary_targets for batch in ordered], axis=0),
        dtype=np.dtype("<f4"),
    )
    auxiliary_mask = _readonly(
        np.concatenate([batch.auxiliary_mask for batch in ordered], axis=0),
        dtype=np.dtype("bool"),
    )
    condition_ids = tuple(batch.condition_id for batch in ordered)
    source_hashes = tuple(batch.batch_sha256 for batch in ordered)
    payload = {
        "auxiliary_mask_sha256": _array_sha256(auxiliary_mask),
        "auxiliary_targets_sha256": _array_sha256(auxiliary),
        "condition_ids": list(condition_ids),
        "endpoint_weights_sha256": _array_sha256(weights),
        "feature_transform_sha256": transform_sha256,
        "resolution_authority_sha256": resolution_authority_sha256,
        "role": role,
        "schema_version": POLYMARKET_ROUND25_SEQUENCE_COLLATION_SCHEMA_VERSION,
        "sequence_values_sha256": _array_sha256(sequences),
        "source_batch_sha256": list(source_hashes),
        "source_dataset_sha256": source_dataset_sha256,
        "terminal_labels_sha256": _array_sha256(labels),
        "terminal_market_prior_sha256": _array_sha256(priors),
        "trading_authority": False,
    }
    return Round25SequenceCollation(
        role=role,
        condition_ids=condition_ids,
        source_batch_sha256=source_hashes,
        source_dataset_sha256=source_dataset_sha256,
        resolution_authority_sha256=resolution_authority_sha256,
        feature_transform_sha256=transform_sha256,
        sequence_values=sequences,
        terminal_labels=labels,
        terminal_market_prior=priors,
        endpoint_weights=weights,
        auxiliary_targets=auxiliary,
        auxiliary_mask=auxiliary_mask,
        collation_sha256=_canonical_sha256(payload),
    )


__all__ = [
    "POLYMARKET_ROUND25_AUXILIARY_HORIZONS_MS",
    "POLYMARKET_ROUND25_MAXIMUM_CONDITIONS_PER_SEQUENCE_BATCH",
    "POLYMARKET_ROUND25_MAXIMUM_SEQUENCE_TENSOR_BYTES",
    "POLYMARKET_ROUND25_SEQUENCE_CADENCE_MS",
    "POLYMARKET_ROUND25_SEQUENCE_INFERENCE_BATCH_SCHEMA_VERSION",
    "POLYMARKET_ROUND25_SEQUENCE_INPUT_WIDTH",
    "POLYMARKET_ROUND25_SEQUENCE_MATERIALIZATION_CONTRACT_SHA256",
    "POLYMARKET_ROUND25_SEQUENCE_ROWS",
    "POLYMARKET_ROUND25_TARGET_FREE_SEQUENCE_INFERENCE_CONTRACT_SHA256",
    "Round25SequenceCollation",
    "Round25SequenceConditionBatch",
    "Round25SequenceInferenceBatch",
    "build_round25_sequence_condition_batch",
    "build_round25_sequence_inference_batch",
    "collate_round25_sequence_batches",
    "round25_feature_transform_sha256",
]
