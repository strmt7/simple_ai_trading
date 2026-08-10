from __future__ import annotations

import hashlib
import json

import numpy as np

from simple_ai_trading.polymarket_round25_candidate_design import (
    POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
    POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
)
from simple_ai_trading.polymarket_round25_controls import (
    POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256,
)
from simple_ai_trading.polymarket_round25_dataset import (
    POLYMARKET_ROUND25_CAMPAIGN_START_MS,
    POLYMARKET_ROUND25_DATASET_SCHEMA_VERSION,
    POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION,
    POLYMARKET_ROUND25_MINIMUM_CONDITIONS,
    POLYMARKET_ROUND25_TRAIN_END_MS,
    Round25DevelopmentDataset,
    Round25DevelopmentSample,
)
from simple_ai_trading.polymarket_round25_joint_features import (
    POLYMARKET_ROUND25_JOINT_FEATURE_NAMES,
    POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
)
from simple_ai_trading.polymarket_round25_sequence import (
    POLYMARKET_ROUND25_AUXILIARY_HORIZONS_MS,
    POLYMARKET_ROUND25_SEQUENCE_BATCH_SCHEMA_VERSION,
    POLYMARKET_ROUND25_SEQUENCE_INPUT_WIDTH,
    POLYMARKET_ROUND25_SEQUENCE_MATERIALIZATION_CONTRACT_SHA256,
    POLYMARKET_ROUND25_SEQUENCE_ROWS,
    Round25SequenceCollation,
    Round25SequenceConditionBatch,
    collate_round25_sequence_batches,
)
from simple_ai_trading.polymarket_round25_twap_features import (
    POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
)


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    selected = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(selected.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(selected.shape), separators=(",", ":")).encode("ascii"))
    digest.update(selected.tobytes(order="C"))
    return digest.hexdigest()


def _readonly(values: np.ndarray, *, dtype: np.dtype[object]) -> np.ndarray:
    result = np.asarray(values, dtype=dtype, order="C")
    result.flags.writeable = False
    return result


def small_round25_dataset(role: str) -> Round25DevelopmentDataset:
    """Build a valid but deliberately minimum-ineligible contract fixture."""

    start = (
        POLYMARKET_ROUND25_CAMPAIGN_START_MS
        if role == "train"
        else POLYMARKET_ROUND25_TRAIN_END_MS + 300_000
    )
    condition_id = "0x" + ("a" if role == "train" else "b") * 64
    resolution_sha256 = ("c" if role == "train" else "d") * 64
    authority_sha256 = "e" * 64
    values = (0.0,) * len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
    samples = []
    for phase in range(4):
        for index in range(4):
            decision = start + phase * 75_000 + (index + 1) * 1_000
            payload = {
                "condition_id": condition_id,
                "decision_time_ms": decision,
                "endpoint_weight": 1.0 / POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION,
                "event_start_ms": start,
                "feature_source_chain_sha256": "f" * 64,
                "feature_values": list(values),
                "market_prior_probability": 0.5,
                "resolution_sha256": resolution_sha256,
                "role": role,
                "target_up": True,
            }
            samples.append(Round25DevelopmentSample(
                role=role,
                condition_id=condition_id,
                event_start_ms=start,
                decision_time_ms=decision,
                feature_values=values,
                market_prior_probability=0.5,
                target_up=True,
                endpoint_weight=1.0 / POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION,
                feature_source_chain_sha256="f" * 64,
                resolution_sha256=resolution_sha256,
                sample_sha256=_canonical_sha256(payload),
            ))
    dataset_payload = {
        "candidate_amendment_sha256": POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
        "candidate_design_sha256": POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
        "condition_count": 1,
        "feature_schema_version": POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
        "minimum_condition_count": POLYMARKET_ROUND25_MINIMUM_CONDITIONS[role],
        "minimum_gate_passed": False,
        "resolution_authority_sha256": authority_sha256,
        "role": role,
        "sample_sha256": [sample.sample_sha256 for sample in samples],
        "schema_version": POLYMARKET_ROUND25_DATASET_SCHEMA_VERSION,
        "trading_authority": False,
    }
    return Round25DevelopmentDataset(
        role=role,
        samples=tuple(samples),
        condition_count=1,
        minimum_condition_count=POLYMARKET_ROUND25_MINIMUM_CONDITIONS[role],
        minimum_gate_passed=False,
        resolution_authority_sha256=authority_sha256,
        dataset_sha256=_canonical_sha256(dataset_payload),
    )


def small_round25_sequence_collation(
    role: str = "train",
) -> Round25SequenceCollation:
    """Build one claim-free condition for tensor-mechanics tests only."""

    return collate_round25_sequence_batches((
        small_round25_sequence_condition_batch(role),
    ))


def small_round25_sequence_condition_batch(
    role: str = "train",
) -> Round25SequenceConditionBatch:
    """Build one hash-valid, minimum-ineligible sequence condition."""

    if role not in {"train", "calibration", "selection"}:
        raise ValueError("Round 25 test sequence role differs")
    condition_id = "0x" + {"train": "a", "calibration": "b", "selection": "c"}[
        role
    ] * 64
    event_start_ms = {"train": 1_000, "calibration": 1_000_000, "selection": 2_000_000}[
        role
    ]
    row_count = POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION
    decision_time_ms = np.asarray(
        [event_start_ms + (index + 1) * 1_000 for index in range(row_count)],
        dtype="<i8",
    )
    sequence_values = np.zeros(
        (
            row_count,
            POLYMARKET_ROUND25_SEQUENCE_ROWS,
            POLYMARKET_ROUND25_SEQUENCE_INPUT_WIDTH,
        ),
        dtype="<f4",
    )
    sequence_values[:, -1, -1] = 1.0
    sequence_values[:, -1, 0] = np.linspace(-0.5, 0.5, row_count, dtype="<f4")
    terminal_labels = np.ones(row_count, dtype="<f4")
    terminal_prior = np.full(row_count, 0.5, dtype="<f8")
    endpoint_weights = np.full(
        row_count,
        1.0 / POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION,
        dtype="<f8",
    )
    auxiliary_targets = np.zeros(
        (row_count, len(POLYMARKET_ROUND25_AUXILIARY_HORIZONS_MS)),
        dtype="<f4",
    )
    auxiliary_mask = np.zeros_like(auxiliary_targets, dtype="bool")
    arrays = tuple(
        _readonly(values, dtype=values.dtype)
        for values in (
            decision_time_ms,
            sequence_values,
            terminal_labels,
            terminal_prior,
            endpoint_weights,
            auxiliary_targets,
            auxiliary_mask,
        )
    )
    (
        decision_time_ms,
        sequence_values,
        terminal_labels,
        terminal_prior,
        endpoint_weights,
        auxiliary_targets,
        auxiliary_mask,
    ) = arrays
    source_dataset_sha256 = "e" * 64
    resolution_authority_sha256 = "f" * 64
    transform_sha256 = "1" * 64
    endpoint_sample_sha256 = tuple(
        hashlib.sha256(f"sample:{condition_id}:{index}".encode("ascii")).hexdigest()
        for index in range(row_count)
    )
    endpoint_source_chain_sha256 = tuple(
        hashlib.sha256(f"source:{condition_id}:{index}".encode("ascii")).hexdigest()
        for index in range(row_count)
    )
    history_source_manifest_sha256 = tuple(
        hashlib.sha256(f"history:{condition_id}:{index}".encode("ascii")).hexdigest()
        for index in range(row_count)
    )
    sequence_values_sha256 = _array_sha256(sequence_values)
    terminal_tensors_sha256 = _canonical_sha256({
        "decision_time_ms_sha256": _array_sha256(decision_time_ms),
        "endpoint_weights_sha256": _array_sha256(endpoint_weights),
        "terminal_labels_sha256": _array_sha256(terminal_labels),
        "terminal_market_prior_sha256": _array_sha256(terminal_prior),
    })
    auxiliary_tensors_sha256 = _canonical_sha256({
        "auxiliary_mask_sha256": _array_sha256(auxiliary_mask),
        "auxiliary_targets_sha256": _array_sha256(auxiliary_targets),
    })
    payload = {
        "auxiliary_tensors_sha256": auxiliary_tensors_sha256,
        "candidate_amendment_sha256": POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
        "candidate_design_sha256": POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
        "condition_id": condition_id,
        "decision_time_ms": decision_time_ms.tolist(),
        "endpoint_sample_sha256": list(endpoint_sample_sha256),
        "endpoint_source_chain_sha256": list(endpoint_source_chain_sha256),
        "event_start_ms": event_start_ms,
        "feature_names_sha256": POLYMARKET_ROUND25_JOINT_FEATURE_NAMES_SHA256,
        "feature_schema_version": POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
        "feature_transform_sha256": transform_sha256,
        "history_source_manifest_sha256": list(history_source_manifest_sha256),
        "materialization_contract_sha256": (
            POLYMARKET_ROUND25_SEQUENCE_MATERIALIZATION_CONTRACT_SHA256
        ),
        "model_design_sha256": POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
        "resolution_authority_sha256": resolution_authority_sha256,
        "role": role,
        "schema_version": POLYMARKET_ROUND25_SEQUENCE_BATCH_SCHEMA_VERSION,
        "sequence_values_sha256": sequence_values_sha256,
        "source_dataset_sha256": source_dataset_sha256,
        "terminal_tensors_sha256": terminal_tensors_sha256,
        "trading_authority": False,
    }
    return Round25SequenceConditionBatch(
        role=role,
        condition_id=condition_id,
        event_start_ms=event_start_ms,
        source_dataset_sha256=source_dataset_sha256,
        resolution_authority_sha256=resolution_authority_sha256,
        decision_time_ms=decision_time_ms,
        sequence_values=sequence_values,
        terminal_labels=terminal_labels,
        terminal_market_prior=terminal_prior,
        endpoint_weights=endpoint_weights,
        auxiliary_targets=auxiliary_targets,
        auxiliary_mask=auxiliary_mask,
        endpoint_sample_sha256=endpoint_sample_sha256,
        endpoint_source_chain_sha256=endpoint_source_chain_sha256,
        history_source_manifest_sha256=history_source_manifest_sha256,
        feature_transform_sha256=transform_sha256,
        sequence_values_sha256=sequence_values_sha256,
        terminal_tensors_sha256=terminal_tensors_sha256,
        auxiliary_tensors_sha256=auxiliary_tensors_sha256,
        batch_sha256=_canonical_sha256(payload),
    )


__all__ = [
    "small_round25_dataset",
    "small_round25_sequence_collation",
    "small_round25_sequence_condition_batch",
]
