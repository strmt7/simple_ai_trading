from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from simple_ai_trading.polymarket_round25_clob_features import (
    POLYMARKET_ROUND25_CLOB_FEATURE_NAMES,
    Round25ClobFeatureSnapshot,
)
from simple_ai_trading.polymarket_round25_controls import round25_logit
from simple_ai_trading.polymarket_round25_dataset import (
    POLYMARKET_ROUND25_CALIBRATION_END_MS,
    POLYMARKET_ROUND25_CAMPAIGN_START_MS,
    POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION,
    Round25DevelopmentSample,
)
from simple_ai_trading.polymarket_round25_joint_features import (
    POLYMARKET_ROUND25_JOINT_FEATURE_NAMES,
    Round25JointFeatureSnapshot,
    combine_round25_features,
)
from simple_ai_trading.polymarket_round25_sequence import (
    POLYMARKET_ROUND25_AUXILIARY_HORIZONS_MS,
    POLYMARKET_ROUND25_MAXIMUM_SEQUENCE_TENSOR_BYTES,
    POLYMARKET_ROUND25_SEQUENCE_INPUT_WIDTH,
    POLYMARKET_ROUND25_SEQUENCE_INFERENCE_BATCH_SCHEMA_VERSION,
    POLYMARKET_ROUND25_SEQUENCE_MATERIALIZATION_CONTRACT_SHA256,
    POLYMARKET_ROUND25_SEQUENCE_ROWS,
    POLYMARKET_ROUND25_TARGET_FREE_SEQUENCE_INFERENCE_CONTRACT_SHA256,
    build_round25_sequence_condition_batch,
    build_round25_sequence_inference_batch,
    collate_round25_sequence_batches,
    round25_feature_transform_sha256,
)
from simple_ai_trading.polymarket_round25_twap_features import (
    POLYMARKET_ROUND25_TWAP_FEATURE_NAMES,
    Round25TwapFeatureSnapshot,
)


CONDITION_A = "0x" + "a" * 64
CONDITION_B = "0x" + "b" * 64
SOURCE_DATASET_SHA256 = "d" * 64
RESOLUTION_AUTHORITY_SHA256 = "e" * 64


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _snapshot(
    index: int,
    *,
    condition_id: str = CONDITION_A,
    start_ms: int = POLYMARKET_ROUND25_CAMPAIGN_START_MS,
    prior_shift: float = 0.0,
) -> Round25JointFeatureSnapshot:
    decision = start_ms + index * 250
    twap_values = [0.0] * len(POLYMARKET_ROUND25_TWAP_FEATURE_NAMES)
    twap_values[0] = float(index)
    for feature_index, name in enumerate(POLYMARKET_ROUND25_TWAP_FEATURE_NAMES):
        if name.endswith("_available"):
            twap_values[feature_index] = 1.0
    clob_values = [0.0] * len(POLYMARKET_ROUND25_CLOB_FEATURE_NAMES)
    clob_values[0] = float(index) / 100.0
    prior = 0.4 + prior_shift + float(index) / 10_000.0
    twap = Round25TwapFeatureSnapshot(
        condition_id=condition_id,
        event_start_ms=start_ms,
        decision_time_ms=decision,
        available=True,
        reasons=(),
        values=tuple(twap_values),
        source_chain_sha256=_digest(f"twap:{condition_id}:{decision}:{prior_shift}"),
        maximum_receipt_ms=decision,
        opening_value_e18=65_000 * 10**18,
        latest_value_e18=(65_000 * 10**18) + index,
    )
    clob = Round25ClobFeatureSnapshot(
        condition_id=condition_id,
        event_start_ms=start_ms,
        decision_time_ms=decision,
        available=True,
        reasons=(),
        market_prior_probability=prior,
        values=tuple(clob_values),
        source_chain_sha256=_digest(f"clob:{condition_id}:{decision}:{prior_shift}"),
        maximum_receipt_ms=decision,
    )
    return combine_round25_features(twap, clob)


def _sample(
    row: Round25JointFeatureSnapshot,
    *,
    target_up: bool = True,
) -> Round25DevelopmentSample:
    payload = {
        "condition_id": row.condition_id,
        "decision_time_ms": row.decision_time_ms,
        "endpoint_weight": 1.0 / POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION,
        "event_start_ms": row.event_start_ms,
        "feature_source_chain_sha256": row.source_chain_sha256,
        "feature_values": list(row.values),
        "market_prior_probability": row.market_prior_probability,
        "resolution_sha256": "c" * 64,
        "role": "train",
        "target_up": target_up,
    }
    return Round25DevelopmentSample(
        role="train",
        condition_id=row.condition_id,
        event_start_ms=row.event_start_ms,
        decision_time_ms=row.decision_time_ms,
        feature_values=row.values,
        market_prior_probability=row.market_prior_probability,
        target_up=target_up,
        endpoint_weight=1.0 / POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION,
        feature_source_chain_sha256=row.source_chain_sha256,
        resolution_sha256="c" * 64,
        sample_sha256=_canonical_sha256(payload),
    )


def _sources(
    *,
    condition_id: str = CONDITION_A,
    start_ms: int = POLYMARKET_ROUND25_CAMPAIGN_START_MS,
    prior_shift: float = 0.0,
) -> tuple[tuple[Round25JointFeatureSnapshot, ...], tuple[Round25DevelopmentSample, ...]]:
    rows = tuple(
        _snapshot(
            index,
            condition_id=condition_id,
            start_ms=start_ms,
            prior_shift=prior_shift,
        )
        for index in range(100)
    )
    endpoint_indices = tuple(range(4, 20))
    return rows, tuple(_sample(rows[index]) for index in endpoint_indices)


def _transform() -> tuple[np.ndarray, np.ndarray]:
    width = len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
    return np.zeros(width, dtype=np.float64), np.ones(width, dtype=np.float64)


def _build_batch(
    *,
    snapshots: tuple[Round25JointFeatureSnapshot, ...] | list[Round25JointFeatureSnapshot],
    endpoint_samples: tuple[Round25DevelopmentSample, ...],
    center: np.ndarray,
    scale: np.ndarray,
) -> object:
    return build_round25_sequence_condition_batch(
        snapshots=snapshots,
        endpoint_samples=endpoint_samples,
        center=center,
        scale=scale,
        source_dataset_sha256=SOURCE_DATASET_SHA256,
        resolution_authority_sha256=RESOLUTION_AUTHORITY_SHA256,
    )


def test_sequence_contract_is_self_hashed_and_claim_free() -> None:
    path = (
        Path(__file__).parents[1]
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-025-sequence-materialization-contract-v1.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    claimed = contract.pop("contract_sha256")

    assert claimed == POLYMARKET_ROUND25_SEQUENCE_MATERIALIZATION_CONTRACT_SHA256
    assert claimed == _canonical_sha256(contract)
    assert contract["causal_history"]["future_receipt_allowed"] is False
    assert contract["auxiliary_targets"]["masked_placeholder_treated_as_target"] is False
    assert contract["resource_policy"]["temporary_sequence_files_allowed"] is False
    assert contract["truth_state"]["tcn_model_fitted"] is False
    assert contract["truth_state"]["live_authority"] is False


def test_target_free_sequence_contract_is_self_hashed_and_preaccess() -> None:
    path = (
        Path(__file__).parents[1]
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-025-target-free-sequence-inference-contract-v1.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    claimed = contract.pop("contract_sha256")

    assert claimed == POLYMARKET_ROUND25_TARGET_FREE_SEQUENCE_INFERENCE_CONTRACT_SHA256
    assert claimed == _canonical_sha256(contract)
    assert contract["integrity"]["target_bearing_training_batch_selection_role_allowed"] is False
    assert contract["integrity"]["selection_target_access_before_prediction_freeze_allowed"] is False
    assert "target_up" in contract["selection_inference"]["forbidden_inputs"]
    assert contract["truth_state"]["selection_target_accessed"] is False


def test_sequence_materialization_is_causal_masked_and_hash_bound() -> None:
    rows, samples = _sources()
    center, scale = _transform()
    batch = _build_batch(
        snapshots=rows,
        endpoint_samples=samples,
        center=center,
        scale=scale,
    )

    assert batch.sequence_values.shape == (
        16,
        POLYMARKET_ROUND25_SEQUENCE_ROWS,
        POLYMARKET_ROUND25_SEQUENCE_INPUT_WIDTH,
    )
    assert batch.sequence_values.flags.writeable is False
    assert np.count_nonzero(batch.sequence_values[0, :, -1]) == 5
    assert np.all(batch.sequence_values[0, -5:, -1] == 1.0)
    assert batch.sequence_values[0, -1, 0] == 4.0
    assert np.all(batch.terminal_labels == 1.0)
    assert np.all(batch.endpoint_weights == 1.0 / 16.0)
    assert batch.feature_transform_sha256 == round25_feature_transform_sha256(
        center,
        scale,
    )
    assert batch.trading_authority is False
    assert batch.validated() is batch
    with pytest.raises(ValueError, match="batch differs"):
        replace(batch, batch_sha256="f" * 64)


def test_selection_sequence_inference_is_target_free_and_hash_bound() -> None:
    start_ms = POLYMARKET_ROUND25_CALIBRATION_END_MS + 300_000
    rows = tuple(
        _snapshot(phase * 300 + index, start_ms=start_ms)
        for phase in range(4)
        for index in range(20)
    )
    center, scale = _transform()
    batch = build_round25_sequence_inference_batch(
        snapshots=rows,
        center=center,
        scale=scale,
        source_receipt_audit_sha256="9" * 64,
    )

    assert batch.schema_version == POLYMARKET_ROUND25_SEQUENCE_INFERENCE_BATCH_SCHEMA_VERSION
    assert batch.role == "selection"
    assert batch.sequence_values.shape == (
        16,
        POLYMARKET_ROUND25_SEQUENCE_ROWS,
        POLYMARKET_ROUND25_SEQUENCE_INPUT_WIDTH,
    )
    assert batch.sequence_values.flags.writeable is False
    assert batch.terminal_market_prior.flags.writeable is False
    assert batch.selection_target_accessed is False
    assert batch.official_resolution_accessed is False
    assert batch.trading_authority is False
    assert not hasattr(batch, "terminal_labels")
    assert not hasattr(batch, "auxiliary_targets")
    assert not hasattr(batch, "resolution_authority_sha256")
    assert not hasattr(batch, "source_dataset_sha256")
    assert batch.validated() is batch
    with pytest.raises(ValueError, match="target-free sequence inference batch"):
        replace(batch, batch_sha256="f" * 64)


def test_target_bearing_sequence_batch_rejects_selection_role() -> None:
    rows, samples = _sources()
    center, scale = _transform()
    batch = _build_batch(
        snapshots=rows,
        endpoint_samples=samples,
        center=center,
        scale=scale,
    )

    with pytest.raises(ValueError, match="sequence condition batch"):
        replace(batch, role="selection")


def test_future_auxiliary_change_cannot_change_causal_sequence() -> None:
    rows, samples = _sources()
    center, scale = _transform()
    baseline = _build_batch(
        snapshots=rows,
        endpoint_samples=samples,
        center=center,
        scale=scale,
    )
    changed_rows = list(rows)
    endpoint_index = 4
    future_index = endpoint_index + POLYMARKET_ROUND25_AUXILIARY_HORIZONS_MS[1] // 250
    changed_rows[future_index] = _snapshot(future_index, prior_shift=0.1)
    changed = _build_batch(
        snapshots=changed_rows,
        endpoint_samples=samples,
        center=center,
        scale=scale,
    )

    assert np.array_equal(baseline.sequence_values[0], changed.sequence_values[0])
    assert baseline.auxiliary_targets[0, 1] != changed.auxiliary_targets[0, 1]
    expected = float(round25_logit(np.asarray([rows[24].market_prior_probability]))[0])
    expected -= float(round25_logit(np.asarray([rows[4].market_prior_probability]))[0])
    assert baseline.auxiliary_targets[0, 1] == pytest.approx(expected)


def test_history_resets_after_gap_without_bridging() -> None:
    rows, _samples = _sources()
    selected_indices = tuple(range(30, 46))
    samples = tuple(_sample(rows[index]) for index in selected_indices)
    gapped = tuple(row for index, row in enumerate(rows) if index != 29)
    center, scale = _transform()
    batch = _build_batch(
        snapshots=gapped,
        endpoint_samples=samples,
        center=center,
        scale=scale,
    )

    assert np.count_nonzero(batch.sequence_values[0, :, -1]) == 1
    assert batch.sequence_values[0, -1, 0] == 30.0
    assert np.all(batch.sequence_values[0, :-1, :] == 0.0)


def test_missing_auxiliary_target_is_masked_and_excluded() -> None:
    rows, samples = _sources()
    missing_time = samples[-1].decision_time_ms + 1_000
    missing = tuple(row for row in rows if row.decision_time_ms != missing_time)
    center, scale = _transform()
    batch = _build_batch(
        snapshots=missing,
        endpoint_samples=samples,
        center=center,
        scale=scale,
    )

    assert batch.auxiliary_mask[-1, 0] == np.bool_(False)
    assert batch.auxiliary_targets[-1, 0] == 0.0
    assert batch.auxiliary_mask[-1, 1] == np.bool_(True)


def test_endpoint_or_transform_drift_fails_closed() -> None:
    rows, samples = _sources()
    center, scale = _transform()
    conflicting = _sample(_snapshot(4, prior_shift=0.1))
    with pytest.raises(ValueError, match="endpoint identity"):
        _build_batch(
            snapshots=rows,
            endpoint_samples=(conflicting, *samples[1:]),
            center=center,
            scale=scale,
        )
    availability_index = next(
        index
        for index, name in enumerate(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
        if name.endswith("_available")
    )
    invalid_center = center.copy()
    invalid_center[availability_index] = 0.5
    with pytest.raises(ValueError, match="feature transform"):
        _build_batch(
            snapshots=rows,
            endpoint_samples=samples,
            center=invalid_center,
            scale=scale,
        )


def test_collation_is_chronological_read_only_and_memory_bounded() -> None:
    center, scale = _transform()
    rows_a, samples_a = _sources()
    rows_b, samples_b = _sources(
        condition_id=CONDITION_B,
        start_ms=POLYMARKET_ROUND25_CAMPAIGN_START_MS + 300_000,
    )
    first = _build_batch(
        snapshots=rows_a,
        endpoint_samples=samples_a,
        center=center,
        scale=scale,
    )
    second = _build_batch(
        snapshots=rows_b,
        endpoint_samples=samples_b,
        center=center,
        scale=scale,
    )
    collated = collate_round25_sequence_batches((second, first))

    assert collated.condition_ids == (CONDITION_A, CONDITION_B)
    assert collated.sequence_values.shape[0] == 32
    assert collated.sequence_values.flags.writeable is False
    assert collated.sequence_values.nbytes <= POLYMARKET_ROUND25_MAXIMUM_SEQUENCE_TENSOR_BYTES
    assert collated.trading_authority is False
    with pytest.raises(ValueError, match="identities differ"):
        collate_round25_sequence_batches((first, first))
