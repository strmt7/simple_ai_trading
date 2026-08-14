from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from simple_ai_trading import polymarket_round25_forensic_model as forensic_model
from simple_ai_trading.polymarket_round25_forensic_model import (
    _Rows,
    _fit_logistic,
    _metrics,
    _predict_logistic,
    _weighted_isotonic,
)
from simple_ai_trading.polymarket_round25_forensic_materialization import (
    POLYMARKET_ROUND25_SALVAGE_CONTRACT_SHA256,
)
from simple_ai_trading.polymarket_round25_forensic_partition import (
    POLYMARKET_ROUND25_FORENSIC_PARTITION_SCHEMA_VERSION,
    partition_round25_forensic_conditions,
)
from simple_ai_trading.polymarket_round25_forensic_resolution import (
    Round25ForensicResolutionTarget,
)
from simple_ai_trading.polymarket_round25_joint_features import (
    POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
    POLYMARKET_ROUND25_JOINT_FEATURE_NAMES,
    Round25JointFeatureSnapshot,
)
from simple_ai_trading.polymarket_round25_twap_features import (
    POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
)


def _rows() -> _Rows:
    condition_ids = tuple(
        f"0x{condition + 1:064x}"
        for condition in range(4)
        for _ in range(16)
    )
    labels = np.repeat(np.asarray([0.0, 1.0, 0.0, 1.0]), 16)
    feature = np.zeros((64, len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)))
    feature[:, 0] = labels * 2.0 - 1.0
    return _Rows(
        role="train",
        condition_ids=condition_ids,
        event_start_ms=np.repeat(np.arange(4, dtype=np.int64) * 300_000 + 300_000, 16),
        decision_time_ms=np.repeat(np.arange(4, dtype=np.int64) * 300_000 + 300_000, 16)
        + np.tile(np.arange(16, dtype=np.int64), 4),
        features=feature,
        prior=np.full(64, 0.5),
        labels=labels,
        source_sha256=("a" * 64,) * 64,
    ).validated(require_labels=True)


def test_logistic_residual_improves_a_real_signal_without_target_shortcuts() -> None:
    rows = _rows()
    intercept, coefficients = _fit_logistic(rows, rows.features, l2=0.1)
    probability = _predict_logistic(rows, rows.features, intercept, coefficients)

    assert _metrics(rows, probability)["condition_equal_log_loss"] < _metrics(
        rows, rows.prior
    )["condition_equal_log_loss"]
    assert coefficients[0] > 0.0


def test_weighted_isotonic_pools_only_monotonicity_violations() -> None:
    x, y = _weighted_isotonic(
        np.asarray([0.1, 0.2, 0.3, 0.4]),
        np.asarray([0.0, 1.0, 0.0, 1.0]),
    )

    assert x == (0.1, 0.2, 0.3, 0.4)
    assert y == (0.0, 0.5, 0.5, 1.0)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_full_fit_freezes_selection_without_selection_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = 1_786_515_300_000
    conditions = tuple(
        (f"0x{index + 1:064x}", start + index * 300_000)
        for index in range(50)
    )
    assigned = partition_round25_forensic_conditions(conditions)
    role_by_id = {condition_id: role for condition_id, _, role in assigned}
    feature_manifest_sha = "1" * 64
    partition_body = {
        "condition_count": 50,
        "conditions": [
            {"condition_id": condition_id, "event_start_ms": event, "role": role}
            for condition_id, event, role in assigned
        ],
        "created_at_ms": start + 51 * 300_000,
        "feature_store_manifest_sha256": feature_manifest_sha,
        "live_trading_authority": False,
        "model_scores_consulted": False,
        "outcomes_consulted": False,
        "paper_trading_authority": False,
        "profitability_claim": False,
        "role_counts": {"train": 29, "calibration": 8, "selection": 9, "purged": 4},
        "salvage_contract_sha256": POLYMARKET_ROUND25_SALVAGE_CONTRACT_SHA256,
        "schema_version": POLYMARKET_ROUND25_FORENSIC_PARTITION_SCHEMA_VERSION,
        "selection_accessed": False,
        "target_accessed": False,
    }
    partition = {
        **partition_body,
        "partition_sha256": _canonical_sha256(partition_body),
    }
    snapshots = []
    targets = []
    for condition_index, (condition_id, event_start_ms) in enumerate(conditions):
        role = role_by_id[condition_id]
        target_up = condition_index % 2 == 1
        for endpoint in range(16):
            decision_time_ms = event_start_ms + 1_000 + endpoint * 18_000
            values = np.zeros(len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES))
            values[3] = 1.0 if target_up else -1.0
            values[38] = 0.30
            values[45] = 10.0
            values[53] = 0.30
            values[60] = 10.0
            twap = "2" * 64
            clob = "3" * 64
            source = _canonical_sha256(
                {
                    "clob_source_chain_sha256": clob,
                    "condition_id": condition_id,
                    "decision_time_ms": decision_time_ms,
                    "feature_schema_version": POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
                    "model_design_sha256": POLYMARKET_ROUND25_TWAP_MODEL_DESIGN_SHA256,
                    "twap_source_chain_sha256": twap,
                }
            )
            snapshots.append(
                Round25JointFeatureSnapshot(
                    condition_id=condition_id,
                    event_start_ms=event_start_ms,
                    decision_time_ms=decision_time_ms,
                    available=True,
                    reasons=(),
                    market_prior_probability=0.5,
                    values=tuple(values),
                    source_chain_sha256=source,
                    twap_source_chain_sha256=twap,
                    clob_source_chain_sha256=clob,
                    maximum_receipt_ms=decision_time_ms,
                )
            )
        if role in {"train", "calibration"}:
            targets.append(
                Round25ForensicResolutionTarget(
                    condition_id=condition_id,
                    event_start_ms=event_start_ms,
                    role=role,
                    target_up=target_up,
                    resolved_at_ms=event_start_ms + 300_001,
                    official_payload_sha256="4" * 64,
                    evidence_sha256=f"{condition_index + 1:064x}",
                )
            )
    monkeypatch.setattr(
        forensic_model,
        "load_round25_joint_endpoint_inputs",
        lambda _path: (
            {"manifest_sha256": feature_manifest_sha},
            {"train": tuple(snapshots), "calibration": (), "selection": ()},
        ),
    )
    fit_claim = {
        "claim_sha256": "5" * 64,
        "feature_store_manifest_sha256": feature_manifest_sha,
        "partition_sha256": partition["partition_sha256"],
        "stage": "fit",
    }
    monkeypatch.setattr(
        forensic_model,
        "load_round25_forensic_resolution_targets",
        lambda _path: (fit_claim, tuple(targets)),
    )

    model_fit, prediction = forensic_model.fit_and_freeze_round25_forensic_models(
        feature_database="unused.duckdb",
        partition_manifest=partition,
        fit_resolution_database="fit.duckdb",
        created_at_ms=start + 52 * 300_000,
    )

    assert model_fit["selected_candidate_id"] == "l2-logistic-residual-v1"
    assert prediction["selection_targets_accessed"] is False
    assert prediction["access_freeze"]["condition_count"] == 9
    assert len(prediction["prediction_rows"]) == 9 * 16
