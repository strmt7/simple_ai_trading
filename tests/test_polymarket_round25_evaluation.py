from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from simple_ai_trading.polymarket_round25_candidate_design import (
    POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
    POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
    POLYMARKET_ROUND25_CANDIDATE_IDS,
)
from simple_ai_trading.polymarket_round25_dataset import (
    POLYMARKET_ROUND25_CALIBRATION_END_MS,
    POLYMARKET_ROUND25_DATASET_SCHEMA_VERSION,
    POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION,
    POLYMARKET_ROUND25_MINIMUM_CONDITIONS,
    Round25DevelopmentDataset,
    Round25DevelopmentSample,
)
from simple_ai_trading.polymarket_round25_evaluation import (
    POLYMARKET_ROUND25_PREDICTIVE_EVALUATION_CONTRACT_SHA256,
    Round25SelectionAccessStore,
    create_round25_prediction_panel,
    evaluate_round25_predictive_candidates,
)
from simple_ai_trading.polymarket_round25_joint_features import (
    POLYMARKET_ROUND25_JOINT_FEATURE_NAMES,
    POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
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


def _selection_fixture() -> Round25DevelopmentDataset:
    condition_count = POLYMARKET_ROUND25_MINIMUM_CONDITIONS["selection"]
    values = (0.0,) * len(POLYMARKET_ROUND25_JOINT_FEATURE_NAMES)
    authority_sha256 = "a" * 64
    samples: list[Round25DevelopmentSample] = []
    first_event = POLYMARKET_ROUND25_CALIBRATION_END_MS + 300_000
    endpoint_offsets = tuple(
        phase * 75_000 + offset
        for phase in range(4)
        for offset in (10_000, 25_000, 40_000, 55_000)
    )
    for condition_index in range(condition_count):
        condition_id = "0x" + format(10_000 + condition_index, "064x")
        event_start = first_event + condition_index * 300_000
        target_up = condition_index % 2 == 0
        resolution_sha256 = hashlib.sha256(
            f"resolution:{condition_id}".encode("ascii")
        ).hexdigest()
        for endpoint_index, endpoint_offset in enumerate(endpoint_offsets):
            decision = event_start + endpoint_offset
            prior = 0.5 + 0.06 * math.sin(
                condition_index * 0.37 + endpoint_index * 0.19
            )
            source_sha256 = hashlib.sha256(
                f"feature:{condition_id}:{decision}".encode("ascii")
            ).hexdigest()
            payload = {
                "condition_id": condition_id,
                "decision_time_ms": decision,
                "endpoint_weight": 1.0 / POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION,
                "event_start_ms": event_start,
                "feature_source_chain_sha256": source_sha256,
                "feature_values": list(values),
                "market_prior_probability": prior,
                "resolution_sha256": resolution_sha256,
                "role": "selection",
                "target_up": target_up,
            }
            samples.append(Round25DevelopmentSample(
                role="selection",
                condition_id=condition_id,
                event_start_ms=event_start,
                decision_time_ms=decision,
                feature_values=values,
                market_prior_probability=prior,
                target_up=target_up,
                endpoint_weight=1.0 / POLYMARKET_ROUND25_ENDPOINTS_PER_CONDITION,
                feature_source_chain_sha256=source_sha256,
                resolution_sha256=resolution_sha256,
                sample_sha256=_canonical_sha256(payload),
            ))
    dataset_payload = {
        "candidate_amendment_sha256": POLYMARKET_ROUND25_CANDIDATE_AMENDMENT_SHA256,
        "candidate_design_sha256": POLYMARKET_ROUND25_CANDIDATE_DESIGN_SHA256,
        "condition_count": condition_count,
        "feature_schema_version": POLYMARKET_ROUND25_JOINT_FEATURE_SCHEMA_VERSION,
        "minimum_condition_count": condition_count,
        "minimum_gate_passed": True,
        "resolution_authority_sha256": authority_sha256,
        "role": "selection",
        "sample_sha256": [sample.sample_sha256 for sample in samples],
        "schema_version": POLYMARKET_ROUND25_DATASET_SCHEMA_VERSION,
        "trading_authority": False,
    }
    return Round25DevelopmentDataset(
        role="selection",
        samples=tuple(samples),
        condition_count=condition_count,
        minimum_condition_count=condition_count,
        minimum_gate_passed=True,
        resolution_authority_sha256=authority_sha256,
        dataset_sha256=_canonical_sha256(dataset_payload),
    )


def _prediction_panel(selection: Round25DevelopmentDataset) -> object:
    labels = np.asarray(
        [sample.target_up for sample in selection.samples],
        dtype=np.float64,
    )
    prior = np.asarray(
        [sample.market_prior_probability for sample in selection.samples],
        dtype=np.float64,
    )
    row_index = np.arange(len(labels), dtype=np.float64)

    def informed(base: float, phase: float) -> np.ndarray:
        confidence = np.clip(
            base + 0.015 * np.sin(row_index * 0.013 + phase),
            0.51,
            0.95,
        )
        return np.where(labels == 1.0, confidence, 1.0 - confidence)

    probabilities = {
        "market-prior-v1": prior,
        "phase-isotonic-market-prior-v1": informed(0.58, 0.1),
        "l2-logistic-residual-v1": informed(0.67, 0.2),
        "lightgbm-residual-depth3-v1": informed(0.71, 0.3),
        "lightgbm-residual-depth5-v1": informed(0.74, 0.4),
        "causal-multitask-tcn-residual-v1": informed(0.79, 0.5),
    }
    artifact_hashes = {
        candidate_id: hashlib.sha256(
            f"artifact:{candidate_id}".encode("ascii")
        ).hexdigest()
        for candidate_id in POLYMARKET_ROUND25_CANDIDATE_IDS
    }
    return create_round25_prediction_panel(
        row_condition_ids=tuple(
            sample.condition_id for sample in selection.samples
        ),
        event_start_ms=tuple(sample.event_start_ms for sample in selection.samples),
        decision_time_ms=tuple(
            sample.decision_time_ms for sample in selection.samples
        ),
        feature_source_chain_sha256=tuple(
            sample.feature_source_chain_sha256 for sample in selection.samples
        ),
        market_prior_probability=prior,
        candidate_probabilities=probabilities,
        candidate_source_artifact_sha256=artifact_hashes,
    )


def test_predictive_evaluation_contract_is_self_hashed_and_claim_free() -> None:
    path = (
        Path(__file__).parents[1]
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-025-predictive-evaluation-contract-v1.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    claimed = contract.pop("contract_sha256")

    assert claimed == POLYMARKET_ROUND25_PREDICTIVE_EVALUATION_CONTRACT_SHA256
    assert claimed == _canonical_sha256(contract)
    assert contract["paired_block_bootstrap"]["replicates"] == 10_000
    assert contract["finite_hypothesis_family"]["hypothesis_count"] == 10
    assert all(value is False for value in contract["truth_state"].values())


def test_prediction_panel_is_target_free_read_only_and_hash_bound() -> None:
    selection = _selection_fixture()
    panel = _prediction_panel(selection)

    assert panel.role == "selection"
    assert panel.target_accessed is False
    assert panel.trading_authority is False
    assert not panel.market_prior_probability.flags.writeable
    assert all(
        not prediction.probabilities.flags.writeable
        for prediction in panel.candidate_predictions
    )
    with pytest.raises(ValueError, match="prediction panel differs"):
        replace(panel, panel_sha256="0" * 64)


def test_paired_stepdown_evaluation_is_deterministic_and_claim_bounded(
    tmp_path: Path,
) -> None:
    selection = _selection_fixture()
    panel = _prediction_panel(selection)
    store = Round25SelectionAccessStore(tmp_path / "selection-access.sqlite3")
    store.freeze_prediction_panel(
        panel=panel,
        one_use_claim_sha256="b" * 64,
    )
    receipt = store.consume_target_access(
        panel=panel,
        selection=selection,
    )

    result = evaluate_round25_predictive_candidates(
        panel=panel,
        selection=selection,
        target_access_receipt=receipt,
        target_access_store=store,
    )
    repeated = evaluate_round25_predictive_candidates(
        panel=panel,
        selection=selection,
        target_access_receipt=receipt,
        target_access_store=store,
    )

    assert result.result_sha256 == repeated.result_sha256
    assert result.bootstrap_mean_sha256 == repeated.bootstrap_mean_sha256
    assert len(result.hypotheses) == 10
    assert result.nominated_candidate_id == "causal-multitask-tcn-residual-v1"
    assert result.predictive_gate_passed is True
    assert all(
        hypothesis.passed
        for hypothesis in result.hypotheses
        if hypothesis.candidate_id == result.nominated_candidate_id
    )
    assert result.development_evidence_only is True
    assert result.edge_verified is False
    assert result.profitability_verified is False
    assert result.ai_uplift_verified is False
    assert result.paper_authority is False
    assert result.live_authority is False
    assert result.validated() is result
    reopened = Round25SelectionAccessStore(
        tmp_path / "selection-access.sqlite3"
    )
    assert reopened.load_consumed_receipt(
        panel=panel,
        selection=selection,
    ) == receipt
    with pytest.raises(ValueError, match="consumed selection access differs"):
        evaluate_round25_predictive_candidates(
            panel=panel,
            selection=selection,
            target_access_receipt=replace(
                receipt,
                selection_dataset_sha256="d" * 64,
                receipt_sha256=_canonical_sha256({
                    **receipt.identity_payload(),
                    "selection_dataset_sha256": "d" * 64,
                }),
            ),
            target_access_store=store,
        )
    with pytest.raises(RuntimeError, match="was not frozen"):
        store.consume_target_access(panel=panel, selection=selection)

    with sqlite3.connect(tmp_path / "selection-access.sqlite3") as connection:
        connection.execute(
            "UPDATE round25_selection_event SET event_json = '{}' WHERE sequence = 1"
        )
    with pytest.raises(ValueError, match="event chain differs"):
        store.validate_consumed(
            receipt=receipt,
            panel=panel,
            selection=selection,
        )
