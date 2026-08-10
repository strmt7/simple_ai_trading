from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_round25_ai_evaluation import (
    POLYMARKET_ROUND25_AI_UPLIFT_CONTRACT_SHA256,
    Round25AIMatchedReplayCondition,
    create_round25_ai_uplift_panel,
    evaluate_round25_ai_uplift,
)
from simple_ai_trading.polymarket_round25_candidate_design import (
    POLYMARKET_ROUND25_CANDIDATE_IDS,
)
from simple_ai_trading.polymarket_round25_evaluation import (
    POLYMARKET_ROUND25_PREDICTIVE_EVALUATION_CONTRACT_SHA256,
    POLYMARKET_ROUND25_PREDICTIVE_RESULT_SCHEMA_VERSION,
    Round25CandidateMetrics,
    Round25PredictiveEvaluationResult,
    Round25PredictiveHypothesis,
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _predictive_result() -> Round25PredictiveEvaluationResult:
    metrics = tuple(
        Round25CandidateMetrics(
            candidate_id=candidate_id,
            condition_equal_log_loss=0.50 + index * 0.01,
            condition_equal_brier_score=0.20 + index * 0.01,
            expected_calibration_error=0.05,
            balanced_accuracy=0.60,
            roc_auc=0.65,
        )
        for index, candidate_id in enumerate(POLYMARKET_ROUND25_CANDIDATE_IDS)
    )
    hypotheses = []
    for candidate_index, candidate_id in enumerate(POLYMARKET_ROUND25_CANDIDATE_IDS[1:]):
        for metric in ("log_loss", "brier_score"):
            passed = candidate_index == 0
            hypotheses.append(Round25PredictiveHypothesis(
                candidate_id=candidate_id,
                metric=metric,
                mean_improvement=0.02 if passed else -0.01,
                bootstrap_standard_error=0.01,
                observed_statistic=2.0 if passed else -1.0,
                adjusted_p_value=0.01 if passed else 1.0,
                step_critical_value=1.5,
                stepdown_lower_bound=0.005 if passed else -0.02,
                passed=passed,
            ))
    prediction_panel_sha256 = _sha("prediction-panel")
    selection_dataset_sha256 = _sha("selection-dataset")
    resolution_authority_sha256 = _sha("resolution-authority")
    target_access_receipt_sha256 = _sha("target-access")
    bootstrap_mean_sha256 = _sha("predictive-bootstrap")
    nominated_candidate_id = POLYMARKET_ROUND25_CANDIDATE_IDS[1]
    values = {
        "ai_uplift_verified": False,
        "bootstrap_mean_sha256": bootstrap_mean_sha256,
        "candidate_metrics": [metric.payload() for metric in metrics],
        "development_evidence_only": True,
        "edge_verified": False,
        "evaluation_contract_sha256": (
            POLYMARKET_ROUND25_PREDICTIVE_EVALUATION_CONTRACT_SHA256
        ),
        "hypotheses": [hypothesis.payload() for hypothesis in hypotheses],
        "live_authority": False,
        "nominated_candidate_id": nominated_candidate_id,
        "paper_authority": False,
        "prediction_panel_sha256": prediction_panel_sha256,
        "predictive_gate_passed": True,
        "profitability_verified": False,
        "resolution_authority_sha256": resolution_authority_sha256,
        "schema_version": POLYMARKET_ROUND25_PREDICTIVE_RESULT_SCHEMA_VERSION,
        "selection_dataset_sha256": selection_dataset_sha256,
        "target_access_receipt_sha256": target_access_receipt_sha256,
    }
    return Round25PredictiveEvaluationResult(
        prediction_panel_sha256=prediction_panel_sha256,
        selection_dataset_sha256=selection_dataset_sha256,
        resolution_authority_sha256=resolution_authority_sha256,
        target_access_receipt_sha256=target_access_receipt_sha256,
        candidate_metrics=metrics,
        hypotheses=tuple(hypotheses),
        bootstrap_mean_sha256=bootstrap_mean_sha256,
        nominated_candidate_id=nominated_candidate_id,
        predictive_gate_passed=True,
        result_sha256=_canonical_sha256(values),
    ).validated()


def _row(index: int, *, violation: bool = False) -> Round25AIMatchedReplayCondition:
    intervention = index < 100 or violation
    control_return = -0.004 if index < 100 else 0.0002
    ai_return = 0.0 if index < 100 else control_return
    return Round25AIMatchedReplayCondition(
        condition_id="0x" + _sha(f"uplift-condition-{index}"),
        event_start_ms=(index + 2) * 300_000,
        selected_candidate_id=POLYMARKET_ROUND25_CANDIDATE_IDS[1],
        selected_model_prediction_sha256=_sha(f"prediction-{index}"),
        deterministic_decision_sha256=_sha(f"decision-{index}"),
        matched_execution_scenario_sha256=_sha(f"execution-{index}"),
        resolution_authority_sha256=_sha("resolution-authority"),
        control_trace_sha256=_sha(f"control-trace-{index}"),
        ai_trace_sha256=_sha(f"ai-trace-{index}"),
        ai_advisory_sha256=_sha(f"advisory-{index}"),
        control_after_cost_return=control_return,
        ai_after_cost_return=ai_return,
        valid_model_response=not violation,
        schema_or_coherence_violation=violation,
        ai_intervened=intervention,
        ai_veto_new_entries=intervention,
        ai_size_multiplier=0.0 if intervention else 1.0,
        ai_cooldown_ms=0,
        ai_response_latency_ms=None if violation else 1500.0,
    )


def _panel(*, violation: bool = False):
    rows = [_row(index, violation=violation and index == 499) for index in range(500)]
    return create_round25_ai_uplift_panel(
        predictive_result=_predictive_result(),
        selection_conditions=(("0x" + _sha("selection-condition"), 0),),
        rows=rows,
    )


def test_round25_ai_uplift_contract_is_self_hashed_and_non_authoritative() -> None:
    path = (
        Path(__file__).parents[1]
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-025-ai-uplift-evaluation-contract-v1.json"
    )
    contract = json.loads(path.read_text(encoding="ascii"))
    claimed = contract.pop("contract_sha256")

    assert claimed == POLYMARKET_ROUND25_AI_UPLIFT_CONTRACT_SHA256
    assert claimed == _canonical_sha256(contract)
    assert contract["population"]["minimum_conditions"] == 500
    assert contract["population"]["selection_condition_reuse_allowed"] is False
    assert contract["statistical_test"]["bootstrap_replicates"] == 10_000
    assert contract["interpretation"]["live_authority"] is False


def test_round25_ai_matched_rows_are_hash_bound_and_panels_are_disjoint() -> None:
    row = _row(0)
    with pytest.raises(ValueError, match="condition hash"):
        replace(row, ai_after_cost_return=0.1)

    with pytest.raises(ValueError, match="not disjoint"):
        create_round25_ai_uplift_panel(
            predictive_result=_predictive_result(),
            selection_conditions=((row.condition_id, 0),),
            rows=(row,),
        )


def test_round25_ai_uplift_gate_uses_full_preregistered_bootstrap() -> None:
    result = evaluate_round25_ai_uplift(_panel())

    assert result.condition_count == 500
    assert result.intervention_count == 100
    assert result.valid_response_ratio == 1.0
    assert result.paired_mean_after_cost_return_delta_ci_lower > 0.0
    assert result.expected_shortfall_95_delta_ci_upper <= 0.0
    assert result.maximum_drawdown_delta <= 0.0
    assert result.development_uplift_gate_passed is True
    assert result.gate_reasons == ()
    payload = result.identity_payload()
    assert payload["development_evidence_only"] is True
    assert payload["ai_uplift_verified"] is False
    assert payload["profitability_verified"] is False
    assert payload["paper_authority"] is False
    assert payload["live_authority"] is False
    assert payload["orders_submitted"] is False


def test_round25_ai_schema_violation_prevents_development_nomination() -> None:
    result = evaluate_round25_ai_uplift(_panel(violation=True))

    assert result.schema_or_coherence_violation_count == 1
    assert result.development_uplift_gate_passed is False
    assert "schema_or_coherence_violation_observed" in result.gate_reasons
