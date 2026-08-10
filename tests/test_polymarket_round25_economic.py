from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_round21_comparison import (
    round21_replay_matrix_sha256,
)
from simple_ai_trading.polymarket_round21_replay import (
    Round21EconomicMatrixAccumulator,
)
from simple_ai_trading.polymarket_round25_candidate_design import (
    POLYMARKET_ROUND25_CANDIDATE_IDS,
)
from simple_ai_trading.polymarket_round25_economic import (
    POLYMARKET_ROUND25_ECONOMIC_REPLAY_CONTRACT_SHA256,
    Round25DevelopmentEconomicResult,
    Round25EconomicLedgerSummary,
    build_round25_primary_condition_series,
    build_round25_probability_envelopes,
    load_round25_economic_result,
    write_round25_economic_result,
)
from simple_ai_trading.polymarket_round25_evaluation import (
    POLYMARKET_ROUND25_GATE_METRICS,
    Round25CandidateMetrics,
    Round25PredictiveEvaluationResult,
    Round25PredictiveHypothesis,
)
from polymarket_round21_support import round21_replay_condition
from test_polymarket_round25_prediction import _prepared_prediction


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


def _predictive_result(
    *,
    nominated: bool,
) -> tuple[object, Round25PredictiveEvaluationResult]:
    prepared = _prepared_prediction()
    metrics = tuple(
        Round25CandidateMetrics(
            candidate_id=candidate_id,
            condition_equal_log_loss=0.70 - index * 0.05,
            condition_equal_brier_score=0.35 - index * 0.03,
            expected_calibration_error=0.12 - index * 0.01,
            balanced_accuracy=0.55 + index * 0.05,
            roc_auc=0.55 + index * 0.05,
        )
        for index, candidate_id in enumerate(POLYMARKET_ROUND25_CANDIDATE_IDS)
    )
    selected_id = POLYMARKET_ROUND25_CANDIDATE_IDS[-1]
    hypotheses = tuple(
        Round25PredictiveHypothesis(
            candidate_id=candidate_id,
            metric=metric,
            mean_improvement=0.1 if nominated and candidate_id == selected_id else -0.1,
            bootstrap_standard_error=0.1,
            observed_statistic=1.0 if nominated and candidate_id == selected_id else -1.0,
            adjusted_p_value=0.01 if nominated and candidate_id == selected_id else 0.5,
            step_critical_value=1.0,
            stepdown_lower_bound=0.01 if nominated and candidate_id == selected_id else -0.1,
            passed=nominated and candidate_id == selected_id,
        )
        for candidate_id in POLYMARKET_ROUND25_CANDIDATE_IDS[1:]
        for metric in POLYMARKET_ROUND25_GATE_METRICS
    )
    nominated_candidate_id = selected_id if nominated else None
    values = {
        "ai_uplift_verified": False,
        "bootstrap_mean_sha256": "d" * 64,
        "candidate_metrics": [metric.payload() for metric in metrics],
        "development_evidence_only": True,
        "edge_verified": False,
        "evaluation_contract_sha256": (
            Round25PredictiveEvaluationResult.__dataclass_fields__[  # type: ignore[index]
                "evaluation_contract_sha256"
            ].default
        ),
        "hypotheses": [hypothesis.payload() for hypothesis in hypotheses],
        "live_authority": False,
        "nominated_candidate_id": nominated_candidate_id,
        "paper_authority": False,
        "prediction_panel_sha256": prepared.panel.panel_sha256,
        "predictive_gate_passed": nominated,
        "profitability_verified": False,
        "resolution_authority_sha256": "b" * 64,
        "schema_version": Round25PredictiveEvaluationResult.__dataclass_fields__[  # type: ignore[index]
            "schema_version"
        ].default,
        "selection_dataset_sha256": "a" * 64,
        "target_access_receipt_sha256": "c" * 64,
    }
    result = Round25PredictiveEvaluationResult(
        prediction_panel_sha256=prepared.panel.panel_sha256,
        selection_dataset_sha256="a" * 64,
        resolution_authority_sha256="b" * 64,
        target_access_receipt_sha256="c" * 64,
        candidate_metrics=metrics,
        hypotheses=hypotheses,
        bootstrap_mean_sha256="d" * 64,
        nominated_candidate_id=nominated_candidate_id,
        predictive_gate_passed=nominated,
        result_sha256=_canonical_sha256(values),
    ).validated()
    return prepared, result


def _economic_result() -> Round25DevelopmentEconomicResult:
    accumulator = Round21EconomicMatrixAccumulator()
    accumulator.observe(round21_replay_condition())
    matrix = accumulator.finish()
    summaries = tuple(
        Round25EconomicLedgerSummary.from_replay(value) for value in matrix
    )
    provisional = Round25DevelopmentEconomicResult(
        terminal_transport_manifest_sha256="1" * 64,
        terminal_receipt_audit_sha256="2" * 64,
        feature_store_manifest_sha256="3" * 64,
        resolution_store_manifest_sha256="4" * 64,
        resolution_authority_sha256="5" * 64,
        prepared_prediction_sha256="6" * 64,
        predictive_result_sha256="7" * 64,
        nominated_candidate_id=POLYMARKET_ROUND25_CANDIDATE_IDS[-1],
        candidate_source_artifact_sha256="8" * 64,
        candidate_probability_batch_sha256="9" * 64,
        source_condition_set_sha256="a" * 64,
        source_condition_count=1,
        ledger_summaries=summaries,
        primary_condition_series=build_round25_primary_condition_series(matrix),
        source_replay_matrix_sha256=round21_replay_matrix_sha256(matrix),
        development_economic_gate_passed=False,
        result_sha256="0" * 64,
    )
    return replace(
        provisional,
        result_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def test_round25_economic_contract_is_self_hashed_and_claim_free() -> None:
    path = (
        Path(__file__).parents[1]
        / "docs"
        / "model-research"
        / "polymarket"
        / "round-025-economic-replay-contract-v1.json"
    )
    contract = json.loads(path.read_text(encoding="utf-8"))
    claimed = contract.pop("contract_sha256")

    assert claimed == POLYMARKET_ROUND25_ECONOMIC_REPLAY_CONTRACT_SHA256
    assert claimed == _canonical_sha256(contract)
    assert contract["execution_matrix"]["ledger_count"] == 81
    assert contract["source_replay"]["post_nomination_full_receipt_scan_count"] == 1
    assert contract["graph_source_evidence"][
        "one_point_per_resolved_condition_and_profile"
    ] is True
    assert contract["graph_source_evidence"][
        "manual_graph_value_edits_allowed"
    ] is False
    assert all(value is False for value in contract["truth_state"].values())


def test_round25_envelopes_use_only_the_frozen_nominated_batch() -> None:
    prepared, result = _predictive_result(nominated=True)

    grouped = build_round25_probability_envelopes(
        prepared_prediction=prepared,
        predictive_result=result,
    )

    assert len(grouped) == 400
    assert all(len(values) == 16 for values in grouped.values())
    first = grouped[prepared.panel.row_condition_ids[0]][0]
    candidate = prepared.panel.candidate_predictions[-1]
    assert first.probability_up == first.lower_up == first.upper_up
    assert float(first.probability_up) == candidate.probabilities[0]
    assert first.source_model_artifact_sha256 == candidate.source_artifact_sha256
    assert first.source_probability_batch_sha256 == candidate.probabilities_sha256
    assert first.trading_authority is False

    no_nomination_prepared, no_nomination = _predictive_result(nominated=False)
    with pytest.raises(RuntimeError, match="nomination gate is closed"):
        build_round25_probability_envelopes(
            prepared_prediction=no_nomination_prepared,
            predictive_result=no_nomination,
        )


def test_round25_economic_result_round_trip_is_hash_and_authority_bound(
    tmp_path: Path,
) -> None:
    result = _economic_result()
    path = tmp_path / "round25-economic-result.json"

    assert write_round25_economic_result(path, result) == path
    assert load_round25_economic_result(path) == result
    assert write_round25_economic_result(path, result) == path
    assert len(result.ledger_summaries) == 81
    assert len(result.primary_condition_series) == 3
    assert {
        value.profile for value in result.primary_condition_series
    } == {"conservative", "regular", "aggressive"}
    assert result.development_economic_gate_passed is False
    assert result.edge_verified is False
    assert result.profitability_verified is False
    assert result.paper_trading_authority is False
    assert result.live_trading_authority is False
    assert result.orders_submitted is False

    payload = json.loads(path.read_text(encoding="ascii"))
    payload["live_trading_authority"] = True
    payload["result_sha256"] = _canonical_sha256({
        key: value for key, value in payload.items() if key != "result_sha256"
    })
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="ascii")
    with pytest.raises(ValueError, match="development economic result differs"):
        load_round25_economic_result(path)


def test_round25_economic_result_rejects_independent_matrix_substitution() -> None:
    result = _economic_result()
    tampered = replace(result, source_replay_matrix_sha256="f" * 64)
    tampered = replace(
        tampered,
        result_sha256=_canonical_sha256(tampered.identity_payload()),
    )

    with pytest.raises(ValueError, match="development economic result differs"):
        tampered.validated()


def test_round25_economic_result_rejects_rehashed_graph_series_drift() -> None:
    result = _economic_result()
    original = result.primary_condition_series[0]
    point = replace(
        original,
        cumulative_net_pnl_quote=original.cumulative_net_pnl_quote + 1,
        point_sha256="0" * 64,
    )
    point = replace(
        point,
        point_sha256=_canonical_sha256(point.identity_payload()),
    ).validated()
    tampered = replace(
        result,
        primary_condition_series=(point, *result.primary_condition_series[1:]),
        result_sha256="0" * 64,
    )
    tampered = replace(
        tampered,
        result_sha256=_canonical_sha256(tampered.identity_payload()),
    )

    with pytest.raises(ValueError, match="development economic result differs"):
        tampered.validated()
