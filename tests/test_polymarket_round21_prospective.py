from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import simple_ai_trading.polymarket_round21_prospective as prospective_module
from simple_ai_trading.polymarket_round21_core_features import (
    POLYMARKET_ROUND21_FEATURE_SCHEMA,
)
from simple_ai_trading.polymarket_round21_dataset import Round21CausalFeatureRow
from simple_ai_trading.polymarket_round21_model import Round21ProbabilityBatch
from simple_ai_trading.polymarket_round21_prospective import (
    Round21ProspectiveScorer,
    build_round21_inference_panel,
    validate_round21_prospective_prediction,
)
from simple_ai_trading.polymarket_round21_sealed import (
    Round21SealedEvaluationResult,
)

from polymarket_round21_support import sha


START_MS = 1_800_000_000_000
CONDITION = "0x" + "9" * 64
MODEL_SHA = sha("round21-prospective-model")
SEALED_SHA = sha("round21-prospective-sealed")


def _row(
    offset_ms: int,
    *,
    condition_id: str = CONDITION,
    spot_available: bool = True,
    usdm_available: bool = True,
) -> Round21CausalFeatureRow:
    decision = START_MS + offset_ms
    schema = POLYMARKET_ROUND21_FEATURE_SCHEMA
    return Round21CausalFeatureRow.create(
        condition_id=condition_id,
        event_start_ms=START_MS,
        decision_time_ms=decision,
        structural_probability=0.51,
        market_prior_probability=0.49,
        core_values=(0.0,) * len(schema.core_names),
        spot_values=(
            (0.0,) * len(schema.spot_names)
            if spot_available
            else (0.0,) * len(schema.spot_names)
        ),
        usdm_values=(
            (0.0,) * len(schema.usdm_names)
            if usdm_available
            else (0.0,) * len(schema.usdm_names)
        ),
        spot_available=spot_available,
        usdm_available=usdm_available,
        feature_schema=schema,
        core_source_chain_sha256=sha(f"core-{condition_id}-{offset_ms}"),
        spot_source_chain_sha256=(
            sha(f"spot-{condition_id}-{offset_ms}") if spot_available else sha("")
        ),
        usdm_source_chain_sha256=(
            sha(f"usdm-{condition_id}-{offset_ms}") if usdm_available else sha("")
        ),
        core_maximum_receipt_ms=decision,
        spot_maximum_receipt_ms=decision if spot_available else 0,
        usdm_maximum_receipt_ms=decision if usdm_available else 0,
    )


def _artifact() -> dict[str, object]:
    schema = POLYMARKET_ROUND21_FEATURE_SCHEMA
    return {
        "artifact_sha256": MODEL_SHA,
        "dataset_and_partition": {
            "core_feature_names_sha256": schema.core_names_sha256,
            "spot_feature_names_sha256": schema.spot_names_sha256,
            "usdm_feature_names_sha256": schema.usdm_names_sha256,
        },
    }


def _sealed(*, layer: str = "core", accepted: bool = True) -> Mock:
    result = Mock(spec=Round21SealedEvaluationResult)
    result.validated.return_value = result
    result.candidate_accepted = accepted
    result.selected_population_layer = layer
    result.predictive = SimpleNamespace(model_artifact_sha256=MODEL_SHA)
    result.result_sha256 = SEALED_SHA
    return result


def _install_boundaries(monkeypatch: pytest.MonkeyPatch) -> Mock:
    predict = Mock()

    def probability_batch(population_layer, panel):
        indices = np.arange(len(panel.condition_ids), dtype=np.int64)
        values = np.linspace(0.52, 0.56, len(indices), dtype=np.float64)
        return Round21ProbabilityBatch.create(
            population_layer=population_layer,
            selected_candidate_id="selected",
            contributing_candidate_ids=("selected", "challenger"),
            indices=indices,
            probability_up=values,
            lower_up=values - 0.02,
            upper_up=values + 0.02,
            feature_support_eligible=np.ones(len(indices), dtype=np.bool_),
            source_model_artifact_sha256=MODEL_SHA,
            feature_batch_sha256=panel.feature_batch_sha256,
        ).validated()

    predict.side_effect = probability_batch

    def compile_predictor(_artifact, *, population_layer):
        return SimpleNamespace(
            artifact_sha256=MODEL_SHA,
            tcn_training_backend_kind="cpu",
            tcn_training_backend_device="cpu",
            tcn_runtime_backend_kind="cpu",
            tcn_runtime_backend_device="cpu",
            tcn_backend_substituted=False,
            tcn_accelerator_fallback=False,
            core_feature_names_sha256=(
                POLYMARKET_ROUND21_FEATURE_SCHEMA.core_names_sha256
            ),
            spot_feature_names_sha256=(
                POLYMARKET_ROUND21_FEATURE_SCHEMA.spot_names_sha256
            ),
            usdm_feature_names_sha256=(
                POLYMARKET_ROUND21_FEATURE_SCHEMA.usdm_names_sha256
            ),
            predict=lambda panel: predict(population_layer, panel),
        )

    monkeypatch.setattr(
        prospective_module,
        "compile_round21_probability_predictor",
        compile_predictor,
    )
    return predict


def test_inference_panel_is_target_free_contiguous_and_hash_bound() -> None:
    rows = (_row(1_000), _row(1_250))
    panel = build_round21_inference_panel(rows)

    assert not hasattr(panel, "labels")
    assert panel.target_accessed is False
    assert panel.trading_authority is False
    assert panel.condition_ids.tolist() == [CONDITION, CONDITION]
    assert all(
        not array.flags.writeable
        for array in (
            panel.condition_ids,
            panel.decision_time_ms,
            panel.core_features,
            panel.spot_features,
            panel.usdm_features,
        )
    )
    assert panel.source_dataset_sha256 != rows[0].row_sha256
    with pytest.raises(ValueError, match="not contiguous"):
        build_round21_inference_panel((_row(1_000), _row(1_500)))


def test_scorer_emits_idempotent_target_free_probability_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predict = _install_boundaries(monkeypatch)
    ticks = iter((100, 140, 200, 250))
    scorer = Round21ProspectiveScorer(
        artifact=_artifact(),
        sealed_result=_sealed(),
        monotonic_ns=lambda: next(ticks),
    )
    row = _row(1_000)

    result = scorer.evaluate(row, observed_at_ms=row.decision_time_ms + 5)
    duplicate = scorer.evaluate(row, observed_at_ms=row.decision_time_ms + 20)

    assert duplicate is result
    assert predict.call_count == 1
    assert result.status == "observed"
    assert result.reset_reason == "initial"
    assert result.inference_latency_ns == 40
    assert result.source_causal_row_sha256 == row.row_sha256
    assert result.envelope is not None
    assert float(result.envelope.probability_up) == pytest.approx(0.52)
    serialized = result.asdict()
    assert serialized["prediction_sha256"] == result.prediction_sha256
    assert serialized["probability_evidence"]["evidence_sha256"] == (
        result.envelope.evidence_sha256
    )
    assert "target" not in serialized
    assert not any(
        (
            result.target_accessed,
            result.credentials_used,
            result.account_connected,
            result.binance_execution_connected,
            result.grants_execution_authority,
            result.profitability_claim,
            result.paper_trading_authority,
            result.live_trading_authority,
        )
    )
    with pytest.raises(ValueError, match="prediction differs"):
        replace(result, live_trading_authority=True).validated()

    restored = validate_round21_prospective_prediction(serialized)
    assert restored == result
    tampered = {**serialized, "live_trading_authority": True}
    with pytest.raises(ValueError, match="prediction differs"):
        validate_round21_prospective_prediction(tampered)
    tampered_evidence = dict(serialized["probability_evidence"])
    tampered_evidence["lower_up"] = "0.51"
    with pytest.raises(ValueError, match="probability evidence differs"):
        validate_round21_prospective_prediction(
            {**serialized, "probability_evidence": tampered_evidence}
        )


def test_scorer_resets_gaps_and_abstains_when_selected_layer_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predict = _install_boundaries(monkeypatch)
    ticks = iter(range(100, 1_000, 10))
    scorer = Round21ProspectiveScorer(
        artifact=_artifact(),
        sealed_result=_sealed(layer="core_spot"),
        monotonic_ns=lambda: next(ticks),
    )
    first = _row(1_000)
    second = _row(1_250, spot_available=False, usdm_available=False)
    third = _row(1_750)

    assert (
        scorer.evaluate(
            first,
            observed_at_ms=first.decision_time_ms,
        ).status
        == "observed"
    )
    abstain = scorer.evaluate(second, observed_at_ms=second.decision_time_ms)
    resumed = scorer.evaluate(third, observed_at_ms=third.decision_time_ms)

    assert abstain.status == "abstain"
    assert abstain.reason == "selected_optional_feature_layer_unavailable"
    assert abstain.envelope is None
    assert abstain.history_row_count == 2
    assert validate_round21_prospective_prediction(abstain.asdict()) == abstain
    invalid_abstention = {
        **abstain.asdict(),
        "probability_evidence_sha256": MODEL_SHA,
    }
    with pytest.raises(ValueError, match="probability evidence differs"):
        validate_round21_prospective_prediction(invalid_abstention)
    assert resumed.status == "observed"
    assert resumed.reset_reason == "cadence_gap"
    assert resumed.history_row_count == 1
    assert predict.call_count == 2


def test_scorer_rejects_unaccepted_or_mismatched_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_boundaries(monkeypatch)
    with pytest.raises(ValueError, match="was not accepted"):
        Round21ProspectiveScorer(
            artifact=_artifact(),
            sealed_result=_sealed(accepted=False),
        )
    mismatched = _sealed()
    mismatched.predictive = SimpleNamespace(model_artifact_sha256=sha("other"))
    with pytest.raises(ValueError, match="model and sealed result differ"):
        Round21ProspectiveScorer(
            artifact=_artifact(),
            sealed_result=mismatched,
        )


def test_scorer_rejects_conflicting_duplicate_and_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_boundaries(monkeypatch)
    ticks = iter(range(100, 1_000, 10))
    scorer = Round21ProspectiveScorer(
        artifact=_artifact(),
        sealed_result=_sealed(),
        monotonic_ns=lambda: next(ticks),
    )
    first = _row(1_000)
    scorer.evaluate(first, observed_at_ms=first.decision_time_ms)
    conflicting = replace(first, structural_probability=0.52)
    with pytest.raises(ValueError, match="causal feature row differs"):
        scorer.evaluate(conflicting, observed_at_ms=conflicting.decision_time_ms)
    later = _row(1_250)
    scorer.evaluate(later, observed_at_ms=later.decision_time_ms)
    with pytest.raises(ValueError, match="chronology regressed"):
        scorer.evaluate(first, observed_at_ms=later.decision_time_ms)
