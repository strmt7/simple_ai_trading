from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
import pytest

import simple_ai_trading.impact_absorption_event_epistemic_evaluation as subject
from simple_ai_trading.impact_absorption_event_epistemic_policy import (
    Round74EpistemicActionFilter,
    evaluate_round74_epistemic_action_replay_challenge,
    fit_round74_epistemic_action_filter,
)
from simple_ai_trading.impact_absorption_event_sequence import (
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_QUANTILES,
    ROUND74_EVENT_PAYOFF_SIDES,
    ROUND74_EVENT_SYMBOLS,
)


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _evaluation_batch(
    run_ordinal: int,
    *,
    rows: int = 300,
    both_outcome_classes: bool = True,
) -> subject.Round74EpistemicEvaluationBatch:
    action_shape = (
        rows,
        len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
        len(ROUND74_EVENT_PAYOFF_SIDES),
    )
    quantile_shape = (*action_shape, len(ROUND74_EVENT_PAYOFF_QUANTILES))
    regime_shape = action_shape[:2]
    row_offset = (run_ordinal - 1) * rows
    uncertainty_row = (
        np.arange(row_offset + 1, row_offset + rows + 1, dtype=np.float64)
        / (6 * rows + 1)
        * 0.20
    )
    uncertainty_action = np.broadcast_to(
        uncertainty_row.reshape(rows, 1, 1),
        action_shape,
    ).copy()
    uncertainty_regime = np.broadcast_to(
        uncertainty_row.reshape(rows, 1),
        regime_shape,
    ).copy()
    if both_outcome_classes:
        long_positive = np.arange(rows) % 2 == 0
    else:
        long_positive = np.zeros(rows, dtype=np.bool_)
    payoff = np.empty(action_shape, dtype=np.float64)
    payoff[:, :, 0] = np.where(long_positive[:, None], 1.0, -1.0)
    payoff[:, :, 1] = -payoff[:, :, 0]
    adverse = (payoff <= 0.0).astype(np.float64)
    payoff_quantiles = np.broadcast_to(
        payoff[..., None],
        quantile_shape,
    ).copy()
    payoff_quantiles -= uncertainty_action[..., None]
    adverse_excursion = np.ones(action_shape, dtype=np.float64)
    adverse_quantiles = np.ones(quantile_shape, dtype=np.float64)
    adverse_quantiles -= uncertainty_action[..., None]
    positive_target = (payoff > 0.0).astype(np.float64)
    positive_probability = np.where(
        positive_target == 1.0,
        1.0 - uncertainty_action,
        uncertainty_action,
    )
    adverse_probability = np.where(
        adverse == 1.0,
        1.0 - uncertainty_action,
        uncertainty_action,
    )
    if both_outcome_classes:
        high_unpredictability = np.arange(rows) % 2 == 1
    else:
        high_unpredictability = np.zeros(rows, dtype=np.bool_)
    unpredictability = np.broadcast_to(
        np.where(high_unpredictability, 0.75, 0.25).reshape(rows, 1),
        regime_shape,
    ).copy()
    unpredictability_probability = np.where(
        unpredictability >= 0.5,
        unpredictability - uncertainty_regime,
        unpredictability + uncertainty_regime,
    )
    result = subject.Round74EpistemicEvaluationBatch(
        batch_sha256=f"{run_ordinal:064x}",
        model_output_sha256=f"{run_ordinal + 100:064x}",
        probability_calibration_sha256="a" * 64,
        tuning_subpartition_sha256="b" * 64,
        run_id=tuple(f"{run_ordinal:032x}" for _ in range(rows)),
        symbol=tuple(
            ROUND74_EVENT_SYMBOLS[row % len(ROUND74_EVENT_SYMBOLS)]
            for row in range(rows)
        ),
        net_payoff_bps=_readonly(payoff),
        maximum_adverse_excursion_bps=_readonly(adverse_excursion),
        adverse_selection=_readonly(adverse),
        regime_unpredictability=_readonly(unpredictability),
        action_eligibility=_readonly(np.ones(action_shape, dtype=np.float64)),
        regime_unpredictability_eligibility=_readonly(
            np.ones(regime_shape, dtype=np.float64)
        ),
        payoff_quantiles_bps=_readonly(payoff_quantiles),
        maximum_adverse_excursion_quantiles_bps=_readonly(adverse_quantiles),
        positive_payoff_probability=_readonly(positive_probability),
        adverse_selection_probability=_readonly(adverse_probability),
        regime_unpredictability_probability=_readonly(unpredictability_probability),
        payoff_quantile_peer_dispersion_bps=_readonly(uncertainty_action.copy()),
        adverse_excursion_quantile_peer_dispersion_bps=_readonly(
            uncertainty_action.copy()
        ),
        positive_payoff_probability_peer_dispersion=_readonly(
            uncertainty_action.copy()
        ),
        adverse_selection_probability_peer_dispersion=_readonly(
            uncertainty_action.copy()
        ),
        regime_unpredictability_probability_peer_dispersion=_readonly(
            uncertainty_regime.copy()
        ),
        peer_count=3,
    )
    result.validate()
    return result


def test_tie_safe_curve_never_splits_equal_uncertainty() -> None:
    curve = subject._tie_safe_curve(  # noqa: SLF001
        np.asarray([0.1, 0.1, 0.2, 0.3], dtype=np.float64),
        np.asarray([0.0, 2.0, 3.0, 4.0], dtype=np.float64),
    )

    assert curve.accepted_rows.tolist() == [2, 3, 4]
    assert curve.coverage.tolist() == [0.5, 0.75, 1.0]
    points = subject._curve_points(curve)  # noqa: SLF001
    assert points[0].target_coverage == 0.05
    assert points[0].attained_coverage == 0.5
    assert points[0].accepted_rows == 2


def test_epistemic_report_requires_every_conditional_stratum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "ROUND74_EPISTEMIC_BOOTSTRAP_SAMPLES", 200)
    batches = tuple(_evaluation_batch(index) for index in range(1, 7))
    run_ids = tuple(f"{index:032x}" for index in range(1, 7))

    report = subject.evaluate_round74_epistemic_risk_coverage(
        batches,
        expected_policy_selection_run_ids=run_ids,
    )

    assert len(report.metrics) == 221
    assert report.missing_required_strata == ()
    assert report.required_strata_complete is True
    assert report.aggregate_ordering_supported is True
    assert report.conditional_ordering_supported is True
    assert report.policy_challenge_eligible is True
    assert all(metric.ordering_supported for metric in report.metrics)
    payload = json.loads(json.dumps(report.as_dict()))
    assert subject.Round74EpistemicRiskCoverageReport.from_dict(payload) == report
    payload["metrics"][0]["curve_points"][0]["accepted_rows"] = True
    unsigned = dict(payload)
    unsigned.pop("report_sha256")
    payload["report_sha256"] = subject._canonical_sha256(unsigned)  # noqa: SLF001
    with pytest.raises(ValueError, match="accepted rows integer differs"):
        subject.Round74EpistemicRiskCoverageReport.from_dict(payload)


def test_epistemic_action_filter_is_profile_specific_and_target_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "ROUND74_EPISTEMIC_BOOTSTRAP_SAMPLES", 200)
    batches = tuple(_evaluation_batch(index) for index in range(1, 7))
    report = subject.evaluate_round74_epistemic_risk_coverage(
        batches,
        expected_policy_selection_run_ids=tuple(
            f"{index:032x}" for index in range(1, 7)
        ),
    )

    conservative = fit_round74_epistemic_action_filter(
        batches,
        report,
        profile="conservative",
    )
    regular = fit_round74_epistemic_action_filter(
        batches,
        report,
        profile="regular",
    )
    aggressive = fit_round74_epistemic_action_filter(
        batches,
        report,
        profile="aggressive",
    )

    assert conservative.component_quantile == pytest.approx(0.95)
    assert regular.component_quantile == pytest.approx(0.97)
    assert aggressive.component_quantile == pytest.approx(0.99)
    assert np.all(conservative.action_thresholds <= regular.action_thresholds)
    assert np.all(regular.action_thresholds <= aggressive.action_thresholds)
    assert np.all(conservative.regime_thresholds <= regular.regime_thresholds)
    assert np.all(regular.regime_thresholds <= aggressive.regime_thresholds)
    payload = json.loads(json.dumps(conservative.as_dict()))
    assert Round74EpistemicActionFilter.from_dict(payload).filter_sha256 == (
        conservative.filter_sha256
    )
    with pytest.raises(ValueError, match="replay ordering gate differs"):
        evaluate_round74_epistemic_action_replay_challenge(
            (),
            (),
            (),
            risk_coverage_report=report,
            action_filter=replace(
                conservative,
                risk_coverage_report_sha256="f" * 64,
            ),
            baseline_policy=None,  # type: ignore[arg-type]
            execution_panel=None,  # type: ignore[arg-type]
        )

    target_mutated = (
        replace(
            batches[0],
            net_payoff_bps=_readonly(-batches[0].net_payoff_bps),
            adverse_selection=_readonly(1.0 - batches[0].adverse_selection),
            regime_unpredictability=_readonly(
                1.0 - batches[0].regime_unpredictability
            ),
        ),
        *batches[1:],
    )
    target_invariant = fit_round74_epistemic_action_filter(
        target_mutated,
        report,
        profile="conservative",
    )
    assert target_invariant.filter_sha256 == conservative.filter_sha256


def test_missing_outcome_classes_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(subject, "ROUND74_EPISTEMIC_BOOTSTRAP_SAMPLES", 200)
    batches = tuple(
        _evaluation_batch(index, both_outcome_classes=False) for index in range(1, 7)
    )

    report = subject.evaluate_round74_epistemic_risk_coverage(
        batches,
        expected_policy_selection_run_ids=tuple(
            f"{index:032x}" for index in range(1, 7)
        ),
    )

    assert report.missing_required_strata
    assert report.required_strata_complete is False
    assert report.conditional_ordering_supported is False
    assert report.policy_challenge_eligible is False
    assert report.as_dict()["policy_effects"] == {
        "candidate_eligibility_changed": False,
        "candidate_ranking_changed": False,
        "position_size_changed": False,
        "leverage_changed": False,
        "automatic_policy_gate_enabled": False,
    }
    with pytest.raises(ValueError, match="action-filter source differs"):
        fit_round74_epistemic_action_filter(
            batches,
            report,
            profile="conservative",
        )
