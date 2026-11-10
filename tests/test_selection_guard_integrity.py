"""Candidate bookkeeping cannot manufacture independent performance evidence."""

from itertools import permutations
import json

import pytest

from simple_ai_trading import training_suite as suite


def _entry(score, selection=None, validation=None):
    return {
        "score": score,
        "selection_score": score if selection is None else selection,
        "validation_score": score if validation is None else validation,
    }


def _report(best, pool, *, trials=None):
    return suite._selection_risk_report(
        best,
        pool,
        base_candidate_count=len(pool) if trials is None else trials,
        local_refinement_candidates=0,
        ensemble_refinement_candidates=0,
        hybrid_rescue_candidates=0,
    )


@pytest.mark.parametrize("values", [(0.00005, 0.00004, 0.00003), (1.0, 0.9, -1.0)])
def test_copies_cannot_reduce_guard_by_sample_size_or_iqr(values):
    original = [_entry(value) for value in values]
    best = original[0]
    reports = [
        _report(best, [*original, *[dict(original[1]) for _ in range(size - 3)]])
        for size in (3, 30, 300, 3000)
    ]
    penalties = [report["trial_penalty"] for report in reports]
    assert penalties == sorted(penalties)
    assert all(report["passed"] is False for report in reports)
    assert reports[-1]["finite_candidate_scores"] == 3000
    assert reports[-1]["effective_trials"] == 3000
    assert reports[-1]["overfit_diagnostics"]["candidate_count"] == 3
    assert reports[-1]["overfit_diagnostics"]["duplicate_score_pairs"] == 2997


def test_copying_a_jointly_losing_panel_profile_cannot_repair_reversal():
    best = _entry(0.9, 0.9, 0.1)
    original = [
        best,
        _entry(0.8, 0.8, 0.7),
        _entry(0.7, 0.7, 0.9),
        _entry(0.69, 0.69, 0.09),
    ]
    for count in (0, 3, 36):
        report = _report(best, [*original, *[dict(original[-1]) for _ in range(count)]])
        proxy = report["overfit_diagnostics"]
        assert proxy["probability_backtest_overfit"] == 1.0
        assert proxy["candidate_count"] == 4
        assert report["passed"] is False


def test_identical_panel_pairs_do_not_supply_required_comparison_profiles():
    best = _entry(1.0)
    report = _report(best, [best, *[dict(best) for _ in range(20)]])
    assert report["passed"] is False
    assert report["overfit_diagnostics"]["candidate_count"] == 1
    assert report["overfit_diagnostics"]["status"] == "skipped"


def test_equivalent_selected_profile_retains_its_rank_without_object_identity():
    best = _entry(1.0)
    report = suite._overfit_diagnostics(best, [dict(best), _entry(0.8), _entry(0.7)])
    assert report["candidate_count"] == 3
    assert report["selected_validation_rank_percentile"] == 1.0


def test_tied_winners_use_worst_opposite_rank_independent_of_input_order():
    best = _entry(2.0, 2.0, 3.0)
    pool = [best, _entry(1.5, 2.0, 0.0), _entry(1.0, 1.0, 1.0)]
    for ordering in permutations(pool):
        report = suite._overfit_diagnostics(best, ordering)
        assert report["probability_backtest_overfit"] == 0.5
        assert report["splits"][0]["winner_out_sample_score"] == 0.0


def test_increasing_declared_search_breadth_never_reduces_penalty():
    best = _entry(1.0)
    pool = [best, _entry(0.99), _entry(0.98)]
    reports = [_report(best, pool, trials=count) for count in (3, 30, 300, 3000)]
    penalties = [report["trial_penalty"] for report in reports]
    assert penalties == sorted(penalties)
    assert all(report["passed"] is True for report in reports)
    assert all(
        report["statistical_significance_qualified"] is False for report in reports
    )
    assert reports[0]["method"] == "monotone_search_score_guard_v1"


def test_nonfinite_rejected_candidates_increase_search_without_adding_observations():
    best = _entry(1.0)
    pool = [best, _entry(0.9), _entry(0.8)]
    before = _report(best, pool)
    after = _report(best, [*pool, {"score": float("nan")}, {"score": float("-inf")}])
    assert after["effective_trials"] > before["effective_trials"]
    assert after["finite_candidate_scores"] == before["finite_candidate_scores"]
    assert after["trial_penalty"] > before["trial_penalty"]


def test_finite_extreme_scores_reject_with_json_safe_evidence():
    best = _entry(1e308)
    report = _report(best, [best, _entry(0.0), _entry(-1e308)])
    assert report["passed"] is False
    assert report["reason"] == "selection_risk_nonfinite_arithmetic"
    assert report["score_range"] is None
    assert report["deflated_score"] is None
    json.dumps(report, allow_nan=False)


def test_extreme_panel_difference_is_unavailable_not_passing_evidence():
    best = _entry(1.0, 1e308, -1e308)
    report = suite._overfit_diagnostics(best, [best, _entry(0.8), _entry(0.7)])
    assert report["passed"] is False
    assert report["status"] == "unavailable"
    assert report["reason"] == "nonfinite_panel_score_difference"
    json.dumps(report, allow_nan=False)


def test_quantile_interpolation_does_not_overflow_opposite_finite_endpoints():
    assert suite._score_quantile([-1e308, 1e308], 0.5) == 0.0
    assert suite._score_quantile([-1e308, 1e308], 0.25) == pytest.approx(-5e307)


@pytest.mark.parametrize("status", [None, "true", "false", 1, {}, []])
def test_proxy_requires_explicit_boolean_pass(monkeypatch, status):
    monkeypatch.setattr(suite, "_overfit_diagnostics", lambda *args: {"passed": status})
    best = _entry(1.0)
    assert _report(best, [best, _entry(0.99), _entry(0.98)])["passed"] is False


def test_no_comparable_proxy_splits_cannot_pass(monkeypatch):
    monkeypatch.setattr(suite, "_mean_panel_score", lambda *args: None)
    best = _entry(1.0)
    report = _report(best, [best, _entry(0.99), _entry(0.98)])
    assert report["passed"] is False
    assert report["overfit_diagnostics"]["reason"] == "no_comparable_cscv_splits"


@pytest.mark.parametrize("value", [True, False, 10**1000])
def test_boolean_or_unrepresentable_selected_score_is_not_numeric_evidence(value):
    report = _report({"score": value}, [_entry(0.9), _entry(0.8)])
    assert report["passed"] is False
    assert report["reason"] == "no_finite_selected_score"


def test_boolean_panel_does_not_supply_a_comparable_profile():
    best = _entry(1.0)
    report = suite._overfit_diagnostics(
        best, [best, _entry(0.9), _entry(0.8, True, 0.8)]
    )
    assert report["passed"] is False
    assert report["candidate_count"] == 2


@pytest.mark.parametrize(
    "count", [-1, True, 1.5, float("nan"), float("inf"), "3", None]
)
def test_invalid_declared_search_count_cannot_be_truncated_or_replaced(count):
    best = _entry(1.0)
    report = suite._selection_risk_report(
        best,
        [best, _entry(0.99), _entry(0.98)],
        base_candidate_count=count,
        local_refinement_candidates=0,
        ensemble_refinement_candidates=0,
        hybrid_rescue_candidates=0,
    )
    assert report["passed"] is False
    assert report["reason"] == "invalid_search_trial_count"
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize(
    "count", [0, -1, True, 1.5, float("nan"), float("inf"), "2", None]
)
def test_explicit_invalid_variant_count_is_not_silently_one_trial(count):
    best = {**_entry(1.0), "internal_variant_count": count}
    report = _report(best, [best, _entry(0.99), _entry(0.98)])
    assert report["passed"] is False
    assert report["reason"] == "invalid_search_trial_count"
    json.dumps(report, allow_nan=False)


def test_large_integer_variant_count_retains_exact_search_breadth():
    best = {**_entry(1.0), "internal_variant_count": 10**1000}
    report = _report(best, [best, _entry(0.99), _entry(0.98)])
    assert report["passed"] is False
    assert report["internal_variants_evaluated"] == 10**1000 + 2
    assert report["effective_trials"] == 10**1000 + 2
    json.dumps(report, allow_nan=False)
