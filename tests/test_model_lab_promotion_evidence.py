"""Promotion evidence must be affirmative, complete and numerically meaningful."""

from types import SimpleNamespace

import pytest

from simple_ai_trading import model_lab


def _walk_forward():
    return {
        "passed": True,
        "reason": None,
        "fold_count": 3,
        "accepted_folds": 3,
        "worst_score": 0.08,
        "worst_realized_pnl": 1.2,
        "worst_max_drawdown": 0.025,
    }


def _outcome(tmp_path, reports, *, score=0.12):
    suite = SimpleNamespace(
        outcomes=[
            SimpleNamespace(
                objective=f"objective_{i}",
                best_score=score,
                hybrid_profile="base_only",
                walk_forward_gate=_walk_forward(),
                selection_risk=report,
            )
            for i, report in enumerate(reports)
        ],
        total_rows=123,
        objectives_run=[f"objective_{i}" for i in range(len(reports))],
        summary_path=tmp_path / "suite.json",
    )
    accepted = SimpleNamespace(accepted=True, asdict=lambda: {})
    coverage = SimpleNamespace(
        integrity_status="pass", integrity_warnings=(), asdict=lambda: {}
    )
    return model_lab._outcome_from_suite(
        "BTCUSDC",
        suite,
        SimpleNamespace(asdict=lambda: {}),
        accepted,
        tmp_path / "stress.json",
        accepted,
        tmp_path / "robustness.json",
        data_coverage=coverage,
    )


@pytest.mark.parametrize(
    "report",
    [
        None,
        {},
        {"passed": None},
        {"passed": "true"},
        {"passed": 1},
        {"passed": True, "reason": "not_evaluated", "reasons": []},
        {"passed": True, "reason": None, "reasons": ["overfit"]},
        "unknown",
    ],
)
def test_unknown_selection_evidence_cannot_promote(tmp_path, report):
    outcome = _outcome(tmp_path, [report])
    assert outcome.accepted is False
    assert outcome.error == "selection_risk_failed"


def test_one_objective_cannot_supply_another_objectives_evidence(tmp_path):
    passed = {"passed": True, "reason": None, "reasons": []}
    assert _outcome(tmp_path, [passed, None]).accepted is False
    assert _outcome(tmp_path, [passed, passed]).accepted is True


@pytest.mark.parametrize("field", ["worst_score", "worst_realized_pnl"])
def test_nonfinite_walk_forward_values_are_not_performance_evidence(field):
    report = _walk_forward()
    report[field] = float("inf")
    assert model_lab._walk_forward_gate_passed(report) is False


def test_infinite_selected_score_does_not_promote(tmp_path):
    passed = {"passed": True, "reason": None, "reasons": []}
    assert _outcome(tmp_path, [passed], score=float("inf")).accepted is False
