from __future__ import annotations

import pytest

from tools.run_head_coherence_screen import (
    _action_gate_reasons,
    _action_rank,
    _top_row,
)


def _action_metrics(*, gross: float, net: float, active: int = 1_000):
    return {
        "active_rows": active,
        "top_rows": [
            {
                "requested_rows": count,
                "rows": count,
                "mean_signed_gross_bps": gross,
                "mean_exact_after_cost_bps": net,
                "exact_after_cost_positive_rate": 0.6,
            }
            for count in (100, 500, 1_000)
        ],
    }


def _forecast_metrics() -> dict[str, float]:
    return {
        "direction_auc": 0.57,
        "spearman_information_coefficient": 0.11,
        "mean_absolute_error_bps": 7.8,
        "zero_baseline_mae_bps": 7.9,
    }


def _gates() -> dict[str, object]:
    return {
        "minimum_development_direction_auc": 0.5,
        "minimum_development_spearman_ic": 0.0,
        "require_development_mae_better_than_zero": True,
        "minimum_policy_active_rows": 500,
        "minimum_development_active_rows": 500,
        "minimum_policy_top_500_signed_gross_bps": 0.0,
        "minimum_development_top_500_signed_gross_bps": 0.0,
        "minimum_policy_top_500_exact_after_cost_bps": 0.0,
        "minimum_development_top_500_exact_after_cost_bps": 0.0,
    }


def test_action_rank_uses_cost_then_gross_then_hit_rate() -> None:
    first = _action_metrics(gross=5.0, net=-2.0)
    second = _action_metrics(gross=4.0, net=-1.0)

    assert _action_rank(second) > _action_rank(first)
    assert _action_rank(None)[0] < -1.0e90


def test_action_gates_require_policy_and_development_economics() -> None:
    accepted = _action_gate_reasons(
        policy=_action_metrics(gross=13.0, net=1.0),
        development=_action_metrics(gross=14.0, net=2.0),
        development_forecast=_forecast_metrics(),
        gates=_gates(),
    )
    rejected = _action_gate_reasons(
        policy=_action_metrics(gross=-1.0, net=-13.0, active=100),
        development=_action_metrics(gross=5.0, net=-7.0),
        development_forecast=_forecast_metrics(),
        gates=_gates(),
    )

    assert accepted == []
    assert "policy_active_rows_gate_failed" in rejected
    assert "policy_top_500_signed_gross_gate_failed" in rejected
    assert "policy_top_500_exact_after_cost_gate_failed" in rejected
    assert "development_top_500_exact_after_cost_gate_failed" in rejected


def test_action_gates_reject_unevaluable_and_ambiguous_evidence() -> None:
    assert _action_gate_reasons(
        policy=None,
        development=None,
        development_forecast=_forecast_metrics(),
        gates=_gates(),
    ) == ["action_score_has_no_eligible_active_rows"]

    metrics = _action_metrics(gross=1.0, net=1.0)
    metrics["top_rows"].append(dict(metrics["top_rows"][1]))
    with pytest.raises(ValueError, match="ambiguous"):
        _top_row(metrics)
