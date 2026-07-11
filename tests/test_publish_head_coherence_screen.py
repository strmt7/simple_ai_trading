from __future__ import annotations

import pytest

from tools.publish_head_coherence_screen import (
    _economics_svg,
    _forecast_svg,
    _funnel_svg,
    _top_row,
)


def _row(model_id: str, method: str, policy: float, development: float):
    return {
        "model_id": model_id,
        "score_method": method,
        "policy_active_rows": 1_000,
        "development_active_rows": 1_000,
        "policy_top_500_exact_after_cost_bps": policy,
        "development_top_500_exact_after_cost_bps": development,
        "policy_top_500_gross_bps": policy + 12.0,
        "development_top_500_gross_bps": development + 12.0,
        "development_direction_auc": 0.57,
        "development_spearman_ic": 0.11,
        "development_head_agreement_ratio": 0.8,
    }


def test_publication_charts_are_accessible_labeled_and_escaped() -> None:
    rows = [
        _row("mlp-huber-direction", "mean", -13.0, -7.0),
        _row("mlp-gmadl-coherence-025", "mean", -14.0, -8.0),
        _row("lightgbm-gross-baseline", "mean", -18.0, -9.0),
    ]

    economics = _economics_svg(rows)
    forecast = _forecast_svg(rows)
    funnel = _funnel_svg(action_rows=rows)

    assert 'role="img"' in economics
    assert "policy -13.00" in economics
    assert "dev -7.00" in economics
    assert "Coherence loss did not improve" in forecast
    assert "Head agreement" in forecast
    assert "Positive" in funnel
    assert "policy gross" in funnel
    assert "Research" in funnel
    assert "candidates" in funnel
    assert "nan" not in (economics + forecast + funnel).lower()


def test_top_row_rejects_ambiguous_and_portfolio_evidence() -> None:
    rows = [
        {
            "requested_rows": count,
            "rows": count,
            "mean_signed_gross_bps": 1.0,
            "mean_exact_after_cost_bps": -11.0,
            "exact_after_cost_positive_rate": 0.4,
            "overlapping_forecasts": True,
            "portfolio_claim": False,
        }
        for count in (100, 500, 1_000)
    ]
    metrics = {"top_rows": rows}
    assert _top_row(metrics, 500)["rows"] == 500

    rows.append(dict(rows[1]))
    with pytest.raises(ValueError, match="ambiguous"):
        _top_row(metrics, 500)

    rows.pop()
    rows[1]["portfolio_claim"] = True
    with pytest.raises(ValueError, match="contract"):
        _top_row(metrics, 500)
