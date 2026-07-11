from __future__ import annotations

import pytest

from tools.publish_gross_architecture_screen import (
    _after_cost_svg,
    _canonical_payload_hash,
    _forecast_svg,
    _funnel_svg,
    _research_progress_svg,
    _top_row,
)
from tools.run_action_value_discovery import _canonical_sha256


def _metrics() -> dict[str, object]:
    return {
        "top_rows": [
            {
                "requested_rows": count,
                "rows": count,
                "portfolio_claim": False,
                "mean_signed_gross_bps": 5.0,
                "mean_exact_after_cost_bps": -7.0,
                "signed_gross_positive_rate": 0.65,
                "exact_after_cost_positive_rate": 0.35,
            }
            for count in (100, 500, 1_000)
        ]
    }


def _row(candidate_id: str) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "development_direction_auc": 0.57,
        "development_spearman_ic": 0.11,
        "development_top_500_gross_bps": 5.0,
        "development_top_500_exact_after_cost_bps": -7.0,
    }


def test_top_row_rejects_ambiguity_and_portfolio_claims() -> None:
    metrics = _metrics()
    assert _top_row(metrics, 500)["rows"] == 500

    metrics["top_rows"].append(dict(metrics["top_rows"][1]))
    with pytest.raises(ValueError, match="ambiguous"):
        _top_row(metrics, 500)

    metrics = _metrics()
    metrics["top_rows"][1]["portfolio_claim"] = True
    with pytest.raises(ValueError, match="contract"):
        _top_row(metrics, 500)


def test_charts_are_accessible_direct_labeled_and_escaped() -> None:
    rows = [_row("mlp-bounded-gmadl"), _row("<unsafe>")]

    cost_svg = _after_cost_svg(rows)
    quality_svg = _forecast_svg(rows)
    funnel_svg = _funnel_svg(neural_screened=3, final_models=3, predictive=3)

    assert 'role="img"' in cost_svg
    assert "gross +5.00" in cost_svg
    assert "exact net -7.00" in cost_svg
    assert "&lt;unsafe&gt;" in cost_svg
    assert "Direction AUC" in quality_svg
    assert "Spearman IC" in quality_svg
    assert "0.570" in quality_svg
    assert "Passed top-500" in funnel_svg
    assert "cost gate" in funnel_svg
    assert "Trading" in funnel_svg
    assert "candidates" in funnel_svg
    assert "nan" not in (cost_svg + quality_svg + funnel_svg).lower()


def test_canonical_hash_detects_evidence_drift() -> None:
    payload: dict[str, object] = {"value": 1}
    payload["sha256"] = _canonical_sha256(payload)
    assert _canonical_payload_hash(payload, "sha256") == payload["sha256"]

    payload["value"] = 2
    with pytest.raises(ValueError, match="binding"):
        _canonical_payload_hash(payload, "sha256")


def test_progress_chart_distinguishes_trades_abstention_and_diagnostic() -> None:
    rows = [
        {"round": 7, "executable_trades": 4, "mean_net_bps": -7.6},
        {"round": 8, "executable_trades": 0, "mean_net_bps": ""},
        {
            "round": 13,
            "executable_trades": 0,
            "mean_net_bps": "",
            "best_top_500_exact_after_cost_bps": -6.6,
        },
    ]

    svg = _research_progress_svg(rows)

    assert "executed mean" in svg
    assert "no executable series" in svg
    assert "overlap diagnostic" in svg
    assert "-7.60 bps" in svg
    assert "-6.60 bps diagnostic" in svg
    assert "No executable trades No executable trades" not in svg
