from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from tools.publish_action_value_discovery import _canonical_sha256
from tools.publish_daily_walkforward_screen import (
    _economics_svg,
    _forecast_svg,
    _funnel_svg,
    _validate_report_identity,
)


def _row(phase: str, candidate: str, day: str, net_bps: float) -> dict[str, object]:
    return {
        "phase": phase,
        "candidate_id": candidate,
        "evaluation_day": day,
        "least_bad_calibration_total_net_bps": net_bps,
        "evaluation_direction_auc": 0.56,
        "evaluation_spearman_ic": 0.08,
    }


def test_publication_charts_are_accessible_parseable_and_truthfully_labeled() -> None:
    rows = [
        _row("policy", "rolling-10d", "2023-06-26", -15.0),
        _row("policy", "rolling-25d", "2023-06-26", -20.0),
        _row("policy", "expanding-half-life-7d", "2023-06-26", -25.0),
        _row("development", "rolling-10d", "2023-07-01", -30.0),
    ]

    economics = _economics_svg(rows)
    forecast = _forecast_svg(rows)
    funnel = _funnel_svg(
        model_fits=21,
        threshold_traces=84,
        accepted_thresholds=0,
        evaluation_trades=0,
        research_candidates=0,
    )

    for chart in (economics, forecast, funnel):
        ET.fromstring(chart)
        assert 'role="img"' in chart
        assert "nan" not in chart.lower()
    assert "-15.00" in economics
    assert "calibration traces, not evaluation trades" in economics
    assert "Direction AUC" in forecast
    assert "2023-07-01" in forecast
    assert ">84<" in funnel
    assert "zero evaluation trades is an abstention result" in funnel


def test_report_identity_uses_canonical_self_hash_and_rejects_tampering() -> None:
    design_sha256 = "d" * 64
    report: dict[str, object] = {
        "schema_version": "daily-walk-forward-screen-report-v1",
        "round": 15,
        "design_sha256": design_sha256,
        "status": "rejected",
        "trading_authority": False,
        "execution_claim": False,
        "profitability_claim": False,
        "portfolio_claim": False,
        "leverage_applied": False,
        "terminal_holdout_accessed": False,
        "development_window_is_consumed": True,
    }
    report["report_sha256"] = _canonical_sha256(report)

    assert (
        _validate_report_identity(report, design_sha256=design_sha256)
        == report["report_sha256"]
    )

    report["status"] = "candidate"
    with pytest.raises(ValueError, match="canonical SHA-256 mismatch"):
        _validate_report_identity(report, design_sha256=design_sha256)
