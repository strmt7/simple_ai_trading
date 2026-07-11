from __future__ import annotations

import pytest

from tools.publish_selective_event_discovery import (
    _canonical_payload_hash,
    _funnel_svg,
    _negative_bar_svg,
    _nice_negative_bound,
    _top_score_row,
)
from tools.run_action_value_discovery import _canonical_sha256


def _row(candidate_id: str, value: float) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "horizon_seconds": 300,
        "risk_level": "conservative",
        "score_method": "event_direct_mean",
        "value": value,
    }


def test_negative_chart_is_direct_labeled_accessible_and_escaped() -> None:
    rows = [_row("first", -2.5), _row("second", -10.0)]
    rows[0]["risk_level"] = "conservative<script>"

    svg = _negative_bar_svg(
        rows,
        value_key="value",
        title="Measured < loss",
        subtitle="After-cost evidence",
        footer="No equity curve",
    )

    assert 'role="img"' in svg
    assert "Measured &lt; loss" in svg
    assert "conservative&lt;script&gt;" in svg
    assert "-2.50" in svg
    assert "-10.00" in svg
    assert "No equity curve" in svg
    assert "nan" not in svg.lower()


def test_negative_chart_scale_and_funnel_keep_zero_visible() -> None:
    assert _nice_negative_bound((-50.69, -621.90)) == -800.0
    assert _nice_negative_bound((0.0, 0.0)) == -1.0

    svg = _funnel_svg(fits=6, candidates=18, evaluable=18, accepted=0)

    assert "Verified model fits" in svg
    assert ">6<" in svg
    assert ">18<" in svg
    assert "Positive policy utility" in svg
    assert "trade quotas" in svg


def test_publisher_rejects_ambiguous_rows_and_hash_drift() -> None:
    with pytest.raises(ValueError, match="ambiguous"):
        _top_score_row(
            [
                {"requested_rows": 100},
                {"requested_rows": 100},
            ]
        )

    payload = {"value": 1}
    payload["sha256"] = _canonical_sha256(payload)
    assert _canonical_payload_hash(payload, "sha256") == payload["sha256"]
    payload["value"] = 2
    with pytest.raises(ValueError, match="binding"):
        _canonical_payload_hash(payload, "sha256")
