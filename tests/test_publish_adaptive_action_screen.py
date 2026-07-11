from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from tools.publish_adaptive_action_screen import (
    _barrier_svg,
    _funnel_svg,
    _forecast_svg,
    _gate_summary,
    _progress_identity,
    _publication_narrative,
    _research_progress_svg,
    _tail_svg,
)


def _forecast_rows() -> list[dict[str, object]]:
    output = []
    for role, start, end in (
        ("calibration", "2023-06-21", "2023-06-25"),
        ("policy", "2023-06-26", "2023-06-30"),
    ):
        for side, auc, top_100, top_500 in (
            ("long", 0.638, -10.95, -10.66),
            ("short", 0.601, -7.90, -13.49),
        ):
            output.append(
                {
                    "role": role,
                    "scenario": "stress",
                    "side": side,
                    "start_date": start,
                    "end_date": end,
                    "auc": auc,
                    "top_100_mean_net_bps": top_100,
                    "top_500_mean_net_bps": top_500,
                }
            )
    return output


def test_round16_charts_are_accessible_parseable_and_truthfully_labeled() -> None:
    profiles = [
        {
            "profile": "conservative",
            "calibration_eligible_rows": 0,
            "policy_eligible_rows": 2,
        },
        {
            "profile": "regular",
            "calibration_eligible_rows": 0,
            "policy_eligible_rows": 3,
        },
        {
            "profile": "aggressive",
            "calibration_eligible_rows": 0,
            "policy_eligible_rows": 13,
        },
    ]
    barrier_rows = [
        {
            "scenario": scenario,
            "side": side,
            "positive_ratio": 0.21,
            "horizon": 150,
            "stop": 50,
            "take": 20,
            "ambiguous_stop": 0,
            "protection_gap_stop": 1 if scenario == "stress" else 0,
        }
        for scenario in ("base", "stress")
        for side in ("long", "short")
    ]
    progress = [
        {
            "round": round_number,
            "mean_net_bps": "",
            "best_top_500_exact_after_cost_bps": -6.0,
            "executable_trades": 0,
        }
        for round_number in range(7, 17)
    ]
    charts = (
        _forecast_svg(_forecast_rows()),
        _tail_svg(_forecast_rows()),
        _funnel_svg(profiles),
        _barrier_svg(barrier_rows),
        _research_progress_svg(progress),
    )

    for chart in charts:
        root = ET.fromstring(chart)
        assert root.attrib["role"] == "img"
        assert root.find("{http://www.w3.org/2000/svg}title") is not None
        assert root.find("{http://www.w3.org/2000/svg}desc") is not None
        assert ">nan<" not in chart.lower()
        assert '="nan"' not in chart.lower()
    assert "Every displayed mean is negative" in charts[1]
    assert "action" + " funnel" not in charts[2].lower()
    assert "pre-trade risk controls" in charts[2].lower()
    assert "No candidate threshold passed all pre-trade risk controls" in charts[2]
    assert "2023-06-21" in charts[2]
    assert "Rounds 15 through 16" in charts[4]


def test_round17_titles_and_extreme_tail_label_have_clearance() -> None:
    rows = _forecast_rows()
    rows[-1]["top_100_mean_net_bps"] = -21.99
    tail = ET.fromstring(_tail_svg(rows, round_number=17))
    namespace = "{http://www.w3.org/2000/svg}"
    texts = [node for node in tail.findall(f"{namespace}text") if node.text]
    value_label = next(node for node in texts if node.text == "-21.99")
    category_label = next(node for node in texts if node.text == "Out-of-sample short")

    assert float(value_label.attrib["y"]) + 20.0 < float(category_label.attrib["y"])
    assert tail.find(f"{namespace}title").text.startswith("Round 17")
    assert "Round 17" in _forecast_svg(rows, round_number=17)
    assert "Round 17" in _funnel_svg(
        [
            {
                "profile": profile,
                "calibration_eligible_rows": 0,
                "policy_eligible_rows": 0,
            }
            for profile in ("conservative", "regular", "aggressive")
        ],
        round_number=17,
    )


def test_round18_gate_summary_preserves_nonzero_candidates() -> None:
    rows = [
        {
            "profile": "conservative",
            "calibration_eligible_rows": 0,
            "calibration_threshold_candidates": 0,
            "calibration_threshold_accepted": False,
            "policy_trades": 0,
            "development_evaluated": False,
        },
        {
            "profile": "regular",
            "calibration_eligible_rows": 0,
            "calibration_threshold_candidates": 0,
            "calibration_threshold_accepted": False,
            "policy_trades": 0,
            "development_evaluated": False,
        },
        {
            "profile": "aggressive",
            "calibration_eligible_rows": 24,
            "calibration_threshold_candidates": 4,
            "calibration_threshold_accepted": False,
            "policy_trades": 0,
            "development_evaluated": False,
        },
    ]

    thresholds = [
        {
            "candidate_available": True,
            "stress_total_net_bps": value,
        }
        for value in (-98.86, -103.68, -59.16, -59.16)
    ]
    summary = _gate_summary(rows, thresholds)

    assert summary["highest_eligible_rows"] == 24
    assert summary["highest_eligible_profile"] == "aggressive"
    assert summary["candidate_count"] == 4
    assert summary["accepted_count"] == 0
    assert summary["all_candidate_stress_nets_negative"] is True
    assert "Aggressive (24)" in str(summary["sentence"])
    assert "all failed the stress-test acceptance criteria" in str(summary["sentence"])


def test_round20_publication_copy_is_specific_and_future_rounds_fail_closed() -> None:
    stage, model_id = _progress_identity(20)
    title, summary, next_step = _publication_narrative(
        20, all_candidate_stress_nets_negative=True
    )

    assert stage == "parameter-matched direction-specific representation ablation"
    assert model_id == "three-seed independent-long-short outcome-mixture"
    assert title == "direction-specific outcome model abstained"
    assert "independent long and short representations" in summary
    assert "decision-objective alignment" in next_step
    with pytest.raises(ValueError, match="undefined for Round 21"):
        _progress_identity(21)
    with pytest.raises(ValueError, match="undefined for Round 21"):
        _publication_narrative(21, all_candidate_stress_nets_negative=True)
