from __future__ import annotations

import xml.etree.ElementTree as ET

from tools.publish_adaptive_action_screen import (
    _barrier_svg,
    _funnel_svg,
    _forecast_svg,
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
    assert "2023-06-21" in charts[2]
    assert "Rounds fifteen and sixteen" in charts[4]


def test_round17_titles_and_extreme_tail_label_have_clearance() -> None:
    rows = _forecast_rows()
    rows[-1]["top_100_mean_net_bps"] = -21.99
    tail = ET.fromstring(_tail_svg(rows, round_number=17))
    namespace = "{http://www.w3.org/2000/svg}"
    texts = [node for node in tail.findall(f"{namespace}text") if node.text]
    value_label = next(node for node in texts if node.text == "-21.99")
    category_label = next(node for node in texts if node.text == "Policy short")

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
