from __future__ import annotations

from pathlib import Path

import numpy as np

from simple_ai_trading.microstructure_direction_screen import DirectionScreenPrediction
from tools.run_consumed_direction_screen import (
    _eligibility_reasons,
    _feature_names,
    _select_candidate,
    _variant_metrics,
    load_direction_screen_design,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-035-consumed-direction-screen-design.json"
)


def test_direction_screen_runner_loads_six_exact_variants() -> None:
    design, design_sha = load_direction_screen_design(DESIGN)

    assert design_sha == (
        "db027483eed1329554bac8c3be057c488bc1899348978ea8b199c5796db6cbaa"
    )
    assert [item["variant"] for item in design["variants"]] == [
        "full_uniqueness",
        "full_utility_margin",
        "noncycle_uniqueness",
        "noncycle_utility_margin",
        "compact_uniqueness",
        "compact_utility_margin",
    ]
    assert [
        len(_feature_names(design, item["feature_set"])) for item in design["variants"]
    ] == [107, 107, 100, 100, 68, 68]


def test_direction_screen_metrics_use_after_cost_side_returns_and_daily_auc() -> None:
    rows = 2_000
    true_long = np.arange(rows) % 2 == 0
    long_actual = np.where(true_long, 12.0, -8.0)
    short_actual = np.where(true_long, -8.0, 12.0)
    conditional_long = np.where(true_long, 0.9, 0.1)
    prediction = DirectionScreenPrediction(
        endpoint_indexes=np.arange(rows, dtype=np.int64),
        long_superiority_probability=conditional_long,
        short_superiority_probability=1.0 - conditional_long,
        conditional_long_probability=conditional_long,
        direction_score=conditional_long - (1.0 - conditional_long),
        selected_side=np.where(true_long, 1, -1).astype(np.int8),
    )
    times = (
        np.repeat(np.arange(5, dtype=np.int64), rows // 5) * 86_400_000
        + np.tile(np.arange(rows // 5, dtype=np.int64), 5) * 5_000
    )

    metrics = _variant_metrics(
        prediction=prediction,
        long_actual=long_actual,
        short_actual=short_actual,
        decision_time_ms=times,
        frozen_opportunity_probability=np.linspace(1.0, 0.0, rows),
    )

    assert metrics["pooled_direction_auc"] == 1.0
    assert metrics["direction_accuracy"] == 1.0
    assert metrics["all_routed_mean_stress_net_bps"] == 12.0
    assert metrics["daily_auc_minimum"] == 1.0
    assert metrics["daily_auc_median"] == 1.0
    assert metrics["days_above_chance"] == 5
    assert len(metrics["daily"]) == 5
    for ranking in ("frozen_opportunity_ranked", "candidate_confidence_ranked"):
        assert metrics[ranking]["100"]["mean_stress_net_bps"] == 12.0
        assert metrics[ranking]["500"]["mean_stress_net_bps"] == 12.0
        assert metrics[ranking]["1000"]["mean_stress_net_bps"] == 12.0


def _passing_metrics() -> dict[str, object]:
    return {
        "pooled_direction_auc": 0.56,
        "daily_auc_minimum": 0.49,
        "daily_auc_median": 0.54,
        "days_above_chance": 4,
        "frozen_opportunity_ranked": {
            "100": {"mean_stress_net_bps": 1.0},
            "500": {"mean_stress_net_bps": 2.0},
        },
        "candidate_confidence_ranked": {
            "500": {"mean_stress_net_bps": 1.0},
        },
    }


def test_direction_screen_eligibility_is_strict_at_zero_return_boundary() -> None:
    design, _design_sha = load_direction_screen_design(DESIGN)
    gates = design["architecture_freeze_eligibility"]
    metrics = _passing_metrics()

    assert _eligibility_reasons(metrics, gates, nonfinite_predictions=0) == []
    metrics["frozen_opportunity_ranked"]["500"]["mean_stress_net_bps"] = 0.0
    assert _eligibility_reasons(metrics, gates, nonfinite_predictions=0) == [
        "frozen_opportunity_top_500_stress_net_gate_failed"
    ]


def test_direction_screen_candidate_selection_uses_frozen_tie_break_order() -> None:
    candidates = [
        {
            "variant": "higher_tail_lower_minimum",
            "feature_count": 68,
            "metrics": {
                "daily_auc_minimum": 0.49,
                "frozen_opportunity_ranked": {"500": {"mean_stress_net_bps": 8.0}},
            },
        },
        {
            "variant": "lower_tail_higher_minimum",
            "feature_count": 107,
            "metrics": {
                "daily_auc_minimum": 0.50,
                "frozen_opportunity_ranked": {"500": {"mean_stress_net_bps": 2.0}},
            },
        },
        {
            "variant": "highest_tail_same_minimum",
            "feature_count": 100,
            "metrics": {
                "daily_auc_minimum": 0.50,
                "frozen_opportunity_ranked": {"500": {"mean_stress_net_bps": 3.0}},
            },
        },
    ]

    assert _select_candidate(candidates) == "highest_tail_same_minimum"
    assert _select_candidate([]) is None
