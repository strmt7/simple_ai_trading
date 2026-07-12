from __future__ import annotations

import numpy as np

from tools.diagnose_round34_failure import _action_labels, _route_diagnostic


def test_round34_action_labels_abstain_on_ties_and_nonpositive_actions() -> None:
    long_values = np.asarray([5.0, -2.0, 3.0, 0.0, 4.0], dtype=np.float64)
    short_values = np.asarray([-1.0, 6.0, 3.0, -1.0, 4.0], dtype=np.float64)

    assert _action_labels(long_values, short_values).tolist() == [0, 2, 1, 1, 1]


def test_round34_route_diagnostic_is_non_promotable_and_does_not_route_ties() -> None:
    rows = 1_200
    long_is_best = np.arange(rows) % 2 == 0
    long_values = np.where(long_is_best, 12.0, -8.0)
    short_values = np.where(long_is_best, -8.0, 12.0)
    labels = _action_labels(long_values, short_values)
    side_score = np.where(long_is_best, 1.0, -1.0)
    side_score[:20] = 0.0
    ranking_score = np.linspace(1.0, 0.0, rows, dtype=np.float64)

    diagnostic = _route_diagnostic(
        name="synthetic_semantics_check",
        long_actual=long_values,
        short_actual=short_values,
        action_labels=labels,
        side_score=side_score,
        ranking_score=ranking_score,
    )

    assert diagnostic["promotion_permitted"] is False
    assert diagnostic["post_hoc_consumed_data_diagnostic"] is True
    assert diagnostic["eligible_rows_before_score_ties"] == rows
    assert diagnostic["eligible_rows"] == rows - 20
    assert diagnostic["direction_auc_on_non_abstain_rows"] > 0.999
    assert diagnostic["direction_accuracy_on_non_abstain_rows"] == (rows - 20) / rows
    assert diagnostic["all_eligible_selected_mean_stress_net_bps"] == 12.0
    assert diagnostic["top_rows"]["100"]["mean_stress_net_bps"] == 12.0


def test_round34_oracle_control_is_explicitly_identified() -> None:
    rows = 1_000
    long_is_best = np.arange(rows) % 2 == 0
    long_values = np.where(long_is_best, 5.0, -5.0)
    short_values = -long_values
    labels = _action_labels(long_values, short_values)

    diagnostic = _route_diagnostic(
        name="oracle_control_semantics_check",
        long_actual=long_values,
        short_actual=short_values,
        action_labels=labels,
        side_score=np.where(long_is_best, 1.0, -1.0),
        ranking_score=np.ones(rows, dtype=np.float64),
        oracle_control=True,
    )

    assert diagnostic["oracle_control"] is True
    assert diagnostic["promotion_permitted"] is False
