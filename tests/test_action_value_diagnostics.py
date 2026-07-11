from __future__ import annotations

import numpy as np
import pytest

from tools.diagnose_action_value_discovery import (
    _average_ranks,
    _top_score_diagnostic,
)


def test_average_ranks_assigns_equal_values_the_same_rank() -> None:
    ranks = _average_ranks(np.asarray([4.0, 1.0, 4.0, 2.0]))

    assert ranks.tolist() == [2.5, 0.0, 2.5, 1.0]


def test_top_score_diagnostic_is_derived_from_eligible_rows_only() -> None:
    edge = np.arange(1_001, dtype=np.float64) - 500.0
    actual = edge / 10.0
    eligible = np.ones(1_001, dtype=bool)
    eligible[-1] = False

    result = _top_score_diagnostic(edge, actual, eligible)
    top_twenty = result["top_score_rows"][0]

    assert result["eligible_rows"] == 1_000
    assert result["positive_predicted_edge_rows"] == 499
    assert result["predicted_edge_actual_spearman_ic"] == pytest.approx(1.0)
    assert top_twenty["requested_rows"] == 20
    assert top_twenty["rows"] == 20
    assert top_twenty["mean_predicted_edge_bps"] == pytest.approx(489.5)
    assert top_twenty["mean_actual_net_bps"] == pytest.approx(48.95)
