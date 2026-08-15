from __future__ import annotations

from pathlib import Path

import numpy as np

from simple_ai_trading import polymarket_round22_diagnostic as diagnostic


REPOSITORY = Path(__file__).resolve().parents[1]


def test_round22_diagnostic_spec_is_exactly_pinned_and_non_promotable() -> None:
    spec = diagnostic.load_round22_diagnostic_model_spec(REPOSITORY)

    assert (
        spec["specification_sha256"]
        == diagnostic.POLYMARKET_ROUND22_DIAGNOSTIC_MODEL_SPEC_SHA256
    )
    assert spec["data_selection"]["sealed_test_access"] is False
    assert spec["authority"] == {
        "ai_edge_claim": False,
        "economic_backtest": False,
        "live_trading": False,
        "model_promotion": False,
        "paper_trading": False,
        "profitability_claim": False,
    }


def test_round22_condition_weights_equalize_unequal_row_counts() -> None:
    condition_ids = np.asarray(["a", "b", "b", "b"], dtype=object)

    weights = diagnostic._condition_weights(condition_ids)

    assert np.isclose(weights[condition_ids == "a"].sum(), 0.5)
    assert np.isclose(weights[condition_ids == "b"].sum(), 0.5)
    assert np.isclose(weights.sum(), 1.0)


def test_round22_offset_residual_fit_improves_controlled_signal() -> None:
    condition_ids = np.repeat(np.asarray(["a", "b", "c", "d"], dtype=object), 20)
    labels = np.repeat(np.asarray([0.0, 0.0, 1.0, 1.0]), 20)
    feature = np.repeat(np.asarray([-2.0, -1.0, 1.0, 2.0]), 20)
    partition = diagnostic._Partition(
        role="train",
        features=feature[:, None],
        priors=np.full(80, 0.5, dtype=np.float64),
        labels=labels,
        condition_ids=condition_ids,
        elapsed_seconds=np.tile(np.arange(20, dtype=np.float64), 4),
    )
    standardizer = diagnostic._fit_standardizer(partition)
    theta = diagnostic._fit_residual(
        partition,
        standardizer=standardizer,
        penalty=0.1,
    )
    fitted, _residual = diagnostic._raw_probabilities(
        partition,
        standardizer=standardizer,
        theta=theta,
    )

    assert diagnostic._weighted_log_loss(
        partition, fitted
    ) < diagnostic._weighted_log_loss(
        partition,
        partition.priors,
    )
