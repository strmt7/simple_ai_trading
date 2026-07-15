from __future__ import annotations

import numpy as np
import pytest

from simple_ai_trading.probability_calibration import (
    apply_platt_scaling,
    fit_platt_scaling,
)


def test_weighted_platt_scaling_preserves_weighted_class_influence() -> None:
    probabilities = np.full(8, 0.5)
    labels = np.asarray([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    weights = np.asarray([5.0, 5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0])

    unweighted = fit_platt_scaling(probabilities, labels)
    weighted = fit_platt_scaling(probabilities, labels, weights)

    assert apply_platt_scaling(np.asarray([0.5]), weighted)[0] > apply_platt_scaling(
        np.asarray([0.5]), unweighted
    )[0]


def test_weighted_platt_scaling_rejects_nonpositive_weights() -> None:
    with pytest.raises(ValueError, match="arrays are inconsistent"):
        fit_platt_scaling(
            np.asarray([0.2, 0.3, 0.7, 0.8]),
            np.asarray([0.0, 0.0, 1.0, 1.0]),
            np.asarray([1.0, 0.0, 1.0, 1.0]),
        )
