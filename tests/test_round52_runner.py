from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from simple_ai_trading.executable_payoff_lightgbm import (
    ExecutablePayoffPredictionBatch,
)
from tools.run_round52_executable_support_hurdle import (
    CANDIDATES,
    _action_score,
    _ensemble_score,
    _specifications,
    _threshold,
)


ROOT = Path(__file__).resolve().parents[1]
DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-052-executable-support-hurdle-fincast-design.json"
)


def _prediction(
    *,
    architecture: str,
    long_expected: np.ndarray,
    short_expected: np.ndarray,
    long_executable: np.ndarray,
    short_executable: np.ndarray,
    probability: float = 0.75,
) -> ExecutablePayoffPredictionBatch:
    rows = len(long_expected)
    hurdle = architecture == "sign_magnitude_hurdle"
    probabilities = np.full(rows, probability, dtype=np.float64)
    gains = np.full(rows, 5.0, dtype=np.float64)
    losses = np.full(rows, 2.0, dtype=np.float64)
    return ExecutablePayoffPredictionBatch(
        architecture=architecture,
        endpoint_indexes=np.arange(1, rows + 1, dtype=np.int64),
        long_expected_net_bps=long_expected,
        short_expected_net_bps=short_expected,
        long_executable=long_executable,
        short_executable=short_executable,
        long_profitable_probability=probabilities if hurdle else None,
        short_profitable_probability=probabilities if hurdle else None,
        long_conditional_gain_bps=gains if hurdle else None,
        short_conditional_gain_bps=gains if hurdle else None,
        long_conditional_loss_bps=losses if hurdle else None,
        short_conditional_loss_bps=losses if hurdle else None,
    )


def test_round52_specifications_match_the_frozen_architectures() -> None:
    design = json.loads(DESIGN.read_text(encoding="utf-8"))
    specifications = _specifications(design)

    assert tuple(specifications) == CANDIDATES
    assert specifications[CANDIDATES[0]].architecture == "direct_mean"
    assert specifications[CANDIDATES[1]].architecture == "sign_magnitude_hurdle"
    assert specifications[CANDIDATES[2]].architecture == "sign_magnitude_hurdle"
    assert all(spec.minimum_leaf_fraction == 0.002 for spec in specifications.values())


def test_ensemble_cannot_select_an_unsupported_high_score() -> None:
    rows = 100
    long_expected = np.linspace(100.0, 1.0, rows)
    short_expected = np.linspace(0.1, 0.2, rows)
    long_executable = np.ones(rows, dtype=bool)
    long_executable[0] = False
    short_executable = np.ones(rows, dtype=bool)
    members = [
        _prediction(
            architecture="direct_mean",
            long_expected=long_expected,
            short_expected=short_expected,
            long_executable=long_executable,
            short_executable=short_executable,
        )
        for _ in range(3)
    ]

    ensemble = _ensemble_score(members)
    threshold = _threshold(ensemble, 0.05)
    score = _action_score(ensemble, threshold)

    assert ensemble.side[0] == -1
    assert not score.eligible[0]
    assert score.side[0] == 0
    assert score.strength_bps[0] == 0.0


def test_hurdle_probability_gate_blocks_positive_expected_value() -> None:
    rows = 100
    members = [
        _prediction(
            architecture="sign_magnitude_hurdle",
            long_expected=np.full(rows, 4.0),
            short_expected=np.full(rows, 1.0),
            long_executable=np.ones(rows, dtype=bool),
            short_executable=np.ones(rows, dtype=bool),
            probability=0.49,
        )
        for _ in range(3)
    ]

    ensemble = _ensemble_score(members)

    assert np.all(ensemble.side == 1)
    assert not np.any(ensemble.eligible)
    assert not np.any(_action_score(ensemble, 0.0).eligible)
