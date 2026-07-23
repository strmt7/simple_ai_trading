from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from simple_ai_trading import impact_absorption_shallow_model as subject


def _threshold_result(
    threshold: float,
    *,
    lower_expectancy_bps: float,
    completed_trades: int = 25,
) -> subject._ThresholdResult:
    return subject._ThresholdResult(
        threshold=threshold,
        completed_trades=completed_trades,
        attempted_actions=completed_trades,
        pre_entry_aborts=0,
        unresolved_risk_count=0,
        lower_expectancy_bps=lower_expectancy_bps,
        selected_row_indexes=np.arange(completed_trades, dtype=np.int64),
    )


def test_threshold_policy_rejects_nonpositive_tuning_lower_bound(monkeypatch) -> None:
    def evaluate(_dataset, _probability, _predicted_net_bps, *, threshold):
        return _threshold_result(threshold, lower_expectancy_bps=0.0)

    monkeypatch.setattr(subject, "_evaluate_tuning_threshold", evaluate)

    threshold, action_enabled, report = subject._select_threshold(
        object(),
        np.asarray([], dtype=np.float64),
        np.asarray([], dtype=np.float64),
    )

    assert threshold == 0.9
    assert action_enabled is False
    assert all(item["admissible"] is False for item in report)
    assert all(item["positive_lower_expectancy_required"] is True for item in report)


def test_threshold_policy_selects_largest_positive_lower_bound(monkeypatch) -> None:
    lower_by_threshold = {
        threshold: 0.1 for threshold in subject.ROUND73_PROBABILITY_THRESHOLDS
    }
    lower_by_threshold[0.7] = 0.4
    lower_by_threshold[0.8] = 0.4

    def evaluate(_dataset, _probability, _predicted_net_bps, *, threshold):
        return _threshold_result(
            threshold,
            lower_expectancy_bps=lower_by_threshold[threshold],
        )

    monkeypatch.setattr(subject, "_evaluate_tuning_threshold", evaluate)

    threshold, action_enabled, _report = subject._select_threshold(
        object(),
        np.asarray([], dtype=np.float64),
        np.asarray([], dtype=np.float64),
    )

    assert threshold == 0.8
    assert action_enabled is True


def test_tuning_overlap_guard_covers_both_execution_lateness_budgets() -> None:
    first_wall_ns = 10_000_000_000
    second_wall_ns = first_wall_ns + 60_750_000_000
    dataset = SimpleNamespace(
        role_mask=lambda role: np.ones(4, dtype=np.bool_) if role == "tuning" else None,
        outcome_status=np.asarray(
            [
                subject.ROUND73_OBSERVED_STATUS,
                subject.ROUND73_OBSERVED_STATUS,
                subject.ROUND73_OBSERVED_STATUS,
                subject.ROUND73_OBSERVED_STATUS,
            ],
            dtype=np.uint8,
        ),
        anchor_wall_ns=np.asarray(
            [first_wall_ns, first_wall_ns, second_wall_ns, second_wall_ns],
            dtype=np.int64,
        ),
        continuous_target_bps=np.asarray([2.0, -2.0, 2.0, -2.0]),
        run_id_binary=np.asarray([b"a" * 16, b"a" * 16, b"b" * 16, b"b" * 16]),
    )
    probability = np.asarray([0.9, 0.1, 0.9, 0.1])
    predicted_net_bps = np.asarray([1.0, -1.0, 1.0, -1.0])

    result = subject._evaluate_tuning_threshold(
        dataset,
        probability,
        predicted_net_bps,
        threshold=0.5,
    )

    assert subject._TUNING_MAXIMUM_POSITION_NS == 61_000_000_000
    assert result.attempted_actions == 1
    assert result.completed_trades == 1


def test_unresolved_tuning_position_blocks_later_same_symbol_actions() -> None:
    first_wall_ns = 10_000_000_000
    dataset = SimpleNamespace(
        role_mask=lambda role: np.ones(4, dtype=np.bool_) if role == "tuning" else None,
        outcome_status=np.asarray(
            [
                subject.ROUND73_POST_ENTRY_UNRESOLVED_STATUS,
                subject.ROUND73_POST_ENTRY_UNRESOLVED_STATUS,
                subject.ROUND73_OBSERVED_STATUS,
                subject.ROUND73_OBSERVED_STATUS,
            ],
            dtype=np.uint8,
        ),
        anchor_wall_ns=np.asarray(
            [
                first_wall_ns,
                first_wall_ns,
                first_wall_ns + 120_000_000_000,
                first_wall_ns + 120_000_000_000,
            ],
            dtype=np.int64,
        ),
        continuous_target_bps=np.asarray([np.nan, np.nan, 2.0, -2.0]),
        run_id_binary=np.asarray([b"a" * 16, b"a" * 16, b"b" * 16, b"b" * 16]),
    )
    probability = np.asarray([0.9, 0.1, 0.9, 0.1])
    predicted_net_bps = np.asarray([1.0, -1.0, 1.0, -1.0])

    result = subject._evaluate_tuning_threshold(
        dataset,
        probability,
        predicted_net_bps,
        threshold=0.5,
    )

    assert result.attempted_actions == 1
    assert result.unresolved_risk_count == 1
    assert result.completed_trades == 0
