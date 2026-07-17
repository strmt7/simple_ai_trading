from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from simple_ai_trading.depth_stress_screen import (
    DEPTH_STRESS_FEATURE_NAMES,
    DepthStressPanel,
    benjamini_hochberg_q_values,
    build_depth_stress_examples,
    paired_blocked_permutation_test,
    utc_month_label,
)


def _panel(*, with_gap: bool = False) -> DepthStressPanel:
    start = int(np.datetime64("2026-01-02T00:00:00", "ms").astype(np.int64))
    first = start + np.arange(80, dtype=np.int64) * 30_000
    if with_gap:
        second = first[-1] + 180_000 + np.arange(80, dtype=np.int64) * 30_000
        timestamps = np.concatenate((first, second))
    else:
        timestamps = start + np.arange(160, dtype=np.int64) * 30_000
    axis = np.arange(len(timestamps), dtype=np.float64)
    descriptors = np.column_stack((axis, axis**2 / 100.0, np.sin(axis / 10.0)))
    return DepthStressPanel(
        symbol="btcusdt",
        timestamp_ms=timestamps,
        descriptors=descriptors,
        source_fingerprint=hashlib.sha256(b"verified-source").hexdigest(),
    )


def test_panel_and_examples_are_hash_bound_and_read_only() -> None:
    panel = _panel()
    examples = build_depth_stress_examples(panel, horizon_seconds=60)
    features = examples.feature_matrix(np.zeros(len(examples.anchor_time_ms), dtype=np.int8))

    assert panel.symbol == "BTCUSDT"
    assert len(panel.panel_sha256) == 64
    assert len(examples.examples_sha256) == 64
    assert features.shape == (len(examples.anchor_time_ms), len(DEPTH_STRESS_FEATURE_NAMES))
    assert np.all(np.diff(examples.anchor_time_ms) == 60_000)
    assert np.all(examples.post_index > examples.pre_index)
    assert examples.base_features.flags.writeable is False
    with pytest.raises(ValueError):
        examples.base_features[0, 0] = 0.0
    with pytest.raises(ValueError, match="digest"):
        replace(examples, examples_sha256="0" * 64)


def test_examples_do_not_cross_detected_data_gap() -> None:
    panel = _panel(with_gap=True)
    examples = build_depth_stress_examples(panel, horizon_seconds=300)
    timestamps = panel.timestamp_ms

    assert np.all(timestamps[examples.post_index] - timestamps[examples.pre_index] <= 390_000)
    gap_start = np.flatnonzero(np.diff(timestamps) > 90_000)[0]
    assert not np.any(
        (examples.pre_index <= gap_start) & (examples.post_index > gap_start)
    )


def test_example_features_use_only_causal_snapshots() -> None:
    panel = _panel()
    examples = build_depth_stress_examples(panel, horizon_seconds=60)
    row = 0
    pre = int(examples.pre_index[row])
    expected_immediate_change = panel.descriptors[pre] - panel.descriptors[pre - 1]

    assert np.allclose(examples.base_features[row, :3], panel.descriptors[pre])
    assert np.allclose(examples.base_features[row, 3:6], expected_immediate_change)
    assert examples.anchor_time_ms[row] <= panel.timestamp_ms[examples.post_index[row]]


def test_paired_blocked_test_detects_consistent_loss_improvement() -> None:
    blocks = np.repeat(np.arange(24), 5)
    baseline = np.full(len(blocks), 0.60)
    challenger = baseline - 0.10
    result = paired_blocked_permutation_test(
        baseline,
        challenger,
        blocks,
        draws=2_000,
        seed=7,
    )

    assert result.rows == 120
    assert result.blocks == 24
    assert result.relative_improvement == pytest.approx(1.0 / 6.0)
    assert result.one_sided_p_value < 0.01


def test_paired_blocked_test_does_not_reward_worse_challenger() -> None:
    blocks = np.repeat(np.arange(20), 3)
    baseline = np.full(len(blocks), 0.50)
    challenger = baseline + 0.05
    result = paired_blocked_permutation_test(
        baseline,
        challenger,
        blocks,
        draws=1_000,
        seed=9,
    )

    assert result.relative_improvement < 0.0
    assert result.one_sided_p_value > 0.90


def test_benjamini_hochberg_correction_and_month_labels() -> None:
    q_values = benjamini_hochberg_q_values([0.01, 0.04, 0.03, 0.20])

    assert q_values == pytest.approx([0.04, 0.0533333333333, 0.0533333333333, 0.20])
    january_2026 = int(np.datetime64("2026-01", "M").astype(np.int64))
    assert utc_month_label(january_2026) == "2026-01"
