from __future__ import annotations

import json

import numpy as np
import pytest

from simple_ai_trading.polymarket_historical_dataset import FEATURE_NAMES
from simple_ai_trading.polymarket_historical_model import HistoricalModelPanel
from simple_ai_trading.polymarket_historical_support import (
    freeze_historical_feature_support,
    load_historical_feature_support,
)


def _panel() -> HistoricalModelPanel:
    conditions = np.repeat(
        np.asarray([f"condition-{index}" for index in range(8)], dtype=object),
        8,
    )
    rows = len(conditions)
    base = np.arange(rows, dtype=np.float32).reshape(-1, 1)
    scale = np.arange(1, len(FEATURE_NAMES) + 1, dtype=np.float32).reshape(1, -1)
    features = base / scale
    labels = np.repeat(np.asarray([0.0, 1.0] * 4), 8)
    panel = HistoricalModelPanel(
        condition_ids=conditions,
        roles=np.full(rows, "train", dtype=object),
        event_start_ms=np.repeat(
            np.arange(8, dtype=np.int64) * 300_000 + 1_000_000,
            8,
        ),
        decision_time_ms=np.arange(rows, dtype=np.int64) * 30_000 + 1_030_000,
        features=features,
        labels=labels,
        dataset_sha256="1" * 64,
    )
    panel.validate(expected_roles=("train",))
    return panel


def test_train_only_profile_round_trips_and_gates_extremes(tmp_path) -> None:
    artifact, artifact_sha = freeze_historical_feature_support(
        _panel(),
        pretest_artifact_sha256="2" * 64,
        source_commit="3" * 40,
    )
    path = tmp_path / "support.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    profile = load_historical_feature_support(
        path,
        expected_pretest_artifact_sha256="2" * 64,
        expected_dataset_sha256="1" * 64,
    )
    assert profile.artifact_sha256 == artifact_sha
    assert profile.training_rows == 64
    in_support = profile.assess(_panel().features[20])
    assert in_support.status == "in_support"
    assert in_support.outside_training_range_count == 0
    extreme_vector = _panel().features[20].copy()
    extreme_vector[0] = 1_000_000
    extreme = profile.assess(extreme_vector)
    assert extreme.status == "abstain"
    assert extreme.extreme_outlier_features == (FEATURE_NAMES[0],)
    assert extreme.trading_authority is False


def test_more_than_four_range_escapes_abstain(tmp_path) -> None:
    artifact, _ = freeze_historical_feature_support(
        _panel(),
        pretest_artifact_sha256="2" * 64,
        source_commit="3" * 40,
    )
    path = tmp_path / "support.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    profile = load_historical_feature_support(
        path,
        expected_pretest_artifact_sha256="2" * 64,
        expected_dataset_sha256="1" * 64,
    )
    vector = _panel().features[-1].copy()
    vector[:5] = profile.maximum[:5] + 1.0
    result = profile.assess(vector)
    assert result.status == "abstain"
    assert result.outside_training_range_count == 5
    assert result.extreme_outlier_count == 0


def test_support_artifact_tampering_is_rejected(tmp_path) -> None:
    artifact, _ = freeze_historical_feature_support(
        _panel(),
        pretest_artifact_sha256="2" * 64,
        source_commit="3" * 40,
    )
    artifact["training_rows"] = 65
    path = tmp_path / "support.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity failed"):
        load_historical_feature_support(
            path,
            expected_pretest_artifact_sha256="2" * 64,
            expected_dataset_sha256="1" * 64,
        )
