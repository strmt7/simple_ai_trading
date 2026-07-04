from __future__ import annotations

import pytest

from simple_ai_trading.model import TrainedModel, serialize_model
from simple_ai_trading.model_readiness import (
    ModelPromotionError,
    assert_model_promoted,
    build_model_readiness_report,
    load_model_readiness_report,
)


def _model(*, promoted: bool = True, deflated_score: float = 0.12) -> TrainedModel:
    return TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
        selection_risk={
            "passed": promoted,
            "effective_trials": 20,
            "selected_score": 0.15,
            "trial_penalty": 0.03,
            "deflated_score": deflated_score,
        } if promoted or deflated_score <= 0.0 else {},
        meta_label_policy={
            "enabled": True,
            "mode": "take_downsize_skip",
            "take_threshold": 0.03,
            "downsize_threshold": 0.01,
            "downsize_fraction": 0.5,
        },
    )


def test_model_readiness_allows_promoted_model_and_round_trips_from_path(tmp_path) -> None:
    model = _model()
    report = build_model_readiness_report(model, model_path=tmp_path / "model.json")

    assert report.allowed is True
    assert report.block_count == 0
    assert any(check.label == "selection risk" and check.status == "ok" for check in report.checks)
    assert report.asdict()["allowed"] is True

    path = tmp_path / "model.json"
    serialize_model(model, path)
    loaded_report = load_model_readiness_report(path)

    assert loaded_report.allowed is True
    assert loaded_report.model_path == str(path)


def test_model_readiness_blocks_missing_or_failed_selection_risk() -> None:
    missing = TrainedModel(
        weights=[0.0],
        bias=0.0,
        feature_dim=1,
        epochs=1,
        feature_means=[0.0],
        feature_stds=[1.0],
    )
    missing_report = build_model_readiness_report(missing)
    assert missing_report.allowed is False
    assert "missing promotion evidence" in missing_report.checks[0].detail

    failed = _model(promoted=False, deflated_score=-0.01)
    failed_report = build_model_readiness_report(failed)
    assert failed_report.allowed is False
    assert failed_report.checks[0].status == "block"

    with pytest.raises(ModelPromotionError, match="selection risk"):
        assert_model_promoted(failed)
