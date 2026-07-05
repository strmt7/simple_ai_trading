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
        execution_validation={
            "passed": True,
            "symbol": "BTCUSDC",
            "stress": {"accepted": True},
            "temporal_robustness": {"accepted": True},
            "portfolio": {"accepted": True},
        } if promoted else {},
        meta_label_policy={
            "enabled": True,
            "mode": "take_downsize_skip",
            "take_threshold": 0.03,
            "downsize_threshold": 0.01,
            "downsize_fraction": 0.5,
        },
        model_candidate_count=3,
        model_selected_candidate="triple_barrier_base",
        model_selection_score=0.42,
    )


def test_model_readiness_allows_promoted_model_and_round_trips_from_path(tmp_path) -> None:
    model = _model()
    report = build_model_readiness_report(model, model_path=tmp_path / "model.json")

    assert report.allowed is True
    assert report.block_count == 0
    assert any(check.label == "selection risk" and check.status == "ok" for check in report.checks)
    assert any(check.label == "model candidate search" and check.status == "ok" for check in report.checks)
    assert report.asdict()["allowed"] is True

    path = tmp_path / "model.json"
    serialize_model(model, path)
    loaded_report = load_model_readiness_report(path)

    assert loaded_report.allowed is True
    assert loaded_report.model_path == str(path)
    assert any("triple_barrier_base" in check.detail for check in loaded_report.checks)


def test_model_readiness_warns_on_single_candidate_evidence() -> None:
    model = _model()
    model.model_candidate_count = 1
    model.model_selected_candidate = "default"
    model.model_selection_score = None

    report = build_model_readiness_report(model)

    assert report.allowed is True
    assert any(
        check.label == "model candidate search" and check.status == "warn"
        for check in report.checks
    )


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


def test_model_readiness_blocks_missing_or_failed_execution_validation() -> None:
    missing = _model()
    missing.execution_validation = {}
    report = build_model_readiness_report(missing)
    assert report.allowed is False
    assert any(check.label == "execution validation" and check.status == "block" for check in report.checks)

    failed = _model()
    failed.execution_validation = {
        "passed": True,
        "symbol": "BTCUSDC",
        "stress": {"accepted": True},
        "temporal_robustness": {"accepted": False},
        "portfolio": {"accepted": True},
    }
    with pytest.raises(ModelPromotionError, match="execution validation"):
        assert_model_promoted(failed)

    failed_portfolio = _model()
    failed_portfolio.execution_validation = {
        "passed": True,
        "symbol": "BTCUSDC",
        "stress": {"accepted": True},
        "temporal_robustness": {"accepted": True},
        "portfolio": {"accepted": False},
    }
    report = build_model_readiness_report(failed_portfolio)
    assert any("portfolio=False" in check.detail for check in report.checks if check.label == "execution validation")


def test_model_readiness_reports_quality_warning_variants() -> None:
    unavailable = _model()
    unavailable.quality_warnings = ["meta_label_policy_unavailable"]
    report = build_model_readiness_report(unavailable)
    assert any(check.label == "model quality warnings" and "unavailable" in check.detail for check in report.checks)

    generic = _model()
    generic.quality_warnings = ["calibration_sparse"]
    report = build_model_readiness_report(generic)
    assert any(check.label == "model quality warnings" and "calibration_sparse" in check.detail for check in report.checks)


def test_model_readiness_blocks_financially_unsound_artifact() -> None:
    bad = _model()
    bad.weights = [0.0, 1.0]
    bad.learning_rate = 2.0

    report = build_model_readiness_report(bad)

    assert report.allowed is False
    assert any(check.label.startswith("financial sanity") for check in report.checks)
