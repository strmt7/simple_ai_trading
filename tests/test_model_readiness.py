from __future__ import annotations

import pytest

from simple_ai_trading.model import TrainedModel, serialize_model
from simple_ai_trading.model_readiness import (
    ModelPromotionError,
    assert_model_promoted,
    build_model_readiness_report,
    load_model_readiness_report,
)


def _live_data_coverage(symbol: str = "BTCUSDC", *, interval: str = "1s", years: float = 2.0) -> dict[str, object]:
    rows = int(365.25 * 24 * 60 * 60 * max(0.1, years))
    return {
        "symbol": symbol,
        "market_type": "futures",
        "interval": interval,
        "source_scope": "sqlite_market_data",
        "expected_interval_ms": 1000 if interval == "1s" else 60_000,
        "integrity_status": "ok",
        "integrity_warnings": [],
        "truth_basis": [
            "prices_from_timestamped_closed_candles",
            "coverage_measured_from_candle_close_time",
            "execution_results_are_simulated_not_exchange_fills",
        ],
        "full_history_requested": True,
        "full_available_history_used": True,
        "candles_available": rows,
        "candles_used": rows,
        "rows_used": rows - 100,
        "requested_start_ms": None,
        "requested_end_ms": None,
        "available_start_ms": 0,
        "available_end_ms": rows * 1000,
        "used_start_ms": 0,
        "used_end_ms": rows * 1000,
        "available_start_utc": "2024-01-01T00:00:00Z",
        "available_end_utc": "2026-01-01T00:00:00Z",
        "used_start_utc": "2024-01-01T00:00:00Z",
        "used_end_utc": "2026-01-01T00:00:00Z",
        "used_duration_days": years * 365.25,
        "used_duration_years": years,
        "gap_count": 0,
        "largest_gap_ms": 1000,
        "largest_gap_intervals": 1.0,
        "coverage_ratio": 1.0,
        "notes": [],
    }


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
            "walk_forward_gate": {
                "passed": True,
                "reason": None,
                "fold_count": 3,
                "accepted_folds": 3,
                "worst_score": 0.08,
                "worst_realized_pnl": 1.2,
                "worst_max_drawdown": 0.025,
            },
            "stress": {"accepted": True},
            "temporal_robustness": {"accepted": True},
            "portfolio": {"accepted": True},
            "data_coverage": _live_data_coverage(),
        } if promoted else {},
        meta_label_policy={
            "enabled": True,
            "mode": "take_downsize_skip",
            "take_threshold": 0.03,
            "downsize_threshold": 0.01,
            "downsize_fraction": 0.5,
        },
        probability_calibration_size=128,
        probability_log_loss_before=0.62,
        probability_log_loss_after=0.58,
        probability_brier_before=0.24,
        probability_brier_after=0.22,
        probability_ece_before=0.10,
        probability_ece_after=0.08,
        probability_calibration_backend_requested="directml",
        probability_calibration_backend_kind="directml",
        probability_calibration_backend_device="privateuseone:0",
        training_backend_requested="directml",
        training_backend_kind="directml",
        training_backend_device="privateuseone:0",
        training_backend_vendor="DirectML",
        model_candidate_count=3,
        model_selected_candidate="triple_barrier_base",
        model_selection_score=0.42,
    )


def test_model_readiness_allows_promoted_model_and_round_trips_from_path(tmp_path) -> None:
    model = _model()
    report = build_model_readiness_report(
        model,
        model_path=tmp_path / "model.json",
        require_model_candidate_search=True,
        require_accelerator_evidence=True,
    )

    assert report.allowed is True
    assert report.block_count == 0
    assert any(check.label == "selection risk" and check.status == "ok" for check in report.checks)
    assert any(check.label == "model candidate search" and check.status == "ok" for check in report.checks)
    assert any(check.label == "training accelerator" and check.status == "ok" for check in report.checks)
    assert any(check.label == "probability calibration accelerator" and check.status == "ok" for check in report.checks)
    assert report.asdict()["allowed"] is True

    path = tmp_path / "model.json"
    serialize_model(model, path)
    loaded_report = load_model_readiness_report(
        path,
        require_model_candidate_search=True,
        require_accelerator_evidence=True,
    )

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


def test_model_readiness_can_require_multi_candidate_evidence() -> None:
    model = _model()
    model.model_candidate_count = 1
    model.model_selected_candidate = "default"
    model.model_selection_score = None

    report = build_model_readiness_report(
        model,
        require_model_candidate_search=True,
        min_model_candidates=2,
    )

    assert report.allowed is False
    assert any(
        check.label == "model candidate search" and check.status == "block"
        for check in report.checks
    )
    with pytest.raises(ModelPromotionError, match="model candidate search"):
        assert_model_promoted(model, require_model_candidate_search=True, min_model_candidates=2)


def test_model_readiness_can_require_accelerator_evidence() -> None:
    model = _model()
    model.training_backend_kind = "cpu"
    model.training_backend_device = "cpu"
    model.training_backend_reason = "DirectML unavailable in test"
    model.probability_calibration_backend_kind = "cpu"
    model.probability_calibration_backend_device = "cpu"
    model.probability_calibration_backend_reason = "calibration fell back in test"

    report = build_model_readiness_report(model, require_accelerator_evidence=True)

    assert report.allowed is False
    assert any(check.label == "training accelerator" and check.status == "block" for check in report.checks)
    assert any(
        check.label == "probability calibration accelerator" and check.status == "block"
        for check in report.checks
    )
    with pytest.raises(ModelPromotionError, match="training accelerator"):
        assert_model_promoted(model, require_accelerator_evidence=True)


def test_model_readiness_can_require_live_data_evidence() -> None:
    model = _model()

    report = build_model_readiness_report(
        model,
        require_live_data_evidence=True,
        expected_symbol="BTCUSDC",
        expected_market_type="futures",
        expected_interval="1s",
        min_live_data_years=1.0,
    )

    assert report.allowed is True
    assert any(check.label == "live data evidence" and check.status == "ok" for check in report.checks)


def test_model_readiness_requires_substantial_hftbacktest_evidence_for_live_promotion() -> None:
    model = _model()
    model.execution_validation["microstructure_replay"] = {
        "passed": True,
        "strategy_replay_passed": True,
        "replay_smoke_passed": True,
        "artifact_hashes_verified": True,
        "immutable_market_data": True,
        "engine": "hftbacktest",
        "engine_version": "2.4.4",
        "schema_version": "binance-usdm-l2-v3",
        "symbol": "BTCUSDC",
        "queue_model": "risk_adverse_queue_model",
        "latency_model": "empirical_feed_and_order_latency",
        "captured_seconds": 20 * 86_400,
        "span_days": 400,
        "unique_days": 20,
        "normalized_rows": 20_000_000,
        "sequence_gap_count": 0,
        "crossed_book_count": 0,
        "invalid_event_count": 0,
        "clock_sync_samples": 100,
    }

    report = build_model_readiness_report(
        model,
        require_microstructure_evidence=True,
        expected_symbol="BTCUSDC",
    )

    assert report.allowed is True
    assert any(check.label == "microstructure replay" and check.status == "ok" for check in report.checks)

    model.execution_validation["microstructure_replay"]["captured_seconds"] = 6
    failed = build_model_readiness_report(
        model,
        require_microstructure_evidence=True,
        expected_symbol="BTCUSDC",
    )
    assert failed.allowed is False
    assert any(
        check.label == "microstructure replay" and "captured_seconds=6.0<1728000" in check.detail
        for check in failed.checks
    )
    with pytest.raises(ModelPromotionError, match="microstructure replay"):
        assert_model_promoted(
            model,
            require_microstructure_evidence=True,
            expected_symbol="BTCUSDC",
        )


@pytest.mark.parametrize(
    ("mutator", "match_text"),
    [
        (lambda payload: payload.pop("data_coverage"), "missing data_coverage"),
        (lambda payload: payload.__setitem__("data_coverage", _live_data_coverage(interval="1m")), "interval=1m"),
        (lambda payload: payload.__setitem__("data_coverage", _live_data_coverage(years=0.5)), "used_duration_years=0.5<1.00"),
        (lambda payload: payload.__setitem__("data_coverage", _live_data_coverage(symbol="ETHUSDC")), "symbol=ETHUSDC!=BTCUSDC"),
    ],
)
def test_model_readiness_blocks_failed_live_data_evidence(mutator, match_text: str) -> None:
    model = _model()
    mutator(model.execution_validation)

    report = build_model_readiness_report(
        model,
        require_live_data_evidence=True,
        expected_symbol="BTCUSDC",
        expected_market_type="futures",
        expected_interval="1s",
    )

    assert report.allowed is False
    assert any(check.label == "live data evidence" and match_text in check.detail for check in report.checks)
    with pytest.raises(ModelPromotionError, match="live data evidence"):
        assert_model_promoted(
            model,
            require_live_data_evidence=True,
            expected_symbol="BTCUSDC",
            expected_market_type="futures",
            expected_interval="1s",
        )


def test_model_readiness_blocks_non_second_runtime_interval() -> None:
    report = build_model_readiness_report(
        _model(),
        require_live_data_evidence=True,
        expected_symbol="BTCUSDC",
        expected_market_type="futures",
        expected_interval="15m",
    )

    assert report.allowed is False
    assert any("runtime_interval=15m!=1s" in check.detail for check in report.checks)


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
        "walk_forward_gate": {
            "passed": True,
            "reason": None,
            "fold_count": 3,
            "accepted_folds": 3,
            "worst_score": 0.08,
            "worst_realized_pnl": 1.2,
            "worst_max_drawdown": 0.025,
        },
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
        "walk_forward_gate": {
            "passed": True,
            "reason": None,
            "fold_count": 3,
            "accepted_folds": 3,
            "worst_score": 0.08,
            "worst_realized_pnl": 1.2,
            "worst_max_drawdown": 0.025,
        },
        "stress": {"accepted": True},
        "temporal_robustness": {"accepted": True},
        "portfolio": {"accepted": False},
    }
    report = build_model_readiness_report(failed_portfolio)
    assert any("portfolio=False" in check.detail for check in report.checks if check.label == "execution validation")


def test_model_readiness_blocks_missing_or_skipped_walk_forward_validation() -> None:
    missing = _model()
    del missing.execution_validation["walk_forward_gate"]
    report = build_model_readiness_report(missing)
    assert report.allowed is False
    assert any("walk_forward=False" in check.detail for check in report.checks)

    skipped = _model()
    skipped.execution_validation["walk_forward_gate"] = {
        "passed": True,
        "reason": "insufficient_rows_for_purged_walk_forward",
        "fold_count": 0,
        "accepted_folds": 0,
        "worst_score": None,
        "worst_realized_pnl": None,
    }
    with pytest.raises(ModelPromotionError, match="walk_forward=False"):
        assert_model_promoted(skipped)


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


def test_model_readiness_blocks_promoted_bad_probability_calibration() -> None:
    bad = _model()
    bad.probability_log_loss_after = 0.90
    bad.probability_brier_after = 0.42
    bad.probability_ece_after = 0.24

    report = build_model_readiness_report(bad)

    assert report.allowed is False
    assert any(
        check.label == "financial sanity: probability Brier score" and check.status == "block"
        for check in report.checks
    )
    with pytest.raises(ModelPromotionError, match="probability Brier score"):
        assert_model_promoted(bad)
