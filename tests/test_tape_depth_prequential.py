from __future__ import annotations

import csv
from dataclasses import replace
from datetime import UTC, datetime
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from simple_ai_trading.tape_depth_features import (
    TAPE_DEPTH_FEATURE_NAMES,
    TAPE_DEPTH_FEATURE_VERSION,
    TAPE_DEPTH_TARGET_MODE,
    TapeDepthForecastDataset,
)
from simple_ai_trading.tape_depth_model import (
    TapeDepthPredictionBatch,
    TapeDepthSignalPolicy,
    save_tape_depth_model_artifact,
    score_tape_depth_evaluation,
    train_tape_depth_forecaster,
)
from simple_ai_trading import tape_depth_prequential as prequential
from simple_ai_trading.tape_depth_prequential import (
    TapeDepthFoldEvaluation,
    TapeDepthFoldPlan,
    TapeDepthSymbolPlan,
    plan_tape_depth_folds,
    read_tape_depth_predictions,
    render_tape_depth_prequential_svg,
    run_tape_depth_prequential,
    write_tape_depth_predictions,
)


def _ms(value: str) -> int:
    return int(datetime.fromisoformat(value).replace(tzinfo=UTC).timestamp() * 1_000)


def test_tape_depth_plan_uses_two_year_training_and_nonoverlapping_evaluation() -> None:
    plan = plan_tape_depth_folds(
        symbol="BTCUSDT",
        source_first_second_ms=_ms("2020-01-01T00:00:00"),
        source_last_second_ms=_ms("2024-12-31T23:59:59"),
    )

    assert plan.training_window_days == 730
    assert len(plan.folds) > 7
    first = plan.folds[0]
    assert first.tuning_start_ms - first.dataset_start_ms == 730 * 86_400_000
    assert first.calibration_start_ms - first.tuning_start_ms == 30 * 86_400_000
    assert first.evaluation_start_ms - first.calibration_start_ms == 30 * 86_400_000
    assert first.estimated_dataset_rows == 880 * 24 * 60 * 3
    assert first.estimated_evaluation_rows == 90 * 24 * 60 * 3
    assert all(
        right.evaluation_start_ms == left.evaluation_end_ms + 20_000
        for left, right in zip(plan.folds, plan.folds[1:])
    )
    assert plan.fold_start == 0
    assert plan.available_fold_count == len(plan.folds)
    assert len(plan.coverage_fingerprint) == 64
    assert len(plan.plan_fingerprint) == 64


def test_tape_depth_plan_is_deterministic_and_contract_sensitive() -> None:
    options = {
        "symbol": "ETHUSDT",
        "source_first_second_ms": _ms("2019-11-27T00:00:00"),
        "source_last_second_ms": _ms("2026-07-09T23:59:59"),
        "max_folds": 2,
    }
    first = plan_tape_depth_folds(**options)
    second = plan_tape_depth_folds(**options)
    changed = plan_tape_depth_folds(**options, horizon_seconds=120)

    assert first == second
    assert len(first.folds) == 2
    assert first.plan_fingerprint != changed.plan_fingerprint


def test_tape_depth_plan_preserves_full_coverage_identity_across_fold_windows() -> None:
    options = {
        "symbol": "BTCUSDT",
        "source_first_second_ms": _ms("2019-01-01T00:00:00"),
        "source_last_second_ms": _ms("2026-07-09T23:59:59"),
    }
    full = plan_tape_depth_folds(**options)
    screening = plan_tape_depth_folds(**options, max_folds=3)
    confirmation = plan_tape_depth_folds(**options, fold_start=3)

    assert [fold.fold_index for fold in screening.folds] == [0, 1, 2]
    assert [fold.fold_index for fold in confirmation.folds] == list(
        range(3, full.available_fold_count)
    )
    assert screening.available_fold_count == confirmation.available_fold_count
    assert screening.coverage_fingerprint == confirmation.coverage_fingerprint
    assert screening.plan_fingerprint != confirmation.plan_fingerprint


def test_tape_depth_plan_rejects_a_fold_above_memory_bound() -> None:
    with pytest.raises(ValueError, match="maximum_rows=100000"):
        plan_tape_depth_folds(
            symbol="SOLUSDT",
            source_first_second_ms=_ms("2020-09-14T00:00:00"),
            source_last_second_ms=_ms("2026-07-09T23:59:59"),
            maximum_rows=100_000,
        )


def test_tape_depth_plan_rejects_negative_fold_window() -> None:
    with pytest.raises(ValueError, match="fold_start and max_folds"):
        plan_tape_depth_folds(
            symbol="BTCUSDT",
            source_first_second_ms=_ms("2020-01-01T00:00:00"),
            source_last_second_ms=_ms("2024-12-31T23:59:59"),
            fold_start=-1,
        )


@pytest.mark.parametrize("cadence", [0, 7, 61])
def test_tape_depth_plan_requires_clock_aligned_cadence(cadence: int) -> None:
    with pytest.raises(ValueError, match="must divide 60"):
        plan_tape_depth_folds(
            symbol="BTCUSDT",
            source_first_second_ms=_ms("2020-01-01T00:00:00"),
            source_last_second_ms=_ms("2024-12-31T23:59:59"),
            decision_cadence_seconds=cadence,
        )


def test_tape_depth_study_stages_fail_before_accessing_unsealed_data(tmp_path) -> None:
    with pytest.raises(ValueError, match="4, 6, 8, or 10 folds"):
        run_tape_depth_prequential(
            object(),  # type: ignore[arg-type]
            symbols=("BTCUSDT",),
            output_dir=tmp_path / "screening",
            study_stage="screening",
            max_folds=1,
        )
    with pytest.raises(ValueError, match="verified selection lock"):
        run_tape_depth_prequential(
            object(),  # type: ignore[arg-type]
            symbols=("BTCUSDT",),
            output_dir=tmp_path / "confirmation",
            study_stage="confirmation",
        )

    with pytest.raises(ValueError, match="study_stage"):
        run_tape_depth_prequential(
            object(),  # type: ignore[arg-type]
            symbols=("BTCUSDT",),
            output_dir=tmp_path / "invalid-stage",
            study_stage="invalid",
        )
    with pytest.raises(ValueError, match="only for confirmation"):
        run_tape_depth_prequential(
            object(),  # type: ignore[arg-type]
            symbols=("BTCUSDT",),
            output_dir=tmp_path / "development-lock",
            selection_lock="selection.json",
        )
    with pytest.raises(ValueError, match="boundaries come only"):
        run_tape_depth_prequential(
            object(),  # type: ignore[arg-type]
            symbols=("BTCUSDT",),
            output_dir=tmp_path / "manual-confirmation-window",
            study_stage="confirmation",
            selection_lock="selection.json",
            max_folds=1,
        )


def test_tape_depth_confirmation_derives_frozen_winner_and_suffix(
    tmp_path,
    monkeypatch,
) -> None:
    plan = plan_tape_depth_folds(
        symbol="BTCUSDT",
        source_first_second_ms=_ms("2019-01-01T00:00:00"),
        source_last_second_ms=_ms("2026-07-09T23:59:59"),
        fold_start=4,
    )
    selection = {
        "selected_horizon_seconds": 300,
        "selected_decision_cadence_seconds": 5,
        "selected_maximum_depth_age_ms": 30_000,
        "selected_model_profile": "expressive",
        "selected_feature_set": "cross_asset",
        "confirmation_fold_start": 4,
    }
    calls: dict[str, object] = {}

    def planned(_warehouse, **options):
        calls["plan_options"] = options
        return (plan,)

    def validated(payload, **options):
        calls["selection"] = payload
        calls["validation"] = options

    monkeypatch.setattr(prequential, "plan_tape_depth_warehouse", planned)
    monkeypatch.setattr(
        "simple_ai_trading.tape_depth_comparison.load_verified_tape_depth_selection",
        lambda _path: (selection, "a" * 64),
    )
    monkeypatch.setattr(
        "simple_ai_trading.tape_depth_comparison.validate_tape_depth_confirmation_request",
        validated,
    )

    report = run_tape_depth_prequential(
        object(),  # type: ignore[arg-type]
        symbols=("BTCUSDT",),
        output_dir=tmp_path / "confirmation-plan",
        study_stage="confirmation",
        selection_lock="selection.json",
        plan_only=True,
    )

    plan_options = calls["plan_options"]
    assert isinstance(plan_options, dict)
    assert plan_options["fold_start"] == 4
    assert plan_options["max_folds"] == 0
    assert plan_options["horizon_seconds"] == 300
    assert plan_options["decision_cadence_seconds"] == 5
    assert report["config"]["maximum_depth_age_ms"] == 30_000  # type: ignore[index]
    assert report["config"]["model_profile"] == "expressive"  # type: ignore[index]
    assert report["config"]["feature_set"] == "cross_asset"  # type: ignore[index]
    assert report["config"]["selection_lock_sha256"] == "a" * 64  # type: ignore[index]
    assert calls["selection"] == selection


@pytest.mark.parametrize(
    ("model_profile", "feature_set", "message"),
    [
        ("balanced", None, "model profile differs"),
        (None, "core", "feature set differs"),
    ],
)
def test_tape_depth_confirmation_rejects_manual_winner_override(
    tmp_path,
    monkeypatch,
    model_profile: str | None,
    feature_set: str | None,
    message: str,
) -> None:
    selection = {
        "selected_horizon_seconds": 60,
        "selected_decision_cadence_seconds": 20,
        "selected_maximum_depth_age_ms": 60_000,
        "selected_model_profile": "expressive",
        "selected_feature_set": "full",
        "confirmation_fold_start": 4,
    }
    monkeypatch.setattr(
        "simple_ai_trading.tape_depth_comparison.load_verified_tape_depth_selection",
        lambda _path: (selection, "b" * 64),
    )

    with pytest.raises(ValueError, match=message):
        run_tape_depth_prequential(
            object(),  # type: ignore[arg-type]
            symbols=("BTCUSDT",),
            output_dir=tmp_path / message.replace(" ", "-"),
            study_stage="confirmation",
            selection_lock="selection.json",
            model_profile=model_profile,
            feature_set=feature_set,
            plan_only=True,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"horizon_seconds": 60}, "horizon differs"),
        ({"decision_cadence_seconds": 20}, "cadence differs"),
        ({"maximum_depth_age_ms": 60_000}, "depth age differs"),
    ],
)
def test_tape_depth_confirmation_rejects_manual_timing_override(
    tmp_path,
    monkeypatch,
    override: dict[str, int],
    message: str,
) -> None:
    selection = {
        "selected_horizon_seconds": 300,
        "selected_decision_cadence_seconds": 5,
        "selected_maximum_depth_age_ms": 30_000,
        "selected_model_profile": "regularized",
        "selected_feature_set": "full",
        "confirmation_fold_start": 4,
    }
    monkeypatch.setattr(
        "simple_ai_trading.tape_depth_comparison.load_verified_tape_depth_selection",
        lambda _path: (selection, "b" * 64),
    )

    with pytest.raises(ValueError, match=message):
        run_tape_depth_prequential(
            object(),  # type: ignore[arg-type]
            symbols=("BTCUSDT",),
            output_dir=tmp_path / message.replace(" ", "-"),
            study_stage="confirmation",
            selection_lock="selection.json",
            plan_only=True,
            **override,
        )


def test_tape_depth_study_rejects_insufficient_screening_and_confirmation_plans(
    tmp_path,
    monkeypatch,
) -> None:
    base = plan_tape_depth_folds(
        symbol="BTCUSDT",
        source_first_second_ms=_ms("2019-01-01T00:00:00"),
        source_last_second_ms=_ms("2026-07-09T23:59:59"),
        max_folds=4,
    )
    shallow_screening = replace(base, available_fold_count=5)
    monkeypatch.setattr(
        prequential,
        "plan_tape_depth_warehouse",
        lambda *_args, **_kwargs: (shallow_screening,),
    )
    with pytest.raises(ValueError, match="at least two sealed folds"):
        run_tape_depth_prequential(
            object(),  # type: ignore[arg-type]
            symbols=("BTCUSDT",),
            output_dir=tmp_path / "shallow-screening",
            study_stage="screening",
            max_folds=4,
            plan_only=True,
        )

    one_fold = replace(
        base,
        fold_start=4,
        max_folds=0,
        available_fold_count=5,
        folds=(replace(base.folds[0], fold_index=4),),
    )
    selection = {
        "selected_horizon_seconds": 60,
        "selected_decision_cadence_seconds": 20,
        "selected_maximum_depth_age_ms": 60_000,
        "selected_model_profile": "regularized",
        "selected_feature_set": "full",
        "confirmation_fold_start": 4,
    }
    monkeypatch.setattr(
        prequential,
        "plan_tape_depth_warehouse",
        lambda *_args, **_kwargs: (one_fold,),
    )
    monkeypatch.setattr(
        "simple_ai_trading.tape_depth_comparison.load_verified_tape_depth_selection",
        lambda _path: (selection, "c" * 64),
    )
    with pytest.raises(ValueError, match="at least two untouched folds"):
        run_tape_depth_prequential(
            object(),  # type: ignore[arg-type]
            symbols=("BTCUSDT",),
            output_dir=tmp_path / "shallow-confirmation",
            study_stage="confirmation",
            selection_lock="selection.json",
            plan_only=True,
        )


def test_tape_depth_prediction_table_is_complete_and_deterministic(tmp_path) -> None:
    batch = TapeDepthPredictionBatch(
        decision_time_ms=np.asarray([1_000, 2_000], dtype=np.int64),
        target_entry_time_ms=np.asarray([2_000, 3_000], dtype=np.int64),
        target_exit_time_ms=np.asarray([62_000, 63_000], dtype=np.int64),
        actual_gross_return_bps=np.asarray([1.25, -0.75]),
        direction_probability=np.asarray([0.8, 0.2]),
        mean_prediction_bps=np.asarray([1.1, -0.5]),
        lower_prediction_bps=np.asarray([-0.2, -1.4]),
        upper_prediction_bps=np.asarray([2.2, 0.3]),
        signal_policy=TapeDepthSignalPolicy(
            risk_level="conservative",
            magnitude_quantile=0.95,
            minimum_direction_probability=0.60,
            interval_width_quantile=0.75,
            signal_threshold_bps=1.0,
            maximum_interval_width_bps=3.0,
            direction_baseline_probability=0.5,
        ),
    )
    path = tmp_path / "predictions.csv.gz"

    first_hash = write_tape_depth_predictions(batch, path)
    first_bytes = path.read_bytes()
    second_hash = write_tape_depth_predictions(batch, path)

    assert first_hash == second_hash
    assert first_bytes == path.read_bytes()
    assert read_tape_depth_predictions(path).fingerprint() == batch.fingerprint()
    with gzip.open(path, "rt", encoding="ascii", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["decision_time_ms"] == "1000"
    assert rows[1]["actual_gross_return_bps"] == "-0.75"
    assert rows[0]["signal_threshold_bps"] == "1"

    rows[1]["minimum_direction_probability"] = "0.61"
    tampered = tmp_path / "tampered-predictions.csv.gz"
    with gzip.open(tampered, "wt", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="numeric contract"):
        read_tape_depth_predictions(tampered)


def test_dataset_range_source_evidence_uses_exact_causal_boundaries(monkeypatch) -> None:
    class Connection:
        prior = (9_098_000,)

        def execute(self, _query, _parameters):
            return self

        def fetchone(self):
            return self.prior

    class Warehouse:
        connection = Connection()

        def connect(self):
            return self.connection

    captured: dict[str, object] = {}

    def source_evidence(_warehouse, symbol, **kwargs):
        captured["symbol"] = symbol
        captured.update(kwargs)
        return {"verified": True, "manifest_fingerprint": "a" * 64}

    monkeypatch.setattr(
        prequential,
        "tape_depth_dataset_source_evidence",
        source_evidence,
    )
    warehouse = Warehouse()
    evidence = prequential._dataset_range_source_evidence(
        warehouse,
        symbol="BTCUSDT",
        start_ms=10_000_000,
        end_ms=20_000_000,
        horizon_seconds=60,
        total_latency_ms=750,
    )

    assert evidence["verified"] is True
    assert captured == {
        "symbol": "BTCUSDT",
        "required_start_ms": 9_098_000,
        "required_end_ms": 20_060_000,
        "peer_feature_start_ms": 9_999_000,
        "peer_feature_end_ms": 19_999_000,
    }
    warehouse.connection.prior = (None,)
    with pytest.raises(ValueError, match="no prior verified trade"):
        prequential._dataset_range_source_evidence(
            warehouse,
            symbol="BTCUSDT",
            start_ms=10_000_000,
            end_ms=20_000_000,
            horizon_seconds=60,
            total_latency_ms=750,
        )


def test_tape_depth_prequential_plan_only_writes_no_model_claim(tmp_path) -> None:
    class Result:
        def fetchone(self):
            return (_ms("2020-01-01T00:00:00"), _ms("2024-12-31T23:59:59"))

    class Connection:
        def execute(self, _query, _parameters):
            return Result()

    class Warehouse:
        def connect(self):
            return Connection()

    report = run_tape_depth_prequential(
        Warehouse(),
        symbols=("BTCUSDT",),
        output_dir=tmp_path,
        max_folds=2,
        plan_only=True,
    )

    assert report["plan_only"] is True
    assert report["trading_authority"] is False
    assert report["execution_claim"] is False
    assert report["total_folds"] == 2
    assert (tmp_path / "plan.json").is_file()
    assert not (tmp_path / "report.json").exists()


def test_tape_depth_diagnostics_chart_uses_real_utc_axis_and_caveat(tmp_path) -> None:
    folds = []
    for symbol_index, symbol in enumerate(("BTCUSDT", "ETHUSDT")):
        for fold_index, date in enumerate(("2025-01-31T00:00:00", "2025-03-02T00:00:00")):
            folds.append(
                {
                    "symbol": symbol,
                    "fold_index": fold_index,
                    "evaluation_end_ms": _ms(date),
                    "metrics": {
                        "direction_auc": 0.52 + 0.01 * symbol_index + 0.005 * fold_index,
                        "spearman_information_coefficient": 0.02 + 0.01 * fold_index,
                        "calibration_threshold_mean_signed_gross_bps": 0.3
                        + 0.1 * symbol_index,
                    },
                }
            )
    path = tmp_path / "diagnostics.svg"

    first_hash = render_tape_depth_prequential_svg(folds, path)
    first_text = path.read_text(encoding="ascii")
    second_hash = render_tape_depth_prequential_svg(folds, path)

    assert first_hash == second_hash
    assert first_text == path.read_text(encoding="ascii")
    assert "2025-01-31" in first_text
    assert "2025-03-02" in first_text
    assert "no spread, fees, fills, or ROI" in first_text
    assert "BTCUSDT" in first_text
    assert "ETHUSDT" in first_text


def _resume_dataset(rows: int = 3_000) -> TapeDepthForecastDataset:
    rng = np.random.default_rng(73)
    features = rng.normal(
        size=(rows, len(TAPE_DEPTH_FEATURE_NAMES))
    ).astype(np.float32)
    signal = features[:, 0] + 0.4 * features[:, 1]
    targets = 2.5 * signal + rng.normal(0.0, 0.25, size=rows)
    base_ms = 1_700_000_000_000
    decision_times = base_ms + np.arange(rows, dtype=np.int64) * 20_000
    prices = 100.0 + np.arange(rows, dtype=np.float64) * 0.001
    return TapeDepthForecastDataset(
        symbol="BTCUSDT",
        feature_version=TAPE_DEPTH_FEATURE_VERSION,
        feature_names=TAPE_DEPTH_FEATURE_NAMES,
        target_mode=TAPE_DEPTH_TARGET_MODE,
        horizon_seconds=60,
        total_latency_ms=750,
        decision_cadence_seconds=20,
        maximum_depth_age_ms=60_000,
        decision_time_ms=decision_times,
        target_entry_time_ms=decision_times + 1_000,
        target_exit_time_ms=decision_times + 61_000,
        target_entry_price=prices,
        target_exit_price=prices * (1.0 + targets / 10_000.0),
        gross_return_bps=targets,
        features=features,
        source_evidence={
            "verified": True,
            "schema_version": "binance-usdm-tick-v6",
            "manifest_fingerprint": "d" * 64,
        },
    )


def test_tape_depth_resume_rejects_path_escape_and_skips_verified_fold(
    tmp_path,
    monkeypatch,
) -> None:
    dataset = _resume_dataset()
    artifact = train_tape_depth_forecaster(
        dataset,
        compute_backend="cpu",
        minimum_segment_rows=128,
    )
    predictions = score_tape_depth_evaluation(artifact, dataset)
    fold = TapeDepthFoldPlan(
        symbol="BTCUSDT",
        fold_index=0,
        dataset_start_ms=int(dataset.decision_time_ms[0]),
        tuning_start_ms=artifact.split.tuning_start_ms,
        calibration_start_ms=artifact.split.calibration_start_ms,
        evaluation_start_ms=artifact.split.evaluation_start_ms,
        evaluation_end_ms=int(predictions.decision_time_ms[-1]),
        estimated_dataset_rows=dataset.rows,
        estimated_evaluation_rows=predictions.rows,
    )
    symbol_plan = TapeDepthSymbolPlan(
        schema_version=prequential.TAPE_DEPTH_PREQUENTIAL_SCHEMA_VERSION,
        symbol="BTCUSDT",
        source_first_second_ms=int(dataset.decision_time_ms[0] - 901_000),
        source_last_second_ms=int(dataset.target_exit_time_ms[-1]),
        first_eligible_decision_ms=int(dataset.decision_time_ms[0]),
        last_eligible_decision_ms=int(dataset.decision_time_ms[-1]),
        training_window_days=730,
        tuning_window_days=30,
        calibration_window_days=30,
        evaluation_window_days=90,
        horizon_seconds=60,
        total_latency_ms=750,
            decision_cadence_seconds=20,
            maximum_rows=5_000_000,
            fold_start=0,
            max_folds=1,
            available_fold_count=1,
            folds=(fold,),
            coverage_fingerprint="d" * 64,
            plan_fingerprint="e" * 64,
    )
    monkeypatch.setattr(
        prequential,
        "plan_tape_depth_warehouse",
        lambda *_args, **_kwargs: (symbol_plan,),
    )
    prequential.run_tape_depth_prequential(
        object(),
        symbols=("BTCUSDT",),
        output_dir=tmp_path,
        max_folds=1,
        plan_only=True,
    )
    cache_events_path = tmp_path / "dataset-cache-events.json"
    cache_events_path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="cache events are unreadable"):
        prequential.run_tape_depth_prequential(
            object(),
            symbols=("BTCUSDT",),
            output_dir=tmp_path,
            max_folds=1,
            resume=True,
        )
    cache_events_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid contract"):
        prequential.run_tape_depth_prequential(
            object(),
            symbols=("BTCUSDT",),
            output_dir=tmp_path,
            max_folds=1,
            resume=True,
        )
    cache_events_path.unlink()
    artifact_relative = Path("models") / "btcusdt-fold-0000.json"
    predictions_relative = Path("predictions") / "btcusdt-fold-0000.csv.gz"
    artifact_path = tmp_path / artifact_relative
    predictions_path = tmp_path / predictions_relative
    save_tape_depth_model_artifact(artifact, artifact_path)
    prediction_hash = write_tape_depth_predictions(predictions, predictions_path)
    evaluation = TapeDepthFoldEvaluation(
        plan=fold,
        artifact=artifact,
        predictions=predictions,
    )
    summary = prequential._fold_summary(
        evaluation,
        artifact_path=artifact_relative,
        artifact_sha256=prequential._sha256_file(artifact_path),
        predictions_path=predictions_relative,
        predictions_sha256=prediction_hash,
    )
    summaries_path = tmp_path / "fold-summaries.json"
    escaped = {**summary, "artifact_path": "../outside.json"}
    summaries_path.write_text(json.dumps([escaped]), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid artifact_path"):
        prequential.run_tape_depth_prequential(
            object(),
            symbols=("BTCUSDT",),
            output_dir=tmp_path,
            max_folds=1,
            resume=True,
        )

    summaries_path.write_text(json.dumps([summary]), encoding="utf-8")
    monkeypatch.setattr(
        prequential,
        "evaluate_tape_depth_fold",
        lambda *_args, **_kwargs: pytest.fail("verified resume retrained a fold"),
    )
    progress: list[str] = []
    report = prequential.run_tape_depth_prequential(
        object(),
        symbols=("BTCUSDT",),
        output_dir=tmp_path,
        max_folds=1,
        resume=True,
        progress=lambda phase, _completed, _total: progress.append(phase),
    )

    assert report["completed_folds"] == 1
    assert progress == ["resume-verified"]
    report_path = tmp_path / "report.json"
    assert report_path.is_file()
    prequential.verify_tape_depth_prequential_report(report_path, report)
    drifted = json.loads(json.dumps(report))
    drifted["folds"][0]["metrics"]["direction_auc"] += 0.01
    with pytest.raises(ValueError, match="binding differs"):
        prequential.verify_tape_depth_prequential_report(report_path, drifted)
    escaped_report = json.loads(json.dumps(report))
    escaped_report["folds"][0]["artifact_path"] = "../outside.json"
    with pytest.raises(ValueError, match="invalid artifact_path"):
        prequential.verify_tape_depth_prequential_report(report_path, escaped_report)
    aggregate_drift = json.loads(json.dumps(report))
    aggregate_drift["aggregate_forecast_metrics"]["rows"] = 0
    with pytest.raises(ValueError, match="aggregate replay differs"):
        prequential.verify_tape_depth_prequential_report(report_path, aggregate_drift)
    plan_path = tmp_path / "plan.json"
    plan_text = plan_path.read_text(encoding="utf-8")
    plan_path.write_text("{}", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="plan evidence differs"):
            prequential.verify_tape_depth_prequential_report(report_path, report)
    finally:
        plan_path.write_text(plan_text, encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be resumed"):
        prequential.run_tape_depth_prequential(
            object(),
            symbols=("BTCUSDT",),
            output_dir=tmp_path,
            max_folds=1,
            resume=True,
        )


def test_tape_depth_prequential_reuses_verified_dataset_cache(
    tmp_path,
    monkeypatch,
) -> None:
    dataset = _resume_dataset()
    artifact = train_tape_depth_forecaster(
        dataset,
        compute_backend="cpu",
        minimum_segment_rows=128,
    )
    predictions = score_tape_depth_evaluation(artifact, dataset)
    fold = TapeDepthFoldPlan(
        symbol="BTCUSDT",
        fold_index=0,
        dataset_start_ms=int(dataset.decision_time_ms[0]),
        tuning_start_ms=artifact.split.tuning_start_ms,
        calibration_start_ms=artifact.split.calibration_start_ms,
        evaluation_start_ms=artifact.split.evaluation_start_ms,
        evaluation_end_ms=int(predictions.decision_time_ms[-1]),
        estimated_dataset_rows=dataset.rows,
        estimated_evaluation_rows=predictions.rows,
    )
    symbol_plan = TapeDepthSymbolPlan(
        schema_version=prequential.TAPE_DEPTH_PREQUENTIAL_SCHEMA_VERSION,
        symbol="BTCUSDT",
        source_first_second_ms=int(dataset.decision_time_ms[0] - 901_000),
        source_last_second_ms=int(dataset.target_exit_time_ms[-1]),
        first_eligible_decision_ms=int(dataset.decision_time_ms[0]),
        last_eligible_decision_ms=int(dataset.decision_time_ms[-1]),
        training_window_days=730,
        tuning_window_days=30,
        calibration_window_days=30,
        evaluation_window_days=90,
        horizon_seconds=60,
        total_latency_ms=750,
            decision_cadence_seconds=20,
            maximum_rows=dataset.rows,
            fold_start=0,
            max_folds=1,
            available_fold_count=1,
            folds=(fold,),
            coverage_fingerprint="d" * 64,
            plan_fingerprint="e" * 64,
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        prequential,
        "plan_tape_depth_warehouse",
        lambda *_args, **_kwargs: (symbol_plan,),
    )
    monkeypatch.setattr(
        prequential,
        "_dataset_range_source_evidence",
        lambda *_args, **_kwargs: dict(dataset.source_evidence),
    )
    monkeypatch.setattr(
        prequential,
        "train_tape_depth_forecaster",
        lambda *_args, **_kwargs: artifact,
    )
    monkeypatch.setattr(
        prequential,
        "score_tape_depth_evaluation",
        lambda *_args, **_kwargs: predictions,
    )
    direct = prequential.evaluate_tape_depth_fold(
        object(),
        plan=fold,
        horizon_seconds=60,
        total_latency_ms=750,
        decision_cadence_seconds=20,
        maximum_depth_age_ms=60_000,
        maximum_rows=dataset.rows,
        risk_level="conservative",
        model_profile="regularized",
        feature_set="full",
        compute_backend="cpu",
        minimum_segment_rows=128,
        prefetched_dataset=dataset,
    )
    assert direct.artifact is artifact

    def load_cache(_warehouse, **kwargs):
        captured["cache_load"] = kwargs
        return dataset

    def evaluate(_warehouse, **kwargs):
        captured["prefetched"] = kwargs["prefetched_dataset"]
        return TapeDepthFoldEvaluation(
            plan=fold,
            artifact=artifact,
            predictions=predictions,
        )

    monkeypatch.setattr(prequential, "load_tape_depth_dataset_cache", load_cache)
    monkeypatch.setattr(
        prequential,
        "tape_depth_dataset_cache_key",
        lambda **_kwargs: "f" * 64,
    )
    monkeypatch.setattr(
        prequential,
        "build_tape_depth_forecast_dataset",
        lambda *_args, **_kwargs: pytest.fail("cache hit rebuilt feature matrix"),
    )
    monkeypatch.setattr(
        prequential,
        "save_tape_depth_dataset_cache",
        lambda *_args, **_kwargs: pytest.fail("cache hit rewrote feature matrix"),
    )
    monkeypatch.setattr(prequential, "evaluate_tape_depth_fold", evaluate)

    report = prequential.run_tape_depth_prequential(
        object(),
        symbols=("BTCUSDT",),
        output_dir=tmp_path,
        maximum_cached_rows=dataset.rows,
        max_folds=1,
        dataset_cache=True,
    )

    assert captured["prefetched"] is dataset
    assert captured["cache_load"]["source_evidence"] == dataset.source_evidence  # type: ignore[index]
    assert report["dataset_cache"]["enabled"] is True  # type: ignore[index]
    cache_event = report["dataset_cache"]["events"][0]  # type: ignore[index]
    assert cache_event["state"] == "hit"
    assert cache_event["cache_key"] == "f" * 64
    assert (tmp_path / "dataset-cache-events.json").is_file()

    def build_cache_miss(_warehouse, **kwargs):
        captured["cache_build"] = kwargs
        return dataset

    def save_cache_miss(_warehouse, cached_dataset):
        captured["cache_save"] = cached_dataset
        return "a" * 64

    monkeypatch.setattr(
        prequential,
        "load_tape_depth_dataset_cache",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        prequential,
        "build_tape_depth_forecast_dataset",
        build_cache_miss,
    )
    monkeypatch.setattr(
        prequential,
        "save_tape_depth_dataset_cache",
        save_cache_miss,
    )
    miss_report = prequential.run_tape_depth_prequential(
        object(),
        symbols=("BTCUSDT",),
        output_dir=tmp_path / "cache-miss",
        maximum_cached_rows=dataset.rows,
        max_folds=1,
        dataset_cache=True,
    )

    assert captured["cache_save"] is dataset
    assert captured["cache_build"]["maximum_rows"] == dataset.rows  # type: ignore[index]
    miss_event = miss_report["dataset_cache"]["events"][0]  # type: ignore[index]
    assert miss_event["state"] == "written"
    assert miss_event["cache_key"] == "a" * 64

    bad_times = dataset.decision_time_ms.copy()
    bad_times[-1] += 20_000
    bad_dataset = replace(dataset, decision_time_ms=bad_times)
    monkeypatch.setattr(
        prequential,
        "load_tape_depth_dataset_cache",
        lambda *_args, **_kwargs: bad_dataset,
    )
    with pytest.raises(ValueError, match="planned row/time interval"):
        prequential.run_tape_depth_prequential(
            object(),
            symbols=("BTCUSDT",),
            output_dir=tmp_path / "bad-cache-hit",
            maximum_cached_rows=dataset.rows,
            max_folds=1,
            dataset_cache=True,
        )

    drifted_dataset = replace(
        dataset,
        source_evidence={
            **dataset.source_evidence,
            "manifest_fingerprint": "b" * 64,
        },
    )
    monkeypatch.setattr(
        prequential,
        "load_tape_depth_dataset_cache",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        prequential,
        "build_tape_depth_forecast_dataset",
        lambda *_args, **_kwargs: drifted_dataset,
    )
    with pytest.raises(ValueError, match="source evidence drifted"):
        prequential.run_tape_depth_prequential(
            object(),
            symbols=("BTCUSDT",),
            output_dir=tmp_path / "drifted-build",
            maximum_cached_rows=dataset.rows,
            max_folds=1,
            dataset_cache=True,
        )
