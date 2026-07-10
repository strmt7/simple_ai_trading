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


def test_tape_depth_plan_rejects_a_fold_above_memory_bound() -> None:
    with pytest.raises(ValueError, match="maximum_rows=100000"):
        plan_tape_depth_folds(
            symbol="SOLUSDT",
            source_first_second_ms=_ms("2020-09-14T00:00:00"),
            source_last_second_ms=_ms("2026-07-09T23:59:59"),
            maximum_rows=100_000,
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
                        "top_decile_mean_signed_gross_bps": 0.3 + 0.1 * symbol_index,
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
        max_folds=1,
        folds=(fold,),
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
    assert (tmp_path / "report.json").is_file()
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
        max_folds=1,
        folds=(fold,),
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
