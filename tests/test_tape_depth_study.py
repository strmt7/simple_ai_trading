from __future__ import annotations

import json
from pathlib import Path

from simple_ai_trading.model_experiment import ChoiceDomain, generate_latin_hypercube_design
from simple_ai_trading.tape_depth_study import run_tape_depth_screening_study


def _design():
    return generate_latin_hypercube_design(
        (
            ChoiceDomain("risk_level", ("conservative",)),
            ChoiceDomain("horizon_seconds", (15, 300)),
            ChoiceDomain("decision_cadence_seconds", (1, 5)),
            ChoiceDomain("maximum_depth_age_ms", (15_000, 60_000)),
            ChoiceDomain("model_profile", ("regularized", "expressive")),
            ChoiceDomain("feature_set", ("core", "full")),
        ),
        sampled_count=2,
        seed=101,
    )


def test_screening_study_plan_is_design_bound_and_has_no_authority(tmp_path) -> None:
    design = _design()
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(design.asdict()), encoding="utf-8")

    report = run_tape_depth_screening_study(
        object(),
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        design_path=design_path,
        output_dir=tmp_path / "study",
        plan_only=True,
    )

    assert report["status"] == "planned_research_only"
    assert report["completed_candidates"] == 0
    assert report["trading_authority"] is False
    assert report["terminal_holdout_consumed"] is False
    assert len(report["candidate_plan"]) == design.trial_burden
    assert (tmp_path / "study" / "study-plan.json").is_file()


def test_screening_study_runs_sequentially_and_resumes_verified_reports(
    tmp_path,
    monkeypatch,
) -> None:
    design = _design()
    design_path = tmp_path / "design.json"
    design_path.write_text(json.dumps(design.asdict()), encoding="utf-8")
    calls: list[dict[str, object]] = []
    selection_calls: list[dict[str, object]] = []

    def run(_warehouse, **options):
        config = {
            key: value
            for key, value in options.items()
            if key
            in {
                "symbols",
                "training_window_days",
                "tuning_window_days",
                "calibration_window_days",
                "evaluation_window_days",
                "horizon_seconds",
                "total_latency_ms",
                "decision_cadence_seconds",
                "maximum_depth_age_ms",
                "maximum_rows",
                "maximum_cached_rows",
                "dataset_cache",
                "study_stage",
                "max_folds",
                "risk_level",
                "model_profile",
                "feature_set",
                "compute_backend",
                "minimum_segment_rows",
            }
        }
        config["symbols"] = list(config["symbols"])
        config["fold_start"] = 0
        config["selection_lock_sha256"] = None
        report = {
            "status": "research_candidate",
            "trading_authority": False,
            "execution_claim": False,
            "profitability_claim": False,
            "completed_folds": 12,
            "total_folds": 12,
            "config": config,
        }
        destination = Path(options["output_dir"])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "report.json").write_text(json.dumps(report), encoding="utf-8")
        calls.append(options)
        return report

    def select(paths, **options):
        selection_calls.append({"paths": tuple(paths), **options})
        selection = {
            "status": "winner_frozen",
            "selected_trial": "candidate",
        }
        Path(options["output"]).write_text(json.dumps(selection), encoding="utf-8")
        return selection

    monkeypatch.setattr("simple_ai_trading.tape_depth_study.run_tape_depth_prequential", run)
    monkeypatch.setattr(
        "simple_ai_trading.tape_depth_study.verify_tape_depth_prequential_report",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "simple_ai_trading.tape_depth_study.load_and_select_tape_depth_reports",
        select,
    )
    output_dir = tmp_path / "study"
    first = run_tape_depth_screening_study(
        object(),
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        design_path=design_path,
        output_dir=output_dir,
        compute_backend="directml",
    )

    assert first["status"] == "winner_frozen"
    assert first["completed_candidates"] == design.trial_burden
    assert len(calls) == design.trial_burden
    assert all(call["study_stage"] == "screening" for call in calls)
    assert all(call["max_folds"] == 4 for call in calls)
    assert selection_calls[0]["design_path"] == design_path.resolve()

    calls.clear()
    resumed = run_tape_depth_screening_study(
        object(),
        symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
        design_path=design_path,
        output_dir=output_dir,
        compute_backend="directml",
        resume=True,
    )

    assert resumed["status"] == "winner_frozen"
    assert calls == []
    assert all(item["reused"] is True for item in resumed["candidate_results"])
