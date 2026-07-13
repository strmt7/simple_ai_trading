from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / "docs" / "model-research" / "action-value" / "latest"


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _csv(path: str) -> list[dict[str, str]]:
    with (LATEST / path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_latest_action_value_publication_is_round39_hash_verified() -> None:
    report = json.loads((LATEST / "report.json").read_text(encoding="utf-8"))
    canonical = dict(report)
    claimed = canonical.pop("publication_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert claimed == "42c27ebf3739b719901f8e71db312cb368c0ce8611468faf8017f03dc4a9219a"
    assert report["schema_version"] == (
        "causal-refit-utility-ai-ablation-publication-v1"
    )
    assert report["round"] == 39
    assert report["status"] == "rejected"
    assert report["source_implementation_commit"] == (
        "474dedb4c3b645cb2d9b8199b6b78619c07e031c"
    )
    assert report["source_report_canonical_sha256"] == (
        "6f95bd8cd238c8a0aee3cb147c477fb6a73edce404ce35bb857a9c411ea7627f"
    )
    assert report["source_report_file_sha256"] == (
        "c987f021b1630cf07a40bad40be76c12f2bef39561251dc8d8e12fa504332598"
    )
    assert report["source_ai_case_file_sha256"] == (
        "eaea3973d67e46c1ef1ab3a31a5c866405445a1629aa245e740c3713cb37acfe"
    )
    assert report["dataset_rows"] == 1_098_105
    assert report["model_feature_count"] == 71
    assert report["gpu_model_artifact_count"] == 60
    assert report["candidate_count"] == 4
    assert report["monthly_result_count"] == 24
    assert report["threshold_cell_count"] == 480
    assert report["ai_case_count"] == 180
    assert report["ai_decision_count"] == 360
    assert report["ai_provider_failure_count"] == 0
    assert report["aggregate_ml_gate_passed_candidate_count"] == 0
    assert report["ai_uplift_gate_passed_model_count"] == 0
    assert report["confidence_capacity_bootstrap_samples"] == 50_000
    assert report["confidence_capacity_bootstrap_seed"] == 3939
    assert report["confidence_capacity_lower_95_mean_net_bps"] == (
        -0.17872218697091233
    )
    for field in (
        "trading_authority",
        "execution_claim",
        "profitability_claim",
        "portfolio_claim",
        "leverage_applied",
    ):
        assert report[field] is False

    for artifact in report["artifact_integrity"]:
        path = LATEST / artifact["path"]
        assert path.is_file()
        assert path.stat().st_size == artifact["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
        if path.suffix == ".csv":
            assert len(_csv(artifact["path"])) == artifact["row_count"]


def test_latest_action_value_source_report_is_exact_and_fail_closed() -> None:
    source = json.loads((LATEST / "screen.json").read_text(encoding="utf-8"))
    canonical = dict(source)
    claimed = canonical.pop("report_canonical_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert source["round"] == 39
    assert source["status"] == "rejected"
    assert source["implementation_commit"] == (
        "474dedb4c3b645cb2d9b8199b6b78619c07e031c"
    )
    assert source["dataset"] == {
        "derivatives_risk_and_accounting_feature_count": 32,
        "feature_count": 103,
        "features_bytes": 452_419_260,
        "features_dtype": "float32",
        "horizons_minutes": [30, 120],
        "model_feature_count": 71,
        "persistent_feature_copy_created": False,
        "rows": 1_098_105,
        "source_exclusions": {
            "base_decision_rows": 367_752,
            "eligible_cross_symbol_rows": 1_098_105,
            "eligible_decision_times": 366_035,
            "premium_quality_excluded_decision_times": 1_717,
        },
    }
    assert source["backend"] == {
        "device": "opencl:auto",
        "gpu_first_requested": True,
        "kind": "opencl",
        "python_dependencies": {
            "lightgbm": "4.6.0",
            "numpy": "2.2.6",
            "scipy": "1.17.1",
        },
    }
    assert source["runtime_evidence"]["memory"]["peak_working_set_bytes"] == (
        4_662_816_768
    )
    assert len(source["model_artifacts"]) == 60
    assert all(
        artifact["reload_max_abs_prediction_error"] == 0.0
        for artifact in source["model_artifacts"]
    )
    assert source["aggregate_ml_gate_passed_candidates"] == []
    assert source["ai_case_set"]["cases"] == 180
    assert source["ai_case_set"]["case_set_sha256"] == (
        "fb94623fc8c5906bfb8f18a951def9ca19bae7d66ca3d2e91ef4d3cb93e5985f"
    )
    assert len(source["ai_reports"]) == 2
    assert all(report["provider_failures"] == 0 for report in source["ai_reports"])
    assert all(not report["uplift_gate_passed"] for report in source["ai_reports"])
    assert source["ai_uplift_gate_passed_models"] == []
    assert source["model_selection_on_evaluation_permitted"] is False
    assert source["ai_model_selection_on_evaluation_permitted"] is False
    assert source["selection_confirmation_accessed"] is False
    assert source["terminal_2026_accessed"] is False
    for field in (
        "trading_authority",
        "execution_claim",
        "profitability_claim",
        "portfolio_claim",
        "leverage_applied",
    ):
        assert source[field] is False


def test_latest_action_value_tables_preserve_round39_graph_data() -> None:
    candidates = _csv("candidates.csv")
    monthly = _csv("monthly.csv")
    thresholds = _csv("thresholds.csv")
    models = _csv("models.csv")
    ai_cases = _csv("ai-cases.csv")
    ai_models = _csv("ai-models.csv")
    ai_decisions = _csv("ai-decisions.csv")
    capacity = _csv("capacity-summary.csv")
    uplift = _csv("utility-uplift.csv")
    sources = _csv("sources.csv")

    assert [
        len(table)
        for table in (
            candidates,
            monthly,
            thresholds,
            models,
            ai_cases,
            ai_models,
            ai_decisions,
            capacity,
            uplift,
            sources,
        )
    ] == [4, 24, 480, 60, 180, 2, 360, 14, 2, 3]
    assert {row["architecture"] for row in candidates} == {
        "shared_two_stage_hurdle_lightgbm",
        "per_symbol_direct_multiclass_lightgbm",
    }
    assert {row["weighting"] for row in candidates} == {
        "equal",
        "bounded_economic_utility",
    }
    assert all(row["ai_entry_support_passed"] == "True" for row in candidates)
    assert all(row["aggregate_ml_gate_passed"] == "False" for row in candidates)
    assert all(float(row["mean_net_bps"]) < 0.0 for row in candidates)
    assert all(float(row["profit_factor"]) < 1.0 for row in candidates)
    assert all(float(row["reload_max_abs_prediction_error"]) == 0.0 for row in models)
    assert all(row["backend_kind"] == "opencl" for row in models)
    assert {row["evaluation_month"] for row in monthly} == {
        "2025-01",
        "2025-02",
        "2025-03",
        "2025-04",
        "2025-05",
        "2025-06",
    }
    assert {row["symbol"] for row in ai_cases} == {
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
    }
    assert all(
        sum(row["symbol"] == symbol for row in ai_cases) == 60
        for symbol in ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    )
    assert all(row["valid"] == "True" for row in ai_decisions)
    assert {row["model"] for row in ai_models} == {"qwen3:8b", "fino1:8b"}
    assert all(row["uplift_gate_passed"] == "False" for row in ai_models)
    assert {row["symbol"] for row in sources} == {
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
    }
    assert all(int(row["price_rows"]) == 1_883_520 for row in sources)
    assert all(int(row["price_gap_count"]) == 0 for row in sources)

    best = max(candidates, key=lambda row: float(row["mean_net_bps"]))
    assert best["candidate_id"] == "rolling_shared_hurdle_h120_utility"
    assert int(best["total_trades"]) == 2_796
    assert float(best["mean_net_bps"]) == -9.908254582381598
    assert float(best["profit_factor"]) == 0.7885771104997189
    assert float(best["day_block_bootstrap_mean_net_bps_lower_95"]) == (
        -15.385173536256483
    )

    overall = next(row for row in capacity if row["scope"] == "overall")
    assert int(overall["trades"]) == 180
    assert float(overall["mean_net_bps"]) == 18.66688779551122
    assert float(overall["day_block_lower_95_mean_net_bps"]) == (
        -0.17872218697091233
    )
    eth = next(
        row
        for row in capacity
        if row["scope"] == "symbol" and row["member"] == "ETHUSDT"
    )
    sol = next(
        row
        for row in capacity
        if row["scope"] == "symbol" and row["member"] == "SOLUSDT"
    )
    assert float(eth["total_net_bps"]) < 0.0
    assert float(sol["total_net_bps"]) == 2944.4864406585693


def test_latest_action_value_progress_extends_to_round39_without_policy_claim() -> None:
    progress = _csv("progress.csv")

    assert [int(row["round"]) for row in progress] == list(range(1, 40))
    latest = progress[-1]
    assert latest["status"] == "rejected"
    assert latest["selection_contaminated"] == "True"
    assert latest["risk_level"] == "research-only; no policy"
    assert latest["selected_signals"] == "2796"
    assert latest["executable_trades"] == "2796"
    assert float(latest["mean_net_bps"]) == -9.908254582381598
    assert latest["accepted_thresholds"] == "0"


def test_latest_action_value_charts_are_accessible_and_prior_files_are_absent() -> None:
    expected_charts = {
        "ai-ablation.svg",
        "confidence-capacity.svg",
        "monthly-economics.svg",
        "research-progress.svg",
        "rolling-candidate-economics.svg",
    }
    charts = {path.name for path in (LATEST / "charts").glob("*.svg")}

    assert charts == expected_charts
    for chart in (LATEST / "charts").glob("*.svg"):
        document = ET.parse(chart).getroot()
        namespace = "{http://www.w3.org/2000/svg}"
        assert document.attrib["role"] == "img"
        assert document.find(f"{namespace}title") is not None
        assert document.find(f"{namespace}desc") is not None
        text = chart.read_text(encoding="utf-8").casefold()
        assert ">nan<" not in text
        assert '="nan"' not in text
        assert ">inf<" not in text
        assert '="inf"' not in text

    for stale in (
        "derivatives-uplift.csv",
        "calibration-to-viability.svg",
        "derivatives-feature-ablation.svg",
        "viability-economics.svg",
    ):
        assert not (LATEST / stale).exists()
        assert not (LATEST / "charts" / stale).exists()

    readme = (LATEST / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Round 39: rolling ML and local AI rejected")
    assert "# Round 38:" not in readme
    assert "completed evaluation month" in readme
    assert "unavailable to a live controller" in readme
    assert "No AI, ML, ROI, portfolio, leverage, execution" in readme
