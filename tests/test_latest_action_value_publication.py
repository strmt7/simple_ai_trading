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


def test_latest_action_value_publication_is_round38_hash_verified() -> None:
    report = json.loads((LATEST / "report.json").read_text(encoding="utf-8"))
    canonical = dict(report)
    claimed = canonical.pop("publication_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert claimed == "89f3ebbf6e37a7c80c6f26e6fedd831fed9a0a8b5f6420a4940c0fcff7b490f9"
    assert report["schema_version"] == (
        "derivatives-hurdle-ai-ablation-publication-v1"
    )
    assert report["round"] == 38
    assert report["status"] == "rejected"
    assert report["source_implementation_commit"] == (
        "dc19f72e69f79e93ed40d97deea855a4c6a0b4aa"
    )
    assert report["source_report_canonical_sha256"] == (
        "7b7f3a5ba4ab1a047fc6d06ca4c3e90b6bdc210c621d7bcca2d887629f9e4c42"
    )
    assert report["source_report_file_sha256"] == (
        "7b39beb7726a2b94b564f77fb766b207592f8979f1828f37bfceeae7970a7346"
    )
    assert report["dataset_rows"] == 1_098_105
    assert report["feature_count"] == 103
    assert report["gpu_model_artifact_count"] == 96
    assert report["candidate_count"] == 32
    assert report["threshold_cell_count"] == 640
    assert report["supported_threshold_cell_count"] == 295
    assert report["selected_threshold_count"] == 32
    assert report["viability_gate_passed_candidate_count"] == 0
    assert report["ai_case_count"] == 0
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
    assert source["status"] == "rejected"
    assert source["dataset"] == {
        "derivatives_feature_count": 32,
        "feature_count": 103,
        "features_bytes": 452_419_260,
        "features_dtype": "float32",
        "horizons_minutes": [15, 30, 60, 120],
        "persistent_feature_copy_created": False,
        "price_flow_feature_count": 71,
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
        4_662_194_176
    )
    assert source["viability_gate_passed_candidates"] == []
    assert source["ai_case_set"]["cases"] == 0
    assert source["ai_reports"] == []
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


def test_latest_action_value_tables_preserve_round38_graph_data() -> None:
    candidates = _csv("candidates.csv")
    thresholds = _csv("thresholds.csv")
    models = _csv("models.csv")
    uplift = _csv("derivatives-uplift.csv")
    sources = _csv("sources.csv")

    assert len(candidates) == 32
    assert len(thresholds) == 640
    assert len(models) == 96
    assert len(uplift) == 16
    assert len(sources) == 3
    assert {row["architecture"] for row in candidates} == {
        "shared_direct_multiclass_lightgbm",
        "per_symbol_direct_multiclass_lightgbm",
        "shared_two_stage_hurdle_lightgbm",
        "per_symbol_two_stage_hurdle_lightgbm",
    }
    assert {row["feature_set"] for row in candidates} == {
        "price_flow_only",
        "price_flow_plus_premium_and_funding",
    }
    assert {int(row["horizon_minutes"]) for row in candidates} == {15, 30, 60, 120}
    assert all(row["viability_gate_passed"] == "False" for row in candidates)
    assert sum(row["support_passed"] == "True" for row in thresholds) == 295
    assert sum(row["selected"] == "True" for row in thresholds) == 32
    assert all(row["backend_kind"] == "opencl" for row in models)
    assert all(float(row["reload_max_abs_prediction_error"]) == 0.0 for row in models)
    assert all(
        float(row["viability_log_loss_delta_augmented_minus_price"]) > 0
        for row in uplift
    )
    assert {row["symbol"] for row in sources} == {
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
    }
    assert all(int(row["price_rows"]) == 1_883_520 for row in sources)
    assert all(int(row["price_gap_count"]) == 0 for row in sources)

    best = max(candidates, key=lambda row: float(row["viability_mean_net_bps"]))
    assert best["candidate_id"] == (
        "price_flow_only_shared_two_stage_hurdle_lightgbm_h120"
    )
    assert int(best["viability_total_trades"]) == 789
    assert float(best["viability_mean_net_bps"]) == -3.9885070425551503
    assert float(best["viability_profit_factor"]) == 0.9246544495467506
    assert float(best["viability_day_block_bootstrap_mean_net_bps_lower_95"]) == (
        -13.44693667520006
    )
    assert min(int(row["viability_total_trades"]) for row in candidates) == 313
    assert max(int(row["viability_total_trades"]) for row in candidates) == 8_515


def test_latest_action_value_progress_extends_to_round38_without_portfolio_claim() -> None:
    progress = _csv("progress.csv")

    assert [int(row["round"]) for row in progress] == list(range(1, 39))
    latest = progress[-1]
    assert latest["status"] == "rejected"
    assert latest["selection_contaminated"] == "True"
    assert latest["risk_level"] == "research-only; no policy"
    assert latest["selected_signals"] == "789"
    assert latest["executable_trades"] == "789"
    assert float(latest["mean_net_bps"]) == -3.9885070425551503
    assert latest["accepted_thresholds"] == "0"


def test_latest_action_value_charts_are_accessible_and_prior_files_are_absent() -> None:
    expected_charts = {
        "calibration-to-viability.svg",
        "derivatives-feature-ablation.svg",
        "research-progress.svg",
        "viability-economics.svg",
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
        "signals.csv",
        "daily.csv",
        "regimes.csv",
        "ranked-event-outcomes.csv",
        "decay.csv",
        "horizon-support.csv",
        "calibration-economics.svg",
        "calibration-support.svg",
        "prediction-quality.svg",
    ):
        assert not (LATEST / stale).exists()
        assert not (LATEST / "charts" / stale).exists()

    readme = (LATEST / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Round 38: activity restored, economic edge rejected")
    assert "# Round 37:" not in readme
    assert "No ROI graph is published" in readme
    assert "not a capital-constrained portfolio, ROI series, or equity curve" in readme
