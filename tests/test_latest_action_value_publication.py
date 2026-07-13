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


def test_latest_action_value_publication_is_round40_hash_verified() -> None:
    report = json.loads((LATEST / "report.json").read_text(encoding="utf-8"))
    canonical = dict(report)
    claimed = canonical.pop("publication_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert claimed == "c7090d138d2243e0d95d288693c6f5e7422e24744d489860e3a81f0071caeb29"
    assert report["schema_version"] == (
        "causal-meta-label-capacity-publication-v1"
    )
    assert report["round"] == 40
    assert report["status"] == "rejected"
    assert report["source_implementation_commit"] == (
        "b7e5852ed0b8fae3a02204474d7ae0831bc9bef5"
    )
    assert report["source_report_canonical_sha256"] == (
        "38e63e5718c811c970835cbf84bbe1c7b371d7ddac50627da9edeb1ac7f4e576"
    )
    assert report["source_report_file_sha256"] == (
        "93ae2a055fc8e1cccc2347b48918c7b513179f190217402bfada051f6f5ba67b"
    )
    assert report["dataset_rows"] == 1_098_105
    assert report["primary_feature_count"] == 71
    assert report["meta_feature_count"] == 81
    assert report["gpu_model_artifact_count"] == 24
    assert report["threshold_cell_count"] == 216
    assert report["monthly_result_count"] == 6
    assert report["selected_threshold_month_count"] == 1
    assert report["aggregate_trade_count"] == 70
    assert report["aggregate_mean_net_bps"] == 36.060967821734295
    assert report["aggregate_day_block_lower_95_mean_net_bps"] == (
        -4.307041800475944
    )
    assert report["ai_case_count"] == 0
    assert report["selection_contaminated"] is True
    assert report["development_only"] is True
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
    assert source["round"] == 40
    assert source["status"] == "rejected"
    assert source["implementation_commit"] == (
        "b7e5852ed0b8fae3a02204474d7ae0831bc9bef5"
    )
    assert source["dataset"]["rows"] == 1_098_105
    assert source["dataset"]["model_feature_count"] == 71
    assert source["dataset"]["meta_feature_count"] == 81
    assert source["dataset"]["features_dtype"] == "float32"
    assert source["backend"]["kind"] == "opencl"
    assert source["backend"]["device"] == "opencl:auto"
    assert len(source["model_artifacts"]) == 24
    assert all(
        artifact["reload_max_abs_prediction_error"] == 0.0
        for artifact in source["model_artifacts"]
    )
    assert source["aggregate_ml_gate_passed"] is False
    assert source["ai_case_set"]["cases"] == 0
    assert source["ai_report"] is None
    assert source["ai_error"] is None
    assert source["ai_uplift_gate_passed"] is False
    assert source["selection_contaminated"] is True
    assert source["development_only"] is True
    assert source["selection_confirmation_accessed"] is False
    assert source["terminal_2026_accessed"] is False
    assert source["runtime_evidence"]["memory"]["peak_working_set_bytes"] == (
        4_686_254_080
    )
    for field in (
        "trading_authority",
        "execution_claim",
        "profitability_claim",
        "portfolio_claim",
        "leverage_applied",
    ):
        assert source[field] is False


def test_latest_action_value_tables_preserve_round40_graph_data() -> None:
    candidate = _csv("candidate.csv")
    monthly = _csv("monthly.csv")
    thresholds = _csv("thresholds.csv")
    models = _csv("models.csv")
    sources = _csv("sources.csv")

    assert [len(candidate), len(monthly), len(thresholds), len(models), len(sources)] == [
        1,
        6,
        216,
        24,
        3,
    ]
    assert candidate[0]["candidate_id"] == (
        "causal_per_symbol_direct_h30_shared_profitability_meta"
    )
    assert candidate[0]["selected_threshold_months"] == "1"
    assert candidate[0]["aggregate_gate_passed"] == "False"
    assert sum(row["support_passed"] == "True" for row in thresholds) == 27
    assert sum(row["economic_gate_passed"] == "True" for row in thresholds) == 3
    assert sum(row["selected"] == "True" for row in thresholds) == 1
    assert {row["model_role"] for row in models} == {"primary", "meta_label"}
    assert sum(row["model_role"] == "primary" for row in models) == 18
    assert sum(row["model_role"] == "meta_label" for row in models) == 6
    assert all(row["backend_kind"] == "opencl" for row in models)
    assert all(float(row["reload_max_abs_prediction_error"]) == 0.0 for row in models)
    assert {row["symbol"] for row in sources} == {
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
    }
    assert all(int(row["price_rows"]) == 1_883_520 for row in sources)
    assert all(int(row["price_gap_count"]) == 0 for row in sources)

    assert [row["evaluation_month"] for row in monthly] == [
        "2024-07",
        "2024-08",
        "2024-09",
        "2024-10",
        "2024-11",
        "2024-12",
    ]
    evaluation_auc = [float(row["meta_evaluation_roc_auc"]) for row in monthly]
    assert min(evaluation_auc) == 0.5636332362683997
    assert max(evaluation_auc) == 0.5866530158317361
    assert all(
        float(row["meta_fit_roc_auc"]) > float(row["meta_evaluation_roc_auc"])
        for row in monthly
    )
    assert all(int(row["evaluation_trades"]) == 0 for row in monthly[:5])
    december = monthly[-1]
    assert december["threshold_selected"] == "True"
    assert float(december["selected_meta_probability"]) == 0.55
    assert int(december["evaluation_trades"]) == 70
    assert [
        int(december[key])
        for key in ("btcusdt_trades", "ethusdt_trades", "solusdt_trades")
    ] == [10, 29, 31]
    assert float(december["evaluation_mean_net_bps"]) == 36.060967821734295
    assert float(december["evaluation_profit_factor"]) == 2.008035067071759
    assert float(december["evaluation_lower_95_mean_net_bps"]) == (
        -2.8312723606804515
    )


def test_latest_action_value_progress_extends_to_round40_without_policy_claim() -> None:
    progress = _csv("progress.csv")

    assert [int(row["round"]) for row in progress] == list(range(1, 41))
    latest = progress[-1]
    assert latest["status"] == "rejected"
    assert latest["selection_contaminated"] == "True"
    assert latest["risk_level"] == "consumed development only; no policy"
    assert latest["selected_signals"] == "70"
    assert latest["executable_trades"] == "70"
    assert float(latest["mean_net_bps"]) == 36.060967821734295
    assert latest["accepted_thresholds"] == "0"


def test_latest_action_value_charts_are_accessible_and_prior_files_are_absent() -> None:
    expected_charts = {
        "calibration-economics.svg",
        "evaluation-activity.svg",
        "meta-label-auc.svg",
        "research-progress.svg",
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
        "ai-cases.csv",
        "ai-decisions.csv",
        "ai-models.csv",
        "capacity-summary.csv",
        "monthly-economics.svg",
        "rolling-candidate-economics.svg",
        "ai-ablation.svg",
    ):
        assert not (LATEST / stale).exists()
        assert not (LATEST / "charts" / stale).exists()

    readme = (LATEST / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Round 40: causal meta-label screen rejected")
    assert "# Round 39:" not in readme
    assert "not a profitability or ROI claim" in readme
    assert "Selection-confirmation 2025-H2 and terminal 2026 remain sealed" in readme
