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


def test_latest_action_value_publication_is_round41_hash_verified() -> None:
    report = json.loads((LATEST / "report.json").read_text(encoding="utf-8"))
    canonical = dict(report)
    claimed = canonical.pop("publication_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert claimed == "244c2b40d10cc0758e6603b1c3fda9265fa804c957aadc0736da782e1e5c72e4"
    assert report["schema_version"] == "prequential-meta-label-publication-v1"
    assert report["round"] == 41
    assert report["status"] == "rejected"
    assert report["source_implementation_commit"] == (
        "ebf8d98263d67de20f52096adfaa0c2e8d1f2c50"
    )
    assert report["source_report_canonical_sha256"] == (
        "718d75fdebf278f359a16dea9cf3b8e6606e49a420b9f8dfd2e262e9173522d7"
    )
    assert report["source_report_file_sha256"] == (
        "980619c4916942314218e923c83ff06bbcdf46679f8f172fd6a7e2f157b25101"
    )
    assert report["dataset_rows"] == 1_098_105
    assert report["primary_feature_count"] == 71
    assert report["meta_feature_count"] == 81
    assert report["gpu_model_artifact_count"] == 48
    assert report["threshold_cell_count"] == 216
    assert report["monthly_result_count"] == 6
    assert report["selected_threshold_month_count"] == 2
    assert report["aggregate_trade_count"] == 466
    assert report["aggregate_mean_net_bps"] == -0.6773770308609685
    assert report["aggregate_total_net_bps"] == -315.6576963812113
    assert report["aggregate_profit_factor"] == 0.9803029043725825
    assert report["aggregate_day_block_lower_95_mean_net_bps"] == (-11.019425264983331)
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
    assert source["round"] == 41
    assert source["status"] == "rejected"
    assert source["implementation_commit"] == (
        "ebf8d98263d67de20f52096adfaa0c2e8d1f2c50"
    )
    assert source["dataset"]["rows"] == 1_098_105
    assert source["dataset"]["primary_feature_count"] == 71
    assert source["dataset"]["meta_feature_count"] == 81
    assert source["dataset"]["features_dtype"] == "float32"
    assert source["backend"]["kind"] == "opencl"
    assert source["backend"]["device"] == "opencl:auto"
    assert len(source["model_artifacts"]) == 48
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
        4_688_977_920
    )
    for field in (
        "trading_authority",
        "execution_claim",
        "profitability_claim",
        "portfolio_claim",
        "leverage_applied",
    ):
        assert source[field] is False


def test_latest_action_value_tables_preserve_round41_graph_data() -> None:
    candidate = _csv("candidate.csv")
    monthly = _csv("monthly.csv")
    thresholds = _csv("thresholds.csv")
    models = _csv("models.csv")
    sources = _csv("sources.csv")

    assert [
        len(candidate),
        len(monthly),
        len(thresholds),
        len(models),
        len(sources),
    ] == [
        1,
        6,
        216,
        48,
        3,
    ]
    assert candidate[0]["candidate_id"] == (
        "prequential_h30_shared_profitability_meta_6m"
    )
    assert candidate[0]["selected_threshold_months"] == "2"
    assert candidate[0]["aggregate_gate_passed"] == "False"
    assert sum(row["support_passed"] == "True" for row in thresholds) == 21
    assert sum(row["economic_gate_passed"] == "True" for row in thresholds) == 4
    assert sum(row["selected"] == "True" for row in thresholds) == 2
    assert {row["model_role"] for row in models} == {
        "prequential_primary",
        "meta_label",
    }
    assert sum(row["model_role"] == "prequential_primary" for row in models) == 42
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
    assert min(evaluation_auc) == 0.5725303340494036
    assert max(evaluation_auc) == 0.5981457892377167
    assert all(
        float(row["meta_fit_roc_auc"]) > float(row["meta_evaluation_roc_auc"])
        for row in monthly
    )
    assert all(int(row["evaluation_trades"]) == 0 for row in monthly[:4])
    november, december = monthly[-2:]
    assert november["threshold_selected"] == "True"
    assert float(november["selected_meta_probability"]) == 0.5
    assert float(november["selected_primary_margin"]) == 0.1
    assert int(november["evaluation_trades"]) == 235
    assert float(november["evaluation_mean_net_bps"]) == -0.7313756379675358
    assert december["threshold_selected"] == "True"
    assert float(december["selected_meta_probability"]) == 0.5
    assert float(december["selected_primary_margin"]) == 0.1
    assert int(december["evaluation_trades"]) == 231
    assert [
        int(december[key])
        for key in ("btcusdt_trades", "ethusdt_trades", "solusdt_trades")
    ] == [43, 86, 102]
    assert float(december["evaluation_mean_net_bps"]) == -0.6224433829387029
    assert float(december["evaluation_profit_factor"]) == 0.9826311840912332
    assert float(december["evaluation_lower_95_mean_net_bps"]) == (-15.5115764058873)


def test_latest_action_value_progress_extends_to_round41_without_policy_claim() -> None:
    progress = _csv("progress.csv")

    assert [int(row["round"]) for row in progress] == list(range(1, 42))
    latest = progress[-1]
    assert latest["status"] == "rejected"
    assert latest["selection_contaminated"] == "True"
    assert latest["risk_level"] == "consumed development only; no policy"
    assert latest["selected_signals"] == "466"
    assert latest["executable_trades"] == "466"
    assert float(latest["mean_net_bps"]) == -0.6773770308609685
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
        assert 'height="-' not in text

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
    assert readme.startswith("# Round 41: prequential meta-label screen rejected")
    assert "# Round 40:" not in readme
    assert "not profitability or ROI" in readme
    assert "Selection-confirmation 2025-H2 and terminal 2026 remain sealed" in readme
