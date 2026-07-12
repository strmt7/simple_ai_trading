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


def test_latest_action_value_publication_is_round37_hash_verified() -> None:
    report = json.loads((LATEST / "report.json").read_text(encoding="utf-8"))
    canonical = dict(report)
    claimed = canonical.pop("publication_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert claimed == "5c25c3374d61e1417c12b7653d1972d6028f1408a2b089b0c08b8056b6a0a9cf"
    assert report["schema_version"] == (
        "cross-asset-cost-aware-ai-ablation-publication-v1"
    )
    assert report["round"] == 37
    assert report["status"] == "rejected"
    assert report["source_implementation_commit"] == (
        "13379db93f661b7a201f580ef02244a6710e59b8"
    )
    assert report["source_report_canonical_sha256"] == (
        "ba1a642169b31751c505aa401b99de55c75fe19d78c1617ea95812d1e4c06bd6"
    )
    assert report["source_report_file_sha256"] == (
        "667296fa31972cd9e5ad32867c1ed02762ca532709a9d44213381aecb2e9ea81"
    )
    assert report["dataset_rows"] == 1_103_328
    assert report["feature_count"] == 71
    assert report["gpu_model_count"] == 16
    assert report["candidate_count"] == 20
    assert report["threshold_cell_count"] == 100
    assert report["selected_threshold_count"] == 0
    assert report["viability_trade_count"] == 0
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
        **source["dataset"],
        "feature_count": 71,
        "features_dtype": "float32",
        "persistent_feature_copy_created": False,
        "rows": 1_103_328,
    }
    assert source["backend"] == {
        "device": "opencl:auto",
        "gpu_first_requested": True,
        "kind": "opencl",
    }
    assert source["runtime_evidence"]["memory"]["peak_working_set_bytes"] == (
        3_557_158_912
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


def test_latest_action_value_tables_preserve_round37_graph_data() -> None:
    candidates = _csv("candidates.csv")
    thresholds = _csv("thresholds.csv")
    models = _csv("models.csv")
    sources = _csv("sources.csv")

    assert len(candidates) == 20
    assert len(thresholds) == 100
    assert len(models) == 16
    assert len(sources) == 3
    assert {row["family"] for row in candidates} == {
        "linear_ridge",
        "per_symbol_lightgbm",
        "persistence",
        "shared_cross_asset_lightgbm",
        "zero_return",
    }
    assert {int(row["horizon_minutes"]) for row in candidates} == {15, 30, 60, 120}
    assert all(row["selected_threshold_bps"] == "" for row in candidates)
    assert all(row["viability_gate_passed"] == "False" for row in candidates)
    assert all(row["support_passed"] == "False" for row in thresholds)
    assert all(row["backend_kind"] == "opencl" for row in models)
    assert all(float(row["reload_max_abs_prediction_error"]) == 0.0 for row in models)
    assert {row["symbol"] for row in sources} == {
        "BTCUSDT",
        "ETHUSDT",
        "SOLUSDT",
    }
    assert all(int(row["rows"]) == 1_883_520 for row in sources)
    assert all(int(row["gap_count"]) == 0 for row in sources)

    best = max(candidates, key=lambda row: float(row["viability_pearson_ic"]))
    assert (best["family"], best["horizon_minutes"]) == (
        "shared_cross_asset_lightgbm",
        "120",
    )
    assert float(best["viability_pearson_ic"]) == 0.017791430387791903
    assert float(best["viability_spearman_ic"]) == 0.0437035343890367

    largest = max(thresholds, key=lambda row: int(row["nonoverlapping_trades"]))
    assert (
        largest["family"],
        largest["horizon_minutes"],
        largest["threshold_bps"],
    ) == ("linear_ridge", "120", "12.0")
    assert (
        int(largest["btc_trades"]),
        int(largest["eth_trades"]),
        int(largest["sol_trades"]),
    ) == (0, 284, 379)


def test_latest_action_value_progress_extends_to_round37_without_equity_claim() -> (
    None
):
    progress = _csv("progress.csv")

    assert [int(row["round"]) for row in progress] == list(range(1, 38))
    latest = progress[-1]
    assert latest["status"] == "rejected"
    assert latest["selection_contaminated"] == "True"
    assert latest["risk_level"] == "research-only; no policy"
    assert latest["selected_signals"] == "0"
    assert latest["executable_trades"] == "0"
    assert latest["direction_auc"] == ""
    assert float(latest["spearman_ic"]) == 0.0437035343890367
    assert latest["accepted_thresholds"] == "0"


def test_latest_action_value_charts_are_accessible_and_round36_files_are_absent() -> (
    None
):
    expected_charts = {
        "calibration-economics.svg",
        "calibration-support.svg",
        "prediction-quality.svg",
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
        "signals.csv",
        "daily.csv",
        "regimes.csv",
        "ranked-event-outcomes.csv",
        "decay.csv",
        "horizon-support.csv",
    ):
        assert not (LATEST / stale).exists()

    readme = (LATEST / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Round 37: diversified candidate selection rejected")
    assert "# Round 36:" not in readme
    assert "No ROI graph exists" in readme
    assert "not selected trades, ROI, an equity curve, or profitability evidence" in readme
