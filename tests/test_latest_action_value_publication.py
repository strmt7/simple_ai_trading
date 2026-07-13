from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from tools.publish_minute_logistic_mixture_tcn_viability import _progress_rows


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "model-research" / "action-value"
LATEST = RESEARCH / "latest"


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


def test_latest_action_value_publication_is_round48_hash_verified() -> None:
    report = json.loads((LATEST / "report.json").read_text(encoding="utf-8"))
    canonical = dict(report)
    claimed = canonical.pop("publication_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert claimed == "50f21514710c71c776ac01e9a6dc1d5d2c3d6db17a39fad5876bf2a2c201693a"
    assert report["schema_version"] == "minute-logistic-mixture-tcn-publication-v1"
    assert report["round"] == 48
    assert report["status"] == "quality_or_economic_gate_rejected"
    assert report["source_implementation_commit"] == (
        "01f6dc21bc0eb1beab918d1dcc133a4781bcbf30"
    )
    assert report["design_sha256"] == (
        "69fe0b4e319d51680c50cc132d9f64d9da4fa01eccaf30f84e0aa51292f61aa3"
    )
    assert report["binding_sha256"] == (
        "504bd325fa27d8d6d444843c54e0695618984ce52540969fc856687b31d159a3"
    )
    assert report["source_report_canonical_sha256"] == (
        "95638932838bfe99dc751ccf18e1909ad601a2882be99c7de913af5d7db5d274"
    )
    assert report["source_report_file_sha256"] == (
        "f438ce711c3b88544c11f0f0b3aee92baa1ad87564023b2e8303ea3d84e2bc70"
    )
    assert report["dataset_sha256"] == (
        "6969a3134049a326024939d5f9c46a99c37a4932e4a1f146a542a77427bba92b"
    )
    assert report["dataset_rows"] == 1_098_105
    assert report["evaluation_timestamps"] == 52_104
    assert report["directml_model_artifact_count"] == 6
    assert report["candidate_distribution_gate_pass_count"] == 0
    assert report["candidate_action_gate_pass_count"] == 2
    assert report["candidate_economic_gate_pass_count"] == 0
    assert report["mixture_ablation_passed"] is True
    assert report["selection_contaminated"] is True
    assert report["development_only"] is True
    for field in (
        "trading_authority",
        "profitability_claim",
        "leverage_applied",
        "ai_uplift_claim",
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
    assert claimed == "95638932838bfe99dc751ccf18e1909ad601a2882be99c7de913af5d7db5d274"
    assert source["round"] == 48
    assert source["status"] == "quality_or_economic_gate_rejected"
    assert source["implementation_commit"] == (
        "01f6dc21bc0eb1beab918d1dcc133a4781bcbf30"
    )
    assert source["dataset"]["dataset_sha256"] == (
        "6969a3134049a326024939d5f9c46a99c37a4932e4a1f146a542a77427bba92b"
    )
    assert source["dataset"]["rows"] == 1_098_105
    assert source["dataset"]["timestamps"] == 366_035
    assert source["dataset"]["feature_count"] == 71
    assert source["dataset"]["symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert source["dataset"]["horizons_minutes"] == [15, 30, 60, 120]
    assert source["dataset"]["persistent_feature_copy_created"] is False
    assert source["backend"]["backend_kind"] == "directml"
    assert source["backend"]["backend_device"] == "privateuseone:0"
    assert source["backend"]["cpu_fallback_warnings"] == 0
    assert source["backend"]["warning_count"] == 0
    assert len(source["external_artifacts"]) == 12
    assert source["runtime_evidence"]["elapsed_seconds"] == 434.03959509998094
    assert source["runtime_evidence"]["memory"]["peak_working_set_bytes"] == (
        4_824_621_056
    )

    candidates = {item["candidate_id"]: item for item in source["candidate_results"]}
    assert set(candidates) == {
        "single_logistic_tcn",
        "state_mixture_logistic_tcn",
    }
    for candidate in candidates.values():
        assert candidate["diagnostics"]["distribution_gate"]["passed"] is False
        assert candidate["diagnostics"]["action_gate"]["passed"] is True
        assert candidate["combined_quality_gate_passed"] is False
        assert candidate["economic_gate"]["passed"] is False
        assert candidate["economic_gate"]["promotion_permitted"] is False
        assert candidate["base"]["metrics"]["total_net_return_fraction"] < 0.0
        assert candidate["base"]["metrics"]["maximum_drawdown_fraction"] > 0.4
        assert candidate["base"]["metrics"]["profit_factor"] < 1.0
        assert candidate["base"]["metrics"]["trades_by_horizon"]["15"] == 0
        assert candidate["stress"]["metrics"]["total_net_return_fraction"] < 0.0
        assert (
            candidate["stress"]["metrics"]["bootstrap_mean_five_minute_portfolio_bps"][
                "upper_bps"
            ]
            < 0.0
        )
        assert len(candidate["artifacts"]) == 3
        assert all(
            artifact[field] == 0.0
            for artifact in candidate["artifacts"]
            for field in (
                "reload_max_abs_location_error",
                "reload_max_abs_scale_error",
                "reload_max_abs_weight_error",
            )
        )

    control = candidates["single_logistic_tcn"]["base"]["metrics"]
    mixture = candidates["state_mixture_logistic_tcn"]["base"]["metrics"]
    assert control["trades"] == 1_808
    assert control["total_net_return_fraction"] == -0.5403046439615072
    assert control["trades_by_horizon"]["120"] == 1_769
    assert mixture["trades"] == 795
    assert mixture["total_net_return_fraction"] == -0.4134917382840835
    assert mixture["trades_by_horizon"]["120"] == 793
    assert source["mixture_ablation_gate"]["passed"] is True
    assert (
        source["mixture_ablation_gate"]["relative_negative_log_likelihood_improvement"]
        == 0.007380897488566296
    )
    assert source["ai_decision"]["executed"] is False
    assert source["ai_decision"]["paired_veto_only_ablation_eligible"] is False
    assert source["selection_confirmation_accessed"] is False
    assert source["terminal_2026_accessed"] is False
    for field in (
        "trading_authority",
        "profitability_claim",
        "promotion_permitted",
        "leverage_applied",
        "ai_uplift_claim",
    ):
        assert source[field] is False


def test_latest_action_value_tables_preserve_round48_graph_data() -> None:
    expected_counts = {
        "horizons.csv": 8,
        "symbol-horizons.csv": 24,
        "action-horizons.csv": 16,
        "seed-stability.csv": 24,
        "monthly-forecast.csv": 48,
        "pit-histogram.csv": 160,
        "routing.csv": 144,
        "prediction-summary.csv": 2,
        "training.csv": 39,
        "models.csv": 6,
        "roles.csv": 4,
        "trades.csv": 2_603,
        "replays.csv": 5_206,
        "monthly.csv": 24,
        "symbols.csv": 12,
        "daily-equity.csv": 724,
        "sources.csv": 3,
    }
    for path, expected_count in expected_counts.items():
        assert len(_csv(path)) == expected_count

    horizons = _csv("horizons.csv")
    assert all(float(row["negative_log_likelihood_skill"]) > 0.10 for row in horizons)
    assert all(float(row["distribution_mean_mse_skill"]) < 0.0 for row in horizons)

    actions_15m = [
        row for row in _csv("action-horizons.csv") if row["horizon_minutes"] == "15"
    ]
    assert len(actions_15m) == 4
    assert all(float(row["roc_auc"]) > 0.617 for row in actions_15m)
    assert all(float(row["brier_skill"]) > 0.039 for row in actions_15m)

    models = _csv("models.csv")
    assert {int(row["seed"]) for row in models} == {4801, 4802, 4803}
    assert {row["backend_kind"] for row in models} == {"directml"}
    assert all(
        float(row[field]) == 0.0
        for row in models
        for field in (
            "reload_max_abs_location_error",
            "reload_max_abs_scale_error",
            "reload_max_abs_weight_error",
        )
    )

    prediction_summary = _csv("prediction-summary.csv")
    assert all(int(row["evaluation_rows"]) == 156_312 for row in prediction_summary)
    assert all(int(row["nonfinite_values"]) == 0 for row in prediction_summary)
    assert all(int(row["quantile_crossing_count"]) == 0 for row in prediction_summary)

    trades = _csv("trades.csv")
    assert sum(row["candidate_id"] == "single_logistic_tcn" for row in trades) == 1_808
    assert (
        sum(row["candidate_id"] == "state_mixture_logistic_tcn" for row in trades)
        == 795
    )
    assert sum(row["horizon_minutes"] == "15" for row in trades) == 0
    assert sum(row["horizon_minutes"] == "120" for row in trades) == 2_562

    sources = _csv("sources.csv")
    assert {row["symbol"] for row in sources} == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    assert all(int(row["rows"]) == 1_883_520 for row in sources)
    assert all(int(row["gap_count"]) == 0 for row in sources)
    assert all(int(row["duplicate_or_regressed_time_count"]) == 0 for row in sources)
    assert all(int(row["invalid_ohlc_rows"]) == 0 for row in sources)

    roles = _csv("roles.csv")
    assert [row["role"] for row in roles] == [
        "training",
        "early_stop",
        "calibration",
        "evaluation",
    ]
    assert [int(row["timestamps"]) for row in roles] == [
        260_918,
        26_469,
        26_472,
        52_104,
    ]


def test_round48_failure_analysis_and_progress_are_truthful() -> None:
    failure_path = RESEARCH / "round-048-failure-analysis.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    canonical = dict(failure)
    claimed = canonical.pop("analysis_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert claimed == "5004312fd7339ae2c569cae10d6e6ac3356417b61f6ffe56630013a1e229f755"
    assert failure["round"] == 48
    assert failure["status"] == "rejected"
    assert failure["profitability_claim"] is False
    assert failure["trading_authority"] is False
    assert failure["leverage_applied"] is False
    assert failure["ai_uplift_claim"] is False
    assert (
        failure["candidate_results"]["single_logistic_tcn"]["fifteen_minute_trades"]
        == 0
    )
    assert (
        failure["candidate_results"]["state_mixture_logistic_tcn"][
            "fifteen_minute_trades"
        ]
        == 0
    )
    assert len(failure["next_model_requirements"]) == 6

    progress = _csv("progress.csv")
    assert [int(row["round"]) for row in progress] == list(range(1, 49))
    latest = progress[-1]
    assert latest["status"] == "rejected"
    assert latest["selection_contaminated"] == "True"
    assert latest["development_consumed"] == "True"
    assert latest["risk_level"] == "consumed development only; unlevered fixed sleeves"
    assert latest["selected_signals"] == "795"
    assert latest["executable_trades"] == "795"
    assert float(latest["mean_net_bps"]) == -0.10057689085167294
    assert latest["architecture_gates_passed"] == "0"
    assert latest["architecture_gate_count"] == "2"

    source = json.loads((LATEST / "screen.json").read_text(encoding="utf-8"))
    fields, rebuilt = _progress_rows(LATEST / "progress.csv", source)
    assert fields == list(progress[0])
    assert [int(row["round"]) for row in rebuilt] == list(range(1, 49))
    assert sum(int(row["round"]) == 48 for row in rebuilt) == 1


def test_latest_action_value_charts_are_accessible_and_stale_files_are_absent() -> None:
    expected_charts = {
        "action-quality.svg",
        "daily-equity.svg",
        "forecast-quality.svg",
        "horizon-allocation.svg",
        "monthly-economics.svg",
        "policy-economics.svg",
        "research-progress.svg",
        "seed-stability.svg",
        "training-dynamics.svg",
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
        "forecast-diagnostics.csv",
        "labels.csv",
        "utility-horizons.csv",
        "utility-quality.svg",
        "calibration-economics.svg",
        "evaluation-activity.svg",
        "meta-label-auc.svg",
    ):
        assert not (LATEST / stale).exists()
        assert not (LATEST / "charts" / stale).exists()

    readme = (LATEST / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Round 48: Minute Logistic-Mixture TCN")
    assert "both policies lost money after fixed costs and were rejected" in readme
    assert "AI was correctly withheld" in readme
    assert "approved for testnet" in readme
    assert "# Round 47:" not in readme
