from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from tools.publish_action_hurdle_tcn_viability import _progress_rows


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


def test_latest_action_value_publication_is_round49_hash_verified() -> None:
    report = json.loads((LATEST / "report.json").read_text(encoding="utf-8"))
    canonical = dict(report)
    claimed = canonical.pop("publication_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert claimed == "af5e16c48f6a23d437a61431a83850dbdb7f8aa75b3785a2d30289863fc10683"
    assert report["schema_version"] == "cost-aware-action-hurdle-tcn-publication-v1"
    assert report["round"] == 49
    assert report["status"] == "quality_or_economic_gate_rejected"
    assert report["source_implementation_commit"] == (
        "f1f55a8db6a9951b33bde5132a68d27eb0da7957"
    )
    assert report["design_sha256"] == (
        "72f114a2ad553c3401f03f1a5d6018566af724c4214408ecc07e1a5e1ae48026"
    )
    assert report["binding_sha256"] == (
        "8977c2788fa38f3d086e26e8b8818236a4ef81da9e92d5901b46d887a94f772b"
    )
    assert report["source_report_canonical_sha256"] == (
        "d07ce85ad0b63e292369d59d5a0c93610c34df17dd73be066d94fe6254a09417"
    )
    assert report["source_report_file_sha256"] == (
        "11f0a61a8bca1fcb5940df5f883c25bd970557a6dd2e875d6565aabbb9dfd9a2"
    )
    assert report["dataset_sha256"] == (
        "37d6c5b29bffd272afe03703d1ff2353f2d6939201be253cfe626e5e12f4b48b"
    )
    assert report["predecessor_dataset_sha256"] == (
        "6969a3134049a326024939d5f9c46a99c37a4932e4a1f146a542a77427bba92b"
    )
    assert report["dataset_rows"] == 1_098_105
    assert report["evaluation_timestamps"] == 52_104
    assert report["directml_model_artifact_count"] == 6
    assert report["candidate_numerical_gate_pass_count"] == 2
    assert report["candidate_action_gate_pass_count"] == 0
    assert report["candidate_economic_gate_pass_count"] == 0
    assert report["mechanism_ablation_passed"] is False
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
    assert claimed == "d07ce85ad0b63e292369d59d5a0c93610c34df17dd73be066d94fe6254a09417"
    assert source["round"] == 49
    assert source["status"] == "quality_or_economic_gate_rejected"
    assert source["implementation_commit"] == (
        "f1f55a8db6a9951b33bde5132a68d27eb0da7957"
    )
    dataset = source["dataset"]
    assert dataset["dataset_sha256"] == (
        "37d6c5b29bffd272afe03703d1ff2353f2d6939201be253cfe626e5e12f4b48b"
    )
    assert dataset["predecessor_dataset_sha256"] == (
        "6969a3134049a326024939d5f9c46a99c37a4932e4a1f146a542a77427bba92b"
    )
    assert dataset["rows"] == 1_098_105
    assert dataset["timestamps"] == 366_035
    assert dataset["feature_count"] == 71
    assert dataset["symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert dataset["horizons_minutes"] == [15, 30]
    assert dataset["persistent_feature_copy_created"] is False
    assert (
        max(
            float(row["identity_absolute_error_bps"])
            for row in dataset["target_geometry"]
        )
        < 1e-10
    )
    assert source["backend"]["backend_kind"] == "directml"
    assert source["backend"]["backend_device"] == "privateuseone:0"
    assert source["backend"]["cpu_fallback_warnings"] == 0
    assert source["backend"]["warning_count"] == 0
    assert len(source["external_artifacts"]) == 12
    assert source["runtime_evidence"]["elapsed_seconds"] == 254.20625259994995
    assert source["runtime_evidence"]["memory"]["peak_working_set_bytes"] == (
        4_824_891_392
    )

    candidates = {item["candidate_id"]: item for item in source["candidate_results"]}
    assert set(candidates) == {
        "direct_action_mean_tcn",
        "hurdle_action_value_tcn",
    }
    for candidate in candidates.values():
        assert candidate["numerical_quality_gate"]["passed"] is True
        assert candidate["diagnostics"]["action_quality_gate"]["passed"] is False
        assert candidate["combined_quality_gate_passed"] is False
        assert candidate["economic_gate"]["passed"] is False
        assert candidate["economic_gate"]["promotion_permitted"] is False
        assert len(candidate["artifacts"]) == 3
        assert all(
            artifact[field] == 0.0
            for artifact in candidate["artifacts"]
            for field in (
                "reload_max_abs_logit_error",
                "reload_max_abs_primary_error",
                "reload_max_abs_secondary_error",
                "reload_max_abs_auxiliary_error",
            )
        )

    direct = candidates["direct_action_mean_tcn"]["base"]["metrics"]
    hurdle = candidates["hurdle_action_value_tcn"]
    assert direct["trades"] == 0
    assert direct["total_net_return_fraction"] == 0.0
    assert hurdle["base"]["metrics"]["trades"] == 165
    assert hurdle["base"]["metrics"]["active_days"] == 35
    assert hurdle["base"]["metrics"]["total_net_return_fraction"] == (
        0.025275283383165315
    )
    assert hurdle["base"]["metrics"]["maximum_drawdown_fraction"] == (
        0.062278649992049684
    )
    assert hurdle["base"]["metrics"]["profit_factor"] == 1.0925769230906202
    assert hurdle["stress"]["metrics"]["total_net_return_fraction"] == (
        0.0029677582378793144
    )
    assert hurdle["stress"]["metrics"]["profit_factor"] == 1.017619475324064
    assert (
        hurdle["stress"]["metrics"]["bootstrap_mean_five_minute_portfolio_bps"][
            "lower_bps"
        ]
        == -0.03456874488678417
    )
    assert hurdle["base"]["metrics"]["trades_by_symbol"] == {
        "BTCUSDT": 5,
        "ETHUSDT": 55,
        "SOLUSDT": 105,
    }
    assert source["mechanism_ablation_gate"]["passed"] is False
    assert (
        source["mechanism_ablation_gate"]["average_expected_net_spearman_improvement"]
        == -0.002939776830973172
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


def test_latest_action_value_tables_preserve_round49_graph_data() -> None:
    expected_counts = {
        "probability.csv": 40,
        "expected-net.csv": 40,
        "severity.csv": 20,
        "seed-stability.csv": 12,
        "training.csv": 22,
        "models.csv": 6,
        "roles.csv": 4,
        "target-geometry.csv": 48,
        "trades.csv": 165,
        "replays.csv": 330,
        "monthly.csv": 24,
        "symbols.csv": 12,
        "daily-equity.csv": 724,
        "gates.csv": 2,
        "mechanism.csv": 1,
        "sources.csv": 3,
    }
    for path, expected_count in expected_counts.items():
        assert len(_csv(path)) == expected_count

    pooled_probability = [
        row for row in _csv("probability.csv") if row["scope"] == "pooled"
    ]
    assert len(pooled_probability) == 4
    assert all(float(row["roc_auc"]) > 0.618 for row in pooled_probability)
    assert all(float(row["log_loss_skill"]) > 0.035 for row in pooled_probability)

    pooled_action = [
        row for row in _csv("expected-net.csv") if row["scope"] == "pooled"
    ]
    assert len(pooled_action) == 4
    assert all(float(row["expected_net_spearman"]) < 0.01 for row in pooled_action)
    assert all(float(row["expected_net_mse_skill"]) < 0.0 for row in pooled_action)

    pooled_severity = [row for row in _csv("severity.csv") if row["scope"] == "pooled"]
    assert len(pooled_severity) == 2
    assert all(
        float(row["conditional_gain_gamma_score_skill"]) > 0.032
        for row in pooled_severity
    )
    assert all(
        float(row["conditional_loss_gamma_score_skill"]) > 0.015
        for row in pooled_severity
    )

    models = _csv("models.csv")
    assert {int(row["seed"]) for row in models} == {4901, 4902, 4903}
    assert {row["backend_kind"] for row in models} == {"directml"}
    assert all(
        float(row[field]) == 0.0
        for row in models
        for field in (
            "reload_max_abs_logit_error",
            "reload_max_abs_primary_error",
            "reload_max_abs_secondary_error",
            "reload_max_abs_auxiliary_error",
        )
    )

    trades = _csv("trades.csv")
    assert {row["candidate_id"] for row in trades} == {"hurdle_action_value_tcn"}
    assert {row["horizon_minutes"] for row in trades} == {"15"}
    assert sum(row["symbol"] == "BTCUSDT" for row in trades) == 5
    assert sum(row["symbol"] == "ETHUSDT" for row in trades) == 55
    assert sum(row["symbol"] == "SOLUSDT" for row in trades) == 105

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
        260_936,
        26_487,
        26_490,
        52_104,
    ]


def test_round49_failure_analysis_and_progress_are_truthful() -> None:
    failure_path = RESEARCH / "round-049-failure-analysis.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    canonical = dict(failure)
    claimed = canonical.pop("analysis_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert claimed == "ec98b3c232b3103328e4cf3172e2cce140afaf9802fb1567824f2f88189b2517"
    assert failure["round"] == 49
    assert failure["status"] == "rejected"
    assert failure["profitability_claim"] is False
    assert failure["trading_authority"] is False
    assert failure["leverage_applied"] is False
    assert failure["ai_uplift_claim"] is False
    assert failure["observed_result"]["direct_control"]["trades"] == 0
    assert failure["observed_result"]["hurdle"]["trades"] == 165
    assert len(failure["next_model_requirements"]) == 6

    progress = _csv("progress.csv")
    assert [int(row["round"]) for row in progress] == list(range(1, 50))
    latest = progress[-1]
    assert latest["status"] == "rejected"
    assert latest["selection_contaminated"] == "True"
    assert latest["development_consumed"] == "True"
    assert latest["risk_level"] == "consumed development only; unlevered fixed sleeves"
    assert latest["selected_signals"] == "165"
    assert latest["executable_trades"] == "165"
    assert float(latest["mean_net_bps"]) == 0.005260859640552832
    assert latest["architecture_gates_passed"] == "0"
    assert latest["architecture_gate_count"] == "3"

    source = json.loads((LATEST / "screen.json").read_text(encoding="utf-8"))
    fields, rebuilt = _progress_rows(LATEST / "progress.csv", source)
    assert fields == list(progress[0])
    assert [int(row["round"]) for row in rebuilt] == list(range(1, 50))
    assert sum(int(row["round"]) == 49 for row in rebuilt) == 1


def test_latest_action_value_charts_are_accessible_and_stale_files_are_absent() -> None:
    expected_charts = {
        "action-value-quality.svg",
        "daily-equity.svg",
        "forecast-quality.svg",
        "monthly-economics.svg",
        "policy-economics.svg",
        "research-progress.svg",
        "seed-stability.svg",
        "severity-quality.svg",
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
        "action-horizons.csv",
        "horizons.csv",
        "monthly-forecast.csv",
        "pit-histogram.csv",
        "prediction-summary.csv",
        "routing.csv",
        "symbol-horizons.csv",
        "action-quality.svg",
        "horizon-allocation.svg",
    ):
        assert not (LATEST / stale).exists()
        assert not (LATEST / "charts" / stale).exists()

    readme = (LATEST / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Round 49: Cost-Aware Action-Hurdle TCN")
    assert "positive point estimate" in readme
    assert "This is not a profitability claim" in readme
    assert "AI was withheld" in readme
    assert "approved for testnet" in readme
    assert "# Round 48:" not in readme
