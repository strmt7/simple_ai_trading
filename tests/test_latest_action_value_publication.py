from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from tools.publish_barrier_competing_risk_viability import _progress_rows


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


def _source() -> dict[str, object]:
    return json.loads((LATEST / "screen.json").read_text(encoding="utf-8"))


def test_latest_action_value_publication_is_round50_hash_verified() -> None:
    report = json.loads((LATEST / "report.json").read_text(encoding="utf-8"))
    canonical = dict(report)
    claimed = canonical.pop("publication_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert claimed == "cff6c65c267efbdb1c726b703b4142ed74b4712fc20bcc475beaa3008ef9f34e"
    assert report["schema_version"] == "path-bounded-competing-risk-tcn-publication-v1"
    assert report["round"] == 50
    assert report["status"] == "rejected"
    assert report["source_implementation_commit"] == (
        "793cdbdd37ce48ec7145c7039f92e2f15adb4e8d"
    )
    assert report["design_sha256"] == (
        "09abb13a55009b4995a3543550b375d39938ff9df7bbfdb7329a1b369570045a"
    )
    assert report["binding_sha256"] == (
        "4bf9248727caec9c848e73d9e39a0ffe1b98f67958bbdeadbcf5057804805d6e"
    )
    assert report["source_report_canonical_sha256"] == (
        "8629a07940c0d8b4b16b35be4d7b651c1625807f8abae82a9e7fa7bfe73b6850"
    )
    assert report["source_report_file_sha256"] == (
        "47385351b7faf6bf1feb19d84f1c6200c5b6d5552e735877ae95f4f2c62245e8"
    )
    assert report["dataset_sha256"] == (
        "31c7713339cff9ad12f3bae02475743d09b2248bfc1b85e02e1f3306a699e774"
    )
    assert report["predecessor_dataset_sha256"] == (
        "37d6c5b29bffd272afe03703d1ff2353f2d6939201be253cfe626e5e12f4b48b"
    )
    assert report["dataset_rows"] == 1_098_105
    assert report["evaluation_timestamps"] == 52_104
    assert report["source_resolution_seconds"] == 60
    assert report["decision_interval_seconds"] == 300
    assert report["directml_model_artifact_count"] == 6
    assert report["external_artifacts_hash_verified"] is True
    assert report["candidate_quality_gate_pass_count"] == 0
    assert report["candidate_economic_gate_pass_count"] == 0
    assert report["mechanism_gate_passed"] is False
    assert report["leverage_sensitivity_run"] is False
    assert report["ai_paired_uplift_run"] is False
    assert report["selection_contaminated"] is True
    assert report["development_only"] is True
    for field in (
        "trading_authority",
        "profitability_claim",
        "leverage_applied",
        "ai_uplift_claim",
    ):
        assert report[field] is False

    declared_paths = {item["path"] for item in report["artifact_integrity"]}
    for chart, sources in report["graph_sources"].items():
        assert chart in declared_paths
        assert all(source in declared_paths for source in sources)
    for artifact in report["artifact_integrity"]:
        path = LATEST / artifact["path"]
        assert path.is_file()
        assert path.stat().st_size == artifact["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
        if path.suffix == ".csv":
            assert len(_csv(artifact["path"])) == artifact["row_count"]


def test_latest_action_value_source_report_is_exact_and_fail_closed() -> None:
    source = _source()
    canonical = dict(source)
    claimed = canonical.pop("report_canonical_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert claimed == "8629a07940c0d8b4b16b35be4d7b651c1625807f8abae82a9e7fa7bfe73b6850"
    assert hashlib.sha256((LATEST / "screen.json").read_bytes()).hexdigest() == (
        "47385351b7faf6bf1feb19d84f1c6200c5b6d5552e735877ae95f4f2c62245e8"
    )
    assert source["round"] == 50
    assert source["implementation_commit"] == (
        "793cdbdd37ce48ec7145c7039f92e2f15adb4e8d"
    )
    dataset = source["dataset"]
    assert dataset["barrier_dataset_sha256"] == (
        "31c7713339cff9ad12f3bae02475743d09b2248bfc1b85e02e1f3306a699e774"
    )
    assert dataset["predecessor_dataset_sha256"] == (
        "37d6c5b29bffd272afe03703d1ff2353f2d6939201be253cfe626e5e12f4b48b"
    )
    assert dataset["rows"] == 1_098_105
    assert dataset["timestamps"] == 366_035
    assert dataset["symbols"] == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    assert dataset["source_resolution_seconds"] == 60
    assert dataset["decision_interval_seconds"] == 300
    assert dataset["synthetic_rows"] == 0
    assert dataset["selection_confirmation_or_terminal_rows_read"] is False
    assert source["backend"]["backend_kind"] == "directml"
    assert source["backend"]["backend_device"] == "privateuseone:0"
    assert source["backend"]["cpu_fallback_warnings"] == 0
    assert source["backend"]["warning_count"] == 0
    assert source["runtime"]["elapsed_seconds"] == 382.08238949999213
    assert source["runtime"]["memory"]["peak_working_set_bytes"] == 7_222_837_248

    assert set(source["diagnostics"]) == {
        "direct_barrier_mean_tcn",
        "competing_risk_barrier_tcn",
    }
    for candidate in source["diagnostics"]:
        assert source["diagnostics"][candidate]["quality_gate"]["passed"] is False
        assert source["economic_gates"][candidate]["passed"] is False
        assert source["leverage_sensitivity"][candidate]["run"] is False
        assert len(source["artifacts"][candidate]) == 3
        assert {item["seed"] for item in source["artifacts"][candidate]} == {
            5001,
            5002,
            5003,
        }
        assert all(
            float(value) == 0.0
            for item in source["artifacts"][candidate]
            for key, value in item.items()
            if key.startswith("reload_max_abs_")
        )
    assert source["mechanism_gate"]["passed"] is False
    assert source["mechanism_gate"]["average_expected_payoff_spearman_improvement"] == (
        -0.01701770760601806
    )
    assert source["ai"]["paired_uplift_run"] is False
    assert source["ai"]["risk_reviewer"]["selected_risk_reviewer"] == "qwen3:8b"
    assert (
        source["ai"]["risk_reviewer"]["financial_edge_tested_by_safety_benchmark"]
        is False
    )
    for field in ("trading_authority", "profitability_claim"):
        assert source["claims"][field] is False


def test_latest_action_value_tables_reconcile_to_round50_report() -> None:
    expected_counts = {
        "forecast.csv": 4,
        "monthly-forecast.csv": 24,
        "symbol-forecast.csv": 12,
        "seed-stability.csv": 12,
        "training.csv": 29,
        "models.csv": 6,
        "scenarios.csv": 4,
        "trades.csv": 700,
        "daily-equity.csv": 724,
        "monthly-performance.csv": 24,
        "symbols.csv": 12,
        "gates.csv": 2,
        "mechanism.csv": 1,
        "target-baselines.csv": 12,
        "roles.csv": 4,
        "sources.csv": 10,
    }
    for path, expected_count in expected_counts.items():
        assert len(_csv(path)) == expected_count

    forecast = _csv("forecast.csv")
    assert all(float(row["event_log_loss_skill"]) > 0.028 for row in forecast)
    assert all(float(row["event_group_brier_skill"]) > 0.052 for row in forecast)
    assert all(float(row["maximum_event_group_ece"]) < 0.022 for row in forecast)
    ranks = {
        (row["candidate_id"], row["side"]): float(row["expected_payoff_spearman"])
        for row in forecast
    }
    assert ranks[("competing_risk_barrier_tcn", "short")] == -0.060433327853639356
    assert ranks[("competing_risk_barrier_tcn", "long")] == 0.048081355987611304

    models = _csv("models.csv")
    assert {int(row["seed"]) for row in models} == {5001, 5002, 5003}
    assert {row["backend_kind"] for row in models} == {"directml"}
    assert all(
        float(value) == 0.0
        for row in models
        for key, value in row.items()
        if key.startswith("reload_max_abs_")
    )

    scenarios = _csv("scenarios.csv")
    daily = _csv("daily-equity.csv")
    monthly = _csv("monthly-performance.csv")
    symbols = _csv("symbols.csv")
    trades = _csv("trades.csv")
    for scenario in scenarios:
        candidate = scenario["candidate_id"]
        name = scenario["scenario"]
        expected_return = float(scenario["total_return_fraction"])
        daily_sum = sum(
            float(row["return_fraction"])
            for row in daily
            if row["candidate_id"] == candidate and row["scenario"] == name
        )
        monthly_rows = [
            row
            for row in monthly
            if row["candidate_id"] == candidate and row["scenario"] == name
        ]
        symbol_rows = [
            row
            for row in symbols
            if row["candidate_id"] == candidate and row["scenario"] == name
        ]
        assert math.isclose(daily_sum, expected_return, rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(
            sum(float(row["return_fraction"]) for row in monthly_rows),
            expected_return,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        assert math.isclose(
            sum(float(row["total_return_fraction"]) for row in symbol_rows),
            expected_return,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        expected_trades = int(scenario["closed_trades"])
        assert sum(int(row["closed_trades"]) for row in monthly_rows) == expected_trades
        assert (
            sum(row["candidate_id"] == candidate for row in trades) == expected_trades
        )

    path_base = next(
        row
        for row in scenarios
        if row["candidate_id"] == "competing_risk_barrier_tcn"
        and row["scenario"] == "base"
    )
    assert int(path_base["closed_trades"]) == 610
    assert float(path_base["total_return_fraction"]) == -0.35726490193704763
    assert float(path_base["maximum_drawdown_fraction"]) == 0.37412205395423015
    assert float(path_base["profit_factor"]) == 0.6839400438906108
    path_trades = [
        row for row in trades if row["candidate_id"] == "competing_risk_barrier_tcn"
    ]
    assert sum(row["side_name"] == "short" for row in path_trades) == 525
    assert sum(row["side_name"] == "long" for row in path_trades) == 85


def test_round50_failure_analysis_and_progress_are_truthful() -> None:
    failure_path = RESEARCH / "round-050-failure-analysis.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    canonical = dict(failure)
    claimed = canonical.pop("analysis_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert claimed == "e5b491e59f91c55118797a705b10ee24db3830d61a809ef7730f00009e665800"
    assert failure["round"] == 50
    assert failure["status"] == "rejected"
    assert failure["profitability_claim"] is False
    assert failure["trading_authority"] is False
    assert failure["leverage_applied"] is False
    assert failure["ai_uplift_claim"] is False
    assert failure["observed_result"]["direct_control"]["trades"] == 90
    assert failure["observed_result"]["competing_risk"]["trades"] == 610
    assert len(failure["next_model_requirements"]) == 6

    progress = _csv("progress.csv")
    assert [int(row["round"]) for row in progress] == list(range(1, 51))
    latest = progress[-1]
    assert latest["status"] == "rejected"
    assert latest["selection_contaminated"] == "True"
    assert latest["development_consumed"] == "True"
    assert latest["risk_level"] == "consumed development only; unlevered fixed sleeves"
    assert latest["selected_signals"] == "610"
    assert latest["executable_trades"] == "610"
    assert float(latest["mean_net_bps"]) == -17.570405013297425
    assert latest["architecture_gates_passed"] == "0"
    assert latest["architecture_gate_count"] == "3"

    fields, rebuilt = _progress_rows(LATEST / "progress.csv", _source())
    assert fields == list(progress[0])
    assert [int(row["round"]) for row in rebuilt] == list(range(1, 51))
    assert sum(int(row["round"]) == 50 for row in rebuilt) == 1


def test_latest_action_value_charts_are_accessible_and_stale_files_are_absent() -> None:
    expected_charts = {
        "daily-equity-drawdown.svg",
        "event-quality.svg",
        "expected-payoff-quality.svg",
        "monthly-performance.svg",
        "policy-economics.svg",
        "research-progress.svg",
        "seed-stability.svg",
        "symbol-performance.svg",
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
        "expected-net.csv",
        "monthly.csv",
        "probability.csv",
        "replays.csv",
        "severity.csv",
        "target-geometry.csv",
        "action-value-quality.svg",
        "daily-equity.svg",
        "forecast-quality.svg",
        "monthly-economics.svg",
        "severity-quality.svg",
    ):
        assert not (LATEST / stale).exists()
        assert not (LATEST / "charts" / stale).exists()

    readme = (LATEST / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Round 50: Path-Bounded Competing-Risk TCN")
    assert "lost `35.73%`" in readme
    assert "This is not a multi-year second-level dataset claim" in readme
    assert "AI uplift was not run" in readme
    assert "approved for testnet" in readme
    assert "# Round 49:" not in readme
