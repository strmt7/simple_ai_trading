from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "model-research" / "action-value"
LATEST = RESEARCH / "latest"
PUBLISHER = ROOT / "tools" / "publish_round52_executable_support_hurdle.py"
CANDIDATES = {
    "executable_direct_mean_lightgbm",
    "executable_hurdle_lightgbm",
    "executable_hurdle_lightgbm_fincast",
}


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


def test_latest_action_value_publication_is_round52_hash_verified() -> None:
    publication = json.loads((LATEST / "report.json").read_text(encoding="utf-8"))
    canonical = dict(publication)
    claimed = canonical.pop("publication_canonical_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert claimed == "3f1feb381ff2e2ee6ba711727cf10784c1f7958e7bd8ecd131a0b962e1504cce"
    assert publication["schema_version"] == "round-052-action-value-publication-v1"
    assert publication["round"] == 52
    assert publication["publisher_path"] == PUBLISHER.relative_to(ROOT).as_posix()
    assert PUBLISHER.is_file()
    assert publication["source"] == {
        "binding_path": (
            "docs/model-research/action-value/round-052-execution-binding.json"
        ),
        "binding_sha256": (
            "e14f6e4b742e0da6d838621355a16fbc030ae2a941b6eef0ee2dd1ab9344568a"
        ),
        "design_path": (
            "docs/model-research/action-value/"
            "round-052-executable-support-hurdle-fincast-design.json"
        ),
        "design_sha256": (
            "af95d80a3adc21b72d6809d43afb3f2446213fe0a4e089b10366691465a0c669"
        ),
        "implementation_commit": "3cb9500fb1c34eeab77c2b294ed8c1fd282e2247",
        "report_canonical_sha256": (
            "ace44ebc33dc0601306841b4c353b43a184b2aa604b49f73d2301257f86d2f7f"
        ),
        "report_file_sha256": (
            "c5b728161535372d934ff9087a24b81c2490246cd17cb56beb3e29a3052d73fa"
        ),
        "report_path": (
            "E:\\SimpleAITradingData\\round52-executable-support-hurdle-"
            "20260713-v1\\report.json"
        ),
    }
    assert publication["claims"] == {
        "ai_uplift_claim": False,
        "leverage_applied": False,
        "live_authority": False,
        "profitability_claim": False,
        "selection_contaminated": True,
        "status": "rejected",
        "testnet_authority": False,
        "trading_authority": False,
        "untouched_data_expansion_authorized": False,
    }
    assert publication["result"] == {
        "ai_expected_payoff_spearman_improvement": 0.0003752313719727132,
        "ai_probability_log_loss_improvement": 0.001463466267459912,
        "deterministic_hurdle_calibration_base_mean_net_bps": (
            -1.0577574655257402
        ),
        "deterministic_hurdle_calibration_stress_mean_net_bps": (
            -2.3910907988590737
        ),
        "deterministic_hurdle_calibration_trades": 9,
        "deterministic_hurdle_consumed_evaluation_base_mean_net_bps": (
            5.162619173795532
        ),
        "deterministic_hurdle_consumed_evaluation_stress_mean_net_bps": (
            3.807621643943038
        ),
        "deterministic_hurdle_consumed_evaluation_trades": 15,
        "formally_selected_policies": 0,
        "models": 27,
        "passed_mechanisms": 0,
        "prediction_artifacts": 54,
    }

    declared = {item["path"] for item in publication["artifacts"]}
    actual = {
        path.relative_to(LATEST).as_posix()
        for path in LATEST.rglob("*")
        if path.is_file()
    }
    assert actual == declared | {"report.json"}
    for artifact in publication["artifacts"]:
        path = LATEST / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]


def test_latest_action_value_source_report_is_exact_and_fail_closed() -> None:
    source = _source()
    canonical = dict(source)
    claimed = canonical.pop("report_canonical_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert claimed == "ace44ebc33dc0601306841b4c353b43a184b2aa604b49f73d2301257f86d2f7f"
    assert hashlib.sha256((LATEST / "screen.json").read_bytes()).hexdigest() == (
        "c5b728161535372d934ff9087a24b81c2490246cd17cb56beb3e29a3052d73fa"
    )
    assert source["schema_version"] == (
        "round-052-executable-support-hurdle-fincast-report-v1"
    )
    assert source["round"] == 52
    assert source["implementation_commit"] == (
        "3cb9500fb1c34eeab77c2b294ed8c1fd282e2247"
    )
    assert source["round_gate"] == {
        "passed": False,
        "reasons": [
            "selection_contaminated_consumed_development_interval",
            "no_candidate_passed_predictive_and_economic_gates",
        ],
    }
    assert source["claims"] == {
        "ai_uplift_claim": False,
        "leverage_applied": False,
        "live_authority": False,
        "profitability_claim": False,
        "selection_contaminated": True,
        "testnet_authority": False,
        "trading_authority": False,
    }
    assert source["mechanism_screen"] == {
        "passed_candidates": [],
        "trading_or_promotion_authorized": False,
        "untouched_data_expansion_authorized": False,
    }
    assert set(source["data"]) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    for evidence in source["data"].values():
        assert evidence["synthetic_rows"] == 0
        assert evidence["source_evidence"]["verified"] is True
        assert evidence["source_evidence"]["source_archive_count"] == 30
        fincast = evidence["fincast"]
        assert fincast["source_round"] == 51
        assert fincast["rerun"] is False
        assert fincast["feature_count"] == 30
        assert fincast["rows"] == evidence["microstructure_rows"]
        assert fincast["artifact"]["bytes"] > 0
        assert len(fincast["artifact"]["sha256"]) == 64
    assert source["source_round_51"] == {
        "report_canonical_sha256": (
            "b97a12764256680402d526fd17ee56999c7f88335d66570a196aae3d0e9d0201"
        ),
        "report_file_sha256": (
            "d2e6c2e1a8ba0a48293f124d148359e4015f9c25aa61fb11b9dc7578d7975a80"
        ),
        "report_path": (
            "E:\\SimpleAITradingData\\round51-categorical-payoff-fincast-"
            "20260713-v2\\report.json"
        ),
    }

    model_records = [
        model
        for symbol_models in source["models"].values()
        for candidate_models in symbol_models.values()
        for model in candidate_models.values()
    ]
    assert len(model_records) == 27
    assert all(model["cache_state"] == "trained" for model in model_records)
    assert all(model["backend_kind"] == "opencl" for model in model_records)
    assert all(model["backend_device"] == "opencl:auto" for model in model_records)

    prediction_records = [
        artifact
        for symbol_predictions in source["prediction_artifacts"].values()
        for candidate_predictions in symbol_predictions.values()
        for seed_predictions in candidate_predictions.values()
        for artifact in seed_predictions.values()
    ]
    assert len(prediction_records) == 54
    assert all(artifact["bytes"] > 0 for artifact in prediction_records)
    assert all(len(artifact["sha256"]) == 64 for artifact in prediction_records)
    assert all(
        gate["passed"] is False for gate in source["predictive_gates"].values()
    )
    assert set(source["selected_policy"]) == CANDIDATES
    assert all(
        policy["selection_passed"] is False
        and policy["selected_coverage"] is None
        for policy in source["selected_policy"].values()
    )


def test_latest_action_value_tables_reconcile_to_round52_report() -> None:
    expected_counts = {
        "support.csv": 3,
        "forecast.csv": 108,
        "models.csv": 27,
        "policy-grid.csv": 18,
        "gates.csv": 10,
        "ai-uplift.csv": 1,
        "daily-policy.csv": 14,
        "progress.csv": 52,
    }
    for path, count in expected_counts.items():
        assert len(_csv(path)) == count

    support = {row["symbol"]: row for row in _csv("support.csv")}
    assert all(int(row["synthetic_rows"]) == 0 for row in support.values())
    assert math.isclose(
        float(support["BTCUSDT"]["long_executable_ratio"]),
        0.9280200275298038,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert math.isclose(
        float(support["ETHUSDT"]["short_executable_ratio"]),
        0.8722887685462928,
        rel_tol=0.0,
        abs_tol=1e-15,
    )
    assert math.isclose(
        float(support["SOLUSDT"]["long_executable_ratio"]),
        0.05465952472551863,
        rel_tol=0.0,
        abs_tol=1e-15,
    )

    forecast = _csv("forecast.csv")
    assert all(int(row["magnitude_floor_count"]) == 0 for row in forecast)
    evaluation = [row for row in forecast if row["role"] == "evaluation"]
    expected_averages = {
        "executable_direct_mean_lightgbm": (
            -0.00343888215809289,
            0.0115781570701407,
        ),
        "executable_hurdle_lightgbm": (
            -0.00779840855538155,
            0.0327730366080526,
        ),
        "executable_hurdle_lightgbm_fincast": (
            -0.00884410602760846,
            0.0331482679800253,
        ),
    }
    for candidate, (expected_mse_skill, expected_spearman) in expected_averages.items():
        rows = [row for row in evaluation if row["candidate"] == candidate]
        assert len(rows) == 18
        assert math.isclose(
            statistics.fmean(float(row["expected_payoff_mse_skill"]) for row in rows),
            expected_mse_skill,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        assert math.isclose(
            statistics.fmean(float(row["expected_payoff_spearman"]) for row in rows),
            expected_spearman,
            rel_tol=0.0,
            abs_tol=1e-15,
        )

    policies = _csv("policy-grid.csv")
    assert all(row["policy_calibration_passed"] == "false" for row in policies)
    assert all(row["formally_selected"] == "false" for row in policies)
    hurdle = next(
        row
        for row in policies
        if row["candidate"] == "executable_hurdle_lightgbm"
        and math.isclose(float(row["coverage"]), 0.0025, abs_tol=1e-12)
    )
    assert int(hurdle["policy_calibration_base_trades"]) == 9
    assert float(hurdle["policy_calibration_base_mean_net_bps"]) == (
        -1.0577574655257402
    )
    assert float(hurdle["policy_calibration_paired_stress_mean_net_bps"]) == (
        -2.3910907988590737
    )
    assert int(hurdle["consumed_evaluation_base_trades"]) == 15
    assert float(hurdle["consumed_evaluation_base_mean_net_bps"]) == (
        5.162619173795532
    )
    assert float(hurdle["consumed_evaluation_paired_stress_mean_net_bps"]) == (
        3.807621643943038
    )

    ai = _csv("ai-uplift.csv")[0]
    assert float(ai["probability_log_loss_improvement"]) == 0.001463466267459912
    assert float(ai["expected_payoff_spearman_improvement"]) == (
        0.0003752313719727132
    )
    assert ai["passed"] == "false"
    assert int(ai["parameter_count"]) == 991_437_160

    daily = _csv("daily-policy.csv")
    assert all(row["formally_selected"] == "false" for row in daily)
    assert all(row["selection_contaminated"] == "true" for row in daily)
    progress = _csv("progress.csv")
    assert [int(row["round"]) for row in progress] == list(range(1, 53))
    latest = progress[-1]
    assert latest["status"] == "rejected"
    assert latest["selection_contaminated"] == "True"
    assert latest["development_consumed"] == "True"
    assert latest["selected_signals"] == "0"
    assert latest["executable_trades"] == "0"
    assert latest["best_policy_trades"] == "15"
    assert latest["best_model_id"].endswith("consumed_diagnostic_not_selected")
    assert latest["architecture_gates_passed"] == "0"
    assert latest["architecture_gate_count"] == "3"


def test_latest_action_value_charts_are_accessible_and_round52_only() -> None:
    expected_charts = {
        "ai-uplift.svg",
        "daily-equity.svg",
        "executable-support.svg",
        "forecast-quality.svg",
        "policy-economics.svg",
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

    readme = (LATEST / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Round 52: Executable-Support Hurdle")
    assert "calibration rejected every policy" in readme
    assert "cannot be selected" in readme
    assert "below both frozen `0.005` gates" in readme
    assert "No profitability" in readme
    assert "# Round 51:" not in readme
