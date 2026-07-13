from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "model-research" / "action-value"
LATEST = RESEARCH / "latest"
PUBLISHER = ROOT / "tools" / "publish_round51_categorical_payoff_fincast.py"


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


def test_latest_action_value_publication_is_round51_hash_verified() -> None:
    publication = json.loads((LATEST / "report.json").read_text(encoding="utf-8"))
    canonical = dict(publication)
    claimed = canonical.pop("publication_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert claimed == "30173edfc3b5656e05aae9a1d4e047e06803804ed8c163cbbbda16231eb9a449"
    assert publication["schema_version"] == "categorical-payoff-fincast-publication-v1"
    assert publication["round"] == 51
    assert publication["status"] == "rejected"
    assert publication["design_sha256"] == (
        "42f9afbda8755807e898fa8bb54ad4039d1f9f6b2f4d6c825afc0b1d02bcfba3"
    )
    assert publication["binding_sha256"] == (
        "4e4f17914bb27411675ffd4dfa7a64c56b3f4515025a95ffa675edcab031b5b1"
    )
    assert publication["source_report_canonical_sha256"] == (
        "b97a12764256680402d526fd17ee56999c7f88335d66570a196aae3d0e9d0201"
    )
    assert publication["source_report_file_sha256"] == (
        "d2e6c2e1a8ba0a48293f124d148359e4015f9c25aa61fb11b9dc7578d7975a80"
    )
    assert publication["source_implementation_commit"] == (
        "ae61bcb87765a5fd1e6610d358808848529de60c"
    )
    assert publication["publisher_sha256"] == hashlib.sha256(
        PUBLISHER.read_bytes()
    ).hexdigest()
    assert publication["source_period"] == "2023-05-16..2023-06-14"
    assert publication["evaluation_period"] == "2023-06-09..2023-06-14"
    assert publication["decision_cadence_seconds"] == 10
    assert publication["target_path_resolution_ms"] == 100
    assert publication["target_horizon_seconds"] == 300
    assert publication["source_market_rows_synthetic"] == 0
    assert publication["model_artifact_count"] == 27
    assert publication["prediction_artifact_count"] == 27
    assert publication["fincast_feature_artifact_count"] == 3
    assert publication["external_artifacts_hash_verified"] is True
    assert publication["external_artifacts_verified_bytes"] == 322_506_457
    assert publication["fincast_parameter_count"] == 991_437_160
    assert publication["fincast_backend_kind"] == "directml"
    assert publication["lightgbm_backend_kind"] == "opencl"
    assert publication["selected_trades"] == 0
    assert publication["round_gate_passed"] is False
    assert publication["distribution_gate_pass_count"] == 0
    assert publication["economic_gate_pass_count"] == 0
    assert publication["ai_uplift_gate_passed"] is False
    for field in (
        "trading_authority",
        "testnet_authority",
        "live_authority",
        "profitability_claim",
        "leverage_applied",
        "ai_uplift_claim",
    ):
        assert publication[field] is False

    declared = {item["path"] for item in publication["artifact_integrity"]}
    for chart, sources in publication["graph_sources"].items():
        assert chart in declared
        assert all(source in declared for source in sources)
    for artifact in publication["artifact_integrity"]:
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
    assert claimed == "b97a12764256680402d526fd17ee56999c7f88335d66570a196aae3d0e9d0201"
    assert hashlib.sha256((LATEST / "screen.json").read_bytes()).hexdigest() == (
        "d2e6c2e1a8ba0a48293f124d148359e4015f9c25aa61fb11b9dc7578d7975a80"
    )
    assert source["schema_version"] == "categorical-payoff-fincast-screen-report-v1"
    assert source["round"] == 51
    assert source["round_gate"] == {"passed": False, "promotion_permitted": False}
    assert source["claims"] == {
        "ai_uplift_claim": False,
        "beta_research_only": True,
        "leverage_applied": False,
        "live_authority": False,
        "profitability_claim": False,
        "selection_contaminated": True,
        "source_market_rows_synthetic": 0,
        "testnet_authority": False,
        "trading_authority": False,
    }
    assert set(source["data"]) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    assert set(source["symbol_results"]) == {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
    assert set(source["portfolio_results"]) == {
        "direct_mean_lightgbm",
        "categorical_payoff_lightgbm",
        "categorical_payoff_lightgbm_fincast",
    }

    model_records = 0
    for symbol, evidence in source["data"].items():
        assert evidence["synthetic_rows"] == 0
        assert evidence["source_evidence"]["verified"] is True
        assert evidence["source_evidence"]["source_archive_count"] == 30
        assert evidence["fincast"]["runtime"]["parameter_count"] == 991_437_160
        assert evidence["fincast"]["runtime"]["backend_kind"] == "directml"
        assert evidence["fincast"]["runtime"]["backend_device"] == "privateuseone:0"
        assert evidence["fincast"]["warning_count"] == 0
        assert evidence["fincast"]["cpu_fallback_warning_count"] == 0
        for candidate, result in source["symbol_results"][symbol].items():
            assert result["selection"]["eligible_rows"] == 0
            assert result["base_trace"]["metrics"]["trades"] == 0
            assert result["stress_trace"]["metrics"]["trades"] == 0
            assert len(result["models"]) == 3
            assert {model["seed"] for model in result["models"]} == {
                5101,
                5102,
                5103,
            }
            assert all(model["backend_kind"] == "opencl" for model in result["models"])
            model_records += len(result["models"])
    assert model_records == 27
    assert all(
        gate["passed"] is False for gate in source["distribution_gates"].values()
    )
    assert all(gate["passed"] is False for gate in source["economic_gates"].values())
    assert source["ai_uplift_gate"]["passed"] is False


def test_latest_action_value_tables_reconcile_to_round51_report() -> None:
    expected_counts = {
        "forecast.csv": 54,
        "prediction-tails.csv": 18,
        "barrier-baselines.csv": 12,
        "scenarios.csv": 6,
        "symbols.csv": 18,
        "models.csv": 27,
        "roles.csv": 12,
        "sources.csv": 3,
        "gates.csv": 53,
        "ai-uplift.csv": 3,
        "daily-policy.csv": 36,
        "progress.csv": 51,
    }
    for path, count in expected_counts.items():
        assert len(_csv(path)) == count

    tails = _csv("prediction-tails.csv")
    assert all(float(row["worst_seed_prediction_max_bps"]) < 0.0 for row in tails)
    assert all(int(row["all_seed_positive_rows"]) == 0 for row in tails)
    assert all(int(row["selected_rows"]) == 0 for row in tails)
    best = max(tails, key=lambda row: float(row["worst_seed_prediction_max_bps"]))
    assert (best["symbol"], best["side"], best["candidate_id"]) == (
        "SOLUSDT",
        "long",
        "categorical_payoff_lightgbm_fincast",
    )
    assert math.isclose(
        float(best["worst_seed_prediction_max_bps"]),
        -1.6180926234723985,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    scenarios = _csv("scenarios.csv")
    assert all(int(row["trades"]) == 0 for row in scenarios)
    assert all(float(row["total_net_bps"]) == 0.0 for row in scenarios)
    assert all(float(row["max_drawdown_bps"]) == 0.0 for row in scenarios)
    assert all(row["profit_factor"] == "" for row in scenarios)
    daily = _csv("daily-policy.csv")
    assert all(int(row["selected_trades"]) == 0 for row in daily)
    assert all(float(row["net_bps"]) == 0.0 for row in daily)
    assert all(row["zero_return_reason"] == "no selected trades" for row in daily)

    barriers = _csv("barrier-baselines.csv")
    base = [row for row in barriers if row["scenario"] == "base"]
    assert len(base) == 6
    assert all(float(row["mean_net_bps"]) < -12.0 for row in base)
    assert all(0.0 < float(row["positive_ratio"]) < 0.25 for row in base)

    ai = _csv("ai-uplift.csv")[0]
    assert float(ai["ranked_probability_skill_improvement"]) == (
        -7.377429005321623e-05
    )
    assert float(ai["expected_payoff_spearman_improvement"]) == (
        0.0020234792018059625
    )
    assert float(ai["expected_payoff_mse_ratio"]) == 1.0011707793594598
    assert ai["passed"] == "False"

    progress = _csv("progress.csv")
    assert [int(row["round"]) for row in progress] == list(range(1, 52))
    latest = progress[-1]
    assert latest["status"] == "rejected"
    assert latest["selection_contaminated"] == "True"
    assert latest["development_consumed"] == "True"
    assert latest["executable_trades"] == "0"
    assert latest["policy_eligible_rows"] == "0"
    assert latest["mean_net_bps"] == ""
    assert latest["architecture_gates_passed"] == "0"
    assert latest["architecture_gate_count"] == "5"


def test_latest_action_value_charts_are_accessible_and_stale_files_are_absent() -> None:
    expected_charts = {
        "ai-uplift.svg",
        "barrier-baselines.svg",
        "calibration.svg",
        "daily-equity.svg",
        "forecast-quality.svg",
        "prediction-tail.svg",
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
        "daily-equity.csv",
        "mechanism.csv",
        "monthly-forecast.csv",
        "monthly-performance.csv",
        "seed-stability.csv",
        "symbol-forecast.csv",
        "target-baselines.csv",
        "trades.csv",
        "training.csv",
        "daily-equity-drawdown.svg",
        "event-quality.svg",
        "expected-payoff-quality.svg",
        "monthly-performance.svg",
        "policy-economics.svg",
        "seed-stability.svg",
        "symbol-performance.svg",
        "training-dynamics.svg",
    ):
        assert not (LATEST / stale).exists()
        assert not (LATEST / "charts" / stale).exists()

    readme = (LATEST / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Round 51: Categorical Payoff + FinCast")
    assert "zero eligible actions" in readme
    assert "not a multi-year claim" in readme
    assert "did not establish economic uplift" in readme
    assert "approved for testnet" in readme
    assert "# Round 50:" not in readme
