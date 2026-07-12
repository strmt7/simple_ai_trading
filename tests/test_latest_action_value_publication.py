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


def test_latest_action_value_publication_is_round36_hash_verified() -> None:
    report = json.loads((LATEST / "report.json").read_text(encoding="utf-8"))
    canonical = dict(report)
    claimed = canonical.pop("publication_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert claimed == "c0b6dceae845b8976148f62d5ac7aad0b90af307216a8062fcf68ce3141d5ff9"
    assert report["schema_version"] == "multi-horizon-signal-decay-publication-v1"
    assert report["round"] == 36
    assert report["status"] == "rejected"
    assert report["source_implementation_commit"] == (
        "3c16bed56370c2dbf393fbb8dd19f54007419ee0"
    )
    assert report["source_report_canonical_sha256"] == (
        "2cb799246b51e610647f330d2b0a5745a6e96f56062782eb23055cb6b0905ee8"
    )
    assert report["source_report_file_sha256"] == (
        "dcad6c89f50cb5a1b5f51816826f7844adafdc895da0c7a1612316e97dcf7633"
    )
    assert report["signal_count"] == 13
    assert report["horizon_count"] == 7
    assert report["signal_horizon_cells"] == 91
    assert report["model_candidate"] is None
    assert report["model_trained"] is False
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
    assert source["status"] == "diagnostic_complete_no_authority"
    assert source["completeness"] == {
        "all_cells_complete": True,
        "expected_daily_records": 455,
        "expected_regime_records": 819,
        "expected_signal_horizon_cells": 91,
        "placebo_replicates_per_cell": 200,
        "reported_daily_records": 455,
        "reported_regime_records": 819,
        "reported_signal_horizon_cells": 91,
    }
    assert source["stage_access"] == {
        "calibration_metrics": True,
        "certified_barrier_targets_recomputed_for_identity_only": True,
        "certified_source_materialized_through_development": True,
        "development_prediction_or_metrics": False,
        "distant_confirmation_prediction_or_metrics": False,
        "distant_confirmation_source_materialized": False,
        "early_stop_prediction_or_metrics": False,
        "policy_prediction_or_metrics": False,
        "train_prediction_or_metrics": False,
    }
    assert source["zero_latency_quote_evidence"]["valid_timestamps"] == 80_307
    assert source["zero_latency_quote_evidence"]["invalid_timestamps"] == 0
    assert source["zero_latency_quote_evidence"]["counterfactual_only"] is True
    assert source["runtime_evidence"]["memory"]["peak_working_set_bytes"] == (
        9_620_840_448
    )
    for field in (
        "trading_authority",
        "execution_claim",
        "profitability_claim",
        "portfolio_claim",
        "leverage_applied",
        "model_trained",
    ):
        assert source[field] is False
    assert source["model_candidate"] is None


def test_latest_action_value_tables_preserve_all_round36_graph_data() -> None:
    signals = _csv("signals.csv")
    daily = _csv("daily.csv")
    regimes = _csv("regimes.csv")
    ranked = _csv("ranked-event-outcomes.csv")
    decay = _csv("decay.csv")
    support = _csv("horizon-support.csv")

    assert len(signals) == 91
    assert len(daily) == 455
    assert len(regimes) == 819
    assert len(ranked) == 273
    assert len(decay) == 91
    assert len(support) == 7
    assert {int(row["horizon_seconds"]) for row in signals} == {
        5,
        15,
        30,
        60,
        120,
        300,
        900,
    }
    assert {row["date"] for row in daily} == {
        "2023-06-21",
        "2023-06-22",
        "2023-06-23",
        "2023-06-24",
        "2023-06-25",
    }
    assert all(row["event_outcomes_not_executable_trades"] == "True" for row in ranked)
    assert all(float(row["mean_delayed_net_return_bps"]) < 0.0 for row in ranked)
    assert all(float(row["mean_delayed_net_return_bps"]) < 0.0 for row in signals)

    best_auc = max(signals, key=lambda row: float(row["weighted_roc_auc"]))
    assert (best_auc["signal"], best_auc["horizon_seconds"]) == (
        "l1_imbalance",
        "5",
    )
    assert float(best_auc["weighted_roc_auc"]) == 0.6072574903425851
    assert float(best_auc["daily_auc_minimum"]) == 0.5579105002109371
    assert int(best_auc["placebo_observed_rank_descending"]) == 1

    best_all = max(signals, key=lambda row: float(row["mean_delayed_net_return_bps"]))
    assert float(best_all["mean_delayed_net_return_bps"]) == -11.578984700095054
    best_100 = max(
        (row for row in ranked if row["requested_rows"] == "100"),
        key=lambda row: float(row["mean_delayed_net_return_bps"]),
    )
    assert float(best_100["mean_delayed_net_return_bps"]) == -3.502678431456879


def test_latest_action_value_progress_extends_to_round36_without_equity_claim() -> None:
    progress = _csv("progress.csv")

    assert [int(row["round"]) for row in progress] == list(range(1, 37))
    latest = progress[-1]
    assert latest["status"] == "rejected"
    assert latest["selection_contaminated"] == "True"
    assert latest["risk_level"] == "research-only; no policy"
    assert latest["selected_signals"] == "0"
    assert latest["executable_trades"] == "0"
    assert latest["direction_auc"] == "0.6072574903425851"
    assert latest["top_100_exact_after_cost_bps"] == "-3.502678431456879"
    assert latest["best_top_500_exact_after_cost_bps"] == "-10.008614095712256"


def test_latest_action_value_charts_are_accessible_and_round35_files_are_absent() -> (
    None
):
    expected_charts = {
        "after-cost-tails.svg",
        "cost-coverage.svg",
        "daily-direction-auc.svg",
        "placebo-comparison.svg",
        "research-progress.svg",
        "signal-decay.svg",
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

    for stale in ("variants.csv", "features.csv", "gates.csv", "models.csv"):
        assert not (LATEST / stale).exists()
    for stale in ("direction-auc.svg", "feature-gain.svg"):
        assert not (LATEST / "charts" / stale).exists()

    readme = (LATEST / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Round 36: direction exists, taker edge rejected")
    assert "# Round 35:" not in readme
    assert "Leverage cannot repair negative unlevered expectancy." in readme
    assert "no out-of-sample, ETHUSDT, SOLUSDT" in readme
    assert "no" in readme.casefold() and "profitability claim" in readme.casefold()
