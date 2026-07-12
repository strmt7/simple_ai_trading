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


def test_latest_action_value_publication_is_round35_hash_verified() -> None:
    report = json.loads((LATEST / "report.json").read_text(encoding="utf-8"))
    canonical = dict(report)
    claimed = canonical.pop("publication_sha256")

    assert claimed == _canonical_sha256(canonical)
    assert report["schema_version"] == "consumed-direction-screen-publication-v1"
    assert report["round"] == 35
    assert report["status"] == "rejected"
    assert report["source_implementation_commit"] == (
        "b89064d2eb594e590134a89bdc45ee6ecc2f93d7"
    )
    assert report["source_report_canonical_sha256"] == (
        "1c6d2b7e5914ef62b110ea6095661f460cdc75f215aee07e04b1cdc0979499ac"
    )
    assert report["source_report_file_sha256"] == (
        "a8a9718e11053002541c6fe77c96df44016c0fdf7a3a187103c0c3c10af9e861"
    )
    assert report["source_corpus_certificate_sha256"] == (
        "113437a381453d53eea811034f9a7e6ad573092e00efe8cc97d070a84f411ebe"
    )
    assert report["source_barrier_targets_sha256"] == (
        "68ba235b7d40abedb953c05c42948592e740070c4aec5e80cc2fcc550eba26fa"
    )
    assert report["source_cache_key"] == (
        "ca5ce2c7f1924717ecdc162a5382925f6f07b85c233b82ad5a8c1ec117ea0d85"
    )
    assert report["variant_count"] == 6
    assert report["architecture_freeze_eligible_variants"] == []
    assert report["architecture_freeze_candidate"] is None
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
    assert source["architecture_freeze_eligible_variants"] == []
    assert source["architecture_freeze_candidate"] is None
    assert source["calibration"]["positive_opportunity_rows"] == 15_222
    assert source["stage_access"] == {
        "certified_source_materialized_through_development": True,
        "train_used_for_fit": True,
        "early_stop_used_for_early_stopping": True,
        "calibration_prediction_and_metrics": True,
        "policy_prediction_or_metrics": False,
        "development_prediction_or_metrics": False,
        "distant_confirmation_source_materialization": False,
        "distant_confirmation_prediction_or_metrics": False,
    }
    for variant in source["variant_results"]:
        assert variant["architecture_freeze_eligible"] is False
        assert variant["rejection_reasons"]
        assert variant["model"]["backend_kind"] == "opencl"
        assert variant["model"]["artifact_reload_max_abs_prediction_error"] == 0.0
        assert variant["model"]["mirror_swap_max_abs_prediction_error"] <= 1e-12
        assert variant["model"]["nonfinite_prediction_count"] == 0
        for field in (
            "promotion_permitted",
            "trading_authority",
            "execution_claim",
            "profitability_claim",
            "portfolio_claim",
            "leverage_applied",
        ):
            assert variant[field] is False


def test_latest_action_value_tables_preserve_all_round35_graph_data() -> None:
    variants = _csv("variants.csv")
    daily = _csv("daily.csv")
    features = _csv("features.csv")
    gates = _csv("gates.csv")
    models = _csv("models.csv")

    expected_variants = [
        "full_uniqueness",
        "full_utility_margin",
        "noncycle_uniqueness",
        "noncycle_utility_margin",
        "compact_uniqueness",
        "compact_utility_margin",
    ]
    assert [row["variant"] for row in variants] == expected_variants
    assert len(daily) == 30
    assert len(features) == 550
    assert len(gates) == 48
    assert len(models) == 6
    assert {row["variant"] for row in daily} == set(expected_variants)
    assert {row["date"] for row in daily} == {
        "2023-06-21",
        "2023-06-22",
        "2023-06-23",
        "2023-06-24",
        "2023-06-25",
    }
    assert {row["variant"] for row in features} == set(expected_variants)
    assert {row["variant"] for row in gates} == set(expected_variants)
    assert all(row["architecture_freeze_eligible"] == "False" for row in variants)
    assert all(row["backend"] == "opencl" for row in variants)
    assert all(row["backend_kind"] == "opencl" for row in models)
    assert all(
        row["artifact_reload_max_abs_prediction_error"] == "0.0" for row in models
    )
    assert all(
        sum(row["passed"] == "True" for row in gates if row["variant"] == variant) <= 3
        for variant in expected_variants
    )

    by_name = {row["variant"]: row for row in variants}
    assert float(by_name["full_uniqueness"]["pooled_direction_auc"]) == (
        0.5425869424605662
    )
    assert (
        float(by_name["noncycle_utility_margin"]["frozen_top_500_mean_stress_net_bps"])
        == 0.5203813963863649
    )
    assert (
        float(
            by_name["noncycle_utility_margin"]["confidence_top_500_mean_stress_net_bps"]
        )
        < 0.0
    )
    assert (
        float(by_name["compact_utility_margin"]["frozen_top_500_mean_stress_net_bps"])
        < 0.0
    )


def test_latest_action_value_progress_extends_to_round35_without_equity_claim() -> None:
    progress = _csv("progress.csv")

    assert [int(row["round"]) for row in progress] == list(range(1, 36))
    latest = progress[-1]
    assert latest["status"] == "rejected"
    assert latest["selection_contaminated"] == "True"
    assert latest["risk_level"] == "research-only; no policy"
    assert latest["selected_signals"] == "0"
    assert latest["executable_trades"] == "0"
    assert latest["calibration_eligible_rows"] == "0"
    assert latest["direction_auc"] == "0.5425869424605662"
    assert latest["best_top_500_exact_after_cost_bps"] == "0.5203813963863649"
    assert latest["architecture_gates_passed"] == "3"
    assert latest["architecture_gate_count"] == "8"


def test_latest_action_value_charts_are_accessible_and_old_round_files_are_absent() -> (
    None
):
    expected_charts = {
        "after-cost-tails.svg",
        "daily-direction-auc.svg",
        "direction-auc.svg",
        "feature-gain.svg",
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
        "architecture.csv",
        "forecast.csv",
        "profiles.csv",
        "stages.csv",
        "candidates.csv",
        "thresholds.csv",
    ):
        assert not (LATEST / stale).exists()
    for stale in (
        "architecture-gates.svg",
        "eligibility.svg",
        "forecast-quality.svg",
        "stage-access.svg",
        "candidate-economics.svg",
        "threshold-economics.svg",
    ):
        assert not (LATEST / "charts" / stale).exists()

    readme = (LATEST / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Round 35: direct direction screen rejected")
    assert "# Round 34:" not in readme
    assert "post-hoc discovery" in readme
    assert "not evidence of profitability" in readme
    assert "No ETHUSDT or SOLUSDT result" in readme
