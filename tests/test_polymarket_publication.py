from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from simple_ai_trading.polymarket_repricing_publication import (
    publish_polymarket_repricing_report,
)


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "docs" / "model-research" / "polymarket"


def test_round_002_publication_is_internally_consistent() -> None:
    report = json.loads(
        (RESEARCH / "round-002-prospective-pipeline-evidence.json").read_text(
            encoding="utf-8"
        )
    )
    with (RESEARCH / "round-002-market-rows.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        markets = list(csv.DictReader(handle))

    assert report["round"] == 2
    assert report["status"] == "pipeline_verified_model_evaluation_blocked"
    assert len(report["recorder"]["report_sha256"]) == 64
    assert len(report["dataset"]["dataset_sha256"]) == 64
    assert len(markets) == report["recorder"]["market_snapshot_count"] == 12
    assert (
        sum(int(row["feature_rows"]) for row in markets)
        == report["dataset"]["row_count"]
    )
    for asset, evidence in report["per_asset"].items():
        asset_rows = [row for row in markets if row["asset"] == asset]
        assert len(asset_rows) == evidence["official_resolutions"] == 4
        assert (
            sum(int(row["feature_rows"]) for row in asset_rows)
            == evidence["feature_rows"]
        )
        assert (
            sum(int(row["feature_rows"]) > 0 for row in asset_rows)
            == report["dataset"]["labeled_market_counts"][asset]
        )

    manifest = {entry["path"]: entry for entry in report["artifact_integrity"]}
    assert set(report["tracked_artifacts"]) - {
        "docs/model-research/polymarket/round-002-prospective-pipeline-evidence.json"
    } == set(manifest)
    for relative_path, expected in manifest.items():
        artifact = ROOT / relative_path
        payload = artifact.read_bytes()
        assert len(payload) == expected["bytes"]
        assert hashlib.sha256(payload).hexdigest() == expected["sha256"]
    market_manifest = manifest[
        "docs/model-research/polymarket/round-002-market-rows.csv"
    ]
    assert market_manifest["row_count"] == len(markets)
    assert market_manifest["columns"] == list(markets[0])


def test_latest_publication_reports_pending_round_13_without_invented_metrics() -> None:
    latest = RESEARCH / "latest"
    readme = (latest / "README.md").read_text(encoding="utf-8").lower()
    chart = (
        (latest / "charts" / "optimization-progress.svg")
        .read_text(encoding="utf-8")
        .lower()
    )
    with (latest / "tables" / "optimization-progress.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = {row["round"]: row for row in csv.DictReader(handle)}

    assert "round 13 is frozen but has not started" in readme
    assert "round 12 is not performance evidence" in readme
    assert "no profitability, roi, acceptable-drawdown" in readme
    assert "round 13" in chart
    assert "n/a simulated fills | frozen; fresh capture not started" in chart
    assert "neither has performance metrics" in chart
    unavailable_metrics = {
        "independent_groups",
        "conditions",
        "selected_filled_conditions",
        "total_utility_quote",
        "maximum_drawdown_quote",
        "bootstrap_lower_mean_group_utility_quote",
    }
    for round_number in ("12", "13"):
        assert all(rows[round_number][field] == "" for field in unavailable_metrics)
        assert rows[round_number]["profitability_claim"] == "False"


def test_pending_round_013_manifest_reconstructs_every_artifact() -> None:
    manifest = json.loads(
        (RESEARCH / "latest" / "publication-integrity.json").read_text(encoding="utf-8")
    )
    claimed = manifest.pop("publication_sha256")
    canonical = json.dumps(
        manifest,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )

    assert hashlib.sha256(canonical.encode("ascii")).hexdigest() == claimed
    assert manifest["schema_version"] == "polymarket-round13-pending-publication-v1"
    assert manifest["latest_round"] == 13
    assert manifest["status"] == "round13_frozen_fresh_capture_not_started"
    assert manifest["profitability_claim"] is False
    assert manifest["roi_claim"] is False
    assert manifest["drawdown_claim"] is False
    assert manifest["paper_authority"] is False
    assert manifest["trading_authority"] is False
    artifact_paths = [entry["path"] for entry in manifest["artifacts"]]
    assert artifact_paths == sorted(set(artifact_paths))
    root = ROOT.resolve()
    for entry in manifest["artifacts"]:
        path = (ROOT / entry["path"]).resolve()
        assert path.is_relative_to(root)
        payload = path.read_bytes()
        assert len(payload) == entry["bytes"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]


def test_round_008_publication_is_deterministic_and_refuses_tampering(
    tmp_path: Path,
) -> None:
    report = RESEARCH / "round-008-executable-repricing-ceiling-report.json"
    capture = RESEARCH / "round-002-prospective-pipeline-evidence.json"
    local_capture = tmp_path / capture.name
    local_capture.write_bytes(capture.read_bytes())
    first = publish_polymarket_repricing_report(report, local_capture, tmp_path)
    second = publish_polymarket_repricing_report(report, local_capture, tmp_path)

    assert first == second
    manifest = json.loads(
        (tmp_path / "latest" / "publication-integrity.json").read_text(encoding="utf-8")
    )
    assert manifest["claims"]["noncausal_oracle_upper_bound"] is True
    assert manifest["claims"]["profitability_claim"] is False
    assert manifest["claims"]["trading_authority"] is False
    assert manifest["source_report_sha256"] == first.report_sha256

    tampered = tmp_path / "tampered.json"
    payload = json.loads(report.read_text(encoding="utf-8"))
    payload["profitability_claim"] = True
    tampered.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="publication validation"):
        publish_polymarket_repricing_report(tampered, local_capture, tmp_path / "bad")


def test_degraded_capture_and_recorder_benchmark_are_arithmetically_truthful() -> None:
    capture = json.loads(
        (RESEARCH / "round-003-degraded-capture-diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    recorder = capture["capture"]
    gaps = capture["gap_diagnostic"]
    assert sum(recorder["stream_counts"].values()) == recorder["raw_message_count"]
    assert math.isclose(
        recorder["duration_seconds"],
        (recorder["ended_at_ms"] - recorder["started_at_ms"]) / 1_000.0,
    )
    assert sum(capture["resolution"]["asset_outcome_counts"].values()) == 105
    assert capture["resolution"]["finalized_condition_count"] == 105
    assert capture["resolution"]["pending_condition_count"] == 0
    for stream in ("binance_spot", "clob_market", "polymarket_rtds"):
        assert sum(gaps[stream]["reasons"].values()) == gaps[stream]["count"]
    assert (
        sum(
            gaps[stream]["count"]
            for stream in ("binance_spot", "clob_market", "polymarket_rtds")
        )
        == gaps["total"]
        == 54
    )
    assert capture["model_evidence"]["eligible_for_model_fit"] is False
    assert capture["model_evidence"]["profitability_result"] is None

    benchmark = json.loads(
        (RESEARCH / "recorder-v2-liveness-2026-07-15.json").read_text(encoding="utf-8")
    )
    assert benchmark["comparison_limits"]["profitability_evidence"] is False
    before = benchmark["measurements"]["before_coalescing"]
    after = benchmark["measurements"]["coalesced_writer"]
    for measurement in (before, after):
        assert measurement["status"] == "complete"
        assert measurement["stream_gap_count"] == 0
        assert measurement["integrity_error_count"] == 0
        assert math.isclose(
            measurement["average_messages_per_chunk"],
            measurement["raw_message_count"] / measurement["raw_chunk_count"],
        )
        assert math.isclose(
            measurement["compressed_to_uncompressed_ratio"],
            measurement["compressed_bytes"] / measurement["uncompressed_bytes"],
        )
    observed = benchmark["observed_change"]
    before_chunks_per_10k = (
        10_000 * before["raw_chunk_count"] / before["raw_message_count"]
    )
    after_chunks_per_10k = (
        10_000 * after["raw_chunk_count"] / after["raw_message_count"]
    )
    assert math.isclose(
        observed["chunks_per_10000_messages_before"], before_chunks_per_10k
    )
    assert math.isclose(
        observed["chunks_per_10000_messages_after"], after_chunks_per_10k
    )
    assert math.isclose(
        observed["chunks_per_10000_messages_reduction_fraction"],
        1.0 - (after_chunks_per_10k / before_chunks_per_10k),
    )
    assert math.isclose(
        observed["messages_per_chunk_factor"],
        after["average_messages_per_chunk"] / before["average_messages_per_chunk"],
    )

    long_tail = json.loads(
        (RESEARCH / "storage-v3-long-tail-benchmark-2026-07-16.json").read_text(
            encoding="utf-8"
        )
    )
    claimed = long_tail.pop("report_sha256")
    canonical = json.dumps(
        long_tail,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    assert hashlib.sha256(canonical.encode("ascii")).hexdigest() == claimed
    assert long_tail["storage_schema_version"] == "polymarket-evidence-storage-v3"
    assert long_tail["persisted_raw_messages"] == long_tail["total_messages"]
    assert long_tail["integrity_errors"] == []
    assert not any(
        constraint[1] in {"PRIMARY KEY", "UNIQUE"}
        for constraint in long_tail["hot_path_constraints"]
    )
    assert (
        min(
            checkpoint["interval_messages_per_second"]
            for checkpoint in long_tail["checkpoints"]
        )
        > 9_700
    )
    assert long_tail["truth_constraints"] == {
        "benchmark_proves_fifteen_hour_capture": False,
        "benchmark_receipt_metadata_is_real": False,
        "financial_edge_claim": False,
        "model_evidence": False,
        "source_payloads_are_real": True,
        "trading_authority": False,
    }


def test_current_ai_risk_evidence_is_truthfully_scoped() -> None:
    latest = RESEARCH / "latest"
    rejected = json.loads(
        (latest / "ai-risk-models-rejected.json").read_text(encoding="utf-8")
    )
    contract = (RESEARCH / "round-003-market-anchored-model-contract.md").read_text(
        encoding="utf-8"
    )

    assert rejected["benchmark_contract"] == "finance-risk-review-adversarial-v7"
    assert rejected["selected_model"] is None
    assert rejected["financial_edge_tested"] is False
    assert rejected["trading_authority"] is False
    assert {item["model"] for item in rejected["results"]} == {
        "qwen3:8b",
        "qwen3.5:9b",
        "fin-r1:8b",
        "fino1:8b",
    }
    assert all(item["passed"] is False for item in rejected["results"])
    assert all(item["valid_json_cases"] == 11 for item in rejected["results"])
    assert "prospective profitability is not established" in contract.lower()
    assert "not market-edge evidence" in contract.lower()
