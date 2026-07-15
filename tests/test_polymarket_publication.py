from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


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
    assert sum(int(row["feature_rows"]) for row in markets) == report["dataset"][
        "row_count"
    ]
    for asset, evidence in report["per_asset"].items():
        asset_rows = [row for row in markets if row["asset"] == asset]
        assert len(asset_rows) == evidence["official_resolutions"] == 4
        assert sum(int(row["feature_rows"]) for row in asset_rows) == evidence[
            "feature_rows"
        ]
        assert sum(int(row["feature_rows"]) > 0 for row in asset_rows) == report[
            "dataset"
        ]["labeled_market_counts"][asset]

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


def test_latest_round_with_disqualified_capture_publishes_no_stale_chart() -> None:
    latest = RESEARCH / "latest"
    readme = (latest / "README.md").read_text(encoding="utf-8").lower()

    assert "run is disqualified" in readme
    assert "no performance graph is published" in readme
    assert not (latest / "charts" / "causal-feature-coverage.svg").exists()


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
    assert sum(
        gaps[stream]["count"]
        for stream in ("binance_spot", "clob_market", "polymarket_rtds")
    ) == gaps["total"] == 54
    assert capture["model_evidence"]["eligible_for_model_fit"] is False
    assert capture["model_evidence"]["profitability_result"] is None

    benchmark = json.loads(
        (RESEARCH / "recorder-v2-liveness-2026-07-15.json").read_text(
            encoding="utf-8"
        )
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


def test_round_003_ai_risk_evidence_is_truthfully_scoped() -> None:
    latest = RESEARCH / "latest"
    selected = json.loads(
        (latest / "ai-risk-selected.json").read_text(encoding="utf-8")
    )
    rejected = json.loads(
        (latest / "ai-risk-challengers-rejected.json").read_text(
            encoding="utf-8"
        )
    )
    contract = (RESEARCH / "round-003-market-anchored-model-contract.md").read_text(
        encoding="utf-8"
    )

    assert selected["benchmark_contract"] == "finance-risk-review-adversarial-v6"
    assert selected["selected_model"] == "qwen3:8b"
    assert selected["financial_edge_tested"] is False
    assert selected["trading_authority"] is False
    selected_result = selected["results"][0]
    assert selected_result["passed"] is True
    assert selected_result["action_match_cases"] == len(selected["tests"]) == 11
    assert selected_result["valid_json_cases"] == 11

    assert rejected["benchmark_contract"] == selected["benchmark_contract"]
    assert rejected["selected_model"] is None
    assert rejected["financial_edge_tested"] is False
    assert {item["model"] for item in rejected["results"]} == {
        "qwen3.5:9b",
        "fin-r1:8b",
    }
    assert all(item["passed"] is False for item in rejected["results"])
    assert "prospective profitability is not established" in contract.lower()
    assert "not market-edge evidence" in contract.lower()
