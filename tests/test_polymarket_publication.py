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


def test_latest_round_publishes_only_noncausal_round_8_ceiling() -> None:
    latest = RESEARCH / "latest"
    readme = (latest / "README.md").read_text(encoding="utf-8").lower()
    chart = (latest / "charts" / "repricing-ceiling.svg").read_text(
        encoding="utf-8"
    )

    assert "noncausal mechanism ceiling" in readme
    assert "not roi or a trading strategy" in readme
    assert "not roi, not a causal strategy" in chart.lower()
    assert "2026-07-15t00:46:38.779z" in chart.lower()
    assert "2026-07-15t00:55:51.787z" in chart.lower()
    assert not (latest / "charts" / "causal-feature-coverage.svg").exists()


def test_round_008_committed_manifest_reconstructs_every_artifact() -> None:
    manifest = json.loads(
        (RESEARCH / "latest" / "publication-integrity.json").read_text(
            encoding="utf-8"
        )
    )
    claimed = manifest.pop("manifest_sha256")
    canonical = json.dumps(
        manifest,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )

    assert hashlib.sha256(canonical.encode("ascii")).hexdigest() == claimed
    assert manifest["claims"]["noncausal_oracle_upper_bound"] is True
    assert manifest["claims"]["profitability_claim"] is False
    assert manifest["claims"]["ai_edge_evaluated"] is False
    for entry in manifest["generated_artifacts"]:
        path = RESEARCH / entry["path"]
        payload = path.read_bytes()
        assert len(payload) == entry["bytes"]
        assert hashlib.sha256(payload).hexdigest() == entry["sha256"]
        if path.suffix == ".csv":
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == entry["row_count"]
            assert list(rows[0]) == entry["columns"]


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
        (tmp_path / "latest" / "publication-integrity.json").read_text(
            encoding="utf-8"
        )
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
        publish_polymarket_repricing_report(
            tampered, local_capture, tmp_path / "bad"
        )


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
