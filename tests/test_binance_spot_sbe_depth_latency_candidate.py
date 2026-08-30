from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "docs/model-research/action-value/"
    "binance-spot-sbe-depth-latency-overlay-source-failure-adjudication-v1-2026-08-30.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    encoded = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_sbe_source_failure_is_preserved_and_retained_bytes_are_adjudicated() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]

    sources = result["sources"]
    for source_name in (
        "retained_SBE_source",
        "retained_JSON_source",
        "retained_changelog",
        "local_unconsumed_preflight_error",
    ):
        source = sources[source_name]
        assert _sha256(ROOT / source["path"]) == source["sha256"]

    contract_source = sources["consumed_contract"]
    contract = json.loads((ROOT / contract_source["path"]).read_bytes())
    assert _canonical_hash(contract, "contract_sha256") == contract_source[
        "contract_sha256"
    ]
    source_result_source = sources["consumed_source_result"]
    source_result_path = ROOT / source_result_source["path"]
    source_result = json.loads(source_result_path.read_bytes())
    assert _sha256(source_result_path) == source_result_source["file_sha256"]
    assert _canonical_hash(source_result, "result_sha256") == source_result_source[
        "result_sha256"
    ]
    assert source_result["source_gate"]["passed"] is False
    assert source_result["source_gate"]["required_phrase_presence"][
        "An API Key is necessary for access."
    ] is False

    sbe_text = (ROOT / sources["retained_SBE_source"]["path"]).read_text(
        encoding="utf-8"
    )
    assert "**An API Key is necessary for access**." in sbe_text
    assert "Only Ed25519 keys are allowed." in sbe_text
    assert "**Update Speed:** 20ms" in sbe_text
    json_text = (ROOT / sources["retained_JSON_source"]["path"]).read_text(
        encoding="utf-8"
    )
    assert "**Update Speed:** 1000ms or 100ms" in json_text


def test_sbe_cadence_candidate_does_not_overclaim_top_of_book_or_profit() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    comparison = result["nominal_cadence_comparison"]
    assert comparison["SBE_diff_depth_updates_per_one_fastest_JSON_interval"] == 5
    assert comparison["documented_interval_reduction_ms"] == 80
    assert comparison["end_to_end_arrival_latency_claim"] is False
    assert result["retained_current_terms"]["best_bid_ask_update_speed"] == "real_time"
    assert result["retained_json_comparator"]["book_ticker_update_speed"] == (
        "real_time"
    )
    assert result["adjudication"]["accepted_edge"] is False
    assert result["adjudication"]["candidate"] is True
    assert result["adjudication"]["standalone_after_cost_profit_floor_USDT"] == "0"

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    rank_5 = next(
        row for row in registry["prioritized_hypotheses"] if row["priority_rank"] == 5
    )
    assert rank_5["sbe_depth_freshness_retry_trigger"] == result["adjudication"][
        "retry_trigger"
    ]
    assert any(
        row["result_sha256"] == result["result_sha256"]
        for row in rank_5["canonical_artifacts"]
    )
    assert any(
        row["canonical_result_sha256"] == result["result_sha256"]
        for row in registry["terminal_do_not_repeat"]
    )
