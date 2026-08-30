from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import _canonical_hash


ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "docs/model-research/action-value"
CONTRACT = ACTION / (
    "polymarket-crypto-sep6-range-threshold-delta-contract-v1-2026-08-30.json"
)
RESULT = ACTION / (
    "polymarket-crypto-sep6-range-threshold-delta-result-v1-2026-08-30.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
RAW = ROOT / (
    "data/polymarket-crypto-current-event-delta-v1/"
    "newest-open-crypto-events.raw"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def test_complete_new_crypto_pair_delta_rejects_before_books() -> None:
    contract = _load(CONTRACT)
    result = _load(RESULT)

    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert result["contract"]["sha256"] == contract["contract_sha256"]
    assert hashlib.sha256(RAW.read_bytes()).hexdigest() == (
        "4f9aadb6a95bdf2612845b3e3bc96146cc1ea5f23b3cb6bf8815ccb43c8ce087"
    )
    assert result["retained_capture"]["delta_complete_through_cutoff"] is True
    assert result["retained_capture"]["event_count"] == 100
    assert result["retained_capture"]["post_cutoff_event_count"] == 85
    assert result["screen"]["package_count"] == 52
    assert result["screen"]["strict_displayed_candidate_count"] == 0
    assert result["screen"]["best_package"]["asset"] == "BTC"
    assert result["screen"]["best_package"]["boundary"] == "88000"
    assert result["screen"]["best_package"]["displayed_price_sum_pUSD"] == "1.0"
    assert result["authority"]["network_requests"] == 0
    assert result["adjudication"]["accepted_edge"] is False


def test_registry_routes_delta_to_rank_31_and_exact_terminal() -> None:
    registry = _load(REGISTRY)
    result = _load(RESULT)

    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    rank_31 = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["priority_rank"] == 31
    )
    assert any(
        artifact["result_sha256"] == result["result_sha256"]
        for artifact in rank_31["canonical_artifacts"]
    )
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["family"]
        == "polymarket_BTC_ETH_SOL_September_6_cross_event_range_threshold_delta_2026_08_30"
    )
    assert terminal["canonical_result_sha256"] == result["result_sha256"]
    assert "zero_strict_displayed_sub_floor_candidates" in terminal["reason"]
