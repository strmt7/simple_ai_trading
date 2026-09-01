import hashlib
import json
from pathlib import Path

from tools.adjudicate_polymarket_gemini_release_projection import adjudicate


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "docs/model-research/action-value/polymarket-gemini-release-projection-contract-v1-2026-09-01.json"
)
RESULT = (
    ROOT
    / "docs/model-research/action-value/polymarket-gemini-release-projection-adjudication-v1-2026-09-01.json"
)
CATALOG_RESULT = (
    ROOT
    / "docs/model-research/action-value/polymarket-sep30-et-negrisk-complete-set-catalog-result-v1-2026-09-01.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _self_hash(payload: dict, field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_retained_projection_is_reproducibly_rejected_before_books() -> None:
    contract = _load(CONTRACT)
    retained = _load(RESULT)
    reproduced = adjudicate(contract, CONTRACT.resolve())
    assert _self_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _self_hash(retained, "result_sha256") == retained["result_sha256"]
    assert reproduced["economics"] == retained["economics"]
    assert retained["scope"] == {
        "deadline_no_leg_count": 1,
        "promotion_eligible": False,
        "quantity_shares_each_leg": "5",
        "release_leg_count": 30,
        "total_leg_count": 31,
    }
    assert retained["payoff_identity"]["source_proved_floor"] is False
    assert retained["economics"]["metadata_cost_pUSD_per_share"] == "1.264"
    assert retained["economics"]["optimistic_metadata_gross_headroom_pUSD"] == "-1.320"
    assert retained["economics"]["after_fee_one_tick_profit_floor_pUSD"] == "-1.64295"
    assert retained["adjudication"]["book_request_justified"] is False


def test_incomplete_catalog_and_registry_remain_fail_closed() -> None:
    catalog = _load(CATALOG_RESULT)
    registry = _load(REGISTRY)
    assert catalog["capture"]["next_cursor_present"] is True
    assert catalog["capture"]["population_complete_under_frozen_filter"] is False
    assert catalog["screen"]["fixed_negrisk_event_count"] == 3
    assert catalog["screen"]["candidate_count_strictly_below_payout_floor"] == 0
    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    assert registry["accepted_edge_count"] == 31
    assert len(registry["prioritized_hypotheses"]) == 48
    assert len(registry["terminal_do_not_repeat"]) == 156
