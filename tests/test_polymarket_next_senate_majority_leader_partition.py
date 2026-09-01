from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "docs/model-research/action-value/polymarket-next-senate-majority-leader-partition-metadata-contract-v1-2026-09-01.json"
)
CAPTURE = (
    ROOT
    / "docs/model-research/action-value/polymarket-next-senate-majority-leader-partition-metadata-capture-result-v1-2026-09-01.json"
)
RESULT = (
    ROOT
    / "docs/model-research/action-value/polymarket-next-senate-majority-leader-partition-metadata-adjudication-v1-2026-09-01.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
AUDIT = (
    ROOT
    / "docs/model-research/action-value/accepted-edge-profitability-durability-audit-v1-2026-08-30.json"
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_generated_hidden_markets_reject_majority_leader_complete_set() -> None:
    contract = _load(CONTRACT)
    capture = _load(CAPTURE)
    result = _load(RESULT)
    registry = _load(REGISTRY)
    audit = _load(AUDIT)

    assert _canonical_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert _canonical_hash(capture, "result_sha256") == capture["result_sha256"]
    assert _canonical_hash(result, "result_sha256") == result["result_sha256"]
    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    assert _canonical_hash(audit, "result_sha256") == audit["result_sha256"]

    source = result["source"]
    assert isinstance(source, dict)
    raw_binding = source["raw"]
    journal_binding = source["journal"]
    assert isinstance(raw_binding, dict)
    assert isinstance(journal_binding, dict)
    raw_path = ROOT / str(raw_binding["path"])
    journal_path = ROOT / str(journal_binding["path"])
    assert _sha256(raw_path) == raw_binding["sha256"]
    assert _sha256(journal_path) == journal_binding["sha256"]

    event = _load(raw_path)
    markets = event["markets"]
    assert isinstance(markets, list)
    assert event["negRisk"] is True
    assert len(markets) == 64
    assert sum(bool(market["active"]) for market in markets) == 11
    assert sum(not bool(market["active"]) for market in markets) == 53
    assert sum(bool(market["closed"]) for market in markets) == 1
    assert (
        sum(
            bool(market["active"])
            and not bool(market["closed"])
            and bool(market["acceptingOrders"])
            for market in markets
        )
        == 10
    )
    assert (
        sum(str(market["groupItemTitle"]).startswith("Person ") for market in markets)
        == 49
    )

    other = next(market for market in markets if market["groupItemTitle"] == "Other")
    assert other["active"] is False
    assert other["acceptingOrders"] is True
    assert other["negRiskOther"] is True
    assert Decimal(str(other["bestAsk"])) == Decimal("1")
    lindsey = next(
        market for market in markets if market["groupItemTitle"] == "Lindsey Graham"
    )
    assert lindsey["active"] is True
    assert lindsey["closed"] is True
    assert lindsey["acceptingOrders"] is False
    assert sum(Decimal(str(market["bestAsk"])) for market in markets) == Decimal(
        "53.931"
    )

    adjudication = result["adjudication"]
    economics = result["economics"]
    authority = result["authority"]
    assert isinstance(adjudication, dict)
    assert isinstance(economics, dict)
    assert isinstance(authority, dict)
    assert adjudication["accepted_edge"] is False
    assert adjudication["book_request_authorized"] is False
    assert adjudication["party_projection_cover_proved"] is False
    assert economics["complete_returned_yes_ask_sum_pUSD_per_share"] == "53.931"
    assert authority["credentials_used"] is False
    assert authority["orders_or_transactions"] == 0
    assert authority["protected_capture_touched"] is False

    terminal = {
        row["family"]: row
        for row in registry["terminal_do_not_repeat"]  # type: ignore[index]
    }
    family = "polymarket_next_Senate_Majority_Leader_generated_inactive_NegRisk_partition_2026_09_01"
    assert terminal[family]["canonical_result_sha256"] == result["result_sha256"]
    binding = audit["source_binding"]
    assert isinstance(binding, dict)
    assert binding["registry_result_sha256"] == registry["result_sha256"]
