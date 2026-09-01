from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/model-research/action-value/polymarket-streaming-service-emmys-partition-metadata-contract-v1-2026-09-01.json"
CAPTURE = ROOT / "docs/model-research/action-value/polymarket-streaming-service-emmys-partition-metadata-capture-result-v1-2026-09-01.json"
RESULT = ROOT / "docs/model-research/action-value/polymarket-streaming-service-emmys-partition-metadata-adjudication-v1-2026-09-01.json"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
AUDIT = ROOT / "docs/model-research/action-value/accepted-edge-profitability-durability-audit-v1-2026-08-30.json"


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


def test_streaming_service_emmys_hidden_other_rejects_complete_set() -> None:
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

    source_binding = result["source_binding"]
    assert isinstance(source_binding, dict)
    raw_binding = source_binding["raw"]
    assert isinstance(raw_binding, dict)
    raw_path = ROOT / str(raw_binding["path"])
    assert _sha256(raw_path) == raw_binding["file_sha256"]
    event = _load(raw_path)
    markets = event["markets"]
    assert isinstance(markets, list)
    assert event["negRisk"] is True
    assert len(markets) == 9

    other = [market for market in markets if market.get("groupItemTitle") == "Other"]
    assert len(other) == 1
    assert other[0]["active"] is False
    assert Decimal(str(other[0]["bestAsk"])) == Decimal("1")
    assert sum(Decimal(str(market["bestAsk"])) for market in markets) == Decimal(
        "1.99"
    )

    decision = result["decision"]
    screen = result["screen"]
    assert isinstance(decision, dict)
    assert isinstance(screen, dict)
    assert decision["accepted_edge"] is False
    assert decision["book_requests_justified"] == 0
    assert screen["strict_complete_subfloor_gate_passed"] is False

    terminal = {
        row["family"]: row for row in registry["terminal_do_not_repeat"]  # type: ignore[index]
    }
    family = "polymarket_streaming_service_most_Emmys_complete_NegRisk_partition_2026_09_01"
    assert terminal[family]["canonical_result_sha256"] == result["result_sha256"]
    source = audit["source_binding"]
    assert isinstance(source, dict)
    assert source["registry_result_sha256"] == registry["result_sha256"]
