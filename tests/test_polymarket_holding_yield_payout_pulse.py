from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path

from tools import reconcile_polymarket_holding_yield_payout_receipt as receipt_tool


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/model-research/polymarket"
PULSE_CONTRACT = BASE / (
    "complete-set-holding-yield-payout-pulse-contract-v9-2026-08-30.json"
)
PULSE_RESULT = BASE / "complete-set-holding-yield-payout-pulse-v9-2026-08-30.json"
RECEIPT_CONTRACT = BASE / (
    "complete-set-holding-yield-payout-receipt-contract-v10-2026-08-30.json"
)
RECEIPT_RESULT = BASE / (
    "complete-set-holding-yield-payout-receipt-v10-2026-08-30.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
PULSE_CONTRACT_HASH = "978945a7bbd5cf9ad171a011909c4f75405739c039a0f6c37e5840e946e6bb42"
PULSE_RESULT_HASH = "47f7d6a9f264cc9740a3c809f245b340cb726e3d715c97a4824832c0a4f2b6ae"
RECEIPT_CONTRACT_HASH = (
    "5e7073f0e6dc72126dca9c48c5181ab9cf8bc974ace12cd7042202aaefce9c4d"
)
RECEIPT_RESULT_HASH = "4e57d6c0216886144fb89f8ae69b11a2eee4db37149ce6c956adecf293b7b927"
REGISTRY_HASH = json.loads(
    (ROOT / "docs/model-research/structural-edge-priority-registry-v1.json").read_text(
        encoding="utf-8"
    )
)["result_sha256"]


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_contracts_results_and_one_use_journals_reconstruct() -> None:
    for path, field, expected in (
        (PULSE_CONTRACT, "contract_sha256", PULSE_CONTRACT_HASH),
        (PULSE_RESULT, "result_sha256", PULSE_RESULT_HASH),
        (RECEIPT_CONTRACT, "contract_sha256", RECEIPT_CONTRACT_HASH),
        (RECEIPT_RESULT, "result_sha256", RECEIPT_RESULT_HASH),
    ):
        payload = _load(path)
        assert payload[field] == expected
        assert _canonical_hash(payload, field) == expected

    for journal_path, contract_hash, result_hash in (
        (
            ROOT / "data/polymarket-holding-yield-payout-pulse-v9/journal.json",
            PULSE_CONTRACT_HASH,
            PULSE_RESULT_HASH,
        ),
        (
            ROOT / "data/polymarket-holding-yield-payout-receipt-v10/journal.json",
            RECEIPT_CONTRACT_HASH,
            RECEIPT_RESULT_HASH,
        ),
    ):
        journal = _load(journal_path)
        assert journal["state"] == "completed"
        assert journal["contract_sha256"] == contract_hash
        assert journal["result_sha256"] == result_hash
        assert journal["request"]["state"] == "received"
        assert journal["request"]["status_code"] == 200


def test_new_payout_and_exact_transfer_reconstruct() -> None:
    pulse = _load(PULSE_RESULT)
    selected = pulse["observation"]["first_new_yield_row"]
    assert pulse["observation"]["new_yield_row_count"] == 1
    assert selected == {
        "amount_pusd": "0.0133",
        "interval_seconds": 86290,
        "timestamp": 1788048820,
        "transaction_hash": (
            "0xc1682058e75cda65542b311964468eaa00c44b27401de609467842aa9045e95b"
        ),
    }

    result = _load(RECEIPT_RESULT)
    contract = _load(RECEIPT_CONTRACT)
    raw_path = ROOT / result["receipt"]["source"]["raw_path"]
    raw = raw_path.read_bytes()
    assert (
        hashlib.sha256(raw).hexdigest()
        == (result["receipt"]["source"]["response_sha256"])
    )
    receipt = json.loads(raw)["result"]
    assert receipt_tool._exact_transfer(receipt, contract)
    assert Decimal(result["payout"]["amount_pusd"]) > 0
    assert result["adjudication"]["current_three_wallet_rate_qualified"] is False
    assert result["adjudication"]["deployment_ready"] is False


def test_registry_keeps_edge_scoped_and_source_bound() -> None:
    registry = _load(REGISTRY)
    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry, "result_sha256") == REGISTRY_HASH
    row = registry["prioritized_hypotheses"][0]
    assert row["canonical_artifacts"][-2:] == [
        {
            "path": PULSE_RESULT.relative_to(ROOT).as_posix(),
            "result_sha256": PULSE_RESULT_HASH,
        },
        {
            "path": RECEIPT_RESULT.relative_to(ROOT).as_posix(),
            "result_sha256": RECEIPT_RESULT_HASH,
        },
    ]
    assert "current_rate_remains_fail_closed_unqualified" in row["current_status"]
    assert "2026_08_31T02_15_30Z" in row["retry_trigger"]
