from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-quarterly-delivery-basis-audit-contract-v1.json"
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def test_quarterly_delivery_basis_contract_is_hash_bound_and_nonadaptive() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))
    expected_hash = contract.pop("result_sha256")

    assert hashlib.sha256(_canonical_json(contract).encode("ascii")).hexdigest() == (
        expected_hash
    )
    assert contract["historical_source"]["pairs"] == ["BTCUSDT", "ETHUSDT"]
    assert contract["historical_source"]["completed_deliveries_per_pair"] == 8
    assert contract["request_contract"]["maximum_public_requests"] == 18
    assert contract["request_contract"]["retry_or_replacement_permitted"] is False
    assert contract["request_contract"]["terminal_failure_receipt_required"] is True
    assert contract["spot_exit_proxies"]["execution_or_depth_claim_permitted"] is False
    assert contract["authority"]["accepted_edge"] is False
    assert contract["fee_sources"]["futures"]["applicability"].startswith(
        "informational_only"
    )
