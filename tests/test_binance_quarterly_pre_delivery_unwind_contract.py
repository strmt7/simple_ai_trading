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
    / "binance-quarterly-pre-delivery-unwind-contract-v1.json"
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def test_pre_delivery_unwind_contract_is_hash_bound_fixed_and_nonpromotional() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))
    expected_hash = contract.pop("result_sha256")

    assert hashlib.sha256(_canonical_json(contract).encode("ascii")).hexdigest() == (
        expected_hash
    )
    assert len(contract["frozen_contracts"]) == 16
    assert [row["pair"] for row in contract["frozen_contracts"]] == [
        *("BTCUSDT" for _ in range(8)),
        *("ETHUSDT" for _ in range(8)),
    ]
    assert {
        row["scheduled_delivery_ms"] % 86_400_000
        for row in contract["frozen_contracts"]
    } == {28_800_000}
    assert len({row["symbol"] for row in contract["frozen_contracts"]}) == 16
    assert contract["metric_contract"]["primary_horizon_minutes"] == 10
    assert contract["metric_contract"]["diagnostics_may_promote"] is False
    assert contract["request_contract"]["maximum_public_requests"] == 32
    assert contract["request_contract"]["retry_or_replacement_permitted"] is False
    assert contract["decision"]["edge_acceptance_possible"] is False
    assert contract["authority"] == {
        "accepted_edge": False,
        "account_credentials_permitted": False,
        "live_trading_authority": False,
        "orders_permitted": False,
        "paper_trading_authority": False,
        "profitability_claim": False,
    }

    for path_field, raw_hash_field, result_hash_field in (
        (
            "current_carry_path",
            "current_carry_raw_file_sha256",
            "current_carry_result_sha256",
        ),
        (
            "timestamp_adjudication_path",
            "timestamp_adjudication_raw_file_sha256",
            "timestamp_adjudication_result_sha256",
        ),
        (
            "timing_semantics_path",
            "timing_semantics_raw_file_sha256",
            "timing_semantics_result_sha256",
        ),
    ):
        source = contract["source_binding"]
        path = ROOT / source[path_field]
        payload = json.loads(path.read_text(encoding="ascii"))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source[raw_hash_field]
        assert payload["result_sha256"] == source[result_hash_field]
