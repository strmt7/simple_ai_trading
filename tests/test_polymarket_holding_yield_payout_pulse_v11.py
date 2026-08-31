from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/model-research/polymarket"
PULSE_CONTRACT = BASE / (
    "complete-set-holding-yield-payout-pulse-contract-v11-2026-08-31.json"
)
PULSE_RESULT = BASE / "complete-set-holding-yield-payout-pulse-v11-2026-08-31.json"
RECEIPT_CONTRACT = BASE / (
    "complete-set-holding-yield-payout-receipt-contract-v12-2026-08-31.json"
)
RECEIPT_RESULT = BASE / (
    "complete-set-holding-yield-payout-receipt-v12-2026-08-31.json"
)
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
PULSE_CONTRACT_HASH = (
    "267040fa9fcdca77ff20e124203de4b128e4f1e9e53a819cd6dd98416c3cc323"
)
PULSE_RESULT_HASH = (
    "0ed8af56f9488757c9351c01b711ece0fbe579ea408cdfc28873863a91555cc9"
)
RECEIPT_CONTRACT_HASH = (
    "f7682f246682bcbff842c6e66776b58990446848a69040fcd1248867e405bb10"
)
RECEIPT_RESULT_HASH = (
    "5befa8d4ed1d93459632537a47a534bf1650d3d36b7ee1967ed1bce012b58309"
)
RAW_HASH = "63d2cbc9b33baee0a9773ecce72d13a38f5654ed08e6863e697d17dda01732c0"
RECEIPT_RAW_HASH = (
    "a102c37099fd5a04bb28ab3705088579d4b053c2bd42a48caf69921448faaebd"
)


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="ascii"))


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_v11_contract_result_raw_and_journal_reconstruct() -> None:
    for path, field, expected in (
        (PULSE_CONTRACT, "contract_sha256", PULSE_CONTRACT_HASH),
        (PULSE_RESULT, "result_sha256", PULSE_RESULT_HASH),
        (RECEIPT_CONTRACT, "contract_sha256", RECEIPT_CONTRACT_HASH),
    ):
        payload = _load(path)
        assert payload[field] == expected
        assert _canonical_hash(payload, field) == expected

    raw = ROOT / "data/polymarket-holding-yield-payout-pulse-v11/raw/btc-activity.raw"
    assert len(raw.read_bytes()) == 8038
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == RAW_HASH
    journal = _load(
        ROOT / "data/polymarket-holding-yield-payout-pulse-v11/journal.json"
    )
    assert journal["state"] == "completed"
    assert journal["contract_sha256"] == PULSE_CONTRACT_HASH
    assert journal["result_sha256"] == PULSE_RESULT_HASH
    assert journal["request"]["state"] == "received"
    assert journal["request"]["status_code"] == 200
    assert journal["request"]["response_sha256"] == RAW_HASH


def test_v11_selected_exactly_one_new_payout_and_remains_fail_closed() -> None:
    pulse = _load(PULSE_RESULT)
    assert pulse["observation"]["new_yield_row_count"] == 1
    assert pulse["observation"]["first_new_yield_row"] == {
        "amount_pusd": "0.0133",
        "interval_seconds": 86190,
        "timestamp": 1788135010,
        "transaction_hash": (
            "0xfb4022e56f217004e89c9a3be838222da491b73abcc0633c1eab55457cc61d5a"
        ),
    }
    assert pulse["adjudication"]["current_rate_qualified"] is False
    assert pulse["adjudication"]["deployment_ready"] is False
    assert pulse["adjudication"]["public_profit_floor_for_new_capital_pusd"] == "0"
    assert pulse["authority"]["credentials_used"] is False
    assert pulse["authority"]["protected_capture_assets_touched"] is False


def test_v12_is_frozen_to_v11_and_exact_receipt_reconstructs() -> None:
    contract = _load(RECEIPT_CONTRACT)
    assert contract["parent_result"] == {
        "path": PULSE_RESULT.relative_to(ROOT).as_posix(),
        "result_sha256": PULSE_RESULT_HASH,
    }
    assert contract["request"]["exact_total_requests"] == 1
    assert contract["request"]["retry_permitted"] is False
    assert contract["request"]["alias_or_parameter_change_permitted"] is False
    assert contract["transaction_hash"] == (
        "0xfb4022e56f217004e89c9a3be838222da491b73abcc0633c1eab55457cc61d5a"
    )
    implementation = ROOT / contract["implementation"]["path"]
    assert hashlib.sha256(implementation.read_bytes()).hexdigest() == (
        contract["implementation"]["sha256"]
    )
    dependency = contract["implementation"]["base_dependency"]
    assert hashlib.sha256((ROOT / dependency["path"]).read_bytes()).hexdigest() == (
        dependency["sha256"]
    )
    result = _load(RECEIPT_RESULT)
    assert result["result_sha256"] == RECEIPT_RESULT_HASH
    assert _canonical_hash(result, "result_sha256") == RECEIPT_RESULT_HASH
    assert result["receipt"]["successful_exact_distributor_pusd_transfer"] is True
    assert result["receipt"]["block_number"] == 92953322
    assert result["adjudication"]["current_three_wallet_rate_qualified"] is False
    assert result["adjudication"]["deployment_ready"] is False
    raw_path = ROOT / contract["outputs"]["raw_path"]
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == RECEIPT_RAW_HASH
    journal = _load(ROOT / contract["outputs"]["journal_path"])
    assert journal["state"] == "completed"
    assert journal["result_sha256"] == RECEIPT_RESULT_HASH
    assert journal["request"]["response_sha256"] == RECEIPT_RAW_HASH


def test_registry_source_binds_consumed_v11_and_v12() -> None:
    registry = _load(REGISTRY)
    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    row = registry["prioritized_hypotheses"][0]
    for path, result_hash in (
        (PULSE_CONTRACT, PULSE_CONTRACT_HASH),
        (PULSE_RESULT, PULSE_RESULT_HASH),
        (RECEIPT_CONTRACT, RECEIPT_CONTRACT_HASH),
        (RECEIPT_RESULT, RECEIPT_RESULT_HASH),
    ):
        assert {
            "path": path.relative_to(ROOT).as_posix(),
            "result_sha256": result_hash,
        } in row["canonical_artifacts"]
    assert "consumed_v11" in row["latest_pulse_delta_status"]
    assert "consumed_exact_successful_transfer_reconciled" in row["receipt_v12_status"]
    assert "current_rate_remains_fail_closed_unqualified" in row["current_status"]
