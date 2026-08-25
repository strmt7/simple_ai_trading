from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-spot-maker-rebate-account-evidence-gate-v1.json"
)
EXPECTED_RESULT_SHA256 = (
    "19e6d69f73a1f723680aec51b82709ab912e7437f6e7889e89fc74ff834ac88f"
)


def test_spot_maker_rebate_gate_hash_and_authority_reconstruct() -> None:
    artifact = json.loads(ARTIFACT_PATH.read_bytes())
    expected = artifact.pop("result_sha256")
    canonical = json.dumps(
        artifact,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")

    assert expected == EXPECTED_RESULT_SHA256
    assert hashlib.sha256(canonical).hexdigest() == EXPECTED_RESULT_SHA256
    contract = artifact["current_account_evidence_contract"]
    assert contract["weekly_final_rebates"]["endpoint"] == (
        "GET /sapi/v1/vip/liquidity-programs/weekly-result"
    )
    assert contract["symbol_commission"]["explicit_exclusions"] == [
        "spot_market_maker_rebate_rate",
        "bnb_discount_effect",
    ]
    assert contract["realized_spot_rebate_history"]["endpoint_security"] == (
        "USER_DATA"
    )
    assert artifact["credential_state"]["key_present"] is False
    assert artifact["credential_state"]["secret_present"] is False
    assert artifact["authority"]["signed_requests_sent"] is False
    assert artifact["authority"]["accepted_edge"] is False
    assert artifact["authority"]["profitability_claim"] is False
