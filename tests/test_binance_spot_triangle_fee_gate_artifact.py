from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-spot-triangle-account-fee-gate-v1.json"
)
EXPECTED_RESULT_SHA256 = (
    "304a78180be3375a3453384ad71948c24e52ffeba2f1482cb97711e59aa4a688"
)


def _artifact() -> dict[str, object]:
    return json.loads(ARTIFACT_PATH.read_bytes())


def test_spot_triangle_fee_gate_hash_and_snapshot_reconstruct() -> None:
    artifact = _artifact()
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
    snapshot = artifact["snapshot"]
    source = ROOT / snapshot["file_path"]
    assert len(source.read_bytes()) == snapshot["file_bytes"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == snapshot["file_sha256"]
    source_payload = json.loads(source.read_bytes())
    assert source_payload["result_sha256"] == snapshot["result_sha256"]


def test_spot_triangle_fee_gate_is_exact_and_non_authoritative() -> None:
    artifact = _artifact()
    path = artifact["snapshot"]["best_zero_fee_path"]
    gross = Decimal(path["gross_multiplier"])
    fee_bips = Decimal(path["break_even_equal_fee_bips_per_leg"])
    after_fee = gross * (Decimal("1") - fee_bips / Decimal("10000")) ** 3

    assert abs(after_fee - Decimal("1")) < Decimal("1e-24")
    assert [
        row["symbol"]
        for row in artifact["exact_fee_evidence_contract"]["required_legs"]
    ] == [
        "BTCUSDC",
        "BTCUSDT",
        "USDCUSDT",
    ]
    assert artifact["credential_state"]["key_present"] is False
    assert artifact["credential_state"]["secret_present"] is False
    assert artifact["authority"]["signed_requests_sent"] is False
    assert artifact["authority"]["accepted_edge"] is False
    assert artifact["authority"]["profitability_claim"] is False
