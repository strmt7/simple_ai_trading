from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/model-research/action-value"
CONTRACT = BASE / (
    "binance-prediction-yield-bearing-source-contract-v1-2026-08-30.json"
)
ARTIFACT = BASE / (
    "binance-prediction-yield-bearing-source-failure-"
    "adjudication-v1-2026-08-30.json"
)
RAW_DIR = ROOT / (
    "docs/model-research/binance/raw/"
    "prediction-yield-bearing-schema-v1-2026-08-30"
)
JOURNAL = RAW_DIR / "request-journal.json"
RAW = RAW_DIR / "prediction-trading-schema.raw.yaml"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _self_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    return _sha256(_canonical(body))


def test_consumed_schema_capture_is_empty_and_terminal() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))
    journal = json.loads(JOURNAL.read_text(encoding="ascii"))
    request = journal["requests"][0]

    assert _self_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert journal["contract_sha256"] == contract["contract_sha256"]
    assert journal["state"] == "failed"
    assert len(journal["requests"]) == 1
    assert request["status_code"] == 202
    assert request["response_bytes"] == len(RAW.read_bytes()) == 0
    assert request["response_sha256"] == _sha256(b"")
    assert "required schema text missing" in journal["error"]


def test_yield_bearing_lead_remains_fail_closed_in_existing_prediction_family() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="ascii"))
    assert _self_hash(artifact, "result_sha256") == artifact["result_sha256"]
    assert artifact["adjudication"]["accepted_edge"] is False
    assert artifact["economic_boundary"]["public_forward_profit_floor"] == "0"
    assert artifact["authority"]["prediction_market_requests"] == 0
    assert artifact["authority"]["credentials_used"] is False
    assert artifact["retained_request"]["response_sha256"] == _sha256(b"")

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    family = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["priority_rank"] == 18
    )
    assert "prediction_collateral_yield_overlay" in family["mechanism"]
    assert {
        "path": ARTIFACT.relative_to(ROOT).as_posix(),
        "result_sha256": artifact["result_sha256"],
    } in family["canonical_artifacts"]
    assert "designated_credentials" in family["retry_trigger"]
