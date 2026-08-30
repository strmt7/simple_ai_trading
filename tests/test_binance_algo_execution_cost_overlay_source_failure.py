from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION = ROOT / "docs/model-research/action-value"
CONTRACT = ACTION / (
    "binance-algo-execution-cost-overlay-source-contract-v1-2026-08-30.json"
)
ARTIFACT = ACTION / (
    "binance-algo-execution-cost-overlay-"
    "source-failure-adjudication-v1-2026-08-30.json"
)
RAW_DIR = ROOT / "docs/model-research/binance/raw" / (
    "algo-execution-cost-overlay-v1-2026-08-30"
)
JOURNAL = RAW_DIR / "00-request-journal.jsonl"
RAW = RAW_DIR / "01-current-openapi-schema.raw.yaml"
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


def test_contract_precedes_and_exactly_matches_the_consumed_request() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))
    rows = [json.loads(row) for row in JOURNAL.read_text(encoding="ascii").splitlines()]

    assert _self_hash(contract, "contract_sha256") == contract["contract_sha256"]
    assert [row["phase"] for row in rows] == ["intent", "completed"]
    assert rows[0]["url"] == contract["request"]["url"] == rows[1]["url"]
    frozen = datetime.fromisoformat(contract["frozen_at_utc"].replace("Z", "+00:00"))
    requested = datetime.fromtimestamp(rows[0]["requested_at_ms"] / 1000, tz=frozen.tzinfo)
    assert frozen < requested


def test_http_202_empty_body_is_retained_and_fails_every_source_term() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="ascii"))
    rows = [json.loads(row) for row in JOURNAL.read_text(encoding="ascii").splitlines()]
    completed = rows[1]
    raw = RAW.read_bytes()

    assert completed["status_code"] == 202
    assert completed["response_bytes"] == len(raw) == 0
    assert completed["response_sha256"] == _sha256(raw)
    assert len(contract["required_utf8_phrases"]) == 6
    assert all(term.encode("utf-8") not in raw for term in contract["required_utf8_phrases"])


def test_failure_adjudication_is_fail_closed_source_bound_and_registered() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="ascii"))
    assert _self_hash(artifact, "result_sha256") == artifact["result_sha256"]
    assert artifact["decision"]["accepted_edge"] is False
    assert artifact["decision"]["profitability_proved"] is False
    assert artifact["decision"]["public_forward_saving_floor"] == "0"
    assert artifact["authority"]["credentials_used"] is False
    assert artifact["authority"]["orders_or_trades"] == 0
    assert artifact["authority"]["state_changes"] == 0
    assert artifact["retry_contract"]["terminal_without_trigger"] is True

    outcome = artifact["frozen_public_source_outcome"]
    assert _sha256(CONTRACT.read_bytes()) == outcome["contract_file_sha256"]
    assert _sha256(JOURNAL.read_bytes()) == outcome["journal_sha256"]
    assert _sha256(RAW.read_bytes()) == outcome["raw_sha256"]

    discovery = artifact["discovery_evidence_not_admitted_as_economics"]
    index_path = ROOT / discovery["retained_index_path"]
    index = index_path.read_bytes()
    assert _sha256(index) == discovery["retained_index_sha256"]
    decoded = index.decode("utf-8")
    for endpoint in (
        "/sapi/v1/algo/spot/newOrderTwap",
        "/sapi/v1/algo/futures/newOrderTwap",
        "/sapi/v1/algo/futures/newOrderVp",
        "/sapi/v1/algo/futures/subOrders",
    ):
        assert endpoint in decoded

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert _self_hash(registry, "result_sha256") == registry["result_sha256"]
    family = next(
        row
        for row in registry["prioritized_hypotheses"]
        if row["mechanism"] == "binance_spot_fee_minimization_overlays"
    )
    assert {
        "path": ARTIFACT.relative_to(ROOT).as_posix(),
        "result_sha256": artifact["result_sha256"],
    } in family["canonical_artifacts"]
    assert family["algo_execution_cost_retry_trigger"] == artifact["retry_contract"][
        "retry_trigger"
    ]
    terminal = next(
        row
        for row in registry["terminal_do_not_repeat"]
        if row["canonical_result_sha256"] == artifact["result_sha256"]
    )
    assert terminal["canonical_result_sha256"] == artifact["result_sha256"]
    assert terminal["family"] == (
        "binance_Algo_Trading_current_OpenAPI_schema_source_gate_2026_08_30"
    )
