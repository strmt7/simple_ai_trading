from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / (
    "docs/model-research/action-value/"
    "binance-rpi-maker-hedge-source-contract-v1-2026-08-30.json"
)
ADJUDICATION = ROOT / (
    "docs/model-research/action-value/"
    "binance-rpi-maker-hedge-source-failure-adjudication-v1-2026-08-30.json"
)
JOURNAL = ROOT / "data/binance-rpi-maker-hedge-source-gate-v1/journal.json"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
CONTRACT_HASH = "bf2523afeca36bf38d20aa2a61e04a8984bca5fc3e221ff854090ad748f1c0e8"
RESULT_HASH = "82245f341e23ab2e8c8e9e3bd4d47805e88aebc3b39f6ac1a6360067491be7ef"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_contract_and_failure_adjudication_reconstruct() -> None:
    contract = _load(CONTRACT)
    adjudication = _load(ADJUDICATION)

    assert contract["contract_sha256"] == CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == CONTRACT_HASH
    assert adjudication["result_sha256"] == RESULT_HASH
    assert _canonical_hash(adjudication, "result_sha256") == RESULT_HASH
    implementation = contract["implementation"]
    assert (
        hashlib.sha256((ROOT / implementation["path"]).read_bytes()).hexdigest()
        == implementation["sha256"]
    )


def test_consumed_202_empty_response_stopped_before_other_requests() -> None:
    adjudication = _load(ADJUDICATION)
    journal = _load(JOURNAL)
    first = adjudication["consumed_evidence"]["first_request"]

    assert journal["state"] == "failed"
    assert len(journal["requests"]) == 1
    assert journal["requests"][0]["status_code"] == 202
    assert journal["requests"][0]["response_bytes"] == 0
    raw = (ROOT / first["raw_path"]).read_bytes()
    assert raw == b""
    assert hashlib.sha256(raw).hexdigest() == first["raw_sha256"]
    decision = adjudication["adjudication"]
    assert decision["accepted_edge"] is False
    assert decision["rpi_depth_requests_made"] == 0
    assert decision["ordinary_book_requests_made"] == 0
    assert decision["commission_requests_made"] == 0


def test_registry_self_hash_and_rpi_family_update_reconstruct() -> None:
    registry = _load(REGISTRY)

    assert _canonical_hash(registry, "result_sha256") == registry["result_sha256"]
    assert len(registry["prioritized_hypotheses"]) == 44
    row = next(
        item
        for item in registry["prioritized_hypotheses"]
        if item["mechanism"] == "binance_spot_fee_minimization_overlays"
    )
    assert {
        "path": ADJUDICATION.relative_to(ROOT).as_posix(),
        "result_sha256": RESULT_HASH,
    } in row["canonical_artifacts"]
    terminal = next(
        item
        for item in registry["terminal_do_not_repeat"]
        if item["family"]
        == "binance_RPI_maker_hedge_dynamic_documentation_source_capture_2026_08_30"
    )
    assert terminal["canonical_result_sha256"] == RESULT_HASH
