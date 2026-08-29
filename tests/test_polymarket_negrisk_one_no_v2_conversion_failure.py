from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACTION_VALUE = ROOT / "docs/model-research/action-value"
CONTRACT = ACTION_VALUE / (
    "polymarket-negrisk-one-no-v2-conversion-contract-v1-2026-08-29.json"
)
ADJUDICATION = ACTION_VALUE / (
    "polymarket-negrisk-one-no-v2-conversion-failure-adjudication-v1-2026-08-29.json"
)
RUNNER = ROOT / "tools/capture_polymarket_negrisk_one_no_v2_conversions.py"
ADJUDICATOR = ROOT / (
    "tools/adjudicate_polymarket_negrisk_one_no_v2_conversion_failure.py"
)
DATA_ROOT = ROOT / "data/polymarket-negrisk-one-no-v2-conversions-v1"
RAW_BLOCK = DATA_ROOT / "raw/block-number.json"
JOURNAL = DATA_ROOT / "request-journal.jsonl"
REGISTRY = ROOT / "docs/model-research/structural-edge-priority-registry-v1.json"
CONTRACT_HASH = "7d8c556225aa5905ca302b7c9409768c8b546a0dc41597ad083c573ce41fd640"
ADJUDICATION_HASH = (
    "7c976cd84795718b63463ea4e32ebeddaf51e807fc5ebe9aa8cb49b476541e19"
)
REGISTRY_HASH = "2baf1b76070e0ef9081f9eb5fba41f3977b5fd1aa74759ed85034947e9ad1c5a"


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


def test_frozen_one_no_contract_is_exact_public_and_action_free() -> None:
    contract = _load(CONTRACT)

    assert contract["contract_sha256"] == CONTRACT_HASH
    assert _canonical_hash(contract, "contract_sha256") == CONTRACT_HASH
    assert contract["authority"]["credentials_used"] is False
    assert contract["authority"]["orders_or_conversions_submitted"] == 0
    assert contract["capture"]["http_request_count"] == 2
    assert contract["capture"]["retry_count"] == 0
    assert contract["capture"]["expected"]["one_no_index_sets"] == [1, 2, 4]
    assert contract["capture"]["log_filter"]["fromBlock"] == "0x5869cf7"
    assert (
        contract["implementation"]["sha256"]
        == hashlib.sha256(RUNNER.read_bytes()).hexdigest()
    )


def test_http_400_is_preserved_as_unknown_and_not_conversion_absence() -> None:
    adjudication = _load(ADJUDICATION)

    assert adjudication["result_sha256"] == ADJUDICATION_HASH
    assert _canonical_hash(adjudication, "result_sha256") == ADJUDICATION_HASH
    assert adjudication["contract"]["sha256"] == CONTRACT_HASH
    assert adjudication["failure_diagnosis"]["terminal_error"] == (
        "HTTP_400_on_the_single_frozen_eth_getLogs_request"
    )
    assert adjudication["failure_diagnosis"]["exact_provider_reason_known"] is False
    assert adjudication["retained_evidence"]["second_response_body_retained"] is False
    assert adjudication["retained_evidence"]["second_request_receipt_retained"] is False
    assert adjudication["adjudication"] == {
        **adjudication["adjudication"],
        "absence_of_conversion_proved": False,
        "accepted_edge": False,
        "exact_one_no_conversion_observed": False,
        "material_public_retry_trigger_satisfied": False,
    }
    assert (
        adjudication["implementation"]["sha256"]
        == hashlib.sha256(ADJUDICATOR.read_bytes()).hexdigest()
    )


def test_only_successful_block_number_receipt_reconstructs() -> None:
    adjudication = _load(ADJUDICATION)
    evidence = adjudication["retained_evidence"]
    receipts = [json.loads(line) for line in JOURNAL.read_text().splitlines()]

    assert len(receipts) == 1
    assert receipts[0]["status_code"] == 200
    assert receipts[0]["response_sha256"] == hashlib.sha256(
        RAW_BLOCK.read_bytes()
    ).hexdigest()
    assert evidence["completed_request_count"] == 1
    assert evidence["observed_latest_block"] == 92874137
    assert evidence["derived_finality_lagged_to_block"] == 92873881
    assert evidence["derived_block_span"] == 165795


def test_registry_binds_failure_without_reopening_or_promotion() -> None:
    registry = _load(REGISTRY)

    assert registry["result_sha256"] == REGISTRY_HASH
    assert _canonical_hash(registry, "result_sha256") == REGISTRY_HASH
    assert registry["accepted_edge_count"] == 19
    assert [row["priority_rank"] for row in registry["prioritized_hypotheses"]] == list(
        range(1, 43)
    )
    row = registry["prioritized_hypotheses"][1]
    assert row["priority_rank"] == 2
    assert row["canonical_artifacts"][-2:] == [
        {
            "path": CONTRACT.relative_to(ROOT).as_posix(),
            "result_sha256": CONTRACT_HASH,
        },
        {
            "path": ADJUDICATION.relative_to(ROOT).as_posix(),
            "result_sha256": ADJUDICATION_HASH,
        },
    ]
    assert "cannot_be_retried_this_interval" in row["current_status"]
    assert "failed_exact_one_NO_log_query" in row["next_action"]
