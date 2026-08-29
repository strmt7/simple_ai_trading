import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STOCK_CONTRACT = ROOT / (
    "docs/model-research/action-value/"
    "binance-stocks-fee-extension-source-contract-v1.json"
)
STOCK_FAILURE = ROOT / (
    "docs/model-research/action-value/"
    "binance-stocks-fee-extension-source-failure-v1-2026-08-29.json"
)
STOCK_JOURNAL = ROOT / (
    "data/binance-stocks-fee-extension-source-v1/request-journal.jsonl"
)
SIDECAR_FAILURE = ROOT / (
    "docs/model-research/polymarket/"
    "round-021-binance-sidecar-terminal-failure-v2-2026-08-29.json"
)


def _canonical_hash(document: dict, field: str) -> str:
    body = dict(document)
    claimed = body.pop(field)
    actual = hashlib.sha256(
        json.dumps(
            body,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    assert actual == claimed
    return actual


def test_stock_fee_extension_capture_is_consumed_and_fail_closed() -> None:
    contract = json.loads(STOCK_CONTRACT.read_text(encoding="utf-8"))
    failure = json.loads(STOCK_FAILURE.read_text(encoding="utf-8"))
    _canonical_hash(contract, "contract_sha256")
    _canonical_hash(failure, "result_sha256")
    assert failure["contract"]["contract_sha256"] == contract["contract_sha256"]

    journal_bytes = STOCK_JOURNAL.read_bytes()
    assert len(journal_bytes) == failure["failure"]["journal_bytes"] == 873
    assert hashlib.sha256(journal_bytes).hexdigest() == (
        failure["failure"]["journal_sha256"]
    )
    entries = [json.loads(line) for line in journal_bytes.splitlines()]
    assert [entry["phase"] for entry in entries] == ["intent", "completed"]
    assert entries[1]["curl_exit_code"] == 23
    assert entries[1]["raw_retained"] is False
    assert not (ROOT / failure["failure"]["raw_path"]).exists()
    assert failure["adjudication"]["accepted_edge_count_change"] == 0
    assert failure["adjudication"]["existing_edge_duration_updated"] is False
    assert failure["retry_policy"]["exact_CMS_request_must_not_be_retried"] is True


def test_round21_sidecar_is_terminally_failed_without_model_admission() -> None:
    failure = json.loads(SIDECAR_FAILURE.read_text(encoding="utf-8"))
    _canonical_hash(failure, "artifact_sha256")
    audit = failure["terminal_process_and_storage_audit"]
    receipt = failure["segment_terminal_receipt"]
    adjudication = failure["adjudication"]
    assert audit["live_previous_process_ids"] == []
    assert audit["campaign_lock_exclusive_read_succeeded"] is True
    assert audit["database_opened"] is False
    assert audit["database_payload_or_outcomes_accessed"] is False
    assert audit["wal_exists"] is False
    assert receipt["segment_count"] == 17
    assert receipt["interrupted_segment_count"] == 16
    assert receipt["failed_segment_count"] == 1
    assert receipt["complete_or_degraded_eligible_segment_count"] == 0
    assert len(receipt["ordered_segment_artifact_sha256"]) == 17
    assert adjudication["campaign_source_continuity_passed"] is False
    assert adjudication["model_data_eligible"] is False
    assert adjudication["profitability_claim"] is False
    assert adjudication["retry_same_campaign"] is False
