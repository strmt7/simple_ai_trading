from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-negrisk-one-no-v2-conversion-contract-v1-2026-08-29.json"
)
RUNNER_PATH = ROOT / "tools/capture_polymarket_negrisk_one_no_v2_conversions.py"
BLOCK_RAW_PATH = ROOT / (
    "data/polymarket-negrisk-one-no-v2-conversions-v1/raw/block-number.json"
)
LOGS_RAW_PATH = ROOT / "data/polymarket-negrisk-one-no-v2-conversions-v1/raw/logs.json"
JOURNAL_PATH = ROOT / (
    "data/polymarket-negrisk-one-no-v2-conversions-v1/request-journal.jsonl"
)
ORIGINAL_RESULT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-negrisk-one-no-v2-conversion-result-v1-2026-08-29.json"
)
OUTPUT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-negrisk-one-no-v2-conversion-failure-adjudication-v1-2026-08-29.json"
)


def _canonical_hash(payload: dict[str, Any], field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    encoded = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    if OUTPUT_PATH.exists():
        raise RuntimeError("failure adjudication already exists")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    contract_hash = contract["contract_sha256"]
    if _canonical_hash(contract, "contract_sha256") != contract_hash:
        raise RuntimeError("contract hash mismatch")
    runner_hash = _sha256_bytes(RUNNER_PATH.read_bytes())
    if runner_hash != contract["implementation"]["sha256"]:
        raise RuntimeError("frozen runner hash mismatch")
    if LOGS_RAW_PATH.exists() or ORIGINAL_RESULT_PATH.exists():
        raise RuntimeError("unexpected logs raw or original result exists")

    block_raw = BLOCK_RAW_PATH.read_bytes()
    block_payload = json.loads(block_raw)
    observed_latest = int(block_payload["result"], 16)
    to_block = observed_latest - contract["capture"]["finality_lag_blocks"]
    from_block = int(contract["capture"]["log_filter"]["fromBlock"], 16)

    journal_lines = JOURNAL_PATH.read_text(encoding="ascii").splitlines()
    if len(journal_lines) != 1:
        raise RuntimeError("expected exactly one completed request receipt")
    receipt = json.loads(journal_lines[0])
    if receipt["response_sha256"] != _sha256_bytes(block_raw):
        raise RuntimeError("block raw response hash mismatch")

    log_filter = dict(contract["capture"]["log_filter"])
    log_filter["toBlock"] = hex(to_block)
    second_request = {
        "id": 2,
        "jsonrpc": "2.0",
        "method": "eth_getLogs",
        "params": [log_filter],
    }
    second_request_bytes = json.dumps(
        second_request,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")

    result: dict[str, Any] = {
        "schema_version": (
            "polymarket-negrisk-one-no-v2-conversion-failure-adjudication-v1"
        ),
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "sha256": contract_hash,
        },
        "frozen_runner": {
            "path": RUNNER_PATH.relative_to(ROOT).as_posix(),
            "sha256": runner_hash,
            "preserved_unchanged": True,
        },
        "retained_evidence": {
            "completed_request_count": 1,
            "block_number_raw_path": BLOCK_RAW_PATH.relative_to(ROOT).as_posix(),
            "block_number_raw_sha256": _sha256_bytes(block_raw),
            "journal_path": JOURNAL_PATH.relative_to(ROOT).as_posix(),
            "journal_sha256": _sha256_bytes(JOURNAL_PATH.read_bytes()),
            "observed_latest_block": observed_latest,
            "derived_finality_lagged_to_block": to_block,
            "frozen_from_block": from_block,
            "derived_block_span": to_block - from_block + 1,
            "reconstructed_second_request_body_sha256": _sha256_bytes(
                second_request_bytes
            ),
            "second_response_status_observed_by_process": 400,
            "second_response_body_retained": False,
            "second_request_receipt_retained": False,
        },
        "failure_diagnosis": {
            "terminal_error": "HTTP_400_on_the_single_frozen_eth_getLogs_request",
            "exact_provider_reason_known": False,
            "reason": (
                "urllib raised HTTPError before the runner wrote the response body or "
                "journal receipt; without that body, attributing the error to block "
                "range, topic syntax, provider policy, or another cause would be an "
                "assumption"
            ),
            "workflow_defect": (
                "the frozen runner retained only successful HTTP responses despite "
                "the contract requiring raw-response retention on failure"
            ),
            "adaptive_retry_or_provider_substitution_permitted": False,
        },
        "adjudication": {
            "status": "terminal_failed_closed_without_hypothesis_observation",
            "exact_one_no_conversion_observed": False,
            "absence_of_conversion_proved": False,
            "material_public_retry_trigger_satisfied": False,
            "accepted_edge": False,
            "profitability_claim": False,
            "deployment_ready": False,
            "next_action": (
                "do_not_retry_this_interval; advance_the_next_ranked_satisfied_"
                "trigger_and_require_future_one_use_HTTP_runners_to_prejournal_"
                "request_identity_and_retain_HTTP_error_bodies"
            ),
        },
        "authority": contract["authority"],
        "implementation": {
            "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": _sha256_bytes(Path(__file__).read_bytes()),
        },
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    OUTPUT_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(json.dumps(result["adjudication"], sort_keys=True))


if __name__ == "__main__":
    main()
