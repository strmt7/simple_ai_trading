from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time
from typing import Any
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-negrisk-one-no-v2-conversion-contract-v1-2026-08-29.json"
)
RESULT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-negrisk-one-no-v2-conversion-result-v1-2026-08-29.json"
)
DATA_ROOT = ROOT / "data/polymarket-negrisk-one-no-v2-conversions-v1"
RAW_ROOT = DATA_ROOT / "raw"
JOURNAL_PATH = DATA_ROOT / "request-journal.jsonl"
RPC_URL = "https://polygon.drpc.org"


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)
    path.write_text(encoded + "\n", encoding="ascii", newline="\n")


def _post_rpc(name: str, payload: dict[str, Any], raw_path: Path) -> bytes:
    request_bytes = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    requested_at_ms = time.time_ns() // 1_000_000
    request = Request(
        RPC_URL,
        data=request_bytes,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "User-Agent": "simple-ai-trading-public-research/1",
        },
    )
    with urlopen(request, timeout=30) as response:
        response_bytes = response.read()
        status_code = response.status
    completed_at_ms = time.time_ns() // 1_000_000

    raw_path.write_bytes(response_bytes)
    receipt = {
        "completed_at_ms": completed_at_ms,
        "method": "POST",
        "name": name,
        "operation": payload["method"],
        "raw_path": raw_path.relative_to(ROOT).as_posix(),
        "request_body_sha256": _sha256_bytes(request_bytes),
        "requested_at_ms": requested_at_ms,
        "response_bytes": len(response_bytes),
        "response_sha256": _sha256_bytes(response_bytes),
        "status_code": status_code,
        "transport": "HTTPS",
        "url": RPC_URL,
    }
    with JOURNAL_PATH.open("a", encoding="ascii", newline="\n") as journal:
        journal.write(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
        )
    return response_bytes


def _load_rpc_result(raw: bytes, expected_id: int) -> Any:
    payload = json.loads(raw)
    if payload.get("jsonrpc") != "2.0" or payload.get("id") != expected_id:
        raise RuntimeError("unexpected JSON-RPC envelope")
    if "error" in payload:
        raise RuntimeError(f"JSON-RPC error: {payload['error']!r}")
    if "result" not in payload:
        raise RuntimeError("JSON-RPC result is absent")
    return payload["result"]


def main() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    claimed_contract_hash = contract["contract_sha256"]
    if _canonical_hash(contract, "contract_sha256") != claimed_contract_hash:
        raise RuntimeError("contract canonical hash mismatch")
    implementation = ROOT / contract["implementation"]["path"]
    if _sha256_bytes(implementation.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("implementation hash mismatch")
    if RESULT_PATH.exists() or DATA_ROOT.exists():
        raise RuntimeError("one-use output already exists; retry is prohibited")

    RAW_ROOT.mkdir(parents=True)
    block_raw = _post_rpc(
        "polygon-finality-bound-block-number",
        {"id": 1, "jsonrpc": "2.0", "method": "eth_blockNumber", "params": []},
        RAW_ROOT / "block-number.json",
    )
    observed_latest_block = int(_load_rpc_result(block_raw, 1), 16)
    finality_lag = contract["capture"]["finality_lag_blocks"]
    to_block = observed_latest_block - finality_lag
    from_block = int(contract["capture"]["log_filter"]["fromBlock"], 16)
    if to_block < from_block:
        raise RuntimeError("finality-lagged block precedes the frozen start block")

    log_filter = dict(contract["capture"]["log_filter"])
    log_filter["toBlock"] = hex(to_block)
    logs_raw = _post_rpc(
        "polymarket-negrisk-exact-one-no-conversion-logs",
        {
            "id": 2,
            "jsonrpc": "2.0",
            "method": "eth_getLogs",
            "params": [log_filter],
        },
        RAW_ROOT / "logs.json",
    )
    logs = _load_rpc_result(logs_raw, 2)
    if not isinstance(logs, list):
        raise RuntimeError("eth_getLogs result is not an array")

    expected = contract["capture"]["expected"]
    decoded: list[dict[str, Any]] = []
    for row in logs:
        topics = row.get("topics")
        if (
            row.get("address", "").lower() != expected["legacy_adapter_address"]
            or not isinstance(topics, list)
            or len(topics) != 4
            or topics[0].lower() != expected["event_topic0"]
            or topics[1].lower() != expected["v2_adapter_topic1"]
            or topics[2].lower() != expected["market_id_topic2"]
            or int(topics[3], 16) not in expected["one_no_index_sets"]
        ):
            raise RuntimeError("provider returned a log outside the frozen filter")
        if row.get("removed") is True:
            raise RuntimeError("provider returned a removed log")
        decoded.append(
            {
                "amount_shares": str(
                    Decimal(int(row["data"], 16)) / Decimal(1_000_000)
                ),
                "block_number": int(row["blockNumber"], 16),
                "index_set": int(topics[3], 16),
                "log_index": int(row["logIndex"], 16),
                "transaction_hash": row["transactionHash"],
            }
        )

    result: dict[str, Any] = {
        "schema_version": "polymarket-negrisk-one-no-v2-conversion-result-v1",
        "created_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "contract": {
            "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "sha256": claimed_contract_hash,
        },
        "authority": contract["authority"],
        "capture": {
            "from_block": from_block,
            "observed_latest_block": observed_latest_block,
            "finality_lag_blocks": finality_lag,
            "to_block": to_block,
            "http_request_count": 2,
            "exact_one_no_conversion_count": len(decoded),
            "exact_one_no_conversions": decoded,
            "raw_block_number_sha256": _sha256_bytes(block_raw),
            "raw_logs_sha256": _sha256_bytes(logs_raw),
        },
        "adjudication": {
            "resolved_blocker": (
                "successful_current_v2_route_exact_one_NO_conversion_observed"
                if decoded
                else "none"
            ),
            "material_public_retry_trigger_satisfied": bool(decoded),
            "accepted_edge": False,
            "profitability_claim": False,
            "deployment_ready": False,
            "next_action": (
                "retain_this_mechanism_evidence_but_require_owned_or_queue_censored_"
                "maker_input_fill_causal_output_books_exact_user_cost_latency_and_"
                "repeated_after_cost_regimes_before_promotion"
                if decoded
                else "do_not_repeat_this_block_interval_wait_for_a_new_exact_one_NO_"
                "conversion_or_other_material_primary_trigger"
            ),
        },
        "implementation": contract["implementation"],
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    _write_json(RESULT_PATH, result)
    print(json.dumps(result["adjudication"], sort_keys=True))


if __name__ == "__main__":
    main()
