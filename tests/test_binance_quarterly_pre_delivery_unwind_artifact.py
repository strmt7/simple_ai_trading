from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-quarterly-pre-delivery-unwind-audit-v1-2026-08-25.json"
)
ADJUDICATION_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-quarterly-pre-delivery-unwind-terminal-adjudication-v1.json"
)
EXPECTED_AUDIT_FILE_SHA256 = (
    "d6f45c2e37f2c2606ff00a321eaae6c5b5a1c5b1ca04ded49fbb947dd1c56444"
)
EXPECTED_AUDIT_RESULT_SHA256 = (
    "07556c4c128fdde32b8bc3ade55134e25eedec157715585aac9e561d87ac9e5a"
)
EXPECTED_ADJUDICATION_RESULT_SHA256 = (
    "e45df8dbffdb8e8e09a542ad3cf2f2f7fe855a775c10f9c07cfa30b290505521"
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _result_hash(payload: dict[str, object]) -> str:
    body = dict(payload)
    body.pop("result_sha256")
    return hashlib.sha256(_canonical_json(body).encode("ascii")).hexdigest()


def test_terminal_audit_and_adjudication_reconstruct_from_source() -> None:
    audit_bytes = AUDIT_PATH.read_bytes()
    audit = json.loads(audit_bytes)
    adjudication = json.loads(ADJUDICATION_PATH.read_bytes())

    assert hashlib.sha256(audit_bytes).hexdigest() == EXPECTED_AUDIT_FILE_SHA256
    assert audit["result_sha256"] == EXPECTED_AUDIT_RESULT_SHA256
    assert _result_hash(audit) == EXPECTED_AUDIT_RESULT_SHA256
    assert adjudication["result_sha256"] == EXPECTED_ADJUDICATION_RESULT_SHA256
    assert _result_hash(adjudication) == EXPECTED_ADJUDICATION_RESULT_SHA256
    assert adjudication["evidence"]["audit_raw_file_sha256"] == (
        EXPECTED_AUDIT_FILE_SHA256
    )
    assert adjudication["evidence"]["audit_result_sha256"] == (
        EXPECTED_AUDIT_RESULT_SHA256
    )


def test_terminal_audit_proves_source_shape_failure_without_edge_authority() -> None:
    audit = json.loads(AUDIT_PATH.read_bytes())
    adjudication = json.loads(ADJUDICATION_PATH.read_bytes())
    ledger = audit["source_contract"]["request_ledger"]
    scheduled_delivery_ms = 1_727_424_000_000

    assert audit["status"] == "terminal_failure_without_retry"
    assert audit["error"] == {
        "type": "ValueError",
        "message": "futures requires exactly 60 bars",
    }
    assert len(ledger) == 2
    futures_rows = ledger[0]["decoded_payload"]
    spot_rows = ledger[1]["decoded_payload"]
    assert ledger[0]["status_code"] == ledger[1]["status_code"] == 200
    assert len(futures_rows) == len(spot_rows) == 70
    assert [row[0] for row in futures_rows] == list(
        range(1_727_420_400_000, 1_727_424_600_000, 60_000)
    )

    pre_schedule = [row for row in futures_rows if row[0] < scheduled_delivery_ms]
    post_schedule = [row for row in futures_rows if row[0] >= scheduled_delivery_ms]
    assert len(pre_schedule) == 60
    assert len(post_schedule) == 10
    assert pre_schedule[-1][0] == scheduled_delivery_ms - 60_000
    assert pre_schedule[-1][5] == "0.097"
    assert pre_schedule[-1][8] == 4
    assert all(row[1:5] == ["65426.0"] * 4 for row in post_schedule)
    assert all(row[5] == "0" and row[8] == 0 for row in post_schedule)

    evidence = adjudication["evidence"]["first_futures_response"]
    assert evidence["pre_schedule_bar_count"] == len(pre_schedule)
    assert evidence["post_schedule_bar_count"] == len(post_schedule)
    assert evidence["raw_response_sha256"] == ledger[0]["raw_response_sha256"]
    assert evidence["canonical_payload_sha256"] == ledger[0]["canonical_payload_sha256"]
    assert adjudication["methodology_correction"]["salvage_or_rerun_permitted"] is False
    assert adjudication["outcome"]["historical_basis_observations_accepted"] == 0
    assert adjudication["outcome"]["remaining_frozen_requests_permitted"] == 0
    assert audit["authority"]["accepted_edge"] is False
    assert audit["authority"]["credentials_used"] is False
    assert audit["authority"]["orders_placed"] is False


def test_terminal_audit_binds_published_implementation_and_contract() -> None:
    audit = json.loads(AUDIT_PATH.read_bytes())
    source = audit["source_contract"]
    implementation = source["implementation"]

    assert (
        source["contract_file_sha256"]
        == hashlib.sha256((ROOT / source["contract_path"]).read_bytes()).hexdigest()
    )
    assert (
        implementation["tool_sha256"]
        == hashlib.sha256((ROOT / implementation["tool_path"]).read_bytes()).hexdigest()
    )
    assert (
        implementation["module_sha256"]
        == hashlib.sha256(
            (ROOT / implementation["module_path"]).read_bytes()
        ).hexdigest()
    )
