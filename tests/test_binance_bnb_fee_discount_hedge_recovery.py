from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Mapping

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools" / "recover_binance_bnb_fee_discount_hedge_history.py"
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-bnb-fee-discount-hedge-recovery-contract-v1.json"
)
INITIAL_SCREEN_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-bnb-fee-discount-hedge-screen-v1-2026-08-25.json"
)
PUBLISHED_RESULT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-bnb-fee-discount-hedge-recovery-v1-2026-08-25.json"
)
PUBLISHED_JOURNAL_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "binance-bnb-fee-discount-hedge-recovery-journal-v1.json"
)
PUBLISHED_RAW_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "raw"
    / "binance-bnb-fee-discount-hedge-recovery-v1"
    / "01-older-futures-funding-history.json"
)
REGISTRY_PATH = (
    ROOT / "docs" / "model-research" / "structural-edge-priority-registry-v1.json"
)
EXPECTED_CONTRACT_SHA256 = (
    "d226b7340aba624e53c6b49450ddcaeea5c31f101c57ca52e2a644c809e8ae5e"
)
EXPECTED_IMPLEMENTATION_SHA256 = (
    "7437aa4c4ed4e66e452e0afa01fd0bff2bc26f40593e491c16b36b30fe81e07a"
)
EXPECTED_PUBLISHED_RESULT_SHA256 = (
    "85d0be66391b53bef87dda33ea73acaf6995d0200e6423de7999d44a8fed3c8f"
)
EXPECTED_PUBLISHED_RESULT_FILE_SHA256 = (
    "7940951ec38fc82409692cad1a47a13eb5d74f5716989824310f114cfd0ec66e"
)
EXPECTED_PUBLISHED_JOURNAL_SHA256 = (
    "347e14abf216c16ffb5cdbed24cb3fdabf0d41e5a011daad03cc41e7ba3946a7"
)
EXPECTED_PUBLISHED_JOURNAL_FILE_SHA256 = (
    "6b1b25904b116063fb50ad4401c32162ac29bc0bf8a2247b08e86b9e7489cc13"
)
EXPECTED_PUBLISHED_RAW_SHA256 = (
    "77e70b92af3492456f653294d9eedae548e17c45969ef67d5cbfc99042416cc7"
)
EXPECTED_REGISTRY_SHA256 = (
    "73363631d72664d21a5458cc773eb1cb44d46000b1f09f51feec3d5442bc208f"
)
SPEC = importlib.util.spec_from_file_location(
    "recover_binance_bnb_fee_discount_hedge_history", TOOL_PATH
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TOOL)


class _Response:
    def __init__(self, payload: object | None, *, raw: bytes | None = None) -> None:
        self.url = "https://fapi.binance.com/fapi/v1/fundingRate"
        self.status_code = 200
        self.headers = {"Content-Type": "application/json"}
        self.content = (
            raw
            if raw is not None
            else json.dumps(payload, separators=(",", ":"), sort_keys=True).encode(
                "ascii"
            )
        )


def _older_rows() -> list[dict[str, object]]:
    initial = json.loads(INITIAL_SCREEN_PATH.read_bytes())
    retained_start = initial["funding_history_payload"][0]["fundingTime"]
    interval_ms = 8 * 60 * 60 * 1000
    start = retained_start - 500 * interval_ms
    price = 650.0
    moves = (1.004, 0.996, 1.0005, 0.9995, 1.0)
    rates = ("0.00010000", "0.00005000", "-0.00002500", "0.00007500")
    rows = []
    for index in range(500):
        price *= moves[index % len(moves)]
        rows.append(
            {
                "symbol": "BNBUSDT",
                "fundingTime": start + index * interval_ms,
                "fundingRate": rates[index % len(rates)],
                "markPrice": f"{price:.8f}",
            }
        )
    assert rows[-1]["fundingTime"] == retained_start - interval_ms
    return rows


class _Session:
    def __init__(self, *, invalid_body: bool = False) -> None:
        self.invalid_body = invalid_body
        self.calls: list[tuple[str, dict[str, object], int]] = []

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, object],
        timeout: int,
    ) -> _Response:
        self.calls.append((url, dict(params), timeout))
        if self.invalid_body:
            return _Response(None, raw=b"not-json")
        return _Response(_older_rows())


def _embedded_hash(payload: dict[str, object], field: str) -> str:
    body = dict(payload)
    body.pop(field)
    canonical = json.dumps(
        body,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(canonical).hexdigest()


def test_recovery_contract_binds_initial_sources_and_exact_implementation() -> None:
    contract = json.loads(CONTRACT_PATH.read_bytes())

    assert contract["result_sha256"] == EXPECTED_CONTRACT_SHA256
    assert _embedded_hash(contract, "result_sha256") == EXPECTED_CONTRACT_SHA256
    assert hashlib.sha256(TOOL_PATH.read_bytes()).hexdigest() == (
        EXPECTED_IMPLEMENTATION_SHA256
    )
    assert contract["source_binding"]["recovery_implementation_file_sha256"] == (
        EXPECTED_IMPLEMENTATION_SHA256
    )
    assert contract["frozen_request"] == TOOL.REQUEST_SPEC
    assert contract["recovery_boundary"]["maximum_requests"] == 1
    assert contract["recovery_boundary"]["decision_threshold_changes"] is False
    assert contract["verdict_limits"]["public_recovery_can_accept_edge"] is False


def test_mock_recovery_fetches_one_non_overlapping_page_and_uses_unchanged_gate(
    tmp_path: Path,
) -> None:
    session = _Session()
    journal_path = tmp_path / "journal.json"
    raw_root = tmp_path / "raw"

    result = TOOL.run(
        session=session,
        journal_path=journal_path,
        raw_root=raw_root,
    )

    assert session.calls == [
        (
            TOOL.REQUEST_SPEC["url"],
            TOOL.REQUEST_SPEC["parameters"],
            30,
        )
    ]
    assert result["new_request_count"] == 1
    assert result["history_recovery"] == {
        "older_row_count": 500,
        "retained_row_count": 500,
        "merged_row_count": 1000,
        "older_start_time_ms": _older_rows()[0]["fundingTime"],
        "older_end_time_ms": _older_rows()[-1]["fundingTime"],
        "retained_start_time_ms": json.loads(INITIAL_SCREEN_PATH.read_bytes())[
            "funding_history_payload"
        ][0]["fundingTime"],
        "retained_end_time_ms": json.loads(INITIAL_SCREEN_PATH.read_bytes())[
            "funding_history_payload"
        ][-1]["fundingTime"],
        "boundary_gap_ms": 8 * 60 * 60 * 1000,
        "overlap_count": 0,
    }
    assert result["funding_evaluation"]["complete_inner_month_count"] >= 6
    assert result["result_sha256"] == _embedded_hash(result, "result_sha256")
    assert result["verdict"]["accepted_edge"] is False
    assert result["verdict"]["profitability_claim"] is False
    assert result["verdict"]["credentials_used"] is False
    assert result["verdict"]["signed_requests_made"] == 0
    assert result["verdict"]["orders_placed"] is False

    journal = json.loads(journal_path.read_bytes())
    assert journal["status"] == "data_complete"
    assert journal["completed_request_count"] == 1
    assert journal["journal_sha256"] == _embedded_hash(journal, "journal_sha256")
    raw_path = raw_root / TOOL.REQUEST_SPEC["raw_filename"]
    assert raw_path.exists()
    assert (
        hashlib.sha256(raw_path.read_bytes()).hexdigest()
        == (journal["responses"][0]["receipt"]["payload_sha256"])
    )


def test_invalid_recovery_body_is_retained_and_cannot_be_retried(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "journal.json"
    raw_root = tmp_path / "raw"
    session = _Session(invalid_body=True)

    with pytest.raises(ValueError, match="raw body is not valid JSON"):
        TOOL.run(
            session=session,
            journal_path=journal_path,
            raw_root=raw_root,
        )

    assert len(session.calls) == 1
    raw_path = raw_root / TOOL.REQUEST_SPEC["raw_filename"]
    assert raw_path.read_bytes() == b"not-json"
    journal = json.loads(journal_path.read_bytes())
    assert journal["status"] == "terminal_failure"
    assert journal["completed_request_count"] == 1
    assert journal["journal_sha256"] == _embedded_hash(journal, "journal_sha256")

    with pytest.raises(RuntimeError, match="rerun is prohibited"):
        TOOL.run(
            session=_Session(),
            journal_path=journal_path,
            raw_root=raw_root,
        )


def test_published_recovery_is_reconstructable_and_terminal() -> None:
    result = json.loads(PUBLISHED_RESULT_PATH.read_bytes())
    journal = json.loads(PUBLISHED_JOURNAL_PATH.read_bytes())
    registry = json.loads(REGISTRY_PATH.read_bytes())

    assert result["result_sha256"] == EXPECTED_PUBLISHED_RESULT_SHA256
    assert _embedded_hash(result, "result_sha256") == EXPECTED_PUBLISHED_RESULT_SHA256
    assert hashlib.sha256(PUBLISHED_RESULT_PATH.read_bytes()).hexdigest() == (
        EXPECTED_PUBLISHED_RESULT_FILE_SHA256
    )
    assert journal["journal_sha256"] == EXPECTED_PUBLISHED_JOURNAL_SHA256
    assert _embedded_hash(journal, "journal_sha256") == (
        EXPECTED_PUBLISHED_JOURNAL_SHA256
    )
    assert hashlib.sha256(PUBLISHED_JOURNAL_PATH.read_bytes()).hexdigest() == (
        EXPECTED_PUBLISHED_JOURNAL_FILE_SHA256
    )
    assert hashlib.sha256(PUBLISHED_RAW_PATH.read_bytes()).hexdigest() == (
        EXPECTED_PUBLISHED_RAW_SHA256
    )
    assert journal["responses"][0]["receipt"]["payload_sha256"] == (
        EXPECTED_PUBLISHED_RAW_SHA256
    )
    assert result["history_recovery"]["overlap_count"] == 0
    assert result["history_recovery"]["boundary_gap_ms"] == 8 * 60 * 60 * 1000
    assert result["history_recovery"]["merged_row_count"] == 1000
    assert result["funding_evaluation"]["complete_inner_month_count"] == 10
    assert (
        result["funding_evaluation"]["worst_complete_inner_month_short_funding_bips"]
        == "-35.61290000"
    )
    assert result["failure_reasons"] == ["primary_worst_monthly_turnover_gate_failed"]
    assert (
        result["turnover_break_even_scenarios"][0][
            "worst_monthly_turnover_multiple_to_cover_negative_funding"
        ]
        == "14.245160"
    )
    assert result["verdict"] == {
        "accepted_edge": False,
        "credentials_used": False,
        "orders_placed": False,
        "profitability_claim": False,
        "qualified_public_prequalification": False,
        "signed_requests_made": 0,
        "status": "rejected_recovered_bnb_fee_discount_hedge_prequalification",
        "trading_authority": False,
    }

    assert registry["result_sha256"] == EXPECTED_REGISTRY_SHA256
    assert _embedded_hash(registry, "result_sha256") == EXPECTED_REGISTRY_SHA256
    terminal = {row["family"]: row for row in registry["terminal_do_not_repeat"]}
    assert (
        terminal["binance_delta_hedged_bnb_spot_fee_discount_inventory"][
            "canonical_result_sha256"
        ]
        == EXPECTED_PUBLISHED_RESULT_SHA256
    )
