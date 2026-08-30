"""Adjudicate a completed NegRisk book batch without another venue request."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

from simple_ai_trading.structural_parity import screen_negative_risk_parity
from tools.adjudicate_polymarket_exact_mlb_monotone_prefilter import (
    _canonical_hash,
    _root_path,
    _sha256,
)
from tools.screen_polymarket_exact_negrisk_books import (
    ZERO_FEE,
    _outcomes,
    _retained_event,
    _screen,
    _stressed_outcomes,
    _tokens,
)


SCHEMA = "polymarket-exact-negrisk-books-retained-adjudication-v1"


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} must contain an object")
    return value


def _load_journal(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="ascii").splitlines()]
    if len(rows) != 2 or any(not isinstance(row, dict) for row in rows):
        raise RuntimeError("request journal must contain exact intent and completion rows")
    return rows


def _validate_contract(
    contract: dict[str, Any], contract_path: Path
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    if _canonical_hash(contract, "contract_sha256") != contract.get("contract_sha256"):
        raise RuntimeError("contract hash mismatch")
    if contract_path != _root_path(contract["contract_path"]):
        raise RuntimeError("contract path mismatch")
    frozen = datetime.fromisoformat(contract["frozen_at_utc"].replace("Z", "+00:00"))
    if frozen.tzinfo is None or frozen > datetime.now(timezone.utc):
        raise RuntimeError("frozen_at_utc is invalid or in the future")
    if contract.get("authority") != {
        "account_requests": 0,
        "book_batch_requests": 0,
        "credentials_used": False,
        "funds_used": False,
        "gamma_requests": 0,
        "onchain_requests": 0,
        "orders_or_transactions": 0,
        "protected_capture_touched": False,
        "public_unauthenticated_read_only_requests": 0,
        "signed_requests": 0,
        "trading_authority": False,
    }:
        raise RuntimeError("authority boundary changed")
    source = contract["retained_sources"]
    source_contract_path = _root_path(source["books_contract_path"])
    source_contract = _load_object(source_contract_path)
    if (
        _canonical_hash(source_contract, "contract_sha256")
        != source["books_contract_sha256"]
        or source_contract["contract_sha256"] != source["books_contract_sha256"]
    ):
        raise RuntimeError("source book contract lineage differs")
    raw_path = _root_path(source["books_raw_path"])
    journal_path = _root_path(source["books_journal_path"])
    raw = raw_path.read_bytes()
    if (
        _sha256(raw) != source["books_raw_sha256"]
        or _sha256(journal_path.read_bytes()) != source["books_journal_sha256"]
    ):
        raise RuntimeError("retained book or journal bytes differ")
    intent, completed = _load_journal(journal_path)
    request = source_contract["request"]
    common = {
        "method": request["method"],
        "name": "exact-negrisk-complete-token-books",
        "request_body_sha256": request["body_sha256"],
        "requested_at_ms": intent.get("requested_at_ms"),
        "url": request["url"],
    }
    if intent != {**common, "phase": "intent"}:
        raise RuntimeError("retained request intent differs")
    if any(completed.get(key) != value for key, value in common.items()):
        raise RuntimeError("retained request completion identity differs")
    if (
        completed.get("phase") != "completed"
        or completed.get("status_code") != 200
        or completed.get("response_bytes") != len(raw)
        or completed.get("response_sha256") != _sha256(raw)
        or completed.get("raw_path") != source["books_raw_path"]
    ):
        raise RuntimeError("retained request completion receipt differs")
    event = _retained_event(source_contract)
    if source_contract["token_ids"] != _tokens(event):
        raise RuntimeError("retained event token population differs")
    for implementation in contract["implementations"]:
        path = _root_path(implementation["path"])
        if _sha256(path.read_bytes()) != implementation["sha256"]:
            raise RuntimeError(f"implementation hash mismatch: {path.name}")
    return source_contract, event, raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    contract = _load_object(contract_path)
    source_contract, event, raw = _validate_contract(contract, contract_path)
    result_path = _root_path(contract["output_path"])
    if result_path.exists():
        raise RuntimeError(f"one-use output already exists: {result_path.name}")
    result_path.parent.mkdir(parents=True, exist_ok=True)

    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) != len(source_contract["token_ids"]):
        raise RuntimeError("retained book batch count differs")
    books = {str(row.get("asset_id") or ""): row for row in rows if isinstance(row, dict)}
    if set(books) != set(source_contract["token_ids"]):
        raise RuntimeError("retained book identities differ")
    journal = _load_journal(_root_path(contract["retained_sources"]["books_journal_path"]))
    receipt = journal[-1]
    timestamps = [int(str(book.get("timestamp"))) for book in books.values()]
    completed_ms = int(receipt["completed_at_ms"])
    elapsed_ms = completed_ms - int(receipt["requested_at_ms"])
    age_ms = completed_ms - min(timestamps)
    skew_ms = max(timestamps) - min(timestamps)
    freshness = source_contract["freshness"]
    freshness_passed = (
        elapsed_ms <= freshness["request_max_elapsed_ms"]
        and 0 <= age_ms <= freshness["book_max_event_age_ms"]
        and skew_ms <= freshness["book_max_timestamp_skew_ms"]
    )
    outcomes = _outcomes(event, books)
    quantity = Decimal(source_contract["quantity_shares"])
    gross = screen_negative_risk_parity(
        tuple(replace(outcome, fee_model=ZERO_FEE) for outcome in outcomes),
        quantity=quantity,
        conversion_fee_bips=0,
    )
    after_fee = screen_negative_risk_parity(
        outcomes, quantity=quantity, conversion_fee_bips=0
    )
    stressed = screen_negative_risk_parity(
        _stressed_outcomes(event, outcomes),
        quantity=quantity,
        conversion_fee_bips=0,
    )
    candidate = bool(
        freshness_passed
        and after_fee.best_path is not None
        and after_fee.best_path.net_quote > 0
        and stressed.best_path is not None
        and stressed.best_path.net_quote > 0
    )
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract": {"path": contract["contract_path"], "sha256": contract["contract_sha256"]},
        "retained_capture": {
            "response_sha256": receipt["response_sha256"],
            "response_bytes": receipt["response_bytes"],
            "request_elapsed_ms": elapsed_ms,
            "oldest_book_event_age_ms": age_ms,
            "book_timestamp_skew_ms": skew_ms,
            "freshness_passed": freshness_passed,
            "book_count": len(books),
        },
        "screen": {
            "zero_fee_no_stress": _screen(gross),
            "gamma_fee_no_stress": _screen(after_fee),
            "gamma_fee_one_adverse_tick_each_leg": _screen(stressed),
            "candidate_after_all_frozen_gates": candidate,
        },
        "adjudication": {
            "status": (
                "source_freshness_failure_no_promotion"
                if not freshness_passed
                else (
                    "candidate_requires_onchain_adapter_fee_gas_latency_atomicity_and_owned_execution_proof"
                    if candidate
                    else "rejected_after_exact_depth_and_current_gamma_fees_before_onchain_requests"
                )
            ),
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "network_requests": 0,
        },
        "authority": contract["authority"],
    }
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    result_path.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "freshness_passed": freshness_passed,
                "gross_best_net": gross.best_path.net_quote if gross.best_path else None,
                "after_fee_best_net": after_fee.best_path.net_quote if after_fee.best_path else None,
                "stressed_best_net": stressed.best_path.net_quote if stressed.best_path else None,
                "candidate": candidate,
                "network_requests": 0,
                "payloads_printed": 0,
            },
            default=str,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
