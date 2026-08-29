"""Adjudicate the retained MLB batch after the CLOB ask-order mismatch."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Mapping

from simple_ai_trading.paper_execution import BookLevel
from simple_ai_trading.storage import write_bytes_atomic
import tools.screen_polymarket_current_mlb_monotone_parity as frozen_screen


SCHEMA = "polymarket-current-mlb-monotone-parity-failure-adjudication-v1"
CONTRACT_FILE_HASH = "63b4831878a8a8dbc9c51404c7f3d767ed9df074c16b35747c11148feafb6d7e"
ORIGINAL_IMPLEMENTATION_HASH = (
    "c180ee3de45352716fe29365f1d47a67815d884b92f56539f64cbc46b1ac1f49"
)
RAW_HASH = "7f2a08ad4988477a3be2da00e392cf4c7a07727982113c0acb4991c2b57597f9"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_hash(payload: Mapping[str, object], *, field: str) -> str:
    body = dict(payload)
    body.pop(field, None)
    return _sha256(_canonical_json(body).encode("ascii"))


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _descending_ask_audit(book: Mapping[str, object]) -> tuple[BookLevel, ...]:
    published = tuple(
        BookLevel(
            price=frozen_screen.Decimal(
                str(_mapping(value, name="ask level").get("price"))
            ),
            quantity=frozen_screen.Decimal(
                str(_mapping(value, name="ask level").get("size"))
            ),
        ).validated()
        for value in _list(book.get("asks"), name="book asks")
    )
    if not published:
        return published
    if (
        tuple(sorted(published, key=lambda level: level.price, reverse=True))
        != published
    ):
        raise ValueError("retained ask array is not strictly descending")
    if len({level.price for level in published}) != len(published):
        raise ValueError("retained ask array contains duplicate prices")
    return tuple(reversed(published))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-result", type=Path, required=True)
    parser.add_argument("--event-raw", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite adjudication: {args.output}")
    contract = _mapping(json.loads(args.contract.read_bytes()), name="contract")
    if (
        _sha256(args.contract.read_bytes()) != CONTRACT_FILE_HASH
        or _canonical_hash(contract, field="contract_sha256")
        != contract.get("contract_sha256")
        or contract.get("implementation", {}).get("sha256")
        != ORIGINAL_IMPLEMENTATION_HASH
        or _sha256(args.raw.read_bytes()) != RAW_HASH
    ):
        raise ValueError("frozen contract, implementation, or raw hash differs")
    receipts = [
        json.loads(line)
        for line in args.journal.read_text(encoding="ascii").splitlines()
    ]
    if (
        len(receipts) != 1
        or receipts[0].get("response_sha256") != RAW_HASH
        or receipts[0].get("status_code") != 200
        or receipts[0].get("request_token_count") != 30
    ):
        raise ValueError("retained request journal differs")
    payload = _list(json.loads(args.raw.read_bytes()), name="retained books")
    if len(payload) != 30:
        raise ValueError("retained book count differs")
    descending_count = sum(
        bool(_descending_ask_audit(_mapping(book, name="book"))) for book in payload
    )
    if descending_count != 30:
        raise ValueError("not every retained book has a populated descending ask array")
    markets, _ = frozen_screen._load_lattice(
        event_result_path=args.event_result,
        event_raw_path=args.event_raw,
    )
    original_ask_parser = frozen_screen._ask_levels
    try:
        frozen_screen._ask_levels = _descending_ask_audit
        sensitivity = frozen_screen._evaluate(
            contract_path=args.contract,
            contract=contract,
            markets=markets,
            payload=payload,
            receipt=receipts[0],
        )
    finally:
        frozen_screen._ask_levels = original_ask_parser
    economics = _mapping(sensitivity["economics"], name="sensitivity economics")
    candidate_count = int(economics["frozen_candidate_count"])
    adjudication: dict[str, object] = {
        "schema_version": SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "original_contract": {
            "path": args.contract.as_posix(),
            "file_sha256": CONTRACT_FILE_HASH,
            "canonical_sha256": contract["contract_sha256"],
            "consumed_and_immutable": True,
        },
        "original_implementation": {
            "path": contract["implementation"]["path"],
            "sha256": ORIGINAL_IMPLEMENTATION_HASH,
            "preserved": True,
        },
        "retained_capture": {
            "raw_path": args.raw.as_posix(),
            "raw_sha256": RAW_HASH,
            "book_count": len(payload),
            "strictly_descending_nonempty_ask_array_count": descending_count,
            "request_count": 1,
            "refetch_count": 0,
        },
        "failure_diagnosis": {
            "original_error": "CLOB asks are not sorted ascending",
            "documentation_claim": "asks sorted by price ascending",
            "retained_observation": "all 30 nonempty ask arrays were strictly descending by price",
            "correction": "reverse each fully audited retained ask array before depth walking",
            "outcome_aware_sensitivity_can_support_snapshot_promotion": False,
        },
        "retained_evidence_sensitivity": {
            "result_sha256": sensitivity["result_sha256"],
            "capture": sensitivity["capture"],
            "economics": economics,
        },
        "adjudication": {
            "accepted_edge": False,
            "profitability_claim": False,
            "deployment_ready": False,
            "trading_authority": False,
            "candidate_for_fresh_preregistered_recurrence": candidate_count > 0,
            "status": (
                "positive_only_in_outcome_aware_retained_ordering_sensitivity"
                if candidate_count > 0
                else "rejected_after_complete_retained_ordering_adjudication"
            ),
            "next_action": (
                "freeze_a_fresh_multi_game_recurrence_contract_that_accepts_documented_or_observed_book_ordering_before_any_access"
                if candidate_count > 0
                else "terminalize_this_exact_event_without_refetch_or_retry"
            ),
        },
        "implementation": {
            "path": "tools/adjudicate_polymarket_current_mlb_monotone_parity_failure.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    adjudication["result_sha256"] = _canonical_hash(adjudication, field="result_sha256")
    write_bytes_atomic(
        args.output,
        (_canonical_json(adjudication) + "\n").encode("ascii"),
    )
    print(json.dumps(adjudication["failure_diagnosis"], indent=2))
    print(json.dumps(economics["best_relation"], indent=2))
    print(json.dumps(adjudication["adjudication"], indent=2))
    print(f"result_sha256={adjudication['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
