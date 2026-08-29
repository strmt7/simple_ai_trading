from __future__ import annotations

from decimal import Decimal
import hashlib
import json
from typing import Any

from tools.screen_polymarket_shutdown_house_identity_parity import (
    CONTRACT_PATH,
    DATA_ROOT,
    QUANTITY,
    ROOT,
    _canonical_hash,
    _fill,
    _load_metadata,
)


RESULT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-shutdown-house-identity-parity-failure-adjudication-v1-2026-08-29.json"
)
FROZEN_RUNNER = ROOT / "tools/screen_polymarket_shutdown_house_identity_parity.py"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _decimal_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(value, "f")


def main() -> None:
    if RESULT_PATH.exists():
        raise RuntimeError("adjudication already exists")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if _canonical_hash(contract, "contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("contract hash mismatch")
    runner_hash = _sha256(FROZEN_RUNNER.read_bytes())
    if runner_hash != contract["implementation"]["sha256"]:
        raise RuntimeError("frozen runner hash mismatch")
    markets = _load_metadata(contract)

    journal_path = DATA_ROOT / "request-journal.jsonl"
    receipts = [json.loads(line) for line in journal_path.read_text().splitlines()]
    if not (
        len(receipts) == 2
        and receipts[0]["phase"] == "intent"
        and receipts[1]["phase"] == "completed"
        and receipts[1]["status_code"] == 200
    ):
        raise RuntimeError("unexpected consumed journal")
    books_path = DATA_ROOT / "raw/books.json"
    books_raw = books_path.read_bytes()
    if _sha256(books_raw) != receipts[1]["response_sha256"]:
        raise RuntimeError("retained books hash mismatch")
    raw_books = json.loads(books_raw)
    books = {str(book["asset_id"]): book for book in raw_books}
    tokens = contract["execution"]["tokens"]
    if len(raw_books) != 8 or set(books) != set(tokens):
        raise RuntimeError("retained book population differs")
    timestamps = [int(book["timestamp"]) for book in raw_books]
    source_skew_ms = max(timestamps) - min(timestamps)
    if source_skew_ms <= contract["execution"]["maximum_book_timestamp_skew_ms"]:
        raise RuntimeError("frozen timing failure no longer reconstructs")

    definitions = {row["name"]: row for row in contract["markets"]}
    rows: list[dict[str, Any]] = []
    for package in contract["packages"]:
        left = _fill(
            books[package["tokens"][0]],
            tick=Decimal(definitions[package["markets"][0]]["tick_size"]),
            adverse_ticks=contract["execution"]["adverse_ticks_per_leg"],
        )
        right = _fill(
            books[package["tokens"][1]],
            tick=Decimal(definitions[package["markets"][1]]["tick_size"]),
            adverse_ticks=contract["execution"]["adverse_ticks_per_leg"],
        )
        if left is None or right is None:
            actual_net = None
            stressed_net = None
        else:
            actual_net = QUANTITY - sum(
                Decimal(fill[key])
                for fill in (left, right)
                for key in ("actual_cost_pUSD", "actual_fee_pUSD")
            )
            stressed_net = QUANTITY - sum(
                Decimal(fill[key])
                for fill in (left, right)
                for key in ("stressed_cost_pUSD", "stressed_fee_pUSD")
            )
        rows.append(
            {
                "name": package["name"],
                "left_fill": left,
                "right_fill": right,
                "actual_after_Gamma_fee_schedule_sensitivity_pUSD": _decimal_text(
                    actual_net
                ),
                "stressed_after_Gamma_fee_schedule_sensitivity_pUSD": _decimal_text(
                    stressed_net
                ),
                "passes_frozen_candidate_gate": False,
                "reason": (
                    "missing_five_share_depth"
                    if stressed_net is None
                    else "frozen_book_timestamp_skew_failed_and_stressed_floor_nonpositive"
                ),
            }
        )
    if any(
        row["stressed_after_Gamma_fee_schedule_sensitivity_pUSD"] is not None
        and Decimal(row["stressed_after_Gamma_fee_schedule_sensitivity_pUSD"]) > 0
        for row in rows
    ):
        raise RuntimeError("retained diagnostic has positive stressed headroom")
    best = max(
        rows,
        key=lambda row: Decimal(
            row["stressed_after_Gamma_fee_schedule_sensitivity_pUSD"] or "-Infinity"
        ),
    )

    result: dict[str, Any] = {
        "schema_version": (
            "polymarket-shutdown-house-identity-parity-failure-adjudication-v1"
        ),
        "contract": {
            "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "sha256": contract["contract_sha256"],
        },
        "frozen_runner": {
            "path": FROZEN_RUNNER.relative_to(ROOT).as_posix(),
            "sha256": runner_hash,
            "preserved_unchanged": True,
        },
        "retained_evidence": {
            "books_path": books_path.relative_to(ROOT).as_posix(),
            "books_sha256": _sha256(books_raw),
            "journal_path": journal_path.relative_to(ROOT).as_posix(),
            "journal_sha256": _sha256(journal_path.read_bytes()),
            "book_count": len(raw_books),
            "book_timestamp_min_ms": min(timestamps),
            "book_timestamp_max_ms": max(timestamps),
            "book_timestamp_skew_ms": source_skew_ms,
            "frozen_maximum_book_timestamp_skew_ms": contract["execution"][
                "maximum_book_timestamp_skew_ms"
            ],
            "fee_requests_made": 0,
        },
        "payoff_proof": {
            "shutdown_final_yes": True,
            "truth_table": markets["truth_table"]["rows"],
            "duplicate_payoff_identity_count": 2,
        },
        "retained_books_diagnostic": {
            "authority": (
                "offline_sensitivity_only_because_the_frozen_timestamp_gate_failed_"
                "before_current_CLOB_fee_rate_cross_checks"
            ),
            "rows": rows,
            "best_package": best["name"],
            "best_stressed_after_Gamma_fee_schedule_sensitivity_pUSD": best[
                "stressed_after_Gamma_fee_schedule_sensitivity_pUSD"
            ],
        },
        "adjudication": {
            "payoff_identity_proved": True,
            "current_executable_candidate_proved": False,
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "same_snapshot_rerun_permitted": False,
            "status": "terminal_rejected_under_consumed_snapshot",
            "next_action": (
                "do_not_rerun_this_snapshot_reopen_only_after_a_material_price_fee_"
                "rule_or_book_architecture_change"
            ),
        },
        "authority": contract["authority"],
        "implementation": {
            "path": (
                "tools/adjudicate_polymarket_shutdown_house_identity_parity_failure.py"
            ),
            "sha256": "IMPLEMENTATION_HASH_PLACEHOLDER",
        },
    }
    implementation_path = ROOT / result["implementation"]["path"]
    result["implementation"]["sha256"] = _sha256(implementation_path.read_bytes())
    result["result_sha256"] = _canonical_hash(result, "result_sha256")
    RESULT_PATH.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(json.dumps(result["adjudication"], sort_keys=True))


if __name__ == "__main__":
    main()
