"""Run an exact sports package screen with a conjoined book-age gate."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from typing import Any

from tools import screen_polymarket_exact_two_leg_package as base
from tools import screen_polymarket_exact_two_leg_sports_package as sports


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise RuntimeError(f"{path.name} must contain an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True)
    args = parser.parse_args()
    contract_path = base._root_path(args.contract)
    contract = _load(contract_path)
    for dependency in contract["dependency_implementations"]:
        path = base._root_path(dependency["path"])
        if base._sha256(path.read_bytes()) != dependency["sha256"]:
            raise RuntimeError(f"dependency hash mismatch: {path.name}")

    with redirect_stdout(io.StringIO()):
        sports.main()

    result_path = base._root_path(contract["outputs"]["result_path"])
    books_path = base._root_path(contract["outputs"]["books_raw_path"])
    result = _load(result_path)
    books = json.loads(books_path.read_bytes())
    timestamps = [int(book["timestamp"]) for book in books]
    completed_at_ms = int(result["capture"]["book_receipt"]["completed_at_ms"])
    oldest_age_ms = completed_at_ms - min(timestamps)
    newest_age_ms = completed_at_ms - max(timestamps)
    maximum_age_ms = int(contract["execution"]["maximum_book_age_ms"])
    source_time_not_future = newest_age_ms >= 0
    within_age_gate = source_time_not_future and oldest_age_ms <= maximum_age_ms
    original_pass = bool(result["adjudication"]["passes_frozen_candidate_gate"])
    final_pass = original_pass and within_age_gate

    result["schema_version"] = "polymarket-exact-two-leg-package-result-v2"
    result["capture"].update(
        {
            "maximum_book_age_ms": maximum_age_ms,
            "oldest_book_age_at_completion_ms": oldest_age_ms,
            "newest_book_age_at_completion_ms": newest_age_ms,
            "source_time_not_future": source_time_not_future,
            "within_frozen_age_gate": within_age_gate,
        }
    )
    result["adjudication"].update(
        {
            "pre_age_candidate_gate_passed": original_pass,
            "passes_frozen_candidate_gate": final_pass,
            "next_action": (
                "require_one_independent_positive_recurrence_before_any_order_capable_work"
                if final_pass
                else "terminalize_this_exact_event_without_refetch_or_retry"
            ),
        }
    )
    result["implementation"] = contract["implementation"]
    result["result_sha256"] = base._canonical_hash(result, "result_sha256")
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "book_request_count": 1,
                "fee_request_count": len(result["capture"]["fee_receipts"]),
                "oldest_book_age_ms": oldest_age_ms,
                "passes_candidate_gate": final_pass,
                "payloads_printed": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
