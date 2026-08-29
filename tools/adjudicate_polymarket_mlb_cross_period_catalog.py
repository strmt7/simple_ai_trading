from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from typing import Any

from tools.screen_polymarket_mlb_cross_period_catalog import _relations


ROOT = Path(__file__).resolve().parents[1]


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


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
    return _sha256(encoded)


def _root_path(value: str) -> Path:
    path = (ROOT / value).resolve()
    path.relative_to(ROOT)
    return path


def adjudicate(
    *, result: dict[str, Any], raw: bytes, result_path: str, raw_path: str
) -> dict[str, Any]:
    if _canonical_hash(result, "result_sha256") != result.get("result_sha256"):
        raise RuntimeError("catalog result hash mismatch")
    receipt = result["capture"]["receipt"]
    if _sha256(raw) != receipt["response_sha256"]:
        raise RuntimeError("catalog raw hash mismatch")
    events = json.loads(raw)
    if not isinstance(events, list):
        raise RuntimeError("catalog raw response is not a list")
    completed_at = datetime.fromtimestamp(
        receipt["completed_at_ms"] / 1000, tz=timezone.utc
    )
    relations, future_event_count = _relations(
        events,
        completed_at=completed_at,
        tag_id="100381",
    )
    relations.sort(
        key=lambda row: (
            Decimal(row["gamma_displayed_price_sum_pUSD"]),
            row["start_time_utc"],
            row["event_slug"],
            Decimal(row["line"]),
        )
    )
    candidates = [row for row in relations if row["passes_strictly_below_payout_gate"]]
    screen = result["screen"]
    if not (
        future_event_count == screen["future_event_count_at_completed_request"]
        and len(relations) == screen["exact_cross_period_relation_count"]
        and len(candidates) == screen["candidate_count_strictly_below_payout_floor"]
    ):
        raise RuntimeError("offline relation inventory differs from consumed result")
    if screen["best_candidate"] != (candidates[0] if candidates else None):
        raise RuntimeError("offline best candidate differs from consumed result")
    best = relations[0] if relations else None
    adjudication: dict[str, Any] = {
        "schema_version": "polymarket-mlb-cross-period-catalog-adjudication-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat().replace(
            "+00:00", "Z"
        ),
        "consumed_result": {
            "path": result_path,
            "sha256": result["result_sha256"],
            "preserved": True,
        },
        "retained_raw": {
            "path": raw_path,
            "sha256": receipt["response_sha256"],
            "refetch_count": 0,
        },
        "population": {
            "returned_event_count": result["capture"]["returned_event_count"],
            "limit": result["capture"]["limit"],
            "partial": not result["capture"][
                "population_complete_under_frozen_filter"
            ],
            "future_event_count_at_completed_request": future_event_count,
        },
        "complete_retained_page_screen": {
            "exact_cross_period_relation_count": len(relations),
            "candidate_count_strictly_below_payout_floor": len(candidates),
            "relations": relations,
            "best_relation": best,
        },
        "adjudication": {
            "status": (
                "retained_partial_page_contains_candidate"
                if candidates
                else "retained_partial_page_rejected_before_books_and_fees"
            ),
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "next_action": (
                "freeze_one_exact_depth_screen_for_the_consumed_best_candidate"
                if candidates
                else "do_not_adaptively_paginate_or_request_books_for_this_consumed_page"
            ),
        },
        "authority": {
            **result["authority"],
            "offline_adjudication": True,
            "book_requests": 0,
            "fee_requests": 0,
            "protected_capture_touched": False,
        },
        "implementation": {
            "path": "tools/adjudicate_polymarket_mlb_cross_period_catalog.py",
            "sha256": _sha256(Path(__file__).read_bytes()),
        },
    }
    adjudication["result_sha256"] = _canonical_hash(adjudication, "result_sha256")
    return adjudication


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adjudicate every relation in a retained MLB catalog page."
    )
    parser.add_argument("--result", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result_path = _root_path(args.result)
    raw_path = _root_path(args.raw)
    output_path = _root_path(args.output)
    if output_path.exists():
        raise RuntimeError("adjudication output already exists")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    payload = adjudicate(
        result=result,
        raw=raw_path.read_bytes(),
        result_path=args.result,
        raw_path=args.raw,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="ascii",
        newline="\n",
    )
    best = payload["complete_retained_page_screen"]["best_relation"]
    print(
        json.dumps(
            {
                "relation_count": payload["complete_retained_page_screen"][
                    "exact_cross_period_relation_count"
                ],
                "candidate_count": payload["complete_retained_page_screen"][
                    "candidate_count_strictly_below_payout_floor"
                ],
                "best_displayed_sum_pUSD": (
                    best["gamma_displayed_price_sum_pUSD"] if best else None
                ),
                "refetch_count": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
