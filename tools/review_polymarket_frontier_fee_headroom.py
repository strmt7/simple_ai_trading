"""Retained-price fee-only rejection audit; never authorizes capture or trading."""

from __future__ import annotations

from decimal import Decimal as D
import json
from pathlib import Path

from tools import screen_polymarket_exact_negrisk_long_only_frontier as frontier
from tools.screen_polymarket_exact_negrisk_long_only_frontier_v2 import _journal

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/review/2026-09-04/nyc-sep5-frontier"


def run() -> None:
    output = BASE / "fee-audit-result.json"
    journal = BASE / "fee-audit-journal.jsonl"
    if output.exists() or journal.exists():
        raise RuntimeError("fee audit already consumed")
    _journal(journal, {"phase": "intent", "network_requests": 0})
    try:
        plan = json.loads((BASE / "fee-audit-plan.json").read_bytes())
        result = json.loads((BASE / "result.json").read_bytes())
        raw = (BASE / "raw/event.json").read_bytes()
        if (
            frontier.base._canonical_hash(result, "result_sha256")
            != plan["input_result_sha256"]
        ):
            raise ValueError("original result binding differs")
        if frontier.base._sha256(raw) != plan["input_raw_sha256"]:
            raise ValueError("original raw binding differs")
        contract_path = BASE / "contract.json"
        contract = json.loads(contract_path.read_bytes())
        frontier._validate_contract(contract, contract_path)
        event = json.loads(raw)
        markets = {str(row["id"]): row for row in frontier._markets(event, 11)}
        rows = []
        for index, row in enumerate(result["screen"]["rows"]):
            quantity = D(row["quantity_shares_each_leg"])
            fees = []
            for leg in row["legs"]:
                market = markets[leg["market_id"]]
                original = (
                    frontier._yes_acquisition(market)
                    if leg["outcome"] == "Yes"
                    else frontier._acquisition_v2(market, outcome="No")
                )
                if original is None or original["token_id"] != leg["token_id"]:
                    raise ValueError("original leg identity differs")
                price = D(leg["price_pUSD_per_share"])
                if original["price_pUSD_per_share"] != price:
                    raise ValueError("original leg price differs")
                fees.append(original["fee_model"](price, quantity, "taker"))
            total_fee = sum(fees, D(0))
            gross = D(row["metadata_profit_floor_pUSD"])
            rows.append(
                {
                    "original_row_index": index,
                    "leg_count": len(fees),
                    "gross_headroom_pUSD": str(gross),
                    "configured_taker_fee_without_ticks_pUSD": str(total_fee),
                    "largest_single_leg_fee_pUSD": str(max(fees)),
                    "after_configured_fee_without_ticks_pUSD": str(gross - total_fee),
                }
            )
        source_paths = [
            BASE / "fee-audit-plan.json",
            BASE / "result.json",
            BASE / "raw/event.json",
            Path(__file__),
        ]
        audit = {
            "schema_version": 1,
            "classification": plan["classification"],
            "rows": rows,
            "new_requests": 0,
            "accepted_edge": False,
            "further_requests_authorized": False,
            "original_gate_changed": False,
            "bindings": [
                {
                    "path": p.relative_to(ROOT).as_posix(),
                    "sha256": frontier.base._sha256(p.read_bytes()),
                }
                for p in source_paths
            ],
        }
        audit["result_sha256"] = frontier.base._canonical_hash(audit, "result_sha256")
        with output.open("x", encoding="ascii", newline="\n") as stream:
            json.dump(audit, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
        _journal(
            journal, {"phase": "completed", "rows": len(rows), "network_requests": 0}
        )
        print(
            json.dumps(
                {
                    "rows": len(rows),
                    "positive_after_fee_rows": sum(
                        D(row["after_configured_fee_without_ticks_pUSD"]) > 0
                        for row in rows
                    ),
                }
            )
        )
    except Exception:
        _journal(journal, {"phase": "terminal_failure", "network_requests": 0})
        raise


if __name__ == "__main__":
    run()
