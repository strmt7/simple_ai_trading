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
)


RESULT_PATH = ROOT / (
    "docs/model-research/action-value/"
    "polymarket-shutdown-house-maker-first-candidate-v1-2026-08-29.json"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _best_bid(book: dict[str, Any]) -> tuple[Decimal, Decimal]:
    levels = [
        (Decimal(str(row["price"])), Decimal(str(row["size"])))
        for row in book.get("bids", [])
    ]
    if not levels:
        raise RuntimeError("maker book has no bid")
    return max(levels, key=lambda row: row[0])


def main() -> None:
    if RESULT_PATH.exists():
        raise RuntimeError("candidate artifact already exists")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if _canonical_hash(contract, "contract_sha256") != contract["contract_sha256"]:
        raise RuntimeError("contract hash mismatch")
    runner_path = ROOT / contract["implementation"]["path"]
    if _sha256(runner_path.read_bytes()) != contract["implementation"]["sha256"]:
        raise RuntimeError("frozen runner hash mismatch")
    books_path = DATA_ROOT / "raw/books.json"
    raw = books_path.read_bytes()
    books = {str(row["asset_id"]): row for row in json.loads(raw)}
    definitions = {row["name"]: row for row in contract["markets"]}

    rows: list[dict[str, Any]] = []
    for package in contract["packages"]:
        for maker_index in (0, 1):
            hedge_index = 1 - maker_index
            maker_token = package["tokens"][maker_index]
            hedge_token = package["tokens"][hedge_index]
            maker_market = package["markets"][maker_index]
            hedge_market = package["markets"][hedge_index]
            maker_price, visible_queue = _best_bid(books[maker_token])
            hedge = _fill(
                books[hedge_token],
                tick=Decimal(definitions[hedge_market]["tick_size"]),
                adverse_ticks=contract["execution"]["adverse_ticks_per_leg"],
            )
            if hedge is None:
                raise RuntimeError("retained hedge lacks five-share depth")
            maker_cost = QUANTITY * maker_price
            actual_net = (
                QUANTITY
                - maker_cost
                - Decimal(hedge["actual_cost_pUSD"])
                - Decimal(hedge["actual_fee_pUSD"])
            )
            stressed_net = (
                QUANTITY
                - maker_cost
                - Decimal(hedge["stressed_cost_pUSD"])
                - Decimal(hedge["stressed_fee_pUSD"])
            )
            source_skew_ms = abs(
                int(books[maker_token]["timestamp"])
                - int(books[hedge_token]["timestamp"])
            )
            rows.append(
                {
                    "package": package["name"],
                    "maker_market": maker_market,
                    "maker_token": maker_token,
                    "maker_bid_price": _decimal_text(maker_price),
                    "visible_queue_ahead_shares": _decimal_text(visible_queue),
                    "maker_fee_pUSD": "0",
                    "hedge_market": hedge_market,
                    "hedge_token": hedge_token,
                    "hedge_fill": hedge,
                    "pair_source_timestamp_skew_ms": source_skew_ms,
                    "actual_after_fee_profit_sensitivity_pUSD": _decimal_text(
                        actual_net
                    ),
                    "two_tick_hedge_stressed_profit_sensitivity_pUSD": (
                        _decimal_text(stressed_net)
                    ),
                    "positive_stressed_economic_lead": stressed_net > 0,
                    "source_continuity_proved": False,
                    "queue_censored_fill_proved": False,
                }
            )
    positive = [row for row in rows if row["positive_stressed_economic_lead"]]
    best = max(
        rows,
        key=lambda row: Decimal(
            row["two_tick_hedge_stressed_profit_sensitivity_pUSD"]
        ),
    )
    result: dict[str, Any] = {
        "schema_version": "polymarket-shutdown-house-maker-first-candidate-v1",
        "source_contract": {
            "path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "sha256": contract["contract_sha256"],
        },
        "retained_books": {
            "path": books_path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(raw),
        },
        "mechanism": (
            "rest_one_five_share_bid_on_one_of_two_exact_duplicate_payoff_tokens_"
            "then_buy_the_complement_of_the_other_immediately_after_an_owned_fill"
        ),
        "market_direction_forecast_required": False,
        "rows": rows,
        "adjudication": {
            "positive_two_tick_hedge_sensitivity_count": len(positive),
            "best_role": {
                "package": best["package"],
                "maker_market": best["maker_market"],
                "hedge_market": best["hedge_market"],
                "stressed_profit_sensitivity_pUSD": best[
                    "two_tick_hedge_stressed_profit_sensitivity_pUSD"
                ],
                "visible_queue_ahead_shares": best["visible_queue_ahead_shares"],
                "pair_source_timestamp_skew_ms": best[
                    "pair_source_timestamp_skew_ms"
                ],
            },
            "candidate_edge": bool(positive),
            "accepted_edge": False,
            "deployment_ready": False,
            "profitability_claim": False,
            "reason": (
                "retained_current_books_show_direction_independent_stressed_"
                "headroom_but_do_not_prove_source_continuity_queue_fill_or_"
                "causally_subsequent_hedge"
            ),
            "next_action": (
                "freeze_one_source_continuous_queue_censored_public_capture_for_"
                "the_best_role_then_require_explicit_authenticated_paper_authority_"
                "before_any_order"
            ),
        },
        "authority": contract["authority"],
        "implementation": {
            "path": "tools/analyze_polymarket_shutdown_house_maker_first.py",
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
