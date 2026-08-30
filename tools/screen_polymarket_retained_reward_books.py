"""Screen one frozen Polymarket two-token book from retained exact reward sources."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import requests

from simple_ai_trading.polymarket_liquidity_rewards import paired_buy_economics
from simple_ai_trading.storage import write_bytes_atomic
from tools.screen_polymarket_exact_paired_maker_reward import (
    _canonical,
    _decimal,
    _levels,
    _list,
    _mapping,
    _request,
    _sha256,
)
from tools.screen_polymarket_exact_reward_source import (
    ROOT,
    _contract,
    _gamma,
    _preflight_destination,
    _reward,
)


def _retained(path: Path, expected_sha256: str) -> object:
    raw = path.read_bytes()
    if _sha256(raw) != expected_sha256:
        raise ValueError(f"retained source hash changed: {path}")
    return json.loads(raw)


def _retained_result(path: Path, expected_result_sha256: str) -> dict[str, object]:
    result = _mapping(json.loads(path.read_bytes()), name="retained result")
    claimed = str(result.pop("result_sha256", ""))
    if claimed != expected_result_sha256 or _sha256(_canonical(result)) != claimed:
        raise ValueError(f"retained result binding changed: {path}")
    return result


def run(*, contract_path: Path, output: Path, journal_dir: Path) -> dict[str, object]:
    started = datetime.now(timezone.utc)
    contract, contract_file_sha = _contract(
        contract_path,
        started,
        schema_version="polymarket-retained-reward-book-contract-v1",
    )
    _preflight_destination(output, name="output")
    if journal_dir.exists():
        raise ValueError("journal directory already exists")
    _preflight_destination(journal_dir, name="journal")
    journal_dir.mkdir()
    candidate = _mapping(contract.get("candidate"), name="candidate")
    retained = _mapping(contract.get("retained_exact_sources"), name="retained sources")
    gamma_source = _mapping(retained.get("gamma"), name="Gamma source")
    reward_source = _mapping(retained.get("reward"), name="reward source")
    prefilter_source = _mapping(
        retained.get("source_prefilter"), name="prefilter source"
    )
    _retained_result(
        ROOT / str(prefilter_source["path"]),
        str(prefilter_source["result_sha256"]),
    )
    market = _gamma(
        _retained(ROOT / str(gamma_source["path"]), str(gamma_source["sha256"])),
        candidate=candidate,
    )
    reward = _reward(
        _retained(ROOT / str(reward_source["path"]), str(reward_source["sha256"])),
        market=market,
        candidate=candidate,
        now=started,
        terminal_cursor=str(reward_source["terminal_cursor"]),
    )
    request = _mapping(contract.get("request_contract"), name="request contract")
    books_body = [{"token_id": token} for token in market["tokens"]]
    books_raw, books_source = _request(
        requests.Session(),
        method="POST",
        url=str(request["books_endpoint"]),
        params=None,
        json_body=books_body,
        journal_dir=journal_dir,
        source_name="01-two-token-books",
        byte_ceiling=int(request["response_byte_ceiling"]),
    )
    rows = [_mapping(value, name="book") for value in _list(books_raw, name="books")]
    books = {str(row.get("asset_id") or ""): row for row in rows}
    if len(rows) != 2 or set(books) != set(market["tokens"]):
        raise ValueError("books response did not contain the two exact tokens")
    timestamps: list[int] = []
    for row in rows:
        if str(row.get("market") or "").lower() != market["condition_id"]:
            raise ValueError("book condition identity changed")
        if (
            _decimal(row.get("tick_size"), name="book tick", positive=True)
            != market["tick_size"]
        ):
            raise ValueError("book and retained Gamma tick sizes disagree")
        if (
            _decimal(row.get("min_order_size"), name="book minimum", positive=True)
            != market["order_minimum"]
        ):
            raise ValueError("book and retained Gamma order minimums disagree")
        _levels(row, side="bids")
        _levels(row, side="asks")
        timestamps.append(int(str(row.get("timestamp"))))
    capture = _mapping(contract.get("capture"), name="capture")
    received_ms = int(books_source["received_after_ms"])
    event_age = received_ms - min(timestamps)
    skew = max(timestamps) - min(timestamps)
    fresh = (
        int(books_source["elapsed_ms"]) <= int(capture["request_max_elapsed_ms"])
        and skew <= int(capture["book_max_timestamp_skew_ms"])
        and 0 <= event_age <= int(capture["book_max_event_age_ms"])
    )
    first, second = (books[token] for token in market["tokens"])
    first_bids, first_asks = _levels(first, side="bids"), _levels(first, side="asks")
    second_bids, second_asks = (
        _levels(second, side="bids"),
        _levels(second, side="asks"),
    )
    first_bid, second_bid = (
        max(x[0] for x in first_bids),
        max(x[0] for x in second_bids),
    )
    first_ask, second_ask = (
        min(x[0] for x in first_asks),
        min(x[0] for x in second_asks),
    )
    quantity = reward["minimum_size"]
    join = paired_buy_economics(
        yes_price=first_bid,
        no_price=second_bid,
        quantity=quantity,
    )
    tick = market["tick_size"]
    improved_prices = (first_bid + tick, second_bid + tick)
    improved_marketable = (
        improved_prices[0] >= first_ask or improved_prices[1] >= second_ask
    )
    improved = paired_buy_economics(
        yes_price=improved_prices[0],
        no_price=improved_prices[1],
        quantity=quantity,
    )
    optimistic_full_pool = reward["daily_rate"] * reward["remaining_days"]
    if not fresh:
        status = "rejected_stale_book_snapshot"
    elif join.both_fill_gross_profit <= 0:
        status = "rejected_no_positive_both_fill_gross_at_best_bid"
    elif join.maximum_orphan_loss > optimistic_full_pool:
        status = "rejected_even_full_reward_pool_cannot_cover_orphan_before_horizon"
    else:
        status = "prospective_paper_candidate_not_an_edge"
    artifact: dict[str, object] = {
        "schema_version": "polymarket-retained-reward-book-screen-v1",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "event_slug": candidate["event_slug"],
            "market_slug": candidate["market_slug"],
            "question": candidate["question"],
            "condition_id": market["condition_id"],
            "tokens": market["tokens"],
            "maker_fee_zero": market["maker_fee_zero"],
            "tick_size": str(tick),
            "reward_minimum_size_shares": str(quantity),
            "reward_daily_rate_pUSD": str(reward["daily_rate"]),
            "remaining_reward_days": str(reward["remaining_days"]),
        },
        "capture": {
            "request_elapsed_ms": books_source["elapsed_ms"],
            "oldest_book_event_age_ms": event_age,
            "book_timestamp_skew_ms": skew,
            "freshness_passed": fresh,
        },
        "economics": {
            "top_of_book": {
                "yes_best_bid": str(first_bid),
                "yes_best_ask": str(first_ask),
                "no_best_bid": str(second_bid),
                "no_best_ask": str(second_ask),
            },
            "best_bid_join": {
                "combined_bid": str(join.combined_price),
                "both_fill_gross_profit_pUSD": str(join.both_fill_gross_profit),
                "maximum_orphan_settlement_loss_pUSD": str(join.maximum_orphan_loss),
            },
            "one_tick_improved": {
                "marketable": improved_marketable,
                "combined_bid": str(improved.combined_price),
                "both_fill_gross_profit_pUSD": str(improved.both_fill_gross_profit),
                "maximum_orphan_settlement_loss_pUSD": str(
                    improved.maximum_orphan_loss
                ),
            },
            "optimistic_full_reward_pool_until_horizon_pUSD": str(optimistic_full_pool),
            "publicly_proven_reward_payout_floor_pUSD": "0",
        },
        "verdict": {
            "status": status,
            "accepted_edge": False,
            "profitability_claim": False,
            "trading_authority": False,
            "retry_permitted": False,
        },
        "authority": {
            "public_unauthenticated_read_only": True,
            "credentials_used": False,
            "orders_or_cancellations": 0,
            "funded_actions": 0,
        },
        "sources": {
            "contract_path": contract_path.resolve().relative_to(ROOT).as_posix(),
            "contract_file_sha256": contract_file_sha,
            "contract_result_sha256": contract["result_sha256"],
            "retained_gamma": gamma_source,
            "retained_reward": reward_source,
            "books_request": books_source,
            "tool_sha256": _sha256(Path(__file__).read_bytes()),
        },
        "limitations": [
            "The full-pool bound is deliberately optimistic: it assumes this hypothetical maker receives 100% of every remaining daily reward.",
            "Public books do not reveal maker identity, queue position, reward persistence, samples, fills, or realized rewards.",
            "One-sided fills are directional inventory; no cancellation latency, hedge cost, adverse selection, or capacity is credited.",
        ],
    }
    artifact["result_sha256"] = _sha256(_canonical(artifact))
    write_bytes_atomic(output, _canonical(artifact) + b"\n")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--journal-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(
            contract_path=args.contract,
            output=args.output,
            journal_dir=args.journal_dir,
        )
    except Exception as exc:
        if args.journal_dir.exists():
            write_bytes_atomic(
                args.journal_dir / "terminal-failure.json",
                _canonical(
                    {
                        "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                        "raw_responses_retained_before_validation": True,
                        "retry_permitted": False,
                    }
                )
                + b"\n",
            )
        raise
    print(
        json.dumps(
            {
                "status": result["verdict"]["status"],
                "result_sha256": result["result_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
