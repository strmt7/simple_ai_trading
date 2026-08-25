"""Screen one frozen Polymarket paired-maker reward candidate without trading."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping

import requests

import simple_ai_trading.polymarket_liquidity_rewards as reward_module
from simple_ai_trading.polymarket_liquidity_rewards import (
    conservative_instantaneous_share,
    maker_minimum_score,
    minimum_reward_days_to_cover,
    paired_buy_economics,
    reward_order_score,
)
from simple_ai_trading.storage import write_bytes_atomic


SCHEMA_VERSION = "polymarket-paired-maker-reward-screen-v1"
CLOB_BASE_URL = "https://clob.polymarket.com"
GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CONDITION_ID = "0x321aa3f3983c61883c1ef938219a2aaa288261c5155fc72fd222285f2fec682f"
EVENT_ID = "727197"
YES_TOKEN_ID = (
    "99279657422049748425702482979776534988406415889924053587239236382885350018424"
)
NO_TOKEN_ID = (
    "81605338347481327415152110254715721970954014495132423843306321337937754395728"
)
MAX_RECEIPT_SPAN_MS = 2_000
MAX_BOOK_EVENT_AGE_MS = 5_000
MAX_BOOK_SKEW_MS = 1_000


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


def _mapping(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _json_list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a JSON string")
    try:
        return _list(json.loads(value), name=name)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not JSON") from exc


def _request(
    session: requests.Session,
    method: str,
    url: str,
    *,
    json_body: object | None = None,
) -> tuple[object, dict[str, object]]:
    before_ms = time.time_ns() // 1_000_000
    response = session.request(method, url, json=json_body, timeout=30)
    after_ms = time.time_ns() // 1_000_000
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        raise RuntimeError(
            "Polymarket rate limit reached; stopped without retry"
            + (f"; Retry-After={retry_after}" if retry_after else "")
        )
    response.raise_for_status()
    try:
        decoded = response.json()
    except requests.JSONDecodeError as exc:
        raise ValueError(f"source {url} did not return JSON") from exc
    return decoded, {
        "method": method,
        "url": response.url,
        "payload_sha256": _sha256(response.content),
        "requested_before_ms": before_ms,
        "received_after_ms": after_ms,
        "request_elapsed_ms": after_ms - before_ms,
    }


def _decimal(value: object, *, name: str, positive: bool = False) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if isinstance(value, bool) or not result.is_finite():
        raise ValueError(f"{name} must be a finite decimal")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _clob_market(raw: object) -> dict[str, object]:
    market = _mapping(raw, name="CLOB market")
    if not (
        market.get("condition_id") == CONDITION_ID
        and market.get("enable_order_book") is True
        and market.get("active") is True
        and market.get("closed") is False
        and market.get("accepting_orders") is True
        and market.get("neg_risk") is True
    ):
        raise ValueError("CLOB market is not the active frozen candidate")
    tokens = [
        _mapping(item, name="CLOB token")
        for item in _list(market.get("tokens"), name="CLOB tokens")
    ]
    token_contract = {
        (str(item.get("outcome")), str(item.get("token_id"))) for item in tokens
    }
    if token_contract != {("Yes", YES_TOKEN_ID), ("No", NO_TOKEN_ID)}:
        raise ValueError("CLOB market token identity changed")
    return market


def _reward_market(raw: object) -> dict[str, object]:
    payload = _mapping(raw, name="reward response")
    rows = [
        _mapping(item, name="reward market")
        for item in _list(payload.get("data"), name="reward markets")
    ]
    if len(rows) != 1 or rows[0].get("condition_id") != CONDITION_ID:
        raise ValueError("reward response did not contain exactly the frozen candidate")
    return rows[0]


def _gamma_market(raw: object) -> tuple[dict[str, object], dict[str, object]]:
    event = _mapping(raw, name="Gamma event")
    if not (
        str(event.get("id")) == EVENT_ID
        and event.get("negRisk") is True
        and event.get("negRiskAugmented") is True
    ):
        raise ValueError("Gamma event identity or augmented negative-risk flag changed")
    matches = [
        _mapping(item, name="Gamma market")
        for item in _list(event.get("markets"), name="Gamma markets")
        if isinstance(item, Mapping) and item.get("conditionId") == CONDITION_ID
    ]
    if len(matches) != 1:
        raise ValueError("Gamma event did not contain exactly the frozen market")
    market = matches[0]
    if not (
        market.get("active") is True
        and market.get("closed") is False
        and market.get("acceptingOrders") is True
        and market.get("enableOrderBook") is True
        and _json_list(market.get("outcomes"), name="Gamma outcomes") == ["Yes", "No"]
        and _json_list(market.get("clobTokenIds"), name="Gamma tokens")
        == [YES_TOKEN_ID, NO_TOKEN_ID]
    ):
        raise ValueError("Gamma market execution identity changed")
    fee = _mapping(market.get("feeSchedule"), name="Gamma fee schedule")
    if market.get("feesEnabled") is not True or fee.get("takerOnly") is not True:
        raise ValueError("maker zero-fee contract is not explicit")
    return event, market


def _book_rows(raw: object) -> dict[str, dict[str, object]]:
    books: dict[str, dict[str, object]] = {}
    for item in _list(raw, name="books response"):
        book = _mapping(item, name="book")
        token_id = str(book.get("asset_id") or "")
        if book.get("market") != CONDITION_ID or token_id not in {
            YES_TOKEN_ID,
            NO_TOKEN_ID,
        }:
            raise ValueError("book identity changed")
        if token_id in books:
            raise ValueError("book token is duplicated")
        books[token_id] = book
    if set(books) != {YES_TOKEN_ID, NO_TOKEN_ID}:
        raise ValueError("both frozen candidate books are required")
    return books


def _levels(
    book: Mapping[str, object], *, side: str
) -> tuple[tuple[Decimal, Decimal], ...]:
    result: list[tuple[Decimal, Decimal]] = []
    for item in _list(book.get(side), name=f"book {side}"):
        level = _mapping(item, name="book level")
        result.append(
            (
                _decimal(level.get("price"), name="book price", positive=True),
                _decimal(level.get("size"), name="book size", positive=True),
            )
        )
    reverse = side == "asks"
    normalized = tuple(result)
    if (
        not normalized
        or tuple(sorted(normalized, key=lambda row: row[0], reverse=reverse))
        != normalized
    ):
        raise ValueError(f"book {side} is empty or not in CLOB response order")
    if len({price for price, _size in normalized}) != len(normalized):
        raise ValueError(f"book {side} has duplicate prices")
    return normalized


def _score_levels(
    levels: tuple[tuple[Decimal, Decimal], ...],
    *,
    midpoint: Decimal,
    maximum_spread: Decimal,
) -> Decimal:
    return sum(
        (
            reward_order_score(
                maximum_spread=maximum_spread,
                distance=abs(price - midpoint),
                size=size,
            )
            for price, size in levels
        ),
        Decimal("0"),
    )


def _quote_payload(
    *,
    yes_book: Mapping[str, object],
    no_book: Mapping[str, object],
    tick_size: Decimal,
    reward_size: Decimal,
    maximum_spread: Decimal,
    daily_reward_rate: Decimal,
) -> dict[str, object]:
    yes_bids = _levels(yes_book, side="bids")
    yes_asks = _levels(yes_book, side="asks")
    no_bids = _levels(no_book, side="bids")
    no_asks = _levels(no_book, side="asks")
    yes_quote = yes_bids[-1][0] + tick_size
    no_quote = no_bids[-1][0] + tick_size
    yes_ask = yes_asks[-1][0]
    no_ask = no_asks[-1][0]
    if yes_quote >= yes_ask or no_quote >= no_ask:
        raise ValueError("one-tick-improved paired bid would be marketable")
    if yes_quote % tick_size != 0 or no_quote % tick_size != 0:
        raise ValueError("hypothetical quotes are not tick aligned")

    # The public methodology names a size-cutoff-adjusted midpoint but does not
    # document its construction. These post-quote top midpoints are therefore a
    # conditional diagnostic, never a claimed production score or payout.
    yes_midpoint = (yes_quote + yes_ask) / 2
    no_midpoint = (no_quote + no_ask) / 2
    yes_score = reward_order_score(
        maximum_spread=maximum_spread,
        distance=abs(yes_quote - yes_midpoint),
        size=reward_size,
    )
    no_score = reward_order_score(
        maximum_spread=maximum_spread,
        distance=abs(no_quote - no_midpoint),
        size=reward_size,
    )
    own_minimum = maker_minimum_score(
        q_one=yes_score,
        q_two=no_score,
        midpoint=yes_midpoint,
    )
    old_q_one = _score_levels(
        yes_bids,
        midpoint=yes_midpoint,
        maximum_spread=maximum_spread,
    ) + _score_levels(
        no_asks,
        midpoint=no_midpoint,
        maximum_spread=maximum_spread,
    )
    old_q_two = _score_levels(
        yes_asks,
        midpoint=yes_midpoint,
        maximum_spread=maximum_spread,
    ) + _score_levels(
        no_bids,
        midpoint=no_midpoint,
        maximum_spread=maximum_spread,
    )
    share = conservative_instantaneous_share(
        own_minimum_score=own_minimum,
        old_aggregate_q_one=old_q_one,
        old_aggregate_q_two=old_q_two,
    )
    daily_equivalent = daily_reward_rate * share
    economics = paired_buy_economics(
        yes_price=yes_quote,
        no_price=no_quote,
        quantity=reward_size,
    )
    payback = minimum_reward_days_to_cover(
        maximum_orphan_loss=economics.maximum_orphan_loss,
        daily_reward_bound=daily_equivalent,
    )
    return {
        "quantity": str(reward_size),
        "yes_bid_price": str(yes_quote),
        "no_bid_price": str(no_quote),
        "combined_bid_price": str(economics.combined_price),
        "both_fill_gross_profit": str(economics.both_fill_gross_profit),
        "yes_only_maximum_settlement_loss": str(economics.yes_only_maximum_loss),
        "no_only_maximum_settlement_loss": str(economics.no_only_maximum_loss),
        "maximum_orphan_settlement_loss": str(economics.maximum_orphan_loss),
        "conditional_post_quote_yes_midpoint": str(yes_midpoint),
        "conditional_post_quote_no_midpoint": str(no_midpoint),
        "conditional_yes_order_score": str(yes_score),
        "conditional_no_order_score": str(no_score),
        "conditional_own_minimum_score": str(own_minimum),
        "conservative_old_aggregate_q_one": str(old_q_one),
        "conservative_old_aggregate_q_two": str(old_q_two),
        "conditional_instantaneous_share_lower_bound": str(share),
        "conditional_daily_rate_equivalent_lower_bound": str(daily_equivalent),
        "conditional_reward_days_to_cover_maximum_orphan_loss": (
            None if payback is None else str(payback)
        ),
        "publicly_proven_reward_payout_lower_bound": "0",
    }


def run(*, session: requests.Session | None = None) -> dict[str, object]:
    """Run one bounded public-data candidate diagnostic without any order."""

    started_ms = time.time_ns() // 1_000_000
    http = session or requests.Session()
    clob_raw, clob_source = _request(
        http, "GET", f"{CLOB_BASE_URL}/markets/{CONDITION_ID}"
    )
    reward_raw, reward_source = _request(
        http, "GET", f"{CLOB_BASE_URL}/rewards/markets/{CONDITION_ID}"
    )
    gamma_raw, gamma_source = _request(
        http, "GET", f"{GAMMA_BASE_URL}/events/{EVENT_ID}"
    )
    books_raw, books_source = _request(
        http,
        "POST",
        f"{CLOB_BASE_URL}/books",
        json_body=[{"token_id": YES_TOKEN_ID}, {"token_id": NO_TOKEN_ID}],
    )
    completed_ms = time.time_ns() // 1_000_000

    clob = _clob_market(clob_raw)
    reward = _reward_market(reward_raw)
    event, gamma = _gamma_market(gamma_raw)
    books = _book_rows(books_raw)
    tick_size = _decimal(clob.get("minimum_tick_size"), name="tick size", positive=True)
    reward_size = _decimal(
        reward.get("rewards_min_size"), name="reward size", positive=True
    )
    maximum_spread_cents = _decimal(
        reward.get("rewards_max_spread"), name="maximum reward spread", positive=True
    )
    maximum_spread = maximum_spread_cents / 100
    rates = [
        _decimal(item.get("rate_per_day"), name="daily reward rate", positive=True)
        for item in (
            _mapping(value, name="reward configuration")
            for value in _list(
                reward.get("rewards_config"), name="reward configurations"
            )
        )
    ]
    daily_rate = sum(rates, Decimal("0"))
    if daily_rate <= 0:
        raise ValueError("daily reward rate must be positive")
    clob_reward = _mapping(clob.get("rewards"), name="CLOB rewards")
    clob_daily_rate = sum(
        (
            _decimal(
                item.get("rewards_daily_rate"),
                name="CLOB reward rate",
                positive=True,
            )
            for item in (
                _mapping(value, name="CLOB reward rate")
                for value in _list(clob_reward.get("rates"), name="CLOB reward rates")
            )
        ),
        Decimal("0"),
    )
    if not (
        daily_rate == clob_daily_rate
        and reward_size
        == _decimal(clob_reward.get("min_size"), name="CLOB reward size")
        and maximum_spread_cents
        == _decimal(clob_reward.get("max_spread"), name="CLOB reward spread")
        and reward_size
        == _decimal(gamma.get("rewardsMinSize"), name="Gamma reward size")
        and maximum_spread_cents
        == _decimal(gamma.get("rewardsMaxSpread"), name="Gamma reward spread")
        and tick_size
        == _decimal(gamma.get("orderPriceMinTickSize"), name="Gamma tick size")
    ):
        raise ValueError(
            "candidate reward or tick configuration disagrees across sources"
        )

    book_timestamps = []
    for book in books.values():
        timestamp = book.get("timestamp")
        try:
            parsed = int(str(timestamp))
        except ValueError as exc:
            raise ValueError(
                "book timestamp must be integer epoch milliseconds"
            ) from exc
        if parsed <= 0:
            raise ValueError("book timestamp must be positive")
        book_timestamps.append(parsed)
    receipt_span_ms = int(books_source["received_after_ms"]) - int(
        clob_source["requested_before_ms"]
    )
    book_skew_ms = max(book_timestamps) - min(book_timestamps)
    book_event_age_ms = int(books_source["received_after_ms"]) - min(book_timestamps)
    freshness_passed = (
        receipt_span_ms <= MAX_RECEIPT_SPAN_MS
        and book_skew_ms <= MAX_BOOK_SKEW_MS
        and 0 <= book_event_age_ms <= MAX_BOOK_EVENT_AGE_MS
    )
    quote = _quote_payload(
        yes_book=books[YES_TOKEN_ID],
        no_book=books[NO_TOKEN_ID],
        tick_size=tick_size,
        reward_size=reward_size,
        maximum_spread=maximum_spread,
        daily_reward_rate=daily_rate,
    )
    fee_schedule = _mapping(gamma.get("feeSchedule"), name="Gamma fee schedule")
    artifact: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "direction_neutral_paired_binary_maker_reward_diagnostic",
        "started_at_ms": started_ms,
        "completed_at_ms": completed_ms,
        "candidate": {
            "event_id": EVENT_ID,
            "condition_id": CONDITION_ID,
            "question": clob.get("question"),
            "yes_token_id": YES_TOKEN_ID,
            "no_token_id": NO_TOKEN_ID,
            "tick_size": str(tick_size),
            "minimum_order_size": str(clob.get("minimum_order_size")),
            "reward_minimum_size": str(reward_size),
            "reward_maximum_spread_cents": str(maximum_spread_cents),
            "reward_daily_rate": str(daily_rate),
            "market_competitiveness": str(reward.get("market_competitiveness")),
            "maker_fee_zero_explicit": fee_schedule.get("takerOnly") is True,
            "fee_schedule": fee_schedule,
            "negative_risk": True,
            "negative_risk_augmented": True,
        },
        "capture": {
            "receipt_span_ms": receipt_span_ms,
            "book_timestamp_skew_ms": book_skew_ms,
            "oldest_book_event_age_ms": book_event_age_ms,
            "freshness_passed": freshness_passed,
        },
        "conditional_quote_diagnostic": quote,
        "source_contract": {
            "clob_market_request": clob_source,
            "reward_request": reward_source,
            "gamma_event_request": gamma_source,
            "books_request": books_source,
            "clob_market": clob,
            "reward_market": reward,
            "gamma_event": event,
            "books": list(books.values()),
            "methodology_urls": [
                "https://docs.polymarket.com/programs/liquidity-rewards",
                "https://docs.polymarket.com/trading/fees",
                "https://docs.polymarket.com/api-reference/rewards/get-raw-rewards-for-a-specific-market",
            ],
            "implementation": {
                "tool_path": Path(__file__).name,
                "tool_sha256": _sha256(Path(__file__).read_bytes()),
                "module_path": Path(reward_module.__file__).name,
                "module_sha256": _sha256(Path(reward_module.__file__).read_bytes()),
            },
        },
        "verdict": {
            "status": (
                "prospective_queue_fill_and_reward_persistence_evidence_required"
                if freshness_passed
                else "rejected_stale_book_snapshot"
            ),
            "displayed_both_fill_gross_positive": Decimal(
                quote["both_fill_gross_profit"]
            )
            > 0,
            "freshness_passed": freshness_passed,
            "publicly_proven_reward_payout_lower_bound": "0",
            "accepted_edge": False,
            "trading_authority": False,
        },
        "safety": {
            "public_data_only": True,
            "credentials_used": False,
            "orders_placed": False,
            "hypothetical_orders_only": True,
            "event_wide_complete_set_assumed": False,
        },
        "limitations": [
            "The public methodology does not specify how the size-cutoff-adjusted midpoint is constructed; conditional scores use an explicitly labeled post-quote top midpoint.",
            "Public books expose price levels, not maker identities, queue positions, or actual per-maker Qmin; the publicly proven reward payout lower bound is therefore zero.",
            "An instantaneous daily-rate equivalent is not a payout forecast because random sampling, order persistence, epoch aggregation, second normalization, and the one-dollar payout floor remain unobserved.",
            "One-sided fills create directional inventory; the maximum settlement loss can exceed both-fill spread income and conditional reward accrual.",
            "The augmented negative-risk parent is not treated as an exhaustive event-wide payout set; only this binary condition's YES+NO identity is used.",
            "No cancellation latency, adverse selection, fill probability, merge gas, operational outage, capacity, or realized reward evidence is available in this snapshot.",
        ],
    }
    artifact["result_sha256"] = _sha256(_canonical_json(artifact).encode("ascii"))
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    write_bytes_atomic(
        args.output,
        (_canonical_json(result) + "\n").encode("ascii"),
    )
    print(json.dumps(result["verdict"], indent=2))
    print(f"result_sha256={result['result_sha256']}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
