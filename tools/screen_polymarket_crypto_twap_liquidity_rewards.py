"""Run the frozen public 15-minute/4-hour crypto TWAP liquidity-reward screen."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import requests

from simple_ai_trading.paper_execution import BookLevel
from simple_ai_trading.polymarket_liquidity_rewards import (
    paired_buy_economics,
    paired_maker_bid_diagnostic,
)
from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs/model-research/polymarket/crypto-twap-liquidity-reward-screen-contract-v1.json"
)
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


def _canonical(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True, allow_nan=False
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


def _decimal(value: object, *, name: str, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a decimal") from exc
    if not parsed.is_finite() or (positive and parsed <= 0):
        raise ValueError(f"{name} is outside the accepted range")
    return parsed


def _json_list(value: object, *, name: str) -> list[object]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} is not JSON") from exc
    return _list(value, name=name)


def _request(
    session: requests.Session,
    method: str,
    url: str,
    *,
    journal_dir: Path,
    source_name: str,
    params: Sequence[tuple[str, str]] | Mapping[str, str] | None = None,
    json_body: object | None = None,
) -> tuple[object, dict[str, object]]:
    before_ms = time.time_ns() // 1_000_000
    journal_dir.mkdir(parents=True, exist_ok=True)
    intent = {
        "method": method,
        "url": url,
        "params": params,
        "json_body_sha256": (
            None
            if json_body is None
            else _sha256(_canonical(json_body).encode("ascii"))
        ),
        "requested_before_ms": before_ms,
    }
    write_bytes_atomic(
        journal_dir / f"{source_name}.intent.json",
        (_canonical(intent) + "\n").encode("ascii"),
    )
    response = session.request(method, url, params=params, json=json_body, timeout=30)
    after_ms = time.time_ns() // 1_000_000
    write_bytes_atomic(
        journal_dir / f"{source_name}.raw",
        response.content,
    )
    response_meta = {
        "status_code": response.status_code,
        "final_url": response.url,
        "payload_bytes": len(response.content),
        "payload_sha256": _sha256(response.content),
        "requested_before_ms": before_ms,
        "received_after_ms": after_ms,
        "elapsed_ms": after_ms - before_ms,
    }
    write_bytes_atomic(
        journal_dir / f"{source_name}.response.json",
        (_canonical(response_meta) + "\n").encode("ascii"),
    )
    if response.status_code == 429:
        raise RuntimeError("Polymarket rate limit reached; stopped without retry")
    response.raise_for_status()
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise ValueError("response exceeded the frozen byte ceiling")
    decoded = response.json()
    return decoded, {
        "method": method,
        "url": response.url,
        "payload_bytes": len(response.content),
        "payload_sha256": _sha256(response.content),
        "requested_before_ms": before_ms,
        "received_after_ms": after_ms,
        "elapsed_ms": after_ms - before_ms,
    }


def _contract(now: datetime) -> tuple[dict[str, object], str]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="ascii"))
    claimed = str(contract.get("result_sha256") or "")
    body = dict(contract)
    body.pop("result_sha256", None)
    if claimed != _sha256(_canonical(body).encode("ascii")):
        raise ValueError("contract embedded hash does not reconstruct")
    capture = _mapping(contract.get("capture"), name="capture")
    lower = datetime.fromisoformat(str(capture["activation_not_before_utc"]).replace("Z", "+00:00"))
    upper = datetime.fromisoformat(str(capture["activation_not_after_utc"]).replace("Z", "+00:00"))
    if not lower <= now <= upper:
        raise ValueError(f"outside frozen activation window: {now.isoformat()}")
    return contract, _sha256(CONTRACT_PATH.read_bytes())


def _market_rows(raw: object, expected: Mapping[str, tuple[str, int, int]]) -> dict[str, dict[str, object]]:
    rows = [_mapping(row, name="Gamma market") for row in _list(raw, name="Gamma response")]
    by_slug = {str(row.get("slug") or ""): row for row in rows}
    if set(by_slug) != set(expected) or len(rows) != len(expected):
        raise ValueError("Gamma did not return the six exact frozen slugs")
    result: dict[str, dict[str, object]] = {}
    for slug, (asset, start_ms, duration_ms) in expected.items():
        row = by_slug[slug]
        if row.get("active") is not True or row.get("closed") is not False:
            raise ValueError(f"{slug} is not active and open")
        if row.get("enableOrderBook") is not True or row.get("acceptingOrders") is not True:
            raise ValueError(f"{slug} is not accepting CLOB orders")
        event_start = datetime.fromisoformat(str(row["eventStartTime"]).replace("Z", "+00:00"))
        end = datetime.fromisoformat(str(row["endDate"]).replace("Z", "+00:00"))
        if int(event_start.timestamp() * 1000) != start_ms or int((end - event_start).total_seconds() * 1000) != duration_ms:
            raise ValueError(f"{slug} time identity changed")
        outcomes = [str(value) for value in _json_list(row.get("outcomes"), name="outcomes")]
        tokens = [str(value) for value in _json_list(row.get("clobTokenIds"), name="clobTokenIds")]
        fee = _mapping(row.get("feeSchedule"), name="feeSchedule")
        if outcomes != ["Up", "Down"] or len(tokens) != 2 or len(set(tokens)) != 2:
            raise ValueError(f"{slug} outcome identity changed")
        if row.get("feesEnabled") is not True or fee.get("takerOnly") is not True:
            raise ValueError(f"{slug} does not prove a taker-only fee schedule")
        result[slug] = {
            "asset": asset,
            "slug": slug,
            "condition_id": str(row.get("conditionId") or "").lower(),
            "tokens": tokens,
            "tick_size": _decimal(row.get("orderPriceMinTickSize"), name="tick size", positive=True),
            "minimum_order_size": _decimal(row.get("orderMinSize"), name="minimum order size", positive=True),
            "duration_ms": duration_ms,
            "end_ms": int(end.timestamp() * 1000),
            "fee_schedule": fee,
            "gamma_payload": row,
        }
    return result


def _reward(raw: object, market: Mapping[str, object], capture_date: str) -> dict[str, object]:
    payload = _mapping(raw, name="reward response")
    rows = [_mapping(row, name="reward row") for row in _list(payload.get("data"), name="reward data")]
    if len(rows) != 1 or rows[0].get("condition_id") != market["condition_id"]:
        raise ValueError("exact reward identity changed")
    row = rows[0]
    size = _decimal(row.get("rewards_min_size"), name="reward minimum size", positive=True)
    spread = _decimal(row.get("rewards_max_spread"), name="reward maximum spread", positive=True)
    gamma = _mapping(market["gamma_payload"], name="Gamma payload")
    if size != _decimal(gamma.get("rewardsMinSize"), name="Gamma reward size", positive=True):
        raise ValueError("Gamma and CLOB reward sizes disagree")
    if spread != _decimal(gamma.get("rewardsMaxSpread"), name="Gamma reward spread", positive=True):
        raise ValueError("Gamma and CLOB reward spreads disagree")
    if size < _decimal(market["minimum_order_size"], name="minimum order size"):
        raise ValueError("reward size is below the executable order minimum")
    active = []
    for value in _list(row.get("rewards_config"), name="reward configurations"):
        config = _mapping(value, name="reward configuration")
        try:
            start_date = datetime.fromisoformat(str(config.get("start_date"))).date()
            end_date = datetime.fromisoformat(str(config.get("end_date"))).date()
            observed_date = datetime.fromisoformat(capture_date).date()
        except ValueError as exc:
            raise ValueError("reward configuration dates are not ISO dates") from exc
        if start_date <= observed_date <= end_date:
            active.append(config)
    daily_rate = sum(
        (_decimal(config.get("rate_per_day"), name="daily rate", positive=True) for config in active),
        Decimal("0"),
    )
    if not active or daily_rate <= 0:
        raise ValueError("no positive exact active daily reward configuration")
    return {"payload": payload, "minimum_size": size, "maximum_spread": spread, "daily_rate": daily_rate, "active": active}


def _levels(book: Mapping[str, object], side: str) -> tuple[BookLevel, ...]:
    levels = tuple(
        BookLevel(
            price=_decimal(row.get("price"), name=f"{side} price", positive=True),
            quantity=_decimal(row.get("size"), name=f"{side} size", positive=True),
        ).validated()
        for row in (_mapping(value, name=f"{side} level") for value in _list(book.get(side), name=side))
    )
    if not levels or tuple(sorted(levels, key=lambda level: level.price, reverse=side == "asks")) != levels:
        raise ValueError(f"{side} levels are empty or out of CLOB order")
    return levels


def _diagnostic(market: Mapping[str, object], reward: Mapping[str, object], books: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    yes = books[str(_list(market["tokens"], name="tokens")[0])]
    no = books[str(_list(market["tokens"], name="tokens")[1])]
    yes_bids, yes_asks = _levels(yes, "bids"), _levels(yes, "asks")
    no_bids, no_asks = _levels(no, "bids"), _levels(no, "asks")
    tick = _decimal(market["tick_size"], name="tick")
    size = _decimal(reward["minimum_size"], name="size")
    economics = paired_buy_economics(
        yes_price=yes_bids[-1].price + tick,
        no_price=no_bids[-1].price + tick,
        quantity=size,
    )
    base = {
        "quantity": str(size),
        "yes_bid": str(economics.yes_price),
        "no_bid": str(economics.no_price),
        "combined_bid": str(economics.combined_price),
        "both_fill_gross_pUSD": str(economics.both_fill_gross_profit),
        "maximum_orphan_loss_pUSD": str(economics.maximum_orphan_loss),
    }
    if economics.combined_price >= 1:
        return {"status": "rejected_combined_bid_not_below_one", "economics": base}
    if economics.yes_price >= min(yes_asks[-1].price, 1 - economics.no_price) or economics.no_price >= min(no_asks[-1].price, 1 - economics.yes_price):
        return {"status": "rejected_post_quote_cross", "economics": base}
    score = paired_maker_bid_diagnostic(
        yes_bids=yes_bids,
        yes_asks=yes_asks,
        no_bids=no_bids,
        no_asks=no_asks,
        tick_size=tick,
        reward_size=size,
        maximum_spread=_decimal(reward["maximum_spread"], name="spread") / 100,
        daily_reward_rate=_decimal(reward["daily_rate"], name="rate"),
    )
    full_market_fraction = Decimal(str(market["duration_ms"])) / Decimal("86400000")
    conditional_full_market_reward = (
        score.conditional_daily_rate_equivalent_lower_bound * full_market_fraction
    )
    stressed_rewards = {
        str(multiplier): conditional_full_market_reward / Decimal(multiplier)
        for multiplier in (1, 10, 100)
    }
    stress_passed = stressed_rewards["100"] >= economics.maximum_orphan_loss
    return {
        "status": (
            "prospective_capture_candidate_not_an_edge"
            if stress_passed
            else "rejected_100x_competition_stress_does_not_cover_orphan_loss"
        ),
        "economics": base,
        "conditional_top_midpoint_diagnostic": {
            "instantaneous_share_bound": str(score.conditional_instantaneous_share_lower_bound),
            "daily_reward_equivalent_pUSD": str(score.conditional_daily_rate_equivalent_lower_bound),
            "reward_days_to_cover_maximum_orphan_loss": None if score.conditional_reward_days_to_cover_maximum_orphan_loss is None else str(score.conditional_reward_days_to_cover_maximum_orphan_loss),
            "exact_market_duration_fraction_of_day": str(full_market_fraction),
            "full_market_reward_pUSD": str(conditional_full_market_reward),
            "competition_stressed_full_market_rewards_pUSD": {
                multiplier: str(value)
                for multiplier, value in stressed_rewards.items()
            },
            "one_hundred_times_stress_covers_maximum_orphan_loss": stress_passed,
            "publicly_proven_reward_payout_floor_pUSD": "0",
        },
    }


def run(
    *,
    journal_dir: Path,
    session: requests.Session | None = None,
) -> dict[str, object]:
    started = datetime.now(timezone.utc)
    contract, contract_file_sha = _contract(started)
    capture = _mapping(contract["capture"], name="capture")
    expected: dict[str, tuple[str, int, int]] = {}
    for asset in _list(capture["assets"], name="assets"):
        asset_text = str(asset)
        for duration, seconds, duration_ms in (
            ("15m", int(capture["fifteen_minute_epoch_seconds"]), 900_000),
            ("4h", int(capture["four_hour_epoch_seconds"]), 14_400_000),
        ):
            expected[f"{asset_text.lower()}-updown-{duration}-{seconds}"] = (asset_text, seconds * 1000, duration_ms)
    http = session or requests.Session()
    gamma_raw, gamma_source = _request(
        http,
        "GET",
        "https://gamma-api.polymarket.com/markets",
        journal_dir=journal_dir,
        source_name="01-gamma-six-markets",
        params=[*(("slug", slug) for slug in expected), ("closed", "false")],
    )
    markets = _market_rows(gamma_raw, expected)
    reward_sources: dict[str, object] = {}
    rewards: dict[str, dict[str, object]] = {}
    capture_date = started.date().isoformat()
    for slug, market in markets.items():
        raw, source = _request(
            http,
            "GET",
            f"https://clob.polymarket.com/rewards/markets/{market['condition_id']}",
            journal_dir=journal_dir,
            source_name=f"reward-{slug}",
            params={"sponsored": "true"},
        )
        rewards[slug] = _reward(raw, market, capture_date)
        reward_sources[slug] = source
    tokens = [str(token) for market in markets.values() for token in _list(market["tokens"], name="tokens")]
    books_raw, books_source = _request(
        http,
        "POST",
        "https://clob.polymarket.com/books",
        journal_dir=journal_dir,
        source_name="08-twelve-token-books",
        json_body=[{"token_id": token} for token in tokens],
    )
    received_ms = int(books_source["received_after_ms"])
    books = {_mapping(value, name="book")["asset_id"]: _mapping(value, name="book") for value in _list(books_raw, name="books")}
    if set(books) != set(tokens) or len(books) != len(tokens):
        raise ValueError("batch books did not return the twelve exact tokens")
    timestamps = []
    for token, book in books.items():
        market = next(value for value in markets.values() if token in value["tokens"])
        if _decimal(book.get("tick_size"), name="book tick", positive=True) != market["tick_size"] or _decimal(book.get("min_order_size"), name="book min", positive=True) != market["minimum_order_size"]:
            raise ValueError("book execution parameters disagree with Gamma")
        _levels(book, "bids")
        _levels(book, "asks")
        timestamps.append(int(str(book["timestamp"])))
    receipt_span = received_ms - int(gamma_source["requested_before_ms"])
    age = received_ms - min(timestamps)
    skew = max(timestamps) - min(timestamps)
    remaining = min(int(market["end_ms"]) for market in markets.values()) - received_ms
    fresh = 0 <= age <= int(capture["book_max_event_age_ms"]) and skew <= int(capture["book_max_timestamp_skew_ms"]) and receipt_span <= int(capture["receipt_max_span_ms"]) and remaining >= int(capture["minimum_remaining_market_ms"])
    results = []
    for slug, market in markets.items():
        reward = rewards[slug]
        results.append({
            "slug": slug,
            "asset": market["asset"],
            "condition_id": market["condition_id"],
            "tokens": market["tokens"],
            "fee_schedule": market["fee_schedule"],
            "reward": {"minimum_size": str(reward["minimum_size"]), "maximum_spread_cents": str(reward["maximum_spread"]), "daily_rate_pUSD": str(reward["daily_rate"]), "active_configurations": reward["active"]},
            "diagnostic": _diagnostic(market, reward, books),
        })
    candidates = [row["slug"] for row in results if row["diagnostic"]["status"] == "prospective_capture_candidate_not_an_edge"] if fresh else []
    artifact: dict[str, object] = {
        "schema_version": "polymarket-crypto-twap-liquidity-reward-screen-v1",
        "started_at_utc": started.isoformat().replace("+00:00", "Z"),
        "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "capture": {"receipt_span_ms": receipt_span, "oldest_book_age_ms": age, "book_timestamp_skew_ms": skew, "remaining_fifteen_minute_market_ms": remaining, "freshness_passed": fresh},
        "market_results": results,
        "verdict": {"status": "prospective_capture_design_eligible_not_an_edge" if candidates else "rejected_without_resampling", "candidate_slugs": candidates, "accepted_edge": False, "profitability_claim": False, "trading_authority": False, "publicly_proven_reward_payout_floor_pUSD": "0"},
        "authority": {"public_read_only": True, "credentials_used": False, "orders_or_cancellations": 0, "funded_actions": 0},
        "sources": {"contract_path": CONTRACT_PATH.relative_to(ROOT).as_posix(), "contract_file_sha256": contract_file_sha, "contract_result_sha256": contract["result_sha256"], "gamma_request": gamma_source, "reward_requests": reward_sources, "books_request": books_source, "gamma_payload": gamma_raw, "reward_payloads": {slug: reward["payload"] for slug, reward in rewards.items()}, "books_payload": books_raw, "tool_sha256": _sha256(Path(__file__).read_bytes())},
        "limitations": ["A public top-midpoint score is conditional because maker identities queue order persistence random sampling and final epoch normalization are unavailable.", "A one-sided fill is directional orphan exposure.", "A one-time public snapshot cannot prove realized reward after-cost recurrence."],
    }
    artifact["result_sha256"] = _sha256(_canonical(artifact).encode("ascii"))
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--journal-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(journal_dir=args.journal_dir)
    except Exception as exc:
        failure = {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "failed_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "raw_responses_retained_before_validation": True,
        }
        write_bytes_atomic(
            args.journal_dir / "terminal-failure.json",
            (_canonical(failure) + "\n").encode("ascii"),
        )
        raise
    write_bytes_atomic(args.output, (_canonical(result) + "\n").encode("ascii"))
    print(json.dumps(result["verdict"], indent=2))
    print(f"result_sha256={result['result_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
