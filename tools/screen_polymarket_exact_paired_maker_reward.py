"""Run one frozen public Polymarket exact-market paired-maker screen."""

from __future__ import annotations

import argparse
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import requests

from simple_ai_trading.polymarket_liquidity_rewards import (
    conservative_instantaneous_share,
    maker_minimum_score,
    minimum_reward_days_to_cover,
    paired_buy_economics,
    reward_order_score,
)
from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs/model-research/polymarket/elon-posts-40-64-paired-maker-reward-contract-v1-2026-08-30.json"
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _list(value: object, *, name: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _json_list(value: object, *, name: str) -> list[object]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} is not JSON") from exc
    return _list(value, name=name)


def _decimal(value: object, *, name: str, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not result.is_finite() or (positive and result <= 0):
        raise ValueError(f"{name} is outside the accepted range")
    return result


def _utc_datetime(value: object, *, end_of_date: bool = False) -> datetime:
    text = str(value)
    try:
        if "T" not in text:
            parsed_date = date.fromisoformat(text)
            parsed = datetime.combine(
                parsed_date + (timedelta(days=1) if end_of_date else timedelta()),
                datetime_time.min,
                tzinfo=timezone.utc,
            )
        else:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid ISO date or timestamp: {text}") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include an offset")
    return parsed.astimezone(timezone.utc)


def _contract(now: datetime) -> tuple[dict[str, Any], str]:
    contract = _mapping(json.loads(CONTRACT_PATH.read_text("ascii")), name="contract")
    claimed = str(contract.get("result_sha256") or "")
    body = dict(contract)
    body.pop("result_sha256", None)
    if claimed != _sha256(_canonical(body)):
        raise ValueError("contract embedded hash does not reconstruct")
    frozen = _utc_datetime(contract["frozen_at_utc"])
    capture = _mapping(contract["capture"], name="capture")
    if frozen > now or now - frozen > timedelta(
        minutes=int(capture["activation_window_minutes"])
    ):
        raise ValueError("frozen contract activation window expired")
    return contract, _sha256(CONTRACT_PATH.read_bytes())


def _request(
    session: requests.Session,
    *,
    method: str,
    url: str,
    params: Mapping[str, str] | None,
    json_body: object | None,
    journal_dir: Path,
    source_name: str,
    byte_ceiling: int,
) -> tuple[object, dict[str, object]]:
    requested_before_ms = time.time_ns() // 1_000_000
    intent = {
        "method": method,
        "url": url,
        "params": params,
        "json_body_sha256": (
            None if json_body is None else _sha256(_canonical(json_body))
        ),
        "requested_before_ms": requested_before_ms,
    }
    prefix = journal_dir / source_name
    write_bytes_atomic(prefix.with_suffix(".intent.json"), _canonical(intent) + b"\n")
    response = session.request(
        method, url, params=params, json=json_body, timeout=30
    )
    received_after_ms = time.time_ns() // 1_000_000
    raw = response.content
    write_bytes_atomic(prefix.with_suffix(".raw"), raw)
    receipt = {
        "method": method,
        "final_url": response.url,
        "status_code": response.status_code,
        "payload_bytes": len(raw),
        "payload_sha256": _sha256(raw),
        "requested_before_ms": requested_before_ms,
        "received_after_ms": received_after_ms,
        "elapsed_ms": received_after_ms - requested_before_ms,
        "retry_after": response.headers.get("Retry-After"),
    }
    write_bytes_atomic(
        prefix.with_suffix(".receipt.json"), _canonical(receipt) + b"\n"
    )
    if response.status_code == 429:
        raise RuntimeError("Polymarket rate limit reached; stopped without retry")
    response.raise_for_status()
    if len(raw) > byte_ceiling:
        raise ValueError("response exceeded frozen byte ceiling")
    try:
        return response.json(), receipt
    except requests.JSONDecodeError as exc:
        raise ValueError("response was not JSON") from exc


def _gamma_market(
    raw: object, *, candidate: Mapping[str, object]
) -> dict[str, Any]:
    rows = [_mapping(row, name="Gamma row") for row in _list(raw, name="Gamma response")]
    if len(rows) != 1:
        raise ValueError("Gamma did not return exactly one frozen market")
    row = rows[0]
    if not (
        row.get("slug") == candidate["market_slug"]
        and row.get("question") == candidate["question"]
        and row.get("active") is True
        and row.get("closed") is False
        and row.get("acceptingOrders") is True
        and row.get("enableOrderBook") is True
    ):
        raise ValueError("Gamma market identity or active-order state changed")
    outcomes = [str(value) for value in _json_list(row.get("outcomes"), name="outcomes")]
    tokens = [str(value) for value in _json_list(row.get("clobTokenIds"), name="tokens")]
    if outcomes != candidate["outcomes"] or len(tokens) != 2 or len(set(tokens)) != 2:
        raise ValueError("Gamma binary token identity changed")
    condition_id = str(row.get("conditionId") or "").lower()
    if not condition_id.startswith("0x") or len(condition_id) != 66:
        raise ValueError("Gamma condition_id is invalid")
    event_end = _utc_datetime(row.get("endDate"), end_of_date=True)
    if event_end.date().isoformat() != candidate["event_end_date_utc"]:
        raise ValueError("Gamma event end date changed")
    fee_schedule = row.get("feeSchedule")
    fee = {} if fee_schedule is None else _mapping(fee_schedule, name="fee schedule")
    maker_fee_zero = row.get("feesEnabled") is False or (
        row.get("feesEnabled") is True and fee.get("takerOnly") is True
    )
    if not maker_fee_zero:
        raise ValueError("Gamma does not establish a zero maker fee")
    return {
        "condition_id": condition_id,
        "tokens": tokens,
        "tick_size": _decimal(
            row.get("orderPriceMinTickSize"), name="Gamma tick size", positive=True
        ),
        "minimum_order_size": _decimal(
            row.get("orderMinSize"), name="Gamma order minimum", positive=True
        ),
        "event_end": event_end,
        "maker_fee_zero": maker_fee_zero,
        "fee_schedule": fee,
        "payload": row,
    }


def _reward(
    raw: object,
    *,
    market: Mapping[str, object],
    candidate: Mapping[str, object],
    now: datetime,
    terminal_cursor: str,
) -> dict[str, Any]:
    payload = _mapping(raw, name="reward response")
    if payload.get("next_cursor") != terminal_cursor:
        raise ValueError("exact reward response did not terminate in one page")
    rows = [_mapping(row, name="reward row") for row in _list(payload.get("data"), name="reward data")]
    if len(rows) != 1 or str(rows[0].get("condition_id") or "").lower() != market["condition_id"]:
        raise ValueError("exact reward response identity changed")
    row = rows[0]
    minimum_size = _decimal(
        row.get("rewards_min_size"), name="reward minimum size", positive=True
    )
    maximum_spread = _decimal(
        row.get("rewards_max_spread"), name="reward maximum spread", positive=True
    )
    discovery = _mapping(candidate["discovery_only"], name="discovery")
    if minimum_size != _decimal(discovery["minimum_size_shares"], name="discovery size"):
        raise ValueError("exact reward minimum size no longer matches discovery")
    if maximum_spread != _decimal(discovery["maximum_spread_cents"], name="discovery spread"):
        raise ValueError("exact reward maximum spread no longer matches discovery")
    if minimum_size < _decimal(market["minimum_order_size"], name="order minimum"):
        raise ValueError("reward size is below executable order minimum")
    active: list[dict[str, Any]] = []
    for value in _list(row.get("rewards_config"), name="reward configurations"):
        config = _mapping(value, name="reward configuration")
        start = _utc_datetime(config.get("start_date"))
        end = _utc_datetime(config.get("end_date"), end_of_date=True)
        if start <= now < end:
            active.append({**config, "_end_utc": end})
    if len(active) != 1:
        raise ValueError("expected exactly one active dated reward configuration")
    daily_rate = _decimal(active[0].get("rate_per_day"), name="daily rate", positive=True)
    if daily_rate != _decimal(discovery["total_daily_reward_pUSD"], name="discovery rate"):
        raise ValueError("exact active daily rate no longer matches discovery")
    return {
        "minimum_size": minimum_size,
        "maximum_spread_cents": maximum_spread,
        "daily_rate": daily_rate,
        "active": active[0],
        "active_end": active[0]["_end_utc"],
        "payload": payload,
    }


def _levels(
    book: Mapping[str, object], *, side: str
) -> tuple[tuple[Decimal, Decimal], ...]:
    rows = tuple(
        (
            _decimal(level.get("price"), name=f"{side} price", positive=True),
            _decimal(level.get("size"), name=f"{side} size", positive=True),
        )
        for level in (
            _mapping(value, name=f"{side} level")
            for value in _list(book.get(side), name=side)
        )
    )
    prices = tuple(price for price, _size in rows)
    if not rows or len(set(prices)) != len(rows):
        raise ValueError(f"{side} levels are empty or duplicate a price")
    if prices != tuple(sorted(prices)) and prices != tuple(sorted(prices, reverse=True)):
        raise ValueError(f"{side} levels are nonmonotone")
    return rows


def _score_levels(
    levels: Sequence[tuple[Decimal, Decimal]],
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


def _diagnostic(
    *,
    market: Mapping[str, object],
    reward: Mapping[str, object],
    books: Mapping[str, Mapping[str, object]],
    stress_multipliers: Sequence[object],
    observed_at: datetime,
) -> dict[str, object]:
    tokens = [str(value) for value in _list(market["tokens"], name="tokens")]
    yes_book, no_book = books[tokens[0]], books[tokens[1]]
    yes_bids, yes_asks = _levels(yes_book, side="bids"), _levels(yes_book, side="asks")
    no_bids, no_asks = _levels(no_book, side="bids"), _levels(no_book, side="asks")
    yes_best_bid, yes_best_ask = max(row[0] for row in yes_bids), min(row[0] for row in yes_asks)
    no_best_bid, no_best_ask = max(row[0] for row in no_bids), min(row[0] for row in no_asks)
    tick = _decimal(market["tick_size"], name="tick", positive=True)
    size = _decimal(reward["minimum_size"], name="reward size", positive=True)
    yes_quote, no_quote = yes_best_bid + tick, no_best_bid + tick
    if yes_quote >= yes_best_ask or no_quote >= no_best_ask:
        raise ValueError("one-tick-improved paired bid would be marketable")
    if yes_quote % tick != 0 or no_quote % tick != 0:
        raise ValueError("hypothetical quote is not tick aligned")
    economics = paired_buy_economics(
        yes_price=yes_quote, no_price=no_quote, quantity=size
    )
    if economics.combined_price >= Decimal("1"):
        raise ValueError("paired bid sum is not below one")
    post_yes_ask = min(yes_best_ask, Decimal("1") - no_quote)
    post_no_ask = min(no_best_ask, Decimal("1") - yes_quote)
    if yes_quote >= post_yes_ask or no_quote >= post_no_ask:
        raise ValueError("paired quotes would cross complementary own asks")
    yes_midpoint = (yes_quote + post_yes_ask) / 2
    no_midpoint = (no_quote + post_no_ask) / 2
    maximum_spread = _decimal(reward["maximum_spread_cents"], name="spread") / 100
    yes_score = reward_order_score(
        maximum_spread=maximum_spread,
        distance=abs(yes_quote - yes_midpoint),
        size=size,
    )
    no_score = reward_order_score(
        maximum_spread=maximum_spread,
        distance=abs(no_quote - no_midpoint),
        size=size,
    )
    own = maker_minimum_score(q_one=yes_score, q_two=no_score, midpoint=yes_midpoint)
    old_q_one = _score_levels(
        yes_bids, midpoint=yes_midpoint, maximum_spread=maximum_spread
    ) + _score_levels(no_asks, midpoint=no_midpoint, maximum_spread=maximum_spread)
    old_q_two = _score_levels(
        yes_asks, midpoint=yes_midpoint, maximum_spread=maximum_spread
    ) + _score_levels(no_bids, midpoint=no_midpoint, maximum_spread=maximum_spread)
    competition = old_q_one + old_q_two
    daily_rate = _decimal(reward["daily_rate"], name="daily rate", positive=True)
    stress: dict[str, object] = {}
    for value in stress_multipliers:
        multiplier = _decimal(value, name="stress multiplier", positive=True)
        share = conservative_instantaneous_share(
            own_minimum_score=own,
            old_aggregate_q_one=old_q_one * multiplier,
            old_aggregate_q_two=old_q_two * multiplier,
        )
        daily_equivalent = daily_rate * share
        payback = minimum_reward_days_to_cover(
            maximum_orphan_loss=economics.maximum_orphan_loss,
            daily_reward_bound=daily_equivalent,
        )
        stress[str(value)] = {
            "conditional_instantaneous_share": str(share),
            "conditional_daily_reward_equivalent_pUSD": str(daily_equivalent),
            "conditional_orphan_payback_days": None if payback is None else str(payback),
        }
    reward_end = _utc_datetime(reward["active_end"])
    event_end = _utc_datetime(market["event_end"])
    remaining_days = Decimal(
        str(max((min(reward_end, event_end) - observed_at).total_seconds(), 0))
    ) / Decimal("86400")
    stressed_payback = stress["100"]["conditional_orphan_payback_days"]
    eligible = (
        economics.both_fill_gross_profit > 0
        and stressed_payback is not None
        and Decimal(str(stressed_payback)) <= remaining_days
    )
    return {
        "status": (
            "prospective_paper_capture_candidate_not_an_edge"
            if eligible
            else "rejected_100x_competition_payback_exceeds_reward_horizon"
        ),
        "top_of_book": {
            "yes_best_bid": str(yes_best_bid),
            "yes_best_ask": str(yes_best_ask),
            "no_best_bid": str(no_best_bid),
            "no_best_ask": str(no_best_ask),
        },
        "hypothetical_quote": {
            "quantity": str(size),
            "yes_bid": str(yes_quote),
            "no_bid": str(no_quote),
            "combined_bid": str(economics.combined_price),
            "both_fill_gross_profit_pUSD": str(economics.both_fill_gross_profit),
            "maximum_orphan_loss_pUSD": str(economics.maximum_orphan_loss),
        },
        "conditional_score": {
            "yes_midpoint": str(yes_midpoint),
            "no_midpoint": str(no_midpoint),
            "own_minimum_score": str(own),
            "old_q_one": str(old_q_one),
            "old_q_two": str(old_q_two),
            "displayed_book_competition": str(competition),
            "stress": stress,
            "remaining_reward_days": str(remaining_days),
            "publicly_proven_reward_payout_floor_pUSD": "0",
        },
    }


def run(*, output: Path, journal_dir: Path) -> dict[str, object]:
    started = datetime.now(timezone.utc)
    contract, contract_file_sha = _contract(started)
    journal_dir.mkdir(parents=True, exist_ok=False)
    candidate = _mapping(contract["candidate"], name="candidate")
    request = _mapping(contract["request_contract"], name="request contract")
    capture = _mapping(contract["capture"], name="capture")
    decision = _mapping(contract["decision"], name="decision")
    byte_ceiling = int(request["response_byte_ceiling"])
    http = requests.Session()
    gamma_raw, gamma_source = _request(
        http,
        method="GET",
        url=str(request["gamma_endpoint"]),
        params=_mapping(request["gamma_params"], name="Gamma params"),
        json_body=None,
        journal_dir=journal_dir,
        source_name="01-exact-gamma-market",
        byte_ceiling=byte_ceiling,
    )
    market = _gamma_market(gamma_raw, candidate=candidate)
    reward_raw, reward_source = _request(
        http,
        method="GET",
        url=str(request["exact_reward_endpoint_pattern"]).format(
            condition_id=market["condition_id"]
        ),
        params=_mapping(request["reward_params"], name="reward params"),
        json_body=None,
        journal_dir=journal_dir,
        source_name="02-exact-sponsored-reward",
        byte_ceiling=byte_ceiling,
    )
    reward = _reward(
        reward_raw,
        market=market,
        candidate=candidate,
        now=started,
        terminal_cursor=str(request["terminal_cursor"]),
    )
    books_body = [{"token_id": token} for token in market["tokens"]]
    books_raw, books_source = _request(
        http,
        method="POST",
        url=str(request["books_endpoint"]),
        params=None,
        json_body=books_body,
        journal_dir=journal_dir,
        source_name="03-two-token-books",
        byte_ceiling=byte_ceiling,
    )
    rows = [_mapping(row, name="book") for row in _list(books_raw, name="books")]
    books = {str(row.get("asset_id") or ""): row for row in rows}
    if len(rows) != 2 or set(books) != set(market["tokens"]):
        raise ValueError("books response did not contain exactly the two frozen tokens")
    timestamps: list[int] = []
    for row in rows:
        if str(row.get("market") or "").lower() != market["condition_id"]:
            raise ValueError("book condition identity changed")
        if _decimal(row.get("tick_size"), name="book tick", positive=True) != market["tick_size"]:
            raise ValueError("book and Gamma tick sizes disagree")
        if _decimal(row.get("min_order_size"), name="book minimum", positive=True) != market["minimum_order_size"]:
            raise ValueError("book and Gamma order minimums disagree")
        _levels(row, side="bids")
        _levels(row, side="asks")
        timestamps.append(int(str(row.get("timestamp"))))
    received_ms = int(books_source["received_after_ms"])
    receipt_span = received_ms - int(gamma_source["requested_before_ms"])
    event_age = received_ms - min(timestamps)
    skew = max(timestamps) - min(timestamps)
    fresh = (
        receipt_span <= int(capture["receipt_max_span_ms"])
        and skew <= int(capture["book_max_timestamp_skew_ms"])
        and 0 <= event_age <= int(capture["book_max_event_age_ms"])
    )
    if not fresh:
        raise ValueError("book capture failed frozen freshness gates")
    observed_at = datetime.fromtimestamp(received_ms / 1000, timezone.utc)
    diagnostic = _diagnostic(
        market=market,
        reward=reward,
        books=books,
        stress_multipliers=_list(decision["stress_multipliers"], name="stress multipliers"),
        observed_at=observed_at,
    )
    active_config = dict(_mapping(reward["active"], name="active configuration"))
    active_config.pop("_end_utc", None)
    artifact: dict[str, object] = {
        "schema_version": "polymarket-exact-paired-maker-reward-screen-v1",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": {
            "event_slug": candidate["event_slug"],
            "market_slug": candidate["market_slug"],
            "question": candidate["question"],
            "condition_id": market["condition_id"],
            "tokens": market["tokens"],
            "maker_fee_zero": market["maker_fee_zero"],
            "fee_schedule": market["fee_schedule"],
            "tick_size": str(market["tick_size"]),
            "minimum_order_size": str(market["minimum_order_size"]),
            "reward_minimum_size": str(reward["minimum_size"]),
            "reward_maximum_spread_cents": str(reward["maximum_spread_cents"]),
            "reward_daily_rate_pUSD": str(reward["daily_rate"]),
            "active_reward_configuration": active_config,
        },
        "capture": {
            "receipt_span_ms": receipt_span,
            "oldest_book_event_age_ms": event_age,
            "book_timestamp_skew_ms": skew,
            "freshness_passed": fresh,
        },
        "diagnostic": diagnostic,
        "verdict": {
            "status": diagnostic["status"],
            "accepted_edge": False,
            "profitability_claim": False,
            "trading_authority": False,
            "publicly_proven_reward_payout_floor_pUSD": "0",
        },
        "authority": {
            "public_unauthenticated_read_only": True,
            "credentials_used": False,
            "orders_or_cancellations": 0,
            "funded_actions": 0,
        },
        "sources": {
            "contract_path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "contract_file_sha256": contract_file_sha,
            "contract_result_sha256": contract["result_sha256"],
            "gamma_request": gamma_source,
            "reward_request": reward_source,
            "books_request": books_source,
            "tool_sha256": _sha256(Path(__file__).read_bytes()),
        },
        "limitations": [
            "Public books do not reveal maker identities, queue position, persistence, random samples, final epoch normalization, fills, or realized rewards.",
            "The score uses an explicitly conditional post-quote top midpoint because the public size-cutoff-adjusted midpoint construction is not fully specified.",
            "A one-sided fill creates directional inventory and the public snapshot proves no positive reward payout floor.",
            "No adverse-selection, cancellation-latency, capacity, outage, tax, custody, or realized after-cost evidence is available from this snapshot.",
        ],
    }
    artifact["result_sha256"] = _sha256(_canonical(artifact))
    write_bytes_atomic(output, _canonical(artifact) + b"\n")
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--journal-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(output=args.output, journal_dir=args.journal_dir)
    except Exception as exc:
        args.journal_dir.mkdir(parents=True, exist_ok=True)
        failure = {
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "raw_responses_retained_before_validation": True,
            "retry_permitted": False,
        }
        write_bytes_atomic(
            args.journal_dir / "terminal-failure.json", _canonical(failure) + b"\n"
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
