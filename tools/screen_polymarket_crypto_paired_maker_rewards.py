"""Run one frozen public BTC/ETH/SOL paired-maker reward screen without trading."""

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

import simple_ai_trading.polymarket as polymarket_module
import simple_ai_trading.polymarket_liquidity_rewards as reward_module
from simple_ai_trading.paper_execution import BookLevel
from simple_ai_trading.polymarket import (
    CLOB_BASE_URL,
    GAMMA_MARKETS_URL,
    SUPPORTED_POLYMARKET_ASSETS,
    PolymarketFiveMinuteMarket,
    parse_polymarket_five_minute_market,
    validate_clob_order_book,
)
from simple_ai_trading.polymarket_liquidity_rewards import (
    paired_buy_economics,
    paired_maker_bid_diagnostic,
)
from simple_ai_trading.storage import write_bytes_atomic


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "crypto-paired-maker-reward-screen-contract-v1.json"
)
SCHEMA_VERSION = "polymarket-crypto-paired-maker-reward-screen-v1"
MARKET_DURATION_MS = 300_000
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


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


def _decimal(
    value: object,
    *,
    name: str,
    positive: bool = False,
) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite() or (positive and parsed <= 0):
        raise ValueError(f"{name} must be a positive finite decimal")
    return parsed


def _request(
    session: requests.Session,
    method: str,
    url: str,
    *,
    params: Sequence[tuple[str, str]] | Mapping[str, str] | None = None,
    json_body: object | None = None,
) -> tuple[object, dict[str, object]]:
    before_ms = time.time_ns() // 1_000_000
    response = session.request(
        method,
        url,
        params=params,
        json=json_body,
        timeout=30,
    )
    after_ms = time.time_ns() // 1_000_000
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        raise RuntimeError(
            "Polymarket rate limit reached; stopped without retry"
            + (f"; Retry-After={retry_after}" if retry_after else "")
        )
    response.raise_for_status()
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise ValueError("Polymarket response exceeded the bounded size")
    try:
        decoded = response.json()
    except requests.JSONDecodeError as exc:
        raise ValueError(f"source {url} did not return JSON") from exc
    return decoded, {
        "method": method,
        "url": response.url,
        "raw_response_sha256": _sha256(response.content),
        "canonical_payload_sha256": _sha256(_canonical_json(decoded).encode("ascii")),
        "requested_before_ms": before_ms,
        "received_after_ms": after_ms,
        "request_elapsed_ms": after_ms - before_ms,
    }


def _contract() -> tuple[dict[str, object], str]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="ascii"))
    expected = str(contract.get("result_sha256") or "")
    body = dict(contract)
    body.pop("result_sha256", None)
    actual = _sha256(_canonical_json(body).encode("ascii"))
    if (
        expected != actual
        or contract.get("status") != "frozen_before_first_live_screen"
    ):
        raise ValueError("paired-maker screen contract is not the frozen source")
    return contract, _sha256(CONTRACT_PATH.read_bytes())


def _gamma_markets(
    raw: object,
    *,
    expected_slugs: Sequence[str],
    expected_epoch_ms: int,
) -> tuple[
    tuple[PolymarketFiveMinuteMarket, ...],
    dict[str, dict[str, object]],
]:
    expected = set(expected_slugs)
    rows = [_mapping(row, name="Gamma market") for row in _list(raw, name="Gamma")]
    if len(rows) != 3 or {str(row.get("slug") or "") for row in rows} != expected:
        raise ValueError("Gamma did not return exactly the frozen three slugs")
    markets = tuple(
        sorted(
            (parse_polymarket_five_minute_market(row) for row in rows),
            key=lambda market: market.asset,
        )
    )
    if tuple(market.asset for market in markets) != SUPPORTED_POLYMARKET_ASSETS or any(
        market.event_start_ms != expected_epoch_ms for market in markets
    ):
        raise ValueError("Gamma market asset or epoch contract changed")
    return markets, {
        market.asset: next(row for row in rows if row["slug"] == market.slug)
        for market in markets
    }


def _reward_market(
    raw: object,
    *,
    market: PolymarketFiveMinuteMarket,
    gamma: Mapping[str, object],
    capture_date: str,
) -> dict[str, object]:
    payload = _mapping(raw, name=f"{market.asset} reward response")
    rows = [
        _mapping(row, name=f"{market.asset} reward market")
        for row in _list(payload.get("data"), name=f"{market.asset} reward data")
    ]
    if len(rows) != 1 or rows[0].get("condition_id") != market.condition_id:
        raise ValueError(f"{market.asset} reward identity changed")
    reward = rows[0]
    minimum_size = _decimal(
        reward.get("rewards_min_size"),
        name=f"{market.asset} reward minimum size",
        positive=True,
    )
    maximum_spread_cents = _decimal(
        reward.get("rewards_max_spread"),
        name=f"{market.asset} reward maximum spread",
        positive=True,
    )
    if (
        minimum_size
        != _decimal(
            gamma.get("rewardsMinSize"),
            name=f"{market.asset} Gamma reward size",
            positive=True,
        )
        or maximum_spread_cents
        != _decimal(
            gamma.get("rewardsMaxSpread"),
            name=f"{market.asset} Gamma reward spread",
            positive=True,
        )
        or minimum_size < market.minimum_order_size
    ):
        raise ValueError(f"{market.asset} reward configuration disagrees with Gamma")
    active: list[dict[str, object]] = []
    for value in _list(
        reward.get("rewards_config"),
        name=f"{market.asset} reward configurations",
    ):
        config = _mapping(value, name=f"{market.asset} reward configuration")
        start = str(config.get("start_date") or "")
        end = str(config.get("end_date") or "")
        try:
            is_active = (
                datetime.fromisoformat(start).date().isoformat()
                <= capture_date
                <= datetime.fromisoformat(end).date().isoformat()
            )
        except ValueError as exc:
            raise ValueError(f"{market.asset} reward dates are not ISO dates") from exc
        if is_active:
            active.append(config)
    if not active or len({str(row.get("asset_address") or "") for row in active}) != 1:
        raise ValueError(f"{market.asset} active reward asset is absent or ambiguous")
    daily_rate = sum(
        (
            _decimal(
                row.get("rate_per_day"),
                name=f"{market.asset} daily reward rate",
                positive=True,
            )
            for row in active
        ),
        Decimal("0"),
    )
    return {
        "payload": payload,
        "minimum_size": minimum_size,
        "maximum_spread_cents": maximum_spread_cents,
        "active_configurations": active,
        "daily_rate": daily_rate,
        "reward_asset_address": str(active[0]["asset_address"]),
        "market_competitiveness": reward.get("market_competitiveness"),
    }


def _raw_levels(
    book: Mapping[str, object],
    *,
    side: str,
) -> tuple[BookLevel, ...]:
    levels = tuple(
        BookLevel(
            price=_decimal(row.get("price"), name=f"{side} price", positive=True),
            quantity=_decimal(row.get("size"), name=f"{side} size", positive=True),
        ).validated()
        for row in (
            _mapping(value, name=f"{side} level")
            for value in _list(book.get(side), name=f"book {side}")
        )
    )
    reverse = side == "asks"
    if (
        not levels
        or tuple(sorted(levels, key=lambda level: level.price, reverse=reverse))
        != levels
        or len({level.price for level in levels}) != len(levels)
    ):
        raise ValueError(f"book {side} is empty, duplicated, or out of CLOB order")
    return levels


def _books(
    raw: object,
    *,
    markets: Sequence[PolymarketFiveMinuteMarket],
    received_wall_ms: int,
    received_monotonic_ns: int,
) -> dict[str, dict[str, object]]:
    expected = {token: market for market in markets for token in market.token_ids}
    books: dict[str, dict[str, object]] = {}
    for value in _list(raw, name="books response"):
        book = _mapping(value, name="book")
        token = str(book.get("asset_id") or "")
        if token not in expected or token in books:
            raise ValueError("book token is unexpected or duplicated")
        market = expected[token]
        validate_clob_order_book(
            market,
            token,
            book,
            received_wall_ms=received_wall_ms,
            received_monotonic_ns=received_monotonic_ns,
        )
        if (
            _decimal(book.get("tick_size"), name="book tick size", positive=True)
            != market.tick_size
            or _decimal(
                book.get("min_order_size"),
                name="book minimum order size",
                positive=True,
            )
            != market.minimum_order_size
        ):
            raise ValueError(f"{market.asset} book execution parameters changed")
        _raw_levels(book, side="bids")
        _raw_levels(book, side="asks")
        books[token] = book
    if set(books) != set(expected):
        raise ValueError("all six exact token books are required")
    return books


def _economics_payload(economics: object) -> dict[str, str]:
    return {
        "quantity": str(economics.quantity),
        "up_bid_price": str(economics.yes_price),
        "down_bid_price": str(economics.no_price),
        "combined_bid_price": str(economics.combined_price),
        "both_fill_gross_profit": str(economics.both_fill_gross_profit),
        "up_only_maximum_settlement_loss": str(economics.yes_only_maximum_loss),
        "down_only_maximum_settlement_loss": str(economics.no_only_maximum_loss),
        "maximum_orphan_settlement_loss": str(economics.maximum_orphan_loss),
    }


def _diagnostic(
    *,
    market: PolymarketFiveMinuteMarket,
    reward: Mapping[str, object],
    books: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    up_book = books[market.up_token_id]
    down_book = books[market.down_token_id]
    up_bids = _raw_levels(up_book, side="bids")
    up_asks = _raw_levels(up_book, side="asks")
    down_bids = _raw_levels(down_book, side="bids")
    down_asks = _raw_levels(down_book, side="asks")
    size = _decimal(reward["minimum_size"], name="reward size", positive=True)
    up_quote = up_bids[-1].price + market.tick_size
    down_quote = down_bids[-1].price + market.tick_size
    economics = paired_buy_economics(
        yes_price=up_quote,
        no_price=down_quote,
        quantity=size,
    )
    output: dict[str, object] = {
        "status": "rejected_combined_bid_not_below_one",
        "economics": _economics_payload(economics),
        "conditional_reward_diagnostic": None,
    }
    if economics.combined_price >= 1:
        return output
    post_up_ask = min(up_asks[-1].price, Decimal("1") - down_quote)
    post_down_ask = min(down_asks[-1].price, Decimal("1") - up_quote)
    if up_quote >= post_up_ask or down_quote >= post_down_ask:
        output["status"] = "rejected_post_quote_cross"
        return output
    result = paired_maker_bid_diagnostic(
        yes_bids=up_bids,
        yes_asks=up_asks,
        no_bids=down_bids,
        no_asks=down_asks,
        tick_size=market.tick_size,
        reward_size=size,
        maximum_spread=_decimal(
            reward["maximum_spread_cents"],
            name="maximum reward spread",
            positive=True,
        )
        / 100,
        daily_reward_rate=_decimal(
            reward["daily_rate"],
            name="daily reward rate",
            positive=True,
        ),
    )
    output["status"] = "prospective_capture_design_candidate"
    output["conditional_reward_diagnostic"] = {
        "assumption": "post_quote_top_midpoint_not_venue_adjusted_midpoint",
        "post_quote_up_ask": str(result.post_quote_yes_ask),
        "post_quote_down_ask": str(result.post_quote_no_ask),
        "conditional_up_midpoint": str(result.conditional_yes_midpoint),
        "conditional_down_midpoint": str(result.conditional_no_midpoint),
        "conditional_instantaneous_share_bound": str(
            result.conditional_instantaneous_share_lower_bound
        ),
        "conditional_daily_rate_equivalent": str(
            result.conditional_daily_rate_equivalent_lower_bound
        ),
        "conditional_reward_days_to_cover_maximum_orphan_loss": (
            None
            if result.conditional_reward_days_to_cover_maximum_orphan_loss is None
            else str(result.conditional_reward_days_to_cover_maximum_orphan_loss)
        ),
        "publicly_proven_reward_payout_lower_bound": "0",
    }
    return output


def run(*, session: requests.Session | None = None) -> dict[str, object]:
    """Execute the one-attempt source screen described by the frozen contract."""

    contract, contract_file_sha256 = _contract()
    started_ms = time.time_ns() // 1_000_000
    epoch_ms = started_ms // MARKET_DURATION_MS * MARKET_DURATION_MS
    epoch_seconds = epoch_ms // 1_000
    slugs = tuple(
        f"{asset.lower()}-updown-5m-{epoch_seconds}"
        for asset in SUPPORTED_POLYMARKET_ASSETS
    )
    http = session or requests.Session()
    gamma_raw, gamma_source = _request(
        http,
        "GET",
        GAMMA_MARKETS_URL,
        params=[*(("slug", slug) for slug in slugs), ("closed", "false")],
    )
    markets, gamma_by_asset = _gamma_markets(
        gamma_raw,
        expected_slugs=slugs,
        expected_epoch_ms=epoch_ms,
    )
    capture_date = (
        datetime.fromtimestamp(
            started_ms / 1_000,
            tz=timezone.utc,
        )
        .date()
        .isoformat()
    )
    reward_rows: dict[str, dict[str, object]] = {}
    reward_sources: dict[str, dict[str, object]] = {}
    reward_payloads: dict[str, object] = {}
    for market in markets:
        raw, source = _request(
            http,
            "GET",
            f"{CLOB_BASE_URL}/rewards/markets/{market.condition_id}",
            params={"sponsored": "true"},
        )
        reward_rows[market.asset] = _reward_market(
            raw,
            market=market,
            gamma=gamma_by_asset[market.asset],
            capture_date=capture_date,
        )
        reward_sources[market.asset] = source
        reward_payloads[market.asset] = raw
    requested_tokens = [token for market in markets for token in market.token_ids]
    books_raw, books_source = _request(
        http,
        "POST",
        f"{CLOB_BASE_URL}/books",
        json_body=[{"token_id": token} for token in requested_tokens],
    )
    completed_ms = time.time_ns() // 1_000_000
    books = _books(
        books_raw,
        markets=markets,
        received_wall_ms=int(books_source["received_after_ms"]),
        received_monotonic_ns=time.monotonic_ns(),
    )
    book_timestamps = [int(str(book["timestamp"])) for book in books.values()]
    capture_contract = _mapping(contract["capture"], name="capture contract")
    receipt_span_ms = int(books_source["received_after_ms"]) - int(
        gamma_source["requested_before_ms"]
    )
    book_skew_ms = max(book_timestamps) - min(book_timestamps)
    oldest_book_age_ms = int(books_source["received_after_ms"]) - min(book_timestamps)
    remaining_market_ms = min(market.end_ms for market in markets) - int(
        books_source["received_after_ms"]
    )
    freshness_passed = (
        0 <= oldest_book_age_ms <= int(capture_contract["book_max_event_age_ms"])
        and book_skew_ms <= int(capture_contract["book_max_timestamp_skew_ms"])
        and receipt_span_ms <= int(capture_contract["receipt_max_span_ms"])
        and remaining_market_ms
        >= int(capture_contract["minimum_remaining_market_ms_for_diagnostic"])
    )
    results = []
    for market in markets:
        reward = reward_rows[market.asset]
        results.append(
            {
                "asset": market.asset,
                "market": market.asdict(),
                "reward": {
                    "minimum_size": str(reward["minimum_size"]),
                    "maximum_spread_cents": str(reward["maximum_spread_cents"]),
                    "daily_rate": str(reward["daily_rate"]),
                    "reward_asset_address": reward["reward_asset_address"],
                    "market_competitiveness": reward["market_competitiveness"],
                    "active_configurations": reward["active_configurations"],
                },
                "diagnostic": _diagnostic(
                    market=market,
                    reward=reward,
                    books=books,
                ),
            }
        )
    candidate_assets = [
        row["asset"]
        for row in results
        if row["diagnostic"]["status"] == "prospective_capture_design_candidate"
    ]
    prospective_design_eligible = freshness_passed and bool(candidate_assets)
    artifact: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "direction_neutral_both_fill_crypto_maker_reward_source_screen",
        "started_at_ms": started_ms,
        "completed_at_ms": completed_ms,
        "market_epoch_seconds": epoch_seconds,
        "capture": {
            "receipt_span_ms": receipt_span_ms,
            "book_timestamp_skew_ms": book_skew_ms,
            "oldest_book_event_age_ms": oldest_book_age_ms,
            "remaining_market_ms_at_book_receipt": remaining_market_ms,
            "freshness_passed": freshness_passed,
        },
        "asset_results": results,
        "source_contract": {
            "screen_contract_path": CONTRACT_PATH.relative_to(ROOT).as_posix(),
            "screen_contract_file_sha256": contract_file_sha256,
            "screen_contract_result_sha256": contract["result_sha256"],
            "gamma_request": gamma_source,
            "reward_requests": reward_sources,
            "books_request": books_source,
            "gamma_payload": gamma_raw,
            "reward_payloads": reward_payloads,
            "books_payload": books_raw,
            "implementation": {
                "tool_path": Path(__file__).relative_to(ROOT).as_posix(),
                "tool_sha256": _sha256(Path(__file__).read_bytes()),
                "polymarket_module_path": Path(polymarket_module.__file__)
                .relative_to(ROOT)
                .as_posix(),
                "polymarket_module_sha256": _sha256(
                    Path(polymarket_module.__file__).read_bytes()
                ),
                "reward_module_path": Path(reward_module.__file__)
                .relative_to(ROOT)
                .as_posix(),
                "reward_module_sha256": _sha256(
                    Path(reward_module.__file__).read_bytes()
                ),
            },
        },
        "verdict": {
            "status": (
                "prospective_capture_design_eligible_not_an_edge"
                if prospective_design_eligible
                else "rejected_without_resampling"
            ),
            "candidate_assets": candidate_assets,
            "prospective_capture_design_eligible": prospective_design_eligible,
            "publicly_proven_reward_payout_lower_bound": "0",
            "accepted_edge": False,
            "profitability_claim": False,
            "trading_authority": False,
        },
        "safety": {
            "public_data_only": True,
            "credentials_used": False,
            "orders_placed": False,
            "hypothetical_orders_only": True,
            "next_epoch_or_replacement_market_used": False,
            "request_count": 5,
            "retry_count": 0,
        },
        "limitations": [
            "The venue does not publicly define the size-cutoff-adjusted midpoint construction, so top-midpoint reward values are conditional only.",
            "Public books do not expose maker grouping, queue position, random sample inclusion, order persistence, final normalization, or realized reward payout.",
            "The official rate_per_day field is a daily reward rate, not evidence of a five-minute payout to this hypothetical maker.",
            "One-sided fills leave directional inventory and require prospective fill, cancellation, adverse-selection, and merge evidence.",
            "A point-in-time screen cannot establish after-cost performance or required cross-regime robustness.",
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
