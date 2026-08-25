"""Screen current BTC/ETH/SOL Polymarket negative-risk payoff parity."""

from __future__ import annotations

import argparse
from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import time
from typing import Mapping, Sequence

import requests

import simple_ai_trading.structural_parity as structural_parity_module
from simple_ai_trading.paper_execution import BookLevel
from simple_ai_trading.polymarket_fees import PolymarketFeeModel
from simple_ai_trading.storage import write_bytes_atomic
from simple_ai_trading.structural_parity import (
    NegativeRiskOutcome,
    NegativeRiskParityPath,
    NegativeRiskParityScreen,
    screen_negative_risk_parity,
)


SCHEMA_VERSION = "polymarket-negative-risk-structural-parity-screen-v1"
GAMMA_BASE_URL = "https://gamma-api.polymarket.com"
CLOB_BASE_URL = "https://clob.polymarket.com"
POLYGON_RPC_URL = "https://polygon-bor-rpc.publicnode.com"
ADAPTER_REPOSITORY = "Polymarket/neg-risk-ctf-adapter"
ASSET_TAGS = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana"}
QUESTION_COUNT_SELECTOR = "0xb7f75d2c"  # getQuestionCount(bytes32)
FEE_BIPS_SELECTOR = "0x2582cb5e"  # getFeeBips(bytes32)
ZERO_FEE = PolymarketFeeModel(False, Decimal("0"), 1, True)


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


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


class _SourceClient:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "simple-ai-trading-structural-research/1.0",
            }
        )

    def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, object] | None = None,
        body: object | None = None,
    ) -> tuple[object, str]:
        response = self.session.request(
            method,
            url,
            params=params,
            json=body,
            timeout=30,
        )
        response.raise_for_status()
        payload = response.content
        try:
            decoded = response.json()
        except requests.JSONDecodeError as exc:
            raise ValueError(f"source {url} did not return JSON") from exc
        return decoded, _sha256(payload)


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
        raise ValueError(f"{name} must be encoded JSON")
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is invalid JSON") from exc
    return _list(decoded, name=name)


def _resolve_pinned_adapter(client: _SourceClient) -> dict[str, object]:
    commit_url = f"https://api.github.com/repos/{ADAPTER_REPOSITORY}/commits/main"
    commit_payload, commit_hash = client.request("GET", commit_url)
    commit = _mapping(commit_payload, name="adapter repository commit")
    sha = str(commit.get("sha") or "").lower()
    if len(sha) != 40 or any(char not in "0123456789abcdef" for char in sha):
        raise ValueError("adapter repository commit SHA is invalid")
    addresses_url = (
        f"https://raw.githubusercontent.com/{ADAPTER_REPOSITORY}/{sha}/addresses.json"
    )
    addresses_payload, addresses_hash = client.request("GET", addresses_url)
    addresses = _mapping(addresses_payload, name="adapter addresses")
    polygon = _mapping(addresses.get("137"), name="Polygon adapter addresses")
    adapter = str(polygon.get("negRiskAdapter") or "")
    if len(adapter) != 42 or not adapter.lower().startswith("0x"):
        raise ValueError("Polygon negative-risk adapter address is invalid")
    return {
        "repository_commit_sha": sha,
        "commit_url": commit_url,
        "commit_payload_sha256": commit_hash,
        "addresses_url": addresses_url,
        "addresses_payload_sha256": addresses_hash,
        "adapter_address": adapter,
    }


def _rpc(
    client: _SourceClient,
    *,
    method: str,
    params: Sequence[object],
) -> tuple[object, str]:
    payload, payload_hash = client.request(
        "POST",
        POLYGON_RPC_URL,
        body={"jsonrpc": "2.0", "id": 1, "method": method, "params": list(params)},
    )
    body = _mapping(payload, name=f"Polygon {method} response")
    if body.get("error") is not None or "result" not in body:
        raise ValueError(f"Polygon {method} failed")
    return body["result"], payload_hash


def _onchain_uint(
    client: _SourceClient,
    *,
    adapter: str,
    selector: str,
    market_id: str,
    block_tag: str,
) -> tuple[int, str]:
    if (
        len(market_id) != 66
        or not market_id.startswith("0x")
        or any(char not in "0123456789abcdef" for char in market_id[2:].lower())
    ):
        raise ValueError("negative-risk market ID is invalid")
    result, payload_hash = _rpc(
        client,
        method="eth_call",
        params=(
            {"to": adapter, "data": selector + market_id[2:]},
            block_tag,
        ),
    )
    if not isinstance(result, str) or not result.startswith("0x"):
        raise ValueError("Polygon eth_call result is invalid")
    return int(result, 16), payload_hash


def _discover_events(client: _SourceClient) -> tuple[list[str], dict[str, object]]:
    seen: set[str] = set()
    fixed: set[str] = set()
    augmented: set[str] = set()
    tag_evidence: list[dict[str, object]] = []
    for asset, slug in ASSET_TAGS.items():
        tag_url = f"{GAMMA_BASE_URL}/tags/slug/{slug}"
        tag_payload, tag_hash = client.request("GET", tag_url)
        tag = _mapping(tag_payload, name=f"{asset} tag")
        if str(tag.get("slug") or "").lower() != slug:
            raise ValueError(f"{asset} tag identity differs")
        tag_id = str(tag.get("id") or "")
        if not tag_id.isdigit():
            raise ValueError(f"{asset} tag ID is invalid")
        offset = 0
        page_hashes: list[str] = []
        tag_event_count = 0
        while True:
            page_payload, page_hash = client.request(
                "GET",
                f"{GAMMA_BASE_URL}/events",
                params={
                    "active": "true",
                    "closed": "false",
                    "tag_id": tag_id,
                    "limit": 100,
                    "offset": offset,
                    "order": "volume24hr",
                    "ascending": "false",
                },
            )
            page = _list(page_payload, name=f"{asset} event page")
            page_hashes.append(page_hash)
            tag_event_count += len(page)
            for raw_event in page:
                event = _mapping(raw_event, name=f"{asset} event")
                event_id = str(event.get("id") or "")
                if not event_id.isdigit():
                    raise ValueError(f"{asset} event ID is invalid")
                if event.get("active") is not True or event.get("closed") is not False:
                    raise ValueError(f"Gamma {asset} open-event filter drifted")
                seen.add(event_id)
                if event.get("negRisk") is True:
                    if event.get("negRiskAugmented") is True:
                        augmented.add(event_id)
                    else:
                        fixed.add(event_id)
            if len(page) < 100:
                break
            offset += 100
            if offset >= 1_000:
                raise ValueError(f"{asset} tag exceeded the bounded Gamma offset")
        tag_evidence.append(
            {
                "asset": asset,
                "slug": slug,
                "tag_id": tag_id,
                "tag_url": tag_url,
                "tag_payload_sha256": tag_hash,
                "event_count_with_duplicates": tag_event_count,
                "event_page_sha256": page_hashes,
            }
        )
    if fixed & augmented:
        raise ValueError("fixed and augmented negative-risk universes overlap")
    return sorted(fixed, key=int), {
        "unique_scoped_active_event_count": len(seen),
        "fixed_negative_risk_event_count": len(fixed),
        "excluded_augmented_negative_risk_event_count": len(augmented),
        "excluded_augmented_event_ids": sorted(augmented, key=int),
        "tags": tag_evidence,
    }


def _levels(raw_book: Mapping[str, object], side: str) -> tuple[BookLevel, ...]:
    raw_levels = _list(raw_book.get(side), name=f"book {side}")
    levels = tuple(
        BookLevel(
            price=Decimal(str(_mapping(level, name="book level").get("price"))),
            quantity=Decimal(str(_mapping(level, name="book level").get("size"))),
        ).validated()
        for level in raw_levels
    )
    descending = side == "bids"
    ordered = tuple(sorted(levels, key=lambda level: level.price, reverse=descending))
    if len({level.price for level in ordered}) != len(ordered):
        raise ValueError("CLOB book contains duplicate price levels")
    return ordered


def _path_payload(path: NegativeRiskParityPath | None) -> dict[str, object] | None:
    if path is None:
        return None
    return {
        "mechanism": path.mechanism,
        "selected_no_outcomes": list(path.selected_no_outcomes),
        "net_quote": _decimal_text(path.net_quote),
        "taker_fees_quote": _decimal_text(path.taker_fees_quote),
        "initial_outlay_quote": _decimal_text(path.initial_outlay_quote),
    }


def _screen_payload(screen: NegativeRiskParityScreen) -> dict[str, object]:
    return {
        "quantity": _decimal_text(screen.quantity),
        "evaluated_path_count": screen.evaluated_path_count,
        "executable_path_count": screen.executable_path_count,
        "profitable_path_count": screen.profitable_path_count,
        "buy_all_yes_hold": _path_payload(screen.buy_all_yes_hold),
        "mint_all_yes_sell": _path_payload(screen.mint_all_yes_sell),
        "best_no_conversion": _path_payload(screen.best_no_conversion),
        "best_path": _path_payload(screen.best_path),
    }


def _screen_event(
    client: _SourceClient,
    *,
    event_id: str,
    adapter: str,
    block_tag: str,
    quantity: Decimal,
) -> dict[str, object]:
    event_url = f"{GAMMA_BASE_URL}/events/{event_id}"
    event_payload, event_hash = client.request("GET", event_url)
    event = _mapping(event_payload, name=f"event {event_id}")
    if (
        str(event.get("id") or "") != event_id
        or event.get("active") is not True
        or event.get("closed") is not False
        or event.get("negRisk") is not True
        or event.get("enableNegRisk") is not True
        or event.get("negRiskAugmented") is not False
    ):
        raise ValueError(f"event {event_id} identity or fixed state differs")
    market_id = str(event.get("negRiskMarketID") or "").lower()
    raw_markets = _list(event.get("markets"), name=f"event {event_id} markets")
    question_count, question_hash = _onchain_uint(
        client,
        adapter=adapter,
        selector=QUESTION_COUNT_SELECTOR,
        market_id=market_id,
        block_tag=block_tag,
    )
    fee_bips, fee_hash = _onchain_uint(
        client,
        adapter=adapter,
        selector=FEE_BIPS_SELECTOR,
        market_id=market_id,
        block_tag=block_tag,
    )
    if question_count != len(raw_markets):
        raise ValueError(f"event {event_id} on-chain question count differs")
    if fee_bips != 0:
        raise ValueError(f"event {event_id} has an unsupported conversion fee")

    parsed_markets: list[dict[str, object]] = []
    requested_tokens: list[str] = []
    for raw_market in raw_markets:
        market = _mapping(raw_market, name=f"event {event_id} market")
        gamma_market_id = str(market.get("id") or "")
        condition_id = str(market.get("conditionId") or "").lower()
        if (
            not gamma_market_id.isdigit()
            or market.get("active") is not True
            or market.get("closed") is not False
            or market.get("acceptingOrders") is not True
            or market.get("enableOrderBook") is not True
            or market.get("negRisk") is not True
            or str(market.get("negRiskMarketID") or "").lower() != market_id
            or len(condition_id) != 66
        ):
            raise ValueError(f"event {event_id} market state or identity differs")
        outcomes = _json_list(market.get("outcomes"), name="market outcomes")
        tokens = [
            str(value)
            for value in _json_list(market.get("clobTokenIds"), name="market token IDs")
        ]
        if outcomes != ["Yes", "No"] or len(tokens) != 2 or len(set(tokens)) != 2:
            raise ValueError(f"event {event_id} token mapping differs")
        schedule = _mapping(market.get("feeSchedule"), name="market fee schedule")
        fees_enabled = market.get("feesEnabled") is True
        fee_rate = Decimal(str(schedule.get("rate")))
        exponent_decimal = Decimal(str(schedule.get("exponent")))
        taker_only = schedule.get("takerOnly") is True
        if (
            not fee_rate.is_finite()
            or fee_rate < 0
            or fee_rate > 1
            or not exponent_decimal.is_finite()
            or exponent_decimal <= 0
            or exponent_decimal != exponent_decimal.to_integral_value()
            or (fees_enabled and (fee_rate <= 0 or not taker_only))
            or (not fees_enabled and fee_rate != 0)
        ):
            raise ValueError(f"event {event_id} fee schedule is unsupported")
        minimum_order_size = Decimal(str(market.get("orderMinSize")))
        tick_size = Decimal(str(market.get("orderPriceMinTickSize")))
        taker_base_fee = market.get("takerBaseFee")
        if (
            not minimum_order_size.is_finite()
            or minimum_order_size <= 0
            or minimum_order_size > quantity
            or not tick_size.is_finite()
            or tick_size <= 0
            or isinstance(taker_base_fee, bool)
            or not isinstance(taker_base_fee, int)
            or taker_base_fee < 0
        ):
            raise ValueError(f"event {event_id} market execution terms are unsupported")
        fee_model = PolymarketFeeModel(
            enabled=fees_enabled,
            rate=fee_rate,
            exponent=int(exponent_decimal),
            taker_only=taker_only,
        )
        label = str(market.get("groupItemTitle") or market.get("question") or "")
        parsed_markets.append(
            {
                "gamma_market_id": gamma_market_id,
                "condition_id": condition_id,
                "label": label,
                "yes_token_id": tokens[0],
                "no_token_id": tokens[1],
                "fee_model": fee_model,
                "taker_base_fee": taker_base_fee,
                "minimum_order_size": minimum_order_size,
                "tick_size": tick_size,
            }
        )
        requested_tokens.extend(tokens)
    if len(set(requested_tokens)) != len(requested_tokens):
        raise ValueError(f"event {event_id} token IDs are duplicated")

    books_url = f"{CLOB_BASE_URL}/books"
    requested_before_ms = time.time_ns() // 1_000_000
    books_payload, books_hash = client.request(
        "POST",
        books_url,
        body=[{"token_id": token} for token in requested_tokens],
    )
    received_after_ms = time.time_ns() // 1_000_000
    raw_books = _list(books_payload, name=f"event {event_id} books")
    books = {
        str(_mapping(book, name="CLOB book").get("asset_id")): _mapping(
            book, name="CLOB book"
        )
        for book in raw_books
    }
    if set(books) != set(requested_tokens) or len(raw_books) != len(requested_tokens):
        raise ValueError(f"event {event_id} CLOB book identities differ")

    outcomes: list[NegativeRiskOutcome] = []
    book_timestamps: list[int] = []
    for market in parsed_markets:
        yes = books[str(market["yes_token_id"])]
        no = books[str(market["no_token_id"])]
        if (
            str(yes.get("market") or "").lower() != market["condition_id"]
            or str(no.get("market") or "").lower() != market["condition_id"]
            or yes.get("neg_risk") is not True
            or no.get("neg_risk") is not True
            or Decimal(str(yes.get("min_order_size"))) != market["minimum_order_size"]
            or Decimal(str(no.get("min_order_size"))) != market["minimum_order_size"]
            or Decimal(str(yes.get("tick_size"))) != market["tick_size"]
            or Decimal(str(no.get("tick_size"))) != market["tick_size"]
        ):
            raise ValueError(f"event {event_id} CLOB condition identity differs")
        book_timestamps.extend(
            (int(yes.get("timestamp", -1)), int(no.get("timestamp", -1)))
        )
        outcomes.append(
            NegativeRiskOutcome(
                label=str(market["label"]),
                yes_bids=_levels(yes, "bids"),
                yes_asks=_levels(yes, "asks"),
                no_asks=_levels(no, "asks"),
                fee_model=market["fee_model"],  # type: ignore[arg-type]
            ).validated()
        )
    if min(book_timestamps) < 0:
        raise ValueError(f"event {event_id} CLOB timestamp is invalid")

    gross_outcomes = tuple(replace(outcome, fee_model=ZERO_FEE) for outcome in outcomes)
    gross = screen_negative_risk_parity(
        gross_outcomes,
        quantity=quantity,
        conversion_fee_bips=fee_bips,
    )
    after_cost = screen_negative_risk_parity(
        outcomes,
        quantity=quantity,
        conversion_fee_bips=fee_bips,
    )
    fee_audit: dict[str, object] | None = None
    if gross.best_path is not None and gross.best_path.net_quote >= 0:
        fee_hashes: list[str] = []
        for market in parsed_markets:
            for token_key in ("yes_token_id", "no_token_id"):
                token = str(market[token_key])
                payload, payload_hash = client.request(
                    "GET", f"{CLOB_BASE_URL}/fee-rate/{token}"
                )
                fee = _mapping(payload, name="CLOB fee-rate response")
                if fee.get("base_fee") != market["taker_base_fee"]:
                    raise ValueError(f"event {event_id} token fee-rate differs")
                fee_hashes.append(payload_hash)
        fee_audit = {
            "audited_token_count": len(requested_tokens),
            "fee_rate_payload_sha256": fee_hashes,
        }
    return {
        "event_id": event_id,
        "title": str(event.get("title") or ""),
        "slug": str(event.get("slug") or ""),
        "end_date": str(event.get("endDate") or ""),
        "negative_risk_market_id": market_id,
        "outcome_count": len(outcomes),
        "onchain_question_count": question_count,
        "onchain_conversion_fee_bips": fee_bips,
        "onchain_question_count_payload_sha256": question_hash,
        "onchain_fee_bips_payload_sha256": fee_hash,
        "event_url": event_url,
        "event_payload_sha256": event_hash,
        "books_url": books_url,
        "books_payload_sha256": books_hash,
        "books_requested_before_ms": requested_before_ms,
        "books_received_after_ms": received_after_ms,
        "books_request_elapsed_ms": received_after_ms - requested_before_ms,
        "book_timestamp_skew_ms": max(book_timestamps) - min(book_timestamps),
        "gross_optimistic_screen": _screen_payload(gross),
        "gamma_fee_schedule_screen": _screen_payload(after_cost),
        "clob_fee_rate_audit": fee_audit,
    }


def run(*, quantity: Decimal) -> dict[str, object]:
    client = _SourceClient()
    started_ms = time.time_ns() // 1_000_000
    adapter = _resolve_pinned_adapter(client)
    block_result, block_hash = _rpc(client, method="eth_blockNumber", params=())
    if not isinstance(block_result, str) or not block_result.startswith("0x"):
        raise ValueError("Polygon block number is invalid")
    event_ids, universe = _discover_events(client)
    events = [
        _screen_event(
            client,
            event_id=event_id,
            adapter=str(adapter["adapter_address"]),
            block_tag=block_result,
            quantity=quantity,
        )
        for event_id in event_ids
    ]
    gross_positive = [
        event["event_id"]
        for event in events
        if (
            isinstance(event["gross_optimistic_screen"], Mapping)
            and isinstance(event["gross_optimistic_screen"].get("best_path"), Mapping)
            and Decimal(str(event["gross_optimistic_screen"]["best_path"]["net_quote"]))
            > 0
        )
    ]
    after_cost_positive = [
        event["event_id"]
        for event in events
        if (
            isinstance(event["gamma_fee_schedule_screen"], Mapping)
            and event["gamma_fee_schedule_screen"].get("profitable_path_count", 0) > 0
        )
    ]
    result: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "target_free_market_direction_independent_payoff_parity_screen",
        "started_at_ms": started_ms,
        "completed_at_ms": time.time_ns() // 1_000_000,
        "scope": {
            "assets": list(ASSET_TAGS),
            "quantity": _decimal_text(quantity),
            "augmented_negative_risk_events": "excluded_fail_closed",
        },
        "source_contract": {
            "gamma_base_url": GAMMA_BASE_URL,
            "clob_base_url": CLOB_BASE_URL,
            "polygon_rpc_url": POLYGON_RPC_URL,
            "adapter": adapter,
            "polygon_block_tag": block_result,
            "polygon_block_number": int(block_result, 16),
            "polygon_block_payload_sha256": block_hash,
            "fee_rule": "Gamma feeSchedule; every token is cross-checked with CLOB fee-rate when the optimistic gross best path is nonnegative",
            "implementation": {
                "tool_path": Path(__file__).name,
                "tool_sha256": _sha256(Path(__file__).read_bytes()),
                "module_path": Path(structural_parity_module.__file__).name,
                "module_sha256": _sha256(
                    Path(structural_parity_module.__file__).read_bytes()
                ),
            },
        },
        "universe": universe,
        "events": events,
        "verdict": {
            "status": (
                "diagnostic_positive_requires_atomicity_latency_and_gas_validation"
                if after_cost_positive
                else "rejected_current_snapshot_no_positive_after_cost_parity"
            ),
            "screened_event_count": len(events),
            "gross_positive_event_ids": gross_positive,
            "after_cost_positive_event_ids": after_cost_positive,
            "edge_claim": False,
            "trading_authority": False,
        },
        "limitations": [
            "A snapshot cannot establish opportunity frequency or persistence.",
            "Positive paths would still require atomicity, latency, gas, fill, inventory, capacity, and settlement validation.",
            "Public books do not prove fills and this tool does not place orders.",
        ],
    }
    result["result_sha256"] = _sha256(_canonical_json(result).encode("ascii"))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantity", default="5")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    quantity = Decimal(str(args.quantity))
    result = run(quantity=quantity)
    payload = (_canonical_json(result) + "\n").encode("ascii")
    write_bytes_atomic(args.output, payload)
    print(json.dumps(result["verdict"], indent=2))
    print(f"result_sha256={result['result_sha256']}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
