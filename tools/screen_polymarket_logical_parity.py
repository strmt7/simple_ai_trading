"""Screen exact BTC/ETH/SOL Polymarket implication bundles at displayed depth."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Mapping, Sequence

import simple_ai_trading.logical_parity as logical_parity_module
from simple_ai_trading.logical_parity import (
    CryptoThresholdQuestion,
    LogicalBinaryOutcome,
    LogicalImplicationBundle,
    parse_crypto_threshold_question,
    screen_logical_implication_bundle,
)
from simple_ai_trading.polymarket_fees import PolymarketFeeModel
from simple_ai_trading.storage import write_bytes_atomic

if __package__:
    from tools.screen_polymarket_structural_parity import (
        ASSET_TAGS,
        CLOB_BASE_URL,
        GAMMA_BASE_URL,
        _SourceClient,
        _canonical_json,
        _decimal_text,
        _json_list,
        _levels,
        _list,
        _mapping,
        _sha256,
    )
    import tools.screen_polymarket_structural_parity as structural_source_helpers
else:
    from screen_polymarket_structural_parity import (
        ASSET_TAGS,
        CLOB_BASE_URL,
        GAMMA_BASE_URL,
        _SourceClient,
        _canonical_json,
        _decimal_text,
        _json_list,
        _levels,
        _list,
        _mapping,
        _sha256,
    )
    import screen_polymarket_structural_parity as structural_source_helpers


SCHEMA_VERSION = "polymarket-logical-implication-parity-screen-v1"
ZERO_FEE = PolymarketFeeModel(False, Decimal("0"), 1, True)
_DEADLINE_ITEM = re.compile(
    r"^(?:by )?(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December) [0-9]{1,2}(?:, [0-9]{4})?$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class _Market:
    event_id: str
    market_id: str
    condition_id: str
    question: str
    description: str
    end_date: str
    group_item_title: str
    threshold: CryptoThresholdQuestion | None
    deadline_stem: str | None
    yes_token_id: str
    no_token_id: str
    fee_model: PolymarketFeeModel
    fee_schedule_key: str
    taker_base_fee: int
    minimum_order_size: Decimal
    tick_size: Decimal


def _deadline_stem(question: str, group_item_title: str) -> str | None:
    item = str(group_item_title or "").strip()
    text = str(question or "").strip()
    if not _DEADLINE_ITEM.fullmatch(item) or not text.endswith(item + "?"):
        return None
    stem = " ".join(text[: -len(item + "?")].strip().lower().split())
    if not stem or (not stem.endswith("by") and not item.lower().startswith("by ")):
        return None
    return stem


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("logical market end date is invalid") from exc
    if parsed.tzinfo is None:
        raise ValueError("logical market end date lacks a timezone")
    return parsed


def _parse_market(
    raw_market: object,
    *,
    event_id: str,
    quantity: Decimal,
) -> _Market | None:
    market = _mapping(raw_market, name=f"event {event_id} market")
    if not (
        market.get("active") is True
        and market.get("closed") is False
        and market.get("acceptingOrders") is True
        and market.get("enableOrderBook") is True
        and market.get("negRisk") is False
    ):
        return None
    question = str(market.get("question") or "")
    group_item_title = str(market.get("groupItemTitle") or "")
    threshold = parse_crypto_threshold_question(question)
    deadline_stem = _deadline_stem(question, group_item_title)
    if threshold is None and deadline_stem is None:
        return None

    market_id = str(market.get("id") or "")
    condition_id = str(market.get("conditionId") or "").lower()
    outcomes = _json_list(market.get("outcomes"), name="logical market outcomes")
    tokens = [
        str(value)
        for value in _json_list(
            market.get("clobTokenIds"),
            name="logical market token IDs",
        )
    ]
    if (
        not market_id.isdigit()
        or len(condition_id) != 66
        or not condition_id.startswith("0x")
        or outcomes != ["Yes", "No"]
        or len(tokens) != 2
        or len(set(tokens)) != 2
    ):
        raise ValueError(f"event {event_id} logical market identity differs")

    if not isinstance(market.get("feeSchedule"), Mapping):
        return None
    schedule = _mapping(market.get("feeSchedule"), name="logical fee schedule")
    enabled = market.get("feesEnabled") is True
    rate = Decimal(str(schedule.get("rate")))
    exponent = Decimal(str(schedule.get("exponent")))
    taker_only = schedule.get("takerOnly") is True
    if (
        not rate.is_finite()
        or rate < 0
        or rate > 1
        or not exponent.is_finite()
        or exponent <= 0
        or exponent != exponent.to_integral_value()
        or (enabled and (rate <= 0 or not taker_only))
        or (not enabled and rate != 0)
    ):
        raise ValueError(f"event {event_id} logical fee schedule is unsupported")
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
        raise ValueError(f"event {event_id} logical execution terms are unsupported")
    end_date = str(market.get("endDate") or "")
    if not end_date:
        return None
    _parse_timestamp(end_date)
    return _Market(
        event_id=event_id,
        market_id=market_id,
        condition_id=condition_id,
        question=question,
        description=str(market.get("description") or ""),
        end_date=end_date,
        group_item_title=group_item_title,
        threshold=threshold,
        deadline_stem=deadline_stem,
        yes_token_id=tokens[0],
        no_token_id=tokens[1],
        fee_model=PolymarketFeeModel(enabled, rate, int(exponent), taker_only),
        fee_schedule_key=_canonical_json(schedule),
        taker_base_fee=taker_base_fee,
        minimum_order_size=minimum_order_size,
        tick_size=tick_size,
    )


def _discover_events(
    client: _SourceClient,
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    seen_event_ids: set[str] = set()
    candidate_event_ids: set[str] = set()
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
        page_hashes: list[str] = []
        page_counts: list[int] = []
        for offset in range(0, 1_000, 100):
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
            page_counts.append(len(page))
            for raw_event in page:
                event = _mapping(raw_event, name=f"{asset} event")
                event_id = str(event.get("id") or "")
                if (
                    not event_id.isdigit()
                    or event.get("active") is not True
                    or event.get("closed") is not False
                ):
                    raise ValueError(f"Gamma {asset} open-event filter drifted")
                seen_event_ids.add(event_id)
                for raw_market in _list(
                    event.get("markets"),
                    name=f"Gamma event {event_id} markets",
                ):
                    market = _mapping(raw_market, name="Gamma embedded market")
                    question = str(market.get("question") or "")
                    item = str(market.get("groupItemTitle") or "")
                    if parse_crypto_threshold_question(question) is not None or (
                        _deadline_stem(question, item) is not None
                    ):
                        candidate_event_ids.add(event_id)
                        break
            if len(page) < 100:
                break
        else:
            raise ValueError(f"{asset} tag exceeded the bounded Gamma offset")
        tag_evidence.append(
            {
                "asset": asset,
                "slug": slug,
                "tag_id": tag_id,
                "tag_url": tag_url,
                "page_counts": page_counts,
                "page_payload_sha256": page_hashes,
            }
        )
    events: dict[str, dict[str, object]] = {}
    event_evidence: list[dict[str, object]] = []
    for event_id in sorted(candidate_event_ids, key=int):
        event_url = f"{GAMMA_BASE_URL}/events/{event_id}"
        event_payload, event_hash = client.request("GET", event_url)
        event = _mapping(event_payload, name=f"canonical event {event_id}")
        if (
            str(event.get("id") or "") != event_id
            or event.get("active") is not True
            or event.get("closed") is not False
        ):
            raise ValueError(f"canonical event {event_id} identity differs")
        events[event_id] = event
        event_evidence.append(
            {
                "event_id": event_id,
                "event_url": event_url,
                "event_payload_sha256": event_hash,
            }
        )
    missing_market_end_dates = [
        {
            "event_id": event_id,
            "market_id": str(market.get("id") or ""),
            "question": question,
        }
        for event_id, event in events.items()
        for raw_market in _list(
            event.get("markets"),
            name=f"canonical event {event_id} markets",
        )
        for market in (_mapping(raw_market, name="canonical embedded market"),)
        for question in (str(market.get("question") or ""),)
        if market.get("active") is True
        and market.get("closed") is False
        and market.get("acceptingOrders") is True
        and not market.get("endDate")
        and (
            parse_crypto_threshold_question(question) is not None
            or _deadline_stem(question, str(market.get("groupItemTitle") or ""))
            is not None
        )
    ]
    missing_fee_schedules = [
        {
            "event_id": event_id,
            "market_id": str(market.get("id") or ""),
            "question": question,
            "fees_enabled": market.get("feesEnabled") is True,
        }
        for event_id, event in events.items()
        for raw_market in _list(
            event.get("markets"),
            name=f"canonical event {event_id} markets",
        )
        for market in (_mapping(raw_market, name="canonical embedded market"),)
        for question in (str(market.get("question") or ""),)
        if market.get("active") is True
        and market.get("closed") is False
        and market.get("acceptingOrders") is True
        and market.get("endDate")
        and not isinstance(market.get("feeSchedule"), Mapping)
        and (
            parse_crypto_threshold_question(question) is not None
            or _deadline_stem(question, str(market.get("groupItemTitle") or ""))
            is not None
        )
    ]
    return events, {
        "unique_scoped_active_event_count": len(seen_event_ids),
        "candidate_event_count": len(candidate_event_ids),
        "excluded_missing_market_end_date_count": len(missing_market_end_dates),
        "excluded_missing_market_end_dates": missing_market_end_dates,
        "excluded_missing_fee_schedule_count": len(missing_fee_schedules),
        "excluded_missing_fee_schedules": missing_fee_schedules,
        "tags": tag_evidence,
        "canonical_events": event_evidence,
    }


def _threshold_groups(markets: Sequence[_Market]) -> list[tuple[str, list[_Market]]]:
    grouped: dict[tuple[str, str, str, str, str, str], list[_Market]] = {}
    for market in markets:
        threshold = market.threshold
        if threshold is None:
            continue
        key = (
            threshold.kind,
            threshold.asset,
            threshold.window,
            market.description,
            market.end_date,
            market.fee_schedule_key,
        )
        grouped.setdefault(key, []).append(market)
    result: list[tuple[str, list[_Market]]] = []
    for rows in grouped.values():
        if len(rows) < 2:
            continue
        values = [row.threshold.threshold for row in rows if row.threshold is not None]
        if len(set(values)) != len(values):
            continue
        result.append(("threshold", sorted(rows, key=lambda row: row.question)))
    return result


def _deadline_groups(markets: Sequence[_Market]) -> list[tuple[str, list[_Market]]]:
    grouped: dict[tuple[str, str, str], list[_Market]] = {}
    for market in markets:
        if market.deadline_stem is None:
            continue
        key = (
            market.deadline_stem,
            market.description,
            market.fee_schedule_key,
        )
        grouped.setdefault(key, []).append(market)
    result: list[tuple[str, list[_Market]]] = []
    for rows in grouped.values():
        if len(rows) < 2:
            continue
        deadlines = [_parse_timestamp(row.end_date) for row in rows]
        if len(set(deadlines)) != len(deadlines):
            continue
        result.append(
            (
                "deadline",
                sorted(rows, key=lambda row: _parse_timestamp(row.end_date)),
            )
        )
    return result


def _market_payload(market: _Market) -> dict[str, object]:
    threshold = market.threshold
    return {
        "event_id": market.event_id,
        "market_id": market.market_id,
        "condition_id": market.condition_id,
        "question": market.question,
        "end_date": market.end_date,
        "group_item_title": market.group_item_title,
        "threshold_identity": (
            None
            if threshold is None
            else {
                "kind": threshold.kind,
                "asset": threshold.asset,
                "threshold": _decimal_text(threshold.threshold),
                "window": threshold.window,
            }
        ),
        "yes_token_id": market.yes_token_id,
        "no_token_id": market.no_token_id,
        "minimum_order_size": _decimal_text(market.minimum_order_size),
        "tick_size": _decimal_text(market.tick_size),
        "fee_schedule": json.loads(market.fee_schedule_key),
        "taker_base_fee": market.taker_base_fee,
    }


def _outcome(
    market: _Market,
    books: Mapping[str, Mapping[str, object]],
) -> LogicalBinaryOutcome:
    yes = books[market.yes_token_id]
    no = books[market.no_token_id]
    errors: list[str] = []
    if (
        str(yes.get("market") or "").lower() != market.condition_id
        or str(no.get("market") or "").lower() != market.condition_id
    ):
        errors.append("condition_id")
    if yes.get("neg_risk") is not False or no.get("neg_risk") is not False:
        errors.append("negative_risk")
    if (
        Decimal(str(yes.get("min_order_size"))) != market.minimum_order_size
        or Decimal(str(no.get("min_order_size"))) != market.minimum_order_size
    ):
        errors.append("minimum_order_size")
    if (
        Decimal(str(yes.get("tick_size"))) != market.tick_size
        or Decimal(str(no.get("tick_size"))) != market.tick_size
    ):
        errors.append("tick_size")
    if errors:
        raise ValueError("CLOB/Gamma mismatch:" + ",".join(errors))
    return LogicalBinaryOutcome(
        label=market.question,
        yes_asks=_levels(yes, "asks"),
        no_asks=_levels(no, "asks"),
        fee_model=market.fee_model,
    ).validated()


def _bundle_payload(
    bundle: LogicalImplicationBundle | None,
) -> dict[str, object] | None:
    if bundle is None:
        return None
    return {
        "weaker_question": bundle.weaker_label,
        "stronger_question": bundle.stronger_label,
        "quantity": _decimal_text(bundle.quantity),
        "gross_cost_quote": _decimal_text(bundle.gross_cost_quote),
        "taker_fees_quote": _decimal_text(bundle.taker_fees_quote),
        "initial_outlay_quote": _decimal_text(bundle.initial_outlay_quote),
        "terminal_payout_floor_quote": _decimal_text(
            bundle.terminal_payout_floor_quote
        ),
        "net_quote": _decimal_text(bundle.net_quote),
    }


def _pair_payload(
    *,
    group_type: str,
    weaker: _Market,
    stronger: _Market,
    outcomes: Mapping[str, LogicalBinaryOutcome],
    quantity: Decimal,
    books: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    weak = outcomes[weaker.market_id]
    strong = outcomes[stronger.market_id]
    gross = screen_logical_implication_bundle(
        replace(weak, fee_model=ZERO_FEE),
        replace(strong, fee_model=ZERO_FEE),
        quantity=quantity,
    )
    after_cost = screen_logical_implication_bundle(
        weak,
        strong,
        quantity=quantity,
    )
    timestamps = (
        int(books[weaker.yes_token_id].get("timestamp", -1)),
        int(books[stronger.no_token_id].get("timestamp", -1)),
    )
    if min(timestamps) < 0:
        raise ValueError("logical CLOB timestamp is invalid")
    return {
        "group_type": group_type,
        "weaker_market_id": weaker.market_id,
        "stronger_market_id": stronger.market_id,
        "weaker_question": weaker.question,
        "stronger_question": stronger.question,
        "book_timestamp_skew_ms": max(timestamps) - min(timestamps),
        "gross_optimistic_bundle": _bundle_payload(gross),
        "gamma_fee_schedule_bundle": _bundle_payload(after_cost),
    }


def _ordered_pairs(
    group_type: str, rows: Sequence[_Market]
) -> list[tuple[_Market, _Market]]:
    pairs: list[tuple[_Market, _Market]] = []
    for left_index, left in enumerate(rows):
        for right in rows[left_index + 1 :]:
            if group_type == "deadline":
                earlier, later = sorted(
                    (left, right),
                    key=lambda row: _parse_timestamp(row.end_date),
                )
                pairs.append((later, earlier))
                continue
            left_threshold = left.threshold
            right_threshold = right.threshold
            if left_threshold is None or right_threshold is None:
                raise ValueError("threshold implication group lost its identity")
            if left_threshold.stronger_than(right_threshold):
                pairs.append((right, left))
            elif right_threshold.stronger_than(left_threshold):
                pairs.append((left, right))
            else:
                raise ValueError("threshold implication group is not strictly ordered")
    return pairs


def _best_pair(
    rows: Sequence[dict[str, object]],
    *,
    key: str,
) -> dict[str, object] | None:
    executable = [
        row
        for row in rows
        if isinstance(row.get(key), Mapping) and row[key].get("net_quote") is not None  # type: ignore[index]
    ]
    return max(
        executable,
        key=lambda row: Decimal(str(row[key]["net_quote"])),  # type: ignore[index]
        default=None,
    )


def _positive_count(rows: Sequence[dict[str, object]], *, key: str) -> int:
    return sum(
        isinstance(row.get(key), Mapping)
        and row[key].get("net_quote") is not None  # type: ignore[index]
        and Decimal(str(row[key]["net_quote"])) > 0  # type: ignore[index]
        for row in rows
    )


def _screen_event(
    client: _SourceClient,
    *,
    event: Mapping[str, object],
    quantity: Decimal,
    fee_cache: dict[str, str],
    execution_exclusions: list[dict[str, object]],
) -> list[dict[str, object]]:
    event_id = str(event.get("id") or "")
    markets = [
        parsed
        for raw in _list(event.get("markets"), name=f"event {event_id} markets")
        if (parsed := _parse_market(raw, event_id=event_id, quantity=quantity))
        is not None
    ]
    groups = _threshold_groups(markets) + _deadline_groups(markets)
    if not groups:
        return []
    requested_tokens = sorted(
        {
            token
            for _, rows in groups
            for market in rows
            for token in (market.yes_token_id, market.no_token_id)
        },
        key=int,
    )
    requested_before_ms = time.time_ns() // 1_000_000
    raw_books_payload, books_hash = client.request(
        "POST",
        f"{CLOB_BASE_URL}/books",
        body=[{"token_id": token} for token in requested_tokens],
    )
    received_after_ms = time.time_ns() // 1_000_000
    raw_books = _list(raw_books_payload, name=f"event {event_id} logical books")
    books = {
        str(_mapping(book, name="logical CLOB book").get("asset_id")): _mapping(
            book,
            name="logical CLOB book",
        )
        for book in raw_books
    }
    if len(raw_books) != len(requested_tokens) or set(books) != set(requested_tokens):
        raise ValueError(f"event {event_id} logical CLOB book identities differ")
    parsed_books: dict[str, Mapping[str, object]] = books
    outcomes: dict[str, LogicalBinaryOutcome] = {}
    market_by_id = {market.market_id: market for _, rows in groups for market in rows}
    for market in market_by_id.values():
        try:
            outcomes[market.market_id] = _outcome(market, parsed_books)
        except ValueError as exc:
            execution_exclusions.append(
                {
                    "event_id": event_id,
                    "market_id": market.market_id,
                    "question": market.question,
                    "reason": str(exc),
                }
            )
    groups = [
        (group_type, valid_rows)
        for group_type, rows in groups
        if len(valid_rows := [row for row in rows if row.market_id in outcomes]) >= 2
    ]

    result: list[dict[str, object]] = []
    for group_index, (group_type, rows) in enumerate(groups):
        pairs = [
            _pair_payload(
                group_type=group_type,
                weaker=weaker,
                stronger=stronger,
                outcomes=outcomes,
                quantity=quantity,
                books=parsed_books,
            )
            for weaker, stronger in _ordered_pairs(group_type, rows)
        ]
        best_gross = _best_pair(pairs, key="gross_optimistic_bundle")
        best_after_cost = _best_pair(pairs, key="gamma_fee_schedule_bundle")
        fee_audit: dict[str, object] | None = None
        if (
            best_gross is not None
            and Decimal(
                str(best_gross["gross_optimistic_bundle"]["net_quote"])  # type: ignore[index]
            )
            >= 0
        ):
            market_by_id = {row.market_id: row for row in rows}
            selected = (
                market_by_id[str(best_gross["weaker_market_id"])],
                market_by_id[str(best_gross["stronger_market_id"])],
            )
            audited: list[dict[str, object]] = []
            for market in selected:
                for token in (market.yes_token_id, market.no_token_id):
                    if token not in fee_cache:
                        payload, payload_hash = client.request(
                            "GET",
                            f"{CLOB_BASE_URL}/fee-rate/{token}",
                        )
                        fee = _mapping(payload, name="logical CLOB fee-rate response")
                        if fee.get("base_fee") != market.taker_base_fee:
                            raise ValueError(
                                f"logical market {market.market_id} fee-rate differs"
                            )
                        fee_cache[token] = payload_hash
                    audited.append(
                        {
                            "token_id": token,
                            "payload_sha256": fee_cache[token],
                        }
                    )
            fee_audit = {"audited_tokens": audited}
        result.append(
            {
                "event_id": event_id,
                "event_title": str(event.get("title") or ""),
                "event_slug": str(event.get("slug") or ""),
                "group_id": f"{event_id}:{group_type}:{group_index}",
                "group_type": group_type,
                "market_contracts": [_market_payload(row) for row in rows],
                "books_url": f"{CLOB_BASE_URL}/books",
                "books_payload_sha256": books_hash,
                "books_requested_before_ms": requested_before_ms,
                "books_received_after_ms": received_after_ms,
                "books_request_elapsed_ms": received_after_ms - requested_before_ms,
                "evaluated_pair_count": len(pairs),
                "executable_pair_count": sum(
                    row["gross_optimistic_bundle"] is not None for row in pairs
                ),
                "gross_positive_pair_count": _positive_count(
                    pairs,
                    key="gross_optimistic_bundle",
                ),
                "after_cost_positive_pair_count": _positive_count(
                    pairs,
                    key="gamma_fee_schedule_bundle",
                ),
                "best_gross_pair": best_gross,
                "best_after_cost_pair": best_after_cost,
                "clob_fee_rate_audit": fee_audit,
            }
        )
    return result


def run(*, quantity: Decimal) -> dict[str, object]:
    client = _SourceClient()
    started_ms = time.time_ns() // 1_000_000
    events, universe = _discover_events(client)
    fee_cache: dict[str, str] = {}
    execution_exclusions: list[dict[str, object]] = []
    groups = [
        group
        for event_id in sorted(events, key=int)
        for group in _screen_event(
            client,
            event=events[event_id],
            quantity=quantity,
            fee_cache=fee_cache,
            execution_exclusions=execution_exclusions,
        )
    ]
    universe["excluded_clob_identity_mismatch_count"] = len(execution_exclusions)
    universe["excluded_clob_identity_mismatches"] = execution_exclusions
    best_gross = _best_pair(
        [
            row["best_gross_pair"]
            for row in groups
            if isinstance(row.get("best_gross_pair"), Mapping)
        ],
        key="gross_optimistic_bundle",
    )
    best_after_cost = _best_pair(
        [
            row["best_after_cost_pair"]
            for row in groups
            if isinstance(row.get("best_after_cost_pair"), Mapping)
        ],
        key="gamma_fee_schedule_bundle",
    )
    claim = {
        "quantity": _decimal_text(quantity),
        "unique_scoped_active_event_count": universe[
            "unique_scoped_active_event_count"
        ],
        "eligible_event_count": len({row["event_id"] for row in groups}),
        "threshold_group_count": sum(
            row["group_type"] == "threshold" for row in groups
        ),
        "deadline_group_count": sum(row["group_type"] == "deadline" for row in groups),
        "evaluated_pair_count": sum(int(row["evaluated_pair_count"]) for row in groups),
        "executable_pair_count": sum(
            int(row["executable_pair_count"]) for row in groups
        ),
        "gross_positive_pair_count": sum(
            int(row["gross_positive_pair_count"]) for row in groups
        ),
        "after_cost_positive_pair_count": sum(
            int(row["after_cost_positive_pair_count"]) for row in groups
        ),
        "best_gross_pair": best_gross,
        "best_after_cost_pair": best_after_cost,
        "accepted_edge": False,
        "promotion_status": "rejected_unpromoted",
    }
    tool_path = Path(__file__).resolve()
    module_path = Path(logical_parity_module.__file__).resolve()
    helper_path = Path(structural_source_helpers.__file__).resolve()
    completed_ms = time.time_ns() // 1_000_000
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "started_at_unix_ms": started_ms,
        "completed_at_unix_ms": completed_ms,
        "source_policy": {
            "universe": "Gamma active=true, closed=false for official bitcoin, ethereum, and solana tags",
            "threshold_contract": "same event, exact parsed asset/window/kind, byte-identical description/endDate/fee schedule, and unique thresholds",
            "deadline_contract": "same event, exact question stem/description/fee schedule, explicit month-day group labels, and unique timezone-aware endDate ordering",
            "guaranteed_bundle": "buy YES(weaker proposition) plus NO(stronger proposition); stronger implies weaker, so terminal payout floor is one quote unit per share",
            "book_source": "one official CLOB POST /books response per eligible event",
            "fee_rule": "Gamma feeSchedule, cross-checked against official CLOB fee-rate for every token in each nonnegative gross-best pair",
            "excluded_costs": [
                "gas",
                "multi-leg non-atomic execution",
                "order latency",
                "settlement delay",
                "adverse selection",
            ],
        },
        "source_evidence": universe,
        "implementation": {
            "tool_path": tool_path.relative_to(tool_path.parents[1]).as_posix(),
            "tool_sha256": hashlib.sha256(tool_path.read_bytes()).hexdigest(),
            "module_path": module_path.relative_to(module_path.parents[2]).as_posix(),
            "module_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
            "source_helper_path": helper_path.relative_to(
                helper_path.parents[1]
            ).as_posix(),
            "source_helper_sha256": hashlib.sha256(
                helper_path.read_bytes()
            ).hexdigest(),
        },
        "groups": groups,
        "result_claim": claim,
        "safety": {
            "credentials_used": False,
            "orders_placed": False,
            "public_books_prove_fills": False,
            "positive_gross_snapshot_would_require_sequential_confirmation": True,
        },
    }
    report["result_claim_sha256"] = _sha256(_canonical_json(claim).encode("utf-8"))
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantity", type=Decimal, default=Decimal("5"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/model-research/polymarket/"
            "logical-implication-parity-snapshot-v1-2026-08-25.json"
        ),
    )
    args = parser.parse_args(argv)
    if not args.quantity.is_finite() or args.quantity <= 0:
        parser.error("--quantity must be a positive finite decimal")
    report = run(quantity=args.quantity)
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_bytes_atomic(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "result_claim": report["result_claim"],
                "result_claim_sha256": report["result_claim_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
