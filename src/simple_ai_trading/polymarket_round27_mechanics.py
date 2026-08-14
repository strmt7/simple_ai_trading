"""Fee- and latency-aware mechanics screen for Round 27 hypotheses."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from decimal import Decimal
import hashlib
from itertools import groupby
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .paper_execution import PaperBookSnapshot
from .polymarket_fees import PolymarketFeeModel
from .polymarket_recorder import PolymarketEvidenceStore
from .polymarket_replay import PolymarketEvidenceReplay, PolymarketRecordedBook
from .storage import write_bytes_atomic


ROUND27_MECHANICS_SCHEMA_VERSION = "polymarket-round27-mechanics-diagnostic-v1"
ROUND27_PAIR_MAX_SKEW_MS = 250
ROUND27_BOOK_SAMPLE_INTERVAL_MS = 50
ROUND27_RESEARCH_QUANTITY = Decimal("5")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _load_claim(path: Path, *, claim: str, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    body = dict(value)
    claimed = str(body.pop(claim, "")).lower()
    if len(claimed) != 64 or claimed != _canonical_sha256(body):
        raise ValueError(f"{label} hash differs")
    return value


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _buy_cost_per_share(
    book: PaperBookSnapshot,
    *,
    quantity: Decimal,
    fee: PolymarketFeeModel,
) -> Decimal | None:
    remaining = quantity
    total = Decimal("0")
    for level in book.asks:
        consumed = min(remaining, level.quantity)
        if consumed > 0:
            total += consumed * level.price
            total += fee(level.price, consumed, "taker")
            remaining -= consumed
        if remaining <= 0:
            return total / quantity
    return None


def _sell_value_per_share(
    book: PaperBookSnapshot,
    *,
    quantity: Decimal,
    fee: PolymarketFeeModel,
) -> Decimal | None:
    remaining = quantity
    total = Decimal("0")
    for level in book.bids:
        consumed = min(remaining, level.quantity)
        if consumed > 0:
            total += consumed * level.price
            total -= fee(level.price, consumed, "taker")
            remaining -= consumed
        if remaining <= 0:
            return total / quantity
    return None


@dataclass(frozen=True, slots=True)
class _PairedQuote:
    condition_id: str
    slug: str
    segment_id: str
    received_monotonic_ns: int
    received_wall_ms: int
    interval_end_ms: int
    market_end_ms: int
    taker_delay_ms: int
    up_best_ask: Decimal | None
    down_best_ask: Decimal | None
    up_buy_cost: Decimal | None
    down_buy_cost: Decimal | None
    up_sell_value: Decimal | None
    down_sell_value: Decimal | None

    @property
    def complete_set_cost(self) -> Decimal | None:
        if self.up_buy_cost is None or self.down_buy_cost is None:
            return None
        return self.up_buy_cost + self.down_buy_cost

    @property
    def split_sell_value(self) -> Decimal | None:
        if self.up_sell_value is None or self.down_sell_value is None:
            return None
        return self.up_sell_value + self.down_sell_value


def _quote_from_pair(
    up: PolymarketRecordedBook,
    down: PolymarketRecordedBook,
    *,
    interval_end_ms: int,
    taker_delay_ms: int,
) -> _PairedQuote | None:
    if (
        up.market.condition_id != down.market.condition_id
        or up.segment_id != down.segment_id
        or up.connection_id != down.connection_id
        or abs(up.received_monotonic_ns - down.received_monotonic_ns)
        > ROUND27_PAIR_MAX_SKEW_MS * 1_000_000
    ):
        return None
    market = up.market
    if market.minimum_order_size > ROUND27_RESEARCH_QUANTITY:
        return None
    now_wall_ms = max(up.received_wall_ms, down.received_wall_ms)
    if now_wall_ms > interval_end_ms:
        return None
    fee = market.fee_schedule.fee_model()
    return _PairedQuote(
        condition_id=market.condition_id,
        slug=market.slug,
        segment_id=up.segment_id,
        received_monotonic_ns=max(
            up.received_monotonic_ns,
            down.received_monotonic_ns,
        ),
        received_wall_ms=now_wall_ms,
        interval_end_ms=interval_end_ms,
        market_end_ms=market.end_ms,
        taker_delay_ms=taker_delay_ms,
        up_best_ask=up.snapshot.asks[0].price if up.snapshot.asks else None,
        down_best_ask=down.snapshot.asks[0].price if down.snapshot.asks else None,
        up_buy_cost=_buy_cost_per_share(
            up.snapshot,
            quantity=ROUND27_RESEARCH_QUANTITY,
            fee=fee,
        ),
        down_buy_cost=_buy_cost_per_share(
            down.snapshot,
            quantity=ROUND27_RESEARCH_QUANTITY,
            fee=fee,
        ),
        up_sell_value=_sell_value_per_share(
            up.snapshot,
            quantity=ROUND27_RESEARCH_QUANTITY,
            fee=fee,
        ),
        down_sell_value=_sell_value_per_share(
            down.snapshot,
            quantity=ROUND27_RESEARCH_QUANTITY,
            fee=fee,
        ),
    )


def _paired_quotes(
    replay: PolymarketEvidenceReplay,
    *,
    intervals: Mapping[tuple[str, str], tuple[int, int]],
) -> tuple[_PairedQuote, ...]:
    delay_by_condition = {
        item.condition_id: item.taker_order_delay_ms
        for item in replay.market_execution_evidence
    }
    expected = {market.condition_id for market in replay.markets}
    if set(delay_by_condition) != expected:
        raise ValueError("Round 27 execution evidence does not cover every market")
    latest: dict[tuple[str, str], dict[str, PolymarketRecordedBook]] = {}
    quotes: list[_PairedQuote] = []
    def batch_key(book: PolymarketRecordedBook) -> tuple[str, int, int]:
        return (
            book.connection_id,
            book.sequence_number,
            book.received_monotonic_ns,
        )

    for _key, batch_iterator in groupby(replay.books, key=batch_key):
        affected: set[tuple[str, str]] = set()
        for book in batch_iterator:
            state_key = (book.market.condition_id, book.segment_id)
            latest.setdefault(state_key, {})[book.outcome] = book
            affected.add(state_key)
        for state_key in sorted(affected):
            state = latest[state_key]
            if set(state) != {"Up", "Down"}:
                continue
            interval = intervals.get(state_key)
            if interval is None:
                continue
            up = state["Up"]
            down = state["Down"]
            quote = _quote_from_pair(
                up,
                down,
                interval_end_ms=interval[1],
                taker_delay_ms=delay_by_condition[state_key[0]],
            )
            if quote is not None and quote.received_wall_ms >= interval[0]:
                quotes.append(quote)
    return tuple(
        sorted(
            quotes,
            key=lambda item: (
                item.condition_id,
                item.segment_id,
                item.received_monotonic_ns,
            ),
        )
    )


def _quote_at_delay(
    quotes: Sequence[_PairedQuote],
    times: Sequence[int],
    source: _PairedQuote,
    delay_ms: int,
) -> _PairedQuote | None:
    target_ns = source.received_monotonic_ns + delay_ms * 1_000_000
    index = bisect_right(times, target_ns) - 1
    if index < 0:
        return None
    candidate = quotes[index]
    target_wall_ms = source.received_wall_ms + delay_ms
    if (
        candidate.segment_id != source.segment_id
        or candidate.condition_id != source.condition_id
        or target_wall_ms > source.interval_end_ms
        or target_wall_ms >= source.market_end_ms
    ):
        return None
    return candidate


def _latency_benchmarks(
    quotes: Sequence[_PairedQuote],
) -> dict[str, object]:
    if not quotes:
        return {
            "same_state_episode_count": 0,
            "venue_delay_survivor_count": 0,
            "minimum_sequential_survivor_count": 0,
            "best_same_state_cost": None,
            "best_venue_delay_cost": None,
            "best_minimum_sequential_cost": None,
        }
    times = [item.received_monotonic_ns for item in quotes]
    starts: list[_PairedQuote] = []
    active = False
    for quote in quotes:
        candidate = quote.complete_set_cost
        current = candidate is not None and candidate < 1
        if current and not active:
            starts.append(quote)
        active = current
    delayed_costs: list[Decimal] = []
    sequential_costs: list[Decimal] = []
    for source in starts:
        first = _quote_at_delay(quotes, times, source, source.taker_delay_ms)
        second = _quote_at_delay(quotes, times, source, 2 * source.taker_delay_ms)
        if first is not None and first.complete_set_cost is not None:
            delayed_costs.append(first.complete_set_cost)
        if first is None or second is None:
            continue
        candidates: list[Decimal] = []
        if first.up_buy_cost is not None and second.down_buy_cost is not None:
            candidates.append(first.up_buy_cost + second.down_buy_cost)
        if first.down_buy_cost is not None and second.up_buy_cost is not None:
            candidates.append(first.down_buy_cost + second.up_buy_cost)
        if candidates:
            sequential_costs.append(min(candidates))
    same_costs = [
        item.complete_set_cost
        for item in starts
        if item.complete_set_cost is not None
    ]
    return {
        "same_state_episode_count": len(starts),
        "venue_delay_survivor_count": sum(value < 1 for value in delayed_costs),
        "minimum_sequential_survivor_count": sum(
            value < 1 for value in sequential_costs
        ),
        "best_same_state_cost": _decimal_text(min(same_costs, default=None)),
        "best_venue_delay_cost": _decimal_text(min(delayed_costs, default=None)),
        "best_minimum_sequential_cost": _decimal_text(
            min(sequential_costs, default=None)
        ),
    }


def _candidate_counts(quotes: Sequence[_PairedQuote]) -> dict[str, object]:
    counts = {
        "extreme_settlement_value": 0,
        "late_strong_favorite": 0,
        "complete_set_after_fee": 0,
        "split_sell_after_fee": 0,
    }
    markets = {key: set() for key in counts}
    for quote in quotes:
        asks = tuple(
            value
            for value in (quote.up_best_ask, quote.down_best_ask)
            if value is not None
        )
        extreme = any(
            Decimal("0.01") <= value <= Decimal("0.05")
            or Decimal("0.95") <= value <= Decimal("0.99")
            for value in asks
        )
        remaining_ms = quote.market_end_ms - quote.received_wall_ms
        flags = {
            "extreme_settlement_value": extreme,
            "late_strong_favorite": (
                5_000 <= remaining_ms <= 60_000
                and any(Decimal("0.95") <= value <= Decimal("0.99") for value in asks)
            ),
            "complete_set_after_fee": (
                quote.complete_set_cost is not None
                and quote.complete_set_cost < Decimal("1")
            ),
            "split_sell_after_fee": (
                quote.split_sell_value is not None
                and quote.split_sell_value > Decimal("1")
            ),
        }
        for name, present in flags.items():
            if present:
                counts[name] += 1
                markets[name].add(quote.condition_id)
    return {
        name: {
            "state_count": counts[name],
            "market_count": len(markets[name]),
        }
        for name in counts
    }


def analyze_round27_mechanics(
    repository: str | Path,
    *,
    database_path: str | Path,
    condition_audit_path: str | Path,
    preregistration_path: str | Path,
    output_path: str | Path,
    progress: Callable[[str, Mapping[str, object]], None] | None = None,
) -> dict[str, object]:
    """Screen mechanics without fitting, settlement labels, or trading authority."""

    root = Path(repository).resolve()
    if not root.is_dir():
        raise ValueError("Round 27 repository root does not exist")
    audit = _load_claim(
        Path(condition_audit_path),
        claim="audit_sha256",
        label="Round 26 condition audit",
    )
    preregistration = _load_claim(
        Path(preregistration_path),
        claim="preregistration_sha256",
        label="Round 27 preregistration",
    )
    if (
        audit.get("target_free") is not True
        or audit.get("model_data_eligible") is not False
        or preregistration.get("schema_version")
        != "polymarket-round27-execution-hypothesis-preregistration-v3"
    ):
        raise ValueError("Round 27 mechanics lineage differs")
    conditions = [
        item
        for item in audit.get("conditions", [])
        if isinstance(item, dict) and item.get("eligible") is True
    ]
    condition_ids = tuple(str(item["condition_id"]) for item in conditions)
    intervals = {
        (str(item["condition_id"]), str(segment["segment_id"])): (
            int(segment["interval_start_ms"]),
            int(segment["interval_end_ms"]),
        )
        for item in conditions
        for segment in item.get("segments", [])
        if isinstance(segment, dict) and segment.get("eligible") is True
    }
    if not condition_ids or not intervals:
        raise ValueError("Round 27 mechanics has no eligible condition intervals")
    if progress is not None:
        progress(
            "replay_started",
            {"condition_count": len(condition_ids), "interval_count": len(intervals)},
        )
    with PolymarketEvidenceStore(
        database_path,
        read_only=True,
        memory_limit="1GB",
        threads=2,
    ) as store:
        replay = PolymarketEvidenceReplay.load(
            store,
            run_id=str(audit["run_id"]),
            allow_segmented_gaps=True,
            include_resolutions=False,
            book_sample_interval_ms=ROUND27_BOOK_SAMPLE_INTERVAL_MS,
            condition_ids=condition_ids,
            maximum_received_wall_ms_by_condition={
                str(item["condition_id"]): int(item["end_ms"]) - 1
                for item in conditions
            },
            materialized_minimum_depth_levels=1,
            cap_materialized_depth_to_minimum_order_size=True,
        )
    if progress is not None:
        progress(
            "replay_complete",
            {
                "materialized_book_count": len(replay.books),
                "market_count": len(replay.markets),
            },
        )
    quotes = _paired_quotes(replay, intervals=intervals)
    grouped: dict[tuple[str, str], list[_PairedQuote]] = {}
    for quote in quotes:
        grouped.setdefault((quote.condition_id, quote.segment_id), []).append(quote)
    segment_benchmarks = [
        {
            "condition_id": condition_id,
            "segment_id": segment_id,
            **_latency_benchmarks(items),
        }
        for (condition_id, segment_id), items in sorted(grouped.items())
    ]
    aggregate = {
        "same_state_episode_count": sum(
            int(item["same_state_episode_count"]) for item in segment_benchmarks
        ),
        "venue_delay_survivor_count": sum(
            int(item["venue_delay_survivor_count"]) for item in segment_benchmarks
        ),
        "minimum_sequential_survivor_count": sum(
            int(item["minimum_sequential_survivor_count"])
            for item in segment_benchmarks
        ),
    }
    body: dict[str, object] = {
        "schema_version": ROUND27_MECHANICS_SCHEMA_VERSION,
        "lineage": {
            "condition_audit_sha256": audit["audit_sha256"],
            "preregistration_sha256": preregistration["preregistration_sha256"],
            "run_id": audit["run_id"],
            "cohort_role": "superseded_round26_v2_diagnostic_only",
            "preregistered_stage_0": False,
        },
        "method": {
            "target_free": True,
            "settlement_labels_loaded": False,
            "research_quantity_shares": _decimal_text(ROUND27_RESEARCH_QUANTITY),
            "book_sample_interval_ms": ROUND27_BOOK_SAMPLE_INTERVAL_MS,
            "maximum_pair_receipt_skew_ms": ROUND27_PAIR_MAX_SKEW_MS,
            "exact_message_batches_applied_before_pair_evaluation": True,
            "fee": "recorded market fee schedule with 0.00001 pUSD ceiling",
            "venue_delay": (
                "recorded itode flag mapped through the protocol-defined delay"
            ),
            "minimum_sequential_delay": "two recorded taker delays; network and response latency excluded",
        },
        "coverage": {
            "eligible_market_count": len(condition_ids),
            "eligible_segment_count": len(intervals),
            "materialized_book_count": len(replay.books),
            "paired_quote_state_count": len(quotes),
        },
        "candidate_counts": _candidate_counts(quotes),
        "complete_set_latency": {
            **aggregate,
            "segment_benchmarks": segment_benchmarks,
            "p50_order_response": "not available",
            "p95_order_response": "not available",
            "p99_order_response": "not available",
        },
        "interpretation": {
            "mechanics_only": True,
            "edge_claim": False,
            "profitability_claim": False,
            "promotion_eligible": False,
            "reason": (
                "the source cohort predates the preregistration and was excluded by its data contract"
            ),
        },
        "authority": {
            "credentials_used": False,
            "execution_connected": False,
            "orders_submitted": False,
            "model_data_eligible": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        },
    }
    body["mechanics_sha256"] = _canonical_sha256(body)
    write_bytes_atomic(
        Path(output_path),
        (json.dumps(body, indent=2, sort_keys=True) + "\n").encode("ascii"),
    )
    if progress is not None:
        progress(
            "complete",
            {
                "mechanics_sha256": body["mechanics_sha256"],
                **aggregate,
            },
        )
    return body


__all__ = [
    "ROUND27_MECHANICS_SCHEMA_VERSION",
    "analyze_round27_mechanics",
]
