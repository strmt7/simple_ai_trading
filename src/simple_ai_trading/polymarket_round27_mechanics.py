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


ROUND27_MECHANICS_SCHEMA_VERSION = "polymarket-round27-mechanics-diagnostic-v2"
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
    components = _buy_cost_and_fee_per_share(
        book,
        quantity=quantity,
        fee=fee,
    )
    return None if components is None else sum(components)


def _buy_cost_and_fee_per_share(
    book: PaperBookSnapshot,
    *,
    quantity: Decimal,
    fee: PolymarketFeeModel,
) -> tuple[Decimal, Decimal] | None:
    remaining = quantity
    gross = Decimal("0")
    taker_fee = Decimal("0")
    for level in book.asks:
        consumed = min(remaining, level.quantity)
        if consumed > 0:
            gross += consumed * level.price
            taker_fee += fee(level.price, consumed, "taker")
            remaining -= consumed
        if remaining <= 0:
            return gross / quantity, taker_fee / quantity
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
    up_buy_fee: Decimal = Decimal("0")
    down_buy_fee: Decimal = Decimal("0")

    @property
    def complete_set_cost(self) -> Decimal | None:
        if self.up_buy_cost is None or self.down_buy_cost is None:
            return None
        return self.up_buy_cost + self.down_buy_cost

    def complete_set_cost_at_rebate(
        self,
        rebate_fraction: Decimal,
    ) -> Decimal | None:
        if rebate_fraction < 0 or rebate_fraction > 1:
            raise ValueError("taker rebate fraction must be between zero and one")
        cost = self.complete_set_cost
        if cost is None:
            return None
        return cost - rebate_fraction * (self.up_buy_fee + self.down_buy_fee)

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
    up_buy = _buy_cost_and_fee_per_share(
        up.snapshot,
        quantity=ROUND27_RESEARCH_QUANTITY,
        fee=fee,
    )
    down_buy = _buy_cost_and_fee_per_share(
        down.snapshot,
        quantity=ROUND27_RESEARCH_QUANTITY,
        fee=fee,
    )
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
        up_buy_cost=None if up_buy is None else sum(up_buy),
        down_buy_cost=None if down_buy is None else sum(down_buy),
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
        up_buy_fee=Decimal("0") if up_buy is None else up_buy[1],
        down_buy_fee=Decimal("0") if down_buy is None else down_buy[1],
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
    *,
    taker_rebate_fraction: Decimal = Decimal("0"),
    include_ordering_details: bool = False,
) -> dict[str, object]:
    if not quotes:
        empty = {
            "same_state_episode_count": 0,
            "venue_delay_survivor_count": 0,
            "minimum_sequential_survivor_count": 0,
            "best_same_state_cost": None,
            "best_venue_delay_cost": None,
            "best_minimum_sequential_cost": None,
        }
        if include_ordering_details:
            empty.update(
                {
                    "up_then_down_survivor_count": 0,
                    "down_then_up_survivor_count": 0,
                    "lower_source_cost_first_survivor_count": 0,
                    "both_orders_survivor_count": 0,
                    "best_up_then_down_cost": None,
                    "best_down_then_up_cost": None,
                    "best_lower_source_cost_first_cost": None,
                    "best_worst_order_cost": None,
                }
            )
        return empty
    times = [item.received_monotonic_ns for item in quotes]

    def complete_set_cost(quote: _PairedQuote) -> Decimal | None:
        return quote.complete_set_cost_at_rebate(taker_rebate_fraction)

    def leg_cost(quote: _PairedQuote, outcome: str) -> Decimal | None:
        cost = quote.up_buy_cost if outcome == "up" else quote.down_buy_cost
        fee = quote.up_buy_fee if outcome == "up" else quote.down_buy_fee
        return None if cost is None else cost - taker_rebate_fraction * fee

    starts: list[_PairedQuote] = []
    active = False
    for quote in quotes:
        candidate = complete_set_cost(quote)
        current = candidate is not None and candidate < 1
        if current and not active:
            starts.append(quote)
        active = current
    delayed_costs: list[Decimal] = []
    sequential_costs: list[Decimal] = []
    up_then_down_costs: list[Decimal] = []
    down_then_up_costs: list[Decimal] = []
    lower_source_cost_first_costs: list[Decimal] = []
    worst_order_costs: list[Decimal] = []
    for source in starts:
        first = _quote_at_delay(quotes, times, source, source.taker_delay_ms)
        second = _quote_at_delay(quotes, times, source, 2 * source.taker_delay_ms)
        if first is not None and complete_set_cost(first) is not None:
            delayed_costs.append(complete_set_cost(first))
        if first is None or second is None:
            continue
        candidates: list[Decimal] = []
        first_up = leg_cost(first, "up")
        first_down = leg_cost(first, "down")
        second_up = leg_cost(second, "up")
        second_down = leg_cost(second, "down")
        up_then_down = None
        down_then_up = None
        if first_up is not None and second_down is not None:
            up_then_down = first_up + second_down
            candidates.append(up_then_down)
            up_then_down_costs.append(up_then_down)
        if first_down is not None and second_up is not None:
            down_then_up = first_down + second_up
            candidates.append(down_then_up)
            down_then_up_costs.append(down_then_up)
        if candidates:
            sequential_costs.append(min(candidates))
        if up_then_down is not None and down_then_up is not None:
            worst_order_costs.append(max(up_then_down, down_then_up))
            source_up = leg_cost(source, "up")
            source_down = leg_cost(source, "down")
            if source_up is not None and source_down is not None:
                lower_source_cost_first_costs.append(
                    up_then_down if source_up <= source_down else down_then_up
                )
    same_costs = [
        complete_set_cost(item)
        for item in starts
        if complete_set_cost(item) is not None
    ]
    result = {
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
    if include_ordering_details:
        result.update(
            {
                "up_then_down_survivor_count": sum(
                    value < 1 for value in up_then_down_costs
                ),
                "down_then_up_survivor_count": sum(
                    value < 1 for value in down_then_up_costs
                ),
                "lower_source_cost_first_survivor_count": sum(
                    value < 1 for value in lower_source_cost_first_costs
                ),
                "both_orders_survivor_count": sum(
                    value < 1 for value in worst_order_costs
                ),
                "best_up_then_down_cost": _decimal_text(
                    min(up_then_down_costs, default=None)
                ),
                "best_down_then_up_cost": _decimal_text(
                    min(down_then_up_costs, default=None)
                ),
                "best_lower_source_cost_first_cost": _decimal_text(
                    min(lower_source_cost_first_costs, default=None)
                ),
                "best_worst_order_cost": _decimal_text(
                    min(worst_order_costs, default=None)
                ),
            }
        )
    return result


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


def _validate_stage0_lineage(
    *,
    audit: Mapping[str, object],
    preregistration: Mapping[str, object],
    capture_contract: Mapping[str, object],
    capture_result: Mapping[str, object],
) -> dict[str, object]:
    capture_report = capture_result.get("capture_report")
    authority = capture_result.get("authority")
    source_quality = capture_result.get("source_quality")
    analysis_policy = capture_contract.get("analysis_policy")
    if (
        audit.get("target_free") is not True
        or audit.get("model_data_eligible") is not False
        or preregistration.get("schema_version")
        != "polymarket-round27-execution-hypothesis-preregistration-v3"
        or capture_contract.get("schema_version")
        != "polymarket-round27-stage0-mechanics-capture-contract-v1"
        or capture_contract.get("phase") != "mechanics_stage0"
        or capture_contract.get("hypothesis_preregistration_sha256")
        != preregistration.get("preregistration_sha256")
        or not isinstance(analysis_policy, Mapping)
        or analysis_policy.get("maximum_resolved_markets") != 60
        or analysis_policy.get("target_access_during_capture") is not False
        or capture_result.get("schema_version")
        != "polymarket-round27-stage0-mechanics-capture-result-v1"
        or capture_result.get("status") != "passed"
        or capture_result.get("failure_reasons") != []
        or capture_result.get("contract_sha256")
        != capture_contract.get("contract_sha256")
        or capture_result.get("run_id") != audit.get("run_id")
        or not isinstance(capture_report, Mapping)
        or capture_report.get("run_id") != audit.get("run_id")
        or capture_report.get("report_sha256") != audit.get("run_report_sha256")
        or capture_report.get("started_at_ms") != audit.get("run_started_at_ms")
        or capture_report.get("ended_at_ms") != audit.get("run_ended_at_ms")
        or not isinstance(authority, Mapping)
        or not isinstance(source_quality, Mapping)
        or source_quality.get("passed") is not True
        or any(
            authority.get(key) is not False
            for key in (
                "credentials_used",
                "execution_connected",
                "orders_submitted",
                "model_data_eligible",
                "edge_claim",
                "profitability_claim",
                "paper_trading_authority",
                "live_trading_authority",
            )
        )
        or int(audit.get("run_started_at_ms", 0))
        <= int(preregistration.get("created_at_ms", 0))
        or int(audit.get("condition_count", 0)) > 60
    ):
        raise ValueError("Round 27 Stage 0 mechanics lineage differs")
    return {
        "condition_audit_sha256": audit["audit_sha256"],
        "preregistration_sha256": preregistration["preregistration_sha256"],
        "capture_contract_sha256": capture_contract["contract_sha256"],
        "capture_result_sha256": capture_result["result_sha256"],
        "run_id": audit["run_id"],
        "cohort_role": "preregistered_stage0_mechanics",
        "preregistered_stage_0": True,
    }


def analyze_round27_mechanics(
    repository: str | Path,
    *,
    database_path: str | Path,
    condition_audit_path: str | Path,
    preregistration_path: str | Path,
    capture_contract_path: str | Path,
    capture_result_path: str | Path,
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
        label="Round 27 Stage 0 condition audit",
    )
    preregistration = _load_claim(
        Path(preregistration_path),
        claim="preregistration_sha256",
        label="Round 27 preregistration",
    )
    capture_contract = _load_claim(
        Path(capture_contract_path),
        claim="contract_sha256",
        label="Round 27 Stage 0 capture contract",
    )
    capture_result = _load_claim(
        Path(capture_result_path),
        claim="result_sha256",
        label="Round 27 Stage 0 capture result",
    )
    lineage = _validate_stage0_lineage(
        audit=audit,
        preregistration=preregistration,
        capture_contract=capture_contract,
        capture_result=capture_result,
    )
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
    materialized_book_count = 0
    market_count = 0
    paired_quote_state_count = 0
    candidate_counts = {
        name: {"state_count": 0, "market_count": 0}
        for name in (
            "extreme_settlement_value",
            "late_strong_favorite",
            "complete_set_after_fee",
            "split_sell_after_fee",
        )
    }
    segment_benchmarks: list[dict[str, object]] = []
    with PolymarketEvidenceStore(
        database_path,
        read_only=True,
        memory_limit="1GB",
        threads=2,
    ) as store:
        for index, item in enumerate(conditions, start=1):
            condition_id = str(item["condition_id"])
            condition_intervals = {
                key: value for key, value in intervals.items() if key[0] == condition_id
            }
            replay = PolymarketEvidenceReplay.load(
                store,
                run_id=str(audit["run_id"]),
                allow_segmented_gaps=True,
                include_resolutions=False,
                book_sample_interval_ms=ROUND27_BOOK_SAMPLE_INTERVAL_MS,
                condition_ids=(condition_id,),
                maximum_received_wall_ms_by_condition={
                    condition_id: int(item["end_ms"]) - 1
                },
                materialized_minimum_depth_levels=1,
                cap_materialized_depth_to_minimum_order_size=True,
            )
            quotes = _paired_quotes(replay, intervals=condition_intervals)
            grouped: dict[tuple[str, str], list[_PairedQuote]] = {}
            for quote in quotes:
                grouped.setdefault((quote.condition_id, quote.segment_id), []).append(
                    quote
                )
            segment_benchmarks.extend(
                {
                    "condition_id": key[0],
                    "segment_id": key[1],
                    **_latency_benchmarks(grouped.get(key, ())),
                }
                for key in sorted(condition_intervals)
            )
            local_counts = _candidate_counts(quotes)
            for name, local in local_counts.items():
                candidate_counts[name]["state_count"] += int(local["state_count"])
                candidate_counts[name]["market_count"] += int(local["market_count"])
            materialized_book_count += len(replay.books)
            market_count += len(replay.markets)
            paired_quote_state_count += len(quotes)
            store.recycle_analytical_connections()
            if progress is not None:
                progress(
                    "condition_complete",
                    {
                        "completed_condition_count": index,
                        "condition_count": len(conditions),
                        "condition_id": condition_id,
                        "materialized_book_count": materialized_book_count,
                        "paired_quote_state_count": paired_quote_state_count,
                    },
                )
    if market_count != len(condition_ids) or len(segment_benchmarks) != len(intervals):
        raise ValueError("Round 27 condition-isolated replay coverage differs")
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
        "lineage": lineage,
        "method": {
            "target_free": True,
            "settlement_labels_loaded": False,
            "research_quantity_shares": _decimal_text(ROUND27_RESEARCH_QUANTITY),
            "book_sample_interval_ms": ROUND27_BOOK_SAMPLE_INTERVAL_MS,
            "maximum_pair_receipt_skew_ms": ROUND27_PAIR_MAX_SKEW_MS,
            "exact_message_batches_applied_before_pair_evaluation": True,
            "condition_isolated_bounded_replay": True,
            "fee": "recorded market fee schedule with 0.00001 pUSD ceiling",
            "venue_delay": (
                "recorded itode flag mapped through the protocol-defined delay"
            ),
            "minimum_sequential_delay": "two recorded taker delays; network and response latency excluded",
        },
        "coverage": {
            "eligible_market_count": len(condition_ids),
            "eligible_segment_count": len(intervals),
            "materialized_book_count": materialized_book_count,
            "paired_quote_state_count": paired_quote_state_count,
        },
        "candidate_counts": candidate_counts,
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
                "Stage 0 is a target-free mechanics screen; no parameter selection or economic claim is allowed"
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
