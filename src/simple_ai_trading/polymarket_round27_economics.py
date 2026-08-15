"""Frozen, causal after-cost execution replay for Polymarket Round 27."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from collections import Counter
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np

from .paper_execution import PaperOrderIntent, simulate_aggressive_order
from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_fees import PolymarketFeeModel
from .polymarket_replay import PolymarketRecordedBook
from .polymarket_round27_model import Round27Partition


POLYMARKET_ROUND27_ECONOMIC_SCHEMA_VERSION = "polymarket-round27-economic-replay-v2"
POLYMARKET_ROUND27_FIXED_DELAYS_MS = (250, 500, 1_000, 2_000)
_SHA256_CHARACTERS = frozenset("0123456789abcdef")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _sha256(value: object, *, name: str) -> str:
    selected = str(value or "").strip().lower()
    if len(selected) != 64 or set(selected) - _SHA256_CHARACTERS:
        raise ValueError(f"Round 27 {name} SHA-256 differs")
    return selected


def _decimal(value: object, *, name: str, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"Round 27 {name} must be a finite decimal")
    try:
        selected = value if isinstance(value, Decimal) else Decimal(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Round 27 {name} must be a finite decimal") from exc
    if not selected.is_finite() or (positive and selected <= 0):
        raise ValueError(f"Round 27 {name} must be a finite positive decimal")
    return selected


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


@dataclass(frozen=True, slots=True)
class Round27EconomicConfig:
    """Precommitted small-size research policy, never an order authority."""

    delays_ms: tuple[int, ...] = POLYMARKET_ROUND27_FIXED_DELAYS_MS
    primary_delay_ms: int = 500
    maximum_execution_observation_delay_ms: int = 500
    maximum_decision_book_age_ms: int = 1_500
    maximum_conditions_per_book_batch: int = 32
    markout_horizon_ms: int = 1_000
    minimum_expected_edge_per_contract: Decimal = Decimal("0.01")
    maximum_entry_cost_quote: Decimal = Decimal("10")
    initial_capital_quote: Decimal = Decimal("1000")
    minimum_executed_trades: int = 100
    minimum_profitable_conditions: int = 20
    bootstrap_draws: int = 5_000
    bootstrap_seed: int = 27_027

    def validated(self) -> "Round27EconomicConfig":
        delays = tuple(int(value) for value in self.delays_ms)
        edge = _decimal(
            self.minimum_expected_edge_per_contract,
            name="minimum expected edge",
        )
        maximum_cost = _decimal(
            self.maximum_entry_cost_quote,
            name="maximum entry cost",
            positive=True,
        )
        capital = _decimal(
            self.initial_capital_quote,
            name="initial capital",
            positive=True,
        )
        if (
            delays != POLYMARKET_ROUND27_FIXED_DELAYS_MS
            or int(self.primary_delay_ms) != 500
            or not 0 <= int(self.maximum_execution_observation_delay_ms) <= 5_000
            or not 0 <= int(self.maximum_decision_book_age_ms) <= 5_000
            or not 1 <= int(self.maximum_conditions_per_book_batch) <= 32
            or not 100 <= int(self.markout_horizon_ms) <= 60_000
            or not Decimal("0") <= edge <= Decimal("0.10")
            or not Decimal("1") <= maximum_cost <= Decimal("100")
            or not Decimal("100") <= capital <= Decimal("1000000")
            or not 1 <= int(self.minimum_executed_trades) <= 100_000
            or not 1 <= int(self.minimum_profitable_conditions) <= 100_000
            or int(self.minimum_profitable_conditions)
            > int(self.minimum_executed_trades)
            or not 1_000 <= int(self.bootstrap_draws) <= 100_000
            or not 0 <= int(self.bootstrap_seed) < 2**32
        ):
            raise ValueError("Round 27 economic configuration differs")
        return replace(
            self,
            delays_ms=delays,
            minimum_expected_edge_per_contract=edge,
            maximum_entry_cost_quote=maximum_cost,
            initial_capital_quote=capital,
        )

    def asdict(self) -> dict[str, object]:
        return {
            "delays_ms": list(self.delays_ms),
            "primary_delay_ms": self.primary_delay_ms,
            "maximum_execution_observation_delay_ms": (
                self.maximum_execution_observation_delay_ms
            ),
            "maximum_decision_book_age_ms": self.maximum_decision_book_age_ms,
            "maximum_conditions_per_book_batch": (
                self.maximum_conditions_per_book_batch
            ),
            "markout_horizon_ms": self.markout_horizon_ms,
            "minimum_expected_edge_per_contract": _decimal_text(
                self.minimum_expected_edge_per_contract
            ),
            "maximum_entry_cost_quote": _decimal_text(
                self.maximum_entry_cost_quote
            ),
            "initial_capital_quote": _decimal_text(self.initial_capital_quote),
            "minimum_executed_trades": self.minimum_executed_trades,
            "minimum_profitable_conditions": self.minimum_profitable_conditions,
            "bootstrap_draws": self.bootstrap_draws,
            "bootstrap_seed": self.bootstrap_seed,
        }


@dataclass(frozen=True, slots=True)
class _DecisionCandidate:
    sample_index: int
    condition_id: str
    event_start_ms: int
    market_end_ms: int
    decision_time_ms: int
    outcome: str
    token_id: str
    predicted_probability: float
    quantity: Decimal
    limit_price: Decimal
    decision_tick_size: Decimal
    decision_average_price: Decimal
    decision_fee_quote: Decimal
    expected_edge_per_contract: Decimal
    segment_id: str
    connection_id: str
    decision_book_event_id: str
    decision_source_payload_sha256: str


@dataclass(frozen=True, slots=True)
class Round27EconomicTrade:
    condition_id: str
    event_start_ms: int
    market_end_ms: int
    decision_time_ms: int
    delay_ms: int
    effective_latency_ms: int
    outcome: str
    predicted_probability: float
    quantity: Decimal
    limit_price: Decimal
    decision_tick_size: Decimal
    execution_tick_size: Decimal | None
    decision_average_price: Decimal
    expected_edge_per_contract: Decimal
    execution_state: str
    execution_reason: str
    execution_book_event_id: str
    filled_quantity: Decimal
    average_fill_price: Decimal
    entry_notional_quote: Decimal
    fee_quote: Decimal
    payout_quote: Decimal
    net_pnl_quote: Decimal
    markout_pnl_per_contract: Decimal | None
    source_payload_sha256: str
    trade_sha256: str

    def asdict(self) -> dict[str, object]:
        return {
            "condition_id": self.condition_id,
            "event_start_ms": self.event_start_ms,
            "market_end_ms": self.market_end_ms,
            "decision_time_ms": self.decision_time_ms,
            "delay_ms": self.delay_ms,
            "effective_latency_ms": self.effective_latency_ms,
            "outcome": self.outcome,
            "predicted_probability": self.predicted_probability,
            "quantity": _decimal_text(self.quantity),
            "limit_price": _decimal_text(self.limit_price),
            "decision_tick_size": _decimal_text(self.decision_tick_size),
            "execution_tick_size": (
                None
                if self.execution_tick_size is None
                else _decimal_text(self.execution_tick_size)
            ),
            "decision_average_price": _decimal_text(self.decision_average_price),
            "expected_edge_per_contract": _decimal_text(
                self.expected_edge_per_contract
            ),
            "execution_state": self.execution_state,
            "execution_reason": self.execution_reason,
            "execution_book_event_id": self.execution_book_event_id,
            "filled_quantity": _decimal_text(self.filled_quantity),
            "average_fill_price": _decimal_text(self.average_fill_price),
            "entry_notional_quote": _decimal_text(self.entry_notional_quote),
            "fee_quote": _decimal_text(self.fee_quote),
            "payout_quote": _decimal_text(self.payout_quote),
            "net_pnl_quote": _decimal_text(self.net_pnl_quote),
            "markout_pnl_per_contract": (
                None
                if self.markout_pnl_per_contract is None
                else _decimal_text(self.markout_pnl_per_contract)
            ),
            "source_payload_sha256": self.source_payload_sha256,
            "trade_sha256": self.trade_sha256,
        }


@dataclass(frozen=True, slots=True)
class Round27EconomicBookBatch:
    """One independently audited, bounded condition batch."""

    condition_ids: tuple[str, ...]
    books: tuple[PolymarketRecordedBook, ...]

    def validated(self) -> "Round27EconomicBookBatch":
        conditions = tuple(str(value).lower() for value in self.condition_ids)
        if (
            not conditions
            or len(conditions) != len(set(conditions))
            or any(not value.startswith("0x") or len(value) != 66 for value in conditions)
            or any(book.market.condition_id not in conditions for book in self.books)
        ):
            raise ValueError("Round 27 economic book batch differs")
        for book in self.books:
            book.snapshot.validated()
        return replace(self, condition_ids=conditions, books=tuple(self.books))


class _BookIndex:
    def __init__(self, books: Sequence[PolymarketRecordedBook]) -> None:
        grouped: dict[str, list[PolymarketRecordedBook]] = {}
        for raw in books:
            book = raw
            _validated_active_tick_size(book)
            grouped.setdefault(book.token_id, []).append(book)
        self.books = {
            token: tuple(
                sorted(
                    values,
                    key=lambda item: (
                        item.received_wall_ms,
                        item.received_monotonic_ns,
                        item.sequence_number,
                        item.sub_index,
                        item.event_id,
                    ),
                )
            )
            for token, values in grouped.items()
        }
        self.times = {
            token: tuple(item.received_wall_ms for item in values)
            for token, values in self.books.items()
        }

    def latest(
        self,
        token_id: str,
        *,
        condition_id: str,
        at_ms: int,
    ) -> PolymarketRecordedBook | None:
        values = self.books.get(token_id, ())
        times = self.times.get(token_id, ())
        index = bisect_right(times, int(at_ms)) - 1
        while index >= 0:
            selected = values[index]
            if selected.market.condition_id == condition_id:
                return selected
            index -= 1
        return None

    def first_at_or_after(
        self,
        token_id: str,
        *,
        condition_id: str,
        target_ms: int,
        maximum_observation_delay_ms: int,
        segment_id: str,
        connection_id: str,
        market_end_ms: int,
    ) -> PolymarketRecordedBook | None:
        values = self.books.get(token_id, ())
        times = self.times.get(token_id, ())
        index = bisect_left(times, int(target_ms))
        deadline = int(target_ms) + int(maximum_observation_delay_ms)
        while index < len(values):
            selected = values[index]
            if selected.market.condition_id != condition_id:
                index += 1
                continue
            if selected.received_wall_ms > deadline:
                return None
            if (
                selected.segment_id != segment_id
                or selected.connection_id != connection_id
                or selected.received_wall_ms >= market_end_ms
            ):
                return None
            return selected
        return None


def _two_sided(book: PolymarketRecordedBook) -> bool:
    snapshot = book.snapshot
    return bool(
        snapshot.connected
        and snapshot.gap_free
        and snapshot.bids
        and snapshot.asks
        and snapshot.bids[0].price < snapshot.asks[0].price
    )


def _validated_active_tick_size(book: PolymarketRecordedBook) -> Decimal:
    snapshot = book.snapshot.validated()
    tick = _decimal(book.tick_size, name="active tick size", positive=True)
    expected_token = (
        book.market.up_token_id
        if book.outcome == "Up"
        else book.market.down_token_id
        if book.outcome == "Down"
        else ""
    )
    if (
        tick >= 1
        or snapshot.venue != "polymarket"
        or snapshot.market_id != book.market.condition_id
        or snapshot.asset_id != expected_token
        or any(
            level.price % tick != 0
            for level in (*snapshot.bids, *snapshot.asks)
        )
    ):
        raise ValueError("Round 27 active book tick lattice differs")
    return tick


def _midpoint(book: PolymarketRecordedBook) -> Decimal:
    if not _two_sided(book):
        raise ValueError("Round 27 economic replay requires a two-sided book")
    return (book.snapshot.bids[0].price + book.snapshot.asks[0].price) / Decimal("2")


def _maximum_economic_limit(
    *,
    fair_probability: float,
    quantity: Decimal,
    tick_size: Decimal,
    minimum_edge_per_contract: Decimal,
    fee: PolymarketFeeModel,
) -> Decimal | None:
    fair = _decimal(
        format(float(fair_probability), ".17g"),
        name="fair probability",
        positive=True,
    )
    tick = _decimal(tick_size, name="tick size", positive=True)
    if fair >= 1 or tick >= 1:
        raise ValueError("Round 27 probability or price lattice differs")
    rough = fair - minimum_edge_per_contract
    ticks = int((rough / tick).to_integral_value(rounding=ROUND_FLOOR))
    while ticks > 0:
        price = tick * ticks
        if price < 1:
            edge = fair * quantity - price * quantity - fee(
                price, quantity, "taker"
            )
            if edge >= minimum_edge_per_contract * quantity:
                return price
        ticks -= 1
    return None


def _walk_asks(
    book: PolymarketRecordedBook,
    *,
    quantity: Decimal,
    limit_price: Decimal,
    fee: PolymarketFeeModel,
) -> tuple[Decimal, Decimal] | None:
    remaining = quantity
    notional = Decimal("0")
    fees = Decimal("0")
    for level in book.snapshot.asks:
        if level.price > limit_price:
            break
        selected = min(level.quantity, remaining)
        if selected <= 0:
            continue
        notional += selected * level.price
        fees += fee(level.price, selected, "taker")
        remaining -= selected
        if remaining <= 0:
            return notional / quantity, fees
    return None


def _decision_books(
    sample_index: int,
    partition: Round27Partition,
    market: PolymarketFiveMinuteMarket,
    index: _BookIndex,
    config: Round27EconomicConfig,
) -> tuple[PolymarketRecordedBook, PolymarketRecordedBook] | None:
    sample = partition.samples[sample_index]
    up = index.latest(
        market.up_token_id,
        condition_id=sample.condition_id,
        at_ms=sample.decision_time_ms,
    )
    down = index.latest(
        market.down_token_id,
        condition_id=sample.condition_id,
        at_ms=sample.decision_time_ms,
    )
    if up is None or down is None:
        return None
    if (
        not _two_sided(up)
        or not _two_sided(down)
        or up.segment_id != down.segment_id
        or up.connection_id != down.connection_id
        or sample.decision_time_ms - up.received_wall_ms
        > config.maximum_decision_book_age_ms
        or sample.decision_time_ms - down.received_wall_ms
        > config.maximum_decision_book_age_ms
        or up.received_wall_ms > sample.decision_time_ms
        or down.received_wall_ms > sample.decision_time_ms
    ):
        return None
    prior = float(_midpoint(up) / (_midpoint(up) + _midpoint(down)))
    if not math.isclose(
        prior,
        sample.market_prior_probability,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Round 27 feature prior and decision books disagree")
    return up, down


def _build_candidates(
    partition: Round27Partition,
    probabilities: np.ndarray,
    market_by_condition: Mapping[str, PolymarketFiveMinuteMarket],
    index: _BookIndex,
    config: Round27EconomicConfig,
) -> tuple[tuple[_DecisionCandidate, ...], dict[str, int]]:
    by_condition: dict[str, list[int]] = {}
    for sample_index, sample in enumerate(partition.samples):
        by_condition.setdefault(sample.condition_id, []).append(sample_index)
    candidates: list[_DecisionCandidate] = []
    reasons: Counter[str] = Counter()
    for condition_id, sample_indices in sorted(by_condition.items()):
        market = market_by_condition[condition_id]
        quantity = _decimal(
            market.minimum_order_size,
            name="minimum order size",
            positive=True,
        )
        fee = market.fee_schedule.fee_model()
        selected_candidate: _DecisionCandidate | None = None
        for sample_index in sorted(
            sample_indices,
            key=lambda value: partition.samples[value].decision_time_ms,
        ):
            books = _decision_books(sample_index, partition, market, index, config)
            if books is None:
                reasons["decision_book_unavailable_or_unsafe"] += 1
                continue
            probability_up = float(probabilities[sample_index])
            choices: list[_DecisionCandidate] = []
            for outcome, probability, book, token_id in (
                ("Up", probability_up, books[0], market.up_token_id),
                ("Down", 1.0 - probability_up, books[1], market.down_token_id),
            ):
                active_tick_size = _validated_active_tick_size(book)
                limit = _maximum_economic_limit(
                    fair_probability=probability,
                    quantity=quantity,
                    tick_size=active_tick_size,
                    minimum_edge_per_contract=(
                        config.minimum_expected_edge_per_contract
                    ),
                    fee=fee,
                )
                if limit is None:
                    continue
                walked = _walk_asks(
                    book,
                    quantity=quantity,
                    limit_price=limit,
                    fee=fee,
                )
                if walked is None:
                    continue
                average, fees = walked
                entry_cost = average * quantity + fees
                edge = (
                    _decimal(
                        format(probability, ".17g"),
                        name="predicted probability",
                    )
                    * quantity
                    - entry_cost
                ) / quantity
                if (
                    edge < config.minimum_expected_edge_per_contract
                    or entry_cost > config.maximum_entry_cost_quote
                ):
                    continue
                choices.append(
                    _DecisionCandidate(
                        sample_index=sample_index,
                        condition_id=condition_id,
                        event_start_ms=market.event_start_ms,
                        market_end_ms=market.end_ms,
                        decision_time_ms=partition.samples[sample_index].decision_time_ms,
                        outcome=outcome,
                        token_id=token_id,
                        predicted_probability=probability,
                        quantity=quantity,
                        limit_price=limit,
                        decision_tick_size=active_tick_size,
                        decision_average_price=average,
                        decision_fee_quote=fees,
                        expected_edge_per_contract=edge,
                        segment_id=book.segment_id,
                        connection_id=book.connection_id,
                        decision_book_event_id=book.event_id,
                        decision_source_payload_sha256=_sha256(
                            book.snapshot.source_payload_sha256,
                            name="decision source payload",
                        ),
                    )
                )
            choices.sort(
                key=lambda item: (-item.expected_edge_per_contract, item.outcome)
            )
            if choices and (
                len(choices) == 1
                or choices[0].expected_edge_per_contract
                != choices[1].expected_edge_per_contract
            ):
                selected_candidate = choices[0]
                break
        if selected_candidate is None:
            reasons["no_positive_after_cost_candidate"] += 1
        else:
            candidates.append(selected_candidate)
    return tuple(candidates), dict(sorted(reasons.items()))


def _condition_bootstrap(
    pnl: Sequence[Decimal],
    *,
    draws: int,
    seed: int,
) -> dict[str, object]:
    values = np.asarray([float(value) for value in pnl], dtype=np.float64)
    if values.size < 20:
        return {
            "eligible": False,
            "condition_count": int(values.size),
            "draw_count": 0,
            "mean_net_pnl_quote": float(np.mean(values)) if values.size else 0.0,
            "ci95_lower": None,
            "ci95_upper": None,
        }
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 500):
        count = min(500, draws - start)
        samples = rng.integers(0, values.size, size=(count, values.size))
        means[start : start + count] = np.mean(values[samples], axis=1)
    return {
        "eligible": True,
        "condition_count": int(values.size),
        "draw_count": draws,
        "mean_net_pnl_quote": float(np.mean(values)),
        "ci95_lower": float(np.quantile(means, 0.025)),
        "ci95_upper": float(np.quantile(means, 0.975)),
    }


def _trade_payload(trade: Round27EconomicTrade) -> dict[str, object]:
    payload = trade.asdict()
    payload.pop("trade_sha256")
    return payload


def _execute_candidate_trades(
    *,
    candidates: Sequence[_DecisionCandidate],
    delay_ms: int,
    market_by_condition: Mapping[str, PolymarketFiveMinuteMarket],
    outcomes_up: Mapping[str, int],
    index: _BookIndex,
    config: Round27EconomicConfig,
) -> tuple[tuple[Round27EconomicTrade, ...], dict[str, int]]:
    reasons: Counter[str] = Counter()
    trades: list[Round27EconomicTrade] = []
    for candidate in candidates:
        market = market_by_condition[candidate.condition_id]
        target_ms = candidate.decision_time_ms + delay_ms
        execution = index.first_at_or_after(
            candidate.token_id,
            condition_id=candidate.condition_id,
            target_ms=target_ms,
            maximum_observation_delay_ms=(
                config.maximum_execution_observation_delay_ms
            ),
            segment_id=candidate.segment_id,
            connection_id=candidate.connection_id,
            market_end_ms=candidate.market_end_ms,
        )
        state = "UNKNOWN"
        reason = "no_gap_free_execution_book_after_delay"
        effective_latency_ms = delay_ms
        execution_event_id = ""
        filled = Decimal("0")
        average = Decimal("0")
        entry_notional = Decimal("0")
        fee_quote = Decimal("0")
        execution_tick_size: Decimal | None = None
        source_sha = candidate.decision_source_payload_sha256
        if execution is not None:
            execution_tick_size = _validated_active_tick_size(execution)
            effective_latency_ms = execution.received_wall_ms - candidate.decision_time_ms
            source_sha = _sha256(
                execution.snapshot.source_payload_sha256,
                name="execution source payload",
            )
            execution_event_id = execution.event_id
            if candidate.limit_price % execution_tick_size != 0:
                state = "REJECTED"
                reason = "limit_price_not_aligned_to_execution_tick_size"
            else:
                intent = PaperOrderIntent(
                    intent_id=_canonical_sha256(
                        {
                            "condition_id": candidate.condition_id,
                            "decision_time_ms": candidate.decision_time_ms,
                            "delay_ms": delay_ms,
                        }
                    ),
                    venue="polymarket",
                    market_id=market.condition_id,
                    asset_id=candidate.token_id,
                    symbol="BTC",
                    outcome=candidate.outcome,
                    side="BUY",
                    order_type="FOK",
                    limit_price=candidate.limit_price,
                    quantity=candidate.quantity,
                    created_at_ms=candidate.decision_time_ms,
                    expires_at_ms=market.end_ms,
                ).validated()
                result = simulate_aggressive_order(
                    intent,
                    execution.snapshot,
                    execution_time_ms=execution.received_wall_ms,
                    submission_latency_ms=delay_ms,
                    maximum_book_age_ms=0,
                    fee=market.fee_schedule.fee_model(),
                )
                state = result.state
                reason = result.reason
                filled = result.filled_quantity
                average = result.average_fill_price
                entry_notional = average * filled
                fee_quote = result.fee_quote
        payout = Decimal("0")
        pnl = Decimal("0")
        markout: Decimal | None = None
        if state == "FILLED":
            payout = (
                filled
                if (candidate.outcome == "Up") is bool(outcomes_up[candidate.condition_id])
                else Decimal("0")
            )
            pnl = payout - entry_notional - fee_quote
            markout_book = index.first_at_or_after(
                candidate.token_id,
                condition_id=candidate.condition_id,
                target_ms=execution.received_wall_ms + config.markout_horizon_ms,
                maximum_observation_delay_ms=(
                    config.maximum_execution_observation_delay_ms
                ),
                segment_id=candidate.segment_id,
                connection_id=candidate.connection_id,
                market_end_ms=candidate.market_end_ms,
            )
            if markout_book is not None and _two_sided(markout_book):
                markout = _midpoint(markout_book) - _midpoint(execution)
        reasons[reason] += 1
        provisional = Round27EconomicTrade(
            condition_id=candidate.condition_id,
            event_start_ms=candidate.event_start_ms,
            market_end_ms=candidate.market_end_ms,
            decision_time_ms=candidate.decision_time_ms,
            delay_ms=delay_ms,
            effective_latency_ms=effective_latency_ms,
            outcome=candidate.outcome,
            predicted_probability=candidate.predicted_probability,
            quantity=candidate.quantity,
            limit_price=candidate.limit_price,
            decision_tick_size=candidate.decision_tick_size,
            execution_tick_size=execution_tick_size,
            decision_average_price=candidate.decision_average_price,
            expected_edge_per_contract=candidate.expected_edge_per_contract,
            execution_state=state,
            execution_reason=reason,
            execution_book_event_id=execution_event_id,
            filled_quantity=filled,
            average_fill_price=average,
            entry_notional_quote=entry_notional,
            fee_quote=fee_quote,
            payout_quote=payout,
            net_pnl_quote=pnl,
            markout_pnl_per_contract=markout,
            source_payload_sha256=source_sha,
            trade_sha256="",
        )
        trades.append(
            replace(
                provisional,
                trade_sha256=_canonical_sha256(_trade_payload(provisional)),
            )
        )
    return tuple(trades), dict(sorted(reasons.items()))


def _scenario_report(
    *,
    trades: Sequence[Round27EconomicTrade],
    candidate_count: int,
    delay_ms: int,
    evaluated_condition_count: int,
    reasons: Mapping[str, int],
    config: Round27EconomicConfig,
) -> dict[str, object]:
    filled_trades = tuple(item for item in trades if item.execution_state == "FILLED")
    pnl_values = tuple(item.net_pnl_quote for item in filled_trades)
    net_pnl = sum(pnl_values, start=Decimal("0"))
    deployed = sum(
        (item.entry_notional_quote + item.fee_quote for item in filled_trades),
        start=Decimal("0"),
    )
    fees = sum((item.fee_quote for item in filled_trades), start=Decimal("0"))
    gross_profit = sum((max(item, Decimal("0")) for item in pnl_values), Decimal("0"))
    gross_loss = -sum((min(item, Decimal("0")) for item in pnl_values), Decimal("0"))
    equity = config.initial_capital_quote
    peak = equity
    maximum_drawdown = Decimal("0")
    for trade in sorted(filled_trades, key=lambda item: (item.market_end_ms, item.condition_id)):
        equity += trade.net_pnl_quote
        peak = max(peak, equity)
        maximum_drawdown = max(maximum_drawdown, peak - equity)
    markouts = tuple(
        item.markout_pnl_per_contract
        for item in filled_trades
        if item.markout_pnl_per_contract is not None
    )
    bootstrap = _condition_bootstrap(
        pnl_values,
        draws=config.bootstrap_draws,
        seed=config.bootstrap_seed + delay_ms,
    )
    profitable_conditions = sum(value > 0 for value in pnl_values)
    unknown_count = sum(item.execution_state == "UNKNOWN" for item in trades)
    checks = {
        "minimum_executed_trades_met": len(filled_trades)
        >= config.minimum_executed_trades,
        "net_pnl_positive": net_pnl > 0,
        "condition_bootstrap_lower_bound_positive": bool(bootstrap["eligible"])
        and float(bootstrap["ci95_lower"]) > 0.0,
        "minimum_profitable_conditions_met": profitable_conditions
        >= config.minimum_profitable_conditions,
        "no_unknown_execution_state": unknown_count == 0,
        "maximum_entry_cost_respected": all(
            item.entry_notional_quote + item.fee_quote
            <= config.maximum_entry_cost_quote
            for item in filled_trades
        ),
    }
    body: dict[str, object] = {
        "delay_ms": delay_ms,
        "evaluated_condition_count": evaluated_condition_count,
        "signal_condition_count": candidate_count,
        "attempted_order_count": len(trades),
        "filled_order_count": len(filled_trades),
        "unknown_order_count": unknown_count,
        "abstained_condition_count": evaluated_condition_count - candidate_count,
        "profitable_condition_count": profitable_conditions,
        "reason_counts": dict(sorted(reasons.items())),
        "gross_deployed_capital_quote": _decimal_text(deployed),
        "total_fees_quote": _decimal_text(fees),
        "net_pnl_quote": _decimal_text(net_pnl),
        "return_on_deployed_capital": _decimal_text(
            net_pnl / deployed if deployed > 0 else Decimal("0")
        ),
        "maximum_drawdown_quote": _decimal_text(maximum_drawdown),
        "maximum_drawdown_fraction": _decimal_text(
            maximum_drawdown / config.initial_capital_quote
        ),
        "profit_factor": (
            "999"
            if gross_loss == 0 and gross_profit > 0
            else _decimal_text(
                gross_profit / gross_loss if gross_loss > 0 else Decimal("0")
            )
        ),
        "win_rate": (
            profitable_conditions / len(filled_trades) if filled_trades else 0.0
        ),
        "turnover_quote": _decimal_text(deployed),
        "fill_rate": len(filled_trades) / len(trades) if trades else 0.0,
        "mean_markout_pnl_per_contract": (
            None
            if not markouts
            else _decimal_text(sum(markouts, Decimal("0")) / len(markouts))
        ),
        "markout_observation_count": len(markouts),
        "condition_bootstrap": bootstrap,
        "gate_checks": checks,
        "scenario_edge_gate_passed": all(checks.values()),
        "trades": [item.asdict() for item in trades],
    }
    body["scenario_sha256"] = _canonical_sha256(body)
    return body


def evaluate_round27_economic_scenarios(
    *,
    partition: Round27Partition,
    predictions: Sequence[float] | np.ndarray,
    markets: Sequence[PolymarketFiveMinuteMarket],
    books: Sequence[PolymarketRecordedBook] | None = None,
    outcomes_up: Mapping[str, int],
    model_name: str,
    model_sha256: str,
    source_audit_sha256: str,
    resolution_evidence_sha256: str,
    config: Round27EconomicConfig | None = None,
    book_batches: Iterable[Round27EconomicBookBatch] | None = None,
) -> dict[str, object]:
    """Replay one frozen candidate population with bounded book residency."""

    cfg = (config or Round27EconomicConfig()).validated()
    probability = np.asarray(predictions, dtype=np.float64)
    if probability.shape != partition.targets.shape or not np.all(
        np.isfinite(probability)
    ) or np.any((probability <= 0.0) | (probability >= 1.0)):
        raise ValueError("Round 27 economic probability population differs")
    conditions = {sample.condition_id for sample in partition.samples}
    if set(outcomes_up) != conditions or any(
        type(value) is not int or value not in {0, 1}
        for value in outcomes_up.values()
    ):
        raise ValueError("Round 27 economic outcome population differs")
    market_by_condition = {
        market.condition_id: market
        for market in markets
        if market.condition_id in conditions
    }
    if set(market_by_condition) != conditions:
        raise ValueError("Round 27 economic market population differs")
    for sample in partition.samples:
        market = market_by_condition[sample.condition_id]
        if (
            market.asset != "BTC"
            or market.event_start_ms != sample.event_start_ms
            or market.end_ms - market.event_start_ms != 300_000
            or not market.event_start_ms
            <= sample.decision_time_ms
            < market.end_ms
        ):
            raise ValueError("Round 27 economic sample metadata differs")
    if (books is None) == (book_batches is None):
        raise ValueError("Round 27 economics requires exactly one book source")
    batches: Iterable[Round27EconomicBookBatch] = (
        (
            Round27EconomicBookBatch(
                condition_ids=tuple(sorted(conditions)),
                books=tuple(
                    book
                    for book in books or ()
                    if book.market.condition_id in conditions
                ),
            ),
        )
        if books is not None
        else book_batches or ()
    )
    sample_indices_by_condition: dict[str, list[int]] = {}
    for sample_index, sample in enumerate(partition.samples):
        sample_indices_by_condition.setdefault(sample.condition_id, []).append(
            sample_index
        )
    seen_conditions: set[str] = set()
    candidates: list[_DecisionCandidate] = []
    selection_reasons: Counter[str] = Counter()
    trades_by_delay: dict[int, list[Round27EconomicTrade]] = {
        delay: [] for delay in cfg.delays_ms
    }
    execution_reasons_by_delay: dict[int, Counter[str]] = {
        delay: Counter() for delay in cfg.delays_ms
    }
    for raw_batch in batches:
        if not isinstance(raw_batch, Round27EconomicBookBatch):
            raise ValueError("Round 27 economic book batch type differs")
        batch = raw_batch.validated()
        batch_conditions = set(batch.condition_ids)
        if (
            not batch_conditions <= conditions
            or batch_conditions & seen_conditions
            or len(batch_conditions) > cfg.maximum_conditions_per_book_batch
        ):
            raise ValueError("Round 27 economic book batch scope differs")
        seen_conditions.update(batch_conditions)
        sample_indices = sorted(
            index
            for condition in batch.condition_ids
            for index in sample_indices_by_condition[condition]
        )
        batch_partition = Round27Partition.from_samples(
            tuple(partition.samples[index] for index in sample_indices),
            role=partition.role,
        )
        batch_probability = probability[np.asarray(sample_indices, dtype=np.int64)]
        batch_market_by_condition = {
            condition: market_by_condition[condition]
            for condition in batch.condition_ids
        }
        book_index = _BookIndex(batch.books)
        batch_candidates, batch_selection_reasons = _build_candidates(
            batch_partition,
            batch_probability,
            batch_market_by_condition,
            book_index,
            cfg,
        )
        candidates.extend(batch_candidates)
        selection_reasons.update(batch_selection_reasons)
        for delay in cfg.delays_ms:
            batch_trades, batch_execution_reasons = _execute_candidate_trades(
                candidates=batch_candidates,
                delay_ms=delay,
                market_by_condition=batch_market_by_condition,
                outcomes_up=outcomes_up,
                index=book_index,
                config=cfg,
            )
            trades_by_delay[delay].extend(batch_trades)
            execution_reasons_by_delay[delay].update(batch_execution_reasons)
    if seen_conditions != conditions:
        raise ValueError("Round 27 economic book batches do not cover the role")
    candidates.sort(
        key=lambda item: (
            item.event_start_ms,
            item.condition_id,
            item.decision_time_ms,
        )
    )
    scenarios = []
    for delay in cfg.delays_ms:
        selected_trades = sorted(
            trades_by_delay[delay],
            key=lambda item: (
                item.event_start_ms,
                item.condition_id,
                item.decision_time_ms,
            ),
        )
        reasons = Counter(selection_reasons)
        reasons.update(execution_reasons_by_delay[delay])
        scenarios.append(
            _scenario_report(
                trades=selected_trades,
                candidate_count=len(candidates),
                delay_ms=delay,
                evaluated_condition_count=len(conditions),
                reasons=dict(sorted(reasons.items())),
                config=cfg,
            )
        )
    scenario_by_delay = {int(item["delay_ms"]): item for item in scenarios}
    all_scenarios_pass = all(
        bool(item["scenario_edge_gate_passed"]) for item in scenarios
    )
    body: dict[str, object] = {
        "schema_version": POLYMARKET_ROUND27_ECONOMIC_SCHEMA_VERSION,
        "partition_role": partition.role,
        "model_name": str(model_name),
        "model_sha256": _sha256(model_sha256, name="model"),
        "source_audit_sha256": _sha256(source_audit_sha256, name="source audit"),
        "resolution_evidence_sha256": _sha256(
            resolution_evidence_sha256,
            name="resolution evidence",
        ),
        "probability_input_sha256": _canonical_sha256(
            {
                "feature_row_sha256": [
                    sample.feature_row_sha256 for sample in partition.samples
                ],
                "probabilities": [format(float(value), ".17g") for value in probability],
            }
        ),
        "config": cfg.asdict(),
        "candidate_condition_count": len(candidates),
        "candidate_population_sha256": _canonical_sha256(
            [
                {
                    "condition_id": item.condition_id,
                    "decision_time_ms": item.decision_time_ms,
                    "outcome": item.outcome,
                    "limit_price": _decimal_text(item.limit_price),
                    "quantity": _decimal_text(item.quantity),
                }
                for item in candidates
            ]
        ),
        "scenarios": scenarios,
        "latency_sensitivity_net_pnl_quote": {
            str(delay): scenario_by_delay[delay]["net_pnl_quote"]
            for delay in cfg.delays_ms
        },
        "primary_500ms_edge_gate_passed": bool(
            scenario_by_delay[cfg.primary_delay_ms]["scenario_edge_gate_passed"]
        ),
        "all_delay_scenarios_edge_gate_passed": all_scenarios_pass,
        "economic_edge_gate_passed": all_scenarios_pass,
        "edge_claim": False,
        "profitability_claim": False,
        "trading_authority": False,
        "orders_submitted": False,
    }
    body["report_sha256"] = _canonical_sha256(body)
    return body


__all__ = [
    "POLYMARKET_ROUND27_ECONOMIC_SCHEMA_VERSION",
    "POLYMARKET_ROUND27_FIXED_DELAYS_MS",
    "Round27EconomicBookBatch",
    "Round27EconomicConfig",
    "Round27EconomicTrade",
    "evaluate_round27_economic_scenarios",
]
