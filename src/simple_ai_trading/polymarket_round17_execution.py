"""Exact Polymarket-only action and execution rules for Round 17.

The predictor may contain optional public Binance observations in its feature
vector.  This module cannot access Binance: it accepts only a probability
envelope, validated Polymarket market metadata, and delayed Polymarket books.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_DOWN
import hashlib
import json
from typing import Mapping

from .paper_execution import (
    BOT_OWNER,
    BookLevel,
    PaperBookSnapshot,
    PaperExecutionResult,
    PaperOrderIntent,
    PolymarketFeeModel,
    simulate_aggressive_order,
)
from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_round14_contract import (
    PolymarketRound14ExecutionScenario,
    PolymarketRound14Program,
    PolymarketRound14RiskProfile,
)
from .polymarket_round17_features import POLYMARKET_ROUND17_CONTRACT_SHA256


POLYMARKET_ROUND17_EXECUTION_SCHEMA_VERSION = (
    "polymarket-round17-btc-5m-execution-decision-v1"
)
POLYMARKET_ROUND17_MAXIMUM_EXECUTION_OBSERVATION_DELAY_MS = 500
POLYMARKET_ROUND17_MINIMUM_REMAINING_MARKET_TIME_MS = 30_000
POLYMARKET_ROUND17_SHARE_QUANTUM = Decimal("0.000001")
_NUMERIC_GUARD = Decimal("0.000000000001")
_OUTCOMES = ("Up", "Down")
_ROUND14_CONTRACT_SHA256 = (
    "60cde01112a749a9971447368b3a5d73b203d095e62a974327004c16cb021f1b"
)
_ROUND14_PROFILE_IDENTITY = (
    ("conservative", "0.001", "0.005", "0.02", 30),
    ("regular", "0.002", "0.01", "0.04", 15),
    ("aggressive", "0.0035", "0.015", "0.06", 5),
)
_ROUND14_SCENARIO_IDENTITY = (
    ("primary", 500, 0, "1", "1"),
    ("latency_250ms", 250, 0, "1", "1"),
    ("latency_750ms", 750, 0, "1", "1"),
    ("latency_1000ms", 1000, 0, "1", "1"),
    ("half_depth", 500, 0, "0.5", "1"),
    ("quarter_depth", 500, 0, "0.25", "1"),
    ("one_tick_adverse", 500, 1, "1", "1"),
    ("combined", 1000, 2, "0.5", "2"),
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        selected = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not selected.is_finite():
        raise ValueError(f"{name} must be a finite decimal")
    return selected


def _floor_share_quantity(value: Decimal) -> Decimal:
    return value.quantize(POLYMARKET_ROUND17_SHARE_QUANTUM, rounding=ROUND_DOWN)


def _ceil_share_quantity(value: Decimal) -> Decimal:
    return value.quantize(POLYMARKET_ROUND17_SHARE_QUANTUM, rounding=ROUND_CEILING)


def _validate_program(program: PolymarketRound14Program) -> None:
    profiles = tuple(
        (
            item.name,
            format(item.maximum_event_loss_capital_fraction, "f"),
            format(item.maximum_daily_loss_capital_fraction, "f"),
            format(item.maximum_drawdown_capital_fraction, "f"),
            item.cooldown_minutes_after_loss_cluster,
        )
        for item in program.risk_profiles
    )
    scenarios = tuple(
        (
            item.name,
            item.submission_latency_ms,
            item.adverse_ticks,
            format(item.displayed_depth_fraction, "f"),
            format(item.fee_multiplier, "f"),
        )
        for item in program.scenarios
    )
    if (
        program.contract_sha256 != _ROUND14_CONTRACT_SHA256
        or profiles != _ROUND14_PROFILE_IDENTITY
        or scenarios != _ROUND14_SCENARIO_IDENTITY
        or program.paper_authority
        or program.live_authority
    ):
        raise ValueError("Round 17 execution program differs from the frozen contract")


def _profile(
    program: PolymarketRound14Program,
    name: str,
) -> PolymarketRound14RiskProfile:
    _validate_program(program)
    matches = tuple(item for item in program.risk_profiles if item.name == name)
    if len(matches) != 1:
        raise ValueError("Round 17 risk profile is absent from the frozen program")
    return matches[0]


def _scenario(
    program: PolymarketRound14Program,
    name: str,
) -> PolymarketRound14ExecutionScenario:
    _validate_program(program)
    matches = tuple(item for item in program.scenarios if item.name == name)
    if len(matches) != 1:
        raise ValueError(
            "Round 17 execution scenario is absent from the frozen program"
        )
    return matches[0]


@dataclass(frozen=True, slots=True)
class Round17ProbabilityEnvelope:
    probability_up: Decimal
    lower_up: Decimal
    upper_up: Decimal
    evidence_sha256: str

    def validated(self) -> "Round17ProbabilityEnvelope":
        point = _decimal(self.probability_up, name="probability_up")
        lower = _decimal(self.lower_up, name="lower_up")
        upper = _decimal(self.upper_up, name="upper_up")
        if (
            not Decimal("0") < lower <= point <= upper < Decimal("1")
            or len(self.evidence_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.evidence_sha256
            )
        ):
            raise ValueError("Round 17 probability envelope is invalid")
        return replace(
            self,
            probability_up=point,
            lower_up=lower,
            upper_up=upper,
        )

    def selected_probability(self, outcome: str) -> Decimal:
        self.validated()
        if outcome == "Up":
            return self.probability_up
        if outcome == "Down":
            return Decimal("1") - self.probability_up
        raise ValueError("Round 17 outcome must be Up or Down")

    def selected_lower_bound(self, outcome: str) -> Decimal:
        self.validated()
        if outcome == "Up":
            return self.lower_up
        if outcome == "Down":
            return Decimal("1") - self.upper_up
        raise ValueError("Round 17 outcome must be Up or Down")


@dataclass(frozen=True, slots=True)
class Round17EntryCandidate:
    market_id: str
    condition_id: str
    decision_time_ms: int
    scenario: str
    outcome: str
    token_id: str
    probability_evidence_sha256: str
    probability: Decimal
    probability_lower_bound: Decimal
    quantity: Decimal
    average_fill_price: Decimal
    fee_quote: Decimal
    total_cost_quote: Decimal
    lower_bound_edge_quote: Decimal
    lower_bound_edge_quote_per_share: Decimal
    execution: PaperExecutionResult
    source_book_sha256: str
    transformed_book_sha256: str

    def validated(self) -> "Round17EntryCandidate":
        if (
            not self.market_id
            or not self.condition_id
            or self.decision_time_ms <= 0
            or not self.scenario
            or self.outcome not in _OUTCOMES
            or not self.token_id
        ):
            raise ValueError("Round 17 entry candidate identity is invalid")
        numeric = (
            self.probability,
            self.probability_lower_bound,
            self.quantity,
            self.average_fill_price,
            self.fee_quote,
            self.total_cost_quote,
            self.lower_bound_edge_quote,
            self.lower_bound_edge_quote_per_share,
        )
        if (
            any(not item.is_finite() for item in numeric)
            or not Decimal("0") < self.probability < Decimal("1")
            or not Decimal("0") < self.probability_lower_bound < Decimal("1")
            or self.quantity <= 0
            or self.average_fill_price <= 0
            or self.fee_quote < 0
            or self.total_cost_quote
            != self.average_fill_price * self.quantity + self.fee_quote
            or self.lower_bound_edge_quote
            != self.probability_lower_bound * self.quantity - self.total_cost_quote
            or self.lower_bound_edge_quote_per_share
            != self.lower_bound_edge_quote / self.quantity
            or self.execution.state != "FILLED"
            or self.execution.filled_quantity != self.quantity
            or self.execution.remaining_quantity != 0
            or self.execution.average_fill_price != self.average_fill_price
            or self.execution.fee_quote != self.fee_quote
            or self.execution.source_payload_sha256 != self.transformed_book_sha256
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in (
                    self.probability_evidence_sha256,
                    self.source_book_sha256,
                    self.transformed_book_sha256,
                )
            )
        ):
            raise ValueError("Round 17 entry candidate accounting is invalid")
        return self


@dataclass(frozen=True, slots=True)
class Round17EntryDecision:
    action: str
    reason: str
    profile: str
    scenario: str
    risk_capital_quote: Decimal
    maximum_event_loss_quote: Decimal
    minimum_expected_edge_quote_per_share: Decimal
    candidate: Round17EntryCandidate | None
    decision_sha256: str = ""

    def identity_payload(self) -> dict[str, object]:
        candidate = self.candidate
        return {
            "schema_version": POLYMARKET_ROUND17_EXECUTION_SCHEMA_VERSION,
            "contract_sha256": POLYMARKET_ROUND17_CONTRACT_SHA256,
            "action": self.action,
            "reason": self.reason,
            "profile": self.profile,
            "scenario": self.scenario,
            "risk_capital_quote": format(self.risk_capital_quote, "f"),
            "maximum_event_loss_quote": format(
                self.maximum_event_loss_quote,
                "f",
            ),
            "minimum_expected_edge_quote_per_share": format(
                self.minimum_expected_edge_quote_per_share,
                "f",
            ),
            "candidate": (
                None
                if candidate is None
                else {
                    "outcome": candidate.outcome,
                    "token_id": candidate.token_id,
                    "market_id": candidate.market_id,
                    "condition_id": candidate.condition_id,
                    "decision_time_ms": candidate.decision_time_ms,
                    "scenario": candidate.scenario,
                    "probability_evidence_sha256": (
                        candidate.probability_evidence_sha256
                    ),
                    "probability": format(candidate.probability, "f"),
                    "probability_lower_bound": format(
                        candidate.probability_lower_bound,
                        "f",
                    ),
                    "quantity": format(candidate.quantity, "f"),
                    "average_fill_price": format(
                        candidate.average_fill_price,
                        "f",
                    ),
                    "fee_quote": format(candidate.fee_quote, "f"),
                    "total_cost_quote": format(candidate.total_cost_quote, "f"),
                    "lower_bound_edge_quote": format(
                        candidate.lower_bound_edge_quote,
                        "f",
                    ),
                    "lower_bound_edge_quote_per_share": format(
                        candidate.lower_bound_edge_quote_per_share,
                        "f",
                    ),
                    "source_book_sha256": candidate.source_book_sha256,
                    "transformed_book_sha256": candidate.transformed_book_sha256,
                }
            ),
            "forced_activity": False,
            "binance_credentials_used": False,
            "binance_execution_connected": False,
            "trading_authority": False,
        }

    def validated(self) -> "Round17EntryDecision":
        actions = {"abstain", "buy_up_fok", "buy_down_fok"}
        if (
            self.action not in actions
            or not self.reason
            or self.risk_capital_quote <= 0
            or self.maximum_event_loss_quote <= 0
            or self.maximum_event_loss_quote > self.risk_capital_quote
            or self.minimum_expected_edge_quote_per_share < 0
            or (self.action == "abstain") != (self.candidate is None)
            or (
                self.candidate is not None
                and (
                    self.candidate.validated().total_cost_quote
                    > self.maximum_event_loss_quote
                    or self.action != f"buy_{self.candidate.outcome.lower()}_fok"
                    or self.candidate.lower_bound_edge_quote_per_share + _NUMERIC_GUARD
                    < self.minimum_expected_edge_quote_per_share
                )
            )
            or self.decision_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 17 entry decision is invalid")
        return self


@dataclass(frozen=True, slots=True)
class Round17OwnedLot:
    owner: str
    parent_intent_id: str
    market_id: str
    token_id: str
    outcome: str
    quantity: Decimal
    entry_cost_quote: Decimal

    def validated(self) -> "Round17OwnedLot":
        quantity = _decimal(self.quantity, name="owned quantity")
        cost = _decimal(self.entry_cost_quote, name="entry cost")
        if (
            self.owner != BOT_OWNER
            or not self.parent_intent_id
            or not self.market_id
            or not self.token_id
            or self.outcome not in _OUTCOMES
            or quantity <= 0
            or quantity.quantize(POLYMARKET_ROUND17_SHARE_QUANTUM) != quantity
            or cost <= 0
        ):
            raise ValueError("Round 17 close requires an exact bot-owned lot")
        return replace(self, quantity=quantity, entry_cost_quote=cost)


@dataclass(frozen=True, slots=True)
class Round17ComplementLock:
    allowed: bool
    guaranteed_payout_quote: Decimal
    combined_cost_quote: Decimal
    guaranteed_net_quote: Decimal
    reason: str


def _transformed_book(
    book: PaperBookSnapshot,
    *,
    scenario: PolymarketRound14ExecutionScenario,
    tick_size: Decimal,
    side: str,
) -> PaperBookSnapshot:
    source = book.validated()
    scale = scenario.displayed_depth_fraction
    adverse = tick_size * scenario.adverse_ticks

    def levels(
        values: tuple[BookLevel, ...],
        *,
        direction: str,
    ) -> tuple[BookLevel, ...]:
        output: list[BookLevel] = []
        for level in values:
            price = (
                level.price + adverse if direction == "ask" else level.price - adverse
            )
            quantity = level.quantity * scale
            if Decimal("0") < price < Decimal("1") and quantity > 0:
                output.append(BookLevel(price=price, quantity=quantity))
        return tuple(output)

    bids = levels(source.bids, direction="bid") if side == "SELL" else source.bids
    asks = levels(source.asks, direction="ask") if side == "BUY" else source.asks
    evidence = {
        "schema_version": "polymarket-round17-transformed-book-v1",
        "contract_sha256": POLYMARKET_ROUND17_CONTRACT_SHA256,
        "source_payload_sha256": source.source_payload_sha256,
        "scenario": scenario.name,
        "side": side,
        "venue": source.venue,
        "market_id": source.market_id,
        "asset_id": source.asset_id,
        "source_time_ms": source.source_time_ms,
        "received_wall_ms": source.received_wall_ms,
        "received_monotonic_ns": source.received_monotonic_ns,
        "connected": source.connected,
        "gap_free": source.gap_free,
        "tick_size": format(tick_size, "f"),
        "adverse_ticks": scenario.adverse_ticks,
        "displayed_depth_fraction": format(
            scenario.displayed_depth_fraction,
            "f",
        ),
        "fee_multiplier": format(scenario.fee_multiplier, "f"),
        "bids": [
            [format(level.price, "f"), format(level.quantity, "f")] for level in bids
        ],
        "asks": [
            [format(level.price, "f"), format(level.quantity, "f")] for level in asks
        ],
    }
    return replace(
        source,
        bids=bids,
        asks=asks,
        source_payload_sha256=_canonical_sha256(evidence),
    ).validated()


def _execution_book(
    market: PolymarketFiveMinuteMarket,
    book: PaperBookSnapshot,
    scenario: PolymarketRound14ExecutionScenario,
    *,
    decision_time_ms: int,
    outcome: str,
    side: str,
) -> PaperBookSnapshot:
    token_id = market.up_token_id if outcome == "Up" else market.down_token_id
    source = book.validated()
    target = decision_time_ms + scenario.submission_latency_ms
    if (
        market.asset != "BTC"
        or market.horizon_minutes != 5
        or source.venue != "polymarket"
        or source.market_id != market.market_id
        or source.asset_id != token_id
        or source.received_wall_ms < target
        or source.received_wall_ms
        > target + POLYMARKET_ROUND17_MAXIMUM_EXECUTION_OBSERVATION_DELAY_MS
    ):
        raise ValueError("Round 17 delayed Polymarket execution book is invalid")
    return _transformed_book(
        source,
        scenario=scenario,
        tick_size=market.tick_size,
        side=side,
    )


def _entry_candidate(
    market: PolymarketFiveMinuteMarket,
    book: PaperBookSnapshot,
    scenario: PolymarketRound14ExecutionScenario,
    envelope: Round17ProbabilityEnvelope,
    *,
    decision_time_ms: int,
    outcome: str,
    risk_budget: Decimal,
    minimum_edge_per_share: Decimal,
) -> Round17EntryCandidate | None:
    probability = envelope.selected_probability(outcome)
    lower_probability = envelope.selected_lower_bound(outcome)
    transformed = _execution_book(
        market,
        book,
        scenario,
        decision_time_ms=decision_time_ms,
        outcome=outcome,
        side="BUY",
    )
    fee_model = PolymarketFeeModel(
        enabled=market.fee_schedule.enabled,
        rate=market.fee_schedule.rate * scenario.fee_multiplier,
        exponent=market.fee_schedule.exponent,
        taker_only=True,
    )
    eligible = tuple(
        level
        for level in transformed.asks
        if level.price + minimum_edge_per_share + _NUMERIC_GUARD < lower_probability
    )
    if not eligible:
        return None
    limit = eligible[-1].price
    maximum_quantity = _floor_share_quantity(
        min(
            sum((level.quantity for level in eligible), Decimal("0")),
            risk_budget / eligible[0].price,
        )
    )
    minimum_quantity = _ceil_share_quantity(
        max(
            market.minimum_order_size,
            POLYMARKET_ROUND17_SHARE_QUANTUM,
        )
    )
    if maximum_quantity < minimum_quantity:
        return None

    upper_units = int(maximum_quantity / POLYMARKET_ROUND17_SHARE_QUANTUM)
    lower_units = int(minimum_quantity / POLYMARKET_ROUND17_SHARE_QUANTUM)
    selected_result: PaperExecutionResult | None = None
    selected_quantity = Decimal("0")
    while lower_units <= upper_units:
        units = (lower_units + upper_units) // 2
        quantity = Decimal(units) * POLYMARKET_ROUND17_SHARE_QUANTUM
        intent = PaperOrderIntent(
            intent_id=(
                "round17-entry-"
                + _canonical_sha256(
                    {
                        "condition_id": market.condition_id,
                        "outcome": outcome,
                        "decision_time_ms": decision_time_ms,
                        "scenario": scenario.name,
                        "quantity": format(quantity, "f"),
                    }
                )[:32]
            ),
            venue="polymarket",
            market_id=market.market_id,
            asset_id=(market.up_token_id if outcome == "Up" else market.down_token_id),
            symbol="BTC",
            outcome=outcome,
            side="BUY",
            order_type="FOK",
            limit_price=limit,
            quantity=quantity,
            created_at_ms=decision_time_ms,
            expires_at_ms=market.end_ms,
        )
        result = simulate_aggressive_order(
            intent,
            transformed,
            execution_time_ms=transformed.received_wall_ms,
            submission_latency_ms=scenario.submission_latency_ms,
            maximum_book_age_ms=0,
            fee=fee_model,
        )
        cost = result.average_fill_price * result.filled_quantity + result.fee_quote
        if result.state == "FILLED" and cost <= risk_budget:
            selected_result = result
            selected_quantity = quantity
            lower_units = units + 1
        else:
            upper_units = units - 1
    if selected_result is None:
        return None

    cost = (
        selected_result.average_fill_price * selected_result.filled_quantity
        + selected_result.fee_quote
    )
    edge = lower_probability * selected_quantity - cost
    edge_per_share = edge / selected_quantity
    if cost > risk_budget or edge_per_share + _NUMERIC_GUARD < minimum_edge_per_share:
        return None
    return Round17EntryCandidate(
        market_id=market.market_id,
        condition_id=market.condition_id,
        decision_time_ms=decision_time_ms,
        scenario=scenario.name,
        outcome=outcome,
        token_id=transformed.asset_id,
        probability_evidence_sha256=envelope.evidence_sha256,
        probability=probability,
        probability_lower_bound=lower_probability,
        quantity=selected_quantity,
        average_fill_price=selected_result.average_fill_price,
        fee_quote=selected_result.fee_quote,
        total_cost_quote=cost,
        lower_bound_edge_quote=edge,
        lower_bound_edge_quote_per_share=edge_per_share,
        execution=selected_result,
        source_book_sha256=book.source_payload_sha256,
        transformed_book_sha256=transformed.source_payload_sha256,
    ).validated()


def select_round17_entry(
    market: PolymarketFiveMinuteMarket,
    books: Mapping[str, PaperBookSnapshot],
    envelope: Round17ProbabilityEnvelope,
    program: PolymarketRound14Program,
    *,
    decision_time_ms: int,
    risk_profile: str,
    scenario_name: str,
    risk_capital_quote: Decimal,
    minimum_expected_edge_quote_per_share: Decimal,
    reconciliation_ok: bool,
    existing_owned_exposure: bool,
) -> Round17EntryDecision:
    """Return one Polymarket action; unsafe or non-economic states abstain."""

    selected_envelope = envelope.validated()
    profile = _profile(program, risk_profile)
    scenario = _scenario(program, scenario_name)
    capital = _decimal(risk_capital_quote, name="risk capital")
    edge_floor = _decimal(
        minimum_expected_edge_quote_per_share,
        name="minimum expected edge",
    )
    if capital <= 0 or edge_floor < 0:
        raise ValueError("Round 17 capital or edge floor is invalid")
    maximum_loss = capital * profile.maximum_event_loss_capital_fraction
    if decision_time_ms < market.event_start_ms or decision_time_ms >= market.end_ms:
        raise ValueError("Round 17 decision is outside the Polymarket event")

    candidate: Round17EntryCandidate | None = None
    if not reconciliation_ok:
        reason = "reconciliation_not_proven"
    elif existing_owned_exposure:
        reason = "existing_owned_exposure_blocks_second_parent"
    elif (
        market.end_ms - decision_time_ms
        < POLYMARKET_ROUND17_MINIMUM_REMAINING_MARKET_TIME_MS
    ):
        reason = "insufficient_remaining_market_time"
    elif set(books) != set(_OUTCOMES):
        reason = "paired_polymarket_books_missing"
    elif any(
        not book.validated().connected or not book.validated().gap_free
        for book in books.values()
    ):
        reason = "polymarket_book_unhealthy"
    else:
        candidates: list[Round17EntryCandidate] = []
        evidence_invalid = False
        for outcome in _OUTCOMES:
            try:
                observed = _entry_candidate(
                    market,
                    books[outcome],
                    scenario,
                    selected_envelope,
                    decision_time_ms=decision_time_ms,
                    outcome=outcome,
                    risk_budget=maximum_loss,
                    minimum_edge_per_share=edge_floor,
                )
            except ValueError:
                evidence_invalid = True
                observed = None
            if observed is not None:
                candidates.append(observed)
        candidates.sort(
            key=lambda item: (
                item.lower_bound_edge_quote,
                item.lower_bound_edge_quote_per_share,
                item.outcome,
            ),
            reverse=True,
        )
        if evidence_invalid:
            candidate = None
            reason = "polymarket_execution_evidence_invalid"
        elif not candidates:
            reason = "no_positive_after_cost_lower_bound_edge"
        elif len(candidates) > 1 and (
            abs(
                candidates[0].lower_bound_edge_quote
                - candidates[1].lower_bound_edge_quote
            )
            <= _NUMERIC_GUARD
        ):
            reason = "contradictory_equal_edge"
        else:
            candidate = candidates[0]
            reason = "eligible_polymarket_fok_entry"

    action = "abstain" if candidate is None else f"buy_{candidate.outcome.lower()}_fok"
    provisional = Round17EntryDecision(
        action=action,
        reason=reason,
        profile=profile.name,
        scenario=scenario.name,
        risk_capital_quote=capital,
        maximum_event_loss_quote=maximum_loss,
        minimum_expected_edge_quote_per_share=edge_floor,
        candidate=candidate,
    )
    return replace(
        provisional,
        decision_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def simulate_round17_owned_close(
    market: PolymarketFiveMinuteMarket,
    lot: Round17OwnedLot,
    book: PaperBookSnapshot,
    program: PolymarketRound14Program,
    *,
    decision_time_ms: int,
    scenario_name: str,
) -> PaperExecutionResult:
    """Simulate a close for one exact parent-bound bot-owned lot."""

    owned = lot.validated()
    scenario = _scenario(program, scenario_name)
    expected_token = (
        market.up_token_id if owned.outcome == "Up" else market.down_token_id
    )
    if (
        owned.market_id != market.market_id
        or owned.token_id != expected_token
        or not market.event_start_ms <= decision_time_ms < market.end_ms
    ):
        raise ValueError("Round 17 owned lot does not match the Polymarket market")
    transformed = _execution_book(
        market,
        book,
        scenario,
        decision_time_ms=decision_time_ms,
        outcome=owned.outcome,
        side="SELL",
    )
    if not transformed.bids:
        return PaperExecutionResult(
            state="CLOSE_PENDING",
            filled_quantity=Decimal("0"),
            remaining_quantity=owned.quantity,
            average_fill_price=Decimal("0"),
            fee_quote=Decimal("0"),
            fills=(),
            reason="no_stressed_displayed_bid_for_owned_close",
            source_payload_sha256=transformed.source_payload_sha256,
        )
    intent = PaperOrderIntent(
        intent_id=(
            "round17-close-"
            + _canonical_sha256(
                {
                    "parent_intent_id": owned.parent_intent_id,
                    "decision_time_ms": decision_time_ms,
                    "scenario": scenario.name,
                }
            )[:32]
        ),
        venue="polymarket",
        market_id=market.market_id,
        asset_id=owned.token_id,
        symbol="BTC",
        outcome=owned.outcome,
        side="SELL",
        order_type="FOK",
        limit_price=transformed.bids[-1].price,
        quantity=owned.quantity,
        created_at_ms=decision_time_ms,
        expires_at_ms=market.end_ms,
        parent_inventory_id=owned.parent_intent_id,
    )
    fee_model = PolymarketFeeModel(
        enabled=market.fee_schedule.enabled,
        rate=market.fee_schedule.rate * scenario.fee_multiplier,
        exponent=market.fee_schedule.exponent,
        taker_only=True,
    )
    return simulate_aggressive_order(
        intent,
        transformed,
        execution_time_ms=transformed.received_wall_ms,
        submission_latency_ms=scenario.submission_latency_ms,
        maximum_book_age_ms=0,
        fee=fee_model,
        owned_quantity=owned.quantity,
        closing_position=True,
    )


def evaluate_round17_complement_lock(
    existing_lot: Round17OwnedLot,
    *,
    complement_quantity: Decimal,
    complement_cost_quote: Decimal,
) -> Round17ComplementLock:
    """Permit a hedge only when exact owned shares lock nonnegative worst PnL."""

    lot = existing_lot.validated()
    quantity = _decimal(complement_quantity, name="complement quantity")
    cost = _decimal(complement_cost_quote, name="complement cost")
    if quantity <= 0 or cost <= 0:
        raise ValueError("Round 17 complement quantity and cost must be positive")
    guaranteed_payout = min(lot.quantity, quantity)
    combined_cost = lot.entry_cost_quote + cost
    guaranteed_net = guaranteed_payout - combined_cost
    allowed = guaranteed_net >= 0
    return Round17ComplementLock(
        allowed=allowed,
        guaranteed_payout_quote=guaranteed_payout,
        combined_cost_quote=combined_cost,
        guaranteed_net_quote=guaranteed_net,
        reason=(
            "nonnegative_worst_case_payout_locked_after_all_costs"
            if allowed
            else "complement_does_not_lock_nonnegative_worst_case_payout"
        ),
    )


__all__ = [
    "POLYMARKET_ROUND17_EXECUTION_SCHEMA_VERSION",
    "Round17ComplementLock",
    "Round17EntryCandidate",
    "Round17EntryDecision",
    "Round17OwnedLot",
    "Round17ProbabilityEnvelope",
    "evaluate_round17_complement_lock",
    "select_round17_entry",
    "simulate_round17_owned_close",
]
