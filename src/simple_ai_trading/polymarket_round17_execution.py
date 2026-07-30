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
POLYMARKET_ROUND17_MAXIMUM_CREATION_BOOK_AGE_MS = 500
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
    limit_price: Decimal
    quantity: Decimal
    maximum_entry_loss_quote: Decimal
    lower_bound_edge_quote: Decimal
    lower_bound_edge_quote_per_share: Decimal
    creation_book_sha256: str
    transformed_creation_book_sha256: str

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
            self.limit_price,
            self.quantity,
            self.maximum_entry_loss_quote,
            self.lower_bound_edge_quote,
            self.lower_bound_edge_quote_per_share,
        )
        if (
            any(not item.is_finite() for item in numeric)
            or not Decimal("0") < self.probability < Decimal("1")
            or not Decimal("0") < self.probability_lower_bound < Decimal("1")
            or not Decimal("0") < self.limit_price < Decimal("1")
            or self.quantity <= 0
            or self.maximum_entry_loss_quote <= 0
            or self.lower_bound_edge_quote
            != self.probability_lower_bound * self.quantity
            - self.maximum_entry_loss_quote
            or self.lower_bound_edge_quote_per_share
            != self.lower_bound_edge_quote / self.quantity
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in (
                    self.probability_evidence_sha256,
                    self.creation_book_sha256,
                    self.transformed_creation_book_sha256,
                )
            )
        ):
            raise ValueError("Round 17 entry candidate accounting is invalid")
        return self


@dataclass(frozen=True, slots=True)
class Round17EntryDecision:
    condition_id: str
    decision_time_ms: int
    probability_evidence_sha256: str
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
            "condition_id": self.condition_id,
            "decision_time_ms": self.decision_time_ms,
            "probability_evidence_sha256": self.probability_evidence_sha256,
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
                    "limit_price": format(candidate.limit_price, "f"),
                    "quantity": format(candidate.quantity, "f"),
                    "maximum_entry_loss_quote": format(
                        candidate.maximum_entry_loss_quote,
                        "f",
                    ),
                    "lower_bound_edge_quote": format(
                        candidate.lower_bound_edge_quote,
                        "f",
                    ),
                    "lower_bound_edge_quote_per_share": format(
                        candidate.lower_bound_edge_quote_per_share,
                        "f",
                    ),
                    "creation_book_sha256": candidate.creation_book_sha256,
                    "transformed_creation_book_sha256": (
                        candidate.transformed_creation_book_sha256
                    ),
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
            not self.condition_id
            or self.decision_time_ms <= 0
            or len(self.probability_evidence_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.probability_evidence_sha256
            )
            or self.action not in actions
            or not self.reason
            or self.risk_capital_quote <= 0
            or self.maximum_event_loss_quote <= 0
            or self.maximum_event_loss_quote > self.risk_capital_quote
            or self.minimum_expected_edge_quote_per_share < 0
            or (self.action == "abstain") != (self.candidate is None)
            or (
                self.candidate is not None
                and (
                    self.candidate.validated().maximum_entry_loss_quote
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
class Round17EntryObservation:
    decision_sha256: str
    state: str
    outcome: str
    maximum_entry_loss_quote: Decimal
    actual_entry_cost_quote: Decimal | None
    conservative_utility_quote: Decimal
    execution: PaperExecutionResult | None
    execution_book_sha256: str
    transformed_execution_book_sha256: str
    observation_sha256: str = ""

    def identity_payload(self) -> dict[str, object]:
        result = self.execution
        return {
            "schema_version": "polymarket-round17-entry-observation-v1",
            "contract_sha256": POLYMARKET_ROUND17_CONTRACT_SHA256,
            "decision_sha256": self.decision_sha256,
            "state": self.state,
            "outcome": self.outcome,
            "maximum_entry_loss_quote": format(
                self.maximum_entry_loss_quote,
                "f",
            ),
            "actual_entry_cost_quote": (
                None
                if self.actual_entry_cost_quote is None
                else format(self.actual_entry_cost_quote, "f")
            ),
            "conservative_utility_quote": format(
                self.conservative_utility_quote,
                "f",
            ),
            "execution_book_sha256": self.execution_book_sha256,
            "transformed_execution_book_sha256": (
                self.transformed_execution_book_sha256
            ),
            "execution": (
                None
                if result is None
                else {
                    "state": result.state,
                    "filled_quantity": format(result.filled_quantity, "f"),
                    "remaining_quantity": format(result.remaining_quantity, "f"),
                    "average_fill_price": format(result.average_fill_price, "f"),
                    "fee_quote": format(result.fee_quote, "f"),
                    "reason": result.reason,
                    "source_payload_sha256": result.source_payload_sha256,
                    "fills": [
                        {
                            "price": format(fill.price, "f"),
                            "quantity": format(fill.quantity, "f"),
                            "fee_quote": format(fill.fee_quote, "f"),
                            "liquidity_role": fill.liquidity_role,
                        }
                        for fill in result.fills
                    ],
                }
            ),
        }

    def validated(self) -> "Round17EntryObservation":
        states = {"filled", "known_no_fill", "unknown_after_submit"}
        actual = self.actual_entry_cost_quote
        result = self.execution
        if (
            len(self.decision_sha256) != 64
            or self.state not in states
            or self.outcome not in _OUTCOMES
            or self.maximum_entry_loss_quote <= 0
            or (actual is not None and (not actual.is_finite() or actual <= 0))
            or not self.conservative_utility_quote.is_finite()
            or any(
                value
                and (
                    len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)
                )
                for value in (
                    self.execution_book_sha256,
                    self.transformed_execution_book_sha256,
                )
            )
            or (
                self.state == "filled"
                and (
                    result is None
                    or result.state != "FILLED"
                    or actual
                    != result.average_fill_price * result.filled_quantity
                    + result.fee_quote
                    or actual > self.maximum_entry_loss_quote
                    or self.conservative_utility_quote != -actual
                    or result.source_payload_sha256
                    != self.transformed_execution_book_sha256
                )
            )
            or (
                self.state == "known_no_fill"
                and (
                    result is None
                    or result.state != "CANCELLED"
                    or actual is not None
                    or self.conservative_utility_quote != 0
                )
            )
            or (
                self.state == "unknown_after_submit"
                and (
                    actual is not None
                    or self.conservative_utility_quote != -self.maximum_entry_loss_quote
                )
            )
            or self.observation_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 17 entry observation is invalid")
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
class Round17OwnedClosePlan:
    lot: Round17OwnedLot
    decision_time_ms: int
    scenario: str
    limit_price: Decimal
    maximum_exit_fee_quote: Decimal
    minimum_exit_proceeds_quote: Decimal
    creation_book_sha256: str
    transformed_creation_book_sha256: str
    plan_sha256: str = ""

    def identity_payload(self) -> dict[str, object]:
        owned = self.lot.validated()
        return {
            "schema_version": "polymarket-round17-owned-close-plan-v1",
            "contract_sha256": POLYMARKET_ROUND17_CONTRACT_SHA256,
            "owner": owned.owner,
            "parent_intent_id": owned.parent_intent_id,
            "market_id": owned.market_id,
            "token_id": owned.token_id,
            "outcome": owned.outcome,
            "quantity": format(owned.quantity, "f"),
            "entry_cost_quote": format(owned.entry_cost_quote, "f"),
            "decision_time_ms": self.decision_time_ms,
            "scenario": self.scenario,
            "limit_price": format(self.limit_price, "f"),
            "maximum_exit_fee_quote": format(
                self.maximum_exit_fee_quote,
                "f",
            ),
            "minimum_exit_proceeds_quote": format(
                self.minimum_exit_proceeds_quote,
                "f",
            ),
            "creation_book_sha256": self.creation_book_sha256,
            "transformed_creation_book_sha256": (self.transformed_creation_book_sha256),
            "closing_only": True,
        }

    def validated(self) -> "Round17OwnedClosePlan":
        self.lot.validated()
        if (
            self.decision_time_ms <= 0
            or not self.scenario
            or not Decimal("0") < self.limit_price < Decimal("1")
            or self.maximum_exit_fee_quote < 0
            or self.minimum_exit_proceeds_quote
            != self.limit_price * self.lot.quantity - self.maximum_exit_fee_quote
            or self.minimum_exit_proceeds_quote < 0
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in (
                    self.creation_book_sha256,
                    self.transformed_creation_book_sha256,
                )
            )
            or self.plan_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 17 owned close plan is invalid")
        return self


@dataclass(frozen=True, slots=True)
class Round17ComplementLock:
    allowed: bool
    guaranteed_payout_quote: Decimal
    combined_cost_quote: Decimal
    guaranteed_net_quote: Decimal
    reason: str


@dataclass(frozen=True, slots=True)
class Round17ComplementPlan:
    lot: Round17OwnedLot
    decision_time_ms: int
    scenario: str
    complement_outcome: str
    complement_token_id: str
    limit_price: Decimal
    quantity: Decimal
    maximum_complement_cost_quote: Decimal
    guaranteed_net_quote_lower_bound: Decimal
    creation_book_sha256: str
    transformed_creation_book_sha256: str
    plan_sha256: str = ""

    def identity_payload(self) -> dict[str, object]:
        owned = self.lot.validated()
        return {
            "schema_version": "polymarket-round17-complement-plan-v1",
            "contract_sha256": POLYMARKET_ROUND17_CONTRACT_SHA256,
            "owner": owned.owner,
            "parent_intent_id": owned.parent_intent_id,
            "market_id": owned.market_id,
            "owned_token_id": owned.token_id,
            "owned_outcome": owned.outcome,
            "owned_quantity": format(owned.quantity, "f"),
            "owned_entry_cost_quote": format(owned.entry_cost_quote, "f"),
            "decision_time_ms": self.decision_time_ms,
            "scenario": self.scenario,
            "complement_outcome": self.complement_outcome,
            "complement_token_id": self.complement_token_id,
            "limit_price": format(self.limit_price, "f"),
            "quantity": format(self.quantity, "f"),
            "maximum_complement_cost_quote": format(
                self.maximum_complement_cost_quote,
                "f",
            ),
            "guaranteed_net_quote_lower_bound": format(
                self.guaranteed_net_quote_lower_bound,
                "f",
            ),
            "creation_book_sha256": self.creation_book_sha256,
            "transformed_creation_book_sha256": (self.transformed_creation_book_sha256),
            "risk_reduction_only": True,
        }

    def validated(self) -> "Round17ComplementPlan":
        owned = self.lot.validated()
        if (
            self.decision_time_ms <= 0
            or not self.scenario
            or self.complement_outcome not in _OUTCOMES
            or self.complement_outcome == owned.outcome
            or not self.complement_token_id
            or not Decimal("0") < self.limit_price < Decimal("1")
            or self.quantity != owned.quantity
            or self.maximum_complement_cost_quote <= 0
            or self.guaranteed_net_quote_lower_bound
            != self.quantity
            - owned.entry_cost_quote
            - self.maximum_complement_cost_quote
            or self.guaranteed_net_quote_lower_bound < 0
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in (
                    self.creation_book_sha256,
                    self.transformed_creation_book_sha256,
                )
            )
            or self.plan_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 17 complement plan is invalid")
        return self


@dataclass(frozen=True, slots=True)
class Round17ComplementObservation:
    plan_sha256: str
    state: str
    complement_cost_quote: Decimal | None
    guaranteed_net_quote: Decimal | None
    conservative_utility_quote: Decimal
    execution: PaperExecutionResult | None
    execution_book_sha256: str
    transformed_execution_book_sha256: str
    observation_sha256: str = ""

    def identity_payload(self) -> dict[str, object]:
        result = self.execution
        return {
            "schema_version": "polymarket-round17-complement-observation-v1",
            "contract_sha256": POLYMARKET_ROUND17_CONTRACT_SHA256,
            "plan_sha256": self.plan_sha256,
            "state": self.state,
            "complement_cost_quote": (
                None
                if self.complement_cost_quote is None
                else format(self.complement_cost_quote, "f")
            ),
            "guaranteed_net_quote": (
                None
                if self.guaranteed_net_quote is None
                else format(self.guaranteed_net_quote, "f")
            ),
            "conservative_utility_quote": format(
                self.conservative_utility_quote,
                "f",
            ),
            "execution_book_sha256": self.execution_book_sha256,
            "transformed_execution_book_sha256": (
                self.transformed_execution_book_sha256
            ),
            "execution": (
                None
                if result is None
                else {
                    "state": result.state,
                    "filled_quantity": format(result.filled_quantity, "f"),
                    "remaining_quantity": format(result.remaining_quantity, "f"),
                    "average_fill_price": format(result.average_fill_price, "f"),
                    "fee_quote": format(result.fee_quote, "f"),
                    "reason": result.reason,
                    "source_payload_sha256": result.source_payload_sha256,
                }
            ),
        }

    def validated(self) -> "Round17ComplementObservation":
        if (
            len(self.plan_sha256) != 64
            or self.state not in {"locked", "known_no_fill", "unknown_after_submit"}
            or not self.conservative_utility_quote.is_finite()
            or (
                self.state == "locked"
                and (
                    self.execution is None
                    or self.execution.state != "FILLED"
                    or self.complement_cost_quote is None
                    or self.guaranteed_net_quote is None
                    or self.guaranteed_net_quote < 0
                    or self.conservative_utility_quote != self.guaranteed_net_quote
                )
            )
            or (
                self.state != "locked"
                and (
                    self.complement_cost_quote is not None
                    or self.guaranteed_net_quote is not None
                    or self.conservative_utility_quote != 0
                )
            )
            or self.observation_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 17 complement observation is invalid")
        return self


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


def _maximum_taker_fee_bound(
    fee_model: PolymarketFeeModel,
    quantity: Decimal,
    *,
    level_count: int,
) -> Decimal:
    maximum = fee_model(Decimal("0.5"), quantity, "taker")
    if maximum > 0:
        maximum += Decimal(max(0, level_count - 1)) * Decimal("0.00001")
    return maximum


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
        or source.market_id != market.condition_id
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


def _creation_book(
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
    age = decision_time_ms - source.received_wall_ms
    if (
        market.asset != "BTC"
        or market.horizon_minutes != 5
        or source.venue != "polymarket"
        or source.market_id != market.condition_id
        or source.asset_id != token_id
        or age < 0
        or age > POLYMARKET_ROUND17_MAXIMUM_CREATION_BOOK_AGE_MS
    ):
        raise ValueError("Round 17 creation Polymarket book is invalid")
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
    transformed = _creation_book(
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
    selected_quantity = Decimal("0")
    selected_maximum_loss = Decimal("0")
    while lower_units <= upper_units:
        units = (lower_units + upper_units) // 2
        quantity = Decimal(units) * POLYMARKET_ROUND17_SHARE_QUANTUM
        maximum_fee = _maximum_taker_fee_bound(
            fee_model,
            quantity,
            level_count=len(eligible),
        )
        maximum_loss = limit * quantity + maximum_fee
        if maximum_loss <= risk_budget:
            selected_quantity = quantity
            selected_maximum_loss = maximum_loss
            lower_units = units + 1
        else:
            upper_units = units - 1
    if selected_quantity <= 0:
        return None

    edge = lower_probability * selected_quantity - selected_maximum_loss
    edge_per_share = edge / selected_quantity
    if edge_per_share + _NUMERIC_GUARD < minimum_edge_per_share:
        return None
    return Round17EntryCandidate(
        market_id=market.condition_id,
        condition_id=market.condition_id,
        decision_time_ms=decision_time_ms,
        scenario=scenario.name,
        outcome=outcome,
        token_id=transformed.asset_id,
        probability_evidence_sha256=envelope.evidence_sha256,
        probability=probability,
        probability_lower_bound=lower_probability,
        limit_price=limit,
        quantity=selected_quantity,
        maximum_entry_loss_quote=selected_maximum_loss,
        lower_bound_edge_quote=edge,
        lower_bound_edge_quote_per_share=edge_per_share,
        creation_book_sha256=book.source_payload_sha256,
        transformed_creation_book_sha256=transformed.source_payload_sha256,
    ).validated()


def observe_round17_entry(
    decision: Round17EntryDecision,
    market: PolymarketFiveMinuteMarket,
    book: PaperBookSnapshot | None,
    program: PolymarketRound14Program,
) -> Round17EntryObservation:
    """Observe one already-selected FOK without using the delayed book to select it."""

    selected = decision.validated()
    candidate = selected.candidate
    if candidate is None:
        raise ValueError("Round 17 cannot observe an abstention as a submission")
    scenario = _scenario(program, selected.scenario)
    result: PaperExecutionResult | None = None
    source_sha = ""
    transformed_sha = ""
    actual_cost: Decimal | None = None
    state = "unknown_after_submit"
    if book is not None:
        source_sha = str(book.source_payload_sha256)
        try:
            transformed = _execution_book(
                market,
                book,
                scenario,
                decision_time_ms=selected.decision_time_ms,
                outcome=candidate.outcome,
                side="BUY",
            )
            transformed_sha = transformed.source_payload_sha256
            intent = PaperOrderIntent(
                intent_id="round17-entry-" + selected.decision_sha256[:32],
                venue="polymarket",
                market_id=market.condition_id,
                asset_id=candidate.token_id,
                symbol="BTC",
                outcome=candidate.outcome,
                side="BUY",
                order_type="FOK",
                limit_price=candidate.limit_price,
                quantity=candidate.quantity,
                created_at_ms=selected.decision_time_ms,
                expires_at_ms=market.end_ms,
            )
            fee_model = PolymarketFeeModel(
                enabled=market.fee_schedule.enabled,
                rate=market.fee_schedule.rate * scenario.fee_multiplier,
                exponent=market.fee_schedule.exponent,
                taker_only=True,
            )
            result = simulate_aggressive_order(
                intent,
                transformed,
                execution_time_ms=transformed.received_wall_ms,
                submission_latency_ms=scenario.submission_latency_ms,
                maximum_book_age_ms=0,
                fee=fee_model,
            )
            if result.state == "FILLED":
                actual_cost = (
                    result.average_fill_price * result.filled_quantity
                    + result.fee_quote
                )
                if actual_cost > candidate.maximum_entry_loss_quote:
                    raise RuntimeError(
                        "Round 17 realized entry exceeded its frozen loss bound"
                    )
                state = "filled"
            elif result.state == "CANCELLED":
                state = "known_no_fill"
        except (TypeError, ValueError):
            result = None
            transformed_sha = ""
    utility = (
        -actual_cost
        if state == "filled" and actual_cost is not None
        else (
            Decimal("0")
            if state == "known_no_fill"
            else -candidate.maximum_entry_loss_quote
        )
    )
    provisional = Round17EntryObservation(
        decision_sha256=selected.decision_sha256,
        state=state,
        outcome=candidate.outcome,
        maximum_entry_loss_quote=candidate.maximum_entry_loss_quote,
        actual_entry_cost_quote=actual_cost,
        conservative_utility_quote=utility,
        execution=result,
        execution_book_sha256=source_sha,
        transformed_execution_book_sha256=transformed_sha,
    )
    return replace(
        provisional,
        observation_sha256=_canonical_sha256(provisional.identity_payload()),
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
        condition_id=market.condition_id,
        decision_time_ms=decision_time_ms,
        probability_evidence_sha256=selected_envelope.evidence_sha256,
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


def plan_round17_owned_close(
    market: PolymarketFiveMinuteMarket,
    lot: Round17OwnedLot,
    book: PaperBookSnapshot,
    program: PolymarketRound14Program,
    *,
    decision_time_ms: int,
    scenario_name: str,
) -> Round17OwnedClosePlan | None:
    """Create one owned-only close plan from a causal creation book."""

    owned = lot.validated()
    scenario = _scenario(program, scenario_name)
    expected_token = (
        market.up_token_id if owned.outcome == "Up" else market.down_token_id
    )
    if (
        owned.market_id != market.condition_id
        or owned.token_id != expected_token
        or not market.event_start_ms <= decision_time_ms < market.end_ms
    ):
        raise ValueError("Round 17 owned lot does not match the Polymarket market")
    transformed = _creation_book(
        market,
        book,
        scenario,
        decision_time_ms=decision_time_ms,
        outcome=owned.outcome,
        side="SELL",
    )
    cumulative = Decimal("0")
    limit: Decimal | None = None
    level_count = 0
    for level in transformed.bids:
        cumulative += level.quantity
        limit = level.price
        level_count += 1
        if cumulative >= owned.quantity:
            break
    if limit is None or cumulative < owned.quantity:
        return None
    fee_model = PolymarketFeeModel(
        enabled=market.fee_schedule.enabled,
        rate=market.fee_schedule.rate * scenario.fee_multiplier,
        exponent=market.fee_schedule.exponent,
        taker_only=True,
    )
    maximum_exit_fee = _maximum_taker_fee_bound(
        fee_model,
        owned.quantity,
        level_count=level_count,
    )
    provisional = Round17OwnedClosePlan(
        lot=owned,
        decision_time_ms=decision_time_ms,
        scenario=scenario.name,
        limit_price=limit,
        maximum_exit_fee_quote=maximum_exit_fee,
        minimum_exit_proceeds_quote=(limit * owned.quantity - maximum_exit_fee),
        creation_book_sha256=book.source_payload_sha256,
        transformed_creation_book_sha256=transformed.source_payload_sha256,
    )
    return replace(
        provisional,
        plan_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def observe_round17_owned_close(
    plan: Round17OwnedClosePlan,
    market: PolymarketFiveMinuteMarket,
    book: PaperBookSnapshot | None,
    program: PolymarketRound14Program,
) -> PaperExecutionResult:
    """Apply one fixed owned-close plan to a delayed execution book."""

    selected = plan.validated()
    owned = selected.lot
    scenario = _scenario(program, selected.scenario)
    if book is None:
        return PaperExecutionResult(
            state="CLOSE_PENDING",
            filled_quantity=Decimal("0"),
            remaining_quantity=owned.quantity,
            average_fill_price=Decimal("0"),
            fee_quote=Decimal("0"),
            fills=(),
            reason="missing_delayed_book_for_owned_close",
            source_payload_sha256=selected.transformed_creation_book_sha256,
        )
    try:
        transformed = _execution_book(
            market,
            book,
            scenario,
            decision_time_ms=selected.decision_time_ms,
            outcome=owned.outcome,
            side="SELL",
        )
    except (TypeError, ValueError):
        return PaperExecutionResult(
            state="CLOSE_PENDING",
            filled_quantity=Decimal("0"),
            remaining_quantity=owned.quantity,
            average_fill_price=Decimal("0"),
            fee_quote=Decimal("0"),
            fills=(),
            reason="invalid_delayed_book_for_owned_close",
            source_payload_sha256=selected.transformed_creation_book_sha256,
        )
    intent = PaperOrderIntent(
        intent_id=(
            "round17-close-"
            + _canonical_sha256(
                {
                    "plan_sha256": selected.plan_sha256,
                }
            )[:32]
        ),
        venue="polymarket",
        market_id=market.condition_id,
        asset_id=owned.token_id,
        symbol="BTC",
        outcome=owned.outcome,
        side="SELL",
        order_type="FOK",
        limit_price=selected.limit_price,
        quantity=owned.quantity,
        created_at_ms=selected.decision_time_ms,
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


def plan_round17_complement_lock(
    market: PolymarketFiveMinuteMarket,
    lot: Round17OwnedLot,
    book: PaperBookSnapshot,
    program: PolymarketRound14Program,
    *,
    decision_time_ms: int,
    scenario_name: str,
) -> Round17ComplementPlan | None:
    """Plan a complement only when its creation-book bound locks nonnegative PnL."""

    owned = lot.validated()
    scenario = _scenario(program, scenario_name)
    if (
        owned.market_id != market.condition_id
        or not market.event_start_ms <= decision_time_ms < market.end_ms
    ):
        raise ValueError("Round 17 complement lot does not match the market")
    complement_outcome = "Down" if owned.outcome == "Up" else "Up"
    complement_token = (
        market.down_token_id if complement_outcome == "Down" else market.up_token_id
    )
    transformed = _creation_book(
        market,
        book,
        scenario,
        decision_time_ms=decision_time_ms,
        outcome=complement_outcome,
        side="BUY",
    )
    cumulative = Decimal("0")
    limit: Decimal | None = None
    level_count = 0
    for level in transformed.asks:
        cumulative += level.quantity
        limit = level.price
        level_count += 1
        if cumulative >= owned.quantity:
            break
    if limit is None or cumulative < owned.quantity:
        return None
    fee_model = PolymarketFeeModel(
        enabled=market.fee_schedule.enabled,
        rate=market.fee_schedule.rate * scenario.fee_multiplier,
        exponent=market.fee_schedule.exponent,
        taker_only=True,
    )
    maximum_cost = limit * owned.quantity + _maximum_taker_fee_bound(
        fee_model,
        owned.quantity,
        level_count=level_count,
    )
    guaranteed = owned.quantity - owned.entry_cost_quote - maximum_cost
    if guaranteed < 0:
        return None
    provisional = Round17ComplementPlan(
        lot=owned,
        decision_time_ms=decision_time_ms,
        scenario=scenario.name,
        complement_outcome=complement_outcome,
        complement_token_id=complement_token,
        limit_price=limit,
        quantity=owned.quantity,
        maximum_complement_cost_quote=maximum_cost,
        guaranteed_net_quote_lower_bound=guaranteed,
        creation_book_sha256=book.source_payload_sha256,
        transformed_creation_book_sha256=transformed.source_payload_sha256,
    )
    return replace(
        provisional,
        plan_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def observe_round17_complement_lock(
    plan: Round17ComplementPlan,
    market: PolymarketFiveMinuteMarket,
    book: PaperBookSnapshot | None,
    program: PolymarketRound14Program,
) -> Round17ComplementObservation:
    """Apply a fixed safe-complement plan to one delayed Polymarket book."""

    selected = plan.validated()
    scenario = _scenario(program, selected.scenario)
    result: PaperExecutionResult | None = None
    source_sha = ""
    transformed_sha = ""
    complement_cost: Decimal | None = None
    guaranteed_net: Decimal | None = None
    state = "unknown_after_submit"
    if book is not None:
        source_sha = str(book.source_payload_sha256)
        try:
            transformed = _execution_book(
                market,
                book,
                scenario,
                decision_time_ms=selected.decision_time_ms,
                outcome=selected.complement_outcome,
                side="BUY",
            )
            transformed_sha = transformed.source_payload_sha256
            intent = PaperOrderIntent(
                intent_id="round17-complement-" + selected.plan_sha256[:32],
                venue="polymarket",
                market_id=market.condition_id,
                asset_id=selected.complement_token_id,
                symbol="BTC",
                outcome=selected.complement_outcome,
                side="BUY",
                order_type="FOK",
                limit_price=selected.limit_price,
                quantity=selected.quantity,
                created_at_ms=selected.decision_time_ms,
                expires_at_ms=market.end_ms,
                parent_inventory_id=selected.lot.parent_intent_id,
            )
            fee_model = PolymarketFeeModel(
                enabled=market.fee_schedule.enabled,
                rate=market.fee_schedule.rate * scenario.fee_multiplier,
                exponent=market.fee_schedule.exponent,
                taker_only=True,
            )
            result = simulate_aggressive_order(
                intent,
                transformed,
                execution_time_ms=transformed.received_wall_ms,
                submission_latency_ms=scenario.submission_latency_ms,
                maximum_book_age_ms=0,
                fee=fee_model,
            )
            if result.state == "FILLED":
                observed_cost = (
                    result.average_fill_price * result.filled_quantity
                    + result.fee_quote
                )
                lock = evaluate_round17_complement_lock(
                    selected.lot,
                    complement_quantity=result.filled_quantity,
                    complement_cost_quote=observed_cost,
                )
                if (
                    observed_cost > selected.maximum_complement_cost_quote
                    or not lock.allowed
                ):
                    raise RuntimeError(
                        "Round 17 complement violated its frozen lock bound"
                    )
                state = "locked"
                complement_cost = observed_cost
                guaranteed_net = lock.guaranteed_net_quote
            elif result.state == "CANCELLED":
                state = "known_no_fill"
        except (TypeError, ValueError):
            result = None
            transformed_sha = ""
    provisional = Round17ComplementObservation(
        plan_sha256=selected.plan_sha256,
        state=state,
        complement_cost_quote=complement_cost,
        guaranteed_net_quote=guaranteed_net,
        conservative_utility_quote=(
            guaranteed_net if guaranteed_net is not None else Decimal("0")
        ),
        execution=result,
        execution_book_sha256=source_sha,
        transformed_execution_book_sha256=transformed_sha,
    )
    return replace(
        provisional,
        observation_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


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
    "Round17ComplementObservation",
    "Round17ComplementPlan",
    "Round17EntryCandidate",
    "Round17EntryDecision",
    "Round17EntryObservation",
    "Round17OwnedClosePlan",
    "Round17OwnedLot",
    "Round17ProbabilityEnvelope",
    "evaluate_round17_complement_lock",
    "observe_round17_complement_lock",
    "observe_round17_entry",
    "observe_round17_owned_close",
    "plan_round17_complement_lock",
    "plan_round17_owned_close",
    "select_round17_entry",
]
