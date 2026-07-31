"""Causal, source-bound economic replay for the independent Round 21 policy."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import hashlib
import json
from math import isqrt
import re
from statistics import median
from typing import Sequence

from .paper_execution import PaperBookSnapshot
from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_round21_contract import POLYMARKET_ROUND21_CONTRACT_SHA256
from .polymarket_round21_dataset import Round21OfficialOutcome
from .polymarket_round21_execution import (
    POLYMARKET_ROUND21_EXECUTION_POLICY_SHA256,
    POLYMARKET_ROUND21_EXECUTION_SCENARIOS,
    POLYMARKET_ROUND21_MAXIMUM_EXECUTION_OBSERVATION_LATENESS_MS,
    Round21AggressiveExecutionObservation,
    Round21AggressiveOrderPlan,
    Round21MarketExecutionEvidence,
    observe_round21_aggressive_execution,
    round21_execution_scenario,
)
from .polymarket_round21_model import (
    POLYMARKET_ROUND21_MODEL_DESIGN_SHA256,
    POLYMARKET_ROUND21_PROBABILITY_ENVELOPE_DESIGN_SHA256,
)
from .polymarket_round21_policy import (
    POLYMARKET_ROUND21_MAXIMUM_CREATION_BOOK_AGE_MS,
    POLYMARKET_ROUND21_MULTI_ACTION_POLICY_SHA256,
    Round21BotInventory,
    Round21OwnedLot,
    Round21ProbabilityEnvelope,
    round21_risk_profile,
    select_round21_action,
)


POLYMARKET_ROUND21_ECONOMIC_REPLAY_SCHEMA_VERSION = (
    "polymarket-round21-economic-replay-v1"
)
POLYMARKET_ROUND21_ECONOMIC_REPLAY_DESIGN_SHA256 = (
    "f075d125699eb9de07b5822c8cee7b69f23dd608c66e4447b92ad264afa8e9f4"
)
POLYMARKET_ROUND21_MINIMUM_SEALED_CONDITIONS = 1_800
POLYMARKET_ROUND21_MINIMUM_SEALED_DAYS = 7
POLYMARKET_ROUND21_MINIMUM_EXECUTED_ACTIONS = 300
POLYMARKET_ROUND21_DAILY_BOOTSTRAP_SAMPLES = 2_000
_DAY_MS = 86_400_000
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_TOKEN_ID = re.compile(r"^[0-9]{20,80}$")


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


def _decimal(value: object, *, name: str, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"Round 21 {name} is not a finite decimal")
    try:
        selected = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Round 21 {name} is not a finite decimal") from exc
    if not selected.is_finite() or (positive and selected <= 0):
        raise ValueError(f"Round 21 {name} is invalid")
    return selected


def _digest(value: object, *, name: str) -> str:
    selected = str(value or "").strip().lower()
    if _SHA256.fullmatch(selected) is None or selected == _EMPTY_SHA256:
        raise ValueError(f"Round 21 {name} identity is invalid")
    return selected


def _book_payload(book: PaperBookSnapshot) -> dict[str, object]:
    selected = book.validated()
    return {
        "venue": selected.venue,
        "market_id": selected.market_id,
        "asset_id": selected.asset_id,
        "bids": [
            [format(level.price, "f"), format(level.quantity, "f")]
            for level in selected.bids
        ],
        "asks": [
            [format(level.price, "f"), format(level.quantity, "f")]
            for level in selected.asks
        ],
        "source_time_ms": selected.source_time_ms,
        "received_wall_ms": selected.received_wall_ms,
        "received_monotonic_ns": selected.received_monotonic_ns,
        "source_payload_sha256": selected.source_payload_sha256,
        "connected": selected.connected,
        "gap_free": selected.gap_free,
    }


def _validate_market(market: PolymarketFiveMinuteMarket) -> None:
    schedule = market.fee_schedule
    decimals = (
        market.tick_size,
        market.minimum_order_size,
        market.liquidity_quote,
        market.volume_quote,
        schedule.rate,
        schedule.rebate_rate,
    )
    if (
        type(market) is not PolymarketFiveMinuteMarket
        or market.asset != "BTC"
        or market.horizon_minutes != 5
        or _CONDITION_ID.fullmatch(market.condition_id) is None
        or _TOKEN_ID.fullmatch(market.up_token_id) is None
        or _TOKEN_ID.fullmatch(market.down_token_id) is None
        or market.up_token_id == market.down_token_id
        or market.end_ms != market.event_start_ms + 300_000
        or market.event_start_ms <= 0
        or market.event_start_ms % 300_000
        or any(not value.is_finite() for value in decimals)
        or not Decimal("0") < market.tick_size < Decimal("1")
        or market.minimum_order_size <= 0
        or market.liquidity_quote < 0
        or market.volume_quote < 0
        or schedule.rate < 0
        or schedule.rate > 1
        or schedule.exponent <= 0
        or type(schedule.enabled) is not bool
        or type(schedule.taker_only) is not bool
        or schedule.rebate_rate < 0
        or _SHA256.fullmatch(market.gamma_payload_sha256) is None
        or market.gamma_payload_sha256 == _EMPTY_SHA256
    ):
        raise ValueError("Round 21 replay market is invalid")


@dataclass(frozen=True, slots=True)
class Round21ReplayCondition:
    market: PolymarketFiveMinuteMarket
    market_evidence: Round21MarketExecutionEvidence
    envelopes: tuple[Round21ProbabilityEnvelope, ...]
    books: tuple[PaperBookSnapshot, ...]
    outcome: Round21OfficialOutcome
    source_manifest_sha256: str
    reconciliation_sha256: str
    condition_input_sha256: str
    trading_authority: bool = False

    @classmethod
    def create(
        cls,
        *,
        market: PolymarketFiveMinuteMarket,
        market_evidence: Round21MarketExecutionEvidence,
        envelopes: Sequence[Round21ProbabilityEnvelope],
        books: Sequence[PaperBookSnapshot],
        outcome: Round21OfficialOutcome,
        source_manifest_sha256: str,
        reconciliation_sha256: str,
    ) -> Round21ReplayCondition:
        _validate_market(market)
        evidence = market_evidence.validated()
        selected_envelopes = tuple(
            sorted(
                (value.validated() for value in envelopes),
                key=lambda value: value.decision_time_ms,
            )
        )
        selected_books = tuple(
            sorted(
                (value.validated() for value in books),
                key=lambda value: (
                    value.received_wall_ms,
                    value.asset_id,
                    value.source_payload_sha256,
                ),
            )
        )
        selected_outcome = outcome.validated()
        source_sha = _digest(source_manifest_sha256, name="source manifest")
        reconciliation_sha = _digest(
            reconciliation_sha256,
            name="reconciliation",
        )
        decision_times = tuple(
            value.decision_time_ms for value in selected_envelopes
        )
        book_keys = tuple(
            (value.asset_id, value.received_wall_ms) for value in selected_books
        )
        if (
            evidence.condition_id != market.condition_id
            or not selected_envelopes
            or len(set(decision_times)) != len(decision_times)
            or any(
                value.condition_id != market.condition_id
                or not market.event_start_ms
                <= value.decision_time_ms
                < market.end_ms
                for value in selected_envelopes
            )
            or evidence.observed_wall_ms > decision_times[0]
            or not selected_books
            or len(set(book_keys)) != len(book_keys)
            or any(
                value.venue != "polymarket"
                or value.market_id != market.condition_id
                or value.asset_id not in market.token_ids
                or value.received_wall_ms
                < market.event_start_ms - POLYMARKET_ROUND21_MAXIMUM_CREATION_BOOK_AGE_MS
                or value.received_wall_ms >= market.end_ms
                or value.source_payload_sha256 == _EMPTY_SHA256
                for value in selected_books
            )
            or selected_outcome.condition_id != market.condition_id
            or selected_outcome.event_start_ms != market.event_start_ms
        ):
            raise ValueError("Round 21 replay condition is invalid")
        payload = {
            "schema_version": POLYMARKET_ROUND21_ECONOMIC_REPLAY_SCHEMA_VERSION,
            "design_sha256": POLYMARKET_ROUND21_ECONOMIC_REPLAY_DESIGN_SHA256,
            "market": market.asdict(),
            "market_execution_evidence_sha256": evidence.evidence_sha256,
            "probability_evidence_sha256": [
                value.evidence_sha256 for value in selected_envelopes
            ],
            "books": [_book_payload(value) for value in selected_books],
            "outcome_sha256": selected_outcome.outcome_sha256,
            "source_manifest_sha256": source_sha,
            "reconciliation_sha256": reconciliation_sha,
            "trading_authority": False,
        }
        return cls(
            market=market,
            market_evidence=evidence,
            envelopes=selected_envelopes,
            books=selected_books,
            outcome=selected_outcome,
            source_manifest_sha256=source_sha,
            reconciliation_sha256=reconciliation_sha,
            condition_input_sha256=_canonical_sha256(payload),
        )

    def validated(self) -> Round21ReplayCondition:
        rebuilt = self.create(
            market=self.market,
            market_evidence=self.market_evidence,
            envelopes=self.envelopes,
            books=self.books,
            outcome=self.outcome,
            source_manifest_sha256=self.source_manifest_sha256,
            reconciliation_sha256=self.reconciliation_sha256,
        )
        if self != rebuilt or self.trading_authority:
            raise ValueError("Round 21 replay condition differs")
        return self

    def creation_book(
        self,
        *,
        outcome: str,
        decision_time_ms: int,
    ) -> PaperBookSnapshot | None:
        token = self.market.up_token_id if outcome == "Up" else self.market.down_token_id
        eligible = tuple(
            value
            for value in self.books
            if value.asset_id == token and value.received_wall_ms <= decision_time_ms
        )
        return eligible[-1] if eligible else None

    def execution_book(
        self,
        plan: Round21AggressiveOrderPlan,
    ) -> PaperBookSnapshot | None:
        selected = plan.validated()
        deadline = (
            selected.effective_execution_target_ms
            + POLYMARKET_ROUND21_MAXIMUM_EXECUTION_OBSERVATION_LATENESS_MS
        )
        return next(
            (
                value
                for value in self.books
                if value.asset_id == selected.token_id
                and selected.effective_execution_target_ms
                <= value.received_wall_ms
                <= deadline
            ),
            None,
        )


@dataclass(frozen=True, slots=True)
class Round21ReplayStep:
    condition_id: str
    decision_time_ms: int
    decision_sha256: str
    action: str
    observation_sha256: str
    execution_state: str
    filled_quantity: Decimal
    cash_after_quote: Decimal
    inventory_sha256: str
    conservative_equity_quote: Decimal
    step_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND21_ECONOMIC_REPLAY_SCHEMA_VERSION,
            "condition_id": self.condition_id,
            "decision_time_ms": self.decision_time_ms,
            "decision_sha256": self.decision_sha256,
            "action": self.action,
            "observation_sha256": self.observation_sha256,
            "execution_state": self.execution_state,
            "filled_quantity": format(self.filled_quantity, "f"),
            "cash_after_quote": format(self.cash_after_quote, "f"),
            "inventory_sha256": self.inventory_sha256,
            "conservative_equity_quote": format(
                self.conservative_equity_quote,
                "f",
            ),
        }

    @classmethod
    def create(
        cls,
        *,
        condition_id: str,
        decision_time_ms: int,
        decision_sha256: str,
        action: str,
        observation: Round21AggressiveExecutionObservation | None,
        cash_after_quote: Decimal,
        inventory: Round21BotInventory,
        conservative_equity_quote: Decimal,
    ) -> Round21ReplayStep:
        decision_sha = _digest(decision_sha256, name="action decision")
        cash = _decimal(cash_after_quote, name="cash after")
        equity = _decimal(conservative_equity_quote, name="conservative equity")
        selected_inventory = inventory.validated()
        if observation is None:
            observation_sha = ""
            state = "abstain"
            filled = Decimal("0")
        else:
            selected_observation = observation.validated()
            observation_sha = selected_observation.observation_sha256
            state = selected_observation.state
            filled = selected_observation.filled_quantity
        provisional = cls(
            condition_id=condition_id,
            decision_time_ms=int(decision_time_ms),
            decision_sha256=decision_sha,
            action=action,
            observation_sha256=observation_sha,
            execution_state=state,
            filled_quantity=filled,
            cash_after_quote=cash,
            inventory_sha256=selected_inventory.inventory_sha256,
            conservative_equity_quote=equity,
            step_sha256=_EMPTY_SHA256,
        )
        return replace(
            provisional,
            step_sha256=_canonical_sha256(provisional.identity_payload()),
        ).validated()

    def validated(self) -> Round21ReplayStep:
        decimals = (
            self.filled_quantity,
            self.cash_after_quote,
            self.conservative_equity_quote,
        )
        if (
            _CONDITION_ID.fullmatch(self.condition_id) is None
            or self.decision_time_ms <= 0
            or _SHA256.fullmatch(self.decision_sha256) is None
            or self.decision_sha256 == _EMPTY_SHA256
            or self.action
            not in {
                "abstain",
                "buy_up",
                "buy_down",
                "reduce_up",
                "reduce_down",
                "lock_up_with_down",
                "lock_down_with_up",
            }
            or self.execution_state
            not in {
                "abstain",
                "filled",
                "partial_fill",
                "known_no_fill",
                "unknown_after_submit",
            }
            or any(not value.is_finite() for value in decimals)
            or min(decimals) < 0
            or _SHA256.fullmatch(self.inventory_sha256) is None
            or self.inventory_sha256 == _EMPTY_SHA256
            or (
                self.action == "abstain"
                and (
                    self.execution_state != "abstain"
                    or self.observation_sha256
                    or self.filled_quantity != 0
                )
            )
            or (
                self.action != "abstain"
                and (
                    self.execution_state == "abstain"
                    or _SHA256.fullmatch(self.observation_sha256) is None
                    or self.observation_sha256 == _EMPTY_SHA256
                )
            )
            or self.step_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 21 replay step differs")
        return self


@dataclass(frozen=True, slots=True)
class Round21ConditionReplay:
    condition_id: str
    event_start_ms: int
    source_input_sha256: str
    outcome_sha256: str
    utility_quote: Decimal
    end_cash_quote: Decimal
    executed_action_count: int
    steps: tuple[Round21ReplayStep, ...]
    condition_result_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "condition_id": self.condition_id,
            "event_start_ms": self.event_start_ms,
            "source_input_sha256": self.source_input_sha256,
            "outcome_sha256": self.outcome_sha256,
            "utility_quote": format(self.utility_quote, "f"),
            "end_cash_quote": format(self.end_cash_quote, "f"),
            "executed_action_count": self.executed_action_count,
            "step_sha256": [value.step_sha256 for value in self.steps],
        }

    def validated(self) -> Round21ConditionReplay:
        steps = tuple(value.validated() for value in self.steps)
        if (
            _CONDITION_ID.fullmatch(self.condition_id) is None
            or self.event_start_ms <= 0
            or self.event_start_ms % 300_000
            or any(
                _SHA256.fullmatch(value) is None or value == _EMPTY_SHA256
                for value in (self.source_input_sha256, self.outcome_sha256)
            )
            or not self.utility_quote.is_finite()
            or not self.end_cash_quote.is_finite()
            or self.end_cash_quote < 0
            or self.executed_action_count
            != sum(value.filled_quantity > 0 for value in steps)
            or any(value.condition_id != self.condition_id for value in steps)
            or any(
                steps[index - 1].decision_time_ms >= steps[index].decision_time_ms
                for index in range(1, len(steps))
            )
            or self.condition_result_sha256
            != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 21 condition replay differs")
        return self


@dataclass(frozen=True, slots=True)
class Round21EconomicMetrics:
    condition_count: int
    calendar_day_count: int
    executed_action_count: int
    net_pnl_quote: Decimal
    mean_event_utility_quote: Decimal
    median_daily_pnl_quote: Decimal
    gross_profit_quote: Decimal
    gross_loss_quote: Decimal
    profit_factor: Decimal
    maximum_drawdown_fraction: Decimal
    realized_maximum_drawdown_fraction: Decimal
    tail_mean_worst_five_percent_quote: Decimal
    daily_mean_pnl_lower_95_quote: Decimal | None
    metric_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "condition_count": self.condition_count,
            "calendar_day_count": self.calendar_day_count,
            "executed_action_count": self.executed_action_count,
            "net_pnl_quote": format(self.net_pnl_quote, "f"),
            "mean_event_utility_quote": format(
                self.mean_event_utility_quote,
                "f",
            ),
            "median_daily_pnl_quote": format(self.median_daily_pnl_quote, "f"),
            "gross_profit_quote": format(self.gross_profit_quote, "f"),
            "gross_loss_quote": format(self.gross_loss_quote, "f"),
            "profit_factor": format(self.profit_factor, "f"),
            "maximum_drawdown_fraction": format(
                self.maximum_drawdown_fraction,
                "f",
            ),
            "realized_maximum_drawdown_fraction": format(
                self.realized_maximum_drawdown_fraction,
                "f",
            ),
            "tail_mean_worst_five_percent_quote": format(
                self.tail_mean_worst_five_percent_quote,
                "f",
            ),
            "daily_mean_pnl_lower_95_quote": (
                None
                if self.daily_mean_pnl_lower_95_quote is None
                else format(self.daily_mean_pnl_lower_95_quote, "f")
            ),
        }

    def validated(self) -> Round21EconomicMetrics:
        decimals = (
            self.net_pnl_quote,
            self.mean_event_utility_quote,
            self.median_daily_pnl_quote,
            self.gross_profit_quote,
            self.gross_loss_quote,
            self.profit_factor,
            self.maximum_drawdown_fraction,
            self.realized_maximum_drawdown_fraction,
            self.tail_mean_worst_five_percent_quote,
        )
        if (
            min(
                self.condition_count,
                self.calendar_day_count,
                self.executed_action_count,
            )
            < 0
            or any(not value.is_finite() for value in decimals)
            or min(
                self.gross_profit_quote,
                self.gross_loss_quote,
                self.profit_factor,
                self.maximum_drawdown_fraction,
                self.realized_maximum_drawdown_fraction,
            )
            < 0
            or self.profit_factor > 999
            or self.net_pnl_quote
            != self.gross_profit_quote - self.gross_loss_quote
            or (
                self.daily_mean_pnl_lower_95_quote is not None
                and not self.daily_mean_pnl_lower_95_quote.is_finite()
            )
            or self.metric_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 21 economic metrics differ")
        return self


@dataclass(frozen=True, slots=True)
class Round21EconomicReplay:
    profile: str
    scenario: str
    initial_capital_quote: Decimal
    final_cash_quote: Decimal
    metrics: Round21EconomicMetrics
    conditions: tuple[Round21ConditionReplay, ...]
    qualification_reasons: tuple[str, ...]
    economic_gate_passed: bool
    qualified: bool
    unknown_state_count: int
    risk_violation_count: int
    replay_sha256: str
    profitability_claim: bool = False
    paper_trading_authority: bool = False
    live_trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND21_ECONOMIC_REPLAY_SCHEMA_VERSION,
            "design_sha256": POLYMARKET_ROUND21_ECONOMIC_REPLAY_DESIGN_SHA256,
            "contract_sha256": POLYMARKET_ROUND21_CONTRACT_SHA256,
            "execution_policy_sha256": POLYMARKET_ROUND21_EXECUTION_POLICY_SHA256,
            "model_design_sha256": POLYMARKET_ROUND21_MODEL_DESIGN_SHA256,
            "probability_envelope_design_sha256": (
                POLYMARKET_ROUND21_PROBABILITY_ENVELOPE_DESIGN_SHA256
            ),
            "multi_action_policy_sha256": (
                POLYMARKET_ROUND21_MULTI_ACTION_POLICY_SHA256
            ),
            "profile": self.profile,
            "scenario": self.scenario,
            "initial_capital_quote": format(self.initial_capital_quote, "f"),
            "final_cash_quote": format(self.final_cash_quote, "f"),
            "metric_sha256": self.metrics.metric_sha256,
            "condition_result_sha256": [
                value.condition_result_sha256 for value in self.conditions
            ],
            "qualification_reasons": list(self.qualification_reasons),
            "economic_gate_passed": self.economic_gate_passed,
            "qualified": self.qualified,
            "unknown_state_count": self.unknown_state_count,
            "risk_violation_count": self.risk_violation_count,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }

    def validated(self) -> Round21EconomicReplay:
        profile = round21_risk_profile(self.profile)
        scenario = round21_execution_scenario(self.scenario)
        metrics = self.metrics.validated()
        conditions = tuple(value.validated() for value in self.conditions)
        if (
            profile.name != self.profile
            or scenario.name != self.scenario
            or not self.initial_capital_quote.is_finite()
            or self.initial_capital_quote <= 0
            or not self.final_cash_quote.is_finite()
            or self.final_cash_quote < 0
            or metrics.condition_count != len(conditions)
            or metrics.executed_action_count
            != sum(value.executed_action_count for value in conditions)
            or len(set(self.qualification_reasons))
            != len(self.qualification_reasons)
            or type(self.economic_gate_passed) is not bool
            or self.economic_gate_passed
            != (
                self.qualification_reasons
                == ("sealed_test_evidence_unavailable",)
            )
            or self.qualified
            or self.unknown_state_count < 0
            or self.risk_violation_count < 0
            or self.profitability_claim
            or self.paper_trading_authority
            or self.live_trading_authority
            or self.replay_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Round 21 economic replay differs")
        return self


def _inventory(
    condition_id: str,
    lots: Sequence[Round21OwnedLot],
    *,
    blocking: bool = False,
) -> Round21BotInventory:
    return Round21BotInventory.create(
        condition_id=condition_id,
        lots=lots,
        blocking_unknown_state=blocking,
    )


def _apply_fill(
    *,
    plan: Round21AggressiveOrderPlan,
    observation: Round21AggressiveExecutionObservation,
    lots: Sequence[Round21OwnedLot],
) -> tuple[Round21OwnedLot, ...]:
    selected = observation.validated()
    if selected.filled_quantity <= 0:
        return tuple(lots)
    output = list(lots)
    if plan.side == "BUY":
        output.append(
            Round21OwnedLot.create(
                condition_id=plan.condition_id,
                outcome=plan.outcome,
                quantity=selected.filled_quantity,
                cost_basis_quote=-selected.execution_cash_flow_quote,
                opened_at_ms=plan.effective_execution_target_ms,
                parent_inventory_id="fill-" + plan.plan_sha256[:48],
            )
        )
        return tuple(output)
    matches = [
        (index, lot)
        for index, lot in enumerate(output)
        if lot.parent_inventory_id == plan.parent_inventory_id
    ]
    if len(matches) != 1:
        raise RuntimeError("Round 21 replay parent inventory differs")
    index, lot = matches[0]
    if selected.filled_quantity > lot.quantity:
        raise RuntimeError("Round 21 replay sold foreign inventory")
    remaining = lot.quantity - selected.filled_quantity
    if remaining == 0:
        output.pop(index)
    else:
        output[index] = Round21OwnedLot.create(
            condition_id=lot.condition_id,
            outcome=lot.outcome,
            quantity=remaining,
            cost_basis_quote=lot.cost_basis_quote * remaining / lot.quantity,
            opened_at_ms=lot.opened_at_ms,
            parent_inventory_id=lot.parent_inventory_id,
        )
    return tuple(output)


def _drawdown_fraction(
    equity: Decimal,
    peak: Decimal,
    capital: Decimal,
) -> Decimal:
    return max(Decimal("0"), peak - equity) / capital


def _daily_lower_95(
    values: Sequence[Decimal],
    *,
    identity: str,
) -> Decimal | None:
    selected = tuple(values)
    if len(selected) < 2:
        return None
    block_length = isqrt(len(selected) - 1) + 1
    block_count = (len(selected) + block_length - 1) // block_length
    means: list[Decimal] = []
    for sample in range(POLYMARKET_ROUND21_DAILY_BOOTSTRAP_SAMPLES):
        draw: list[Decimal] = []
        for block in range(block_count):
            seed = hashlib.sha256(
                f"{identity}:{sample}:{block}".encode("ascii")
            ).digest()
            start = int.from_bytes(seed[:8], "big") % len(selected)
            draw.extend(
                selected[(start + offset) % len(selected)]
                for offset in range(block_length)
            )
        means.append(sum(draw[: len(selected)], start=Decimal("0")) / len(selected))
    means.sort()
    rank = (
        Decimal("0.025") * POLYMARKET_ROUND21_DAILY_BOOTSTRAP_SAMPLES
    ).to_integral_value(rounding=ROUND_CEILING)
    return means[max(0, int(rank) - 1)]


def _metrics(
    *,
    utilities: Sequence[Decimal],
    daily_values: Sequence[Decimal],
    executed_actions: int,
    maximum_drawdown: Decimal,
    realized_maximum_drawdown: Decimal,
    bootstrap_identity: str,
) -> Round21EconomicMetrics:
    utility = tuple(utilities)
    daily = tuple(daily_values)
    gains = sum((value for value in utility if value > 0), start=Decimal("0"))
    losses = -sum((value for value in utility if value < 0), start=Decimal("0"))
    if losses > 0:
        profit_factor = min(Decimal("999"), gains / losses)
    elif gains > 0:
        profit_factor = Decimal("999")
    else:
        profit_factor = Decimal("0")
    tail_count = max(1, (len(utility) * 5 + 99) // 100) if utility else 0
    tail = (
        sum(sorted(utility)[:tail_count], start=Decimal("0")) / tail_count
        if tail_count
        else Decimal("0")
    )
    provisional = Round21EconomicMetrics(
        condition_count=len(utility),
        calendar_day_count=len(daily),
        executed_action_count=executed_actions,
        net_pnl_quote=sum(utility, start=Decimal("0")),
        mean_event_utility_quote=(
            sum(utility, start=Decimal("0")) / len(utility)
            if utility
            else Decimal("0")
        ),
        median_daily_pnl_quote=(
            Decimal(median(daily)) if daily else Decimal("0")
        ),
        gross_profit_quote=gains,
        gross_loss_quote=losses,
        profit_factor=profit_factor,
        maximum_drawdown_fraction=maximum_drawdown,
        realized_maximum_drawdown_fraction=realized_maximum_drawdown,
        tail_mean_worst_five_percent_quote=tail,
        daily_mean_pnl_lower_95_quote=_daily_lower_95(
            daily,
            identity=bootstrap_identity,
        ),
        metric_sha256=_EMPTY_SHA256,
    )
    return replace(
        provisional,
        metric_sha256=_canonical_sha256(provisional.identity_payload()),
    )


def replay_round21_economics(
    conditions: Sequence[Round21ReplayCondition],
    *,
    scenario_name: str,
    risk_profile: str = "conservative",
    initial_capital_quote: Decimal = Decimal("10000"),
    minimum_edge_per_share: Decimal = Decimal("0.02"),
    builder_taker_fee_bps: Decimal = Decimal("0"),
) -> Round21EconomicReplay:
    """Replay one independent profile/scenario ledger without trading authority."""

    scenario = round21_execution_scenario(scenario_name)
    profile = round21_risk_profile(risk_profile)
    capital = _decimal(
        initial_capital_quote,
        name="initial capital",
        positive=True,
    )
    minimum_edge = _decimal(
        minimum_edge_per_share,
        name="minimum edge",
        positive=True,
    )
    builder_fee = _decimal(builder_taker_fee_bps, name="builder taker fee")
    selected = tuple(value.validated() for value in conditions)
    if (
        not selected
        or any(
            selected[index - 1].market.end_ms > selected[index].market.event_start_ms
            for index in range(1, len(selected))
        )
        or any(
            selected[index - 1].market.event_start_ms
            >= selected[index].market.event_start_ms
            for index in range(1, len(selected))
        )
    ):
        raise ValueError("Round 21 replay condition order is invalid")

    cash = capital
    settled_peak = capital
    conservative_peak = capital
    maximum_drawdown = Decimal("0")
    realized_maximum_drawdown = Decimal("0")
    cooldown_until_ms = 0
    consecutive_losses = 0
    day_values: dict[int, Decimal] = {}
    results: list[Round21ConditionReplay] = []
    utilities: list[Decimal] = []
    executed_actions = 0
    unknown_states = 0
    risk_violations = 0

    for condition in selected:
        market = condition.market
        condition_start_cash = cash
        day = market.event_start_ms // _DAY_MS
        lots: tuple[Round21OwnedLot, ...] = ()
        steps: list[Round21ReplayStep] = []
        condition_actions = 0
        unresolved = False
        for envelope in condition.envelopes:
            inventory = _inventory(market.condition_id, lots)
            settled_drawdown = _drawdown_fraction(cash, settled_peak, capital)
            decision = select_round21_action(
                market=market,
                market_evidence=condition.market_evidence,
                books={
                    "Up": condition.creation_book(
                        outcome="Up",
                        decision_time_ms=envelope.decision_time_ms,
                    ),
                    "Down": condition.creation_book(
                        outcome="Down",
                        decision_time_ms=envelope.decision_time_ms,
                    ),
                },
                envelope=envelope,
                inventory=inventory,
                decision_time_ms=envelope.decision_time_ms,
                risk_capital_quote=capital,
                available_cash_quote=cash,
                daily_realized_pnl_quote=day_values.get(day, Decimal("0")),
                drawdown_capital_fraction=settled_drawdown,
                cooldown_until_ms=cooldown_until_ms,
                transition_pending=False,
                reconciliation_ok=True,
                reconciliation_sha256=condition.reconciliation_sha256,
                minimum_edge_per_share=minimum_edge,
                risk_profile=profile.name,
                scenario_name=scenario.name,
                builder_taker_fee_bps=builder_fee,
            )
            observation: Round21AggressiveExecutionObservation | None = None
            if decision.plan is not None:
                observation = observe_round21_aggressive_execution(
                    decision.plan,
                    condition.execution_book(decision.plan),
                )
                if observation.state == "unknown_after_submit":
                    unknown_states += 1
                    inventory = _inventory(
                        market.condition_id,
                        lots,
                        blocking=True,
                    )
                    unresolved = True
                else:
                    cash += observation.execution_cash_flow_quote
                    lots = _apply_fill(
                        plan=decision.plan,
                        observation=observation,
                        lots=lots,
                    )
                    inventory = _inventory(market.condition_id, lots)
                    if observation.filled_quantity > 0:
                        condition_actions += 1
                        executed_actions += 1
            guaranteed = min(
                inventory.quantity("Up"),
                inventory.quantity("Down"),
            )
            conservative_equity = cash + guaranteed
            conservative_peak = max(conservative_peak, conservative_equity)
            maximum_drawdown = max(
                maximum_drawdown,
                _drawdown_fraction(
                    conservative_equity,
                    conservative_peak,
                    capital,
                ),
            )
            event_downside = max(
                Decimal("0"),
                condition_start_cash - conservative_equity,
            )
            settled_drawdown_quote = max(Decimal("0"), settled_peak - condition_start_cash)
            if (
                event_downside
                > capital * profile.maximum_event_loss_capital_fraction
                or max(
                    Decimal("0"),
                    -day_values.get(day, Decimal("0")),
                )
                + event_downside
                > capital * profile.maximum_daily_loss_capital_fraction
                or settled_drawdown_quote + event_downside
                > capital * profile.maximum_drawdown_capital_fraction
                or cash < 0
            ):
                risk_violations += 1
            steps.append(
                Round21ReplayStep.create(
                    condition_id=market.condition_id,
                    decision_time_ms=envelope.decision_time_ms,
                    decision_sha256=decision.decision_sha256,
                    action=decision.action,
                    observation=observation,
                    cash_after_quote=cash,
                    inventory=inventory,
                    conservative_equity_quote=conservative_equity,
                )
            )
            if unresolved:
                break
        if unresolved:
            break

        # The official outcome is consulted only after every causal decision.
        outcome = condition.outcome.validated()
        winning = "Up" if outcome.resolved_up else "Down"
        cash += _inventory(market.condition_id, lots).quantity(winning)
        utility = cash - condition_start_cash
        utilities.append(utility)
        day_values[day] = day_values.get(day, Decimal("0")) + utility
        if utility < 0:
            consecutive_losses += 1
            if consecutive_losses >= 2:
                cooldown_until_ms = (
                    market.end_ms + profile.loss_cluster_cooldown_minutes * 60_000
                )
        else:
            consecutive_losses = 0
        settled_peak = max(settled_peak, cash)
        realized_maximum_drawdown = max(
            realized_maximum_drawdown,
            _drawdown_fraction(cash, settled_peak, capital),
        )
        provisional_result = Round21ConditionReplay(
            condition_id=market.condition_id,
            event_start_ms=market.event_start_ms,
            source_input_sha256=condition.condition_input_sha256,
            outcome_sha256=outcome.outcome_sha256,
            utility_quote=utility,
            end_cash_quote=cash,
            executed_action_count=condition_actions,
            steps=tuple(steps),
            condition_result_sha256=_EMPTY_SHA256,
        )
        results.append(
            replace(
                provisional_result,
                condition_result_sha256=_canonical_sha256(
                    provisional_result.identity_payload()
                ),
            ).validated()
        )

    if unknown_states:
        metrics = _metrics(
            utilities=utilities,
            daily_values=tuple(day_values.values()),
            executed_actions=executed_actions,
            maximum_drawdown=maximum_drawdown,
            realized_maximum_drawdown=realized_maximum_drawdown,
            bootstrap_identity=f"{profile.name}:{scenario.name}:unknown",
        )
    else:
        metrics = _metrics(
            utilities=utilities,
            daily_values=tuple(day_values.values()),
            executed_actions=executed_actions,
            maximum_drawdown=maximum_drawdown,
            realized_maximum_drawdown=realized_maximum_drawdown,
            bootstrap_identity=f"{profile.name}:{scenario.name}",
        )
    reasons: list[str] = []
    if unknown_states:
        reasons.append("unknown_post_submit_state")
    if risk_violations:
        reasons.append("risk_limit_violation")
    if metrics.condition_count < POLYMARKET_ROUND21_MINIMUM_SEALED_CONDITIONS:
        reasons.append("insufficient_resolved_conditions")
    if metrics.calendar_day_count < POLYMARKET_ROUND21_MINIMUM_SEALED_DAYS:
        reasons.append("insufficient_calendar_days")
    if metrics.executed_action_count < POLYMARKET_ROUND21_MINIMUM_EXECUTED_ACTIONS:
        reasons.append("insufficient_executed_actions")
    if metrics.net_pnl_quote <= 0:
        reasons.append("net_pnl_not_positive")
    if metrics.mean_event_utility_quote <= 0:
        reasons.append("mean_event_utility_not_positive")
    if metrics.median_daily_pnl_quote <= 0:
        reasons.append("median_daily_pnl_not_positive")
    if metrics.profit_factor <= 1:
        reasons.append("profit_factor_not_above_one")
    if (
        metrics.daily_mean_pnl_lower_95_quote is None
        or metrics.daily_mean_pnl_lower_95_quote <= 0
    ):
        reasons.append("daily_pnl_lower_95_not_positive")
    if metrics.maximum_drawdown_fraction > profile.maximum_drawdown_capital_fraction:
        reasons.append("maximum_drawdown_limit_exceeded")
    reasons = list(dict.fromkeys(reasons))
    economic_gate_passed = not reasons
    reasons.append("sealed_test_evidence_unavailable")
    provisional_replay = Round21EconomicReplay(
        profile=profile.name,
        scenario=scenario.name,
        initial_capital_quote=capital,
        final_cash_quote=cash,
        metrics=metrics,
        conditions=tuple(results),
        qualification_reasons=tuple(reasons),
        economic_gate_passed=economic_gate_passed,
        qualified=False,
        unknown_state_count=unknown_states,
        risk_violation_count=risk_violations,
        replay_sha256=_EMPTY_SHA256,
    )
    return replace(
        provisional_replay,
        replay_sha256=_canonical_sha256(provisional_replay.identity_payload()),
    ).validated()


def replay_round21_full_matrix(
    conditions: Sequence[Round21ReplayCondition],
    *,
    initial_capital_quote: Decimal = Decimal("10000"),
    minimum_edge_per_share: Decimal = Decimal("0.02"),
    builder_taker_fee_bps: Decimal = Decimal("0"),
) -> tuple[Round21EconomicReplay, ...]:
    """Replay all 81 independent profile/scenario ledgers."""

    results = tuple(
        replay_round21_economics(
            conditions,
            scenario_name=scenario.name,
            risk_profile=profile,
            initial_capital_quote=initial_capital_quote,
            minimum_edge_per_share=minimum_edge_per_share,
            builder_taker_fee_bps=builder_taker_fee_bps,
        )
        for profile in ("conservative", "regular", "aggressive")
        for scenario in POLYMARKET_ROUND21_EXECUTION_SCENARIOS
    )
    if (
        len(results) != 81
        or len({(value.profile, value.scenario) for value in results}) != 81
    ):
        raise RuntimeError("Round 21 economic replay matrix differs")
    return results


credentials_used = False
account_connected = False
binance_execution_connected = False
paper_trading_authority = False
live_trading_authority = False


__all__ = [
    "POLYMARKET_ROUND21_ECONOMIC_REPLAY_DESIGN_SHA256",
    "POLYMARKET_ROUND21_ECONOMIC_REPLAY_SCHEMA_VERSION",
    "Round21ConditionReplay",
    "Round21EconomicMetrics",
    "Round21EconomicReplay",
    "Round21ReplayCondition",
    "Round21ReplayStep",
    "replay_round21_economics",
    "replay_round21_full_matrix",
]
