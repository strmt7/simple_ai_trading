"""Causal hold-to-settlement execution diagnostics for Polymarket models."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, replace
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np

from .paper_execution import (
    PaperOrderIntent,
    PolymarketFeeModel,
    paper_intent_id,
    simulate_aggressive_order,
)
from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_model import PolymarketModelSample
from .polymarket_replay import PolymarketEvidenceReplay, PolymarketRecordedBook


POLYMARKET_EXECUTION_CONFIG_SCHEMA_VERSION = "polymarket-execution-config-v1"
POLYMARKET_EXECUTION_TRADE_SCHEMA_VERSION = "polymarket-execution-trade-v2"
POLYMARKET_EXECUTION_REPORT_SCHEMA_VERSION = "polymarket-execution-report-v2"


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


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


def _finite_decimal(value: object, *, name: str, positive: bool = False) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite() or (positive and parsed <= 0):
        raise ValueError(f"{name} must be a finite positive decimal")
    return parsed


@dataclass(frozen=True)
class PolymarketExecutionResearchConfig:
    """Conservative fixed contract for prospective execution diagnostics."""

    submission_latency_ms: int = 100
    maximum_book_age_ms: int = 2_000
    order_ttl_ms: int = 30_000
    minimum_expected_edge_per_contract: Decimal = Decimal("0.02")
    initial_capital_quote: Decimal = Decimal("1000")
    maximum_loss_fraction_per_market: Decimal = Decimal("0.005")
    maximum_loss_fraction_per_time_group: Decimal = Decimal("0.015")

    def validated(self) -> "PolymarketExecutionResearchConfig":
        edge = _finite_decimal(
            self.minimum_expected_edge_per_contract,
            name="minimum_expected_edge_per_contract",
        )
        capital = _finite_decimal(
            self.initial_capital_quote,
            name="initial_capital_quote",
            positive=True,
        )
        per_market = _finite_decimal(
            self.maximum_loss_fraction_per_market,
            name="maximum_loss_fraction_per_market",
            positive=True,
        )
        per_group = _finite_decimal(
            self.maximum_loss_fraction_per_time_group,
            name="maximum_loss_fraction_per_time_group",
            positive=True,
        )
        if (
            not 1 <= int(self.submission_latency_ms) <= 60_000
            or not 0 <= int(self.maximum_book_age_ms) <= 60_000
            or not 1_000 <= int(self.order_ttl_ms) <= 300_000
            or not Decimal("0") <= edge <= Decimal("0.25")
            or not Decimal("10") <= capital <= Decimal("1000000000")
            or not Decimal("0") < per_market <= Decimal("0.10")
            or not per_market <= per_group <= Decimal("0.30")
        ):
            raise ValueError("Polymarket execution research configuration is invalid")
        return replace(
            self,
            minimum_expected_edge_per_contract=edge,
            initial_capital_quote=capital,
            maximum_loss_fraction_per_market=per_market,
            maximum_loss_fraction_per_time_group=per_group,
        )

    def asdict(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_EXECUTION_CONFIG_SCHEMA_VERSION,
            "submission_latency_ms": self.submission_latency_ms,
            "maximum_book_age_ms": self.maximum_book_age_ms,
            "order_ttl_ms": self.order_ttl_ms,
            "minimum_expected_edge_per_contract": _decimal_text(
                self.minimum_expected_edge_per_contract
            ),
            "initial_capital_quote": _decimal_text(self.initial_capital_quote),
            "maximum_loss_fraction_per_market": _decimal_text(
                self.maximum_loss_fraction_per_market
            ),
            "maximum_loss_fraction_per_time_group": _decimal_text(
                self.maximum_loss_fraction_per_time_group
            ),
        }


@dataclass(frozen=True)
class PolymarketExecutionTrade:
    trade_id: str
    sample_id: str
    condition_id: str
    market_id: str
    asset: str
    event_start_ms: int
    end_ms: int
    decision_received_wall_ms: int
    decision_received_monotonic_ns: int
    decision_delay_ms: int
    submission_latency_ms: int
    effective_latency_ms: int
    outcome: str
    predicted_probability: float
    expected_edge_per_contract: Decimal
    quantity: Decimal
    decision_best_ask: Decimal
    limit_price: Decimal
    decision_book_event_id: str
    execution_book_event_id: str
    execution_state: str
    execution_reason: str
    filled_quantity: Decimal
    average_fill_price: Decimal
    fee_quote: Decimal
    gross_payout_quote: Decimal
    realized_pnl_quote: Decimal
    official_resolution_event_id: str
    source_payload_sha256: str
    trade_sha256: str

    @property
    def filled(self) -> bool:
        return self.execution_state == "FILLED"


@dataclass(frozen=True)
class PolymarketEquityPoint:
    settled_at_ms: int
    group_realized_pnl_quote: Decimal
    equity_quote: Decimal
    peak_equity_quote: Decimal
    drawdown_quote: Decimal
    drawdown_fraction: Decimal

    def asdict(self) -> dict[str, object]:
        return {
            "settled_at_ms": self.settled_at_ms,
            "group_realized_pnl_quote": _decimal_text(
                self.group_realized_pnl_quote
            ),
            "equity_quote": _decimal_text(self.equity_quote),
            "peak_equity_quote": _decimal_text(self.peak_equity_quote),
            "drawdown_quote": _decimal_text(self.drawdown_quote),
            "drawdown_fraction": _decimal_text(self.drawdown_fraction),
        }


@dataclass(frozen=True)
class PolymarketExecutionReport:
    schema_version: str
    replay_run_id: str
    probability_input_sha256: str
    market_permission_sha256: str
    market_permissions: Mapping[str, bool]
    decision_delay_input_sha256: str
    decision_delay_ms_by_condition: Mapping[str, int]
    config: PolymarketExecutionResearchConfig
    evaluated_market_count: int
    signal_market_count: int
    attempted_order_count: int
    filled_order_count: int
    winning_order_count: int
    losing_order_count: int
    abstained_market_count: int
    reason_counts: Mapping[str, int]
    gross_deployed_capital_quote: Decimal
    gross_payout_quote: Decimal
    total_fees_quote: Decimal
    net_realized_pnl_quote: Decimal
    initial_capital_quote: Decimal
    final_equity_quote: Decimal
    return_on_initial_capital: Decimal
    return_on_deployed_capital: Decimal
    maximum_drawdown_quote: Decimal
    maximum_drawdown_fraction: Decimal
    trades: tuple[PolymarketExecutionTrade, ...]
    equity_curve: tuple[PolymarketEquityPoint, ...]
    report_sha256: str
    trading_authority: bool = False
    execution_claim: bool = False
    profitability_claim: bool = False
    portfolio_claim: bool = False
    leverage_applied: bool = False

    def asdict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "replay_run_id": self.replay_run_id,
            "probability_input_sha256": self.probability_input_sha256,
            "market_permission_sha256": self.market_permission_sha256,
            "market_permissions": dict(sorted(self.market_permissions.items())),
            "decision_delay_input_sha256": self.decision_delay_input_sha256,
            "decision_delay_ms_by_condition": dict(
                sorted(self.decision_delay_ms_by_condition.items())
            ),
            "config": self.config.asdict(),
            "evaluated_market_count": self.evaluated_market_count,
            "signal_market_count": self.signal_market_count,
            "attempted_order_count": self.attempted_order_count,
            "filled_order_count": self.filled_order_count,
            "winning_order_count": self.winning_order_count,
            "losing_order_count": self.losing_order_count,
            "abstained_market_count": self.abstained_market_count,
            "reason_counts": dict(self.reason_counts),
            "gross_deployed_capital_quote": _decimal_text(
                self.gross_deployed_capital_quote
            ),
            "gross_payout_quote": _decimal_text(self.gross_payout_quote),
            "total_fees_quote": _decimal_text(self.total_fees_quote),
            "net_realized_pnl_quote": _decimal_text(self.net_realized_pnl_quote),
            "initial_capital_quote": _decimal_text(self.initial_capital_quote),
            "final_equity_quote": _decimal_text(self.final_equity_quote),
            "return_on_initial_capital": _decimal_text(
                self.return_on_initial_capital
            ),
            "return_on_deployed_capital": _decimal_text(
                self.return_on_deployed_capital
            ),
            "maximum_drawdown_quote": _decimal_text(self.maximum_drawdown_quote),
            "maximum_drawdown_fraction": _decimal_text(
                self.maximum_drawdown_fraction
            ),
            "trades": [_trade_payload(item) for item in self.trades],
            "equity_curve": [item.asdict() for item in self.equity_curve],
            "report_sha256": self.report_sha256,
            "trading_authority": self.trading_authority,
            "execution_claim": self.execution_claim,
            "profitability_claim": self.profitability_claim,
            "portfolio_claim": self.portfolio_claim,
            "leverage_applied": self.leverage_applied,
        }


@dataclass(frozen=True)
class PolymarketPolicyCandidate:
    sample: PolymarketModelSample
    outcome: str
    predicted_probability: float
    expected_edge_per_contract: Decimal
    decision_best_ask: Decimal
    limit_price: Decimal

    def asdict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample.sample_id,
            "condition_id": self.sample.condition_id,
            "asset": self.sample.asset,
            "event_start_ms": self.sample.event_start_ms,
            "decision_received_wall_ms": self.sample.decision_received_wall_ms,
            "outcome": self.outcome,
            "predicted_probability": format(self.predicted_probability, ".17g"),
            "expected_edge_per_contract": _decimal_text(
                self.expected_edge_per_contract
            ),
            "decision_best_ask": _decimal_text(self.decision_best_ask),
            "limit_price": _decimal_text(self.limit_price),
        }


@dataclass(frozen=True)
class PolymarketPolicySelection:
    evaluated_market_count: int
    candidates: tuple[PolymarketPolicyCandidate, ...]
    reason_counts: Mapping[str, int]
    selection_sha256: str

    def asdict(self) -> dict[str, object]:
        return {
            "evaluated_market_count": self.evaluated_market_count,
            "candidate_count": len(self.candidates),
            "candidates": [item.asdict() for item in self.candidates],
            "reason_counts": dict(self.reason_counts),
            "selection_sha256": self.selection_sha256,
        }


class _ReplayBookIndex:
    def __init__(self, replay: PolymarketEvidenceReplay) -> None:
        books: dict[str, list[PolymarketRecordedBook]] = {}
        for book in replay.books:
            books.setdefault(book.token_id, []).append(book)
        self.books = {
            token: tuple(
                sorted(
                    values,
                    key=lambda item: (
                        item.received_monotonic_ns,
                        item.received_wall_ms,
                        item.sequence_number,
                        item.sub_index,
                        item.event_id,
                    ),
                )
            )
            for token, values in books.items()
        }
        self.timestamps = {
            token: tuple(item.received_monotonic_ns for item in values)
            for token, values in self.books.items()
        }

    def decision_book(
        self,
        token_id: str,
        *,
        condition_id: str,
        decision_monotonic_ns: int,
    ) -> PolymarketRecordedBook | None:
        values = self.books.get(token_id, ())
        timestamps = self.timestamps.get(token_id, ())
        index = bisect_right(timestamps, int(decision_monotonic_ns)) - 1
        while index >= 0:
            candidate = values[index]
            if candidate.market.condition_id == condition_id:
                return candidate
            index -= 1
        return None

    def execution_book(
        self,
        token_id: str,
        *,
        condition_id: str,
        decision_monotonic_ns: int,
        latency_ms: int,
        segment_id: str,
        market_end_ms: int,
    ) -> PolymarketRecordedBook | None:
        values = self.books.get(token_id, ())
        timestamps = self.timestamps.get(token_id, ())
        target = int(decision_monotonic_ns) + int(latency_ms) * 1_000_000
        index = bisect_right(timestamps, target) - 1
        while index >= 0:
            candidate = values[index]
            if candidate.market.condition_id != condition_id:
                index -= 1
                continue
            if candidate.segment_id != segment_id:
                return None
            if candidate.received_wall_ms >= market_end_ms:
                return None
            return candidate
        return None


def _maximum_economic_limit(
    *,
    fair_probability: float,
    quantity: Decimal,
    tick_size: Decimal,
    minimum_edge_per_contract: Decimal,
    fee: PolymarketFeeModel,
) -> Decimal | None:
    fair = _finite_decimal(
        format(float(fair_probability), ".17g"),
        name="fair probability",
        positive=True,
    )
    tick = _finite_decimal(tick_size, name="tick size", positive=True)
    if fair >= 1 or tick >= 1:
        raise ValueError("Polymarket probability or tick size is invalid")
    rough_maximum = fair - minimum_edge_per_contract
    if rough_maximum <= 0:
        return None
    ticks = int((rough_maximum / tick).to_integral_value(rounding=ROUND_FLOOR))
    while ticks > 0:
        price = tick * ticks
        if price >= 1:
            ticks -= 1
            continue
        edge = fair * quantity - price * quantity - fee(price, quantity, "taker")
        if edge >= minimum_edge_per_contract * quantity:
            return price
        ticks -= 1
    return None


def _policy_candidate(
    sample: PolymarketModelSample,
    predicted_up_probability: float,
    *,
    minimum_edge_per_contract: Decimal,
    fee: PolymarketFeeModel,
    quantity: Decimal,
    tick_size: Decimal,
) -> PolymarketPolicyCandidate | None:
    probability_up = float(predicted_up_probability)
    if not math.isfinite(probability_up) or not 0.0 < probability_up < 1.0:
        raise ValueError("Polymarket model probability must lie inside (0, 1)")
    choices: list[PolymarketPolicyCandidate] = []
    for outcome, fair_probability, raw_ask in (
        ("Up", probability_up, sample.up_best_ask),
        ("Down", 1.0 - probability_up, sample.down_best_ask),
    ):
        ask = _finite_decimal(format(raw_ask, ".17g"), name="decision best ask")
        edge_total = (
            _finite_decimal(
                format(fair_probability, ".17g"),
                name="predicted outcome probability",
            )
            * quantity
            - ask * quantity
            - fee(ask, quantity, "taker")
        )
        edge_per_contract = edge_total / quantity
        if edge_per_contract < minimum_edge_per_contract:
            continue
        limit = _maximum_economic_limit(
            fair_probability=fair_probability,
            quantity=quantity,
            tick_size=tick_size,
            minimum_edge_per_contract=minimum_edge_per_contract,
            fee=fee,
        )
        if limit is None or limit < ask:
            continue
        choices.append(
            PolymarketPolicyCandidate(
                sample=sample,
                outcome=outcome,
                predicted_probability=fair_probability,
                expected_edge_per_contract=edge_per_contract,
                decision_best_ask=ask,
                limit_price=limit,
            )
        )
    if not choices:
        return None
    choices.sort(
        key=lambda item: (
            -item.expected_edge_per_contract,
            item.outcome,
        )
    )
    if (
        len(choices) > 1
        and choices[0].expected_edge_per_contract
        == choices[1].expected_edge_per_contract
    ):
        return None
    return choices[0]


def build_polymarket_policy_selection(
    samples: Sequence[PolymarketModelSample],
    probabilities: Sequence[float] | np.ndarray,
    markets: Sequence[PolymarketFiveMinuteMarket],
    *,
    config: PolymarketExecutionResearchConfig | None = None,
) -> PolymarketPolicySelection:
    """Freeze the first positive after-cost proposal per market before execution."""

    cfg = (config or PolymarketExecutionResearchConfig()).validated()
    if not samples:
        raise ValueError("Polymarket policy selection requires model samples")
    predicted = np.asarray(probabilities, dtype=np.float64)
    if predicted.shape != (len(samples),) or not np.all(np.isfinite(predicted)):
        raise ValueError("Polymarket policy probability array is invalid")
    if len({item.sample_id for item in samples}) != len(samples):
        raise ValueError("Polymarket policy samples are duplicated")
    market_by_condition = {market.condition_id: market for market in markets}
    if len(market_by_condition) != len(markets):
        raise ValueError("Polymarket policy market metadata is duplicated")
    probability_by_sample = {
        sample.sample_id: float(probability)
        for sample, probability in zip(samples, predicted, strict=True)
    }
    samples_by_condition: dict[str, list[PolymarketModelSample]] = {}
    for sample in samples:
        market = market_by_condition.get(sample.condition_id)
        if (
            market is None
            or market.market_id != sample.market_id
            or market.asset != sample.asset
            or market.event_start_ms != sample.event_start_ms
            or market.end_ms != sample.end_ms
        ):
            raise ValueError("Polymarket policy sample and market metadata disagree")
        samples_by_condition.setdefault(sample.condition_id, []).append(sample)
    reasons: Counter[str] = Counter()
    candidates: list[PolymarketPolicyCandidate] = []
    for condition in sorted(samples_by_condition):
        market = market_by_condition[condition]
        candidate = None
        for sample in sorted(
            samples_by_condition[condition],
            key=lambda item: (
                item.decision_received_monotonic_ns,
                item.decision_received_wall_ms,
                item.sample_id,
            ),
        ):
            candidate = _policy_candidate(
                sample,
                probability_by_sample[sample.sample_id],
                minimum_edge_per_contract=cfg.minimum_expected_edge_per_contract,
                fee=market.fee_schedule.fee_model(),
                quantity=market.minimum_order_size,
                tick_size=market.tick_size,
            )
            if candidate is not None:
                break
        if candidate is None:
            reasons["no_positive_after_cost_edge"] += 1
            continue
        candidates.append(candidate)
    candidates.sort(
        key=lambda item: (
            item.sample.decision_received_monotonic_ns,
            item.sample.asset,
            item.sample.condition_id,
            item.sample.sample_id,
        )
    )
    payload = {
        "schema_version": "polymarket-policy-selection-v1",
        "config": cfg.asdict(),
        "sample_ids": [item.sample_id for item in samples],
        "probabilities": [format(float(value), ".17g") for value in predicted],
        "candidates": [item.asdict() for item in candidates],
        "reason_counts": dict(sorted(reasons.items())),
    }
    return PolymarketPolicySelection(
        evaluated_market_count=len(samples_by_condition),
        candidates=tuple(candidates),
        reason_counts=dict(sorted(reasons.items())),
        selection_sha256=_canonical_sha256(payload),
    )


def _trade_payload(trade: PolymarketExecutionTrade) -> dict[str, object]:
    return {
        "schema_version": POLYMARKET_EXECUTION_TRADE_SCHEMA_VERSION,
        "trade_id": trade.trade_id,
        "sample_id": trade.sample_id,
        "condition_id": trade.condition_id,
        "market_id": trade.market_id,
        "asset": trade.asset,
        "event_start_ms": trade.event_start_ms,
        "end_ms": trade.end_ms,
        "decision_received_wall_ms": trade.decision_received_wall_ms,
        "decision_received_monotonic_ns": trade.decision_received_monotonic_ns,
        "decision_delay_ms": trade.decision_delay_ms,
        "submission_latency_ms": trade.submission_latency_ms,
        "effective_latency_ms": trade.effective_latency_ms,
        "outcome": trade.outcome,
        "predicted_probability": format(trade.predicted_probability, ".17g"),
        "expected_edge_per_contract": _decimal_text(
            trade.expected_edge_per_contract
        ),
        "quantity": _decimal_text(trade.quantity),
        "decision_best_ask": _decimal_text(trade.decision_best_ask),
        "limit_price": _decimal_text(trade.limit_price),
        "decision_book_event_id": trade.decision_book_event_id,
        "execution_book_event_id": trade.execution_book_event_id,
        "execution_state": trade.execution_state,
        "execution_reason": trade.execution_reason,
        "filled_quantity": _decimal_text(trade.filled_quantity),
        "average_fill_price": _decimal_text(trade.average_fill_price),
        "fee_quote": _decimal_text(trade.fee_quote),
        "gross_payout_quote": _decimal_text(trade.gross_payout_quote),
        "realized_pnl_quote": _decimal_text(trade.realized_pnl_quote),
        "official_resolution_event_id": trade.official_resolution_event_id,
        "source_payload_sha256": trade.source_payload_sha256,
    }


def _report_payload(report: PolymarketExecutionReport) -> dict[str, object]:
    payload = report.asdict()
    payload.pop("report_sha256", None)
    return payload


def evaluate_polymarket_execution_policy(
    samples: Sequence[PolymarketModelSample],
    probabilities: Sequence[float] | np.ndarray,
    replay: PolymarketEvidenceReplay,
    *,
    config: PolymarketExecutionResearchConfig | None = None,
    market_permissions: Mapping[str, bool] | None = None,
    decision_delay_ms_by_condition: Mapping[str, int] | None = None,
) -> PolymarketExecutionReport:
    """Replay one causal FOK entry per market and settle only from official evidence."""

    cfg = (config or PolymarketExecutionResearchConfig()).validated()
    if not samples:
        raise ValueError("Polymarket execution evaluation requires model samples")
    predicted = np.asarray(probabilities, dtype=np.float64)
    if predicted.shape != (len(samples),) or not np.all(np.isfinite(predicted)):
        raise ValueError("Polymarket execution probability array is invalid")
    if len({item.sample_id for item in samples}) != len(samples):
        raise ValueError("Polymarket execution samples are duplicated")
    sample_conditions = {item.condition_id for item in samples}
    if market_permissions is None:
        permissions = {condition: True for condition in sample_conditions}
    else:
        permissions = dict(market_permissions)
        if set(permissions) != sample_conditions or any(
            not isinstance(value, bool) for value in permissions.values()
        ):
            raise ValueError(
                "Polymarket market permissions must bind every evaluated market"
            )
    if decision_delay_ms_by_condition is None:
        decision_delays = {condition: 0 for condition in sample_conditions}
    else:
        raw_delays = dict(decision_delay_ms_by_condition)
        try:
            decision_delays = {
                condition: int(value) for condition, value in raw_delays.items()
            }
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Polymarket decision delays must bind every evaluated market"
            ) from exc
        if (
            set(decision_delays) != sample_conditions
            or any(isinstance(value, bool) for value in raw_delays.values())
            or any(raw_delays[key] != value for key, value in decision_delays.items())
            or any(not 0 <= value <= 300_000 for value in decision_delays.values())
        ):
            raise ValueError(
                "Polymarket decision delays must bind every evaluated market"
            )
    market_permission_sha256 = _canonical_sha256(
        {
            "schema_version": "polymarket-market-permission-v1",
            "permissions": dict(sorted(permissions.items())),
        }
    )
    decision_delay_input_sha256 = _canonical_sha256(
        {
            "schema_version": "polymarket-decision-delay-input-v1",
            "decision_delay_ms_by_condition": dict(sorted(decision_delays.items())),
        }
    )
    if (
        replay.diagnostics.continuity_mode != "strict"
        or replay.diagnostics.stream_gap_count != 0
        or replay.diagnostics.book_sample_interval_ms != 0
    ):
        raise ValueError(
            "Polymarket execution evaluation requires strict unsampled gap-free replay"
        )
    if any(item.source_run_id != replay.run_id for item in samples):
        raise ValueError("Polymarket execution samples belong to another recorder run")
    probability_input_sha256 = _canonical_sha256(
        {
            "schema_version": "polymarket-probability-input-v1",
            "sample_ids": [item.sample_id for item in samples],
            "probabilities": [format(float(value), ".17g") for value in predicted],
            "market_permission_sha256": market_permission_sha256,
            "decision_delay_input_sha256": decision_delay_input_sha256,
        }
    )
    market_by_condition = {market.condition_id: market for market in replay.markets}
    if len(market_by_condition) != len(replay.markets):
        raise ValueError("Polymarket replay market metadata is duplicated")
    resolution_by_condition = {
        resolution.condition_id: resolution for resolution in replay.resolutions
    }
    if len(resolution_by_condition) != len(replay.resolutions):
        raise ValueError("Polymarket replay resolution evidence is duplicated")
    for sample in samples:
        market = market_by_condition.get(sample.condition_id)
        resolution = resolution_by_condition.get(sample.condition_id)
        if market is None or resolution is None:
            raise ValueError("Polymarket execution sample lacks official replay evidence")
        if (
            market.market_id != sample.market_id
            or market.asset != sample.asset
            or market.event_start_ms != sample.event_start_ms
            or market.end_ms != sample.end_ms
            or (resolution.winning_outcome == "Up") is not sample.official_up
            or resolution.event_id != sample.resolution_event_id
        ):
            raise ValueError("Polymarket execution evidence disagrees with model labels")

    selection = build_polymarket_policy_selection(
        samples,
        predicted,
        replay.markets,
        config=cfg,
    )
    candidates_by_group: dict[int, list[PolymarketPolicyCandidate]] = {}
    reasons: Counter[str] = Counter(selection.reason_counts)
    for candidate in selection.candidates:
        if not permissions[candidate.sample.condition_id]:
            reasons["external_fail_closed_veto"] += 1
            continue
        candidates_by_group.setdefault(candidate.sample.event_start_ms, []).append(
            candidate
        )

    book_index = _ReplayBookIndex(replay)
    equity = cfg.initial_capital_quote
    peak_equity = equity
    maximum_drawdown = Decimal("0")
    maximum_drawdown_fraction = Decimal("0")
    trades: list[PolymarketExecutionTrade] = []
    equity_curve: list[PolymarketEquityPoint] = []
    gross_deployed = Decimal("0")
    gross_payout = Decimal("0")
    total_fees = Decimal("0")
    winning = 0
    losing = 0
    group_end_by_start: dict[int, int] = {}
    for sample in samples:
        existing_end = group_end_by_start.setdefault(
            sample.event_start_ms,
            sample.end_ms,
        )
        if existing_end != sample.end_ms:
            raise ValueError("Polymarket time group has inconsistent market ends")

    for group_start, expected_group_end in sorted(group_end_by_start.items()):
        group_equity_start = equity
        market_risk_limit = (
            group_equity_start * cfg.maximum_loss_fraction_per_market
        )
        group_risk_limit = (
            group_equity_start * cfg.maximum_loss_fraction_per_time_group
        )
        deployed_group_risk = Decimal("0")
        group_pnl = Decimal("0")
        group_end = expected_group_end
        for candidate in sorted(
            candidates_by_group.get(group_start, ()),
            key=lambda item: (
                item.sample.decision_received_monotonic_ns,
                item.sample.asset,
                item.sample.sample_id,
            ),
        ):
            sample = candidate.sample
            market = market_by_condition[sample.condition_id]
            resolution = resolution_by_condition[sample.condition_id]
            fee = market.fee_schedule.fee_model()
            quantity = market.minimum_order_size
            worst_cost = (
                candidate.limit_price * quantity
                + fee(candidate.limit_price, quantity, "taker")
            )
            if worst_cost > market_risk_limit:
                reasons["minimum_order_exceeds_market_risk_budget"] += 1
                continue
            if deployed_group_risk + worst_cost > group_risk_limit:
                reasons["time_group_risk_budget_exhausted"] += 1
                continue
            token_id = (
                market.up_token_id
                if candidate.outcome == "Up"
                else market.down_token_id
            )
            decision_book = book_index.decision_book(
                token_id,
                condition_id=sample.condition_id,
                decision_monotonic_ns=sample.decision_received_monotonic_ns,
            )
            if decision_book is None:
                reasons["missing_causal_decision_book"] += 1
                continue
            if decision_book.received_wall_ms > sample.decision_received_wall_ms:
                raise ValueError("Polymarket decision book arrived after the model decision")
            expected_ask = (
                sample.up_best_ask
                if candidate.outcome == "Up"
                else sample.down_best_ask
            )
            if (
                not decision_book.snapshot.asks
                or float(decision_book.snapshot.asks[0].price) != expected_ask
            ):
                raise ValueError("Polymarket model quote disagrees with replay book state")
            decision_delay_ms = decision_delays[sample.condition_id]
            effective_latency_ms = decision_delay_ms + cfg.submission_latency_ms
            order_created_wall_ms = (
                sample.decision_received_wall_ms + decision_delay_ms
            )
            execution_wall_ms = (
                sample.decision_received_wall_ms + effective_latency_ms
            )
            execution_book = None
            if execution_wall_ms < market.end_ms:
                execution_book = book_index.execution_book(
                    token_id,
                    condition_id=sample.condition_id,
                    decision_monotonic_ns=sample.decision_received_monotonic_ns,
                    latency_ms=effective_latency_ms,
                    segment_id=decision_book.segment_id,
                    market_end_ms=market.end_ms,
                )
            if order_created_wall_ms >= market.end_ms or execution_wall_ms >= market.end_ms:
                execution_state = "EXPIRED"
                execution_reason = "decision_or_submission_completed_after_market_end"
                filled_quantity = Decimal("0")
                average_fill_price = Decimal("0")
                fee_quote = Decimal("0")
                source_payload_sha256 = decision_book.snapshot.source_payload_sha256
                execution_event_id = ""
            elif execution_book is None:
                execution_state = "UNKNOWN"
                execution_reason = "no_gap_free_causal_execution_book_at_latency"
                filled_quantity = Decimal("0")
                average_fill_price = Decimal("0")
                fee_quote = Decimal("0")
                source_payload_sha256 = decision_book.snapshot.source_payload_sha256
                execution_event_id = ""
            else:
                intent_id = paper_intent_id(
                    "polymarket",
                    sample.sample_id,
                    "open",
                )
                intent = PaperOrderIntent(
                    intent_id=intent_id,
                    venue="polymarket",
                    market_id=market.market_id,
                    asset_id=token_id,
                    symbol=market.asset,
                    outcome=candidate.outcome,
                    side="BUY",
                    order_type="FOK",
                    limit_price=candidate.limit_price,
                    quantity=quantity,
                    created_at_ms=order_created_wall_ms,
                    expires_at_ms=min(
                        market.end_ms,
                        order_created_wall_ms + cfg.order_ttl_ms,
                    ),
                ).validated()
                result = simulate_aggressive_order(
                    intent,
                    execution_book.snapshot,
                    execution_time_ms=execution_wall_ms,
                    submission_latency_ms=cfg.submission_latency_ms,
                    maximum_book_age_ms=cfg.maximum_book_age_ms,
                    fee=fee,
                )
                execution_state = result.state
                execution_reason = result.reason
                filled_quantity = result.filled_quantity
                average_fill_price = result.average_fill_price
                fee_quote = result.fee_quote
                source_payload_sha256 = result.source_payload_sha256
                execution_event_id = execution_book.event_id
            payout = Decimal("0")
            realized = Decimal("0")
            if execution_state == "FILLED":
                entry_cost = average_fill_price * filled_quantity + fee_quote
                payout = (
                    filled_quantity
                    if resolution.winning_outcome == candidate.outcome
                    else Decimal("0")
                )
                realized = payout - entry_cost
                deployed_group_risk += entry_cost
                gross_deployed += entry_cost
                gross_payout += payout
                total_fees += fee_quote
                group_pnl += realized
                if realized > 0:
                    winning += 1
                else:
                    losing += 1
            elif execution_state == "UNKNOWN":
                deployed_group_risk += worst_cost
            reasons[execution_reason] += 1
            trade_id = _canonical_sha256(
                {
                    "replay_run_id": replay.run_id,
                    "sample_id": sample.sample_id,
                    "config": cfg.asdict(),
                    "decision_delay_ms": decision_delay_ms,
                }
            )
            trade = PolymarketExecutionTrade(
                trade_id=trade_id,
                sample_id=sample.sample_id,
                condition_id=sample.condition_id,
                market_id=sample.market_id,
                asset=sample.asset,
                event_start_ms=sample.event_start_ms,
                end_ms=sample.end_ms,
                decision_received_wall_ms=sample.decision_received_wall_ms,
                decision_received_monotonic_ns=sample.decision_received_monotonic_ns,
                decision_delay_ms=decision_delay_ms,
                submission_latency_ms=cfg.submission_latency_ms,
                effective_latency_ms=effective_latency_ms,
                outcome=candidate.outcome,
                predicted_probability=candidate.predicted_probability,
                expected_edge_per_contract=candidate.expected_edge_per_contract,
                quantity=quantity,
                decision_best_ask=candidate.decision_best_ask,
                limit_price=candidate.limit_price,
                decision_book_event_id=decision_book.event_id,
                execution_book_event_id=execution_event_id,
                execution_state=execution_state,
                execution_reason=execution_reason,
                filled_quantity=filled_quantity,
                average_fill_price=average_fill_price,
                fee_quote=fee_quote,
                gross_payout_quote=payout,
                realized_pnl_quote=realized,
                official_resolution_event_id=resolution.event_id,
                source_payload_sha256=source_payload_sha256,
                trade_sha256="",
            )
            trades.append(
                replace(trade, trade_sha256=_canonical_sha256(_trade_payload(trade)))
            )
            group_end = max(group_end, market.end_ms)
        equity += group_pnl
        peak_equity = max(peak_equity, equity)
        drawdown = peak_equity - equity
        drawdown_fraction = (
            drawdown / peak_equity if peak_equity > 0 else Decimal("0")
        )
        maximum_drawdown = max(maximum_drawdown, drawdown)
        maximum_drawdown_fraction = max(
            maximum_drawdown_fraction,
            drawdown_fraction,
        )
        if group_end:
            equity_curve.append(
                PolymarketEquityPoint(
                    settled_at_ms=group_end,
                    group_realized_pnl_quote=group_pnl,
                    equity_quote=equity,
                    peak_equity_quote=peak_equity,
                    drawdown_quote=drawdown,
                    drawdown_fraction=drawdown_fraction,
                )
            )

    evaluated_markets = selection.evaluated_market_count
    signal_markets = sum(len(values) for values in candidates_by_group.values())
    filled_count = sum(item.filled for item in trades)
    net_pnl = equity - cfg.initial_capital_quote
    provisional = PolymarketExecutionReport(
        schema_version=POLYMARKET_EXECUTION_REPORT_SCHEMA_VERSION,
        replay_run_id=replay.run_id,
        probability_input_sha256=probability_input_sha256,
        market_permission_sha256=market_permission_sha256,
        market_permissions=dict(sorted(permissions.items())),
        decision_delay_input_sha256=decision_delay_input_sha256,
        decision_delay_ms_by_condition=dict(sorted(decision_delays.items())),
        config=cfg,
        evaluated_market_count=evaluated_markets,
        signal_market_count=signal_markets,
        attempted_order_count=len(trades),
        filled_order_count=filled_count,
        winning_order_count=winning,
        losing_order_count=losing,
        abstained_market_count=evaluated_markets - signal_markets,
        reason_counts=dict(sorted(reasons.items())),
        gross_deployed_capital_quote=gross_deployed,
        gross_payout_quote=gross_payout,
        total_fees_quote=total_fees,
        net_realized_pnl_quote=net_pnl,
        initial_capital_quote=cfg.initial_capital_quote,
        final_equity_quote=equity,
        return_on_initial_capital=net_pnl / cfg.initial_capital_quote,
        return_on_deployed_capital=(
            net_pnl / gross_deployed if gross_deployed > 0 else Decimal("0")
        ),
        maximum_drawdown_quote=maximum_drawdown,
        maximum_drawdown_fraction=maximum_drawdown_fraction,
        trades=tuple(trades),
        equity_curve=tuple(equity_curve),
        report_sha256="",
    )
    return replace(
        provisional,
        report_sha256=_canonical_sha256(_report_payload(provisional)),
    )


__all__ = [
    "POLYMARKET_EXECUTION_CONFIG_SCHEMA_VERSION",
    "POLYMARKET_EXECUTION_REPORT_SCHEMA_VERSION",
    "POLYMARKET_EXECUTION_TRADE_SCHEMA_VERSION",
    "PolymarketEquityPoint",
    "PolymarketExecutionReport",
    "PolymarketExecutionResearchConfig",
    "PolymarketExecutionTrade",
    "PolymarketPolicyCandidate",
    "PolymarketPolicySelection",
    "build_polymarket_policy_selection",
    "evaluate_polymarket_execution_policy",
]
