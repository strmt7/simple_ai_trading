"""Public after-cost settlement-value screen for BTC Polymarket shadow telemetry."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import json
import time

from .paper_execution import PaperBookSnapshot
from .polymarket import (
    PolymarketFiveMinuteMarket,
    PolymarketPublicClient,
    validate_clob_market_info,
    validate_clob_order_book,
)
from .polymarket_historical_shadow import PolymarketHistoricalShadowDecision


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
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite() or (positive and parsed <= 0):
        raise ValueError(f"{name} must be a finite decimal")
    return parsed


def _decimal_text(value: Decimal) -> str:
    return format(value, "f")


@dataclass(frozen=True, slots=True)
class PolymarketShadowMarketState:
    market: PolymarketFiveMinuteMarket
    up_book: PaperBookSnapshot
    down_book: PaperBookSnapshot
    clob_market_info_sha256: str
    general_order_delay_seconds: int
    taker_order_delay_enabled: bool
    observed_at_ms: int
    trading_authority: bool = False

    def __post_init__(self) -> None:
        if self.trading_authority:
            raise ValueError("Polymarket shadow market state cannot have authority")
        if self.market.asset != "BTC":
            raise ValueError("Polymarket shadow market state is BTC-only")
        if (
            self.up_book.market_id != self.market.condition_id
            or self.down_book.market_id != self.market.condition_id
            or self.up_book.asset_id != self.market.up_token_id
            or self.down_book.asset_id != self.market.down_token_id
            or not self.up_book.connected
            or not self.down_book.connected
            or not self.up_book.gap_free
            or not self.down_book.gap_free
        ):
            raise ValueError("Polymarket shadow book identity or continuity differs")
        if (
            len(self.clob_market_info_sha256) != 64
            or int(self.general_order_delay_seconds) < 0
            or int(self.observed_at_ms) <= 0
            or type(self.taker_order_delay_enabled) is not bool
        ):
            raise ValueError("Polymarket shadow market metadata is invalid")


@dataclass(frozen=True, slots=True)
class PolymarketShadowFillQuote:
    outcome: str
    token_id: str
    quantity: Decimal
    filled_quantity: Decimal
    average_price: Decimal
    worst_price: Decimal
    notional_quote: Decimal
    fee_quote: Decimal
    total_cost_quote: Decimal
    total_cost_per_share: Decimal
    displayed_fillable: bool
    book_source_time_ms: int
    book_payload_sha256: str

    def asdict(self) -> dict[str, object]:
        return {
            "outcome": self.outcome,
            "token_id": self.token_id,
            "quantity": _decimal_text(self.quantity),
            "filled_quantity": _decimal_text(self.filled_quantity),
            "average_price": _decimal_text(self.average_price),
            "worst_price": _decimal_text(self.worst_price),
            "notional_quote": _decimal_text(self.notional_quote),
            "fee_quote": _decimal_text(self.fee_quote),
            "total_cost_quote": _decimal_text(self.total_cost_quote),
            "total_cost_per_share": _decimal_text(self.total_cost_per_share),
            "displayed_fillable": self.displayed_fillable,
            "book_source_time_ms": self.book_source_time_ms,
            "book_payload_sha256": self.book_payload_sha256,
        }


@dataclass(frozen=True, slots=True)
class PolymarketShadowOpportunity:
    status: str
    reason: str
    gamma_market_id: str
    condition_id: str
    slug: str
    gamma_payload_sha256: str
    event_start_ms: int
    event_end_ms: int
    decision_time_ms: int
    observed_at_ms: int
    quote_observation_latency_ms: int
    probability_up: str
    selected_outcome: str
    expected_terminal_value_per_share: str
    minimum_required_edge_per_share: str
    maximum_loss_quote: str
    up_quote: PolymarketShadowFillQuote
    down_quote: PolymarketShadowFillQuote
    model_candidate_id: str
    model_pretest_sha256: str
    model_evaluation_sha256: str
    model_support_profile_sha256: str
    outside_training_range_count: int
    extreme_outlier_count: int
    clob_market_info_sha256: str
    tick_size: str
    minimum_order_size: str
    fee_rate: str
    fee_exponent: int
    general_order_delay_seconds: int
    taker_order_delay_enabled: bool
    artifact_sha256: str
    trading_authority: bool = False
    proposal_authority: bool = False
    execution_or_profitability_claim: bool = False

    def __post_init__(self) -> None:
        if self.status not in {"candidate", "abstain"}:
            raise ValueError("Polymarket shadow opportunity status is invalid")
        if self.status == "candidate" and self.reason:
            raise ValueError("candidate Polymarket shadow opportunity has a reason")
        if self.status == "abstain" and not self.reason:
            raise ValueError("abstained Polymarket shadow opportunity lacks a reason")
        if (
            self.trading_authority
            or self.proposal_authority
            or self.execution_or_profitability_claim
        ):
            raise ValueError("Polymarket shadow opportunity cannot grant authority")
        if (
            not self.gamma_market_id
            or len(self.gamma_payload_sha256) != 64
            or not 0 <= int(self.quote_observation_latency_ms) <= 30_000
            or int(self.fee_exponent) <= 0
            or int(self.general_order_delay_seconds) < 0
            or type(self.taker_order_delay_enabled) is not bool
        ):
            raise ValueError("Polymarket shadow opportunity metadata is invalid")
        tick = _decimal(self.tick_size, name="tick size", positive=True)
        minimum = _decimal(
            self.minimum_order_size,
            name="minimum order size",
            positive=True,
        )
        fee_rate = _decimal(self.fee_rate, name="fee rate")
        if tick >= 1 or minimum <= 0 or not Decimal("0") <= fee_rate <= Decimal("1"):
            raise ValueError("Polymarket shadow opportunity economics are invalid")
        payload = self.asdict()
        claimed = payload.pop("artifact_sha256")
        if claimed != _canonical_sha256(payload):
            raise ValueError("Polymarket shadow opportunity hash differs")

    def asdict(self) -> dict[str, object]:
        return {
            "schema_version": "polymarket-btc-shadow-opportunity-v1",
            "status": self.status,
            "reason": self.reason,
            "gamma_market_id": self.gamma_market_id,
            "condition_id": self.condition_id,
            "slug": self.slug,
            "gamma_payload_sha256": self.gamma_payload_sha256,
            "event_start_ms": self.event_start_ms,
            "event_end_ms": self.event_end_ms,
            "decision_time_ms": self.decision_time_ms,
            "observed_at_ms": self.observed_at_ms,
            "quote_observation_latency_ms": self.quote_observation_latency_ms,
            "probability_up": self.probability_up,
            "selected_outcome": self.selected_outcome,
            "expected_terminal_value_per_share": (
                self.expected_terminal_value_per_share
            ),
            "minimum_required_edge_per_share": self.minimum_required_edge_per_share,
            "maximum_loss_quote": self.maximum_loss_quote,
            "up_quote": self.up_quote.asdict(),
            "down_quote": self.down_quote.asdict(),
            "model_candidate_id": self.model_candidate_id,
            "model_pretest_sha256": self.model_pretest_sha256,
            "model_evaluation_sha256": self.model_evaluation_sha256,
            "model_support_profile_sha256": (
                self.model_support_profile_sha256
            ),
            "outside_training_range_count": (
                self.outside_training_range_count
            ),
            "extreme_outlier_count": self.extreme_outlier_count,
            "clob_market_info_sha256": self.clob_market_info_sha256,
            "tick_size": self.tick_size,
            "minimum_order_size": self.minimum_order_size,
            "fee_rate": self.fee_rate,
            "fee_exponent": self.fee_exponent,
            "general_order_delay_seconds": self.general_order_delay_seconds,
            "taker_order_delay_enabled": self.taker_order_delay_enabled,
            "artifact_sha256": self.artifact_sha256,
            "trading_authority": self.trading_authority,
            "proposal_authority": self.proposal_authority,
            "execution_or_profitability_claim": self.execution_or_profitability_claim,
        }


def fetch_current_btc_shadow_market_state(
    client: PolymarketPublicClient,
    *,
    now_ms: int | None = None,
    maximum_book_age_ms: int = 2_000,
    maximum_cross_book_skew_ms: int = 1_000,
) -> PolymarketShadowMarketState:
    """Fetch and cross-check current public Gamma/CLOB market state."""

    if not isinstance(client, PolymarketPublicClient):
        raise TypeError("client must be PolymarketPublicClient")
    requested_at = time.time_ns() // 1_000_000 if now_ms is None else int(now_ms)
    maximum_age = int(maximum_book_age_ms)
    maximum_skew = int(maximum_cross_book_skew_ms)
    if requested_at <= 0 or not 100 <= maximum_age <= 10_000:
        raise ValueError("Polymarket shadow market timing is invalid")
    if not 0 <= maximum_skew <= 5_000:
        raise ValueError("Polymarket shadow cross-book skew is invalid")
    markets = client.discover_five_minute_markets(
        now_ms=requested_at,
        include_next=True,
        require_all_assets=True,
        assets=("BTC",),
    )
    current = [
        market
        for market in markets
        if market.event_start_ms <= requested_at < market.end_ms
    ]
    if len(current) != 1:
        raise ValueError("current BTC five-minute Polymarket identity is unavailable")
    market = current[0]
    market_info = validate_clob_market_info(
        market,
        client.clob_market_info(market.condition_id),
    )

    def book(token_id: str) -> PaperBookSnapshot:
        payload = client.order_book(token_id)
        received_wall_ms = time.time_ns() // 1_000_000
        return validate_clob_order_book(
            market,
            token_id,
            payload,
            received_wall_ms=received_wall_ms,
            received_monotonic_ns=time.monotonic_ns(),
        )

    up = book(market.up_token_id)
    down = book(market.down_token_id)
    observed_at = time.time_ns() // 1_000_000
    for snapshot in (up, down):
        if (
            snapshot.source_time_ms > snapshot.received_wall_ms + 1_000
            or snapshot.received_wall_ms > observed_at
            or observed_at - snapshot.source_time_ms > maximum_age
        ):
            raise ValueError("Polymarket shadow order book is stale or future-dated")
    if abs(up.source_time_ms - down.source_time_ms) > maximum_skew:
        raise ValueError("Polymarket shadow Up/Down books are not synchronized")
    return PolymarketShadowMarketState(
        market=market,
        up_book=up,
        down_book=down,
        clob_market_info_sha256=str(market_info["payload_sha256"]),
        general_order_delay_seconds=int(
            market_info["general_order_delay_seconds"]
        ),
        taker_order_delay_enabled=bool(
            market_info["taker_order_delay_enabled"]
        ),
        observed_at_ms=observed_at,
    )


def _fill_quote(
    state: PolymarketShadowMarketState,
    *,
    outcome: str,
    quantity: Decimal,
) -> PolymarketShadowFillQuote:
    if outcome == "Up":
        book = state.up_book
        token_id = state.market.up_token_id
    elif outcome == "Down":
        book = state.down_book
        token_id = state.market.down_token_id
    else:
        raise ValueError("Polymarket shadow outcome is invalid")
    remaining = quantity
    filled = Decimal("0")
    notional = Decimal("0")
    fee_quote = Decimal("0")
    worst_price = Decimal("0")
    fee = state.market.fee_schedule.fee_model()
    for level in book.asks:
        consumed = min(remaining, level.quantity)
        if consumed <= 0:
            continue
        filled += consumed
        notional += consumed * level.price
        fee_quote += fee(level.price, consumed, "taker")
        worst_price = level.price
        remaining -= consumed
        if remaining <= 0:
            break
    fillable = remaining <= 0
    total = notional + fee_quote
    return PolymarketShadowFillQuote(
        outcome=outcome,
        token_id=token_id,
        quantity=quantity,
        filled_quantity=filled,
        average_price=notional / filled if filled > 0 else Decimal("0"),
        worst_price=worst_price,
        notional_quote=notional,
        fee_quote=fee_quote,
        total_cost_quote=total,
        total_cost_per_share=total / filled if filled > 0 else Decimal("0"),
        displayed_fillable=fillable,
        book_source_time_ms=book.source_time_ms,
        book_payload_sha256=book.source_payload_sha256,
    )


def evaluate_shadow_settlement_opportunity(
    prediction: PolymarketHistoricalShadowDecision,
    state: PolymarketShadowMarketState,
    *,
    quantity: Decimal | None = None,
    minimum_edge_per_share: Decimal = Decimal("0.01"),
    maximum_prediction_age_ms: int = 5_000,
    minimum_remaining_seconds: int = 20,
) -> PolymarketShadowOpportunity:
    """Compare frozen endpoint probability with current executable ask depth."""

    if not isinstance(prediction, PolymarketHistoricalShadowDecision):
        raise TypeError("prediction must be PolymarketHistoricalShadowDecision")
    if not isinstance(state, PolymarketShadowMarketState):
        raise TypeError("state must be PolymarketShadowMarketState")
    if prediction.status != "observed" or prediction.probability_up is None:
        raise ValueError("Polymarket shadow opportunity requires an observed prediction")
    market = state.market
    if (
        prediction.event_start_ms != market.event_start_ms
        or prediction.decision_time_ms < market.event_start_ms
        or prediction.decision_time_ms >= market.end_ms
    ):
        raise ValueError("Polymarket shadow prediction and market windows differ")
    selected_quantity = (
        market.minimum_order_size
        if quantity is None
        else _decimal(quantity, name="quantity", positive=True)
    )
    minimum_edge = _decimal(
        minimum_edge_per_share,
        name="minimum edge",
        positive=True,
    )
    maximum_age = int(maximum_prediction_age_ms)
    minimum_remaining = int(minimum_remaining_seconds)
    if selected_quantity < market.minimum_order_size:
        raise ValueError("Polymarket shadow quantity is below the venue minimum")
    if not 100 <= maximum_age <= 30_000 or not 1 <= minimum_remaining <= 240:
        raise ValueError("Polymarket shadow opportunity timing limits are invalid")
    probability_up = _decimal(
        format(prediction.probability_up, ".17g"),
        name="probability_up",
    )
    if not Decimal("0") < probability_up < Decimal("1"):
        raise ValueError("Polymarket shadow probability is invalid")
    up_quote = _fill_quote(state, outcome="Up", quantity=selected_quantity)
    down_quote = _fill_quote(state, outcome="Down", quantity=selected_quantity)
    fair = {"Up": probability_up, "Down": Decimal("1") - probability_up}
    quotes = {"Up": up_quote, "Down": down_quote}
    edges = {
        outcome: (
            fair[outcome] - quote.total_cost_per_share
            if quote.displayed_fillable
            else Decimal("-1")
        )
        for outcome, quote in quotes.items()
    }
    outcome = max(edges, key=lambda value: (edges[value], value == "Up"))
    edge = edges[outcome]
    reasons: list[str] = []
    if state.observed_at_ms - prediction.decision_time_ms > maximum_age:
        reasons.append("prediction_stale")
    if market.end_ms - state.observed_at_ms < minimum_remaining * 1_000:
        reasons.append("insufficient_market_time_remaining")
    if state.general_order_delay_seconds * 1_000 >= (
        market.end_ms - state.observed_at_ms
    ):
        reasons.append("venue_order_delay_exceeds_remaining_time")
    if not quotes[outcome].displayed_fillable:
        reasons.append("displayed_depth_insufficient")
    if edge < minimum_edge:
        reasons.append("after_cost_edge_below_threshold")
    status = "abstain" if reasons else "candidate"
    body: dict[str, object] = {
        "schema_version": "polymarket-btc-shadow-opportunity-v1",
        "status": status,
        "reason": ",".join(reasons),
        "gamma_market_id": market.market_id,
        "condition_id": market.condition_id,
        "slug": market.slug,
        "gamma_payload_sha256": market.gamma_payload_sha256,
        "event_start_ms": market.event_start_ms,
        "event_end_ms": market.end_ms,
        "decision_time_ms": prediction.decision_time_ms,
        "observed_at_ms": state.observed_at_ms,
        "quote_observation_latency_ms": (
            state.observed_at_ms - prediction.decision_time_ms
        ),
        "probability_up": _decimal_text(probability_up),
        "selected_outcome": outcome,
        "expected_terminal_value_per_share": _decimal_text(edge),
        "minimum_required_edge_per_share": _decimal_text(minimum_edge),
        "maximum_loss_quote": _decimal_text(quotes[outcome].total_cost_quote),
        "up_quote": up_quote.asdict(),
        "down_quote": down_quote.asdict(),
        "model_candidate_id": prediction.candidate_id,
        "model_pretest_sha256": prediction.pretest_artifact_sha256,
        "model_evaluation_sha256": prediction.evaluation_artifact_sha256,
        "model_support_profile_sha256": prediction.support_profile_sha256,
        "outside_training_range_count": (
            prediction.outside_training_range_count
        ),
        "extreme_outlier_count": prediction.extreme_outlier_count,
        "clob_market_info_sha256": state.clob_market_info_sha256,
        "tick_size": _decimal_text(market.tick_size),
        "minimum_order_size": _decimal_text(market.minimum_order_size),
        "fee_rate": _decimal_text(market.fee_schedule.rate),
        "fee_exponent": market.fee_schedule.exponent,
        "general_order_delay_seconds": state.general_order_delay_seconds,
        "taker_order_delay_enabled": state.taker_order_delay_enabled,
        "trading_authority": False,
        "proposal_authority": False,
        "execution_or_profitability_claim": False,
    }
    artifact_sha = _canonical_sha256(body)
    return PolymarketShadowOpportunity(
        status=status,
        reason=str(body["reason"]),
        gamma_market_id=market.market_id,
        condition_id=market.condition_id,
        slug=market.slug,
        gamma_payload_sha256=market.gamma_payload_sha256,
        event_start_ms=market.event_start_ms,
        event_end_ms=market.end_ms,
        decision_time_ms=prediction.decision_time_ms,
        observed_at_ms=state.observed_at_ms,
        quote_observation_latency_ms=(
            state.observed_at_ms - prediction.decision_time_ms
        ),
        probability_up=str(body["probability_up"]),
        selected_outcome=outcome,
        expected_terminal_value_per_share=str(
            body["expected_terminal_value_per_share"]
        ),
        minimum_required_edge_per_share=str(
            body["minimum_required_edge_per_share"]
        ),
        maximum_loss_quote=str(body["maximum_loss_quote"]),
        up_quote=up_quote,
        down_quote=down_quote,
        model_candidate_id=prediction.candidate_id,
        model_pretest_sha256=prediction.pretest_artifact_sha256,
        model_evaluation_sha256=prediction.evaluation_artifact_sha256,
        model_support_profile_sha256=prediction.support_profile_sha256,
        outside_training_range_count=(
            prediction.outside_training_range_count
        ),
        extreme_outlier_count=prediction.extreme_outlier_count,
        clob_market_info_sha256=state.clob_market_info_sha256,
        tick_size=_decimal_text(market.tick_size),
        minimum_order_size=_decimal_text(market.minimum_order_size),
        fee_rate=_decimal_text(market.fee_schedule.rate),
        fee_exponent=market.fee_schedule.exponent,
        general_order_delay_seconds=state.general_order_delay_seconds,
        taker_order_delay_enabled=state.taker_order_delay_enabled,
        artifact_sha256=artifact_sha,
    )


__all__ = [
    "PolymarketShadowFillQuote",
    "PolymarketShadowMarketState",
    "PolymarketShadowOpportunity",
    "evaluate_shadow_settlement_opportunity",
    "fetch_current_btc_shadow_market_state",
]
