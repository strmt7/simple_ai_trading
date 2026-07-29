"""Promotion-gated autonomous opening for the independent BTC Polymarket bot.

Binance observations may enter only through the read-only advisory decision.
They cannot grant authority, increase size, select a wallet, or bypass any
Polymarket ownership, reconciliation, funding, or execution gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_DOWN
import hashlib
import json
import re
import time
from typing import Callable

from .paper_execution import PolymarketFeeModel
from .polymarket_external_signal import PolymarketExternalSignalDecision
from .polymarket_live import (
    PolymarketLiveBlocked,
    PolymarketLiveCoordinator,
    PolymarketLiveOrderIntent,
    PolymarketLiveOrderRecord,
    PolymarketOpenQuote,
)
from .polymarket_live_promotion import VerifiedPolymarketLivePromotion


POLYMARKET_AUTONOMOUS_PROPOSAL_SCHEMA_VERSION = (
    "polymarket-autonomous-open-proposal-v1"
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_TOKEN_ID = re.compile(r"^[0-9]{20,80}$")
_QUANTITY_STEP = Decimal("0.000001")
_HALF = Decimal("0.5")
_MARKET_VARIANT_DURATION_MS = {
    "fiveminute": 300_000,
    "fifteenminute": 900_000,
}


def _decimal(value: object, *, name: str) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite decimal") from exc
    if not parsed.is_finite():
        raise ValueError(f"{name} must be a finite decimal")
    return parsed


def _sha(value: object, *, name: str) -> str:
    normalized = str(value or "").strip().lower()
    if _SHA256.fullmatch(normalized) is None:
        raise ValueError(f"{name} must be a SHA-256 digest")
    return normalized


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class PolymarketAutonomousOpenProposal:
    proposal_id: str
    input_sha256: str
    model_artifact_sha256: str
    promotion_sha256: str
    market_id: str
    token_id: str
    symbol: str
    market_variant: str
    outcome: str
    selected_outcome_probability: Decimal
    requested_quantity: Decimal
    event_start_time_ms: int
    event_end_time_ms: int
    decision_time_ms: int
    expires_at_ms: int
    proposal_sha256: str = ""

    def __post_init__(self) -> None:
        proposal_id = str(self.proposal_id or "").strip()
        if _IDENTIFIER.fullmatch(proposal_id) is None:
            raise ValueError("Polymarket proposal_id is invalid")
        object.__setattr__(self, "proposal_id", proposal_id)
        for field_name in (
            "input_sha256",
            "model_artifact_sha256",
            "promotion_sha256",
        ):
            object.__setattr__(
                self,
                field_name,
                _sha(getattr(self, field_name), name=field_name),
            )
        market_id = str(self.market_id or "").strip().lower()
        token_id = str(self.token_id or "").strip()
        if _CONDITION_ID.fullmatch(market_id) is None:
            raise ValueError("Polymarket proposal market_id is invalid")
        if _TOKEN_ID.fullmatch(token_id) is None:
            raise ValueError("Polymarket proposal token_id is invalid")
        object.__setattr__(self, "market_id", market_id)
        object.__setattr__(self, "token_id", token_id)
        symbol = str(self.symbol or "").strip().upper()
        variant = str(self.market_variant or "").strip().lower()
        outcome = str(self.outcome or "").strip().title()
        if symbol != "BTC" or variant not in _MARKET_VARIANT_DURATION_MS:
            raise ValueError(
                "autonomous live Polymarket proposals must be BTC 5m or 15m"
            )
        if outcome not in {"Up", "Down"}:
            raise ValueError("Polymarket proposal outcome must be Up or Down")
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "market_variant", variant)
        object.__setattr__(self, "outcome", outcome)
        probability = _decimal(
            self.selected_outcome_probability,
            name="selected_outcome_probability",
        )
        quantity = _decimal(self.requested_quantity, name="requested_quantity")
        if probability <= 0 or probability >= 1:
            raise ValueError("selected outcome probability must lie inside (0, 1)")
        if quantity <= 0 or quantity.quantize(_QUANTITY_STEP) != quantity:
            raise ValueError(
                "requested quantity must be positive with at most six decimals"
            )
        object.__setattr__(self, "selected_outcome_probability", probability)
        object.__setattr__(self, "requested_quantity", quantity)
        start = int(self.event_start_time_ms)
        end = int(self.event_end_time_ms)
        decision = int(self.decision_time_ms)
        expires = int(self.expires_at_ms)
        duration_ms = _MARKET_VARIANT_DURATION_MS[variant]
        if (
            start <= 0
            or start % duration_ms
            or end - start != duration_ms
            or not start <= decision < end
            or not decision < expires <= end
        ):
            raise ValueError("Polymarket proposal chronology is invalid")
        object.__setattr__(self, "event_start_time_ms", start)
        object.__setattr__(self, "event_end_time_ms", end)
        object.__setattr__(self, "decision_time_ms", decision)
        object.__setattr__(self, "expires_at_ms", expires)
        claimed = str(self.proposal_sha256 or "").strip().lower()
        actual = _canonical_sha256(self.body())
        if claimed and claimed != actual:
            raise ValueError("Polymarket proposal hash differs")
        object.__setattr__(self, "proposal_sha256", actual)

    def body(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_AUTONOMOUS_PROPOSAL_SCHEMA_VERSION,
            "proposal_id": self.proposal_id,
            "input_sha256": self.input_sha256,
            "model_artifact_sha256": self.model_artifact_sha256,
            "promotion_sha256": self.promotion_sha256,
            "venue": "polymarket",
            "protocol_version": 2,
            "market_id": self.market_id,
            "token_id": self.token_id,
            "symbol": self.symbol,
            "market_variant": self.market_variant,
            "outcome": self.outcome,
            "selected_outcome_probability": format(
                self.selected_outcome_probability,
                "f",
            ),
            "requested_quantity": format(self.requested_quantity, "f"),
            "event_start_time_ms": self.event_start_time_ms,
            "event_end_time_ms": self.event_end_time_ms,
            "decision_time_ms": self.decision_time_ms,
            "expires_at_ms": self.expires_at_ms,
        }

    def asdict(self) -> dict[str, object]:
        return {**self.body(), "proposal_sha256": self.proposal_sha256}


@dataclass(frozen=True, slots=True)
class PolymarketAutonomousOpenResult:
    record: PolymarketLiveOrderRecord
    quote: PolymarketOpenQuote
    effective_quantity: Decimal
    worst_notional_quote: Decimal
    worst_fee_quote: Decimal
    net_edge_quote_per_share: Decimal


def _assert_current(
    proposal: PolymarketAutonomousOpenProposal,
    promotion: VerifiedPolymarketLivePromotion,
    *,
    observed_at_ms: int,
) -> None:
    now = int(observed_at_ms)
    policy = promotion.promotion
    promotion.assert_live_authority(observed_at_ms=now)
    if now < proposal.event_start_time_ms or now >= proposal.event_end_time_ms:
        raise PolymarketLiveBlocked("Polymarket proposal is outside its event")
    if now < proposal.decision_time_ms:
        raise PolymarketLiveBlocked("Polymarket proposal timestamp is in the future")
    if now - proposal.decision_time_ms > policy.maximum_prediction_age_ms:
        raise PolymarketLiveBlocked("Polymarket proposal prediction is stale")
    if proposal.expires_at_ms <= now:
        raise PolymarketLiveBlocked("Polymarket proposal expired")
    remaining = proposal.event_end_time_ms - now
    if remaining < policy.minimum_remaining_seconds * 1_000:
        raise PolymarketLiveBlocked(
            "Polymarket proposal has too little event time remaining"
        )


def _effective_quantity(
    proposal: PolymarketAutonomousOpenProposal,
    signal: PolymarketExternalSignalDecision | None,
    *,
    observed_at_ms: int,
    maximum_signal_age_ms: int,
) -> Decimal:
    if signal is None:
        return proposal.requested_quantity
    if signal.action == "abstain":
        raise PolymarketLiveBlocked("external BTC price discovery vetoed the proposal")
    if signal.features is None:
        raise PolymarketLiveBlocked("external BTC price discovery lacks features")
    signal_time = int(signal.features.observed_at_ms)
    if signal_time > observed_at_ms or observed_at_ms - signal_time > maximum_signal_age_ms:
        raise PolymarketLiveBlocked("external BTC price discovery is stale")
    multiplier = signal.maximum_size_multiplier
    if signal.action == "preserve" and multiplier != 1:
        raise PolymarketLiveBlocked("external preserve decision changed proposal size")
    if signal.action == "reduce" and not Decimal("0") < multiplier < Decimal("1"):
        raise PolymarketLiveBlocked("external reduce decision is not strictly reducing")
    quantity = (proposal.requested_quantity * multiplier).quantize(
        _QUANTITY_STEP,
        rounding=ROUND_DOWN,
    )
    if quantity <= 0 or quantity > proposal.requested_quantity:
        raise PolymarketLiveBlocked("external BTC price discovery produced invalid size")
    return quantity


def submit_promoted_open(
    proposal: PolymarketAutonomousOpenProposal,
    promotion: VerifiedPolymarketLivePromotion,
    coordinator: PolymarketLiveCoordinator,
    *,
    external_signal: PolymarketExternalSignalDecision | None = None,
    clock_ms: Callable[[], int] | None = None,
) -> PolymarketAutonomousOpenResult:
    """Submit one immediate-or-cancelled open only after every authority gate."""

    if not isinstance(proposal, PolymarketAutonomousOpenProposal):
        raise TypeError("proposal must be PolymarketAutonomousOpenProposal")
    if not isinstance(promotion, VerifiedPolymarketLivePromotion):
        raise PolymarketLiveBlocked(
            "Polymarket autonomous open requires verified promotion evidence"
        )
    now_fn = clock_ms or (lambda: int(time.time() * 1_000))
    observed_at_ms = int(now_fn())
    policy = promotion.promotion
    _assert_current(proposal, promotion, observed_at_ms=observed_at_ms)
    if proposal.promotion_sha256 != policy.promotion_sha256:
        raise PolymarketLiveBlocked("Polymarket proposal promotion hash differs")
    if proposal.model_artifact_sha256 != policy.model_artifact.sha256:
        raise PolymarketLiveBlocked("Polymarket proposal model hash differs")
    if proposal.market_variant != policy.market_variant:
        raise PolymarketLiveBlocked("Polymarket proposal market variant differs")
    quantity = _effective_quantity(
        proposal,
        external_signal,
        observed_at_ms=observed_at_ms,
        maximum_signal_age_ms=policy.maximum_prediction_age_ms,
    )
    quote = coordinator.venue.open_quote(
        market_id=proposal.market_id,
        token_id=proposal.token_id,
        outcome=proposal.outcome,
        quantity=quantity,
        maximum_book_age_ms=policy.maximum_prediction_age_ms,
    )
    submit_at_ms = int(now_fn())
    _assert_current(proposal, promotion, observed_at_ms=submit_at_ms)
    if (
        quote.market_id != proposal.market_id
        or quote.token_id != proposal.token_id
        or quote.outcome != proposal.outcome
        or quote.quantity != quantity
    ):
        raise PolymarketLiveBlocked("Polymarket execution quote differs from proposal")
    if (
        quote.observed_at_ms > submit_at_ms + 250
        or submit_at_ms - quote.observed_at_ms > policy.maximum_prediction_age_ms
    ):
        raise PolymarketLiveBlocked("Polymarket execution quote is stale")
    if quantity < quote.minimum_order_size:
        raise PolymarketLiveBlocked(
            "reduced Polymarket quantity is below the venue minimum"
        )
    fee_model = PolymarketFeeModel(
        enabled=quote.fee_rate > 0,
        rate=quote.fee_rate,
        exponent=quote.fee_exponent,
        taker_only=True,
    )
    peak_fee_price = min(quote.limit_price, _HALF)
    worst_fee = fee_model(peak_fee_price, quantity, "taker")
    worst_notional = quote.limit_price * quantity
    net_edge = (
        proposal.selected_outcome_probability
        - quote.limit_price
        - worst_fee / quantity
    )
    if (
        net_edge <= 0
        or net_edge < policy.minimum_expected_edge_quote_per_share
    ):
        raise PolymarketLiveBlocked(
            "Polymarket proposal has insufficient after-cost edge"
        )
    intent_binding = {
        "schema_version": "polymarket-autonomous-intent-binding-v1",
        "proposal_sha256": proposal.proposal_sha256,
        "promotion_sha256": policy.promotion_sha256,
        "book_payload_sha256": quote.book_payload_sha256,
        "input_sha256": proposal.input_sha256,
        "model_artifact_sha256": proposal.model_artifact_sha256,
        "quantity": format(quantity, "f"),
        "limit_price": format(quote.limit_price, "f"),
        "worst_fee_quote": format(worst_fee, "f"),
    }
    intent_id = "poly-open-" + _canonical_sha256(intent_binding)[:32]
    expires_at_ms = min(
        proposal.expires_at_ms,
        proposal.event_end_time_ms,
        policy.expires_at_ms,
        submit_at_ms + policy.maximum_prediction_age_ms,
    )
    if expires_at_ms <= submit_at_ms:
        raise PolymarketLiveBlocked("Polymarket execution authorization expired")
    intent = PolymarketLiveOrderIntent(
        intent_id=intent_id,
        bot_id=policy.bot_id,
        market_id=proposal.market_id,
        token_id=proposal.token_id,
        symbol="BTC",
        outcome=proposal.outcome,
        side="BUY",
        order_type="FOK",
        limit_price=quote.limit_price,
        quantity=quantity,
        fee_reserve_quote=worst_fee,
        created_at_ms=submit_at_ms,
        expires_at_ms=expires_at_ms,
    )
    record = coordinator.submit(
        intent,
        tick_size=quote.tick_size,
        neg_risk=quote.neg_risk,
    )
    return PolymarketAutonomousOpenResult(
        record=record,
        quote=quote,
        effective_quantity=quantity,
        worst_notional_quote=worst_notional,
        worst_fee_quote=worst_fee,
        net_edge_quote_per_share=net_edge,
    )


__all__ = [
    "POLYMARKET_AUTONOMOUS_PROPOSAL_SCHEMA_VERSION",
    "PolymarketAutonomousOpenProposal",
    "PolymarketAutonomousOpenResult",
    "submit_promoted_open",
]
