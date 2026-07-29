"""Promoted BTC fifteen-minute decisions for the independent Polymarket bot."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
from threading import Lock
import time
from typing import Callable

from .paper_execution import PolymarketFeeModel
from .polymarket import (
    PolymarketFiveMinuteMarket,
    PolymarketPublicClient,
    validate_clob_order_book,
)
from .polymarket_autonomous import PolymarketAutonomousOpenProposal
from .polymarket_autonomous_runtime import PolymarketAutonomousDecision
from .polymarket_live import PolymarketLiveBlocked
from .polymarket_live_promotion import VerifiedPolymarketLivePromotion
from .polymarket_round16 import ROUND16_DECISION_OFFSETS_SECONDS
from .polymarket_round16_shadow import (
    PolymarketRound16ShadowScorer,
)


_QUANTITY_STEP = Decimal("0.000001")
_HALF = Decimal("0.5")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def _quantity(value: object) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("requested quantity must be a finite decimal")
    try:
        parsed = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError("requested quantity must be a finite decimal") from exc
    if (
        not parsed.is_finite()
        or parsed <= 0
        or parsed.quantize(_QUANTITY_STEP) != parsed
    ):
        raise ValueError(
            "requested quantity must be positive with at most six decimals"
        )
    return parsed


class PolymarketRound16PromotedDecisionProvider:
    """Turn one accepted model timestamp into at most one live proposal."""

    def __init__(
        self,
        *,
        public_client: PolymarketPublicClient,
        scorer: PolymarketRound16ShadowScorer,
        promotion: VerifiedPolymarketLivePromotion,
        requested_quantity: Decimal,
        clock_ms: Callable[[], int] | None = None,
        monotonic_ns: Callable[[], int] | None = None,
    ) -> None:
        if not isinstance(public_client, PolymarketPublicClient):
            raise TypeError("public_client must be PolymarketPublicClient")
        if not isinstance(scorer, PolymarketRound16ShadowScorer):
            raise TypeError("scorer must be a Round 16 shadow scorer")
        if not isinstance(promotion, VerifiedPolymarketLivePromotion):
            raise PolymarketLiveBlocked(
                "Round 16 decisions require verified promotion evidence"
            )
        policy = promotion.promotion
        predictor = scorer.predictor
        if policy.market_variant != "fifteenminute":
            raise PolymarketLiveBlocked(
                "Round 16 decisions require a fifteen-minute promotion"
            )
        if (
            policy.model_artifact.sha256
            != predictor.pretest_file_sha256
            or policy.evaluation_report.sha256
            != predictor.evaluation_file_sha256
        ):
            raise PolymarketLiveBlocked(
                "Round 16 predictor files differ from promotion evidence"
            )
        self.public_client = public_client
        self.scorer = scorer
        self.promotion = promotion
        self.requested_quantity = _quantity(requested_quantity)
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1_000))
        self._monotonic_ns = monotonic_ns or time.monotonic_ns
        self._consumed: set[tuple[str, int]] = set()
        self._lock = Lock()

    def _scheduled_decision(
        self,
        market: PolymarketFiveMinuteMarket,
        *,
        observed_at_ms: int,
    ) -> int | None:
        maximum_age = self.promotion.promotion.maximum_prediction_age_ms
        eligible = [
            market.event_start_ms + offset * 1_000
            for offset in ROUND16_DECISION_OFFSETS_SECONDS
            if 0
            <= observed_at_ms - (market.event_start_ms + offset * 1_000)
            <= maximum_age
        ]
        return max(eligible, default=None)

    @staticmethod
    def _no_proposal(reason: str) -> PolymarketAutonomousDecision:
        return PolymarketAutonomousDecision(reasons=(reason,))

    def decide(
        self,
        *,
        markets: tuple[PolymarketFiveMinuteMarket, ...],
        observed_at_ms: int,
    ) -> PolymarketAutonomousDecision:
        now = int(observed_at_ms)
        self.promotion.assert_live_authority(observed_at_ms=now)
        if len(markets) != 1:
            raise PolymarketLiveBlocked(
                "Round 16 decision requires one active Polymarket market"
            )
        market = markets[0]
        if market.asset != "BTC" or market.horizon_minutes != 15:
            raise PolymarketLiveBlocked(
                "Round 16 decision market identity differs"
            )
        decision_time = self._scheduled_decision(
            market,
            observed_at_ms=now,
        )
        if decision_time is None:
            return self._no_proposal("no_scheduled_model_timestamp")
        key = (market.condition_id, decision_time)
        with self._lock:
            if key in self._consumed:
                return self._no_proposal("model_timestamp_already_consumed")
            self._consumed.add(key)
        shadow = self.scorer.evaluate(
            event_start_ms=market.event_start_ms,
            decision_time_ms=decision_time,
            observed_at_ms=now,
        )
        if shadow.status != "observed" or shadow.probability_up is None:
            return self._no_proposal(f"model_abstained:{shadow.reason}")
        if shadow.probability_up >= 0.5:
            outcome = "Up"
            token_id = market.up_token_id
            selected_probability = Decimal(
                format(shadow.probability_up, ".17g")
            )
        else:
            outcome = "Down"
            token_id = market.down_token_id
            selected_probability = Decimal(
                format(1.0 - shadow.probability_up, ".17g")
            )
        requested_at_ms = int(self._clock_ms())
        if requested_at_ms < now:
            raise PolymarketLiveBlocked(
                "Round 16 decision clock regressed"
            )
        book_payload = self.public_client.order_book(token_id)
        received_at_ms = int(self._clock_ms())
        if received_at_ms < requested_at_ms:
            raise PolymarketLiveBlocked(
                "Round 16 book receipt clock regressed"
            )
        book = validate_clob_order_book(
            market,
            token_id,
            book_payload,
            received_wall_ms=received_at_ms,
            received_monotonic_ns=int(self._monotonic_ns()),
        )
        maximum_age = self.promotion.promotion.maximum_prediction_age_ms
        if (
            book.source_time_ms > received_at_ms + 250
            or received_at_ms - book.source_time_ms > maximum_age
        ):
            return self._no_proposal("polymarket_book_stale")
        if self.requested_quantity < market.minimum_order_size:
            return self._no_proposal("requested_quantity_below_market_minimum")
        remaining = self.requested_quantity
        worst_price = Decimal("0")
        for level in book.asks:
            consumed = min(remaining, level.quantity)
            if consumed > 0:
                worst_price = level.price
                remaining -= consumed
            if remaining == 0:
                break
        if remaining > 0 or worst_price <= 0:
            return self._no_proposal("insufficient_displayed_ask_depth")
        fee_model = PolymarketFeeModel(
            enabled=market.fee_schedule.enabled,
            rate=market.fee_schedule.rate,
            exponent=market.fee_schedule.exponent,
            taker_only=market.fee_schedule.taker_only,
        )
        worst_fee = fee_model(
            min(worst_price, _HALF),
            self.requested_quantity,
            "taker",
        )
        edge = (
            selected_probability
            - worst_price
            - worst_fee / self.requested_quantity
        )
        if (
            edge
            < self.promotion.promotion.minimum_expected_edge_quote_per_share
        ):
            return self._no_proposal("insufficient_after_cost_edge")
        expires_at_ms = min(
            market.end_ms,
            decision_time + maximum_age,
            self.promotion.promotion.expires_at_ms,
        )
        if received_at_ms >= expires_at_ms:
            return self._no_proposal("proposal_expired_during_book_fetch")
        input_sha = _canonical_sha256(
            {
                "schema_version": "polymarket-round16-proposal-input-v1",
                "shadow_input_sha256": shadow.input_sha256,
                "book_payload_sha256": book.source_payload_sha256,
                "market_id": market.condition_id,
                "token_id": token_id,
                "outcome": outcome,
                "selected_outcome_probability": format(
                    selected_probability,
                    "f",
                ),
                "requested_quantity": format(
                    self.requested_quantity,
                    "f",
                ),
                "decision_time_ms": decision_time,
                "book_requested_at_ms": requested_at_ms,
                "book_received_at_ms": received_at_ms,
            }
        )
        proposal = PolymarketAutonomousOpenProposal(
            proposal_id=(
                "round16-" + market.condition_id[2:18] + f"-{decision_time}"
            ),
            input_sha256=input_sha,
            model_artifact_sha256=(
                self.promotion.promotion.model_artifact.sha256
            ),
            promotion_sha256=self.promotion.promotion.promotion_sha256,
            market_id=market.condition_id,
            token_id=token_id,
            symbol="BTC",
            market_variant="fifteenminute",
            outcome=outcome,
            selected_outcome_probability=selected_probability,
            requested_quantity=self.requested_quantity,
            event_start_time_ms=market.event_start_ms,
            event_end_time_ms=market.end_ms,
            decision_time_ms=decision_time,
            expires_at_ms=expires_at_ms,
        )
        return PolymarketAutonomousDecision(proposals=(proposal,))


__all__ = ["PolymarketRound16PromotedDecisionProvider"]
