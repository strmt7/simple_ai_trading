"""Promotion-gated Round 21 decisions for the independent Polymarket bot."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import hashlib
import json
from threading import Lock
import time
from typing import Callable

from .polymarket import (
    PolymarketFiveMinuteMarket,
    PolymarketPublicClient,
    validate_clob_order_book,
)
from .polymarket_autonomous import PolymarketAutonomousOpenProposal
from .polymarket_autonomous import (
    PolymarketAutonomousLockProposal,
    PolymarketAutonomousReduceProposal,
)
from .polymarket_autonomous_runtime import (
    PolymarketAutonomousDecision,
    PolymarketAutonomousPortfolio,
)
from .polymarket_live import PolymarketLiveBlocked
from .polymarket_live_promotion import VerifiedPolymarketLivePromotion
from .polymarket_fees import PolymarketFeeModel
from .polymarket_round21_live_features import Round21CoordinatedPrediction
from .polymarket_round21_model import (
    VerifiedRound21DevelopmentArtifact,
    load_verified_round21_development_artifact,
)
from .polymarket_round21_policy import round21_risk_profile
from .polymarket_round21_session import Round21RollingPublicDataService


POLYMARKET_ROUND21_PROMOTED_DECISION_SCHEMA_VERSION = (
    "polymarket-round21-promoted-decision-v1"
)
_QUANTITY_STEP = Decimal("0.000001")
_HALF = Decimal("0.5")
_MAXIMUM_CREATION_BOOK_AGE_MS = 500


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


class PolymarketRound21PromotedDecisionProvider:
    """Turn one promoted conservative probability envelope into one proposal."""

    credentials_used = False
    account_connected = False
    binance_connected = False
    trading_authority = False

    def __init__(
        self,
        *,
        public_client: PolymarketPublicClient,
        data_service: Round21RollingPublicDataService,
        promotion: VerifiedPolymarketLivePromotion,
        requested_quantity: Decimal,
        risk_level: str = "conservative",
        artifact_evidence: VerifiedRound21DevelopmentArtifact | None = None,
        clock_ms: Callable[[], int] | None = None,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(public_client, PolymarketPublicClient):
            raise TypeError("Round 21 decision public client differs")
        if not isinstance(data_service, Round21RollingPublicDataService):
            raise TypeError("Round 21 decision data service differs")
        if not isinstance(promotion, VerifiedPolymarketLivePromotion):
            raise PolymarketLiveBlocked(
                "Round 21 decisions require verified promotion evidence"
            )
        policy = promotion.promotion
        if policy.market_variant != "fiveminute":
            raise PolymarketLiveBlocked(
                "Round 21 decisions require a five-minute promotion"
            )
        evidence = artifact_evidence or load_verified_round21_development_artifact(
            promotion.model_artifact_path,
            expected_file_sha256=policy.model_artifact.sha256,
        )
        if (
            not isinstance(evidence, VerifiedRound21DevelopmentArtifact)
            or evidence.path != promotion.model_artifact_path.resolve()
            or evidence.file_sha256 != policy.model_artifact.sha256
        ):
            raise PolymarketLiveBlocked(
                "Round 21 model file differs from promotion evidence"
            )
        model_sha = evidence.artifact_sha256
        if data_service.scorer.source_model_artifact_sha256 != model_sha:
            raise PolymarketLiveBlocked(
                "Round 21 scorer differs from promotion evidence"
            )
        if not callable(monotonic_ns):
            raise TypeError("Round 21 decision monotonic clock is invalid")
        if clock_ms is not None and not callable(clock_ms):
            raise TypeError("Round 21 decision wall clock is invalid")
        self.public_client = public_client
        self.data_service = data_service
        self.promotion = promotion
        self.artifact_evidence = evidence
        self.source_model_artifact_sha256 = model_sha
        self.requested_quantity = _quantity(requested_quantity)
        self.risk_profile = round21_risk_profile(risk_level)
        self._clock_ms = clock_ms or (lambda: int(time.time() * 1_000))
        self._monotonic_ns = monotonic_ns
        self._consumed: set[tuple[str, int]] = set()
        self._active_condition_id = ""
        self._lock = Lock()

    @staticmethod
    def _no_proposal(reason: str) -> PolymarketAutonomousDecision:
        selected = str(reason or "round21_decision_unavailable").strip()[:160]
        return PolymarketAutonomousDecision(reasons=(selected,))

    def _claim_timestamp(self, condition_id: str, decision_time_ms: int) -> bool:
        key = (condition_id, int(decision_time_ms))
        with self._lock:
            if condition_id != self._active_condition_id:
                self._consumed.clear()
                self._active_condition_id = condition_id
            if key in self._consumed:
                return False
            self._consumed.add(key)
            return True

    @staticmethod
    def _walk_price(
        levels: tuple[object, ...],
        quantity: Decimal,
    ) -> Decimal | None:
        remaining = quantity
        worst = Decimal("0")
        for level in levels:
            level_quantity = Decimal(str(getattr(level, "quantity")))
            consumed = min(remaining, level_quantity)
            if consumed > 0:
                worst = Decimal(str(getattr(level, "price")))
                remaining -= consumed
            if remaining == 0:
                return worst
        return None

    def _build_inventory_transition(
        self,
        market: PolymarketFiveMinuteMarket,
        *,
        coordinated: Round21CoordinatedPrediction,
        portfolio: PolymarketAutonomousPortfolio,
        observed_at_ms: int,
    ) -> PolymarketAutonomousDecision:
        result = coordinated.validated()
        if result.status != "observed" or result.prediction is None:
            reason = result.reasons[0] if result.reasons else "round21_model_abstained"
            return self._no_proposal(reason)
        prediction = result.prediction.validated()
        envelope = prediction.envelope
        if envelope is None:
            return self._no_proposal("round21_probability_envelope_unavailable")
        evidence = envelope.validated()
        maximum_book_age = min(
            _MAXIMUM_CREATION_BOOK_AGE_MS,
            self.promotion.promotion.maximum_prediction_age_ms,
        )
        books: dict[str, object] = {}
        received_at_ms = observed_at_ms
        for outcome, token_id in (
            ("Up", market.up_token_id),
            ("Down", market.down_token_id),
        ):
            payload = self.public_client.order_book(token_id)
            received_at_ms = int(self._clock_ms())
            book = validate_clob_order_book(
                market,
                token_id,
                payload,
                received_wall_ms=received_at_ms,
                received_monotonic_ns=int(self._monotonic_ns()),
            )
            if (
                book.source_time_ms > received_at_ms + 250
                or received_at_ms - book.source_time_ms > maximum_book_age
            ):
                return self._no_proposal("polymarket_book_stale")
            books[outcome] = book
        if (
            market.end_ms - received_at_ms
            < self.promotion.promotion.minimum_remaining_seconds * 1_000
        ):
            return self._no_proposal("insufficient_event_time_remaining")
        expires_at_ms = min(
            market.end_ms,
            evidence.decision_time_ms
            + self.promotion.promotion.maximum_prediction_age_ms,
            self.promotion.promotion.expires_at_ms,
        )
        if received_at_ms >= expires_at_ms:
            return self._no_proposal("proposal_expired_during_book_fetch")
        fee_model = PolymarketFeeModel(
            enabled=market.fee_schedule.enabled,
            rate=market.fee_schedule.rate,
            exponent=market.fee_schedule.exponent,
            taker_only=market.fee_schedule.taker_only,
        )
        minimum_edge = self.promotion.promotion.minimum_expected_edge_quote_per_share
        accounting = portfolio.accounting
        lock_candidates: list[tuple[Decimal, PolymarketAutonomousLockProposal]] = []
        for owned_outcome, complement_outcome in (("Up", "Down"), ("Down", "Up")):
            owned_quantity = accounting.quantity(owned_outcome)
            complement_quantity = accounting.quantity(complement_outcome)
            unpaired = max(Decimal("0"), owned_quantity - complement_quantity)
            quantity = min(self.requested_quantity, unpaired)
            quantity = quantity.quantize(_QUANTITY_STEP)
            if quantity < market.minimum_order_size or owned_quantity <= 0:
                continue
            book = books[complement_outcome]
            asks = tuple(getattr(book, "asks"))
            worst_price = self._walk_price(asks, quantity)
            if worst_price is None:
                continue
            worst_fee = fee_model(min(worst_price, _HALF), quantity, "taker")
            owned_average_cost = (
                accounting.cost_basis_quote(owned_outcome) / owned_quantity
            )
            edge = (
                Decimal("1") - owned_average_cost - worst_price - worst_fee / quantity
            )
            if edge < minimum_edge:
                continue
            input_sha = _canonical_sha256(
                {
                    "schema_version": (
                        POLYMARKET_ROUND21_PROMOTED_DECISION_SCHEMA_VERSION
                    ),
                    "action": f"lock_{owned_outcome.lower()}_with_"
                    f"{complement_outcome.lower()}",
                    "coordinated_decision_sha256": result.decision_sha256,
                    "probability_evidence_sha256": evidence.evidence_sha256,
                    "portfolio_sha256": portfolio.portfolio_sha256,
                    "up_book_sha256": getattr(
                        books["Up"],
                        "source_payload_sha256",
                    ),
                    "down_book_sha256": getattr(
                        books["Down"],
                        "source_payload_sha256",
                    ),
                    "quantity": format(quantity, "f"),
                    "worst_price": format(worst_price, "f"),
                    "guaranteed_edge_per_share": format(edge, "f"),
                }
            )
            lock_candidates.append(
                (
                    edge,
                    PolymarketAutonomousLockProposal(
                        proposal_id=(
                            "round21-lock-"
                            + market.condition_id[2:18]
                            + f"-{evidence.decision_time_ms}"
                        ),
                        input_sha256=input_sha,
                        model_artifact_sha256=(
                            self.promotion.promotion.model_artifact.sha256
                        ),
                        promotion_sha256=(self.promotion.promotion.promotion_sha256),
                        market_id=market.condition_id,
                        token_id=(
                            market.up_token_id
                            if complement_outcome == "Up"
                            else market.down_token_id
                        ),
                        symbol="BTC",
                        market_variant="fiveminute",
                        outcome=complement_outcome,
                        owned_outcome=owned_outcome,
                        requested_quantity=quantity,
                        event_start_time_ms=market.event_start_ms,
                        event_end_time_ms=market.end_ms,
                        decision_time_ms=evidence.decision_time_ms,
                        expires_at_ms=expires_at_ms,
                    ),
                )
            )
        if lock_candidates:
            _, proposal = max(
                lock_candidates,
                key=lambda item: (item[0], item[1].proposal_sha256),
            )
            return PolymarketAutonomousDecision(locks=(proposal,))

        reduction_candidates: list[
            tuple[Decimal, PolymarketAutonomousReduceProposal]
        ] = []
        for lot in portfolio.lots:
            quantity = min(self.requested_quantity, lot.quantity).quantize(
                _QUANTITY_STEP
            )
            if quantity < market.minimum_order_size:
                continue
            book = books[lot.outcome]
            bids = tuple(getattr(book, "bids"))
            worst_price = self._walk_price(bids, quantity)
            if worst_price is None:
                continue
            worst_fee = fee_model(min(worst_price, _HALF), quantity, "taker")
            maximum_probability = evidence.upper(lot.outcome)
            edge = worst_price - worst_fee / quantity - maximum_probability
            if edge < minimum_edge:
                continue
            input_sha = _canonical_sha256(
                {
                    "schema_version": (
                        POLYMARKET_ROUND21_PROMOTED_DECISION_SCHEMA_VERSION
                    ),
                    "action": f"reduce_{lot.outcome.lower()}",
                    "coordinated_decision_sha256": result.decision_sha256,
                    "probability_evidence_sha256": evidence.evidence_sha256,
                    "portfolio_sha256": portfolio.portfolio_sha256,
                    "parent_intent_id": lot.parent_intent_id,
                    "up_book_sha256": getattr(
                        books["Up"],
                        "source_payload_sha256",
                    ),
                    "down_book_sha256": getattr(
                        books["Down"],
                        "source_payload_sha256",
                    ),
                    "quantity": format(quantity, "f"),
                    "worst_price": format(worst_price, "f"),
                    "edge_per_share": format(edge, "f"),
                }
            )
            reduction_candidates.append(
                (
                    edge,
                    PolymarketAutonomousReduceProposal(
                        proposal_id=(
                            "round21-reduce-"
                            + market.condition_id[2:18]
                            + f"-{evidence.decision_time_ms}"
                        ),
                        input_sha256=input_sha,
                        model_artifact_sha256=(
                            self.promotion.promotion.model_artifact.sha256
                        ),
                        promotion_sha256=(self.promotion.promotion.promotion_sha256),
                        market_id=market.condition_id,
                        token_id=lot.token_id,
                        symbol="BTC",
                        market_variant="fiveminute",
                        outcome=lot.outcome,
                        parent_intent_id=lot.parent_intent_id,
                        maximum_outcome_probability=maximum_probability,
                        requested_quantity=quantity,
                        event_start_time_ms=market.event_start_ms,
                        event_end_time_ms=market.end_ms,
                        decision_time_ms=evidence.decision_time_ms,
                        expires_at_ms=expires_at_ms,
                    ),
                )
            )
        if reduction_candidates:
            _, proposal = max(
                reduction_candidates,
                key=lambda item: (item[0], item[1].proposal_sha256),
            )
            return PolymarketAutonomousDecision(reductions=(proposal,))
        return self._no_proposal("no_positive_inventory_transition")

    def _build_proposal(
        self,
        market: PolymarketFiveMinuteMarket,
        *,
        coordinated: Round21CoordinatedPrediction,
        portfolio: PolymarketAutonomousPortfolio,
        observed_at_ms: int,
    ) -> PolymarketAutonomousDecision:
        if portfolio.lots:
            return self._build_inventory_transition(
                market,
                coordinated=coordinated,
                portfolio=portfolio,
                observed_at_ms=observed_at_ms,
            )
        if (
            portfolio.accounting.confirmed_fill_count
            or portfolio.accounting.gross_buy_cost_quote > 0
        ):
            return self._no_proposal(
                "realized_event_loss_reentry_blocked"
                if portfolio.accounting.maximum_loss_quote > 0
                else "historical_event_reentry_blocked"
            )
        risk_state = portfolio.risk_state
        if risk_state.risk_profile != self.risk_profile.name:
            raise PolymarketLiveBlocked(
                "Round 21 decision risk profile differs from live risk state"
            )
        if not risk_state.entry_allowed:
            reason = (
                risk_state.entry_block_reasons[0]
                if risk_state.entry_block_reasons
                else "live_risk_entry_blocked"
            )
            return self._no_proposal(reason)
        result = coordinated.validated()
        if result.status != "observed" or result.prediction is None:
            reason = result.reasons[0] if result.reasons else "round21_model_abstained"
            return self._no_proposal(reason)
        prediction = result.prediction.validated()
        envelope = prediction.envelope
        if envelope is None:
            return self._no_proposal("round21_probability_envelope_unavailable")
        evidence = envelope.validated()
        model_sha = self.source_model_artifact_sha256
        if (
            evidence.source_model_artifact_sha256 != model_sha
            or prediction.source_model_artifact_sha256 != model_sha
            or prediction.sealed_result_sha256
            != self.data_service.scorer.sealed_result.result_sha256
        ):
            raise PolymarketLiveBlocked(
                "Round 21 probability evidence differs from promoted scorer"
            )
        robust_up = evidence.lower("Up")
        robust_down = evidence.lower("Down")
        if robust_up >= robust_down:
            outcome = "Up"
            token_id = market.up_token_id
            selected_probability = robust_up
        else:
            outcome = "Down"
            token_id = market.down_token_id
            selected_probability = robust_down
        requested_at_ms = int(self._clock_ms())
        if requested_at_ms < observed_at_ms:
            raise PolymarketLiveBlocked("Round 21 decision clock regressed")
        payload = self.public_client.order_book(token_id)
        received_at_ms = int(self._clock_ms())
        if received_at_ms < requested_at_ms:
            raise PolymarketLiveBlocked("Round 21 book receipt clock regressed")
        book = validate_clob_order_book(
            market,
            token_id,
            payload,
            received_wall_ms=received_at_ms,
            received_monotonic_ns=int(self._monotonic_ns()),
        )
        maximum_book_age = min(
            _MAXIMUM_CREATION_BOOK_AGE_MS,
            self.promotion.promotion.maximum_prediction_age_ms,
        )
        if (
            book.source_time_ms > received_at_ms + 250
            or received_at_ms - book.source_time_ms > maximum_book_age
        ):
            return self._no_proposal("polymarket_book_stale")
        if self.requested_quantity < market.minimum_order_size:
            return self._no_proposal("requested_quantity_below_market_minimum")
        remaining = self.requested_quantity
        worst_price = Decimal("0")
        executable_depth = Decimal("0")
        for level in book.asks:
            executable_depth += level.quantity
            consumed = min(remaining, level.quantity)
            if consumed > 0:
                worst_price = level.price
                remaining -= consumed
            if remaining == 0:
                break
        if remaining > 0 or worst_price <= 0:
            return self._no_proposal("insufficient_displayed_ask_depth")
        if (
            executable_depth <= 0
            or self.requested_quantity
            > executable_depth * self.risk_profile.maximum_displayed_depth_participation
        ):
            return self._no_proposal("displayed_depth_participation_exceeded")
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
        projected_inventory_downside = worst_price * self.requested_quantity + worst_fee
        if projected_inventory_downside > (
            risk_state.maximum_current_condition_downside_quote
        ):
            return self._no_proposal("live_risk_headroom_insufficient")
        edge = selected_probability - worst_price - worst_fee / self.requested_quantity
        if edge < self.promotion.promotion.minimum_expected_edge_quote_per_share:
            return self._no_proposal("insufficient_conservative_after_cost_edge")
        if (
            market.end_ms - received_at_ms
            < self.promotion.promotion.minimum_remaining_seconds * 1_000
        ):
            return self._no_proposal("insufficient_event_time_remaining")
        expires_at_ms = min(
            market.end_ms,
            evidence.decision_time_ms
            + self.promotion.promotion.maximum_prediction_age_ms,
            self.promotion.promotion.expires_at_ms,
        )
        if received_at_ms >= expires_at_ms:
            return self._no_proposal("proposal_expired_during_book_fetch")
        input_sha = _canonical_sha256(
            {
                "schema_version": POLYMARKET_ROUND21_PROMOTED_DECISION_SCHEMA_VERSION,
                "coordinated_decision_sha256": result.decision_sha256,
                "prospective_prediction_sha256": prediction.prediction_sha256,
                "probability_evidence_sha256": evidence.evidence_sha256,
                "book_payload_sha256": book.source_payload_sha256,
                "market_id": market.condition_id,
                "token_id": token_id,
                "outcome": outcome,
                "conservative_outcome_probability": format(
                    selected_probability,
                    "f",
                ),
                "requested_quantity": format(self.requested_quantity, "f"),
                "portfolio_sha256": portfolio.portfolio_sha256,
                "risk_state_sha256": risk_state.risk_state_sha256,
                "maximum_projected_inventory_downside_quote": format(
                    risk_state.maximum_current_condition_downside_quote,
                    "f",
                ),
                "decision_time_ms": evidence.decision_time_ms,
                "book_requested_at_ms": requested_at_ms,
                "book_received_at_ms": received_at_ms,
                "risk_profile": self.risk_profile.name,
            }
        )
        proposal = PolymarketAutonomousOpenProposal(
            proposal_id=(
                "round21-" + market.condition_id[2:18] + f"-{evidence.decision_time_ms}"
            ),
            input_sha256=input_sha,
            model_artifact_sha256=(self.promotion.promotion.model_artifact.sha256),
            promotion_sha256=self.promotion.promotion.promotion_sha256,
            market_id=market.condition_id,
            token_id=token_id,
            symbol="BTC",
            market_variant="fiveminute",
            outcome=outcome,
            selected_outcome_probability=selected_probability,
            requested_quantity=self.requested_quantity,
            risk_state_sha256=risk_state.risk_state_sha256,
            maximum_projected_inventory_downside_quote=(
                risk_state.maximum_current_condition_downside_quote
            ),
            event_start_time_ms=market.event_start_ms,
            event_end_time_ms=market.end_ms,
            decision_time_ms=evidence.decision_time_ms,
            expires_at_ms=expires_at_ms,
        )
        return PolymarketAutonomousDecision(proposals=(proposal,))

    def decide(
        self,
        *,
        markets: tuple[PolymarketFiveMinuteMarket, ...],
        observed_at_ms: int,
        portfolio: PolymarketAutonomousPortfolio,
    ) -> PolymarketAutonomousDecision:
        now = int(observed_at_ms)
        self.promotion.assert_live_authority(observed_at_ms=now)
        if len(markets) != 1:
            raise PolymarketLiveBlocked(
                "Round 21 decision requires one active Polymarket market"
            )
        market = markets[0]
        if (
            market.asset != "BTC"
            or market.horizon_minutes != 5
            or not market.event_start_ms <= now < market.end_ms
        ):
            raise PolymarketLiveBlocked("Round 21 decision market identity differs")
        if (
            not isinstance(portfolio, PolymarketAutonomousPortfolio)
            or portfolio.condition_id != market.condition_id
        ):
            raise PolymarketLiveBlocked("Round 21 decision portfolio identity differs")
        coordinated = self.data_service.evaluate(
            market,
            observed_at_ms=now,
        )
        if coordinated is None:
            return self._no_proposal("round21_public_state_unavailable")
        selected = coordinated.validated()
        if (
            selected.condition_id != market.condition_id
            or selected.event_start_ms != market.event_start_ms
            or selected.observed_at_ms != now
        ):
            raise PolymarketLiveBlocked(
                "Round 21 coordinated decision differs from current market"
            )
        decision_time = selected.decision_time_ms
        key = (market.condition_id, decision_time)
        if not self._claim_timestamp(*key):
            return self._no_proposal("round21_timestamp_already_consumed")
        try:
            return self._build_proposal(
                market,
                coordinated=selected,
                portfolio=portfolio,
                observed_at_ms=now,
            )
        except BaseException:
            with self._lock:
                self._consumed.discard(key)
            raise


credentials_used = False
account_connected = False
binance_connected = False
trading_authority = False


__all__ = [
    "POLYMARKET_ROUND21_PROMOTED_DECISION_SCHEMA_VERSION",
    "PolymarketRound21PromotedDecisionProvider",
]
