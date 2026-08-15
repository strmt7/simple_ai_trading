from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import simple_ai_trading.polymarket_round21_decision as decision_module

from simple_ai_trading.polymarket import PolymarketPublicClient
from simple_ai_trading.polymarket_autonomous_runtime import (
    PolymarketAutonomousPortfolio,
    PolymarketAutonomousPortfolioLot,
)
from simple_ai_trading.polymarket_live import (
    PolymarketConditionAccounting,
    PolymarketLedgerRevision,
    PolymarketLiveBlocked,
)
from simple_ai_trading.polymarket_live_risk import PolymarketLiveRiskState
from simple_ai_trading.polymarket_live_promotion import (
    VerifiedPolymarketLivePromotion,
)
from simple_ai_trading.polymarket_round21_model import (
    VerifiedRound21DevelopmentArtifact,
)
from simple_ai_trading.polymarket_round21_decision import (
    PolymarketRound21PromotedDecisionProvider,
)
from simple_ai_trading.polymarket_round21_live_features import (
    Round21CoordinatedPrediction,
)
from simple_ai_trading.polymarket_round21_policy import (
    Round21ProbabilityEnvelope,
)
from simple_ai_trading.polymarket_round21_prospective import (
    Round21ProspectivePrediction,
)
from simple_ai_trading.polymarket_round21_session import (
    Round21RollingPublicDataService,
)

from polymarket_round21_support import round21_replay_condition, sha


MARKET = round21_replay_condition().market
NOW_MS = MARKET.event_start_ms + 120_000
MODEL_SHA = sha("round21-promoted-model")
SEALED_SHA = sha("round21-promoted-sealed")
PROMOTION_SHA = sha("round21-live-promotion")
MODEL_FILE_SHA = sha("round21-promoted-model-file")


@pytest.fixture(autouse=True)
def _model_artifact_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    evidence = Mock(spec=VerifiedRound21DevelopmentArtifact)
    evidence.path = Path("model.json").resolve()
    evidence.file_sha256 = MODEL_FILE_SHA
    evidence.artifact_sha256 = MODEL_SHA
    monkeypatch.setattr(
        decision_module,
        "load_verified_round21_development_artifact",
        lambda _path, *, expected_file_sha256: (
            evidence
            if expected_file_sha256 == MODEL_FILE_SHA
            else pytest.fail("unexpected model evidence hash")
        ),
    )


class _PublicClient(PolymarketPublicClient):
    def __init__(
        self,
        *,
        ask_price: str = "0.40",
        ask_size: str = "100",
        source_time_ms: int = NOW_MS,
        fail_once: bool = False,
    ) -> None:
        self.ask_price = ask_price
        self.ask_size = ask_size
        self.source_time_ms = source_time_ms
        self.fail_once = fail_once
        self.calls = 0

    def order_book(self, token_id: str) -> dict[str, object]:
        self.calls += 1
        if self.fail_once and self.calls == 1:
            raise TimeoutError("transient public book timeout")
        return {
            "market": MARKET.condition_id,
            "asset_id": token_id,
            "timestamp": str(self.source_time_ms),
            "bids": [{"price": "0.39", "size": "100"}],
            "asks": [{"price": self.ask_price, "size": self.ask_size}],
        }


def _promotion(*, variant: str = "fiveminute"):
    verified = Mock(spec=VerifiedPolymarketLivePromotion)
    verified.promotion = SimpleNamespace(
        market_variant=variant,
        model_artifact=SimpleNamespace(sha256=MODEL_FILE_SHA),
        evaluation_report=SimpleNamespace(sha256=sha("evaluation")),
        promotion_sha256=PROMOTION_SHA,
        maximum_prediction_age_ms=1_000,
        minimum_expected_edge_quote_per_share=Decimal("0.02"),
        minimum_remaining_seconds=30,
        expires_at_ms=NOW_MS + 60_000,
    )
    verified.model_artifact_path = Path("model.json").resolve()
    return verified


def _coordinated(
    *,
    probability: str = "0.70",
    lower: str = "0.68",
    upper: str = "0.72",
    status: str = "observed",
):
    envelope = Round21ProbabilityEnvelope.create(
        condition_id=MARKET.condition_id,
        decision_time_ms=NOW_MS,
        probability_up=Decimal(probability),
        lower_up=Decimal(lower),
        upper_up=Decimal(upper),
        model_layer="core",
        source_model_artifact_sha256=MODEL_SHA,
        source_probability_batch_sha256=sha("probability-batch"),
        feature_row_sha256=sha("feature-row"),
    )
    prediction = Mock(spec=Round21ProspectivePrediction)
    prediction.validated.return_value = prediction
    prediction.envelope = envelope
    prediction.source_model_artifact_sha256 = MODEL_SHA
    prediction.sealed_result_sha256 = SEALED_SHA
    prediction.prediction_sha256 = sha("prediction")
    coordinated = Mock(spec=Round21CoordinatedPrediction)
    coordinated.validated.return_value = coordinated
    coordinated.status = status
    coordinated.reasons = () if status == "observed" else ("model_abstained",)
    coordinated.prediction = prediction if status == "observed" else None
    coordinated.condition_id = MARKET.condition_id
    coordinated.event_start_ms = MARKET.event_start_ms
    coordinated.observed_at_ms = NOW_MS
    coordinated.decision_time_ms = NOW_MS
    coordinated.decision_sha256 = sha("coordinated")
    return coordinated


def _service(result) -> Mock:
    service = Mock(spec=Round21RollingPublicDataService)
    service.scorer = SimpleNamespace(
        source_model_artifact_sha256=MODEL_SHA,
        sealed_result=SimpleNamespace(result_sha256=SEALED_SHA),
    )
    service.evaluate.return_value = result
    return service


def _provider(*, client=None, service=None, promotion=None, risk_level="conservative"):
    return PolymarketRound21PromotedDecisionProvider(
        public_client=client or _PublicClient(),
        data_service=service or _service(_coordinated()),
        promotion=promotion or _promotion(),
        requested_quantity=Decimal("5"),
        risk_level=risk_level,
        clock_ms=lambda: NOW_MS,
        monotonic_ns=lambda: 1,
    )


def _portfolio(
    *,
    outcome: str | None = None,
    quantity: str = "0",
    cost_basis: str = "0",
    gross_buy_cost: str | None = None,
    gross_sell_proceeds: str = "0",
) -> PolymarketAutonomousPortfolio:
    selected_quantity = Decimal(quantity)
    selected_cost = Decimal(cost_basis)
    lots = ()
    up_quantity = Decimal("0")
    down_quantity = Decimal("0")
    up_cost = Decimal("0")
    down_cost = Decimal("0")
    if outcome is not None:
        token = MARKET.up_token_id if outcome == "Up" else MARKET.down_token_id
        lots = (
            PolymarketAutonomousPortfolioLot(
                parent_intent_id="poly-open-parent-0001",
                market_id=MARKET.condition_id,
                token_id=token,
                outcome=outcome,
                quantity=selected_quantity,
            ),
        )
        if outcome == "Up":
            up_quantity = selected_quantity
            up_cost = selected_cost
        else:
            down_quantity = selected_quantity
            down_cost = selected_cost
    return PolymarketAutonomousPortfolio(
        condition_id=MARKET.condition_id,
        lots=lots,
        accounting=PolymarketConditionAccounting(
            condition_id=MARKET.condition_id,
            gross_buy_cost_quote=Decimal(
                gross_buy_cost if gross_buy_cost is not None else cost_basis
            ),
            gross_sell_proceeds_quote=Decimal(gross_sell_proceeds),
            confirmed_redemption_payout_quote=Decimal("0"),
            up_quantity=up_quantity,
            down_quantity=down_quantity,
            up_cost_basis_quote=up_cost,
            down_cost_basis_quote=down_cost,
            confirmed_fill_count=0 if outcome is None else 1,
        ),
        risk_state=PolymarketLiveRiskState(
            condition_id=MARKET.condition_id,
            risk_profile="conservative",
            risk_capital_quote=Decimal("10000"),
            observed_at_ms=NOW_MS,
            utc_day_index=NOW_MS // 86_400_000,
            ledger_revision=PolymarketLedgerRevision(0, "0" * 64),
            realized_event_count=0,
            realized_condition_count=0,
            daily_realized_pnl_quote=Decimal("0"),
            lifetime_realized_pnl_quote=Decimal("0"),
            settled_equity_quote=Decimal("10000"),
            settled_peak_equity_quote=Decimal("10000"),
            drawdown_capital_fraction=Decimal("0"),
            consecutive_losing_conditions=0,
            cooldown_until_ms=0,
            cooldown_active=False,
            current_condition_inventory_downside_quote=max(
                Decimal("0"),
                up_cost + down_cost - min(up_quantity, down_quantity),
            ),
            other_condition_inventory_downside_quote=Decimal("0"),
            total_inventory_downside_quote=max(
                Decimal("0"),
                up_cost + down_cost - min(up_quantity, down_quantity),
            ),
            maximum_current_condition_downside_quote=Decimal("10"),
            entry_allowed=True,
            entry_block_reasons=(),
        ),
    )


def test_round21_provider_builds_one_conservative_hash_bound_proposal() -> None:
    client = _PublicClient()
    service = _service(_coordinated())
    promotion = _promotion()
    provider = _provider(client=client, service=service, promotion=promotion)

    first = provider.decide(
        markets=(MARKET,), observed_at_ms=NOW_MS, portfolio=_portfolio()
    )
    duplicate = provider.decide(
        markets=(MARKET,), observed_at_ms=NOW_MS, portfolio=_portfolio()
    )

    assert len(first.proposals) == 1
    proposal = first.proposals[0]
    assert proposal.outcome == "Up"
    assert proposal.token_id == MARKET.up_token_id
    assert proposal.selected_outcome_probability == Decimal("0.68")
    assert proposal.market_variant == "fiveminute"
    assert proposal.model_artifact_sha256 == MODEL_FILE_SHA
    assert proposal.promotion_sha256 == PROMOTION_SHA
    assert duplicate.proposals == ()
    assert duplicate.reasons == ("round21_timestamp_already_consumed",)
    assert service.evaluate.call_count == 2
    assert client.calls == 1
    promotion.assert_live_authority.assert_called()
    assert not any(
        (
            provider.credentials_used,
            provider.account_connected,
            provider.binance_connected,
            provider.trading_authority,
        )
    )


def test_round21_provider_selects_down_from_the_conservative_hull() -> None:
    provider = _provider(
        service=_service(_coordinated(probability="0.28", lower="0.25", upper="0.30"))
    )

    decision = provider.decide(
        markets=(MARKET,), observed_at_ms=NOW_MS, portfolio=_portfolio()
    )

    assert decision.proposals[0].outcome == "Down"
    assert decision.proposals[0].token_id == MARKET.down_token_id
    assert decision.proposals[0].selected_outcome_probability == Decimal("0.70")


def test_round21_provider_prioritizes_positive_guaranteed_complement_lock() -> None:
    provider = _provider()

    decision = provider.decide(
        markets=(MARKET,),
        observed_at_ms=NOW_MS,
        portfolio=_portfolio(
            outcome="Up",
            quantity="5",
            cost_basis="2.00",
        ),
    )

    assert decision.proposals == ()
    assert decision.reductions == ()
    assert len(decision.locks) == 1
    lock = decision.locks[0]
    assert lock.outcome == "Down"
    assert lock.owned_outcome == "Up"
    assert lock.requested_quantity == Decimal("5")
    assert lock.token_id == MARKET.down_token_id


def test_round21_provider_can_reduce_exact_parent_when_lock_is_unprofitable() -> None:
    provider = _provider(
        service=_service(_coordinated(probability="0.10", lower="0.05", upper="0.20"))
    )

    decision = provider.decide(
        markets=(MARKET,),
        observed_at_ms=NOW_MS,
        portfolio=_portfolio(
            outcome="Up",
            quantity="5",
            cost_basis="3.50",
        ),
    )

    assert decision.proposals == ()
    assert decision.locks == ()
    assert len(decision.reductions) == 1
    reduction = decision.reductions[0]
    assert reduction.outcome == "Up"
    assert reduction.parent_intent_id == "poly-open-parent-0001"
    assert reduction.maximum_outcome_probability == Decimal("0.20")


def test_round21_provider_blocks_reentry_after_realized_same_event_loss() -> None:
    decision = _provider().decide(
        markets=(MARKET,),
        observed_at_ms=NOW_MS,
        portfolio=_portfolio(
            gross_buy_cost="1.00",
            gross_sell_proceeds="0.80",
        ),
    )

    assert decision.proposals == ()
    assert decision.reasons == ("realized_event_loss_reentry_blocked",)


@pytest.mark.parametrize(
    ("client", "reason"),
    (
        (
            _PublicClient(ask_price="0.67"),
            "insufficient_conservative_after_cost_edge",
        ),
        (
            _PublicClient(ask_size="49.999999"),
            "displayed_depth_participation_exceeded",
        ),
        (
            _PublicClient(source_time_ms=NOW_MS - 501),
            "polymarket_book_stale",
        ),
    ),
)
def test_round21_provider_rejects_unsafe_book_or_edge(client, reason) -> None:
    decision = _provider(client=client).decide(
        markets=(MARKET,),
        observed_at_ms=NOW_MS,
        portfolio=_portfolio(),
    )

    assert decision.proposals == ()
    assert decision.reasons == (reason,)


def test_round21_provider_retries_after_transient_book_failure() -> None:
    client = _PublicClient(fail_once=True)
    provider = _provider(client=client)
    with pytest.raises(TimeoutError, match="transient public book timeout"):
        provider.decide(
            markets=(MARKET,), observed_at_ms=NOW_MS, portfolio=_portfolio()
        )

    recovered = provider.decide(
        markets=(MARKET,), observed_at_ms=NOW_MS, portfolio=_portfolio()
    )

    assert len(recovered.proposals) == 1
    assert client.calls == 2


def test_round21_provider_discards_consumed_timestamps_at_market_rollover() -> None:
    provider = _provider()

    assert provider._claim_timestamp(MARKET.condition_id, NOW_MS) is True
    assert provider._claim_timestamp(MARKET.condition_id, NOW_MS) is False
    assert provider._claim_timestamp("0x" + "d" * 64, NOW_MS + 300_000) is True
    assert provider._consumed == {("0x" + "d" * 64, NOW_MS + 300_000)}


def test_round21_provider_fails_closed_without_state_or_matching_promotion() -> None:
    unavailable = _provider(service=_service(None)).decide(
        markets=(MARKET,),
        observed_at_ms=NOW_MS,
        portfolio=_portfolio(),
    )
    abstained = _provider(service=_service(_coordinated(status="abstain"))).decide(
        markets=(MARKET,),
        observed_at_ms=NOW_MS,
        portfolio=_portfolio(),
    )

    assert unavailable.reasons == ("round21_public_state_unavailable",)
    assert abstained.reasons == ("model_abstained",)
    mismatched_service = _service(_coordinated())
    mismatched_service.scorer.source_model_artifact_sha256 = sha("other-model")
    with pytest.raises(PolymarketLiveBlocked, match="scorer differs"):
        _provider(service=mismatched_service)
    with pytest.raises(PolymarketLiveBlocked, match="five-minute promotion"):
        _provider(promotion=_promotion(variant="fifteenminute"))
