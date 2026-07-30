from __future__ import annotations

from decimal import Decimal
import hashlib
from pathlib import Path

import pytest

from simple_ai_trading.paper_execution import (
    BOT_OWNER,
    BookLevel,
    PaperBookSnapshot,
)
from simple_ai_trading.polymarket import (
    PolymarketFeeSchedule,
    PolymarketFiveMinuteMarket,
)
from simple_ai_trading.polymarket_round14_contract import load_round14_contract
from simple_ai_trading.polymarket_round17_execution import (
    Round17OwnedLot,
    Round17ProbabilityEnvelope,
    evaluate_round17_complement_lock,
    select_round17_entry,
    simulate_round17_owned_close,
)


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-014-btc-5m-prospective-contract-v1.json"
)
START_MS = 1_800_000_000_000
DECISION_MS = START_MS + 120_000


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _market() -> PolymarketFiveMinuteMarket:
    return PolymarketFiveMinuteMarket(
        asset="BTC",
        market_id="0x" + "1" * 64,
        condition_id="0x" + "2" * 64,
        slug=f"btc-updown-5m-{START_MS // 1000}",
        question="Bitcoin Up or Down",
        event_start_ms=START_MS,
        end_ms=START_MS + 300_000,
        up_token_id="1" * 40,
        down_token_id="2" * 40,
        tick_size=Decimal("0.01"),
        minimum_order_size=Decimal("1"),
        fee_schedule=PolymarketFeeSchedule(
            enabled=False,
            rate=Decimal("0"),
            exponent=2,
            taker_only=True,
            rebate_rate=Decimal("0"),
        ),
        liquidity_quote=Decimal("100000"),
        volume_quote=Decimal("100000"),
        resolution_source="chainlink",
        gamma_payload_sha256=_digest("market"),
        gamma_payload_json="{}",
    )


def _book(
    outcome: str,
    *,
    bid: str,
    ask: str,
    quantity: str = "100",
    connected: bool = True,
    gap_free: bool = True,
) -> PaperBookSnapshot:
    market = _market()
    token_id = market.up_token_id if outcome == "Up" else market.down_token_id
    received = DECISION_MS + 500
    return PaperBookSnapshot(
        venue="polymarket",
        market_id=market.condition_id,
        asset_id=token_id,
        bids=(BookLevel(Decimal(bid), Decimal(quantity)),),
        asks=(BookLevel(Decimal(ask), Decimal(quantity)),),
        source_time_ms=received - 10,
        received_wall_ms=received,
        received_monotonic_ns=received * 1_000_000,
        source_payload_sha256=_digest(f"{outcome}-{bid}-{ask}-{quantity}"),
        connected=connected,
        gap_free=gap_free,
    ).validated()


def _envelope() -> Round17ProbabilityEnvelope:
    return Round17ProbabilityEnvelope(
        probability_up=Decimal("0.70"),
        lower_up=Decimal("0.65"),
        upper_up=Decimal("0.75"),
        evidence_sha256=_digest("probability"),
    ).validated()


def test_round17_selects_only_positive_delayed_after_cost_polymarket_entry() -> None:
    program = load_round14_contract(CONTRACT_PATH)

    decision = select_round17_entry(
        _market(),
        {
            "Up": _book("Up", bid="0.44", ask="0.45"),
            "Down": _book("Down", bid="0.53", ask="0.54"),
        },
        _envelope(),
        program,
        decision_time_ms=DECISION_MS,
        risk_profile="conservative",
        scenario_name="primary",
        risk_capital_quote=Decimal("1000"),
        minimum_expected_edge_quote_per_share=Decimal("0.02"),
        reconciliation_ok=True,
        existing_owned_exposure=False,
    )

    assert decision.action == "buy_up_fok"
    assert decision.candidate is not None
    assert decision.candidate.total_cost_quote <= Decimal("1")
    assert decision.candidate.probability_lower_bound == Decimal("0.65")
    assert decision.candidate.lower_bound_edge_quote_per_share >= Decimal("0.02")
    assert decision.identity_payload()["binance_credentials_used"] is False
    assert decision.identity_payload()["binance_execution_connected"] is False
    assert decision.identity_payload()["trading_authority"] is False


def test_round17_abstains_without_edge_or_reconciliation() -> None:
    program = load_round14_contract(CONTRACT_PATH)
    books = {
        "Up": _book("Up", bid="0.55", ask="0.56"),
        "Down": _book("Down", bid="0.55", ask="0.56"),
    }

    no_edge = select_round17_entry(
        _market(),
        books,
        Round17ProbabilityEnvelope(
            probability_up=Decimal("0.50"),
            lower_up=Decimal("0.48"),
            upper_up=Decimal("0.52"),
            evidence_sha256=_digest("flat"),
        ),
        program,
        decision_time_ms=DECISION_MS,
        risk_profile="conservative",
        scenario_name="primary",
        risk_capital_quote=Decimal("1000"),
        minimum_expected_edge_quote_per_share=Decimal("0.02"),
        reconciliation_ok=True,
        existing_owned_exposure=False,
    )
    blocked = select_round17_entry(
        _market(),
        books,
        _envelope(),
        program,
        decision_time_ms=DECISION_MS,
        risk_profile="conservative",
        scenario_name="primary",
        risk_capital_quote=Decimal("1000"),
        minimum_expected_edge_quote_per_share=Decimal("0.02"),
        reconciliation_ok=False,
        existing_owned_exposure=False,
    )

    assert no_edge.action == "abstain"
    assert no_edge.reason == "no_positive_after_cost_lower_bound_edge"
    assert blocked.action == "abstain"
    assert blocked.reason == "reconciliation_not_proven"


def test_round17_stress_depth_and_gap_never_receive_fill_credit() -> None:
    program = load_round14_contract(CONTRACT_PATH)
    disconnected = select_round17_entry(
        _market(),
        {
            "Up": _book(
                "Up",
                bid="0.44",
                ask="0.45",
                connected=False,
                gap_free=False,
            ),
            "Down": _book("Down", bid="0.53", ask="0.54"),
        },
        _envelope(),
        program,
        decision_time_ms=DECISION_MS,
        risk_profile="conservative",
        scenario_name="combined",
        risk_capital_quote=Decimal("1000"),
        minimum_expected_edge_quote_per_share=Decimal("0.02"),
        reconciliation_ok=True,
        existing_owned_exposure=False,
    )

    assert disconnected.action == "abstain"
    assert disconnected.reason == "polymarket_book_unhealthy"


def test_round17_close_requires_exact_bot_owned_parent_lot() -> None:
    program = load_round14_contract(CONTRACT_PATH)
    lot = Round17OwnedLot(
        owner=BOT_OWNER,
        parent_intent_id="round17-parent-00000001",
        market_id=_market().condition_id,
        token_id=_market().up_token_id,
        outcome="Up",
        quantity=Decimal("1.000000"),
        entry_cost_quote=Decimal("0.45"),
    )

    result = simulate_round17_owned_close(
        _market(),
        lot,
        _book("Up", bid="0.60", ask="0.61"),
        program,
        decision_time_ms=DECISION_MS,
        scenario_name="primary",
    )

    assert result.state == "FILLED"
    assert result.filled_quantity == lot.quantity
    with pytest.raises(ValueError, match="exact bot-owned lot"):
        simulate_round17_owned_close(
            _market(),
            Round17OwnedLot(
                owner="foreign-user",
                parent_intent_id=lot.parent_intent_id,
                market_id=lot.market_id,
                token_id=lot.token_id,
                outcome=lot.outcome,
                quantity=lot.quantity,
                entry_cost_quote=lot.entry_cost_quote,
            ),
            _book("Up", bid="0.60", ask="0.61"),
            program,
            decision_time_ms=DECISION_MS,
            scenario_name="primary",
        )


def test_round17_complement_requires_nonnegative_worst_case_lock() -> None:
    lot = Round17OwnedLot(
        owner=BOT_OWNER,
        parent_intent_id="round17-parent-00000001",
        market_id=_market().condition_id,
        token_id=_market().up_token_id,
        outcome="Up",
        quantity=Decimal("2.000000"),
        entry_cost_quote=Decimal("0.80"),
    )

    locked = evaluate_round17_complement_lock(
        lot,
        complement_quantity=Decimal("2.000000"),
        complement_cost_quote=Decimal("1.10"),
    )
    unsafe = evaluate_round17_complement_lock(
        lot,
        complement_quantity=Decimal("2.000000"),
        complement_cost_quote=Decimal("1.30"),
    )

    assert locked.allowed is True
    assert locked.guaranteed_net_quote == Decimal("0.10")
    assert unsafe.allowed is False
    assert unsafe.guaranteed_net_quote == Decimal("-0.10")
