from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

import simple_ai_trading.polymarket_round21_execution as execution_module
from simple_ai_trading.paper_execution import BookLevel, PaperBookSnapshot
from simple_ai_trading.polymarket import (
    PolymarketFeeSchedule,
    PolymarketFiveMinuteMarket,
)
from simple_ai_trading.polymarket_round21_execution import (
    POLYMARKET_ROUND21_EXECUTION_POLICY_SHA256,
    POLYMARKET_ROUND21_EXECUTION_SCENARIOS,
    Round21AggressiveOrderPlan,
    Round21MarketExecutionEvidence,
    load_round21_execution_policy,
    observe_round21_aggressive_execution,
    round21_execution_scenario,
    validate_round21_execution_policy,
)


START_MS = 1_800_000_000_000
DECISION_MS = START_MS + 120_000
CONDITION_ID = "0x" + "2" * 64
UP_TOKEN = "1" * 40
DOWN_TOKEN = "2" * 40
POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-021-executable-action-policy-v1.json"
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def test_round21_price_lattice_count_always_floors_partial_ticks() -> None:
    assert (
        execution_module._maximum_executable_price_levels(
            side="SELL",
            limit_price=Decimal("0.48"),
            tick_size=Decimal("0.03"),
        )
        == 17
    )


def _market() -> PolymarketFiveMinuteMarket:
    return PolymarketFiveMinuteMarket(
        asset="BTC",
        market_id="12345",
        condition_id=CONDITION_ID,
        slug=f"btc-updown-5m-{START_MS // 1000}",
        question="Bitcoin Up or Down",
        event_start_ms=START_MS,
        end_ms=START_MS + 300_000,
        up_token_id=UP_TOKEN,
        down_token_id=DOWN_TOKEN,
        tick_size=Decimal("0.01"),
        minimum_order_size=Decimal("1"),
        fee_schedule=PolymarketFeeSchedule(
            enabled=True,
            rate=Decimal("0.07"),
            exponent=1,
            taker_only=True,
            rebate_rate=Decimal("0.20"),
        ),
        liquidity_quote=Decimal("100000"),
        volume_quote=Decimal("100000"),
        resolution_source="chainlink",
        gamma_payload_sha256=_digest("market"),
        gamma_payload_json="{}",
    )


def _evidence(
    *,
    taker_delay: bool = True,
    general_delay_seconds: int = 0,
) -> Round21MarketExecutionEvidence:
    return Round21MarketExecutionEvidence.create(
        condition_id=CONDITION_ID,
        observed_wall_ms=DECISION_MS - 1_000,
        observed_monotonic_ns=(DECISION_MS - 1_000) * 1_000_000,
        maker_base_fee=0,
        taker_base_fee=700,
        taker_order_delay_enabled=taker_delay,
        general_order_delay_seconds=general_delay_seconds,
        minimum_order_age_seconds=0,
        clob_info_sha256=_digest("clob"),
        up_fee_rate_sha256=_digest("up-fee"),
        down_fee_rate_sha256=_digest("down-fee"),
        snapshot_sha256=_digest("snapshot"),
    )


def _plan(
    *,
    scenario_name: str = "primary",
    side: str = "BUY",
    outcome: str = "Up",
    quantity: str = "10",
    limit_price: str = "0.55",
    evidence: Round21MarketExecutionEvidence | None = None,
) -> Round21AggressiveOrderPlan:
    selected_side = side.upper()
    return Round21AggressiveOrderPlan.create(
        market=_market(),
        market_evidence=_evidence() if evidence is None else evidence,
        scenario_name=scenario_name,
        decision_time_ms=DECISION_MS,
        outcome=outcome,
        side=selected_side,
        quantity=Decimal(quantity),
        limit_price=Decimal(limit_price),
        effective_tick_size=Decimal("0.01"),
        builder_taker_fee_bps=Decimal("10"),
        owned_cost_basis_quote=(
            Decimal("4") if selected_side == "SELL" else Decimal("0")
        ),
        parent_inventory_id=(
            "round21-owned-inventory" if selected_side == "SELL" else ""
        ),
        predictor_evidence_sha256=_digest("predictor"),
        reconciliation_sha256=_digest("reconciliation"),
    )


def _book(
    plan: Round21AggressiveOrderPlan,
    *,
    bids: tuple[tuple[str, str], ...] = (("0.49", "20"),),
    asks: tuple[tuple[str, str], ...] = (("0.50", "6"), ("0.51", "10")),
    offset_ms: int = 0,
    connected: bool = True,
    gap_free: bool = True,
) -> PaperBookSnapshot:
    received = plan.effective_execution_target_ms + offset_ms
    return PaperBookSnapshot(
        venue="polymarket",
        market_id=CONDITION_ID,
        asset_id=plan.token_id,
        bids=tuple(
            BookLevel(Decimal(price), Decimal(quantity)) for price, quantity in bids
        ),
        asks=tuple(
            BookLevel(Decimal(price), Decimal(quantity)) for price, quantity in asks
        ),
        source_time_ms=received - 25,
        received_wall_ms=received,
        received_monotonic_ns=received * 1_000_000,
        source_payload_sha256=_digest(
            f"book-{plan.plan_sha256}-{received}-{connected}-{gap_free}"
        ),
        connected=connected,
        gap_free=gap_free,
    ).validated()


def _rehash_policy(value: dict[str, object]) -> dict[str, object]:
    body = dict(value)
    body.pop("design_sha256", None)
    body["design_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return body


def test_round21_execution_policy_and_full_stress_matrix_are_frozen() -> None:
    policy = load_round21_execution_policy(POLICY_PATH)

    assert policy["design_sha256"] == POLYMARKET_ROUND21_EXECUTION_POLICY_SHA256
    assert len(POLYMARKET_ROUND21_EXECUTION_SCENARIOS) == 27
    assert len({item.name for item in POLYMARKET_ROUND21_EXECUTION_SCENARIOS}) == 27
    primary = round21_execution_scenario("primary")
    assert primary.submission_latency_ms == 500
    assert primary.displayed_depth_fraction == Decimal("1")
    assert primary.adverse_ticks == 0
    assert policy["independence"]["execution_venue"] == "polymarket_only"
    assert execution_module.credentials_used is False
    assert execution_module.account_connected is False
    assert execution_module.binance_execution_connected is False
    assert execution_module.live_trading_authority is False

    changed = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    changed["independence"]["binance_execution"] = True
    with pytest.raises(ValueError, match="executable action policy differs"):
        validate_round21_execution_policy(_rehash_policy(changed))


def test_round21_plan_adds_captured_venue_and_general_delay() -> None:
    plan = _plan(
        scenario_name="latency_250ms_depth_100pct_adverse_0ticks",
        evidence=_evidence(general_delay_seconds=1),
    )

    assert plan.effective_execution_target_ms == DECISION_MS + 1_500
    assert plan.maximum_loss_quote == Decimal("5.681094")
    assert plan.identity_payload()["binance_execution_connected"] is False
    assert plan.trading_authority is False


def test_round21_full_fill_reconciles_per_level_platform_and_builder_fees() -> None:
    plan = _plan()

    observation = observe_round21_aggressive_execution(plan, _book(plan))

    assert observation.state == "filled"
    assert observation.filled_quantity == Decimal("10")
    assert observation.unfilled_quantity == 0
    assert observation.average_fill_price == Decimal("0.504")
    assert observation.platform_fee_quote == Decimal("0.17498")
    assert observation.builder_fee_quote == Decimal("0.005040")
    assert observation.total_fee_quote == Decimal("0.180020")
    assert observation.execution_cash_flow_quote == Decimal("-5.220020")
    assert -observation.execution_cash_flow_quote <= plan.maximum_loss_quote
    assert observation.blocks_new_exposure is False
    assert observation.trading_authority is False


def test_round21_maximum_loss_covers_fee_rounding_at_every_price_level() -> None:
    plan = _plan(limit_price="0.50")
    asks = tuple((format(Decimal(index) / 100, "f"), "0.2") for index in range(1, 51))

    observation = observe_round21_aggressive_execution(
        plan,
        _book(plan, bids=(), asks=asks),
    )

    assert observation.filled_quantity == plan.quantity
    assert observation.platform_fee_quote > 0
    assert observation.builder_fee_quote > 0
    assert -observation.execution_cash_flow_quote <= plan.maximum_loss_quote


def test_round21_depth_and_adverse_price_stress_produce_real_partial_fak() -> None:
    plan = _plan(
        scenario_name="latency_500ms_depth_25pct_adverse_2ticks",
        limit_price="0.53",
    )
    book = _book(
        plan,
        bids=(("0.49", "20"),),
        asks=(("0.50", "20"),),
    )

    observation = observe_round21_aggressive_execution(plan, book)

    assert observation.state == "partial_fill"
    assert observation.filled_quantity == Decimal("5.00")
    assert observation.unfilled_quantity == Decimal("5.00")
    assert observation.average_fill_price == Decimal("0.52")
    assert observation.execution_cash_flow_quote < Decimal("-2.6")
    assert observation.reason == "fak_unfilled_remainder_cancelled"


def test_round21_zero_fill_fak_is_known_without_phantom_inventory() -> None:
    plan = _plan(limit_price="0.49")

    observation = observe_round21_aggressive_execution(plan, _book(plan))

    assert observation.state == "known_no_fill"
    assert observation.filled_quantity == 0
    assert observation.unfilled_quantity == plan.quantity
    assert observation.total_fee_quote == 0
    assert observation.execution_cash_flow_quote == 0
    assert observation.blocks_new_exposure is False
    assert observation.execution_book_sha256


def test_round21_missing_or_gapped_execution_evidence_blocks_new_exposure() -> None:
    plan = _plan()

    missing = observe_round21_aggressive_execution(plan, None)
    gapped = observe_round21_aggressive_execution(
        plan,
        _book(plan, gap_free=False),
    )

    for observation in (missing, gapped):
        assert observation.state == "unknown_after_submit"
        assert observation.filled_quantity == 0
        assert observation.conservative_utility_bound_quote == (
            -plan.maximum_loss_quote
        )
        assert observation.blocks_new_exposure is True
    assert missing.execution_book_sha256 == ""
    assert (
        gapped.execution_book_sha256
        == _book(
            plan,
            gap_free=False,
        ).source_payload_sha256
    )


def test_round21_wrong_time_market_or_tampered_plan_is_not_a_no_fill() -> None:
    plan = _plan()

    with pytest.raises(ValueError, match="delayed execution book is invalid"):
        observe_round21_aggressive_execution(
            plan,
            _book(plan, offset_ms=-1),
        )
    with pytest.raises(ValueError, match="delayed execution book is invalid"):
        observe_round21_aggressive_execution(
            plan,
            _book(plan, offset_ms=501),
        )
    with pytest.raises(ValueError, match="aggressive order plan differs"):
        observe_round21_aggressive_execution(
            replace(plan, plan_sha256="f" * 64),
            None,
        )
    with pytest.raises(ValueError, match="aggressive order plan is invalid"):
        Round21AggressiveOrderPlan.create(
            market=_market(),
            market_evidence=_evidence(),
            scenario_name="primary",
            decision_time_ms=DECISION_MS,
            outcome="Up",
            side="BUY",
            quantity=Decimal("10.0000005"),
            limit_price=Decimal("0.55"),
            effective_tick_size=Decimal("0.01"),
            builder_taker_fee_bps=Decimal("0"),
            predictor_evidence_sha256=_digest("predictor"),
            reconciliation_sha256=_digest("reconciliation"),
        )


def test_round21_sell_requires_owned_parent_and_uses_net_proceeds() -> None:
    plan = _plan(
        side="SELL",
        quantity="10",
        limit_price="0.45",
    )
    observation = observe_round21_aggressive_execution(
        plan,
        _book(
            plan,
            bids=(("0.50", "10"),),
            asks=(("0.51", "20"),),
        ),
    )

    assert observation.state == "filled"
    assert observation.average_fill_price == Decimal("0.50")
    assert observation.execution_cash_flow_quote == Decimal("4.820000")
    assert observation.total_fee_quote == Decimal("0.180000")
    assert observation.execution_cash_flow_quote > 0

    with pytest.raises(ValueError, match="aggressive order plan is invalid"):
        Round21AggressiveOrderPlan.create(
            market=_market(),
            market_evidence=_evidence(),
            scenario_name="primary",
            decision_time_ms=DECISION_MS,
            outcome="Up",
            side="SELL",
            quantity=Decimal("10"),
            limit_price=Decimal("0.45"),
            effective_tick_size=Decimal("0.01"),
            builder_taker_fee_bps=Decimal("0"),
            owned_cost_basis_quote=Decimal("4"),
            parent_inventory_id="",
            predictor_evidence_sha256=_digest("predictor"),
            reconciliation_sha256=_digest("reconciliation"),
        )
