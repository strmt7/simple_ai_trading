from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import hashlib
import json
from pathlib import Path

import pytest

import simple_ai_trading.polymarket_round21_policy as policy_module
from simple_ai_trading.paper_execution import BookLevel, PaperBookSnapshot
from simple_ai_trading.polymarket import (
    PolymarketFeeSchedule,
    PolymarketFiveMinuteMarket,
)
from simple_ai_trading.polymarket_round21_execution import (
    Round21MarketExecutionEvidence,
)
from simple_ai_trading.polymarket_round21_policy import (
    POLYMARKET_ROUND21_MULTI_ACTION_POLICY_SHA256,
    Round21BotInventory,
    Round21OwnedLot,
    Round21ProbabilityEnvelope,
    load_round21_multi_action_policy,
    round21_risk_profile,
    select_round21_action,
    validate_round21_multi_action_policy,
)


START_MS = 1_800_000_000_000
DECISION_MS = START_MS + 120_000
CONDITION_ID = "0x" + "3" * 64
UP_TOKEN = "1" * 40
DOWN_TOKEN = "2" * 40
POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-021-multi-action-policy-design-v8.json"
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


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


def _evidence() -> Round21MarketExecutionEvidence:
    return Round21MarketExecutionEvidence.create(
        condition_id=CONDITION_ID,
        observed_wall_ms=DECISION_MS - 1_000,
        observed_monotonic_ns=(DECISION_MS - 1_000) * 1_000_000,
        maker_base_fee=0,
        taker_base_fee=700,
        taker_order_delay_enabled=True,
        general_order_delay_seconds=0,
        minimum_order_age_seconds=0,
        clob_info_sha256=_digest("clob"),
        up_fee_rate_sha256=_digest("up-fee"),
        down_fee_rate_sha256=_digest("down-fee"),
        snapshot_sha256=_digest("snapshot"),
    )


def _book(
    outcome: str,
    *,
    bid: str,
    ask: str,
    quantity: str = "1000",
    age_ms: int = 50,
    connected: bool = True,
    gap_free: bool = True,
) -> PaperBookSnapshot:
    token = UP_TOKEN if outcome == "Up" else DOWN_TOKEN
    received = DECISION_MS - age_ms
    return PaperBookSnapshot(
        venue="polymarket",
        market_id=CONDITION_ID,
        asset_id=token,
        bids=(BookLevel(Decimal(bid), Decimal(quantity)),),
        asks=(BookLevel(Decimal(ask), Decimal(quantity)),),
        source_time_ms=received - 10,
        received_wall_ms=received,
        received_monotonic_ns=received * 1_000_000,
        source_payload_sha256=_digest(
            f"{outcome}-{bid}-{ask}-{quantity}-{age_ms}-{connected}-{gap_free}"
        ),
        connected=connected,
        gap_free=gap_free,
    ).validated()


def _envelope(
    probability: str = "0.80",
    lower: str = "0.75",
    upper: str = "0.85",
    *,
    layer: str = "core",
    condition_id: str = CONDITION_ID,
    decision_time_ms: int = DECISION_MS,
    feature_support_eligible: bool = True,
) -> Round21ProbabilityEnvelope:
    return Round21ProbabilityEnvelope.create(
        condition_id=condition_id,
        decision_time_ms=decision_time_ms,
        probability_up=Decimal(probability),
        lower_up=Decimal(lower),
        upper_up=Decimal(upper),
        model_layer=layer,
        source_model_artifact_sha256=_digest(
            f"model-{probability}-{lower}-{upper}-{layer}"
        ),
        source_probability_batch_sha256=_digest("probability-batch"),
        feature_row_sha256=_digest("causal-feature-row"),
        feature_support_eligible=feature_support_eligible,
    )


def _inventory(
    *lots: Round21OwnedLot,
    blocking: bool = False,
) -> Round21BotInventory:
    return Round21BotInventory.create(
        condition_id=CONDITION_ID,
        lots=lots,
        blocking_unknown_state=blocking,
    )


def _lot(
    outcome: str,
    *,
    quantity: str = "10",
    cost: str = "4",
    parent: str | None = None,
    offset_ms: int = 10_000,
) -> Round21OwnedLot:
    return Round21OwnedLot.create(
        condition_id=CONDITION_ID,
        outcome=outcome,
        quantity=Decimal(quantity),
        cost_basis_quote=Decimal(cost),
        opened_at_ms=START_MS + offset_ms,
        parent_inventory_id=(
            f"owned-{outcome.lower()}-{offset_ms}" if parent is None else parent
        ),
    )


def _select(
    *,
    envelope: Round21ProbabilityEnvelope | None = None,
    inventory: Round21BotInventory | None = None,
    up_book: PaperBookSnapshot | None = None,
    down_book: PaperBookSnapshot | None = None,
    daily_pnl: str = "0",
    drawdown: str = "0",
    cooldown_until_ms: int = 0,
    transition_pending: bool = False,
    reconciliation_ok: bool = True,
    capital: str = "10000",
    cash: str = "1000",
    condition_start_cash: str | None = None,
    directional_entry_allowed: bool = True,
    directional_entry_permission_sha256: str = "",
):
    return select_round21_action(
        market=_market(),
        market_evidence=_evidence(),
        books={
            "Up": (_book("Up", bid="0.49", ask="0.50") if up_book is None else up_book),
            "Down": (
                _book("Down", bid="0.49", ask="0.50")
                if down_book is None
                else down_book
            ),
        },
        envelope=_envelope() if envelope is None else envelope,
        inventory=_inventory() if inventory is None else inventory,
        decision_time_ms=DECISION_MS,
        risk_capital_quote=Decimal(capital),
        available_cash_quote=Decimal(cash),
        condition_start_cash_quote=Decimal(
            cash if condition_start_cash is None else condition_start_cash
        ),
        daily_realized_pnl_quote=Decimal(daily_pnl),
        drawdown_capital_fraction=Decimal(drawdown),
        cooldown_until_ms=cooldown_until_ms,
        transition_pending=transition_pending,
        reconciliation_ok=reconciliation_ok,
        reconciliation_sha256=_digest("reconciliation"),
        minimum_edge_per_share=Decimal("0.02"),
        directional_entry_allowed=directional_entry_allowed,
        directional_entry_permission_sha256=(directional_entry_permission_sha256),
    )


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


def test_round21_multi_action_policy_freezes_profiles_and_independence() -> None:
    policy = load_round21_multi_action_policy(POLICY_PATH)

    assert policy["design_sha256"] == POLYMARKET_ROUND21_MULTI_ACTION_POLICY_SHA256
    assert round21_risk_profile().name == "conservative"
    assert round21_risk_profile().default is True
    assert round21_risk_profile("aggressive").default is False
    assert policy["inventory"]["foreign_positions"].startswith("never_")
    assert policy_module.credentials_used is False
    assert policy_module.account_connected is False
    assert policy_module.binance_execution_connected is False
    assert policy_module.live_trading_authority is False

    changed = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    changed["inventory"]["foreign_positions"] = "include_account_positions"
    with pytest.raises(ValueError, match="multi-action policy differs"):
        validate_round21_multi_action_policy(_rehash_policy(changed))


def test_round21_selects_directional_entry_with_true_loss_bound() -> None:
    decision = _select()

    assert decision.action == "buy_up"
    assert decision.plan is not None
    assert decision.plan.side == "BUY"
    assert decision.plan.outcome == "Up"
    assert decision.plan.maximum_loss_quote <= Decimal("10")
    assert decision.plan.quantity >= _market().minimum_order_size
    assert decision.edge_per_share >= Decimal("0.02")
    assert decision.identity_payload()["binance_execution_connected"] is False
    assert decision.trading_authority is False


def test_round21_abstains_when_no_probability_bound_clears_fees() -> None:
    decision = _select(
        envelope=_envelope("0.50", "0.48", "0.52"),
    )

    assert decision.action == "abstain"
    assert decision.reason == "no_positive_after_cost_action"
    assert decision.plan is None


def test_round21_probability_endpoints_are_valid_but_cannot_force_action() -> None:
    decision = _select(
        envelope=_envelope("0", "0", "1"),
    )

    assert decision.action == "abstain"
    assert decision.reason == "no_positive_after_cost_action"


def test_round21_reduces_exact_oldest_owned_lot_before_new_entry() -> None:
    lot = _lot("Up", parent="exact-owned-up")
    decision = _select(
        envelope=_envelope("0.40", "0.35", "0.45"),
        inventory=_inventory(lot),
        up_book=_book("Up", bid="0.60", ask="0.61", quantity="10"),
        down_book=_book("Down", bid="0.69", ask="0.70", quantity="10"),
    )

    assert decision.action == "reduce_up"
    assert decision.plan is not None
    assert decision.plan.side == "SELL"
    assert decision.plan.parent_inventory_id == "exact-owned-up"
    assert decision.plan.quantity == lot.quantity


def test_round21_positive_complement_lock_precedes_reduction() -> None:
    lot = _lot("Up", cost="4", parent="exact-owned-up")
    decision = _select(
        inventory=_inventory(lot),
        up_book=_book("Up", bid="0.80", ask="0.81", quantity="20"),
        down_book=_book("Down", bid="0.49", ask="0.50", quantity="20"),
        cash="100",
    )

    assert decision.action == "lock_up_with_down"
    assert decision.plan is not None
    assert decision.plan.side == "BUY"
    assert decision.plan.outcome == "Down"
    assert decision.plan.quantity == Decimal("10")
    assert decision.edge_per_share >= Decimal("0.02")


def test_round21_loss_and_cooldown_gates_block_only_new_directional_risk() -> None:
    daily = _select(daily_pnl="-50")
    cooldown = _select(cooldown_until_ms=DECISION_MS + 60_000)
    reduction = _select(
        envelope=_envelope("0.40", "0.35", "0.45"),
        inventory=_inventory(_lot("Up")),
        up_book=_book("Up", bid="0.60", ask="0.61", quantity="10"),
        down_book=_book("Down", bid="0.69", ask="0.70", quantity="10"),
        daily_pnl="-50",
    )

    assert daily.action == "abstain"
    assert daily.reason == "daily_loss_gate_no_positive_reduction"
    assert cooldown.action == "abstain"
    assert cooldown.reason == "cooldown_gate_no_positive_reduction"
    assert reduction.action == "reduce_up"


def test_round21_out_of_support_blocks_entry_but_preserves_owned_reduction() -> None:
    unsupported = _envelope(feature_support_eligible=False)
    entry = _select(envelope=unsupported)
    reduction = _select(
        envelope=_envelope(
            "0.40",
            "0.35",
            "0.45",
            feature_support_eligible=False,
        ),
        inventory=_inventory(_lot("Up")),
        up_book=_book("Up", bid="0.60", ask="0.61", quantity="10"),
        down_book=_book("Down", bid="0.69", ask="0.70", quantity="10"),
    )

    assert entry.action == "abstain"
    assert entry.reason == "feature_support_out_of_distribution_no_risk_reduction"
    assert reduction.action == "reduce_up"


def test_round21_directional_size_cannot_overshoot_remaining_loss_headroom() -> None:
    near_daily_limit = _select(daily_pnl="-49")
    near_drawdown_limit = _select(drawdown="0.0199")

    assert near_daily_limit.plan is not None
    assert near_daily_limit.plan.maximum_loss_quote <= Decimal("1")
    assert near_drawdown_limit.plan is not None
    assert near_drawdown_limit.plan.maximum_loss_quote <= Decimal("1")


def test_round21_realized_intra_event_loss_cannot_be_forgotten_after_exit() -> None:
    decision = _select(
        cash="989",
        condition_start_cash="1000",
    )

    assert decision.action == "abstain"
    assert decision.reason == "event_loss_gate_no_risk_reducing_action"


def test_round21_paired_reduction_cannot_breach_remaining_event_cap() -> None:
    decision = _select(
        envelope=_envelope("0.40", "0.35", "0.45"),
        inventory=_inventory(
            _lot("Up", quantity="10", cost="4", parent="owned-up"),
            _lot("Down", quantity="10", cost="4", parent="owned-down"),
        ),
        up_book=_book("Up", bid="0.60", ask="0.61", quantity="10"),
        down_book=_book("Down", bid="0.69", ask="0.70", quantity="10"),
        cash="980.5",
        condition_start_cash="1000",
    )

    assert decision.action == "abstain"
    assert decision.reason == "candidate_would_breach_risk_cap"


def test_round21_risk_reducing_complement_lock_remains_available_over_cap() -> None:
    decision = _select(
        inventory=_inventory(_lot("Up", cost="4", parent="owned-up")),
        cash="989",
        condition_start_cash="1000",
    )

    assert decision.action == "lock_up_with_down"
    assert decision.plan is not None
    assert decision.plan.outcome == "Down"


def test_round21_ai_veto_blocks_only_new_directional_entries() -> None:
    permission_sha = _digest("ai-permission")
    entry = _select(
        directional_entry_allowed=False,
        directional_entry_permission_sha256=permission_sha,
    )
    reduction = _select(
        envelope=_envelope("0.40", "0.35", "0.45"),
        inventory=_inventory(_lot("Up")),
        up_book=_book("Up", bid="0.60", ask="0.61", quantity="10"),
        down_book=_book("Down", bid="0.69", ask="0.70", quantity="10"),
        directional_entry_allowed=False,
        directional_entry_permission_sha256=permission_sha,
    )

    assert entry.action == "abstain"
    assert entry.reason == "ai_veto_no_positive_reduction"
    assert reduction.action == "reduce_up"
    with pytest.raises(ValueError, match="requires bound permission evidence"):
        _select(directional_entry_allowed=False)


def test_round21_reconciliation_transition_and_unknown_state_fail_closed() -> None:
    reconciliation = _select(reconciliation_ok=False)
    transition = _select(transition_pending=True)
    unknown = _select(inventory=_inventory(blocking=True))

    assert reconciliation.reason == "ownership_reconciliation_failed"
    assert transition.reason == "active_transition_pending"
    assert unknown.reason == "unknown_order_or_fill_state"
    assert all(
        decision.action == "abstain"
        for decision in (reconciliation, transition, unknown)
    )


def test_round21_stale_books_and_tampered_owned_lot_cannot_create_action() -> None:
    stale_up = _book("Up", bid="0.49", ask="0.50", age_ms=501)
    stale_down = _book("Down", bid="0.49", ask="0.50", age_ms=501)
    decision = _select(up_book=stale_up, down_book=stale_down)

    assert decision.action == "abstain"
    assert decision.reason == "creation_book_context_unavailable"
    with pytest.raises(ValueError, match="owned lot differs"):
        _inventory(replace(_lot("Up"), cost_basis_quote=Decimal("0.01")))


def test_round21_rejects_cross_condition_or_wrong_time_probability_evidence() -> None:
    with pytest.raises(ValueError, match="action context is invalid"):
        _select(envelope=_envelope(condition_id="0x" + "4" * 64))
    with pytest.raises(ValueError, match="action context is invalid"):
        _select(envelope=_envelope(decision_time_ms=DECISION_MS - 250))


def test_round21_validates_execution_scenario_even_when_abstaining() -> None:
    with pytest.raises(ValueError, match="execution scenario is unknown"):
        select_round21_action(
            market=_market(),
            market_evidence=_evidence(),
            books={
                "Up": _book("Up", bid="0.49", ask="0.50"),
                "Down": _book("Down", bid="0.49", ask="0.50"),
            },
            envelope=_envelope("0.50", "0.48", "0.52"),
            inventory=_inventory(),
            decision_time_ms=DECISION_MS,
            risk_capital_quote=Decimal("10000"),
            available_cash_quote=Decimal("1000"),
            condition_start_cash_quote=Decimal("1000"),
            daily_realized_pnl_quote=Decimal("0"),
            drawdown_capital_fraction=Decimal("0"),
            cooldown_until_ms=0,
            transition_pending=False,
            reconciliation_ok=True,
            reconciliation_sha256=_digest("reconciliation"),
            minimum_edge_per_share=Decimal("0.02"),
            scenario_name="unregistered",
        )


def test_round21_decision_hash_rejects_action_or_parent_drift() -> None:
    decision = _select()
    assert decision.plan is not None

    with pytest.raises(ValueError, match="action decision differs"):
        replace(decision, action="buy_down").validated()
    with pytest.raises(ValueError, match="probability envelope differs"):
        replace(_envelope(), lower_up=Decimal("0.70")).validated()
    with pytest.raises(ValueError, match="probability envelope differs"):
        replace(
            _envelope(),
            source_probability_batch_sha256=_digest("other-probability-batch"),
        ).validated()
    with pytest.raises(ValueError, match="aggressive order plan differs"):
        replace(
            decision.plan,
            parent_inventory_id="foreign-position",
        ).validated()
