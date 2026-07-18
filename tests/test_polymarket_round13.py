from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, ROUND_CEILING, localcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from simple_ai_trading import polymarket_round13 as round13_module
from simple_ai_trading.command_contract import command_specs, workflow_commands
from simple_ai_trading.paper_execution import BookLevel, PaperBookSnapshot
from simple_ai_trading.polymarket_replay import PolymarketRecordedBook
from simple_ai_trading.polymarket_repricing import PolymarketRepricingDecision
from simple_ai_trading.polymarket_round12_reference import (
    load_round12_reference_from_round11_artifact,
    polymarket_round12_primary_policy,
)
from simple_ai_trading.polymarket_round13 import (
    PolymarketRound13Attempt,
    PolymarketRound13CalibrationSnapshot,
    PolymarketRound13EntryObservation,
    PolymarketRound13LabelFreeDataset,
    PolymarketRound13Program,
    PolymarketRound13Scenario,
    load_round13_confirmation_contract,
    polymarket_round13_evaluation_gates,
    polymarket_round13_scenarios,
    polymarket_round13_upstream_order_semantics,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "docs/model-research/polymarket/round-011-single-leg-directional-value-artifact.json"
)
SEALED_CONTRACT = (
    ROOT / "docs/model-research/polymarket/round-013-sealed-confirmation-contract.json"
)
CONTRACT_SHA = "a" * 64
SEGMENT_SHA = "b" * 64


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _program() -> PolymarketRound13Program:
    return PolymarketRound13Program(
        contract={"contract_sha256": CONTRACT_SHA},
        contract_sha256=CONTRACT_SHA,
        model=load_round12_reference_from_round11_artifact(ARTIFACT),
        policy=polymarket_round12_primary_policy(),
        scenarios=polymarket_round13_scenarios(),
        evaluation_gates=polymarket_round13_evaluation_gates(),
        confirmation_capital_quote=Decimal("1000"),
    ).validated()


def _observation(
    *,
    condition_id: str = "condition-btc",
    action_sha: str = "c" * 64,
    decision_ns: int = 10_000_000_000,
    state: str = "simulated_fill",
    entry_cost: str | None = "4.0",
    order_amount: str = "4",
    entry_quantity: str | None = "5",
    entry_fee: str | None = "0",
    maximum_loss: str = "4.1",
) -> PolymarketRound13EntryObservation:
    observed = state in {"simulated_fill", "simulated_no_fill"}
    reasons = {
        "simulated_fill": "stressed_displayed_depth_walk_complete",
        "simulated_no_fill": ("insufficient_stressed_displayed_depth_within_fok_limit"),
        "unknown_after_submit": "missing_same_segment_entry_observation",
        "not_submitted": "missing_execution_parameters",
    }
    submitted = state != "not_submitted"
    provisional = PolymarketRound13EntryObservation(
        scenario="primary",
        action_feature_sha256=action_sha,
        condition_id=condition_id,
        outcome="Up",
        decision_event_id=f"decision-{decision_ns}",
        decision_segment_id=SEGMENT_SHA,
        decision_monotonic_ns=decision_ns,
        creation_book_event_id=f"creation-{decision_ns}",
        fok_tick_size="0.01",
        fok_limit_price="0.8",
        order_amount_quote=order_amount,
        execution_parameter_sha256="d" * 64 if submitted else "",
        execution_target_monotonic_ns=(
            decision_ns + 500_000_000 if submitted else None
        ),
        entry_book_event_id=f"entry-{decision_ns}" if observed else "",
        entry_book_segment_id=SEGMENT_SHA if observed else "",
        entry_book_monotonic_ns=(decision_ns + 510_000_000 if observed else None),
        entry_book_tick_size="0.01" if observed else None,
        submission_attempted=submitted,
        observation_state=state,
        entry_modeled_quantity=entry_quantity if state == "simulated_fill" else None,
        entry_fee_quote=entry_fee if state == "simulated_fill" else None,
        entry_cost_quote=entry_cost if state == "simulated_fill" else None,
        maximum_entry_loss_quote=maximum_loss,
        reason=reasons[state],
        source_evidence_sha256="e" * 64,
        evidence_sha256="",
    )
    return replace(
        provisional,
        evidence_sha256=_sha(provisional.identity_payload()),
    ).validated()


def _attempt(
    *,
    index: int = 0,
    decision_ns: int = 10_000_000_000,
    observation: PolymarketRound13EntryObservation | None = None,
    probability: float = 0.9,
    order_amount: str = "4",
    minimum_quantity: str = "5",
    creation_quantity: str = "5",
    creation_fee: str = "0",
    conservative_cost: str = "4",
) -> PolymarketRound13Attempt:
    selected = observation or _observation(decision_ns=decision_ns)
    edge = Decimal(minimum_quantity) * Decimal(str(probability)) - Decimal(
        conservative_cost
    )
    provisional = PolymarketRound13Attempt(
        scenario="primary",
        policy="calibrated",
        condition_id=selected.condition_id,
        asset="BTC",
        event_start_ms=1_700_000_000_000,
        attempt_index=index,
        action_feature_sha256=selected.action_feature_sha256,
        decision_event_id=selected.decision_event_id,
        decision_monotonic_ns=selected.decision_monotonic_ns,
        remaining_seconds=180.0,
        outcome="Up",
        probability=probability,
        expected_edge_quote=format(edge, "f"),
        order_amount_quote=order_amount,
        minimum_signed_quantity=minimum_quantity,
        creation_modeled_quantity=creation_quantity,
        creation_fee_quote=creation_fee,
        conservative_entry_cost_quote=conservative_cost,
        observation=selected,
        attempt_sha256="",
    )
    return replace(
        provisional,
        attempt_sha256=_sha(provisional.identity_payload()),
    ).validated()


def _snapshot(
    condition: str, asset: str, suffix: str
) -> PolymarketRound13CalibrationSnapshot:
    provisional = PolymarketRound13CalibrationSnapshot(
        condition_id=condition,
        asset=asset,
        event_start_ms=1_700_000_000_000,
        action_feature_up_sha256=suffix * 64,
        action_feature_down_sha256=(str((int(suffix) + 1) % 10)) * 64,
        decision_event_id=f"snapshot-{asset}",
        decision_monotonic_ns=9_000_000_000,
        remaining_seconds=120.0,
        market_prior_up=0.75,
        calibrated_probability_up=0.82,
        snapshot_sha256="",
    )
    return replace(
        provisional,
        snapshot_sha256=_sha(provisional.identity_payload()),
    ).validated()


def _dataset(
    attempts: tuple[PolymarketRound13Attempt, ...],
) -> PolymarketRound13LabelFreeDataset:
    conditions = ("condition-btc", "condition-eth", "condition-sol")
    provisional = PolymarketRound13LabelFreeDataset(
        contract_sha256=CONTRACT_SHA,
        source_run_id="run",
        source_feature_dataset_sha256="1" * 64,
        source_action_dataset_sha256="2" * 64,
        model_sha256=_program().model.model_sha256,
        policy_sha256=_program().policy.policy_sha256,
        event_start_ms=1_700_000_000_000,
        condition_ids=conditions,
        calibration_snapshots=(
            _snapshot(conditions[0], "BTC", "3"),
            _snapshot(conditions[1], "ETH", "5"),
            _snapshot(conditions[2], "SOL", "7"),
        ),
        attempts=attempts,
        abstention_counts={},
        dataset_sha256="",
    )
    return replace(
        provisional,
        dataset_sha256=_sha(provisional.identity_payload()),
    )


def test_round13_program_is_host_neutral_and_exact() -> None:
    program = _program()

    assert program.validated() is program
    assert program.confirmation_capital_quote == 1000
    assert program.evaluation_gates.bootstrap_samples == 10_000
    assert len(program.scenarios) == 7
    semantics = polymarket_round13_upstream_order_semantics()
    assert semantics["market_buy_side"] == "quote_amount"
    assert semantics["book_depth_unit"] == "shares"
    assert semantics["buy_fee_unit"] == "quote_collateral"
    assert semantics["required_live_clob_protocol_version"] == 2
    assert semantics["minimum_order_size_documented_unit"] == "unspecified"
    assert semantics["minimum_order_size_unit_assumption"] is False
    assert semantics["utility_share_quantity"] == (
        "signed_minimum_not_modeled_price_improvement"
    )
    assert len(semantics["source_files"]) == 12
    assert all(len(item["sha256"]) == 64 for item in semantics["source_files"])


def test_round13_sealed_contract_is_relocatable_and_rejects_semantic_tamper(
    tmp_path: Path,
) -> None:
    program = load_round13_confirmation_contract(SEALED_CONTRACT)
    relocated_contract = tmp_path / SEALED_CONTRACT.name
    relocated_predecessor = tmp_path / ARTIFACT.name
    relocated_contract.write_bytes(SEALED_CONTRACT.read_bytes())
    relocated_predecessor.write_bytes(ARTIFACT.read_bytes())

    relocated = load_round13_confirmation_contract(relocated_contract)
    assert relocated.contract_sha256 == program.contract_sha256

    payload = json.loads(relocated_contract.read_text(encoding="utf-8"))
    payload.pop("contract_sha256")
    payload["authority"]["profitability_claim"] = True
    payload["contract_sha256"] = _sha(payload)
    relocated_contract.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with pytest.raises(ValueError, match="authority"):
        load_round13_confirmation_contract(relocated_contract)


def test_round13_scenario_rejects_valid_values_in_the_wrong_combination() -> None:
    changed = PolymarketRound13Scenario("primary", 1000, 2, 1, 1, 2)

    with pytest.raises(ValueError, match="scenario is invalid"):
        changed.validated()


def test_round13_fok_limit_is_tick_aligned_fee_bounded_and_stress_monotone() -> None:
    market = SimpleNamespace(
        minimum_order_size=Decimal("5"),
        tick_size=Decimal("0.01"),
        fee_schedule=SimpleNamespace(
            enabled=True,
            rate=Decimal("0.07"),
            exponent=1,
        ),
    )
    scenarios = {item.name: item for item in polymarket_round13_scenarios()}

    primary = round13_module._conservative_fok_buy_bound(
        market, scenarios["primary"], 0.8, 0.02
    )
    doubled_fee = round13_module._conservative_fok_buy_bound(
        market, scenarios["double_taker_fee"], 0.8, 0.02
    )

    assert primary is not None and doubled_fee is not None
    assert primary.limit_price % market.tick_size == 0
    assert doubled_fee.limit_price % market.tick_size == 0
    assert primary.amount_quote % Decimal("0.01") == 0
    assert doubled_fee.amount_quote % Decimal("0.01") == 0
    assert primary.amount_quote >= market.minimum_order_size
    assert doubled_fee.amount_quote >= market.minimum_order_size
    assert primary.minimum_signed_quantity >= Decimal("5")
    assert doubled_fee.minimum_signed_quantity >= Decimal("5")
    assert primary.minimum_signed_quantity * Decimal(
        "0.8"
    ) - primary.maximum_entry_loss_quote > Decimal("0.02")
    assert doubled_fee.minimum_signed_quantity * Decimal(
        "0.8"
    ) - doubled_fee.maximum_entry_loss_quote > Decimal("0.02")
    assert doubled_fee.limit_price <= primary.limit_price

    with localcontext() as caller_context:
        caller_context.prec = 6
        caller_context.rounding = ROUND_CEILING
        changed_context = round13_module._conservative_fok_buy_bound(
            market,
            scenarios["primary"],
            0.8,
            0.02,
        )

    assert changed_context == primary


def test_round13_fok_bound_fails_closed_on_unpinned_tick_size() -> None:
    market = SimpleNamespace(
        minimum_order_size=Decimal("5"),
        tick_size=Decimal("0.003"),
        fee_schedule=SimpleNamespace(
            enabled=True,
            rate=Decimal("0.07"),
            exponent=1,
        ),
    )

    with pytest.raises(ValueError, match="unsupported by the pinned V2 client"):
        round13_module._conservative_fok_buy_bound(
            market,
            polymarket_round13_scenarios()[0],
            0.8,
            0.02,
        )


def test_round13_numeric_guard_abstains_at_probability_boundary() -> None:
    policy = polymarket_round12_primary_policy()
    executions = {
        "Up": (Decimal("5"), Decimal("3")),
        "Down": (Decimal("5"), Decimal("4.9")),
    }

    outcome, probability, edge, reason = round13_module._select_outcome(
        policy.minimum_direction_probability,
        executions,
        policy,
    )

    assert (outcome, probability, edge) == (None, None, None)
    assert reason == "no_positive_conservative_edge"


def test_round13_fok_buy_spends_quote_and_derives_observed_shares() -> None:
    market = SimpleNamespace(
        condition_id="condition",
        minimum_order_size=Decimal("5"),
        tick_size=Decimal("0.01"),
        fee_schedule=SimpleNamespace(
            enabled=True,
            rate=Decimal("0.07"),
            exponent=1,
        ),
    )
    snapshot = PaperBookSnapshot(
        venue="polymarket",
        market_id="condition",
        asset_id="token",
        bids=(BookLevel(Decimal("0.4"), Decimal("20")),),
        asks=(
            BookLevel(Decimal("0.5"), Decimal("4")),
            BookLevel(Decimal("0.6"), Decimal("10")),
        ),
        source_time_ms=1,
        received_wall_ms=2,
        received_monotonic_ns=3,
        source_payload_sha256="f" * 64,
    ).validated()
    book = PolymarketRecordedBook(
        run_id="run",
        event_id="book",
        event_type="book",
        connection_id="connection",
        segment_id=SEGMENT_SHA,
        sequence_number=1,
        sub_index=0,
        market=market,
        outcome="Up",
        tick_size=Decimal("0.01"),
        snapshot=snapshot,
    )
    decision = PolymarketRepricingDecision.from_book(book)

    result = round13_module._walk_transformed_asks(
        book,
        decision,
        polymarket_round13_scenarios()[0],
        limit_price=Decimal("0.6"),
        order_amount_quote=Decimal("3"),
    )

    assert result is not None
    cost, filled_quantity, fee, fills = result
    assert sum((Decimal(item[2]) for item in fills), Decimal("0")) == Decimal("3")
    assert filled_quantity == Decimal("5.666666")
    assert cost == Decimal("3") + fee
    assert cost > Decimal("3")


def test_round13_attempt_recomputes_expected_edge() -> None:
    attempt = _attempt()
    tampered = replace(attempt, expected_edge_quote="0.49", attempt_sha256="")
    tampered = replace(
        tampered,
        attempt_sha256=_sha(tampered.identity_payload()),
    )

    with pytest.raises(ValueError, match="attempt is invalid"):
        tampered.validated()


def test_round13_attempt_does_not_credit_modeled_price_improvement() -> None:
    attempt = _attempt(
        minimum_quantity="5",
        creation_quantity="6",
        conservative_cost="4",
        probability=0.9,
    )

    assert attempt.expected_edge_quote == "0.5"


def test_round13_attempt_rejects_fok_loss_bound_that_erases_frozen_edge() -> None:
    observation = _observation(maximum_loss="4.49")

    with pytest.raises(ValueError, match="attempt is invalid"):
        _attempt(observation=observation)


def test_round13_entry_rejects_observation_after_frozen_horizon() -> None:
    observation = _observation()
    changed = replace(
        observation,
        entry_book_monotonic_ns=observation.execution_target_monotonic_ns + 500_000_001,  # type: ignore[operator]
        evidence_sha256="",
    )
    changed = replace(changed, evidence_sha256=_sha(changed.identity_payload()))

    with pytest.raises(ValueError, match="timing evidence"):
        changed.validated()


def test_round13_tick_drift_preserves_the_observed_entry_book() -> None:
    observation = _observation(state="unknown_after_submit", entry_cost=None)
    changed = replace(
        observation,
        entry_book_event_id="entry-tick-drift",
        entry_book_segment_id=SEGMENT_SHA,
        entry_book_monotonic_ns=(
            observation.execution_target_monotonic_ns + 10_000_000  # type: ignore[operator]
        ),
        entry_book_tick_size="0.02",
        reason="post_submit_tick_size_drift",
        source_evidence_sha256="f" * 64,
        evidence_sha256="",
    )
    changed = replace(changed, evidence_sha256=_sha(changed.identity_payload()))

    assert changed.validated().entry_book_event_id == "entry-tick-drift"


def test_round13_dataset_rejects_retry_after_terminal_state() -> None:
    first = _attempt()
    second_observation = _observation(
        action_sha="f" * 64,
        decision_ns=11_100_000_000,
        state="simulated_no_fill",
        entry_cost=None,
    )
    second = _attempt(
        index=1,
        decision_ns=11_100_000_000,
        observation=second_observation,
    )
    dataset = _dataset((first, second))

    with pytest.raises(ValueError, match="dataset is invalid"):
        dataset.validated()


def test_round13_dataset_rejects_exposure_above_explicit_capital() -> None:
    observation = _observation(
        entry_cost="1100",
        order_amount="1000",
        entry_quantity="2000",
        entry_fee="100",
        maximum_loss="1100",
    )
    attempt = _attempt(
        observation=observation,
        probability=0.9,
        order_amount="1000",
        minimum_quantity="2000",
        creation_quantity="2000",
        creation_fee="0",
        conservative_cost="1000",
    )
    dataset = _dataset((attempt,))

    with pytest.raises(ValueError, match="dataset is invalid"):
        dataset.validated()


def test_round13_dataset_accepts_one_hash_linked_terminal_attempt() -> None:
    dataset = _dataset((_attempt(),))
    validated = replace(
        dataset,
        dataset_sha256=_sha(dataset.identity_payload()),
    ).validated()

    assert validated.condition_ids == (
        "condition-btc",
        "condition-eth",
        "condition-sol",
    )


def test_round13_cli_and_windows_contract_expose_the_same_confirmation_workflow() -> (
    None
):
    specs = {item.name: item for item in command_specs()}
    workflows = {item.name: item for item in workflow_commands()}

    evaluate = specs["polymarket-round13-evaluate"]
    publish = specs["polymarket-round13-publish"]
    assert {item.dest for item in evaluate.options} >= {
        "database",
        "run_id",
        "pipeline_report_sha256",
        "contract",
        "resolution_wait_seconds",
    }
    assert {item.dest for item in publish.options} >= {
        "database",
        "report_sha256",
        "research_root",
    }
    assert workflows[evaluate.name].group == "Polymarket confirmation"
    assert workflows[publish.name].group == "Polymarket confirmation"
