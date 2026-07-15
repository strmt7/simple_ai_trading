from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tools.probe_round57_passive_fill_support import (
    _canonical_json,
    _fill_counts,
    _grouped_trade_flow,
)


ROOT = Path(__file__).resolve().parents[1]
ROUND57_DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "action-value"
    / "round-057-queue-censored-make-take-design.json"
)
POLYMARKET_DESIGN = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-001-paper-parity-design.json"
)
POLYMARKET_RETRY_CONTRACT = (
    ROOT
    / "docs"
    / "model-research"
    / "polymarket"
    / "round-004-causal-retry-contract.json"
)


def _assert_design_identity(path: Path) -> dict[str, object]:
    design = json.loads(path.read_text(encoding="utf-8"))
    claimed = str(design.pop("design_sha256"))
    actual = hashlib.sha256(_canonical_json(design).encode("ascii")).hexdigest()
    assert claimed == actual
    return design


def _assert_contract_identity(path: Path) -> dict[str, object]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    claimed = str(contract.pop("contract_sha256"))
    actual = hashlib.sha256(_canonical_json(contract).encode("ascii")).hexdigest()
    assert claimed == actual
    return contract


def test_passive_fill_requires_queue_and_own_quantity_after_arrival() -> None:
    trade_times, _, trade_quantities, groups = _grouped_trade_flow(
        trade_time_ms=np.array([1_000, 2_000, 3_000, 4_000, 7_000]),
        trade_price=np.array([100.0, 100.0, 100.0, 101.0, 100.0]),
        trade_quantity=np.array([100.0, 2.0, 100.0, 100.0, 1.0]),
        buyer_is_maker=np.array([True, True, False, True, True]),
    )

    fills = _fill_counts(
        arrivals_ms=np.array([1_000]),
        prices=np.array([100.0]),
        queue_ahead=np.array([2.0]),
        buyer_is_maker=True,
        order_notional_quote=100.0,
        trade_times=trade_times,
        trade_quantities=trade_quantities,
        groups=groups,
    )

    assert fills == {5_000: 0, 15_000: 1, 30_000: 1}


def test_frozen_designs_have_valid_identities_and_shared_lifecycle() -> None:
    round57 = _assert_design_identity(ROUND57_DESIGN)
    polymarket = _assert_design_identity(POLYMARKET_DESIGN)

    assert round57["status"] == "frozen"
    assert polymarket["status"] == "frozen_for_implementation"
    parity = polymarket["lifecycle_parity"]
    assert parity["venue_fork_permitted"] is False
    assert set(parity["required_shared_core"]) == {
        "ownership",
        "idempotency",
        "reconciliation",
        "outage_recovery",
        "pause",
        "stop_and_close",
        "loss_budgets",
        "data_freshness",
        "api_budget",
        "coordinator_deadlines",
    }


def test_polymarket_retry_contract_preserves_binance_paper_safety() -> None:
    contract = _assert_contract_identity(POLYMARKET_RETRY_CONTRACT)

    assert contract["status"] == "frozen_before_round_003_outcomes"
    policy = contract["execution_policy"]
    assert policy["venue_order_type"] == "FOK"
    assert policy["stop_after_fill"] is True
    assert policy["future_book_selection_forbidden"] is True
    assert policy["future_outcome_selection_forbidden"] is True
    assert policy["terminal_zero_fill_states"] == ["CANCELLED", "EXPIRED"]
    assert contract["lifecycle_contract"]["binance_paper_parity_required"] is True
    assert contract["lifecycle_contract"]["external_inventory_may_be_touched"] is False
    assert contract["promotion_gates"]["control_may_not_be_replaced_if_challenger_only_increases_activity"] is True
    assert contract["truth_constraints"]["untouched_confirmation_requires_a_later_prospective_capture"] is True
