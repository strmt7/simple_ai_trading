"""Supplied transition-aware cash labels, not forecasts or trading admission."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
import hashlib
import json
import re

from .funding_cash_inventory import (
    InventoryCashPath,
    InventoryCashReplay,
    replay_fixed_base_inventory,
)


@dataclass(frozen=True)
class InventoryCashAction:
    action_id: str
    target_signed_base_quantity: Fraction

    def __post_init__(self) -> None:
        if (
            not isinstance(self.action_id, str)
            or re.fullmatch(r"[a-z][a-z0-9_]{0,63}", self.action_id) is None
            or not isinstance(self.target_signed_base_quantity, Fraction)
        ):
            raise ValueError("inventory cash action contract is invalid")


@dataclass(frozen=True)
class InventoryCashActionValue:
    action: InventoryCashAction
    cash_replay: InventoryCashReplay
    surplus_cash_lower: Fraction
    surplus_cash_upper: Fraction
    surplus_reference_lower_bps: Fraction
    surplus_reference_upper_bps: Fraction


@dataclass(frozen=True)
class InventoryCashActionTable:
    symbol: str
    state_as_of_ms: int
    decision_time_ms: int
    initial_signed_base_quantity: Fraction
    state_source_sha256: str
    reference_quote_notional: Fraction
    values: tuple[InventoryCashActionValue, ...]
    input_sha256: str
    schema_version: str = field(
        default="supplied-inventory-cash-actions-v1", init=False
    )
    financially_qualified: bool = field(default=False, init=False)


def evaluate_inventory_cash_actions(
    path: InventoryCashPath,
    actions: tuple[InventoryCashAction, ...],
    *,
    state_as_of_ms: int,
    decision_time_ms: int,
    state_source_sha256: str,
) -> InventoryCashActionTable:
    """Value exact target quantities against immediate incumbent liquidation.

    Every candidate closes at the supplied terminal boundary. This finite-horizon
    label is not a continuous continuation value or an inference-time forecast.
    Price/funding outcomes may never decide a live action. State timestamp and
    digest bind supplied declarations, not native origin or availability proof.
    Fees are proportional modeled quote costs, not native receipts. Hedging,
    financing, basis, margin and partial execution require separate qualification.
    """
    if (
        not isinstance(path, InventoryCashPath)
        or len(path.desired_position) != 1
        or path.one_way_cost_fraction <= 0
        or type(state_as_of_ms) is not int
        or type(decision_time_ms) is not int
        or not 0 <= state_as_of_ms <= decision_time_ms < path.boundary_time_ms[0]
        or not isinstance(state_source_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", state_source_sha256) is None
        or not isinstance(actions, tuple)
        or not actions
        or any(not isinstance(action, InventoryCashAction) for action in actions)
    ):
        raise ValueError("inventory action state, horizon or cost contract is invalid")
    if (
        len({action.action_id for action in actions}) != len(actions)
        or len({action.target_signed_base_quantity for action in actions})
        != len(actions)
        or sum(
            action.action_id == "flatten" and action.target_signed_base_quantity == 0
            for action in actions
        )
        != 1
        or any(
            (action.action_id == "flatten") != (action.target_signed_base_quantity == 0)
            for action in actions
        )
    ):
        raise ValueError(
            "inventory actions require unique targets and a flatten comparator"
        )
    replays = tuple(
        replay_fixed_base_inventory(
            replace(
                path,
                desired_position=(
                    (action.target_signed_base_quantity > 0)
                    - (action.target_signed_base_quantity < 0),
                ),
                target_signed_base_quantity=(action.target_signed_base_quantity,),
            )
        )
        for action in actions
    )
    flatten = next(
        replay
        for action, replay in zip(actions, replays, strict=True)
        if action.action_id == "flatten"
    )
    values: list[InventoryCashActionValue] = []
    for action, replay in zip(actions, replays, strict=True):
        if action.action_id == "flatten":
            lower = upper = Fraction(0)
        else:
            # Independent bounds are conservative; they do not claim jointly
            # attainable extrema or cancel uncertain entry-boundary funding.
            lower = replay.total_net_cash_lower - flatten.total_net_cash_upper
            upper = replay.total_net_cash_upper - flatten.total_net_cash_lower
        values.append(
            InventoryCashActionValue(
                action,
                replay,
                lower,
                upper,
                lower / path.opening_quote_notional * 10_000,
                upper / path.opening_quote_notional * 10_000,
            )
        )
    binding = {
        "schema": "supplied-inventory-cash-actions-v1",
        "symbol": path.symbol,
        "state_as_of_ms": state_as_of_ms,
        "decision_time_ms": decision_time_ms,
        "initial_signed_base_quantity": str(path.initial_signed_base_quantity),
        "state_source_sha256": state_source_sha256,
        "reference_quote_notional": str(path.opening_quote_notional),
        "one_way_cost_fraction": str(path.one_way_cost_fraction),
        "boundary_time_ms": path.boundary_time_ms,
        "boundary_price": [str(price) for price in path.boundary_price],
        "price_source_sha256": path.price_source_sha256,
        "population_certificate_sha256": path.population_certificate_sha256,
        "coverage": [path.coverage_start_ms, path.coverage_end_exclusive_ms],
        "funding": [
            [
                event.funding_time_ms,
                str(event.rate),
                str(event.settlement_mark),
                event.source_body_sha256,
                event.rate_type,
            ]
            for event in path.settlements
        ],
        "actions": [
            [
                value.action.action_id,
                str(value.action.target_signed_base_quantity),
                str(value.cash_replay.total_net_cash_lower),
                str(value.cash_replay.total_net_cash_upper),
                str(value.surplus_cash_lower),
                str(value.surplus_cash_upper),
            ]
            for value in values
        ],
        "terminal": "mandatory_modeled_liquidation",
        "comparison": "conservative_cash_minus_immediate_flatten_not_signed_contrast",
        "admission": "supplied_counterfactual_only_not_hedge_or_training_admission",
    }
    digest = hashlib.sha256(
        json.dumps(
            binding, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("ascii")
    ).hexdigest()
    return InventoryCashActionTable(
        path.symbol,
        state_as_of_ms,
        decision_time_ms,
        path.initial_signed_base_quantity,
        state_source_sha256,
        path.opening_quote_notional,
        tuple(values),
        digest,
    )
