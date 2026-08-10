"""Deterministic independent Polymarket multi-action policy for Round 21."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation, ROUND_CEILING, ROUND_DOWN
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping, Sequence

from .paper_execution import PaperBookSnapshot, PolymarketFeeModel
from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_round21_contract import POLYMARKET_ROUND21_CONTRACT_SHA256
from .polymarket_round21_execution import (
    POLYMARKET_ROUND21_BUILDER_FEE_QUANTUM,
    POLYMARKET_ROUND21_EXECUTION_POLICY_SHA256,
    POLYMARKET_ROUND21_MAXIMUM_BUILDER_TAKER_FEE_BPS,
    POLYMARKET_ROUND21_PLATFORM_FEE_QUANTUM,
    POLYMARKET_ROUND21_SHARE_QUANTUM,
    Round21AggressiveOrderPlan,
    Round21MarketExecutionEvidence,
    round21_execution_scenario,
)


POLYMARKET_ROUND21_MULTI_ACTION_POLICY_SCHEMA_VERSION = (
    "polymarket-round21-multi-action-policy-design-v8"
)
POLYMARKET_ROUND21_MULTI_ACTION_POLICY_SHA256 = (
    "3286725dcdcc01b4b1f82717b59e30e94a123c5d1cdb0726f887ff3c4517308e"
)
POLYMARKET_ROUND21_PROBABILITY_ENVELOPE_DESIGN_SHA256 = (
    "049b4f88e1746286176009ab8fe974fcceff2ae3b262d7142f10b996a874a125"
)
POLYMARKET_ROUND21_AI_VETO_DESIGN_SHA256 = (
    "65ebb61934cac403af4369e862d851db09126efbe5d35e4b091d851ee251d7c5"
)
POLYMARKET_ROUND21_DETERMINISTIC_PERMISSION_SHA256 = hashlib.sha256(
    b"polymarket-round21-deterministic-core-permission-v1"
).hexdigest()
POLYMARKET_ROUND21_ACTION_DECISION_SCHEMA_VERSION = (
    "polymarket-round21-multi-action-decision-v2"
)
POLYMARKET_ROUND21_MAXIMUM_CREATION_BOOK_AGE_MS = 500
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONDITION_ID = re.compile(r"^0x[0-9a-f]{64}$")
_INVENTORY_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
_EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
_MAX_POLICY_BYTES = 256 * 1024


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Round 21 action policy has duplicate JSON keys")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> object:
    raise ValueError(f"Round 21 action policy contains {value}")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _decimal(
    value: object,
    *,
    name: str,
    positive: bool = False,
) -> Decimal:
    if isinstance(value, bool):
        raise ValueError(f"Round 21 {name} is not a finite decimal")
    try:
        selected = value if isinstance(value, Decimal) else Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"Round 21 {name} is not a finite decimal") from exc
    if not selected.is_finite() or (positive and selected <= 0):
        raise ValueError(f"Round 21 {name} is not a valid decimal")
    return selected


def _digest(value: object, *, name: str) -> str:
    selected = str(value or "").strip().lower()
    if _SHA256.fullmatch(selected) is None or selected == _EMPTY_SHA256:
        raise ValueError(f"Round 21 {name} identity is invalid")
    return selected


def _floor_quantity(value: Decimal) -> Decimal:
    return value.quantize(POLYMARKET_ROUND21_SHARE_QUANTUM, rounding=ROUND_DOWN)


@dataclass(frozen=True, slots=True)
class Round21RiskProfile:
    name: str
    maximum_event_loss_capital_fraction: Decimal
    maximum_daily_loss_capital_fraction: Decimal
    maximum_drawdown_capital_fraction: Decimal
    loss_cluster_cooldown_minutes: int
    maximum_displayed_depth_participation: Decimal
    default: bool

    def validated(self) -> Round21RiskProfile:
        name = str(self.name or "").strip().lower()
        event = _decimal(
            self.maximum_event_loss_capital_fraction,
            name="maximum event loss fraction",
            positive=True,
        )
        daily = _decimal(
            self.maximum_daily_loss_capital_fraction,
            name="maximum daily loss fraction",
            positive=True,
        )
        drawdown = _decimal(
            self.maximum_drawdown_capital_fraction,
            name="maximum drawdown fraction",
            positive=True,
        )
        participation = _decimal(
            self.maximum_displayed_depth_participation,
            name="displayed depth participation",
            positive=True,
        )
        expected = {
            "conservative": (
                Decimal("0.001"),
                Decimal("0.005"),
                Decimal("0.02"),
                30,
                Decimal("0.10"),
                True,
            ),
            "regular": (
                Decimal("0.002"),
                Decimal("0.01"),
                Decimal("0.04"),
                15,
                Decimal("0.20"),
                False,
            ),
            "aggressive": (
                Decimal("0.0035"),
                Decimal("0.015"),
                Decimal("0.06"),
                5,
                Decimal("0.30"),
                False,
            ),
        }
        if expected.get(name) != (
            event,
            daily,
            drawdown,
            int(self.loss_cluster_cooldown_minutes),
            participation,
            self.default,
        ):
            raise ValueError("Round 21 risk profile differs")
        return replace(
            self,
            name=name,
            maximum_event_loss_capital_fraction=event,
            maximum_daily_loss_capital_fraction=daily,
            maximum_drawdown_capital_fraction=drawdown,
            loss_cluster_cooldown_minutes=int(self.loss_cluster_cooldown_minutes),
            maximum_displayed_depth_participation=participation,
        )


POLYMARKET_ROUND21_RISK_PROFILES = tuple(
    Round21RiskProfile(
        name=name,
        maximum_event_loss_capital_fraction=event,
        maximum_daily_loss_capital_fraction=daily,
        maximum_drawdown_capital_fraction=drawdown,
        loss_cluster_cooldown_minutes=cooldown,
        maximum_displayed_depth_participation=participation,
        default=default,
    ).validated()
    for name, event, daily, drawdown, cooldown, participation, default in (
        (
            "conservative",
            Decimal("0.001"),
            Decimal("0.005"),
            Decimal("0.02"),
            30,
            Decimal("0.10"),
            True,
        ),
        (
            "regular",
            Decimal("0.002"),
            Decimal("0.01"),
            Decimal("0.04"),
            15,
            Decimal("0.20"),
            False,
        ),
        (
            "aggressive",
            Decimal("0.0035"),
            Decimal("0.015"),
            Decimal("0.06"),
            5,
            Decimal("0.30"),
            False,
        ),
    )
)


def round21_risk_profile(name: str = "conservative") -> Round21RiskProfile:
    selected = str(name or "conservative").strip().lower()
    matches = tuple(
        profile
        for profile in POLYMARKET_ROUND21_RISK_PROFILES
        if profile.name == selected
    )
    if len(matches) != 1:
        raise ValueError("Round 21 risk profile is unknown")
    return matches[0]


def validate_round21_multi_action_policy(
    value: Mapping[str, object],
) -> dict[str, object]:
    """Reject rehashed changes to ownership, risk, or venue independence."""

    policy = dict(value)
    claimed = str(policy.pop("design_sha256", "")).strip().lower()
    parents = policy.get("parents")
    decision = policy.get("decision")
    actions = policy.get("actions")
    inventory = policy.get("inventory")
    profiles = policy.get("risk_profiles")
    safety = policy.get("safety")
    independence = policy.get("independence")
    authority = policy.get("authority")
    expected_profiles = {
        profile.name: {
            "default": profile.default,
            "maximum_event_loss_capital_fraction": format(
                profile.maximum_event_loss_capital_fraction,
                "f",
            ),
            "maximum_daily_loss_capital_fraction": format(
                profile.maximum_daily_loss_capital_fraction,
                "f",
            ),
            "maximum_drawdown_capital_fraction": format(
                profile.maximum_drawdown_capital_fraction,
                "f",
            ),
            "loss_cluster_cooldown_minutes": (profile.loss_cluster_cooldown_minutes),
            "maximum_displayed_depth_participation": format(
                profile.maximum_displayed_depth_participation,
                "f",
            ),
        }
        for profile in POLYMARKET_ROUND21_RISK_PROFILES
    }
    expected_decision = {
        "cadence_ms": 250,
        "maximum_creation_book_age_ms": 500,
        "probability_input": (
            "condition_decision_model_artifact_feature_row_and_probability_batch_"
            "bound_calibrated_point_lower_and_upper_up_probability"
        ),
        "probability_range": "inclusive_zero_to_one",
        "minimum_edge_per_share": "immutable_development_selected_parameter",
        "creation_book_context": (
            "exact_fresh_gap_free_up_and_down_structured_book_identities"
        ),
        "future_books_or_outcomes": False,
        "one_active_transition_at_a_time": True,
        "forced_activity": False,
        "feature_support_evidence_required": True,
    }
    expected_actions = {
        "allowed": [
            "abstain",
            "buy_up",
            "buy_down",
            "reduce_up",
            "reduce_down",
            "lock_up_with_down",
            "lock_down_with_up",
        ],
        "priority": [
            "positive_guaranteed_complement_lock",
            "positive_conservative_inventory_reduction",
            "positive_conservative_directional_entry",
            "abstain",
        ],
        "directional_buy_edge": (
            "outcome_lower_probability_minus_limit_price_minus_all_taker_fees"
        ),
        "inventory_reduction_edge": (
            "net_limit_proceeds_minus_outcome_upper_probability"
        ),
        "complement_lock_edge": (
            "one_minus_owned_average_cost_minus_complement_limit_cost_and_all_"
            "taker_fees"
        ),
        "limit_price": ("most_aggressive_tick_still_meeting_the_frozen_edge"),
        "order_type": "FAK",
    }
    expected_inventory = {
        "source": "bot_owned_confirmed_parent_bound_lots_only",
        "foreign_positions": "never_read_as_bot_inventory_and_never_modified",
        "unknown_order_or_fill": "block_new_decisions",
        "event_worst_case_loss": (
            "total_owned_cost_basis_minus_minimum_up_down_owned_quantity_"
            "bounded_below_by_zero"
        ),
        "sell_quantity": "bot_owned_quantity_only",
        "complement_quantity": "unpaired_bot_owned_quantity_only",
        "partial_fills": (
            "persist_exact_filled_inventory_and_cancel_unfilled_remainder"
        ),
    }
    expected_safety = {
        "default_profile": "conservative",
        "daily_loss_or_drawdown_gate": (
            "no_new_directional_entry_reductions_and_positive_locks_remain_available"
        ),
        "cooldown_gate": (
            "no_new_directional_entry_reductions_and_positive_locks_remain_available"
        ),
        "feature_support_gate": (
            "no_new_directional_entry_bot_owned_reductions_and_positive_locks_"
            "remain_available"
        ),
        "feature_support_gate_population": (
            "train_only_condition_equal_weighted_with_tcn_history_contamination"
        ),
        "unsupported_rows_remain_in_proper_scoring": True,
        "new_directional_risk_budget": (
            "minimum_remaining_event_daily_and_drawdown_loss_headroom_after_"
            "current_event_worst_case_loss"
        ),
        "positive_daily_pnl_increases_risk_budget": False,
        "stale_disconnected_gapped_or_wrong_market_book": "abstain",
        "reconciliation_failure": "abstain",
        "cash_and_event_loss_cap": ("both_must_cover_the_maximum_buy_loss_bound"),
        "maker_rebate_credit": False,
        "leverage": False,
        "reinvestment": False,
    }
    expected_independence = {
        "execution_venue": "polymarket_only",
        "binance_execution": False,
        "binance_account": False,
        "binance_risk_or_stop_dependency": False,
        "optional_binance_data_may_only_be_inside_the_hash_bound_probability_"
        "evidence": True,
    }
    if (
        set(policy)
        != {
            "schema_version",
            "round",
            "status",
            "supersedes",
            "parents",
            "decision",
            "actions",
            "inventory",
            "risk_profiles",
            "safety",
            "independence",
            "authority",
        }
        or claimed != POLYMARKET_ROUND21_MULTI_ACTION_POLICY_SHA256
        or claimed != _canonical_sha256(policy)
        or policy.get("schema_version")
        != POLYMARKET_ROUND21_MULTI_ACTION_POLICY_SCHEMA_VERSION
        or policy.get("round") != 21
        or policy.get("status") != "preregistered_during_target_and_model_blind_capture"
        or policy.get("supersedes")
        != "polymarket-round21-multi-action-policy-design-v7"
        or parents
        != {
            "round21_contract_sha256": POLYMARKET_ROUND21_CONTRACT_SHA256,
            "round21_execution_policy_sha256": (
                POLYMARKET_ROUND21_EXECUTION_POLICY_SHA256
            ),
            "round21_probability_envelope_design_sha256": (
                POLYMARKET_ROUND21_PROBABILITY_ENVELOPE_DESIGN_SHA256
            ),
        }
        or decision != expected_decision
        or actions != expected_actions
        or inventory != expected_inventory
        or safety != expected_safety
        or independence != expected_independence
        or not isinstance(decision, Mapping)
        or decision.get("cadence_ms") != 250
        or decision.get("maximum_creation_book_age_ms")
        != POLYMARKET_ROUND21_MAXIMUM_CREATION_BOOK_AGE_MS
        or decision.get("future_books_or_outcomes") is not False
        or decision.get("one_active_transition_at_a_time") is not True
        or decision.get("forced_activity") is not False
        or not isinstance(actions, Mapping)
        or actions.get("order_type") != "FAK"
        or actions.get("priority")
        != [
            "positive_guaranteed_complement_lock",
            "positive_conservative_inventory_reduction",
            "positive_conservative_directional_entry",
            "abstain",
        ]
        or not isinstance(inventory, Mapping)
        or inventory.get("source") != "bot_owned_confirmed_parent_bound_lots_only"
        or inventory.get("foreign_positions")
        != "never_read_as_bot_inventory_and_never_modified"
        or inventory.get("unknown_order_or_fill") != "block_new_decisions"
        or profiles != expected_profiles
        or not isinstance(safety, Mapping)
        or safety.get("default_profile") != "conservative"
        or safety.get("reconciliation_failure") != "abstain"
        or safety.get("maker_rebate_credit") is not False
        or safety.get("leverage") is not False
        or safety.get("reinvestment") is not False
        or not isinstance(independence, Mapping)
        or independence.get("execution_venue") != "polymarket_only"
        or independence.get("binance_execution") is not False
        or independence.get("binance_account") is not False
        or independence.get("binance_risk_or_stop_dependency") is not False
        or authority
        != {
            "model_selected": False,
            "ai_edge_claim": False,
            "profitability_claim": False,
            "paper_trading_authority": False,
            "live_trading_authority": False,
        }
    ):
        raise ValueError("Round 21 multi-action policy differs")
    return {**policy, "design_sha256": claimed}


def load_round21_multi_action_policy(path: str | Path) -> dict[str, object]:
    selected = Path(path)
    if (
        selected.is_symlink()
        or not selected.is_file()
        or not 2 <= selected.stat().st_size <= _MAX_POLICY_BYTES
    ):
        raise ValueError("Round 21 multi-action policy is unavailable")
    try:
        value = json.loads(
            selected.read_text(encoding="utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("Round 21 multi-action policy is unavailable") from exc
    if not isinstance(value, Mapping):
        raise ValueError("Round 21 multi-action policy is not an object")
    return validate_round21_multi_action_policy(value)


@dataclass(frozen=True, slots=True)
class Round21ProbabilityEnvelope:
    condition_id: str
    decision_time_ms: int
    probability_up: Decimal
    lower_up: Decimal
    upper_up: Decimal
    model_layer: str
    source_model_artifact_sha256: str
    source_probability_batch_sha256: str
    feature_row_sha256: str
    feature_support_eligible: bool
    evidence_sha256: str
    trading_authority: bool = False

    @classmethod
    def from_probability_batch(
        cls,
        *,
        batch: object,
        panel: object,
        panel_row_index: int,
    ) -> Round21ProbabilityEnvelope:
        from .polymarket_round21_model import (  # noqa: PLC0415
            Round21InferencePanel,
            Round21ProbabilityBatch,
        )

        if not isinstance(batch, Round21ProbabilityBatch) or not isinstance(
            panel,
            Round21InferencePanel,
        ):
            raise ValueError("Round 21 probability evidence type is invalid")
        selected_batch = batch.validated()
        selected_panel = panel.validate()
        if selected_batch.feature_batch_sha256 != selected_panel.feature_batch_sha256:
            raise ValueError("Round 21 probability evidence population differs")
        index = int(panel_row_index)
        probability, lower, upper = selected_batch.row(index)
        feature_support_eligible = selected_batch.support_eligible(index)
        if index < 0 or index >= len(selected_panel.condition_ids):
            raise ValueError("Round 21 probability evidence row is unavailable")
        return cls.create(
            condition_id=str(selected_panel.condition_ids[index]),
            decision_time_ms=int(selected_panel.decision_time_ms[index]),
            probability_up=Decimal(format(probability, ".17g")),
            lower_up=Decimal(format(lower, ".17g")),
            upper_up=Decimal(format(upper, ".17g")),
            model_layer=selected_batch.population_layer,
            source_model_artifact_sha256=(selected_batch.source_model_artifact_sha256),
            source_probability_batch_sha256=selected_batch.prediction_sha256,
            feature_row_sha256=selected_panel.row_sha256(index),
            feature_support_eligible=feature_support_eligible,
        )

    @classmethod
    def create(
        cls,
        *,
        condition_id: str,
        decision_time_ms: int,
        probability_up: Decimal,
        lower_up: Decimal,
        upper_up: Decimal,
        model_layer: str,
        source_model_artifact_sha256: str,
        source_probability_batch_sha256: str,
        feature_row_sha256: str,
        feature_support_eligible: bool = True,
    ) -> Round21ProbabilityEnvelope:
        condition = str(condition_id or "").strip().lower()
        decision = int(decision_time_ms)
        probability = _decimal(probability_up, name="probability up")
        lower = _decimal(lower_up, name="lower probability")
        upper = _decimal(upper_up, name="upper probability")
        layer = str(model_layer or "").strip().lower()
        model_sha = _digest(
            source_model_artifact_sha256,
            name="source model artifact",
        )
        probability_batch_sha = _digest(
            source_probability_batch_sha256,
            name="source probability batch",
        )
        row_sha = _digest(feature_row_sha256, name="causal feature row")
        if type(feature_support_eligible) is not bool:
            raise ValueError("Round 21 feature support evidence is invalid")
        if (
            _CONDITION_ID.fullmatch(condition) is None
            or decision <= 0
            or not Decimal("0") <= lower <= probability <= upper <= Decimal("1")
            or layer not in {"core", "core_spot", "core_spot_usdm"}
        ):
            raise ValueError("Round 21 probability envelope is invalid")
        payload = {
            "schema_version": POLYMARKET_ROUND21_ACTION_DECISION_SCHEMA_VERSION,
            "multi_action_policy_sha256": (
                POLYMARKET_ROUND21_MULTI_ACTION_POLICY_SHA256
            ),
            "condition_id": condition,
            "decision_time_ms": decision,
            "probability_up": format(probability, "f"),
            "lower_up": format(lower, "f"),
            "upper_up": format(upper, "f"),
            "model_layer": layer,
            "source_model_artifact_sha256": model_sha,
            "source_probability_batch_sha256": probability_batch_sha,
            "feature_row_sha256": row_sha,
            "feature_support_eligible": feature_support_eligible,
            "trading_authority": False,
        }
        return cls(
            condition_id=condition,
            decision_time_ms=decision,
            probability_up=probability,
            lower_up=lower,
            upper_up=upper,
            model_layer=layer,
            source_model_artifact_sha256=model_sha,
            source_probability_batch_sha256=probability_batch_sha,
            feature_row_sha256=row_sha,
            feature_support_eligible=feature_support_eligible,
            evidence_sha256=_canonical_sha256(payload),
        )

    def validated(self) -> Round21ProbabilityEnvelope:
        rebuilt = self.create(
            condition_id=self.condition_id,
            decision_time_ms=self.decision_time_ms,
            probability_up=self.probability_up,
            lower_up=self.lower_up,
            upper_up=self.upper_up,
            model_layer=self.model_layer,
            source_model_artifact_sha256=self.source_model_artifact_sha256,
            source_probability_batch_sha256=self.source_probability_batch_sha256,
            feature_row_sha256=self.feature_row_sha256,
            feature_support_eligible=self.feature_support_eligible,
        )
        if self != rebuilt or self.trading_authority:
            raise ValueError("Round 21 probability envelope differs")
        return self

    def lower(self, outcome: str) -> Decimal:
        item = self.validated()
        if outcome == "Up":
            return item.lower_up
        if outcome == "Down":
            return Decimal("1") - item.upper_up
        raise ValueError("Round 21 outcome is invalid")

    def upper(self, outcome: str) -> Decimal:
        item = self.validated()
        if outcome == "Up":
            return item.upper_up
        if outcome == "Down":
            return Decimal("1") - item.lower_up
        raise ValueError("Round 21 outcome is invalid")


def build_round21_probability_envelopes(
    *,
    batch: object,
    panel: object,
) -> tuple[Round21ProbabilityEnvelope, ...]:
    """Create a probability evidence population with one panel validation pass."""

    from .polymarket_round21_model import (  # noqa: PLC0415
        Round21InferencePanel,
        Round21ProbabilityBatch,
    )

    if not isinstance(batch, Round21ProbabilityBatch) or not isinstance(
        panel,
        Round21InferencePanel,
    ):
        raise ValueError("Round 21 probability evidence type is invalid")
    selected_batch = batch.validated()
    selected_panel = panel.validate()
    if selected_batch.feature_batch_sha256 != selected_panel.feature_batch_sha256:
        raise ValueError("Round 21 probability evidence population differs")
    indices = tuple(int(value) for value in selected_batch.indices)
    row_hashes = selected_panel.row_sha256_many(indices)
    return tuple(
        Round21ProbabilityEnvelope.create(
            condition_id=str(selected_panel.condition_ids[index]),
            decision_time_ms=int(selected_panel.decision_time_ms[index]),
            probability_up=Decimal(
                format(float(selected_batch.probability_up[position]), ".17g")
            ),
            lower_up=Decimal(format(float(selected_batch.lower_up[position]), ".17g")),
            upper_up=Decimal(format(float(selected_batch.upper_up[position]), ".17g")),
            model_layer=selected_batch.population_layer,
            source_model_artifact_sha256=(selected_batch.source_model_artifact_sha256),
            source_probability_batch_sha256=selected_batch.prediction_sha256,
            feature_row_sha256=row_hashes[position],
            feature_support_eligible=bool(
                selected_batch.feature_support_eligible[position]
            ),
        )
        for position, index in enumerate(indices)
    )


@dataclass(frozen=True, slots=True)
class Round21OwnedLot:
    condition_id: str
    outcome: str
    quantity: Decimal
    cost_basis_quote: Decimal
    opened_at_ms: int
    parent_inventory_id: str
    lot_sha256: str

    @classmethod
    def create(
        cls,
        *,
        condition_id: str,
        outcome: str,
        quantity: Decimal,
        cost_basis_quote: Decimal,
        opened_at_ms: int,
        parent_inventory_id: str,
    ) -> Round21OwnedLot:
        condition = str(condition_id or "").strip().lower()
        selected_outcome = str(outcome or "").strip().title()
        selected_quantity = _decimal(quantity, name="owned quantity", positive=True)
        cost = _decimal(
            cost_basis_quote,
            name="owned cost basis",
            positive=True,
        )
        opened = int(opened_at_ms)
        parent = str(parent_inventory_id or "").strip()
        if (
            _CONDITION_ID.fullmatch(condition) is None
            or selected_outcome not in {"Up", "Down"}
            or not _floor_quantity(selected_quantity) == selected_quantity
            or opened <= 0
            or _INVENTORY_ID.fullmatch(parent) is None
        ):
            raise ValueError("Round 21 owned lot is invalid")
        payload = {
            "schema_version": POLYMARKET_ROUND21_ACTION_DECISION_SCHEMA_VERSION,
            "condition_id": condition,
            "outcome": selected_outcome,
            "quantity": format(selected_quantity, "f"),
            "cost_basis_quote": format(cost, "f"),
            "opened_at_ms": opened,
            "parent_inventory_id": parent,
        }
        return cls(
            condition_id=condition,
            outcome=selected_outcome,
            quantity=selected_quantity,
            cost_basis_quote=cost,
            opened_at_ms=opened,
            parent_inventory_id=parent,
            lot_sha256=_canonical_sha256(payload),
        )

    def validated(self) -> Round21OwnedLot:
        rebuilt = self.create(
            condition_id=self.condition_id,
            outcome=self.outcome,
            quantity=self.quantity,
            cost_basis_quote=self.cost_basis_quote,
            opened_at_ms=self.opened_at_ms,
            parent_inventory_id=self.parent_inventory_id,
        )
        if self != rebuilt:
            raise ValueError("Round 21 owned lot differs")
        return self

    @property
    def average_cost(self) -> Decimal:
        return self.cost_basis_quote / self.quantity


@dataclass(frozen=True, slots=True)
class Round21BotInventory:
    condition_id: str
    lots: tuple[Round21OwnedLot, ...]
    blocking_unknown_state: bool
    inventory_sha256: str

    @classmethod
    def create(
        cls,
        *,
        condition_id: str,
        lots: Sequence[Round21OwnedLot],
        blocking_unknown_state: bool = False,
    ) -> Round21BotInventory:
        condition = str(condition_id or "").strip().lower()
        selected = tuple(
            sorted(
                (lot.validated() for lot in lots),
                key=lambda lot: (
                    lot.opened_at_ms,
                    lot.parent_inventory_id,
                    lot.lot_sha256,
                ),
            )
        )
        if (
            _CONDITION_ID.fullmatch(condition) is None
            or type(blocking_unknown_state) is not bool
            or any(lot.condition_id != condition for lot in selected)
            or len({lot.parent_inventory_id for lot in selected}) != len(selected)
        ):
            raise ValueError("Round 21 bot inventory is invalid")
        payload = {
            "schema_version": POLYMARKET_ROUND21_ACTION_DECISION_SCHEMA_VERSION,
            "condition_id": condition,
            "lot_sha256": [lot.lot_sha256 for lot in selected],
            "blocking_unknown_state": blocking_unknown_state,
        }
        return cls(
            condition_id=condition,
            lots=selected,
            blocking_unknown_state=blocking_unknown_state,
            inventory_sha256=_canonical_sha256(payload),
        )

    def validated(self) -> Round21BotInventory:
        rebuilt = self.create(
            condition_id=self.condition_id,
            lots=self.lots,
            blocking_unknown_state=self.blocking_unknown_state,
        )
        if self != rebuilt:
            raise ValueError("Round 21 bot inventory differs")
        return self

    def quantity(self, outcome: str) -> Decimal:
        return sum(
            (lot.quantity for lot in self.lots if lot.outcome == outcome),
            start=Decimal("0"),
        )

    def cost_basis(self, outcome: str) -> Decimal:
        return sum(
            (lot.cost_basis_quote for lot in self.lots if lot.outcome == outcome),
            start=Decimal("0"),
        )

    @property
    def worst_case_loss_quote(self) -> Decimal:
        total_cost = self.cost_basis("Up") + self.cost_basis("Down")
        locked_payout = min(self.quantity("Up"), self.quantity("Down"))
        return max(Decimal("0"), total_cost - locked_payout)


def _fresh_book(
    market: PolymarketFiveMinuteMarket,
    book: PaperBookSnapshot | None,
    *,
    outcome: str,
    decision_time_ms: int,
    already_validated: bool = False,
) -> PaperBookSnapshot | None:
    if book is None:
        return None
    if already_validated:
        selected = book
    else:
        try:
            selected = book.validated()
        except (TypeError, ValueError):
            return None
    token_id = market.up_token_id if outcome == "Up" else market.down_token_id
    age = decision_time_ms - selected.received_wall_ms
    if (
        selected.venue != "polymarket"
        or selected.market_id != market.condition_id
        or selected.asset_id != token_id
        or not selected.connected
        or not selected.gap_free
        or age < 0
        or age > POLYMARKET_ROUND21_MAXIMUM_CREATION_BOOK_AGE_MS
    ):
        return None
    return selected


@lru_cache(maxsize=2_048)
def _book_identity_sha256(book: PaperBookSnapshot) -> str:
    selected = book.validated()
    return _canonical_sha256(
        {
            "venue": selected.venue,
            "market_id": selected.market_id,
            "asset_id": selected.asset_id,
            "bids": [
                [format(level.price, "f"), format(level.quantity, "f")]
                for level in selected.bids
            ],
            "asks": [
                [format(level.price, "f"), format(level.quantity, "f")]
                for level in selected.asks
            ],
            "source_time_ms": selected.source_time_ms,
            "received_wall_ms": selected.received_wall_ms,
            "received_monotonic_ns": selected.received_monotonic_ns,
            "source_payload_sha256": selected.source_payload_sha256,
            "connected": selected.connected,
            "gap_free": selected.gap_free,
        }
    )


def _builder_fee_per_share(price: Decimal, builder_fee_bps: Decimal) -> Decimal:
    raw = price * builder_fee_bps / Decimal("10000")
    if raw == 0:
        return Decimal("0")
    return raw.quantize(
        POLYMARKET_ROUND21_BUILDER_FEE_QUANTUM,
        rounding=ROUND_CEILING,
    )


def _buy_cost_per_share(
    price: Decimal,
    fee_model: PolymarketFeeModel,
    builder_fee_bps: Decimal,
) -> Decimal:
    return (
        price
        + fee_model(price, Decimal("1"), "taker")
        + _builder_fee_per_share(price, builder_fee_bps)
    )


def _sell_proceeds_per_share(
    price: Decimal,
    fee_model: PolymarketFeeModel,
    builder_fee_bps: Decimal,
) -> Decimal:
    return (
        price
        - fee_model(price, Decimal("1"), "taker")
        - _builder_fee_per_share(price, builder_fee_bps)
    )


def _maximum_buy_limit(
    *,
    best_ask: Decimal,
    tick_size: Decimal,
    fair_lower: Decimal,
    minimum_edge: Decimal,
    fee_model: PolymarketFeeModel,
    builder_fee_bps: Decimal,
) -> Decimal | None:
    price = best_ask
    selected: Decimal | None = None
    while price <= Decimal("1") - tick_size:
        edge = fair_lower - _buy_cost_per_share(
            price,
            fee_model,
            builder_fee_bps,
        )
        if edge < minimum_edge:
            break
        selected = price
        price += tick_size
    return selected


def _minimum_sell_limit(
    *,
    best_bid: Decimal,
    tick_size: Decimal,
    fair_upper: Decimal,
    minimum_edge: Decimal,
    fee_model: PolymarketFeeModel,
    builder_fee_bps: Decimal,
) -> Decimal | None:
    price = best_bid
    selected: Decimal | None = None
    while price >= tick_size:
        edge = (
            _sell_proceeds_per_share(
                price,
                fee_model,
                builder_fee_bps,
            )
            - fair_upper
        )
        if edge < minimum_edge:
            break
        selected = price
        price -= tick_size
    return selected


def _displayed_quantity(
    book: PaperBookSnapshot,
    *,
    side: str,
    limit_price: Decimal,
) -> Decimal:
    levels = book.asks if side == "BUY" else book.bids
    return sum(
        (
            level.quantity
            for level in levels
            if (
                level.price <= limit_price
                if side == "BUY"
                else level.price >= limit_price
            )
        ),
        start=Decimal("0"),
    )


@dataclass(frozen=True, slots=True)
class _Candidate:
    priority: int
    action: str
    edge_per_share: Decimal
    plan: Round21AggressiveOrderPlan


def _minimum_sell_cash_flow_quote(
    plan: Round21AggressiveOrderPlan,
    fee_model: PolymarketFeeModel,
) -> Decimal:
    """Bound a full FAK sell fill without assuming favorable price improvement."""

    maximum_price_levels = (
        int((Decimal("1") - plan.tick_size - plan.limit_price) / plan.tick_size) + 1
    )
    maximum_platform_fee = fee_model(
        Decimal("0.5"),
        plan.quantity,
        "taker",
    )
    if maximum_platform_fee > 0:
        maximum_platform_fee += (
            Decimal(maximum_price_levels - 1) * POLYMARKET_ROUND21_PLATFORM_FEE_QUANTUM
        )
    maximum_builder_fee = (
        plan.quantity
        * (Decimal("1") - plan.tick_size)
        * plan.builder_taker_fee_bps
        / Decimal("10000")
    )
    if maximum_builder_fee > 0:
        maximum_builder_fee = maximum_builder_fee.quantize(
            POLYMARKET_ROUND21_BUILDER_FEE_QUANTUM,
            rounding=ROUND_CEILING,
        ) + (Decimal(maximum_price_levels - 1) * POLYMARKET_ROUND21_BUILDER_FEE_QUANTUM)
    return max(
        Decimal("0"),
        plan.quantity * plan.limit_price - maximum_platform_fee - maximum_builder_fee,
    )


def _maximum_event_downside_after_plan(
    *,
    plan: Round21AggressiveOrderPlan,
    inventory: Round21BotInventory,
    available_cash_quote: Decimal,
    condition_start_cash_quote: Decimal,
    fee_model: PolymarketFeeModel,
) -> Decimal:
    """Bound event downside at both endpoints of any possible partial fill."""

    up_quantity = inventory.quantity("Up")
    down_quantity = inventory.quantity("Down")
    current_equity = available_cash_quote + min(up_quantity, down_quantity)
    if plan.side == "BUY":
        post_cash = available_cash_quote - plan.maximum_loss_quote
        if plan.outcome == "Up":
            up_quantity += plan.quantity
        else:
            down_quantity += plan.quantity
    else:
        post_cash = available_cash_quote + _minimum_sell_cash_flow_quote(
            plan,
            fee_model,
        )
        if plan.outcome == "Up":
            up_quantity -= plan.quantity
        else:
            down_quantity -= plan.quantity
    if up_quantity < 0 or down_quantity < 0:
        raise ValueError("Round 21 candidate exceeds bot-owned inventory")
    post_equity = post_cash + min(up_quantity, down_quantity)
    return max(
        Decimal("0"),
        condition_start_cash_quote - min(current_equity, post_equity),
    )


@dataclass(frozen=True, slots=True)
class Round21ActionDecision:
    condition_id: str
    decision_time_ms: int
    action: str
    reason: str
    edge_per_share: Decimal
    minimum_edge_per_share: Decimal
    risk_profile: str
    probability_evidence_sha256: str
    inventory_sha256: str
    reconciliation_sha256: str
    policy_context_sha256: str
    plan: Round21AggressiveOrderPlan | None
    decision_sha256: str
    trading_authority: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ROUND21_ACTION_DECISION_SCHEMA_VERSION,
            "multi_action_policy_sha256": (
                POLYMARKET_ROUND21_MULTI_ACTION_POLICY_SHA256
            ),
            "condition_id": self.condition_id,
            "decision_time_ms": self.decision_time_ms,
            "action": self.action,
            "reason": self.reason,
            "edge_per_share": format(self.edge_per_share, "f"),
            "minimum_edge_per_share": format(
                self.minimum_edge_per_share,
                "f",
            ),
            "risk_profile": self.risk_profile,
            "probability_evidence_sha256": (self.probability_evidence_sha256),
            "inventory_sha256": self.inventory_sha256,
            "reconciliation_sha256": self.reconciliation_sha256,
            "policy_context_sha256": self.policy_context_sha256,
            "plan_sha256": None if self.plan is None else self.plan.plan_sha256,
            "binance_credentials_used": False,
            "binance_execution_connected": False,
            "trading_authority": False,
        }

    def validated(self) -> Round21ActionDecision:
        plan = None if self.plan is None else self.plan.validated()
        if (
            _CONDITION_ID.fullmatch(self.condition_id) is None
            or self.decision_time_ms <= 0
            or self.action
            not in {
                "abstain",
                "buy_up",
                "buy_down",
                "reduce_up",
                "reduce_down",
                "lock_up_with_down",
                "lock_down_with_up",
            }
            or not self.reason
            or not self.edge_per_share.is_finite()
            or self.edge_per_share < 0
            or not self.minimum_edge_per_share.is_finite()
            or self.minimum_edge_per_share <= 0
            or self.risk_profile
            not in {profile.name for profile in POLYMARKET_ROUND21_RISK_PROFILES}
            or any(
                _SHA256.fullmatch(value) is None or value == _EMPTY_SHA256
                for value in (
                    self.probability_evidence_sha256,
                    self.inventory_sha256,
                    self.reconciliation_sha256,
                    self.policy_context_sha256,
                )
            )
            or (self.action == "abstain") != (plan is None)
            or (
                self.action != "abstain"
                and self.edge_per_share < self.minimum_edge_per_share
            )
            or (
                plan is not None
                and (
                    plan.condition_id != self.condition_id
                    or plan.decision_time_ms != self.decision_time_ms
                    or plan.predictor_evidence_sha256
                    != self.probability_evidence_sha256
                    or plan.reconciliation_sha256 != self.reconciliation_sha256
                )
            )
            or self.decision_sha256 != _canonical_sha256(self.identity_payload())
            or self.trading_authority
        ):
            raise ValueError("Round 21 action decision differs")
        return self


def _decision(
    *,
    condition_id: str,
    decision_time_ms: int,
    action: str,
    reason: str,
    edge_per_share: Decimal,
    minimum_edge_per_share: Decimal,
    profile: Round21RiskProfile,
    envelope: Round21ProbabilityEnvelope,
    inventory: Round21BotInventory,
    reconciliation_sha256: str,
    policy_context_sha256: str,
    plan: Round21AggressiveOrderPlan | None,
) -> Round21ActionDecision:
    provisional = Round21ActionDecision(
        condition_id=condition_id,
        decision_time_ms=decision_time_ms,
        action=action,
        reason=reason,
        edge_per_share=edge_per_share,
        minimum_edge_per_share=minimum_edge_per_share,
        risk_profile=profile.name,
        probability_evidence_sha256=envelope.evidence_sha256,
        inventory_sha256=inventory.inventory_sha256,
        reconciliation_sha256=reconciliation_sha256,
        policy_context_sha256=policy_context_sha256,
        plan=plan,
        decision_sha256=_EMPTY_SHA256,
    )
    return replace(
        provisional,
        decision_sha256=_canonical_sha256(provisional.identity_payload()),
    )


def select_round21_action(
    *,
    market: PolymarketFiveMinuteMarket,
    market_evidence: Round21MarketExecutionEvidence,
    books: Mapping[str, PaperBookSnapshot | None],
    envelope: Round21ProbabilityEnvelope,
    inventory: Round21BotInventory,
    decision_time_ms: int,
    risk_capital_quote: Decimal,
    available_cash_quote: Decimal,
    condition_start_cash_quote: Decimal,
    daily_realized_pnl_quote: Decimal,
    drawdown_capital_fraction: Decimal,
    cooldown_until_ms: int,
    transition_pending: bool,
    reconciliation_ok: bool,
    reconciliation_sha256: str,
    minimum_edge_per_share: Decimal,
    risk_profile: str = "conservative",
    scenario_name: str = "primary",
    builder_taker_fee_bps: Decimal = Decimal("0"),
    directional_entry_allowed: bool = True,
    directional_entry_permission_sha256: str = "",
    _validated_replay_inputs: bool = False,
) -> Round21ActionDecision:
    """Select at most one target-blind Polymarket action from bot-owned state."""

    if type(_validated_replay_inputs) is not bool:
        raise ValueError("Round 21 action validation mode is invalid")
    selected_envelope = envelope if _validated_replay_inputs else envelope.validated()
    selected_inventory = (
        inventory if _validated_replay_inputs else inventory.validated()
    )
    selected_profile = round21_risk_profile(risk_profile)
    selected_scenario = round21_execution_scenario(scenario_name)
    evidence = (
        market_evidence if _validated_replay_inputs else market_evidence.validated()
    )
    decision_time = int(decision_time_ms)
    capital = _decimal(
        risk_capital_quote,
        name="risk capital",
        positive=True,
    )
    available_cash = _decimal(
        available_cash_quote,
        name="available cash",
    )
    condition_start_cash = _decimal(
        condition_start_cash_quote,
        name="condition start cash",
    )
    daily_pnl = _decimal(
        daily_realized_pnl_quote,
        name="daily realized PnL",
    )
    drawdown = _decimal(
        drawdown_capital_fraction,
        name="drawdown fraction",
    )
    minimum_edge = _decimal(
        minimum_edge_per_share,
        name="minimum edge",
        positive=True,
    )
    builder_fee_bps = _decimal(
        builder_taker_fee_bps,
        name="builder taker fee",
    )
    permission_sha = str(directional_entry_permission_sha256 or "").strip().lower()
    if type(directional_entry_allowed) is not bool:
        raise ValueError("Round 21 action context is invalid")
    if permission_sha:
        permission_sha = _digest(
            permission_sha,
            name="directional entry permission",
        )
    elif directional_entry_allowed:
        permission_sha = POLYMARKET_ROUND21_DETERMINISTIC_PERMISSION_SHA256
    else:
        raise ValueError("Round 21 AI veto requires bound permission evidence")
    reconciliation = _digest(
        reconciliation_sha256,
        name="reconciliation",
    )
    try:
        cooldown = int(cooldown_until_ms)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Round 21 action context is invalid") from exc
    if (
        market.asset != "BTC"
        or market.condition_id != selected_inventory.condition_id
        or market.condition_id != evidence.condition_id
        or market.condition_id != selected_envelope.condition_id
        or decision_time != selected_envelope.decision_time_ms
        or not market.event_start_ms <= decision_time < market.end_ms
        or (decision_time - market.event_start_ms) % 250
        or available_cash < 0
        or condition_start_cash < 0
        or not Decimal("0") <= drawdown <= Decimal("1")
        or minimum_edge >= Decimal("1")
        or not Decimal("0")
        <= builder_fee_bps
        <= POLYMARKET_ROUND21_MAXIMUM_BUILDER_TAKER_FEE_BPS
        or cooldown < 0
        or type(transition_pending) is not bool
        or type(reconciliation_ok) is not bool
        or set(books) != {"Up", "Down"}
    ):
        raise ValueError("Round 21 action context is invalid")

    up_book = _fresh_book(
        market,
        books.get("Up"),
        outcome="Up",
        decision_time_ms=decision_time,
        already_validated=_validated_replay_inputs,
    )
    down_book = _fresh_book(
        market,
        books.get("Down"),
        outcome="Down",
        decision_time_ms=decision_time,
        already_validated=_validated_replay_inputs,
    )
    policy_context_sha256 = _canonical_sha256(
        {
            "schema_version": POLYMARKET_ROUND21_ACTION_DECISION_SCHEMA_VERSION,
            "multi_action_policy_sha256": (
                POLYMARKET_ROUND21_MULTI_ACTION_POLICY_SHA256
            ),
            "condition_id": market.condition_id,
            "decision_time_ms": decision_time,
            "market_execution_evidence_sha256": evidence.evidence_sha256,
            "probability_evidence_sha256": (selected_envelope.evidence_sha256),
            "inventory_sha256": selected_inventory.inventory_sha256,
            "risk_profile": selected_profile.name,
            "risk_capital_quote": format(capital, "f"),
            "available_cash_quote": format(available_cash, "f"),
            "condition_start_cash_quote": format(condition_start_cash, "f"),
            "daily_realized_pnl_quote": format(daily_pnl, "f"),
            "drawdown_capital_fraction": format(drawdown, "f"),
            "cooldown_until_ms": cooldown,
            "transition_pending": transition_pending,
            "reconciliation_ok": reconciliation_ok,
            "reconciliation_sha256": reconciliation,
            "minimum_edge_per_share": format(minimum_edge, "f"),
            "scenario_name": selected_scenario.name,
            "builder_taker_fee_bps": format(builder_fee_bps, "f"),
            "ai_veto_design_sha256": POLYMARKET_ROUND21_AI_VETO_DESIGN_SHA256,
            "directional_entry_allowed": directional_entry_allowed,
            "directional_entry_permission_sha256": permission_sha,
            "creation_books": {
                "Up": (
                    None
                    if up_book is None
                    else {
                        "book_identity_sha256": _book_identity_sha256(up_book),
                        "source_payload_sha256": up_book.source_payload_sha256,
                        "received_wall_ms": up_book.received_wall_ms,
                    }
                ),
                "Down": (
                    None
                    if down_book is None
                    else {
                        "book_identity_sha256": _book_identity_sha256(down_book),
                        "source_payload_sha256": (down_book.source_payload_sha256),
                        "received_wall_ms": down_book.received_wall_ms,
                    }
                ),
            },
            "trading_authority": False,
        }
    )

    def abstain(reason: str) -> Round21ActionDecision:
        return _decision(
            condition_id=market.condition_id,
            decision_time_ms=decision_time,
            action="abstain",
            reason=reason,
            edge_per_share=Decimal("0"),
            minimum_edge_per_share=minimum_edge,
            profile=selected_profile,
            envelope=selected_envelope,
            inventory=selected_inventory,
            reconciliation_sha256=reconciliation,
            policy_context_sha256=policy_context_sha256,
            plan=None,
        )

    if not reconciliation_ok:
        return abstain("ownership_reconciliation_failed")
    if selected_inventory.blocking_unknown_state:
        return abstain("unknown_order_or_fill_state")
    if transition_pending:
        return abstain("active_transition_pending")
    if up_book is None or down_book is None:
        return abstain("creation_book_context_unavailable")
    fee_model = market.fee_schedule.fee_model()
    candidates: list[_Candidate] = []
    current_guaranteed_payout = min(
        selected_inventory.quantity("Up"),
        selected_inventory.quantity("Down"),
    )
    current_event_downside = max(
        Decimal("0"),
        condition_start_cash - available_cash - current_guaranteed_payout,
    )
    event_cap = capital * selected_profile.maximum_event_loss_capital_fraction
    daily_cap = capital * selected_profile.maximum_daily_loss_capital_fraction
    drawdown_cap = capital * selected_profile.maximum_drawdown_capital_fraction
    prior_daily_loss = max(Decimal("0"), -daily_pnl)
    prior_drawdown_loss = capital * drawdown
    event_gate = current_event_downside >= event_cap
    daily_gate = prior_daily_loss + current_event_downside >= daily_cap
    drawdown_gate = prior_drawdown_loss + current_event_downside >= drawdown_cap
    cooldown_gate = decision_time < cooldown

    def add_buy(
        *,
        action: str,
        outcome: str,
        fair_lower: Decimal,
        book: PaperBookSnapshot | None,
        priority: int,
        participation: Decimal,
        maximum_quantity: Decimal | None,
        event_loss_budget: Decimal | None,
    ) -> None:
        if book is None or not book.asks:
            return
        limit = _maximum_buy_limit(
            best_ask=book.asks[0].price,
            tick_size=market.tick_size,
            fair_lower=fair_lower,
            minimum_edge=minimum_edge,
            fee_model=fee_model,
            builder_fee_bps=builder_fee_bps,
        )
        if limit is None:
            return
        displayed = _displayed_quantity(
            book,
            side="BUY",
            limit_price=limit,
        )
        quantity = displayed * participation
        if maximum_quantity is not None:
            quantity = min(quantity, maximum_quantity)
        unit_loss = _buy_cost_per_share(limit, fee_model, builder_fee_bps)
        quantity = min(quantity, available_cash / unit_loss)
        if event_loss_budget is not None:
            quantity = min(quantity, event_loss_budget / unit_loss)
        quantity = _floor_quantity(quantity)
        if quantity < market.minimum_order_size:
            return
        loss_cap = available_cash
        if event_loss_budget is not None:
            loss_cap = min(loss_cap, event_loss_budget)
        plan: Round21AggressiveOrderPlan | None = None
        for _attempt in range(3):
            candidate = Round21AggressiveOrderPlan.create(
                market=market,
                market_evidence=evidence,
                scenario_name=selected_scenario.name,
                decision_time_ms=decision_time,
                outcome=outcome,
                side="BUY",
                quantity=quantity,
                limit_price=limit,
                effective_tick_size=market.tick_size,
                builder_taker_fee_bps=builder_fee_bps,
                predictor_evidence_sha256=(selected_envelope.evidence_sha256),
                reconciliation_sha256=reconciliation,
            )
            if candidate.maximum_loss_quote <= loss_cap:
                plan = candidate
                break
            quantity = _floor_quantity(
                quantity * loss_cap / candidate.maximum_loss_quote
            )
            if quantity < market.minimum_order_size:
                return
        if plan is None:
            return
        edge = fair_lower - _buy_cost_per_share(
            limit,
            fee_model,
            builder_fee_bps,
        )
        candidates.append(_Candidate(priority, action, edge, plan))

    for owned_outcome, complement_outcome, book, action in (
        ("Up", "Down", down_book, "lock_up_with_down"),
        ("Down", "Up", up_book, "lock_down_with_up"),
    ):
        owned_quantity = selected_inventory.quantity(owned_outcome)
        complement_quantity = selected_inventory.quantity(complement_outcome)
        unpaired = max(Decimal("0"), owned_quantity - complement_quantity)
        if unpaired <= 0 or owned_quantity <= 0:
            continue
        average_owned_cost = (
            selected_inventory.cost_basis(owned_outcome) / owned_quantity
        )
        add_buy(
            action=action,
            outcome=complement_outcome,
            fair_lower=Decimal("1") - average_owned_cost,
            book=book,
            priority=0,
            participation=Decimal("1"),
            maximum_quantity=unpaired,
            event_loss_budget=None,
        )

    for outcome, book, action in (
        ("Up", up_book, "reduce_up"),
        ("Down", down_book, "reduce_down"),
    ):
        if book is None or not book.bids:
            continue
        lot = next(
            (item for item in selected_inventory.lots if item.outcome == outcome),
            None,
        )
        if lot is None:
            continue
        limit = _minimum_sell_limit(
            best_bid=book.bids[0].price,
            tick_size=market.tick_size,
            fair_upper=selected_envelope.upper(outcome),
            minimum_edge=minimum_edge,
            fee_model=fee_model,
            builder_fee_bps=builder_fee_bps,
        )
        if limit is None:
            continue
        displayed = _displayed_quantity(
            book,
            side="SELL",
            limit_price=limit,
        )
        quantity = _floor_quantity(min(lot.quantity, displayed))
        if quantity < market.minimum_order_size:
            continue
        owned_cost = lot.cost_basis_quote * quantity / lot.quantity
        plan = Round21AggressiveOrderPlan.create(
            market=market,
            market_evidence=evidence,
            scenario_name=selected_scenario.name,
            decision_time_ms=decision_time,
            outcome=outcome,
            side="SELL",
            quantity=quantity,
            limit_price=limit,
            effective_tick_size=market.tick_size,
            builder_taker_fee_bps=builder_fee_bps,
            owned_cost_basis_quote=owned_cost,
            parent_inventory_id=lot.parent_inventory_id,
            predictor_evidence_sha256=selected_envelope.evidence_sha256,
            reconciliation_sha256=reconciliation,
        )
        edge = _sell_proceeds_per_share(
            limit,
            fee_model,
            builder_fee_bps,
        ) - selected_envelope.upper(outcome)
        candidates.append(_Candidate(1, action, edge, plan))

    if not (
        event_gate
        or daily_gate
        or drawdown_gate
        or cooldown_gate
        or not selected_envelope.feature_support_eligible
        or not directional_entry_allowed
    ):
        remaining_event_loss = max(
            Decimal("0"),
            event_cap - current_event_downside,
        )
        remaining_daily_loss = max(
            Decimal("0"),
            daily_cap - prior_daily_loss - current_event_downside,
        )
        remaining_drawdown_loss = max(
            Decimal("0"),
            drawdown_cap - prior_drawdown_loss - current_event_downside,
        )
        remaining_directional_loss = min(
            remaining_event_loss,
            remaining_daily_loss,
            remaining_drawdown_loss,
        )
        for outcome, book, action in (
            ("Up", up_book, "buy_up"),
            ("Down", down_book, "buy_down"),
        ):
            add_buy(
                action=action,
                outcome=outcome,
                fair_lower=selected_envelope.lower(outcome),
                book=book,
                priority=2,
                participation=(selected_profile.maximum_displayed_depth_participation),
                maximum_quantity=None,
                event_loss_budget=remaining_directional_loss,
            )

    risk_rejected_candidate = False
    admissible_candidates: list[_Candidate] = []
    for candidate in candidates:
        maximum_event_downside = _maximum_event_downside_after_plan(
            plan=candidate.plan,
            inventory=selected_inventory,
            available_cash_quote=available_cash,
            condition_start_cash_quote=condition_start_cash,
            fee_model=fee_model,
        )
        increases_downside = maximum_event_downside > current_event_downside
        breaches_cap = (
            maximum_event_downside > event_cap
            or prior_daily_loss + maximum_event_downside > daily_cap
            or prior_drawdown_loss + maximum_event_downside > drawdown_cap
        )
        if increases_downside and (breaches_cap or cooldown_gate):
            risk_rejected_candidate = True
            continue
        admissible_candidates.append(candidate)
    candidates = admissible_candidates

    if not candidates:
        if event_gate:
            return abstain("event_loss_gate_no_risk_reducing_action")
        if daily_gate:
            return abstain("daily_loss_gate_no_positive_reduction")
        if drawdown_gate:
            return abstain("drawdown_gate_no_positive_reduction")
        if cooldown_gate:
            return abstain("cooldown_gate_no_positive_reduction")
        if not selected_envelope.feature_support_eligible:
            return abstain("feature_support_out_of_distribution_no_risk_reduction")
        if not directional_entry_allowed:
            return abstain("ai_veto_no_positive_reduction")
        if risk_rejected_candidate:
            return abstain("candidate_would_breach_risk_cap")
        return abstain("no_positive_after_cost_action")
    selected_candidate = min(
        candidates,
        key=lambda item: (
            item.priority,
            -item.edge_per_share,
            item.action,
            item.plan.plan_sha256,
        ),
    )
    return _decision(
        condition_id=market.condition_id,
        decision_time_ms=decision_time,
        action=selected_candidate.action,
        reason="positive_conservative_after_cost_edge",
        edge_per_share=selected_candidate.edge_per_share,
        minimum_edge_per_share=minimum_edge,
        profile=selected_profile,
        envelope=selected_envelope,
        inventory=selected_inventory,
        reconciliation_sha256=reconciliation,
        policy_context_sha256=policy_context_sha256,
        plan=selected_candidate.plan,
    )


credentials_used = False
account_connected = False
binance_execution_connected = False
paper_trading_authority = False
live_trading_authority = False


__all__ = [
    "POLYMARKET_ROUND21_ACTION_DECISION_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_AI_VETO_DESIGN_SHA256",
    "POLYMARKET_ROUND21_DETERMINISTIC_PERMISSION_SHA256",
    "POLYMARKET_ROUND21_MAXIMUM_CREATION_BOOK_AGE_MS",
    "POLYMARKET_ROUND21_MULTI_ACTION_POLICY_SCHEMA_VERSION",
    "POLYMARKET_ROUND21_MULTI_ACTION_POLICY_SHA256",
    "POLYMARKET_ROUND21_PROBABILITY_ENVELOPE_DESIGN_SHA256",
    "POLYMARKET_ROUND21_RISK_PROFILES",
    "Round21ActionDecision",
    "Round21BotInventory",
    "Round21OwnedLot",
    "Round21ProbabilityEnvelope",
    "Round21RiskProfile",
    "build_round21_probability_envelopes",
    "load_round21_multi_action_policy",
    "round21_risk_profile",
    "select_round21_action",
    "validate_round21_multi_action_policy",
]
