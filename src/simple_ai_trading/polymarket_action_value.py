"""Causal Polymarket action features and exact executable value labels."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from decimal import Decimal
import hashlib
import json
import math
from typing import Sequence

from .assets import SUPPORTED_MAJOR_BASE_ASSETS
from .paper_execution import PaperExecutionResult
from .polymarket import PolymarketFiveMinuteMarket
from .polymarket_features import PolymarketFeatureDataset, PolymarketFeatureRow
from .polymarket_model import (
    POLYMARKET_MODEL_FEATURE_NAMES,
    POLYMARKET_MODEL_RISK_CONTEXT_NAMES,
    build_polymarket_model_features,
    build_polymarket_risk_context,
)
from .polymarket_replay import (
    PolymarketMarketExecutionEvidence,
    PolymarketRecordedBook,
)
from .polymarket_recorder import PolymarketEvidenceStore
from .polymarket_repricing import (
    PolymarketRepricingDecisionExecution,
    PolymarketRepricingExecutionContext,
)


POLYMARKET_ACTION_VALUE_CONTRACT_SHA256 = (
    "c8988fd548cff295800b977d6e6c92c39e9f2867b6c6e4b5f7e3d0b2b96f9800"
)
POLYMARKET_ACTION_FEATURE_SCHEMA_VERSION = "polymarket-causal-action-feature-v1"
POLYMARKET_ACTION_LABEL_SCHEMA_VERSION = "polymarket-causal-action-label-v1"
POLYMARKET_ACTION_DATASET_SCHEMA_VERSION = "polymarket-causal-action-dataset-v1"

_ASSETS = tuple(SUPPORTED_MAJOR_BASE_ASSETS)
_OUTCOMES = ("Up", "Down")
_SIGNED_MODEL_FEATURES = frozenset(
    {
        "direct_distance_from_chainlink_open_bps",
        "direct_chainlink_basis_bps",
        "direct_return_100ms_bps",
        "direct_return_250ms_bps",
        "direct_return_1000ms_bps",
        "direct_return_5000ms_bps",
        "direct_diffusion_market_logit_gap",
        "chainlink_diffusion_market_logit_gap",
        "direct_trade_imbalance_100ms",
        "direct_trade_imbalance_250ms",
        "direct_trade_imbalance_1000ms",
        "direct_trade_imbalance_5000ms",
        "direct_top_imbalance",
    }
)
_MODEL_SWAP_PAIRS = (
    ("up_microprice_deviation_bps", "down_microprice_deviation_bps"),
    ("up_top_imbalance", "down_top_imbalance"),
)
_RISK_SWAP_PAIRS = (
    ("up_book_age_ms", "down_book_age_ms"),
    ("up_bid_depth_3_contracts", "down_bid_depth_3_contracts"),
    ("up_ask_depth_3_contracts", "down_ask_depth_3_contracts"),
)

POLYMARKET_ACTION_FEATURE_NAMES = (
    "remaining_seconds",
    "chosen_distance_from_chainlink_open_bps",
    "chosen_direct_chainlink_basis_bps",
    "chosen_return_100ms_bps",
    "chosen_return_250ms_bps",
    "chosen_return_1000ms_bps",
    "chosen_return_5000ms_bps",
    "direct_realized_volatility_100ms_bps",
    "direct_realized_volatility_1000ms_bps",
    "direct_realized_volatility_5000ms_bps",
    "chosen_direct_diffusion_market_logit_gap",
    "chosen_chainlink_diffusion_market_logit_gap",
    "chosen_direct_trade_imbalance_100ms",
    "chosen_direct_trade_imbalance_250ms",
    "chosen_direct_trade_imbalance_1000ms",
    "chosen_direct_trade_imbalance_5000ms",
    "chosen_direct_top_imbalance",
    "direct_spread_bps",
    "chosen_microprice_deviation_bps",
    "opposite_microprice_deviation_bps",
    "chosen_outcome_top_imbalance",
    "opposite_outcome_top_imbalance",
    "outcome_midpoint_sum_error_bps",
    "executable_ask_pair_premium_bps",
    "executable_bid_pair_discount_bps",
    "asset_is_eth",
    "asset_is_sol",
    "chosen_book_age_ms",
    "opposite_book_age_ms",
    "direct_binance_age_ms",
    "chainlink_source_age_ms",
    "chainlink_arrival_age_ms",
    "chainlink_anchor_gap_ms",
    "chosen_bid_depth_3_contracts",
    "chosen_ask_depth_3_contracts",
    "opposite_bid_depth_3_contracts",
    "opposite_ask_depth_3_contracts",
    "log1p_market_liquidity_quote",
    "log1p_market_volume_quote",
)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _is_sha256(value: object) -> bool:
    text = str(value)
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _float_text(values: Sequence[float]) -> list[str]:
    return [format(float(value), ".17g") for value in values]


@dataclass(frozen=True)
class PolymarketActionValueConfig:
    """Frozen primary execution path from the Round 9 preregistration."""

    decision_cadence_ms: int = 250
    per_leg_submission_latency_ms: int = 500
    holding_period_ms: int = 1_000
    minimum_remaining_market_time_ms: int = 30_000
    maximum_order_creation_book_age_ms: int = 500
    maximum_post_target_execution_observation_delay_ms: int = 500

    def validated(self) -> "PolymarketActionValueConfig":
        if asdict(self) != {
            "decision_cadence_ms": 250,
            "per_leg_submission_latency_ms": 500,
            "holding_period_ms": 1_000,
            "minimum_remaining_market_time_ms": 30_000,
            "maximum_order_creation_book_age_ms": 500,
            "maximum_post_target_execution_observation_delay_ms": 500,
        }:
            raise ValueError("Polymarket action-value config drifted from Round 9")
        return self

    def asdict(self) -> dict[str, int]:
        self.validated()
        return {name: int(value) for name, value in asdict(self).items()}


@dataclass(frozen=True)
class PolymarketActionFeature:
    action_feature_id: str
    source_run_id: str
    source_feature_id: str
    source_input_provenance_sha256: str
    source_label_free_sha256: str
    condition_id: str
    market_id: str
    asset: str
    outcome: str
    token_id: str
    decision_event_id: str
    decision_received_wall_ms: int
    decision_received_monotonic_ns: int
    feature_values: tuple[float, ...]
    action_feature_sha256: str

    def feature_map(self) -> dict[str, float]:
        return dict(
            zip(POLYMARKET_ACTION_FEATURE_NAMES, self.feature_values, strict=True)
        )

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ACTION_FEATURE_SCHEMA_VERSION,
            "contract_sha256": POLYMARKET_ACTION_VALUE_CONTRACT_SHA256,
            "source_run_id": self.source_run_id,
            "source_feature_id": self.source_feature_id,
            "source_input_provenance_sha256": self.source_input_provenance_sha256,
            "source_label_free_sha256": self.source_label_free_sha256,
            "condition_id": self.condition_id,
            "market_id": self.market_id,
            "asset": self.asset,
            "outcome": self.outcome,
            "token_id": self.token_id,
            "decision_event_id": self.decision_event_id,
            "decision_received_wall_ms": self.decision_received_wall_ms,
            "decision_received_monotonic_ns": self.decision_received_monotonic_ns,
            "feature_names": list(POLYMARKET_ACTION_FEATURE_NAMES),
            "feature_values": _float_text(self.feature_values),
        }

    def asdict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "action_feature_id": self.action_feature_id,
            "features": self.feature_map(),
            "action_feature_sha256": self.action_feature_sha256,
        }

    def validated(self) -> "PolymarketActionFeature":
        if (
            not _is_sha256(self.action_feature_id)
            or not _is_sha256(self.source_input_provenance_sha256)
            or not _is_sha256(self.source_label_free_sha256)
            or not _is_sha256(self.action_feature_sha256)
            or self.action_feature_id != self.action_feature_sha256
            or self.action_feature_sha256 != _sha256(self.identity_payload())
            or not self.source_run_id
            or not self.source_feature_id
            or not self.condition_id
            or not self.market_id
            or self.asset not in _ASSETS
            or self.outcome not in _OUTCOMES
            or not self.token_id
            or not self.decision_event_id
            or self.decision_received_wall_ms < 0
            or self.decision_received_monotonic_ns < 0
            or len(self.feature_values) != len(POLYMARKET_ACTION_FEATURE_NAMES)
            or not all(math.isfinite(value) for value in self.feature_values)
        ):
            raise ValueError("Polymarket action feature identity is invalid")
        return self


@dataclass(frozen=True)
class PolymarketActionLabel:
    action_label_id: str
    action_feature_sha256: str
    terminal_reason: str
    category: str
    classifier_eligible: bool
    positive_complete: bool
    condition_blocked: bool
    entry_filled: bool
    exit_filled: bool
    stress_utility_quote: Decimal
    entry_cost_quote: Decimal | None
    exit_proceeds_quote: Decimal | None
    net_quote: Decimal | None
    creation_book_event_id: str
    entry_book_event_id: str
    exit_decision_book_event_id: str
    exit_book_event_id: str
    entry_execution_parameter_sha256: str
    exit_execution_parameter_sha256: str
    execution_evidence_sha256: str
    action_label_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ACTION_LABEL_SCHEMA_VERSION,
            "contract_sha256": POLYMARKET_ACTION_VALUE_CONTRACT_SHA256,
            "action_feature_sha256": self.action_feature_sha256,
            "terminal_reason": self.terminal_reason,
            "category": self.category,
            "classifier_eligible": self.classifier_eligible,
            "positive_complete": self.positive_complete,
            "condition_blocked": self.condition_blocked,
            "entry_filled": self.entry_filled,
            "exit_filled": self.exit_filled,
            "stress_utility_quote": _decimal_text(self.stress_utility_quote),
            "entry_cost_quote": _decimal_text(self.entry_cost_quote),
            "exit_proceeds_quote": _decimal_text(self.exit_proceeds_quote),
            "net_quote": _decimal_text(self.net_quote),
            "creation_book_event_id": self.creation_book_event_id,
            "entry_book_event_id": self.entry_book_event_id,
            "exit_decision_book_event_id": self.exit_decision_book_event_id,
            "exit_book_event_id": self.exit_book_event_id,
            "entry_execution_parameter_sha256": (
                self.entry_execution_parameter_sha256
            ),
            "exit_execution_parameter_sha256": (
                self.exit_execution_parameter_sha256
            ),
            "execution_evidence_sha256": self.execution_evidence_sha256,
        }

    def asdict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "action_label_id": self.action_label_id,
            "action_label_sha256": self.action_label_sha256,
        }

    def validated(self) -> "PolymarketActionLabel":
        categories = {
            "action_unavailable",
            "entry_no_fill",
            "filled_entry_failed_exit",
            "successful_round_trip",
        }
        basic_invalid = (
            not _is_sha256(self.action_label_id)
            or not _is_sha256(self.action_feature_sha256)
            or not _is_sha256(self.execution_evidence_sha256)
            or not _is_sha256(self.action_label_sha256)
            or self.action_label_id != self.action_label_sha256
            or self.action_label_sha256 != _sha256(self.identity_payload())
            or not self.terminal_reason
            or self.category not in categories
            or not self.stress_utility_quote.is_finite()
            or any(
                value is not None and not value.is_finite()
                for value in (
                    self.entry_cost_quote,
                    self.exit_proceeds_quote,
                    self.net_quote,
                )
            )
            or any(
                value and not _is_sha256(value)
                for value in (
                    self.entry_execution_parameter_sha256,
                    self.exit_execution_parameter_sha256,
                )
            )
        )
        unavailable_invalid = self.category == "action_unavailable" and (
            self.classifier_eligible
            or self.positive_complete
            or self.condition_blocked
            or self.entry_filled
            or self.exit_filled
            or self.stress_utility_quote != 0
            or self.entry_cost_quote is not None
            or self.exit_proceeds_quote is not None
            or self.net_quote is not None
        )
        no_fill_invalid = self.category == "entry_no_fill" and (
            not self.classifier_eligible
            or self.positive_complete
            or self.condition_blocked
            or self.entry_filled
            or self.exit_filled
            or self.stress_utility_quote != 0
            or self.entry_cost_quote is not None
            or self.exit_proceeds_quote is not None
            or self.net_quote is not None
        )
        failed_exit_invalid = self.category == "filled_entry_failed_exit" and (
            not self.classifier_eligible
            or self.positive_complete
            or not self.condition_blocked
            or not self.entry_filled
            or self.exit_filled
            or self.entry_cost_quote is None
            or self.entry_cost_quote <= 0
            or self.stress_utility_quote != -self.entry_cost_quote
            or self.exit_proceeds_quote is not None
            or self.net_quote is not None
        )
        complete_invalid = self.category == "successful_round_trip" and (
            self.terminal_reason != "complete_round_trip"
            or not self.classifier_eligible
            or self.condition_blocked
            or not self.entry_filled
            or not self.exit_filled
            or self.entry_cost_quote is None
            or self.exit_proceeds_quote is None
            or self.net_quote is None
            or self.stress_utility_quote != self.net_quote
            or self.positive_complete != (self.net_quote > 0)
        )
        if basic_invalid or unavailable_invalid or no_fill_invalid or failed_exit_invalid or complete_invalid:
            raise ValueError("Polymarket action label identity is invalid")
        return self


@dataclass(frozen=True)
class PolymarketActionValueDataset:
    source_feature_dataset_sha256: str
    source_run_id: str
    config: PolymarketActionValueConfig
    features: tuple[PolymarketActionFeature, ...]
    labels: tuple[PolymarketActionLabel, ...]
    category_counts: dict[str, int]
    terminal_reason_counts: dict[str, int]
    dataset_sha256: str

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": POLYMARKET_ACTION_DATASET_SCHEMA_VERSION,
            "contract_sha256": POLYMARKET_ACTION_VALUE_CONTRACT_SHA256,
            "source_feature_dataset_sha256": self.source_feature_dataset_sha256,
            "source_run_id": self.source_run_id,
            "config": self.config.asdict(),
            "action_feature_sha256": [
                item.action_feature_sha256 for item in self.features
            ],
            "action_label_sha256": [item.action_label_sha256 for item in self.labels],
            "category_counts": dict(sorted(self.category_counts.items())),
            "terminal_reason_counts": dict(
                sorted(self.terminal_reason_counts.items())
            ),
        }

    def summary(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "action_count": len(self.features),
            "classifier_eligible_count": sum(
                item.classifier_eligible for item in self.labels
            ),
            "positive_complete_count": sum(
                item.positive_complete for item in self.labels
            ),
            "dataset_sha256": self.dataset_sha256,
            "training_authority": False,
            "trading_authority": False,
            "profitability_claim": False,
        }

    def validated(self) -> "PolymarketActionValueDataset":
        expected_categories: dict[str, int] = {}
        expected_reasons: dict[str, int] = {}
        for label in self.labels:
            label.validated()
            expected_categories[label.category] = expected_categories.get(label.category, 0) + 1
            expected_reasons[label.terminal_reason] = expected_reasons.get(label.terminal_reason, 0) + 1
        if (
            not _is_sha256(self.source_feature_dataset_sha256)
            or not self.source_run_id
            or len(self.features) != len(self.labels)
            or any(item.validated().source_run_id != self.source_run_id for item in self.features)
            or any(
                feature.action_feature_sha256 != label.action_feature_sha256
                for feature, label in zip(self.features, self.labels, strict=True)
            )
            or len({item.action_feature_sha256 for item in self.features}) != len(self.features)
            or dict(self.category_counts) != expected_categories
            or dict(self.terminal_reason_counts) != expected_reasons
            or not _is_sha256(self.dataset_sha256)
            or self.dataset_sha256 != _sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket action-value dataset identity is invalid")
        return self


@dataclass(frozen=True)
class PolymarketActionValueMaterialization:
    dataset_sha256: str
    status: str
    action_count: int
    classifier_eligible_count: int
    positive_complete_count: int

    def asdict(self) -> dict[str, object]:
        return asdict(self)


def _source_label_free_payload(row: PolymarketFeatureRow) -> dict[str, object]:
    return {
        "schema_version": "polymarket-label-free-feature-source-v1",
        "source_run_id": row.run_id,
        "source_feature_id": row.feature_id,
        "condition_id": row.condition_id,
        "market_id": row.market_id,
        "asset": row.asset,
        "decision_event_id": row.decision_event_id,
        "decision_received_wall_ms": row.decision_received_wall_ms,
        "decision_received_monotonic_ns": row.decision_received_monotonic_ns,
        "feature_values": _float_text(row.feature_values),
        "input_provenance_sha256": row.input_provenance_sha256,
    }


def _oriented_feature_values(
    row: PolymarketFeatureRow,
    outcome: str,
) -> tuple[float, ...]:
    raw = row.feature_map()
    midpoint_total = raw["up_midpoint"] + raw["down_midpoint"]
    if midpoint_total <= 0:
        raise ValueError("Polymarket action feature has invalid outcome midpoint sum")
    baseline_up = raw["up_midpoint"] / midpoint_total
    model_values = dict(
        zip(
            POLYMARKET_MODEL_FEATURE_NAMES,
            build_polymarket_model_features(
                raw,
                row.asset,
                baseline_up_probability=baseline_up,
            ),
            strict=True,
        )
    )
    risk_values = dict(
        zip(
            POLYMARKET_MODEL_RISK_CONTEXT_NAMES,
            build_polymarket_risk_context(raw),
            strict=True,
        )
    )
    if outcome == "Down":
        for name in _SIGNED_MODEL_FEATURES:
            model_values[name] = -model_values[name]
        for left, right in _MODEL_SWAP_PAIRS:
            model_values[left], model_values[right] = (
                model_values[right],
                model_values[left],
            )
        for left, right in _RISK_SWAP_PAIRS:
            risk_values[left], risk_values[right] = risk_values[right], risk_values[left]
    return tuple(model_values.values()) + tuple(risk_values.values())


def build_polymarket_action_feature(
    row: PolymarketFeatureRow,
    market: PolymarketFiveMinuteMarket,
    outcome: str,
) -> PolymarketActionFeature:
    """Create one label-free chosen-outcome feature vector."""

    row.validated()
    if (
        market.condition_id != row.condition_id
        or market.market_id != row.market_id
        or market.asset != row.asset
        or outcome not in _OUTCOMES
    ):
        raise ValueError("Polymarket action feature source identity is invalid")
    values = _oriented_feature_values(row, outcome)
    if len(values) != len(POLYMARKET_ACTION_FEATURE_NAMES):
        raise ValueError("Polymarket action feature width is invalid")
    source_label_free_sha256 = _sha256(_source_label_free_payload(row))
    token_id = market.up_token_id if outcome == "Up" else market.down_token_id
    provisional = PolymarketActionFeature(
        action_feature_id="",
        source_run_id=row.run_id,
        source_feature_id=row.feature_id,
        source_input_provenance_sha256=row.input_provenance_sha256,
        source_label_free_sha256=source_label_free_sha256,
        condition_id=row.condition_id,
        market_id=row.market_id,
        asset=row.asset,
        outcome=outcome,
        token_id=token_id,
        decision_event_id=row.decision_event_id,
        decision_received_wall_ms=row.decision_received_wall_ms,
        decision_received_monotonic_ns=row.decision_received_monotonic_ns,
        feature_values=values,
        action_feature_sha256="",
    )
    digest = _sha256(provisional.identity_payload())
    feature = replace(
        provisional,
        action_feature_id=digest,
        action_feature_sha256=digest,
    )
    return feature.validated()


def _book_payload(book: PolymarketRecordedBook | None) -> object:
    if book is None:
        return None
    snapshot = book.snapshot.validated()
    return {
        "run_id": book.run_id,
        "event_id": book.event_id,
        "event_type": book.event_type,
        "connection_id": book.connection_id,
        "segment_id": book.segment_id,
        "sequence_number": book.sequence_number,
        "sub_index": book.sub_index,
        "condition_id": book.market.condition_id,
        "token_id": book.token_id,
        "outcome": book.outcome,
        "tick_size": format(book.tick_size, "f"),
        "source_time_ms": snapshot.source_time_ms,
        "received_wall_ms": snapshot.received_wall_ms,
        "received_monotonic_ns": snapshot.received_monotonic_ns,
        "source_payload_sha256": snapshot.source_payload_sha256,
        "bids": [
            [format(level.price, "f"), format(level.quantity, "f")]
            for level in snapshot.bids
        ],
        "asks": [
            [format(level.price, "f"), format(level.quantity, "f")]
            for level in snapshot.asks
        ],
    }


def _parameter_payload(
    value: PolymarketMarketExecutionEvidence | None,
) -> object:
    return None if value is None else value.asdict()


def _result_payload(value: PaperExecutionResult | None) -> object:
    if value is None:
        return None
    return {
        "state": value.state,
        "filled_quantity": format(value.filled_quantity, "f"),
        "remaining_quantity": format(value.remaining_quantity, "f"),
        "average_fill_price": format(value.average_fill_price, "f"),
        "fee_quote": format(value.fee_quote, "f"),
        "fills": [
            {
                "price": format(fill.price, "f"),
                "quantity": format(fill.quantity, "f"),
                "fee_quote": format(fill.fee_quote, "f"),
                "liquidity_role": fill.liquidity_role,
            }
            for fill in value.fills
        ],
        "reason": value.reason,
        "source_payload_sha256": value.source_payload_sha256,
    }


def _execution_payload(
    feature: PolymarketActionFeature,
    execution: PolymarketRepricingDecisionExecution | None,
    *,
    unavailable_reason: str = "",
) -> dict[str, object]:
    if execution is None:
        return {
            "schema_version": "polymarket-action-execution-evidence-v1",
            "contract_sha256": POLYMARKET_ACTION_VALUE_CONTRACT_SHA256,
            "action_feature_sha256": feature.action_feature_sha256,
            "terminal_reason": unavailable_reason,
            "decision": {
                "event_id": feature.decision_event_id,
                "received_wall_ms": feature.decision_received_wall_ms,
                "received_monotonic_ns": feature.decision_received_monotonic_ns,
            },
        }
    return {
        "schema_version": "polymarket-action-execution-evidence-v1",
        "contract_sha256": POLYMARKET_ACTION_VALUE_CONTRACT_SHA256,
        "action_feature_sha256": feature.action_feature_sha256,
        "terminal_reason": execution.terminal_reason,
        "decision": {
            "event_id": execution.decision.event_id,
            "condition_id": execution.decision.condition_id,
            "token_id": execution.decision.token_id,
            "outcome": execution.decision.outcome,
            "segment_id": execution.decision.segment_id,
            "received_wall_ms": execution.decision.received_wall_ms,
            "received_monotonic_ns": execution.decision.received_monotonic_ns,
        },
        "creation_book": _book_payload(execution.decision.creation_book),
        "entry_book": _book_payload(execution.entry_book),
        "exit_decision_book": _book_payload(execution.exit_decision_book),
        "exit_book": _book_payload(execution.exit_book),
        "entry_parameter": _parameter_payload(execution.entry_parameter),
        "exit_parameter": _parameter_payload(execution.exit_parameter),
        "entry_result": _result_payload(execution.entry_result),
        "exit_result": _result_payload(execution.exit_result),
        "entry_venue_taker_delay_ms": execution.entry_venue_taker_delay_ms,
        "exit_venue_taker_delay_ms": execution.exit_venue_taker_delay_ms,
        "entry_execution_target_wall_ms": execution.entry_execution_target_wall_ms,
        "exit_decision_target_wall_ms": execution.exit_decision_target_wall_ms,
        "exit_execution_target_wall_ms": execution.exit_execution_target_wall_ms,
        "entry_execution_target_monotonic_ns": (
            execution.entry_execution_target_monotonic_ns
        ),
        "exit_decision_target_monotonic_ns": (
            execution.exit_decision_target_monotonic_ns
        ),
        "exit_execution_target_monotonic_ns": (
            execution.exit_execution_target_monotonic_ns
        ),
        "entry_cost_quote": _decimal_text(execution.entry_cost_quote),
        "exit_proceeds_quote": _decimal_text(execution.exit_proceeds_quote),
        "net_quote": _decimal_text(execution.net_quote),
    }


def build_polymarket_action_label(
    feature: PolymarketActionFeature,
    execution: PolymarketRepricingDecisionExecution | None,
    *,
    unavailable_reason: str = "missing_entry_creation_book",
) -> PolymarketActionLabel:
    """Map one exact execution path to its preregistered stress label."""

    feature.validated()
    if execution is None:
        terminal_reason = str(unavailable_reason).strip()
        if not terminal_reason:
            raise ValueError("Polymarket unavailable action requires a reason")
        category = "action_unavailable"
        classifier_eligible = False
        positive_complete = False
        condition_blocked = False
        stress_utility = Decimal("0")
    else:
        if (
            execution.decision.condition_id != feature.condition_id
            or execution.decision.token_id != feature.token_id
            or execution.decision.outcome != feature.outcome
            or execution.decision.event_id != feature.decision_event_id
            or execution.decision.received_wall_ms
            != feature.decision_received_wall_ms
            or execution.decision.received_monotonic_ns
            != feature.decision_received_monotonic_ns
        ):
            raise ValueError("Polymarket action execution and feature are misaligned")
        terminal_reason = execution.terminal_reason
        if terminal_reason == "complete_round_trip":
            if execution.net_quote is None:
                raise ValueError("complete Polymarket action has no net value")
            category = "successful_round_trip"
            classifier_eligible = True
            positive_complete = execution.net_quote > 0
            condition_blocked = False
            stress_utility = execution.net_quote
        elif execution.entry_filled:
            if execution.entry_cost_quote is None:
                raise ValueError("filled Polymarket entry has no exact cost")
            category = "filled_entry_failed_exit"
            classifier_eligible = True
            positive_complete = False
            condition_blocked = True
            stress_utility = -execution.entry_cost_quote
        elif terminal_reason == "entry_not_filled":
            category = "entry_no_fill"
            classifier_eligible = True
            positive_complete = False
            condition_blocked = False
            stress_utility = Decimal("0")
        else:
            category = "action_unavailable"
            classifier_eligible = False
            positive_complete = False
            condition_blocked = False
            stress_utility = Decimal("0")

    evidence_payload = _execution_payload(
        feature,
        execution,
        unavailable_reason=terminal_reason,
    )
    evidence_sha256 = _sha256(evidence_payload)
    provisional = PolymarketActionLabel(
        action_label_id="",
        action_feature_sha256=feature.action_feature_sha256,
        terminal_reason=terminal_reason,
        category=category,
        classifier_eligible=classifier_eligible,
        positive_complete=positive_complete,
        condition_blocked=condition_blocked,
        entry_filled=False if execution is None else execution.entry_filled,
        exit_filled=False if execution is None else execution.exit_filled,
        stress_utility_quote=stress_utility,
        entry_cost_quote=None if execution is None else execution.entry_cost_quote,
        exit_proceeds_quote=(
            None if execution is None else execution.exit_proceeds_quote
        ),
        net_quote=None if execution is None else execution.net_quote,
        creation_book_event_id=(
            "" if execution is None else execution.decision.creation_book.event_id
        ),
        entry_book_event_id=(
            "" if execution is None or execution.entry_book is None else execution.entry_book.event_id
        ),
        exit_decision_book_event_id=(
            ""
            if execution is None or execution.exit_decision_book is None
            else execution.exit_decision_book.event_id
        ),
        exit_book_event_id=(
            "" if execution is None or execution.exit_book is None else execution.exit_book.event_id
        ),
        entry_execution_parameter_sha256=(
            ""
            if execution is None or execution.entry_parameter is None
            else execution.entry_parameter.snapshot_sha256
        ),
        exit_execution_parameter_sha256=(
            ""
            if execution is None or execution.exit_parameter is None
            else execution.exit_parameter.snapshot_sha256
        ),
        execution_evidence_sha256=evidence_sha256,
        action_label_sha256="",
    )
    digest = _sha256(provisional.identity_payload())
    return replace(
        provisional,
        action_label_id=digest,
        action_label_sha256=digest,
    ).validated()


def build_polymarket_action_value_dataset(
    source: PolymarketFeatureDataset,
    execution_context: PolymarketRepricingExecutionContext,
    *,
    config: PolymarketActionValueConfig | None = None,
) -> PolymarketActionValueDataset:
    """Build both chosen outcomes for every causal decision row."""

    cfg = (config or PolymarketActionValueConfig()).validated()
    if (
        source.dataset_id != source.dataset_sha256
        or not _is_sha256(source.dataset_sha256)
        or source.run_id != execution_context.run_id
        or execution_context.book_sample_interval_ms != 0
        or int(source.config.cadence_ms) != cfg.decision_cadence_ms
    ):
        raise ValueError("Polymarket action-value source contract is invalid")
    markets = execution_context.market_by_condition
    features: list[PolymarketActionFeature] = []
    labels: list[PolymarketActionLabel] = []
    category_counts: dict[str, int] = {}
    terminal_reason_counts: dict[str, int] = {}
    for row in source.rows:
        row.validated()
        market = markets.get(row.condition_id)
        if market is None:
            raise ValueError("Polymarket action feature market is absent from replay")
        for outcome in _OUTCOMES:
            feature = build_polymarket_action_feature(row, market, outcome)
            decision = execution_context.decision_at(
                market,
                event_id=row.decision_event_id,
                received_wall_ms=row.decision_received_wall_ms,
                received_monotonic_ns=row.decision_received_monotonic_ns,
                outcome=outcome,
                maximum_creation_book_age_ms=(
                    cfg.maximum_order_creation_book_age_ms
                ),
            )
            execution = (
                None
                if decision is None
                else execution_context.execute(
                    market,
                    decision,
                    latency_ms=cfg.per_leg_submission_latency_ms,
                    holding_period_ms=cfg.holding_period_ms,
                    minimum_remaining_market_time_ms=(
                        cfg.minimum_remaining_market_time_ms
                    ),
                    maximum_order_creation_book_age_ms=(
                        cfg.maximum_order_creation_book_age_ms
                    ),
                    maximum_post_target_execution_observation_delay_ms=(
                        cfg.maximum_post_target_execution_observation_delay_ms
                    ),
                )
            )
            label = build_polymarket_action_label(feature, execution)
            features.append(feature)
            labels.append(label)
            category_counts[label.category] = category_counts.get(label.category, 0) + 1
            terminal_reason_counts[label.terminal_reason] = (
                terminal_reason_counts.get(label.terminal_reason, 0) + 1
            )
    provisional = PolymarketActionValueDataset(
        source_feature_dataset_sha256=source.dataset_sha256,
        source_run_id=source.run_id,
        config=cfg,
        features=tuple(features),
        labels=tuple(labels),
        category_counts=dict(sorted(category_counts.items())),
        terminal_reason_counts=dict(sorted(terminal_reason_counts.items())),
        dataset_sha256="",
    )
    return replace(
        provisional,
        dataset_sha256=_sha256(provisional.identity_payload()),
    ).validated()


def materialize_polymarket_action_value_dataset(
    store: PolymarketEvidenceStore,
    dataset: PolymarketActionValueDataset,
) -> PolymarketActionValueMaterialization:
    """Persist one immutable action batch with exact source-feature linkage."""

    dataset.validated()
    connection = store.connect()
    source_manifest = connection.execute(
        """
        SELECT run_id, dataset_sha256
        FROM polymarket_feature_dataset
        WHERE dataset_id = ?
        """,
        [dataset.source_feature_dataset_sha256],
    ).fetchone()
    if source_manifest is None:
        raise ValueError("Polymarket action source feature dataset is not materialized")
    if tuple(map(str, source_manifest)) != (
        dataset.source_run_id,
        dataset.source_feature_dataset_sha256,
    ):
        raise ValueError("Polymarket action source feature manifest is inconsistent")
    stored_sources = {
        str(row[0]): tuple(row[1:])
        for row in connection.execute(
            """
            SELECT feature_id, run_id, condition_id, market_id, asset,
                   decision_event_id, decision_received_wall_ms,
                   decision_received_monotonic_ns, feature_values_json,
                   input_provenance_sha256
            FROM polymarket_feature_row
            WHERE dataset_id = ?
            """,
            [dataset.source_feature_dataset_sha256],
        ).fetchall()
    }
    for feature in dataset.features:
        source = stored_sources.get(feature.source_feature_id)
        if source is None:
            raise ValueError("Polymarket action source feature linkage is invalid")
        try:
            source_values = json.loads(str(source[7]))
        except json.JSONDecodeError as exc:
            raise ValueError("stored Polymarket source feature JSON is invalid") from exc
        expected_source = (
            feature.source_run_id,
            feature.condition_id,
            feature.market_id,
            feature.asset,
            feature.decision_event_id,
            feature.decision_received_wall_ms,
            feature.decision_received_monotonic_ns,
            source[7],
            feature.source_input_provenance_sha256,
        )
        source_label_free_sha256 = _sha256(
            {
                "schema_version": "polymarket-label-free-feature-source-v1",
                "source_run_id": str(source[0]),
                "source_feature_id": feature.source_feature_id,
                "condition_id": str(source[1]),
                "market_id": str(source[2]),
                "asset": str(source[3]),
                "decision_event_id": str(source[4]),
                "decision_received_wall_ms": int(source[5]),
                "decision_received_monotonic_ns": int(source[6]),
                "feature_values": source_values,
                "input_provenance_sha256": str(source[8]),
            }
        )
        if source != expected_source or (
            source_label_free_sha256 != feature.source_label_free_sha256
        ):
            raise ValueError("Polymarket action source feature linkage is invalid")

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS polymarket_action_value_dataset (
            dataset_sha256 VARCHAR PRIMARY KEY,
            schema_version VARCHAR NOT NULL,
            contract_sha256 VARCHAR NOT NULL,
            source_feature_dataset_sha256 VARCHAR NOT NULL,
            source_run_id VARCHAR NOT NULL,
            config_json VARCHAR NOT NULL,
            feature_names_json VARCHAR NOT NULL,
            action_count UBIGINT NOT NULL,
            classifier_eligible_count UBIGINT NOT NULL,
            positive_complete_count UBIGINT NOT NULL,
            category_counts_json VARCHAR NOT NULL,
            terminal_reason_counts_json VARCHAR NOT NULL,
            manifest_sha256 VARCHAR NOT NULL
        );

        CREATE TABLE IF NOT EXISTS polymarket_action_value_row (
            dataset_sha256 VARCHAR NOT NULL,
            action_index UBIGINT NOT NULL,
            action_feature_sha256 VARCHAR NOT NULL,
            action_label_sha256 VARCHAR NOT NULL,
            source_feature_id VARCHAR NOT NULL,
            source_input_provenance_sha256 VARCHAR NOT NULL,
            source_label_free_sha256 VARCHAR NOT NULL,
            condition_id VARCHAR NOT NULL,
            market_id VARCHAR NOT NULL,
            asset VARCHAR NOT NULL,
            outcome VARCHAR NOT NULL,
            token_id VARCHAR NOT NULL,
            decision_event_id VARCHAR NOT NULL,
            decision_received_wall_ms BIGINT NOT NULL,
            decision_received_monotonic_ns UBIGINT NOT NULL,
            feature_values_json VARCHAR NOT NULL,
            terminal_reason VARCHAR NOT NULL,
            category VARCHAR NOT NULL,
            classifier_eligible BOOLEAN NOT NULL,
            positive_complete BOOLEAN NOT NULL,
            condition_blocked BOOLEAN NOT NULL,
            entry_filled BOOLEAN NOT NULL,
            exit_filled BOOLEAN NOT NULL,
            stress_utility_quote VARCHAR NOT NULL,
            entry_cost_quote VARCHAR,
            exit_proceeds_quote VARCHAR,
            net_quote VARCHAR,
            creation_book_event_id VARCHAR NOT NULL,
            entry_book_event_id VARCHAR NOT NULL,
            exit_decision_book_event_id VARCHAR NOT NULL,
            exit_book_event_id VARCHAR NOT NULL,
            entry_execution_parameter_sha256 VARCHAR NOT NULL,
            exit_execution_parameter_sha256 VARCHAR NOT NULL,
            execution_evidence_sha256 VARCHAR NOT NULL,
            PRIMARY KEY(dataset_sha256, action_index),
            UNIQUE(dataset_sha256, action_feature_sha256)
        );
        """
    )
    eligible_count = sum(label.classifier_eligible for label in dataset.labels)
    positive_count = sum(label.positive_complete for label in dataset.labels)
    manifest_values = (
        dataset.dataset_sha256,
        POLYMARKET_ACTION_DATASET_SCHEMA_VERSION,
        POLYMARKET_ACTION_VALUE_CONTRACT_SHA256,
        dataset.source_feature_dataset_sha256,
        dataset.source_run_id,
        _canonical_json(dataset.config.asdict()),
        _canonical_json(list(POLYMARKET_ACTION_FEATURE_NAMES)),
        len(dataset.features),
        eligible_count,
        positive_count,
        _canonical_json(dict(sorted(dataset.category_counts.items()))),
        _canonical_json(dict(sorted(dataset.terminal_reason_counts.items()))),
        dataset.dataset_sha256,
    )
    expected_rows = [
        (
            dataset.dataset_sha256,
            index,
            feature.action_feature_sha256,
            label.action_label_sha256,
            feature.source_feature_id,
            feature.source_input_provenance_sha256,
            feature.source_label_free_sha256,
            feature.condition_id,
            feature.market_id,
            feature.asset,
            feature.outcome,
            feature.token_id,
            feature.decision_event_id,
            feature.decision_received_wall_ms,
            feature.decision_received_monotonic_ns,
            _canonical_json(_float_text(feature.feature_values)),
            label.terminal_reason,
            label.category,
            label.classifier_eligible,
            label.positive_complete,
            label.condition_blocked,
            label.entry_filled,
            label.exit_filled,
            _decimal_text(label.stress_utility_quote),
            _decimal_text(label.entry_cost_quote),
            _decimal_text(label.exit_proceeds_quote),
            _decimal_text(label.net_quote),
            label.creation_book_event_id,
            label.entry_book_event_id,
            label.exit_decision_book_event_id,
            label.exit_book_event_id,
            label.entry_execution_parameter_sha256,
            label.exit_execution_parameter_sha256,
            label.execution_evidence_sha256,
        )
        for index, (feature, label) in enumerate(
            zip(dataset.features, dataset.labels, strict=True)
        )
    ]
    existing = connection.execute(
        """
        SELECT dataset_sha256, schema_version, contract_sha256,
               source_feature_dataset_sha256, source_run_id, config_json,
               feature_names_json, action_count, classifier_eligible_count,
               positive_complete_count, category_counts_json,
               terminal_reason_counts_json, manifest_sha256
        FROM polymarket_action_value_dataset
        WHERE dataset_sha256 = ?
        """,
        [dataset.dataset_sha256],
    ).fetchone()
    if existing is not None:
        if tuple(existing) != manifest_values:
            raise ValueError("stored Polymarket action manifest is inconsistent")
        stored_rows = connection.execute(
            """
            SELECT dataset_sha256, action_index, action_feature_sha256,
                   action_label_sha256, source_feature_id,
                   source_input_provenance_sha256, source_label_free_sha256,
                   condition_id, market_id, asset, outcome, token_id,
                   decision_event_id, decision_received_wall_ms,
                   decision_received_monotonic_ns, feature_values_json,
                   terminal_reason, category, classifier_eligible,
                   positive_complete, condition_blocked, entry_filled,
                   exit_filled, stress_utility_quote, entry_cost_quote,
                   exit_proceeds_quote, net_quote, creation_book_event_id,
                   entry_book_event_id, exit_decision_book_event_id,
                   exit_book_event_id, entry_execution_parameter_sha256,
                   exit_execution_parameter_sha256, execution_evidence_sha256
            FROM polymarket_action_value_row
            WHERE dataset_sha256 = ? ORDER BY action_index
            """,
            [dataset.dataset_sha256],
        ).fetchall()
        if [tuple(row) for row in stored_rows] != expected_rows:
            raise ValueError("stored Polymarket action rows are inconsistent")
        status = "existing"
    else:
        connection.execute("BEGIN TRANSACTION")
        try:
            connection.execute(
                """
                INSERT INTO polymarket_action_value_dataset VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                manifest_values,
            )
            if expected_rows:
                placeholders = ", ".join("?" for _ in expected_rows[0])
                connection.executemany(
                    f"INSERT INTO polymarket_action_value_row VALUES ({placeholders})",
                    expected_rows,
                )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        status = "created"
    return PolymarketActionValueMaterialization(
        dataset_sha256=dataset.dataset_sha256,
        status=status,
        action_count=len(dataset.features),
        classifier_eligible_count=eligible_count,
        positive_complete_count=positive_count,
    )


__all__ = [
    "POLYMARKET_ACTION_DATASET_SCHEMA_VERSION",
    "POLYMARKET_ACTION_FEATURE_NAMES",
    "POLYMARKET_ACTION_FEATURE_SCHEMA_VERSION",
    "POLYMARKET_ACTION_LABEL_SCHEMA_VERSION",
    "POLYMARKET_ACTION_VALUE_CONTRACT_SHA256",
    "PolymarketActionFeature",
    "PolymarketActionLabel",
    "PolymarketActionValueConfig",
    "PolymarketActionValueDataset",
    "PolymarketActionValueMaterialization",
    "build_polymarket_action_feature",
    "build_polymarket_action_label",
    "build_polymarket_action_value_dataset",
    "materialize_polymarket_action_value_dataset",
]
