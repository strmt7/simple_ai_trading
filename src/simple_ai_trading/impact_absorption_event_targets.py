"""Prospective executable targets for Round 74 causal event windows.

The engine is pure research infrastructure. It accepts already synchronized
L2 states, uses only marketable book walks, and censors any path that cannot be
represented without interpolation or invented liquidity.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Mapping, Sequence

from .impact_absorption import L2BookState
from .impact_absorption_event_sequence import (
    ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS,
    ROUND74_EVENT_PAYOFF_SIDES,
)
from .impact_absorption_targets import (
    Round73BookWalk,
    Round73MarketQuantityRules,
    walk_round73_book,
)


ROUND74_EVENT_TARGET_SCHEMA_VERSION = "round-074-executable-event-target-v1"
ROUND74_EVENT_TARGET_MAXIMUM_LATENCY_NS = 5_000_000_000
ROUND74_EVENT_TARGET_MAXIMUM_SLIPPAGE_BPS_PER_SIDE = 1_000.0
ROUND74_EVENT_TARGET_MAXIMUM_STATE_LATENESS_NS = 250_000_000
ROUND74_EVENT_TARGET_MAXIMUM_DECISION_STATE_AGE_NS = 250_000_000
ROUND74_EVENT_TARGET_MAXIMUM_PATH_STATE_GAP_NS = 250_000_000
ROUND74_EVENT_TARGET_MINIMUM_ANCHOR_SPACING_NS = 1_000_000_000
ROUND74_EVENT_TARGET_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
ROUND74_EVENT_TARGET_INELIGIBLE_REASONS = frozenset(
    {
        "decision_state_missing",
        "decision_state_late",
        "quantity_ineligible",
        "entry_state_late",
        "entry_capacity",
        "entry_minimum_notional",
        "funding_boundary",
        "path_capacity",
        "path_state_gap",
        "exit_state_late",
        "coverage_end",
    }
)
ReceiptOrderKey = tuple[int, int, int]


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()


def _sha256_digest(value: object, label: str) -> str:
    selected = str(value)
    if len(selected) != 64 or any(
        character not in "0123456789abcdef" for character in selected
    ):
        raise ValueError(f"Round 74 {label} digest is invalid")
    return selected


def _finite_nonnegative(value: object, label: str) -> float:
    selected = float(value)
    if not math.isfinite(selected) or selected < 0.0:
        raise ValueError(f"Round 74 {label} must be finite and nonnegative")
    return selected


@dataclass(frozen=True)
class Round74EventTargetSpec:
    """Hash-bound latency, cost, size, and sampling assumptions."""

    reference_quote_notional: float
    decision_to_entry_latency_ns: int
    taker_fee_bps_by_symbol: tuple[tuple[str, float], ...]
    additional_slippage_bps_per_side: float
    commission_evidence_sha256: str
    latency_evidence_sha256: str
    slippage_evidence_sha256: str
    maximum_state_lateness_ns: int = (
        ROUND74_EVENT_TARGET_MAXIMUM_STATE_LATENESS_NS
    )
    maximum_decision_state_age_ns: int = (
        ROUND74_EVENT_TARGET_MAXIMUM_DECISION_STATE_AGE_NS
    )
    maximum_path_state_gap_ns: int = (
        ROUND74_EVENT_TARGET_MAXIMUM_PATH_STATE_GAP_NS
    )
    minimum_anchor_spacing_ns: int = (
        ROUND74_EVENT_TARGET_MINIMUM_ANCHOR_SPACING_NS
    )
    schema_version: str = ROUND74_EVENT_TARGET_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        reference_quote_notional: float,
        decision_to_entry_latency_ns: int,
        taker_fee_bps_by_symbol: Mapping[str, float],
        additional_slippage_bps_per_side: float,
        commission_evidence_sha256: str,
        latency_evidence_sha256: str,
        slippage_evidence_sha256: str,
        maximum_state_lateness_ns: int = (
            ROUND74_EVENT_TARGET_MAXIMUM_STATE_LATENESS_NS
        ),
        maximum_decision_state_age_ns: int = (
            ROUND74_EVENT_TARGET_MAXIMUM_DECISION_STATE_AGE_NS
        ),
        maximum_path_state_gap_ns: int = (
            ROUND74_EVENT_TARGET_MAXIMUM_PATH_STATE_GAP_NS
        ),
        minimum_anchor_spacing_ns: int = (
            ROUND74_EVENT_TARGET_MINIMUM_ANCHOR_SPACING_NS
        ),
    ) -> Round74EventTargetSpec:
        fees = tuple(
            sorted(
                (str(symbol).strip().upper(), float(value))
                for symbol, value in taker_fee_bps_by_symbol.items()
            )
        )
        return cls(
            reference_quote_notional=float(reference_quote_notional),
            decision_to_entry_latency_ns=int(decision_to_entry_latency_ns),
            taker_fee_bps_by_symbol=fees,
            additional_slippage_bps_per_side=float(
                additional_slippage_bps_per_side
            ),
            commission_evidence_sha256=str(commission_evidence_sha256),
            latency_evidence_sha256=str(latency_evidence_sha256),
            slippage_evidence_sha256=str(slippage_evidence_sha256),
            maximum_state_lateness_ns=int(maximum_state_lateness_ns),
            maximum_decision_state_age_ns=int(
                maximum_decision_state_age_ns
            ),
            maximum_path_state_gap_ns=int(maximum_path_state_gap_ns),
            minimum_anchor_spacing_ns=int(minimum_anchor_spacing_ns),
        )

    def __post_init__(self) -> None:
        reference = float(self.reference_quote_notional)
        if not math.isfinite(reference) or reference <= 0.0:
            raise ValueError("Round 74 target reference notional is invalid")
        latency = int(self.decision_to_entry_latency_ns)
        if latency <= 0 or latency > ROUND74_EVENT_TARGET_MAXIMUM_LATENCY_NS:
            raise ValueError("Round 74 target latency is invalid")
        fees = tuple(self.taker_fee_bps_by_symbol)
        if tuple(symbol for symbol, _fee in fees) != ROUND74_EVENT_TARGET_SYMBOLS:
            raise ValueError("Round 74 target fee universe differs")
        if any(
            not math.isfinite(float(fee)) or not 0.0 <= float(fee) <= 100.0
            for _symbol, fee in fees
        ):
            raise ValueError("Round 74 target fee is invalid")
        slippage = _finite_nonnegative(
            self.additional_slippage_bps_per_side,
            "target additional slippage",
        )
        if slippage > ROUND74_EVENT_TARGET_MAXIMUM_SLIPPAGE_BPS_PER_SIDE:
            raise ValueError("Round 74 target additional slippage is too large")
        _sha256_digest(
            self.commission_evidence_sha256,
            "commission evidence",
        )
        _sha256_digest(self.latency_evidence_sha256, "latency evidence")
        _sha256_digest(self.slippage_evidence_sha256, "slippage evidence")
        if (
            int(self.maximum_state_lateness_ns) < 0
            or int(self.maximum_state_lateness_ns)
            > ROUND74_EVENT_TARGET_MAXIMUM_STATE_LATENESS_NS
            or int(self.maximum_decision_state_age_ns) < 0
            or int(self.maximum_decision_state_age_ns)
            > ROUND74_EVENT_TARGET_MAXIMUM_DECISION_STATE_AGE_NS
            or int(self.maximum_path_state_gap_ns) <= 0
            or int(self.maximum_path_state_gap_ns)
            > ROUND74_EVENT_TARGET_MAXIMUM_PATH_STATE_GAP_NS
            or int(self.minimum_anchor_spacing_ns)
            < ROUND74_EVENT_TARGET_MINIMUM_ANCHOR_SPACING_NS
        ):
            raise ValueError("Round 74 target timing guard differs")
        if self.schema_version != ROUND74_EVENT_TARGET_SCHEMA_VERSION:
            raise ValueError("Round 74 target schema differs")

    @property
    def spec_sha256(self) -> str:
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def fee_bps(self, symbol: str) -> float:
        selected = str(symbol).strip().upper()
        try:
            return float(dict(self.taker_fee_bps_by_symbol)[selected])
        except KeyError as exc:
            raise ValueError("Round 74 target fee symbol differs") from exc

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "reference_quote_notional": self.reference_quote_notional,
            "decision_to_entry_latency_ns": self.decision_to_entry_latency_ns,
            "taker_fee_bps_by_symbol": dict(self.taker_fee_bps_by_symbol),
            "additional_slippage_bps_per_side": (
                self.additional_slippage_bps_per_side
            ),
            "commission_evidence_sha256": self.commission_evidence_sha256,
            "latency_evidence_sha256": self.latency_evidence_sha256,
            "slippage_evidence_sha256": self.slippage_evidence_sha256,
            "maximum_state_lateness_ns": self.maximum_state_lateness_ns,
            "maximum_decision_state_age_ns": (
                self.maximum_decision_state_age_ns
            ),
            "maximum_path_state_gap_ns": self.maximum_path_state_gap_ns,
            "minimum_anchor_spacing_ns": self.minimum_anchor_spacing_ns,
            "horizons_seconds": list(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS),
            "sides": list(ROUND74_EVENT_PAYOFF_SIDES),
            "passive_fill_model": False,
            "marketable_l2_walk_only": True,
        }
        if include_sha256:
            value["spec_sha256"] = _canonical_sha256(value)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> Round74EventTargetSpec:
        payload = dict(value)
        claimed = str(payload.pop("spec_sha256", ""))
        if claimed != _canonical_sha256(payload):
            raise ValueError("Round 74 target spec payload digest differs")
        fees = payload.get("taker_fee_bps_by_symbol")
        if not isinstance(fees, Mapping):
            raise ValueError("Round 74 target fee payload differs")
        selected = cls.create(
            reference_quote_notional=float(
                payload["reference_quote_notional"]
            ),
            decision_to_entry_latency_ns=int(
                payload["decision_to_entry_latency_ns"]
            ),
            taker_fee_bps_by_symbol={
                str(symbol): float(fee) for symbol, fee in fees.items()
            },
            additional_slippage_bps_per_side=float(
                payload["additional_slippage_bps_per_side"]
            ),
            commission_evidence_sha256=str(
                payload["commission_evidence_sha256"]
            ),
            latency_evidence_sha256=str(
                payload["latency_evidence_sha256"]
            ),
            slippage_evidence_sha256=str(
                payload["slippage_evidence_sha256"]
            ),
            maximum_state_lateness_ns=int(
                payload["maximum_state_lateness_ns"]
            ),
            maximum_decision_state_age_ns=int(
                payload["maximum_decision_state_age_ns"]
            ),
            maximum_path_state_gap_ns=int(
                payload["maximum_path_state_gap_ns"]
            ),
            minimum_anchor_spacing_ns=int(
                payload["minimum_anchor_spacing_ns"]
            ),
        )
        if selected.as_dict(include_sha256=False) != payload:
            raise ValueError("Round 74 target spec static policy differs")
        return selected


@dataclass(frozen=True)
class Round74EventTargetAnchor:
    symbol: str
    anchor_index: int
    decision_monotonic_ns: int
    decision_wall_ns: int
    endpoint_frame_index: int
    endpoint_message_index: int
    feature_window_sha256: str

    @property
    def decision_order_key(self) -> ReceiptOrderKey:
        return (
            int(self.decision_monotonic_ns),
            int(self.endpoint_frame_index),
            int(self.endpoint_message_index),
        )

    def validate(self) -> None:
        if self.symbol not in ROUND74_EVENT_TARGET_SYMBOLS:
            raise ValueError("Round 74 target anchor symbol differs")
        if min(
            int(self.anchor_index),
            int(self.decision_monotonic_ns),
            int(self.decision_wall_ns),
            int(self.endpoint_frame_index),
            int(self.endpoint_message_index),
        ) < 0:
            raise ValueError("Round 74 target anchor metadata is negative")
        _sha256_digest(self.feature_window_sha256, "feature window")


@dataclass(frozen=True)
class Round74EventActionPayoff:
    gross_payoff_quote: float
    gross_payoff_bps: float
    commission_quote: float
    additional_slippage_quote: float
    total_cost_quote: float
    net_payoff_quote: float
    net_payoff_bps: float


def round74_event_action_payoff(
    *,
    side: str,
    entry_walk: Round73BookWalk,
    exit_walk: Round73BookWalk,
    taker_fee_bps: float,
    additional_slippage_bps_per_side: float,
) -> Round74EventActionPayoff:
    """Value one marketable round trip from actual walked quote notionals."""

    selected_side = str(side)
    if selected_side not in ROUND74_EVENT_PAYOFF_SIDES:
        raise ValueError("Round 74 payoff side differs")
    if not math.isclose(
        entry_walk.filled_base_quantity,
        exit_walk.filled_base_quantity,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ValueError("Round 74 payoff entry and exit quantities differ")
    fee = _finite_nonnegative(taker_fee_bps, "payoff taker fee")
    slippage = _finite_nonnegative(
        additional_slippage_bps_per_side,
        "payoff additional slippage",
    )
    entry_quote = float(entry_walk.quote_notional)
    exit_quote = float(exit_walk.quote_notional)
    if (
        not math.isfinite(entry_quote)
        or not math.isfinite(exit_quote)
        or entry_quote <= 0.0
        or exit_quote <= 0.0
    ):
        raise ValueError("Round 74 payoff quote notional is invalid")
    gross = (
        exit_quote - entry_quote
        if selected_side == "long"
        else entry_quote - exit_quote
    )
    gross_bps = gross / entry_quote * 10_000.0
    commission = fee / 10_000.0 * (entry_quote + exit_quote)
    residual_slippage = slippage / 10_000.0 * (entry_quote + exit_quote)
    total_cost = commission + residual_slippage
    net = gross - total_cost
    net_bps = net / entry_quote * 10_000.0
    values = (
        gross,
        gross_bps,
        commission,
        residual_slippage,
        total_cost,
        net,
        net_bps,
    )
    if not all(math.isfinite(value) for value in values):
        raise ArithmeticError("Round 74 payoff is nonfinite")
    return Round74EventActionPayoff(
        gross_payoff_quote=gross,
        gross_payoff_bps=gross_bps,
        commission_quote=commission,
        additional_slippage_quote=residual_slippage,
        total_cost_quote=total_cost,
        net_payoff_quote=net,
        net_payoff_bps=net_bps,
    )


@dataclass(frozen=True)
class Round74EventTargetOutcome:
    symbol: str
    anchor_index: int
    horizon_seconds: int
    side: str
    eligible: bool
    ineligible_reason: str
    requested_entry_monotonic_ns: int
    actual_entry_monotonic_ns: int | None
    actual_entry_frame_index: int | None
    actual_entry_message_index: int | None
    requested_exit_monotonic_ns: int | None
    actual_exit_monotonic_ns: int | None
    actual_exit_frame_index: int | None
    actual_exit_message_index: int | None
    base_quantity: float | None
    entry_average_price: float | None
    exit_average_price: float | None
    gross_payoff_bps: float | None
    total_cost_quote: float | None
    net_payoff_bps: float | None
    positive_net_payoff: bool | None
    maximum_adverse_excursion_bps: float | None
    maximum_favorable_excursion_bps: float | None
    adverse_selection: bool | None
    regime_unpredictability: float | None
    maximum_spread_bps: float | None
    minimum_exit_side_capacity_ratio: float | None
    entry_update_id: int | None
    exit_update_id: int | None
    target_spec_sha256: str
    target_context_sha256: str
    feature_window_sha256: str

    @property
    def key(self) -> tuple[str, int, int, str]:
        return (
            self.symbol,
            int(self.anchor_index),
            int(self.horizon_seconds),
            self.side,
        )

    @property
    def outcome_sha256(self) -> str:
        return _canonical_sha256(self.as_dict())

    def validate(self) -> None:
        if self.symbol not in ROUND74_EVENT_TARGET_SYMBOLS:
            raise ValueError("Round 74 outcome symbol differs")
        if self.horizon_seconds not in ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS:
            raise ValueError("Round 74 outcome horizon differs")
        if self.side not in ROUND74_EVENT_PAYOFF_SIDES:
            raise ValueError("Round 74 outcome side differs")
        _sha256_digest(self.target_spec_sha256, "target spec")
        _sha256_digest(self.target_context_sha256, "target context")
        _sha256_digest(self.feature_window_sha256, "feature window")
        if int(self.requested_entry_monotonic_ns) < 0:
            raise ValueError("Round 74 outcome requested entry is negative")
        for label, coordinates in (
            (
                "entry",
                (
                    self.actual_entry_monotonic_ns,
                    self.actual_entry_frame_index,
                    self.actual_entry_message_index,
                ),
            ),
            (
                "exit",
                (
                    self.actual_exit_monotonic_ns,
                    self.actual_exit_frame_index,
                    self.actual_exit_message_index,
                ),
            ),
        ):
            present = tuple(value is not None for value in coordinates)
            if any(present) and not all(present):
                raise ValueError(f"Round 74 outcome {label} order key is partial")
            if all(present) and min(int(value) for value in coordinates) < 0:
                raise ValueError(f"Round 74 outcome {label} order key is negative")
        financial = (
            self.base_quantity,
            self.entry_average_price,
            self.exit_average_price,
            self.gross_payoff_bps,
            self.total_cost_quote,
            self.net_payoff_bps,
            self.maximum_adverse_excursion_bps,
            self.maximum_favorable_excursion_bps,
            self.regime_unpredictability,
            self.maximum_spread_bps,
            self.minimum_exit_side_capacity_ratio,
        )
        if self.eligible:
            if self.ineligible_reason:
                raise ValueError("Round 74 eligible outcome has a failure reason")
            if (
                self.actual_entry_monotonic_ns is None
                or self.actual_entry_frame_index is None
                or self.actual_entry_message_index is None
                or self.requested_exit_monotonic_ns is None
                or self.actual_exit_monotonic_ns is None
                or self.actual_exit_frame_index is None
                or self.actual_exit_message_index is None
                or self.positive_net_payoff is None
                or self.adverse_selection is None
                or self.entry_update_id is None
                or self.exit_update_id is None
                or any(value is None for value in financial)
            ):
                raise ValueError("Round 74 eligible outcome is incomplete")
            if min(
                self.actual_entry_frame_index,
                self.actual_entry_message_index,
                self.actual_exit_frame_index,
                self.actual_exit_message_index,
            ) < 0:
                raise ValueError("Round 74 eligible outcome order key is negative")
            numeric = tuple(float(value) for value in financial if value is not None)
            if not all(math.isfinite(value) for value in numeric):
                raise ValueError("Round 74 eligible outcome is nonfinite")
            if (
                float(self.base_quantity) <= 0.0
                or float(self.entry_average_price) <= 0.0
                or float(self.exit_average_price) <= 0.0
                or float(self.total_cost_quote) < 0.0
                or float(self.maximum_adverse_excursion_bps) < 0.0
                or float(self.maximum_favorable_excursion_bps) < 0.0
                or not 0.0 <= float(self.regime_unpredictability) <= 1.0
                or float(self.minimum_exit_side_capacity_ratio) < 1.0
            ):
                raise ValueError("Round 74 eligible outcome bounds differ")
        else:
            if self.ineligible_reason not in ROUND74_EVENT_TARGET_INELIGIBLE_REASONS:
                raise ValueError("Round 74 ineligible outcome reason differs")
            if any(value is not None for value in financial) or any(
                value is not None
                for value in (
                    self.positive_net_payoff,
                    self.adverse_selection,
                    self.entry_update_id,
                    self.exit_update_id,
                )
            ):
                raise ValueError("Round 74 ineligible outcome has financial values")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": ROUND74_EVENT_TARGET_SCHEMA_VERSION,
            "symbol": self.symbol,
            "anchor_index": self.anchor_index,
            "horizon_seconds": self.horizon_seconds,
            "side": self.side,
            "eligible": self.eligible,
            "ineligible_reason": self.ineligible_reason,
            "requested_entry_monotonic_ns": self.requested_entry_monotonic_ns,
            "actual_entry_monotonic_ns": self.actual_entry_monotonic_ns,
            "actual_entry_frame_index": self.actual_entry_frame_index,
            "actual_entry_message_index": self.actual_entry_message_index,
            "requested_exit_monotonic_ns": self.requested_exit_monotonic_ns,
            "actual_exit_monotonic_ns": self.actual_exit_monotonic_ns,
            "actual_exit_frame_index": self.actual_exit_frame_index,
            "actual_exit_message_index": self.actual_exit_message_index,
            "base_quantity": self.base_quantity,
            "entry_average_price": self.entry_average_price,
            "exit_average_price": self.exit_average_price,
            "gross_payoff_bps": self.gross_payoff_bps,
            "total_cost_quote": self.total_cost_quote,
            "net_payoff_bps": self.net_payoff_bps,
            "positive_net_payoff": self.positive_net_payoff,
            "maximum_adverse_excursion_bps": (
                self.maximum_adverse_excursion_bps
            ),
            "maximum_favorable_excursion_bps": (
                self.maximum_favorable_excursion_bps
            ),
            "adverse_selection": self.adverse_selection,
            "regime_unpredictability": self.regime_unpredictability,
            "maximum_spread_bps": self.maximum_spread_bps,
            "minimum_exit_side_capacity_ratio": (
                self.minimum_exit_side_capacity_ratio
            ),
            "entry_update_id": self.entry_update_id,
            "exit_update_id": self.exit_update_id,
            "target_spec_sha256": self.target_spec_sha256,
            "target_context_sha256": self.target_context_sha256,
            "feature_window_sha256": self.feature_window_sha256,
        }


@dataclass(frozen=True)
class _Decision:
    anchor: Round74EventTargetAnchor
    state_order_key: ReceiptOrderKey
    state: L2BookState
    base_quantity: float | None


@dataclass(frozen=True)
class _PendingEntry:
    decision: _Decision
    requested_entry_monotonic_ns: int


@dataclass
class _ActivePosition:
    identifier: int
    decision: _Decision
    horizon_seconds: int
    side: str
    requested_entry_monotonic_ns: int
    actual_entry_monotonic_ns: int
    actual_entry_frame_index: int
    actual_entry_message_index: int
    requested_exit_monotonic_ns: int
    base_quantity: float
    entry_walk: Round73BookWalk
    entry_mid: float
    prior_mid: float
    prior_state_received_monotonic_ns: int
    total_mid_variation_bps: float
    minimum_net_payoff_bps: float
    maximum_net_payoff_bps: float
    maximum_spread_bps: float
    minimum_exit_side_capacity_ratio: float
    entry_update_id: int


class Round74EventTargetEngine:
    """Create one complete marketable target panel from causal L2 states."""

    def __init__(
        self,
        *,
        spec: Round74EventTargetSpec,
        anchors: Sequence[Round74EventTargetAnchor],
        quantity_rules: Mapping[str, Round73MarketQuantityRules],
        funding_boundaries_monotonic_ns: Mapping[str, Sequence[int]] | None = None,
    ) -> None:
        self.spec = spec
        self.anchors: dict[str, deque[Round74EventTargetAnchor]] = {
            symbol: deque() for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        }
        self._anchor_prior_by_symbol: dict[str, int] = {}
        self._anchor_keys: set[tuple[str, int]] = set()
        self._anchor_count = 0
        self._global_prior_order_key: ReceiptOrderKey | None = None
        self._finished = False
        for anchor in sorted(
            anchors,
            key=lambda item: (
                item.decision_monotonic_ns,
                item.symbol,
                item.anchor_index,
            ),
        ):
            self.add_anchor(anchor)
        rules = {
            str(symbol).upper(): value for symbol, value in quantity_rules.items()
        }
        if tuple(sorted(rules)) != ROUND74_EVENT_TARGET_SYMBOLS or any(
            rules[symbol].symbol != symbol for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ):
            raise ValueError("Round 74 target quantity-rule universe differs")
        validated_rules = {
            symbol: Round73MarketQuantityRules.create(
                symbol=symbol,
                step_size=rules[symbol].step_size,
                minimum_quantity=rules[symbol].minimum_quantity,
                maximum_quantity=rules[symbol].maximum_quantity,
                minimum_notional=rules[symbol].minimum_notional,
            )
            for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        }
        self.quantity_rules = validated_rules
        boundaries = funding_boundaries_monotonic_ns or {}
        self.funding_boundaries = {
            symbol: tuple(sorted(int(value) for value in boundaries.get(symbol, ())))
            for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        }
        if any(
            value < 0
            for rows in self.funding_boundaries.values()
            for value in rows
        ):
            raise ValueError("Round 74 target funding boundary is negative")
        if any(
            len(rows) != len(set(rows))
            for rows in self.funding_boundaries.values()
        ):
            raise ValueError("Round 74 target funding boundary is duplicated")
        self.target_context_sha256 = _canonical_sha256(
            {
                "target_spec_sha256": self.spec.spec_sha256,
                "quantity_rules": {
                    symbol: {
                        "step_size": format(
                            validated_rules[symbol].step_size,
                            "f",
                        ),
                        "minimum_quantity": format(
                            validated_rules[symbol].minimum_quantity,
                            "f",
                        ),
                        "maximum_quantity": format(
                            validated_rules[symbol].maximum_quantity,
                            "f",
                        ),
                        "minimum_notional": format(
                            validated_rules[symbol].minimum_notional,
                            "f",
                        ),
                    }
                    for symbol in ROUND74_EVENT_TARGET_SYMBOLS
                },
                "funding_boundaries_monotonic_ns": {
                    symbol: list(self.funding_boundaries[symbol])
                    for symbol in ROUND74_EVENT_TARGET_SYMBOLS
                },
            }
        )
        self.latest_state: dict[
            str,
            tuple[ReceiptOrderKey, L2BookState],
        ] = {}
        self.pending: dict[str, deque[_PendingEntry]] = {
            symbol: deque() for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        }
        self.active: dict[str, dict[int, _ActivePosition]] = {
            symbol: {} for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        }
        self.outcomes: list[Round74EventTargetOutcome] = []
        self._next_identifier = 0

    @property
    def anchor_count(self) -> int:
        return self._anchor_count

    @staticmethod
    def _walk(
        state: L2BookState,
        *,
        side: str,
        entering: bool,
        base_quantity: float,
    ) -> Round73BookWalk | None:
        buy = (side == "long") == entering
        return walk_round73_book(
            state.ask_levels if buy else state.bid_levels,
            base_quantity=base_quantity,
            ascending_prices=buy,
        )

    def add_anchor(self, anchor: Round74EventTargetAnchor) -> None:
        """Append one causal anchor before any later receipt is observed."""

        if self._finished:
            raise ValueError("Round 74 target engine is already finished")
        anchor.validate()
        key = (anchor.symbol, int(anchor.anchor_index))
        if key in self._anchor_keys:
            raise ValueError("Round 74 target anchor key is duplicated")
        prior = self._anchor_prior_by_symbol.get(anchor.symbol)
        if (
            prior is not None
            and anchor.decision_monotonic_ns - prior
            < self.spec.minimum_anchor_spacing_ns
        ):
            raise ValueError("Round 74 target anchors are oversampled")
        if (
            self._global_prior_order_key is not None
            and anchor.decision_order_key < self._global_prior_order_key
        ):
            raise ValueError("Round 74 target anchor was added after its receipt")
        queue = self.anchors[anchor.symbol]
        if queue and anchor.decision_order_key <= queue[-1].decision_order_key:
            raise ValueError("Round 74 target anchor order did not advance")
        self._anchor_keys.add(key)
        self._anchor_prior_by_symbol[anchor.symbol] = (
            anchor.decision_monotonic_ns
        )
        queue.append(anchor)
        self._anchor_count += 1

    def _record_ineligible(
        self,
        *,
        anchor: Round74EventTargetAnchor,
        horizon_seconds: int,
        side: str,
        reason: str,
        requested_entry_monotonic_ns: int,
        actual_entry_monotonic_ns: int | None = None,
        actual_entry_frame_index: int | None = None,
        actual_entry_message_index: int | None = None,
        requested_exit_monotonic_ns: int | None = None,
        actual_exit_monotonic_ns: int | None = None,
        actual_exit_frame_index: int | None = None,
        actual_exit_message_index: int | None = None,
    ) -> None:
        outcome = Round74EventTargetOutcome(
            symbol=anchor.symbol,
            anchor_index=anchor.anchor_index,
            horizon_seconds=int(horizon_seconds),
            side=side,
            eligible=False,
            ineligible_reason=reason,
            requested_entry_monotonic_ns=int(requested_entry_monotonic_ns),
            actual_entry_monotonic_ns=actual_entry_monotonic_ns,
            actual_entry_frame_index=actual_entry_frame_index,
            actual_entry_message_index=actual_entry_message_index,
            requested_exit_monotonic_ns=requested_exit_monotonic_ns,
            actual_exit_monotonic_ns=actual_exit_monotonic_ns,
            actual_exit_frame_index=actual_exit_frame_index,
            actual_exit_message_index=actual_exit_message_index,
            base_quantity=None,
            entry_average_price=None,
            exit_average_price=None,
            gross_payoff_bps=None,
            total_cost_quote=None,
            net_payoff_bps=None,
            positive_net_payoff=None,
            maximum_adverse_excursion_bps=None,
            maximum_favorable_excursion_bps=None,
            adverse_selection=None,
            regime_unpredictability=None,
            maximum_spread_bps=None,
            minimum_exit_side_capacity_ratio=None,
            entry_update_id=None,
            exit_update_id=None,
            target_spec_sha256=self.spec.spec_sha256,
            target_context_sha256=self.target_context_sha256,
            feature_window_sha256=anchor.feature_window_sha256,
        )
        outcome.validate()
        self.outcomes.append(outcome)

    def _record_anchor_panel_ineligible(
        self,
        anchor: Round74EventTargetAnchor,
        *,
        reason: str,
        requested_entry_monotonic_ns: int,
        actual_entry_monotonic_ns: int | None = None,
        actual_entry_frame_index: int | None = None,
        actual_entry_message_index: int | None = None,
    ) -> None:
        for horizon in ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS:
            for side in ROUND74_EVENT_PAYOFF_SIDES:
                self._record_ineligible(
                    anchor=anchor,
                    horizon_seconds=horizon,
                    side=side,
                    reason=reason,
                    requested_entry_monotonic_ns=requested_entry_monotonic_ns,
                    actual_entry_monotonic_ns=actual_entry_monotonic_ns,
                    actual_entry_frame_index=actual_entry_frame_index,
                    actual_entry_message_index=actual_entry_message_index,
                )

    def _schedule_anchors_before(
        self,
        current_order_key: ReceiptOrderKey,
    ) -> None:
        for symbol in ROUND74_EVENT_TARGET_SYMBOLS:
            queue = self.anchors[symbol]
            while queue and queue[0].decision_order_key < current_order_key:
                anchor = queue.popleft()
                requested = (
                    anchor.decision_monotonic_ns
                    + self.spec.decision_to_entry_latency_ns
                )
                latest = self.latest_state.get(symbol)
                if latest is None:
                    self._record_anchor_panel_ineligible(
                        anchor,
                        reason="decision_state_missing",
                        requested_entry_monotonic_ns=requested,
                    )
                    continue
                state_order_key, state = latest
                if (
                    state_order_key > anchor.decision_order_key
                    or anchor.decision_monotonic_ns - state_order_key[0]
                    > self.spec.maximum_decision_state_age_ns
                ):
                    self._record_anchor_panel_ineligible(
                        anchor,
                        reason="decision_state_late",
                        requested_entry_monotonic_ns=requested,
                    )
                    continue
                quantity = self.quantity_rules[
                    symbol
                ].quantize_reference_quantity(
                    reference_quote_notional=self.spec.reference_quote_notional,
                    decision_mid=state.mid,
                )
                self.pending[symbol].append(
                    _PendingEntry(
                        decision=_Decision(
                            anchor=anchor,
                            state_order_key=state_order_key,
                            state=state,
                            base_quantity=quantity,
                        ),
                        requested_entry_monotonic_ns=requested,
                    )
                )

    def _crosses_funding(
        self,
        symbol: str,
        *,
        entry_ns: int,
        exit_ns: int,
    ) -> bool:
        return any(
            entry_ns < boundary <= exit_ns
            for boundary in self.funding_boundaries[symbol]
        )

    def _fulfill_entries(
        self,
        *,
        symbol: str,
        received_monotonic_ns: int,
        frame_index: int,
        message_index: int,
        state: L2BookState,
    ) -> None:
        queue = self.pending[symbol]
        while queue and queue[0].requested_entry_monotonic_ns <= received_monotonic_ns:
            pending = queue.popleft()
            anchor = pending.decision.anchor
            if (
                received_monotonic_ns - pending.requested_entry_monotonic_ns
                > self.spec.maximum_state_lateness_ns
            ):
                self._record_anchor_panel_ineligible(
                    anchor,
                    reason="entry_state_late",
                    requested_entry_monotonic_ns=(
                        pending.requested_entry_monotonic_ns
                    ),
                    actual_entry_monotonic_ns=received_monotonic_ns,
                    actual_entry_frame_index=frame_index,
                    actual_entry_message_index=message_index,
                )
                continue
            quantity = pending.decision.base_quantity
            if quantity is None:
                self._record_anchor_panel_ineligible(
                    anchor,
                    reason="quantity_ineligible",
                    requested_entry_monotonic_ns=(
                        pending.requested_entry_monotonic_ns
                    ),
                    actual_entry_monotonic_ns=received_monotonic_ns,
                    actual_entry_frame_index=frame_index,
                    actual_entry_message_index=message_index,
                )
                continue
            for side in ROUND74_EVENT_PAYOFF_SIDES:
                entry_walk = self._walk(
                    state,
                    side=side,
                    entering=True,
                    base_quantity=quantity,
                )
                close_walk = self._walk(
                    state,
                    side=side,
                    entering=False,
                    base_quantity=quantity,
                )
                reason = ""
                if entry_walk is None:
                    reason = "entry_capacity"
                elif (
                    entry_walk.quote_notional
                    < float(self.quantity_rules[symbol].minimum_notional)
                ):
                    reason = "entry_minimum_notional"
                elif close_walk is None:
                    reason = "path_capacity"
                for horizon in ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS:
                    requested_exit = received_monotonic_ns + horizon * 1_000_000_000
                    if reason or self._crosses_funding(
                        symbol,
                        entry_ns=received_monotonic_ns,
                        exit_ns=requested_exit,
                    ):
                        self._record_ineligible(
                            anchor=anchor,
                            horizon_seconds=horizon,
                            side=side,
                            reason=reason or "funding_boundary",
                            requested_entry_monotonic_ns=(
                                pending.requested_entry_monotonic_ns
                            ),
                            actual_entry_monotonic_ns=received_monotonic_ns,
                            actual_entry_frame_index=frame_index,
                            actual_entry_message_index=message_index,
                            requested_exit_monotonic_ns=requested_exit,
                        )
                        continue
                    assert entry_walk is not None and close_walk is not None
                    initial = round74_event_action_payoff(
                        side=side,
                        entry_walk=entry_walk,
                        exit_walk=close_walk,
                        taker_fee_bps=self.spec.fee_bps(symbol),
                        additional_slippage_bps_per_side=(
                            self.spec.additional_slippage_bps_per_side
                        ),
                    )
                    identifier = self._next_identifier
                    self._next_identifier += 1
                    self.active[symbol][identifier] = _ActivePosition(
                        identifier=identifier,
                        decision=pending.decision,
                        horizon_seconds=horizon,
                        side=side,
                        requested_entry_monotonic_ns=(
                            pending.requested_entry_monotonic_ns
                        ),
                        actual_entry_monotonic_ns=received_monotonic_ns,
                        actual_entry_frame_index=frame_index,
                        actual_entry_message_index=message_index,
                        requested_exit_monotonic_ns=requested_exit,
                        base_quantity=quantity,
                        entry_walk=entry_walk,
                        entry_mid=state.mid,
                        prior_mid=state.mid,
                        prior_state_received_monotonic_ns=(
                            received_monotonic_ns
                        ),
                        total_mid_variation_bps=0.0,
                        minimum_net_payoff_bps=initial.net_payoff_bps,
                        maximum_net_payoff_bps=initial.net_payoff_bps,
                        maximum_spread_bps=state.spread_bps,
                        minimum_exit_side_capacity_ratio=(
                            close_walk.capacity_ratio
                        ),
                        entry_update_id=state.update_id,
                    )

    def _complete_position(
        self,
        position: _ActivePosition,
        *,
        received_monotonic_ns: int,
        frame_index: int,
        message_index: int,
        state: L2BookState,
        exit_walk: Round73BookWalk,
        payoff: Round74EventActionPayoff,
    ) -> None:
        lateness = received_monotonic_ns - position.requested_exit_monotonic_ns
        if lateness > self.spec.maximum_state_lateness_ns:
            self._record_ineligible(
                anchor=position.decision.anchor,
                horizon_seconds=position.horizon_seconds,
                side=position.side,
                reason="exit_state_late",
                requested_entry_monotonic_ns=(
                    position.requested_entry_monotonic_ns
                ),
                actual_entry_monotonic_ns=position.actual_entry_monotonic_ns,
                actual_entry_frame_index=position.actual_entry_frame_index,
                actual_entry_message_index=position.actual_entry_message_index,
                requested_exit_monotonic_ns=(
                    position.requested_exit_monotonic_ns
                ),
                actual_exit_monotonic_ns=received_monotonic_ns,
                actual_exit_frame_index=frame_index,
                actual_exit_message_index=message_index,
            )
            return
        directional_move = abs(
            math.log(state.mid / position.entry_mid) * 10_000.0
        )
        unpredictability = (
            1.0
            if position.total_mid_variation_bps <= 1e-12
            else 1.0
            - min(
                1.0,
                directional_move / position.total_mid_variation_bps,
            )
        )
        outcome = Round74EventTargetOutcome(
            symbol=position.decision.anchor.symbol,
            anchor_index=position.decision.anchor.anchor_index,
            horizon_seconds=position.horizon_seconds,
            side=position.side,
            eligible=True,
            ineligible_reason="",
            requested_entry_monotonic_ns=(
                position.requested_entry_monotonic_ns
            ),
            actual_entry_monotonic_ns=position.actual_entry_monotonic_ns,
            actual_entry_frame_index=position.actual_entry_frame_index,
            actual_entry_message_index=position.actual_entry_message_index,
            requested_exit_monotonic_ns=position.requested_exit_monotonic_ns,
            actual_exit_monotonic_ns=received_monotonic_ns,
            actual_exit_frame_index=frame_index,
            actual_exit_message_index=message_index,
            base_quantity=position.base_quantity,
            entry_average_price=position.entry_walk.average_price,
            exit_average_price=exit_walk.average_price,
            gross_payoff_bps=payoff.gross_payoff_bps,
            total_cost_quote=payoff.total_cost_quote,
            net_payoff_bps=payoff.net_payoff_bps,
            positive_net_payoff=payoff.net_payoff_bps > 0.0,
            maximum_adverse_excursion_bps=max(
                0.0,
                -position.minimum_net_payoff_bps,
            ),
            maximum_favorable_excursion_bps=max(
                0.0,
                position.maximum_net_payoff_bps,
            ),
            adverse_selection=payoff.gross_payoff_bps < 0.0,
            regime_unpredictability=unpredictability,
            maximum_spread_bps=position.maximum_spread_bps,
            minimum_exit_side_capacity_ratio=(
                position.minimum_exit_side_capacity_ratio
            ),
            entry_update_id=position.entry_update_id,
            exit_update_id=state.update_id,
            target_spec_sha256=self.spec.spec_sha256,
            target_context_sha256=self.target_context_sha256,
            feature_window_sha256=(
                position.decision.anchor.feature_window_sha256
            ),
        )
        outcome.validate()
        self.outcomes.append(outcome)

    def _update_active(
        self,
        *,
        symbol: str,
        received_monotonic_ns: int,
        frame_index: int,
        message_index: int,
        state: L2BookState,
    ) -> None:
        walk_cache: dict[tuple[str, float], Round73BookWalk | None] = {}
        completed: list[int] = []
        for identifier, position in tuple(self.active[symbol].items()):
            if (
                received_monotonic_ns
                - position.prior_state_received_monotonic_ns
                > self.spec.maximum_path_state_gap_ns
            ):
                self._record_ineligible(
                    anchor=position.decision.anchor,
                    horizon_seconds=position.horizon_seconds,
                    side=position.side,
                    reason="path_state_gap",
                    requested_entry_monotonic_ns=(
                        position.requested_entry_monotonic_ns
                    ),
                    actual_entry_monotonic_ns=(
                        position.actual_entry_monotonic_ns
                    ),
                    actual_entry_frame_index=(
                        position.actual_entry_frame_index
                    ),
                    actual_entry_message_index=(
                        position.actual_entry_message_index
                    ),
                    requested_exit_monotonic_ns=(
                        position.requested_exit_monotonic_ns
                    ),
                )
                completed.append(identifier)
                continue
            key = (position.side, position.base_quantity)
            if key not in walk_cache:
                walk_cache[key] = self._walk(
                    state,
                    side=position.side,
                    entering=False,
                    base_quantity=position.base_quantity,
                )
            exit_walk = walk_cache[key]
            if exit_walk is None:
                self._record_ineligible(
                    anchor=position.decision.anchor,
                    horizon_seconds=position.horizon_seconds,
                    side=position.side,
                    reason="path_capacity",
                    requested_entry_monotonic_ns=(
                        position.requested_entry_monotonic_ns
                    ),
                    actual_entry_monotonic_ns=(
                        position.actual_entry_monotonic_ns
                    ),
                    actual_entry_frame_index=(
                        position.actual_entry_frame_index
                    ),
                    actual_entry_message_index=(
                        position.actual_entry_message_index
                    ),
                    requested_exit_monotonic_ns=(
                        position.requested_exit_monotonic_ns
                    ),
                )
                completed.append(identifier)
                continue
            payoff = round74_event_action_payoff(
                side=position.side,
                entry_walk=position.entry_walk,
                exit_walk=exit_walk,
                taker_fee_bps=self.spec.fee_bps(symbol),
                additional_slippage_bps_per_side=(
                    self.spec.additional_slippage_bps_per_side
                ),
            )
            position.minimum_net_payoff_bps = min(
                position.minimum_net_payoff_bps,
                payoff.net_payoff_bps,
            )
            position.maximum_net_payoff_bps = max(
                position.maximum_net_payoff_bps,
                payoff.net_payoff_bps,
            )
            position.maximum_spread_bps = max(
                position.maximum_spread_bps,
                state.spread_bps,
            )
            position.minimum_exit_side_capacity_ratio = min(
                position.minimum_exit_side_capacity_ratio,
                exit_walk.capacity_ratio,
            )
            position.total_mid_variation_bps += abs(
                math.log(state.mid / position.prior_mid) * 10_000.0
            )
            position.prior_mid = state.mid
            position.prior_state_received_monotonic_ns = (
                received_monotonic_ns
            )
            if received_monotonic_ns >= position.requested_exit_monotonic_ns:
                self._complete_position(
                    position,
                    received_monotonic_ns=received_monotonic_ns,
                    frame_index=frame_index,
                    message_index=message_index,
                    state=state,
                    exit_walk=exit_walk,
                    payoff=payoff,
                )
                completed.append(identifier)
        for identifier in completed:
            self.active[symbol].pop(identifier, None)

    def observe_depth(
        self,
        *,
        received_monotonic_ns: int,
        frame_index: int,
        message_index: int,
        state: L2BookState,
    ) -> None:
        if self._finished:
            raise ValueError("Round 74 target engine is already finished")
        received = int(received_monotonic_ns)
        order_key = (received, int(frame_index), int(message_index))
        if min(order_key) < 0:
            raise ValueError("Round 74 target receipt order key is negative")
        if (
            self._global_prior_order_key is not None
            and order_key <= self._global_prior_order_key
        ):
            raise ValueError("Round 74 target global receipt order regressed")
        self._global_prior_order_key = order_key
        if state.symbol not in ROUND74_EVENT_TARGET_SYMBOLS:
            raise ValueError("Round 74 target depth symbol differs")
        prior = self.latest_state.get(state.symbol)
        if prior is not None and order_key <= prior[0]:
            raise ValueError("Round 74 target symbol depth order did not advance")
        self._schedule_anchors_before(order_key)
        self.latest_state[state.symbol] = (order_key, state)
        self._update_active(
            symbol=state.symbol,
            received_monotonic_ns=received,
            frame_index=int(frame_index),
            message_index=int(message_index),
            state=state,
        )
        self._fulfill_entries(
            symbol=state.symbol,
            received_monotonic_ns=received,
            frame_index=int(frame_index),
            message_index=int(message_index),
            state=state,
        )

    def finish(self) -> tuple[Round74EventTargetOutcome, ...]:
        if self._finished:
            return tuple(sorted(self.outcomes, key=lambda outcome: outcome.key))
        for symbol in ROUND74_EVENT_TARGET_SYMBOLS:
            while self.anchors[symbol]:
                anchor = self.anchors[symbol].popleft()
                self._record_anchor_panel_ineligible(
                    anchor,
                    reason="coverage_end",
                    requested_entry_monotonic_ns=(
                        anchor.decision_monotonic_ns
                        + self.spec.decision_to_entry_latency_ns
                    ),
                )
            while self.pending[symbol]:
                pending = self.pending[symbol].popleft()
                self._record_anchor_panel_ineligible(
                    pending.decision.anchor,
                    reason="coverage_end",
                    requested_entry_monotonic_ns=(
                        pending.requested_entry_monotonic_ns
                    ),
                )
            for position in tuple(self.active[symbol].values()):
                self._record_ineligible(
                    anchor=position.decision.anchor,
                    horizon_seconds=position.horizon_seconds,
                    side=position.side,
                    reason="coverage_end",
                    requested_entry_monotonic_ns=(
                        position.requested_entry_monotonic_ns
                    ),
                    actual_entry_monotonic_ns=(
                        position.actual_entry_monotonic_ns
                    ),
                    actual_entry_frame_index=(
                        position.actual_entry_frame_index
                    ),
                    actual_entry_message_index=(
                        position.actual_entry_message_index
                    ),
                    requested_exit_monotonic_ns=(
                        position.requested_exit_monotonic_ns
                    ),
                )
            self.active[symbol].clear()
        expected = (
            self._anchor_count
            * len(ROUND74_EVENT_PAYOFF_HORIZONS_SECONDS)
            * len(ROUND74_EVENT_PAYOFF_SIDES)
        )
        if len(self.outcomes) != expected:
            raise ValueError(
                "Round 74 target outcome count differs: "
                f"expected={expected} actual={len(self.outcomes)}"
            )
        ordered = sorted(self.outcomes, key=lambda outcome: outcome.key)
        keys = [outcome.key for outcome in ordered]
        if len(keys) != len(set(keys)):
            raise ValueError("Round 74 target outcome keys are duplicated")
        for outcome in ordered:
            outcome.validate()
        self._finished = True
        return tuple(ordered)


__all__ = [
    "ROUND74_EVENT_TARGET_INELIGIBLE_REASONS",
    "ROUND74_EVENT_TARGET_MAXIMUM_DECISION_STATE_AGE_NS",
    "ROUND74_EVENT_TARGET_MAXIMUM_LATENCY_NS",
    "ROUND74_EVENT_TARGET_MAXIMUM_SLIPPAGE_BPS_PER_SIDE",
    "ROUND74_EVENT_TARGET_MAXIMUM_STATE_LATENESS_NS",
    "ROUND74_EVENT_TARGET_MINIMUM_ANCHOR_SPACING_NS",
    "ROUND74_EVENT_TARGET_SCHEMA_VERSION",
    "ROUND74_EVENT_TARGET_SYMBOLS",
    "Round74EventActionPayoff",
    "Round74EventTargetAnchor",
    "Round74EventTargetEngine",
    "Round74EventTargetOutcome",
    "Round74EventTargetSpec",
    "round74_event_action_payoff",
]
