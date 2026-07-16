"""Noncausal executable ceiling for fast Polymarket repricing research."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass, replace
from decimal import Decimal
import hashlib
from heapq import merge
import json
from statistics import median
from typing import Iterable, Mapping, Sequence

from .assets import SUPPORTED_MAJOR_BASE_ASSETS
from .paper_execution import (
    PaperOrderIntent,
    PaperExecutionResult,
    paper_intent_id,
    simulate_aggressive_order,
)
from .polymarket import POLYMARKET_TAKER_ORDER_DELAY_MS, PolymarketFiveMinuteMarket
from .polymarket_replay import (
    PolymarketEvidenceReplay,
    PolymarketMarketExecutionEvidence,
    PolymarketRecordedBook,
)


POLYMARKET_REPRICING_REPORT_SCHEMA_VERSION = (
    "polymarket-executable-repricing-ceiling-report-v1"
)
POLYMARKET_REPRICING_CONTRACT_SHA256 = (
    "54a4101f2d9e1ae7bc8825fa023d88ee999282b9681d2e09350c0624db2fc9c7"
)
_ASSETS = tuple(SUPPORTED_MAJOR_BASE_ASSETS)
_LATENCIES_MS = (100, 250, 500, 1_000)
_HOLDING_PERIODS_MS = (250, 500, 1_000, 2_000, 5_000)
_MAX_ORDER_CREATION_BOOK_AGE_MS = 500
_MAX_POST_TARGET_EXECUTION_OBSERVATION_DELAY_MS = 500
_TERMINAL_REASONS = (
    "complete_round_trip",
    "entry_confirmation_enters_excluded_close_window",
    "entry_not_filled",
    "entry_enters_excluded_close_window",
    "entry_tick_drift",
    "exit_not_filled",
    "exit_enters_excluded_close_window",
    "exit_tick_drift",
    "missing_entry_execution_book",
    "missing_entry_execution_parameters",
    "missing_exit_decision_book",
    "missing_exit_execution_book",
    "missing_exit_execution_parameters",
    "unsupported_entry_minimum_order_age",
    "unsupported_exit_minimum_order_age",
)


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


def _decimal_text(value: Decimal | None) -> str | None:
    return None if value is None else format(value, "f")


def _replay_book_order_key(
    item: PolymarketRecordedBook,
) -> tuple[int, int, str, int, int, str]:
    return (
        item.received_monotonic_ns,
        item.received_wall_ms,
        item.connection_id,
        item.sequence_number,
        item.sub_index,
        item.token_id,
    )


@dataclass(frozen=True)
class PolymarketRepricingConfig:
    """Immutable Round 8 grid; changing it requires another contract."""

    minimum_decision_spacing_ms: int = 250
    holding_periods_ms: tuple[int, ...] = _HOLDING_PERIODS_MS
    minimum_remaining_market_time_ms: int = 30_000
    maximum_order_creation_book_age_ms: int = _MAX_ORDER_CREATION_BOOK_AGE_MS
    maximum_post_target_execution_observation_delay_ms: int = (
        _MAX_POST_TARGET_EXECUTION_OBSERVATION_DELAY_MS
    )
    per_leg_submission_latencies_ms: tuple[int, ...] = _LATENCIES_MS
    primary_holding_period_ms: int = 1_000
    primary_per_leg_submission_latency_ms: int = 500
    minimum_complete_markets_per_asset: int = 30
    minimum_positive_markets_per_asset: int = 10
    minimum_positive_market_fraction_per_asset: Decimal = Decimal("0.1")
    minimum_positive_markets_per_outcome_asset: int = 5

    def validated(self) -> "PolymarketRepricingConfig":
        if (
            int(self.minimum_decision_spacing_ms) != 250
            or tuple(self.holding_periods_ms) != _HOLDING_PERIODS_MS
            or int(self.minimum_remaining_market_time_ms) != 30_000
            or int(self.maximum_order_creation_book_age_ms)
            != _MAX_ORDER_CREATION_BOOK_AGE_MS
            or int(self.maximum_post_target_execution_observation_delay_ms)
            != _MAX_POST_TARGET_EXECUTION_OBSERVATION_DELAY_MS
            or tuple(self.per_leg_submission_latencies_ms) != _LATENCIES_MS
            or int(self.primary_holding_period_ms) != 1_000
            or int(self.primary_per_leg_submission_latency_ms) != 500
            or int(self.minimum_complete_markets_per_asset) != 30
            or int(self.minimum_positive_markets_per_asset) != 10
            or self.minimum_positive_market_fraction_per_asset
            != Decimal("0.1")
            or int(self.minimum_positive_markets_per_outcome_asset) != 5
        ):
            raise ValueError(
                "Polymarket repricing configuration differs from the frozen contract"
            )
        return self

    def asdict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["holding_periods_ms"] = list(self.holding_periods_ms)
        payload["per_leg_submission_latencies_ms"] = list(
            self.per_leg_submission_latencies_ms
        )
        payload["minimum_positive_market_fraction_per_asset"] = _decimal_text(
            self.minimum_positive_market_fraction_per_asset
        )
        return payload


@dataclass(frozen=True)
class PolymarketRepricingDecision:
    """One causal decision clock and its latest known chosen-token book."""

    event_id: str
    condition_id: str
    token_id: str
    outcome: str
    segment_id: str
    received_wall_ms: int
    received_monotonic_ns: int
    creation_book: PolymarketRecordedBook

    @classmethod
    def from_book(
        cls, book: PolymarketRecordedBook
    ) -> "PolymarketRepricingDecision":
        return cls(
            event_id=book.event_id,
            condition_id=book.market.condition_id,
            token_id=book.token_id,
            outcome=book.outcome,
            segment_id=book.segment_id,
            received_wall_ms=book.received_wall_ms,
            received_monotonic_ns=book.received_monotonic_ns,
            creation_book=book,
        )

    def validated(self, *, maximum_creation_book_age_ms: int) -> "PolymarketRepricingDecision":
        book = self.creation_book
        age_ns = self.received_monotonic_ns - book.received_monotonic_ns
        if (
            not self.event_id
            or not self.condition_id
            or not self.token_id
            or self.outcome not in {"Up", "Down"}
            or not self.segment_id
            or self.received_wall_ms < 0
            or self.received_monotonic_ns < 0
            or book.market.condition_id != self.condition_id
            or book.token_id != self.token_id
            or book.outcome != self.outcome
            or book.segment_id != self.segment_id
            or age_ns < 0
            or age_ns > int(maximum_creation_book_age_ms) * 1_000_000
        ):
            raise ValueError("Polymarket repricing decision is invalid")
        return self


@dataclass(frozen=True)
class PolymarketRepricingDecisionExecution:
    """Exact shared two-leg execution result for one decision and outcome."""

    terminal_reason: str
    decision: PolymarketRepricingDecision
    entry_book: PolymarketRecordedBook | None = None
    exit_decision_book: PolymarketRecordedBook | None = None
    exit_book: PolymarketRecordedBook | None = None
    entry_parameter: PolymarketMarketExecutionEvidence | None = None
    exit_parameter: PolymarketMarketExecutionEvidence | None = None
    entry_result: PaperExecutionResult | None = None
    exit_result: PaperExecutionResult | None = None
    entry_venue_taker_delay_ms: int | None = None
    exit_venue_taker_delay_ms: int | None = None
    entry_execution_target_wall_ms: int | None = None
    exit_decision_target_wall_ms: int | None = None
    exit_execution_target_wall_ms: int | None = None
    entry_execution_target_monotonic_ns: int | None = None
    exit_decision_target_monotonic_ns: int | None = None
    exit_execution_target_monotonic_ns: int | None = None
    entry_cost_quote: Decimal | None = None
    exit_proceeds_quote: Decimal | None = None
    net_quote: Decimal | None = None

    @property
    def entry_filled(self) -> bool:
        return self.entry_result is not None and self.entry_result.state == "FILLED"

    @property
    def exit_filled(self) -> bool:
        return (
            self.entry_result is not None
            and self.exit_result is not None
            and self.exit_result.state == "FILLED"
            and self.exit_result.filled_quantity
            == self.entry_result.filled_quantity
        )


@dataclass(frozen=True)
class PolymarketRepricingOpportunity:
    condition_id: str
    market_id: str
    asset: str
    outcome: str
    token_id: str
    per_leg_submission_latency_ms: int
    holding_period_ms: int
    decision_count: int
    complete_round_trip_count: int
    terminal_reason_counts: Mapping[str, int]
    best_decision_event_id: str
    best_entry_event_id: str
    best_exit_event_id: str
    best_exit_decision_event_id: str
    best_decision_received_wall_ms: int | None
    best_entry_received_wall_ms: int | None
    best_exit_received_wall_ms: int | None
    best_exit_decision_received_wall_ms: int | None
    best_decision_received_monotonic_ns: int | None
    best_entry_received_monotonic_ns: int | None
    best_exit_received_monotonic_ns: int | None
    best_exit_decision_received_monotonic_ns: int | None
    best_entry_execution_target_wall_ms: int | None
    best_exit_decision_target_wall_ms: int | None
    best_exit_execution_target_wall_ms: int | None
    best_entry_execution_target_monotonic_ns: int | None
    best_exit_decision_target_monotonic_ns: int | None
    best_exit_execution_target_monotonic_ns: int | None
    best_entry_venue_taker_delay_ms: int | None
    best_exit_venue_taker_delay_ms: int | None
    quantity: Decimal
    best_entry_cost_quote: Decimal | None
    best_exit_proceeds_quote: Decimal | None
    best_net_quote: Decimal | None
    best_net_bps_on_entry_cost: Decimal | None
    opportunity_sha256: str

    @property
    def positive(self) -> bool:
        return self.best_net_quote is not None and self.best_net_quote > 0

    def identity_payload(self) -> dict[str, object]:
        return {
            "condition_id": self.condition_id,
            "market_id": self.market_id,
            "asset": self.asset,
            "outcome": self.outcome,
            "token_id": self.token_id,
            "per_leg_submission_latency_ms": self.per_leg_submission_latency_ms,
            "holding_period_ms": self.holding_period_ms,
            "decision_count": self.decision_count,
            "complete_round_trip_count": self.complete_round_trip_count,
            "terminal_reason_counts": dict(sorted(self.terminal_reason_counts.items())),
            "best_decision_event_id": self.best_decision_event_id,
            "best_entry_event_id": self.best_entry_event_id,
            "best_exit_event_id": self.best_exit_event_id,
            "best_exit_decision_event_id": self.best_exit_decision_event_id,
            "best_decision_received_wall_ms": (
                self.best_decision_received_wall_ms
            ),
            "best_entry_received_wall_ms": self.best_entry_received_wall_ms,
            "best_exit_received_wall_ms": self.best_exit_received_wall_ms,
            "best_exit_decision_received_wall_ms": (
                self.best_exit_decision_received_wall_ms
            ),
            "best_decision_received_monotonic_ns": (
                self.best_decision_received_monotonic_ns
            ),
            "best_entry_received_monotonic_ns": (
                self.best_entry_received_monotonic_ns
            ),
            "best_exit_received_monotonic_ns": (
                self.best_exit_received_monotonic_ns
            ),
            "best_exit_decision_received_monotonic_ns": (
                self.best_exit_decision_received_monotonic_ns
            ),
            "best_entry_execution_target_wall_ms": (
                self.best_entry_execution_target_wall_ms
            ),
            "best_exit_decision_target_wall_ms": (
                self.best_exit_decision_target_wall_ms
            ),
            "best_exit_execution_target_wall_ms": (
                self.best_exit_execution_target_wall_ms
            ),
            "best_entry_execution_target_monotonic_ns": (
                self.best_entry_execution_target_monotonic_ns
            ),
            "best_exit_decision_target_monotonic_ns": (
                self.best_exit_decision_target_monotonic_ns
            ),
            "best_exit_execution_target_monotonic_ns": (
                self.best_exit_execution_target_monotonic_ns
            ),
            "best_entry_venue_taker_delay_ms": (
                self.best_entry_venue_taker_delay_ms
            ),
            "best_exit_venue_taker_delay_ms": self.best_exit_venue_taker_delay_ms,
            "quantity": _decimal_text(self.quantity),
            "best_entry_cost_quote": _decimal_text(self.best_entry_cost_quote),
            "best_exit_proceeds_quote": _decimal_text(
                self.best_exit_proceeds_quote
            ),
            "best_net_quote": _decimal_text(self.best_net_quote),
            "best_net_bps_on_entry_cost": _decimal_text(
                self.best_net_bps_on_entry_cost
            ),
        }

    def asdict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "positive": self.positive,
            "opportunity_sha256": self.opportunity_sha256,
        }

    def validated(self) -> "PolymarketRepricingOpportunity":
        empty = self.complete_round_trip_count == 0
        optional_values = (
            self.best_entry_cost_quote,
            self.best_exit_proceeds_quote,
            self.best_net_quote,
            self.best_net_bps_on_entry_cost,
        )
        optional_times = (
            self.best_decision_received_wall_ms,
            self.best_entry_received_wall_ms,
            self.best_exit_decision_received_wall_ms,
            self.best_exit_received_wall_ms,
            self.best_entry_execution_target_wall_ms,
            self.best_exit_decision_target_wall_ms,
            self.best_exit_execution_target_wall_ms,
            self.best_decision_received_monotonic_ns,
            self.best_entry_received_monotonic_ns,
            self.best_exit_decision_received_monotonic_ns,
            self.best_exit_received_monotonic_ns,
            self.best_entry_execution_target_monotonic_ns,
            self.best_exit_decision_target_monotonic_ns,
            self.best_exit_execution_target_monotonic_ns,
        )
        optional_delays = (
            self.best_entry_venue_taker_delay_ms,
            self.best_exit_venue_taker_delay_ms,
        )
        event_ids = (
            self.best_decision_event_id,
            self.best_entry_event_id,
            self.best_exit_decision_event_id,
            self.best_exit_event_id,
        )
        optional_state_valid = (
            all(value is None for value in optional_values)
            and all(value is None for value in optional_times)
            and all(value is None for value in optional_delays)
            and all(not value for value in event_ids)
            if empty
            else all(value is not None for value in optional_values)
            and all(value is not None for value in optional_times)
            and all(value is not None for value in optional_delays)
            and all(bool(value) for value in event_ids)
        )
        reason_counts = dict(self.terminal_reason_counts)
        if (
            self.asset not in _ASSETS
            or self.outcome not in {"Up", "Down"}
            or not self.condition_id
            or not self.market_id
            or not self.token_id
            or self.per_leg_submission_latency_ms not in _LATENCIES_MS
            or self.holding_period_ms not in _HOLDING_PERIODS_MS
            or self.decision_count < self.complete_round_trip_count
            or self.complete_round_trip_count < 0
            or set(reason_counts) != set(_TERMINAL_REASONS)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in reason_counts.values()
            )
            or sum(reason_counts.values()) != self.decision_count
            or reason_counts["complete_round_trip"]
            != self.complete_round_trip_count
            or self.quantity <= 0
            or not optional_state_valid
            or not empty
            and (
                self.best_entry_cost_quote is None
                or self.best_entry_cost_quote <= 0
                or self.best_exit_proceeds_quote is None
                or self.best_exit_proceeds_quote < 0
                or self.best_net_quote
                != self.best_exit_proceeds_quote - self.best_entry_cost_quote
                or self.best_net_bps_on_entry_cost
                != self.best_net_quote
                / self.best_entry_cost_quote
                * Decimal("10000")
                or self.best_entry_execution_target_wall_ms
                != self.best_decision_received_wall_ms
                + self.per_leg_submission_latency_ms
                + self.best_entry_venue_taker_delay_ms
                or self.best_exit_decision_target_wall_ms
                != self.best_entry_received_wall_ms
                + self.holding_period_ms
                or self.best_exit_execution_target_wall_ms
                != self.best_exit_decision_target_wall_ms
                + self.per_leg_submission_latency_ms
                + self.best_exit_venue_taker_delay_ms
                or self.best_entry_execution_target_monotonic_ns
                != self.best_decision_received_monotonic_ns
                + (
                    self.per_leg_submission_latency_ms
                    + self.best_entry_venue_taker_delay_ms
                )
                * 1_000_000
                or self.best_exit_decision_target_monotonic_ns
                != self.best_entry_received_monotonic_ns
                + self.holding_period_ms * 1_000_000
                or self.best_exit_execution_target_monotonic_ns
                != self.best_exit_decision_target_monotonic_ns
                + (
                    self.per_leg_submission_latency_ms
                    + self.best_exit_venue_taker_delay_ms
                )
                * 1_000_000
                or self.best_decision_received_monotonic_ns
                > self.best_entry_received_monotonic_ns
                or self.best_entry_received_monotonic_ns
                > self.best_exit_decision_received_monotonic_ns
                or self.best_exit_decision_received_monotonic_ns
                > self.best_exit_received_monotonic_ns
                or self.best_entry_received_monotonic_ns
                < self.best_entry_execution_target_monotonic_ns
                or self.best_entry_received_monotonic_ns
                - self.best_entry_execution_target_monotonic_ns
                > _MAX_POST_TARGET_EXECUTION_OBSERVATION_DELAY_MS * 1_000_000
                or self.best_exit_decision_received_monotonic_ns
                > self.best_exit_decision_target_monotonic_ns
                or self.best_exit_decision_target_monotonic_ns
                - self.best_exit_decision_received_monotonic_ns
                > _MAX_ORDER_CREATION_BOOK_AGE_MS * 1_000_000
                or self.best_exit_received_monotonic_ns
                < self.best_exit_execution_target_monotonic_ns
                or self.best_exit_received_monotonic_ns
                - self.best_exit_execution_target_monotonic_ns
                > _MAX_POST_TARGET_EXECUTION_OBSERVATION_DELAY_MS * 1_000_000
                or self.best_entry_venue_taker_delay_ms
                not in {0, POLYMARKET_TAKER_ORDER_DELAY_MS}
                or self.best_exit_venue_taker_delay_ms
                not in {0, POLYMARKET_TAKER_ORDER_DELAY_MS}
            )
            or self.opportunity_sha256
            != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket repricing opportunity is invalid")
        return self


@dataclass(frozen=True)
class PolymarketRepricingReport:
    schema_version: str
    contract_sha256: str
    source_run_id: str
    source_replay_evidence_sha256: str
    market_execution_parameter_set_sha256: str
    market_execution_evidence: tuple[PolymarketMarketExecutionEvidence, ...]
    replay_continuity_mode: str
    replay_stream_gap_count: int
    replay_book_sample_interval_ms: int
    config: PolymarketRepricingConfig
    market_counts: Mapping[str, int]
    opportunities: tuple[PolymarketRepricingOpportunity, ...]
    cells: tuple[Mapping[str, object], ...]
    primary_gate: Mapping[str, object]
    status: str
    confirmation_eligible: bool
    oracle_support_observed: bool
    report_sha256: str
    noncausal_oracle_upper_bound: bool = True
    prepositioned_inventory_assumption: bool = True
    training_authority: bool = False
    trading_authority: bool = False
    profitability_claim: bool = False

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "contract_sha256": self.contract_sha256,
            "source_run_id": self.source_run_id,
            "source_replay_evidence_sha256": self.source_replay_evidence_sha256,
            "market_execution_parameter_set_sha256": (
                self.market_execution_parameter_set_sha256
            ),
            "market_execution_evidence": [
                item.asdict() for item in self.market_execution_evidence
            ],
            "replay_continuity_mode": self.replay_continuity_mode,
            "replay_stream_gap_count": self.replay_stream_gap_count,
            "replay_book_sample_interval_ms": self.replay_book_sample_interval_ms,
            "config": self.config.asdict(),
            "market_counts": dict(sorted(self.market_counts.items())),
            "opportunities": [item.asdict() for item in self.opportunities],
            "cells": [dict(item) for item in self.cells],
            "primary_gate": dict(self.primary_gate),
            "status": self.status,
            "confirmation_eligible": self.confirmation_eligible,
            "oracle_support_observed": self.oracle_support_observed,
            "noncausal_oracle_upper_bound": self.noncausal_oracle_upper_bound,
            "prepositioned_inventory_assumption": (
                self.prepositioned_inventory_assumption
            ),
            "training_authority": self.training_authority,
            "trading_authority": self.trading_authority,
            "profitability_claim": self.profitability_claim,
        }

    def asdict(self) -> dict[str, object]:
        return {**self.identity_payload(), "report_sha256": self.report_sha256}

    def validated(self) -> "PolymarketRepricingReport":
        config = self.config.validated()
        opportunities = tuple(item.validated() for item in self.opportunities)
        evidence = tuple(item.validated() for item in self.market_execution_evidence)
        expected_evidence_order = tuple(
            sorted(
                evidence,
                key=lambda item: (
                    item.observed_monotonic_ns,
                    item.condition_id,
                    item.snapshot_sha256,
                ),
            )
        )
        expected_opportunity_order = tuple(sorted(opportunities, key=_opportunity_sort_key))
        opportunity_keys = {
            (
                item.condition_id,
                item.outcome,
                item.per_leg_submission_latency_ms,
                item.holding_period_ms,
            )
            for item in opportunities
        }
        condition_assets: dict[str, set[str]] = {}
        condition_markets: dict[str, set[str]] = {}
        condition_quantities: dict[str, set[Decimal]] = {}
        outcome_tokens: dict[tuple[str, str], set[str]] = {}
        outcome_decision_counts: dict[tuple[str, str], set[int]] = {}
        for item in opportunities:
            condition_assets.setdefault(item.condition_id, set()).add(item.asset)
            condition_markets.setdefault(item.condition_id, set()).add(item.market_id)
            condition_quantities.setdefault(item.condition_id, set()).add(item.quantity)
            key = (item.condition_id, item.outcome)
            outcome_tokens.setdefault(key, set()).add(item.token_id)
            outcome_decision_counts.setdefault(key, set()).add(item.decision_count)
        reconstructed_market_counts = {
            asset: sum(values == {asset} for values in condition_assets.values())
            for asset in _ASSETS
        }
        confirmation_eligible = (
            self.replay_continuity_mode == "strict"
            and self.replay_stream_gap_count == 0
            and self.replay_book_sample_interval_ms == 0
        )
        expected_cells = _cell_rows(opportunities)
        expected_primary_gate, economic_support = _primary_gate(
            opportunities,
            reconstructed_market_counts,
            config,
            confirmation_eligible=confirmation_eligible,
        )
        expected_status = _report_status(
            expected_primary_gate,
            economic_support=economic_support,
            confirmation_eligible=confirmation_eligible,
            config=config,
        )
        source_hashes = (
            self.source_replay_evidence_sha256,
            self.market_execution_parameter_set_sha256,
        )
        if (
            self.schema_version != POLYMARKET_REPRICING_REPORT_SCHEMA_VERSION
            or self.contract_sha256 != POLYMARKET_REPRICING_CONTRACT_SHA256
            or not self.source_run_id
            or any(
                len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
                for digest in source_hashes
            )
            or not evidence
            or evidence != expected_evidence_order
            or any(item.run_id != self.source_run_id for item in evidence)
            or len(
                {
                    (
                        item.condition_id,
                        item.observed_monotonic_ns,
                        item.snapshot_sha256,
                    )
                    for item in evidence
                }
            )
            != len(evidence)
            or {item.condition_id for item in evidence} != set(condition_assets)
            or self.market_execution_parameter_set_sha256
            != _market_execution_parameter_set_sha256(evidence)
            or not opportunities
            or opportunities != expected_opportunity_order
            or len(opportunity_keys) != len(opportunities)
            or len(opportunities)
            != len(condition_assets) * 2 * len(_LATENCIES_MS) * len(_HOLDING_PERIODS_MS)
            or any(len(values) != 1 for values in condition_assets.values())
            or any(len(values) != 1 for values in condition_markets.values())
            or any(len(values) != 1 for values in condition_quantities.values())
            or any(len(values) != 1 for values in outcome_tokens.values())
            or any(len(values) != 1 for values in outcome_decision_counts.values())
            or any(
                next(iter(outcome_tokens[(condition_id, "Up")]))
                == next(iter(outcome_tokens[(condition_id, "Down")]))
                for condition_id in condition_assets
            )
            or set(self.market_counts) != set(_ASSETS)
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in self.market_counts.values()
            )
            or dict(self.market_counts) != reconstructed_market_counts
            or self.replay_stream_gap_count < 0
            or self.replay_book_sample_interval_ms < 0
            or self.confirmation_eligible != confirmation_eligible
            or self.cells != expected_cells
            or dict(self.primary_gate) != dict(expected_primary_gate)
            or self.status != expected_status
            or self.oracle_support_observed != economic_support
            or not self.noncausal_oracle_upper_bound
            or not self.prepositioned_inventory_assumption
            or self.training_authority
            or self.trading_authority
            or self.profitability_claim
            or self.report_sha256 != _canonical_sha256(self.identity_payload())
        ):
            raise ValueError("Polymarket repricing report is invalid")
        return self


class _BookIndex:
    def __init__(self, replay: PolymarketEvidenceReplay) -> None:
        grouped: dict[tuple[str, str], list[PolymarketRecordedBook]] = {}
        for book in replay.books:
            grouped.setdefault((book.token_id, book.segment_id), []).append(book)
        for values in grouped.values():
            if any(
                _replay_book_order_key(current) < _replay_book_order_key(previous)
                for previous, current in zip(values, values[1:], strict=False)
            ):
                raise ValueError("Polymarket repricing books are not replay ordered")
        self.books = grouped
        segment_keys_by_token: dict[str, list[tuple[str, str]]] = {}
        for key in grouped:
            segment_keys_by_token.setdefault(key[0], []).append(key)
        self.segment_keys_by_token = {
            token_id: tuple(keys) for token_id, keys in segment_keys_by_token.items()
        }
        self.segment_deadline_ns: dict[tuple[str, str], int | None] = {}
        for token_id, keys in self.segment_keys_by_token.items():
            ordered_keys = sorted(
                keys,
                key=lambda key: _replay_book_order_key(self.books[key][0]),
            )
            self.segment_keys_by_token[token_id] = tuple(ordered_keys)
            for position, key in enumerate(ordered_keys):
                self.segment_deadline_ns[key] = (
                    None
                    if position + 1 == len(ordered_keys)
                    else self.books[ordered_keys[position + 1]][0].received_monotonic_ns
                )

    def books_for_token(self, token_id: str) -> Iterable[PolymarketRecordedBook]:
        groups = tuple(
            self.books[key] for key in self.segment_keys_by_token.get(token_id, ())
        )
        return merge(*groups, key=_replay_book_order_key)

    @staticmethod
    def _lower_bound(values: Sequence[PolymarketRecordedBook], target: int) -> int:
        left = 0
        right = len(values)
        while left < right:
            middle = (left + right) // 2
            if values[middle].received_monotonic_ns < target:
                left = middle + 1
            else:
                right = middle
        return left

    @staticmethod
    def _upper_bound(values: Sequence[PolymarketRecordedBook], target: int) -> int:
        left = 0
        right = len(values)
        while left < right:
            middle = (left + right) // 2
            if values[middle].received_monotonic_ns <= target:
                left = middle + 1
            else:
                right = middle
        return left

    def first_after(
        self,
        reference: PolymarketRecordedBook,
        delay_ms: int,
        maximum_observation_delay_ms: int,
    ) -> PolymarketRecordedBook | None:
        return self.first_at_or_after(
            token_id=reference.token_id,
            segment_id=reference.segment_id,
            condition_id=reference.market.condition_id,
            target_monotonic_ns=(
                reference.received_monotonic_ns + int(delay_ms) * 1_000_000
            ),
            maximum_observation_delay_ms=maximum_observation_delay_ms,
        )

    def first_at_or_after(
        self,
        *,
        token_id: str,
        segment_id: str,
        condition_id: str,
        target_monotonic_ns: int,
        maximum_observation_delay_ms: int,
    ) -> PolymarketRecordedBook | None:
        key = (token_id, segment_id)
        values = self.books.get(key, ())
        target = int(target_monotonic_ns)
        deadline = target + int(maximum_observation_delay_ms) * 1_000_000
        index = self._lower_bound(values, target)
        if index >= len(values):
            return None
        candidate = values[index]
        if (
            candidate.received_monotonic_ns > deadline
            or candidate.market.condition_id != condition_id
            or candidate.received_wall_ms >= candidate.market.end_ms
        ):
            return None
        return candidate

    def latest_at_or_before(
        self,
        reference: PolymarketRecordedBook,
        delay_ms: int,
        maximum_age_ms: int,
    ) -> PolymarketRecordedBook | None:
        return self.latest_at_or_before_target(
            token_id=reference.token_id,
            segment_id=reference.segment_id,
            condition_id=reference.market.condition_id,
            target_monotonic_ns=(
                reference.received_monotonic_ns + int(delay_ms) * 1_000_000
            ),
            maximum_age_ms=maximum_age_ms,
        )

    def latest_at_or_before_target(
        self,
        *,
        token_id: str,
        segment_id: str,
        condition_id: str,
        target_monotonic_ns: int,
        maximum_age_ms: int,
    ) -> PolymarketRecordedBook | None:
        key = (token_id, segment_id)
        values = self.books.get(key, ())
        target = int(target_monotonic_ns)
        segment_deadline = self.segment_deadline_ns.get(key)
        if segment_deadline is not None and target >= segment_deadline:
            return None
        index = self._upper_bound(values, target) - 1
        if index < 0:
            return None
        candidate = values[index]
        if (
            candidate.market.condition_id != condition_id
            or candidate.received_wall_ms >= candidate.market.end_ms
            or target - candidate.received_monotonic_ns
            > int(maximum_age_ms) * 1_000_000
        ):
            return None
        return candidate


class _ExecutionParameterIndex:
    def __init__(
        self,
        evidence: Sequence[PolymarketMarketExecutionEvidence],
    ) -> None:
        grouped: dict[str, list[PolymarketMarketExecutionEvidence]] = {}
        for item in evidence:
            grouped.setdefault(item.condition_id, []).append(item.validated())
        self.values = {
            condition_id: tuple(
                sorted(
                    rows,
                    key=lambda item: (
                        item.observed_monotonic_ns,
                        item.observed_wall_ms,
                        item.snapshot_sha256,
                    ),
                )
            )
            for condition_id, rows in grouped.items()
        }
        self.timestamps = {
            condition_id: tuple(item.observed_monotonic_ns for item in rows)
            for condition_id, rows in self.values.items()
        }

    def latest_at_or_before(
        self,
        condition_id: str,
        monotonic_ns: int,
    ) -> PolymarketMarketExecutionEvidence | None:
        values = self.values.get(condition_id, ())
        timestamps = self.timestamps.get(condition_id, ())
        index = bisect_right(timestamps, int(monotonic_ns)) - 1
        return None if index < 0 else values[index]


def _valid_limit_for_tick(price: Decimal, tick_size: Decimal) -> bool:
    if tick_size <= 0 or tick_size > Decimal("0.1"):
        return False
    return (
        tick_size <= price <= Decimal("1") - tick_size
        and price % tick_size == 0
    )


class PolymarketRepricingExecutionContext:
    """Build shared replay indexes once and execute exact two-leg decisions."""

    def __init__(self, replay: PolymarketEvidenceReplay) -> None:
        evidence = tuple(item.validated() for item in replay.market_execution_evidence)
        conditions = {market.condition_id for market in replay.markets}
        if not evidence or {item.condition_id for item in evidence} != conditions:
            raise ValueError(
                "Polymarket repricing execution parameters do not cover markets"
            )
        if any(item.run_id != replay.run_id for item in evidence):
            raise ValueError("Polymarket repricing execution evidence run differs")
        self.run_id = replay.run_id
        self.book_sample_interval_ms = replay.diagnostics.book_sample_interval_ms
        self.market_by_condition = {
            market.condition_id: market for market in replay.markets
        }
        decision_events: dict[
            tuple[str, str], list[PolymarketRecordedBook]
        ] = {}
        for book in replay.books:
            decision_events.setdefault(
                (book.market.condition_id, book.event_id), []
            ).append(book)
        self.decision_events = {
            key: tuple(values) for key, values in decision_events.items()
        }
        self.book_index = _BookIndex(replay)
        self.parameter_index = _ExecutionParameterIndex(evidence)

    def decision_at(
        self,
        market: PolymarketFiveMinuteMarket,
        *,
        event_id: str,
        received_wall_ms: int,
        received_monotonic_ns: int,
        outcome: str,
        maximum_creation_book_age_ms: int,
    ) -> PolymarketRepricingDecision | None:
        """Resolve one hash-bound feature clock to its latest causal token book."""

        if (
            self.market_by_condition.get(market.condition_id) != market
            or not event_id
            or received_wall_ms < 0
            or received_monotonic_ns < 0
            or outcome not in {"Up", "Down"}
            or maximum_creation_book_age_ms < 0
        ):
            raise ValueError("Polymarket repricing decision clock is invalid")
        anchors = tuple(
            book
            for book in self.decision_events.get(
                (market.condition_id, event_id), ()
            )
            if book.received_wall_ms == received_wall_ms
            and book.received_monotonic_ns == received_monotonic_ns
        )
        if not anchors:
            raise ValueError("Polymarket decision event is absent from replay evidence")
        segments = {book.segment_id for book in anchors}
        if len(segments) != 1:
            raise ValueError("Polymarket decision event crosses continuity segments")
        segment_id = next(iter(segments))
        token_id = market.up_token_id if outcome == "Up" else market.down_token_id
        creation_book = self.book_index.latest_at_or_before_target(
            token_id=token_id,
            segment_id=segment_id,
            condition_id=market.condition_id,
            target_monotonic_ns=received_monotonic_ns,
            maximum_age_ms=maximum_creation_book_age_ms,
        )
        if creation_book is None:
            return None
        return PolymarketRepricingDecision(
            event_id=event_id,
            condition_id=market.condition_id,
            token_id=token_id,
            outcome=outcome,
            segment_id=segment_id,
            received_wall_ms=received_wall_ms,
            received_monotonic_ns=received_monotonic_ns,
            creation_book=creation_book,
        ).validated(maximum_creation_book_age_ms=maximum_creation_book_age_ms)

    def execute(
        self,
        market: PolymarketFiveMinuteMarket,
        decision: PolymarketRepricingDecision,
        *,
        latency_ms: int,
        holding_period_ms: int,
        minimum_remaining_market_time_ms: int,
        maximum_order_creation_book_age_ms: int,
        maximum_post_target_execution_observation_delay_ms: int,
    ) -> PolymarketRepricingDecisionExecution:
        """Replay one share-sized V2 FOK entry and owned-quantity FOK exit."""

        selected = decision.validated(
            maximum_creation_book_age_ms=maximum_order_creation_book_age_ms
        )
        if (
            market.condition_id != selected.condition_id
            or selected.token_id not in market.token_ids
            or selected.outcome
            != ("Up" if selected.token_id == market.up_token_id else "Down")
            or latency_ms <= 0
            or holding_period_ms <= 0
        ):
            raise ValueError("Polymarket repricing execution request is invalid")

        def terminal(
            reason: str,
            **values: object,
        ) -> PolymarketRepricingDecisionExecution:
            if reason not in _TERMINAL_REASONS:
                raise ValueError("Polymarket repricing terminal reason is invalid")
            return PolymarketRepricingDecisionExecution(
                terminal_reason=reason,
                decision=selected,
                **values,
            )

        quantity = market.minimum_order_size
        fee = market.fee_schedule.fee_model()
        entry_parameter = self.parameter_index.latest_at_or_before(
            market.condition_id,
            selected.received_monotonic_ns + latency_ms * 1_000_000,
        )
        if entry_parameter is None:
            return terminal("missing_entry_execution_parameters")
        if entry_parameter.minimum_order_age_seconds != 0:
            return terminal(
                "unsupported_entry_minimum_order_age",
                entry_parameter=entry_parameter,
            )
        entry_venue_delay_ms = entry_parameter.taker_order_delay_ms
        entry_total_latency_ms = latency_ms + entry_venue_delay_ms
        entry_target_wall_ms = selected.received_wall_ms + entry_total_latency_ms
        entry_target_monotonic_ns = (
            selected.received_monotonic_ns + entry_total_latency_ms * 1_000_000
        )
        entry_values = {
            "entry_parameter": entry_parameter,
            "entry_venue_taker_delay_ms": entry_venue_delay_ms,
            "entry_execution_target_wall_ms": entry_target_wall_ms,
            "entry_execution_target_monotonic_ns": entry_target_monotonic_ns,
        }
        if (
            market.end_ms - entry_target_wall_ms
            < minimum_remaining_market_time_ms
        ):
            return terminal("entry_enters_excluded_close_window", **entry_values)
        entry = self.book_index.first_at_or_after(
            token_id=selected.token_id,
            segment_id=selected.segment_id,
            condition_id=selected.condition_id,
            target_monotonic_ns=entry_target_monotonic_ns,
            maximum_observation_delay_ms=(
                maximum_post_target_execution_observation_delay_ms
            ),
        )
        if entry is None:
            return terminal("missing_entry_execution_book", **entry_values)
        entry_values["entry_book"] = entry
        if (
            market.end_ms - entry.received_wall_ms
            < minimum_remaining_market_time_ms
        ):
            return terminal(
                "entry_confirmation_enters_excluded_close_window",
                **entry_values,
            )
        entry_limit = Decimal("1") - selected.creation_book.tick_size
        if not _valid_limit_for_tick(
            entry_limit, selected.creation_book.tick_size
        ) or not _valid_limit_for_tick(entry_limit, entry.tick_size):
            return terminal("entry_tick_drift", **entry_values)

        entry_seed = _canonical_sha256(
            {
                "leg": "entry",
                "condition_id": market.condition_id,
                "outcome": selected.outcome,
                "decision_event_id": selected.event_id,
                "per_leg_submission_latency_ms": latency_ms,
                "holding_period_ms": holding_period_ms,
                "entry_execution_parameter_sha256": (
                    entry_parameter.snapshot_sha256
                ),
            }
        )
        entry_intent = PaperOrderIntent(
            intent_id=paper_intent_id("polymarket", entry_seed, "open"),
            venue="polymarket",
            market_id=market.condition_id,
            asset_id=selected.token_id,
            symbol=market.asset,
            outcome=selected.outcome,
            side="BUY",
            order_type="FOK",
            limit_price=entry_limit,
            quantity=quantity,
            created_at_ms=selected.received_wall_ms,
            expires_at_ms=market.end_ms,
        ).validated()
        entry_result = simulate_aggressive_order(
            entry_intent,
            entry.snapshot,
            execution_time_ms=entry.received_wall_ms,
            submission_latency_ms=entry_total_latency_ms,
            maximum_book_age_ms=(
                maximum_post_target_execution_observation_delay_ms
            ),
            fee=fee,
        )
        entry_values["entry_result"] = entry_result
        if entry_result.state != "FILLED":
            return terminal("entry_not_filled", **entry_values)
        entry_cost = (
            entry_result.average_fill_price * entry_result.filled_quantity
            + entry_result.fee_quote
        )
        entry_values["entry_cost_quote"] = entry_cost

        exit_target_monotonic_ns = (
            entry.received_monotonic_ns + holding_period_ms * 1_000_000
        )
        exit_decision = self.book_index.latest_at_or_before_target(
            token_id=selected.token_id,
            segment_id=selected.segment_id,
            condition_id=selected.condition_id,
            target_monotonic_ns=exit_target_monotonic_ns,
            maximum_age_ms=maximum_order_creation_book_age_ms,
        )
        if exit_decision is None:
            return terminal("missing_exit_decision_book", **entry_values)
        exit_parameter = self.parameter_index.latest_at_or_before(
            market.condition_id,
            exit_target_monotonic_ns + latency_ms * 1_000_000,
        )
        exit_created_at_ms = entry.received_wall_ms + holding_period_ms
        exit_values = {
            **entry_values,
            "exit_decision_book": exit_decision,
            "exit_decision_target_wall_ms": exit_created_at_ms,
            "exit_decision_target_monotonic_ns": exit_target_monotonic_ns,
        }
        if exit_parameter is None:
            return terminal("missing_exit_execution_parameters", **exit_values)
        exit_values["exit_parameter"] = exit_parameter
        if exit_parameter.minimum_order_age_seconds != 0:
            return terminal("unsupported_exit_minimum_order_age", **exit_values)
        exit_venue_delay_ms = exit_parameter.taker_order_delay_ms
        exit_total_latency_ms = latency_ms + exit_venue_delay_ms
        exit_execution_target_wall_ms = (
            exit_created_at_ms + exit_total_latency_ms
        )
        exit_execution_target_monotonic_ns = (
            exit_target_monotonic_ns + exit_total_latency_ms * 1_000_000
        )
        exit_values.update(
            {
                "exit_venue_taker_delay_ms": exit_venue_delay_ms,
                "exit_execution_target_wall_ms": exit_execution_target_wall_ms,
                "exit_execution_target_monotonic_ns": (
                    exit_execution_target_monotonic_ns
                ),
            }
        )
        if (
            market.end_ms - exit_execution_target_wall_ms
            < minimum_remaining_market_time_ms
        ):
            return terminal("exit_enters_excluded_close_window", **exit_values)
        exit_book = self.book_index.first_at_or_after(
            token_id=selected.token_id,
            segment_id=selected.segment_id,
            condition_id=selected.condition_id,
            target_monotonic_ns=exit_execution_target_monotonic_ns,
            maximum_observation_delay_ms=(
                maximum_post_target_execution_observation_delay_ms
            ),
        )
        if exit_book is None:
            return terminal("missing_exit_execution_book", **exit_values)
        exit_values["exit_book"] = exit_book
        if (
            market.end_ms - exit_book.received_wall_ms
            < minimum_remaining_market_time_ms
        ):
            return terminal("exit_enters_excluded_close_window", **exit_values)
        exit_limit = exit_decision.tick_size
        if not _valid_limit_for_tick(
            exit_limit, exit_decision.tick_size
        ) or not _valid_limit_for_tick(exit_limit, exit_book.tick_size):
            return terminal("exit_tick_drift", **exit_values)

        exit_seed = _canonical_sha256(
            {
                "leg": "exit",
                "condition_id": market.condition_id,
                "outcome": selected.outcome,
                "decision_event_id": selected.event_id,
                "entry_intent_id": entry_intent.intent_id,
                "exit_creation_book_event_id": exit_decision.event_id,
                "per_leg_submission_latency_ms": latency_ms,
                "holding_period_ms": holding_period_ms,
                "exit_execution_parameter_sha256": (
                    exit_parameter.snapshot_sha256
                ),
            }
        )
        exit_intent = PaperOrderIntent(
            intent_id=paper_intent_id("polymarket", exit_seed, "close"),
            venue="polymarket",
            market_id=market.condition_id,
            asset_id=selected.token_id,
            symbol=market.asset,
            outcome=selected.outcome,
            side="SELL",
            order_type="FOK",
            limit_price=exit_limit,
            quantity=entry_result.filled_quantity,
            created_at_ms=exit_created_at_ms,
            expires_at_ms=market.end_ms,
        ).validated()
        exit_result = simulate_aggressive_order(
            exit_intent,
            exit_book.snapshot,
            execution_time_ms=exit_book.received_wall_ms,
            submission_latency_ms=exit_total_latency_ms,
            maximum_book_age_ms=(
                maximum_post_target_execution_observation_delay_ms
            ),
            fee=fee,
            owned_quantity=entry_result.filled_quantity,
            closing_position=True,
        )
        exit_values["exit_result"] = exit_result
        if (
            exit_result.state != "FILLED"
            or exit_result.filled_quantity != entry_result.filled_quantity
        ):
            return terminal("exit_not_filled", **exit_values)
        exit_proceeds = (
            exit_result.average_fill_price * exit_result.filled_quantity
            - exit_result.fee_quote
        )
        return terminal(
            "complete_round_trip",
            **exit_values,
            exit_proceeds_quote=exit_proceeds,
            net_quote=exit_proceeds - entry_cost,
        )


def _source_replay_evidence_sha256(replay: PolymarketEvidenceReplay) -> str:
    digest = hashlib.sha256()
    digest.update(b'{"books":[')
    previous_key: tuple[int, int, str, int, int, str] | None = None
    for index, book in enumerate(replay.books):
        order_key = _replay_book_order_key(book)
        if previous_key is not None and order_key < previous_key:
            raise ValueError("Polymarket repricing source books are not replay ordered")
        previous_key = order_key
        snapshot = book.snapshot.validated()
        row = {
            "event_id": book.event_id,
            "event_type": book.event_type,
            "connection_id": book.connection_id,
            "segment_id": book.segment_id,
            "sequence_number": book.sequence_number,
            "sub_index": book.sub_index,
            "condition_id": book.market.condition_id,
            "token_id": book.token_id,
            "outcome": book.outcome,
            "tick_size": _decimal_text(book.tick_size),
            "received_wall_ms": book.received_wall_ms,
            "received_monotonic_ns": book.received_monotonic_ns,
            "snapshot": {
                "venue": snapshot.venue,
                "market_id": snapshot.market_id,
                "asset_id": snapshot.asset_id,
                "bids": [
                    {
                        "price": _decimal_text(level.price),
                        "quantity": _decimal_text(level.quantity),
                    }
                    for level in snapshot.bids
                ],
                "asks": [
                    {
                        "price": _decimal_text(level.price),
                        "quantity": _decimal_text(level.quantity),
                    }
                    for level in snapshot.asks
                ],
                "source_time_ms": snapshot.source_time_ms,
                "received_wall_ms": snapshot.received_wall_ms,
                "received_monotonic_ns": snapshot.received_monotonic_ns,
                "connected": snapshot.connected,
                "gap_free": snapshot.gap_free,
                "source_payload_sha256": snapshot.source_payload_sha256,
            },
        }
        if index:
            digest.update(b",")
        digest.update(_canonical_json(row).encode("ascii"))
    digest.update(b'],"markets":')
    markets = [
        market.asdict()
        for market in sorted(
            replay.markets,
            key=lambda item: (item.event_start_ms, item.asset, item.condition_id),
        )
    ]
    digest.update(_canonical_json(markets).encode("ascii"))
    digest.update(b"}")
    return digest.hexdigest()


def _market_execution_parameter_set_sha256(
    evidence: Sequence[PolymarketMarketExecutionEvidence],
) -> str:
    return _canonical_sha256([item.asdict() for item in evidence])


def _scenario_opportunity(
    market: PolymarketFiveMinuteMarket,
    outcome: str,
    token_id: str,
    decisions: Sequence[PolymarketRecordedBook],
    execution_context: PolymarketRepricingExecutionContext,
    *,
    latency_ms: int,
    holding_period_ms: int,
    config: PolymarketRepricingConfig,
) -> PolymarketRepricingOpportunity:
    quantity = market.minimum_order_size
    terminal_reason_counts = {reason: 0 for reason in _TERMINAL_REASONS}
    best: tuple[
        Decimal,
        PolymarketRecordedBook,
        PolymarketRecordedBook,
        PolymarketRecordedBook,
        PolymarketRecordedBook,
        Decimal,
        Decimal,
        int,
        int,
        int,
        int,
        int,
    ] | None = None

    for decision in decisions:
        execution = execution_context.execute(
            market,
            PolymarketRepricingDecision.from_book(decision),
            latency_ms=latency_ms,
            holding_period_ms=holding_period_ms,
            minimum_remaining_market_time_ms=(
                config.minimum_remaining_market_time_ms
            ),
            maximum_order_creation_book_age_ms=(
                config.maximum_order_creation_book_age_ms
            ),
            maximum_post_target_execution_observation_delay_ms=(
                config.maximum_post_target_execution_observation_delay_ms
            ),
        )
        terminal_reason_counts[execution.terminal_reason] += 1
        if execution.terminal_reason != "complete_round_trip":
            continue
        if (
            execution.entry_book is None
            or execution.exit_decision_book is None
            or execution.exit_book is None
            or execution.entry_cost_quote is None
            or execution.exit_proceeds_quote is None
            or execution.net_quote is None
            or execution.entry_venue_taker_delay_ms is None
            or execution.exit_venue_taker_delay_ms is None
            or execution.entry_execution_target_wall_ms is None
            or execution.exit_decision_target_wall_ms is None
            or execution.exit_execution_target_wall_ms is None
        ):
            raise ValueError("complete Polymarket repricing execution is incomplete")
        net = execution.net_quote
        if best is None or net > best[0]:
            best = (
                net,
                decision,
                execution.entry_book,
                execution.exit_decision_book,
                execution.exit_book,
                execution.entry_cost_quote,
                execution.exit_proceeds_quote,
                execution.entry_venue_taker_delay_ms,
                execution.exit_venue_taker_delay_ms,
                execution.entry_execution_target_wall_ms,
                execution.exit_decision_target_wall_ms,
                execution.exit_execution_target_wall_ms,
            )

    provisional = PolymarketRepricingOpportunity(
        condition_id=market.condition_id,
        market_id=market.market_id,
        asset=market.asset,
        outcome=outcome,
        token_id=token_id,
        per_leg_submission_latency_ms=latency_ms,
        holding_period_ms=holding_period_ms,
        decision_count=len(decisions),
        complete_round_trip_count=terminal_reason_counts["complete_round_trip"],
        terminal_reason_counts=terminal_reason_counts,
        best_decision_event_id="" if best is None else best[1].event_id,
        best_entry_event_id="" if best is None else best[2].event_id,
        best_exit_decision_event_id="" if best is None else best[3].event_id,
        best_exit_event_id="" if best is None else best[4].event_id,
        best_decision_received_wall_ms=(
            None if best is None else best[1].received_wall_ms
        ),
        best_entry_received_wall_ms=(
            None if best is None else best[2].received_wall_ms
        ),
        best_exit_decision_received_wall_ms=(
            None if best is None else best[3].received_wall_ms
        ),
        best_exit_received_wall_ms=(
            None if best is None else best[4].received_wall_ms
        ),
        best_decision_received_monotonic_ns=(
            None if best is None else best[1].received_monotonic_ns
        ),
        best_entry_received_monotonic_ns=(
            None if best is None else best[2].received_monotonic_ns
        ),
        best_exit_decision_received_monotonic_ns=(
            None if best is None else best[3].received_monotonic_ns
        ),
        best_exit_received_monotonic_ns=(
            None if best is None else best[4].received_monotonic_ns
        ),
        best_entry_venue_taker_delay_ms=None if best is None else best[7],
        best_exit_venue_taker_delay_ms=None if best is None else best[8],
        best_entry_execution_target_wall_ms=None if best is None else best[9],
        best_exit_decision_target_wall_ms=None if best is None else best[10],
        best_exit_execution_target_wall_ms=None if best is None else best[11],
        best_entry_execution_target_monotonic_ns=(
            None
            if best is None
            else best[1].received_monotonic_ns
            + (latency_ms + best[7]) * 1_000_000
        ),
        best_exit_decision_target_monotonic_ns=(
            None
            if best is None
            else best[2].received_monotonic_ns
            + holding_period_ms * 1_000_000
        ),
        best_exit_execution_target_monotonic_ns=(
            None
            if best is None
            else best[2].received_monotonic_ns
            + (holding_period_ms + latency_ms + best[8]) * 1_000_000
        ),
        quantity=quantity,
        best_entry_cost_quote=None if best is None else best[5],
        best_exit_proceeds_quote=None if best is None else best[6],
        best_net_quote=None if best is None else best[0],
        best_net_bps_on_entry_cost=(
            None if best is None else best[0] / best[5] * Decimal("10000")
        ),
        opportunity_sha256="",
    )
    return replace(
        provisional,
        opportunity_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


def _sample_decisions(
    books: Iterable[PolymarketRecordedBook],
    config: PolymarketRepricingConfig,
) -> tuple[PolymarketRecordedBook, ...]:
    selected: list[PolymarketRecordedBook] = []
    last_by_segment: dict[str, int] = {}
    previous_key: tuple[int, int, str, int, int, str] | None = None
    for book in books:
        order_key = _replay_book_order_key(book)
        if previous_key is not None and order_key < previous_key:
            raise ValueError("Polymarket repricing decisions are not replay ordered")
        previous_key = order_key
        if (
            book.received_wall_ms < book.market.event_start_ms
            or book.market.end_ms - book.received_wall_ms
            < config.minimum_remaining_market_time_ms
        ):
            continue
        previous = last_by_segment.get(book.segment_id)
        if (
            previous is not None
            and book.received_monotonic_ns - previous
            < config.minimum_decision_spacing_ms * 1_000_000
        ):
            continue
        selected.append(book)
        last_by_segment[book.segment_id] = book.received_monotonic_ns
    return tuple(selected)


def _opportunity_sort_key(
    item: PolymarketRepricingOpportunity,
) -> tuple[str, str, str, int, int]:
    return (
        item.asset,
        item.condition_id,
        item.outcome,
        item.per_leg_submission_latency_ms,
        item.holding_period_ms,
    )


def _cell_rows(
    opportunities: Sequence[PolymarketRepricingOpportunity],
) -> tuple[Mapping[str, object], ...]:
    rows: list[Mapping[str, object]] = []
    for asset in _ASSETS:
        for latency_ms in _LATENCIES_MS:
            for holding_period_ms in _HOLDING_PERIODS_MS:
                selected = tuple(
                    item
                    for item in opportunities
                    if item.asset == asset
                    and item.per_leg_submission_latency_ms == latency_ms
                    and item.holding_period_ms == holding_period_ms
                )
                completed = tuple(
                    item
                    for item in selected
                    if item.best_net_bps_on_entry_cost is not None
                )
                positive_conditions = {
                    item.condition_id for item in completed if item.positive
                }
                complete_conditions = {item.condition_id for item in completed}
                values = tuple(
                    item.best_net_bps_on_entry_cost
                    for item in completed
                    if item.best_net_bps_on_entry_cost is not None
                )
                rows.append(
                    {
                        "asset": asset,
                        "per_leg_submission_latency_ms": latency_ms,
                        "holding_period_ms": holding_period_ms,
                        "market_count": len({item.condition_id for item in selected}),
                        "complete_market_count": len(complete_conditions),
                        "positive_market_count": len(positive_conditions),
                        "positive_market_fraction": (
                            "0"
                            if not complete_conditions
                            else _decimal_text(
                                Decimal(len(positive_conditions))
                                / Decimal(len(complete_conditions))
                            )
                        ),
                        "median_market_outcome_best_net_bps": (
                            None if not values else _decimal_text(median(values))
                        ),
                        "maximum_market_outcome_best_net_bps": (
                            None if not values else _decimal_text(max(values))
                        ),
                    }
                )
    return tuple(rows)


def _primary_gate(
    opportunities: Sequence[PolymarketRepricingOpportunity],
    market_counts: Mapping[str, int],
    config: PolymarketRepricingConfig,
    *,
    confirmation_eligible: bool,
) -> tuple[Mapping[str, object], bool]:
    per_asset: list[dict[str, object]] = []
    economic_pass = True
    for asset in _ASSETS:
        selected = tuple(
            item
            for item in opportunities
            if item.asset == asset
            and item.per_leg_submission_latency_ms
            == config.primary_per_leg_submission_latency_ms
            and item.holding_period_ms == config.primary_holding_period_ms
        )
        complete_conditions = {
            item.condition_id
            for item in selected
            if item.best_net_quote is not None
        }
        positive_conditions = {
            item.condition_id for item in selected if item.positive
        }
        positive_up = len(
            {item.condition_id for item in selected if item.outcome == "Up" and item.positive}
        )
        positive_down = len(
            {
                item.condition_id
                for item in selected
                if item.outcome == "Down" and item.positive
            }
        )
        fraction = (
            Decimal("0")
            if not complete_conditions
            else Decimal(len(positive_conditions)) / Decimal(len(complete_conditions))
        )
        passed = (
            len(complete_conditions)
            >= config.minimum_complete_markets_per_asset
            and len(positive_conditions)
            >= config.minimum_positive_markets_per_asset
            and fraction >= config.minimum_positive_market_fraction_per_asset
            and positive_up >= config.minimum_positive_markets_per_outcome_asset
            and positive_down >= config.minimum_positive_markets_per_outcome_asset
        )
        economic_pass = economic_pass and passed
        per_asset.append(
            {
                "asset": asset,
                "source_market_count": int(market_counts[asset]),
                "complete_round_trip_market_count": len(complete_conditions),
                "positive_market_count": len(positive_conditions),
                "positive_market_fraction": _decimal_text(fraction),
                "positive_up_market_count": positive_up,
                "positive_down_market_count": positive_down,
                "economic_breadth_passed": passed,
            }
        )
    return (
        {
            "per_leg_submission_latency_ms": (
                config.primary_per_leg_submission_latency_ms
            ),
            "holding_period_ms": config.primary_holding_period_ms,
            "source_confirmation_eligible": confirmation_eligible,
            "economic_breadth_passed": economic_pass,
            "gate_passed": confirmation_eligible and economic_pass,
            "per_asset": per_asset,
            "training_authority_if_passed": False,
            "trading_authority_if_passed": False,
        },
        economic_pass,
    )


def _report_status(
    primary_gate: Mapping[str, object],
    *,
    economic_support: bool,
    confirmation_eligible: bool,
    config: PolymarketRepricingConfig,
) -> str:
    per_asset = primary_gate.get("per_asset")
    if not isinstance(per_asset, Sequence):
        raise ValueError("Polymarket repricing primary gate is malformed")
    enough_markets = all(
        isinstance(row, Mapping)
        and int(row["complete_round_trip_market_count"])
        >= config.minimum_complete_markets_per_asset
        for row in per_asset
    )
    if not enough_markets:
        return "insufficient_market_support"
    if not economic_support:
        return "rejected_oracle_ceiling"
    if not confirmation_eligible:
        return "diagnostic_oracle_support_not_confirmatory"
    return "oracle_support_research_only"


def evaluate_polymarket_repricing_ceiling(
    replay: PolymarketEvidenceReplay,
    *,
    config: PolymarketRepricingConfig | None = None,
) -> PolymarketRepricingReport:
    """Measure a deliberately optimistic two-taker-leg opportunity ceiling."""

    cfg = (config or PolymarketRepricingConfig()).validated()
    markets = tuple(replay.markets)
    if not markets or not replay.books:
        raise ValueError("Polymarket repricing screen requires replay books and markets")
    market_by_condition = {market.condition_id: market for market in markets}
    if len(market_by_condition) != len(markets):
        raise ValueError("Polymarket repricing markets are duplicated")
    if any(market.asset not in _ASSETS for market in markets):
        raise ValueError("Polymarket repricing screen found an unsupported asset")
    market_counts = {
        asset: sum(market.asset == asset for market in markets) for asset in _ASSETS
    }
    execution_evidence = tuple(
        sorted(
            (item.validated() for item in replay.market_execution_evidence),
            key=lambda item: (
                item.observed_monotonic_ns,
                item.condition_id,
                item.snapshot_sha256,
            ),
        )
    )
    if not execution_evidence:
        raise ValueError(
            "Polymarket repricing screen requires recorded execution parameters"
        )
    if any(item.run_id != replay.run_id for item in execution_evidence):
        raise ValueError("Polymarket execution evidence belongs to another run")
    if {item.condition_id for item in execution_evidence} != set(market_by_condition):
        raise ValueError(
            "Polymarket execution evidence does not exactly cover replay markets"
        )
    for book in replay.books:
        market = market_by_condition.get(book.market.condition_id)
        if (
            market is None
            or market != book.market
            or book.token_id not in market.token_ids
        ):
            raise ValueError("Polymarket repricing book metadata is inconsistent")

    execution_context = PolymarketRepricingExecutionContext(replay)
    opportunities: list[PolymarketRepricingOpportunity] = []
    for market in sorted(
        markets,
        key=lambda item: (item.event_start_ms, item.asset, item.condition_id),
    ):
        for outcome, token_id in (
            ("Up", market.up_token_id),
            ("Down", market.down_token_id),
        ):
            decisions = _sample_decisions(
                execution_context.book_index.books_for_token(token_id), cfg
            )
            for latency_ms in cfg.per_leg_submission_latencies_ms:
                for holding_period_ms in cfg.holding_periods_ms:
                    opportunities.append(
                        _scenario_opportunity(
                            market,
                            outcome,
                            token_id,
                            decisions,
                            execution_context,
                            latency_ms=latency_ms,
                            holding_period_ms=holding_period_ms,
                            config=cfg,
                        )
                    )
    opportunities.sort(key=_opportunity_sort_key)
    confirmation_eligible = (
        replay.diagnostics.continuity_mode == "strict"
        and replay.diagnostics.stream_gap_count == 0
        and replay.diagnostics.book_sample_interval_ms == 0
    )
    cells = _cell_rows(opportunities)
    primary_gate, economic_support = _primary_gate(
        opportunities,
        market_counts,
        cfg,
        confirmation_eligible=confirmation_eligible,
    )
    status = _report_status(
        primary_gate,
        economic_support=economic_support,
        confirmation_eligible=confirmation_eligible,
        config=cfg,
    )
    provisional = PolymarketRepricingReport(
        schema_version=POLYMARKET_REPRICING_REPORT_SCHEMA_VERSION,
        contract_sha256=POLYMARKET_REPRICING_CONTRACT_SHA256,
        source_run_id=replay.run_id,
        source_replay_evidence_sha256=_source_replay_evidence_sha256(replay),
        market_execution_parameter_set_sha256=(
            _market_execution_parameter_set_sha256(execution_evidence)
        ),
        market_execution_evidence=execution_evidence,
        replay_continuity_mode=replay.diagnostics.continuity_mode,
        replay_stream_gap_count=replay.diagnostics.stream_gap_count,
        replay_book_sample_interval_ms=(
            replay.diagnostics.book_sample_interval_ms
        ),
        config=cfg,
        market_counts=market_counts,
        opportunities=tuple(opportunities),
        cells=cells,
        primary_gate=primary_gate,
        status=status,
        confirmation_eligible=confirmation_eligible,
        oracle_support_observed=economic_support,
        report_sha256="",
    )
    return replace(
        provisional,
        report_sha256=_canonical_sha256(provisional.identity_payload()),
    ).validated()


__all__ = [
    "POLYMARKET_REPRICING_CONTRACT_SHA256",
    "POLYMARKET_REPRICING_REPORT_SCHEMA_VERSION",
    "PolymarketRepricingConfig",
    "PolymarketRepricingDecision",
    "PolymarketRepricingDecisionExecution",
    "PolymarketRepricingExecutionContext",
    "PolymarketRepricingOpportunity",
    "PolymarketRepricingReport",
    "evaluate_polymarket_repricing_ceiling",
]
