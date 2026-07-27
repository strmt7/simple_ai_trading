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


ROUND74_EVENT_TARGET_SCHEMA_VERSION = "round-074-executable-event-target-v9"
ROUND74_EVENT_TARGET_EVIDENCE_SCHEMA_VERSION = (
    "round-074-target-evidence-v1"
)
ROUND74_EVENT_TARGET_MAXIMUM_LATENCY_NS = 5_000_000_000
ROUND74_EVENT_TARGET_MAXIMUM_SLIPPAGE_BPS_PER_SIDE = 1_000.0
ROUND74_EVENT_TARGET_MAXIMUM_STATE_LATENESS_NS = 250_000_000
ROUND74_EVENT_TARGET_MAXIMUM_DECISION_STATE_AGE_NS = 250_000_000
ROUND74_EVENT_TARGET_MAXIMUM_PATH_STATE_GAP_NS = 250_000_000
ROUND74_EVENT_TARGET_MINIMUM_ANCHOR_SPACING_NS = 1_000_000_000
ROUND74_EVENT_TARGET_MAXIMUM_ADDITIONAL_ENTRY_LATENCY_NS = 30_000_000_000
ROUND74_EVENT_EXECUTION_OVERRIDE_SCHEMA_VERSION = (
    "round-074-event-execution-override-v1"
)
ROUND74_EVENT_TARGET_SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
ROUND74_EVENT_TARGET_ENVIRONMENTS = (
    "binance_usdm_mainnet",
    "binance_usdm_testnet",
)
ROUND74_EVENT_TARGET_EVIDENCE_SOURCES = {
    "quantity_rules": "binance_usdm_fapi_v1_exchange_info",
    "commission": "binance_usdm_fapi_v1_commission_rate",
    "entry_exit_latency": (
        "round74_host_submission_to_execution_latency_v1"
    ),
    "residual_slippage": (
        "round74_realized_execution_shortfall_calibration_v1"
    ),
    "funding_schedule": "binance_usdm_fapi_v1_funding_rate",
}
ROUND74_EVENT_TARGET_INELIGIBLE_REASONS = frozenset(
    {
        "decision_state_missing",
        "decision_state_late",
        "quantity_ineligible",
        "entry_state_late",
        "entry_capacity",
        "entry_minimum_notional",
        "funding_boundary",
        "funding_coverage",
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


def round74_commission_evidence_claims(
    taker_fee_bps_by_symbol: Mapping[str, float],
) -> dict[str, object]:
    return {
        "taker_fee_bps_by_symbol": {
            str(symbol).strip().upper(): float(value)
            for symbol, value in sorted(taker_fee_bps_by_symbol.items())
        }
    }


def round74_quantity_rules_evidence_claims(
    quantity_rules: Mapping[str, Round73MarketQuantityRules],
) -> dict[str, object]:
    return {
        "market_quantity_rules_by_symbol": {
            str(symbol).strip().upper(): {
                "step_size": format(value.step_size, "f"),
                "minimum_quantity": format(value.minimum_quantity, "f"),
                "maximum_quantity": format(value.maximum_quantity, "f"),
                "minimum_notional": format(value.minimum_notional, "f"),
            }
            for symbol, value in sorted(quantity_rules.items())
        },
        "quantity_filter": "MARKET_LOT_SIZE",
        "notional_filter": "MIN_NOTIONAL",
    }


def round74_latency_evidence_claims(
    *,
    decision_to_entry_latency_ns_by_symbol: Mapping[str, int],
    decision_to_exit_latency_ns_by_symbol: Mapping[str, int],
) -> dict[str, object]:
    return {
        "decision_to_entry_latency_ns_by_symbol": {
            str(symbol).strip().upper(): int(value)
            for symbol, value in sorted(
                decision_to_entry_latency_ns_by_symbol.items()
            )
        },
        "decision_to_exit_latency_ns_by_symbol": {
            str(symbol).strip().upper(): int(value)
            for symbol, value in sorted(
                decision_to_exit_latency_ns_by_symbol.items()
            )
        },
        "latency_semantics": "submission to terminal execution report",
    }


def round74_slippage_evidence_claims(
    *,
    reference_quote_notional: float,
    additional_slippage_bps_per_side_by_symbol: Mapping[str, float],
) -> dict[str, object]:
    return {
        "reference_quote_notional": float(reference_quote_notional),
        "additional_slippage_bps_per_side_by_symbol": {
            str(symbol).strip().upper(): float(value)
            for symbol, value in sorted(
                additional_slippage_bps_per_side_by_symbol.items()
            )
        },
        "slippage_semantics": (
            "realized execution shortfall residual after captured L2 book walk"
        ),
    }


def round74_funding_schedule_evidence_claims(
    *,
    funding_boundary_intervals_monotonic_ns: Mapping[
        str,
        Sequence[Sequence[int]],
    ],
    funding_schedule_coverage_monotonic_ns: Mapping[str, Sequence[int]],
) -> dict[str, object]:
    return {
        "funding_boundary_intervals_monotonic_ns": {
            str(symbol).strip().upper(): sorted(
                [int(interval[0]), int(interval[1])]
                for interval in intervals
            )
            for symbol, intervals in sorted(
                funding_boundary_intervals_monotonic_ns.items()
            )
        },
        "funding_schedule_coverage_monotonic_ns": {
            str(symbol).strip().upper(): [
                int(value) for value in coverage
            ]
            for symbol, coverage in sorted(
                funding_schedule_coverage_monotonic_ns.items()
            )
        },
        "funding_crossing_policy": "censor",
        "funding_coverage_policy": "censor",
    }


@dataclass(frozen=True)
class Round74EventTargetEvidence:
    """One immutable source record bound to an exact target claim."""

    kind: str
    source_id: str
    environment: str
    observed_wall_ns: int
    record_count: int
    source_query_or_protocol_sha256: str
    source_payload_sha256: str
    claims_sha256: str
    schema_version: str = ROUND74_EVENT_TARGET_EVIDENCE_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        kind: str,
        environment: str,
        observed_wall_ns: int,
        record_count: int,
        source_query_or_protocol_sha256: str,
        source_payload_sha256: str,
        claims: object,
    ) -> Round74EventTargetEvidence:
        selected_kind = str(kind).strip()
        try:
            source_id = ROUND74_EVENT_TARGET_EVIDENCE_SOURCES[selected_kind]
        except KeyError as exc:
            raise ValueError("Round 74 target evidence kind differs") from exc
        return cls(
            kind=selected_kind,
            source_id=source_id,
            environment=str(environment).strip(),
            observed_wall_ns=int(observed_wall_ns),
            record_count=int(record_count),
            source_query_or_protocol_sha256=str(
                source_query_or_protocol_sha256
            ),
            source_payload_sha256=str(source_payload_sha256),
            claims_sha256=_canonical_sha256(claims),
        )

    def __post_init__(self) -> None:
        if (
            self.kind not in ROUND74_EVENT_TARGET_EVIDENCE_SOURCES
            or self.source_id
            != ROUND74_EVENT_TARGET_EVIDENCE_SOURCES[self.kind]
        ):
            raise ValueError("Round 74 target evidence source differs")
        if self.environment not in ROUND74_EVENT_TARGET_ENVIRONMENTS:
            raise ValueError("Round 74 target evidence environment differs")
        if int(self.observed_wall_ns) <= 0:
            raise ValueError("Round 74 target evidence observation time differs")
        if int(self.record_count) < len(ROUND74_EVENT_TARGET_SYMBOLS):
            raise ValueError("Round 74 target evidence record count differs")
        _sha256_digest(
            self.source_query_or_protocol_sha256,
            "evidence query or protocol",
        )
        _sha256_digest(
            self.source_payload_sha256,
            "evidence source payload",
        )
        _sha256_digest(self.claims_sha256, "evidence claims")
        if self.schema_version != ROUND74_EVENT_TARGET_EVIDENCE_SCHEMA_VERSION:
            raise ValueError("Round 74 target evidence schema differs")

    @property
    def evidence_sha256(self) -> str:
        return _canonical_sha256(self.as_dict(include_sha256=False))

    def binds(self, claims: object) -> bool:
        return self.claims_sha256 == _canonical_sha256(claims)

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "source_id": self.source_id,
            "environment": self.environment,
            "observed_wall_ns": self.observed_wall_ns,
            "record_count": self.record_count,
            "source_query_or_protocol_sha256": (
                self.source_query_or_protocol_sha256
            ),
            "source_payload_sha256": self.source_payload_sha256,
            "claims_sha256": self.claims_sha256,
        }
        if include_sha256:
            value["evidence_sha256"] = _canonical_sha256(value)
        return value

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> Round74EventTargetEvidence:
        payload = dict(value)
        claimed = str(payload.pop("evidence_sha256", ""))
        if claimed != _canonical_sha256(payload):
            raise ValueError("Round 74 target evidence payload digest differs")
        selected = cls(
            kind=str(payload["kind"]),
            source_id=str(payload["source_id"]),
            environment=str(payload["environment"]),
            observed_wall_ns=int(payload["observed_wall_ns"]),
            record_count=int(payload["record_count"]),
            source_query_or_protocol_sha256=str(
                payload["source_query_or_protocol_sha256"]
            ),
            source_payload_sha256=str(payload["source_payload_sha256"]),
            claims_sha256=str(payload["claims_sha256"]),
            schema_version=str(payload["schema_version"]),
        )
        if selected.as_dict(include_sha256=False) != payload:
            raise ValueError("Round 74 target evidence static policy differs")
        return selected


@dataclass(frozen=True)
class Round74EventTargetSpec:
    """Hash-bound latency, cost, size, and sampling assumptions."""

    reference_quote_notional: float
    decision_to_entry_latency_ns_by_symbol: tuple[tuple[str, int], ...]
    decision_to_exit_latency_ns_by_symbol: tuple[tuple[str, int], ...]
    taker_fee_bps_by_symbol: tuple[tuple[str, float], ...]
    funding_boundary_intervals_monotonic_ns: tuple[
        tuple[str, tuple[tuple[int, int], ...]], ...
    ]
    funding_schedule_coverage_monotonic_ns: tuple[
        tuple[str, tuple[int, int]], ...
    ]
    additional_slippage_bps_per_side_by_symbol: tuple[
        tuple[str, float], ...
    ]
    quantity_rules_evidence: Round74EventTargetEvidence
    commission_evidence: Round74EventTargetEvidence
    entry_exit_latency_evidence: Round74EventTargetEvidence
    slippage_evidence: Round74EventTargetEvidence
    funding_schedule_evidence: Round74EventTargetEvidence
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
        decision_to_entry_latency_ns_by_symbol: Mapping[str, int],
        decision_to_exit_latency_ns_by_symbol: Mapping[str, int],
        taker_fee_bps_by_symbol: Mapping[str, float],
        funding_boundary_intervals_monotonic_ns: Mapping[
            str,
            Sequence[Sequence[int]],
        ],
        funding_schedule_coverage_monotonic_ns: Mapping[str, Sequence[int]],
        additional_slippage_bps_per_side_by_symbol: Mapping[str, float],
        quantity_rules_evidence: Round74EventTargetEvidence,
        commission_evidence: Round74EventTargetEvidence,
        entry_exit_latency_evidence: Round74EventTargetEvidence,
        slippage_evidence: Round74EventTargetEvidence,
        funding_schedule_evidence: Round74EventTargetEvidence,
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
        if any(
            len(interval) != 2
            for intervals in funding_boundary_intervals_monotonic_ns.values()
            for interval in intervals
        ):
            raise ValueError("Round 74 target funding schedule differs")
        funding = tuple(
            sorted(
                (
                    str(symbol).strip().upper(),
                    tuple(
                        sorted(
                            (int(interval[0]), int(interval[1]))
                            for interval in intervals
                        )
                    ),
                )
                for symbol, intervals in (
                    funding_boundary_intervals_monotonic_ns.items()
                )
            )
        )
        funding_coverage = tuple(
            sorted(
                (
                    str(symbol).strip().upper(),
                    (int(coverage[0]), int(coverage[1])),
                )
                for symbol, coverage in (
                    funding_schedule_coverage_monotonic_ns.items()
                )
                if len(coverage) == 2
            )
        )
        if len(funding_coverage) != len(
            funding_schedule_coverage_monotonic_ns
        ):
            raise ValueError("Round 74 target funding coverage differs")
        entry_latencies = tuple(
            sorted(
                (str(symbol).strip().upper(), int(value))
                for symbol, value in (
                    decision_to_entry_latency_ns_by_symbol.items()
                )
            )
        )
        exit_latencies = tuple(
            sorted(
                (str(symbol).strip().upper(), int(value))
                for symbol, value in (
                    decision_to_exit_latency_ns_by_symbol.items()
                )
            )
        )
        slippage = tuple(
            sorted(
                (str(symbol).strip().upper(), float(value))
                for symbol, value in (
                    additional_slippage_bps_per_side_by_symbol.items()
                )
            )
        )
        return cls(
            reference_quote_notional=float(reference_quote_notional),
            decision_to_entry_latency_ns_by_symbol=entry_latencies,
            decision_to_exit_latency_ns_by_symbol=exit_latencies,
            taker_fee_bps_by_symbol=fees,
            funding_boundary_intervals_monotonic_ns=funding,
            funding_schedule_coverage_monotonic_ns=funding_coverage,
            additional_slippage_bps_per_side_by_symbol=slippage,
            quantity_rules_evidence=quantity_rules_evidence,
            commission_evidence=commission_evidence,
            entry_exit_latency_evidence=entry_exit_latency_evidence,
            slippage_evidence=slippage_evidence,
            funding_schedule_evidence=funding_schedule_evidence,
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
        entry_latencies = tuple(self.decision_to_entry_latency_ns_by_symbol)
        exit_latencies = tuple(self.decision_to_exit_latency_ns_by_symbol)
        if (
            tuple(symbol for symbol, _value in entry_latencies)
            != ROUND74_EVENT_TARGET_SYMBOLS
            or tuple(symbol for symbol, _value in exit_latencies)
            != ROUND74_EVENT_TARGET_SYMBOLS
        ):
            raise ValueError("Round 74 target latency universe differs")
        latencies = tuple(
            int(value)
            for panel in (entry_latencies, exit_latencies)
            for _symbol, value in panel
        )
        if any(
            latency <= 0 or latency > ROUND74_EVENT_TARGET_MAXIMUM_LATENCY_NS
            for latency in latencies
        ):
            raise ValueError("Round 74 target latency is invalid")
        fees = tuple(self.taker_fee_bps_by_symbol)
        if tuple(symbol for symbol, _fee in fees) != ROUND74_EVENT_TARGET_SYMBOLS:
            raise ValueError("Round 74 target fee universe differs")
        if any(
            not math.isfinite(float(fee)) or not 0.0 <= float(fee) <= 100.0
            for _symbol, fee in fees
        ):
            raise ValueError("Round 74 target fee is invalid")
        funding = tuple(self.funding_boundary_intervals_monotonic_ns)
        if (
            tuple(symbol for symbol, _intervals in funding)
            != ROUND74_EVENT_TARGET_SYMBOLS
            or any(
                int(interval[0]) < 0 or int(interval[1]) < int(interval[0])
                for _symbol, intervals in funding
                for interval in intervals
            )
            or any(
                tuple(intervals) != tuple(sorted(intervals))
                or len(intervals) != len(set(intervals))
                or any(
                    int(intervals[index][0])
                    <= int(intervals[index - 1][1])
                    for index in range(1, len(intervals))
                )
                for _symbol, intervals in funding
            )
        ):
            raise ValueError("Round 74 target funding schedule differs")
        funding_coverage = tuple(
            self.funding_schedule_coverage_monotonic_ns
        )
        if (
            tuple(symbol for symbol, _coverage in funding_coverage)
            != ROUND74_EVENT_TARGET_SYMBOLS
            or any(
                int(coverage[0]) < 0
                or int(coverage[1]) <= int(coverage[0])
                for _symbol, coverage in funding_coverage
            )
            or any(
                any(
                    int(interval[0]) < int(coverage[0])
                    or int(interval[1]) > int(coverage[1])
                    for interval in dict(funding)[symbol]
                )
                for symbol, coverage in funding_coverage
            )
        ):
            raise ValueError("Round 74 target funding coverage differs")
        slippage = tuple(self.additional_slippage_bps_per_side_by_symbol)
        if (
            tuple(symbol for symbol, _value in slippage)
            != ROUND74_EVENT_TARGET_SYMBOLS
        ):
            raise ValueError("Round 74 target slippage universe differs")
        if any(
            _finite_nonnegative(value, "target additional slippage")
            > ROUND74_EVENT_TARGET_MAXIMUM_SLIPPAGE_BPS_PER_SIDE
            for _symbol, value in slippage
        ):
            raise ValueError("Round 74 target additional slippage is too large")
        evidence_panel = (
            (
                "commission",
                self.commission_evidence,
                round74_commission_evidence_claims(dict(fees)),
            ),
            (
                "entry_exit_latency",
                self.entry_exit_latency_evidence,
                round74_latency_evidence_claims(
                    decision_to_entry_latency_ns_by_symbol=dict(
                        entry_latencies
                    ),
                    decision_to_exit_latency_ns_by_symbol=dict(
                        exit_latencies
                    ),
                ),
            ),
            (
                "residual_slippage",
                self.slippage_evidence,
                round74_slippage_evidence_claims(
                    reference_quote_notional=reference,
                    additional_slippage_bps_per_side_by_symbol=dict(
                        slippage
                    ),
                ),
            ),
            (
                "funding_schedule",
                self.funding_schedule_evidence,
                round74_funding_schedule_evidence_claims(
                    funding_boundary_intervals_monotonic_ns=dict(funding),
                    funding_schedule_coverage_monotonic_ns=dict(
                        funding_coverage
                    ),
                ),
            ),
        )
        if (
            not isinstance(
                self.quantity_rules_evidence,
                Round74EventTargetEvidence,
            )
            or self.quantity_rules_evidence.kind != "quantity_rules"
        ):
            raise ValueError("Round 74 target quantity-rules evidence differs")
        if any(
            not isinstance(evidence, Round74EventTargetEvidence)
            or evidence.kind != kind
            or not evidence.binds(claims)
            for kind, evidence, claims in evidence_panel
        ):
            raise ValueError("Round 74 target evidence claims differ")
        if len(
            {
                evidence.environment
                for evidence in (
                    self.quantity_rules_evidence,
                    *(item[1] for item in evidence_panel),
                )
            }
        ) != 1:
            raise ValueError("Round 74 target evidence environments differ")
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

    @property
    def execution_environment(self) -> str:
        return self.commission_evidence.environment

    @staticmethod
    def _symbol_value(
        panel: tuple[tuple[str, int | float], ...],
        symbol: str,
        *,
        label: str,
    ) -> int | float:
        selected = str(symbol).strip().upper()
        try:
            return dict(panel)[selected]
        except KeyError as exc:
            raise ValueError(f"Round 74 target {label} symbol differs") from exc

    def entry_latency_ns(self, symbol: str) -> int:
        return int(
            self._symbol_value(
                self.decision_to_entry_latency_ns_by_symbol,
                symbol,
                label="entry latency",
            )
        )

    def exit_latency_ns(self, symbol: str) -> int:
        return int(
            self._symbol_value(
                self.decision_to_exit_latency_ns_by_symbol,
                symbol,
                label="exit latency",
            )
        )

    def fee_bps(self, symbol: str) -> float:
        return float(
            self._symbol_value(
                self.taker_fee_bps_by_symbol,
                symbol,
                label="fee",
            )
        )

    def slippage_bps_per_side(self, symbol: str) -> float:
        return float(
            self._symbol_value(
                self.additional_slippage_bps_per_side_by_symbol,
                symbol,
                label="slippage",
            )
        )

    def as_dict(self, *, include_sha256: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "reference_quote_notional": self.reference_quote_notional,
            "decision_to_entry_latency_ns_by_symbol": dict(
                self.decision_to_entry_latency_ns_by_symbol
            ),
            "decision_to_exit_latency_ns_by_symbol": dict(
                self.decision_to_exit_latency_ns_by_symbol
            ),
            "taker_fee_bps_by_symbol": dict(self.taker_fee_bps_by_symbol),
            "funding_boundary_intervals_monotonic_ns": {
                symbol: [list(interval) for interval in intervals]
                for symbol, intervals in (
                    self.funding_boundary_intervals_monotonic_ns
                )
            },
            "funding_schedule_coverage_monotonic_ns": {
                symbol: list(coverage)
                for symbol, coverage in (
                    self.funding_schedule_coverage_monotonic_ns
                )
            },
            "additional_slippage_bps_per_side_by_symbol": dict(
                self.additional_slippage_bps_per_side_by_symbol
            ),
            "evidence": {
                "quantity_rules": self.quantity_rules_evidence.as_dict(),
                "commission": self.commission_evidence.as_dict(),
                "entry_exit_latency": (
                    self.entry_exit_latency_evidence.as_dict()
                ),
                "residual_slippage": self.slippage_evidence.as_dict(),
                "funding_schedule": self.funding_schedule_evidence.as_dict(),
            },
            "execution_environment": self.execution_environment,
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
            "funding_schedule_is_mandatory_and_hash_bound": True,
            "latency_and_residual_slippage_are_symbol_specific": True,
            "latency_semantics": (
                "separately measured symbol-specific entry and exit "
                "submission-to-terminal-execution-report delays"
            ),
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
        funding = payload.get("funding_boundary_intervals_monotonic_ns")
        funding_coverage = payload.get(
            "funding_schedule_coverage_monotonic_ns"
        )
        entry_latencies = payload.get(
            "decision_to_entry_latency_ns_by_symbol"
        )
        exit_latencies = payload.get("decision_to_exit_latency_ns_by_symbol")
        slippage = payload.get(
            "additional_slippage_bps_per_side_by_symbol"
        )
        evidence = payload.get("evidence")
        if not all(
            isinstance(item, Mapping)
            for item in (
                fees,
                funding,
                funding_coverage,
                entry_latencies,
                exit_latencies,
                slippage,
                evidence,
            )
        ):
            raise ValueError("Round 74 target mapping payload differs")
        assert isinstance(evidence, Mapping)
        if set(evidence) != {
            "quantity_rules",
            "commission",
            "entry_exit_latency",
            "residual_slippage",
            "funding_schedule",
        } or not all(isinstance(item, Mapping) for item in evidence.values()):
            raise ValueError("Round 74 target evidence panel differs")
        assert isinstance(fees, Mapping)
        assert isinstance(funding, Mapping)
        assert isinstance(funding_coverage, Mapping)
        assert isinstance(entry_latencies, Mapping)
        assert isinstance(exit_latencies, Mapping)
        assert isinstance(slippage, Mapping)
        selected = cls.create(
            reference_quote_notional=float(
                payload["reference_quote_notional"]
            ),
            decision_to_entry_latency_ns_by_symbol={
                str(symbol): int(latency)
                for symbol, latency in entry_latencies.items()
            },
            decision_to_exit_latency_ns_by_symbol={
                str(symbol): int(latency)
                for symbol, latency in exit_latencies.items()
            },
            taker_fee_bps_by_symbol={
                str(symbol): float(fee) for symbol, fee in fees.items()
            },
            funding_boundary_intervals_monotonic_ns={
                str(symbol): tuple(
                    tuple(int(value) for value in interval)
                    for interval in intervals
                )
                for symbol, intervals in funding.items()
            },
            funding_schedule_coverage_monotonic_ns={
                str(symbol): tuple(int(value) for value in coverage)
                for symbol, coverage in funding_coverage.items()
            },
            additional_slippage_bps_per_side_by_symbol={
                str(symbol): float(value)
                for symbol, value in slippage.items()
            },
            quantity_rules_evidence=Round74EventTargetEvidence.from_dict(
                evidence["quantity_rules"]
            ),
            commission_evidence=Round74EventTargetEvidence.from_dict(
                evidence["commission"]
            ),
            entry_exit_latency_evidence=(
                Round74EventTargetEvidence.from_dict(
                    evidence["entry_exit_latency"]
                )
            ),
            slippage_evidence=Round74EventTargetEvidence.from_dict(
                evidence["residual_slippage"]
            ),
            funding_schedule_evidence=Round74EventTargetEvidence.from_dict(
                evidence["funding_schedule"]
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
class Round74EventExecutionOverride:
    """Review-specific delay and size applied before an exact L2 book walk."""

    symbol: str
    anchor_index: int
    feature_window_sha256: str
    additional_entry_latency_ns: int
    quote_size_multiplier_bps: int
    source_review_sha256: str
    schema_version: str = ROUND74_EVENT_EXECUTION_OVERRIDE_SCHEMA_VERSION

    @property
    def key(self) -> tuple[str, int]:
        return self.symbol, int(self.anchor_index)

    def validate(self) -> None:
        if (
            self.schema_version
            != ROUND74_EVENT_EXECUTION_OVERRIDE_SCHEMA_VERSION
            or self.symbol not in ROUND74_EVENT_TARGET_SYMBOLS
            or isinstance(self.anchor_index, bool)
            or not isinstance(self.anchor_index, int)
            or self.anchor_index < 0
            or isinstance(self.additional_entry_latency_ns, bool)
            or not isinstance(self.additional_entry_latency_ns, int)
            or not 0
            <= self.additional_entry_latency_ns
            <= ROUND74_EVENT_TARGET_MAXIMUM_ADDITIONAL_ENTRY_LATENCY_NS
            or isinstance(self.quote_size_multiplier_bps, bool)
            or not isinstance(self.quote_size_multiplier_bps, int)
            or not 1 <= self.quote_size_multiplier_bps <= 10_000
        ):
            raise ValueError("Round 74 event execution override differs")
        _sha256_digest(self.feature_window_sha256, "override feature window")
        _sha256_digest(self.source_review_sha256, "override source review")

    def as_dict(self) -> dict[str, object]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "anchor_index": self.anchor_index,
            "feature_window_sha256": self.feature_window_sha256,
            "additional_entry_latency_ns": self.additional_entry_latency_ns,
            "quote_size_multiplier_bps": self.quote_size_multiplier_bps,
            "source_review_sha256": self.source_review_sha256,
        }


@dataclass(frozen=True)
class Round74EventActionPayoff:
    midpoint_payoff_quote: float
    midpoint_payoff_bps: float
    book_walk_implementation_shortfall_quote: float
    book_walk_implementation_shortfall_bps: float
    gross_payoff_quote: float
    gross_payoff_bps: float
    commission_quote: float
    commission_bps: float
    additional_slippage_quote: float
    additional_slippage_bps: float
    explicit_cost_quote: float
    explicit_cost_bps: float
    total_implementation_shortfall_quote: float
    total_implementation_shortfall_bps: float
    net_payoff_quote: float
    net_payoff_bps: float


def round74_event_action_payoff(
    *,
    side: str,
    entry_walk: Round73BookWalk,
    exit_walk: Round73BookWalk,
    entry_mid: float,
    exit_mid: float,
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
    entry_midpoint = float(entry_mid)
    exit_midpoint = float(exit_mid)
    if (
        not math.isfinite(entry_quote)
        or not math.isfinite(exit_quote)
        or not math.isfinite(entry_midpoint)
        or not math.isfinite(exit_midpoint)
        or entry_quote <= 0.0
        or exit_quote <= 0.0
        or entry_midpoint <= 0.0
        or exit_midpoint <= 0.0
    ):
        raise ValueError("Round 74 payoff price or quote notional is invalid")
    base_quantity = float(entry_walk.filled_base_quantity)
    midpoint = base_quantity * (
        exit_midpoint - entry_midpoint
        if selected_side == "long"
        else entry_midpoint - exit_midpoint
    )
    gross = (
        exit_quote - entry_quote
        if selected_side == "long"
        else entry_quote - exit_quote
    )
    book_walk_shortfall = midpoint - gross
    tolerance = max(1e-12, abs(midpoint) * 1e-12, entry_quote * 1e-12)
    if book_walk_shortfall < -tolerance:
        raise ValueError("Round 74 payoff book walk improves on both midpoints")
    if book_walk_shortfall < 0.0:
        book_walk_shortfall = 0.0
    midpoint_bps = midpoint / entry_quote * 10_000.0
    book_walk_shortfall_bps = (
        book_walk_shortfall / entry_quote * 10_000.0
    )
    gross_bps = gross / entry_quote * 10_000.0
    commission = fee / 10_000.0 * (entry_quote + exit_quote)
    residual_slippage = slippage / 10_000.0 * (entry_quote + exit_quote)
    explicit_cost = commission + residual_slippage
    total_implementation_shortfall = book_walk_shortfall + explicit_cost
    commission_bps = commission / entry_quote * 10_000.0
    residual_slippage_bps = residual_slippage / entry_quote * 10_000.0
    explicit_cost_bps = explicit_cost / entry_quote * 10_000.0
    total_implementation_shortfall_bps = (
        total_implementation_shortfall / entry_quote * 10_000.0
    )
    net = gross - explicit_cost
    net_bps = net / entry_quote * 10_000.0
    values = (
        midpoint,
        midpoint_bps,
        book_walk_shortfall,
        book_walk_shortfall_bps,
        gross,
        gross_bps,
        commission,
        commission_bps,
        residual_slippage,
        residual_slippage_bps,
        explicit_cost,
        explicit_cost_bps,
        total_implementation_shortfall,
        total_implementation_shortfall_bps,
        net,
        net_bps,
    )
    if not all(math.isfinite(value) for value in values):
        raise ArithmeticError("Round 74 payoff is nonfinite")
    if not math.isclose(
        midpoint - total_implementation_shortfall,
        net,
        rel_tol=1e-12,
        abs_tol=tolerance,
    ):
        raise ArithmeticError("Round 74 payoff reconciliation differs")
    return Round74EventActionPayoff(
        midpoint_payoff_quote=midpoint,
        midpoint_payoff_bps=midpoint_bps,
        book_walk_implementation_shortfall_quote=book_walk_shortfall,
        book_walk_implementation_shortfall_bps=book_walk_shortfall_bps,
        gross_payoff_quote=gross,
        gross_payoff_bps=gross_bps,
        commission_quote=commission,
        commission_bps=commission_bps,
        additional_slippage_quote=residual_slippage,
        additional_slippage_bps=residual_slippage_bps,
        explicit_cost_quote=explicit_cost,
        explicit_cost_bps=explicit_cost_bps,
        total_implementation_shortfall_quote=(
            total_implementation_shortfall
        ),
        total_implementation_shortfall_bps=(
            total_implementation_shortfall_bps
        ),
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
    midpoint_payoff_bps: float | None
    book_walk_implementation_shortfall_quote: float | None
    book_walk_implementation_shortfall_bps: float | None
    gross_payoff_bps: float | None
    commission_quote: float | None
    commission_bps: float | None
    additional_slippage_quote: float | None
    additional_slippage_bps: float | None
    explicit_cost_quote: float | None
    explicit_cost_bps: float | None
    total_implementation_shortfall_quote: float | None
    total_implementation_shortfall_bps: float | None
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
            self.midpoint_payoff_bps,
            self.book_walk_implementation_shortfall_quote,
            self.book_walk_implementation_shortfall_bps,
            self.gross_payoff_bps,
            self.commission_quote,
            self.commission_bps,
            self.additional_slippage_quote,
            self.additional_slippage_bps,
            self.explicit_cost_quote,
            self.explicit_cost_bps,
            self.total_implementation_shortfall_quote,
            self.total_implementation_shortfall_bps,
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
                or float(self.book_walk_implementation_shortfall_quote) < 0.0
                or float(self.book_walk_implementation_shortfall_bps) < 0.0
                or float(self.commission_quote) < 0.0
                or float(self.commission_bps) < 0.0
                or float(self.additional_slippage_quote) < 0.0
                or float(self.additional_slippage_bps) < 0.0
                or float(self.explicit_cost_quote) < 0.0
                or float(self.explicit_cost_bps) < 0.0
                or float(self.total_implementation_shortfall_quote) < 0.0
                or float(self.total_implementation_shortfall_bps) < 0.0
                or float(self.maximum_adverse_excursion_bps) < 0.0
                or float(self.maximum_favorable_excursion_bps) < 0.0
                or not 0.0 <= float(self.regime_unpredictability) <= 1.0
                or float(self.minimum_exit_side_capacity_ratio) < 1.0
            ):
                raise ValueError("Round 74 eligible outcome bounds differ")
            if not (
                math.isclose(
                    float(self.midpoint_payoff_bps)
                    - float(self.book_walk_implementation_shortfall_bps),
                    float(self.gross_payoff_bps),
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                )
                and math.isclose(
                    float(self.commission_quote)
                    + float(self.additional_slippage_quote),
                    float(self.explicit_cost_quote),
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                )
                and math.isclose(
                    float(self.commission_bps)
                    + float(self.additional_slippage_bps),
                    float(self.explicit_cost_bps),
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                )
                and math.isclose(
                    float(self.book_walk_implementation_shortfall_quote)
                    + float(self.explicit_cost_quote),
                    float(self.total_implementation_shortfall_quote),
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                )
                and math.isclose(
                    float(self.book_walk_implementation_shortfall_bps)
                    + float(self.explicit_cost_bps),
                    float(self.total_implementation_shortfall_bps),
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                )
                and math.isclose(
                    float(self.midpoint_payoff_bps)
                    - float(self.total_implementation_shortfall_bps),
                    float(self.net_payoff_bps),
                    rel_tol=1e-10,
                    abs_tol=1e-10,
                )
            ):
                raise ValueError("Round 74 eligible outcome accounting differs")
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
            "midpoint_payoff_bps": self.midpoint_payoff_bps,
            "book_walk_implementation_shortfall_quote": (
                self.book_walk_implementation_shortfall_quote
            ),
            "book_walk_implementation_shortfall_bps": (
                self.book_walk_implementation_shortfall_bps
            ),
            "gross_payoff_bps": self.gross_payoff_bps,
            "commission_quote": self.commission_quote,
            "commission_bps": self.commission_bps,
            "additional_slippage_quote": self.additional_slippage_quote,
            "additional_slippage_bps": self.additional_slippage_bps,
            "explicit_cost_quote": self.explicit_cost_quote,
            "explicit_cost_bps": self.explicit_cost_bps,
            "total_implementation_shortfall_quote": (
                self.total_implementation_shortfall_quote
            ),
            "total_implementation_shortfall_bps": (
                self.total_implementation_shortfall_bps
            ),
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
        execution_overrides: Sequence[Round74EventExecutionOverride] = (),
    ) -> None:
        self.spec = spec
        overrides: dict[tuple[str, int], Round74EventExecutionOverride] = {}
        for override in execution_overrides:
            if not isinstance(override, Round74EventExecutionOverride):
                raise TypeError("Round 74 event execution override type differs")
            override.validate()
            if override.key in overrides:
                raise ValueError("Round 74 event execution override is duplicated")
            overrides[override.key] = override
        self.execution_overrides = overrides
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
        if self.execution_overrides and set(self.execution_overrides) != (
            self._anchor_keys
        ):
            raise ValueError("Round 74 event execution override coverage differs")
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
        if not self.spec.quantity_rules_evidence.binds(
            round74_quantity_rules_evidence_claims(validated_rules)
        ):
            raise ValueError("Round 74 target quantity-rules claims differ")
        self.funding_boundary_intervals = {
            symbol: tuple(intervals)
            for symbol, intervals in (
                self.spec.funding_boundary_intervals_monotonic_ns
            )
        }
        self.funding_coverage = {
            symbol: tuple(coverage)
            for symbol, coverage in (
                self.spec.funding_schedule_coverage_monotonic_ns
            )
        }
        context: dict[str, object] = {
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
            "funding_boundary_intervals_monotonic_ns": {
                symbol: [
                    list(interval)
                    for interval in self.funding_boundary_intervals[symbol]
                ]
                for symbol in ROUND74_EVENT_TARGET_SYMBOLS
            },
            "funding_schedule_coverage_monotonic_ns": {
                symbol: list(self.funding_coverage[symbol])
                for symbol in ROUND74_EVENT_TARGET_SYMBOLS
            },
        }
        if self.execution_overrides:
            context["execution_overrides"] = [
                self.execution_overrides[key].as_dict()
                for key in sorted(self.execution_overrides)
            ]
        self.target_context_sha256 = _canonical_sha256(context)
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
        override = self.execution_overrides.get(key)
        if self.execution_overrides and (
            override is None
            or override.feature_window_sha256 != anchor.feature_window_sha256
        ):
            raise ValueError("Round 74 event execution override identity differs")
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
            midpoint_payoff_bps=None,
            book_walk_implementation_shortfall_quote=None,
            book_walk_implementation_shortfall_bps=None,
            gross_payoff_bps=None,
            commission_quote=None,
            commission_bps=None,
            additional_slippage_quote=None,
            additional_slippage_bps=None,
            explicit_cost_quote=None,
            explicit_cost_bps=None,
            total_implementation_shortfall_quote=None,
            total_implementation_shortfall_bps=None,
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
                    + self.spec.entry_latency_ns(symbol)
                )
                override = self.execution_overrides.get(
                    (anchor.symbol, anchor.anchor_index)
                )
                if override is not None:
                    requested += override.additional_entry_latency_ns
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
                    reference_quote_notional=(
                        self.spec.reference_quote_notional
                        * (
                            override.quote_size_multiplier_bps
                            if override is not None
                            else 10_000
                        )
                        / 10_000.0
                    ),
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

    def _funding_ineligibility(
        self,
        symbol: str,
        *,
        entry_ns: int,
        exit_ns: int,
    ) -> str:
        coverage_start, coverage_end = self.funding_coverage[symbol]
        if int(entry_ns) < coverage_start or int(exit_ns) > coverage_end:
            return "funding_coverage"
        if any(
            entry_ns <= interval[1] and exit_ns >= interval[0]
            for interval in self.funding_boundary_intervals[symbol]
        ):
            return "funding_boundary"
        return ""

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
                    requested_exit = (
                        received_monotonic_ns
                        + horizon * 1_000_000_000
                        + self.spec.exit_latency_ns(symbol)
                    )
                    funding_reason = self._funding_ineligibility(
                        symbol,
                        entry_ns=received_monotonic_ns,
                        exit_ns=requested_exit,
                    )
                    if reason or funding_reason:
                        self._record_ineligible(
                            anchor=anchor,
                            horizon_seconds=horizon,
                            side=side,
                            reason=reason or funding_reason,
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
                        entry_mid=state.mid,
                        exit_mid=state.mid,
                        taker_fee_bps=self.spec.fee_bps(symbol),
                        additional_slippage_bps_per_side=(
                            self.spec.slippage_bps_per_side(symbol)
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
        funding_reason = self._funding_ineligibility(
            position.decision.anchor.symbol,
            entry_ns=position.actual_entry_monotonic_ns,
            exit_ns=received_monotonic_ns,
        )
        if funding_reason:
            self._record_ineligible(
                anchor=position.decision.anchor,
                horizon_seconds=position.horizon_seconds,
                side=position.side,
                reason=funding_reason,
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
            midpoint_payoff_bps=payoff.midpoint_payoff_bps,
            book_walk_implementation_shortfall_quote=(
                payoff.book_walk_implementation_shortfall_quote
            ),
            book_walk_implementation_shortfall_bps=(
                payoff.book_walk_implementation_shortfall_bps
            ),
            gross_payoff_bps=payoff.gross_payoff_bps,
            commission_quote=payoff.commission_quote,
            commission_bps=payoff.commission_bps,
            additional_slippage_quote=payoff.additional_slippage_quote,
            additional_slippage_bps=payoff.additional_slippage_bps,
            explicit_cost_quote=payoff.explicit_cost_quote,
            explicit_cost_bps=payoff.explicit_cost_bps,
            total_implementation_shortfall_quote=(
                payoff.total_implementation_shortfall_quote
            ),
            total_implementation_shortfall_bps=(
                payoff.total_implementation_shortfall_bps
            ),
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
            adverse_selection=payoff.midpoint_payoff_bps < 0.0,
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
                entry_mid=position.entry_mid,
                exit_mid=state.mid,
                taker_fee_bps=self.spec.fee_bps(symbol),
                additional_slippage_bps_per_side=(
                    self.spec.slippage_bps_per_side(symbol)
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
                        + self.spec.entry_latency_ns(symbol)
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
    "ROUND74_EVENT_TARGET_ENVIRONMENTS",
    "ROUND74_EVENT_TARGET_EVIDENCE_SCHEMA_VERSION",
    "ROUND74_EVENT_TARGET_EVIDENCE_SOURCES",
    "ROUND74_EVENT_EXECUTION_OVERRIDE_SCHEMA_VERSION",
    "ROUND74_EVENT_TARGET_INELIGIBLE_REASONS",
    "ROUND74_EVENT_TARGET_MAXIMUM_ADDITIONAL_ENTRY_LATENCY_NS",
    "ROUND74_EVENT_TARGET_MAXIMUM_DECISION_STATE_AGE_NS",
    "ROUND74_EVENT_TARGET_MAXIMUM_LATENCY_NS",
    "ROUND74_EVENT_TARGET_MAXIMUM_SLIPPAGE_BPS_PER_SIDE",
    "ROUND74_EVENT_TARGET_MAXIMUM_STATE_LATENESS_NS",
    "ROUND74_EVENT_TARGET_MINIMUM_ANCHOR_SPACING_NS",
    "ROUND74_EVENT_TARGET_SCHEMA_VERSION",
    "ROUND74_EVENT_TARGET_SYMBOLS",
    "Round74EventActionPayoff",
    "Round74EventExecutionOverride",
    "Round74EventTargetAnchor",
    "Round74EventTargetEngine",
    "Round74EventTargetEvidence",
    "Round74EventTargetOutcome",
    "Round74EventTargetSpec",
    "round74_commission_evidence_claims",
    "round74_event_action_payoff",
    "round74_funding_schedule_evidence_claims",
    "round74_latency_evidence_claims",
    "round74_quantity_rules_evidence_claims",
    "round74_slippage_evidence_claims",
]
