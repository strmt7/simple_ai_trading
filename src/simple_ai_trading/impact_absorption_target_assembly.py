"""Source-derived assembly for Round 74 executable event targets."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping, Sequence

from .impact_absorption_event_evidence import (
    Round74BinanceClockProbe,
    build_round74_commission_evidence,
    build_round74_funding_evidence,
    build_round74_quantity_rules_evidence,
)
from .impact_absorption_event_targets import (
    ROUND74_EVENT_TARGET_SYMBOLS,
    Round74EventTargetAnchor,
    Round74EventTargetEngine,
    Round74EventTargetSpec,
    round74_quantity_rules_evidence_claims,
)
from .impact_absorption_execution_evidence import (
    build_round74_execution_calibration_evidence,
)
from .impact_absorption_targets import Round73MarketQuantityRules


ROUND74_SOURCE_TARGET_ASSEMBLY_SCHEMA_VERSION = (
    "round-074-source-target-assembly-v1"
)


def _canonical_sha256(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError("Round 74 source target assembly is not canonical") from exc
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class Round74SourceTargetAssembly:
    """A spec and market rules built from the mandatory source validators."""

    spec: Round74EventTargetSpec
    quantity_rules_by_symbol: tuple[
        tuple[str, Round73MarketQuantityRules],
        ...,
    ]
    schema_version: str = ROUND74_SOURCE_TARGET_ASSEMBLY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if (
            not isinstance(self.spec, Round74EventTargetSpec)
            or self.schema_version
            != ROUND74_SOURCE_TARGET_ASSEMBLY_SCHEMA_VERSION
        ):
            raise ValueError("Round 74 source target assembly contract differs")
        rules = self.quantity_rules_mapping()
        if tuple(rules) != ROUND74_EVENT_TARGET_SYMBOLS or any(
            not isinstance(rules[symbol], Round73MarketQuantityRules)
            or rules[symbol].symbol != symbol
            for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ):
            raise ValueError("Round 74 source target assembly rules differ")
        if not self.spec.quantity_rules_evidence.binds(
            round74_quantity_rules_evidence_claims(rules)
        ):
            raise ValueError(
                "Round 74 source target assembly evidence claims differ"
            )
        Round74EventTargetEngine(
            spec=self.spec,
            anchors=(),
            quantity_rules=rules,
        )

    def quantity_rules_mapping(
        self,
    ) -> dict[str, Round73MarketQuantityRules]:
        return dict(self.quantity_rules_by_symbol)

    @property
    def assembly_sha256(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": self.schema_version,
                "target_spec_sha256": self.spec.spec_sha256,
                "quantity_rules": round74_quantity_rules_evidence_claims(
                    self.quantity_rules_mapping()
                ),
            }
        )

    def create_engine(
        self,
        *,
        anchors: Sequence[Round74EventTargetAnchor],
    ) -> Round74EventTargetEngine:
        return Round74EventTargetEngine(
            spec=self.spec,
            anchors=anchors,
            quantity_rules=self.quantity_rules_mapping(),
        )


def assemble_round74_source_target(
    *,
    exchange_info_payload: Mapping[str, object],
    commission_payload_by_symbol: Mapping[str, Mapping[str, object]],
    funding_payload_by_symbol: Mapping[
        str,
        Sequence[Mapping[str, object]],
    ],
    execution_calibration_records: Sequence[Mapping[str, object]],
    funding_clock_probes: Sequence[Round74BinanceClockProbe],
    environment: str,
    exchange_info_observed_wall_ns: int,
    commission_observed_wall_ns: int,
    funding_observed_wall_ns: int,
    execution_observed_wall_ns: int,
    funding_start_time_ms: int,
    funding_end_time_ms: int,
    funding_limit: int,
    reference_quote_notional: float,
) -> Round74SourceTargetAssembly:
    """Assemble a target spec without accepting caller-supplied target values."""

    quantity = build_round74_quantity_rules_evidence(
        payload=exchange_info_payload,
        environment=environment,
        observed_wall_ns=exchange_info_observed_wall_ns,
    )
    commission = build_round74_commission_evidence(
        payload_by_symbol=commission_payload_by_symbol,
        environment=environment,
        observed_wall_ns=commission_observed_wall_ns,
    )
    funding = build_round74_funding_evidence(
        payload_by_symbol=funding_payload_by_symbol,
        environment=environment,
        observed_wall_ns=funding_observed_wall_ns,
        start_time_ms=funding_start_time_ms,
        end_time_ms=funding_end_time_ms,
        limit=funding_limit,
        clock_probes=funding_clock_probes,
    )
    execution = build_round74_execution_calibration_evidence(
        records=execution_calibration_records,
        environment=environment,
        observed_wall_ns=execution_observed_wall_ns,
        reference_quote_notional=reference_quote_notional,
    )
    rules = quantity.as_mapping()
    spec = Round74EventTargetSpec.create(
        reference_quote_notional=execution.reference_quote_notional,
        decision_to_entry_latency_ns_by_symbol=(
            execution.entry_latency_mapping()
        ),
        decision_to_exit_latency_ns_by_symbol=(
            execution.exit_latency_mapping()
        ),
        taker_fee_bps_by_symbol=commission.as_mapping(),
        funding_boundary_intervals_monotonic_ns=(
            funding.boundary_mapping()
        ),
        funding_schedule_coverage_monotonic_ns=(
            funding.coverage_mapping()
        ),
        additional_slippage_bps_per_side_by_symbol=(
            execution.slippage_mapping()
        ),
        quantity_rules_evidence=quantity.evidence,
        commission_evidence=commission.evidence,
        entry_exit_latency_evidence=(
            execution.entry_exit_latency_evidence
        ),
        slippage_evidence=execution.residual_slippage_evidence,
        funding_schedule_evidence=funding.evidence,
    )
    return Round74SourceTargetAssembly(
        spec=spec,
        quantity_rules_by_symbol=tuple(
            (symbol, rules[symbol]) for symbol in ROUND74_EVENT_TARGET_SYMBOLS
        ),
    )


__all__ = [
    "ROUND74_SOURCE_TARGET_ASSEMBLY_SCHEMA_VERSION",
    "Round74SourceTargetAssembly",
    "assemble_round74_source_target",
]
